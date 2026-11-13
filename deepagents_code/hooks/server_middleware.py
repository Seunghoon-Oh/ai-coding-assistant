"""Server-owned Hooks v2 lifecycle middleware.

Emits `PreCompact`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop`,
`SubagentStart`, and `SubagentStop` through the LangGraph interrupt channel so the
client runtime can execute matching handlers and return typed decisions.
"""

# [해설] 모듈 개요: 사용자가 설정한 훅(Hooks v2)을 '서버 에이전트 그래프의 미들웨어
# [해설]   체인' 안에서 발화시키는 `ServerHooksMiddleware`를 정의한다.
# [해설] 핵심 아이디어: 훅 핸들러는 서버가 아니라 클라이언트(TUI)에서 실행된다. 서버는
# [해설]   이벤트를 LangGraph `interrupt()`로 올려 보내고, 클라이언트가 핸들러를 돌린 뒤
# [해설]   결정을 resume 값으로 돌려준다(`_invoke_hook`).
# [해설] 재정의하는 langchain `AgentMiddleware` 훅(동기/비동기 쌍): before_agent(턴
# [해설]   게이트 저장), before_model(PostToolUse 등 사후 훅),
# [해설]   after_model(PreToolUse/PreCompact 사전 훅),
# [해설] wrap_tool_call(차단 적용, SubagentStart, 실행 시간 측정), after_agent(Stop).
# [해설]   `wrap_model_call`은 재정의하지 않는다.
# [해설][흐름] 한 턴의 순서: before_agent -> (모델 호출) -> after_model: PreToolUse
# [해설]   결과를 상태에 저장 -> [승인 미들웨어 `AutoModeHITLMiddleware` 등이 훅 결과를
# [해설]   참고해 승인 라우팅] -> wrap_tool_call: 차단이면 실행 없이 거부 메시지, 아니면
# [해설]   실행 후 대기열 기록
# [해설][흐름] -> 다음 before_model: 체크포인트가 확정된 뒤
# [해설]   PostToolUse/PostToolUseFailure/SubagentStop 실행 -> 모델이 도구 호출 없이
# [해설]   끝나면 after_agent: Stop 훅이 '계속 작업'을 요청할 수
# [해설]   있음(`jump_to="model"`).
# [해설][설계] 사후 훅을 도구 실행 직후가 아니라 '다음 before_model'로 미룬 이유:
# [해설]   interrupt가 도구 실행 중에 나면 resume 시 도구가 다시 실행될 수 있어서,
# [해설]   결과가 체크포인트된 안전한 경계에서만 훅을 돌린다(대기 기록은
# [해설]   _record_pending_post_tool, 재실행이 없음은
# [해설]   test_post_tool_resumes_do_not_reexecute_parallel_tools가 검증).
# [해설] 미들웨어 순서는 `deepagents_code/agent.py`의 `create_cli_agent`에서 정해지며,
# [해설]   이 미들웨어는 HITL(승인) 미들웨어 뒤에 추가된다. after_model은 목록 역순으로
# [해설]   실행되므로 PreToolUse가 승인 라우팅보다 먼저 결정된다(langchain factory.py의
# [해설]   after_model 간선 순서, test_server_pre_tool_node_runs_before_stock_hitl로
# [해설]   확인).
# [해설][주의] 훅 이벤트는 런타임 컨텍스트의 `hooks_snapshot_id`/`hooks_server_events`로
# [해설]   켜진다(`_session_gate`). 설정된 훅이 없으면 interrupt 왕복이 생기지 않는다.
# [해설] 승인 정책과의 접점은 공개 함수
# [해설]   `hook_permission_behavior`/`hook_decided_permission`이며, `auto_mode.py`의
# [해설]   `AutoModeHITLMiddleware.aafter_model`이 이를 읽어 훅이 이미 allow/deny한
# [해설]   호출은 다시 묻지 않는다.
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
    TypeGuard,
    cast,
    override,
)
from uuid import UUID, uuid5

from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    HITLRequest,
    ReviewConfig,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
    TracePolicy,
    hook_config,
    omit_payload,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command, interrupt
from pydantic import ValidationError
from typing_extensions import TypedDict

from deepagents_code.approval_mode import ApprovalMode, coerce_approval_mode
from deepagents_code.hooks.interrupt import (
    build_hook_interrupt_payload,
    parse_hook_resume_value,
)
from deepagents_code.hooks.models.domain import (
    AgentIdentity,
    BaseHookDecision,
    CompactTrigger,
    HookContext,
    HookDecision,
    HookDiagnostic,
    HookEvent,
    HookInvocation,
    PermissionEffect,
    PostToolUseDecision,
    PostToolUseEvent,
    PostToolUseFailureDecision,
    PostToolUseFailureEvent,
    PreCompactDecision,
    PreCompactEvent,
    PreToolUseDecision,
    PreToolUseEvent,
    StopDecision,
    StopEvent,
    SubagentStartDecision,
    SubagentStartEvent,
    SubagentStopDecision,
    SubagentStopEvent,
    ToolCallData,
)
from deepagents_code.hooks.models.transport import HookInvocationRequest
from deepagents_code.hooks.reducer import reduce_hook_results
from deepagents_code.hooks.tools import to_wire_tool_name

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from pathlib import Path

    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages.tool import ToolCall
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from deepagents_code.json_types import JsonObject

# [해설] 훅 요청의 클라이언트 실행 마감 기본값은 600초다. 아래 `_..._STATE_KEY` 상수들은
# [해설]   상태(State)의 비공개 채널 이름이다.
_DEFAULT_DEADLINE = timedelta(seconds=600)
_STOP_STATE_KEY = "_hooks_stop_continuation_count"
_PRE_TOOL_STATE_KEY = "_hooks_pre_tool_outcomes"
_PENDING_POST_TOOL_STATE_KEY = "_hooks_pending_post_tools"
_TASK_TOOL_NAME = "task"
_COMPACT_TOOL_NAME = "compact_conversation"
_INVOCATION_NAMESPACE = UUID("f2896d18-cf2a-4e7d-b11a-d5b10fc0e335")


# [해설][설계] /offload 같은 '그래프 밖 서버 작업'에서는 `interrupt()`를 쓸 수 없어서,
# [해설]   훅 요청을 HTTP 경계까지 올리는 용도의 예외다. `Exception`이 아니라
# [해설]   `BaseException`을 상속한 것은 중간의 `except Exception`이 삼키지 못하게
# [해설]   하려는 의도다.
class HookTransportInterruptError(BaseException):
    """Carry a hook request across a non-graph server operation boundary.

    Derives from `BaseException`, not `Exception`, for the same reason
    `asyncio.CancelledError` does: it is a control signal that must reach the
    HTTP boundary intact. The compaction chain it crosses is lined with broad
    `except Exception` handlers, any of which would otherwise turn a resumable
    hook request into a permanent `"failed"` result.
    """

    def __init__(self, request: HookInvocationRequest) -> None:
        """Initialize the transport interrupt.

        Args:
            request: Hook invocation the client must fulfill.
        """
        super().__init__(str(request.invocation_id))
        self.request = request


logger = logging.getLogger(__name__)

_HOOK_RESPONSES: ContextVar[Mapping[str, object] | None] = ContextVar(
    "deepagents_code_hook_responses",
    default=None,
)


@contextmanager
# [해설] '작업 모드'(그래프 밖)에서 이미 받은 훅 응답을 ContextVar에 넣어 둔다. 작업이
# [해설]   처음부터 다시 실행될 때 답이 있는 호출은 재호출하지 않고 이 응답을 재생하므로
# [해설]   훅이 여러 개여도 끝난다.
def operation_hook_responses(
    responses: Mapping[str, object],
) -> Iterator[None]:
    """Serve hook responses while a server operation replays from the top.

    Args:
        responses: Resume payloads keyed by deterministic hook invocation ID.
    """
    token = _HOOK_RESPONSES.set(responses)
    try:
        yield
    finally:
        _HOOK_RESPONSES.reset(token)


# [해설] ContextVar가 None이면 그래프 모드, 빈 매핑이라도 있으면 작업 모드다. 그래서
# [해설]   truthiness가 아니라 `is not None`으로 판정한다.
def _in_server_operation() -> bool:
    """Report whether hooks are running under a non-graph server operation.

    `None` means graph mode; an empty mapping means operation mode with no
    answers accumulated yet, which is why this cannot be a truthiness check.

    Returns:
        `True` when the caller is inside `operation_hook_responses`.
    """
    return _HOOK_RESPONSES.get() is not None


type PreToolBehavior = Literal["allow", "deny", "none"]
_DEFAULT_DENY_REASON = "Blocked by PreToolUse hook"


# [해설] 사전 훅 결과 타입 두 가지. 거부(deny)는 항상 reason을 가지고,
# [해설]   통과(allow/none)는 훅이 덧붙인 context 문자열만 가진다.
class _PreToolDenied(TypedDict):
    """Outcome for a call a hook refused. A denial always carries a reason."""

    behavior: Literal["deny"]
    reason: str
    context: list[str]


class _PreToolPassed(TypedDict):
    """Outcome for a call a hook allowed or had no opinion on."""

    behavior: Literal["allow", "none"]
    context: list[str]


type _PreToolState = _PreToolDenied | _PreToolPassed

# Maps a tool-call id to the measured execution duration while the call awaits
# its post-execution hook. The value is overloaded as a tombstone: a `None`
# value means "delete this key", not "no duration". This mirrors LangGraph's
# `RemoveMessage` sentinel -- a LangGraph reducer merges an update into the
# channel and returns the whole new value, so removal is expressed by writing
# a `None` sentinel that `_merge_pending_post_tools` pops, rather than by
# omitting the key (a plain merge can only add/overwrite, never remove).
# `_pending_post_tools` filters these tombstones out, so consumers only ever
# see real `int` durations.
type _PendingPostToolState = dict[str, int | None]


# [해설] 리듀서. 값이 None이면 그 키를 지우는 '삭제 표식(tombstone)'으로 쓴다. LangGraph
# [해설]   리듀서는 추가/덮어쓰기만 하므로 소비한 항목을 지우려면 이런 표식이 필요하다.
def _merge_pending_post_tools(
    current: _PendingPostToolState,
    update: _PendingPostToolState,
) -> _PendingPostToolState:
    """Merge pending entries, treating a `None` value as a deletion sentinel.

    LangGraph reducers return the entire new channel value, so writing
    `{call_id: None}` removes `call_id` from the merged result instead of
    storing the `None`. A merge can only add/overwrite keys, so this sentinel
    is the mechanism for removing a consumed entry.

    Args:
        current: Current channel value.
        update: Incoming update; `None` values delete their keys.

    Returns:
        The merged channel value with tombstoned keys removed.
    """
    merged = dict(current)
    for call_id, duration_ms in update.items():
        if duration_ms is None:
            merged.pop(call_id, None)
        else:
            merged[call_id] = duration_ms
    return merged


# [해설] 이 미들웨어 전용 상태 확장. 모두 `PrivateStateAttr`라 공개 입출력 스키마에는 안
# [해설]   보이지만 체크포인트에는 저장되어 interrupt/resume을 넘어 살아남는다.
# [해설][SDK] SubAgentMiddleware가 서브에이전트 결과를 합칠 때 이 필드들을 제거해 부모
# [해설]   턴의 훅 장부가 오염되지 않는다.
class ServerHooksState(AgentState[Any]):
    """Agent state extensions for server-owned hook middleware.

    All fields are per-turn bookkeeping owned by `ServerHooksMiddleware` and
    marked `PrivateStateAttr`: they are omitted from the public graph I/O schema,
    and `SubAgentMiddleware` strips them from subagent result merges.

    `PrivateStateAttr` only omits the fields from the input and output schemas;
    the channels remain checkpointed and visible to graph nodes, so values flow
    across lifecycle boundaries and survive interrupt/resume.

    Note:
        Reducers must be placed *after* `PrivateStateAttr` in the `Annotated`
        metadata. LangGraph only inspects the last metadata entry when detecting
        reducers, so a reducer added before the marker is silently ignored.
    """

    _hooks_session_gate: NotRequired[
        Annotated[_SessionHookGate | None, PrivateStateAttr]
    ]
    """Expected hook context, replaced at the start of each independent turn."""

    _hooks_stop_continuation_count: NotRequired[Annotated[int, PrivateStateAttr]]
    """Stop-hook continuations in the current turn; reset to 0 when the loop ends."""

    _hooks_pre_tool_outcomes: NotRequired[
        Annotated[dict[str, _PreToolState], PrivateStateAttr]
    ]
    """Pre-execution hook verdicts keyed by tool-call id.

    A full snapshot of the *current* turn's calls, not an accumulator: every
    `_after_model` replaces the whole dict (including with `{}`) so stale ids
    cannot survive into a later turn.
    """

    _hooks_pending_post_tools: NotRequired[
        Annotated[
            _PendingPostToolState,
            PrivateStateAttr,
            _merge_pending_post_tools,
        ]
    ]
    """Executed calls awaiting post-tool hooks at a checkpointed boundary."""


# [해설] 이번 턴에 켜진 훅 스냅샷 id와 이벤트 목록. 턴 시작에 체크포인트에 저장하고,
# [해설]   이후 resume마다 같은 값인지 검증한다(`_validated_session_gate`).
class _SessionHookGate(TypedDict):
    snapshot_id: str
    events: frozenset[str]


@dataclass(slots=True)
# [해설] 도구 실행 직전 `wrap_tool_call`이 쓰는 요약. `blocked`가 있으면 도구를 실행하지
# [해설]   않고 그 메시지를 돌려준다.
class _PreToolOutcome:
    """Pre-execution gate result for the tool-call wrapper."""

    blocked: ToolMessage | None = None
    context: tuple[str, ...] = field(default_factory=tuple)


@contextmanager
# [해설] `task` 도구를 실행하는 동안만 자식 실행 설정의 metadata에 호출 id를 넣어,
# [해설]   서브에이전트의 대화 기록을 이 호출과 묶는다. 다른 도구는 그대로 지나간다.
def _subagent_transcript_config(
    call: ToolCallData,
    config: RunnableConfig,
) -> Iterator[None]:
    if call.name != _TASK_TOOL_NAME:
        yield
        return

    from langchain_core.runnables.config import var_child_runnable_config

    from deepagents_code.hooks.transcript import (
        SUBAGENT_TRANSCRIPT_ID_METADATA_KEY,
    )

    metadata = config.get("metadata")
    child_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    child_metadata[SUBAGENT_TRANSCRIPT_ID_METADATA_KEY] = call.id
    child_config: RunnableConfig = {**config, "metadata": child_metadata}
    token = var_child_runnable_config.set(child_config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


# [해설][SDK] langchain `AgentMiddleware` 하위 클래스. 재정의 훅:
# [해설]   before_agent/abefore_agent, before_model/abefore_model,
# [해설]   after_model/aafter_model, wrap_tool_call/awrap_tool_call,
# [해설]   after_agent/aafter_agent.
# [해설] 동기/비동기 메서드는 같은 `_before_model`/`_after_model`/`_after_agent` 구현을
# [해설]   공유하고, `wrap_tool_call` 쌍만 `handler` 호출 방식(await 여부)이 다른 두
# [해설]   벌의 본문이다. 고칠 때 둘을 같이 고쳐야 한다.
# [해설] 서브에이전트 그래프는 `emit_stop=False`로 만들어 도구 래핑은 하되 부모의 Stop
# [해설]   훅은 발화하지 않는다.
class ServerHooksMiddleware(AgentMiddleware[ServerHooksState, ContextT, ResponseT]):
    """Emit server-owned lifecycle events over the hook interrupt transport."""

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    state_schema = ServerHooksState

    def __init__(
        self,
        *,
        cwd: Path,
        default_deadline: timedelta = _DEFAULT_DEADLINE,
        emit_stop: bool = True,
        mcp_tools: Sequence[BaseTool] = (),
    ) -> None:
        """Initialize middleware.

        Args:
            cwd: Session working directory projected into hook context.
            default_deadline: Client execution deadline attached to requests.
            emit_stop: Whether to emit the main-agent `Stop` event from
                `after_agent`. Subagent graphs set this to `False` so they still
                wrap tools without firing parent `Stop` handlers.
            mcp_tools: MCP tools whose server metadata is needed before tool
                execution for compatible hook projection.
        """
        super().__init__()
        self._cwd = cwd
        self._default_deadline = default_deadline
        self._emit_stop = emit_stop
        self._mcp_servers = {
            name: server
            for tool in mcp_tools
            if (name := getattr(tool, "name", None))
            and isinstance(name, str)
            and (server := _mcp_server_from_tool(tool)) is not None
        }

    @override
    # [해설] 턴 시작 훅: 이번 턴의 훅 게이트를 체크포인트에 기록한다. 이후 어떤 훅이
    # [해설]   interrupt를 내도 resume 때 게이트가 바뀌었는지 비교할 수 있다.
    def before_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        """Checkpoint the turn's hook gate before any hook can interrupt.

        Returns:
            The expected hook gate for this independent turn.
        """
        return {"_hooks_session_gate": _session_gate(runtime.context)}

    async def abefore_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        """Checkpoint the turn's hook gate on async graph entry.

        Returns:
            The expected hook gate for this independent turn.
        """
        return self.before_agent(state, runtime)

    # [해설] 모델 호출 직전 훅. 여기서는 모델이 아니라 '이전 도구 결과'를
    # [해설]   후처리한다(PostToolUse, PostToolUseFailure, SubagentStop). 결과가
    # [해설]   체크포인트된 안전한 경계이기 때문이다.
    def before_model(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Run post-execution hooks after tool results are checkpointed.

        Returns:
            State updates for rewritten results and completed hook bookkeeping.
        """
        return self._before_model(state, runtime)

    async def abefore_model(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Run async post-execution hooks at the same safe boundary.

        Returns:
            State updates for rewritten results and completed hook bookkeeping.
        """
        return self._before_model(state, runtime)

    # [해설] 모델 응답 직후 훅: 도구 호출마다 PreCompact/PreToolUse 훅을 돌려
    # [해설]   허용/거부/묻기 결과를 상태에 저장한다. 승인(HITL) 미들웨어보다 먼저
    # [해설]   실행되도록 체인에 배치된다.
    def after_model(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        """Run pre-execution hooks before downstream HITL middleware.

        Returns:
            State update carrying per-tool hook outcomes.
        """
        return self._after_model(state, runtime)

    async def aafter_model(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        """Run the async graph path through the same interrupt sequence.

        Returns:
            State update carrying per-tool hook outcomes.
        """
        return self._after_model(state, runtime)

    # [해설][SDK] langchain `wrap_tool_call` 훅(동기). `handler`는 다음 미들웨어 또는
    # [해설]   실제 도구 실행이다.
    # [해설][흐름] 1) 턴 게이트 검증 2) after_model에서 저장한 사전 훅 결과 확인: 거부면
    # [해설]   실행하지 않고 거부 ToolMessage 반환 3) `task` 도구면 SubagentStart
    # [해설]   훅(차단/컨텍스트 주입)
    # [해설][흐름] 4) 실행 시간을 재면서 `handler` 실행 5) 사전 훅이 준 context 문자열을
    # [해설]   결과 뒤에 덧붙임 6) 사후 훅이 켜져 있으면 결과를 '대기열'에
    # [해설]   기록(`_record_pending_post_tool`)해 다음 before_model에서 처리.
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run pre-tool hooks and record synchronous results for post hooks.

        Returns:
            Tool result with checkpointed post-hook bookkeeping when needed.
        """
        gate = _validated_session_gate(request.state, request.runtime.context)
        call = _tool_call_data(request)
        pre = _pre_tool_outcome(request.state, call)
        context = _hook_context(
            request.runtime.context, request.runtime.config, self._cwd
        )
        if pre.blocked is not None:
            return _append_message_text(pre.blocked, pre.context, call.id)
        started_or_blocked = self._maybe_subagent_start(request, call, context, gate)
        if isinstance(started_or_blocked, ToolMessage):
            return started_or_blocked
        request = started_or_blocked
        started = time.perf_counter()
        with _subagent_transcript_config(call, request.runtime.config):
            result = handler(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        result = _append_message_text(result, pre.context, call.id)
        if _post_tool_boundary_enabled(gate, call):
            return _record_pending_post_tool(result, call.id, duration_ms)
        return result

    # [해설][SDK] langchain `awrap_tool_call`(비동기). 위 `wrap_tool_call`과 흐름이 같고
    # [해설]   `await handler(request)`만 다르다.
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Run pre-tool hooks and record asynchronous results for post hooks.

        Returns:
            Tool result with checkpointed post-hook bookkeeping when needed.
        """
        gate = _validated_session_gate(request.state, request.runtime.context)
        call = _tool_call_data(request)
        pre = _pre_tool_outcome(request.state, call)
        context = _hook_context(
            request.runtime.context, request.runtime.config, self._cwd
        )
        if pre.blocked is not None:
            return _append_message_text(pre.blocked, pre.context, call.id)
        started_or_blocked = self._maybe_subagent_start(request, call, context, gate)
        if isinstance(started_or_blocked, ToolMessage):
            return started_or_blocked
        request = started_or_blocked
        started = time.perf_counter()
        with _subagent_transcript_config(call, request.runtime.config):
            result = await handler(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        result = _append_message_text(result, pre.context, call.id)
        if _post_tool_boundary_enabled(gate, call):
            return _record_pending_post_tool(result, call.id, duration_ms)
        return result

    # [해설][SDK] `hook_config(can_jump_to=["model"])`: 이 훅이 반환값의
    # [해설]   `jump_to="model"`로 그래프를 모델 노드로 되돌릴 수 있음을 langchain에
    # [해설]   선언한다. Stop 훅의 '계속 작업' 기능에 필요하다.
    @hook_config(can_jump_to=["model"])
    def after_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Emit `Stop` when the agent reaches a natural end.

        Returns:
            Optional state update that may jump back to the model.
        """
        return self._after_agent(state, runtime)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async `Stop` emission; mirrors `after_agent`.

        Returns:
            Optional state update that may jump back to the model.
        """
        return self._after_agent(state, runtime)

    # [해설] `task` 도구이고 SubagentStart 이벤트가 켜져 있을 때만 훅을 부른다. 훅이
    # [해설]   중단을 요청하면 거부 메시지를, 컨텍스트를 주면 task의 description 앞에
    # [해설]   붙여 서브에이전트에게 전달한다.
    def _maybe_subagent_start(
        self,
        request: ToolCallRequest,
        call: ToolCallData,
        context: HookContext,
        gate: _SessionHookGate | None,
    ) -> ToolCallRequest | ToolMessage:
        if call.name != _TASK_TOOL_NAME or not _event_enabled(
            gate, HookEvent.SUBAGENT_START
        ):
            return request
        agent = _task_agent_identity(call)
        decision = _invoke_hook(
            context,
            SubagentStartEvent(event=HookEvent.SUBAGENT_START, agent=agent),
            gate=gate,
            config=request.runtime.config,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, SubagentStartDecision)
        if not decision.continue_processing:
            return _denied_tool_message(
                call,
                PermissionEffect(
                    behavior="deny",
                    reason=decision.stop_reason or "Blocked by SubagentStart hook",
                ),
            )
        return _inject_subagent_start_context(request, decision)

    # [해설][흐름] 1) 대기열(`_hooks_pending_post_tools`)이 비어 있으면 아무것도 안 함
    # [해설]   2) 대기열을 '삭제 표식'으로 바꿔 두고 3) 마지막 AI 도구 호출 메시지 뒤의
    # [해설]   ToolMessage를 호출 id로 매칭
    # [해설][흐름] 4) 호출마다 PostToolUse(실패면 PostToolUseFailure) 훅 -> `task`면
    # [해설]   SubagentStop 훅 5) 갱신된 ToolMessage와 비워진 대기열을 상태 업데이트로
    # [해설]   반환. 훅은 결과를 바꿀 수 있어도 ToolMessage 자체는 보존해야 한다(아니면
    # [해설]   TypeError).
    def _before_model(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        gate = _validated_session_gate(state, runtime.context)
        pending = _pending_post_tools(state)
        if not pending:
            return None
        # Construct a new _PendingPostToolState where `duration_ms` is None
        # for each entry. This causes the pending state to be evicted during
        # graph state reconciliation in _merge_pending_post_tools
        completed: _PendingPostToolState = dict.fromkeys(pending)
        messages = state.get("messages", ())
        latest_call_message = _latest_tool_call_message(messages)
        # If there is an extant _PendingPostToolState but no corresponding
        # tool message, mark the _PendingPostToolState as resolved.
        if latest_call_message is None:
            return {_PENDING_POST_TOOL_STATE_KEY: completed}
        message_index, ai_message = latest_call_message
        results = {
            message.tool_call_id: message
            for message in messages[message_index + 1 :]
            if isinstance(message, ToolMessage)
        }
        config = _runtime_hook_config(runtime)
        context = _hook_context(runtime.context, config, self._cwd)
        updates: list[ToolMessage] = []
        for tool_call in ai_message.tool_calls:
            call = _tool_call_data_from_call(
                tool_call,
                mcp_server=self._mcp_servers.get(str(tool_call.get("name") or "")),
            )
            duration_ms = pending.get(call.id)
            result = results.get(call.id)
            if duration_ms is None or result is None:
                # This pending entry has already been consumed, continue
                continue
            updated = self._maybe_post_tool_use(
                call,
                context,
                gate,
                config,
                result,
                duration_ms,
            )
            updated = self._maybe_subagent_stop(
                call,
                context,
                gate,
                config,
                updated,
            )
            if not isinstance(updated, ToolMessage):
                msg = "Post-tool hooks must preserve committed ToolMessage results"
                raise TypeError(msg)
            updates.append(updated)
        state_update: dict[str, Any] = {
            _PENDING_POST_TOOL_STATE_KEY: completed,
        }
        if updates:
            state_update["messages"] = updates
        return state_update

    # [해설][흐름] 1) PreCompact/PreToolUse 둘 다 꺼져 있으면 빈 결과로 종료 2) 마지막
    # [해설]   AI 메시지의 도구 호출을 하나씩 처리
    # [해설][흐름] 3) `compact_conversation` 호출이면 PreCompact 훅(수동/자동 구분).
    # [해설]   중단 요청이면 해당 호출을 deny로 기록 4) PreToolUse 훅 호출. `execute`
    # [해설]   도구의 명령은 훅이 `updated_command`를 주면 바뀐 명령으로 AI 메시지를
    # [해설]   수정(MCP 도구는 제외)
    # [해설][흐름] 5) 훅 결정을 해석: 중단/deny -> 거부(사유 필수), ask ->
    # [해설]   `_ask_permission_via_hitl`로 사람에게 확인(승인되면 allow, 거절이면
    # [해설]   deny), allow -> 허용, 의견 없음 -> none
    # [해설][흐름] 6) 전체 결과 딕셔너리를 '현재 턴 스냅샷'으로 통째로 교체 저장(이전 턴
    # [해설]   id가 남지 않게 함). 이 결과를 `wrap_tool_call`(거부 적용)과 승인
    # [해설]   미들웨어(`hook_permission_behavior`)가 읽는다.
    def _after_model(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        gate = _validated_session_gate(state, runtime.context)
        precompact_enabled = _event_enabled(gate, HookEvent.PRE_COMPACT)
        pretool_enabled = _event_enabled(gate, HookEvent.PRE_TOOL_USE)
        if not precompact_enabled and not pretool_enabled:
            return {_PRE_TOOL_STATE_KEY: {}}
        message = _last_ai_message(state.get("messages", ()))
        if message is None:
            return {_PRE_TOOL_STATE_KEY: {}}
        context = _hook_context(runtime.context, None, self._cwd)
        outcomes: dict[str, _PreToolState] = {}
        tool_calls = [dict(call) for call in message.tool_calls]
        for index, tool_call in enumerate(message.tool_calls):
            call = _tool_call_data_from_call(
                tool_call,
                mcp_server=self._mcp_servers.get(str(tool_call.get("name") or "")),
            )
            behavior: PreToolBehavior = "none"
            reason: str | None = None
            hook_context: list[str] = []
            if precompact_enabled and call.name == _COMPACT_TOOL_NAME:
                trigger = (
                    CompactTrigger.MANUAL
                    if call.args.get("force") is True
                    else CompactTrigger.AUTO
                )
                compact = _invoke_hook(
                    context,
                    PreCompactEvent(event=HookEvent.PRE_COMPACT, trigger=trigger),
                    gate=gate,
                    config=None,
                    deadline=self._default_deadline,
                    logical_event_id=call.id,
                )
                compact = _require_decision(compact, PreCompactDecision)
                if not compact.continue_processing:
                    outcomes[call.id] = {
                        "behavior": "deny",
                        "reason": compact.stop_reason or "Blocked by PreCompact hook",
                        "context": hook_context,
                    }
                    continue
            if pretool_enabled:
                decision = _invoke_hook(
                    context,
                    PreToolUseEvent(event=HookEvent.PRE_TOOL_USE, call=call),
                    gate=gate,
                    config=None,
                    deadline=self._default_deadline,
                )
                decision = _require_decision(decision, PreToolUseDecision)
                if (
                    decision.updated_command is not None
                    and decision.continue_processing
                    and decision.permission.behavior != "deny"
                    and call.name == "execute"
                    and call.mcp_server is None
                ):
                    call = call.model_copy(
                        update={
                            "args": {**call.args, "command": decision.updated_command}
                        }
                    )
                    tool_calls[index] = {**tool_call, "args": dict(call.args)}
                permission = decision.permission
                hook_context.extend(decision.context)
                if not decision.continue_processing or permission.behavior == "deny":
                    behavior = "deny"
                    reason = (
                        permission.reason
                        or decision.stop_reason
                        or _DEFAULT_DENY_REASON
                    )
                elif permission.behavior == "ask":
                    blocked = _ask_permission_via_hitl(call, permission)
                    if blocked is None:
                        behavior = "allow"
                    else:
                        behavior = "deny"
                        blocked_content = blocked.content
                        reason = (
                            blocked_content
                            if isinstance(blocked_content, str)
                            else str(blocked_content)
                        )
                elif permission.behavior == "allow":
                    behavior = "allow"
            if behavior == "deny":
                outcomes[call.id] = {
                    "behavior": "deny",
                    # Every deny path above resolves a reason; the guard keeps the
                    # "a denial always explains itself" invariant checkable here.
                    "reason": reason if reason is not None else _DEFAULT_DENY_REASON,
                    "context": hook_context,
                }
            else:
                outcomes[call.id] = {
                    "behavior": behavior,
                    "context": hook_context,
                }
        update: dict[str, Any] = {_PRE_TOOL_STATE_KEY: outcomes}
        if tool_calls != message.tool_calls:
            update["messages"] = [message.model_copy(update={"tool_calls": tool_calls})]
        return update

    # [해설] 도구 결과가 오류인지에 따라 PostToolUseFailure 또는 PostToolUse 이벤트를
    # [해설]   고른다. 해당 이벤트가 꺼져 있으면 결과를 그대로 반환하고, 훅의
    # [해설]   feedback/context는 결과 텍스트 뒤에 덧붙는다.
    def _maybe_post_tool_use(
        self,
        call: ToolCallData,
        context: HookContext,
        gate: _SessionHookGate | None,
        config: Mapping[str, Any] | None,
        result: ToolMessage | Command[Any],
        duration_ms: int,
    ) -> ToolMessage | Command[Any]:
        error = _tool_result_error(result, call)
        event = (
            HookEvent.POST_TOOL_USE_FAILURE
            if error is not None
            else HookEvent.POST_TOOL_USE
        )
        if not _event_enabled(gate, event):
            return result
        if error is not None:
            hook_event = PostToolUseFailureEvent(
                event=HookEvent.POST_TOOL_USE_FAILURE,
                call=call,
                error=error,
                duration_ms=duration_ms,
            )
            decision_type = PostToolUseFailureDecision
        else:
            hook_event = PostToolUseEvent.from_tool_result(
                result,
                call=call,
                duration_ms=duration_ms,
            )
            decision_type = PostToolUseDecision
        decision = _require_decision(
            _invoke_hook(
                context,
                hook_event,
                gate=gate,
                config=config,
                deadline=self._default_deadline,
            ),
            decision_type,
        )
        return _apply_post_tool_use(result, decision, call.id)

    # [해설] `task` 도구 결과에 대해 SubagentStop 훅을 호출하고, 훅이 준 context를 결과
    # [해설]   뒤에 덧붙인다. 이 훅은 결과를 막지는 않는다(여기서는 context만 사용).
    def _maybe_subagent_stop(
        self,
        call: ToolCallData,
        context: HookContext,
        gate: _SessionHookGate | None,
        config: Mapping[str, Any] | None,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        if call.name != _TASK_TOOL_NAME or not _event_enabled(
            gate, HookEvent.SUBAGENT_STOP
        ):
            return result
        agent = _task_agent_identity(call)
        decision = _invoke_hook(
            context,
            SubagentStopEvent(
                event=HookEvent.SUBAGENT_STOP,
                agent=agent,
                continuation_count=0,
                last_assistant_message=_tool_result_text(result, call.id),
            ),
            gate=gate,
            config=config,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, SubagentStopDecision)
        return _apply_subagent_stop(result, decision, call.id)

    # [해설][흐름] Stop 훅: 1) `emit_stop`이 꺼졌거나(서브에이전트) Stop 이벤트가 꺼져
    # [해설]   있으면 건너뜀 2) 지금까지의 연속 횟수와 마지막 어시스턴트 메시지를 담아
    # [해설]   Stop 이벤트 발화
    # [해설][흐름] 3) 훅이 계속(loop)을 원하지 않으면 카운터를 0으로 리셋하고 종료 4)
    # [해설]   원하면 피드백을 HumanMessage로 넣고 `jump_to="model"`로 모델에게 돌아가
    # [해설]   카운터를 1 올린다.
    def _after_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        gate = _validated_session_gate(state, runtime.context)
        if not self._emit_stop:
            return None
        if not _event_enabled(gate, HookEvent.STOP):
            return None
        continuation = int(state.get(_STOP_STATE_KEY, 0) or 0)
        context = _hook_context(runtime.context, None, self._cwd)
        decision = _invoke_hook(
            context,
            StopEvent(
                event=HookEvent.STOP,
                continuation_count=continuation,
                last_assistant_message=_last_assistant_text(state.get("messages", ())),
            ),
            gate=gate,
            config=None,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, StopDecision)
        if not decision.continue_processing or not decision.continue_loop:
            # Reset so a later independent turn does not inherit the count.
            if continuation:
                return {_STOP_STATE_KEY: 0}
            return None
        feedback = "\n".join(decision.feedback).strip() or (
            decision.stop_reason or "Continue working."
        )
        return {
            "messages": [HumanMessage(content=feedback)],
            "jump_to": "model",
            _STOP_STATE_KEY: continuation + 1,
        }


# [해설] 클라이언트가 돌려준 결정이 기대한 타입(예: PreToolUseDecision)인지 확인하는
# [해설]   타입 가드. 다르면 TypeError.
def _require_decision[DecisionT: BaseHookDecision](
    decision: HookDecision,
    expected: type[DecisionT],
) -> DecisionT:
    if not isinstance(decision, expected):
        msg = f"Expected {expected.__name__}, got {type(decision).__name__}"
        raise TypeError(msg)
    return decision


# [해설] 런타임 컨텍스트에서 `hooks_snapshot_id`와 `hooks_server_events`를 읽어 게이트를
# [해설]   만든다. 둘 중 하나라도 없거나 비어 있으면 None이며, 그러면 서버 훅은 전부
# [해설]   꺼진 것으로 동작한다.
def _session_gate(runtime_context: object) -> _SessionHookGate | None:
    fields = _context_mapping(runtime_context)
    snapshot_id = fields.get("hooks_snapshot_id")
    events = fields.get("hooks_server_events")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        return None
    if not isinstance(events, list) or not events:
        return None
    return {
        "snapshot_id": snapshot_id,
        "events": frozenset(str(item) for item in events),
    }


# [해설][주의] 턴 도중 resume에서 훅 컨텍스트가 달라지면 RuntimeError다. 스냅샷이 바뀌면
# [해설]   이미 받은 훅 응답을 새 훅 설정에 잘못 적용할 수 있기 때문이다.
def _validated_session_gate(
    state: Mapping[str, Any], runtime_context: object
) -> _SessionHookGate | None:
    gate = _session_gate(runtime_context)
    expected = state.get("_hooks_session_gate")
    if expected is not None and gate != expected:
        msg = (
            "Server hook context changed during an active turn. Pass the same "
            "hooks_snapshot_id and hooks_server_events on every resume."
        )
        raise RuntimeError(msg)
    return gate


# [해설] 게이트가 있고 이벤트 이름이 켜진 목록에 있을 때만 참. 모든 훅 발화의 입구
# [해설]   조건이다.
def _event_enabled(gate: _SessionHookGate | None, event: HookEvent) -> bool:
    return gate is not None and event.value in gate["events"]


def _post_tool_boundary_enabled(
    gate: _SessionHookGate | None,
    call: ToolCallData,
) -> bool:
    return (
        _event_enabled(gate, HookEvent.POST_TOOL_USE)
        or _event_enabled(gate, HookEvent.POST_TOOL_USE_FAILURE)
        or (
            call.name == _TASK_TOOL_NAME
            and _event_enabled(gate, HookEvent.SUBAGENT_STOP)
        )
    )


def _pending_post_tools(state: ServerHooksState) -> dict[str, int]:
    raw = state.get(_PENDING_POST_TOOL_STATE_KEY)
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(call_id): duration_ms
        for call_id, duration_ms in raw.items()
        if isinstance(duration_ms, int) and not isinstance(duration_ms, bool)
    }


# [해설] 승인 정책과의 접점 (1). 훅이 명시적으로 allow/deny를 정했으면 True이고, 그러면
# [해설]   기본 승인 흐름이 같은 호출을 다시 묻지 않는다.
def hook_decided_permission(state: object, tool_call_id: str) -> bool:
    """Report whether a pre-execution hook already settled permission for a call.

    Args:
        state: Agent state carrying the current turn's hook outcomes.
        tool_call_id: Tool call to look up.

    Returns:
        `True` when a hook explicitly allowed or denied the call, so stock
        approval flows must not prompt again. `False` when no hook ran, the hook
        expressed no opinion, or no outcome was recorded -- in every one of those
        cases normal approval still applies.
    """
    return hook_permission_behavior(state, tool_call_id) is not None


# [해설] 승인 정책과의 접점 (2). 상태에서 호출 id의 사전 훅 결과를 읽어
# [해설]   'allow'/'deny'/None을 돌려준다. `auto_mode.py`의
# [해설]   `AutoModeHITLMiddleware.aafter_model`이 이 값으로 훅이 정한 호출을
# [해설]   사람/분류기 검토에서 빼낸다.
def hook_permission_behavior(
    state: object, tool_call_id: str
) -> Literal["allow", "deny"] | None:
    """Return the explicit pre-execution hook permission for a call.

    Args:
        state: Agent state carrying the current turn's hook outcomes.
        tool_call_id: Tool call to look up.

    Returns:
        The hook's explicit `allow` or `deny`, or `None` when normal approval
            routing still decides permission.
    """
    outcome = _pre_tool_state(state, tool_call_id)
    if outcome is None:
        return None
    behavior = outcome.get("behavior")
    if behavior == "allow":
        return "allow"
    if behavior == "deny":
        return "deny"
    return None


def _pre_tool_state(state: object, tool_call_id: str) -> Mapping[str, object] | None:
    if not isinstance(state, Mapping):
        return None
    raw = state.get(_PRE_TOOL_STATE_KEY)
    if not isinstance(raw, Mapping):
        return None
    outcome = raw.get(tool_call_id)
    if not isinstance(outcome, Mapping):
        return None
    return {str(key): value for key, value in outcome.items()}


# [해설] 저장된 사전 훅 결과를 `wrap_tool_call`용으로 변환한다. 거부면 `blocked`에 거부
# [해설]   ToolMessage를, 아니면 덧붙일 context만 담는다.
def _pre_tool_outcome(state: object, call: ToolCallData) -> _PreToolOutcome:
    outcome = _pre_tool_state(state, call.id)
    if outcome is None:
        return _PreToolOutcome()
    raw_context = outcome.get("context")
    context = (
        tuple(item for item in raw_context if isinstance(item, str))
        if isinstance(raw_context, Sequence) and not isinstance(raw_context, str)
        else ()
    )
    if outcome.get("behavior") != "deny":
        return _PreToolOutcome(context=context)
    raw_reason = outcome.get("reason")
    reason = raw_reason if isinstance(raw_reason, str) else None
    return _PreToolOutcome(
        blocked=_denied_tool_message(
            call,
            PermissionEffect(behavior="deny", reason=reason),
        ),
        context=context,
    )


# [해설][흐름] 훅 한 번 호출: 1) 게이트가 없으면 RuntimeError 2) 결정론적 invocation_id
# [해설]   생성(`_invocation_id`) 3) `HookInvocationRequest` 조립(마감시간 포함)
# [해설][흐름] 4) 그래프 모드면 `interrupt(build_hook_interrupt_payload(request))`로
# [해설]   클라이언트에 넘기고, 작업 모드면 저장된 응답을 찾거나
# [해설]   `HookTransportInterruptError`를 올림 5) resume 값을
# [해설]   `parse_hook_resume_value`로 해석, 모양이 틀리면 경고 후 '결정 없음'으로 처리
# [해설][설계] invocation_id를 입력(thread, snapshot, prompt, 이벤트, 논리 이벤트
# [해설]   id)으로 uuid5 생성하기 때문에 노드가 재실행되어도 같은 요청은 같은 id가 되어
# [해설]   응답이 중복 없이 재생된다.
def _invoke_hook(
    context: HookContext,
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | PostToolUseFailureEvent
        | PreCompactEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
    *,
    gate: _SessionHookGate | None,
    config: Mapping[str, Any] | None,
    deadline: timedelta,
    logical_event_id: str | None = None,
) -> HookDecision:
    if gate is None:
        msg = "hooks_snapshot_id is required to emit server-owned hook events"
        raise RuntimeError(msg)
    run_id = _run_id(config, context.thread_id)
    invocation_id = _invocation_id(
        snapshot_id=gate["snapshot_id"],
        context=context,
        event=event,
        logical_event_id=logical_event_id,
    )
    request = HookInvocationRequest(
        protocol_version=1,
        invocation_id=invocation_id,
        snapshot_id=gate["snapshot_id"],
        run_id=run_id,
        invocation=HookInvocation(context=context, event=event),
        deadline=datetime.now(UTC) + deadline,
    )
    operation_responses = _HOOK_RESPONSES.get()
    if operation_responses is None:
        raw = interrupt(build_hook_interrupt_payload(request))
    else:
        # Operation mode: `interrupt()` is unusable outside a Pregel task, so a
        # request the client has not answered yet is raised out to the HTTP
        # boundary instead. Because the operation re-executes from the top on
        # every resume round, an already-answered invocation is replayed from
        # this mapping rather than re-invoked -- that is what makes an operation
        # with several hooks terminate instead of looping forever.
        key = str(request.invocation_id)
        if key not in operation_responses:
            raise HookTransportInterruptError(request)
        raw = operation_responses[key]
    try:
        response = parse_hook_resume_value(
            raw,
            invocation_id=request.invocation_id,
            snapshot_id=request.snapshot_id,
        )
    except ValidationError:
        # Only shape errors degrade to a neutral decision. A plain `ValueError`
        # means the client answered a different request, so it stays fatal.
        #
        # Log it too: the diagnostic is only rendered by the client-side hook
        # presenter, and the offload operation reads just the pre-tool channel
        # from this update, so on that path the diagnostic is dropped and the
        # hook is silently ignored.
        logger.warning(
            "Malformed hook resume value for invocation %s; treating it as no decision",
            request.invocation_id,
            exc_info=True,
        )
        diagnostic = HookDiagnostic(
            code="invalid_resume",
            severity="warning",
            message="Malformed hook resume value; treating it as no decision",
        )
        return reduce_hook_results(request.invocation, (), diagnostics=(diagnostic,))
    return response.decision


# [해설] 훅에 전달할 컨텍스트(스레드 id, cwd, prompt id, 승인 모드). 승인 모드는
# [해설]   `coerce_approval_mode`를 거치므로 잘못된 값이면 manual로 닫힌다.
def _hook_context(
    runtime_context: object,
    config: Mapping[str, Any] | None,
    cwd: Path,
) -> HookContext:
    fields = _context_mapping(runtime_context)
    thread_id = fields.get("thread_id") or _config_thread_id(config) or "unknown"
    if not isinstance(thread_id, str):
        thread_id = "unknown"
    approval = coerce_approval_mode(fields.get("approval_mode", "manual"))
    prompt_raw = fields.get("prompt_id")
    prompt_id = UUID(prompt_raw) if isinstance(prompt_raw, str) and prompt_raw else None
    return HookContext(
        thread_id=thread_id,
        cwd=cwd,
        prompt_id=prompt_id,
        approval_mode=(
            approval if isinstance(approval, ApprovalMode) else ApprovalMode.MANUAL
        ),
    )


def _context_mapping(runtime_context: object) -> dict[str, Any]:
    """Project LangGraph run context (dataclass or mapping) into a plain dict.

    In-process graphs coerce `context=` into `CLIContextSchema`; RemoteGraph
    delivers a plain mapping. Both shapes are accepted here.

    Returns:
        A shallow string-keyed dict of the hook-relevant context fields.
    """
    if runtime_context is None:
        return {}
    if isinstance(runtime_context, Mapping):
        return {str(key): value for key, value in runtime_context.items()}
    result: dict[str, Any] = {}
    for key in (
        "hooks_snapshot_id",
        "hooks_server_events",
        "thread_id",
        "approval_mode",
        "prompt_id",
    ):
        value = getattr(runtime_context, key, None)
        if value is not None:
            result[key] = value
    return result


def _runtime_hook_config(runtime: Runtime[Any]) -> dict[str, Any] | None:
    info = runtime.execution_info
    if info is None:
        return None
    configurable = {
        key: value
        for key, value in (
            ("run_id", info.run_id),
            ("thread_id", info.thread_id),
        )
        if value
    }
    return {"configurable": configurable} if configurable else None


def _run_id(config: Mapping[str, Any] | None, thread_id: str) -> str:
    if isinstance(config, Mapping):
        configurable = config.get("configurable")
        if isinstance(configurable, Mapping):
            for key in ("run_id", "thread_id"):
                value = configurable.get(key)
                if isinstance(value, UUID):
                    return str(value)
                if isinstance(value, str) and value:
                    return value
    return thread_id


# [해설] 요청 식별 정보를 JSON으로 정렬 직렬화한 뒤 uuid5로 해시한 결정론적 id.
def _invocation_id(
    *,
    snapshot_id: str,
    context: HookContext,
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | PostToolUseFailureEvent
        | PreCompactEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
    logical_event_id: str | None = None,
) -> UUID:
    identity = {
        "thread_id": context.thread_id,
        "snapshot_id": snapshot_id,
        "prompt_id": str(context.prompt_id) if context.prompt_id is not None else "",
        "event": event.event.value,
        "logical_event": _logical_event_identity(
            event,
            logical_event_id=logical_event_id,
        ),
    }
    return uuid5(
        _INVOCATION_NAMESPACE,
        json.dumps(identity, sort_keys=True, separators=(",", ":")),
    )


# [해설] 이벤트 종류별 '논리적 이벤트' 식별자: 도구 이벤트는 호출 id, PreCompact는
# [해설]   전달받은 id, SubagentStart/Stop은 서브에이전트 id(+횟수), Stop은 연속 횟수와
# [해설]   마지막 메시지 해시.
def _logical_event_identity(
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | PostToolUseFailureEvent
        | PreCompactEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
    *,
    logical_event_id: str | None = None,
) -> str:
    if isinstance(
        event,
        PreToolUseEvent | PostToolUseEvent | PostToolUseFailureEvent,
    ):
        return event.call.id
    if isinstance(event, PreCompactEvent):
        if logical_event_id:
            return logical_event_id
        msg = "PreCompact requires a stable tool-call identity"
        raise ValueError(msg)
    if isinstance(event, SubagentStartEvent):
        return event.agent.id
    if isinstance(event, SubagentStopEvent):
        return f"{event.agent.id}:{event.continuation_count}"
    message_hash = hashlib.sha256(event.last_assistant_message.encode()).hexdigest()
    return f"{event.continuation_count}:{message_hash}"


def _config_thread_id(config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return None
    value = configurable.get("thread_id")
    return value if isinstance(value, str) and value else None


def _tool_call_data(request: ToolCallRequest) -> ToolCallData:
    return _tool_call_data_from_call(
        request.tool_call,
        mcp_server=_mcp_server_from_tool(request.tool),
    )


def _tool_call_data_from_call(
    tool_call: Mapping[str, object],
    *,
    mcp_server: str | None,
) -> ToolCallData:
    raw_args = tool_call.get("args")
    args: dict[str, Any]
    if isinstance(raw_args, dict):
        args = {str(key): value for key, value in raw_args.items()}
    else:
        args = {}
    return ToolCallData(
        id=str(tool_call.get("id") or ""),
        name=str(tool_call.get("name") or ""),
        args=cast("JsonObject", args),
        mcp_server=mcp_server,
    )


def _mcp_server_from_tool(tool: object | None) -> str | None:
    if tool is None:
        return None
    metadata = getattr(tool, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    for key in ("mcp_server", "mcp_server_name", "server_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# [해설] 훅 거부를 모델이 읽을 수 있는 오류 ToolMessage로 바꾼다. 도구는 실행되지 않고
# [해설]   이 메시지가 결과가 된다.
def _denied_tool_message(
    call: ToolCallData,
    permission: PermissionEffect,
) -> ToolMessage:
    reason = permission.reason or "Blocked by PreToolUse hook"
    wire_name = to_wire_tool_name(call.name, mcp_server=call.mcp_server)
    return ToolMessage(
        content=f"{wire_name} blocked by hook: {reason}",
        name=call.name,
        tool_call_id=call.id,
        status="error",
    )


# [해설][흐름] PreToolUse 훅이 `ask`를 돌려줬을 때: 1) 그래프 밖 작업 모드면 물을 수
# [해설]   없으므로 즉시 거부(사유 안내) 2) 아니면 표준 HITL 요청(`HITLRequest`,
# [해설]   approve/reject)으로 `interrupt`
# [해설][흐름] 3) 응답의 첫 결정이 approve면 None(진행), 응답이 없거나 reject면 거부
# [해설]   메시지. 사용자가 거절 메시지를 줬으면 그것이 사유가 된다.
# [해설] 같은 승인 UI를 재사용하므로 사용자 입장에서는 일반 승인 프롬프트와 동일하게
# [해설]   보인다.
def _ask_permission_via_hitl(
    call: ToolCallData,
    permission: PermissionEffect,
) -> ToolMessage | None:
    """Escalate PreToolUse `ask` through the existing HITL interrupt channel.

    Returns:
        A deny ToolMessage when the user rejects, otherwise `None` to proceed.
    """
    if _in_server_operation():
        # `interrupt()` is only usable inside a Pregel task: it reaches into the
        # run's scratchpad, which a server operation's fabricated config has no
        # equivalent of. Deny with an actionable reason instead of raising a
        # `KeyError` on an internal LangGraph config key. The operation
        # transport carries hook *invocations*, not HITL review requests, so
        # there is no channel to prompt the user on here.
        return _denied_tool_message(
            call,
            PermissionEffect(
                behavior="deny",
                reason=(
                    f"PreToolUse returned `ask` for {call.name}, which cannot "
                    "prompt for approval during a server-side operation such as "
                    "/offload. Return `allow` or `deny` for this tool instead."
                ),
            ),
        )
    description = permission.reason or "PreToolUse hook requested approval"
    response = interrupt(
        HITLRequest(
            action_requests=[
                ActionRequest(
                    name=call.name,
                    args=dict(call.args),
                    description=description,
                )
            ],
            review_configs=[
                ReviewConfig(
                    action_name=call.name,
                    allowed_decisions=["approve", "reject"],
                )
            ],
        )
    )
    decisions: Sequence[Any]
    if isinstance(response, Mapping):
        raw = response.get("decisions", ())
        decisions = raw if isinstance(raw, Sequence) else ()
    else:
        decisions = ()
    if not decisions:
        return _denied_tool_message(
            call,
            PermissionEffect(
                behavior="deny",
                reason="PreToolUse ask was not answered",
            ),
        )
    first = decisions[0]
    decision_type = first.get("type") if isinstance(first, Mapping) else None
    if decision_type != "approve":
        reject_message = None
        if isinstance(first, Mapping):
            raw_message = first.get("message")
            if isinstance(raw_message, str) and raw_message:
                reject_message = raw_message
        return _denied_tool_message(
            call,
            PermissionEffect(
                behavior="deny",
                reason=reject_message or description,
            ),
        )
    return None


# [해설] 도구 결과를 `Command(update=...)`로 감싸 결과 메시지와 '사후 훅 대기 항목(호출
# [해설]   id -> 실행 시간 ms)'을 한 번에 체크포인트한다. 결과가 이미 Command이면 그
# [해설]   update에 합친다.
def _record_pending_post_tool(
    result: ToolMessage | Command[Any],
    call_id: str,
    duration_ms: int,
) -> Command[Any]:
    pending = {_PENDING_POST_TOOL_STATE_KEY: {call_id: duration_ms}}
    if isinstance(result, ToolMessage):
        return Command(update={"messages": [result], **pending})
    update = result.update
    if not isinstance(update, Mapping):
        return result
    return replace(result, update={**update, **pending})


# [해설] 사전 훅이 준 context 문자열들을 결과 텍스트 뒤에 이어 붙인다(없으면 그대로).
def _append_message_text(
    result: ToolMessage | Command[Any],
    parts: Sequence[str],
    call_id: str,
) -> ToolMessage | Command[Any]:
    if not parts:
        return result
    return _append_tool_result_text(result, "\n".join(parts), call_id)


# [해설] PostToolUse 훅의 feedback, context, (중단 시) stop_reason을 결과 뒤에 덧붙인다.
# [해설]   결과 본문을 지우거나 바꾸지는 않는다.
def _apply_post_tool_use(
    result: ToolMessage | Command[Any],
    decision: PostToolUseDecision | PostToolUseFailureDecision,
    call_id: str,
) -> ToolMessage | Command[Any]:
    extras: list[str] = []
    if decision.feedback:
        extras.append("\n".join(decision.feedback))
    if decision.context:
        extras.append("\n".join(decision.context))
    if decision.stop_reason and not decision.continue_processing:
        extras.append(decision.stop_reason)
    if not extras:
        return result
    return _append_tool_result_text(
        result,
        "\n\n".join(part for part in extras if part),
        call_id,
    )


def _apply_subagent_stop(
    result: ToolMessage | Command[Any],
    decision: SubagentStopDecision,
    call_id: str,
) -> ToolMessage | Command[Any]:
    if not decision.context:
        return result
    return _append_tool_result_text(result, "\n".join(decision.context), call_id)


def _append_tool_result_text(
    result: ToolMessage | Command[Any],
    suffix: str,
    call_id: str,
) -> ToolMessage | Command[Any]:
    if isinstance(result, ToolMessage):
        return _merge_tool_message_content(result, suffix)
    update = result.update
    if not isinstance(update, Mapping):
        return result
    changed = False
    messages: list[object] = []
    for message in _command_messages(result):
        if _is_call_result(message, call_id):
            messages.append(_merge_tool_message_content(message, suffix))
            changed = True
        else:
            messages.append(message)
    if not changed:
        return result
    return replace(result, update={**update, "messages": messages})


def _tool_result_error(
    result: ToolMessage | Command[Any],
    call: ToolCallData,
) -> str | None:
    messages = (
        [result] if isinstance(result, ToolMessage) else _command_messages(result)
    )
    for message in messages:
        if not _is_call_result(message, call.id):
            continue
        if message.status == "error":
            return _tool_result_text(result, call.id)
        artifact = message.artifact
        if call.name != "execute" or not isinstance(artifact, Mapping):
            continue
        exit_code = artifact.get("exit_code")
        if (
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code != 0
        ):
            return f"Command exited with non-zero status code {exit_code}"
    return None


def _command_messages(result: Command[Any]) -> Sequence[object]:
    """Return the `messages` list carried by a `Command` update.

    Returns:
        The update's messages, or an empty sequence when absent or malformed.
    """
    update = result.update
    if not isinstance(update, Mapping):
        return ()
    messages = update.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str):
        return ()
    return messages


def _is_call_result(message: object, call_id: str) -> TypeGuard[ToolMessage]:
    """Check whether a message is the `ToolMessage` for the in-flight call.

    A `Command` update may carry results for several calls, so hook context must
    only read from and write to the one this wrapper is handling.

    Returns:
        `True` when the message answers `call_id`.
    """
    return isinstance(message, ToolMessage) and message.tool_call_id == call_id


def _merge_tool_message_content(result: ToolMessage, suffix: str) -> ToolMessage:
    if not suffix:
        return result
    content = result.content
    if isinstance(content, str):
        merged = f"{content}\n\n{suffix}" if content else suffix
    # Preserve structured content blocks; append a text block.
    elif isinstance(content, list):
        merged = [*content, {"type": "text", "text": suffix}]
    else:
        merged = f"{content!s}\n\n{suffix}"
    return result.model_copy(update={"content": merged})


# [해설] SubagentStart 훅의 context를 `task` 호출의 `description` 앞에 붙여 새 호출로
# [해설]   교체한다(`request.override`). 원본 호출은 수정하지 않는다.
def _inject_subagent_start_context(
    request: ToolCallRequest,
    decision: SubagentStartDecision,
) -> ToolCallRequest:
    if not decision.context:
        return request

    original = request.tool_call
    raw_args = original.get("args")
    args: dict[str, Any]
    if isinstance(raw_args, dict):
        args = {str(key): value for key, value in raw_args.items()}
    else:
        args = {}
    description = args.get("description")
    prefix = "\n".join(decision.context)
    if isinstance(description, str) and description:
        args["description"] = f"{prefix}\n\n{description}"
    else:
        args["description"] = prefix
    tool_call = cast(
        "ToolCall",
        {
            "name": str(original.get("name") or ""),
            "args": args,
            "id": original.get("id"),
            "type": "tool_call",
        },
    )
    return request.override(tool_call=tool_call)


def _task_agent_identity(call: ToolCallData) -> AgentIdentity:
    name = call.args.get("subagent_type")
    if not isinstance(name, str) or not name:
        name = "unknown"
    return AgentIdentity(id=call.id or name, name=name)


def _tool_result_text(result: ToolMessage | Command[Any], call_id: str) -> str:
    if isinstance(result, ToolMessage):
        content = result.content
        return content if isinstance(content, str) else str(content)
    return "\n".join(
        str(message.content)
        for message in _command_messages(result)
        if _is_call_result(message, call_id)
    )


def _latest_tool_call_message(
    messages: Sequence[Any],
) -> tuple[int, AIMessage] | None:
    return next(
        (
            (index, message)
            for index, message in reversed(list(enumerate(messages)))
            if isinstance(message, AIMessage) and message.tool_calls
        ),
        None,
    )


def _last_ai_message(messages: Sequence[Any]) -> AIMessage | None:
    return next(
        (message for message in reversed(messages) if isinstance(message, AIMessage)),
        None,
    )


def _last_assistant_text(messages: Sequence[Any]) -> str:
    message = _last_ai_message(messages)
    if message is None:
        return ""
    content = message.content
    return content if isinstance(content, str) else str(content)
