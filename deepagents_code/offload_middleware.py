"""CLI-specific conversation compaction middleware."""

# [해설] 모듈 역할: dcode 의 '컨텍스트 오프로드(압축)' 구현. 대화가 길어지면 오래된
# [해설]   메시지를 요약으로 바꾸고 원문은 아카이브 파일로 내보낸다. 자동 압축, 모델이
# [해설]   부르는 compact_conversation 도구, 사용자의 /offload 명령(서버 연산) 세 경로를
# [해설]   모두 다룬다.
# [해설] 실행 프로세스: 모두 LangGraph 서버 프로세스 안에서 돈다.
# [해설]   create_cli_agent(agent.py)가 _create_cli_compaction_middleware 로
# [해설]   CLICompactionMiddleware 를 만들어 스택에 넣고, OffloadOperation 은
# [해설]   attach_offload_operation 으로 CompositeBackend 에 '게시'되어 서버 HTTP
# [해설]   라우트(offload_api.py, 이 레인 밖)가 호출한다.
# [해설] 주요 심볼: CLICompactionMiddleware(자동/도구/강제 압축), OffloadOperation(서버
# [해설]   /offload 실행: 훅 -> 계획 -> 결과), _PendingArchive/_ArchiveAppend(아카이브
# [해설]   쓰기와 롤백), _LazySummaryModel(요약 전용 모델의 지연 생성),
# [해설]   _ArchiveReadGuard(읽기 실패 시 덮어쓰기 방지).
# [해설][SDK] 엔진은 SDK deepagents.middleware.summarization 의
# [해설]   SummarizationMiddleware/SummarizationToolMiddleware 이다. 이 파일은 그것을
# [해설]   상속/감싸서 (1) PreCompact 훅 (2) 런타임 모델 변경 반영 (3) 요약 모델 재시도
# [해설]   정책 (4) 서버 전용 강제 압축을 추가한다.
# [해설][주의] 구현이 SDK 의 비공개 속성(_lc_helper, _summary_model, _create_summary,
# [해설]   _backend 등)을 직접 바꾼다. _require_helper_slot 이 슬롯이 사라지면 조용히
# [해설]   무시되는 대신 AttributeError 로 크게 실패하게 하지만, SDK 업그레이드 때 가장
# [해설]   먼저 점검해야 할 파일이다.
# [해설] 저장 위치 규칙(어디에 파일을 쓰는가)은 offload.py 가, 백엔드
# [해설]   라우팅(conversation_history 경로)은 agent.py 의 CompositeBackend 조립부가
# [해설]   정한다.
# [해설] 이 파일은 크게 (1) 요약 모델 설치 헬퍼 (2) 결과/상태 타입 (3) 아카이브
# [해설]   쓰기/롤백 (4) CLICompactionMiddleware (5) 팩토리 (6) OffloadOperation 순서로
# [해설]   읽으면 된다.
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, NotRequired, Protocol, cast
from uuid import NAMESPACE_URL, uuid4, uuid5
from weakref import WeakValueDictionary

from deepagents.backends.protocol import FILE_NOT_FOUND
from deepagents.middleware.summarization import (
    SummarizationState,
    SummarizationToolMiddleware,
    create_summarization_middleware,
    create_summarization_tool_middleware,
)
from langchain.agents.middleware.summarization import (
    _get_approximate_token_counter,  # noqa: PLC2701  # keep summary trimming aligned with LangChain
)
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # inspected for runtime injection
)
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from langgraph.types import Command  # noqa: TC002  # inspected for tool schema
from typing_extensions import TypedDict

from deepagents_code._cli_context import (
    INHERIT_SUMMARIZATION_MODEL,
    CLIContextSchema,
)
from deepagents_code.cost_tracking import CostState
from deepagents_code.hooks.models.domain import (
    CompactTrigger,
    HookEvent,
    PreCompactDecision,
    PreCompactEvent,
)
from deepagents_code.hooks.server_middleware import (
    _DEFAULT_DEADLINE,
    _PRE_TOOL_STATE_KEY,
    HookTransportInterruptError,
    _event_enabled,
    _hook_context,
    _invoke_hook,
    _require_decision,
    _session_gate,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.protocol import (
        BackendProtocol,
        EditResult,
        FileDownloadResponse,
        WriteResult,
    )
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
    )
    from langchain_core.messages import AnyMessage
    from langgraph.runtime import Runtime

    from deepagents_code.hooks.server_middleware import ServerHooksMiddleware

logger = logging.getLogger(__name__)

# [해설] 요약 모델의 입력 한도를 알 수 없을 때 쓰는 보수적 기본값(토큰).
# [해설]   _summary_trim_limit 이 사용한다.
_SUMMARY_TRIM_FALLBACK = 4_000
"""Conservative history budget when a summary model exposes no context limit."""


# [해설] 요약용 보조 모델 호출을 dcode 의 재시도 정책(model_retry.retry_model_call)으로
# [해설]   감싸는 Runnable. LangChain 요약 엔진의 _summary_model 슬롯에 들어가
# [해설]   'LangChain 기본 3회 with_retry' 를 대체한다. --max-retries 0 이 요약에도
# [해설]   적용되게 하려는 장치다.
class _RetryingModelInvoker(Runnable[LanguageModelInput, AIMessage]):
    """Apply dcode's retry budget to one auxiliary model runnable."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def invoke(
        self,
        input: LanguageModelInput,  # noqa: A002  # `Runnable` keyword contract
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the model with dcode-owned retries.

        Returns:
            The model response.
        """
        from deepagents_code.model_retry import retry_model_call

        return retry_model_call(
            self._model,
            lambda: self._model.invoke(input, config=config, **kwargs),
        )

    async def ainvoke(
        self,
        input: LanguageModelInput,  # noqa: A002  # `Runnable` keyword contract
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Asynchronously invoke the model with dcode-owned retries.

        Returns:
            The model response.
        """
        from deepagents_code.model_retry import aretry_model_call

        return await aretry_model_call(
            self._model,
            lambda: self._model.ainvoke(input, config=config, **kwargs),
        )


# [해설][주의] SDK 내부 슬롯이 이름이 바뀌어 사라졌을 때를 감지하는 안전장치. 존재하지
# [해설]   않는 속성에 대입하면 파이썬은 새 속성을 만들 뿐이라 아무 효과도 없이 조용히
# [해설]   넘어간다. 그 '조용한 무동작'을 AttributeError 로 바꾼다.
def _require_helper_slot(helper: object, name: str, purpose: str) -> None:
    """Fail loudly when an SDK summarization slot dcode overwrites is gone.

    Every write dcode makes into `_lc_helper` replaces behavior the SDK would
    otherwise supply. Assigning to a renamed slot would create a new attribute
    nothing reads, leaving the SDK's own behavior in place -- a silent no-op
    that reports nothing. Checking first turns that into a loud failure.

    Args:
        helper: The SDK summarization helper being patched.
        name: Attribute dcode is about to overwrite.
        purpose: What dcode loses if the slot is gone, phrased to follow
            "so dcode cannot ...".

    Raises:
        AttributeError: If the SDK no longer exposes the named slot.
    """
    if hasattr(helper, name):
        return
    msg = (
        f"{type(helper).__name__} exposes no {name!r} slot, so dcode cannot "
        f"{purpose}. The SDK's summarization internals have changed."
    )
    raise AttributeError(msg)


# [해설][SDK] SDK 가 감싼 LangChain 요약 엔진(summarization._lc_helper)의 _summary_model
# [해설]   을 _RetryingModelInvoker 로 교체한다. --summarization-model 로 지정한 모델이
# [해설]   실제로 요약을 만드는 유일한 지점이기도 하다.
def _install_summary_model_retries(
    summarization: SummarizationMiddleware,
    model: BaseChatModel | None = None,
) -> None:
    """Replace LangChain's generic summary retries with dcode's exact policy.

    Also selects the model summaries are generated with, which is the only
    place a `--summarization-model` override takes effect.

    Args:
        summarization: Middleware whose summary-model slot is replaced.
        model: Model to generate summaries with. `None` reuses
            `summarization.model`, the model driving thresholds and counting.
    """
    helper = summarization._lc_helper
    # A renamed slot would leave LangChain's unconditional three-attempt
    # `with_retry` in place, silently defeating `--max-retries 0`.
    _require_helper_slot(
        helper, "_summary_model", "own compaction summarization retries"
    )
    helper._summary_model = _RetryingModelInvoker(
        model if model is not None else summarization.model
    )


# [해설] 요약 전용 모델에 보낼 이력의 토큰 상한(trim_tokens_to_summarize)을 설정한다.
# [해설]   요약 모델이 메인보다 작을 수 있어 필요하다. 생성자 인자가 아니라 사후에
# [해설]   설정하는 이유는 요약 모델이 지연 생성되기 때문이다.
def _install_summary_trim_limit(
    summarization: SummarizationMiddleware, model: BaseChatModel
) -> None:
    """Bound the history sent to a dedicated summary model.

    Set after construction rather than through the
    `create_summarization_middleware` kwarg because the summary model is built
    lazily, long after the middleware exists. Raises `AttributeError` through
    `_require_helper_slot` if the slot is gone.

    Args:
        summarization: Middleware whose trim budget is replaced.
        model: The model summaries are generated with.
    """
    helper = summarization._lc_helper
    # dcode leaves trimming off by default: `create_summarization_middleware`
    # passes `trim_tokens_to_summarize=None`, overriding LangChain's 4000, and
    # the helper then skips trimming entirely. So a renamed slot would send the
    # whole conversation to a model picked for being smaller than the main one.
    _require_helper_slot(
        helper,
        "trim_tokens_to_summarize",
        "bound the history sent to a dedicated summary model",
    )
    helper.trim_tokens_to_summarize = _summary_trim_limit(model)


# [해설] 상한 계산: 모델 입력 창의 80%(* 4 // 5)를 쓰고 20%는 프롬프트, 직렬화 오버헤드,
# [해설]   근사 토큰 카운터의 오차 몫으로 남긴다. bool 은 int 의 하위 클래스라 True 가 1
# [해설]   로 해석되는 것을 막는 검사도 있다.
def _summary_trim_limit(model: BaseChatModel) -> int:
    """Return the token budget for history sent to the summary model.

    Reserves a fifth of the model's input window for the summary prompt,
    message serialization overhead, and the imprecision of the approximate
    token counter that measures against this budget -- the provider limit is
    hard, the measurement is not.

    Args:
        model: The model summaries are generated with.

    Returns:
        The summary-history budget, falling back to
            `_SUMMARY_TRIM_FALLBACK` when the model exposes no usable positive
            `max_input_tokens`.
    """
    profile = getattr(model, "profile", None)
    if not isinstance(profile, dict):
        return _SUMMARY_TRIM_FALLBACK
    context_limit = profile.get("max_input_tokens")
    # `bool` is an `int` subclass: `max_input_tokens=True` would otherwise
    # reserve its way down to a one-token summary budget.
    if (
        not isinstance(context_limit, int)
        or isinstance(context_limit, bool)
        or context_limit <= 0
    ):
        return _SUMMARY_TRIM_FALLBACK
    return max(1, context_limit * 4 // 5)


# [해설] 요약 모델의 공급자 특성에 맞는 토큰 카운터를 설치한다. 안 하면 메인 모델
# [해설]   기준으로 재면서 요약 모델의 창에 쓰게 되어 예산이 어긋난다.
def _install_summary_token_counter(
    summarization: SummarizationMiddleware, model: BaseChatModel
) -> None:
    """Count dedicated-summary input with that model's provider tuning.

    Raises `AttributeError` through `_require_helper_slot` if a slot is gone.

    Args:
        summarization: Middleware whose summary token counters are replaced.
        model: The model summaries are generated with.
    """
    helper = summarization._lc_helper
    # A renamed slot would measure the summary budget with the main model's
    # tuning while spending it against the summary model's window.
    for slot in ("token_counter", "_partial_token_counter"):
        _require_helper_slot(
            helper, slot, "count dedicated-summary input for the right provider"
        )
    counter = _get_approximate_token_counter(model)
    helper.token_counter = counter
    helper._partial_token_counter = partial(
        counter,
        use_usage_metadata_scaling=False,  # ty: ignore[unknown-argument]
    )


# [해설] 이벤트 루프 블로킹 감시(blockbuster)의 BlockingError 인지 클래스 이름으로
# [해설]   판별한다(테스트 전용 의존성을 import 하지 않으려고). 이 오류는 호출자
# [해설]   결함이므로 '메인 모델로 후퇴' 대신 그대로 올려 보낸다.
def _is_blocking_error(exc: BaseException) -> bool:
    """Whether `exc` is the event-loop blocking guard (`blockbuster`).

    `BlockingError` marks a defect in the caller, not a bad model spec or a
    provider failure, so the degrade-to-main-model fallbacks re-raise it rather
    than absorb it. Matched by class name, as elsewhere in the package, because
    the guard is a test-only dependency that must not be imported here.

    Returns:
        Whether any class in the exception's MRO is named `BlockingError`.
    """
    return any(cls.__name__ == "BlockingError" for cls in type(exc).__mro__)


# [해설] 요약 전용 모델(--summarization-model 또는 /offload model)을 '요약이 처음
# [해설]   필요해질 때' 만들어 설치하는 지연 래퍼. SDK 요약 엔진의
# [해설]   _create_summary/_acreate_summary 를 이 객체의 메서드로 바꿔 끼운다.
# [해설][설계] 모델을 못 만들거나 호출에 실패하면 예외 대신 메인 모델로 요약하도록
# [해설]   '후퇴(degrade)'하고 다음 압축 때 다시 시도한다. 압축 실패가 세션을 멈추거나
# [해설]   /compact 를 망가뜨리면, 컨텍스트가 가득 찬 순간에 사용자가 쓸 수단이 사라지기
# [해설]   때문이다.
# [해설][주의] 경고는 요약기당 한 번만 남긴다. 스레드 락(_lock, _invocation_lock)으로
# [해설]   동시 호출을 직렬화하며, 비동기 경로는 락 획득을 to_thread 로 넘겨 이벤트
# [해설]   루프를 막지 않는다.
class _LazySummaryModel:
    """Configure a dedicated model immediately before summary generation."""

    def __init__(
        self,
        summarization: SummarizationMiddleware,
        model_spec: str,
        cli_max_retries: int | None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._summarization = summarization
        self._model_spec = model_spec
        self._cli_max_retries = cli_max_retries
        self._environ = environ
        self._lock = Lock()
        self._invocation_lock = Lock()
        self._configured = False
        self._warned = False
        helper = summarization._lc_helper
        # Main-model summary tuning, captured before any override installs so
        # `_degrade_to_main_model` can uninstall the override.
        self._main_summary_tuning = (
            helper.token_counter,
            helper._partial_token_counter,
            helper.trim_tokens_to_summarize,
        )
        self._create_summary = helper._create_summary
        self._acreate_summary = helper._acreate_summary

    # [해설][흐름] 1) 이미 설정됐으면 반환 2) create_model 로 요약 모델
    # [해설]   생성(워크스페이스 환경 적용) 3) 실패하면 메인 모델 재시도 정책만 설치하고
    # [해설]   경고 후 반환 4) 성공하면 재시도, 토큰 카운터, 트림 상한을 요약 모델용으로
    # [해설]   설치하고 _configured=True.
    def _configure(self) -> None:
        """Install the model once, degrading to the main model on failure.

        A summary model that cannot be built is a broken optimization, not a
        broken session: raising here would fail the turn that triggered
        compaction and would also break `/compact`, the tool the user reaches
        for when the context is full. So the summary runs on the main model
        instead and the next compaction retries the override.
        """
        with self._lock:
            if self._configured:
                return
            from deepagents_code.config import create_model

            try:
                from deepagents_code.config import use_environment

                with use_environment(self._environ):
                    model = create_model(
                        self._model_spec,
                        cli_max_retries=self._cli_max_retries,
                    ).model
            except Exception as exc:
                # BlockingError means this ran on the server event loop; that is
                # a defect in the caller, not a bad model spec, so let it out.
                if _is_blocking_error(exc):
                    raise
                _install_summary_model_retries(self._summarization)
                if not self._warned:
                    # Warned once per summarizer: compaction is rare, but a
                    # broken spec would otherwise log on every attempt.
                    self._warned = True
                    logger.warning(
                        "Could not build the summarization model %r; compaction "
                        "summaries use the main agent model instead. Run "
                        "`/offload model clear` to stop trying it.",
                        self._model_spec,
                        exc_info=True,
                    )
                return
            _install_summary_model_retries(self._summarization, model)
            _install_summary_token_counter(self._summarization, model)
            _install_summary_trim_limit(self._summarization, model)
            self._configured = True

    # [해설] 요약 모델 호출 실패 시 설치했던 오버라이드를 걷어 내고(메인 모델의 토큰
    # [해설]   카운터/트림 값으로 복원) _configured 를 되돌린다. exc 를 인자로 받는
    # [해설]   이유는 비동기 경로가 워커 스레드에서 실행되어 sys.exc_info() 가 비기
    # [해설]   때문이다.
    def _degrade_to_main_model(self, exc: Exception) -> None:
        """Restore main-model summary generation after an invocation failure.

        A dedicated model that builds but cannot generate (an unknown model ID,
        say, or missing provider access) is the same broken optimization as one
        that cannot build: the summary runs on the main model instead, and
        `_configured` resets so the next compaction retries the override.

        Args:
            exc: The failure being logged. Passed explicitly because the async
                path runs this on a worker thread, where `sys.exc_info()` is
                thread-local and would log `NoneType: None`.
        """
        with self._lock:
            _install_summary_model_retries(self._summarization)
            helper = self._summarization._lc_helper
            (
                helper.token_counter,
                helper._partial_token_counter,
                helper.trim_tokens_to_summarize,
            ) = self._main_summary_tuning
            self._configured = False
            if not self._warned:
                # Shared with the build-failure warning: either failure mode
                # means the override is suspect, and one warning per summarizer
                # is enough.
                self._warned = True
                logger.warning(
                    "The summarization model %r failed to generate a summary; "
                    "compaction summaries use the main agent model instead. Run "
                    "`/offload model clear` to stop trying it.",
                    self._model_spec,
                    exc_info=exc,
                )

    # [해설] 동기 요약. 요약 모델 실패 시 한 번 메인 모델로 후퇴해 재시도하고, 메인 모델
    # [해설]   자체의 실패는 그대로 올린다(후퇴할 곳이 없으므로).
    def create_summary(self, messages: list[AnyMessage]) -> str:
        """Generate a synchronous summary after lazy configuration.

        Returns:
            The generated summary.
        """
        with self._invocation_lock:
            self._configure()
            try:
                return self._create_summary(messages)
            except Exception as exc:
                # `_configured` means the override, not the main model, was behind
                # the failed call; a main-model failure has nothing to fall back to.
                if _is_blocking_error(exc) or not self._configured:
                    raise
                self._degrade_to_main_model(exc)
            return self._create_summary(messages)

    @asynccontextmanager
    # [해설] 스레드 락을 이벤트 루프를 막지 않고 잡는 비동기 컨텍스트 매니저.
    # [해설]   취소(BaseException) 시에도 획득 태스크를 끝까지 기다렸다가 락을 풀어
    # [해설]   누수를 막는다.
    async def _async_invocation_lock(self) -> AsyncIterator[None]:
        """Hold the shared invocation lock without blocking the event loop."""
        acquisition = asyncio.create_task(
            asyncio.to_thread(self._invocation_lock.acquire)
        )
        try:
            await asyncio.shield(acquisition)
        except BaseException:
            await acquisition
            self._invocation_lock.release()
            raise
        try:
            yield
        finally:
            self._invocation_lock.release()

    # [해설] 비동기 요약. 동기판과 같은 로직이며 락, 설정, 후퇴를 to_thread 로
    # [해설]   오프로드한다.
    async def acreate_summary(self, messages: list[AnyMessage]) -> str:
        """Generate an asynchronous summary after lazy configuration.

        Returns:
            The generated summary.
        """
        async with self._async_invocation_lock():
            await asyncio.to_thread(self._configure)
            try:
                return await self._acreate_summary(messages)
            except Exception as exc:
                # `_configured` means the override, not the main model, was behind
                # the failed call; a main-model failure has nothing to fall back to.
                if _is_blocking_error(exc) or not self._configured:
                    raise
                # Locked like `_configure`, so it stays off the event loop.
                await asyncio.to_thread(self._degrade_to_main_model, exc)
            return await self._acreate_summary(messages)


# [해설][SDK] SDK 요약 엔진의 _create_summary/_acreate_summary 를 _LazySummaryModel 의
# [해설]   메서드로 교체한다(요약기 인스턴스 단위라 다른 요약기에는 영향 없음).
def _install_lazy_summary_model(
    summarization: SummarizationMiddleware,
    model_spec: str,
    cli_max_retries: int | None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Defer dedicated model construction until a summary is requested.

    Raises `AttributeError` through `_require_helper_slot` if a hook is gone.

    Args:
        summarization: Middleware whose summary hooks are wrapped.
        model_spec: Spec to resolve when a summary is first requested.
        cli_max_retries: Explicit `--max-retries` value, or `None`.
        environ: Workspace environment retained for lazy model construction.
    """
    for slot in ("_create_summary", "_acreate_summary"):
        _require_helper_slot(
            summarization._lc_helper, slot, "defer summary-model construction"
        )
    lazy = _LazySummaryModel(summarization, model_spec, cli_max_retries, environ)
    # LangChain annotates these private attributes as unbound method shapes;
    # instance-local bound wrappers are intentional so only this summarizer is lazy.
    summarization._lc_helper._create_summary = (  # ty: ignore[invalid-assignment]
        lazy.create_summary
    )
    summarization._lc_helper._acreate_summary = (  # ty: ignore[invalid-assignment]
        lazy.acreate_summary
    )


# [해설] 강제 압축(/offload)이 읽고 쓰는 체크포인트 채널 모음 타입. SDK 의
# [해설]   SummarizationState 에서 _summarization_event 채널 정의를 상속해 주석대로 SDK
# [해설]   와 어긋난 복사본이 생기지 않게 한다.
class _OffloadState(CostState, SummarizationState, total=False):
    """Checkpoint channels server-owned forced compaction reads and writes.

    `_summarization_event` is inherited from `SummarizationState` rather than
    re-declared so it keeps the SDK's own annotation (`NotRequired`, `| None`,
    and the `PrivateStateAttr` marker) instead of a divergent copy that claimed
    the value is always present.

    `total=False` describes the inherited shape only -- this body declares no
    keys of its own, so the modifier is deliberate documentation rather than a
    constraint on anything written here.
    """


# [해설] 오프로드 시도의 결과 상태 6종: compacted(압축됨), summarized(핸드오프 요약),
# [해설]   empty(빈 스레드), noop(압축할 것 없음), denied(훅이 거부), failed(실패).
type OffloadStatus = Literal[
    "compacted", "summarized", "empty", "noop", "denied", "failed"
]
"""Outcome of one offload attempt. Aliased so the result type and the private
`_result` factory cannot drift apart."""


# [해설] 서버 연산이 클라이언트에 돌려주는 결과 타입(전선 형식). tokens_before/after 로
# [해설]   압축 효과를, archive_path 로 원문 아카이브 위치를 알린다. archive_ephemeral
# [해설]   은 임시 저장소라 재시작 시 사라질 수 있다는 뜻이다(offload.py).
class OffloadResult(TypedDict):
    """Typed result emitted by the server-owned offload operation.

    Handoff message and token counts describe the unchanged source thread.
    """

    status: OffloadStatus
    messages_offloaded: int
    messages_kept: int
    tokens_before: int
    tokens_after: int
    archive_path: str | None
    archive_ephemeral: bool
    error: str | None
    summary: NotRequired[str]
    """Summary text for a handoff; the source thread stays uncompacted."""


# [해설][설계] 서버 연산이 체크포인트에 쓸 수 있는 채널을 타입으로 한정한다. 'messages
# [해설]   는 절대 쓰지 않는다'는 불변식을 문자열 검사가 아니라 타입의 성질로 만든
# [해설]   것이다. 런타임 검사는 offload_api 에 보조로 남아 있다.
class OffloadStateUpdate(TypedDict, total=False):
    """The only checkpoint channels a server-owned operation may write.

    Naming the permitted channels makes the load-bearing invariant -- that this
    route can never write `messages` -- a property of the type rather than a
    single string check performed after the summarizer has already been billed.
    The runtime check in `offload_api` stays as a backstop for the `Any`-typed
    values the summarization SDK hands back.
    """

    _summarization_event: dict[str, Any]
    _summarization_session_id: str
    _session_cost_usd: float
    _session_cost_breakdown: dict[str, Any]


# [해설] 한 번의 서버 연산이 만든 (상태 갱신, 결과, 보류 중인 아카이브) 묶음. archive 는
# [해설]   체크포인트 예약이 끝난 뒤 쓰도록 미뤄 둔 쓰기다.
class OffloadExecution(NamedTuple):
    """State update and typed result produced by one server operation."""

    update: OffloadStateUpdate
    result: OffloadResult
    archive: _PendingArchive | None = None


# [해설] 세션(스레드) ID 별 프로세스 내부 락 저장소. 약한 참조라 쓰이지 않는 락은 알아서
# [해설]   사라진다.
_archive_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


# [해설] 같은 아카이브 파일의 읽기-덧붙이기-쓰기 사이클을 직렬화하는 asyncio 락. 자동
# [해설]   압축, 도구 압축, /offload 가 같은 파일을 동시에 갱신해 이력이 사라지는 것을
# [해설]   막는다.
# [해설][주의] asyncio 락이라 프로세스 안에서만 유효하다. 동기 경로(_run_compact)는 이
# [해설]   락을 잡을 수 없어 SDK 동작과 같게 둔다.
def _archive_lock(session_id: str) -> asyncio.Lock:
    """Return the process-local lock serializing one archive's read/write cycle."""
    lock = _archive_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _archive_locks[session_id] = lock
    return lock


# [해설] '체크포인트 요약이 예약된 뒤에 쓰려고 미뤄 둔 아카이브 덧붙이기'. write() 는
# [해설]   쓰기 전 기존 내용을 읽어 두어 실패 시 되돌릴 수 있게 한다.
class _PendingArchive(NamedTuple):
    """Archive append deferred until the checkpoint summary is reserved."""

    summarization: SummarizationMiddleware
    backend: BackendProtocol
    messages: list[AnyMessage]
    session_id: str
    summary: str
    state_cutoff: int

    def update(self, file_path: str | None) -> dict[str, Any]:
        """Build the summary update with the archive's settled path.

        Returns:
            State update containing the summary, cutoff, and archive path.
        """
        return CLICompactionMiddleware._forced_compaction_update(
            self.summarization,
            self.summary,
            file_path,
            self.state_cutoff,
            self.session_id,
        )

    # [해설] 롤백용으로 아카이브의 현재 내용을 읽는다. 없으면 (False, ""), 읽기 오류면
    # [해설]   RuntimeError.
    async def _previous_content(self, path: str) -> tuple[bool, str]:
        """Read the archive snapshot needed to undo an uncommitted append.

        Returns:
            Whether the archive existed and its prior UTF-8 content.

        Raises:
            RuntimeError: If the backend cannot return the archive snapshot.
        """
        responses = await self.backend.adownload_files([path])
        if not responses:
            msg = f"archive backend returned no response for {path}"
            raise RuntimeError(msg)
        response = responses[0]
        if response.error == FILE_NOT_FOUND:
            return False, ""
        if response.error is not None:
            msg = f"archive read failed for {path}: {response.error}"
            raise RuntimeError(msg)
        content = response.content or b""
        return True, content.decode("utf-8")

    # [해설][흐름] 1) 경로 계산 2) 이전 내용 스냅샷 3) _ArchiveReadGuard 로 감싼
    # [해설]   백엔드로 SDK _aoffload_to_backend 실행 4) SDK 가 쓰기에 실패(None)하면
    # [해설]   즉시 롤백하고 None 반환 5) 성공하면 되돌릴 수 있는 _ArchiveAppend 반환.
    async def write(self) -> _ArchiveAppend | None:
        """Append staged messages and retain enough state for rollback.

        Returns:
            The reversible append, or `None` when the SDK could not write it.
        """
        path = self.summarization._get_history_path(self.session_id)
        existed, previous = await self._previous_content(path)
        guard = cast("BackendProtocol", _ArchiveReadGuard(self.backend))
        written_path = await self.summarization._aoffload_to_backend(
            guard, self.messages, self.session_id
        )
        append = _ArchiveAppend(self.backend, path, existed, previous)
        if written_path is None:
            await append.rollback()
            return None
        return append


# [해설] 완료된 아카이브 덧붙이기를 체크포인트가 확정될 때까지 되돌릴 수 있게 하는 기록.
# [해설]   rollback() 은 이전 내용을 복원하거나(기존 파일) 파일을 삭제한다(새 파일).
class _ArchiveAppend(NamedTuple):
    """Completed archive append that can be restored until checkpointed."""

    backend: BackendProtocol
    path: str
    existed: bool
    previous: str

    async def rollback(self) -> None:
        """Restore the exact archive snapshot from before the append.

        Raises:
            RuntimeError: If the backend cannot restore the snapshot.
        """
        result = (
            await self.backend.awrite(self.path, self.previous)
            if self.existed
            else await self.backend.adelete(self.path)
        )
        if result.error is not None:
            msg = f"archive rollback failed for {self.path}: {result.error}"
            raise RuntimeError(msg)


# [해설] 강제 압축의 '계획': 요약 문자열, 절대 cutoff 인덱스, 아직 쓰지 않은 아카이브.
# [해설]   계획과 쓰기를 분리해 체크포인트 예약이 성공한 뒤에만 파일을 쓸 수 있다.
class _ForcedCompactionPlan(NamedTuple):
    """Checkpoint update plus its not-yet-written archive append."""

    summarization: SummarizationMiddleware
    summary: str
    state_cutoff: int
    archive: _PendingArchive

    def update(self, file_path: str | None) -> dict[str, Any]:
        """Build the summary update with the archive's settled path.

        Returns:
            State update containing the summary, cutoff, and archive path.
        """
        return self.archive.update(file_path)


# [해설] 상태를 바꾸지 않은 결과(empty/noop/denied/failed) 생성. HTTP 경계가 서버
# [해설]   런타임을 만들지 않고도 호출할 수 있게 모듈 수준 함수로 두었다. denied/failed
# [해설]   는 이유(error)가 없으면 ValueError 로 막는다(이유 없는 거부는 사용자에게 아무
# [해설]   정보도 주지 못하므로).
def unchanged_offload_result(
    status: OffloadStatus,
    *,
    messages: int,
    tokens: int,
    error: str | None = None,
) -> OffloadResult:
    """Build a result for an operation that did not compact state.

    Module-level so the HTTP boundary can report an unchanged outcome without
    resolving the server runtime first -- an empty thread has nothing to
    compact, and building the agent only to describe that is both wasteful and
    a way for a construction failure to turn "nothing to do" into a 500.

    Args:
        status: A non-compacting outcome.
        messages: Messages left in the conversation.
        tokens: Context estimate, unchanged by definition.
        error: Reason, required for `denied` and `failed`.

    Returns:
        Typed result containing unchanged context statistics.

    Raises:
        ValueError: If a refusal carries no reason.
    """
    if status in {"denied", "failed"} and not error:
        # `error` is `str | None` on every status because the wire shape is one
        # flat object, so the checker cannot make "a refusal has a reason" a
        # compile-time fact. Enforce it at the single construction point
        # instead: a reasonless refusal renders as the client's generic "the
        # server rejected the operation", which tells the user nothing.
        msg = f"An offload {status!r} result must carry a reason."
        raise ValueError(msg)
    return {
        "status": status,
        "messages_offloaded": 0,
        "messages_kept": messages,
        "tokens_before": tokens,
        "tokens_after": tokens,
        "archive_path": None,
        "archive_ephemeral": False,
        "error": error,
    }


# [해설] 전선 응답 2종 중 하나: 클라이언트 개입 없이 끝난 시도. 다른
# [해설]   하나(OffloadInterruptResponse)는 훅 요청을 클라이언트가 이행해야 하는 경우다.
class OffloadCompleteResponse(TypedDict):
    """Wire response for an attempt that finished without needing the client."""

    status: Literal["complete"]
    result: OffloadResult


class OffloadInterruptResponse(TypedDict):
    """Wire response carrying a hook request the client must fulfill.

    `request` stays a plain mapping on purpose: the client transports it back
    without inspecting it, and only `hooks.interrupt` owns its shape.
    """

    status: Literal["interrupt"]
    request: dict[str, Any]


# [해설] 오프로드 프로토콜 한 라운드의 응답 타입. 서버(offload_api._execute_offload)와
# [해설]   클라이언트(RemoteAgent.aoffload)가 하나의 정의를 공유한다(HTTP 를 건너므로
# [해설]   런타임 검증은 별도).
type OffloadResponse = OffloadCompleteResponse | OffloadInterruptResponse
"""One round of the offload operation protocol.

Tagged on `status` so the producer (`offload_api._execute_offload`) and the
consumer (`RemoteAgent.aoffload`) are checked against one definition instead of
two independently hand-written `isinstance` ladders. The client still validates
at runtime -- this crosses HTTP, so the type is a contract, not a guarantee.
"""


# [해설] OffloadOperation 을 백엔드 객체에 붙일 때 쓰는 속성 이름.
_OFFLOAD_OPERATION_ATTR = "_dcode_offload_operation"


# [해설] create_cli_agent 가 호출한다. 서버 HTTP 라우트는 같은 CompositeBackend 를 통해
# [해설]   이 연산을 찾는다(offload_operation_from).
# [해설][주의] 압축 미들웨어가 다른 백엔드에 쓰고 있으면(요약기의 _backend 가 이
# [해설]   백엔드와 다르면) ValueError. 아카이브가 엉뚱한 곳에 써지는 것을 막는 검사다.
def attach_offload_operation(
    backend: CompositeBackend,
    operation: OffloadOperation,
) -> None:
    """Publish the operation on the backend shared with the server runtime.

    Args:
        backend: Composite backend owned by the agent server.
        operation: Offload implementation bound to that backend.

    Raises:
        ValueError: If compaction writes through a different backend.
    """
    # The SDK requires `backend` in its constructor, so a real summarization
    # middleware always has one; `None` here means a test double, which is
    # allowed through rather than asserted against.
    bound = getattr(operation._compaction._summarization, "_backend", None)
    if bound is not None and bound is not backend:
        msg = "Offload operation must use the agent's composite backend"
        raise ValueError(msg)
    setattr(backend, _OFFLOAD_OPERATION_ATTR, operation)


# [해설] 백엔드에 게시된 OffloadOperation 을 꺼낸다. 없거나 타입이 다르면 None.
def offload_operation_from(backend: CompositeBackend) -> OffloadOperation | None:
    """Return the server operation published on `backend`, when available."""
    operation = getattr(backend, _OFFLOAD_OPERATION_ATTR, None)
    return operation if isinstance(operation, OffloadOperation) else None


# [해설] 저장된 _summarization_event 에서 cutoff_index(절대 인덱스)를 읽는다. 형식이
# [해설]   이상하면 0. bool 은 int 로 통과해 버리므로 명시적으로 제외한다.
def _event_cutoff(event: object) -> int:
    """Return the absolute cutoff index carried by a `_summarization_event`.

    Args:
        event: A `_summarization_event` mapping (as persisted in state), or
            `None`.

    Returns:
        The `cutoff_index`, or `0` when the event is missing or malformed.
    """
    if isinstance(event, dict):
        cutoff = event.get("cutoff_index")
        # `bool` is excluded explicitly: it passes `isinstance(_, int)`, so a
        # malformed `cutoff_index: true` would otherwise read as cutoff 1 and
        # silently shift the offloaded/kept counts by one message. The HTTP
        # boundary rejects bools for `model_context_limit` for the same reason.
        if isinstance(cutoff, int) and not isinstance(cutoff, bool):
            return cutoff
    return 0


# [해설] 훅이 자동 압축을 막았을 때, 원래의 공급자 컨텍스트 초과
# [해설]   오류(ContextOverflowError)를 SDK 의 폴백 처리기를 통과시켜 되돌려 주기 위한
# [해설]   운반용 예외.
class _AutoCompactionBlockedError(Exception):
    """Carry a blocked provider overflow past the SDK fallback handler."""

    def __init__(self, overflow: ContextOverflowError) -> None:
        super().__init__(str(overflow))
        self.overflow = overflow


# [해설] 실행 중인 요청의 모델 설정(모델 스펙, 요약 모델 스펙, 모델 파라미터, 프로필
# [해설]   덮어쓰기, 컨텍스트 한도). 이름 있는 튜플이라 같은 모양의 슬롯을 위치로
# [해설]   헷갈리지 않는다.
class RuntimeModelConfig(NamedTuple):
    """Active model configuration read from a tool runtime.

    A named tuple rather than a bare tuple so the structurally identical slots
    — the two `str | None` model specs and the two `dict` overrides — are
    addressed by name at both the construction sites (keyword args) and the
    read site (attribute access) — a silent positional transposition the type
    checker would not catch is thereby avoided. Positional
    construction/unpacking is still possible and would defeat this, so call
    sites must keep using names.
    """

    model_spec: str | None
    summarization_model_spec: str | None
    model_params: dict[str, Any]
    profile_overrides: dict[str, Any]
    context_limit: int | None


# [해설] context 속성만 가진 객체를 뜻하는 구조적 타입. 도구에 주입되는 ToolRuntime 과
# [해설]   서버 연산이 받는 LangGraph Runtime 둘 다 받아들이기 위한 최소 의존성
# [해설]   표현이다.
class _HasRunContext(Protocol):
    """Anything carrying a per-run context object.

    The compaction helpers read `context` and nothing else, so they accept both
    the `ToolRuntime` injected into the tool and the plain LangGraph `Runtime`
    the server operation receives -- which is not a `ToolRuntime`. Stating
    the dependency this narrowly means a helper that starts touching, say,
    `tool_call_id` fails to type-check instead of breaking the server operation
    at runtime.
    """

    @property
    def context(self) -> object:
        """The run's context object.

        Typed as `object` rather than `Any`: every consumer narrows the shape
        with `isinstance` before touching it, so `object` type-checks the same
        code while still rejecting an unnarrowed attribute access.
        """
        ...


# [해설] 실행에 바인딩된 작업공간 cwd 를 컨텍스트에서 읽고, 없으면 프로세스의 cwd 를
# [해설]   쓴다. 훅 실행 위치를 정할 때 쓴다.
def _runtime_cwd(runtime: _HasRunContext) -> Path:
    """Return the bound workspace directory, or preserve the process fallback."""
    context = CLIContextSchema.from_payload(runtime.context)
    if context is not None:
        cwd = context.workspace.get("cwd")
        if isinstance(cwd, str) and cwd:
            return Path(cwd)
    return Path.cwd()


# [해설] 런타임 컨텍스트(CLIContextSchema, 서버에서는 JSON dict 일 수 있음)에서 모델
# [해설]   설정을 읽는다. 컨텍스트가 없으면 빈 설정. /model 이나 /offload model 로 바뀐
# [해설]   값이 여기로 들어온다.
def _runtime_model_config(runtime: _HasRunContext) -> RuntimeModelConfig:
    """Read the active model configuration from a run context carrier.

    Args:
        runtime: Runtime carrying the current `CLIContext`.

    Returns:
        The active model specification, invocation parameters, profile
            overrides, and effective context-window limit.
    """
    # `from_payload` narrows both the in-process schema and the remote JSON
    # dict, so this stays in step with every other context consumer.
    context = CLIContextSchema.from_payload(runtime.context)
    if context is None:
        return RuntimeModelConfig(
            model_spec=None,
            summarization_model_spec=None,
            model_params={},
            profile_overrides={},
            context_limit=None,
        )
    return RuntimeModelConfig(
        model_spec=context.model,
        summarization_model_spec=context.summarization_model,
        model_params=context.model_params,
        profile_overrides=context.profile_overrides,
        context_limit=context.model_context_limit,
    )


# [해설][설계] 읽기 실패를 '파일 없음'으로 오해해 기존 이력을 덮어쓰는 것을 막는 백엔드
# [해설]   어댑터. SDK 아카이브 도우미는 읽기가 실패하면 '파일 없음'처럼 취급하고 이어서
# [해설]   파일을 새로 쓰는데, 이 래퍼는 읽기 오류를 기록해 두었다가 이후 write/edit 를
# [해설]   RuntimeError 로 거부한다(fail closed).
# [해설][주의] 이 래퍼를 SDK 요약기 생성자에 넘기면 안 된다(아래
# [해설]   _summarization_for_runtime 주석 참고): CompositeBackend 인스턴스 검사가
# [해설]   실패해 아카이브 접두사가 / 로 돌아가면 대화 이력이 사용자 프로젝트 트리에
# [해설]   써진다.
class _ArchiveReadGuard:
    """Prevent an archive write after its prerequisite read fails.

    The SDK archive helper treats any unsuccessful read like a missing file and
    follows it with a truncating `write`. This narrow backend adapter preserves
    the SDK formatting and append behavior while making that fallback fail closed.
    """

    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend
        self._read_failed = False

    def _record_response_errors(
        self, responses: list[FileDownloadResponse]
    ) -> list[FileDownloadResponse]:
        """Record read errors other than an expected missing archive.

        Args:
            responses: Backend download responses to inspect.

        Returns:
            The unchanged backend download responses.
        """
        if any(
            response.error is not None and response.error != FILE_NOT_FOUND
            for response in responses
        ):
            self._read_failed = True
        return responses

    def _ensure_read_succeeded(self) -> None:
        """Raise when a prior archive read failed in this operation.

        Raises:
            RuntimeError: If the prerequisite archive read failed.
        """
        if self._read_failed:
            msg = "archive read failed; refusing to overwrite existing history"
            raise RuntimeError(msg)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Delegate a synchronous read while recording failures.

        Args:
            paths: Backend paths to read.

        Returns:
            The backend download responses.
        """
        try:
            responses = self._backend.download_files(paths)
        except Exception:
            self._read_failed = True
            raise
        return self._record_response_errors(responses)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Delegate an asynchronous read while recording failures.

        Args:
            paths: Backend paths to read.

        Returns:
            The backend download responses.
        """
        try:
            responses = await self._backend.adownload_files(paths)
        except Exception:
            self._read_failed = True
            raise
        return self._record_response_errors(responses)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write only when the prerequisite archive read succeeded.

        Args:
            file_path: Backend path to write.
            content: Complete archive content.

        Returns:
            The backend write result.
        """
        self._ensure_read_succeeded()
        return self._backend.write(file_path, content)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Asynchronously write only after a successful archive read.

        Args:
            file_path: Backend path to write.
            content: Complete archive content.

        Returns:
            The backend write result.
        """
        self._ensure_read_succeeded()
        return await self._backend.awrite(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit only when the prerequisite archive read did not raise.

        Args:
            file_path: Backend path to edit.
            old_string: Existing archive content.
            new_string: Archive content with the new section appended.
            replace_all: Whether to replace every match.

        Returns:
            The backend edit result.
        """
        self._ensure_read_succeeded()
        return self._backend.edit(
            file_path, old_string, new_string, replace_all=replace_all
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Asynchronously edit only after a successful archive read.

        Args:
            file_path: Backend path to edit.
            old_string: Existing archive content.
            new_string: Archive content with the new section appended.
            replace_all: Whether to replace every match.

        Returns:
            The backend edit result.
        """
        self._ensure_read_succeeded()
        return await self._backend.aedit(
            file_path, old_string, new_string, replace_all=replace_all
        )


# [해설] === 이 파일의 중심 클래스 ===
# [해설][SDK] SDK SummarizationToolMiddleware(compact_conversation 도구를 제공하고 자동
# [해설]   요약 엔진을 감쌈)를 상속한다. name 속성이 내부 SummarizationMiddleware 의
# [해설]   이름을 돌려주므로, SDK 가 기본으로 넣는 요약 미들웨어 슬롯을 '교체'하고
# [해설]   도구도 함께 제공한다(agent.py 의 create_cli_agent 주석 참조).
# [해설] 세 가지 압축 경로: (1) 자동: wrap_model_call/awrap_model_call 이 모델 호출 전에
# [해설]   토큰 임계를 확인해 요약 (2) 도구: compact_conversation
# [해설]   도구(_create_compact_tool) (3) 강제: 서버 /offload 의
# [해설]   _aplan_forced_compaction_update.
# [해설] dcode 가 추가한 것: PreCompact 훅 게이트, 런타임 모델 변경
# [해설]   반영(_summarization_for_runtime), 요청 단위 아카이브 락, 컨텍스트 초과 오류
# [해설]   시의 훅 게이트.
class CLICompactionMiddleware(SummarizationToolMiddleware):
    """Add hook-aware automatic and server-owned compaction for dcode.

    The SDK tool's normal, model-initiated behavior remains unchanged.
    `_aplan_forced_compaction_update` is the state-only planning entry point
    used by the server-owned `/offload` operation.
    """

    # [해설] 요약 엔진과 재시도/요약 모델/환경 설정을 보관한다. _runtime_summarization
    # [해설]   은 1칸짜리 캐시로, /model 이나 /offload model 이 바뀔 때만 요약기를 다시
    # [해설]   만들게 해 매 턴 create_model 의 자격 증명 읽기와 클라이언트 생성을
    # [해설]   피한다.
    def __init__(
        self,
        summarization: SummarizationMiddleware,
        *,
        system_prompt: str | None = None,
        cli_max_retries: int | None = None,
        summarization_model_spec: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the CLI compaction middleware.

        Args:
            summarization: Summarization engine used by every compaction path.
            system_prompt: Optional prompt fragment advertising the compact tool.
            cli_max_retries: Explicit `--max-retries` value to retain when
                rebuilding a runtime-selected model for `/offload`.
            summarization_model_spec: Startup model spec to use only for summaries.
            environ: Workspace environment retained for lazy model construction.
        """
        super().__init__(summarization, system_prompt=system_prompt)
        self._cli_max_retries = cli_max_retries
        self._summarization_model_spec = summarization_model_spec
        self._environ = environ
        # One-entry memo for `_summarization_for_runtime`. Every model call
        # consults it, but the runtime model configuration only changes on
        # `/model` or `/offload model`, so a single slot hits almost
        # always and keeps `create_model`'s credential reads and client
        # construction off the per-turn path. It also lets one
        # `_LazySummaryModel` survive across compactions, which is what its
        # `_configured` and `_warned` memos are sized for.
        self._runtime_summarization: tuple[str, SummarizationMiddleware] | None = None

    @property
    # [해설][설계] 내부 요약 엔진의 이름을 그대로 보고한다. SDK 의 이름 기반 병합이 이
    # [해설]   객체를 'SummarizationMiddleware' 로 보고 SDK 기본 요약 미들웨어를
    # [해설]   대체하게 만드는 핵심 한 줄이다.
    def name(self) -> str:
        """Replace the SDK auto-summarizer while retaining the compact tool."""
        return self._summarization.name

    @staticmethod
    # [해설] 모델 입력 스냅샷의 안정적인 식별자(메시지 수 + 마지막 메시지 ID/해시). 같은
    # [해설]   시도가 훅 이벤트에서 같은 logical_event_id 를 갖게 한다.
    def _auto_compaction_id(request: ModelRequest) -> str:
        """Return a stable identity for one model-input snapshot."""
        messages = request.messages
        last = messages[-1]
        identity = (
            last.id or hashlib.sha256(last.model_dump_json().encode()).hexdigest()
        )
        return f"{len(messages)}:{identity}"

    # [해설] 자동 요약 직전에 PreCompact 훅을 실행하고 continue_processing 여부를
    # [해설]   돌려준다. 훅이 비활성이면 True(그대로 진행).
    def _pre_auto_compact(self, request: ModelRequest) -> bool:
        """Run `PreCompact` before automatic summarization.

        Returns:
            Whether summarization may continue.
        """
        from langgraph.config import get_config

        runtime = request.runtime
        gate = _session_gate(runtime.context)
        if not _event_enabled(gate, HookEvent.PRE_COMPACT):
            return True
        try:
            config = get_config()
        except RuntimeError:
            config = None
        decision = _invoke_hook(
            _hook_context(runtime.context, config, _runtime_cwd(runtime)),
            PreCompactEvent(event=HookEvent.PRE_COMPACT, trigger=CompactTrigger.AUTO),
            gate=gate,
            config=config,
            deadline=_DEFAULT_DEADLINE,
            logical_event_id=self._auto_compaction_id(request),
        )
        return _require_decision(decision, PreCompactDecision).continue_processing

    @staticmethod
    # [해설] 임계 기반 자동 압축이 실제로 일어날지 미리 판정한다. 압축 대상이면 정리된
    # [해설]   요청을, 아니면 None 을 돌려준다. 이때 SDK 요약기의 내부
    # [해설]   메서드(_get_effective_messages, _count_tokens, _truncate_args,
    # [해설]   _should_summarize, _determine_cutoff_index)를 호출한다.
    # [해설] 요청 단위 summarizer 를 인자로 받는 정적 메서드인 이유: self._summarization
    # [해설]   을 직접 읽으면 런타임 모델 오버라이드를 우회할 수 있어, 구조적으로 막은
    # [해설]   것이다.
    def _auto_compaction_request(
        request: ModelRequest,
        summarization: SummarizationMiddleware,
    ) -> ModelRequest | None:
        """Return the prepared request when threshold compaction will run.

        Args:
            request: The pending model request.
            summarization: Request-local summarizer for this runtime. Passed in
                rather than read from `self._summarization` so a summarization
                override cannot be bypassed; `@staticmethod` makes that
                structural rather than a convention.

        Returns:
            The prepared request, or `None` when threshold compaction will not
                run.
        """
        messages = summarization._get_effective_messages(request)
        tokens = summarization._count_tokens(
            messages, request.system_message, request.tools
        )
        messages, modified = summarization._truncate_args(messages, tokens)
        if modified:
            tokens = summarization._count_tokens(
                messages, request.system_message, request.tools
            )
        if (
            not summarization._should_summarize(messages, tokens)
            or summarization._determine_cutoff_index(messages) <= 0
        ):
            return None
        return request.override(messages=messages)

    # [해설] 공급자가 컨텍스트 초과 오류를 낸 뒤 SDK 의 폴백 압축이 실행되기 전에 훅으로
    # [해설]   게이트한다. 훅이 막으면 _AutoCompactionBlockedError 로 신호한다.
    def _pre_overflow_compact(
        self,
        request: ModelRequest,
        overflow: ContextOverflowError,
        summarization: SummarizationMiddleware,
    ) -> None:
        """Gate a provider-overflow fallback before it compacts.

        Args:
            request: The request that overflowed.
            overflow: The provider overflow being recovered from.
            summarization: Request-local summarizer for this runtime, passed in
                so an override cannot be bypassed.

        Raises:
            _AutoCompactionBlockedError: If the hook blocks compaction.
        """
        if summarization._determine_cutoff_index(
            request.messages
        ) > 0 and not self._pre_auto_compact(request):
            raise _AutoCompactionBlockedError(overflow) from overflow

    # [해설][SDK] langchain AgentMiddleware 의 wrap_model_call 훅. 모델 호출을 감싸는
    # [해설]   양파 구조에서, 이 미들웨어는 모델 호출 직전에 컨텍스트를 확인한다.
    # [해설][흐름] 1) 원래 모델/설정을 보관 2) 요청 단위 요약기 선택 3) 런타임에 선택된
    # [해설]   메인 모델 기준으로 요청 정리 4) 임계 초과면 PreCompact 훅 -> 통과하면 SDK
    # [해설]   요약기가 요약+아카이브 후 호출, 훅이 거부하면 압축 없이 호출 5) 임계
    # [해설]   미달이면 모델 호출을 하되 공급자가 컨텍스트 초과 오류를 내면 한 번만 훅
    # [해설]   게이트를 거쳐 SDK 폴백이 압축하게 한다.
    # [해설][주의] handler 로 내려보내기 전 request.override 로 모델과 설정을 원래대로
    # [해설]   되돌린다. 요약 때 쓴 모델/설정이 실제 모델 호출로 새지 않게 하려는
    # [해설]   것이다.
    def wrap_model_call(  # ty: ignore[invalid-method-override]  # delegates auto summarizer
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Run `PreCompact` before synchronous automatic summarization.

        Returns:
            The model response.
        """
        original = request
        call_model = partial(
            super().wrap_model_call,
            handler=lambda updated: handler(
                updated.override(
                    model=original.model, model_settings=original.model_settings
                )
            ),
        )
        summarization = self._summarization_for_runtime(request.runtime)
        request = self._summarization_request(request, summarization)
        prepared = self._auto_compaction_request(request, summarization)
        if prepared is not None:
            if not self._pre_auto_compact(prepared):
                return call_model(prepared)
            return summarization.wrap_model_call(request, call_model)

        overflow_gated = False

        def gated_handler(next_request: ModelRequest) -> ModelResponse:
            nonlocal overflow_gated
            try:
                return call_model(next_request)
            except ContextOverflowError as overflow:
                if not overflow_gated:
                    overflow_gated = True
                    self._pre_overflow_compact(next_request, overflow, summarization)
                raise

        try:
            return summarization.wrap_model_call(request, gated_handler)
        except _AutoCompactionBlockedError as blocked:
            raise blocked.overflow from None

    @staticmethod
    # [해설] 자동 압축의 아카이브 덧붙이기를 다른 압축 경로와 직렬화하기 위해 세션별 락
    # [해설]   안에서 SDK 요약기를 호출한다.
    async def _awrap_with_archive_lock(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        summarization: SummarizationMiddleware,
    ) -> ModelResponse | ExtendedModelResponse:
        """Serialize an automatic archive append with other compactions.

        Args:
            request: The pending model request.
            handler: The downstream model handler.
            summarization: Request-local summarizer for this runtime. Passed in
                rather than read from `self._summarization` so a summarization
                override cannot be bypassed; `@staticmethod` makes that
                structural rather than a convention.

        Returns:
            The wrapped model response.
        """
        session_id = summarization._get_session_id(request.state)
        async with _archive_lock(session_id):
            return await summarization.awrap_model_call(request, handler)

    # [해설] wrap_model_call 의 비동기판(서버는 비동기로 동작). 요약기 선택을 to_thread
    # [해설]   로 넘기고 아카이브 락을 잡는 점만 다르다.
    async def awrap_model_call(  # ty: ignore[invalid-method-override]  # delegates auto summarizer
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Run `PreCompact` before asynchronous automatic summarization.

        Returns:
            The model response.
        """
        original = request
        call_model = partial(
            super().awrap_model_call,
            handler=lambda updated: handler(
                updated.override(
                    model=original.model, model_settings=original.model_settings
                )
            ),
        )
        summarization = await asyncio.to_thread(
            self._summarization_for_runtime, request.runtime
        )
        request = self._summarization_request(request, summarization)
        prepared = self._auto_compaction_request(request, summarization)
        if prepared is not None:
            if not self._pre_auto_compact(prepared):
                return await call_model(prepared)
            return await self._awrap_with_archive_lock(
                request, call_model, summarization
            )

        overflow_gated = False

        async def gated_handler(next_request: ModelRequest) -> ModelResponse:
            nonlocal overflow_gated
            try:
                return await call_model(next_request)
            except ContextOverflowError as overflow:
                if not overflow_gated:
                    overflow_gated = True
                    self._pre_overflow_compact(next_request, overflow, summarization)
                raise

        try:
            return await self._awrap_with_archive_lock(
                request, gated_handler, summarization
            )
        except _AutoCompactionBlockedError as blocked:
            raise blocked.overflow from None

    # [해설] 모델이 compact_conversation 도구를 동기로 부를 때의 구현. 요청 단위
    # [해설]   요약기를 쓰는 임시 호스트에 SDK 구현을 위임한다. 동기 경로는 asyncio 락을
    # [해설]   못 잡는다.
    def _run_compact(self, runtime: ToolRuntime[Any, Any]) -> Command:
        """Run model-initiated compaction with a request-local summarizer.

        Takes no archive lock, unlike `_arun_compact`: `_archive_lock` is an
        asyncio lock, so only the async path can serialize. This matches the
        SDK's sync behavior.

        Args:
            runtime: The tool runtime for this compaction.

        Returns:
            The compact tool's state command.
        """
        tool = self._compact_host(self._summarization_for_runtime(runtime))
        return tool._run_compact(runtime)

    # [해설] compact_conversation 의 비동기판. 세션별 아카이브 락 안에서 SDK 구현을
    # [해설]   호출한다.
    async def _arun_compact(self, runtime: ToolRuntime[Any, Any]) -> Command:
        """Serialize model-initiated compaction with a request-local summarizer.

        Args:
            runtime: The tool runtime for this compaction.

        Returns:
            The compact tool's state command.
        """
        summarization = await asyncio.to_thread(
            self._summarization_for_runtime, runtime
        )
        session_id = summarization._get_session_id(runtime.state)
        tool = self._compact_host(summarization)
        async with _archive_lock(session_id):
            return await tool._arun_compact(runtime)

    @staticmethod
    # [해설][SDK] SDK 의 _run_compact 는 self._summarization 을 읽으므로, 요청 단위
    # [해설]   요약기를 가리키는 일회용 SummarizationToolMiddleware 를 만들어 구현을
    # [해설]   빌려 쓴다. system_prompt=None 인 이유는 이 인스턴스가 모델 호출을 감싸지
    # [해설]   않기 때문이다.
    def _compact_host(
        summarization: SummarizationMiddleware,
    ) -> SummarizationToolMiddleware:
        """Host the SDK's compact implementation over a chosen summarizer.

        The SDK's `_run_compact` reads `self._summarization`, so a throwaway
        host is the only way to point it at the request-local summarizer.

        Args:
            summarization: Summarizer the compaction must run on.

        Returns:
            A middleware instance used only for its compact implementation.
                `system_prompt` is `None` because only `wrap_model_call` reads
                it and this instance never wraps a call.
        """
        return SummarizationToolMiddleware(summarization, system_prompt=None)

    # [해설][SDK] SDK 가 제공하는 compact_conversation 도구를 dcode 버전으로 교체한다.
    # [해설]   같은 이름의 StructuredTool 이라 모델에게는 똑같이 보이고, 내부에서 위의
    # [해설]   _run_compact/_arun_compact 를 호출한다. agent.py 의 _add_interrupt_on 이
    # [해설]   이 도구를 승인 대상에 넣는다.
    def _create_compact_tool(self) -> StructuredTool:
        """Create the CLI variant of `compact_conversation`.

        Returns:
            The model-initiated compaction tool.
        """
        middleware = self

        def sync_compact(runtime: ToolRuntime[Any, Any]) -> Command:
            return middleware._run_compact(runtime)

        async def async_compact(runtime: ToolRuntime[Any, Any]) -> Command:
            return await middleware._arun_compact(runtime)

        return StructuredTool.from_function(
            name="compact_conversation",
            description=(
                "Compact the conversation by summarizing older messages into "
                "a concise summary. Use this proactively when the conversation "
                "is getting long to free up context window space."
            ),
            func=sync_compact,
            coroutine=async_compact,
        )

    @staticmethod
    # [해설] 사용자가 /model 로 바꾼 모델 기준으로 임계 계산을 하도록 요청의
    # [해설]   model/model_settings 를 요약기의 모델 기준으로 덮어쓴다(실제 호출 모델은
    # [해설]   바뀌지 않음).
    def _summarization_request(
        request: ModelRequest, summarization: SummarizationMiddleware
    ) -> ModelRequest:
        """Prepare selected-model budget checks without changing downstream selection.

        Returns:
            A request using the cached summarizer's main model and runtime settings.
        """
        config = _runtime_model_config(request.runtime)
        if not config.model_spec:
            return request
        return request.override(
            model=summarization.model,
            model_settings={**request.model_settings, **config.model_params},
        )

    # [해설] 요청 단위 요약기 결정. 런타임에서 모델이나 요약 모델이 바뀌지 않았으면 시작
    # [해설]   시점의 요약기를 그대로 쓴다.
    # [해설][흐름] 1) 런타임 모델 설정과 요약 모델 스펙 해석(INHERIT 센티널이면 메인
    # [해설]   상속) 2) 둘 다 없으면 기본 요약기 3) 설정 JSON 을 캐시 키로 1칸 캐시 확인
    # [해설]   4) 새 모델 생성(create_model; 환경 적용) 5) 새로 만든 모델에만 컨텍스트
    # [해설]   한도 프로필을 합성(공유 모델 변형 방지) 6)
    # [해설]   create_summarization_middleware(model, backend) 로 새 SDK 요약기 생성 7)
    # [해설]   요약 전용 모델이면 지연 설치, 아니면 재시도만 설치 8) 캐시에 저장.
    # [해설][주의] 생성자에는 _ArchiveReadGuard 가 아닌 '진짜 composite 백엔드'를
    # [해설]   넘긴다. SDK 는 __init__ 에서 isinstance(backend, CompositeBackend) 일
    # [해설]   때만 artifacts_root 를 읽어 아카이브 접두사를 정하므로, 가드를 넘기면
    # [해설]   접두사가 / 가 되어 대화 이력이 사용자 프로젝트에 써질 수 있다.
    def _summarization_for_runtime(
        self, runtime: _HasRunContext
    ) -> SummarizationMiddleware:
        """Build a summarizer when the runtime overrides either model.

        Args:
            runtime: Runtime carrying the current `CLIContext`.

        Returns:
            The startup summarizer when neither a runtime model nor a
                summarization model is selected. Otherwise a summarizer over
                the same configured backend whose thresholds and token counting
                track the main model, while summary generation uses the
                summarization override when one is set -- so the two can be
                different models.
        """
        config = _runtime_model_config(runtime)
        summary_model_spec = config.summarization_model_spec
        if summary_model_spec == INHERIT_SUMMARIZATION_MODEL:
            summary_model_spec = None
        elif summary_model_spec is None:
            summary_model_spec = self._summarization_model_spec
        if not config.model_spec and not summary_model_spec:
            return self._summarization

        cache_key = json.dumps(
            [
                config.model_spec,
                summary_model_spec,
                config.model_params,
                config.profile_overrides,
                config.context_limit,
            ],
            sort_keys=True,
            default=repr,
        )
        cached = self._runtime_summarization
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        from deepagents_code.config import create_model

        if config.model_spec:
            from deepagents_code.config import use_environment

            with use_environment(self._environ):
                model = create_model(
                    config.model_spec,
                    extra_kwargs=config.model_params or None,
                    profile_overrides=config.profile_overrides or None,
                    cli_max_retries=self._cli_max_retries,
                ).model
        else:
            model = self._summarization.model
        # Only a freshly built model may have its profile mutated. On the
        # `else` branch above, `model` is the shared startup summarizer's own
        # instance, so applying a per-request limit here would leak into every
        # later turn. The limit is also derived from that same profile, so
        # applying it there would be a no-op anyway.
        context_limit = config.context_limit if config.model_spec else None
        if context_limit is not None:
            profile = getattr(model, "profile", None)
            native = (
                profile.get("max_input_tokens") if isinstance(profile, dict) else None
            )
            if native != context_limit:
                merged = (
                    {**profile, "max_input_tokens": context_limit}
                    if isinstance(profile, dict)
                    else {"max_input_tokens": context_limit}
                )
                try:
                    model.profile = merged  # ty: ignore[invalid-assignment]
                except AttributeError, TypeError, ValueError:
                    logger.warning(
                        "Could not apply runtime context limit %d to the offload "
                        "model profile; using its resolved profile",
                        context_limit,
                        exc_info=True,
                    )
        # Never pass the `_ArchiveReadGuard` wrapper to the constructor: the SDK
        # resolves the archive prefix once in `__init__` via
        # `backend.artifacts_root if isinstance(backend, CompositeBackend)`, and
        # the guard is not a `CompositeBackend`, so that check would fall back
        # to a `/` prefix. The archive write would then miss the
        # `conversation_history` route and land in the default backend --
        # silently writing into the user's project tree. An `artifacts_root`
        # passthrough on the guard would not help; the `isinstance` is what
        # fails.
        #
        # This is a forward-looking constraint on the *constructor argument*,
        # not a bug being fixed: the previous code also passed the real
        # composite backend here and only swapped `_backend` for the guard
        # afterwards, so the prefix was correct then too. `_PendingArchive`
        # applies the guard separately when writing.
        summarization = create_summarization_middleware(
            model, self._summarization._backend
        )
        if summary_model_spec:
            _install_lazy_summary_model(
                summarization,
                summary_model_spec,
                self._cli_max_retries,
                self._environ,
            )
        else:
            _install_summary_model_retries(summarization)
        self._runtime_summarization = (cache_key, summarization)
        return summarization

    # [해설] 서버 /offload 가 쓰는 '상태 전용 계획' 단계. 파일은 쓰지 않고 요약만
    # [해설]   만든다.
    # [해설][흐름] 1) 요약기 선택 2) 이벤트가 이미 적용된 유효 메시지 계산 3) cutoff
    # [해설]   결정(handoff 면 전부, 아니면 SDK 규칙) 4) cutoff 0 이면 None 5) 절대
    # [해설]   cutoff 로 변환하고 이전 이벤트의 cutoff 를 넘지 못하면(요약의 요약만 다시
    # [해설]   하게 되므로) 모델 호출 전에 None 6) 요약 대상을 분리해 요약 생성 7) 세션
    # [해설]   ID 와 아카이브 계획(_PendingArchive) 구성.
    # [해설][주의] 도구 경로와 달리 실패를 ToolMessage 가 아니라 예외로 올린다(서버
    # [해설]   연산에는 도구 노드가 없으므로). handoff 는 새 스레드를 위한 전체 요약이라
    # [해설]   session_id 를 새로 만들고 원본 전체 메시지를 아카이브하며 원본 cutoff 는
    # [해설]   올리지 않는다.
    async def _aplan_forced_compaction_update(
        self,
        state: _OffloadState,
        runtime: _HasRunContext,
        *,
        handoff: bool = False,
    ) -> _ForcedCompactionPlan | None:
        """Summarize forced-compaction history without writing its archive.

        Unlike the tool paths, this raises on failure instead of returning a
        `ToolMessage`: the server operation has no tool node to carry one, so it
        converts the exception into a client-visible error. Any summarizer,
        or planning failure therefore propagates out of this method rather than
        being folded into the return value.

        Args:
            state: Checkpointed conversation and prior summarization event.
            runtime: Run context carrier used to select the summarizer model.
            handoff: Summarize every message, including the recent tail that
                compaction normally keeps verbatim.

        Returns:
            The checkpoint/archive plan, or `None` when nothing can be compacted.

        Raises:
            ValueError: If called directly with no messages. The owning server
                operation handles an empty thread before reaching this helper.
        """
        summarization = await asyncio.to_thread(
            self._summarization_for_runtime, runtime
        )
        messages = state.get("messages", [])
        event = state.get("_summarization_event")
        if not messages:
            msg = "Offload compaction requires checkpointed conversation messages."
            raise ValueError(msg)
        effective = summarization._apply_event_to_messages(messages, event)
        cutoff = (
            len(effective)
            if handoff
            else summarization._determine_cutoff_index(effective)
        )
        if cutoff == 0:
            return None
        # Resolved once and threaded into the update below: the SDK call is the
        # relative-to-absolute conversion, and computing it twice would let the
        # value checked here drift from the value committed.
        state_cutoff = summarization._compute_state_cutoff(event, cutoff)
        if state_cutoff <= _event_cutoff(event):
            # Degenerate chained compaction: everything eligible is already
            # behind the prior event's cutoff, so only the previous summary
            # would be re-summarized. Committing would spend a model call to
            # replace the in-context summary with a lossier summary-of-a-summary
            # and drop the prior `file_path` from the event -- while the client,
            # which keys its report on the *absolute* cutoff advancing, still
            # reported "nothing to offload". Stop before the model call so the
            # report and the state agree.
            return None
        to_summarize, _ = summarization._partition_messages(effective, cutoff)
        summary = await summarization._acreate_summary(to_summarize)
        # Handoffs snapshot every checkpointed message without advancing the
        # source cutoff. Appending to its compaction archive would repeat history.
        # The server adds source-thread ownership before committing the snapshot.
        session_id = uuid4().hex if handoff else summarization._get_session_id(state)
        archive = _PendingArchive(
            summarization,
            self._summarization._backend,
            # A handoff promises a complete recovery transcript even when an
            # earlier compaction archive failed or was swept. The effective
            # messages contain only its summary and the recent tail.
            messages if handoff else to_summarize,
            session_id,
            summary,
            state_cutoff,
        )
        return _ForcedCompactionPlan(summarization, summary, state_cutoff, archive)

    # [해설] 계획과 아카이브 쓰기를 한 번에 하는 이전 방식의 편의 메서드. HTTP 연산은
    # [해설]   체크포인트를 먼저 예약하려고 계획 메서드를 직접 쓴다.
    # [해설][주의] 아카이브 쓰기가 실패해도 예외를 올리지 않고 error 로그만 남긴 채 요약
    # [해설]   갱신은 진행한다(요약은 컨텍스트에 있고 원문은 체크포인트에 남아 있어 압축
    # [해설]   자체는 유용하므로). 대신 원문이 저장소에서 복구되지 않는다는 로그를
    # [해설]   남기며, 클라이언트는 파일 경로가 없는 것을 오류로 알린다.
    async def arun_forced_compaction_update(
        self, state: _OffloadState, runtime: _HasRunContext
    ) -> dict[str, Any] | None:
        """Run forced compaction and persist its archive immediately.

        The HTTP operation uses `_aplan_forced_compaction_update` directly so
        it can reserve the checkpoint before this side effect. Direct callers
        retain the historical all-in-one behavior.

        Returns:
            The completed state update, or `None` when nothing can be compacted.
        """
        plan = await self._aplan_forced_compaction_update(state, runtime)
        if plan is None:
            return None
        try:
            async with _archive_lock(plan.archive.session_id):
                append = await plan.archive.write()
        except Exception:
            logger.exception("/offload archive append failed")
            append = None
        if append is None:
            # `_aoffload_to_backend` catches every write failure and returns
            # `None`, which also swallows `_ArchiveReadGuard`'s deliberate
            # "refusing to overwrite existing history" `RuntimeError`. Its own
            # log names neither the thread nor this call site, so record one
            # here that does.
            #
            # Not raised: the compaction is still useful (the summary is
            # in-context and the raw messages remain in the checkpoint), and the
            # client reports the missing archive to the user as an error rather
            # than a success. Escalating here would change that policy, not just
            # its observability.
            logger.error(
                "/offload compacted %d messages but the archive write failed; "
                "those messages are not recoverable from storage",
                len(plan.archive.messages),
            )
        return plan.update(append.path if append is not None else None)

    @staticmethod
    # [해설] 상태 갱신 dict 를 만든다: _summarization_event(절대 cutoff, 요약 메시지,
    # [해설]   아카이브 경로)와 _summarization_session_id. 다음 /offload 가 같은
    # [해설]   아카이브에 덧붙이도록 세션 ID 를 보존한다.
    # [해설] SDK _build_new_messages_with_path 의 첫 메시지가 HumanMessage 가 아니면
    # [해설]   TypeError 로 실패시켜, 스키마를 어긴 이벤트가 체크포인트에 저장되는 것을
    # [해설]   막는다.
    def _forced_compaction_update(
        summarization: SummarizationMiddleware,
        summary: str,
        file_path: str | None,
        state_cutoff: int,
        session_id: str,
    ) -> dict[str, Any]:
        """Build the state-only result used by the server `/offload` operation.

        The returned dict carries the `_summarization_event` payload plus the
        session id, so it is not itself a `SummarizationEvent` and cannot be
        annotated as one. That the event's `summary_message` really is the
        `HumanMessage` the channel expects is therefore enforced at runtime by
        the `isinstance` check below, not by the type checker.

        Args:
            summarization: SDK summarization middleware building the message.
            summary: Generated summary text.
            file_path: Archive path, or `None` when the write failed.
            state_cutoff: **Absolute** cutoff index, already converted from the
                relative one by `_compute_state_cutoff`. Taken pre-resolved
                rather than converted here so the caller's no-advance check and
                the committed value cannot disagree.
            session_id: The id that named the history file. Persisted under
                `_summarization_session_id` (mirroring the SDK's compact and
                auto-summarize paths) so a later offload appends to the same
                archive instead of minting a fresh file.

        Returns:
            The summarization state update.

        Raises:
            TypeError: If the summarizer's first message is not the
                `HumanMessage` the event schema declares.
        """
        summary_message = summarization._build_new_messages_with_path(
            summary, file_path
        )[0]
        if not isinstance(summary_message, HumanMessage):
            # `_build_new_messages_with_path` is annotated `list[AnyMessage]`
            # but documents (and the SDK's own call site assumes, with a type
            # suppression) that element 0 is the summary `HumanMessage`. Check
            # rather than suppress: the node turns this into a visible
            # "Compaction failed" instead of checkpointing an event whose
            # `summary_message` violates its own schema.
            msg = (
                "Summarizer returned a "
                f"{type(summary_message).__name__} summary message; expected "
                "HumanMessage."
            )
            raise TypeError(msg)
        return {
            "_summarization_event": {
                # Absolute, not relative: a second `/offload` on the same thread
                # reads this back as its base.
                "cutoff_index": state_cutoff,
                "summary_message": summary_message,
                "file_path": file_path,
            },
            "_summarization_session_id": session_id,
        }


# [해설][SDK] 팩토리. create_summarization_tool_middleware(model, backend)가 만든 SDK
# [해설]   엔진과 시스템 프롬프트 조각을 받아 CLICompactionMiddleware 로 감싼다. 요약
# [해설]   모델 재시도는 즉시 dcode 정책으로 교체한다. agent.py 의 create_cli_agent 가
# [해설]   호출한다.
def _create_cli_compaction_middleware(
    model: str | BaseChatModel,
    backend: BackendProtocol,
    *,
    cli_max_retries: int | None = None,
    summarization_model_spec: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> CLICompactionMiddleware:
    """Create the dcode compaction middleware from the SDK configuration.

    Args:
        model: Startup model or model specification.
        backend: Agent backend used for archive persistence.
        cli_max_retries: Explicit `--max-retries` value for runtime rebuilds.
        summarization_model_spec: Startup model spec to use only for summaries.
        environ: Workspace environment retained for lazy model construction.

    Returns:
        CLI compaction middleware with the SDK's model-aware defaults.
    """
    sdk_middleware = create_summarization_tool_middleware(model, backend)
    _install_summary_model_retries(sdk_middleware._summarization)
    return CLICompactionMiddleware(
        sdk_middleware._summarization,
        system_prompt=sdk_middleware.system_prompt,
        cli_max_retries=cli_max_retries,
        summarization_model_spec=summarization_model_spec,
        environ=environ,
    )


# [해설] /offload 훅 디스패치용 '가짜 도구 호출 ID' 를 결정적으로 만들 때 쓰는 UUID5
# [해설]   네임스페이스.
_OFFLOAD_CALL_NAMESPACE = uuid5(NAMESPACE_URL, "https://deepagents/offload/forced-call")
"""Namespace for deriving the `/offload` hook dispatch's forced tool-call id."""


# [해설][주의] 겉보기엔 사소한 ID 생성이지만 두 요구가 충돌한다. (1) 훅 응답으로 연산이
# [해설]   '처음부터 다시 실행'되므로 요청과 재개에서 같은 ID 여야 한다. (2) 같은 턴의
# [해설]   두 번째 /offload 가 첫 번째의 거부 결정을 재사용하지 않도록 시도마다 달라야
# [해설]   한다. configurable.checkpoint_ns 가 둘 다 만족하며, offload_api 가
# [해설]   dcode_offload:{operation_id} 로 만든다. 그 형식이나 operation_id 재사용
# [해설]   방식을 바꾸면 훅이 설정된 사용자에게만 /offload 재개가 조용히 깨진다.
def _forced_offload_call_id() -> str:
    """Return the tool-call id the `/offload` hook dispatch runs against.

    Two requirements pull in opposite directions, and both are load-bearing:

    *Stable across resumes.* `ServerHooksMiddleware` folds this id into its hook
    `invocation_id`, and answering a hook request re-executes the operation
    **from the top** rather than resuming mid-coroutine. An id minted fresh here
    would therefore differ between the request and the resume, and
    `parse_hook_resume_value` rejects a mismatched invocation id as fatal ("the
    client answered a different request") -- which would break `/offload` for
    exactly those users who have a `PreCompact`/`PreToolUse` hook configured,
    and make the client's whole fulfill/resume loop unreachable.

    *Distinct across attempts.* The client memoizes fulfillments by
    `(snapshot_id, invocation_id)` for the session, and the hook `prompt_id`
    only rotates on user-prompt submit. A constant would make two `/offload`s
    within one turn collide and replay the first attempt's decision -- including
    a denial -- instead of re-running the user's hook.

    `configurable.checkpoint_ns` satisfies both, and
    `offload_api._execute_offload` is the invariant's owner: it derives the
    namespace as `dcode_offload:{operation_id}` from the client's per-attempt
    `operation_id`, which `RemoteAgent.aoffload` mints once and reuses across
    every resume round of that attempt. Changing either that namespace format or
    the client's reuse of `operation_id` breaks hook resume, silently, for hook
    users only.

    Returns:
        An id stable across this attempt's resume rounds and distinct from every
            other attempt's.
    """
    try:
        config = get_config()
    except RuntimeError:
        # No runnable context at all -- a direct call outside a graph run.
        # Nothing can interrupt or resume such a call, so uniqueness is the only
        # property left to preserve, and the `uuid4()` fallback is correct.
        return f"offload-precompact-{uuid4()}"
    configurable = config.get("configurable")
    namespace = (
        configurable.get("checkpoint_ns") if isinstance(configurable, dict) else None
    )
    if not namespace:
        # A runnable context *without* a usable `checkpoint_ns` is a different
        # situation entirely, and a silent fallback here is the failure mode
        # this function exists to prevent: the id would differ between the
        # request and the resume, `parse_hook_resume_value` would reject the
        # mismatch as fatal, and `/offload` would die with "the client answered
        # a different request" -- but only for users with hooks configured, and
        # with nothing in the logs pointing here. Say so loudly.
        logger.warning(
            "Deriving the /offload hook call id inside a run but "
            "`configurable.checkpoint_ns` is %r; falling back to a random id. "
            "Configured PreCompact/PreToolUse hooks will fail to resume this "
            "run. This usually means LangGraph moved or renamed the key.",
            namespace,
        )
        return f"offload-precompact-{uuid4()}"
    return f"offload-precompact-{uuid5(_OFFLOAD_CALL_NAMESPACE, namespace)}"


# [해설] 서버 소유 /offload 연산의 본체. HTTP 경계(offload_api.py)가 체크포인트 접근과
# [해설]   저장을 맡고, 이 클래스는 훅 실행 -> 압축 계획 -> 결과 구성까지를 맡는다.
# [해설]   에이전트가 쓰는 것과 같은 compaction/hooks 인스턴스를 공유해 정책이 어긋나지
# [해설]   않는다.
# [해설] create_cli_agent 가 OffloadOperation(compaction_middleware,
# [해설]   server_hooks_middleware) 를 만들어 attach_offload_operation 으로 백엔드에
# [해설]   게시한다.
class OffloadOperation:
    """Compact checkpoint state behind dcode's server-owned HTTP boundary."""

    # [해설] 압축 미들웨어와 서버 훅 미들웨어(에이전트가 쓰는 것과 같은 인스턴스)를
    # [해설]   보관한다.
    def __init__(
        self,
        compaction: CLICompactionMiddleware,
        hooks: ServerHooksMiddleware,
    ) -> None:
        """Initialize the operation with the agent's own policy and hooks.

        Args:
            compaction: Compaction middleware bound to the agent backend.
            hooks: Server hook middleware used by the interactive graph.
        """
        self._compaction = compaction
        self._hooks = hooks

    _result = staticmethod(unchanged_offload_result)

    # [해설] 강제 압축을 'compact_conversation 도구 호출'처럼 가장해
    # [해설]   ServerHooksMiddleware 의 aafter_model 에 통과시켜 PreCompact/PreToolUse
    # [해설]   훅을 실행한다. 반환: 거부/실패 상태와 사유, 허용이면 None.
    # [해설][설계] 훅 결과 채널이 없거나 모양이 다르면 '허용'으로 간주하지 않고 실패로
    # [해설]   닫는다(fail closed). 기본값 처리(.get(..., {}))를 하면 사용자의 거부를
    # [해설]   '결과 없음'으로 읽어 아무 로그 없이 압축해 버릴 수 있기 때문이다.
    async def _run_hooks(
        self, runtime: Runtime[CLIContextSchema]
    ) -> tuple[Literal["denied", "failed"], str] | None:
        """Dispatch the forced call through `PreCompact` and `PreToolUse`.

        Returns:
            Failure status and detail, or `None` when hooks allow compaction.

        Raises:
            HookTransportInterruptError: If the client must fulfill a hook request.
        """
        try:
            forced_call_id = _forced_offload_call_id()
            hook_update = await self._hooks.aafter_model(
                cast(
                    "Any",
                    {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "compact_conversation",
                                        "args": {"force": True},
                                        "id": forced_call_id,
                                    }
                                ],
                            )
                        ]
                    },
                ),
                cast("Runtime[Any]", runtime),
            )
        except HookTransportInterruptError:
            raise
        except Exception as exc:
            logger.exception("/offload hook dispatch failed")
            return "failed", f"Offload hooks failed: {type(exc).__name__}: {exc}"

        # Fail closed on a missing channel rather than defaulting to allow.
        # `_after_model` always returns this key, so its absence means the
        # channel, the id derivation, or the outcome shape drifted -- and a
        # `.get(..., {})` chain would read a user's *denial* as "no outcome" and
        # compact straight through it, with no log.
        outcomes = hook_update.get(_PRE_TOOL_STATE_KEY)
        if not isinstance(outcomes, dict):
            logger.error(
                "Compaction hooks returned no %s channel for /offload; refusing "
                "rather than treating a possible denial as an allow",
                _PRE_TOOL_STATE_KEY,
            )
            return "failed", (
                "Could not read the compaction hook decision; offload refused."
            )
        outcome = outcomes.get(forced_call_id) or {}
        if outcome.get("behavior") == "deny":
            reason = outcome.get("reason") or "Blocked by a compaction hook"
            return "denied", str(reason)
        if outcome.get("context"):
            logger.warning(
                "Discarding PreToolUse additionalContext for the /offload "
                "operation; no tool result or model turn exists to carry it"
            )
        return None

    # [해설][흐름] 1) 체크포인트 메시지/이벤트로 유효 메시지와 압축 전 토큰 추정 2)
    # [해설]   메시지가 없으면 empty 3) 훅 실행(거부/실패면 상태 변경 없이 결과만) 4)
    # [해설]   강제 압축 계획 수립(예외는 failed 로 변환, 훅 전송 중단 예외는 그대로
    # [해설]   전파) 5) 계획이 None 이면 noop 6) handoff 면 summarized 결과와 아카이브만
    # [해설]   돌려주고 원본 상태는 건드리지 않음 7) 일반이면 갱신을 구성하고 전후 토큰,
    # [해설]   이동한 메시지 수를 계산해 compacted 결과 반환.
    # [해설][주의] 반환하는 상태 갱신은 SDK 가 돌려준 dict 전체가 아니라 허용된 두
    # [해설]   채널(_summarization_event, _summarization_session_id)만 복사한다. 이후
    # [해설]   SDK 가 키를 추가해도(특히 messages) 체크포인트 쓰기로 새지 않게 하는
    # [해설]   방어다. 아카이브 파일은 이 메서드가 아니라 호출자가 체크포인트 예약 후에
    # [해설]   쓴다.
    async def execute(
        self,
        state: _OffloadState,
        runtime: Runtime[CLIContextSchema],
        *,
        handoff: bool = False,
    ) -> OffloadExecution:
        """Run one offload against server-read checkpoint state.

        Args:
            state: Checkpointed conversation and prior summarization event.
            runtime: Runtime context for hooks and the summarizer model.
            handoff: Summarize every message for a new thread. The caller
                commits only cost and the transcript, not the summary event.

        Returns:
            State update for the server to persist and the typed client result.

        Raises:
            HookTransportInterruptError: If the client must fulfill a hook request.
        """
        from langchain_core.messages.utils import count_tokens_approximately

        messages = list(state.get("messages", []))
        event = state.get("_summarization_event")
        effective = self._compaction._summarization._apply_event_to_messages(
            messages, event
        )
        tokens_before = count_tokens_approximately(effective)
        if not messages:
            result = self._result("empty", messages=0, tokens=0)
            return OffloadExecution({}, result)

        hook_failure = await self._run_hooks(runtime)
        if hook_failure is not None:
            status, error = hook_failure
            result = self._result(
                status,
                messages=max(0, len(messages) - _event_cutoff(event)),
                tokens=tokens_before,
                error=error,
            )
            return OffloadExecution({}, result)

        try:
            plan = await self._compaction._aplan_forced_compaction_update(
                state, runtime, handoff=handoff
            )
        except HookTransportInterruptError:
            raise
        except Exception as exc:
            logger.exception("forced /offload compaction failed")
            result = self._result(
                "failed",
                messages=max(0, len(messages) - _event_cutoff(event)),
                tokens=tokens_before,
                error=f"Compaction failed: {type(exc).__name__}: {exc}",
            )
            return OffloadExecution({}, result)

        if plan is None:
            result = self._result(
                "noop",
                messages=max(0, len(messages) - _event_cutoff(event)),
                tokens=tokens_before,
            )
            return OffloadExecution({}, result)

        from deepagents_code.offload import offload_storage_is_ephemeral

        if handoff:
            result = self._result(
                "summarized",
                messages=max(0, len(messages) - _event_cutoff(event)),
                tokens=tokens_before,
            )
            result["archive_ephemeral"] = offload_storage_is_ephemeral()
            return OffloadExecution({}, result, plan.archive)

        update = plan.update(None)
        new_event = update["_summarization_event"]
        new_cutoff = _event_cutoff(new_event)
        prior_cutoff = _event_cutoff(event)
        effective_after = self._compaction._summarization._apply_event_to_messages(
            messages, new_event
        )
        file_path = new_event.get("file_path")
        result: OffloadResult = {
            "status": "compacted",
            "messages_offloaded": max(0, new_cutoff - prior_cutoff),
            "messages_kept": max(0, len(messages) - new_cutoff),
            "tokens_before": tokens_before,
            "tokens_after": count_tokens_approximately(effective_after),
            "archive_path": file_path if isinstance(file_path, str) else None,
            "archive_ephemeral": offload_storage_is_ephemeral(),
            "error": None,
        }
        # Forward only the permitted channels rather than the SDK's whole update
        # dict, so a future SDK change that adds keys (`messages` above all)
        # cannot reach the checkpoint write through this operation.
        return OffloadExecution(
            {
                "_summarization_event": new_event,
                "_summarization_session_id": update["_summarization_session_id"],
            },
            result,
            plan.archive,
        )
