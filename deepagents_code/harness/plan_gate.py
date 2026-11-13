"""Plan gate: code changes only after a reviewed and approved plan.

The gate is a middleware around tool calls. Until the thread has an approved
plan it refuses every tool that can change the workspace, except writes to the
thread's plan file. Approval is a three-step path the model cannot shortcut:

1. The model writes the plan file.
2. `request_plan_review` checks the plan's structure and runs the AI reviewer.
3. `request_plan_approval` asks the human. Only that answer sets the phase to
   approved.

After approval, file edits are limited to the plan's "수정 대상" entries, and a
plan file whose content no longer matches the approved fingerprint sends the
thread back to planning.

An approval covers one user request. When the next request arrives the
thread is back in planning, and the approved plan is kept as a dated copy.

The phase lives in graph state, so it is per thread and survives a resume.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, cast, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ToolCallRequest,
    TracePolicy,
    omit_payload,
)
from langchain.tools import InjectedToolCallId, ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.config import get_config
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from deepagents_code.harness import (
    experience,
    improvement,
    plan as plan_rules,
    plan_report,
    trace_log,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path

    from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
    from langgraph.runtime import Runtime

    from deepagents_code.harness.improvement import SelfImprover
    from deepagents_code.harness.plan_report import (
        Comparison,
        TurnActivity,
        WorkReview,
        WorkReviewer,
    )
    from deepagents_code.harness.plan_review import PlanReview, PlanReviewer

logger = logging.getLogger(__name__)

Phase = Literal["planning", "reviewed", "approved"]

PHASE_PLANNING: Phase = "planning"
PHASE_REVIEWED: Phase = "reviewed"
PHASE_APPROVED: Phase = "approved"

REVIEW_TOOL = "request_plan_review"
APPROVAL_TOOL = "request_plan_approval"

MAX_REVIEW_ROUNDS = 3
"""AI review rounds per plan before the remaining findings go to the human."""

FILE_TOOLS = frozenset({"write_file", "edit_file", "delete"})
"""Gated tools that name the file they change in `file_path`."""

GATED_TOOLS = FILE_TOOLS | frozenset(
    {
        "execute",
        "js_eval",
        "start_async_task",
        "task",
        "update_async_task",
    }
)
"""Tools that can change the workspace, directly or through another agent."""

APPROVE_CHOICE = "승인"
REVISE_CHOICE = "수정 요청"

_REQUEST_CHAR_LIMIT = 6000
_EARLIER_REQUESTS = 5
_EARLIER_REQUEST_CHAR_LIMIT = 400
_PLAN_TEMPLATE = "\n".join(f"## {title}" for title in plan_rules.REQUIRED_SECTIONS)


class PlanGateState(AgentState[Any]):
    """Per-thread plan state. All channels are private to this graph."""

    _plan_phase: Annotated[NotRequired[Phase], PrivateStateAttr]
    """Current phase. Absent means planning."""

    _plan_reviewed_hash: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Fingerprint of the plan text the AI review cleared for approval."""

    _plan_approved_hash: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Fingerprint of the plan text the human approved."""

    _plan_review_rounds: Annotated[NotRequired[int], PrivateStateAttr]
    """AI review rounds spent since the last approval."""

    _plan_review_findings: Annotated[NotRequired[list[str]], PrivateStateAttr]
    """Findings of the latest review that the plan has not resolved."""

    _plan_turn_started: Annotated[NotRequired[float], PrivateStateAttr]
    """Unix time at which the current turn started, for the turn metrics."""


def thread_id_of(runtime: object) -> str:
    """Return the thread a tool call or model call belongs to.

    Args:
        runtime: Tool or middleware runtime.

    Returns:
        Thread id, or `default` when the run is not bound to a thread.
    """
    config = getattr(runtime, "config", None)
    if isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id
    execution_info = getattr(runtime, "execution_info", None)
    thread_id = getattr(execution_info, "thread_id", None)
    return thread_id if isinstance(thread_id, str) and thread_id else "default"


def _hook_thread_id(runtime: object) -> str:
    """Return the thread a middleware hook is running for.

    Hook runtimes do not carry the run config, so it is read from the active
    runnable context first.

    Args:
        runtime: Middleware runtime.

    Returns:
        Thread id, or `default` when the run is not bound to a thread.
    """
    try:
        config = get_config()
    except RuntimeError:
        return thread_id_of(runtime)
    thread_id = config.get("configurable", {}).get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return thread_id_of(runtime)


def split_requests(messages: list[Any]) -> tuple[str, tuple[str, ...]]:
    """Separate the request of the current turn from those of earlier turns.

    Args:
        messages: Conversation messages in order.

    Returns:
        The latest human message, and the human messages before it, oldest
            first. Each is cut to a bounded length and only the most recent
            earlier ones are kept.
    """
    parts = [
        " ".join(str(message.text).split())
        for message in messages
        if isinstance(message, HumanMessage)
    ]
    parts = [part for part in parts if part]
    if not parts:
        return "", ()
    earlier = tuple(
        part[:_EARLIER_REQUEST_CHAR_LIMIT] for part in parts[:-1][-_EARLIER_REQUESTS:]
    )
    return parts[-1][:_REQUEST_CHAR_LIMIT], earlier


def _phase_of(state: Mapping[str, Any] | None) -> Phase:
    """Read the phase from state, defaulting to planning.

    Args:
        state: Graph state.

    Returns:
        The stored phase, or planning when it is absent or unknown.
    """
    phase = (state or {}).get("_plan_phase")
    return phase if phase in {PHASE_REVIEWED, PHASE_APPROVED} else PHASE_PLANNING


def _is_gated(request: ToolCallRequest) -> bool:
    """Report whether a tool call needs an approved plan. No disk access.

    Only dcode's own tools are gated, by name. MCP tools are not: what one
    does cannot be told from the call, and treating every unmarked MCP tool
    as a change would also stop the lookups a plan is written from.

    Args:
        request: Tool call.

    Returns:
        `True` for a tool in `GATED_TOOLS`.
    """
    return request.tool_call["name"] in GATED_TOOLS


def _file_argument(request: ToolCallRequest) -> str | None:
    """Return the path a file tool call targets.

    Args:
        request: Tool call.

    Returns:
        The `file_path` argument, or `None` when it is missing or not a string.
    """
    args = request.tool_call.get("args") or {}
    file_path = args.get("file_path")
    return file_path if isinstance(file_path, str) and file_path else None


def _review_record(round_number: int, review: PlanReview, plan_hash: str) -> str:
    """Format one review round for the review record file.

    Args:
        round_number: One-based review round.
        review: Verdict of the round.
        plan_hash: Fingerprint of the reviewed plan text.

    Returns:
        Markdown block to append to the record.
    """
    findings = "\n".join(f"- {item}" for item in review.findings) or "- 없음"
    return (
        f"## 리뷰 {round_number}회차\n\n"
        f"- 시각: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        f"- 판정: {'통과' if review.passed else '수정 필요'}\n"
        f"- 계획서 해시: `{plan_hash}`\n\n"
        f"### 의견\n\n{findings}\n\n"
    )


def _approval_record(decision: str, approver: str, plan_hash: str, note: str) -> str:
    """Format an approval decision for the review record file.

    Args:
        decision: Outcome shown to the reader.
        approver: Who decided.
        plan_hash: Fingerprint of the plan text the decision applies to.
        note: Free-text note from the approver, possibly empty.

    Returns:
        Markdown block to append to the record.
    """
    return (
        f"## 승인 결정\n\n"
        f"- 시각: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        f"- 결정: {decision}\n"
        f"- 결정한 주체: {approver}\n"
        f"- 계획서 해시: `{plan_hash}`\n"
        f"- 의견: {note.strip() or '없음'}\n\n"
    )


def _append(path: Path, text: str) -> None:
    """Append text to a file, creating the folder and a title when needed.

    Args:
        path: File to append to.
        text: Text to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        ""
        if path.exists()
        else f"# 계획 리뷰 기록: {path.stem.removesuffix('.review')}\n\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(header + text)


class PlanGateMiddleware(AgentMiddleware[PlanGateState, ContextT]):
    """Refuse workspace changes until the thread's plan is approved."""

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces; they carry the whole message history."""

    state_schema = PlanGateState

    def __init__(
        self,
        root: Path,
        *,
        reviewer: PlanReviewer | None = None,
        work_reviewer: WorkReviewer | None = None,
        interactive: bool = True,
        allow_ai_approval: bool = False,
        record_experience: bool = False,
        improver: SelfImprover | None = None,
        retry_counter: trace_log.RetryCounter | None = None,
        run_counter: trace_log.RunCounter | None = None,
    ) -> None:
        """Create the gate for one project.

        Args:
            root: Project root. Plans live under it and "수정 대상" entries are
                resolved against it.
            reviewer: AI reviewer. Without one, a structurally valid plan goes
                straight to the approval step.
            work_reviewer: AI reviewer of finished work. Without one, the
                post-work report holds only what the harness collected.
            interactive: Whether a human can answer an approval question.
            allow_ai_approval: Accept a passed AI review as approval when no
                human can answer. Has no effect in interactive runs.
            record_experience: Save each post-work report as an experience
                record in the project's memory folder.
            improver: Looks for an improvement after a turn whose report shows
                failure signals.
            retry_counter: Counter of failed model attempts, filled by
                `ModelRetryCounterMiddleware`. Without one, the turn
                metrics have no retry count.
            run_counter: Counter of every agent's model and tool calls,
                filled by `RunCounterCallback`. Without one, the turn
                metrics count the main agent's messages only.
        """
        super().__init__()
        self._improver = improver
        self._retry_counter = retry_counter
        self._run_counter = run_counter
        self._record_experience = record_experience
        self._root = root
        self._reviewer = reviewer
        self._work_reviewer = work_reviewer
        self._interactive = interactive
        self._allow_ai_approval = allow_ai_approval
        self.tools = [self._build_review_tool(), self._build_approval_tool()]

    # ----- tools ---------------------------------------------------------

    def _build_review_tool(self) -> Any:  # noqa: ANN401  # StructuredTool
        """Build the tool the agent calls to have its plan reviewed.

        Returns:
            The `request_plan_review` tool bound to this middleware.
        """
        gate = self

        @tool(REVIEW_TOOL)
        async def request_plan_review(
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict[str, Any], InjectedState],
            runtime: ToolRuntime[Any, Any],
        ) -> Command[Any]:
            """Check the plan file and have it reviewed before asking for approval.

            Call this after writing or revising the plan file named in the plan
            gate notice. The structure is checked first; a structurally valid
            plan is then reviewed by a separate reviewer. Fix every problem that
            is reported in the plan file and call this tool again. When the
            review passes, call `request_plan_approval`.

            Returns:
                Command carrying the review result and the updated plan phase.
            """
            return await gate._review(tool_call_id, state, runtime)

        return request_plan_review

    def _build_approval_tool(self) -> Any:  # noqa: ANN401  # StructuredTool
        """Build the tool the agent calls to ask the user for approval.

        Returns:
            The `request_plan_approval` tool bound to this middleware.
        """
        gate = self

        @tool(APPROVAL_TOOL)
        async def request_plan_approval(
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict[str, Any], InjectedState],
            runtime: ToolRuntime[Any, Any],
        ) -> Command[Any]:
            """Ask the user to approve the reviewed plan.

            Call this only after `request_plan_review` reported that the review
            passed, and without changing the plan file in between. Approval
            unlocks code changes limited to the files listed under
            "수정 대상". If the user asks for changes, revise the plan file and
            request a review again.

            Returns:
                Command carrying the user's decision and the updated plan phase.
            """
            return await gate._approve(tool_call_id, state, runtime)

        return request_plan_approval

    async def _review(
        self,
        tool_call_id: str,
        state: dict[str, Any],
        runtime: ToolRuntime[Any, Any],
    ) -> Command[Any]:
        """Validate and review the thread's plan.

        Args:
            tool_call_id: Id of the tool call being answered.
            state: Graph state.
            runtime: Tool runtime.

        Returns:
            Command with the result message and phase update.
        """
        path = plan_rules.plan_file(self._root, thread_id_of(runtime))
        text = await asyncio.to_thread(plan_rules.read_text, path)
        if text is None:
            await self._log(
                runtime, "plan.invalid", {"problems": ["계획서 파일 없음"]}, failed=True
            )
            return self._reply(
                tool_call_id,
                REVIEW_TOOL,
                f"계획서 파일이 없습니다: {path}\n"
                f"이 경로에 계획서를 쓴 뒤 다시 요청하세요. 필수 절:\n{_PLAN_TEMPLATE}",
                error=True,
            )
        document = plan_rules.parse_plan(text)
        problems = await asyncio.to_thread(
            plan_rules.validate_plan,
            document,
            self._root,
            memory_ids=tuple(state.get(experience.HITS_STATE_KEY) or ()),
        )
        if problems:
            await self._log(
                runtime, "plan.invalid", {"problems": problems}, failed=True
            )
            listed = "\n".join(f"- {item}" for item in problems)
            return self._reply(
                tool_call_id,
                REVIEW_TOOL,
                "계획서 형식 검증에 실패했습니다. 아래를 고친 뒤 다시 "
                f"요청하세요.\n{listed}",
                error=True,
                update={"_plan_phase": PHASE_PLANNING, "_plan_reviewed_hash": None},
            )
        plan_hash = plan_rules.hash_text(text)
        rounds = int(state.get("_plan_review_rounds") or 0)
        if rounds >= MAX_REVIEW_ROUNDS:
            findings = list(state.get("_plan_review_findings") or [])
            await self._log(
                runtime,
                "plan.review",
                {
                    "round": rounds,
                    "verdict": "limit",
                    "findings": findings,
                    "plan_hash": plan_hash,
                },
                failed=bool(findings),
            )
            return self._reply(
                tool_call_id,
                REVIEW_TOOL,
                f"AI 리뷰 상한({MAX_REVIEW_ROUNDS}회)에 닿았습니다. 남은 의견을 "
                f"사용자에게 함께 보여 줍니다. `{APPROVAL_TOOL}`을 호출하세요.",
                update={
                    "_plan_phase": PHASE_REVIEWED,
                    "_plan_reviewed_hash": plan_hash,
                    "_plan_review_findings": findings,
                },
            )
        review = await self._run_reviewer(state, text, str(path), runtime)
        record = path.with_name(path.stem + plan_rules.REVIEW_SUFFIX)
        await asyncio.to_thread(
            _append, record, _review_record(rounds + 1, review, plan_hash)
        )
        await self._log(
            runtime,
            "plan.review",
            {
                "round": rounds + 1,
                "verdict": review.verdict,
                "findings": list(review.findings),
                "plan_hash": plan_hash,
            },
            failed=not review.passed,
        )
        if review.passed:
            return self._reply(
                tool_call_id,
                REVIEW_TOOL,
                f"AI 리뷰를 통과했습니다({rounds + 1}회차). 계획서를 바꾸지 말고 "
                f"`{APPROVAL_TOOL}`을 호출하세요.",
                update={
                    "_plan_phase": PHASE_REVIEWED,
                    "_plan_reviewed_hash": plan_hash,
                    "_plan_review_rounds": rounds + 1,
                    "_plan_review_findings": [],
                },
            )
        listed = "\n".join(f"- {item}" for item in review.findings)
        return self._reply(
            tool_call_id,
            REVIEW_TOOL,
            "AI 리뷰 결과 수정이 필요합니다"
            f"({rounds + 1}/{MAX_REVIEW_ROUNDS}회차).\n{listed}\n"
            f"계획서 파일을 고친 뒤 `{REVIEW_TOOL}`을 다시 호출하세요.",
            update={
                "_plan_phase": PHASE_PLANNING,
                "_plan_reviewed_hash": None,
                "_plan_review_rounds": rounds + 1,
                "_plan_review_findings": list(review.findings),
            },
        )

    async def _run_reviewer(
        self,
        state: dict[str, Any],
        plan_text: str,
        plan_path: str,
        runtime: ToolRuntime[Any, Any],
    ) -> PlanReview:
        """Run the AI reviewer, or pass the plan when none is configured.

        Args:
            state: Graph state, the source of the user's request.
            plan_text: Plan file content.
            plan_path: Location of the plan file.
            runtime: Tool runtime whose context the reviewer inherits.

        Returns:
            The review verdict.
        """
        from deepagents_code.harness.plan_review import PlanReview

        if self._reviewer is None:
            return PlanReview(verdict="approve", findings=[])
        request, earlier = split_requests(list(state.get("messages") or []))
        return await self._reviewer.review(
            request_text=request,
            plan_text=plan_text,
            plan_path=plan_path,
            project_root=str(self._root),
            earlier_requests=earlier,
            context=runtime.context,
        )

    async def _approve(
        self,
        tool_call_id: str,
        state: dict[str, Any],
        runtime: ToolRuntime[Any, Any],
    ) -> Command[Any]:
        """Ask for approval of the reviewed plan and apply the decision.

        Args:
            tool_call_id: Id of the tool call being answered.
            state: Graph state.
            runtime: Tool runtime.

        Returns:
            Command with the decision message and phase update.
        """
        path = plan_rules.plan_file(self._root, thread_id_of(runtime))
        text = await asyncio.to_thread(plan_rules.read_text, path)
        plan_hash = plan_rules.hash_text(text) if text is not None else None
        reviewed_hash = state.get("_plan_reviewed_hash")
        if plan_hash is None or reviewed_hash is None or plan_hash != reviewed_hash:
            await self._log(
                runtime,
                "plan.approval",
                {"decision": "불가", "reason": "리뷰를 통과한 계획서가 없음"},
                failed=True,
            )
            return self._reply(
                tool_call_id,
                APPROVAL_TOOL,
                "리뷰를 통과한 계획서가 없거나, 리뷰 뒤에 계획서가 바뀌었습니다. "
                f"`{REVIEW_TOOL}`을 먼저 호출하세요.",
                error=True,
                update={"_plan_phase": PHASE_PLANNING, "_plan_reviewed_hash": None},
            )
        record = path.with_name(path.stem + plan_rules.REVIEW_SUFFIX)
        findings = list(state.get("_plan_review_findings") or [])
        if self._interactive:
            approved, note = self._ask_human(tool_call_id, path, findings)
            approver = "사용자"
        elif self._allow_ai_approval and findings:
            # Findings left at the review limit mean no review passed. The
            # option accepts a passed review; without a human, nothing else
            # may approve the plan.
            await self._log(
                runtime,
                "plan.approval",
                {
                    "decision": "불가",
                    "reason": "리뷰 상한에 닿아 남은 의견이 있음",
                    "findings": findings,
                },
                failed=True,
            )
            listed = "\n".join(f"- {item}" for item in findings)
            return self._reply(
                tool_call_id,
                APPROVAL_TOOL,
                "AI 리뷰를 통과하지 못한 계획은 비대화 실행에서 승인할 수 없습니다. "
                "코드 변경 없이 계획서와 아래 남은 의견을 보고하고 끝내세요.\n"
                f"{listed}",
                error=True,
            )
        elif self._allow_ai_approval:
            approved, note = True, "비대화 실행: AI 리뷰 통과를 승인으로 인정"
            approver = "AI 리뷰 (비대화 옵션)"
        else:
            await self._log(
                runtime,
                "plan.approval",
                {"decision": "불가", "reason": "비대화 실행에 승인 옵션이 없음"},
                failed=True,
            )
            return self._reply(
                tool_call_id,
                APPROVAL_TOOL,
                "비대화 실행에서는 계획을 승인할 사람이 없습니다. 코드 변경 없이 "
                "계획서와 리뷰 결과를 보고하고 끝내세요.",
                error=True,
            )
        decision = "승인" if approved else "수정 요청"
        await asyncio.to_thread(
            _append, record, _approval_record(decision, approver, plan_hash, note)
        )
        await self._log(
            runtime,
            "plan.approval",
            {
                "decision": decision,
                "approver": approver,
                "note": note,
                "plan_hash": plan_hash,
            },
            failed=not approved,
        )
        if approved:
            # Remember what was already changed, so the post-work report does
            # not count it as this plan's work.
            await asyncio.to_thread(plan_report.write_baseline, self._root, path)
            return self._reply(
                tool_call_id,
                APPROVAL_TOOL,
                "계획이 승인됐습니다. 이제 '## 수정 대상'에 적은 경로만 편집할 수 "
                "있습니다. 다른 파일이 필요하면 계획서를 고치고 리뷰와 승인을 다시 "
                "받으세요.",
                update={
                    "_plan_phase": PHASE_APPROVED,
                    "_plan_approved_hash": plan_hash,
                    "_plan_review_rounds": 0,
                    "_plan_review_findings": [],
                },
            )
        return self._reply(
            tool_call_id,
            APPROVAL_TOOL,
            f"사용자가 계획 수정을 요청했습니다. 의견: {note.strip() or '(없음)'}\n"
            f"계획서를 고친 뒤 `{REVIEW_TOOL}`을 다시 호출하세요.",
            update={
                "_plan_phase": PHASE_PLANNING,
                "_plan_reviewed_hash": None,
                "_plan_review_rounds": 0,
            },
        )

    @staticmethod
    def _ask_human(
        tool_call_id: str, path: Path, findings: list[str]
    ) -> tuple[bool, str]:
        """Interrupt the run to ask the human for a decision.

        The question uses the `ask_user` interrupt format, so the client shows
        it with its existing question prompt. Unlike tool approvals, it is not
        skipped by an auto-approve mode: a plan is approved by a person.

        Args:
            tool_call_id: Id of the approval tool call.
            path: Plan file the human should read.
            findings: Review findings the plan did not resolve.

        Returns:
            Whether the human approved, and their free-text note.
        """
        question = f"계획서를 승인하시겠습니까?\n계획서: {path}"
        if findings:
            listed = "\n".join(f"- {item}" for item in findings)
            question += f"\n\nAI 리뷰에서 해결되지 않은 의견:\n{listed}"
        response = interrupt(
            {
                "type": "ask_user",
                "tool_call_id": tool_call_id,
                "questions": [
                    {
                        "question": question,
                        "type": "multiple_choice",
                        "choices": [
                            {"value": APPROVE_CHOICE},
                            {"value": REVISE_CHOICE},
                        ],
                    },
                    {
                        "question": "수정 요청이면 무엇을 바꿀지 적어 주세요. (선택)",
                        "type": "text",
                        "required": False,
                    },
                ],
            }
        )
        answers: list[Any] = []
        if (
            isinstance(response, dict)
            and response.get("status", "answered") == "answered"
        ):
            raw = response.get("answers")
            answers = raw if isinstance(raw, list) else []
        choice = answers[0] if answers else ""
        note = answers[1] if len(answers) > 1 and isinstance(answers[1], str) else ""
        # Anything other than the exact approve choice is a refusal, including a
        # cancelled or malformed answer.
        return choice == APPROVE_CHOICE, note

    @staticmethod
    def _reply(
        tool_call_id: str,
        name: str,
        text: str,
        *,
        error: bool = False,
        update: dict[str, Any] | None = None,
    ) -> Command[Any]:
        """Build a tool result that may also update plan state.

        Args:
            tool_call_id: Id of the tool call being answered.
            name: Tool name.
            text: Message for the model.
            error: Whether the result is an error.
            update: Plan state channels to set.

        Returns:
            Command carrying the tool message and the state update.
        """
        message = ToolMessage(
            content=text,
            tool_call_id=tool_call_id,
            name=name,
            status="error" if error else "success",
        )
        return Command(update={**(update or {}), "messages": [message]})

    # ----- gate ----------------------------------------------------------

    def will_refuse_without_io(self, request: ToolCallRequest) -> bool:
        """Predict a refusal using only the request, with no disk access.

        Tool approval prompts are decided on the event loop before a tool
        runs. Asking a human to approve a call that the gate will refuse anyway
        is noise, so the approval layer skips its prompt when this returns
        `True`. The prediction is conservative: whenever it returns `True` the
        gate does refuse.

        Args:
            request: Tool call about to be approved.

        Returns:
            `True` when the call is certain to be refused by the gate.
        """
        if not _is_gated(request) or _phase_of(request.state) == PHASE_APPROVED:
            return False
        if trace_log.in_subagent():
            return False
        return not self._names_plan_file(request)

    def will_refuse_call(self, tool_call: Mapping[str, Any], state: object) -> bool:
        """Predict a refusal from a tool call and the graph state alone.

        Same answer as `will_refuse_without_io`. The interactive approval
        layer picks the calls to ask about before any `ToolCallRequest`
        exists and does not consult `when` predicates, so it asks here.

        Args:
            tool_call: Tool call proposed by the model.
            state: Graph state the call was proposed in.

        Returns:
            `True` when the call is certain to be refused by the gate.
        """
        request = ToolCallRequest(
            tool_call=cast("Any", tool_call),
            tool=None,
            state=cast("Any", state),
            runtime=cast("Any", None),
        )
        return self.will_refuse_without_io(request)

    def _names_plan_file(self, request: ToolCallRequest) -> bool:
        """Report whether a call writes a plan file, judged without disk access.

        Args:
            request: Tool call.

        Returns:
            `True` for `write_file`/`edit_file` on a plan file in the plan
                folder.
        """
        if request.tool_call["name"] not in {"write_file", "edit_file"}:
            return False
        raw = _file_argument(request)
        if raw is None:
            return False
        folder = plan_rules.plan_directory(self._root)
        name = raw.replace("\\", "/").rsplit("/", 1)[-1]
        return plan_rules.is_plan_file_name(name) and plan_rules.lexically_inside(
            raw, self._root, folder
        )

    def _writes_plan_file(self, request: ToolCallRequest) -> bool:
        """Report whether a call writes a plan file, following symlinks.

        Both the lexical and the resolved check must agree, so a symlink can
        neither smuggle a write out of the plan folder nor make the gate allow
        a call that `will_refuse_without_io` said it would refuse.

        Args:
            request: Tool call.

        Returns:
            `True` when the call may always run because it only writes a plan.
        """
        if not self._names_plan_file(request):
            return False
        raw = _file_argument(request)
        if raw is None:
            return False
        folder = plan_rules.plan_directory(self._root).resolve()
        return plan_rules.is_inside(plan_rules.resolve_path(raw, self._root), folder)

    def _decide(self, request: ToolCallRequest) -> ToolMessage | Command[Any] | None:
        """Decide whether a tool call may run. Blocking: reads the plan file.

        Args:
            request: Tool call.

        Returns:
            `None` to let the call run, or the refusal to return in its place.
        """
        name = request.tool_call["name"]
        if not _is_gated(request) or self._writes_plan_file(request):
            return None
        path = plan_rules.plan_file(self._root, thread_id_of(request.runtime))
        if _phase_of(request.state) != PHASE_APPROVED:
            return self._refusal(
                request,
                f"아직 승인된 계획이 없어 `{name}` 호출을 실행하지 않았습니다.\n"
                f"1. 계획서를 `{path}`에 씁니다. 필수 절:\n{_PLAN_TEMPLATE}\n"
                f"2. `{REVIEW_TOOL}`을 호출합니다.\n"
                f"3. 리뷰를 통과하면 `{APPROVAL_TOOL}`을 호출합니다.\n"
                "그동안 읽기 도구(read_file, grep, glob, ls)는 쓸 수 있습니다.",
            )
        text = plan_rules.read_text(path)
        approved_hash = (request.state or {}).get("_plan_approved_hash")
        if text is None or plan_rules.hash_text(text) != approved_hash:
            message = self._refusal(
                request,
                "승인된 뒤에 계획서가 바뀌었습니다. 승인이 무효가 되어 "
                f"`{name}` 호출을 실행하지 않았습니다. `{REVIEW_TOOL}`과 "
                f"`{APPROVAL_TOOL}`을 다시 거치세요.",
            )
            return Command(
                update={
                    "_plan_phase": PHASE_PLANNING,
                    "_plan_reviewed_hash": None,
                    "_plan_approved_hash": None,
                    "messages": [message],
                }
            )
        if name not in FILE_TOOLS:
            return None
        raw = _file_argument(request)
        targets = plan_rules.parse_plan(text).targets
        if raw is not None and plan_rules.is_target(
            plan_rules.resolve_path(raw, self._root), self._root, targets
        ):
            return None
        listed = "\n".join(f"- `{entry}`" for entry in targets)
        return self._refusal(
            request,
            f"`{raw}`는 승인된 계획의 '## 수정 대상'에 없어 `{name}` 호출을 "
            f"실행하지 않았습니다. 승인된 대상:\n{listed}\n"
            "이 파일이 필요하면 계획서의 '## 수정 대상'을 고치고 "
            f"`{REVIEW_TOOL}`과 `{APPROVAL_TOOL}`을 다시 거치세요.",
        )

    async def _log(
        self,
        runtime: object,
        kind: str,
        data: Mapping[str, Any],
        *,
        failed: bool = False,
    ) -> None:
        """Record a harness step from a tool, off the event loop.

        Args:
            runtime: Tool runtime, the source of the thread id.
            kind: Step kind.
            data: Details of the step.
            failed: Whether the step is a failure.
        """
        await asyncio.to_thread(
            trace_log.record,
            self._root,
            thread_id_of(runtime),
            kind,
            data,
            failed=failed,
        )

    def _decide_and_log(
        self, request: ToolCallRequest
    ) -> ToolMessage | Command[Any] | None:
        """Decide whether a tool call may run and record the decision. Blocking.

        A refusal and a write to the plan file are harness steps; any other
        permitted call is traced as an ordinary tool run.

        Args:
            request: Tool call.

        Returns:
            `None` to let the call run, or the refusal to return in its place.
        """
        refusal = self._decide(request)
        name = request.tool_call["name"]
        thread_id = thread_id_of(request.runtime)
        if refusal is not None:
            message = (
                refusal
                if isinstance(refusal, ToolMessage)
                else (refusal.update or {}).get("messages", [None])[0]
            )
            trace_log.record(
                self._root,
                thread_id,
                "gate.refused",
                {
                    "tool": name,
                    "file_path": _file_argument(request),
                    "phase": _phase_of(request.state),
                    "message": str(getattr(message, "text", ""))[:300],
                },
                failed=True,
            )
        elif name in FILE_TOOLS and self._names_plan_file(request):
            trace_log.record(
                self._root,
                thread_id,
                "plan.written",
                {"tool": name, "file_path": _file_argument(request)},
            )
        return refusal

    @staticmethod
    def _refusal(request: ToolCallRequest, text: str) -> ToolMessage:
        """Build the message returned in place of a refused tool call.

        Args:
            request: Refused tool call.
            text: Explanation and next steps for the model.

        Returns:
            Error tool message answering the call.
        """
        return ToolMessage(
            content=f"[계획 관문] {text}",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run a tool call only when the gate allows it.

        Returns:
            The tool result, or the refusal when the gate blocks the call.
        """
        if trace_log.in_subagent():
            return handler(request)
        refusal = self._decide_and_log(request)
        return handler(request) if refusal is None else refusal

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Run a tool call only when the gate allows it.

        Returns:
            The tool result, or the refusal when the gate blocks the call.
        """
        # Most calls are not gated. Check that inline so those calls skip the
        # thread hop that reading the plan needs.
        if not _is_gated(request) or trace_log.in_subagent():
            return await handler(request)
        refusal = await asyncio.to_thread(self._decide_and_log, request)
        return await handler(request) if refusal is None else refusal

    # ----- post-work report ----------------------------------------------

    def _report_inputs(
        self, state: Mapping[str, Any], thread_id: str
    ) -> tuple[Path, str, TurnActivity, Comparison] | None:
        """Gather what a post-work report needs. Blocking: reads disk, runs git.

        Args:
            state: Graph state at the end of the turn.
            thread_id: Thread the turn belongs to.

        Returns:
            Plan path, plan text, the turn's activity and its comparison with
                the plan, or `None` when the turn calls for no report: no
                approved plan, a plan changed since approval, or a turn that
                changed nothing outside the harness folder and ran no command.
        """
        if _phase_of(state) != PHASE_APPROVED:
            return None
        path = plan_rules.plan_file(self._root, thread_id)
        text = plan_rules.read_text(path)
        if text is None or plan_rules.hash_text(text) != state.get(
            "_plan_approved_hash"
        ):
            return None
        activity = plan_report.collect_turn_activity(list(state.get("messages") or []))
        if not activity.changed_something:
            return None
        comparison = plan_report.compare_with_plan(self._root, path, text, activity)
        if not (comparison.changed or activity.commands or activity.other_calls):
            # Only the harness's own files were written, such as the plan.
            return None
        return path, text, activity, comparison

    def _write_report(
        self,
        path: Path,
        text: str,
        activity: TurnActivity,
        comparison: Comparison,
        review: WorkReview | None,
        review_note: str,
        request_text: str,
        thread_id: str,
    ) -> None:
        """Append the turn's report entry and save it as experience. Blocking.

        Args:
            path: Plan file the report belongs to.
            text: Approved plan content.
            activity: Tool activity of the turn.
            comparison: Harness comparison with the plan.
            review: Reviewer's assessment, if any.
            review_note: Why there is no assessment.
            request_text: The request the turn answered.
            thread_id: Thread the turn belongs to.
        """
        entry = plan_report.render_report(
            plan_hash=plan_rules.hash_text(text),
            activity=activity,
            comparison=comparison,
            review=review,
            review_note=review_note,
        )
        report = path.with_name(path.stem + plan_rules.REPORT_SUFFIX)
        plan_report.append_report(report, entry)
        trace_log.record(
            self._root,
            thread_id,
            "work.review",
            {
                "changed": list(comparison.changed),
                "unplanned_paths": list(comparison.unplanned),
                "untouched_targets": list(comparison.untouched),
                "commands": len(activity.commands),
                "outcome": review.outcome if review else "unknown",
                "missing": list(review.missing) if review else [],
                "unplanned": list(review.unplanned) if review else [],
                "tests": review.tests if review else review_note,
                "report": str(report),
            },
            failed=bool(
                comparison.unplanned
                or (review and (review.missing or review.outcome == "failure"))
            ),
        )
        if self._record_experience:
            # The record is built from the same review, so remembering the
            # work costs no further model call.
            saved = experience.save(
                self._root,
                experience.Experience(
                    id=experience.new_id(thread_id),
                    request=request_text,
                    summary=review.summary if review else "",
                    outcome=review.outcome if review else "unknown",
                    files=comparison.changed,
                    lessons=tuple(review.lessons) if review else (),
                    tags=tuple(review.tags) if review else (),
                    date=datetime.now(UTC).isoformat(timespec="seconds"),
                    thread_id=thread_id,
                ),
            )
            trace_log.record(self._root, thread_id, "memory.saved", {"id": saved.stem})

    def _record_turn(self, state: Mapping[str, Any], thread_id: str) -> None:
        """Record the turn's metrics locally and as trace scores. Blocking.

        Args:
            state: Graph state at the end of the turn.
            thread_id: Thread the turn belongs to.
        """
        try:
            started = state.get("_plan_turn_started")
            metrics = trace_log.turn_metrics(
                list(state.get("messages") or []),
                started if isinstance(started, float) else None,
            )
            if self._retry_counter is not None:
                metrics["model_retries"] = self._retry_counter.take(thread_id)
            if self._run_counter is not None:
                counts = self._run_counter.take(thread_id, trace_log.AGENT_WORK_ROLES)
                metrics.update(trace_log.agent_work_metrics(counts))
            trace_log.record(
                self._root, thread_id, "turn.metrics", trace_log.metrics_step(metrics)
            )
            trace_log.record_scores(metrics)
        except Exception:
            logger.warning("Turn metrics could not be recorded", exc_info=True)

    def _record_post_work(
        self, state: Mapping[str, Any], thread_id: str, hook_started: float
    ) -> None:
        """Record how long the harness worked after the agent finished. Blocking.

        `turn.metrics` is taken when the agent's own work ends. The post-work
        review, the memory record and self-improvement run after that and add
        to what the user waits for, so their time is recorded separately,
        together with the model and tool calls they made when a run counter
        is present.

        Args:
            state: Graph state at the end of the turn.
            thread_id: Thread the turn belongs to.
            hook_started: Unix time at which the end-of-turn hook started.
        """
        try:
            now = time.time()
            times: dict[str, float] = {
                "post_work_seconds": round(max(now - hook_started, 0.0), 3)
            }
            started = state.get("_plan_turn_started")
            if isinstance(started, float):
                times["total_seconds"] = round(max(now - started, 0.0), 3)
            if self._run_counter is not None:
                counts = self._run_counter.take(thread_id)
                times.update(trace_log.post_work_metrics(counts))
            trace_log.record(self._root, thread_id, "turn.post_work", times)
            trace_log.record_scores(times)
        except Exception:
            logger.warning("Post-work time could not be recorded", exc_info=True)

    def _record_improvement(
        self,
        thread_id: str,
        signals: improvement.Signals,
        outcome: improvement.Outcome | None,
    ) -> None:
        """Record what self-improvement did after a failed turn. Blocking.

        Args:
            thread_id: Thread the turn belongs to.
            signals: Failure signals of the turn.
            outcome: Decision of the run, or `None` when nothing was proposed.
        """
        data: dict[str, Any] = {"signals": signals.describe()}
        if outcome is None:
            data["decision"] = "개선안 없음"
        else:
            data.update(
                id=outcome.id,
                decision="반영" if outcome.applied else "보류",
                reason=outcome.reason,
                scores=[
                    {"case": s.case_id, "before": s.before, "after": s.after}
                    for s in outcome.scores
                ],
            )
        trace_log.record(
            self._root,
            thread_id,
            "improvement.decision",
            data,
            failed=outcome is not None and not outcome.applied,
        )

    def _start_turn(self, state: Mapping[str, Any], thread_id: str) -> dict[str, Any]:
        """Open a new turn: a new request starts without an approved plan. Blocking.

        An approval covers the request it was given for. Keeping it for the
        next request would let that request change the approved files without
        a plan of its own, so the thread goes back to planning. The plan that
        was approved is kept as a dated copy, because the next plan is written
        to the same file.

        A run that resumes after an approval question is the same turn and
        does not come through here.

        Args:
            state: Graph state at the start of the turn.
            thread_id: Thread the turn belongs to.

        Returns:
            State update with the turn's start time and, when the previous
                request left plan state behind, that state cleared.
        """
        if self._retry_counter is not None:
            self._retry_counter.reset(thread_id)
        if self._run_counter is not None:
            self._run_counter.reset(thread_id)
        update: dict[str, Any] = {"_plan_turn_started": time.time()}
        phase = _phase_of(state)
        if phase == PHASE_PLANNING and not state.get("_plan_review_rounds"):
            return update
        archived: Path | None = None
        if phase == PHASE_APPROVED:
            try:
                archived = self._archive_plan(state, thread_id)
            except OSError:
                logger.warning("Approved plan could not be archived", exc_info=True)
        trace_log.record(
            self._root,
            thread_id,
            "plan.reset",
            {
                "previous_phase": phase,
                "archived": str(archived) if archived else None,
                "reason": "새 사용자 요청",
            },
        )
        update.update(
            _plan_phase=PHASE_PLANNING,
            _plan_reviewed_hash=None,
            _plan_approved_hash=None,
            _plan_review_rounds=0,
            _plan_review_findings=[],
        )
        return update

    def _archive_plan(self, state: Mapping[str, Any], thread_id: str) -> Path | None:
        """Keep a copy of the plan that was approved for the previous request.

        Args:
            state: Graph state holding the approved fingerprint.
            thread_id: Thread whose plan is copied.

        Returns:
            Path of the copy, or `None` when the plan file no longer holds the
                approved text.
        """
        path = plan_rules.plan_file(self._root, thread_id)
        text = plan_rules.read_text(path)
        if text is None or plan_rules.hash_text(text) != state.get(
            "_plan_approved_hash"
        ):
            return None
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        copy = path.with_name(f"{path.stem}.{stamp}{plan_rules.APPROVED_SUFFIX}")
        copy.write_text(text, encoding="utf-8")
        return copy

    @override
    def before_agent(
        self, state: PlanGateState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Open a new turn: note its start and require a plan of its own.

        Returns:
            State update from `_start_turn`.
        """
        if trace_log.in_subagent():
            return None
        return self._start_turn(state, _hook_thread_id(runtime))

    @override
    async def abefore_agent(
        self, state: PlanGateState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Open a new turn: note its start and require a plan of its own.

        Returns:
            State update from `_start_turn`.
        """
        if trace_log.in_subagent():
            return None
        return await asyncio.to_thread(
            self._start_turn, state, _hook_thread_id(runtime)
        )

    @override
    def after_agent(self, state: PlanGateState, runtime: Runtime[ContextT]) -> None:
        """Write the post-work report without an AI assessment.

        The reviewer agent is asynchronous, so a synchronous run records only
        what the harness collected. The report is a file; no state changes.
        """
        if trace_log.in_subagent():
            return
        thread_id = _hook_thread_id(runtime)
        self._record_turn(state, thread_id)
        try:
            inputs = self._report_inputs(state, thread_id)
            if inputs is not None:
                self._write_report(
                    *inputs,
                    None,
                    "동기 실행에서는 AI 리뷰를 실행하지 않습니다",
                    experience.latest_request(list(state.get("messages") or [])),
                    thread_id,
                )
        except Exception:
            # A failed report must not fail a turn whose work is already done.
            logger.warning("Post-work report failed", exc_info=True)

    @override
    async def aafter_agent(
        self, state: PlanGateState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Write the post-work report once the turn's work is finished.

        Returns:
            `None`; the report is a file and changes no state.
        """
        if trace_log.in_subagent():
            return None
        thread_id = _hook_thread_id(runtime)
        hook_started = time.time()
        await asyncio.to_thread(self._record_turn, state, thread_id)
        try:
            inputs = await asyncio.to_thread(self._report_inputs, state, thread_id)
            if inputs is None:
                return None
            path, text, activity, comparison = inputs
            review: WorkReview | None = None
            note = ""
            if self._work_reviewer is not None:
                request, earlier = split_requests(list(state.get("messages") or []))
                review = await self._work_reviewer.review(
                    request_text=request,
                    plan_text=text,
                    evidence=plan_report.evidence_text(activity, comparison),
                    project_root=str(self._root),
                    earlier_requests=earlier,
                    context=runtime.context,
                )
                if review is None:
                    note = "리뷰어가 평가를 내지 못했습니다"
            messages = list(state.get("messages") or [])
            request_text = experience.latest_request(messages)
            await asyncio.to_thread(
                self._write_report,
                path,
                text,
                activity,
                comparison,
                review,
                note,
                request_text,
                thread_id,
            )
        except Exception:
            # A failed report must not fail a turn whose work is already done.
            logger.warning("Post-work report failed", exc_info=True)
            return None
        if self._improver is not None:
            try:
                signals = improvement.collect_signals(messages, review, comparison)
                outcome = await self._improver.run(
                    signals=signals,
                    request_text=request_text,
                    thread_id=thread_id,
                    context=runtime.context,
                )
                if signals.failed:
                    await asyncio.to_thread(
                        self._record_improvement, thread_id, signals, outcome
                    )
            except Exception:
                # Improvement is optional follow-up work, never part of the turn.
                logger.warning("Self-improvement failed", exc_info=True)
        await asyncio.to_thread(self._record_post_work, state, thread_id, hook_started)
        return None

    # ----- model notice --------------------------------------------------

    def notice(self, state: Mapping[str, Any] | None, runtime: object) -> str:
        """Describe the gate and the current phase for the model.

        Args:
            state: Graph state.
            runtime: Runtime used to name the thread's plan file.

        Returns:
            Text appended to the system message.
        """
        path = plan_rules.plan_file(self._root, thread_id_of(runtime))
        phase = _phase_of(state)
        if phase == PHASE_APPROVED:
            status = (
                "Phase: approved. You may edit only the paths listed under "
                "'## 수정 대상' of the approved plan. Changing the plan file "
                "voids the approval."
            )
        elif phase == PHASE_REVIEWED:
            status = (
                f"Phase: reviewed. Call `{APPROVAL_TOOL}` now, without changing "
                "the plan file."
            )
        else:
            status = (
                "Phase: planning. Tools that change the workspace (write_file, "
                "edit_file, execute, task) are refused, except writing the "
                "plan file. An approval covers one user request: every new "
                "request that changes code needs its own reviewed and approved "
                "plan, written to the same plan file."
            )
        return (
            "## Plan gate\n\n"
            "Code changes in this session require an approved plan. Questions "
            "and read-only investigation need no plan.\n\n"
            f"{status}\n\n"
            f"Plan file for this thread: `{path}`\n\n"
            "Workflow: write the plan file, call "
            f"`{REVIEW_TOOL}`, fix reported problems and repeat, then call "
            f"`{APPROVAL_TOOL}`. Write the plan in the user's language and use "
            "exactly these level-two headings:\n\n"
            f"{_PLAN_TEMPLATE}\n\n"
            "Under '## 수정 대상' list each file to change as a bullet with the "
            "path in backticks, relative to the project root; end a path with "
            "`/` to cover a folder. Under '## 작업 순서' use a numbered list. "
            "Base '## 테스트 방법' on the project's existing test setup."
        )

    def _with_notice(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Return the request with the gate notice appended to its system message.

        Args:
            request: Model request.

        Returns:
            Updated request.
        """
        if trace_log.in_subagent():
            # The phase shown would be the subagent's own, which is never the
            # main agent's. The gate does not act there, so it says nothing.
            return request
        blocks = (
            list(request.system_message.content_blocks)
            if request.system_message
            else []
        )
        text = self.notice(request.state, request.runtime)
        blocks.append({"type": "text", "text": f"\n\n{text}" if blocks else text})
        return request.override(system_message=SystemMessage(content_blocks=blocks))

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Tell the model the current plan phase on every call.

        Returns:
            The model response.
        """
        return handler(self._with_notice(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Tell the model the current plan phase on every call.

        Returns:
            The model response.
        """
        return await handler(self._with_notice(request))

    # ----- approval layer integration ------------------------------------

    def guard_interrupt_on(
        self, interrupt_on: Mapping[str, bool | InterruptOnConfig]
    ) -> dict[str, bool | InterruptOnConfig]:
        """Stop tool approval prompts for calls the gate will refuse.

        Args:
            interrupt_on: Per-tool approval settings of the main agent.

        Returns:
            Copy of the settings in which each gated tool's `when` predicate
                first asks the gate. Tools without a `when` predicate are left
                unchanged.
        """
        guarded: dict[str, bool | InterruptOnConfig] = dict(interrupt_on)
        for name in GATED_TOOLS:
            config = guarded.get(name)
            if not isinstance(config, dict):
                continue
            original = config.get("when")
            if original is None:
                continue
            guarded[name] = {**config, "when": self._guarded_when(original)}
        return guarded

    def _guarded_when(
        self, original: Callable[[ToolCallRequest], bool]
    ) -> Callable[[ToolCallRequest], bool]:
        """Wrap an approval predicate so refused calls are not prompted for.

        Args:
            original: Existing predicate deciding whether to ask the human.

        Returns:
            Predicate that returns `False` for calls the gate will refuse and
                otherwise defers to `original`.
        """

        def when(request: ToolCallRequest) -> bool:
            """Skip the wrapped check for calls the gate will refuse anyway.

            Args:
                request: Tool call about to run.

            Returns:
                Whether the original condition applies to this call.
            """
            return not self.will_refuse_without_io(request) and original(request)

        return when
