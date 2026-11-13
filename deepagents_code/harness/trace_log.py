"""Step records of the harness: one function, two destinations.

Model calls and tool runs are traced by callbacks without any help. What the
harness itself decides is not: a refused tool call, a review verdict, an
approval, a retrieved memory, an applied improvement. Each such step goes
through `record`, which writes it to

1. a local file, `.deepagents/traces/<thread id>.jsonl`, so the flow can be
   inspected without a tracing service and in tests without a network
   (on unless `DEEPAGENTS_CODE_PLAN_STEP_LOG` is set to a falsy value);
2. the active trace, as a named run under the step that produced it, so
   LangSmith and Langfuse show it in place.

Turn metrics are also attached to the Langfuse trace as numeric scores.

Recording is an aid. Nothing here may fail the work it describes.

A step marked as failed is also given a level by `failure_level`: `warning`
for a refusal that is part of the procedure (the gate or a review doing its
job) and `error` for a step where the work actually went wrong. The viewer's
first failure is the first `error`.

Run `python -m deepagents_code.harness.trace_log` in a project to list the
recorded threads, or pass a thread id to print its steps and first failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphBubbleUp

from deepagents_code._env_vars import PLAN_STEP_LOG, is_env_truthy
from deepagents_code.harness.plan_review import (
    FAILED_REVIEW_PREFIX,
    HELPER_TAG_PREFIX,
    ROLE_IMPROVEMENT,
    ROLE_PLAN_REVIEW,
    ROLE_WORK_REVIEW,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from uuid import UUID

    from langgraph.types import Command

logger = logging.getLogger(__name__)

TRACE_DIRECTORY = Path(".deepagents") / "traces"
"""Local step record folder, relative to the project root."""

RUN_NAME_PREFIX = "harness."
"""Prefix of the run name a step gets in the active trace."""

SCORE_PREFIX = "harness."
"""Prefix of the Langfuse score name a turn metric gets."""

REVIEW_TOOL = "request_plan_review"
GATE_REFUSAL_PREFIX = "[계획 관문]"

LEVEL_WARNING = "warning"
"""Level of a failed step that is part of the procedure, such as a refusal."""

LEVEL_ERROR = "error"
"""Level of a failed step where the work went wrong."""

# Wording the plan gate writes into step data. The gate imports this module,
# so the values are repeated here; the gate's recording tests fail when the
# two drift apart.
REVIEW_LIMIT_VERDICT = "limit"
APPROVAL_UNAVAILABLE = "불가"
NO_APPROVAL_OPTION_REASON = "비대화 실행에 승인 옵션이 없음"
OPEN_FINDINGS_REASON = "리뷰 상한에 닿아 남은 의견이 있음"
IMPROVEMENT_HELD = "보류"

_PROCEDURAL_KINDS = frozenset({"gate.refused", "plan.invalid"})
"""Step kinds whose failure is always a refusal the agent can act on."""

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")
_VALUE_CHAR_LIMIT = 2000
_SUMMARY_CHAR_LIMIT = 100

_anonymizer: Callable[[Any], Any] | None = None


def trace_file(root: Path, thread_id: str) -> Path:
    """Return the local step record of one thread.

    Args:
        root: Project root.
        thread_id: Thread identifier. Characters outside `[A-Za-z0-9_-]` are
            replaced so the result is always a single file name.

    Returns:
        Path of the thread's record. The file may not exist yet.
    """
    name = _UNSAFE_NAME.sub("_", thread_id) or "thread"
    return root / TRACE_DIRECTORY / f"{name}.jsonl"


def _redact(data: Mapping[str, Any]) -> dict[str, Any]:
    """Mask secrets in step data and bound the size of its text values.

    Uses the same anonymizer dcode applies to LangSmith traces. When it
    cannot be built, text values are dropped rather than written unmasked.

    Args:
        data: Step data.

    Returns:
        A copy that is safe to store and send.
    """
    global _anonymizer  # noqa: PLW0603  # built once, on first use
    clipped = json.loads(json.dumps(data, ensure_ascii=False, default=str))

    def clip(value: Any) -> Any:  # noqa: ANN401  # JSON value
        """Shorten every string inside the value to the log's length limit.

        Args:
            value: JSON-like value; lists and dicts are walked.

        Returns:
            The same structure with long strings cut.
        """
        if isinstance(value, str):
            return value[:_VALUE_CHAR_LIMIT]
        if isinstance(value, list):
            return [clip(item) for item in value]
        if isinstance(value, dict):
            return {key: clip(item) for key, item in value.items()}
        return value

    clipped = clip(clipped)
    try:
        if _anonymizer is None:
            from deepagents_code._trace_redaction import (
                create_cached_secret_anonymizer,
            )

            _anonymizer = create_cached_secret_anonymizer()
        masked = _anonymizer(clipped)
    except Exception:
        logger.warning("Step data could not be masked; text dropped", exc_info=True)
        return {
            key: value
            for key, value in clipped.items()
            if isinstance(value, (bool, int, float)) or value is None
        }
    return masked if isinstance(masked, dict) else clipped


def _emit_run(kind: str, data: dict[str, Any]) -> None:
    """Show a step in the active trace as a named run.

    The run is a function that returns its input, started inside the current
    runnable context, so every attached tracer records it under the step that
    produced it. Outside a runnable context nothing is emitted.

    Args:
        kind: Step kind.
        data: Masked step data.
    """
    from langchain_core.runnables import RunnableLambda
    from langgraph.config import get_config

    try:
        get_config()
    except RuntimeError:
        return
    RunnableLambda(lambda value: value).invoke(
        data, {"run_name": f"{RUN_NAME_PREFIX}{kind}"}
    )


def _failure_level(kind: str, data: Mapping[str, Any]) -> str:
    """Classify a failed step by its kind and recorded details.

    Args:
        kind: Step kind.
        data: Recorded details of the step.

    Returns:
        `LEVEL_WARNING` or `LEVEL_ERROR`.
    """
    if kind in _PROCEDURAL_KINDS:
        return LEVEL_WARNING
    if kind == "plan.review":
        findings = data.get("findings") or []
        no_verdict = any(str(f).startswith(FAILED_REVIEW_PREFIX) for f in findings)
        if data.get("verdict") == REVIEW_LIMIT_VERDICT or no_verdict:
            return LEVEL_ERROR
        return LEVEL_WARNING
    if kind == "plan.approval":
        stuck = data.get("decision") == APPROVAL_UNAVAILABLE and data.get("reason") in {
            NO_APPROVAL_OPTION_REASON,
            OPEN_FINDINGS_REASON,
        }
        return LEVEL_ERROR if stuck else LEVEL_WARNING
    if kind == "improvement.decision" and data.get("decision") == IMPROVEMENT_HELD:
        return LEVEL_WARNING
    return LEVEL_ERROR


def failure_level(step: Mapping[str, Any]) -> str | None:
    """Return how serious a recorded step's failure is.

    A refusal that belongs to the procedure (the gate refusing a call before
    approval, a review asking for changes, a human sending the plan back, an
    improvement held by its check) is a `warning`. A step where the work went
    wrong (a work review finding gaps, a review that could not reach a verdict
    or did not converge, a run that cannot be approved) is an `error`. An
    unknown failed step is an `error`.

    Steps recorded before levels existed carry no `level`; they are classified
    from their kind and data, so older records read the same way.

    Args:
        step: Recorded step, as returned by `read`.

    Returns:
        `None` for a step that did not fail, otherwise `LEVEL_WARNING` or
            `LEVEL_ERROR`.
    """
    if not step.get("failed"):
        return None
    level = step.get("level")
    if level in {LEVEL_WARNING, LEVEL_ERROR}:
        return level
    data = step.get("data")
    try:
        return _failure_level(
            str(step.get("kind", "")), data if isinstance(data, dict) else {}
        )
    except Exception:
        logger.warning("Step level could not be determined", exc_info=True)
        return LEVEL_ERROR


def record(
    root: Path,
    thread_id: str,
    kind: str,
    data: Mapping[str, Any] | None = None,
    *,
    failed: bool = False,
) -> None:
    """Record one harness step locally and in the active trace.

    Args:
        root: Project root.
        thread_id: Thread the step belongs to.
        kind: Step kind, such as `plan.review` or `gate.refused`.
        data: Details of the step. Values must be JSON-serializable; anything
            else is stored as its string form.
        failed: Whether the step is a failure a reader should look at first.
            A failed step is stored with its `failure_level`.
    """
    level: str | None = None
    try:
        payload = _redact(data or {})
        level = failure_level({"kind": kind, "failed": failed, "data": payload})
        if is_env_truthy(PLAN_STEP_LOG, default=True):
            entry = {
                "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "thread_id": thread_id,
                "kind": kind,
                "failed": failed,
                "level": level,
                "data": payload,
            }
            path = trace_file(root, thread_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("Step %s could not be recorded locally", kind, exc_info=True)
        return
    try:
        details = {**payload, "failed": failed}
        if level is not None:
            details["level"] = level
        _emit_run(kind, details)
    except Exception:
        logger.warning("Step %s could not be added to the trace", kind, exc_info=True)


def record_scores(metrics: Mapping[str, float]) -> None:
    """Attach turn metrics to the current Langfuse trace as numeric scores.

    Does nothing when Langfuse tracing is off.

    Args:
        metrics: Metric values keyed by name.
    """
    from deepagents_code.harness import observability

    if not observability.is_enabled():
        return
    try:
        from langfuse import get_client

        client = get_client()
        for name, value in metrics.items():
            client.score_current_trace(
                name=f"{SCORE_PREFIX}{name}", value=float(value), data_type="NUMERIC"
            )
    except Exception:
        logger.warning("Turn metrics could not be sent to Langfuse", exc_info=True)


def current_thread_id() -> str:
    """Return the thread of the run this code is executing in.

    Returns:
        Thread id, or `default` outside a run or when the run has no thread.
    """
    from langgraph.config import get_config

    try:
        thread_id = get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:
        return "default"
    return thread_id if isinstance(thread_id, str) and thread_id else "default"


SUBAGENT_METADATA_KEY = "lc_subagent_tool_call_id"
"""Run metadata the SDK sets on everything a subagent does."""


def in_subagent() -> bool:
    """Report whether this code is running inside a subagent.

    A forked subagent is built from the main agent's middleware, so the
    harness middleware runs in it as well. Its private state does not come
    along, though: inside the subagent the plan looks unapproved even when
    the main agent's plan was approved. The harness therefore stands aside in
    a subagent. That is safe because the main agent cannot start one before
    its plan is approved: `task` is itself a gated tool.

    Returns:
        `True` inside a subagent run, `False` in the main agent or outside
            any run.
    """
    from langgraph.config import get_config

    try:
        metadata = get_config().get("metadata")
    except RuntimeError:
        return False
    return isinstance(metadata, dict) and bool(metadata.get(SUBAGENT_METADATA_KEY))


class RetryCounter:
    """Failed model attempts per thread, shared by two middlewares.

    The counting middleware adds to it from inside the retry loop; the plan
    gate clears it when a turn starts and reads it when the turn ends. The
    count lives in the server process, so it is lost if the process restarts
    in the middle of a turn.
    """

    def __init__(self) -> None:
        """Create an empty counter."""
        self._failed: dict[str, int] = {}
        self._lock = threading.Lock()

    def reset(self, thread_id: str) -> None:
        """Forget a thread's count at the start of a turn.

        Args:
            thread_id: Thread whose turn is starting.
        """
        with self._lock:
            self._failed.pop(thread_id, None)

    def add(self, thread_id: str) -> None:
        """Count one failed model attempt.

        Args:
            thread_id: Thread the attempt belongs to.
        """
        with self._lock:
            self._failed[thread_id] = self._failed.get(thread_id, 0) + 1

    def take(self, thread_id: str) -> int:
        """Return a thread's count and forget it.

        Args:
            thread_id: Thread whose turn is ending.

        Returns:
            Failed model attempts since the last reset.
        """
        with self._lock:
            return self._failed.pop(thread_id, 0)


class ModelRetryCounterMiddleware(AgentMiddleware[Any, ContextT]):
    """Count failed model attempts from inside the retry loop.

    dcode's retry middleware calls the rest of the model stack once per
    attempt. Placed directly after it, this middleware sees every attempt and
    counts the ones that raise. In a turn that completes, each failed attempt
    was followed by a retry, so the count is the turn's number of retries.
    """

    def __init__(self, counter: RetryCounter) -> None:
        """Create the middleware.

        Args:
            counter: Counter shared with the plan gate.
        """
        super().__init__()
        self._counter = counter

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Pass the call through, counting it when it fails.

        Returns:
            The model response.
        """
        try:
            return handler(request)
        except Exception as exc:
            # Graph control flow, such as an interrupt, is not a failed attempt.
            if not isinstance(exc, GraphBubbleUp):
                self._counter.add(current_thread_id())
            raise

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Pass the call through, counting it when it fails.

        Returns:
            The model response.
        """
        try:
            return await handler(request)
        except Exception as exc:
            # Graph control flow, such as an interrupt, is not a failed attempt.
            if not isinstance(exc, GraphBubbleUp):
                self._counter.add(current_thread_id())
            raise


# ----- counting every role's calls ------------------------------------------

ROLE_MAIN = "main"
ROLE_SUBAGENT = "subagent"

AGENT_WORK_ROLES = (ROLE_MAIN, ROLE_PLAN_REVIEW, ROLE_SUBAGENT)
"""Roles that run while the agent works, reported in `turn.metrics`."""

POST_WORK_ROLES = (ROLE_WORK_REVIEW, ROLE_IMPROVEMENT)
"""Roles that run after the agent finished, reported in `turn.post_work`."""

MODEL_CALLS = "model_calls"
MODEL_ERRORS = "model_errors"
TOOL_CALLS = "tool_calls"
TOOL_ERRORS = "tool_errors"
_COUNTS = (MODEL_CALLS, MODEL_ERRORS, TOOL_CALLS, TOOL_ERRORS)

type RoleCounts = dict[str, dict[str, int]]
"""Counts keyed by role, then by `MODEL_CALLS`, `MODEL_ERRORS`, ..."""


class RunCounter:
    """Model and tool calls per thread and role, for every agent in a turn.

    `RunCounterCallback` adds to it from the callbacks of every run, so it
    also sees the helper agents and subagents that never appear in the main
    agent's messages. The plan gate clears it when a turn starts and takes
    the counts when the turn's work and its post-work end. The counts live in
    the server process, so they are lost if the process restarts in the
    middle of a turn.
    """

    def __init__(self) -> None:
        """Create an empty counter."""
        self._counts: dict[str, RoleCounts] = {}
        self._lock = threading.Lock()

    def reset(self, thread_id: str) -> None:
        """Forget a thread's counts at the start of a turn.

        Args:
            thread_id: Thread whose turn is starting.
        """
        with self._lock:
            self._counts.pop(thread_id, None)

    def add(self, thread_id: str, role: str, count: str) -> None:
        """Count one event.

        Args:
            thread_id: Thread the event belongs to.
            role: Role of the agent that caused it, such as `main`.
            count: Which count to raise, such as `MODEL_CALLS`.
        """
        with self._lock:
            roles = self._counts.setdefault(thread_id, {})
            counts = roles.setdefault(role, dict.fromkeys(_COUNTS, 0))
            counts[count] = counts.get(count, 0) + 1

    def take(self, thread_id: str, roles: Sequence[str] | None = None) -> RoleCounts:
        """Return a thread's counts and forget them.

        Args:
            thread_id: Thread whose counts to take.
            roles: Roles to take. Omit to take every role left.

        Returns:
            Counts of the taken roles. A role without events is absent.
        """
        with self._lock:
            held = self._counts.get(thread_id, {})
            names = list(held) if roles is None else [r for r in roles if r in held]
            taken = {name: held.pop(name) for name in names}
            if not held:
                self._counts.pop(thread_id, None)
            return taken


def run_role(tags: Sequence[str] | None, metadata: Mapping[str, Any] | None) -> str:
    """Tell which agent a run belongs to.

    Args:
        tags: Run tags, inherited from the enclosing runs.
        metadata: Run metadata, inherited from the enclosing runs.

    Returns:
        The role from a `harness:<role>` tag of a helper agent, `subagent`
            inside a subagent, else `main`.
    """
    for tag in tags or ():
        if tag.startswith(HELPER_TAG_PREFIX):
            return tag.removeprefix(HELPER_TAG_PREFIX)
    if metadata and metadata.get(SUBAGENT_METADATA_KEY):
        return ROLE_SUBAGENT
    return ROLE_MAIN


def _is_error_result(output: object) -> bool:
    """Tell whether a tool's result reports an error.

    A tool returns either a `ToolMessage` or, like the plan gate's own tools,
    a `Command` whose update carries the `ToolMessage`.

    Args:
        output: What the tool returned.

    Returns:
        `True` when the result, or a tool message in it, has error status.
    """
    if isinstance(output, ToolMessage):
        return output.status == "error"
    update = getattr(output, "update", None)
    if not isinstance(update, dict):
        return False
    return any(
        isinstance(message, ToolMessage) and message.status == "error"
        for message in update.get("messages") or ()
    )


class RunCounterCallback(BaseCallbackHandler):
    """Count model calls of every run into a `RunCounter`.

    Bound to the agent graph, the callback is inherited by everything the
    graph runs, including helper agents invoked from tools and hooks. Tool
    calls are counted by `ToolResultCounterMiddleware`, which reports them
    through `count_tool_result`. The
    thread comes from the run metadata, which LangGraph fills from the
    run's `thread_id`; events without a thread are not counted. Counting is
    an aid: nothing here may fail the run.
    """

    def __init__(self, counter: RunCounter) -> None:
        """Create the callback.

        Args:
            counter: Counter shared with the plan gate.
        """
        super().__init__()
        self._counter = counter
        # End and error events carry no tags or metadata, so a model run's
        # own are kept from its start until it ends.
        self._open: dict[UUID, tuple[Any, Any]] = {}
        self._open_lock = threading.Lock()

    def _add(
        self,
        count: str,
        tags: Sequence[str] | None,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        """Count one event of the run's thread and role, never raising.

        Args:
            count: Which count to raise.
            tags: Run tags.
            metadata: Run metadata.
        """
        try:
            thread_id = (metadata or {}).get("thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                return
            self._counter.add(thread_id, run_role(tags, metadata), count)
        except Exception:
            logger.warning("A run event could not be counted", exc_info=True)

    def _start(
        self,
        run_id: UUID,
        count: str | None,
        tags: Sequence[str] | None,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        """Keep a starting run's tags and metadata for its end, counting `count`.

        Args:
            run_id: Run identifier.
            count: Count to raise for the start, or `None`.
            tags: Run tags.
            metadata: Run metadata.
        """
        with self._open_lock:
            self._open[run_id] = (tags, metadata)
        if count is not None:
            self._add(count, tags, metadata)

    def _end(self, run_id: UUID, *counts: str) -> None:
        """Forget a finished run, counting each of `counts` for it.

        Args:
            run_id: Run identifier.
            *counts: Counts to raise for the run.
        """
        with self._open_lock:
            tags, metadata = self._open.pop(run_id, (None, None))
        for count in counts:
            self._add(count, tags, metadata)

    @override
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Count a model call."""
        del serialized, messages, kwargs
        self._start(run_id, MODEL_CALLS, tags, metadata)

    @override
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Count a call to a completion-style model."""
        del serialized, prompts, kwargs
        self._start(run_id, MODEL_CALLS, tags, metadata)

    @override
    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Forget a finished model call."""
        del response, kwargs
        self._end(run_id)

    @override
    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Count a failed model attempt; a retry follows it in a finished turn."""
        del kwargs
        if isinstance(error, GraphBubbleUp):
            self._end(run_id)
        else:
            self._end(run_id, MODEL_ERRORS)

    def count_tool_result(
        self,
        result: object,
        tags: Sequence[str] | None,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        """Count one finished tool call reported by `ToolResultCounterMiddleware`.

        A refusal of the plan gate is a call but not an error; the turn
        metrics count refusals on their own.

        Args:
            result: What the tool call returned to the agent.
            tags: Tags of the run the call was made in.
            metadata: Metadata of the run the call was made in.
        """
        self._add(TOOL_CALLS, tags, metadata)
        if _is_error_result(result) and not _is_gate_refusal(result):
            self._add(TOOL_ERRORS, tags, metadata)


def _active_counting_callback() -> (
    tuple[RunCounterCallback, list[str], dict[str, Any]] | None
):
    """Find the counting callback of the current run, with the run's tags and metadata.

    Returns:
        The callback, the run's tags and its metadata, or `None` outside a
            run or when no counting callback is bound to it.
    """
    from langgraph.config import get_config

    try:
        config = get_config()
    except RuntimeError:
        return None
    callbacks = config.get("callbacks")
    handlers = (
        callbacks
        if isinstance(callbacks, list)
        else getattr(callbacks, "handlers", None)
    )
    found = next((h for h in handlers or () if isinstance(h, RunCounterCallback)), None)
    if found is None:
        return None
    tags = [
        *(config.get("tags") or ()),
        *(getattr(callbacks, "inheritable_tags", None) or ()),
    ]
    metadata = {
        **(getattr(callbacks, "inheritable_metadata", None) or {}),
        **(config.get("metadata") or {}),
    }
    return found, tags, metadata


def _is_gate_refusal(result: object) -> bool:
    """Tell whether a tool result is the plan gate's refusal.

    Args:
        result: What the tool call returned.

    Returns:
        `True` when the result's text starts with the gate's refusal prefix.
    """
    messages = [result] if isinstance(result, ToolMessage) else []
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        messages += [
            m for m in update.get("messages") or () if isinstance(m, ToolMessage)
        ]
    return any(str(m.text).startswith(GATE_REFUSAL_PREFIX) for m in messages)


class ToolResultCounterMiddleware(AgentMiddleware[Any, ContextT]):
    """Count every tool call by the result it returns to the agent.

    Placed first in an agent's middleware list, it is the outermost tool
    wrapper and sees every result: those of tools that ran and those that a
    middleware returned without running the tool, such as a shell command the
    allow-list rejected or a call over a tool call limit. Tool callbacks see
    only tools that ran, which is why tool calls are counted here and not in
    `RunCounterCallback`.

    The middleware holds no counter. It uses the `RunCounterCallback` bound
    to the current run, so helper agents and subagents need no wiring, and it
    does nothing where no such callback is bound. An interrupt is not a
    result and is not counted; the resumed call is counted once.
    """

    @staticmethod
    def _count(result: object) -> None:
        """Count a result with the run's counting callback, never raising.

        Args:
            result: What the tool call returned.
        """
        try:
            active = _active_counting_callback()
            if active is not None:
                callback, tags, metadata = active
                callback.count_tool_result(result, tags, metadata)
        except Exception:
            logger.warning("A tool result could not be counted", exc_info=True)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Pass the call through and count its result.

        Returns:
            The tool call's result.
        """
        result = handler(request)
        self._count(result)
        return result

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Pass the call through and count its result.

        Returns:
            The tool call's result.
        """
        result = await handler(request)
        self._count(result)
        return result


def _total(counts: RoleCounts, count: str) -> int:
    """Sum one count over every role in `counts`.

    Returns:
        The sum.
    """
    return sum(by_role.get(count, 0) for by_role in counts.values())


def agent_work_metrics(counts: RoleCounts) -> dict[str, float]:
    """Shape the counts of the agent's working roles as turn metrics.

    Args:
        counts: Counts taken for `AGENT_WORK_ROLES`.

    Returns:
        Totals as `all_*` and the model calls of each working role as
            `model_calls.<role>`.
    """
    metrics: dict[str, float] = {
        f"all_{count}": _total(counts, count) for count in _COUNTS
    }
    for role in AGENT_WORK_ROLES:
        metrics[f"{MODEL_CALLS}.{role}"] = counts.get(role, {}).get(MODEL_CALLS, 0)
    return metrics


def post_work_metrics(counts: RoleCounts) -> dict[str, float]:
    """Shape the counts of everything after the agent's work as metrics.

    Args:
        counts: Counts left after the working roles were taken.

    Returns:
        Totals as `post_work_*` and the model calls of each post-work role
            as `model_calls.<role>`.
    """
    metrics: dict[str, float] = {
        f"post_work_{count}": _total(counts, count) for count in _COUNTS
    }
    for role in POST_WORK_ROLES:
        metrics[f"{MODEL_CALLS}.{role}"] = counts.get(role, {}).get(MODEL_CALLS, 0)
    return metrics


def turn_metrics(messages: Sequence[Any], started_at: float | None) -> dict[str, float]:
    """Count what happened since the latest user message.

    Args:
        messages: Conversation messages in order.
        started_at: Unix time at which the turn started, if known.

    Returns:
        Metric values keyed by name. Model retries are not among them:
            they are not visible in the conversation and are counted by
            `ModelRetryCounterMiddleware`. Token counts are what the
            provider reported. `input_tokens` is left out when the
            provider reported output tokens but no input tokens, so a
            missing count is not recorded as zero.
    """
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index + 1
    turn = messages[start:]
    replies = [m for m in turn if isinstance(m, AIMessage)]
    results = [m for m in turn if isinstance(m, ToolMessage)]
    errors = [m for m in results if m.status == "error"]
    refusals = [m for m in errors if str(m.text).startswith(GATE_REFUSAL_PREFIX)]
    usage = [m.usage_metadata or {} for m in replies]
    metrics: dict[str, float] = {
        "model_calls": len(replies),
        "tool_calls": len(results),
        "tool_errors": len(errors) - len(refusals),
        "gate_refusals": len(refusals),
        "review_rounds": sum(
            1
            for message in replies
            for call in message.tool_calls
            if call["name"] == REVIEW_TOOL
        ),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in usage),
    }
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in usage)
    # Every model call sends input. A zero next to real output means the
    # provider did not report the count, not that nothing was sent.
    if input_tokens or not metrics["output_tokens"]:
        metrics["input_tokens"] = input_tokens
    if started_at is not None:
        elapsed = datetime.now(UTC).timestamp() - started_at
        metrics["duration_seconds"] = round(max(elapsed, 0.0), 3)
    return metrics


def metrics_step(metrics: Mapping[str, float]) -> dict[str, Any]:
    """Shape turn metrics for the step log, naming what is missing.

    Args:
        metrics: Metric values from `turn_metrics`.

    Returns:
        The same values, plus `input_tokens` set to `None` and listed
            under `unreported` when the provider did not report it.
    """
    step: dict[str, Any] = dict(metrics)
    if "input_tokens" not in step:
        step["input_tokens"] = None
        step["unreported"] = ["input_tokens"]
    return step


# ----- reading ---------------------------------------------------------------


def read(root: Path, thread_id: str) -> list[dict[str, Any]]:
    """Read the recorded steps of a thread.

    Args:
        root: Project root.
        thread_id: Thread identifier.

    Returns:
        Steps in the order they were recorded. Lines that are not valid
            records are skipped; a missing file reads as no steps.
    """
    try:
        lines = trace_file(root, thread_id).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    steps: list[dict[str, Any]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("kind"), str):
            steps.append(entry)
    return steps


def threads(root: Path) -> list[str]:
    """List the threads that have a local step record.

    Args:
        root: Project root.

    Returns:
        Thread ids, most recently written first.
    """
    try:
        paths = sorted(
            (root / TRACE_DIRECTORY).glob("*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    return [path.stem for path in paths]


def first_failure(steps: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Find the first step where the work went wrong.

    Procedural refusals (`warning` steps) are passed over, so the result
    points at the cause rather than at the gate or a review doing its job.

    Args:
        steps: Steps in recorded order.

    Returns:
        The first `error` step, or `None` when there is none.
    """
    return next((s for s in steps if failure_level(s) == LEVEL_ERROR), None)


def procedural_refusals(steps: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """List the failed steps that are part of the procedure.

    Args:
        steps: Steps in recorded order.

    Returns:
        The `warning` steps, in recorded order.
    """
    return [s for s in steps if failure_level(s) == LEVEL_WARNING]


def _summary(step: Mapping[str, Any]) -> str:
    """Describe a step's data in one short line.

    Args:
        step: Recorded step.

    Returns:
        `key=value` pairs of the step's data, cut to a bounded length.
    """
    data = step.get("data")
    if not isinstance(data, dict):
        return ""
    parts = []
    for key, value in data.items():
        text = (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
        parts.append(f"{key}={' '.join(text.split())}")
    line = "  ".join(parts)
    if len(line) <= _SUMMARY_CHAR_LIMIT:
        return line
    return line[:_SUMMARY_CHAR_LIMIT] + "..."


def format_steps(steps: Sequence[Mapping[str, Any]]) -> str:
    """Format a thread's steps as a numbered list with its first failure.

    Args:
        steps: Steps in recorded order.

    Returns:
        One line per step, `✗` marking errors and `!` procedural refusals,
            then the number of procedural refusals and the first error in
            full, or a note that none was recorded.
    """
    if not steps:
        return "기록된 단계가 없습니다."
    marks = {LEVEL_ERROR: "✗", LEVEL_WARNING: "!"}
    lines = []
    for number, step in enumerate(steps, start=1):
        mark = marks.get(failure_level(step) or "", " ")
        time = str(step.get("time", ""))[11:19]
        lines.append(f"{number:3d} {mark} {time}  {step['kind']:<22} {_summary(step)}")
    refusals = procedural_refusals(steps)
    if refusals:
        kinds = ", ".join(sorted({str(step["kind"]) for step in refusals}))
        lines.append(f"\n절차상 거부(!): {len(refusals)}건 ({kinds})")
    failure = first_failure(steps)
    if failure is None:
        lines.append("\n첫 실패 지점: 없음")
    else:
        number = steps.index(failure) + 1
        detail = json.dumps(failure.get("data"), ensure_ascii=False, indent=2)
        lines.append(f"\n첫 실패 지점: {number}번 {failure['kind']}\n{detail}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the recorded threads of a project, or the steps of one thread.

    Args:
        argv: Command-line arguments; the process arguments when omitted.

    Returns:
        Process exit code: 0 on success, 1 when the thread has no record.
    """
    parser = argparse.ArgumentParser(
        prog="python -m deepagents_code.harness.trace_log",
        description="Show the harness steps recorded for a thread.",
    )
    parser.add_argument(
        "thread_id", nargs="?", help="Thread to show. Omit to list recorded threads."
    )
    parser.add_argument(
        "--root", default=".", help="Project root (default: current directory)."
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    out = sys.stdout
    if args.thread_id is None:
        found = threads(root)
        out.write("\n".join(found) + "\n" if found else "기록된 thread가 없습니다.\n")
        return 0
    steps = read(root, args.thread_id)
    out.write(format_steps(steps) + "\n")
    return 0 if steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
