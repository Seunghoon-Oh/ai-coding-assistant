"""AI review of a plan before the human is asked to approve it.

The reviewer is a separate agent with its own context. It sees the user's
request and the plan, and can read the repository with read-only tools. It
returns a structured verdict; anything it fails to return counts as a request to
revise, never as approval.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

MAX_FINDINGS = 5
"""Upper bound on findings a review may return."""

FAILED_REVIEW_PREFIX = "리뷰어가 판정을 내지 못했습니다:"
"""Start of the only finding of a review that produced no verdict."""

HELPER_TAG_PREFIX = "harness:"
"""Prefix of the run tag naming the role of a helper agent.

Tags are inherited by every run below the tagged one, so the model and tool
calls a helper makes can be counted under its role.
"""

ROLE_PLAN_REVIEW = "plan_review"
ROLE_WORK_REVIEW = "work_review"
ROLE_IMPROVEMENT = "improvement"

_HELPER_ROLES = {
    "plan_reviewer": ROLE_PLAN_REVIEW,
    "plan_work_reviewer": ROLE_WORK_REVIEW,
}
_IMPROVEMENT_NAME_PREFIX = "improvement_"


def helper_role(name: str) -> str:
    """Return the role a helper agent is counted under.

    Args:
        name: Graph name given to `build_review_agent`.

    Returns:
        `plan_review`, `work_review` or `improvement` for the harness's own
            helpers; the name itself for any other helper.
    """
    if name.startswith(_IMPROVEMENT_NAME_PREFIX):
        return ROLE_IMPROVEMENT
    return _HELPER_ROLES.get(name, name)


REVIEW_ATTEMPTS = 2
"""Times a review is run before it is reported as having no verdict.

A model sometimes ends with a plain text answer instead of the structured
verdict. One more run usually gets the verdict, and costs less than a
review round spent on a failure.
"""

REVIEW_TOOL_CALL_LIMIT = 12
"""Read-only tool calls a reviewer may spend on one review."""

REVIEW_RECURSION_LIMIT = REVIEW_TOOL_CALL_LIMIT * 4 + 12
"""Graph steps one review may take.

A tool call costs three steps here: the model node, the tool-call-limit
hook after it, and the tools node. The limit leaves room for the whole tool
budget, for a few calls refused once the budget is spent, and for the final
verdict. A smaller limit ends the review with a recursion error before the
budget is used.
"""

REVIEW_SYSTEM_PROMPT = """\
You review an implementation plan for a coding task before any code is changed.

You are given the user's request and the plan. You may read the repository with
the read-only tools to check the plan against the actual code. You cannot change
anything.

Check the plan for these problems:
- A requirement in the request that the plan does not cover.
- Scope that goes beyond the request, or a missing statement of what is out of
  scope.
- Completion criteria that cannot be verified.
- Files under "수정 대상" that do not exist where the plan expects them, that are
  missing for the described work, or that are broader than the work needs (for
  example a whole source folder for a one-file change).
- Steps under "작업 순서" that are out of order or leave out a necessary step.
- A "테스트 방법" that does not say what to run or what result counts as passing.

Return `approve` only when you found no problem that should block the work.
Otherwise return `revise` with at most five findings. State each finding as one
concrete sentence the author can act on, written in the language of the
plan. Do not rewrite the plan.
"""


class PlanReview(BaseModel):
    """Verdict of one plan review."""

    verdict: Literal["approve", "revise"] = Field(
        description="`approve` when the plan can go to the human, else `revise`."
    )
    findings: list[str] = Field(
        default_factory=list,
        max_length=MAX_FINDINGS,
        description="Concrete problems to fix. Empty when the verdict is `approve`.",
    )

    @property
    def passed(self) -> bool:
        """Whether the plan may be shown to the human for approval."""
        return self.verdict == "approve"


def read_structured[ResponseT: BaseModel](
    result: object, schema: type[ResponseT]
) -> ResponseT | None:
    """Read a helper agent's structured answer from its result state.

    The answer normally arrives as `structured_response`. When the model
    answered in plain text instead, the last message is searched for a JSON
    object of the expected shape, which recovers a verdict the model wrote out
    but did not submit through the structured-output tool.

    Args:
        result: Result state of a helper agent run.
        schema: Expected answer type.

    Returns:
        The answer, or `None` when neither form holds a valid one.
    """
    if not isinstance(result, dict):
        return None
    response = result.get("structured_response")
    if isinstance(response, schema):
        return response
    if isinstance(response, dict):
        try:
            return schema.model_validate(response)
        except ValueError:
            logger.warning("Invalid %s response", schema.__name__, exc_info=True)
    messages = result.get("messages")
    last = messages[-1] if isinstance(messages, list) and messages else None
    text = str(getattr(last, "text", ""))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return schema.model_validate_json(text[start : end + 1])
    except ValueError:
        return None


def failed_review(reason: str) -> PlanReview:
    """Build the verdict used when the reviewer produced no usable answer.

    Args:
        reason: Why no verdict is available.

    Returns:
        A `revise` verdict carrying the reason as its only finding.
    """
    return PlanReview(verdict="revise", findings=[f"{FAILED_REVIEW_PREFIX} {reason}"])


HARNESS_NOTE = (
    "The `.deepagents/` folder under the project root belongs to the harness "
    "that runs the agent. It holds the plan, review records, post-work "
    "reports, memory records, improvement records and step logs. The harness "
    "writes these files itself; the agent can write only the plan file. They "
    "are not changes made by the work. Do not count them, or their showing up "
    "in `git status`, as unplanned changes, and do not ask a plan to list, "
    "explain or clean them up.\n\n"
)
"""Paragraph telling a reviewer which files are not the agent's doing.

Without it a reviewer sees these files in `git status` output and reports them
as changes the plan does not cover.
"""


def earlier_requests_note(earlier_requests: Sequence[str]) -> str:
    """Tell a reviewer which requests belong to earlier turns.

    An approval covers one request, so a review is about the latest request
    only. Shown the whole conversation as "the request", a reviewer asks the
    plan to cover work that was finished in an earlier turn.

    Args:
        earlier_requests: Requests of earlier turns, oldest first.

    Returns:
        A tagged block for the reviewer's input, or an empty string when the
            conversation has no earlier request.
    """
    if not earlier_requests:
        return ""
    listed = "\n".join(f"- {item}" for item in earlier_requests)
    return (
        "<earlier_requests>\n"
        "Requests from earlier turns of this conversation. They were handled "
        "in those turns and are context only: do not expect the plan or the "
        "work to cover them again.\n"
        f"{listed}\n"
        "</earlier_requests>\n\n"
    )


def no_verdict_reason(result: object) -> str:
    """Say why a helper run produced no structured answer.

    A reasoning model can spend its whole output budget on thinking and stop
    before it answers. The reply is then empty: no text and no tool call. That
    case is named, because the remedy is a setting and not a better plan.

    Args:
        result: Result state of the run.

    Returns:
        Reason in words, for the agent and the review record.
    """
    messages = result.get("messages") if isinstance(result, dict) else None
    last = messages[-1] if isinstance(messages, list) and messages else None
    if (
        last is not None
        and not str(getattr(last, "text", "")).strip()
        and not getattr(last, "tool_calls", None)
    ):
        return (
            "리뷰어가 판정을 내기 전에 응답이 끝났습니다(본문과 판정이 없는 빈 "
            "응답). 모델이 생각에 출력 토큰을 다 썼을 수 있습니다. 모델 설정의 "
            "`max_tokens`를 올려야 합니다"
        )
    return "구조화된 응답이 없습니다"


def root_note(project_root: str) -> str:
    """Explain to a reviewer how host paths map to its file tools.

    A reviewer's file tools show the project root as `/`. Without this note a
    reviewer looks up an absolute host path, finds nothing, and reports a
    correct path as missing.

    Args:
        project_root: Absolute path of the project root on the host, or an
            empty string when it is unknown.

    Returns:
        A paragraph for the reviewer's input, or an empty string when the root
            is unknown.
    """
    if not project_root:
        return ""
    return (
        f"The project root on the host is `{project_root}`. Your file tools "
        "show that folder as `/`: read the host path "
        f"`{project_root}/src/app.py` as `/src/app.py`. A path that starts "
        "with the project root is valid; do not report it as missing or as "
        "hard-coded only because it is absolute.\n\n"
    )


def review_prompt(
    request_text: str,
    plan_text: str,
    plan_path: str,
    project_root: str = "",
    earlier_requests: Sequence[str] = (),
) -> str:
    """Build the reviewer's input message.

    Args:
        request_text: The request the plan answers.
        plan_text: Plan file content.
        plan_path: Location of the plan file, for the reviewer's reference.
        project_root: Absolute path of the project root on the host.
        earlier_requests: Requests of earlier turns, given as context only.

    Returns:
        Message text with the request and the plan in separate tagged blocks.
    """
    return (
        "Review this plan.\n\n"
        f"{root_note(project_root)}"
        f"{HARNESS_NOTE}"
        f"{earlier_requests_note(earlier_requests)}"
        f"<user_request>\n{request_text.strip()}\n</user_request>\n\n"
        f'<plan path="{plan_path}">\n{plan_text.strip()}\n</plan>'
    )


class PlanReviewer:
    """Runs the review agent and converts its output to a verdict."""

    def __init__(
        self,
        agent: Any = None,  # noqa: ANN401  # compiled graph
        *,
        factory: Callable[[], Any] | None = None,
    ) -> None:
        """Wrap a review agent, or a factory that builds one on first use.

        Building the agent resolves the chat model, so the factory form keeps
        that work out of agent assembly: a session that never requests a
        review never builds a reviewer.

        Args:
            agent: Agent returning a `PlanReview` as its structured response.
            factory: Builds the agent on the first review when `agent` is not
                given.
        """
        self._agent = agent
        self._factory = factory

    def _resolve_agent(self) -> Any:  # noqa: ANN401  # compiled graph
        """Return the review agent, building it on first use.

        Returns:
            The compiled review agent.

        Raises:
            RuntimeError: If neither an agent nor a factory was provided.
        """
        if self._agent is None:
            if self._factory is None:
                msg = "PlanReviewer has neither an agent nor a factory"
                raise RuntimeError(msg)
            self._agent = self._factory()
        return self._agent

    async def review(
        self,
        *,
        request_text: str,
        plan_text: str,
        plan_path: str,
        project_root: str = "",
        earlier_requests: Sequence[str] = (),
        context: object | None = None,
    ) -> PlanReview:
        """Review one plan.

        Args:
            request_text: The request the plan answers.
            plan_text: Plan file content.
            plan_path: Location of the plan file.
            project_root: Absolute path of the project root on the host.
            earlier_requests: Requests of earlier turns, given as context
                only.
            context: Runtime context of the calling agent, passed through so
                the reviewer uses the same model selection and credentials.

        Returns:
            The reviewer's verdict, or a `revise` verdict when the reviewer
                failed or returned something that is not a verdict.
        """
        from langchain_core.messages import HumanMessage

        message = HumanMessage(
            review_prompt(
                request_text, plan_text, plan_path, project_root, earlier_requests
            )
        )
        reason = "구조화된 응답이 없습니다"
        for _ in range(REVIEW_ATTEMPTS):
            try:
                agent = self._resolve_agent()
                result = await agent.ainvoke({"messages": [message]}, context=context)
            except Exception as exc:
                logger.warning("Plan reviewer failed", exc_info=True)
                reason = type(exc).__name__
                continue
            review = read_structured(result, PlanReview)
            if review is not None:
                return review
            reason = no_verdict_reason(result)
        return failed_review(reason)


def create_plan_reviewer(
    *,
    model: str | BaseChatModel,
    root: Path,
    model_retries: int,
    cli_max_retries: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> PlanReviewer:
    """Build the review agent for a project.

    The agent mirrors how dcode builds its other read-only helper agents: the
    same runtime model selection and retry handling as the main agent, and a
    filesystem view confined to the project root.

    Args:
        model: Chat model or model identifier of the main agent.
        root: Project root the reviewer may read.
        model_retries: Model-node retry attempts after the first call.
        cli_max_retries: Explicit `--max-retries` value for runtime switches.
        environ: Workspace environment retained for runtime model switches.

    Returns:
        Reviewer bound to the project. Its agent is built on the first review.
    """

    def build() -> Any:  # noqa: ANN401  # compiled graph
        """Build the read-only agent that reviews the plan.

        Returns:
            The compiled review agent.
        """
        return build_review_agent(
            system_prompt=REVIEW_SYSTEM_PROMPT,
            schema=PlanReview,
            name="plan_reviewer",
            run_name="Deep Agents Code plan review",
            model=model,
            root=root,
            model_retries=model_retries,
            cli_max_retries=cli_max_retries,
            environ=environ,
        )

    return PlanReviewer(factory=build)


def build_review_agent(
    *,
    system_prompt: str,
    schema: type[BaseModel] | None,
    name: str,
    run_name: str,
    model: str | BaseChatModel,
    root: Path,
    model_retries: int,
    cli_max_retries: int | None,
    environ: Mapping[str, str] | None,
    read_tools: bool = True,
) -> Any:  # noqa: ANN401  # compiled graph
    """Compile a helper agent that cannot change anything.

    Args:
        system_prompt: Instructions for the agent.
        schema: Structured response the agent must return, or `None` for a
            plain text answer.
        name: Graph name of the agent. It also decides the role tag
            (`helper_role`) under which the agent's calls are counted.
        run_name: Run name shown in traces.
        model: Chat model or model identifier of the main agent.
        root: Project root the reviewer may read.
        model_retries: Model-node retry attempts after the first call.
        cli_max_retries: Explicit `--max-retries` value for runtime switches.
        environ: Workspace environment retained for runtime model switches.
        read_tools: Whether the agent may read the project with `ls`,
            `read_file`, `glob` and `grep`. Without them it answers from its
            input alone.

    Returns:
        Compiled agent. With a `schema` it returns an instance of it as its
            structured response.
    """
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware import FilesystemMiddleware
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware
    from langchain.agents.structured_output import ToolStrategy

    from deepagents_code._cli_context import CLIContextSchema
    from deepagents_code.configurable_model import ConfigurableModelMiddleware
    from deepagents_code.harness.trace_log import ToolResultCounterMiddleware
    from deepagents_code.model_retry import CodeModelRetryMiddleware

    middleware: list[AgentMiddleware[Any, Any]] = [
        # Outermost tool wrapper: counts results the tool call limit returns too.
        ToolResultCounterMiddleware(),
        ConfigurableModelMiddleware(
            persist_model_state=False,
            cli_max_retries=cli_max_retries,
            environ=environ,
        ),
        CodeModelRetryMiddleware(
            max_retries=model_retries,
            stream_output_is_visible=False,
        ),
    ]
    if read_tools:
        middleware += [
            FilesystemMiddleware(
                backend=FilesystemBackend(root_dir=root, virtual_mode=True),
                tools=["ls", "read_file", "glob", "grep"],
            ),
            ToolCallLimitMiddleware(
                run_limit=REVIEW_TOOL_CALL_LIMIT,
                exit_behavior="continue",
            ),
        ]
    return create_agent(
        model=model,
        tools=[],
        middleware=middleware,
        system_prompt=system_prompt,
        response_format=ToolStrategy(schema=schema) if schema is not None else None,
        context_schema=CLIContextSchema,
        name=name,
    ).with_config(
        {
            "recursion_limit": REVIEW_RECURSION_LIMIT,
            "run_name": run_name,
            "tags": [f"{HELPER_TAG_PREFIX}{helper_role(name)}"],
        }
    )
