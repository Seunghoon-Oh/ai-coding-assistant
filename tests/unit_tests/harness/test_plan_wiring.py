"""Tests for how the plan gate is wired into the dcode agent assembly."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_code._env_vars import (
    EXPERIENCE_MEMORY,
    PLAN_GATE,
    PLAN_SELF_IMPROVE,
)
from deepagents_code._paths import get_built_in_skills_dir
from deepagents_code.agent import create_cli_agent
from deepagents_code.harness import plan, plan_review, trace_log
from deepagents_code.harness.experience import ExperienceMemoryMiddleware
from deepagents_code.harness.improvement import PromptAddendumMiddleware
from deepagents_code.harness.plan_gate import PlanGateMiddleware
from deepagents_code.harness.plan_review import (
    PlanReview,
    PlanReviewer,
    create_plan_reviewer,
)
from deepagents_code.harness.trace_log import (
    ModelRetryCounterMiddleware,
    RunCounter,
    RunCounterCallback,
)
from tests.unit_tests.harness.test_plan import VALID_PLAN

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel


def _settings(tmp_path: Path) -> Mock:
    """Build a settings stand-in that keeps agent assembly inside `tmp_path`.

    Args:
        tmp_path: Directory for the agent and skills folders.

    Returns:
        Mock exposing the settings accessors `create_cli_agent` reads.
    """
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    settings = Mock()
    settings.ensure_agent_dir.return_value = agent_dir
    settings.ensure_user_skills_dir.return_value = skills_dir
    settings.get_project_skills_dir.return_value = None
    settings.get_built_in_skills_dir.return_value = get_built_in_skills_dir()
    settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
    settings.get_project_agent_md_path.return_value = []
    settings.get_user_agents_dir.return_value = tmp_path / "agents"
    settings.get_project_agents_dir.return_value = None
    settings.project_root = None
    return settings


def _assembled_middleware(tmp_path: Path, **kwargs: Any) -> list[Any]:
    """Assemble the dcode agent and return the main agent's middleware.

    Args:
        tmp_path: Directory for the agent and skills folders.
        **kwargs: Extra arguments for `create_cli_agent`.

    Returns:
        The middleware list handed to `create_deep_agent`.
    """
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    model.profile = {"max_input_tokens": 200000}
    compiled = Mock()
    compiled.with_config.return_value = compiled
    with (
        patch("deepagents_code.agent.credentials", _settings(tmp_path)),
        patch("deepagents_code.agent.PluginSkillsMiddleware"),
        patch("deepagents_code.agent.MemoryMiddleware"),
        patch(
            "deepagents_code.agent.create_deep_agent", return_value=compiled
        ) as create,
        patch("deepagents._models.init_chat_model", return_value=model),
    ):
        create_cli_agent(
            model="fake-model",
            assistant_id="test",
            enable_memory=False,
            enable_skills=False,
            cwd=tmp_path,
            **kwargs,
        )
    return list(create.call_args.kwargs["middleware"])


def test_gate_is_installed_by_default(tmp_path: Path) -> None:
    middleware = _assembled_middleware(tmp_path)

    gates = [m for m in middleware if isinstance(m, PlanGateMiddleware)]
    assert len(gates) == 1


def test_gate_can_be_turned_off_by_environment(tmp_path: Path) -> None:
    environ = {**os.environ, PLAN_GATE: "0"}

    middleware = _assembled_middleware(tmp_path, environ=environ)

    assert not any(isinstance(m, PlanGateMiddleware) for m in middleware)


def test_experience_memory_is_installed_ahead_of_the_gate(tmp_path: Path) -> None:
    middleware = _assembled_middleware(tmp_path)

    kinds = [type(m) for m in middleware]
    assert kinds.count(ExperienceMemoryMiddleware) == 1
    assert kinds.index(ExperienceMemoryMiddleware) < kinds.index(PlanGateMiddleware)


def test_experience_memory_can_be_turned_off_by_environment(tmp_path: Path) -> None:
    environ = {**os.environ, EXPERIENCE_MEMORY: "0"}

    middleware = _assembled_middleware(tmp_path, environ=environ)

    kinds = [type(m) for m in middleware]
    assert ExperienceMemoryMiddleware not in kinds
    assert PlanGateMiddleware in kinds


def test_no_experience_memory_without_the_gate(tmp_path: Path) -> None:
    environ = {**os.environ, PLAN_GATE: "0"}

    middleware = _assembled_middleware(tmp_path, environ=environ)

    assert not any(isinstance(m, ExperienceMemoryMiddleware) for m in middleware)


def test_self_improvement_is_wired_by_default(tmp_path: Path) -> None:
    with patch("deepagents_code.harness.improvement.SelfImprover") as improver:
        middleware = _assembled_middleware(tmp_path)

    improver.assert_called_once()
    assert any(isinstance(m, PromptAddendumMiddleware) for m in middleware)


def test_self_improvement_can_be_turned_off_by_environment(tmp_path: Path) -> None:
    environ = {**os.environ, PLAN_SELF_IMPROVE: "0"}

    with patch("deepagents_code.harness.improvement.SelfImprover") as improver:
        middleware = _assembled_middleware(tmp_path, environ=environ)

    improver.assert_not_called()
    # Improvements applied earlier still reach the prompt.
    assert any(isinstance(m, PromptAddendumMiddleware) for m in middleware)


def test_retry_counter_sits_inside_the_retry_middleware(tmp_path: Path) -> None:
    from deepagents_code.model_retry import CodeModelRetryMiddleware

    middleware = _assembled_middleware(tmp_path)

    kinds = [type(m) for m in middleware]
    assert kinds.count(ModelRetryCounterMiddleware) == 1
    # Later in the list means further inside: the retry loop then calls the
    # counter once per attempt.
    assert kinds.index(ModelRetryCounterMiddleware) > kinds.index(
        CodeModelRetryMiddleware
    )


def test_no_retry_counter_without_the_gate(tmp_path: Path) -> None:
    environ = {**os.environ, PLAN_GATE: "0"}

    middleware = _assembled_middleware(tmp_path, environ=environ)

    assert not any(isinstance(m, ModelRetryCounterMiddleware) for m in middleware)


def test_reviewer_is_not_built_during_assembly(tmp_path: Path) -> None:
    with patch("deepagents_code.harness.plan_review.build_review_agent") as build_agent:
        reviewer = create_plan_reviewer(
            model="fake-model", root=tmp_path, model_retries=0
        )

    build_agent.assert_not_called()
    assert isinstance(reviewer, PlanReviewer)


async def test_reviewer_is_built_once_on_first_review() -> None:
    verdict = PlanReview(verdict="approve")
    built: list[_Agent] = []

    class _Agent:
        async def ainvoke(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"structured_response": verdict}

    def factory() -> _Agent:
        built.append(_Agent())
        return built[-1]

    reviewer = PlanReviewer(factory=factory)
    for _ in range(2):
        result = await reviewer.review(
            request_text="요청", plan_text="계획", plan_path="plan.md"
        )
        assert result.passed

    assert len(built) == 1


async def test_reviewer_build_failure_is_a_revise_verdict() -> None:
    def factory() -> object:
        msg = "model unavailable"
        raise ValueError(msg)

    reviewer = PlanReviewer(factory=factory)

    result = await reviewer.review(
        request_text="요청", plan_text="계획", plan_path="plan.md"
    )

    assert not result.passed
    assert "ValueError" in result.findings[0]


class _ToolCallingModel(GenericFakeChatModel):
    """Fake model that replays prepared messages and ignores tool binding."""

    def bind_tools(self, tools: object, **kwargs: object) -> _ToolCallingModel:
        del tools, kwargs
        return self


async def test_assembled_agent_blocks_code_writes_but_allows_the_plan_file(
    tmp_path: Path,
) -> None:
    thread_id = "wiring-thread"
    source = tmp_path / "src" / "greeting.py"
    plan_path = plan.plan_file(tmp_path, thread_id)

    def write(call_id: str, path: Path, content: str) -> AIMessage:
        args = {"file_path": str(path), "content": content}
        call = {"name": "write_file", "args": args, "id": call_id}
        return AIMessage(content="", tool_calls=[call])

    model = _ToolCallingModel(
        messages=iter(
            [
                write("code", source, "def greet(): ...\n"),
                write("plan", plan_path, VALID_PLAN),
                AIMessage(content="done"),
            ]
        ),
        profile={"max_input_tokens": 200000},
        disable_streaming=True,
    )
    graph, _ = create_cli_agent(
        model,
        "plan-wiring-test",
        cwd=tmp_path,
        checkpointer=InMemorySaver(),
        auto_approve=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="인사 함수를 추가해 줘")]},
        {"configurable": {"thread_id": thread_id}},
    )

    replies = {
        m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)
    }
    assert not source.exists()
    assert replies["code"].status == "error"
    assert plan_path.read_text(encoding="utf-8") == VALID_PLAN
    assert replies["plan"].status != "error"


# ----- the reviewer agent itself --------------------------------------------


class _ReviewerModel(GenericFakeChatModel):
    """Fake model for a reviewer agent; ignores tool binding."""

    def bind_tools(self, tools: object, **kwargs: object) -> _ReviewerModel:
        del tools, kwargs
        return self


def _reviewer_script(reads: int, final: AIMessage) -> list[AIMessage]:
    """Script a reviewer that lists the project `reads` times, then answers."""
    calls = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ls",
                    "args": {"path": "/"},
                    "id": f"r{n}",
                    "type": "tool_call",
                }
            ],
        )
        for n in range(reads)
    ]
    return [*calls, final]


def _verdict_call(verdict: str = "approve") -> AIMessage:
    call = {
        "name": "PlanReview",
        "args": {"verdict": verdict, "findings": []},
        "id": "verdict",
        "type": "tool_call",
    }
    return AIMessage(content="", tool_calls=[call])


def _reviewer(tmp_path: Path, *scripts: list[AIMessage]) -> PlanReviewer:
    """Build a real reviewer whose model replays one script per attempt."""
    pending = list(scripts)

    def build() -> object:
        model = _ReviewerModel(messages=iter(pending.pop(0)))
        model.profile = {"max_input_tokens": 200000}
        return plan_review.build_review_agent(
            system_prompt=plan_review.REVIEW_SYSTEM_PROMPT,
            schema=PlanReview,
            name="plan_reviewer",
            run_name="test",
            model=model,
            root=tmp_path,
            model_retries=0,
            cli_max_retries=None,
            environ=None,
        )

    class _FreshAgentReviewer(PlanReviewer):
        """Reviewer that builds a new agent, and so a new script, per attempt."""

        def _resolve_agent(self) -> object:
            return build()

    return _FreshAgentReviewer()


async def _run_review(reviewer: PlanReviewer) -> PlanReview:
    return await reviewer.review(
        request_text="요청", plan_text="계획", plan_path="plan.md"
    )


@pytest.mark.parametrize(
    "reads",
    [
        0,
        plan_review.REVIEW_TOOL_CALL_LIMIT - 2,
        plan_review.REVIEW_TOOL_CALL_LIMIT,
        plan_review.REVIEW_TOOL_CALL_LIMIT + 6,
    ],
)
async def test_reviewer_returns_its_verdict_after_spending_its_read_budget(
    tmp_path: Path, reads: int
) -> None:
    reviewer = _reviewer(tmp_path, _reviewer_script(reads, _verdict_call()))

    review = await _run_review(reviewer)

    assert review.passed


async def test_verdict_written_as_text_is_still_read(tmp_path: Path) -> None:
    text = (
        'Here is my verdict:\n{"verdict": "revise", "findings": ["테스트 명령 없음"]}'
    )
    reviewer = _reviewer(tmp_path, _reviewer_script(1, AIMessage(content=text)))

    review = await _run_review(reviewer)

    assert not review.passed
    assert review.findings == ["테스트 명령 없음"]


async def test_review_without_a_verdict_is_run_once_more(tmp_path: Path) -> None:
    reviewer = _reviewer(
        tmp_path,
        _reviewer_script(0, AIMessage(content="계획이 괜찮아 보입니다.")),
        _reviewer_script(0, _verdict_call()),
    )

    review = await _run_review(reviewer)

    assert review.passed


async def test_two_runs_without_a_verdict_are_a_failed_review(tmp_path: Path) -> None:
    reviewer = _reviewer(
        tmp_path,
        _reviewer_script(0, AIMessage(content="괜찮습니다.")),
        _reviewer_script(0, AIMessage(content="여전히 괜찮습니다.")),
    )

    review = await _run_review(reviewer)

    assert not review.passed
    assert "구조화된 응답이 없습니다" in review.findings[0]


# ----- subagents ------------------------------------------------------------


def _tool_call(call_id: str, name: str, **args: object) -> AIMessage:
    call = {"name": name, "args": args, "id": call_id, "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


@pytest.mark.parametrize("forked", ["1", "0"])
async def test_subagent_started_after_approval_can_write_the_approved_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forked: str
) -> None:
    """A forked subagent inherits the gate but not the approval.

    Before the gate learned to stand aside in a subagent, the forked child's
    write was refused as "no approved plan" although the parent's plan had
    been approved. Forking is the default.
    """
    monkeypatch.setenv("DEEPAGENTS_CODE_FORKED_SUBAGENTS", forked)
    monkeypatch.setenv("DEEPAGENTS_CODE_PLAN_AUTO_APPROVE", "1")
    monkeypatch.setenv("DEEPAGENTS_CODE_PLAN_SELF_IMPROVE", "0")
    monkeypatch.setenv("DEEPAGENTS_CODE_EXPERIENCE_MEMORY", "0")
    thread_id = "subagent-thread"
    plan_path = plan.plan_file(tmp_path, thread_id)
    target = tmp_path / "src" / "greeting.py"
    # One model plays every part in turn: the main agent, the plan reviewer
    # (third message) and the subagent (sixth and seventh).
    model = _ToolCallingModel(
        messages=iter(
            [
                _tool_call(
                    "p1", "write_file", file_path=str(plan_path), content=VALID_PLAN
                ),
                _tool_call("p2", "request_plan_review"),
                _tool_call("r1", "PlanReview", verdict="approve", findings=[]),
                _tool_call("p3", "request_plan_approval"),
                _tool_call(
                    "p4",
                    "task",
                    description="implement greet",
                    subagent_type="general-purpose",
                ),
                _tool_call("c1", "write_file", file_path=str(target), content="x\n"),
                AIMessage(content="child done"),
                AIMessage(content="parent done"),
            ]
        ),
        profile={"max_input_tokens": 200000},
        disable_streaming=True,
    )
    graph, _ = create_cli_agent(
        model,
        "subagent-test",
        cwd=tmp_path,
        checkpointer=InMemorySaver(),
        auto_approve=True,
        interactive=False,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
    )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="인사 함수를 추가해 줘")]},
        {"configurable": {"thread_id": thread_id}},
    )

    assert target.read_text(encoding="utf-8") == "x\n"
    kinds = [step["kind"] for step in trace_log.read(tmp_path, thread_id)]
    assert "gate.refused" not in kinds
    # The subagent's run is not a turn of its own.
    assert kinds.count("turn.metrics") == 1


def test_code_outside_any_run_is_not_in_a_subagent() -> None:
    assert trace_log.in_subagent() is False


async def test_reply_cut_off_before_the_verdict_is_reported_as_such(
    tmp_path: Path,
) -> None:
    """A reasoning model that spends its output budget thinking replies empty."""
    reviewer = _reviewer(
        tmp_path,
        _reviewer_script(0, AIMessage(content="")),
        _reviewer_script(0, AIMessage(content="")),
    )

    review = await _run_review(reviewer)

    assert not review.passed
    assert "max_tokens" in review.findings[0]
    assert "구조화된 응답이 없습니다" not in review.findings[0]


# ----- the interactive approval layer ---------------------------------------


async def _interactive_first_call(
    tmp_path: Path, name: str, args: dict[str, str]
) -> dict[str, Any]:
    """Run one scripted tool call through the agent the TUI assembles.

    The TUI builds the agent with Auto mode available, which installs a
    different approval middleware than the headless agent.

    Args:
        tmp_path: Project root.
        name: Tool the scripted model calls first.
        args: Arguments of that call.

    Returns:
        Final graph state, including any pending approval interrupt.
    """
    call = {"name": name, "args": args, "id": "first"}
    model = _ToolCallingModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[call]), AIMessage(content="done")]
        ),
        profile={"max_input_tokens": 200000},
        disable_streaming=True,
    )
    graph, _ = create_cli_agent(
        model,
        "plan-wiring-test",
        cwd=tmp_path,
        checkpointer=InMemorySaver(),
        auto_approve=False,
        interactive=True,
        auto_mode_enabled=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=True,
    )
    return await graph.ainvoke(
        {"messages": [HumanMessage(content="인사 함수를 추가해 줘")]},
        {"configurable": {"thread_id": "interactive-thread"}},
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("execute", {"command": "ls -a"}),
        ("write_file", {"file_path": "src/greeting.py", "content": "x = 1\n"}),
    ],
)
async def test_interactive_agent_does_not_ask_about_calls_the_gate_refuses(
    tmp_path: Path, name: str, args: dict[str, str]
) -> None:
    """Approving a call that the gate then refuses would be a pointless prompt."""
    result = await _interactive_first_call(tmp_path, name, args)

    replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert not result.get("__interrupt__")
    assert [reply.status for reply in replies] == ["error"]
    assert str(replies[0].content).startswith("[계획 관문]")


async def test_interactive_agent_still_asks_before_writing_the_plan_file(
    tmp_path: Path,
) -> None:
    """The exemption covers refused calls only, not every call before approval."""
    plan_path = plan.plan_file(tmp_path, "interactive-thread")

    result = await _interactive_first_call(
        tmp_path, "write_file", {"file_path": str(plan_path), "content": VALID_PLAN}
    )

    assert result.get("__interrupt__")
    assert not plan_path.exists()


# ----- counting every role's calls ------------------------------------------


def _assembled_graph(
    tmp_path: Path,
    model: BaseChatModel,
    *,
    auto_approve: bool = True,
    enable_shell: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    environ: dict[str, str] | None = None,
) -> Any:  # noqa: ANN401
    graph, _ = create_cli_agent(
        model,
        "count-wiring-test",
        cwd=tmp_path,
        checkpointer=InMemorySaver(),
        auto_approve=auto_approve,
        enable_memory=False,
        enable_skills=False,
        enable_shell=enable_shell,
        interrupt_shell_only=interrupt_shell_only,
        shell_allow_list=shell_allow_list,
        environ=environ,
    )
    return graph


def _counting_callbacks(graph: Any) -> list[RunCounterCallback]:  # noqa: ANN401
    callbacks = (graph.config or {}).get("callbacks") or []
    return [c for c in callbacks if isinstance(c, RunCounterCallback)]


def _replying_model(*replies: AIMessage) -> _ToolCallingModel:
    model = _ToolCallingModel(messages=iter(replies), disable_streaming=True)
    model.profile = {"max_input_tokens": 200000}
    return model


def test_counting_callback_is_bound_to_the_assembled_graph(tmp_path: Path) -> None:
    graph = _assembled_graph(tmp_path, _replying_model(AIMessage(content="ok")))

    assert len(_counting_callbacks(graph)) == 1


def test_no_counting_callback_without_the_gate(tmp_path: Path) -> None:
    graph = _assembled_graph(
        tmp_path,
        _replying_model(AIMessage(content="ok")),
        environ={**os.environ, PLAN_GATE: "0"},
    )

    assert _counting_callbacks(graph) == []


async def test_assembled_agent_records_the_calls_it_made(tmp_path: Path) -> None:
    thread_id = "count-thread"
    read = {"name": "read_file", "args": {"file_path": "a.py"}, "id": "r1"}
    graph = _assembled_graph(
        tmp_path,
        _replying_model(
            AIMessage(content="", tool_calls=[read]), AIMessage(content="done")
        ),
    )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="a.py를 읽어 줘")]},
        {"configurable": {"thread_id": thread_id}},
    )

    steps = trace_log.read(tmp_path, thread_id)
    (metrics,) = [s["data"] for s in steps if s["kind"] == "turn.metrics"]
    assert metrics["all_model_calls"] == metrics["model_calls.main"] == 2
    assert metrics["all_tool_calls"] == 1
    assert metrics["all_tool_errors"] == 1  # a.py does not exist
    # Nothing changed, so no post-work ran and none was recorded.
    assert "turn.post_work" not in [s["kind"] for s in steps]


async def test_reviewer_calls_are_counted_under_the_plan_review_role(
    tmp_path: Path,
) -> None:
    counter = RunCounter()
    reviewer = _reviewer(tmp_path, _reviewer_script(2, _verdict_call()))

    async def review(_: object) -> PlanReview:
        return await _run_review(reviewer)

    await RunnableLambda(review).ainvoke(
        None,
        {
            "configurable": {"thread_id": "t"},
            "callbacks": [RunCounterCallback(counter)],
        },
    )

    counts = counter.take("t")
    assert set(counts) == {plan_review.ROLE_PLAN_REVIEW}
    # Two `ls` reads and the verdict: three model calls, two tool calls.
    assert counts["plan_review"][trace_log.MODEL_CALLS] == 3
    assert counts["plan_review"][trace_log.TOOL_CALLS] == 2


async def test_a_command_the_allow_list_rejects_is_counted_as_a_tool_error(
    tmp_path: Path,
) -> None:
    """A middleware's own error reply never runs the tool; it still counts."""
    thread_id = "allow-list-thread"
    rejected = {"name": "execute", "args": {"command": "rm -rf build"}, "id": "x1"}
    graph = _assembled_graph(
        tmp_path,
        _replying_model(
            AIMessage(content="", tool_calls=[rejected]), AIMessage(content="done")
        ),
        enable_shell=True,
        interrupt_shell_only=True,
        shell_allow_list=["ls"],
        auto_approve=False,
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="빌드 폴더를 지워 줘")]},
        {"configurable": {"thread_id": thread_id}},
    )

    (reply,) = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert str(reply.text).startswith("Shell command rejected")
    steps = trace_log.read(tmp_path, thread_id)
    (metrics,) = [s["data"] for s in steps if s["kind"] == "turn.metrics"]
    assert metrics["tool_errors"] == 1
    assert metrics["all_tool_errors"] == metrics["tool_errors"]
    assert metrics["all_tool_calls"] == metrics["tool_calls"] == 1


async def test_reviewer_reads_are_counted_up_to_its_tool_call_limit(
    tmp_path: Path,
) -> None:
    """Calls over the limit never reach the tools and are not counted.

    `ToolCallLimitMiddleware` answers them from its `after_model` hook, not
    through the tool-call path, so no tool wrapper sees them (README 7절).
    """
    reads = plan_review.REVIEW_TOOL_CALL_LIMIT + 2
    counter = RunCounter()
    reviewer = _reviewer(tmp_path, _reviewer_script(reads, _verdict_call()))

    async def review(_: object) -> PlanReview:
        return await _run_review(reviewer)

    await RunnableLambda(review).ainvoke(
        None,
        {
            "configurable": {"thread_id": "t"},
            "callbacks": [RunCounterCallback(counter)],
        },
    )

    counts = counter.take("t")[plan_review.ROLE_PLAN_REVIEW]
    assert counts[trace_log.TOOL_CALLS] == plan_review.REVIEW_TOOL_CALL_LIMIT
    assert counts[trace_log.TOOL_ERRORS] == 0
