"""Tests for the plan gate, driven through a real agent with a scripted model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents_code.harness import plan, plan_gate, trace_log
from deepagents_code.harness.plan_gate import PlanGateMiddleware
from deepagents_code.harness.plan_review import PlanReview, PlanReviewer
from tests.unit_tests.harness.test_plan import VALID_PLAN

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool

    from deepagents_code.harness.plan_report import WorkReviewer

THREAD = "thread-1"
CONFIG: RunnableConfig = {"configurable": {"thread_id": THREAD}}


class _ScriptedModel(GenericFakeChatModel):
    """Fake model that replays a fixed list of messages and ignores tools."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        del tools, tool_choice, kwargs
        return self


class _ScriptedReviewer(PlanReviewer):
    """Reviewer stand-in that returns prepared verdicts in order."""

    def __init__(self, *verdicts: PlanReview) -> None:
        super().__init__(agent=None)
        self._verdicts = list(verdicts)
        self.requests: list[str] = []

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
        del plan_text, plan_path, project_root, earlier_requests, context
        self.requests.append(request_text)
        return self._verdicts.pop(0)


class _Harness:
    """A small agent with stand-in workspace tools behind the plan gate."""

    def __init__(
        self,
        root: Path,
        *,
        reviewer: _ScriptedReviewer | None = None,
        work_reviewer: WorkReviewer | None = None,
        interactive: bool = True,
        allow_ai_approval: bool = False,
        record_experience: bool = False,
        extra_middleware: Sequence[AgentMiddleware[Any, Any]] = (),
    ) -> None:
        self.root = root
        self.request = "인사 함수를 추가해 줘"
        self.extra_middleware = list(extra_middleware)
        self.inner_middleware: list[AgentMiddleware[Any, Any]] = []
        self.extra_tools: list[BaseTool] = []
        self.executed: list[tuple[str, str]] = []
        self.gate = PlanGateMiddleware(
            root,
            reviewer=reviewer,
            work_reviewer=work_reviewer,
            interactive=interactive,
            allow_ai_approval=allow_ai_approval,
            record_experience=record_experience,
        )
        self._saver = InMemorySaver()
        self._calls = 0

    def _tools(self) -> list[BaseTool]:
        executed = self.executed
        root = self.root

        @tool
        def write_file(file_path: str, content: str) -> str:
            """Write a file."""
            target = plan.resolve_path(file_path, root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            executed.append(("write_file", file_path))
            return "written"

        @tool
        def edit_file(file_path: str) -> str:
            """Edit a file."""
            executed.append(("edit_file", file_path))
            return "edited"

        @tool
        def execute(command: str) -> str:
            """Run a shell command."""
            executed.append(("execute", command))
            return "ran"

        @tool
        def task(description: str) -> str:
            """Delegate to a subagent."""
            executed.append(("task", description))
            return "delegated"

        @tool
        def read_file(file_path: str) -> str:
            """Read a file."""
            executed.append(("read_file", file_path))
            return "content"

        return [write_file, edit_file, execute, task, read_file, *self.extra_tools]

    def call(self, name: str, **args: Any) -> AIMessage:
        self._calls += 1
        tool_call = {
            "name": name,
            "args": args,
            "id": f"call-{self._calls}",
            "type": "tool_call",
        }
        return AIMessage(content="", tool_calls=[tool_call])

    async def run(self, *steps: AIMessage, resume: object = None) -> dict[str, Any]:
        """Run one turn: the scripted tool calls, then a closing reply."""
        script = [*steps, AIMessage(content="done")]
        middleware: list[AgentMiddleware[Any, Any]] = [
            *self.extra_middleware,
            self.gate,
            *self.inner_middleware,
        ]
        agent = create_agent(
            model=_ScriptedModel(messages=iter(script)),
            tools=self._tools(),
            middleware=middleware,
            checkpointer=self._saver,
        )
        payload: Any = (
            Command(resume=resume)
            if resume is not None
            else {"messages": [HumanMessage(self.request)]}
        )
        result = await agent.ainvoke(payload, CONFIG)
        state = await agent.aget_state(CONFIG)
        return {"result": result, "state": state.values}

    @property
    def plan_path(self) -> Path:
        return plan.plan_file(self.root, THREAD)

    def write_plan(self, text: str = VALID_PLAN) -> None:
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        self.plan_path.write_text(text, encoding="utf-8")


def _tool_messages(outcome: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in outcome["result"]["messages"] if isinstance(m, ToolMessage)]


def _last_tool_message(outcome: dict[str, Any]) -> ToolMessage:
    return _tool_messages(outcome)[-1]


def _phase(outcome: dict[str, Any]) -> str:
    return outcome["state"].get("_plan_phase", plan_gate.PHASE_PLANNING)


async def _approved(harness: _Harness, *steps: AIMessage) -> dict[str, Any]:
    """Get the valid plan reviewed and approved, then run `steps` in that turn.

    An approval covers one user request, so the work that needs it is scripted
    to follow the approval before the turn ends. Calling this again starts a
    new request with a fresh review and approval.

    Returns:
        Outcome of the run that resumed after the approval.
    """
    harness.write_plan()
    await harness.run(
        harness.call(plan_gate.REVIEW_TOOL), harness.call(plan_gate.APPROVAL_TOOL)
    )
    outcome = await harness.run(
        *steps,
        resume={"status": "answered", "answers": [plan_gate.APPROVE_CHOICE, ""]},
    )
    if not steps:
        assert _phase(outcome) == plan_gate.PHASE_APPROVED
    return outcome


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("write_file", {"file_path": "src/app.py", "content": "x"}),
        ("edit_file", {"file_path": "src/app.py"}),
        ("execute", {"command": "echo x > src/app.py"}),
        ("task", {"description": "edit the file"}),
    ],
)
async def test_workspace_changes_are_refused_before_approval(
    tmp_path: Path, name: str, args: dict[str, Any]
) -> None:
    harness = _Harness(tmp_path)

    outcome = await harness.run(harness.call(name, **args))

    message = _last_tool_message(outcome)
    assert message.status == "error"
    assert "계획 관문" in str(message.content)
    assert str(harness.plan_path) in str(message.content)
    assert harness.executed == []
    assert not (tmp_path / "src" / "app.py").exists()


async def test_read_only_tools_run_without_a_plan(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)

    outcome = await harness.run(harness.call("read_file", file_path="src/app.py"))

    assert _last_tool_message(outcome).status == "success"
    assert harness.executed == [("read_file", "src/app.py")]


async def test_plan_file_can_be_written_while_planning(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)

    outcome = await harness.run(
        harness.call("write_file", file_path=str(harness.plan_path), content="# plan")
    )

    assert _last_tool_message(outcome).status == "success"
    assert harness.plan_path.read_text(encoding="utf-8") == "# plan"


@pytest.mark.parametrize(
    "relative",
    [
        ".deepagents/plans/thread-1.review.md",
        ".deepagents/plans/thread-1.report.md",
        ".deepagents/plans/notes.txt",
        ".deepagents/plans/../../src/app.py",
        ".deepagents/plansX/thread-1.md",
    ],
)
async def test_only_plan_files_in_the_plan_folder_are_exempt(
    tmp_path: Path, relative: str
) -> None:
    harness = _Harness(tmp_path)

    outcome = await harness.run(
        harness.call("write_file", file_path=relative, content="x")
    )

    assert _last_tool_message(outcome).status == "error"
    assert harness.executed == []


async def test_symlink_inside_the_plan_folder_is_not_an_exemption(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    outside = tmp_path / "src"
    outside.mkdir()
    harness.plan_path.parent.mkdir(parents=True)
    (harness.plan_path.parent / "link").symlink_to(outside)

    outcome = await harness.run(
        harness.call(
            "write_file", file_path=".deepagents/plans/link/evil.md", content="x"
        )
    )

    assert _last_tool_message(outcome).status == "error"
    assert not (outside / "evil.md").exists()


async def test_review_without_a_plan_file_reports_where_to_write_it(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)

    outcome = await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    message = _last_tool_message(outcome)
    assert message.status == "error"
    assert str(harness.plan_path) in str(message.content)
    assert plan.SECTION_TARGETS in str(message.content)


async def test_review_rejects_a_structurally_invalid_plan(tmp_path: Path) -> None:
    reviewer = _ScriptedReviewer()
    harness = _Harness(tmp_path, reviewer=reviewer)
    harness.write_plan(VALID_PLAN.replace("## 테스트 방법", "## 기타"))

    outcome = await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    message = _last_tool_message(outcome)
    assert message.status == "error"
    assert plan.SECTION_TESTS in str(message.content)
    assert reviewer.requests == []
    assert _phase(outcome) == plan_gate.PHASE_PLANNING


async def test_review_findings_send_the_plan_back(tmp_path: Path) -> None:
    reviewer = _ScriptedReviewer(
        PlanReview(verdict="revise", findings=["테스트 명령이 없습니다."])
    )
    harness = _Harness(tmp_path, reviewer=reviewer)
    harness.write_plan()

    outcome = await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    message = _last_tool_message(outcome)
    assert "테스트 명령이 없습니다." in str(message.content)
    assert _phase(outcome) == plan_gate.PHASE_PLANNING
    assert outcome["state"]["_plan_review_rounds"] == 1
    assert reviewer.requests == ["인사 함수를 추가해 줘"]
    record = harness.plan_path.with_name("thread-1.review.md")
    assert "수정 필요" in record.read_text(encoding="utf-8")


async def test_passed_review_moves_to_reviewed(tmp_path: Path) -> None:
    reviewer = _ScriptedReviewer(PlanReview(verdict="approve"))
    harness = _Harness(tmp_path, reviewer=reviewer)
    harness.write_plan()

    outcome = await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    assert _last_tool_message(outcome).status == "success"
    assert _phase(outcome) == plan_gate.PHASE_REVIEWED
    assert outcome["state"]["_plan_reviewed_hash"] == plan.hash_text(VALID_PLAN)


async def test_review_does_not_open_the_gate(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    harness.write_plan()

    outcome = await harness.run(
        harness.call(plan_gate.REVIEW_TOOL),
        harness.call("write_file", file_path="src/greeting.py", content="x"),
    )

    assert _phase(outcome) == plan_gate.PHASE_REVIEWED
    assert _last_tool_message(outcome).status == "error"
    assert harness.executed == []


async def test_approval_needs_a_passed_review(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    harness.write_plan()

    outcome = await harness.run(harness.call(plan_gate.APPROVAL_TOOL))

    assert "__interrupt__" not in outcome["result"]
    assert _last_tool_message(outcome).status == "error"
    assert _phase(outcome) == plan_gate.PHASE_PLANNING


async def test_approval_asks_the_human_with_an_ask_user_question(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    harness.write_plan()

    outcome = await harness.run(
        harness.call(plan_gate.REVIEW_TOOL), harness.call(plan_gate.APPROVAL_TOOL)
    )

    (pending,) = outcome["result"]["__interrupt__"]
    assert pending.value["type"] == "ask_user"
    question = pending.value["questions"][0]
    assert str(harness.plan_path) in question["question"]
    assert [choice["value"] for choice in question["choices"]] == [
        plan_gate.APPROVE_CHOICE,
        plan_gate.REVISE_CHOICE,
    ]
    assert _phase(outcome) == plan_gate.PHASE_REVIEWED


async def test_human_approval_opens_the_gate_for_planned_files(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    outcome = await _approved(
        harness,
        harness.call("write_file", file_path="src/greeting.py", content="x"),
        harness.call("write_file", file_path="tests/test_greeting.py", content="y"),
        harness.call("execute", command="pytest"),
    )

    assert [m.status for m in _tool_messages(outcome)[-3:]] == ["success"] * 3
    assert harness.executed == [
        ("write_file", "src/greeting.py"),
        ("write_file", "tests/test_greeting.py"),
        ("execute", "pytest"),
    ]
    record = harness.plan_path.with_name("thread-1.review.md").read_text("utf-8")
    assert "결정: 승인" in record
    assert "사용자" in record


async def test_files_outside_the_plan_stay_refused_after_approval(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    outcome = await _approved(
        harness, harness.call("write_file", file_path="src/other.py", content="x")
    )

    message = _last_tool_message(outcome)
    assert message.status == "error"
    assert "수정 대상" in str(message.content)
    assert harness.executed == []
    assert _phase(outcome) == plan_gate.PHASE_APPROVED


@pytest.mark.parametrize(
    "answer",
    [
        {
            "status": "answered",
            "answers": [plan_gate.REVISE_CHOICE, "범위를 줄여 주세요"],
        },
        {"status": "cancelled"},
        {"status": "answered", "answers": []},
        "approve",
    ],
)
async def test_anything_but_an_explicit_approval_keeps_the_gate_closed(
    tmp_path: Path, answer: object
) -> None:
    harness = _Harness(tmp_path)
    harness.write_plan()
    await harness.run(
        harness.call(plan_gate.REVIEW_TOOL), harness.call(plan_gate.APPROVAL_TOOL)
    )

    outcome = await harness.run(
        harness.call("write_file", file_path="src/greeting.py", content="x"),
        resume=answer,
    )

    assert _phase(outcome) == plan_gate.PHASE_PLANNING
    assert harness.executed == []
    assert _last_tool_message(outcome).status == "error"


async def test_rejection_note_reaches_the_model(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    harness.write_plan()
    await harness.run(
        harness.call(plan_gate.REVIEW_TOOL), harness.call(plan_gate.APPROVAL_TOOL)
    )

    outcome = await harness.run(
        resume={
            "status": "answered",
            "answers": [plan_gate.REVISE_CHOICE, "범위를 줄여 주세요"],
        }
    )

    assert "범위를 줄여 주세요" in str(_last_tool_message(outcome).content)


async def test_plan_changed_after_review_cannot_be_approved(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    harness.write_plan()
    await harness.run(harness.call(plan_gate.REVIEW_TOOL))
    harness.write_plan(VALID_PLAN + "\n추가 내용\n")

    outcome = await harness.run(harness.call(plan_gate.APPROVAL_TOOL))

    assert "__interrupt__" not in outcome["result"]
    assert _last_tool_message(outcome).status == "error"
    assert _phase(outcome) == plan_gate.PHASE_PLANNING


async def test_plan_changed_after_approval_voids_the_approval(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    changed = VALID_PLAN.replace("`src/greeting.py`", "`src/other.py`")
    plan_path = str(harness.plan_path)

    outcome = await _approved(
        harness,
        harness.call("write_file", file_path=plan_path, content=changed),
        harness.call("write_file", file_path="src/other.py", content="x"),
    )

    assert _last_tool_message(outcome).status == "error"
    assert harness.executed == [("write_file", plan_path)]
    assert _phase(outcome) == plan_gate.PHASE_PLANNING
    assert outcome["state"]["_plan_approved_hash"] is None


async def test_agent_edit_of_the_approved_plan_closes_the_gate(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    outcome = await _approved(
        harness,
        harness.call(
            "write_file",
            file_path=str(harness.plan_path),
            content=VALID_PLAN.replace("`src/greeting.py`", "`src/other.py`"),
        ),
        harness.call("write_file", file_path="src/other.py", content="x"),
    )

    assert _last_tool_message(outcome).status == "error"
    assert ("write_file", "src/other.py") not in harness.executed
    assert _phase(outcome) == plan_gate.PHASE_PLANNING


async def test_non_interactive_run_cannot_approve_by_default(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, interactive=False)
    harness.write_plan()

    outcome = await harness.run(
        harness.call(plan_gate.REVIEW_TOOL),
        harness.call(plan_gate.APPROVAL_TOOL),
        harness.call("write_file", file_path="src/greeting.py", content="x"),
    )

    assert "__interrupt__" not in outcome["result"]
    assert _phase(outcome) == plan_gate.PHASE_REVIEWED
    assert harness.executed == []


async def test_non_interactive_run_can_opt_in_to_ai_approval(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, interactive=False, allow_ai_approval=True)
    harness.write_plan()

    outcome = await harness.run(
        harness.call(plan_gate.REVIEW_TOOL),
        harness.call(plan_gate.APPROVAL_TOOL),
        harness.call("write_file", file_path="src/greeting.py", content="x"),
    )

    assert "__interrupt__" not in outcome["result"]
    assert _phase(outcome) == plan_gate.PHASE_APPROVED
    assert harness.executed == [("write_file", "src/greeting.py")]
    record = harness.plan_path.with_name("thread-1.review.md").read_text("utf-8")
    assert "AI 리뷰" in record


async def test_ai_approval_option_does_not_skip_the_review(tmp_path: Path) -> None:
    reviewer = _ScriptedReviewer(PlanReview(verdict="revise", findings=["부족합니다."]))
    harness = _Harness(
        tmp_path, reviewer=reviewer, interactive=False, allow_ai_approval=True
    )
    harness.write_plan()

    outcome = await harness.run(
        harness.call(plan_gate.REVIEW_TOOL),
        harness.call(plan_gate.APPROVAL_TOOL),
        harness.call("write_file", file_path="src/greeting.py", content="x"),
    )

    assert _phase(outcome) == plan_gate.PHASE_PLANNING
    assert harness.executed == []


async def test_ai_approval_refuses_a_plan_left_with_open_findings(
    tmp_path: Path,
) -> None:
    """At the review limit no review passed; with no human, nothing approves it."""
    rounds = plan_gate.MAX_REVIEW_ROUNDS
    reviewer = _ScriptedReviewer(
        *[PlanReview(verdict="revise", findings=["부족합니다."])] * rounds
    )
    harness = _Harness(
        tmp_path, reviewer=reviewer, interactive=False, allow_ai_approval=True
    )
    harness.write_plan()

    outcome = await harness.run(
        *[harness.call(plan_gate.REVIEW_TOOL) for _ in range(rounds + 1)],
        harness.call(plan_gate.APPROVAL_TOOL),
        harness.call("write_file", file_path="src/greeting.py", content="x"),
    )

    assert harness.executed == []
    assert _phase(outcome) != plan_gate.PHASE_APPROVED
    approval = [
        s for s in trace_log.read(tmp_path, THREAD) if s["kind"] == "plan.approval"
    ][-1]
    assert approval["data"]["decision"] == trace_log.APPROVAL_UNAVAILABLE
    assert approval["data"]["reason"] == trace_log.OPEN_FINDINGS_REASON
    assert approval["level"] == trace_log.LEVEL_ERROR
    record = harness.plan_path.with_name("thread-1.review.md").read_text("utf-8")
    assert "AI 리뷰 통과를 승인으로 인정" not in record


async def test_review_limit_hands_open_findings_to_the_human(tmp_path: Path) -> None:
    finding = "완료 조건을 검증할 수 없습니다."
    reviewer = _ScriptedReviewer(
        *[PlanReview(verdict="revise", findings=[finding])]
        * plan_gate.MAX_REVIEW_ROUNDS
    )
    harness = _Harness(tmp_path, reviewer=reviewer)
    harness.write_plan()
    reviews = [
        harness.call(plan_gate.REVIEW_TOOL) for _ in range(plan_gate.MAX_REVIEW_ROUNDS)
    ]

    # One request: the review rounds are counted within it.
    outcome = await harness.run(
        *reviews,
        harness.call(plan_gate.REVIEW_TOOL),
        harness.call(plan_gate.APPROVAL_TOOL),
    )

    assert len(reviewer.requests) == plan_gate.MAX_REVIEW_ROUNDS
    (pending,) = outcome["result"]["__interrupt__"]
    assert finding in pending.value["questions"][0]["question"]


def _request(name: str, phase: str | None = None, **args: object) -> ToolCallRequest:
    """Build a tool call request the way the approval layer sees one."""
    state = {} if phase is None else {"_plan_phase": phase}
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "c", "type": "tool_call"},
        tool=None,
        state=state,
        runtime=cast("Any", None),
    )


def test_refusal_is_predicted_without_touching_the_disk(tmp_path: Path) -> None:
    gate = PlanGateMiddleware(tmp_path)
    plan_path = str(plan.plan_file(tmp_path, THREAD))

    assert gate.will_refuse_without_io(_request("execute", command="ls"))
    assert gate.will_refuse_without_io(_request("write_file", file_path="src/a.py"))
    assert not gate.will_refuse_without_io(_request("write_file", file_path=plan_path))
    assert not gate.will_refuse_without_io(_request("read_file", file_path="src/a.py"))
    assert not gate.will_refuse_without_io(
        _request("execute", plan_gate.PHASE_APPROVED, command="ls")
    )


def test_guarded_approval_settings_skip_prompts_for_refused_calls(
    tmp_path: Path,
) -> None:
    gate = PlanGateMiddleware(tmp_path)
    asked: list[str] = []

    def when(request: ToolCallRequest) -> bool:
        asked.append(request.tool_call["name"])
        return True

    settings: dict[str, bool | InterruptOnConfig] = {
        "execute": {"allowed_decisions": ["approve"], "when": when},
        "web_search": {"allowed_decisions": ["approve"], "when": when},
        "task": True,
    }

    guarded = gate.guard_interrupt_on(settings)

    execute = guarded["execute"]
    web_search = guarded["web_search"]
    assert isinstance(execute, dict)
    assert isinstance(web_search, dict)
    assert execute["when"](_request("execute")) is False
    assert asked == []
    assert execute["when"](_request("execute", plan_gate.PHASE_APPROVED)) is True
    assert asked == ["execute"]
    assert web_search["when"] is when
    assert guarded["task"] is True


def test_notice_names_the_phase_and_the_plan_file(tmp_path: Path) -> None:
    gate = PlanGateMiddleware(tmp_path)
    runtime = SimpleNamespace(config=CONFIG)

    planning = gate.notice({}, runtime)
    approved = gate.notice({"_plan_phase": plan_gate.PHASE_APPROVED}, runtime)

    assert "Phase: planning" in planning
    assert str(plan.plan_file(tmp_path, THREAD)) in planning
    assert all(f"## {title}" in planning for title in plan.REQUIRED_SECTIONS)
    assert "Phase: approved" in approved


# ----- an approval covers one user request ----------------------------------


def _approved_copies(harness: _Harness) -> list[Path]:
    return sorted(harness.plan_path.parent.glob(f"*{plan.APPROVED_SUFFIX}"))


async def test_a_new_request_needs_its_own_approved_plan(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    first = await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )
    assert _last_tool_message(first).status == "success"

    second = await harness.run(
        harness.call("write_file", file_path="src/greeting.py", content="y")
    )

    message = _last_tool_message(second)
    assert message.status == "error"
    assert "아직 승인된 계획이 없어" in str(message.content)
    assert _phase(second) == plan_gate.PHASE_PLANNING
    assert second["state"]["_plan_approved_hash"] is None
    assert harness.executed == [("write_file", "src/greeting.py")]


async def test_the_approved_plan_is_kept_when_the_next_request_arrives(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    await _approved(harness)
    assert _approved_copies(harness) == []

    await harness.run(harness.call("read_file", file_path="src/app.py"))

    (copy,) = _approved_copies(harness)
    assert copy.name.startswith("thread-1.")
    assert copy.read_text(encoding="utf-8") == VALID_PLAN
    assert not plan.is_plan_file_name(copy.name)


async def test_the_agent_cannot_write_a_kept_plan_copy(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    forged = ".deepagents/plans/thread-1.20261112T000000.approved.md"

    outcome = await harness.run(
        harness.call("write_file", file_path=forged, content=VALID_PLAN)
    )

    assert _last_tool_message(outcome).status == "error"
    assert not (tmp_path / forged).exists()


async def test_an_unfinished_review_does_not_carry_over_to_the_next_request(
    tmp_path: Path,
) -> None:
    reviewer = _ScriptedReviewer(PlanReview(verdict="revise", findings=["부족합니다."]))
    harness = _Harness(tmp_path, reviewer=reviewer)
    harness.write_plan()
    first = await harness.run(harness.call(plan_gate.REVIEW_TOOL))
    assert first["state"]["_plan_review_rounds"] == 1

    second = await harness.run(harness.call("read_file", file_path="src/app.py"))

    assert second["state"]["_plan_review_rounds"] == 0
    assert second["state"]["_plan_review_findings"] == []


async def test_a_reviewed_plan_cannot_be_approved_in_a_later_request(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    harness.write_plan()
    await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    outcome = await harness.run(harness.call(plan_gate.APPROVAL_TOOL))

    message = _last_tool_message(outcome)
    assert message.status == "error"
    assert plan_gate.REVIEW_TOOL in str(message.content)
    assert _phase(outcome) == plan_gate.PHASE_PLANNING


async def test_a_thread_without_plan_state_is_left_alone(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)

    first = await harness.run(harness.call("read_file", file_path="src/app.py"))
    second = await harness.run(harness.call("read_file", file_path="src/app.py"))

    assert "_plan_phase" not in first["state"]
    assert "_plan_phase" not in second["state"]
    assert _approved_copies(harness) == []


# ----- MCP tools ------------------------------------------------------------


async def test_mcp_tools_are_not_gated(tmp_path: Path) -> None:
    """The gate covers dcode's own tools only.

    An MCP tool runs without a plan, whatever its annotations say. Gating
    every MCP tool that is not marked read-only was tried and taken back: it
    also blocked the lookups a plan is written from, and the assignment does
    not ask for it.
    """
    harness = _Harness(tmp_path)
    executed = harness.executed

    @tool
    def tracker_update(title: str) -> str:
        """Stand-in for a tool loaded from an MCP server."""
        executed.append(("tracker_update", title))
        return "updated"

    tracker_update.metadata = {"_deepagents_code_mcp": True}
    harness.extra_tools = [tracker_update]

    outcome = await harness.run(harness.call("tracker_update", title="x"))

    assert _last_tool_message(outcome).status == "success"
    assert harness.executed == [("tracker_update", "x")]


# ----- the reviewer judges the current request ------------------------------


def test_requests_are_split_into_the_current_one_and_earlier_ones() -> None:
    messages = [
        HumanMessage("첫 요청"),
        AIMessage(content="답"),
        HumanMessage("둘째\n요청"),
        AIMessage(content="답"),
        HumanMessage("지금 요청"),
    ]

    current, earlier = plan_gate.split_requests(messages)

    assert current == "지금 요청"
    assert earlier == ("첫 요청", "둘째 요청")
    assert plan_gate.split_requests([AIMessage(content="답")]) == ("", ())


async def test_plan_review_is_about_the_latest_request_only(tmp_path: Path) -> None:
    reviewer = _ScriptedReviewer(
        PlanReview(verdict="approve"), PlanReview(verdict="approve")
    )
    harness = _Harness(tmp_path, reviewer=reviewer)
    harness.write_plan()
    harness.request = "함수에 docstring을 추가해 줘"
    await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    harness.request = "NOTES.md를 만들어 줘"
    await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    assert reviewer.requests == ["함수에 docstring을 추가해 줘", "NOTES.md를 만들어 줘"]


def test_refusal_is_predicted_from_a_bare_tool_call(tmp_path: Path) -> None:
    """The interactive approval layer has tool calls, not requests."""
    gate = PlanGateMiddleware(tmp_path)
    plan_path = str(plan.plan_file(tmp_path, THREAD))

    def call(name: str, **args: object) -> dict[str, Any]:
        return {"name": name, "args": args, "id": "c", "type": "tool_call"}

    approved = {"_plan_phase": plan_gate.PHASE_APPROVED}

    assert gate.will_refuse_call(call("execute", command="ls"), {})
    assert gate.will_refuse_call(call("write_file", file_path="src/a.py"), {})
    assert not gate.will_refuse_call(call("write_file", file_path=plan_path), {})
    assert not gate.will_refuse_call(call("read_file", file_path="src/a.py"), {})
    assert not gate.will_refuse_call(call("execute", command="ls"), approved)


def test_failing_exemption_check_keeps_the_call_under_review(tmp_path: Path) -> None:
    """A broken predicate must not silently drop a call from approval."""
    from deepagents_code.auto_mode import AutoModeHITLMiddleware

    def broken(call: object, state: object) -> bool:
        del call, state
        msg = "boom"
        raise RuntimeError(msg)

    settings: dict[str, bool | InterruptOnConfig] = {"execute": True}
    execute = cast("Any", {"name": "execute", "args": {}, "id": "c"})
    read = cast("Any", {"name": "read_file", "args": {}, "id": "d"})

    failing = AutoModeHITLMiddleware(
        settings, worktree_root=tmp_path, review_exempt=broken
    )
    exempting = AutoModeHITLMiddleware(
        settings, worktree_root=tmp_path, review_exempt=lambda _call, _state: True
    )
    plain = AutoModeHITLMiddleware(settings, worktree_root=tmp_path)

    assert failing._needs_review(execute, {})
    assert not exempting._needs_review(execute, {})
    assert plain._needs_review(execute, {})
    assert not plain._needs_review(read, {})
