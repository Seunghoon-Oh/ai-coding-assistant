"""Tests for the post-work report: evidence collection, comparison, writing."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents_code.harness import plan, plan_report, plan_review
from deepagents_code.harness.plan_report import (
    TurnActivity,
    WorkReview,
    WorkReviewer,
)
from tests.unit_tests.harness.test_plan import VALID_PLAN
from tests.unit_tests.harness.test_plan_gate import THREAD, _approved, _Harness

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from langchain_core.messages import BaseMessage


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git work tree with one committed file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "README.md").write_text("readme\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _call(call_id: str, name: str, **args: Any) -> AIMessage:
    tool_call = {"name": name, "args": args, "id": call_id, "type": "tool_call"}
    return AIMessage(content="", tool_calls=[tool_call])


def _result(call_id: str, text: str = "ok", status: str = "success") -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=call_id, status=status)


def _plan_path(root: Path) -> Path:
    path = plan.plan_file(root, THREAD)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(VALID_PLAN, encoding="utf-8")
    return path


def _activity(*edited: str) -> TurnActivity:
    return TurnActivity(edited=edited, commands=(), final_reply="")


# ----- collecting the turn's activity ---------------------------------------


def test_activity_covers_only_the_turn_after_the_latest_user_message() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("첫 요청"),
        _call("a", "write_file", file_path="src/old.py", content="x"),
        _result("a"),
        AIMessage(content="첫 답"),
        HumanMessage("둘째 요청"),
        _call("b", "edit_file", file_path="src/new.py"),
        _result("b"),
        _call("c", "execute", command="pytest -q"),
        _result("c", "2 passed"),
        AIMessage(content="끝났습니다"),
    ]

    activity = plan_report.collect_turn_activity(messages)

    assert activity.edited == ("src/new.py",)
    assert activity.commands == (("pytest -q", "2 passed"),)
    assert activity.final_reply == "끝났습니다"
    assert activity.changed_something


def test_refused_or_failed_calls_are_not_activity() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("요청"),
        _call("a", "write_file", file_path="src/a.py", content="x"),
        _result("a", "[계획 관문] 거부", status="error"),
        _call("b", "execute", command="rm -rf build"),
        _result("b", "[계획 관문] 거부", status="error"),
        _call("c", "read_file", file_path="src/a.py"),
        _result("c"),
        AIMessage(content="계획이 필요합니다"),
    ]

    activity = plan_report.collect_turn_activity(messages)

    assert activity.edited == ()
    assert activity.commands == ()
    assert not activity.changed_something


def test_long_command_output_keeps_its_start_and_its_final_summary() -> None:
    output = "collected 300 items\n" + "." * 5000 + "\n3 failed, 297 passed"
    messages: list[BaseMessage] = [
        HumanMessage("요청"),
        _call("a", "execute", command="pytest"),
        _result("a", output),
    ]

    activity = plan_report.collect_turn_activity(messages)

    excerpt = activity.commands[0][1]
    assert len(excerpt) < 1000
    assert excerpt.startswith("collected 300 items")
    assert excerpt.endswith("3 failed, 297 passed")


# ----- git state ------------------------------------------------------------


def test_dirty_files_are_none_outside_a_git_work_tree(tmp_path: Path) -> None:
    assert plan_report.git_dirty_files(tmp_path) is None


def test_dirty_files_list_modified_untracked_and_renamed(repo: Path) -> None:
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "new file.py").write_text("x\n", encoding="utf-8")
    (repo / "old.txt").write_text("content\n", encoding="utf-8")
    _git(repo, "add", "old.txt")
    _git(repo, "commit", "-q", "-m", "add old")
    _git(repo, "mv", "old.txt", "renamed.txt")

    dirty = plan_report.git_dirty_files(repo)

    assert dirty is not None
    names = {str((repo / name).resolve()) for name in dirty}
    assert names == {
        str((repo / "README.md").resolve()),
        str((repo / "src" / "new file.py").resolve()),
        str((repo / "renamed.txt").resolve()),
    }


# ----- comparison with the plan ---------------------------------------------


def test_changes_since_approval_are_checked_against_targets(repo: Path) -> None:
    path = _plan_path(repo)
    (repo / "notes.txt").write_text("before approval\n", encoding="utf-8")
    plan_report.write_baseline(repo, path)
    (repo / "src").mkdir()
    (repo / "src" / "greeting.py").write_text("x\n", encoding="utf-8")
    (repo / "setup.cfg").write_text("made by a command\n", encoding="utf-8")

    comparison = plan_report.compare_with_plan(
        repo, path, VALID_PLAN, _activity("src/greeting.py")
    )

    assert comparison.used_git
    assert set(comparison.changed) == {"src/greeting.py", "setup.cfg"}
    assert comparison.unplanned == ("setup.cfg",)
    assert comparison.untouched == ("tests/",)


def test_file_changed_again_after_approval_is_reported(repo: Path) -> None:
    path = _plan_path(repo)
    notes = repo / "notes.txt"
    notes.write_text("before approval\n", encoding="utf-8")
    plan_report.write_baseline(repo, path)
    notes.write_text("changed again, with a different size\n", encoding="utf-8")

    comparison = plan_report.compare_with_plan(repo, path, VALID_PLAN, _activity())

    assert comparison.unplanned == ("notes.txt",)


def test_without_git_only_file_tool_calls_are_known(tmp_path: Path) -> None:
    path = _plan_path(tmp_path)
    (tmp_path / "setup.cfg").write_text("made by a command\n", encoding="utf-8")

    comparison = plan_report.compare_with_plan(
        tmp_path, path, VALID_PLAN, _activity("src/greeting.py", "tests/test_g.py")
    )

    assert not comparison.used_git
    assert comparison.changed == ("src/greeting.py", "tests/test_g.py")
    assert comparison.unplanned == ()
    assert comparison.untouched == ()


def test_plan_folder_files_are_not_reported_as_changes(repo: Path) -> None:
    path = _plan_path(repo)
    plan_report.write_baseline(repo, path)
    path.with_name("thread-1.review.md").write_text("record\n", encoding="utf-8")

    comparison = plan_report.compare_with_plan(repo, path, VALID_PLAN, _activity())

    assert comparison.changed == ()


# ----- rendering ------------------------------------------------------------


def test_report_keeps_harness_findings_when_there_is_no_review(
    tmp_path: Path,
) -> None:
    path = _plan_path(tmp_path)
    activity = TurnActivity(
        edited=("docs/extra.md",), commands=(("pytest -q", "1 failed"),), final_reply=""
    )
    comparison = plan_report.compare_with_plan(tmp_path, path, VALID_PLAN, activity)

    text = plan_report.render_report(
        plan_hash="abc",
        activity=activity,
        comparison=comparison,
        review=None,
        review_note="리뷰어가 평가를 내지 못했습니다",
    )

    assert "계획에 없는 변경:\n\n- `docs/extra.md`" in text
    assert "- `src/greeting.py`" in text
    assert "- `pytest -q`" in text
    assert "AI 리뷰 없음: 리뷰어가 평가를 내지 못했습니다" in text


def test_report_shows_the_four_parts_of_a_review(tmp_path: Path) -> None:
    path = _plan_path(tmp_path)
    activity = _activity("src/greeting.py")
    comparison = plan_report.compare_with_plan(tmp_path, path, VALID_PLAN, activity)
    review = WorkReview(
        followed=["`greet` 함수를 추가했다."],
        missing=["테스트를 쓰지 않았다."],
        unplanned=[],
        tests="테스트 실행 기록이 없다.",
    )

    text = plan_report.render_report(
        plan_hash="abc", activity=activity, comparison=comparison, review=review
    )

    for heading in ("계획대로 한 것", "빠진 단계", "계획에 없는 변경", "테스트 결과"):
        assert f"#### {heading}" in text
    assert "- 테스트를 쓰지 않았다." in text
    assert "테스트 실행 기록이 없다." in text


# ----- reviewer -------------------------------------------------------------


class _Agent:
    """Review agent stand-in returning a prepared structured response."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def ainvoke(self, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        self.prompts.append(str(payload["messages"][0].text))
        if isinstance(self.response, Exception):
            raise self.response
        return {"structured_response": self.response}


async def _review(reviewer: WorkReviewer) -> WorkReview | None:
    return await reviewer.review(
        request_text="요청 본문", plan_text="계획 본문", evidence="증거 본문"
    )


async def test_reviewer_passes_request_plan_and_evidence_to_the_agent() -> None:
    agent = _Agent(WorkReview(tests="통과"))

    result = await _review(WorkReviewer(agent))

    assert result is not None
    assert result.tests == "통과"
    for part in ("요청 본문", "계획 본문", "증거 본문"):
        assert part in agent.prompts[0]


@pytest.mark.parametrize(
    "response",
    [None, "not a review", {"followed": "not a list"}, ValueError("model down")],
)
async def test_unusable_reviewer_output_is_no_review(response: object) -> None:
    assert await _review(WorkReviewer(_Agent(response))) is None


async def test_reviewer_agent_is_built_on_first_use() -> None:
    built: list[_Agent] = []

    def factory() -> _Agent:
        built.append(_Agent({"tests": "통과"}))
        return built[-1]

    reviewer = WorkReviewer(factory=factory)
    assert built == []

    first = await _review(reviewer)
    await _review(reviewer)

    assert first is not None
    assert first.tests == "통과"
    assert len(built) == 1


# ----- end of turn, through the gate ----------------------------------------


class _ScriptedWorkReviewer(WorkReviewer):
    """Work reviewer stand-in that records what it was asked to review."""

    def __init__(self, review: WorkReview | None) -> None:
        super().__init__()
        self._review = review
        self.evidence: list[str] = []

    async def review(
        self,
        *,
        request_text: str,
        plan_text: str,
        evidence: str,
        project_root: str = "",
        earlier_requests: Sequence[str] = (),
        context: object | None = None,
    ) -> WorkReview | None:
        del request_text, plan_text, project_root, earlier_requests, context
        self.evidence.append(evidence)
        return self._review


def _report_path(harness: _Harness) -> Path:
    return harness.plan_path.with_name(harness.plan_path.stem + plan.REPORT_SUFFIX)


async def test_turn_with_changes_after_approval_writes_a_report(
    tmp_path: Path,
) -> None:
    reviewer = _ScriptedWorkReviewer(
        WorkReview(followed=["함수를 추가했다."], tests="pytest가 통과했다.")
    )
    harness = _Harness(tmp_path, work_reviewer=reviewer)
    assert not _report_path(harness).exists()

    await _approved(
        harness,
        harness.call("write_file", file_path="src/greeting.py", content="x"),
        harness.call("execute", command="pytest tests/test_greeting.py"),
    )

    report = _report_path(harness).read_text(encoding="utf-8")
    assert report.startswith("# 작업 후 보고서: thread-1")
    assert "- `src/greeting.py`" in report
    assert "- `pytest tests/test_greeting.py`" in report
    assert "변경이 없는 수정 대상:\n\n- `tests/`" in report
    assert "- 함수를 추가했다." in report
    assert "pytest tests/test_greeting.py" in reviewer.evidence[0]


async def test_each_request_with_changes_appends_an_entry(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)

    for name in ("tests/test_a.py", "tests/test_b.py"):
        await _approved(
            harness, harness.call("write_file", file_path=name, content="x")
        )

    report = _report_path(harness).read_text(encoding="utf-8")
    assert report.count("## 작업 후 리뷰") == 2
    assert report.count("# 작업 후 보고서") == 1


async def test_turn_without_changes_writes_no_report(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)

    await _approved(harness, harness.call("read_file", file_path="src/greeting.py"))

    assert not _report_path(harness).exists()


async def test_no_report_before_approval(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    harness.write_plan()

    await harness.run(
        harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    assert not _report_path(harness).exists()


async def test_failed_reviewer_still_leaves_the_harness_findings(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path, work_reviewer=_ScriptedWorkReviewer(None))
    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    report = _report_path(harness).read_text(encoding="utf-8")
    assert "- `src/greeting.py`" in report
    assert "AI 리뷰 없음: 리뷰어가 평가를 내지 못했습니다" in report


async def test_report_failure_does_not_fail_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(tmp_path)

    def broken(*_args: object, **_kwargs: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(plan_report, "append_report", broken)

    outcome = await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    assert outcome["result"]["messages"][-1].content == "done"


# ----- findings from the first runs with a real model -----------------------


def test_subagent_and_interpreter_calls_are_activity() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("요청"),
        _call("a", "task", description="run the\ntests", subagent_type="general"),
        _result("a", "done"),
        _call("b", "js_eval", code="1 + 1"),
        _result("b", "2"),
        _call("c", "task", description="refused"),
        _result("c", "[계획 관문] 거부", status="error"),
        AIMessage(content="끝"),
    ]

    activity = plan_report.collect_turn_activity(messages)

    assert activity.other_calls == ("task: run the tests", "js_eval: 1 + 1")
    assert activity.changed_something


def test_report_and_evidence_list_the_other_calls(tmp_path: Path) -> None:
    path = _plan_path(tmp_path)
    activity = TurnActivity(
        edited=(), commands=(), final_reply="", other_calls=("task: run the tests",)
    )
    comparison = plan_report.compare_with_plan(tmp_path, path, VALID_PLAN, activity)

    report = plan_report.render_report(
        plan_hash="abc", activity=activity, comparison=comparison, review=None
    )

    assert "- task: run the tests" in report
    assert "- task: run the tests" in plan_report.evidence_text(activity, comparison)


async def test_turn_with_only_a_subagent_call_writes_a_report(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)

    await _approved(harness, harness.call("task", description="run the tests"))

    report = _report_path(harness).read_text(encoding="utf-8")
    assert "- task: run the tests" in report


def test_reviewer_prompts_explain_how_host_paths_map_to_the_tools() -> None:
    root = "/home/user/project"

    work = plan_report.review_prompt("요청", "계획", "증거", project_root=root)
    planned = plan_review.review_prompt("요청", "계획", "plan.md", root)

    for prompt in (work, planned):
        assert f"`{root}`" in prompt
        assert "`/src/app.py`" in prompt
    assert root not in plan_review.review_prompt("요청", "계획", "plan.md")


async def test_gate_tells_the_work_reviewer_the_project_root(tmp_path: Path) -> None:
    roots: list[str] = []

    class _Recorder(WorkReviewer):
        async def review(
            self,
            *,
            request_text: str,
            plan_text: str,
            evidence: str,
            project_root: str = "",
            earlier_requests: Sequence[str] = (),
            context: object | None = None,
        ) -> WorkReview | None:
            del request_text, plan_text, evidence, earlier_requests, context
            roots.append(project_root)
            return None

    harness = _Harness(tmp_path, work_reviewer=_Recorder())
    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    assert roots == [str(tmp_path)]


def test_reviewers_are_told_which_files_belong_to_the_harness() -> None:
    work = plan_report.review_prompt("요청", "계획", "증거")
    planned = plan_review.review_prompt("요청", "계획", "plan.md")

    for prompt in (work, planned):
        assert "`.deepagents/`" in prompt
        assert "not changes made by the work" in prompt


@pytest.mark.parametrize(
    "placeholder",
    [
        "없음",
        "없음.",
        "없음: 편집된 파일은 하나뿐이다.",
        "해당 없음",
        "None",
        "N/A",
        "-",
    ],
)
def test_entries_that_only_say_nothing_was_found_are_dropped(placeholder: str) -> None:
    review = WorkReview.model_validate(
        {
            "tests": "통과",
            "missing": [placeholder],
            "unplanned": [placeholder, "계획에 없던 README 수정"],
        }
    )

    assert review.missing == []
    assert review.unplanned == ["계획에 없던 README 수정"]


def test_findings_that_merely_start_like_a_placeholder_are_kept() -> None:
    review = WorkReview(tests="통과", missing=["없음직한 단계 하나가 빠졌다"])

    assert review.missing == ["없음직한 단계 하나가 빠졌다"]


def test_reviewer_prompts_separate_earlier_requests_from_the_current_one() -> None:
    earlier = ("함수에 docstring을 추가해 줘",)

    work = plan_report.review_prompt(
        "NOTES.md를 만들어 줘", "계획", "증거", earlier_requests=earlier
    )
    planned = plan_review.review_prompt(
        "NOTES.md를 만들어 줘", "계획", "plan.md", earlier_requests=earlier
    )

    for prompt in (work, planned):
        request = prompt.split("<user_request>")[1].split("</user_request>")[0]
        assert "NOTES.md를 만들어 줘" in request
        assert "docstring" not in request
        assert "<earlier_requests>" in prompt
        assert "context only" in prompt
    assert "<earlier_requests>" not in plan_review.review_prompt("요청", "계획", "p.md")
