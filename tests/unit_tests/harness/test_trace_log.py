"""Tests for harness step records: local file, trace runs, metrics, viewer."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx
import pytest
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from deepagents_code.harness import (
    experience,
    plan_gate,
    plan_report,
    plan_review,
    trace_log,
)
from deepagents_code.harness.experience import ExperienceMemoryMiddleware
from deepagents_code.harness.improvement import Outcome, SelfImprover, Signals
from deepagents_code.harness.plan_report import WorkReview, WorkReviewer
from deepagents_code.harness.plan_review import PlanReview, failed_review
from deepagents_code.harness.trace_log import (
    ModelRetryCounterMiddleware,
    RetryCounter,
    RunCounter,
    RunCounterCallback,
    ToolResultCounterMiddleware,
)
from deepagents_code.model_retry import CodeModelRetryMiddleware
from tests.unit_tests.harness import test_plan_gate
from tests.unit_tests.harness.test_plan import VALID_PLAN
from tests.unit_tests.harness.test_plan_gate import (
    THREAD,
    _approved,
    _Harness,
    _ScriptedReviewer,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path
    from uuid import UUID

    from langchain.agents.middleware.types import ModelRequest, ModelResponse

SECRET = "sk-proj-" + "a1B2" * 12


def _kinds(root: Path, thread_id: str = THREAD) -> list[str]:
    return [step["kind"] for step in trace_log.read(root, thread_id)]


def _steps(root: Path, kind: str, thread_id: str = THREAD) -> list[dict[str, Any]]:
    return [s for s in trace_log.read(root, thread_id) if s["kind"] == kind]


# ----- the local record -----------------------------------------------------


def test_step_is_appended_to_the_threads_file(tmp_path: Path) -> None:
    trace_log.record(tmp_path, "t-1", "plan.review", {"round": 1, "verdict": "revise"})
    trace_log.record(tmp_path, "t-1", "gate.refused", {"tool": "execute"}, failed=True)

    path = tmp_path / ".deepagents" / "traces" / "t-1.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first, second = (json.loads(line) for line in lines)
    assert first["kind"] == "plan.review"
    assert first["data"] == {"round": 1, "verdict": "revise"}
    assert first["failed"] is False
    assert first["thread_id"] == "t-1"
    assert second["failed"] is True
    assert trace_log.read(tmp_path, "t-1") == [first, second]


def test_thread_id_cannot_escape_the_trace_folder(tmp_path: Path) -> None:
    trace_log.record(tmp_path, "../../etc/passwd", "x")

    assert trace_log.trace_file(tmp_path, "../../etc/passwd").parent == (
        tmp_path / ".deepagents" / "traces"
    )
    assert trace_log.threads(tmp_path) == ["______etc_passwd"]


def test_secrets_are_masked_before_they_are_stored(tmp_path: Path) -> None:
    data = {"message": f"key {SECRET} leaked", "nested": [{"token": SECRET}]}

    trace_log.record(tmp_path, "t", "gate.refused", data)

    text = trace_log.trace_file(tmp_path, "t").read_text(encoding="utf-8")
    assert SECRET not in text
    assert "[SECRET_DETECTED]" in text


def test_long_values_are_cut_and_odd_values_become_text(tmp_path: Path) -> None:
    trace_log.record(tmp_path, "t", "x", {"long": "가" * 5000, "path": tmp_path})

    (step,) = trace_log.read(tmp_path, "t")
    assert len(step["data"]["long"]) == 2000
    assert step["data"]["path"] == str(tmp_path)


def test_recording_never_raises(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("x", encoding="utf-8")

    trace_log.record(blocked, "t", "x", {"a": 1})

    assert trace_log.read(blocked, "t") == []


def test_unreadable_lines_are_skipped(tmp_path: Path) -> None:
    trace_log.record(tmp_path, "t", "x")
    with trace_log.trace_file(tmp_path, "t").open("a", encoding="utf-8") as handle:
        handle.write("not json\n[1, 2]\n")

    assert [step["kind"] for step in trace_log.read(tmp_path, "t")] == ["x"]


# ----- the active trace -----------------------------------------------------


class _RunNames(BaseCallbackHandler):
    """Collects the name and parent of every chain run."""

    def __init__(self) -> None:
        self.runs: dict[str, tuple[UUID, UUID | None]] = {}
        self.inputs: dict[str, object] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: object,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized
        name = kwargs.get("name") or ""
        self.runs[name] = (run_id, parent_run_id)
        self.inputs[name] = inputs


def test_step_appears_in_the_trace_under_the_run_that_made_it(
    tmp_path: Path,
) -> None:
    def review(value: dict[str, Any]) -> dict[str, Any]:
        trace_log.record(tmp_path, "t", "plan.review", {"round": 2}, failed=True)
        return value

    handler = _RunNames()

    RunnableLambda(review).invoke({}, {"callbacks": [handler]})

    run_id, parent = handler.runs["harness.plan.review"]
    assert parent == handler.runs["review"][0]
    assert run_id != parent
    assert handler.inputs["harness.plan.review"] == {
        "round": 2,
        "failed": True,
        "level": trace_log.LEVEL_WARNING,
    }


def test_step_outside_a_run_is_still_recorded_locally(tmp_path: Path) -> None:
    trace_log.record(tmp_path, "t", "plan.review", {"round": 1})

    assert _kinds(tmp_path, "t") == ["plan.review"]


def test_scores_are_not_sent_when_langfuse_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with patch("langfuse.get_client") as get_client:
        trace_log.record_scores({"model_calls": 3})

    get_client.assert_not_called()


def test_each_metric_becomes_a_numeric_langfuse_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    client = Mock()

    with patch("langfuse.get_client", return_value=client):
        trace_log.record_scores({"model_calls": 3, "duration_seconds": 1.5})

    calls = {c.kwargs["name"]: c.kwargs for c in client.score_current_trace.mock_calls}
    assert calls["harness.model_calls"]["value"] == pytest.approx(3.0)
    assert calls["harness.duration_seconds"]["value"] == pytest.approx(1.5)
    assert calls["harness.model_calls"]["data_type"] == "NUMERIC"


def test_a_langfuse_error_does_not_reach_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    with patch("langfuse.get_client", side_effect=RuntimeError("down")):
        trace_log.record_scores({"model_calls": 3})


# ----- turn metrics ---------------------------------------------------------


def test_metrics_count_the_current_turn() -> None:
    def call(call_id: str, name: str, tokens: tuple[int, int]) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": {}, "id": call_id, "type": "tool_call"}],
            usage_metadata={
                "input_tokens": tokens[0],
                "output_tokens": tokens[1],
                "total_tokens": sum(tokens),
            },
        )

    def result(call_id: str, text: str = "ok", status: str = "success") -> ToolMessage:
        return ToolMessage(content=text, tool_call_id=call_id, status=status)

    messages: list[BaseMessage] = [
        HumanMessage("이전 요청"),
        call("old", "execute", (999, 999)),
        result("old", "[계획 관문] 거부", "error"),
        HumanMessage("이번 요청"),
        call("a", "write_file", (100, 10)),
        result("a", "[계획 관문] 거부", "error"),
        call("b", "request_plan_review", (110, 20)),
        result("b"),
        call("c", "request_plan_review", (120, 30)),
        result("c"),
        call("d", "read_file", (130, 40)),
        result("d", "File not found", "error"),
        AIMessage(content="끝"),
    ]

    metrics = trace_log.turn_metrics(messages, time.time() - 2)

    assert metrics["model_calls"] == 5
    assert metrics["tool_calls"] == 4
    assert metrics["gate_refusals"] == 1
    assert metrics["tool_errors"] == 1
    assert metrics["review_rounds"] == 2
    assert metrics["input_tokens"] == 460
    assert metrics["output_tokens"] == 100
    assert 2 <= metrics["duration_seconds"] < 10


def _reply(input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        content="답",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def test_unreported_input_tokens_are_left_out_rather_than_zero() -> None:
    """Some gateways report output tokens only; zero input would be false."""
    messages: list[BaseMessage] = [HumanMessage("요청"), _reply(0, 50), _reply(0, 70)]

    metrics = trace_log.turn_metrics(messages, None)
    step = trace_log.metrics_step(metrics)

    assert "input_tokens" not in metrics
    assert metrics["output_tokens"] == 120
    assert step["input_tokens"] is None
    assert step["unreported"] == ["input_tokens"]


def test_reported_and_absent_token_counts_are_kept_as_numbers() -> None:
    reported = trace_log.turn_metrics([HumanMessage("요청"), _reply(30, 5)], None)
    silent = trace_log.turn_metrics([HumanMessage("요청")], None)

    assert reported["input_tokens"] == 30
    assert silent["input_tokens"] == 0
    assert "unreported" not in trace_log.metrics_step(reported)
    assert "unreported" not in trace_log.metrics_step(silent)


def test_metrics_without_a_start_time_have_no_duration() -> None:
    metrics = trace_log.turn_metrics([HumanMessage("요청")], None)

    assert "duration_seconds" not in metrics
    assert metrics["model_calls"] == 0


# ----- reading and the viewer -----------------------------------------------


def test_first_failure_is_the_earliest_error_not_a_refusal(tmp_path: Path) -> None:
    trace_log.record(tmp_path, "t", "memory.search", {"hits": []})
    trace_log.record(tmp_path, "t", "gate.refused", {"tool": "execute"}, failed=True)
    trace_log.record(tmp_path, "t", "plan.review", {"round": 1}, failed=True)
    trace_log.record(tmp_path, "t", "work.review", {"missing": ["x"]}, failed=True)
    trace_log.record(tmp_path, "t", "work.review", {"missing": ["y"]}, failed=True)
    steps = trace_log.read(tmp_path, "t")

    failure = trace_log.first_failure(steps)

    assert failure is not None
    assert failure["kind"] == "work.review"
    assert failure["data"] == {"missing": ["x"]}
    assert trace_log.first_failure(steps[:3]) is None
    assert [s["kind"] for s in trace_log.procedural_refusals(steps)] == [
        "gate.refused",
        "plan.review",
    ]


def test_viewer_lists_steps_and_shows_the_first_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_log.record(tmp_path, "t", "memory.search", {"hits": []})
    trace_log.record(tmp_path, "t", "gate.refused", {"tool": "execute"}, failed=True)
    trace_log.record(tmp_path, "t", "work.review", {"outcome": "partial"}, failed=True)

    code = trace_log.main(["t", "--root", str(tmp_path)])

    out = capsys.readouterr().out
    assert code == 0
    assert "  1" in out
    assert "memory.search" in out
    assert "  2 ! " in out
    assert "  3 ✗ " in out
    assert "절차상 거부(!): 1건 (gate.refused)" in out
    assert "첫 실패 지점: 3번 work.review" in out
    assert '"outcome": "partial"' in out


def test_viewer_shows_no_failure_when_only_the_procedure_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_log.record(tmp_path, "t", "gate.refused", {"tool": "execute"}, failed=True)

    trace_log.main(["t", "--root", str(tmp_path)])

    out = capsys.readouterr().out
    assert "절차상 거부(!): 1건 (gate.refused)" in out
    assert "첫 실패 지점: 없음" in out
    assert "✗" not in out


_REVIEW_LIMIT = {"verdict": trace_log.REVIEW_LIMIT_VERDICT, "findings": ["x"]}
_NO_VERDICT = {
    "verdict": "revise",
    "findings": [failed_review("시간 초과").findings[0]],
}
_UNORDERED = {"decision": "불가", "reason": "리뷰를 통과한 계획서가 없음"}
_NO_OPTION = {"decision": "불가", "reason": trace_log.NO_APPROVAL_OPTION_REASON}


@pytest.mark.parametrize(
    ("kind", "data", "level"),
    [
        ("gate.refused", {"tool": "execute"}, trace_log.LEVEL_WARNING),
        ("plan.invalid", {"problems": ["x"]}, trace_log.LEVEL_WARNING),
        (
            "plan.review",
            {"verdict": "revise", "findings": ["x"]},
            trace_log.LEVEL_WARNING,
        ),
        ("plan.review", _NO_VERDICT, trace_log.LEVEL_ERROR),
        ("plan.review", _REVIEW_LIMIT, trace_log.LEVEL_ERROR),
        ("plan.approval", _UNORDERED, trace_log.LEVEL_WARNING),
        ("plan.approval", _NO_OPTION, trace_log.LEVEL_ERROR),
        ("plan.approval", {"decision": "수정 요청"}, trace_log.LEVEL_WARNING),
        ("work.review", {"outcome": "partial"}, trace_log.LEVEL_ERROR),
        ("improvement.decision", {"decision": "보류"}, trace_log.LEVEL_WARNING),
        ("something.new", {}, trace_log.LEVEL_ERROR),
    ],
)
def test_failure_level_follows_the_kind_and_its_details(
    kind: str, data: dict[str, Any], level: str
) -> None:
    step = {"kind": kind, "failed": True, "data": data}

    assert trace_log.failure_level(step) == level


def test_steps_that_did_not_fail_have_no_level(tmp_path: Path) -> None:
    trace_log.record(tmp_path, "t", "plan.review", {"verdict": "approve"})

    (step,) = trace_log.read(tmp_path, "t")

    assert step["level"] is None
    assert trace_log.failure_level(step) is None


def test_a_failed_step_is_stored_with_its_level(tmp_path: Path) -> None:
    trace_log.record(tmp_path, "t", "gate.refused", {"tool": "x"}, failed=True)
    trace_log.record(tmp_path, "t", "work.review", {"outcome": "x"}, failed=True)

    levels = [step["level"] for step in trace_log.read(tmp_path, "t")]

    assert levels == [trace_log.LEVEL_WARNING, trace_log.LEVEL_ERROR]


def test_records_without_a_level_are_classified_from_their_kind(
    tmp_path: Path,
) -> None:
    path = trace_log.trace_file(tmp_path, "t")
    path.parent.mkdir(parents=True)
    old = [
        {"kind": "gate.refused", "failed": True, "data": {"tool": "execute"}},
        {"kind": "work.review", "failed": True, "data": {"outcome": "partial"}},
    ]
    path.write_text("".join(json.dumps(s) + "\n" for s in old), encoding="utf-8")

    steps = trace_log.read(tmp_path, "t")

    failure = trace_log.first_failure(steps)
    assert failure is not None
    assert failure["kind"] == "work.review"
    assert len(trace_log.procedural_refusals(steps)) == 1


def test_a_stored_level_is_trusted() -> None:
    step = {"kind": "work.review", "failed": True, "level": "warning", "data": {}}

    assert trace_log.failure_level(step) == trace_log.LEVEL_WARNING


def test_unreadable_details_count_as_an_error() -> None:
    step = {"kind": "plan.review", "failed": True, "data": {"findings": 3}}

    assert trace_log.failure_level(step) == trace_log.LEVEL_ERROR


def test_viewer_reports_a_thread_without_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_log.record(tmp_path, "t", "turn.metrics", {"model_calls": 1})

    trace_log.main(["t", "--root", str(tmp_path)])

    assert "첫 실패 지점: 없음" in capsys.readouterr().out


def test_viewer_without_a_thread_lists_recorded_threads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert trace_log.main(["--root", str(tmp_path)]) == 0
    assert "기록된 thread가 없습니다." in capsys.readouterr().out
    trace_log.record(tmp_path, "thread-a", "x")
    trace_log.record(tmp_path, "thread-b", "x")

    trace_log.main(["--root", str(tmp_path)])

    assert set(capsys.readouterr().out.split()) == {"thread-a", "thread-b"}


def test_viewer_exits_with_an_error_for_an_unknown_thread(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert trace_log.main(["missing", "--root", str(tmp_path)]) == 1
    assert "기록된 단계가 없습니다." in capsys.readouterr().out


# ----- what the gate records ------------------------------------------------


class _FixedReviewer(WorkReviewer):
    def __init__(self, review: WorkReview | None) -> None:
        super().__init__()
        self._review = review

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
        del request_text, plan_text, evidence, project_root, earlier_requests, context
        return self._review


async def test_a_refused_call_is_recorded_as_a_failure(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)

    await harness.run(harness.call("execute", command="echo x > a.py"))

    (step,) = _steps(tmp_path, "gate.refused")
    assert step["failed"] is True
    assert step["level"] == trace_log.LEVEL_WARNING
    assert step["data"]["tool"] == "execute"
    assert step["data"]["phase"] == plan_gate.PHASE_PLANNING
    assert step["data"]["message"].startswith("[계획 관문]")


async def test_plan_review_and_approval_are_recorded_in_order(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    await harness.run(
        harness.call(
            "write_file", file_path=str(harness.plan_path), content=VALID_PLAN
        ),
        harness.call(plan_gate.REVIEW_TOOL),
        harness.call(plan_gate.APPROVAL_TOOL),
    )
    await harness.run(
        resume={"status": "answered", "answers": [plan_gate.APPROVE_CHOICE, ""]}
    )

    kinds = [k for k in _kinds(tmp_path) if k != "turn.metrics"]
    assert kinds == ["plan.written", "plan.review", "plan.approval"]
    (review,) = _steps(tmp_path, "plan.review")
    assert review["data"]["round"] == 1
    assert review["data"]["verdict"] == "approve"
    (approval,) = _steps(tmp_path, "plan.approval")
    assert approval["data"]["decision"] == "승인"
    assert approval["data"]["approver"] == "사용자"
    assert approval["failed"] is False


async def test_a_rejected_plan_and_an_invalid_plan_are_failures(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    harness.write_plan("# 계획서\n\n## 요구사항\n내용\n")
    await harness.run(harness.call(plan_gate.REVIEW_TOOL))
    harness.write_plan()
    await harness.run(
        harness.call(plan_gate.REVIEW_TOOL), harness.call(plan_gate.APPROVAL_TOOL)
    )
    await harness.run(
        resume={"status": "answered", "answers": [plan_gate.REVISE_CHOICE, "더 좁게"]}
    )

    (invalid,) = _steps(tmp_path, "plan.invalid")
    assert invalid["failed"] is True
    assert invalid["level"] == trace_log.LEVEL_WARNING
    assert invalid["data"]["problems"]
    (approval,) = _steps(tmp_path, "plan.approval")
    assert approval["failed"] is True
    assert approval["level"] == trace_log.LEVEL_WARNING
    assert approval["data"]["decision"] == "수정 요청"
    assert approval["data"]["note"] == "더 좁게"


async def test_approval_without_a_reviewed_plan_is_a_refusal(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)

    await harness.run(harness.call(plan_gate.APPROVAL_TOOL))

    (approval,) = _steps(tmp_path, "plan.approval")
    assert approval["data"]["decision"] == trace_log.APPROVAL_UNAVAILABLE
    assert approval["level"] == trace_log.LEVEL_WARNING


async def test_a_run_that_cannot_be_approved_is_an_error(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, interactive=False)
    harness.write_plan()

    await harness.run(
        harness.call(plan_gate.REVIEW_TOOL), harness.call(plan_gate.APPROVAL_TOOL)
    )

    (approval,) = _steps(tmp_path, "plan.approval")
    assert approval["data"]["decision"] == trace_log.APPROVAL_UNAVAILABLE
    assert approval["data"]["reason"] == trace_log.NO_APPROVAL_OPTION_REASON
    assert approval["level"] == trace_log.LEVEL_ERROR


async def test_a_review_without_a_verdict_is_an_error(tmp_path: Path) -> None:
    harness = _Harness(
        tmp_path, reviewer=_ScriptedReviewer(failed_review("응답이 잘림"))
    )
    harness.write_plan()

    await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    (review,) = _steps(tmp_path, "plan.review")
    assert review["level"] == trace_log.LEVEL_ERROR


async def test_a_review_that_does_not_converge_is_an_error(tmp_path: Path) -> None:
    rounds = plan_gate.MAX_REVIEW_ROUNDS
    harness = _Harness(
        tmp_path,
        reviewer=_ScriptedReviewer(
            *[PlanReview(verdict="revise", findings=["부족합니다."])] * rounds
        ),
    )
    harness.write_plan()

    await harness.run(*[harness.call(plan_gate.REVIEW_TOOL) for _ in range(rounds + 1)])

    levels = [step["level"] for step in _steps(tmp_path, "plan.review")]
    assert levels == [trace_log.LEVEL_WARNING] * rounds + [trace_log.LEVEL_ERROR]
    final = _steps(tmp_path, "plan.review")[-1]
    assert final["data"]["verdict"] == trace_log.REVIEW_LIMIT_VERDICT


async def test_work_review_memory_and_metrics_are_recorded(tmp_path: Path) -> None:
    review = WorkReview(
        tests="실행 기록 없음", missing=["테스트를 실행하지 않았다."], outcome="partial"
    )
    harness = _Harness(
        tmp_path, work_reviewer=_FixedReviewer(review), record_experience=True
    )
    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    (work,) = _steps(tmp_path, "work.review")
    assert work["failed"] is True
    assert work["level"] == trace_log.LEVEL_ERROR
    assert work["data"]["changed"] == ["src/greeting.py"]
    assert work["data"]["missing"] == ["테스트를 실행하지 않았다."]
    assert work["data"]["outcome"] == "partial"
    (saved,) = _steps(tmp_path, "memory.saved")
    assert saved["data"]["id"] == experience.load_all(tmp_path)[0].id
    metrics = _steps(tmp_path, "turn.metrics")[-1]["data"]
    # Review, approval, the write and the closing reply happen in one turn.
    assert metrics["model_calls"] == 4
    assert metrics["tool_calls"] == 3
    assert metrics["duration_seconds"] >= 0


async def test_every_turn_records_metrics_even_without_changes(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)

    await harness.run(harness.call("read_file", file_path="src/app.py"))

    assert _kinds(tmp_path) == ["turn.metrics"]


async def test_memory_search_is_recorded_with_its_hits(tmp_path: Path) -> None:
    record = experience.Experience(
        id="20261110T000000-other",
        request="calc.py에 multiply 함수를 추가하고 테스트를 추가해 줘",
        summary="calc.py에 multiply 함수를 추가했다.",
        outcome="success",
        files=("calc.py", "tests/test_calc.py"),
        lessons=(),
        tags=("calc",),
        date="2026-11-10T00:00:00+00:00",
        thread_id="other",
    )
    experience.save(tmp_path, record)
    harness = _Harness(
        tmp_path, extra_middleware=[ExperienceMemoryMiddleware(tmp_path)]
    )
    harness.request = "calc 모듈의 테스트를 보강해 줘"

    await harness.run()

    (search,) = _steps(tmp_path, "memory.search")
    assert search["data"]["query"] == "calc 모듈의 테스트를 보강해 줘"
    assert [hit["id"] for hit in search["data"]["hits"]] == [record.id]
    assert search["data"]["hits"][0]["score"] >= experience.MIN_SCORE


class _FixedImprover(SelfImprover):
    def __init__(self, root: Path, outcome: Outcome | None) -> None:
        super().__init__(root, Mock())
        self._outcome = outcome

    async def run(
        self,
        *,
        signals: Signals,
        request_text: str,
        thread_id: str,
        context: object | None = None,
    ) -> Outcome | None:
        del signals, request_text, thread_id, context
        return self._outcome


@pytest.mark.parametrize(
    ("outcome", "decision", "failed", "level"),
    [
        (Outcome("id-1", applied=True, reason="올랐다"), "반영", False, None),
        (
            Outcome("id-1", applied=False, reason="내용이 비어 있습니다."),
            "보류",
            True,
            trace_log.LEVEL_WARNING,
        ),
        (None, "개선안 없음", False, None),
    ],
)
async def test_improvement_decision_is_recorded(
    tmp_path: Path,
    outcome: Outcome | None,
    decision: str,
    failed: bool,
    level: str | None,
) -> None:
    review = WorkReview(tests="없음", missing=["테스트를 실행하지 않았다."])
    harness = _Harness(tmp_path, work_reviewer=_FixedReviewer(review))
    harness.gate._improver = _FixedImprover(tmp_path, outcome)
    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    (step,) = _steps(tmp_path, "improvement.decision")
    assert step["data"]["decision"] == decision
    assert step["failed"] is failed
    assert step["level"] == level
    assert "테스트를 실행하지 않았다." in step["data"]["signals"]


async def test_no_improvement_step_for_a_clean_turn(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, work_reviewer=_FixedReviewer(None))
    harness.gate._improver = _FixedImprover(tmp_path, None)
    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    assert _steps(tmp_path, "improvement.decision") == []


# ----- the harness's own files are not the agent's changes ------------------


def test_harness_records_are_not_reported_as_changes(tmp_path: Path) -> None:
    path = tmp_path / ".deepagents" / "plans" / "thread-1.md"
    path.parent.mkdir(parents=True)
    path.write_text(VALID_PLAN, encoding="utf-8")
    activity = plan_report.TurnActivity(
        edited=(
            ".deepagents/traces/thread-1.jsonl",
            ".deepagents/memory/x.md",
            "src/greeting.py",
        ),
        commands=(),
        final_reply="",
    )

    comparison = plan_report.compare_with_plan(tmp_path, path, VALID_PLAN, activity)

    assert comparison.changed == ("src/greeting.py",)
    assert comparison.unplanned == ()


async def test_a_turn_that_only_wrote_the_plan_gets_no_work_report(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path, record_experience=True)
    await harness.run(
        harness.call(
            "write_file", file_path=str(harness.plan_path), content=VALID_PLAN
        ),
        harness.call(plan_gate.REVIEW_TOOL),
        harness.call(plan_gate.APPROVAL_TOOL),
    )

    await harness.run(
        resume={"status": "answered", "answers": [plan_gate.APPROVE_CHOICE, ""]}
    )

    assert _steps(tmp_path, "work.review") == []
    assert experience.load_all(tmp_path) == []
    assert not harness.plan_path.with_name("thread-1.report.md").exists()


def test_local_record_can_be_turned_off_while_the_trace_run_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPAGENTS_CODE_PLAN_STEP_LOG", "0")

    def review(value: dict[str, Any]) -> dict[str, Any]:
        trace_log.record(tmp_path, "t", "plan.review", {"round": 1})
        return value

    handler = _RunNames()
    RunnableLambda(review).invoke({}, {"callbacks": [handler]})

    assert not (tmp_path / ".deepagents").exists()
    assert "harness.plan.review" in handler.runs


# ----- model retries ---------------------------------------------------------


class _FailFirstAttempts(AgentMiddleware[Any, Any]):
    """Innermost stand-in for a provider that drops the first attempts."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.remaining = failures

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if self.remaining > 0:
            self.remaining -= 1
            msg = "connection refused"
            raise httpx.ConnectError(msg)
        return await handler(request)


async def _no_sleep(_seconds: float) -> None:  # noqa: RUF029  # replaces asyncio.sleep
    return None


def test_retry_counter_counts_per_thread_until_it_is_read() -> None:
    counter = RetryCounter()
    counter.add("a")
    counter.add("a")
    counter.add("b")

    assert counter.take("a") == 2
    assert counter.take("a") == 0
    counter.reset("b")
    assert counter.take("b") == 0


async def test_retried_model_attempts_are_counted_in_the_turn_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    counter = RetryCounter()
    harness = _Harness(tmp_path)
    harness.gate._retry_counter = counter
    flaky = _FailFirstAttempts(2)
    harness.inner_middleware = [
        CodeModelRetryMiddleware(max_retries=3),
        ModelRetryCounterMiddleware(counter),
        flaky,
    ]

    outcome = await harness.run(harness.call("read_file", file_path="src/app.py"))

    assert outcome["result"]["messages"][-1].content == "done"
    (metrics,) = _steps(tmp_path, "turn.metrics")
    assert metrics["data"]["model_retries"] == 2
    assert metrics["data"]["model_calls"] == 2

    await harness.run(harness.call("read_file", file_path="src/app.py"))

    assert _steps(tmp_path, "turn.metrics")[-1]["data"]["model_retries"] == 0


async def test_turn_metrics_have_no_retry_count_without_a_counter(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)

    await harness.run(harness.call("read_file", file_path="src/app.py"))

    (metrics,) = _steps(tmp_path, "turn.metrics")
    assert "model_retries" not in metrics["data"]


async def test_the_reset_for_a_new_request_is_recorded(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    await _approved(harness)
    assert _steps(tmp_path, "plan.reset") == []

    await harness.run(harness.call("read_file", file_path="src/app.py"))

    (reset,) = _steps(tmp_path, "plan.reset")
    assert reset["data"]["previous_phase"] == plan_gate.PHASE_APPROVED
    assert reset["data"]["archived"].endswith(".approved.md")
    assert reset["failed"] is False


async def test_time_spent_after_the_work_is_recorded_separately(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path, work_reviewer=_FixedReviewer(None))

    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    (post,) = _steps(tmp_path, "turn.post_work")
    assert post["data"]["post_work_seconds"] >= 0
    assert post["data"]["total_seconds"] >= post["data"]["post_work_seconds"]
    kinds = _kinds(tmp_path)
    assert kinds.index("turn.post_work") > kinds.index("work.review")


async def test_no_post_work_step_without_a_work_report(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)

    await harness.run(harness.call("read_file", file_path="src/app.py"))

    assert _steps(tmp_path, "turn.post_work") == []


# ----- counting every role's calls ------------------------------------------


def test_run_counter_takes_the_roles_asked_for_and_keeps_the_rest() -> None:
    counter = RunCounter()
    counter.add("t", "main", trace_log.MODEL_CALLS)
    counter.add("t", "main", trace_log.MODEL_CALLS)
    counter.add("t", "work_review", trace_log.MODEL_CALLS)
    counter.add("other", "main", trace_log.TOOL_CALLS)

    working = counter.take("t", trace_log.AGENT_WORK_ROLES)
    rest = counter.take("t")

    assert working["main"][trace_log.MODEL_CALLS] == 2
    assert set(working) == {"main"}
    assert rest["work_review"][trace_log.MODEL_CALLS] == 1
    assert counter.take("t") == {}
    counter.reset("other")
    assert counter.take("other") == {}


@pytest.mark.parametrize(
    ("tags", "metadata", "role"),
    [
        (["seq:step:1", "harness:plan_review"], {}, "plan_review"),
        (None, {trace_log.SUBAGENT_METADATA_KEY: "call-1"}, "subagent"),
        (
            ["harness:improvement"],
            {trace_log.SUBAGENT_METADATA_KEY: "x"},
            "improvement",
        ),
        (["seq:step:1"], {"thread_id": "t"}, "main"),
        (None, None, "main"),
    ],
)
def test_run_role_reads_the_helper_tag_then_the_subagent_marker(
    tags: list[str] | None, metadata: dict[str, Any] | None, role: str
) -> None:
    assert trace_log.run_role(tags, metadata) == role


def test_helper_names_map_to_their_roles() -> None:
    assert plan_review.helper_role("plan_reviewer") == plan_review.ROLE_PLAN_REVIEW
    assert plan_review.helper_role("plan_work_reviewer") == "work_review"
    assert plan_review.helper_role("improvement_judge") == "improvement"
    assert plan_review.helper_role("custom") == "custom"


def _callback_events(callback: RunCounterCallback, metadata: dict[str, Any]) -> None:
    """Feed one failed and one good model call, and three tool results."""
    failed, good = uuid4(), uuid4()
    callback.on_chat_model_start({}, [[]], run_id=failed, metadata=metadata)
    callback.on_llm_error(RuntimeError("503"), run_id=failed)
    callback.on_chat_model_start({}, [[]], run_id=good, metadata=metadata)
    callback.on_llm_end(Mock(), run_id=good)
    error = ToolMessage(content="no such file", tool_call_id="c1", status="error")
    callback.count_tool_result(error, None, metadata)
    callback.count_tool_result(
        ToolMessage(content="ok", tool_call_id="c2"), None, metadata
    )
    refusal = ToolMessage(
        content=f"{trace_log.GATE_REFUSAL_PREFIX} 아직 승인된 계획이 없어",
        tool_call_id="c3",
        status="error",
    )
    callback.count_tool_result(refusal, None, metadata)


def test_callback_counts_calls_failed_attempts_and_tool_errors() -> None:
    counter = RunCounter()

    _callback_events(RunCounterCallback(counter), {"thread_id": "t"})

    # The gate's refusal is a call, not an error: refusals are counted apart.
    assert counter.take("t") == {
        "main": {
            trace_log.MODEL_CALLS: 2,
            trace_log.MODEL_ERRORS: 1,
            trace_log.TOOL_CALLS: 3,
            trace_log.TOOL_ERRORS: 1,
        }
    }


def test_an_error_reply_inside_a_command_is_a_tool_error() -> None:
    counter = RunCounter()
    callback = RunCounterCallback(counter)
    metadata = {"thread_id": "t"}
    error = ToolMessage(content="상한", tool_call_id="c1", status="error")
    success = ToolMessage(content="ok", tool_call_id="c2")

    callback.count_tool_result(Command(update={"messages": [error]}), None, metadata)
    callback.count_tool_result(Command(update={"messages": [success]}), None, metadata)

    counts = counter.take("t")["main"]
    assert counts[trace_log.TOOL_CALLS] == 2
    assert counts[trace_log.TOOL_ERRORS] == 1


async def test_a_gate_tool_error_reply_is_counted_like_the_message_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(tmp_path)
    _counted(monkeypatch, harness)

    # No plan file: the review tool replies with an error.
    await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    metrics = _steps(tmp_path, "turn.metrics")[-1]["data"]
    assert metrics["tool_errors"] == metrics["all_tool_errors"] == 1


def test_callback_skips_events_without_a_thread_and_control_flow() -> None:
    counter = RunCounter()
    callback = RunCounterCallback(counter)
    _callback_events(callback, {})
    run = uuid4()
    callback.on_chat_model_start({}, [[]], run_id=run, metadata={"thread_id": "t"})

    callback.on_llm_error(GraphBubbleUp(), run_id=run)

    assert counter.take("t")["main"][trace_log.MODEL_ERRORS] == 0


def test_callback_never_raises_when_counting_fails() -> None:
    counter = Mock(spec=RunCounter)
    counter.add.side_effect = RuntimeError("broken")
    callback = RunCounterCallback(counter)

    _callback_events(callback, {"thread_id": "t"})

    # Two model calls, one failed attempt, three tool calls, one tool error.
    assert counter.add.call_count == 7


async def test_middleware_counts_a_result_no_tool_produced() -> None:
    counter = RunCounter()
    middleware = ToolResultCounterMiddleware()
    rejected = ToolMessage(content="rejected", tool_call_id="c", status="error")

    async def short_circuit(_: object) -> ToolMessage:
        await asyncio.sleep(0)
        return rejected

    async def call(_: object) -> ToolMessage:
        return await middleware.awrap_tool_call(Mock(), short_circuit)

    result = await RunnableLambda(call).ainvoke(
        None,
        {
            "metadata": {"thread_id": "t"},
            "callbacks": [RunCounterCallback(counter)],
        },
    )

    assert result is rejected
    assert counter.take("t")["main"] == {
        trace_log.MODEL_CALLS: 0,
        trace_log.MODEL_ERRORS: 0,
        trace_log.TOOL_CALLS: 1,
        trace_log.TOOL_ERRORS: 1,
    }


def test_middleware_does_nothing_without_a_counting_callback() -> None:
    middleware = ToolResultCounterMiddleware()
    reply = ToolMessage(content="ok", tool_call_id="c")

    assert middleware.wrap_tool_call(Mock(), lambda _: reply) is reply


def test_middleware_does_not_count_an_interrupt() -> None:
    counter = RunCounter()
    middleware = ToolResultCounterMiddleware()

    def interrupted(_: object) -> ToolMessage:
        raise GraphBubbleUp

    def call(_: object) -> ToolMessage:
        return middleware.wrap_tool_call(Mock(), interrupted)

    with pytest.raises(GraphBubbleUp):
        RunnableLambda(call).invoke(
            None,
            {
                "metadata": {"thread_id": "t"},
                "callbacks": [RunCounterCallback(counter)],
            },
        )

    assert counter.take("t") == {}


def test_metrics_are_split_into_the_work_and_what_follows_it() -> None:
    working = {
        "main": {
            "model_calls": 3,
            "model_errors": 1,
            "tool_calls": 4,
            "tool_errors": 0,
        },
        "plan_review": {
            "model_calls": 2,
            "model_errors": 0,
            "tool_calls": 1,
            "tool_errors": 1,
        },
    }
    after = {"work_review": {"model_calls": 5, "tool_errors": 2}}

    work = trace_log.agent_work_metrics(working)
    post = trace_log.post_work_metrics(after)

    assert work["all_model_calls"] == 5
    assert work["all_model_errors"] == 1
    assert work["all_tool_errors"] == 1
    assert work["model_calls.main"] == 3
    assert work["model_calls.plan_review"] == 2
    assert work["model_calls.subagent"] == 0
    assert post["post_work_model_calls"] == 5
    assert post["post_work_tool_errors"] == 2
    assert post["model_calls.work_review"] == 5
    assert post["model_calls.improvement"] == 0


def _counted(monkeypatch: pytest.MonkeyPatch, harness: _Harness) -> RunCounter:
    """Give the harness's gate a run counter and bind its callback to runs."""
    counter = RunCounter()
    harness.gate._run_counter = counter
    # Outermost, as in the assembled agent, so it sees the gate's replies too.
    harness.extra_middleware.insert(0, ToolResultCounterMiddleware())
    config = {
        **test_plan_gate.CONFIG,
        "callbacks": [RunCounterCallback(counter)],
    }
    monkeypatch.setattr(test_plan_gate, "CONFIG", config)
    return counter


async def test_turn_metrics_count_the_calls_of_every_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = WorkReview(tests="없음", missing=["테스트를 실행하지 않았다."])
    harness = _Harness(tmp_path, work_reviewer=_FixedReviewer(review))
    _counted(monkeypatch, harness)

    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    metrics = _steps(tmp_path, "turn.metrics")[-1]["data"]
    # The scripted main model is called for the review, the approval, the
    # write and the closing reply; the stand-in reviewers call no model.
    assert metrics["all_model_calls"] == metrics["model_calls"] == 4
    assert metrics["model_calls.main"] == 4
    assert metrics["all_tool_calls"] == 3
    post = _steps(tmp_path, "turn.post_work")[-1]["data"]
    assert post["post_work_model_calls"] == 0


async def test_a_refused_call_is_not_counted_as_a_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(tmp_path)
    _counted(monkeypatch, harness)

    await harness.run(harness.call("execute", command="echo x > a.py"))

    metrics = _steps(tmp_path, "turn.metrics")[-1]["data"]
    assert metrics["gate_refusals"] == 1
    assert metrics["all_tool_errors"] == 0


async def test_a_new_turn_starts_its_counts_from_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(tmp_path)
    counter = _counted(monkeypatch, harness)
    counter.add(THREAD, "main", trace_log.MODEL_CALLS)

    await harness.run()

    metrics = _steps(tmp_path, "turn.metrics")[-1]["data"]
    assert metrics["all_model_calls"] == 1
