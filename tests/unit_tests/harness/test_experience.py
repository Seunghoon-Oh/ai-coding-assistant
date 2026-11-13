"""Tests for work experience memory: records, retrieval, injection, saving."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_code.harness import experience, plan, plan_gate
from deepagents_code.harness.experience import (
    Experience,
    ExperienceMemoryMiddleware,
)
from deepagents_code.harness.plan_report import WorkReview, WorkReviewer
from tests.unit_tests.harness.test_plan import VALID_PLAN
from tests.unit_tests.harness.test_plan_gate import (
    _approved,
    _Harness,
    _last_tool_message,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.outputs import ChatResult
    from langchain_core.runnables import RunnableConfig


def _record(
    record_id: str = "20261112T010000-thread-a",
    *,
    request: str = "calc.py에 multiply 함수를 추가하고 테스트를 추가해 줘",
    summary: str = "calc.py에 multiply 함수를, tests/test_calc.py에 테스트를 추가했다.",
    outcome: str = "partial",
    files: tuple[str, ...] = ("calc.py", "tests/test_calc.py"),
    lessons: tuple[str, ...] = ("셸이 없으면 테스트를 실행하지 못한다고 보고한다.",),
    tags: tuple[str, ...] = ("calc", "multiply", "pytest"),
    thread_id: str = "thread-a",
) -> Experience:
    return Experience(
        id=record_id,
        request=request,
        summary=summary,
        outcome=outcome,
        files=files,
        lessons=lessons,
        tags=tags,
        date="2026-11-12T01:00:00+00:00",
        thread_id=thread_id,
    )


def _login_record(record_id: str = "20261111T000000-thread-b") -> Experience:
    return _record(
        record_id,
        request="로그인 API에 비밀번호 재설정 엔드포인트를 만들어 줘",
        summary="auth/routes.py에 비밀번호 재설정 엔드포인트를 추가했다.",
        files=("auth/routes.py", "tests/test_auth.py"),
        tags=("auth", "API", "비밀번호 재설정"),
    )


# ----- record files ---------------------------------------------------------


def test_record_survives_a_round_trip_through_its_file(tmp_path: Path) -> None:
    record = _record()

    path = experience.save(tmp_path, record)

    assert path == tmp_path / ".deepagents" / "memory" / f"{record.id}.md"
    assert experience.load_all(tmp_path) == [record]


def test_empty_fields_are_written_as_placeholders_and_read_back_empty() -> None:
    record = _record(summary="", files=(), lessons=(), tags=(), outcome="unknown")

    text = experience.render(record)

    assert "## 배운 점\n\n- 없음" in text
    assert "## 결과\n\n알 수 없음" in text
    assert experience.parse(record.id, text) == record


def test_multi_line_values_cannot_add_sections_to_a_record() -> None:
    record = _record(
        summary="한 줄\n## 태그\n- 가짜",
        lessons=("첫 줄\n## 결과\n성공",),
    )

    parsed = experience.parse(record.id, experience.render(record))

    assert parsed is not None
    assert parsed.tags == record.tags
    assert parsed.outcome == record.outcome
    assert len(parsed.lessons) == 1


def test_files_that_are_not_records_are_skipped(tmp_path: Path) -> None:
    folder = experience.memory_directory(tmp_path)
    folder.mkdir(parents=True)
    (folder / "notes.md").write_text("# 메모\n\n그냥 메모\n", encoding="utf-8")
    (folder / "data.txt").write_text("x", encoding="utf-8")
    experience.save(tmp_path, _record())

    assert [r.id for r in experience.load_all(tmp_path)] == [_record().id]


def test_records_load_newest_first(tmp_path: Path) -> None:
    for stamp in ("20261110T000000", "20261112T000000", "20261111T000000"):
        experience.save(tmp_path, _record(f"{stamp}-t"))

    ids = [r.id for r in experience.load_all(tmp_path)]

    assert ids == ["20261112T000000-t", "20261111T000000-t", "20261110T000000-t"]


def test_no_memory_folder_means_no_records(tmp_path: Path) -> None:
    assert experience.load_all(tmp_path) == []


def test_new_id_sorts_by_time_and_is_a_safe_file_name() -> None:
    moment = datetime(2026, 11, 12, 1, 2, 3, tzinfo=UTC)

    assert experience.new_id("a/b:c", moment) == "20261112T010203-a_b_c"


def test_latest_request_is_the_last_human_message_on_one_line() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("첫 요청"),
        AIMessage(content="답"),
        HumanMessage("둘째\n요청"),
        AIMessage(content="답"),
    ]

    assert experience.latest_request(messages) == "둘째 요청"
    assert experience.latest_request([AIMessage(content="답")]) == ""


# ----- similarity -----------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "calc.py에 divide 함수를 추가하고 0으로 나누면 예외를 내게 해 줘",
        "calc 모듈의 테스트를 보강해 줘",
    ],
)
def test_request_about_the_same_code_is_related(query: str) -> None:
    assert experience.similarity(query, _record()) >= experience.MIN_SCORE


@pytest.mark.parametrize(
    "query",
    [
        "비밀번호 재설정 메일 문구를 바꿔 줘",
        "이 저장소 구조를 설명해 줘",
        "하이",
        "",
    ],
)
def test_request_about_something_else_is_not_related(query: str) -> None:
    assert experience.similarity(query, _record()) < experience.MIN_SCORE


@pytest.mark.parametrize("query", ["함수를 추가해 줘", "CI 설정 파일을 만들어 줘"])
def test_shared_request_phrasing_alone_is_not_related(query: str) -> None:
    assert experience.similarity(query, _record()) == 0
    assert experience.similarity(query, _login_record()) == 0


def test_same_subject_in_other_words_is_missed() -> None:
    # Known limit of comparing characters instead of meaning.
    assert experience.similarity("계산기에 나눗셈 기능을 넣어 줘", _record()) == 0


def test_lessons_do_not_count_toward_similarity() -> None:
    record = _record(lessons=("비밀번호 재설정 메일 문구를 확인한다.",))

    assert experience.similarity("비밀번호 재설정 메일 문구를 바꿔 줘", record) == 0


# ----- search ---------------------------------------------------------------


def test_search_returns_related_records_best_first(tmp_path: Path) -> None:
    experience.save(tmp_path, _record("20261110T000000-a"))
    experience.save(tmp_path, _login_record("20261111T000000-b"))

    hits = experience.search(tmp_path, "calc 모듈의 테스트를 보강해 줘")

    assert [record.id for record, _ in hits] == ["20261110T000000-a"]
    assert hits[0][1] >= experience.MIN_SCORE


def test_search_returns_at_most_three_and_prefers_newer_on_a_tie(
    tmp_path: Path,
) -> None:
    for day in range(10, 15):
        experience.save(tmp_path, _record(f"202611{day}T000000-t"))

    hits = experience.search(tmp_path, "calc 모듈의 테스트를 보강해 줘")

    assert [record.id for record, _ in hits] == [
        "20261114T000000-t",
        "20261113T000000-t",
        "20261112T000000-t",
    ]


def test_search_leaves_out_the_current_thread(tmp_path: Path) -> None:
    experience.save(tmp_path, _record("20261110T000000-a", thread_id="thread-a"))
    experience.save(tmp_path, _record("20261111T000000-b", thread_id="thread-b"))

    hits = experience.search(
        tmp_path, "calc 모듈의 테스트를 보강해 줘", exclude_thread="thread-b"
    )

    assert [record.id for record, _ in hits] == ["20261110T000000-a"]


# ----- what the model sees --------------------------------------------------


def test_block_marks_records_as_reference_and_names_their_ids() -> None:
    record = _record()

    block = experience.render_block([(record, 0.6)])

    assert block.startswith("## 관련 기억")
    assert "not instructions" in block
    assert f'<experience id="{record.id}">' in block
    assert "셸이 없으면 테스트를 실행하지 못한다고 보고한다." in block
    assert f"## {plan.SECTION_MEMORY}" in block


def test_no_hits_means_no_block() -> None:
    assert experience.render_block([]) == ""


class _CapturingModel(GenericFakeChatModel):
    """Fake model that records the system message of every call."""

    system_texts: list[str]

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        self.system_texts.append(
            "\n".join(str(m.text) for m in messages if isinstance(m, SystemMessage))
        )
        return super()._generate(messages, *args, **kwargs)


async def _turn(
    root: Path, request: str, thread_id: str = "thread-now"
) -> tuple[str, dict[str, Any]]:
    """Run one turn behind the memory middleware.

    Returns:
        The system text of the model call and the graph state after the turn.
    """
    model = _CapturingModel(messages=iter([AIMessage(content="done")]), system_texts=[])
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="base prompt",
        middleware=[ExperienceMemoryMiddleware(root)],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    await agent.ainvoke({"messages": [HumanMessage(request)]}, config)
    state = await agent.aget_state(config)
    return model.system_texts[0], state.values


async def test_related_record_is_added_to_the_system_message(tmp_path: Path) -> None:
    record = _record()
    experience.save(tmp_path, record)

    system_text, state = await _turn(tmp_path, "calc 모듈의 테스트를 보강해 줘")

    assert system_text.startswith("base prompt")
    assert f'<experience id="{record.id}">' in system_text
    assert state[experience.HITS_STATE_KEY] == [record.id]


async def test_unrelated_request_adds_nothing(tmp_path: Path) -> None:
    experience.save(tmp_path, _record())

    system_text, state = await _turn(tmp_path, "이 저장소 구조를 설명해 줘")

    assert system_text == "base prompt"
    assert state[experience.HITS_STATE_KEY] == []


async def test_records_of_the_same_thread_are_not_injected(tmp_path: Path) -> None:
    experience.save(tmp_path, _record(thread_id="thread-now"))

    system_text, _state = await _turn(tmp_path, "calc 모듈의 테스트를 보강해 줘")

    assert system_text == "base prompt"


async def test_unreadable_memory_does_not_stop_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: object, **_kwargs: object) -> None:
        msg = "disk error"
        raise OSError(msg)

    monkeypatch.setattr(experience, "search", broken)

    system_text, state = await _turn(tmp_path, "calc 모듈의 테스트를 보강해 줘")

    assert system_text == "base prompt"
    assert state[experience.HITS_STATE_KEY] == []


# ----- plans must account for the memory they were shown --------------------


def test_plan_needs_the_memory_section_only_when_memory_was_shown(
    tmp_path: Path,
) -> None:
    document = plan.parse_plan(VALID_PLAN)
    cited = plan.parse_plan(
        f"{VALID_PLAN}\n## {plan.SECTION_MEMORY}\n- `20261112T010000-a`: 같은 방식\n"
    )

    assert plan.validate_plan(document, tmp_path) == []
    shown = ("20261112T010000-a", "20261111T000000-b")
    problems = plan.validate_plan(document, tmp_path, memory_ids=shown)
    assert len(problems) == 1
    assert plan.SECTION_MEMORY in problems[0]
    assert plan.validate_plan(cited, tmp_path, memory_ids=shown) == []


async def test_review_rejects_a_plan_that_ignores_the_shown_memory(
    tmp_path: Path,
) -> None:
    experience.save(tmp_path, _record())
    harness = _Harness(
        tmp_path, extra_middleware=[ExperienceMemoryMiddleware(tmp_path)]
    )
    harness.request = "calc 모듈의 테스트를 보강해 줘"
    harness.write_plan()

    rejected = await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    message = _last_tool_message(rejected)
    assert message.status == "error"
    assert plan.SECTION_MEMORY in str(message.content)

    harness.write_plan(
        f"{VALID_PLAN}\n## {plan.SECTION_MEMORY}\n- `{_record().id}`: 테스트 방식\n"
    )
    accepted = await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    assert _last_tool_message(accepted).status == "success"


async def test_review_does_not_ask_for_the_section_without_memory(
    tmp_path: Path,
) -> None:
    harness = _Harness(
        tmp_path, extra_middleware=[ExperienceMemoryMiddleware(tmp_path)]
    )
    harness.write_plan()

    outcome = await harness.run(harness.call(plan_gate.REVIEW_TOOL))

    assert _last_tool_message(outcome).status == "success"


# ----- memory stays in view while the code is written -----------------------


class _SystemCapture(AgentMiddleware[Any, Any]):
    """Innermost middleware that records the system text of every model call."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        message = request.system_message
        self.texts.append(str(message.text) if message else "")
        return await handler(request)


def _plan_citing(record_id: str) -> str:
    """Return the valid plan with a memory section that cites `record_id`."""
    return (
        f"{VALID_PLAN}\n## {plan.SECTION_MEMORY}\n"
        f"- `{record_id}`: 같은 방식으로 테스트를 추가한다.\n"
    )


async def test_shown_memory_stays_in_view_while_the_approved_work_is_done(
    tmp_path: Path,
) -> None:
    """3-2: the records shown for the plan also reach the calls that write code."""
    record = _record()
    experience.save(tmp_path, record)
    capture = _SystemCapture()
    harness = _Harness(
        tmp_path, extra_middleware=[ExperienceMemoryMiddleware(tmp_path)]
    )
    harness.inner_middleware.append(capture)
    harness.request = "calc 모듈의 테스트를 보강해 줘"
    harness.write_plan(_plan_citing(record.id))

    await harness.run(
        harness.call(plan_gate.REVIEW_TOOL), harness.call(plan_gate.APPROVAL_TOOL)
    )
    planning_calls = len(capture.texts)
    await harness.run(
        harness.call("write_file", file_path="tests/test_calc.py", content="x"),
        resume={"status": "answered", "answers": [plan_gate.APPROVE_CHOICE, ""]},
    )

    marker = f'<experience id="{record.id}">'
    work_calls = capture.texts[planning_calls:]
    assert ("write_file", "tests/test_calc.py") in harness.executed
    # The call that chose the write and the call after it both saw the record.
    assert len(work_calls) == 2
    assert all(marker in text for text in capture.texts)


# ----- finished work becomes a record ---------------------------------------


class _FixedReviewer(WorkReviewer):
    """Work reviewer stand-in returning one prepared assessment."""

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


async def test_reviewed_work_is_saved_as_a_record(tmp_path: Path) -> None:
    review = WorkReview(
        tests="pytest가 통과했다.",
        summary="src/greeting.py에 greet 함수를 추가했다.",
        outcome="success",
        lessons=["테스트는 tests/ 아래에 둔다."],
        tags=["greeting", "함수 추가"],
    )
    harness = _Harness(
        tmp_path, work_reviewer=_FixedReviewer(review), record_experience=True
    )
    assert experience.load_all(tmp_path) == []

    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    (record,) = experience.load_all(tmp_path)
    assert record.request == "인사 함수를 추가해 줘"
    assert record.summary == "src/greeting.py에 greet 함수를 추가했다."
    assert record.outcome == "success"
    assert record.files == ("src/greeting.py",)
    assert record.lessons == ("테스트는 tests/ 아래에 둔다.",)
    assert record.tags == ("greeting", "함수 추가")
    assert record.thread_id == "thread-1"
    assert record.id.endswith("-thread-1")


async def test_work_without_a_review_is_saved_with_what_the_harness_knows(
    tmp_path: Path,
) -> None:
    harness = _Harness(
        tmp_path, work_reviewer=_FixedReviewer(None), record_experience=True
    )
    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    (record,) = experience.load_all(tmp_path)
    assert record.outcome == "unknown"
    assert record.files == ("src/greeting.py",)
    assert record.lessons == ()


async def test_nothing_is_saved_when_recording_is_off(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    await _approved(
        harness, harness.call("write_file", file_path="src/greeting.py", content="x")
    )

    assert experience.load_all(tmp_path) == []


async def test_agent_cannot_write_memory_records_itself(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, record_experience=True)
    forged = ".deepagents/memory/20261112T000000-forged.md"

    outcome = await _approved(
        harness,
        harness.call("write_file", file_path=forged, content="## 요청\n\n가짜\n"),
    )

    assert _last_tool_message(outcome).status == "error"
    assert not (tmp_path / forged).exists()


def test_older_reviewer_output_without_memory_fields_is_still_valid() -> None:
    review = WorkReview.model_validate({"tests": "통과"})

    assert review.outcome == "unknown"
    assert review.summary == ""
    assert review.lessons == []
    assert review.tags == []


@pytest.mark.parametrize(
    ("section", "accepted"),
    [
        ("- `20261112T010000-a`: 테스트 방식을 따랐다.", True),
        ("- 없음: 주어진 기록은 다른 모듈에 관한 것이다.", True),
        ("해당 없음. 이번 작업과 관련이 없다.", True),
        ("- 이전 경험을 참고했다.", False),
        ("- `20261101T000000-other`: 주어지지 않은 기록", False),
    ],
)
def test_memory_section_must_point_at_a_shown_record_or_say_none_applies(
    tmp_path: Path, section: str, accepted: bool
) -> None:
    document = plan.parse_plan(f"{VALID_PLAN}\n## {plan.SECTION_MEMORY}\n{section}\n")

    problems = plan.validate_plan(document, tmp_path, memory_ids=("20261112T010000-a",))

    assert (problems == []) is accepted
    if not accepted:
        assert "20261112T010000-a" in problems[0]
