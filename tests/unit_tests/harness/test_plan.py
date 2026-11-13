"""Tests for plan document rules: location, structure, and scope."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_code.harness import plan

if TYPE_CHECKING:
    from pathlib import Path

VALID_PLAN = """\
# 계획서

## 요구사항
사용자 이름을 입력받아 인사말을 돌려준다.

## 작업 범위
`greet` 함수 추가. 범위 밖: CLI 인자 처리.

## 완료 조건
- `greet("A")`가 `"Hello, A"`를 돌려준다.
- 기존 테스트가 통과한다.

## 수정 대상
- `src/greeting.py` 새 함수
- `tests/` 테스트 추가

## 작업 순서
1. 실패하는 테스트를 쓴다.
2. 함수를 구현한다.

## 테스트 방법
`pytest tests/test_greeting.py`가 통과한다.
"""


def test_plan_file_is_per_thread_under_the_plan_folder(tmp_path: Path) -> None:
    path = plan.plan_file(tmp_path, "thread-1")

    assert path == tmp_path / ".deepagents" / "plans" / "thread-1.md"


def test_plan_file_name_cannot_escape_the_plan_folder(tmp_path: Path) -> None:
    path = plan.plan_file(tmp_path, "../../etc/passwd")

    assert path.parent == plan.plan_directory(tmp_path)
    assert "/" not in path.name
    assert ".." not in path.name


def test_hash_changes_with_the_text() -> None:
    assert plan.hash_text("a") == plan.hash_text("a")
    assert plan.hash_text("a") != plan.hash_text("a ")


def test_parse_plan_reads_sections_and_targets() -> None:
    document = plan.parse_plan(VALID_PLAN)

    assert set(plan.REQUIRED_SECTIONS) <= set(document.sections)
    assert document.targets == ("src/greeting.py", "tests/")
    assert document.sections[plan.SECTION_TESTS].startswith("`pytest")


def test_parse_plan_accepts_numbered_headings() -> None:
    document = plan.parse_plan("## 1. 요구사항\n내용\n## 2) 작업 범위\n범위\n")

    assert document.sections == {"요구사항": "내용", "작업 범위": "범위"}


def test_parse_targets_uses_first_word_without_backticks() -> None:
    assert plan.parse_targets("- src/a.py 설명\n* src/b.py\n본문 줄\n") == (
        "src/a.py",
        "src/b.py",
    )


def test_valid_plan_has_no_problems(tmp_path: Path) -> None:
    assert plan.validate_plan(plan.parse_plan(VALID_PLAN), tmp_path) == []


@pytest.mark.parametrize("title", plan.REQUIRED_SECTIONS)
def test_missing_section_is_reported(tmp_path: Path, title: str) -> None:
    text = VALID_PLAN.replace(f"## {title}", "## 다른 제목")

    problems = plan.validate_plan(plan.parse_plan(text), tmp_path)

    assert any(title in problem for problem in problems)


def test_empty_section_is_reported(tmp_path: Path) -> None:
    text = VALID_PLAN.replace("사용자 이름을 입력받아 인사말을 돌려준다.", "")

    problems = plan.validate_plan(plan.parse_plan(text), tmp_path)

    assert any(plan.SECTION_REQUIREMENTS in problem for problem in problems)


def test_done_criteria_need_a_list(tmp_path: Path) -> None:
    text = VALID_PLAN.replace(
        '- `greet("A")`가 `"Hello, A"`를 돌려준다.\n- 기존 테스트가 통과한다.',
        "잘 동작한다.",
    )

    problems = plan.validate_plan(plan.parse_plan(text), tmp_path)

    assert any(plan.SECTION_DONE in problem for problem in problems)


def test_steps_need_a_numbered_list(tmp_path: Path) -> None:
    text = VALID_PLAN.replace(
        "1. 실패하는 테스트를 쓴다.\n2. 함수를 구현한다.", "- 구현한다."
    )

    problems = plan.validate_plan(plan.parse_plan(text), tmp_path)

    assert any(plan.SECTION_STEPS in problem for problem in problems)


@pytest.mark.parametrize("entry", [".", "./", "src/.."])
def test_target_naming_the_whole_project_is_rejected(
    tmp_path: Path, entry: str
) -> None:
    text = VALID_PLAN.replace("- `src/greeting.py` 새 함수", f"- `{entry}`")

    problems = plan.validate_plan(plan.parse_plan(text), tmp_path)

    assert any("프로젝트 전체" in problem for problem in problems)


@pytest.mark.parametrize("entry", ["../other/a.py", "/etc/passwd"])
def test_target_outside_the_project_is_rejected(tmp_path: Path, entry: str) -> None:
    text = VALID_PLAN.replace("- `src/greeting.py` 새 함수", f"- `{entry}`")

    problems = plan.validate_plan(plan.parse_plan(text), tmp_path)

    assert any("프로젝트 밖" in problem for problem in problems)


def test_targets_without_paths_are_reported(tmp_path: Path) -> None:
    text = VALID_PLAN.replace(
        "- `src/greeting.py` 새 함수\n- `tests/` 테스트 추가", "관련 파일 전부"
    )

    problems = plan.validate_plan(plan.parse_plan(text), tmp_path)

    assert any("파일 경로 목록" in problem for problem in problems)


def test_is_target_matches_files_exactly_and_folders_by_prefix(
    tmp_path: Path,
) -> None:
    targets = ("src/greeting.py", "tests/")

    def covered(raw: str) -> bool:
        return plan.is_target(plan.resolve_path(raw, tmp_path), tmp_path, targets)

    assert covered("src/greeting.py")
    assert covered(str(tmp_path / "src" / "greeting.py"))
    assert covered("tests/test_greeting.py")
    assert covered("tests/unit/test_more.py")
    assert not covered("src/other.py")
    assert not covered("src/greeting.py.bak")
    assert not covered("tests/../src/other.py")
    assert not covered("testsuite/a.py")


def test_file_target_does_not_cover_a_folder_of_the_same_name(
    tmp_path: Path,
) -> None:
    path = plan.resolve_path("src/pkg/module.py", tmp_path)

    assert not plan.is_target(path, tmp_path, ("src/pkg",))
    assert plan.is_target(path, tmp_path, ("src/pkg/",))


def test_symlink_does_not_extend_a_folder_target(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "secret").mkdir()
    (tmp_path / "tests" / "link").symlink_to(tmp_path / "secret")

    path = plan.resolve_path("tests/link/key.txt", tmp_path)

    assert not plan.is_target(path, tmp_path, ("tests/",))


def test_lexically_inside_normalizes_parent_segments(tmp_path: Path) -> None:
    folder = plan.plan_directory(tmp_path)

    assert plan.lexically_inside(".deepagents/plans/t.md", tmp_path, folder)
    assert plan.lexically_inside(str(folder / "t.md"), tmp_path, folder)
    assert not plan.lexically_inside(".deepagents/plans/../x.md", tmp_path, folder)
    assert not plan.lexically_inside(".deepagents/plansX/t.md", tmp_path, folder)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("thread-1.md", True),
        ("thread-1.review.md", False),
        ("thread-1.report.md", False),
        ("thread-1.txt", False),
    ],
)
def test_only_plan_files_are_writable_names(name: str, expected: bool) -> None:
    assert plan.is_plan_file_name(name) is expected


def test_read_text_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    assert plan.read_text(tmp_path / "missing.md") is None
