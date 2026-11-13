"""Plan documents: where they live, what they must contain, and their scope.

A plan is a Markdown file the agent writes before it may change code. This
module holds the rules that do not need a model: the file location, the required
sections, the structural checks, and the test for whether a path is covered by
the plan's "수정 대상" (files to change) section.

Functions that resolve paths touch the filesystem and are therefore blocking.
Callers on an event loop run them in a worker thread.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

HARNESS_DIRECTORY = Path(".deepagents")
"""Project folder holding what the harness itself writes."""

PLAN_DIRECTORY = HARNESS_DIRECTORY / "plans"
"""Plan folder, relative to the project root."""

SECTION_REQUIREMENTS = "요구사항"
SECTION_SCOPE = "작업 범위"
SECTION_DONE = "완료 조건"
SECTION_TARGETS = "수정 대상"
SECTION_STEPS = "작업 순서"
SECTION_TESTS = "테스트 방법"

SECTION_MEMORY = "참고한 기억"
"""Section a plan needs only when related experience was shown to the agent."""

REQUIRED_SECTIONS = (
    SECTION_REQUIREMENTS,
    SECTION_SCOPE,
    SECTION_DONE,
    SECTION_TARGETS,
    SECTION_STEPS,
    SECTION_TESTS,
)
"""Section titles every plan must have, in the order they are listed."""

REVIEW_SUFFIX = ".review.md"
"""Suffix of the review record written next to a plan by the harness."""

REPORT_SUFFIX = ".report.md"
"""Suffix of the post-work report written next to a plan by the harness."""

APPROVED_SUFFIX = ".approved.md"
"""Suffix of the copy the harness keeps of a plan once its request is over."""

BASELINE_SUFFIX = ".baseline.json"
"""Suffix of the file recording which files were changed before approval."""

_HEADING = re.compile(r"^##\s+(?:\d+[.)]\s*)?(?P<title>.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?P<body>\S.*)$")
_NUMBERED_ITEM = re.compile(r"^\s*\d+[.)]\s+\S")
_BACKTICKED = re.compile(r"`([^`]+)`")
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")
_NONE_APPLIES = re.compile(r"없음|해당 없음|none", re.IGNORECASE)


@dataclass(frozen=True)
class PlanDocument:
    """A parsed plan.

    Attributes:
        sections: Section body text keyed by section title.
        targets: Entries of the "수정 대상" section. An entry ending in `/` names
            a folder; any other entry names one file.
    """

    sections: dict[str, str]
    targets: tuple[str, ...]


def plan_directory(root: Path) -> Path:
    """Return the plan folder of a project.

    Args:
        root: Project root.

    Returns:
        Absolute plan folder path. The folder may not exist yet.
    """
    return root / PLAN_DIRECTORY


def plan_file(root: Path, thread_id: str) -> Path:
    """Return the plan file of one conversation thread.

    Args:
        root: Project root.
        thread_id: Thread identifier. Characters outside `[A-Za-z0-9_-]` are
            replaced so the result is always a single file name.

    Returns:
        Path of the thread's plan file. The file may not exist yet.
    """
    name = _UNSAFE_NAME.sub("_", thread_id) or "thread"
    return plan_directory(root) / f"{name}.md"


def hash_text(text: str) -> str:
    """Return the fingerprint used to detect a changed plan.

    Args:
        text: Plan file content.

    Returns:
        Hex SHA-256 digest of the UTF-8 encoded text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str | None:
    """Read a UTF-8 file, treating a missing or unreadable file as absent.

    Args:
        path: File to read.

    Returns:
        File content, or `None` when the file cannot be read.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        return None


def parse_targets(section: str) -> tuple[str, ...]:
    """Extract path entries from a "수정 대상" section.

    Each list item contributes one entry: the first backticked span if the item
    has one, otherwise its first whitespace-delimited word.

    Args:
        section: Body text of the section.

    Returns:
        Entries in document order, without duplicates.
    """
    entries: list[str] = []
    for line in section.splitlines():
        item = _LIST_ITEM.match(line)
        if item is None:
            continue
        body = item.group("body")
        quoted = _BACKTICKED.search(body)
        entry = (quoted.group(1) if quoted else body.split()[0]).strip().strip("'\"")
        if entry and entry not in entries:
            entries.append(entry)
    return tuple(entries)


def parse_plan(text: str) -> PlanDocument:
    """Split a plan into its level-two sections.

    Args:
        text: Plan file content.

    Returns:
        Parsed plan. A section that appears twice keeps its first body.
    """
    sections: dict[str, str] = {}
    title: str | None = None
    body: list[str] = []

    def close() -> None:
        """Store the section collected so far; a repeated title keeps the first."""
        if title is not None and title not in sections:
            sections[title] = "\n".join(body).strip()

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading is not None:
            close()
            title = heading.group("title")
            body = []
        elif line.startswith("# "):
            close()
            title = None
        elif title is not None:
            body.append(line)
    close()
    return PlanDocument(
        sections=sections,
        targets=parse_targets(sections.get(SECTION_TARGETS, "")),
    )


def resolve_path(raw: str, root: Path) -> Path:
    """Resolve a path the way file tools see it, following symlinks.

    Args:
        raw: Path as given to a tool or written in a plan.
        root: Base for relative paths.

    Returns:
        Absolute path with symlinks and `..` segments resolved.
    """
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def is_inside(path: Path, folder: Path) -> bool:
    """Report whether `path` is `folder` itself or lies below it.

    Args:
        path: Resolved path to test.
        folder: Resolved folder.

    Returns:
        `True` when `path` is inside `folder`.
    """
    return path == folder or folder in path.parents


def lexically_inside(raw: str, root: Path, folder: Path) -> bool:
    """Report whether a path is inside a folder without touching the disk.

    The check normalizes `..` segments but does not follow symlinks. It is used
    where blocking I/O is not allowed and as a second condition next to the
    resolved check, so a symlink cannot make the two disagree in the permissive
    direction.

    Args:
        raw: Path as given to a tool.
        root: Base for relative paths.
        folder: Folder to test against.

    Returns:
        `True` when the normalized path is `folder` or lies below it.
    """
    absolute = os.path.normpath(root / Path(raw).expanduser())
    base = os.path.normpath(folder)
    return absolute == base or absolute.startswith(base + os.sep)


def is_plan_file_name(name: str) -> bool:
    """Report whether a file name is a plan the agent may write.

    Review records, post-work reports and kept copies of approved plans share
    the plan folder but are written by the harness only.

    Args:
        name: File name without folders.

    Returns:
        `True` for a Markdown file that is none of those harness records.
    """
    return name.endswith(".md") and not name.endswith(
        (REVIEW_SUFFIX, REPORT_SUFFIX, APPROVED_SUFFIX)
    )


def is_target(path: Path, root: Path, targets: tuple[str, ...]) -> bool:
    """Report whether a path is covered by a plan's "수정 대상" entries.

    Args:
        path: Resolved path a tool wants to change.
        root: Project root that relative entries are resolved against.
        targets: Entries from the plan.

    Returns:
        `True` when `path` equals a file entry or lies below a folder entry.
    """
    for entry in targets:
        target = resolve_path(entry, root)
        if entry.endswith("/"):
            if is_inside(path, target):
                return True
        elif path == target:
            return True
    return False


def validate_plan(
    document: PlanDocument, root: Path, *, memory_ids: Sequence[str] = ()
) -> list[str]:
    """Check the structure of a plan.

    The checks are structural only. Whether the content is sensible is left to
    the AI reviewer.

    Args:
        document: Parsed plan.
        root: Project root, used to reject entries outside the project.
        memory_ids: Ids of the experience records shown to the agent for
            this request. When there are any, the plan must name at least
            one of them or say that none applies.

    Returns:
        Problems found, as messages for the agent. Empty when the plan is
            structurally valid.
    """
    problems: list[str] = [
        f"'## {title}' 절이 없거나 비어 있습니다."
        for title in REQUIRED_SECTIONS
        if not document.sections.get(title, "").strip()
    ]
    if memory_ids:
        cited = document.sections.get(SECTION_MEMORY, "")
        if not cited.strip():
            problems.append(
                f"이 요청에는 관련 기억이 주어졌습니다. '## {SECTION_MEMORY}' 절에 "
                "참고한 기억의 id와 반영한 내용을 적거나, 해당하는 것이 없다면 그 "
                "이유를 적습니다."
            )
        elif not any(record in cited for record in memory_ids) and not (
            _NONE_APPLIES.search(cited)
        ):
            listed = ", ".join(f"`{record}`" for record in memory_ids)
            problems.append(
                f"'## {SECTION_MEMORY}' 절이 이 요청에 주어진 기억을 가리키지 "
                f"않습니다. 주어진 기억의 id({listed}) 가운데 참고한 것을 적거나, "
                "해당하는 것이 없으면 '없음'과 그 이유를 적습니다."
            )
    done = document.sections.get(SECTION_DONE, "")
    if done.strip() and not any(_LIST_ITEM.match(line) for line in done.splitlines()):
        problems.append(f"'## {SECTION_DONE}'에 목록 항목이 하나 이상 있어야 합니다.")
    steps = document.sections.get(SECTION_STEPS, "")
    if steps.strip() and not any(
        _NUMBERED_ITEM.match(line) for line in steps.splitlines()
    ):
        problems.append(f"'## {SECTION_STEPS}'에 번호 목록이 하나 이상 있어야 합니다.")
    if document.sections.get(SECTION_TARGETS, "").strip():
        problems.extend(_target_problems(document.targets, root))
    return problems


def _target_problems(targets: tuple[str, ...], root: Path) -> list[str]:
    """Check the entries of a "수정 대상" section.

    Args:
        targets: Entries from the plan.
        root: Project root.

    Returns:
        Problems found. Empty when every entry stays inside the project and
            none of them names the project root itself.
    """
    if not targets:
        return [
            (
                f"'## {SECTION_TARGETS}'에 파일 경로 목록이 필요합니다. "
                "항목마다 경로를 백틱으로 감싸 적습니다."
            )
        ]
    problems: list[str] = []
    resolved_root = root.resolve()
    harness_folder = (root / HARNESS_DIRECTORY).resolve()
    for entry in targets:
        target = resolve_path(entry, root)
        if target == resolved_root:
            problems.append(
                f"'## {SECTION_TARGETS}'의 `{entry}`는 프로젝트 전체를 가리킵니다. "
                "더 좁은 파일이나 폴더를 적습니다."
            )
        elif not is_inside(target, resolved_root):
            problems.append(
                f"'## {SECTION_TARGETS}'의 `{entry}`는 프로젝트 밖의 경로입니다."
            )
        elif is_inside(target, harness_folder):
            problems.append(
                f"'## {SECTION_TARGETS}'의 `{entry}`는 하네스가 관리하는 "
                f"`{HARNESS_DIRECTORY}/` 폴더 안입니다. 계획서, 기억, 개선 기록은 "
                "수정 대상이 될 수 없습니다."
            )
    return problems
