"""Post-work report: an approved plan compared with what actually changed.

The report has two parts with different sources. The harness collects what
happened in the turn (file tools that succeeded, commands that ran, files git
sees as changed since approval) and checks it against the plan's "수정 대상"
entries. A separate reviewer agent then reads the plan and that evidence and
writes its assessment. The first part does not depend on a model, so it stays
in the report when the reviewer fails.

The report is a record. It never sends the agent back to redo work.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess  # noqa: S404  # Bounded read-only Git subprocesses.
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator

from deepagents_code.harness import plan as plan_rules
from deepagents_code.harness.plan_review import (
    HARNESS_NOTE,
    REVIEW_ATTEMPTS,
    earlier_requests_note,
    read_structured,
    root_note,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

MAX_ITEMS = 8
"""Upper bound on entries the reviewer may return per list."""

MAX_LESSONS = 5
"""Upper bound on lessons the reviewer may return."""

MAX_COMMANDS = 20
"""Commands of one turn passed to the reviewer and listed in the report."""

FILE_TOOLS = frozenset({"write_file", "edit_file", "delete"})
"""Tools whose successful call changed the file named in `file_path`."""

COMMAND_TOOL = "execute"

OTHER_CHANGE_TOOLS = frozenset(
    {"js_eval", "start_async_task", "task", "update_async_task"}
)
"""Tools that can change the workspace without naming a file or a command."""

_GIT_TIMEOUT_SECONDS = 10
_PLACEHOLDER = re.compile(r"^(없음|해당 없음|none|n/a|-)\s*([:.,(]|$)", re.IGNORECASE)
"""A list entry that states the absence of findings instead of a finding."""
_OUTPUT_HEAD_CHARS = 200
_OUTPUT_TAIL_CHARS = 600
_REPLY_EXCERPT_CHARS = 2000
_CALL_SUMMARY_CHARS = 160
_STATUS_FIELD_WIDTH = 3
"""Width of the `XY ` status prefix of a `git status --porcelain` entry."""

WORK_REVIEW_SYSTEM_PROMPT = """\
You review finished work against the plan that was approved for it.

You are given the user's request, the approved plan, and evidence the harness
collected: files changed, commands run with the start of their output, the
harness's own comparison with the plan's "수정 대상" list, and the agent's final
reply. You may read the repository with the read-only tools to check the
changed files. You cannot change anything.

Report four things, in the language of the plan:
- followed: steps or completion criteria of the plan that the work fulfils.
- missing: steps or completion criteria of the plan with no matching work.
- unplanned: changes that the plan does not describe.
- tests: what the evidence shows about tests - which command ran and whether it
  passed. Say so plainly when no test run appears in the evidence. Do not infer
  a passing result from the agent's reply alone.

State each entry as one concrete sentence. Do not propose new work.

Also fill in the record that is kept as project memory for later tasks:
- summary: one sentence saying what was changed and where.
- outcome: `success` when the work fulfils the plan and the evidence shows
  passing tests; `partial` when something is missing or unverified; `failure`
  when the work does not do what was asked.
- lessons: what a later task of this kind should do the same or differently,
  drawn from this work and specific to this project. Leave the list empty
  rather than writing general advice.
- tags: short keywords for finding this record again, such as the feature,
  the module, and the kind of change.
"""


class WorkReview(BaseModel):
    """Reviewer's assessment of finished work against the approved plan."""

    followed: list[str] = Field(
        default_factory=list,
        max_length=MAX_ITEMS,
        description="Plan steps or completion criteria the work fulfils.",
    )
    missing: list[str] = Field(
        default_factory=list,
        max_length=MAX_ITEMS,
        description="Plan steps or completion criteria with no matching work.",
    )
    unplanned: list[str] = Field(
        default_factory=list,
        max_length=MAX_ITEMS,
        description="Changes the plan does not describe.",
    )
    tests: str = Field(
        description="What the evidence shows about test runs and their results."
    )
    summary: str = Field(
        default="", description="One sentence: what was changed and where."
    )
    outcome: Literal["success", "partial", "failure", "unknown"] = Field(
        default="unknown",
        description="`success`, `partial` or `failure`, judged from the evidence.",
    )
    lessons: list[str] = Field(
        default_factory=list,
        max_length=MAX_LESSONS,
        description="Project-specific lessons for a later task of this kind.",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=MAX_ITEMS,
        description="Short keywords for finding this record again.",
    )

    @field_validator(
        "followed", "missing", "unplanned", "lessons", "tags", mode="before"
    )
    @classmethod
    def _drop_placeholders(cls, value: object) -> object:
        """Remove entries that only say there is nothing to report.

        A reviewer sometimes fills an empty list with a sentence such as
        "없음: 편집된 파일은 하나뿐이다". Left in, that sentence would count as
        a finding: an entry under `unplanned` is a failure signal.

        Args:
            value: Raw list from the reviewer.

        Returns:
            The list without placeholder entries; other values unchanged.
        """
        if not isinstance(value, list):
            return value
        return [
            item
            for item in value
            if not (isinstance(item, str) and _PLACEHOLDER.match(item.strip()))
        ]


@dataclass(frozen=True)
class TurnActivity:
    """What the agent did since the latest user message.

    Attributes:
        edited: Paths that file tools changed successfully, as given to the
            tools, in call order and without duplicates.
        commands: Shell commands that ran, each with an excerpt of its output.
        final_reply: Text of the agent's last message, possibly empty.
        other_calls: Other calls that can change the workspace, such as a
            subagent task or interpreter code, each as the tool name and the
            start of its main argument.
    """

    edited: tuple[str, ...]
    commands: tuple[tuple[str, str], ...]
    final_reply: str
    other_calls: tuple[str, ...] = ()

    @property
    def changed_something(self) -> bool:
        """Whether the turn ran any tool that can change the workspace."""
        return bool(self.edited or self.commands or self.other_calls)


@dataclass(frozen=True)
class Comparison:
    """Harness comparison of actual changes with the plan's "수정 대상".

    Attributes:
        changed: Changed paths relative to the project root where possible.
        unplanned: Changed paths that no "수정 대상" entry covers.
        untouched: "수정 대상" entries that cover no changed path.
        used_git: Whether git contributed to `changed`. Without git only file
            tool calls are known, so changes made by commands are not listed.
    """

    changed: tuple[str, ...]
    unplanned: tuple[str, ...]
    untouched: tuple[str, ...]
    used_git: bool


def _excerpt(output: str) -> str:
    """Shorten command output, keeping its start and its end.

    Test runners print their pass/fail summary last, so the end is kept in
    full and the start only briefly.

    Args:
        output: Command output.

    Returns:
        The output itself when short, otherwise its start and end joined by
            an omission marker.
    """
    text = output.strip()
    if len(text) <= _OUTPUT_HEAD_CHARS + _OUTPUT_TAIL_CHARS:
        return text
    return f"{text[:_OUTPUT_HEAD_CHARS]}\n...\n{text[-_OUTPUT_TAIL_CHARS:]}"


def _call_summary(name: str, args: Mapping[str, Any]) -> str:
    """Describe a tool call in one line.

    Args:
        name: Tool name.
        args: Tool arguments.

    Returns:
        The tool name, followed by the start of its first text argument when
            it has one.
    """
    for value in args.values():
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())[:_CALL_SUMMARY_CHARS]
            return f"{name}: {text}"
    return name


def collect_turn_activity(messages: Sequence[Any]) -> TurnActivity:
    """Collect the workspace-changing tool calls of the current turn.

    The turn starts after the latest human message. A tool call counts only
    when its result is present and is not an error, so calls the plan gate
    refused are left out.

    Args:
        messages: Conversation messages in order.

    Returns:
        The turn's successful file changes, commands and final reply.
    """
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index + 1
    turn = messages[start:]

    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in turn:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            call_id = call.get("id")
            if call_id:
                calls[call_id] = (call["name"], call.get("args") or {})

    edited: list[str] = []
    commands: list[tuple[str, str]] = []
    other_calls: list[str] = []
    for message in turn:
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        name, args = calls.get(message.tool_call_id, ("", {}))
        if name in FILE_TOOLS:
            file_path = args.get("file_path")
            if isinstance(file_path, str) and file_path and file_path not in edited:
                edited.append(file_path)
        elif name == COMMAND_TOOL:
            command = args.get("command")
            if isinstance(command, str) and command:
                commands.append((command, _excerpt(str(message.text))))
        elif name in OTHER_CHANGE_TOOLS:
            other_calls.append(_call_summary(name, args))

    final_reply = ""
    for message in reversed(turn):
        if isinstance(message, AIMessage) and not message.tool_calls:
            final_reply = str(message.text).strip()
            break
    return TurnActivity(
        edited=tuple(edited),
        commands=tuple(commands[-MAX_COMMANDS:]),
        final_reply=final_reply[:_REPLY_EXCERPT_CHARS],
        other_calls=tuple(other_calls[-MAX_COMMANDS:]),
    )


def _git(root: Path, *args: str) -> str | None:
    """Run a read-only git command in the project.

    Args:
        root: Project root.
        *args: Arguments after `git -C <root>`.

    Returns:
        Standard output, or `None` when git is missing, fails or times out.
    """
    try:
        completed = subprocess.run(  # noqa: S603  # fixed argv, no shell
            ["git", "-C", str(root), *args],  # noqa: S607  # git from PATH
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return completed.stdout if completed.returncode == 0 else None


def _fingerprint(path: Path) -> str:
    """Describe a file's current state cheaply, without reading it.

    Args:
        path: File to describe.

    Returns:
        Size and modification time, or `absent` when the file does not exist.
    """
    try:
        stat = path.stat()
    except OSError:
        return "absent"
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def git_dirty_files(root: Path) -> dict[str, str] | None:
    """List files git reports as changed or untracked under the project.

    Args:
        root: Project root.

    Returns:
        Fingerprint of each changed file keyed by absolute path, or `None`
            when the project is not a git work tree or git cannot be run.
    """
    top = _git(root, "rev-parse", "--show-toplevel")
    status = _git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."
    )
    if top is None or status is None:
        return None
    base = plan_rules.resolve_path(top.strip(), root)
    dirty: dict[str, str] = {}
    entries = iter(status.split("\0"))
    for entry in entries:
        if len(entry) <= _STATUS_FIELD_WIDTH:
            continue
        code, name = entry[:2], entry[_STATUS_FIELD_WIDTH:]
        if "R" in code or "C" in code:
            # A rename or copy is followed by its source path as its own entry.
            next(entries, None)
        path = base / name
        dirty[str(path)] = _fingerprint(path)
    return dirty


def baseline_file(plan_path: Path) -> Path:
    """Return where the approval-time git state of a plan is stored.

    Args:
        plan_path: Plan file.

    Returns:
        Path of the baseline file next to the plan.
    """
    return plan_path.with_name(plan_path.stem + plan_rules.BASELINE_SUFFIX)


def write_baseline(root: Path, plan_path: Path) -> None:
    """Record which files were already changed when the plan was approved.

    The report compares against this record, so changes that predate the plan
    are not reported as the plan's work. Nothing is written outside a git work
    tree.

    Args:
        root: Project root.
        plan_path: Approved plan file.
    """
    dirty = git_dirty_files(root)
    if dirty is None:
        return
    target = baseline_file(plan_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dirty, ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.warning("Could not write plan baseline %s", target, exc_info=True)


def _read_baseline(plan_path: Path) -> dict[str, str]:
    """Read the approval-time git state of a plan.

    Args:
        plan_path: Plan file.

    Returns:
        Fingerprints keyed by absolute path; empty when no usable record exists.
    """
    text = plan_rules.read_text(baseline_file(plan_path))
    if text is None:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _display(path: Path, root: Path) -> str:
    """Format a path for the report.

    Args:
        path: Absolute path.
        root: Project root.

    Returns:
        POSIX path relative to the root, or the absolute path outside it.
    """
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def compare_with_plan(
    root: Path, plan_path: Path, plan_text: str, activity: TurnActivity
) -> Comparison:
    """Compare actual changes with the plan's "수정 대상" entries.

    Changed files are the union of successful file tool calls in the turn and
    files git reports as changed that differ from the approval-time baseline.
    Files under `.deepagents/` are the harness's own records (plans, memory,
    improvements, step records) and are left out.

    Args:
        root: Project root.
        plan_path: Approved plan file.
        plan_text: Content of the approved plan.
        activity: Tool activity of the turn.

    Returns:
        Changed paths and how they relate to the plan.
    """
    targets = plan_rules.parse_plan(plan_text).targets
    harness_folder = (root / plan_rules.HARNESS_DIRECTORY).resolve()
    changed: list[Path] = []

    def add(path: Path) -> None:
        """Record a changed path once, leaving out the harness's own folder."""
        if path not in changed and not plan_rules.is_inside(path, harness_folder):
            changed.append(path)

    for raw in activity.edited:
        add(plan_rules.resolve_path(raw, root))
    dirty = git_dirty_files(root)
    if dirty is not None:
        baseline = _read_baseline(plan_path)
        for name, fingerprint in dirty.items():
            if baseline.get(name) != fingerprint:
                add(plan_rules.resolve_path(name, root))

    unplanned = [p for p in changed if not plan_rules.is_target(p, root, targets)]
    untouched = [
        entry
        for entry in targets
        if not any(plan_rules.is_target(p, root, (entry,)) for p in changed)
    ]
    return Comparison(
        changed=tuple(_display(p, root) for p in changed),
        unplanned=tuple(_display(p, root) for p in unplanned),
        untouched=tuple(untouched),
        used_git=dirty is not None,
    )


def _bullets(items: Sequence[str], *, code: bool = False) -> str:
    """Format a Markdown bullet list.

    Args:
        items: Entries in order.
        code: Whether to render each entry as inline code.

    Returns:
        One bullet per entry, or a single "없음" bullet for an empty list.
    """
    if not items:
        return "- 없음"
    return "\n".join(f"- `{item}`" if code else f"- {item}" for item in items)


def evidence_text(activity: TurnActivity, comparison: Comparison) -> str:
    """Format the harness's findings for the reviewer.

    Args:
        activity: Tool activity of the turn.
        comparison: Comparison with the plan.

    Returns:
        Plain-text evidence block.
    """
    commands = "\n".join(
        f"$ {command}\n{output or '(no output)'}"
        for command, output in activity.commands
    )
    source = (
        "file tool calls and git"
        if comparison.used_git
        else "file tool calls only (not a git work tree)"
    )
    return (
        f"Changed files ({source}):\n{_bullets(comparison.changed)}\n\n"
        f"Changed but not under '수정 대상':\n{_bullets(comparison.unplanned)}\n\n"
        f"'수정 대상' entries with no change:\n{_bullets(comparison.untouched)}\n\n"
        f"Commands run:\n{commands or '(none)'}\n\n"
        "Other calls that can change files (subagent tasks, interpreter "
        f"code):\n{_bullets(activity.other_calls)}\n\n"
        f"Agent's final reply:\n{activity.final_reply or '(empty)'}"
    )


def render_report(
    *,
    plan_hash: str,
    activity: TurnActivity,
    comparison: Comparison,
    review: WorkReview | None,
    review_note: str = "",
) -> str:
    """Format one report entry.

    Args:
        plan_hash: Fingerprint of the approved plan text.
        activity: Tool activity of the turn.
        comparison: Harness comparison with the plan.
        review: Reviewer's assessment, or `None` when there is none.
        review_note: Why there is no assessment. Used when `review` is `None`.

    Returns:
        Markdown block to append to the report file.
    """
    source = (
        "파일 도구 호출과 git 상태(승인 시점 이후 바뀐 것)"
        if comparison.used_git
        else "파일 도구 호출만 (git 저장소가 아니어서 명령으로 바꾼 파일은 빠짐)"
    )
    commands = _bullets([command for command, _ in activity.commands], code=True)
    if review is None:
        assessment = f"AI 리뷰 없음: {review_note or '리뷰어가 설정되지 않았습니다'}\n"
    else:
        assessment = (
            f"#### 계획대로 한 것\n\n{_bullets(review.followed)}\n\n"
            f"#### 빠진 단계\n\n{_bullets(review.missing)}\n\n"
            f"#### 계획에 없는 변경\n\n{_bullets(review.unplanned)}\n\n"
            f"#### 테스트 결과\n\n{review.tests.strip() or '없음'}\n"
        )
    return (
        "## 작업 후 리뷰\n\n"
        f"- 시각: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        f"- 계획서 해시: `{plan_hash}`\n\n"
        "### 실제 변경 (하네스가 수집)\n\n"
        f"출처: {source}\n\n"
        f"변경된 파일:\n\n{_bullets(comparison.changed, code=True)}\n\n"
        f"실행한 명령:\n\n{commands}\n\n"
        "그 밖에 작업 공간을 바꿀 수 있는 호출 (서브에이전트, 인터프리터):\n\n"
        f"{_bullets(activity.other_calls)}\n\n"
        "### 수정 대상과의 대조 (하네스가 판정)\n\n"
        "계획에 없는 변경:\n\n"
        f"{_bullets(comparison.unplanned, code=True)}\n\n"
        "변경이 없는 수정 대상:\n\n"
        f"{_bullets(comparison.untouched, code=True)}\n\n"
        f"### AI 리뷰\n\n{assessment}\n"
    )


def append_report(path: Path, text: str) -> None:
    """Append a report entry, creating the folder and a title when needed.

    Args:
        path: Report file.
        text: Entry to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    title = path.name.removesuffix(plan_rules.REPORT_SUFFIX)
    header = "" if path.exists() else f"# 작업 후 보고서: {title}\n\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(header + text)


def review_prompt(
    request_text: str,
    plan_text: str,
    evidence: str,
    project_root: str = "",
    earlier_requests: Sequence[str] = (),
) -> str:
    """Build the reviewer's input message.

    Args:
        request_text: The request the work answers.
        plan_text: Approved plan content.
        evidence: Harness findings from `evidence_text`.
        project_root: Absolute path of the project root on the host.
        earlier_requests: Requests of earlier turns, given as context only.

    Returns:
        Message text with request, plan and evidence in separate tagged blocks.
    """
    return (
        "Review the finished work against its plan.\n\n"
        f"{root_note(project_root)}"
        f"{HARNESS_NOTE}"
        f"{earlier_requests_note(earlier_requests)}"
        f"<user_request>\n{request_text.strip()}\n</user_request>\n\n"
        f"<approved_plan>\n{plan_text.strip()}\n</approved_plan>\n\n"
        f"<evidence>\n{evidence.strip()}\n</evidence>"
    )


class WorkReviewer:
    """Runs the post-work review agent and returns its assessment."""

    def __init__(
        self,
        agent: Any = None,  # noqa: ANN401  # compiled graph
        *,
        factory: Callable[[], Any] | None = None,
    ) -> None:
        """Wrap a review agent, or a factory that builds one on first use.

        Args:
            agent: Agent returning a `WorkReview` as its structured response.
            factory: Builds the agent on the first review when `agent` is not
                given.
        """
        self._agent = agent
        self._factory = factory

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
        """Assess finished work against its plan.

        Args:
            request_text: The request the work answers.
            plan_text: Approved plan content.
            evidence: Harness findings from `evidence_text`.
            project_root: Absolute path of the project root on the host.
            earlier_requests: Requests of earlier turns, given as context
                only.
            context: Runtime context of the calling agent, passed through so
                the reviewer uses the same model selection and credentials.

        Returns:
            The assessment, or `None` when the reviewer failed or returned
                something that is not an assessment.
        """
        message = HumanMessage(
            review_prompt(
                request_text, plan_text, evidence, project_root, earlier_requests
            )
        )
        for _ in range(REVIEW_ATTEMPTS):
            try:
                if self._agent is None and self._factory is not None:
                    self._agent = self._factory()
                if self._agent is None:
                    return None
                result = await self._agent.ainvoke(
                    {"messages": [message]}, context=context
                )
            except Exception:
                logger.warning("Post-work reviewer failed", exc_info=True)
                continue
            review = read_structured(result, WorkReview)
            if review is not None:
                return review
        return None


def create_work_reviewer(
    *,
    model: str | BaseChatModel,
    root: Path,
    model_retries: int,
    cli_max_retries: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> WorkReviewer:
    """Build the post-work reviewer for a project.

    Args:
        model: Chat model or model identifier of the main agent.
        root: Project root the reviewer may read.
        model_retries: Model-node retry attempts after the first call.
        cli_max_retries: Explicit `--max-retries` value for runtime switches.
        environ: Workspace environment retained for runtime model switches.

    Returns:
        Reviewer bound to the project. Its agent is built on the first review.
    """
    from deepagents_code.harness.plan_review import build_review_agent

    def build() -> Any:  # noqa: ANN401  # compiled graph
        """Build the read-only agent that reviews the finished work.

        Returns:
            The compiled review agent.
        """
        return build_review_agent(
            system_prompt=WORK_REVIEW_SYSTEM_PROMPT,
            schema=WorkReview,
            name="plan_work_reviewer",
            run_name="Deep Agents Code post-work review",
            model=model,
            root=root,
            model_retries=model_retries,
            cli_max_retries=cli_max_retries,
            environ=environ,
        )

    return WorkReviewer(factory=build)
