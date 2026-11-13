"""Self-improvement: turn a failed piece of work into a checked change.

When a turn ends with failure signals, an improver agent proposes one change to
what later turns are told: a project rule, a prompt addendum, or a skill. The
change is not trusted. It is applied only after two checks:

1. Static checks that need no model: size, format, and a target inside the
   project's `.deepagents/` folder.
2. A before/after comparison. For a few stored cases the model is asked what
   plan it would make, once with the current project instructions and once
   with the change added, and a judge marks which expectations each plan
   meets. The change is applied only when the case it came from scores higher
   and no case scores lower.

The case for a failed turn is written by a separate examiner that sees the
failure but not the proposed change, and the improver does not see the case.
Otherwise the expectations would restate the change and the comparison would
only show that the model follows instructions.

The comparison is about plans, not about executed work: it can show that the
change makes the agent plan better, not that it makes the agent code better.

Every decision, applied or held, is appended to a history file. An applied
change can be taken back:

    python -m deepagents_code.harness.improvement list
    python -m deepagents_code.harness.improvement revert <id>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import re
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
    Protocol,
    override,
)

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    TracePolicy,
    omit_payload,
)
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from deepagents_code.harness import experience, plan as plan_rules
from deepagents_code.harness.plan_review import read_structured

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from langchain_core.language_models import BaseChatModel
    from langgraph.runtime import Runtime

    from deepagents_code.harness.plan_report import Comparison, WorkReview

logger = logging.getLogger(__name__)

HARNESS_DIRECTORY = Path(".deepagents")
"""Project folder that holds everything the harness may change on its own."""

IMPROVEMENT_DIRECTORY = HARNESS_DIRECTORY / "improvements"
RULES_FILE = HARNESS_DIRECTORY / "AGENTS.md"
"""Project rules. dcode loads this file into the prompt as project memory."""

ADDENDUM_FILE = HARNESS_DIRECTORY / "prompt_addendum.md"
"""Text appended to the system prompt by `PromptAddendumMiddleware`."""

SKILLS_DIRECTORY = HARNESS_DIRECTORY / "skills"
"""Project skills. dcode discovers `<name>/SKILL.md` files here."""

MAX_EVAL_CASES = 3
"""Cases compared for one proposal: the new one and the most recent others."""

MAX_EXPECTATIONS = 5
"""Expectations one case may hold."""

REFUSAL_SIGNAL = 3
"""Plan gate refusals in one turn that count as a failure signal."""

TOOL_ERROR_SIGNAL = 3
"""Other failed tool calls in one turn that count as a failure signal."""

GATE_REFUSAL_PREFIX = "[계획 관문]"

CONTENT_LIMITS = {"rule": 600, "prompt": 1500, "skill": 6000}
"""Longest content, in characters, a proposal may carry per target."""

FILE_LIMITS = {"rule": 20_000, "prompt": 8_000}
"""Size, in characters, past which an append-only target takes no more."""

_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SKILL_NAME_LIMIT = 64
_SKILL_DESCRIPTION_LIMIT = 1024
_CONFIG_SKILL_CHARS = 1500
_RULES_HEADING = "## 자기 개선으로 추가된 규칙"
_ITEM = re.compile(r"^\s*[-*+]\s+(?P<body>\S.*)$")

Target = Literal["none", "rule", "prompt", "skill"]

TARGET_LABELS = {
    "rule": "작업 메모리 (프로젝트 규칙)",
    "prompt": "시스템 프롬프트 (덧붙임)",
    "skill": "Skill",
}

IMPROVER_SYSTEM_PROMPT = """\
You improve how a coding agent works in one project, based on a piece of work
that went wrong.

You are given the failure signals of that work, the project's current
instructions, and records of recent work. Propose at most one change that
would have prevented the failure, choosing the narrowest target that fits:

- rule: one short project rule, as a single sentence or two. For a fact or
  convention of this project the agent must respect every time.
- prompt: a short paragraph on how to work, appended to the system prompt. For
  working habits that are not specific to one file or feature.
- skill: a named procedure with steps, loaded only when relevant. For a
  multi-step task the agent will repeat. Give `name` (lowercase letters,
  digits and hyphens), `description` (when to use it), and the steps as
  `content`.
- none: when the failure is a one-off, when the current instructions already
  cover it, or when no instruction would have helped.

Write the content in the language of the project instructions. Do not repeat
what the current instructions already say. Do not weaken a safety rule, and do
not tell the agent to skip planning, review, approval or tests.

Put the text to add in `content`, complete and ready to use as written. The
`reason` explains the change; it is not a substitute for `content`. A proposal
with an empty `content` is discarded.
"""

EXAMINER_SYSTEM_PROMPT = """\
You write one evaluation case from a piece of work by a coding agent that went
wrong. The case will be used to check whether a later change to the agent's
instructions helps: the agent is asked to plan for the case's request, and the
plan is checked against the case's expectations.

You are given the request the agent worked on and the failure signals of that
work. You are not told what change will be tried, and you must not guess it.

- `input`: a request a user could make in this project that leads into the
  same situation. Make it a different task of the same kind, not a copy of the
  original request.
- `expectations`: one to five things a plan for that request must contain to
  avoid this failure. Describe what the plan must do or state, as checkable
  sentences. Do not prescribe wording, and do not name a rule or instruction.

The case must be decidable from a plan alone, without running anything. Write
it in the language of the request.
"""

PLANNER_SYSTEM_PROMPT = """\
You are a coding agent working in a project. You are given the project's
instructions and a user request. Write the plan you would follow for the
request: what you would check first, which files you would change, the steps
in order, and how you would verify the result. You cannot use tools here; do
not ask questions. Answer with the plan only.
"""

JUDGE_SYSTEM_PROMPT = """\
You compare two plans for the same request against a list of expectations.

For each expectation, decide separately for the first plan and for the second
plan whether the plan clearly meets it. A plan meets an expectation only when
it says so itself; do not give credit for what a plan might imply. Judge each
plan on its own: the plans may be equally good or equally bad.

Return two lists of booleans, `first` and `second`, each with one entry per
expectation in the order given.
"""


class Improvement(BaseModel):
    """One proposed change to what later turns are told."""

    target: Target = Field(description="What to change, or `none`.")
    name: str = Field(default="", description="Skill name. Only for `skill`.")
    description: str = Field(
        default="", description="When to use the skill. Only for `skill`."
    )
    content: str = Field(
        description=(
            "The exact text to add: the rule sentence, the prompt paragraph, or "
            "the skill's steps. Required for every target except `none`, where "
            "it is an empty string."
        )
    )
    reason: str = Field(description="Why this change would have prevented the failure.")


class CaseDraft(BaseModel):
    """An evaluation case written from a failure, without sight of the change."""

    input: str = Field(description="A user request that leads into the same situation.")
    expectations: list[str] = Field(
        max_length=MAX_EXPECTATIONS,
        description="One to five things a plan for that request must contain.",
    )


class Judgement(BaseModel):
    """Which expectations each of two plans meets."""

    first: list[bool] = Field(description="One entry per expectation, first plan.")
    second: list[bool] = Field(description="One entry per expectation, second plan.")


@dataclass(frozen=True)
class Signals:
    """Signs that a turn's work went wrong.

    Attributes:
        missing: Plan steps the post-work review found no work for.
        unplanned: Changes the review found that the plan does not describe.
        unplanned_paths: Changed files outside the plan's "수정 대상".
        outcome: Review outcome, a key of `experience.OUTCOME_LABELS`.
        refusals: Tool calls the plan gate refused in the turn.
        tool_errors: Other tool calls that failed in the turn.
    """

    missing: tuple[str, ...] = ()
    unplanned: tuple[str, ...] = ()
    unplanned_paths: tuple[str, ...] = ()
    outcome: str = "unknown"
    refusals: int = 0
    tool_errors: int = 0

    @property
    def failed(self) -> bool:
        """Whether the turn gives a reason to look for an improvement.

        Extra work the review found but the plan did not describe
        (`unplanned`) is a minor signal. In a turn the review judged a
        success it is left out: an improvement learned from it would teach
        the agent to do less, such as asking before adding a useful test.
        Every other signal, and `unplanned` in any other outcome, still
        counts. Files outside the plan's "수정 대상" (`unplanned_paths`) are
        checked by the harness itself and always count.
        """
        checked = bool(
            self.missing
            or self.unplanned_paths
            or self.outcome == "failure"
            or self.refusals >= REFUSAL_SIGNAL
            or self.tool_errors >= TOOL_ERROR_SIGNAL
        )
        return checked or (bool(self.unplanned) and self.outcome != "success")

    def describe(self) -> str:
        """Format the signals for the improver and the history file.

        Returns:
            One line per kind of signal that is present.
        """
        lines: list[str] = []
        lines += [f"- 빠진 단계: {item}" for item in self.missing]
        lines += [f"- 계획에 없는 변경: {item}" for item in self.unplanned]
        lines += [f"- 수정 대상 밖의 파일: `{item}`" for item in self.unplanned_paths]
        if self.outcome == "failure":
            lines.append("- 작업 후 리뷰의 결과: 실패")
        if self.refusals:
            lines.append(f"- 계획 관문이 거부한 호출: {self.refusals}회")
        if self.tool_errors:
            lines.append(f"- 실패한 도구 호출: {self.tool_errors}회")
        return "\n".join(lines) or "- 없음"


def collect_signals(
    messages: Sequence[Any], review: WorkReview | None, comparison: Comparison
) -> Signals:
    """Gather the failure signals of the current turn.

    Args:
        messages: Conversation messages in order.
        review: Post-work review of the turn, if there is one.
        comparison: Harness comparison of the turn's changes with the plan.

    Returns:
        The turn's signals. Model retries are not among them: they are not
            visible in the conversation.
    """
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index + 1
    refusals = 0
    tool_errors = 0
    for message in messages[start:]:
        if not isinstance(message, ToolMessage) or message.status != "error":
            continue
        if str(message.text).startswith(GATE_REFUSAL_PREFIX):
            refusals += 1
        else:
            tool_errors += 1
    return Signals(
        missing=tuple(review.missing) if review else (),
        unplanned=tuple(review.unplanned) if review else (),
        unplanned_paths=comparison.unplanned,
        outcome=review.outcome if review else "unknown",
        refusals=refusals,
        tool_errors=tool_errors,
    )


# ----- evaluation cases -----------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    """A situation a good plan must handle.

    Attributes:
        id: Case id; also the file name without its extension.
        input: The user request.
        expectations: What a good plan for the request must contain.
        source: Where the case came from.
    """

    id: str
    input: str
    expectations: tuple[str, ...]
    source: str


def _one_line(text: str) -> str:
    """Collapse all whitespace runs so the text fits on one line.

    Args:
        text: Text that may span several lines.

    Returns:
        The text with every whitespace run replaced by one space.
    """
    return " ".join(text.split())


def render_case(case: EvalCase) -> str:
    """Format a case as the Markdown stored on disk.

    Args:
        case: Case to format.

    Returns:
        File content with one level-two section per field.
    """
    expectations = "\n".join(f"- {_one_line(item)}" for item in case.expectations)
    return (
        f"# 평가 사례: {case.id}\n\n"
        f"## 입력\n\n{_one_line(case.input)}\n\n"
        f"## 기대 항목\n\n{expectations}\n\n"
        f"## 출처\n\n{_one_line(case.source)}\n"
    )


def parse_case(case_id: str, text: str) -> EvalCase | None:
    """Read a case file.

    Args:
        case_id: Case id, taken from the file name.
        text: File content.

    Returns:
        The case, or `None` when it has no input or no expectation.
    """
    sections = plan_rules.parse_plan(text).sections
    expectations = tuple(
        match.group("body").strip()
        for line in sections.get("기대 항목", "").splitlines()
        if (match := _ITEM.match(line)) is not None
    )
    request = sections.get("입력", "").strip()
    if not request or not expectations:
        return None
    return EvalCase(
        id=case_id,
        input=request,
        expectations=expectations,
        source=sections.get("출처", "").strip(),
    )


def cases_directory(root: Path) -> Path:
    """Return the folder holding a project's evaluation cases.

    Args:
        root: Project root.

    Returns:
        Absolute folder path. The folder may not exist yet.
    """
    return root / IMPROVEMENT_DIRECTORY / "cases"


def save_case(root: Path, case: EvalCase) -> Path:
    """Write a case to the project's case folder.

    Args:
        root: Project root.
        case: Case to write.

    Returns:
        Path of the written file.
    """
    folder = cases_directory(root)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{case.id}.md"
    path.write_text(render_case(case), encoding="utf-8")
    return path


def load_cases(root: Path, limit: int) -> list[EvalCase]:
    """Read the project's most recent cases.

    Args:
        root: Project root.
        limit: Most cases to return.

    Returns:
        Cases, newest first. Unreadable files are skipped.
    """
    if limit <= 0:
        return []
    try:
        paths = sorted(cases_directory(root).glob("*.md"), reverse=True)
    except OSError:
        return []
    cases: list[EvalCase] = []
    for path in paths:
        text = plan_rules.read_text(path)
        case = parse_case(path.stem, text) if text is not None else None
        if case is not None:
            cases.append(case)
        if len(cases) == limit:
            break
    return cases


# ----- the project's instructions and a change to them ----------------------


@dataclass(frozen=True)
class ProjectConfig:
    """The instructions a project adds to the agent.

    Attributes:
        rules: Content of the project rules file.
        addendum: Content of the prompt addendum file.
        skills: Content of each project skill file, keyed by skill name.
    """

    rules: str = ""
    addendum: str = ""
    skills: tuple[tuple[str, str], ...] = ()


def load_config(root: Path) -> ProjectConfig:
    """Read the project's current instructions.

    Args:
        root: Project root.

    Returns:
        The instructions; missing files read as empty.
    """
    skills: list[tuple[str, str]] = []
    try:
        folders = sorted((root / SKILLS_DIRECTORY).iterdir())
    except OSError:
        folders = []
    for folder in folders:
        text = plan_rules.read_text(folder / "SKILL.md")
        if text is not None:
            skills.append((folder.name, text))
    return ProjectConfig(
        rules=plan_rules.read_text(root / RULES_FILE) or "",
        addendum=plan_rules.read_text(root / ADDENDUM_FILE) or "",
        skills=tuple(skills),
    )


def config_text(config: ProjectConfig) -> str:
    """Format project instructions as one text for a model.

    Args:
        config: Project instructions.

    Returns:
        The rules, the addendum and each skill under their own headings, with
            long skills cut.
    """
    skills = "\n\n".join(
        f"### {name}\n{text.strip()[:_CONFIG_SKILL_CHARS]}"
        for name, text in config.skills
    )
    return (
        f"## Project rules\n{config.rules.strip() or '(none)'}\n\n"
        f"## Working notes\n{config.addendum.strip() or '(none)'}\n\n"
        f"## Skills\n{skills or '(none)'}"
    )


def skill_text(improvement: Improvement) -> str:
    """Build the `SKILL.md` content of a skill proposal.

    The harness writes the front matter itself, so a proposal cannot produce a
    malformed header.

    Args:
        improvement: A `skill` proposal.

    Returns:
        File content: front matter with name and description, then the steps.
    """
    return (
        "---\n"
        f"name: {improvement.name.strip()}\n"
        f"description: {_one_line(improvement.description)}\n"
        "---\n\n"
        f"{improvement.content.strip()}\n"
    )


def _appended(existing: str, addition: str, heading: str = "") -> str:
    """Return a file's content with a paragraph added at its end.

    Args:
        existing: Current content.
        addition: Text to add.
        heading: Heading to place before the first addition, if not present.

    Returns:
        The new content.
    """
    body = existing.rstrip()
    if heading and heading not in body:
        body = f"{body}\n\n{heading}" if body else heading
    return f"{body}\n\n{addition.strip()}\n" if body else f"{addition.strip()}\n"


def with_change(config: ProjectConfig, improvement: Improvement) -> ProjectConfig:
    """Return the instructions as they would be after a change.

    Args:
        config: Current instructions.
        improvement: Proposed change.

    Returns:
        The changed instructions. A `none` proposal changes nothing.
    """
    if improvement.target == "rule":
        rule = f"- {_one_line(improvement.content)}"
        return replace(config, rules=_appended(config.rules, rule, _RULES_HEADING))
    if improvement.target == "prompt":
        return replace(config, addendum=_appended(config.addendum, improvement.content))
    if improvement.target == "skill":
        name = improvement.name.strip()
        others = tuple(item for item in config.skills if item[0] != name)
        return replace(config, skills=(*others, (name, skill_text(improvement))))
    return config


def target_path(root: Path, improvement: Improvement) -> Path | None:
    """Return the file a proposal would change.

    Args:
        root: Project root.
        improvement: Proposed change.

    Returns:
        The target file, or `None` for a `none` proposal.
    """
    if improvement.target == "rule":
        return root / RULES_FILE
    if improvement.target == "prompt":
        return root / ADDENDUM_FILE
    if improvement.target == "skill":
        return root / SKILLS_DIRECTORY / improvement.name.strip() / "SKILL.md"
    return None


def static_problems(
    root: Path,
    improvement: Improvement,
    config: ProjectConfig,
    case: CaseDraft | None,
) -> list[str]:
    """Check a proposal and its evaluation case without calling a model.

    Args:
        root: Project root.
        improvement: Proposed change.
        config: Current project instructions.
        case: Evaluation case written for the failure, if the examiner
            produced one.

    Returns:
        Reasons to hold the proposal. Empty when it may go on to evaluation.
    """
    target = improvement.target
    if target == "none":
        return ["개선 대상이 없습니다."]
    problems: list[str] = []
    content = improvement.content.strip()
    if not content:
        problems.append("내용이 비어 있습니다.")
    limit = CONTENT_LIMITS[target]
    if len(content) > limit:
        problems.append(f"내용이 {len(content)}자로 상한 {limit}자를 넘습니다.")
    if target == "skill":
        name = improvement.name.strip()
        if not _SKILL_NAME.match(name) or len(name) > _SKILL_NAME_LIMIT:
            problems.append(
                f"Skill 이름 `{name}`이 형식(소문자·숫자·하이픈, "
                f"{_SKILL_NAME_LIMIT}자 이하)에 맞지 않습니다."
            )
        description = improvement.description.strip()
        if not description or len(description) > _SKILL_DESCRIPTION_LIMIT:
            problems.append(
                f"Skill 설명이 없거나 {_SKILL_DESCRIPTION_LIMIT}자를 넘습니다."
            )
    elif content and content in (config.rules if target == "rule" else config.addendum):
        problems.append("같은 내용이 이미 대상 파일에 있습니다.")
    else:
        current = config.rules if target == "rule" else config.addendum
        if len(current) + len(content) > FILE_LIMITS[target]:
            problems.append(
                f"대상 파일이 상한 {FILE_LIMITS[target]}자에 닿아 더 넣지 않습니다."
            )
    path = target_path(root, improvement)
    scope = (root / HARNESS_DIRECTORY).resolve()
    if path is None or not plan_rules.is_inside(path.resolve(), scope):
        problems.append("대상 경로가 프로젝트의 `.deepagents/` 밖입니다.")
    if (
        case is None
        or not case.input.strip()
        or not any(item.strip() for item in case.expectations)
    ):
        problems.append("평가 사례(입력과 기대 항목)가 없습니다.")
    return problems


def apply(root: Path, improvement: Improvement) -> Path:
    """Write a proposal to its target file.

    Args:
        root: Project root.
        improvement: Proposal that passed `static_problems`.

    Returns:
        The file that was written.

    Raises:
        ValueError: If the proposal has no target.
    """
    path = target_path(root, improvement)
    if path is None:
        msg = "A proposal without a target cannot be applied"
        raise ValueError(msg)
    changed = with_change(load_config(root), improvement)
    if improvement.target == "rule":
        text = changed.rules
    elif improvement.target == "prompt":
        text = changed.addendum
    else:
        text = skill_text(improvement)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ----- decision -------------------------------------------------------------


@dataclass(frozen=True)
class CaseScore:
    """Share of a case's expectations met before and after a change.

    Attributes:
        case_id: Case the scores belong to.
        before: Share met by the plan made with the current instructions.
        after: Share met by the plan made with the change added.
    """

    case_id: str
    before: float
    after: float


def decide(source_case_id: str, scores: Sequence[CaseScore]) -> tuple[bool, str]:
    """Decide whether a change is applied.

    Args:
        source_case_id: Case the change was proposed for.
        scores: Scores of every compared case.

    Returns:
        Whether to apply, and the reason in words. A change is applied only
            when its own case scores higher and no case scores lower.
    """
    source = next((s for s in scores if s.case_id == source_case_id), None)
    if source is None:
        return False, "근거 사례의 점수가 없습니다."
    worse = [s.case_id for s in scores if s.after < s.before]
    if worse:
        return False, "점수가 떨어진 사례가 있습니다: " + ", ".join(worse)
    if source.after <= source.before:
        return False, (
            f"근거 사례의 점수가 오르지 않았습니다"
            f"({source.before:.2f} → {source.after:.2f})."
        )
    return True, (
        f"근거 사례의 점수가 올랐고({source.before:.2f} → {source.after:.2f}) "
        "떨어진 사례가 없습니다."
    )


def judge_prompt(case: EvalCase, first: str, second: str) -> str:
    """Build the judge's input message.

    Args:
        case: Case whose expectations are checked.
        first: Plan shown first.
        second: Plan shown second.

    Returns:
        Message text with the request, expectations and both plans.
    """
    expectations = "\n".join(
        f"{index}. {item}" for index, item in enumerate(case.expectations, start=1)
    )
    return (
        f"<request>\n{case.input}\n</request>\n\n"
        f"<expectations>\n{expectations}\n</expectations>\n\n"
        f"<first_plan>\n{first.strip()}\n</first_plan>\n\n"
        f"<second_plan>\n{second.strip()}\n</second_plan>"
    )


def case_prompt(signals: Signals, request_text: str) -> str:
    """Build the examiner's input message.

    It holds the request and the failure signals only. The current
    instructions and the proposed change are left out on purpose.

    Args:
        signals: Failure signals of the turn.
        request_text: The request the turn answered.

    Returns:
        Message text with the request and the signals in tagged blocks.
    """
    return (
        f"<request>\n{request_text.strip()}\n</request>\n\n"
        f"<failure_signals>\n{signals.describe()}\n</failure_signals>"
    )


def changed_plan_is_first(case_id: str) -> bool:
    """Decide the order in which a case's two plans are shown to the judge.

    The order depends only on the case id, so it is repeatable, and it differs
    between cases, so the judge cannot learn which position holds the change.

    Args:
        case_id: Case id.

    Returns:
        `True` when the plan made with the change is shown first.
    """
    return hashlib.sha256(case_id.encode("utf-8")).digest()[0] % 2 == 0


class ImprovementModels(Protocol):
    """The four model roles self-improvement needs."""

    async def propose(self, prompt: str, context: object | None) -> Improvement | None:
        """Propose a change for a failed turn."""
        ...

    async def examine(self, prompt: str, context: object | None) -> CaseDraft | None:
        """Write an evaluation case for a failed turn, without seeing the change."""
        ...

    async def plan(
        self, instructions: str, request: str, context: object | None
    ) -> str | None:
        """Write a plan for a request under the given project instructions."""
        ...

    async def judge(self, prompt: str, context: object | None) -> Judgement | None:
        """Mark which expectations two plans meet."""
        ...


@dataclass(frozen=True)
class Outcome:
    """Result of one self-improvement run.

    Attributes:
        id: Id shared by the proposal, its case and its history entry.
        applied: Whether the change was written to its target.
        reason: Why it was applied or held.
        scores: Scores of the compared cases; empty when evaluation did not
            run.
    """

    id: str
    applied: bool
    reason: str
    scores: tuple[CaseScore, ...] = ()


def render_proposal(
    proposal_id: str, improvement: Improvement, signals: Signals
) -> str:
    """Format a proposal as the file kept in the pending or applied folder.

    Args:
        proposal_id: Proposal id.
        improvement: Proposed change.
        signals: Failure signals that led to it.

    Returns:
        Markdown content.
    """
    label = TARGET_LABELS.get(improvement.target, improvement.target)
    name = f" `{improvement.name.strip()}`" if improvement.target == "skill" else ""
    return (
        f"# 개선안: {proposal_id}\n\n"
        f"## 대상\n\n{label}{name}\n\n"
        f"## 바꿀 내용\n\n{improvement.content.strip() or '없음'}\n\n"
        f"## 이유\n\n{_one_line(improvement.reason) or '없음'}\n\n"
        f"## 근거가 된 실패\n\n{signals.describe()}\n"
    )


def render_history(
    outcome: Outcome,
    improvement: Improvement,
    root: Path,
    thread_id: str,
) -> str:
    """Format one history entry.

    Args:
        outcome: Decision and scores.
        improvement: Proposed change.
        root: Project root, used to show the target path relative to it.
        thread_id: Thread whose turn led to the proposal.

    Returns:
        Markdown block to append to the history file.
    """
    path = target_path(root, improvement)
    target = path.relative_to(root).as_posix() if path is not None else "없음"
    digest = hashlib.sha256(improvement.content.encode("utf-8")).hexdigest()
    rows = "\n".join(
        f"| `{score.case_id}` | {score.before:.2f} | {score.after:.2f} |"
        for score in outcome.scores
    )
    table = (
        f"| 사례 | 개선 전 | 개선 후 |\n|---|---|---|\n{rows}\n"
        if rows
        else "평가를 실행하지 않았습니다.\n"
    )
    return (
        f"## {outcome.id}\n\n"
        f"- 시각: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        f"- thread: {thread_id}\n"
        f"- 대상: {TARGET_LABELS.get(improvement.target, improvement.target)} "
        f"(`{target}`)\n"
        f"- 결정: {'반영' if outcome.applied else '보류'}\n"
        f"- 사유: {outcome.reason}\n"
        f"- 내용 해시: `{digest}`\n\n"
        f"{table}\n"
    )


def _append(path: Path, text: str, title: str) -> None:
    """Append the text to the file, writing the title heading when creating it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "" if path.exists() else f"# {title}\n\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(header + text)


class SelfImprover:
    """Propose, check and apply one improvement after a failed turn."""

    def __init__(
        self,
        root: Path,
        models: ImprovementModels,
        *,
        max_cases: int = MAX_EVAL_CASES,
    ) -> None:
        """Create the improver for one project.

        Args:
            root: Project root.
            models: The model roles to use.
            max_cases: Most cases compared for one proposal.
        """
        self._root = root
        self._models = models
        self._max_cases = max_cases

    @property
    def directory(self) -> Path:
        """Folder holding proposals, cases and the history file."""
        return self._root / IMPROVEMENT_DIRECTORY

    def _prompt(self, signals: Signals, request_text: str) -> str:
        """Build the improver's input. Blocking: reads project files.

        Args:
            signals: Failure signals of the turn.
            request_text: The request the turn answered.

        Returns:
            Message text with the request, signals, instructions and records.
        """
        records = "\n\n".join(
            experience.render(record) for record in experience.load_all(self._root)[:3]
        )
        return (
            f"<request>\n{request_text.strip()}\n</request>\n\n"
            f"<failure_signals>\n{signals.describe()}\n</failure_signals>\n\n"
            "<current_instructions>\n"
            f"{config_text(load_config(self._root))}\n"
            "</current_instructions>\n\n"
            f"<recent_work>\n{records or '(none)'}\n</recent_work>"
        )

    def _record(
        self,
        outcome: Outcome,
        improvement: Improvement,
        signals: Signals,
        thread_id: str,
    ) -> Outcome:
        """Write the proposal file and the history entry. Blocking.

        An applied proposal is filed under `applied/`, a held one stays under
        `pending/` where a person can read it.

        Args:
            outcome: Decision and scores.
            improvement: Proposed change.
            signals: Failure signals of the turn.
            thread_id: Thread the turn belongs to.

        Returns:
            The outcome, unchanged.
        """
        folder = self.directory / ("applied" if outcome.applied else "pending")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{outcome.id}.md").write_text(
            render_proposal(outcome.id, improvement, signals), encoding="utf-8"
        )
        if outcome.applied:
            # Exact content, kept for `revert`; the Markdown file is for people.
            (folder / f"{outcome.id}.json").write_text(
                improvement.model_dump_json(indent=2), encoding="utf-8"
            )
        _append(
            self.directory / "history.md",
            render_history(outcome, improvement, self._root, thread_id),
            "자기 개선 이력",
        )
        return outcome

    async def _score(
        self,
        case: EvalCase,
        before: ProjectConfig,
        after: ProjectConfig,
        context: object | None,
    ) -> CaseScore | None:
        """Compare the plans made for one case before and after a change.

        Args:
            case: Case to evaluate.
            before: Current project instructions.
            after: Instructions with the change added.
            context: Runtime context of the calling agent.

        Returns:
            The case's scores, or `None` when a plan or the judgement is
                missing or malformed.
        """
        plan_before, plan_after = await asyncio.gather(
            self._models.plan(config_text(before), case.input, context),
            self._models.plan(config_text(after), case.input, context),
        )
        if not plan_before or not plan_after:
            return None
        flipped = changed_plan_is_first(case.id)
        first, second = (
            (plan_after, plan_before) if flipped else (plan_before, plan_after)
        )
        judgement = await self._models.judge(judge_prompt(case, first, second), context)
        count = len(case.expectations)
        if (
            judgement is None
            or len(judgement.first) != count
            or len(judgement.second) != count
        ):
            return None
        first_score = sum(judgement.first) / count
        second_score = sum(judgement.second) / count
        return CaseScore(
            case_id=case.id,
            before=second_score if flipped else first_score,
            after=first_score if flipped else second_score,
        )

    async def run(
        self,
        *,
        signals: Signals,
        request_text: str,
        thread_id: str,
        context: object | None = None,
    ) -> Outcome | None:
        """Look for an improvement after a turn and apply it if it checks out.

        Args:
            signals: Failure signals of the turn.
            request_text: The request the turn answered.
            thread_id: Thread the turn belongs to.
            context: Runtime context of the calling agent, passed through so
                the helper models use the same model selection.

        Returns:
            The decision, or `None` when the turn showed no failure or the
                improver proposed nothing.
        """
        if not signals.failed:
            return None
        prompt = await asyncio.to_thread(self._prompt, signals, request_text)
        # The improver and the examiner run side by side and never see each
        # other's output: the case must not be written to fit the change, nor
        # the change to fit the case.
        improvement, draft = await asyncio.gather(
            self._models.propose(prompt, context),
            self._models.examine(case_prompt(signals, request_text), context),
        )
        if improvement is None or improvement.target == "none":
            return None
        proposal_id = experience.new_id(thread_id)
        config = await asyncio.to_thread(load_config, self._root)
        problems = await asyncio.to_thread(
            static_problems, self._root, improvement, config, draft
        )
        if problems:
            held = Outcome(proposal_id, applied=False, reason=" ".join(problems))
            return await asyncio.to_thread(
                self._record, held, improvement, signals, thread_id
            )

        if draft is None:  # Unreachable: a missing case is a static problem.
            return None
        case = EvalCase(
            id=proposal_id,
            input=_one_line(draft.input),
            expectations=tuple(
                _one_line(item) for item in draft.expectations if item.strip()
            ),
            source=f"thread {thread_id}의 작업 후 리뷰",
        )
        earlier = await asyncio.to_thread(load_cases, self._root, self._max_cases - 1)
        await asyncio.to_thread(save_case, self._root, case)
        changed = with_change(config, improvement)
        # The cases are independent, so they are compared at the same time.
        # Run one after another, three cases kept the user waiting for minutes
        # after the work itself was done.
        cases = (case, *earlier)
        results = await asyncio.gather(
            *(self._score(item, config, changed, context) for item in cases)
        )
        scores = [score for score in results if score is not None]
        unread = [
            item.id for item, score in zip(cases, results, strict=True) if score is None
        ]
        if unread:
            held = Outcome(
                proposal_id,
                applied=False,
                reason=f"사례 `{unread[0]}`의 평가 결과를 해석하지 못했습니다.",
                scores=tuple(scores),
            )
            return await asyncio.to_thread(
                self._record, held, improvement, signals, thread_id
            )

        applied, reason = decide(case.id, scores)
        if applied:
            await asyncio.to_thread(apply, self._root, improvement)
        outcome = Outcome(
            proposal_id, applied=applied, reason=reason, scores=tuple(scores)
        )
        return await asyncio.to_thread(
            self._record, outcome, improvement, signals, thread_id
        )


def applied_improvements(root: Path) -> list[tuple[str, Improvement]]:
    """List the improvements that were applied and can be taken back.

    Args:
        root: Project root.

    Returns:
        Proposal ids with their content, newest first. Entries whose stored
            content cannot be read are left out.
    """
    folder = root / IMPROVEMENT_DIRECTORY / "applied"
    try:
        paths = sorted(folder.glob("*.json"), reverse=True)
    except OSError:
        return []
    found: list[tuple[str, Improvement]] = []
    for path in paths:
        text = plan_rules.read_text(path)
        if text is None:
            continue
        try:
            found.append((path.stem, Improvement.model_validate_json(text)))
        except ValueError:
            continue
    return found


def without_change(
    config: ProjectConfig, improvement: Improvement
) -> ProjectConfig | None:
    """Return the instructions with an applied change taken out.

    Args:
        config: Current instructions.
        improvement: The change as it was applied.

    Returns:
        The instructions without the change, or `None` when the change is no
            longer there in the form it was written.
    """
    if improvement.target == "rule":
        line = f"- {_one_line(improvement.content)}"
        lines = config.rules.splitlines()
        if line not in lines:
            return None
        lines.remove(line)
        return replace(config, rules="\n".join(lines).rstrip() + "\n")
    if improvement.target == "prompt":
        paragraph = improvement.content.strip()
        if paragraph not in config.addendum:
            return None
        rest = config.addendum.replace(paragraph, "", 1)
        rest = re.sub(r"\n{3,}", "\n\n", rest).strip()
        return replace(config, addendum=f"{rest}\n" if rest else "")
    if improvement.target == "skill":
        name = improvement.name.strip()
        current = dict(config.skills).get(name)
        if current is None or current != skill_text(improvement):
            return None
        return replace(
            config, skills=tuple(item for item in config.skills if item[0] != name)
        )
    return None


def revert(root: Path, proposal_id: str) -> tuple[bool, str]:
    """Take an applied improvement back out of the project's instructions.

    The change is removed only when it is still present exactly as it was
    written; a target that was edited since is left alone. The proposal file
    moves to `reverted/` and the history gets an entry.

    Args:
        root: Project root.
        proposal_id: Id of the applied proposal.

    Returns:
        Whether the change was removed, and a message for the user.
    """
    applied = dict(applied_improvements(root))
    improvement = applied.get(proposal_id)
    if improvement is None:
        return False, f"반영된 개선안 `{proposal_id}`를 찾지 못했습니다."
    path = target_path(root, improvement)
    remaining = without_change(load_config(root), improvement)
    if path is None or remaining is None:
        return False, (
            f"`{proposal_id}`의 내용이 대상 파일에 반영된 형태 그대로 남아 있지 "
            "않아 되돌리지 않았습니다. 대상 파일을 직접 확인하세요."
        )
    if improvement.target == "rule":
        path.write_text(remaining.rules, encoding="utf-8")
    elif improvement.target == "prompt":
        path.write_text(remaining.addendum, encoding="utf-8")
    else:
        path.unlink()
        if not any(path.parent.iterdir()):
            path.parent.rmdir()
    folder = root / IMPROVEMENT_DIRECTORY
    (folder / "reverted").mkdir(parents=True, exist_ok=True)
    for suffix in (".md", ".json"):
        source = folder / "applied" / f"{proposal_id}{suffix}"
        if source.exists():
            source.rename(folder / "reverted" / source.name)
    target = path.relative_to(root).as_posix()
    _append(
        folder / "history.md",
        f"## {proposal_id} 되돌림\n\n"
        f"- 시각: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        f"- 대상: {TARGET_LABELS.get(improvement.target, improvement.target)} "
        f"(`{target}`)\n"
        "- 결정: 되돌림\n"
        "- 사유: 사용자가 되돌리기 명령을 실행했습니다.\n\n",
        "자기 개선 이력",
    )
    return True, f"`{proposal_id}`를 `{target}`에서 되돌렸습니다."


def main(argv: Sequence[str] | None = None) -> int:
    """List the applied improvements of a project, or take one back.

    Args:
        argv: Command-line arguments; the process arguments when omitted.

    Returns:
        Process exit code: 0 on success, 1 when a revert was not carried out.
    """
    parser = argparse.ArgumentParser(
        prog="python -m deepagents_code.harness.improvement",
        description="Show or revert the self-improvements applied to a project.",
    )
    parser.add_argument(
        "--root", default=".", help="Project root (default: current directory)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List applied improvements.")
    reverting = commands.add_parser("revert", help="Take an applied improvement back.")
    reverting.add_argument("id", help="Id of the applied improvement.")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    out = sys.stdout
    if args.command == "list":
        applied = applied_improvements(root)
        if not applied:
            out.write("반영된 개선안이 없습니다.\n")
        for proposal_id, improvement in applied:
            label = TARGET_LABELS.get(improvement.target, improvement.target)
            summary = _one_line(improvement.content)[:100]
            out.write(f"{proposal_id}  [{label}]  {summary}\n")
        return 0
    done, message = revert(root, args.id)
    out.write(message + "\n")
    return 0 if done else 1


class AgentImprovementModels:
    """The four model roles, each a helper agent built on first use."""

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        root: Path,
        model_retries: int,
        cli_max_retries: int | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Remember how to build the helper agents.

        Args:
            model: Chat model or model identifier of the main agent.
            root: Project root the improver may read.
            model_retries: Model-node retry attempts after the first call.
            cli_max_retries: Explicit `--max-retries` value for runtime switches.
            environ: Workspace environment retained for runtime model switches.
        """
        self._settings: dict[str, Any] = {
            "model": model,
            "root": root,
            "model_retries": model_retries,
            "cli_max_retries": cli_max_retries,
            "environ": environ,
        }
        self._agents: dict[str, Any] = {}

    def _agent(
        self,
        role: str,
        system_prompt: str,
        schema: type[BaseModel] | None,
        *,
        read_tools: bool,
    ) -> Any:  # noqa: ANN401  # compiled graph
        """Return the agent for a role, building it on first use.

        Args:
            role: Role name; also names the agent in traces.
            system_prompt: Instructions for the role.
            schema: Structured response the role returns, if any.
            read_tools: Whether the role may read the project.

        Returns:
            The compiled agent.
        """
        if role not in self._agents:
            from deepagents_code.harness.plan_review import build_review_agent

            self._agents[role] = build_review_agent(
                system_prompt=system_prompt,
                schema=schema,
                name=f"improvement_{role}",
                run_name=f"Deep Agents Code self-improvement {role}",
                read_tools=read_tools,
                **self._settings,
            )
        return self._agents[role]

    async def _invoke(
        self,
        role: str,
        system_prompt: str,
        schema: type[BaseModel] | None,
        prompt: str,
        context: object | None,
        *,
        read_tools: bool = False,
    ) -> dict[str, Any] | None:
        """Run a role once.

        Args:
            role: Role name.
            system_prompt: Instructions for the role.
            schema: Structured response the role returns, if any.
            prompt: Input message.
            context: Runtime context of the calling agent.
            read_tools: Whether the role may read the project.

        Returns:
            The agent's result state, or `None` when the run failed.
        """
        try:
            agent = self._agent(role, system_prompt, schema, read_tools=read_tools)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(prompt)]}, context=context
            )
        except Exception:
            logger.warning("Self-improvement %s failed", role, exc_info=True)
            return None
        return result if isinstance(result, dict) else None

    async def propose(self, prompt: str, context: object | None) -> Improvement | None:
        """Propose a change for a failed turn.

        Args:
            prompt: Failure signals, instructions and recent work.
            context: Runtime context of the calling agent.

        Returns:
            The proposal, or `None` when the improver failed.
        """
        result = await self._invoke(
            "improver",
            IMPROVER_SYSTEM_PROMPT,
            Improvement,
            prompt,
            context,
            read_tools=True,
        )
        return read_structured(result, Improvement)

    async def examine(self, prompt: str, context: object | None) -> CaseDraft | None:
        """Write an evaluation case for a failed turn.

        Args:
            prompt: The request and the failure signals.
            context: Runtime context of the calling agent.

        Returns:
            The case, or `None` when the examiner failed.
        """
        result = await self._invoke(
            "examiner", EXAMINER_SYSTEM_PROMPT, CaseDraft, prompt, context
        )
        return read_structured(result, CaseDraft)

    async def plan(
        self, instructions: str, request: str, context: object | None
    ) -> str | None:
        """Write a plan for a request under the given project instructions.

        Args:
            instructions: Project instructions the plan must follow.
            request: The user request.
            context: Runtime context of the calling agent.

        Returns:
            The plan text, or `None` when the planner failed or said nothing.
        """
        prompt = (
            f"<project_instructions>\n{instructions}\n</project_instructions>\n\n"
            f"<request>\n{request}\n</request>"
        )
        result = await self._invoke(
            "planner", PLANNER_SYSTEM_PROMPT, None, prompt, context
        )
        messages = result.get("messages") if result else None
        if not messages:
            return None
        return str(messages[-1].text).strip() or None

    async def judge(self, prompt: str, context: object | None) -> Judgement | None:
        """Mark which expectations two plans meet.

        Args:
            prompt: Request, expectations and both plans.
            context: Runtime context of the calling agent.

        Returns:
            The judgement, or `None` when the judge failed.
        """
        result = await self._invoke(
            "judge", JUDGE_SYSTEM_PROMPT, Judgement, prompt, context
        )
        return read_structured(result, Judgement)


# ----- prompt addendum ------------------------------------------------------


class AddendumState(AgentState[Any]):
    """Per-thread addendum state. The channel is private to this graph."""

    _prompt_addendum: Annotated[NotRequired[str], PrivateStateAttr]
    """Addendum text read at the start of the current turn."""


class PromptAddendumMiddleware(AgentMiddleware[AddendumState, ContextT]):
    """Append the project's prompt addendum file to the system prompt.

    The packaged system prompt is never edited. Improvements to how the agent
    works go into a project file instead, which this middleware reads once per
    turn.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces; they carry the whole message history."""

    state_schema = AddendumState

    def __init__(self, root: Path) -> None:
        """Create the middleware for one project.

        Args:
            root: Project root whose addendum file is read.
        """
        super().__init__()
        self._root = root

    def _load(self) -> dict[str, Any]:
        """Read the addendum file. Blocking.

        Returns:
            State update holding the text to append; empty text when the file
                is missing or blank.
        """
        text = (plan_rules.read_text(self._root / ADDENDUM_FILE) or "").strip()
        text = text[: FILE_LIMITS["prompt"]]
        block = (
            "## Working notes for this project\n\n"
            "Added by the harness after earlier work in this project was "
            f"reviewed.\n\n{text}"
            if text
            else ""
        )
        return {"_prompt_addendum": block}

    @override
    def before_agent(
        self, state: AddendumState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Read the addendum once, at the start of the turn.

        Returns:
            State update holding the text to append.
        """
        return self._load()

    @override
    async def abefore_agent(
        self, state: AddendumState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Read the addendum once, at the start of the turn.

        Returns:
            State update holding the text to append.
        """
        return await asyncio.to_thread(self._load)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Show the addendum to the model on every call.

        Returns:
            The model response.
        """
        text = (request.state or {}).get("_prompt_addendum")
        return handler(experience.append_system_text(request, text))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Show the addendum to the model on every call.

        Returns:
            The model response.
        """
        text = (request.state or {}).get("_prompt_addendum")
        return await handler(experience.append_system_text(request, text))


if __name__ == "__main__":
    raise SystemExit(main())
