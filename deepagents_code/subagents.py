"""Subagent loader for app.

Loads custom subagent definitions from the filesystem. Subagents are defined
as markdown files with YAML frontmatter in the agents/ directory.

Directory structure:
    .deepagents/agents/{agent_name}/AGENTS.md

Example file (researcher/AGENTS.md):
    ---
    name: researcher  # optional; defaults to the folder name
    description: Research topics on the web before writing content
    model: anthropic:claude-haiku-4-5-20251001
    ---

    You are a research assistant with access to web search.

    ## Your Process
    1. Search for relevant information
    2. Summarize findings clearly

The `name` field is optional; when omitted it defaults to the folder name
(e.g. `researcher`). This diverges from the Agent Skills specification
(`deepagents.middleware.skills`), which requires `name` in frontmatter and
warns when it does not match the parent directory name. Subagents use the
folder name as an implicit fallback instead because subagent definitions are
already uniquely identified by their folder — requiring a redundant `name`
field adds friction without adding information.
"""

# [해설] 모듈 역할: 사용자가 파일시스템에 직접 적어 둔 '사용자 정의
# [해설]   서브에이전트'(마크다운 + YAML frontmatter)를 읽어 메타데이터 목록으로 만든다.
# [해설] 실행 위치: 서브에이전트 정의를 쓰는 쪽은 에이전트 그래프를 조립하는
# [해설]   프로세스(LangGraph 서버 또는 --acp 같은 in-process 경로)이다.
# [해설] 주요 진입점: list_subagents(). 내부 함수 _load_subagents_from_dir() ->
# [해설]   _parse_subagent_file() 순서로 호출된다.
# [해설] 이 모듈은 모델을 만들거나 SDK 를 호출하지 않는다. 읽기만 하며,
# [해설]   결과(SubagentMetadata 목록)를 agent.py 의 create_cli_agent 가 SDK
# [해설]   서브에이전트 설정으로 바꿔 create_deep_agent(subagents=...) 에 넘긴다(변환은
# [해설]   create_cli_agent 의 'for subagent_meta in list_subagents(' 반복문).
# [해설] 우선순위 규칙: 같은 이름이면 프로젝트(.deepagents/agents) 정의가 사용자(홈)
# [해설]   정의를 덮어쓴다.
# [해설] 관련 레인 모듈: agent.py(조립), tools.py(도구), offload*.py(컨텍스트 오프로드).

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, TypedDict

import yaml

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# [해설] 파일 하나에서 읽어 낸 서브에이전트 정의를 담는 TypedDict. SDK 의 SubAgent
# [해설]   스펙과는 다른 '로딩 중간 표현'이다.
# [해설] source 는 'user' 또는 'project' 로, 호출자(_load_subagents_from_dir)가 나중에
# [해설]   채운다.
class SubagentMetadata(TypedDict):
    """Metadata for a custom subagent loaded from filesystem."""

    name: str
    """Unique identifier for the subagent, used with the task tool."""

    description: str
    """What this subagent does. Main agent uses this to decide when to delegate."""

    system_prompt: str
    """Instructions for the subagent (body of the markdown file)."""

    model: str | None
    """Optional model override in 'provider:model-name' format."""

    source: str
    """Where this subagent was loaded from ('user' or 'project')."""

    path: str
    """Absolute path to the subagent definition file."""


# [해설] AGENTS.md 한 개를 파싱하는 함수. 실패하면 예외를 던지지 않고 경고 로그를 남긴
# [해설]   뒤 None 을 돌려준다(잘못된 파일 하나가 전체 로딩을 막지 않게 하려는 설계).
# [해설][설계] name 은 선택이며 생략하면 폴더 이름을 쓴다. 단, name 키가 있는데 비었거나
# [해설]   문자열이 아니면 폴백하지 않고 거부한다 - 오타가 조용히 가려지지 않게 하기
# [해설]   위함.
def _parse_subagent_file(
    file_path: Path, *, fallback_name: str | None = None
) -> SubagentMetadata | None:
    """Parse a subagent markdown file with YAML frontmatter.

    The file must have YAML frontmatter (delimited by ---) containing at minimum
    a 'description' field. The body of the file becomes the system_prompt.

    Unlike the Agent Skills spec, `name` is optional here — when omitted the
    folder name passed via `fallback_name` is used instead. Skills require
    `name` in frontmatter and warn when it doesn't match the directory name;
    subagents relax that to a fallback so users don't repeat the folder name
    redundantly.

    Args:
        file_path: Path to the markdown file.
        fallback_name: Name to use when the frontmatter omits `name` entirely.
            A present-but-empty, whitespace-only, or non-string `name` is
            treated as invalid and rejected rather than falling back, so a typo
            surfaces loudly instead of being silently masked by the folder name.

    Returns:
        SubagentMetadata if parsing succeeds, None otherwise.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Skipping subagent %s: could not read file (%s)", file_path, exc)
        return None

    # [해설][흐름] 1) 파일 읽기 2) --- 로 감싼 frontmatter 와 본문 분리(정규식) 3) YAML
    # [해설]   파싱 4) 필드 타입 검증 5) dict 반환. 단계마다 실패하면 None.
    # Extract YAML frontmatter (--- delimited)
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        logger.warning(
            "Skipping subagent %s: missing YAML frontmatter. The file must start "
            "with a '---' delimited block containing at least 'description'.",
            file_path,
        )
        return None

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning(
            "Skipping subagent %s: invalid YAML frontmatter (%s)", file_path, exc
        )
        return None

    # Validate frontmatter structure and required fields
    if not isinstance(frontmatter, dict):
        logger.warning(
            "Skipping subagent %s: frontmatter must be a mapping with a "
            "'description' field.",
            file_path,
        )
        return None

    name_value = frontmatter.get("name", fallback_name)
    description_value = frontmatter.get("description")
    model = frontmatter.get("model")

    # Validate types: name and description must be non-empty strings (leading and
    # trailing whitespace is stripped, so a whitespace-only value is rejected).
    # model is optional but must be a string if present.
    name = (
        name_value.strip()
        if isinstance(name_value, str) and name_value.strip()
        else None
    )
    description = (
        description_value.strip()
        if isinstance(description_value, str) and description_value.strip()
        else None
    )
    model_valid = model is None or isinstance(model, str)

    if name is None or description is None or not model_valid:
        invalid_fields: list[str] = []
        if name is None:
            invalid_fields.append("name (non-empty string required)")
        if description is None:
            invalid_fields.append("description (non-empty string required)")
        if not model_valid:
            invalid_fields.append("model (string required when present)")
        logger.warning(
            "Skipping subagent %s: invalid or missing frontmatter field(s): %s",
            file_path,
            ", ".join(invalid_fields),
        )
        return None

    if "name" not in frontmatter:
        # Fallback engaged. Log it so a typo'd key (e.g. `nmae:`) that silently
        # resolves to the folder name is at least diagnosable at debug level.
        logger.debug(
            "Subagent %s: 'name' omitted from frontmatter; using folder name %r.",
            file_path,
            name,
        )

    # [해설] 본문(match.group(2))이 서브에이전트의 system_prompt 가 된다. source 는 빈
    # [해설]   문자열로 두고 호출자가 채운다.
    return {
        "name": name,
        "description": description,
        "system_prompt": match.group(2).strip(),
        "model": model,
        "source": "",  # Set by caller
        "path": str(file_path),
    }


# [해설] agents_dir/{이름}/AGENTS.md 구조를 순회해 {이름: 메타데이터} dict 를 만든다.
# [해설][주의] 폴더 순회 순서가 파일시스템에 의존하므로, 서로 다른 폴더가 같은 name 으로
# [해설]   해석되면 어느 쪽이 남을지 불확정이다. 코드는 충돌 경고만 남기고 마지막으로
# [해설]   순회된 것이 이긴다.
def _load_subagents_from_dir(
    agents_dir: Path, source: str
) -> dict[str, SubagentMetadata]:
    """Load subagents from a directory.

    Expects structure: agents_dir/{subagent_name}/AGENTS.md

    Args:
        agents_dir: Directory containing subagent folders.
        source: Source identifier ('user' or 'project').

    Returns:
        Dict mapping subagent name to metadata.
    """
    subagents: dict[str, SubagentMetadata] = {}

    if not agents_dir.exists() or not agents_dir.is_dir():
        return subagents

    for entry in agents_dir.iterdir():
        if not entry.is_dir():
            # A stray file directly under agents/ is a common mistake: subagents
            # must live at agents/{name}/AGENTS.md, not agents/{name}.md.
            if entry.suffix.lower() == ".md":
                logger.warning(
                    "Ignoring %s subagent file %s: subagents must be defined at "
                    "%s/{subagent-name}/AGENTS.md, not as a file directly in the "
                    "agents directory.",
                    source,
                    entry,
                    agents_dir,
                )
            continue

        # [해설] 파일명은 반드시 AGENTS.md. agent.md 나 {name}.md 로 두면 무시되고
        # [해설]   경고만 나온다(아래 stray_md 분기).
        # Look for {folder_name}/AGENTS.md
        subagent_file = entry / "AGENTS.md"
        if not subagent_file.exists():
            # The folder exists but holds a differently-named markdown file
            # (e.g. agent.md or {name}.md) instead of the required AGENTS.md.
            stray_md = [p.name for p in entry.glob("*.md")]
            if stray_md:
                logger.warning(
                    "Ignoring %s subagent folder %s: expected an AGENTS.md file "
                    "but found %s. Rename the definition to AGENTS.md.",
                    source,
                    entry,
                    ", ".join(sorted(stray_md)),
                )
            continue

        subagent = _parse_subagent_file(subagent_file, fallback_name=entry.name)
        if subagent:
            subagent["source"] = source
            # The folder name and a declared `name` can differ, so two folders can
            # resolve to the same subagent name and silently collapse to one entry.
            # Iteration order is filesystem-dependent, so warn rather than let a
            # definition vanish without explanation.
            existing = subagents.get(subagent["name"])
            if existing is not None:
                logger.warning(
                    "Subagent name collision in %s: %s and %s both resolve to "
                    "name=%r. Using %s; give each subagent a unique folder or "
                    "frontmatter 'name'.",
                    agents_dir,
                    existing["path"],
                    subagent["path"],
                    subagent["name"],
                    subagent["path"],
                )
            subagents[subagent["name"]] = subagent

    return subagents


# [해설] 공개 API. 사용자 디렉터리를 먼저 넣고 프로젝트 디렉터리를 나중에 update 하여,
# [해설]   dict 키(name) 기준으로 프로젝트가 사용자 정의를 덮어쓴다.
def list_subagents(
    *,
    user_agents_dir: Path | None = None,
    project_agents_dir: Path | None = None,
) -> list[SubagentMetadata]:
    """List subagents from user and/or project directories.

    Scans for subagent definitions in the provided directories.
    Project subagents override user subagents with the same name.

    Args:
        user_agents_dir: Path to user-level agents directory.
        project_agents_dir: Path to project-level agents directory.

    Returns:
        List of subagent metadata, with project subagents taking precedence.
    """
    all_subagents: dict[str, SubagentMetadata] = {}

    # Load user subagents first (lower priority)
    if user_agents_dir is not None:
        all_subagents.update(_load_subagents_from_dir(user_agents_dir, "user"))

    # Load project subagents second (override user)
    if project_agents_dir is not None:
        all_subagents.update(_load_subagents_from_dir(project_agents_dir, "project"))

    return list(all_subagents.values())
