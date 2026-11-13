"""Lightweight theme preference coercion shared by config providers and the UI."""
# [해설] 테마 이름 해석 헬퍼 모음. 설정 제공자(`providers.py`)와 UI가 같은 규칙으로 테마
# [해설]   이름을 정규화하도록 한 곳에 모았다.
# [해설] 입력은 사용자가 손으로 편집하는 값이다: TOML의 `[ui] theme`,
# [해설]   `[ui.terminal_themes]` 표, 환경변수 `DEEPAGENTS_CODE_THEME`.
# [해설] 해석 결과는 `deepagents_code/theme.py`의 레지스트리(`theme.get_registry()`)에
# [해설]   실제로 등록된 키로 한정된다. 등록되지 않은 이름은 `None`이고 경고만 남긴다.
# [해설] 모듈 이름처럼 "lightweight"를 유지하려는 파일이다. 무거운 UI 의존을 끌어오지
# [해설]   않고 `theme` 모듈만 import 한다.

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, cast

from deepagents_code import theme

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


# [해설] 레지스트리 키 또는 표시 라벨을 대소문자 무시로 받아 정식 레지스트리 키로
# [해설]   바꾼다. 옛 이름 `textual-ansi`는 `ansi-light`로 이전(migration)한다.
# [해설] 문자열이 아니거나 등록 이름이 아니면 `None`.
def resolve_theme_name(value: object) -> str | None:
    """Resolve a user-supplied theme name to a canonical registry key.

    Accepts the registry key or the human-readable label, case-insensitive on
    both, with surrounding whitespace stripped - config values (especially
    `[ui.terminal_themes]`) and the `DEEPAGENTS_CODE_THEME` env var are commonly
    hand-edited. Also applies the legacy `textual-ansi` to `ansi-light`
    migration, which predates Textual 8.2.5 and can be dropped once no
    supported config can still carry the old name.

    Args:
        value: Raw value read from TOML or an environment variable.

    Returns:
        Canonical registry key, or `None` when the value is not a string or
        names no registered theme.
    """
    if not isinstance(value, str):
        return None
    name = value.strip()
    if name == "textual-ansi":
        name = "ansi-light"
    registry = theme.get_registry()
    if name in registry:
        return name
    folded = name.casefold()
    for registered, entry in registry.items():
        if registered.casefold() == folded or entry.label.casefold() == folded:
            return registered
    return None


# [해설][설계] `tomllib`이 만든 TOML 표는 문자열 키 dict인데 타입 검사기가 이를 추론하지
# [해설]   못해, 캐스트를 이 한 곳에만 가둔다. 범용 우회 수단으로 번지지 않게 하려는
# [해설]   의도가 docstring에 적혀 있다.
def as_toml_table(value: object) -> dict[str, object] | None:
    """Return `value` as a TOML table when it has the expected runtime shape.

    `tomllib` parses TOML tables as string-keyed dicts, which `ty` cannot infer
    from a runtime `dict` check. Keep the cast at this boundary so it does not
    become a general-purpose escape hatch - now more important, not less, since
    the helper became importable from a shared module.
    """
    if not isinstance(value, dict):
        return None
    return cast("dict[str, object]", value)


# [해설][흐름] 1) `[ui]` 표에서 `terminal_themes`를 찾는다 2) 표 모양인지 검사한다 3)
# [해설]   환경변수 `TERM_PROGRAM` 값을 키로 매핑을 찾는다 4) `resolve_theme_name`으로
# [해설]   정식 이름을 얻는다.
# [해설] 각 단계의 잘못된 설정은 경고 로그를 남기고 `None`을 반환한다. 예외를 던지지
# [해설]   않아 잘못된 테마 설정이 기동을 막지 않는다.
def resolve_terminal_mapping(ui: Mapping[str, object]) -> str | None:
    """Resolve `[ui.terminal_themes][TERM_PROGRAM]` to a registered theme.

    Centralizes both the lookup and the misconfiguration warnings, which are
    logged exactly once per call. Three callers now share that contract: the
    managed and user TOML tiers and the UI.

    Args:
        ui: An `[ui]` table parsed from a managed or user TOML source.

    Returns:
        Canonical registry key, or `None` when no valid mapping applies.
    """
    terminal_themes = ui.get("terminal_themes")
    if terminal_themes is None:
        return None
    terminal_themes_table = as_toml_table(terminal_themes)
    if terminal_themes_table is None:
        logger.warning(
            "[ui.terminal_themes] should be a table mapping TERM_PROGRAM "
            "values to theme names; got %s",
            type(terminal_themes).__name__,
        )
        return None
    term_program = os.environ.get("TERM_PROGRAM", "").strip()
    if not term_program:
        if terminal_themes_table:
            logger.warning(
                "[ui.terminal_themes] is configured but TERM_PROGRAM is unset; "
                "no per-terminal theme will be applied",
            )
        return None
    mapped = terminal_themes_table.get(term_program)
    resolved = resolve_theme_name(mapped)
    if resolved is not None:
        return resolved
    if isinstance(mapped, str):
        logger.warning(
            "Unknown theme '%s' mapped to TERM_PROGRAM='%s' "
            "in [ui.terminal_themes]; ignoring",
            mapped,
            term_program,
        )
    elif mapped is not None:
        logger.warning(
            "Expected string theme name for TERM_PROGRAM='%s' in "
            "[ui.terminal_themes], got %s; ignoring",
            term_program,
            type(mapped).__name__,
        )
    return None
