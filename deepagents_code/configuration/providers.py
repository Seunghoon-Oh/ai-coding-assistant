"""Synchronous providers and provider-domain option coercion."""

# [해설] 이 모듈은 설정 계층의 '출처(provider)' 구현이다. resolver.py가 순위로 승자를
# [해설]   고른다면, 여기는 각 출처에서 값을 읽어
# [해설]   매니페스트 옵션 타입(ConfigOption.kind)에 맞게 변환(coerce)한 뒤 Found /
# [해설]   Unset / Invalid 중 하나로 돌려준다.
# [해설] 출처 세 가지: TomlFileProvider(관리형·사용자 TOML 파일, durable),
# [해설]   EnvProvider(환경변수, 비영속),
# [해설]   DefaultProvider(매니페스트 기본값, 영속). 순위는 resolver.py의
# [해설]   MANAGED/ENVIRONMENT/USER/DEFAULT_RANK가 정한다.
# [해설] 의미 구분: Unset=이 출처는 선언하지 않음(다음 계층으로), Invalid=선언했지만
# [해설]   읽을 수 없음(경고 후 다음 계층으로).
# [해설] 관리형 정책이 원격이면 RemoteTomlProvider가 HTTPS로 받아온다. 이때 프록시 우회,
# [해설]   리다이렉트 거부,
# [해설]   크기 상한(1MiB)·시간 상한(5초)을 모두 강제한다. 이 상수들은 아래
# [해설]   REMOTE_MANAGED_CONFIG_* 에 있다.
# [해설] 파일을 못 읽는 경우 provider는 예외를 던지지 않고 빈 스냅샷 +
# [해설]   상태(ProviderHealth)를 돌려준다. 그 상태를 해석해
# [해설]   '시작을 막을지'를 정하는 곳은 service.py다.
# [해설][주의] SHADOWED/UNUSABLE/RETAINED_SOURCE_SUFFIX 상수는 config_manifest.py가
# [해설]   문구를 비교해 경고를 중복 제거하는 데 쓰므로 문구를 바꾸면 안 된다.
from __future__ import annotations

import logging
import os
import threading
import time
import tomllib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, assert_never, cast, override

from deepagents_code._env_vars import classify_env_bool

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from concurrent.futures import Future
    from http.client import HTTPResponse
    from pathlib import Path
    from typing import Protocol
    from urllib.request import Request

    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.configuration.resolver import RankedProviderValue

    class _TimeoutSocket(Protocol):
        """Socket operation needed to tighten an active HTTP response."""

        def settimeout(self, value: float | None) -> None:
            """Set the maximum wait for the next socket operation."""

        def shutdown(self, how: int) -> None:
            """Stop an in-flight socket operation."""

    class _RemoteOpener(Protocol):
        """Minimal opener surface used by the remote provider.

        An implementation must bypass environment proxies and refuse
        redirects. The `cast` in `_build_remote_opener` erases that, so it is
        recorded here: an opener that satisfies this protocol without both
        properties silently reopens two destination-control holes.
        """

        def open(self, request: Request, *, timeout: float) -> HTTPResponse:
            """Open one HTTP response."""


from deepagents_code.configuration.resolver import (
    DEFAULT_RANK,
    ENVIRONMENT_RANK,
    USER_RANK,
)
from deepagents_code.configuration.types import (
    Found,
    Invalid,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    TomlSnapshot,
    Unset,
    _validate_remote_source_url as _validate_remote_url,
)

logger = logging.getLogger(__name__)

# [해설] 원격 관리형 설정의 한도: 본문 1MiB, 전체 5초. 다운로드가 잘리거나 오래 걸려
# [해설]   정책이 일부만 적용되는 일을 막는다.
REMOTE_MANAGED_CONFIG_MAX_BYTES = 1024 * 1024
REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS = 5
_HTTP_OK = 200
_REFUSED_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
"""Statuses `urllib` would have followed had `_RejectRedirects` allowed it.

Not the whole 3xx range: a `300`, `304`, or `305` reaches the same handler
without ever naming a redirect target, so reporting it as a refused redirect
would misdescribe the server's answer.
"""

SHADOWED_TABLE_SUFFIX = "— every option under it falls back to its next source"
"""Tail of the rejection raised when a scalar shadows a whole TOML table.

`config_manifest._emit_ranked_diagnostics` matches this text to deduplicate the
warning across a full-manifest pass. Both sides must share one constant: a
reworded message that no longer matches would silently restore roughly one
duplicated line per option for a single typo.
"""

UNUSABLE_SOURCE_SUFFIX = "— using defaults for every option it would have set"
"""Tail of the rejection raised when a whole TOML source could not be read.

Deduplicated the same way, and for the same reason: a rejected file affects
every option at once, so the warning belongs to the file, not to each key.
"""

RETAINED_SOURCE_SUFFIX = "— still applying the last readable version of it"
"""Tail of the rejection raised when a failed reload kept the previous values.

Distinct from `UNUSABLE_SOURCE_SUFFIX` because the consequence is different:
nothing fell back to a default, but the file on disk no longer describes what
the process is enforcing, and an edit the user just made is not in effect.
"""


# [해설] 타입 검사기를 달래는 cast 헬퍼. 값 검증은 하지 않으며, 호출하는 쪽이
# [해설]   option.kind로 이미 검증했다는 전제다.
def _found_for[T](option: ConfigOption[T], value: object) -> Found[T]:
    """Assert a kind-validated provider value into its manifest option type.

    This checks nothing -- the body is a `cast`, and `option` is taken only to
    bind `T` at the call site. What makes it sound is the caller: every branch
    that reaches here sits behind a test on `option.kind`, and the
    `ConfigOption.__new__` overloads map that same `kind` to `T`. A type
    checker cannot follow an enum comparison to a class-level type parameter,
    so the relationship the overloads establish is re-stated here by hand.

    Keep the two in step: a branch that returns a shape the overload table does
    not name for that kind is a silent lie, not a type error.

    Returns:
        The validated value wrapped in a typed provider result.
    """
    del option
    return Found(cast("T", value))


def _ranked_for[T](
    option: ConfigOption[T], ranked: RankedProviderValue[object]
) -> RankedProviderValue[T]:
    """Assert an option-specific provider path into its value type.

    Unchecked, for the same reason as `_found_for`: `option` binds `T` and the
    body is a `cast`. Used by the theme/startup paths, which build their ranked
    value before the kind is available to narrow on.

    Returns:
        The ranked value associated with the option's value type.
    """
    del option
    return cast("RankedProviderValue[T]", ranked)


# [해설] 환경변수 문자열 하나를 옵션 종류(bool, int, enum 등)에 맞춰 변환한다. 실패하면
# [해설]   Invalid(이유)를 돌려줄 뿐 로그는 남기지 않는다(상태 조회가 로그를 일으키지
# [해설]   않도록).
def coerce_environment_value[T](
    option: ConfigOption[T], raw: str, name: str
) -> ProviderResult[T]:
    """Coerce one present environment value within the env provider domain.

    The returned reason preserves the established diagnostic text. Resolution
    decides when to emit it so health inspection does not log a rejection as a
    side effect of merely reading provider state.

    Args:
        option: Manifest declaration that defines the output type.
        raw: Present environment string.
        name: Environment variable spelling that supplied `raw`.

    Returns:
        `Found` with the typed value or `Invalid` with the rejection reason.
    """
    from deepagents_code.config_manifest import VALID_CURSOR_STYLES, OptionKind

    kind = option.kind
    if kind in {OptionKind.BOOL, OptionKind.BOOL_MODE_DEFAULT}:
        classified = classify_env_bool(raw)
        if classified is None:
            return Invalid(f"Ignoring {name}={raw!r} (expected bool)")
        return _found_for(option, classified)
    if kind is OptionKind.BOOL_PRESENCE:
        return _found_for(option, bool(raw))
    if kind is OptionKind.STR:
        return _found_for(option, raw)
    if kind is OptionKind.NON_EMPTY_STR:
        value = raw.strip()
        if value:
            return _found_for(option, value)
        return Invalid(f"Ignoring {name}={raw!r} (expected non-empty string)")
    if kind is OptionKind.ISO_DATETIME:
        from deepagents_code.config_manifest import normalize_iso_datetime

        if value := normalize_iso_datetime(raw):
            return _found_for(option, value)
        return Invalid(
            f"Ignoring {name}={raw!r} (expected ISO 8601 date or aware datetime)"
        )
    if kind is OptionKind.DURATION_SECONDS:
        from deepagents_code.config_manifest import normalize_duration

        if value := normalize_duration(raw):
            return _found_for(option, value)
        return Invalid(f"Ignoring {name}={raw!r} (expected duration such as 7d)")
    if kind is OptionKind.EXTENSION_TRUST_DELEGATE:
        from deepagents_code.extensions.settings import parse_trust_policy

        policy = parse_trust_policy(raw)
        if policy is not None:
            return _found_for(option, policy.value)
        return Invalid(f"Ignoring {name}={raw!r} (expected ask, always, or never)")
    if kind is OptionKind.LOG_LEVEL_DELEGATE:
        from deepagents_code._debug import LOG_LEVELS

        level = raw.strip().upper()
        if level in LOG_LEVELS:
            return _found_for(option, level)
        valid = ", ".join(LOG_LEVELS)
        return Invalid(f"Ignoring {name}={raw!r} (expected one of {valid})")
    if kind is OptionKind.INT:
        try:
            return _found_for(option, int(raw.strip()))
        except ValueError:
            return Invalid(f"Ignoring {name}={raw!r} (expected int)")
    if kind is OptionKind.NON_NEGATIVE_INT:
        try:
            value = int(raw.strip())
        except ValueError:
            return Invalid(f"Ignoring {name}={raw!r} (expected int >= 0)")
        if value >= 0:
            return _found_for(option, value)
        return Invalid(f"Ignoring {name}={raw!r} (expected int >= 0)")
    if kind is OptionKind.FLOAT:
        try:
            return _found_for(option, float(raw.strip()))
        except ValueError:
            return Invalid(f"Ignoring {name}={raw!r} (expected number)")
    if kind is OptionKind.SHELL_LIST_DELEGATE:
        from deepagents_code.config import parse_shell_allow_list

        try:
            return _found_for(option, parse_shell_allow_list(raw))
        except ValueError:
            return Invalid(f"Ignoring invalid {name}")
    if kind is OptionKind.SKILLS_DIRS_DELEGATE:
        from deepagents_code.config import _parse_extra_skills_dirs

        try:
            return _found_for(option, _parse_extra_skills_dirs(raw, None))
        except OSError, ValueError, RuntimeError:
            return Invalid(f"Ignoring {name} (could not resolve a path)")
    if kind is OptionKind.THEME_DELEGATE:
        # Theme names are resolved by the theme-aware provider path. Keep this
        # defensive passthrough for the compatibility wrapper.
        return _found_for(option, raw)
    if kind is OptionKind.CURSOR_STYLE_DELEGATE:
        if raw in VALID_CURSOR_STYLES:
            return _found_for(option, raw)
        return Invalid(f"Ignoring {name}={raw!r} (expected 'block' or 'underline')")
    if kind in {
        OptionKind.MODEL_LIST_DELEGATE,
        OptionKind.PTC_DELEGATE,
        OptionKind.STRUCTURED,
    }:
        # No option of these kinds declares `env_var` *or* `fallback_env_vars`,
        # so the env provider never reaches this branch; only the `_coerce_env`
        # compatibility wrapper, which has no env-var guard, can. If a future
        # option gains either name, reject rather than pass the raw string
        # through: an uncoerced value would bypass the delegate parser's
        # validation. Rejection drops to the next-weaker tier (TOML, then the
        # manifest default) -- note that for a policy option such as
        # `models.allowed` the default is "unrestricted", so what keeps that
        # fallback safe is managed config outranking the environment, not the
        # default itself being conservative.
        return Invalid(f"{option.key} is not env-backed; ignoring {name}={raw!r}")
    if kind is OptionKind.STARTUP_MODE_DELEGATE:
        from deepagents_code.model_config import VALID_STARTUP_MODES

        if raw in VALID_STARTUP_MODES:
            return _found_for(option, raw)
        return Invalid(
            f"Ignoring {name}={raw!r} (expected 'manual', 'auto', or 'yolo')"
        )
    assert_never(kind)


# [해설] TOML 값 하나를 옵션 종류에 맞춰 변환한다. coerce_environment_value의 파일 판.
# [해설]   환경변수는 문자열, TOML은 이미 타입이 있어 규칙이 다르다.
def coerce_toml_value[T](
    option: ConfigOption[T], raw: object, *, source: str
) -> ProviderResult[T]:
    """Coerce one present TOML value within the file-provider domain.

    Args:
        option: Manifest declaration that defines the output type.
        raw: Parsed TOML value.
        source: Human-readable provider name used in diagnostic text.

    Returns:
        `Found` with the typed value or `Invalid` with the rejection reason.
    """
    from deepagents_code.config_manifest import VALID_CURSOR_STYLES, OptionKind

    kind = option.kind
    label = option.toml_path or option.key

    if option.key == "threads.sort_order":
        if isinstance(raw, str) and raw in {"created_at", "updated_at"}:
            return _found_for(option, raw)
        return Invalid(
            f"Ignoring {label}={raw!r} in {source} "
            "(expected 'created_at' or 'updated_at')"
        )
    if kind in {
        OptionKind.BOOL,
        OptionKind.BOOL_MODE_DEFAULT,
        OptionKind.BOOL_PRESENCE,
    }:
        if isinstance(raw, bool):
            value = not raw if option.invert_toml_bool else raw
            return _found_for(option, value)
    elif kind is OptionKind.INT:
        if isinstance(raw, int) and not isinstance(raw, bool):
            return _found_for(option, raw)
    elif kind is OptionKind.NON_NEGATIVE_INT:
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            return _found_for(option, raw)
    elif kind is OptionKind.FLOAT:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return _found_for(option, float(raw))
    elif kind is OptionKind.STR:
        if isinstance(raw, str):
            return _found_for(option, raw)
    elif kind is OptionKind.NON_EMPTY_STR:
        if isinstance(raw, str) and (value := raw.strip()):
            return _found_for(option, value)
    elif kind is OptionKind.ISO_DATETIME:
        from deepagents_code.config_manifest import normalize_iso_datetime

        if value := normalize_iso_datetime(raw):
            return _found_for(option, value)
    elif kind is OptionKind.DURATION_SECONDS:
        from deepagents_code.config_manifest import normalize_duration

        if value := normalize_duration(raw):
            return _found_for(option, value)
    elif kind is OptionKind.MODEL_LIST_DELEGATE:
        from deepagents_code.model_config import parse_model_allowlist

        try:
            return _found_for(option, parse_model_allowlist(raw))
        except (TypeError, ValueError) as exc:
            return Invalid(f"Ignoring {label} in {source}: {exc}")
    elif kind is OptionKind.EXTENSION_TRUST_DELEGATE:
        from deepagents_code.extensions.settings import parse_trust_policy

        policy = parse_trust_policy(raw)
        if policy is not None:
            return _found_for(option, policy.value)
    elif kind is OptionKind.SKILLS_DIRS_DELEGATE:
        if isinstance(raw, list):
            from deepagents_code.config import _parse_extra_skills_dirs

            try:
                return _found_for(
                    option, _parse_extra_skills_dirs(None, cast("list[str]", raw))
                )
            except ValueError, RuntimeError:
                return Invalid(
                    f"Ignoring {label} in {source} (could not resolve a path)"
                )
    elif kind is OptionKind.PTC_DELEGATE:
        from deepagents_code.config import _parse_interpreter_ptc

        try:
            return _found_for(option, _parse_interpreter_ptc(raw))
        except ValueError as exc:
            return Invalid(f"Ignoring {label} in {source}: {exc}")
    elif kind is OptionKind.CURSOR_STYLE_DELEGATE:
        if isinstance(raw, str) and raw in VALID_CURSOR_STYLES:
            return _found_for(option, raw)
        return Invalid(
            f"Ignoring {label}={raw!r} in {source} (expected 'block' or 'underline')"
        )
    elif kind is OptionKind.STARTUP_MODE_DELEGATE:
        from deepagents_code.model_config import VALID_STARTUP_MODES

        if isinstance(raw, str) and raw in VALID_STARTUP_MODES:
            return _found_for(option, raw)
        return Invalid(
            f"Ignoring {label}={raw!r} in {source} "
            "(expected 'manual', 'auto', or 'yolo')"
        )
    elif kind is OptionKind.STRUCTURED:
        return _found_for(option, raw)
    elif kind is OptionKind.SHELL_LIST_DELEGATE:
        from deepagents_code.config import (
            parse_shell_allow_list,
            parse_shell_allow_list_items,
        )

        try:
            if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                return _found_for(
                    option,
                    parse_shell_allow_list_items(cast("list[str]", raw)),
                )
            if isinstance(raw, str):
                return _found_for(option, parse_shell_allow_list(raw))
        except ValueError as exc:
            return Invalid(f"Ignoring {label} in {source}: {exc}")

    return Invalid(f"Ignoring {label}={raw!r} in {source} (expected {option.type})")


# [해설] 파싱된 TOML 표에서 옵션의 toml_keys 경로를 따라 내려가 값을 읽는다. 중간 경로가
# [해설]   표가 아니면(스칼라가 표를 가림) Invalid + SHADOWED 문구.
def ranked_toml_value[T](
    option: ConfigOption[T],
    data: Mapping[str, Any],
    *,
    rank: int,
    durable: bool,
    status: ProviderStatus,
) -> RankedProviderValue[T]:
    """Read and coerce one option from a parsed TOML provider.

    Args:
        option: Manifest option to read.
        data: Parsed provider table.
        rank: Numeric precedence rank.
        durable: Whether this tier masks lower-priority ephemeral tiers.
        status: Provider health and display metadata.

    Returns:
        Ranked `Found`, `Unset`, or `Invalid` provider result.
    """
    from deepagents_code.configuration.resolver import RankedProviderValue

    if not status.usable or not option.toml_keys:
        result: ProviderResult[T] = Unset()
    else:
        node: object = data
        result = Unset()
        for index, key in enumerate(option.toml_keys):
            if not isinstance(node, dict):
                path = option.toml_keys[:index]
                result = Invalid(
                    f"Ignoring {status.name} [{'.'.join(path)}]; expected a "
                    f"table, got {type(node).__name__} {SHADOWED_TABLE_SUFFIX}"
                )
                break
            if key not in node:
                break
            node = node[key]
        else:
            result = coerce_toml_value(option, node, source=status.name)
    return RankedProviderValue(rank, durable, status, result)


# [해설] 환경변수 이름 해석: DEEPAGENTS_CODE_ 접두사가 붙은 이름이 존재하면(빈 값이어도)
# [해설]   그것을 쓰고, 없으면 원래 이름을 쓴다.
def _prefix_aware_env_name(name: str, environ: Mapping[str, str]) -> str:
    """Select a prefix override by presence, including an explicitly empty one.

    Args:
        name: Canonical environment variable name.
        environ: Environment mapping being resolved.

    Returns:
        The prefixed name when present, otherwise the canonical name.
    """
    prefixed = (
        name if name.startswith("DEEPAGENTS_CODE_") else f"DEEPAGENTS_CODE_{name}"
    )
    return prefixed if prefixed in environ else name


# [해설] 환경변수 계층. 옵션의 env_var와 fallback_env_vars를 차례로 보고 첫 Found를
# [해설]   쓰며, 그 사이의 경고는 diagnostics에 모은다.
# [해설]   공백만 있는 값은 '미설정'으로 보되 경고한다(empty_env_is_false 옵션이면
# [해설]   False로 해석). 환경변수는 durable=False다.
def ranked_environment_value[T](
    option: ConfigOption[T],
    environ: Mapping[str, str],
    *,
    rank: int,
) -> RankedProviderValue[T]:
    """Read and coerce one option from the process-environment domain.

    Args:
        option: Manifest option to read.
        environ: Environment mapping, normally `os.environ`.
        rank: Numeric precedence rank.

    Returns:
        Ranked provider result. Fallback names remain one provider tier.
    """
    from deepagents_code.configuration.resolver import RankedProviderValue

    names: list[str] = []
    if option.env_var:
        names.append(_prefix_aware_env_name(option.env_var, environ))
    for fallback in option.fallback_env_vars:
        name = (
            _prefix_aware_env_name(fallback, environ)
            if option.prefix_aware_fallbacks
            else fallback
        )
        # A prefix-aware fallback can resolve to a name the primary already
        # selected; reading it twice would emit a duplicate diagnostic.
        if name not in names:
            names.append(name)

    status = ProviderStatus("environment", None, ProviderHealth.OK)
    last_invalid: Invalid | None = None
    diagnostics: list[str] = []
    for name in names:
        raw = environ.get(name)
        if raw is None:
            continue
        status = replace(status, name=f"env ({name})")
        if not raw.strip():
            if option.empty_env_is_false:
                return RankedProviderValue(
                    rank, False, status, _found_for(option, False)
                )
            if raw:
                last_invalid = Invalid(
                    f"Ignoring {name}={raw!r} (whitespace-only; treated as unset)"
                )
                diagnostics.append(last_invalid.reason)
            continue
        result = coerce_environment_value(option, raw, name)
        if isinstance(result, Found):
            return RankedProviderValue(
                rank,
                False,
                status,
                result,
                tuple(diagnostics),
            )
        if isinstance(result, Invalid):
            last_invalid = result
            diagnostics.append(result.reason)
    return RankedProviderValue(
        rank,
        False,
        status,
        last_invalid or Unset(),
        tuple(diagnostics),
    )


# [해설] 테마는 일반 옵션과 달리 [ui.terminal_themes.<TERM_PROGRAM>] 매핑을 먼저 보고
# [해설]   없으면 [ui].theme를 본다. 둘은 한 provider 안의 내부 순서다.
def ranked_theme_toml_value(
    data: Mapping[str, Any],
    *,
    rank: int,
    durable: bool,
    status: ProviderStatus,
) -> RankedProviderValue[object]:
    """Resolve one file provider's terminal-aware theme preference.

    The terminal mapping and `[ui].theme` fallback are one provider domain:
    they share a durability boundary and source rank. Their internal ordering
    stays inside this provider while precedence between managed, environment,
    user, and default remains the ranked resolver's responsibility.

    Args:
        data: Parsed TOML provider table.
        rank: Numeric provider rank.
        durable: Whether this file tier masks lower ephemeral tiers.
        status: Provider health and display metadata.

    Returns:
        Ranked theme result with the selected TOML path in its display status.
    """
    from deepagents_code.configuration.resolver import RankedProviderValue
    from deepagents_code.configuration.theme_resolution import (
        resolve_terminal_mapping,
        resolve_theme_name,
    )

    if not status.usable:
        return RankedProviderValue(rank, durable, status, Unset())
    ui = data.get("ui")
    if ui is None:
        return RankedProviderValue(rank, durable, status, Unset())
    if not isinstance(ui, dict):
        result: ProviderResult[object] = Invalid(
            f"[ui] in {status.name} should be a table; got "
            f"{type(ui).__name__} while resolving theme"
        )
        return RankedProviderValue(rank, durable, status, result)

    resolved = resolve_terminal_mapping(ui)
    if resolved is not None:
        term_program = os.environ.get("TERM_PROGRAM", "").strip()
        selected = replace(
            status,
            name=f"{status.name} [ui.terminal_themes.{term_program}]",
        )
        return RankedProviderValue(rank, durable, selected, Found(resolved))

    saved = ui.get("theme")
    resolved = resolve_theme_name(saved)
    if resolved is not None:
        selected = replace(status, name=f"{status.name} [ui.theme]")
        return RankedProviderValue(rank, durable, selected, Found(resolved))
    if isinstance(saved, str):
        result = Invalid(f"Unknown theme '{saved}' in {status.name}; ignoring it")
        return RankedProviderValue(rank, durable, status, result)
    return RankedProviderValue(rank, durable, status, Unset())


# [해설] 테마 전용 환경변수 처리. 알 수 없는 이름이면 Invalid로 돌려 다음 계층으로
# [해설]   넘어가게 한다.
def ranked_theme_environment_value(
    environ: Mapping[str, str], *, rank: int
) -> RankedProviderValue[object]:
    """Resolve the theme environment provider.

    Args:
        environ: Environment mapping, normally `os.environ`.
        rank: Numeric environment rank.

    Returns:
        Ranked theme result with the concrete variable name in its status.
    """
    from deepagents_code._env_vars import THEME
    from deepagents_code.configuration.resolver import RankedProviderValue
    from deepagents_code.configuration.theme_resolution import resolve_theme_name

    status = ProviderStatus(f"env ({THEME})", None, ProviderHealth.OK)
    raw = environ.get(THEME)
    if raw is None:
        return RankedProviderValue(rank, False, status, Unset())
    resolved = resolve_theme_name(raw)
    if resolved is not None:
        return RankedProviderValue(rank, False, status, Found(resolved))
    return RankedProviderValue(
        rank,
        False,
        status,
        Invalid(f"Unknown theme '{raw}' in {THEME}; falling through"),
    )


# [해설] 기본값 계층. 일부 옵션은 상수가 아니라 DEBUG·EXPERIMENTAL 환경변수나 테마
# [해설]   모듈에 따라 기본값이 달라진다(BOOL_MODE_DEFAULT 등).
def ranked_default_value[T](
    option: ConfigOption[T], *, rank: int
) -> RankedProviderValue[T]:
    """Produce an option's typed or mode-dependent default provider result.

    Args:
        option: Manifest option whose default should be produced.
        rank: Numeric precedence rank.

    Returns:
        Durable ranked default result.
    """
    from deepagents_code.config_manifest import OptionKind
    from deepagents_code.configuration.resolver import RankedProviderValue

    if option.kind is OptionKind.BOOL_MODE_DEFAULT:
        from deepagents_code._env_vars import DEBUG, EXPERIMENTAL, is_env_truthy

        value: object = is_env_truthy(DEBUG) or is_env_truthy(EXPERIMENTAL)
    elif option.kind is OptionKind.LOG_LEVEL_DELEGATE:
        from deepagents_code._env_vars import DEBUG, is_env_truthy

        value = "DEBUG" if is_env_truthy(DEBUG) else "INFO"
    elif option.kind is OptionKind.THEME_DELEGATE:
        from deepagents_code import theme

        value = theme.DEFAULT_THEME
    elif option.kind is OptionKind.STRUCTURED:
        status = ProviderStatus("default", None, ProviderHealth.OK)
        return RankedProviderValue(rank, True, status, Unset())
    else:
        value = option.default
    status = ProviderStatus("default", None, ProviderHealth.OK)
    return RankedProviderValue(rank, True, status, _found_for(option, value))


# [해설][설계] 원격 관리형 설정 보안: 환경 프록시를 무시하고 리다이렉트를 거부하는
# [해설]   opener. 관리자가 지정한 URL 외 다른 곳으로 흘러가는 길을 막는다.
def _build_remote_opener() -> _RemoteOpener:
    """Build a redirect-refusing, proxy-bypassing opener.

    The `urllib.request` imports are deferred into the body so an install with
    no remote descriptor never pays for them.

    Returns:
        An opener that bypasses environment proxies and rejects redirects.
    """
    from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

    class _RejectRedirects(HTTPRedirectHandler):
        @override
        def redirect_request(self, *_args: object, **_kwargs: object) -> None:
            return None

    return cast(
        "_RemoteOpener",
        build_opener(ProxyHandler({}), _RejectRedirects()),
    )


def _remote_status(
    name: str,
    path: Path | None,
    health: ProviderHealth,
    detail: str,
    source: str | None = None,
) -> TomlSnapshot:
    """Build an empty snapshot with safe remote-source diagnostics.

    Pass `source` only after validation accepted it. Pass `None` for a
    rejected source: it can hold a secret query token, and no detail string
    this module writes contains the URL itself.

    Returns:
        An unhealthy snapshot carrying no policy data.
    """
    return TomlSnapshot(
        {},
        ProviderStatus(name, path, health, detail, remote_source=source),
    )


_TIMED_OUT_DETAIL = "remote source timed out"
"""Shared by the two deadline guards and by `load`'s timeout arms.

`_fail_if_expired` and `_remaining_timeout` test the same condition from
opposite directions, and `load` renders whichever one fired. One string keeps
the three from drifting into differently-worded reports of one failure.
"""

_READ_CHUNK_SIZE = 65536
_remote_open_lock = threading.Lock()
_remote_open_requests: dict[str, Future[HTTPResponse]] = {}


# [해설] 아래 _fail_if_expired ~ _read_limited_response 는 모두 '전체 5초 데드라인'과
# [해설]   '1MiB 상한'을 소켓 수준에서 강제하기 위한 헬퍼들이다.
# [해설]   각각 데드라인 검사, 남은 시간 계산, 별도 스레드로 연결 열기, 헤더 검증, 청크
# [해설]   읽기를 맡는다.
def _fail_if_expired(deadline: float) -> None:
    """Raise once the fetch has run past its end-to-end time boundary.

    Raises:
        TimeoutError: If the current monotonic time is at or past *deadline*.
    """
    if time.monotonic() >= deadline:
        raise TimeoutError(_TIMED_OUT_DETAIL)


def _remaining_timeout(deadline: float) -> float:
    """Return positive seconds left before the end-to-end deadline.

    Returns:
        The remaining timeout in seconds.

    Raises:
        TimeoutError: If the deadline has already passed.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(_TIMED_OUT_DETAIL)
    return remaining


def _close_completed_response(future: Future[HTTPResponse]) -> None:
    """Close a response that arrived after its caller stopped waiting.

    The caller has already reported `"remote source timed out"`, so the outcome
    here cannot change what it returns. Record it anyway: "the host answered
    just past the deadline" and "the host never answered" call for different
    administrator action, and a handshake failure that only shows up on a slow
    connection is otherwise invisible.
    """
    from http.client import HTTPException
    from urllib.error import HTTPError

    try:
        response = future.result()
    except HTTPError as exc:
        logger.debug("Abandoned managed-config fetch returned HTTP %s", exc.code)
        exc.close()
        return
    except (HTTPException, OSError, ValueError) as exc:
        logger.debug("Abandoned managed-config fetch failed (%s)", type(exc).__name__)
        return
    logger.debug("Abandoned managed-config fetch completed after the deadline")
    response.close()


# [해설] 같은 URL로의 이전 연결이 아직 안 끝났을 때의 예외. 목적지(URL)별로 슬롯을
# [해설]   하나만 허용해 멈춘 서버가 스레드를 쌓지 못하게 한다.
class _RemoteOpenInProgressError(OSError):
    """Raised when an unfinished open to the same destination is still stuck.

    The slot is keyed by destination URL, so this means "the previous fetch of
    this source has not returned", never "some other source is slow": a
    stalled host must not block a descriptor the administrator has since
    pointed at a healthy one.
    """


def _run_remote_open(
    future: Future[HTTPResponse],
    opener: _RemoteOpener,
    request: Request,
) -> None:
    """Publish one blocking opener result to its waiting caller."""
    try:
        response = opener.open(
            request,
            timeout=REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS,
        )
    except BaseException as exc:  # See the comment below.
        # Every exception, not only the wire-level ones. An exception this
        # worker does not hand back leaves the future pending forever: the
        # caller reports the administrator's server as timed out, and because
        # `_release_remote_open` runs as a done callback the destination's open
        # slot stays occupied for the process lifetime -- wedging every later
        # fetch of a source that has since recovered.
        future.set_exception(exc)
        if not isinstance(exc, Exception):
            raise
    else:
        future.set_result(response)


def _release_remote_open(destination: str, future: Future[HTTPResponse]) -> None:
    """Release the destination's worker slot after its blocking open returns."""
    with _remote_open_lock:
        if _remote_open_requests.get(destination) is future:
            del _remote_open_requests[destination]


# [해설] 블로킹 연결 열기를 데몬 스레드에 맡기고 Future로 돌려준다. 호출자는 데드라인
# [해설]   안에서만 기다린다.
def _start_remote_open(
    opener: _RemoteOpener,
    request: Request,
) -> Future[HTTPResponse]:
    """Start one bounded blocking open, rejecting overlap to its destination.

    Slots are keyed by `request.full_url` (the normalized HTTPS source): one
    stalled host blocks only retries of that same URL, so a descriptor change
    to a healthy source still recovers while the abandoned worker runs on.
    Live slots stay bounded because each remote source gets exactly one.

    Returns:
        Future carrying the response opened by the bounded worker.

    Raises:
        _RemoteOpenInProgressError: If an earlier open to the same destination
            has not returned.
        OSError: If the daemon worker cannot be started.
    """
    from concurrent.futures import Future
    from functools import partial
    from threading import Thread

    destination = request.full_url
    with _remote_open_lock:
        active = _remote_open_requests.get(destination)
        if active is not None and not active.done():
            msg = "the previous fetch of this source has not returned yet"
            raise _RemoteOpenInProgressError(msg)
        future: Future[HTTPResponse] = Future()
        _remote_open_requests[destination] = future
        future.add_done_callback(partial(_release_remote_open, destination))
    worker = Thread(
        target=_run_remote_open,
        args=(future, opener, request),
        name="managed-config-open",
        daemon=True,
    )
    try:
        worker.start()
    except RuntimeError as exc:
        msg = "remote source open worker could not start"
        future.set_exception(OSError(msg))
        raise OSError(msg) from exc
    return future


# [해설] 연결 열기가 데드라인을 넘기면 TimeoutError를 던진다. 늦게 완료된 응답은
# [해설]   콜백으로 닫아 소켓이 새지 않게 한다.
def _open_response_with_deadline(
    opener: _RemoteOpener,
    request: Request,
    *,
    deadline: float,
) -> HTTPResponse:
    """Open a response without letting blocking setup exceed the deadline.

    Returns:
        The opened response.

    Raises:
        TimeoutError: If setup does not finish within the deadline.
    """
    future = _start_remote_open(opener, request)
    try:
        response = future.result(timeout=_remaining_timeout(deadline))
    except TimeoutError:
        future.add_done_callback(_close_completed_response)
        raise
    try:
        _fail_if_expired(deadline)
    except TimeoutError:
        response.close()
        raise
    return response


class _RemotePolicyRejectedError(ValueError):
    """Raised when this module rejects a response it did read.

    `load()` maps it to `CORRUPT`, which tells the administrator to repair the
    published document. Only rejections this module authors may carry that
    instruction: a `ValueError` from a library on the same path (a closed
    `http.client` file, for one) has nothing to do with the document, and
    sending an administrator to edit a byte-perfect file is worse than saying
    the fetch failed.
    """


class _TimeoutNotEnforceableError(OSError):
    """Raised when a live response exposes no socket to time out.

    Its own class, not a bare `OSError`, because the two mean different things
    to an operator: this one says the read deadline is not being enforced on
    this connection -- an internal invariant break -- while every sibling
    `OSError` on this path is a network fault on the administrator's server.
    Subclasses `OSError` so any handler that misses it still fails closed.
    """


def _response_socket(response: HTTPResponse) -> _TimeoutSocket | None:
    """Return the CPython socket behind a live response, if it is reachable.

    `fp.raw._sock` is a CPython internal, so a layout change can make this
    return `None` where a socket used to be reachable. Callers that treat that
    as an invariant break must say so themselves.

    Returns:
        The socket, or `None` when the response is exhausted or the internal
        layout no longer exposes one.
    """
    if response.fp is None:
        return None
    raw = getattr(response.fp, "raw", None)
    return cast("_TimeoutSocket | None", getattr(raw, "_sock", None))


def _set_response_timeout(response: HTTPResponse, timeout: float) -> None:
    """Tighten the active response socket to the remaining timeout.

    A response with no `fp` is already exhausted: `read1` returns `b""` and
    `_parse_remote_toml` rejects the empty body, so there is nothing to bound.

    Raises:
        _TimeoutNotEnforceableError: If a live response has no controllable socket.
    """
    if response.fp is not None:
        sock = _response_socket(response)
        # Check the method too, not just for `None`: a layout change that
        # leaves `_sock` holding some other object would make the `cast` a lie
        # and raise `AttributeError`, which no arm of `load()` catches, so it
        # would escape as a traceback and bypass the `ManagedConfigError` exit
        # path.
        if sock is None or not hasattr(sock, "settimeout"):
            msg = "remote source read timeout could not be enforced"
            raise _TimeoutNotEnforceableError(msg)
        sock.settimeout(timeout)


def _abort_response_read(response: HTTPResponse) -> None:
    """Close a response after stopping its in-flight socket operation."""
    from contextlib import suppress
    from socket import SHUT_RDWR

    sock = _response_socket(response)
    if sock is not None:
        with suppress(OSError):
            sock.shutdown(SHUT_RDWR)
    response.close()


def _read_response_chunk(
    response: HTTPResponse,
    size: int,
    *,
    deadline: float,
) -> bytes:
    """Read at most one socket chunk behind an absolute-time wait.

    Returns:
        The next available body chunk.

    Raises:
        OSError: If the daemon read worker cannot be started.
        TimeoutError: If the absolute fetch deadline expires during the read.
    """
    from concurrent.futures import Future
    from threading import Thread

    timeout = _remaining_timeout(deadline)
    _set_response_timeout(response, timeout)
    future: Future[bytes] = Future()

    def read_chunk() -> None:
        try:
            future.set_result(response.read1(size))
        except BaseException as exc:  # See `_run_remote_open`.
            future.set_exception(exc)
            if not isinstance(exc, Exception):
                raise

    worker = Thread(target=read_chunk, name="managed-config-read", daemon=True)
    try:
        worker.start()
    except RuntimeError as exc:
        msg = "remote source read worker could not start"
        raise OSError(msg) from exc
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        _abort_response_read(response)
        raise


_ALLOWED_REMOTE_MEDIA_TYPES = frozenset(
    {"application/toml", "text/plain", "application/octet-stream"}
)


def _unexpected_payload_encoding(response: HTTPResponse) -> str | None:
    """Reject a 200 whose body cannot be the TOML policy that was requested.

    The request sends `Accept: application/toml`, but nothing obliges a server
    to honour it. A captive portal, an SSO interstitial, or a gateway error
    page all answer 200 with HTML, and `urllib` does not decode a body that
    arrives compressed. Without this check both reach `tomllib` and report
    `CORRUPT`, which tells the administrator to repair a document that is
    fine -- and hides that something answered in its place.

    Not a complete guard, and it must not be documented as one. A response
    carrying no `Content-Type`, or one that claims `application/octet-stream`,
    is read: a bare object store answers that way, and refusing it would
    reject a correctly published policy. Something answering in its place with
    either shape still reaches `tomllib` and reports `CORRUPT`.

    Returns:
        A safe operator-facing detail, or `None` when the body may be read.
    """
    encoding = (response.headers.get("Content-Encoding") or "").strip().lower()
    if encoding and encoding != "identity":
        return "remote source sent a compressed body"
    content_type = response.headers.get("Content-Type")
    if content_type is None:
        return None
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not media_type or media_type in _ALLOWED_REMOTE_MEDIA_TYPES:
        return None
    # The media type is echoed because it is the only way to tell an
    # administrator what answered instead. It is server-controlled, so it is
    # bounded and stripped of anything that could reflow operator output.
    safe = "".join(char for char in media_type if char.isascii() and char.isprintable())
    return f"remote source returned {safe[:64]!r}, not TOML"


def _declared_body_length(response: HTTPResponse) -> int | None:
    """Return the body length this response's HTTP framing promises.

    Returns:
        The declared length, or `None` when chunked framing delimits the body.

    Raises:
        _RemotePolicyRejectedError: If framing cannot show that the body arrived
            whole, or the declared length is over the fixed limit.
    """
    transfer_encodings = response.headers.get_all("Transfer-Encoding") or ()
    # Trust the framing mode `http.client` actually selected, not a looser
    # token search over the header. It decodes only one exact `chunked` value;
    # accepting combinations or whitespace variants here would treat a raw,
    # connection-delimited body as self-delimiting.
    chunked = (
        response.chunked
        and len(transfer_encodings) == 1
        and transfer_encodings[0].lower() == "chunked"
    )
    if transfer_encodings and not chunked:
        msg = "remote source sent an unsupported transfer encoding"
        raise _RemotePolicyRejectedError(msg)
    # `get_all` rather than `get`: a repeated header returns only its first
    # value, and `http.client` reads it the same way. `Content-Length: 200`
    # followed by `Content-Length: 5000` would then agree with itself at 200
    # bytes and accept the prefix of a 5000-byte policy as a whole document --
    # the exact truncation the rest of this function exists to reject.
    declared_lengths = {
        value.strip() for value in response.headers.get_all("Content-Length") or ()
    }
    if len(declared_lengths) > 1:
        msg = "remote source sent conflicting body lengths"
        raise _RemotePolicyRejectedError(msg)
    if declared_lengths and chunked:
        # `http.client` honours chunked framing and ignores the length, so the
        # two disagree about where the body ends. Neither reading is safe to
        # guess at on this boundary.
        msg = "remote source sent conflicting body framing"
        raise _RemotePolicyRejectedError(msg)
    if not declared_lengths:
        if chunked:
            # Chunked framing detects its own truncation: `http.client` raises
            # `IncompleteRead` when the terminating chunk never arrives.
            return None
        # Connection-close framing cannot distinguish a complete policy from
        # one cut short, and TOML truncated at a line boundary still parses --
        # a deny list can lose its entries and still look healthy.
        msg = "remote source did not delimit the response body"
        raise _RemotePolicyRejectedError(msg)
    msg = "remote source declared an invalid body length"
    try:
        declared = int(next(iter(declared_lengths)))
    except ValueError as exc:
        raise _RemotePolicyRejectedError(msg) from exc
    if declared < 0:
        raise _RemotePolicyRejectedError(msg)
    if declared > REMOTE_MANAGED_CONFIG_MAX_BYTES:
        msg = "remote source response exceeds the size limit"
        raise _RemotePolicyRejectedError(msg)
    return declared


# [해설] 본문을 청크로 읽으며 선언된 길이와 1MiB 상한을 모두 검사한다. 잘린 TOML도
# [해설]   문법상 파싱될 수 있어 부분 정책이 적용되는 것을 막기 위함이다.
def _read_limited_response(
    response: HTTPResponse,
    *,
    deadline: float,
) -> bytes:
    """Read one response without crossing the size or time boundary.

    Returns:
        The bounded response body.

    The read helpers it calls also raise `TimeoutError` once the end-to-end
    deadline expires, and `_TimeoutNotEnforceableError` when the deadline
    cannot be applied to the socket.

    Raises:
        IncompleteRead: If HTTP framing reports a truncated response.
        _RemotePolicyRejectedError: If the declared or actual body breaks the
            framing or size limits.
    """
    declared_length = _declared_body_length(response)
    chunks: list[bytes] = []
    remaining = REMOTE_MANAGED_CONFIG_MAX_BYTES + 1
    while remaining > 0:
        chunk = _read_response_chunk(
            response,
            min(_READ_CHUNK_SIZE, remaining),
            deadline=deadline,
        )
        _fail_if_expired(deadline)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > REMOTE_MANAGED_CONFIG_MAX_BYTES:
        msg = "remote source response exceeds the size limit"
        raise _RemotePolicyRejectedError(msg)
    if declared_length is not None and len(payload) != declared_length:
        if len(payload) > declared_length:
            msg = "remote source sent more than its declared body length"
            raise _RemotePolicyRejectedError(msg)
        from http.client import IncompleteRead

        raise IncompleteRead(payload, declared_length - len(payload))
    return payload


# [해설] 받은 바이트를 UTF-8 TOML로 파싱한다. 형식이 틀리면 CORRUPT 상태의 빈 스냅샷을
# [해설]   돌려준다.
def _parse_remote_toml(
    payload: bytes,
    *,
    name: str,
    path: Path | None,
    source: str,
) -> TomlSnapshot:
    """Parse one remote policy body into a managed snapshot.

    Returns:
        A readable policy snapshot or a safe corrupt-source status.
    """
    try:
        data = tomllib.loads(payload.decode("utf-8"))
    except UnicodeDecodeError:
        return _remote_status(
            name,
            path,
            ProviderHealth.CORRUPT,
            "remote source is not UTF-8",
            source,
        )
    except tomllib.TOMLDecodeError, RecursionError:
        return _remote_status(
            name,
            path,
            ProviderHealth.CORRUPT,
            "remote source contains invalid TOML",
            source,
        )
    if not data:
        # An empty local file is a deliberate administrator act. Here the
        # descriptor asserts that this URL holds the whole managed policy, so a
        # document with no keys contradicts it -- a botched publish, a
        # truncated object store overwrite, or an edge answering while the
        # origin is down. Reading it as "nothing is enforced" would let any of
        # those evict live policy.
        return _remote_status(
            name,
            path,
            ProviderHealth.CORRUPT,
            "remote source declared no policy",
            source,
        )
    if "managed_config" in data:
        return _remote_status(
            name,
            path,
            ProviderHealth.CORRUPT,
            "remote policy must not declare [managed_config]",
            source,
        )
    return TomlSnapshot(
        data,
        ProviderStatus(
            name,
            path,
            ProviderHealth.OK,
            None,
            remote_source=source,
        ),
    )


# [해설] 관리형 정책을 원격 HTTPS URL에서 읽는 provider. 로컬 관리형 파일은 URL만 담은
# [해설]   '기술자'이고 실제 정책은 여기서 받아온다(service.py의
# [해설]   _remote_managed_snapshot이 호출).
@dataclass(frozen=True, slots=True)
class RemoteTomlProvider:
    """Managed TOML provider backed by one administrator-selected HTTPS URL."""

    name: str
    source: str
    path: Path | None

    def _status(
        self,
        health: ProviderHealth,
        detail: str,
        source: str | None = None,
    ) -> TomlSnapshot:
        """Build an empty snapshot carrying this provider's identity.

        Args:
            health: Failure class driving the remediation each surface prints.
            detail: Operator-facing cause, already safe to render.
            source: Validated URL, or `None` while it is still unvalidated.

        Returns:
            An unhealthy snapshot that declares no policy.
        """
        return _remote_status(self.name, self.path, health, detail, source)

    def load(self) -> TomlSnapshot:
        """Fetch, bound, and parse the remote managed policy.

        Returns:
            The parsed policy or a safe provider failure status.
        """
        from http.client import HTTPException
        from urllib.error import HTTPError, URLError
        from urllib.request import Request

        # [해설][흐름] 1) URL을 HTTPS 전용으로 검증한다(자격증명·쿼리·프래그먼트가
        # [해설]   있으면 거부).
        try:
            source = _validate_remote_url(self.source)
        except ValueError as exc:
            return self._status(ProviderHealth.CORRUPT, str(exc))
        request = Request(  # noqa: S310  # `_validate_remote_url` permits HTTPS only
            source,
            headers={"Accept": "application/toml"},
        )
        # [해설][흐름] 2) 프록시 우회·리다이렉트 거부 opener를 만들고 5초 데드라인을
        # [해설]   건다. 3) 응답 상태가 정확히 200이어야 한다(204/206 부분 응답 거부).
        # [해설]   4) 인코딩 확인 후 크기 상한 안에서 본문을 읽는다.
        opener = _build_remote_opener()
        deadline = time.monotonic() + REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS
        try:
            with _open_response_with_deadline(
                opener,
                request,
                deadline=deadline,
            ) as response:
                # `urllib` only raises for a status outside 200..299, so a
                # `204` or a `206` reaches here as a success. A partial policy
                # is the dangerous one: TOML cut at a line boundary parses, so
                # a deny list can silently lose its entries.
                if response.status != _HTTP_OK:
                    return self._status(
                        ProviderHealth.UNREADABLE,
                        f"remote source returned HTTP {response.status}",
                        source,
                    )
                unexpected = _unexpected_payload_encoding(response)
                if unexpected is not None:
                    return self._status(ProviderHealth.UNREADABLE, unexpected, source)
                payload = _read_limited_response(response, deadline=deadline)
        # [해설][흐름] 5) 모든 실패(HTTP 오류, 타임아웃, TLS·DNS 오류 등)는 예외로 새지
        # [해설]   않고 UNREADABLE/CORRUPT 상태의 빈 스냅샷으로 바뀐다.
        # [해설]   예외 순서가 중요하다: 구체적인 사설 예외를 OSError 처리보다 먼저 둬야
        # [해설]   원인이 가려지지 않는다.
        except HTTPError as exc:
            try:
                # Only the statuses `HTTPRedirectHandler` would have followed
                # are redirects this fetch refused. A `300`, `304`, or `305`
                # also arrives as an `HTTPError` in the 3xx range, and naming
                # those a refused redirect would send an administrator looking
                # for a `Location` header their server never sent.
                detail = (
                    f"remote source refused a redirect (HTTP {exc.code})"
                    if exc.code in _REFUSED_REDIRECT_STATUSES
                    else (f"remote source returned HTTP {exc.code}")
                )
                return self._status(ProviderHealth.UNREADABLE, detail, source)
            finally:
                exc.close()
        except TimeoutError:
            return self._status(ProviderHealth.UNREADABLE, _TIMED_OUT_DETAIL, source)
        except _TimeoutNotEnforceableError as exc:
            # Ahead of the `OSError` arm below, which would render this as the
            # generic "could not be read (OSError)" and lose the one detail
            # that matters: the read deadline is not being applied to this
            # connection, which is a fault here and not on the policy host.
            return self._status(ProviderHealth.UNREADABLE, str(exc), source)
        except _RemoteOpenInProgressError:
            # Ahead of the `OSError` arm, which would render this as
            # "could not be read (_RemoteOpenInProgressError)" -- a private
            # class name -- and append the "verify that the source is
            # reachable" remediation for a source that may be answering
            # perfectly well. The stuck fetch is this process's state.
            return self._status(
                ProviderHealth.UNREADABLE,
                "an earlier fetch of the remote source has not returned yet",
                source,
            )
        except _RemotePolicyRejectedError as exc:
            # Ahead of the `OSError` arm too: `_TimeoutNotEnforceableError` is an
            # `OSError`, and this one is the only `ValueError` that may claim
            # the published document is at fault.
            return self._status(ProviderHealth.CORRUPT, str(exc), source)
        except (HTTPException, URLError, OSError) as exc:
            # `http.client.HTTPException` derives from `Exception`, not
            # `OSError`, so every wire-level parse failure it covers --
            # `IncompleteRead` from a truncated chunked body, `BadStatusLine`,
            # `LineTooLong` -- escapes the other arms. Catching one subclass
            # leaves the siblings to propagate as a traceback, which bypasses
            # the `ManagedConfigError` exit path and crashes `doctor`.
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                # `urllib` wraps a connect-phase stall as `URLError`, which is
                # not a `TimeoutError`, so a blackholed host would otherwise
                # report the generic read failure.
                return self._status(
                    ProviderHealth.UNREADABLE, _TIMED_OUT_DETAIL, source
                )
            # Name the failure class. One fixed string cannot separate an
            # expired certificate on the policy host -- the most
            # security-relevant failure on this boundary -- from a DNS miss, a
            # refused connection, or a local descriptor exhaustion that has
            # nothing to do with the administrator's server. The class name
            # carries no untrusted text: it is a type, not server output.
            return self._status(
                ProviderHealth.UNREADABLE,
                f"remote source could not be read ({type(reason).__name__})",
                source,
            )
        except ValueError as exc:
            # Not `_RemotePolicyRejectedError`, so this came from a library on the
            # fetch path rather than from a check in this module. It says
            # nothing about the published document, so it must not carry the
            # "repair the document" remediation `CORRUPT` selects, and its
            # message is not ours to forward.
            return self._status(
                ProviderHealth.UNREADABLE,
                f"remote source could not be read ({type(exc).__name__})",
                source,
            )
        # [해설][흐름] 6) 성공한 본문을 TOML로 파싱해 스냅샷으로 만든다.
        return _parse_remote_toml(
            payload,
            name=self.name,
            path=self.path,
            source=source,
        )


# [해설] 불변(frozen) provider 안에서 바뀌는 스냅샷을 담는 가변 칸. value=마지막 사용
# [해설]   가능 스냅샷, failure=최근 reload 실패 상태.
@dataclass(slots=True)
class _TomlSnapshotState:
    """Mutable snapshot cell owned by a frozen provider."""

    value: TomlSnapshot | None = None
    """Last usable snapshot, or the empty failed snapshot from an initial read."""

    failure: ProviderStatus | None = None
    """Status of the most recent failed reload, kept for health reporting."""


# [해설] 로컬 TOML 파일 provider. 관리형(MANAGED_RANK)과 사용자(USER_RANK) 둘 다 이
# [해설]   클래스다. rank·durable·loader로 구분한다.
@dataclass(frozen=True, slots=True)
class TomlFileProvider:
    """Ranked provider backed by one local TOML file snapshot."""

    name: str
    path: Path | None
    """File this provider reads, or `None` for a snapshot with no known origin.

    A `None` path is not a filename to guess at. Inventing one would make
    `load` read a relative path against the process working directory, and for
    the managed tier that is a trust boundary, not a cosmetic default.
    """

    rank: int = USER_RANK
    durable: bool = True
    snapshot: TomlSnapshot | None = field(default=None, repr=False, compare=False)
    loader: Callable[[], TomlSnapshot] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _state: _TomlSnapshotState = field(
        default_factory=_TomlSnapshotState,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Seed the mutable snapshot cell when a generation is supplied."""
        self._state.value = self.snapshot

    # [해설] 파일을 읽어 상태를 분류한다: 경로 없음=INDETERMINATE, 없음=MISSING, 권한 등
    # [해설]   OSError=UNREADABLE, 문법 오류=CORRUPT, 정상=OK.
    # [해설]   읽지 못한 경우에도 예외 대신 빈 표 + 상태를 돌려준다.
    def load(self) -> TomlSnapshot:
        """Parse the file and classify missing, unreadable, or corrupt states.

        Returns:
            Parsed data and provider health. A provider with no path reports
            `INDETERMINATE`, which is not usable: an empty read proves nothing
            about a file whose location is unknown.
        """
        if self.path is None:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    None,
                    ProviderHealth.INDETERMINATE,
                    "no path is known for this source, so it cannot be re-read",
                ),
            )
        try:
            with self.path.open("rb") as handle:
                data = tomllib.load(handle)
        except FileNotFoundError:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.MISSING,
                ),
            )
        except OSError as exc:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.UNREADABLE,
                    type(exc).__name__,
                ),
            )
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            detail = (
                "not UTF-8 encoded" if isinstance(exc, UnicodeDecodeError) else str(exc)
            )
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.CORRUPT,
                    detail,
                ),
            )
        if not isinstance(data, dict):
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.CORRUPT,
                    "top-level TOML value is not a table",
                ),
            )
        return TomlSnapshot(
            data,
            ProviderStatus(self.name, self.path, ProviderHealth.OK),
        )

    # [해설] 현재 스냅샷에서 옵션을 읽는다. 파일 전체가 거부된 상태라면 그 사실을
    # [해설]   diagnostics로 붙여, '선언 안 함'으로 오인되지 않게 한다.
    def get[T](self, option: ConfigOption[T]) -> RankedProviderValue[T]:
        """Read one option from the current file snapshot.

        Args:
            option: Manifest option to read.

        Returns:
            Ranked and coerced provider result.
        """
        from deepagents_code.config_manifest import OptionKind

        snapshot = self.current_snapshot()
        if option.kind is OptionKind.THEME_DELEGATE:
            ranked = _ranked_for(
                option,
                ranked_theme_toml_value(
                    snapshot.data,
                    rank=self.rank,
                    durable=self.durable,
                    status=snapshot.status,
                ),
            )
        else:
            ranked = ranked_toml_value(
                option,
                snapshot.data,
                rank=self.rank,
                durable=self.durable,
                status=snapshot.status,
            )
        # The status of the file on disk, which is not the status of the
        # snapshot resolution just read: a failed reload keeps enforcing the
        # last readable generation while `failure` records why the current
        # contents were refused.
        failure = self._state.failure
        return self._with_rejection_diagnostic(
            ranked,
            failure or snapshot.status,
            # Values were retained only when the generation in hand is itself
            # usable. A first read that fails records a failure too, but there
            # is no earlier generation behind it - those options fall back.
            retained=snapshot.status.usable,
        )

    @staticmethod
    def _with_rejection_diagnostic[T](
        ranked: RankedProviderValue[T],
        status: ProviderStatus,
        *,
        retained: bool,
    ) -> RankedProviderValue[T]:
        """Attach the reason when the file on disk was rejected as a whole.

        Neither rejection is visible in the result otherwise. A file refused on
        first read coerces to `Unset` for every option, which resolution reads
        as "this source declares nothing" — indistinguishable from a file that
        omits the key. A file refused on *reload* is quieter still: resolution
        keeps returning the previous generation's values, so nothing looks
        wrong at all while the user's latest edit silently fails to apply.

        Args:
            ranked: Result already coerced from the snapshot.
            status: Health of the file on disk.
            retained: Whether resolution is still serving an earlier generation
                rather than falling through to lower tiers.

        Returns:
            The result, with a rejection diagnostic when the source is unusable.
        """
        if status.usable:
            return ranked
        location = f" ({status.path})" if status.path is not None else ""
        detail = f": {status.detail}" if status.detail else ""
        suffix = RETAINED_SOURCE_SUFFIX if retained else UNUSABLE_SOURCE_SUFFIX
        reason = (
            f"Ignoring {status.name}{location} — it is "
            f"{status.health.value}{detail} {suffix}"
        )
        return replace(ranked, diagnostics=(reason, *ranked.diagnostics))

    def status(self) -> ProviderStatus:
        """Return health for the current file snapshot.

        A failed reload reports its own health even though resolution keeps
        using the retained snapshot: diagnostics must describe the file on
        disk, not the generation still being enforced.

        Raises:
            RuntimeError: If a reload produces no snapshot.
        """
        state = self._state
        if state.value is None:
            self.reload()
        if state.failure is not None:
            return state.failure
        snapshot = state.value
        if snapshot is None:
            msg = f"{self.name} reload produced no snapshot"
            raise RuntimeError(msg)
        return snapshot.status

    # [해설][주의] reload는 사용 불가한 새 스냅샷으로 직전의 정상 스냅샷을 교체하지
    # [해설]   않는다. 빈 표로 교체하면 제한이 사라져 약한 계층이 이기기 때문이다.
    # [해설]   실패는 failure에만 기록해 진단 화면이 디스크 파일의 상태를 보여 주게
    # [해설]   한다.
    def reload(self) -> None:
        """Replace the current snapshot with a fresh file read.

        A reload the source cannot use never replaces the last usable
        snapshot. An unusable candidate carries an empty table, which
        resolution reads as "this source declares nothing"; installing it
        would drop the source's restrictions and let lower ranks win. The
        failed status is still recorded so health surfaces report the file on
        disk.
        """
        snapshot = self.loader() if self.loader is not None else self.load()
        self.reload_from_snapshot(snapshot)

    def reload_from_snapshot(self, snapshot: TomlSnapshot) -> None:
        """Install an already-read candidate with normal retention semantics.

        Args:
            snapshot: Candidate loaded before the resolver generation lock.
        """
        if snapshot.status.usable:
            self._state.value = snapshot
            self._state.failure = None
        else:
            if self._state.value is None:
                self._state.value = snapshot
            self._state.failure = snapshot.status

    def current_snapshot(self) -> TomlSnapshot:
        """Return the cached snapshot, loading it on first access.

        Public so `ConfigResolver.toml_snapshot` can hand the generation this
        provider is serving to a caller building a masked variant resolver.

        Returns:
            Current parsed file snapshot.

        Raises:
            RuntimeError: If a reload produces no snapshot.
        """
        if self._state.value is None:
            self.reload()
        snapshot = self._state.value
        if snapshot is None:
            msg = f"{self.name} reload produced no snapshot"
            raise RuntimeError(msg)
        return snapshot


# [해설] 환경변수 provider. 호출 때마다 활성 환경(active_environment)을 직접 읽어 캐시가
# [해설]   없다. durable은 항상 False(프로세스가 끝나면 사라짐).
@dataclass(frozen=True, slots=True)
class EnvProvider:
    """Live process-environment configuration provider."""

    name: str = "environment"
    rank: int = ENVIRONMENT_RANK
    environ: Mapping[str, str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def durable(self) -> bool:
        """Never durable: the environment does not survive the process.

        A property rather than a field because the coercion helpers below stamp
        durability onto every value they emit. A settable field would
        type-check, enter `__eq__`, and change nothing about masking - a lie in
        the one attribute that decides whether a tier can hide another.
        """
        return False

    def get[T](self, option: ConfigOption[T]) -> RankedProviderValue[T]:
        """Read one option from the live environment.

        Args:
            option: Manifest option to read.

        Returns:
            Ranked and coerced provider result.
        """
        from deepagents_code.config import active_environment
        from deepagents_code.config_manifest import OptionKind

        environ = active_environment() if self.environ is None else self.environ
        if option.kind is OptionKind.THEME_DELEGATE:
            return _ranked_for(
                option,
                ranked_theme_environment_value(environ, rank=self.rank),
            )
        return ranked_environment_value(option, environ, rank=self.rank)

    def status(self) -> ProviderStatus:
        """Return the always-healthy environment provider status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Keep reading the live environment without cached state."""


# [해설] 매니페스트 기본값 provider. 항상 마지막 순위(DEFAULT_RANK)이고 durable이 True라
# [해설]   어떤 계층도 못 이길 때만 쓰인다.
@dataclass(frozen=True, slots=True)
class DefaultProvider:
    """Typed manifest-default configuration provider."""

    name: str = "default"
    rank: int = DEFAULT_RANK

    @property
    def durable(self) -> bool:
        """Always durable: manifest defaults are compiled into the process.

        A property for the same reason as `EnvProvider.durable`: the value the
        helpers stamp on each result is the truth, so the attribute must not be
        able to disagree with it.
        """
        return True

    def get[T](self, option: ConfigOption[T]) -> RankedProviderValue[T]:
        """Return one option's manifest default.

        Args:
            option: Manifest option whose default should be returned.

        Returns:
            Ranked default provider result.
        """
        ranked = ranked_default_value(option, rank=self.rank)
        if isinstance(ranked.result, Unset):
            return replace(ranked, result=_found_for(option, option.default))
        return ranked

    def status(self) -> ProviderStatus:
        """Return the always-healthy default provider status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Retain immutable manifest defaults without cached state."""
