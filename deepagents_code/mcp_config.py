"""Validation and environment-variable expansion for MCP server config.

Resolves `${VAR}` and `${VAR:-default}` references in the supported
configuration fields (`command`, `url`, `args`, `env`, `headers`, `oauth`) and
validates their types. A `${VAR:-default}` reference falls back to
`default` when `VAR` is unset *or* empty (POSIX `:-` semantics).
"""

# [해설] 이 모듈은 MCP 서버 설정 하나에 들어 있는 ${VAR} / ${VAR:-기본값} 참조를
# [해설]   환경변수 값으로 치환하고 필드 타입을 검증한다.
# [해설] 호출 시점: mcp_tools.py의 _load_tools_from_config 안 _prepare_transport가
# [해설]   서버를 활성화할 때, 서버마다 resolve_mcp_server_env를 부른다.
# [해설]   파일을 읽을 때가 아니라 서버를 켤 때 치환하므로 환경변수 하나가 없어도 그
# [해설]   서버만 실패하고 나머지는 계속 로드된다.
# [해설] 치환 대상 필드: command, url, args(목록), env, headers, oauth(딕셔너리). 그
# [해설]   밖의 필드는 그대로 복사한다.
# [해설] 문법: 중괄호가 있는 형태만 치환한다. $VAR나 단독 $는 그대로 둔다.
# [해설]   ${VAR-default}처럼 틀린 형태는 조용히 통과시키지 않고 오류를 낸다.
# [해설] 값이 비어 있으면(POSIX :- 규칙) 기본값이 쓰이고, ${VAR}가 빈 문자열로 설정돼
# [해설]   있으면 빈 값을 그대로 낸다. 미설정이고 기본값도 없을 때만 RuntimeError.
# [해설] 환경변수는 os.environ이 아니라 deepagents_code.config의
# [해설]   active_environment()에서 읽는다(프로필·.env 계층을 반영한다, 추정).
# [해설][주의] 오류 메시지에는 변수 이름만 넣고 값은 넣지 않는다(비밀 노출 방지). Slack
# [해설]   OAuth 설정은 값 내용을 노출하지 않고 형식만 검사한다.
from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Mapping

# [해설] 허용 형태의 정규식. 그룹1=변수명, 그룹2=:- 뒤의 기본값(없으면 None).
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^{}]*))?\}")
"""Matches a supported reference: `${VAR}` or `${VAR:-default}`.

Group 1 is the variable name; group 2 (present only for the `:-` form) is
the default. A bare `$VAR` and a literal `$` are intentionally not matched.
"""

_ENV_BRACE_RE = re.compile(r"\$\{")
"""Matches a `${` brace-open, used to catch malformed `${...}` references."""


# [해설] 문자열 하나를 치환한다. 아래 replace가 한 참조의 해석 순서를 정한다: 값이
# [해설]   있으면 사용 -> 기본값이 있으면 기본값 -> 빈 값이면 빈 값 -> 아니면 오류.
# [해설]   그 뒤 형식이 틀린 ${...}는 치환 결과가 아니라 원문에서 찾아 거부한다(치환된
# [해설]   값에 ${가 있어도 오류가 되지 않게).
def _interpolate_env(value: str, *, field: str) -> str:
    """Expand `${VAR}` / `${VAR:-default}` references in one config string.

    A bare `$VAR` (no braces) and a literal `$` pass through untouched;
    only the braced forms expand. `${VAR:-default}` uses `default` when
    `VAR` is unset or empty. A `${...}` that does not parse as one of the
    supported forms (e.g. `${VAR-default}` or an unterminated `${VAR`) is
    rejected rather than silently emitted, so a typo cannot inject a
    garbage value into a URL, command, or header.

    Args:
        value: Raw configuration string.
        field: Fully qualified field path for error messages.

    Returns:
        The interpolated string.

    Raises:
        RuntimeError: If a required environment variable is unset, or the
            string contains a malformed `${...}` reference.
    """
    from deepagents_code.config import active_environment

    environ = active_environment()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        resolved = environ.get(name)
        # A non-empty value always wins, for both `${VAR}` and `${VAR:-default}`.
        if resolved:
            return resolved
        # `resolved` is now "" (set but empty) or None (unset).
        if default is not None:
            # `${VAR:-default}`: `:-` falls back for empty *and* unset (POSIX).
            return default
        if resolved is not None:
            # `${VAR}` set to "": no default, so emit the empty value.
            return resolved
        # `${VAR}` unset with no default: the only hard error.
        msg = (
            f"{field} references unset env var {name}. "
            f"Set {name} in the environment or provide a default."
        )
        raise RuntimeError(msg)

    # Reject any `${` that isn't the start of a well-formed reference. The
    # check is against the raw `value` (not the substituted result) so a
    # resolved value that happens to contain `${` never trips it.
    ref_spans = [match.span() for match in _ENV_REF_RE.finditer(value)]
    for brace in _ENV_BRACE_RE.finditer(value):
        if not any(start <= brace.start() < end for start, end in ref_spans):
            msg = (
                f"{field} contains a malformed '${{...}}' reference. "
                "Use '${VAR}' or '${VAR:-default}'."
            )
            raise RuntimeError(msg)

    return _ENV_REF_RE.sub(replace, value)


# [해설] 문자열 타입을 확인하고 _interpolate_env로 넘긴다. 문자열이 아니면 TypeError.
def _resolve_string(value: object, *, field: str) -> str:
    """Validate and interpolate one string field.

    Args:
        value: Raw field value.
        field: Fully qualified field path for error messages.

    Returns:
        The validated and interpolated string.

    Raises:
        TypeError: If the field value is not a string.
    """
    if not isinstance(value, str):
        msg = f"{field} must be a string, got {type(value).__name__}"
        raise TypeError(msg)
    return _interpolate_env(value, field=field)


# [해설] 딕셔너리 필드(env, headers, oauth)의 모든 값을 같은 방식으로 치환한다. 오류
# [해설]   메시지에 mcpServers.<이름>.<필드>.<키> 경로가 들어간다.
def _resolve_mapping_values(
    values: Mapping[str, object],
    *,
    field: str,
) -> dict[str, str]:
    """Validate and interpolate string values in a mapping field.

    Args:
        values: Raw mapping values.
        field: Fully qualified field path for error messages.

    Returns:
        A new mapping with validated and interpolated values.
    """
    return {
        name: _resolve_string(value, field=f"{field}.{name}")
        for name, value in values.items()
    }


# [해설] 이 모듈의 공개 진입점. 원본을 변경하지 않고 deepcopy한 사본에서 필드별로 치환한
# [해설]   결과를 반환한다.
# [해설][흐름] 1) command, url 문자열 치환 2) args는 리스트여야 하며 항목별 치환 3) env,
# [해설]   headers, oauth는 딕셔너리여야 하며 값별 치환
# [해설]   4) 마지막에 Slack OAuth 설정 검증.
def resolve_mcp_server_env(
    server_name: str,
    server_config: Mapping[str, object],
) -> dict[str, Any]:
    """Resolve `${VAR}` references in one MCP server's supported fields.

    Interpolates the `command`, `url`, `args`, `env`, `headers`, and `oauth`
    fields (see `_interpolate_env` for the reference syntax); every other
    field is copied through verbatim. The input is not mutated.

    Args:
        server_name: Server name used in field-specific error messages.
        server_config: Raw server configuration.

    Returns:
        A resolved copy of the server configuration.

    Raises:
        TypeError: If a supported field has the wrong type — a non-string
            scalar value, or `args`/`env`/`headers` with the wrong container
            type.
        RuntimeError: If a required environment variable is unset.
        ValueError: If Slack app settings are invalid.
    """  # noqa: DOC502 - `RuntimeError` is raised by `_interpolate_env`
    resolved: dict[str, Any] = copy.deepcopy(dict(server_config))
    prefix = f"mcpServers.{server_name}"

    for name in ("command", "url"):
        if name in resolved:
            resolved[name] = _resolve_string(resolved[name], field=f"{prefix}.{name}")

    if "args" in resolved:
        args = resolved["args"]
        if not isinstance(args, list):
            msg = f"{prefix}.args must be a list, got {type(args).__name__}"
            raise TypeError(msg)
        resolved["args"] = [
            _resolve_string(value, field=f"{prefix}.args[{index}]")
            for index, value in enumerate(args)
        ]

    for name in ("env", "headers", "oauth"):
        if name not in resolved:
            continue
        values = resolved[name]
        if not isinstance(values, dict):
            msg = f"{prefix}.{name} must be a dictionary, got {type(values).__name__}"
            raise TypeError(msg)
        resolved[name] = _resolve_mapping_values(values, field=f"{prefix}.{name}")

    _validate_slack_oauth_config(server_name, resolved)
    return resolved


# [해설][주의] oauth 필드는 Slack 서버(slack.com 또는 *.slack.com) + auth: oauth
# [해설]   조합에서만 허용한다. client_id 필수, client_secret 선택이며
# [해설]   제어문자나 빈 문자열이 든 값은 거부한다. 오류에 자격증명 값을 싣지 않는다.
def _validate_slack_oauth_config(
    server_name: str, server_config: Mapping[str, object]
) -> None:
    """Reject invalid Slack app settings without exposing credential values.

    Raises:
        TypeError: If `oauth` is not a mapping.
        ValueError: If the settings or server are unsupported.
    """
    if "oauth" not in server_config:
        return
    prefix = f"mcpServers.{server_name}.oauth"
    oauth = server_config["oauth"]
    if not isinstance(oauth, dict):
        msg = f"{prefix} must be a dictionary"
        raise TypeError(msg)
    if oauth.keys() - {"client_id", "client_secret"} or "client_id" not in oauth:
        msg = f"{prefix} requires client_id and supports only an optional client_secret"
        raise ValueError(msg)
    for name, value in oauth.items():
        if (
            not isinstance(value, str)
            or not value.strip()
            or re.search(r"[\x00-\x1f\x7f]", value)
        ):
            msg = (
                f"{prefix}.{name} must be a non-empty string without control characters"
            )
            raise ValueError(msg)
    url = server_config.get("url")
    host = urlparse(url).hostname if isinstance(url, str) else None
    if server_config.get("auth") != "oauth" or not (
        host and (host == "slack.com" or host.endswith(".slack.com"))
    ):
        msg = f"{prefix} is supported only for Slack servers with auth: oauth"
        raise ValueError(msg)
