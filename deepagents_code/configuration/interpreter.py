"""Resolver-backed interpreter configuration snapshots."""
# [해설] 코드 인터프리터(`CodeInterpreterMiddleware`) 설정을 `interpreter.*` 옵션 6개로
# [해설]   모아 불변 스냅샷(`InterpreterConfig`)으로 만드는 모듈이다.
# [해설] 값은 직접 읽지 않고 `configuration/resolver.py`의 `ConfigResolver`가 관리형 >
# [해설]   CLI > 환경변수 > 사용자 TOML > 기본값 순서로 해석한 결과를 받는다.
# [해설] 옵션의 정의(키 이름, 기본값, 타입)는 `deepagents_code/config_manifest.py`의
# [해설]   매니페스트가 가지고 있고, 이 파일은 그 키를 이름으로 조회만 한다.
# [해설] [흐름] 1) `_resolved_values`가 매니페스트에서 옵션 6개를 찾는다 2) resolver가
# [해설]   한 세대(generation) 기준으로 한꺼번에 해석한다 3)
# [해설]   `InterpreterConfig.from_resolver`가 서버가 넘긴 호출 단위 override를 얹어
# [해설]   dataclass를 만든다.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option
from deepagents_code.configuration.resolver import (
    CLI_RANK,
    get_config_resolver,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.configuration.resolver import ConfigResolver, ResolvedValue

# [해설] 이 스냅샷이 다루는 매니페스트 옵션 키. 키가 하나라도 매니페스트에 없으면
# [해설]   `_resolved_values`가 `RuntimeError`를 낸다.
_INTERPRETER_KEYS = (
    "interpreter.timeout_seconds",
    "interpreter.memory_limit_mb",
    "interpreter.max_ptc_calls",
    "interpreter.max_result_chars",
    "interpreter.ptc",
    "interpreter.ptc_acknowledge_unsafe",
)


# [해설][흐름] 1) 키마다 `get_option`으로 매니페스트 항목을 찾고, 없으면 즉시 실패한다
# [해설]   2) `resolve_options`로 같은 세대 기준의 해석 결과를 한 번에 받는다 3)
# [해설]   옵션마다 `_emit_ranked_diagnostics`로 계층 충돌 진단을 남긴다.
# [해설] 옵션을 하나씩 따로 해석하지 않는 이유는 중간에 설정이 다시 로드돼 값들이 서로
# [해설]   다른 세대에서 섞이는 것을 막기 위해서다(추정: docstring의 "one consistent
# [해설]   generation" 표현 기준).
def _resolved_values(
    resolver: ConfigResolver,
) -> Mapping[str, ResolvedValue[object]]:
    """Resolve one consistent generation of interpreter options.

    Returns:
        Resolved values keyed by manifest option.

    Raises:
        RuntimeError: If an interpreter option is absent from the manifest.
    """
    options = tuple(get_option(key) for key in _INTERPRETER_KEYS)
    if any(option is None for option in options):
        msg = "interpreter options are missing from the configuration manifest"
        raise RuntimeError(msg)
    required = cast("tuple[ConfigOption, ...]", options)
    resolved = resolver.resolve_options(required)
    for option in required:
        _emit_ranked_diagnostics(option, resolved[option.key])
    return resolved


# [해설] 인터프리터 실행 한도와 PTC(프로그램 방식 도구 호출) 정책을 담는 불변 스냅샷.
# [해설]   `frozen=True, slots=True`라 한 번 만들면 바뀌지 않는다.
# [해설] `ptc`는 `str | bool | list[str]`이다.
# [해설]   `parse_interpreter_tools`(`provider.py`)가 만드는 "safe"/"all" 문자열 또는
# [해설]   도구 이름 목록이 여기로 들어온다.
@dataclass(frozen=True, slots=True)
class InterpreterConfig:
    """Configuration consumed by `CodeInterpreterMiddleware`."""

    timeout_seconds: float
    memory_limit_mb: int
    max_ptc_calls: int
    max_result_chars: int
    ptc: str | bool | list[str]
    ptc_acknowledge_unsafe: bool

    # [해설] resolver(기본값은 프로세스 공용 `get_config_resolver()`)에서 값을 읽고,
    # [해설]   서버 부모가 넘긴 호출 단위 `ptc`, `ptc_acknowledge_unsafe`를 덮어쓴다.
    @classmethod
    def from_resolver(
        cls,
        resolver: ConfigResolver | None = None,
        *,
        ptc: str | list[str] | None = None,
        ptc_acknowledge_unsafe: bool = False,
    ) -> InterpreterConfig:
        """Build a snapshot from the resolver and optional server overrides.

        Args:
            resolver: Resolver generation to read. Defaults to the process resolver.
            ptc: Invocation-scoped PTC override forwarded by the server parent.
            ptc_acknowledge_unsafe: Invocation-scoped unsafe-PTC acknowledgement.

        Returns:
            Resolved interpreter configuration for one agent build.
        """
        values = _resolved_values(resolver or get_config_resolver())
        resolved_ptc = cast("str | bool | list[str]", values["interpreter.ptc"].value)
        ptc_policy = values["interpreter.ptc"]
        acknowledgement = values["interpreter.ptc_acknowledge_unsafe"]
        # [해설][주의] `rank`는 숫자가 작을수록 우선이다(`resolver.py`: MANAGED_RANK=200
        # [해설]   < CLI_RANK=300). CLI보다 앞선 순위, 즉 관리형 정책이 이 옵션에
        # [해설]   관여했다면 호출 단위 override를 버린다.
        # [해설] 이렇게 해서 서버 부모가 넘긴 값이 관리자 정책을 뚫고 PTC를 켜는 일을
        # [해설]   막는다. 아래 `ptc_acknowledge_unsafe`도 같은 규칙이다.
        if any(rank < CLI_RANK for rank in ptc_policy.ranks):
            ptc = None
        if any(rank < CLI_RANK for rank in acknowledgement.ranks):
            ptc_acknowledge_unsafe = False
        return cls(
            timeout_seconds=cast("float", values["interpreter.timeout_seconds"].value),
            memory_limit_mb=cast("int", values["interpreter.memory_limit_mb"].value),
            max_ptc_calls=cast("int", values["interpreter.max_ptc_calls"].value),
            max_result_chars=cast("int", values["interpreter.max_result_chars"].value),
            ptc=resolved_ptc if ptc is None else ptc,
            # [해설] 호출 단위 확인(acknowledge)과 해석된 설정 중 하나라도 True면
            # [해설]   True다. 단, 위에서 관리형 정책이 관여했으면 호출 단위 값은 이미
            # [해설]   False로 꺼졌다.
            ptc_acknowledge_unsafe=(
                ptc_acknowledge_unsafe
                or bool(values["interpreter.ptc_acknowledge_unsafe"].value)
            ),
        )
