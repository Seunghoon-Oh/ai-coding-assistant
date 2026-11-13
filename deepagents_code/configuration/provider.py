"""Structural contract and CLI implementation for configuration providers."""
# [해설] 설정 제공자(ConfigProvider)의 구조적 계약(Protocol)과, 그 구현 중 CLI 인자를
# [해설]   읽는 `CliProvider`를 담은 모듈이다. 나머지 구현(환경변수, TOML 등)은
# [해설]   `configuration/providers.py`에 있다.
# [해설] 제공자 하나는 이름(`name`), 우선순위 숫자(`rank`), 지속 여부(`durable`), 옵션
# [해설]   읽기(`get`), 상태(`status`), 재로드(`reload`)를 가진다.
# [해설] `get`은 매니페스트 옵션(`config_manifest.py`의 `ConfigOption`)을 받아
# [해설]   `RankedProviderValue`(rank + 결과 `Found`/`Unset`/`Invalid`)를 돌려준다. 결과
# [해설]   타입은 `types.py`에 있다.
# [해설] 우선순위는 `resolver.py`가 숫자로 정한다. 작을수록 강하다: 관리형 200 < CLI 300
# [해설]   < 환경변수 400 < 사용자 TOML 500 < 기본값 1000.
# [해설] `CliProvider`는 argparse가 이미 파싱한 값을 불변 스냅샷으로 얼려
# [해설]   두고(`MappingProxyType`), 옵션마다 매니페스트가 지정한 `cli.dest_name`으로
# [해설]   값을 꺼낸다.
# [해설] [설계] `providers.py`의 변환 함수(`_found_for`, `coerce_toml_value` 등)는 순환
# [해설]   import를 피하려고 이 파일 안에서 함수 내부 import로 가져온다.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import argparse

    from deepagents_code.config_manifest import CliSpec, ConfigOption

from deepagents_code.configuration.resolver import CLI_RANK, RankedProviderValue
from deepagents_code.configuration.types import (
    Invalid,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    Unset,
)


# [해설] `--interpreter-tools` 플래그 문법의 단일 구현. "safe"/"all" 문자열, 쉼표 구분
# [해설]   도구 이름 목록, 또는 사유를 담은 `Invalid`를 반환한다.
# [해설] `CliProvider`는 `Invalid`를 경고로 바꾸고 다음 계층으로 넘기고, `main`은 같은
# [해설]   사유를 사용 오류(종료 코드 2)로 바꾼다. 두 호출자가 문구를 따로 복사해
# [해설]   어긋났던 문제를 한 규칙으로 합친 것이다.
def parse_interpreter_tools(raw: str) -> str | list[str] | Invalid:
    """Parse the shared `--interpreter-tools` grammar.

    One rule serves both callers. `CliProvider` frames a rejection as a
    warning and falls through to the next tier; `main` frames the same reason
    as a usage error and exits 2. Two hand-maintained copies had already
    drifted in wording.

    Args:
        raw: Raw flag text as the user typed it.

    Returns:
        The `"safe"`/`"all"` sentinel, a list of tool names, or `Invalid`
        carrying the bare reason for the caller to frame.
    """
    text = raw.strip()
    if not text:
        return Invalid(
            "requires a value: 'safe', 'all', or a comma-separated list of tool names"
        )
    normalized = text.lower()
    if normalized in {"safe", "all"}:
        return normalized
    names = [token.strip() for token in text.split(",") if token.strip()]
    if not names:
        return Invalid("list must contain at least one non-empty tool name")
    if any(name.lower() == "all" for name in names):
        return Invalid(
            "'all' cannot be combined with other tools; use 'all' on its own "
            "or list explicit tool names (optionally with the 'safe' preset)"
        )
    return names


# [해설] 설정 제공자가 만족해야 하는 구조적 계약. `@runtime_checkable`이라
# [해설]   `isinstance`로도 검사할 수 있다. 상속이 아니라 메서드 모양으로 판단한다.
@runtime_checkable
class ConfigProvider(Protocol):
    """A ranked source of typed configuration values."""

    @property
    def name(self) -> str:
        """Provider display label."""
        ...

    @property
    def rank(self) -> int:
        """Numeric precedence rank."""
        ...

    @property
    def durable(self) -> bool:
        """Whether the source survives the process."""
        ...

    def get[T](self, option: ConfigOption[T]) -> RankedProviderValue[T]:
        """Read and coerce one manifest option."""
        ...

    def status(self) -> ProviderStatus:
        """Return current provider health and display metadata."""
        ...

    def reload(self) -> None:
        """Refresh provider state when the source supports it."""
        ...


# [해설] 파싱된 CLI 인자를 읽는 제공자. `rank=CLI_RANK`(300)이고 프로세스와 함께
# [해설]   사라지므로 `durable`은 항상 False다.
# [해설] `init=False`이고 `__init__`을 직접 정의한다(필드 `args`는 `repr`에서 숨김).
@dataclass(frozen=True, slots=True, init=False)
class CliProvider:
    """Config provider backed by parsed CLI arguments."""

    args: Mapping[str, object] = field(repr=False)
    name: str = "CLI argument"
    rank: int = CLI_RANK

    # [해설][주의] `args`가 Mapping이면 그대로, 아니면 `vars(args)`로 dict를 만든다.
    # [해설]   아래 기존 주석대로 `hasattr(__dict__)`로 가르면 일부 Mapping이 잘못된
    # [해설]   분기로 가서 모든 옵션이 조용히 `Unset`이 되기 때문에 선언된 타입으로
    # [해설]   구분한다.
    def __init__(self, args: argparse.Namespace | Mapping[str, object]) -> None:
        """Snapshot an already-parsed CLI namespace.

        Args:
            args: Values produced by `argparse`, or an equivalent mapping.
        """
        # Discriminate on the declared type, not on `__dict__`: every `Mapping`
        # implementation other than `dict`/`MappingProxyType` also has one, so
        # `hasattr` sent them down the `vars()` branch and snapshotted the
        # object's attributes instead of its items -- every option then read
        # back `Unset` with no error anywhere.
        values = args if isinstance(args, Mapping) else vars(args)
        object.__setattr__(self, "args", MappingProxyType(dict(values)))
        object.__setattr__(self, "name", "CLI argument")
        object.__setattr__(self, "rank", CLI_RANK)

    # [해설] CLI 상태는 프로세스와 수명이 같다. 지속되지 않는다.
    @property
    def durable(self) -> bool:
        """Never durable: CLI state dies with this process."""
        return False

    # [해설][흐름] 1) 매니페스트의 `option.cli`가 None이면 CLI로 설정할 수 없는
    # [해설]   옵션이므로 `Unset` 2) `startup.mode`는 `_startup_mode`로 특별 처리 3) 그
    # [해설]   외는 `args`에서 `dest_name`으로 읽어 `_coerce`로 변환 4) rank, durable,
    # [해설]   status와 함께 `RankedProviderValue`로 포장.
    def get[T](self, option: ConfigOption[T]) -> RankedProviderValue[T]:
        """Read and coerce one option from the parsed namespace.

        Args:
            option: Manifest option to read.

        Returns:
            Ranked `Found`, `Unset`, or `Invalid` provider result.
        """
        status = self.status()
        spec = option.cli
        if spec is None:
            return RankedProviderValue(self.rank, self.durable, status, Unset())

        if option.key == "startup.mode":
            result = self._startup_mode(option, spec)
        else:
            raw = self.args.get(spec.dest_name)
            result = (
                Unset() if raw is None else self._coerce(option, raw, flag=spec.flag)
            )
        return RankedProviderValue(self.rank, self.durable, status, result)

    # [해설] `--auto`, `--yolo` 같은 상호 배타적 승인 플래그를 하나의 `startup.mode`
    # [해설]   값("auto"/"yolo")으로 합친다. 동반 플래그가 먼저 검사되므로 둘 다
    # [해설]   주어지면 `--yolo`가 이긴다.
    def _startup_mode[T](
        self, option: ConfigOption[T], spec: CliSpec
    ) -> ProviderResult[T]:
        """Map the mutually exclusive approval flags to one config value.

        The companion destinations are derived from the spec rather than
        spelled out here. A literal `"yolo"` made `CliSpec` the source of truth
        for the flag's *display* while this method independently re-derived the
        destination it *reads*, so an argparse `dest=` rename would silently
        degrade YOLO to `Unset` -- the same class of failure the `Mapping`
        discrimination above guards against.

        Args:
            option: Manifest option that owns the startup-mode value type.
            spec: CLI binding for `startup.mode`, carrying both flags.

        Returns:
            `Found` for an explicit mode, otherwise `Unset`.
        """
        from deepagents_code.config_manifest import CliSpec
        from deepagents_code.configuration.providers import _found_for

        # Companions are checked first: `--yolo` is the stronger mode and wins
        # if a caller somehow supplies both. Each companion flag is spelled
        # exactly like the mode it selects (`--yolo` -> `"yolo"`), which
        # `test_companion_flags_spell_their_startup_mode` pins.
        for flag in spec.companion_flags:
            if self.args.get(CliSpec(flag).dest_name) is True:
                return _found_for(option, flag.removeprefix("--"))
        if self.args.get(spec.dest_name) is True:
            return _found_for(option, "auto")
        return Unset()

    # [해설][흐름] CLI 값 하나를 매니페스트 타입으로 변환한다. 옵션 종류에 따라 분기:
    # [해설]   `threads.sort_order`(created/updated 별칭 변환), 셸 목록 위임형, PTC
    # [해설]   위임형(`parse_interpreter_tools` 사용), 구조화 값, 빈 문자열 처리, 그 외
    # [해설]   일반 TOML 변환(`coerce_toml_value`).
    @staticmethod
    def _coerce[T](
        option: ConfigOption[T], raw: object, *, flag: str
    ) -> ProviderResult[T]:
        """Coerce one already-parsed CLI value to its manifest type.

        Returns:
            Typed `Found` or an `Invalid` rejection.
        """
        from deepagents_code.config_manifest import OptionKind
        from deepagents_code.configuration.providers import (
            _found_for,
            coerce_environment_value,
            coerce_toml_value,
        )

        if option.key == "threads.sort_order":
            if raw == "created":
                return _found_for(option, "created_at")
            if raw == "updated":
                return _found_for(option, "updated_at")
            return Invalid(f"Ignoring {flag}={raw!r} (expected 'created' or 'updated')")
        if option.kind is OptionKind.SHELL_LIST_DELEGATE and isinstance(raw, str):
            return coerce_environment_value(option, raw, flag)
        if option.kind is OptionKind.PTC_DELEGATE and isinstance(raw, str):
            parsed = parse_interpreter_tools(raw)
            if isinstance(parsed, Invalid):
                return Invalid(f"Ignoring {flag}={raw!r} ({parsed.reason})")
            return coerce_toml_value(option, parsed, source=flag)
        if option.kind is OptionKind.STRUCTURED and isinstance(raw, (dict, list)):
            return _found_for(option, raw)
        # [해설][주의] CLI의 빈 문자열은 "값 없음"이다(TOML의 빈 문자열과 다르다).
        # [해설]   그렇지 않으면 `--flag ""`가 환경변수와 `config.toml`의 실제 값을
        # [해설]   가려버린다. 예외는 아래 `models.auto_classifier`뿐이며 "메인 에이전트
        # [해설]   모델을 상속"한다는 명시적 의미를 가진다.
        if isinstance(raw, str) and not raw.strip():
            # A CLI value is a shell string, not a TOML literal: `--flag ""` is
            # an absent value, not an empty one. `coerce_toml_value` accepts the
            # empty string, so without this the blank flag wins its rank and
            # masks every real value in env and `config.toml`.
            if option.key == "models.auto_classifier":
                # Except this one: `parse_args` strips the flag's value, so an
                # empty string here can only be `--auto-classifier-model ""` --
                # the explicit "inherit the main agent model" instruction that
                # `run_textual_cli_async` maps to `INHERIT_CLASSIFIER_MODEL`
                # ahead of env and `config.toml`. `Unset` would abstain the CLI
                # tier, and `dcode --auto-classifier-model "" config get
                # models.auto_classifier` would report the very env/TOML
                # classifier the launch ignores. Returning the sentinel keeps
                # introspection and the launch path on the same value.
                from deepagents_code._cli_context import INHERIT_CLASSIFIER_MODEL

                return _found_for(option, INHERIT_CLASSIFIER_MODEL)
            if raw:
                return Invalid(
                    f"Ignoring {flag}={raw!r} (whitespace-only; treated as unset)"
                )
            return Unset()
        return coerce_toml_value(option, raw, source=flag)

    # [해설] 메모리 안의 CLI 상태는 항상 OK다.
    def status(self) -> ProviderStatus:
        """Return the always-healthy in-memory CLI status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    # [해설] 아무 일도 하지 않는다. 파싱된 인자 스냅샷은 불변이라 다시 읽을 것이 없다.
    def reload(self) -> None:
        """Retain the immutable parsed argument snapshot."""
