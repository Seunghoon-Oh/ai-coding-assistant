"""Typed configuration-provider results and health metadata."""
# [해설] 설정 제공자(provider)가 "옵션 하나를 읽은 결과"와 "소스 건강 상태"를 표현하는
# [해설]   기본 타입 모음이다. configuration 패키지의 가장 아래층이라 다른 모듈이 이
# [해설]   파일을 가져다 쓴다.
# [해설] 읽기 결과 3종: `Found[T]`(값을 선언했고 변환 성공), `Unset`(선언 없음),
# [해설]   `Invalid`(선언했으나 변환 실패). 합쳐서 `ProviderResult[T]`.
# [해설] 건강 상태: `ProviderHealth`(OK/MISSING/INDETERMINATE/UNREADABLE/CORRUPT)와,
# [해설]   이름·경로·상태를 묶은 `ProviderStatus`.
# [해설] TOML 한 장의 파싱 결과: `TomlSnapshot`(데이터 + 상태). `data`는 상태가 OK가
# [해설]   아니면 반드시 비어 있어야 한다.
# [해설] [설계] "값 없음"과 "읽을 수 없음"을 같은 빈 dict로 합치지 않는 것이 이 파일의
# [해설]   핵심 의도다. 그래서 `data`만 따로 읽지 말고 항상 `status`와 함께 읽도록
# [해설]   강제한다.
# [해설] 사용처: `provider.py`, `providers.py`, `resolver.py`, `service.py`, 그리고
# [해설]   `deepagents_code/model_config.py`가 `ProviderStatus`를 타입 힌트로 import
# [해설]   한다.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


# [해설] 제공자가 값을 선언했고 매니페스트 타입으로 변환에도 성공한 결과. `value`는 이미
# [해설]   변환된 값이다. (PEP 695 제네릭 문법이라 Python 3.12 이상 필요)
@dataclass(frozen=True, slots=True)
class Found[T]:
    """A provider declared a value and coerced it successfully."""

    value: T


# [해설] 제공자가 이 옵션에 대해 아무 선언도 하지 않았다는 뜻. resolver는 이 경우 다음
# [해설]   우선순위 제공자로 넘어간다.
@dataclass(frozen=True, slots=True)
class Unset:
    """A provider made no declaration for an option."""


# [해설] 값은 있었지만 변환할 수 없었다는 뜻. `reason`에는 호출자가 문장으로 감쌀 수
# [해설]   있는 짧은 사유만 담는다(`provider.py`의 `parse_interpreter_tools` 참고).
@dataclass(frozen=True, slots=True)
class Invalid:
    """A provider declared a value that its input domain could not coerce."""

    reason: str


# [해설] `type` 문(Python 3.12+)으로 만든 타입 별칭. 제공자 한 곳에서 옵션 하나를 읽은
# [해설]   결과가 취할 수 있는 모든 경우.
type ProviderResult[T] = Found[T] | Unset | Invalid
"""Complete result of reading one option from one provider."""


# [해설] 설정 소스 하나의 건강 상태.
# [해설] OK: 정상. MISSING: 권위 있는 경로에 파일이 없음(정책 미배포). INDETERMINATE:
# [해설]   경로 자체가 추측이라 판단 불가. UNREADABLE: 읽기 실패. CORRUPT: TOML 문법
# [해설]   오류.
class ProviderHealth(StrEnum):
    """Health of one configuration source."""

    OK = "OK"
    MISSING = "MISSING"
    INDETERMINATE = "INDETERMINATE"
    UNREADABLE = "UNREADABLE"
    CORRUPT = "CORRUPT"


# [해설] 원격 관리형 소스 URL을 진단 메시지에 남길 최대 길이.
REMOTE_SOURCE_MAX_CHARS = 2048
"""Longest remote source URL retained in operator-facing diagnostics."""


# [해설][주의] 원격 정책 URL 검증기. 절대 HTTPS URL만 통과시키고, 공백·제어문자,
# [해설]   비ASCII, 계정정보(user:pass), 쿼리·프래그먼트, 잘못된 포트는 모두
# [해설]   `ValueError`로 거부한다.
# [해설] `ProviderStatus`가 이 URL을 운영자용 문장에 그대로 끼워 넣으므로, 검증되지 않은
# [해설]   문자열이 화면·로그에 섞이지 않게 하는 보안 경계다.
def _validate_remote_source_url(source: str) -> str:
    """Validate and normalize one configured remote source URL.

    Returns:
        The normalized absolute HTTPS URL.

    Raises:
        ValueError: If the source is unsafe or is not an absolute HTTPS URL.
    """
    from urllib.parse import urlsplit

    if len(source) > REMOTE_SOURCE_MAX_CHARS:
        msg = "remote source is too long"
        raise ValueError(msg)
    if any(char.isspace() or not char.isprintable() for char in source):
        msg = "remote source must not contain whitespace or control characters"
        raise ValueError(msg)
    try:
        parsed = urlsplit(source)
    except ValueError as exc:
        msg = "remote source is not a valid URL"
        raise ValueError(msg) from exc
    if not source.isascii():
        msg = "remote source must contain only ASCII URI characters"
        raise ValueError(msg)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        msg = "remote source must be an absolute HTTPS URL"
        raise ValueError(msg)
    if parsed.username is not None or parsed.password is not None:
        msg = "remote source must not contain credentials"
        raise ValueError(msg)
    if parsed.query or parsed.fragment:
        msg = "remote source must not contain a query string or fragment"
        raise ValueError(msg)
    try:
        port = parsed.port
    except ValueError as exc:
        msg = "remote source has an invalid port"
        raise ValueError(msg) from exc
    host = parsed.hostname.rstrip(".")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return parsed._replace(scheme="https", netloc=netloc).geturl()


# [해설] 제공자 하나의 건강 상태와 안전한 진단 정보. `name`(표시 이름), `path`(로컬 파일
# [해설]   경로 또는 None), `health`, `detail`(사유), `remote_source`(검증된 원격 URL)로
# [해설]   구성된다.
@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """Health and safe diagnostic detail for one provider."""

    name: str
    path: Path | None
    health: ProviderHealth
    detail: str | None = None
    remote_source: str | None = field(default=None, kw_only=True)
    """Validated URL this status came from, when `path` is only a trust anchor.

    A remote managed policy reports the *local* descriptor file as its `path`,
    because that is the file an operator can edit. Without this field an error
    renders as "repair or remove" that file, which is not broken -- and
    removing it drops all policy. Set only after `_validate_remote_url`
    accepts the URL, so a rejected source is never echoed back --
    `__post_init__` enforces that rather than trusting every construction site.
    """

    # [해설][주의] `remote_source`가 "검증기를 통과한 정규화 URL"과 정확히 같지 않으면
    # [해설]   `ValueError`. 생성 지점마다 검증을 믿지 않고 타입 자체에서 불변식을
    # [해설]   강제한다.
    def __post_init__(self) -> None:
        """Reject a remote source that never passed URL validation.

        Every surface interpolates this field straight into operator-facing
        text, so "only a validated URL reaches here" is a security invariant.
        It held by convention -- one construction site in each of two modules
        -- and `ProviderStatus` is public, so a third site is a rejected
        source, credentials and all, in a `doctor` row. This restates the
        canonical remote-source validator and also requires its normalized
        output: the point is that an unvalidated string cannot get in.

        Raises:
            ValueError: If `remote_source` is not the normalized output of the
                remote-source validator.
        """
        source = self.remote_source
        if source is None:
            return
        msg = "remote_source must be a validated absolute HTTPS URL"
        try:
            normalized = _validate_remote_source_url(source)
        except ValueError as exc:
            raise ValueError(msg) from exc
        if normalized != source:
            raise ValueError(msg)

    # [해설] 이 소스가 해석(resolution)에 참여해도 안전한가. OK와 MISSING만 True다.
    # [해설] MISSING은 "관리자가 정책을 배포하지 않음"이라 안전하지만, INDETERMINATE는
    # [해설]   경로가 추측이라 빈 결과를 증거로 삼을 수 없으므로 False다.
    @property
    def usable(self) -> bool:
        """Whether the provider can safely participate in resolution.

        `MISSING` is usable because no file at an authoritative path means the
        administrator deployed no policy. `INDETERMINATE` is not: the path
        itself is a guess, so an empty read proves nothing about what policy
        the administrator deployed.
        """
        return self.health in {ProviderHealth.OK, ProviderHealth.MISSING}


# [해설] 파싱된 TOML 한 장(`data`)과 그 건강 상태(`status`)를 한 쌍으로 묶는다.
# [해설]   관리형·사용자 두 계층 모두 이 타입으로 `service.py`에서 전달된다.
@dataclass(frozen=True, slots=True)
class TomlSnapshot:
    """One parsed TOML source and its health.

    `data` is empty whenever `status.health` is not `OK`, so it must always be
    read together with `status`: an empty table alone cannot distinguish "this
    source declares nothing" from "this source could not be read".
    """

    data: dict[str, Any]
    status: ProviderStatus

    # [해설][주의] 상태가 OK가 아닌데 `data`가 비어 있지 않은 스냅샷은 만들 수
    # [해설]   없다(`ValueError`).
    # [해설] `data`를 `MappingProxyType`으로 감싸지 않고 일반 dict로 둔 것도 의도다.
    # [해설]   변환기가 `isinstance(value, dict)`로 표와 스칼라를 구분하기 때문에 다른
    # [해설]   Mapping을 넣으면 조용히 다음 소스로 넘어가 버린다.
    def __post_init__(self) -> None:
        """Reject a snapshot that carries data it could not have read.

        `data` is a `dict`, not a `Mapping`, and deliberately not copied
        behind a `MappingProxyType` the way `ResolvedValue.__post_init__`
        copies its own mappings. It is a raw TOML table, and the coercers test
        nested values with `isinstance(value, dict)` to tell a table from a
        scalar -- so any `Mapping` that is not a `dict` fails that test and
        every option under it falls back to its next source, silently. The
        narrower annotation makes a `mappingproxy` (or any other `Mapping`) a
        type error at the call site rather than a fall-through at runtime.

        The immutability this type promises is therefore a convention, kept by
        the providers that build the snapshot and by callers handed one.

        Raises:
            ValueError: If an unhealthy snapshot carries a non-empty table.
        """
        if self.status.health is not ProviderHealth.OK and self.data:
            msg = (
                f"a {self.status.health.value} snapshot must carry no data; an "
                "empty table is what every reader reads as 'nothing declared'"
            )
            raise ValueError(msg)

    # [해설] 이미 파싱된 표를 가진 호출자용. 상태 정보가 없으므로 OK 스냅샷을 만든다.
    @classmethod
    def from_table(cls, name: str, data: dict[str, Any]) -> TomlSnapshot:
        """Build a readable snapshot around an already-parsed table.

        For a caller that holds the parsed data and no health metadata of its
        own. A caller that has real health passes both halves directly.

        Args:
            name: Human-readable source label.
            data: Parsed TOML table.

        Returns:
            An `OK` snapshot carrying `data`.
        """
        return cls(data, ProviderStatus(name, None, ProviderHealth.OK))

    # [해설][설계] "읽을 수 있지만 아무것도 선언하지 않은" 소스(OK + 빈 데이터).
    # [해설]   `absent`, `unknown_origin`과 건강이 달라 해석 참여 여부가 달라지므로,
    # [해설]   복붙으로 건강을 고르지 않도록 이름을 붙여 둔 생성자다.
    @classmethod
    def declaring_nothing(cls, name: str) -> TomlSnapshot:
        """Build a readable snapshot for a source that declares nothing.

        `ProviderStatus.usable` gates whether a source takes part in
        resolution, so the difference between this and `absent` or
        `unknown_origin` is a real decision. Naming it keeps that decision from
        being made by copy-paste: three of the call sites this replaced picked
        three different healths for a literally empty user table.

        Args:
            name: Human-readable source label.

        Returns:
            An `OK` snapshot carrying no data.
        """
        return cls({}, ProviderStatus(name, None, ProviderHealth.OK))

    # [해설] 소스가 아예 없는 경우(MISSING + 빈 데이터).
    @classmethod
    def absent(cls, name: str) -> TomlSnapshot:
        """Build a snapshot for a source that is not there at all.

        Args:
            name: Human-readable source label.

        Returns:
            A `MISSING` snapshot carrying no data.
        """
        return cls({}, ProviderStatus(name, None, ProviderHealth.MISSING))

    # [해설] 소스의 상태를 알 수 없는 경우(INDETERMINATE + 빈 데이터)를 만드는 생성자.
    # [해설]   이 저장소 안에 호출하는 곳은 없고, 추측한 관리형
    # [해설]   경로(`ResolvedManagedPath.fallback`)의 경우는 service.py의
    # [해설]   _load_managed가 직접 INDETERMINATE로 바꿔 만든다.
    @classmethod
    def unknown_origin(cls, name: str) -> TomlSnapshot:
        """Build a snapshot for a source whose state could not be determined.

        Args:
            name: Human-readable source label.

        Returns:
            An `INDETERMINATE` snapshot carrying no data.
        """
        return cls({}, ProviderStatus(name, None, ProviderHealth.INDETERMINATE))
