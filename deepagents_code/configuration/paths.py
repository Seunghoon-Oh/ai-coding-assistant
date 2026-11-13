"""Fixed operating-system paths for managed configuration."""
# [해설] 관리형(managed) 설정 파일의 위치를 OS별로 고정해서 알려주는 모듈이다. 사용자
# [해설]   설정 위치(`~/.deepagents/config.toml`)는 여기가 아니라
# [해설]   `deepagents_code/_paths.py`의 `PATHS.profile.config_file`이다.
# [해설] 관리형 경로는 고정이다. macOS: `/Library/Application
# [해설]   Support/dcode/managed_config.toml`, Windows:
# [해설]   `<ProgramData>/dcode/managed_config.toml`, 그 외(Linux 등):
# [해설]   `/etc/dcode/managed_config.toml`.
# [해설] [설계] 관리형 정책은 관리자가 배포하는 것이므로 일반 사용자가 환경변수나
# [해설]   설정으로 위치를 바꿀 수 없어야 한다. 그래서 환경변수 대신
# [해설]   레지스트리(Windows)를 읽고, 경로 자체를 하드코딩한다.
# [해설] 공개 함수는 두 개다. 표시용 `managed_config_path`(경로만 반환)와 판단용
# [해설]   `resolve_managed_path`(경로 + 추측 여부 `fallback`). 둘 다 내부 `_resolve`를
# [해설]   직접 호출한다.
# [해설] 소비자: `configuration/__init__.py`가 재노출하고, `configuration/service.py`가
# [해설]   스냅샷을 읽을 때, `configuration/writer.py`가 쓰기 거부 판단에 쓴다.

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# [해설] Windows 레지스트리 조회가 실패했을 때만 쓰는 최후의 추측값.
_PROGRAM_DATA_DEFAULT = "C:/ProgramData"


# [해설] 관리형 경로와 "그 경로를 믿어도 되는가"를 한 쌍으로 묶는 불변 값.
# [해설] `fallback`이 `None`이면 권위 있는 경로이고, 문자열이면 추측한 경로라는 이유다.
# [해설]   추측한 경로에 파일이 없을 때 "정책 없음"(MISSING)과 "모름"(INDETERMINATE)을
# [해설]   혼동하지 않게 하려는 설계다(`types.py`의 `ProviderHealth` 참고).
@dataclass(frozen=True, slots=True)
class ResolvedManagedPath:
    """Where managed policy is read from, and whether that location is certain.

    `fallback` holds why the path is a guess, or `None` when the path is
    authoritative. A failed registry query leaves the lookup pointing at the
    hardcoded default. On a host whose ProgramData is relocated, the guessed
    path holds no file, which reads as "no administrator deployed policy" — the
    same state as a machine with no policy at all. The reason travels with the
    path so the two can never be confused: a guessed path yields an
    `INDETERMINATE` snapshot rather than a clean `MISSING` one.
    """

    path: Path
    fallback: str | None = None


# [해설][설계] Windows 전용 레지스트리 조회. 이 레지스트리 키(`Shell Folders`의 `Common
# [해설]   AppData`)는 일반 사용자가 환경변수처럼 쉽게 바꿀 수 없는 값을 읽기 위해
# [해설]   선택했다. 표준 라이브러리 `winreg`를 쓴다.
def _program_data_from_registry() -> tuple[str | None, str | None]:
    """Read ProgramData from the Windows registry, if available.

    Returns:
        The registry-reported ProgramData path and `None`, or `None` and the
        reason the lookup failed. Off-Windows both are `None`: that platform
        needs no registry lookup, so the path is not a guess.
    """
    if sys.platform != "win32":
        return None, None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "Common AppData")
    except (ImportError, OSError) as exc:
        logger.warning(
            "Could not read ProgramData from the registry (%s); falling back to "
            "%s. Managed policy stored elsewhere will not be found.",
            type(exc).__name__,
            _PROGRAM_DATA_DEFAULT,
        )
        return None, (
            f"ProgramData could not be read from the registry ({type(exc).__name__}); "
            f"looked under {_PROGRAM_DATA_DEFAULT}"
        )
    if isinstance(value, str) and value:
        return value, None
    logger.warning(
        "Registry ProgramData value is unusable (%r); falling back to %s. "
        "Managed policy stored elsewhere will not be found.",
        value,
        _PROGRAM_DATA_DEFAULT,
    )
    return None, (
        f"registry ProgramData value is unusable; looked under {_PROGRAM_DATA_DEFAULT}"
    )


# [해설][주의] `%ProgramData%` 환경변수를 신뢰하지 않는다. 일반 사용자가 자기 셸에서 이
# [해설]   값을 바꾸면 관리형 정책 위치가 사용자 통제 경로로 바뀌어 정책이 조용히 사라질
# [해설]   수 있다.
# [해설] `environ` 인자는 테스트용 주입 지점이며, 운영에서는 `None`이어야 한다.
def _windows_program_data(
    environ: Mapping[str, str] | None,
) -> tuple[str, str | None]:
    r"""Resolve the real ProgramData directory, ignoring process env vars.

    `%ProgramData%` can be redefined by any unprivileged user in their own
    shell, which would redirect the managed-config lookup to a user-controlled
    path and silently drop (or replace) administrator policy. Read the value
    from the registry (`HKLM\...\Shell Folders\Common AppData`) and fall
    back to the hardcoded default only if the registry query fails.

    Returns:
        The administrator-owned ProgramData directory, and why it is a guess
        when the registry query failed.
    """
    if environ is not None:
        # Tests inject a fake environ to exercise platform logic off-Windows.
        return (
            environ.get("ProgramData")
            or environ.get("PROGRAMDATA")
            or _PROGRAM_DATA_DEFAULT
        ), None
    value, fallback = _program_data_from_registry()
    return value or _PROGRAM_DATA_DEFAULT, fallback


# [해설] 플랫폼 문자열(`sys.platform`) -> 관리형 파일 경로 매핑의 유일한 구현. 두 공개
# [해설]   함수가 서로를 호출하지 않고 여기로 위임하므로, 테스트가 한쪽을 바꿔도 다른 쪽
# [해설]   결과는 변하지 않는다.
def _resolve(
    platform: str | None, environ: Mapping[str, str] | None
) -> ResolvedManagedPath:
    """Map one platform to its fixed managed-config path.

    Both public entry points delegate here rather than to each other, so a test
    that redirects one of them cannot change what the other computes.

    Returns:
        The path for the platform, paired with the reason it is a guess.
    """
    active_platform = sys.platform if platform is None else platform
    if active_platform == "darwin":
        return ResolvedManagedPath(
            Path("/Library/Application Support/dcode/managed_config.toml")
        )
    if active_platform == "win32":
        root, fallback = _windows_program_data(environ)
        return ResolvedManagedPath(
            Path(root) / "dcode" / "managed_config.toml", fallback
        )
    return ResolvedManagedPath(Path("/etc/dcode/managed_config.toml"))


# [해설] 스냅샷 로더가 실제로 읽는 쪽. 건강 상태를 보고하는 호출자는
# [해설]   `managed_config_path`가 아니라 이 함수를 써야 추측 경로를 권위 경로로
# [해설]   착각하지 않는다.
def resolve_managed_path(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None
) -> ResolvedManagedPath:
    """Return the fixed managed-config path and whether it is authoritative.

    What the snapshot loader reads. Callers that report health need this rather
    than `managed_config_path`, so a guessed path is never mistaken for an
    authoritative one.

    Args:
        platform: Override the detected platform; intended for tests.
        environ: Test-only injection point for the Windows branch. Production
            passes `None` so the ProgramData directory comes from the registry.
            Passing a real environment restores the redirection this module
            exists to prevent.

    Returns:
        The path for the platform, paired with the reason it is a guess.
    """
    return _resolve(platform, environ)


# [해설] 화면 표시와 오류 메시지용 경로 반환. 정책 적용 가능 여부를 판단하려면
# [해설]   `resolve_managed_path`를 써야 한다.
def managed_config_path(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    """Return the fixed managed-config path for the current operating system.

    For display and error messages. Anything that decides whether policy is
    enforceable wants `resolve_managed_path`.

    Args:
        platform: Override the detected platform; intended for tests.
        environ: Test-only injection point for the Windows branch. See
            `resolve_managed_path`.

    Returns:
        The managed-config path for the platform.
    """
    return _resolve(platform, environ).path
