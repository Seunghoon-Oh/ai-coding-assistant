"""User-config-only atomic TOML writes."""
# [해설] 사용자 설정 파일(`~/.deepagents/config.toml`, 경로는
# [해설]   `deepagents_code/model_config.py`의 `DEFAULT_CONFIG_PATH`)을 안전하게
# [해설]   수정하는 쓰기 계층이다. 관리형 설정은 쓰지 않는다.
# [해설] 핵심은 "읽기-수정-쓰기" 트랜잭션이다: 잠금 획득 -> TOML 읽기 -> 호출자의
# [해설]   `mutate`가 표를 제자리 수정 -> 임시 파일에 쓰고 `replace`로 원자적 교체 ->
# [해설]   공용 resolver 갱신.
# [해설] `USER_CONFIG_WRITE_LOCK`은 `model_config.py`의 `_config_write_lock`과 같은
# [해설]   객체라서, 두 쓰기 경로가 서로를 배제한다.
# [해설] 결과는 예외 대신 `WriteResult(ok, changed, error)`로 돌려주므로, UI는 반환값만
# [해설]   보고 분기한다.
# [해설] [설계] 쓰기 도구(`tomli_w`)는 함수 안에서 import 한다. 설치가 빠진 환경도 "쓰기
# [해설]   실패" 메시지로 보고되고, 모듈 import 단계에서 죽지 않는다.

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# [해설][설계] 프로세스 전역 재진입 잠금. 파일 전체를 교체하는 방식이라 섹션별로 다른
# [해설]   잠금을 쓰면 서로 배제되지 않아 마지막 `replace`가 앞선 변경을 지운다. 그래서
# [해설]   모든 `config.toml` 쓰기가 이 하나를 공유한다. 프로세스 간 경쟁은 범위 밖이다.
USER_CONFIG_WRITE_LOCK = threading.RLock()


# [해설] 쓰기 트랜잭션 한 번의 결과. `ok`(성공), `changed`(실제로 파일이 바뀜),
# [해설]   `error`(실패 사유).
@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome of one user config transaction."""

    ok: bool
    changed: bool
    error: str | None = None

    # [해설][주의] 세 필드가 모순되면 생성 자체를 막는다. 실패인데 사유가 없거나,
    # [해설]   실패인데 changed가 True거나, 성공인데 error가 있는 경우는 모두
    # [해설]   `ValueError`.
    def __post_init__(self) -> None:
        """Reject outcomes that cannot describe a real transaction.

        Callers branch on `ok` alone, so a failure with no detail would surface
        as a bare "could not be saved" with nothing to act on, and a change
        recorded against a failed write would report an edit that never
        reached the file.

        Raises:
            ValueError: If the three fields do not describe one outcome.
        """
        if not self.ok and self.error is None:
            msg = "a failed write must carry an error detail"
            raise ValueError(msg)
        if self.changed and not self.ok:
            msg = "a failed write cannot have changed the file"
            raise ValueError(msg)
        if self.ok and self.error is not None:
            msg = "a successful write cannot carry an error detail"
            raise ValueError(msg)


# [해설][흐름] `mutate`를 받아 사용자 `config.toml` 한 번 수정. 1) 경로 결정 2) 관리형
# [해설]   경로 거부 3) 잠금 획득 후 읽기 4) `mutate` 실행 5) 임시 파일에 쓰고 원자적
# [해설]   교체 6) 공용 resolver 갱신.
# [해설] 성공 경로에서 `WriteResult(True, True)`, 변경이 없으면 `WriteResult(True,
# [해설]   False)`로 파일을 건드리지 않는다.
def update_user_config(
    mutate: Callable[[dict[str, Any]], bool],
    *,
    config_path: Path | None = None,
) -> WriteResult:
    """Serialize a read-modify-write of the user config and replace it atomically.

    Writes the user tier only. The managed path is refused rather than trusted
    to be unreachable.

    A committed write to the default path also refreshes the shared process
    resolver, so later reads see the new value. That refresh is best-effort and
    never turns a landed write into a reported failure; see
    `refresh_shared_resolver`.

    Args:
        mutate: Edit applied to the table parsed inside the write lock. It must
            edit that table in place and return whether anything changed;
            overwriting it wholesale discards concurrent edits to sibling
            tables, which is the hazard the shared lock exists to prevent.
        config_path: Override the default config location; intended for tests.
            The managed-config path is rejected.

    Returns:
        Transaction success, changed state, and safe error detail.
    """
    # [해설][흐름] 1) 기본 경로를 쓰는 호출자는 `DEFAULT_CONFIG_PATH`를 쓴다. 순환
    # [해설]   import를 피하려고 함수 안에서 `model_config`를 import 한다.
    if config_path is None:
        from deepagents_code.model_config import DEFAULT_CONFIG_PATH

        config_path = DEFAULT_CONFIG_PATH
    from deepagents_code.configuration.paths import managed_config_path

    # [해설][주의] 2) 관리형 경로를 사용자 쓰기 대상으로 주면 거부한다. "관리형은 읽기
    # [해설]   전용"이라는 보안 속성(`THREAT_MODEL.md`)을 호출자 관례가 아니라 구조로
    # [해설]   보장한다.
    if config_path == managed_config_path():
        # The managed tier is read-only, and `THREAT_MODEL.md` states that as a
        # security property. It held only because no caller passed this path;
        # make it structural, so the claim does not depend on every future
        # caller knowing it.
        return WriteResult(False, False, "managed config is read-only")
    # [해설][흐름] 3) 잠금 구간 시작. 파일이 있으면 읽고(`tomllib.load`), 없으면 빈
    # [해설]   dict로 시작한다. 읽기 실패는 `WriteResult(False, ...)`로 반환한다.
    with USER_CONFIG_WRITE_LOCK:
        try:
            if config_path.exists():
                with config_path.open("rb") as handle:
                    data = tomllib.load(handle)
            else:
                data = {}
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            # `tomllib` decodes the bytes itself, so a file that is not UTF-8
            # raises `UnicodeDecodeError` rather than `TOMLDecodeError`.
            # Letting it escape turns a mis-encoded config into a traceback the
            # caller reports as a generic failure, with the real reason lost.
            return WriteResult(False, False, f"could not update {config_path}: {exc}")
        # [해설][흐름] 4) `mutate(data)`는 일부러 예외 처리 바깥에서 호출한다. 호출자
        # [해설]   클로저의 버그가 "파일 권한 문제"로 오인되지 않게 하기 위해서다.
        # Outside the exception handling on purpose: a `TypeError` or
        # `ValueError` raised inside a caller's closure is a bug in that
        # caller, and reporting it as "could not update <path>" sends the user
        # to check permissions and disk space for a defect in a lambda.
        changed = mutate(data)
        if not changed:
            return WriteResult(True, False)
        try:
            # [해설][흐름] 5) 부모 디렉터리를 만들고, 같은 디렉터리에 임시
            # [해설]   파일(`mkstemp`)을 만들어 쓴 뒤 `replace`로 교체한다. 같은
            # [해설]   디렉터리여야 교체가 원자적이다. 실패하면 임시 파일을 지운다.
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # Imported before `mkstemp` so a missing writer dependency cannot
            # leak the descriptor: the cleanup below can unlink the temp path,
            # but only `os.fdopen` takes ownership of the descriptor, so an
            # `ImportError` between the two left it open.
            import tomli_w

            descriptor, temporary = tempfile.mkstemp(
                dir=config_path.parent,
                suffix=".tmp",
            )
            temporary_path = Path(temporary)
            try:
                handle = os.fdopen(descriptor, "wb")
            except BaseException:
                # Nothing adopted the descriptor, so close it here. Closing it
                # in the handler below instead would risk closing an unrelated
                # file: once `os.fdopen` succeeds and its `with` block exits,
                # the number is free for another thread to reuse.
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                with contextlib.suppress(OSError):
                    temporary_path.unlink()
                raise
            try:
                with handle:
                    tomli_w.dump(data, handle)
                temporary_path.replace(config_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    temporary_path.unlink()
                raise
        except (
            OSError,
            ImportError,
            tomllib.TOMLDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            # `ImportError` covers the `tomli_w` import inside this block: an
            # install without the writer dependency must report "could not
            # update <path>" like any other write failure.
            return WriteResult(False, False, f"could not update {config_path}: {exc}")
    # [해설][흐름] 6) 쓰기가 성공했으면 공용 resolver를 갱신한다. 이 단계의 실패는 쓰기
    # [해설]   성공을 뒤집지 않는다(내부에서 로그만 남김).
    refresh_shared_resolver(config_path)
    return WriteResult(True, True)


# [해설][주의] 기본 경로에 쓴 경우에만 공용 resolver를 `refresh_managed=True`로 다시
# [해설]   로드한다. 테스트가 임시 경로에 쓴 경우 실제 사용자·관리형 파일을 읽게 되는
# [해설]   것을 막기 위해서다.
# [해설] 모든 예외를 삼키고 경고만 남긴다(`except Exception`). 이미 디스크에 기록된
# [해설]   뒤라서, 갱신 실패를 쓰기 실패로 보고하면 사용자가 올바른 파일을 다시 고치려
# [해설]   하기 때문이다.
def refresh_shared_resolver(config_path: Path) -> None:
    """Make a committed write visible to the shared process resolver.

    Only the default path is refreshed. `get_config_resolver` is keyed on
    `DEFAULT_CONFIG_PATH`, so reloading it after a write to an override path
    would re-read the real user config and the managed policy file - live
    filesystem reads inside tests that passed a `tmp_path` - while leaving the
    written path's own view stale anyway.

    Refreshes every tier, managed policy included. The managed snapshot is
    fetched before the resolver takes its generation lock, then installed as
    an already-refreshed replacement. An in-app preference toggle therefore
    still picks up policy installed since startup without making ordinary
    event-loop config reads wait on remote I/O. Leaving the managed provider
    alone would let the user tier advance past the policy tier, which is the
    split-generation state the whole design exists to prevent.

    Failures are logged rather than returned. The write already landed and was
    replaced into place; reporting a stale in-process view as a failed write
    sends the user to retry or hand-edit a file that is already correct.

    Every failure, not just `OSError`: a reload can also raise `ValueError`
    from the snapshot and resolved-value invariants, or `RuntimeError` from a
    provider that produced no snapshot. Callers such as
    `model_config._save_toml_field` invoke this from the success branch of
    their own write and return `bool`, so anything escaping here surfaces as a
    crash after the bytes are already on disk -- the exact outcome the
    paragraph above says must not happen.

    Args:
        config_path: Path the caller just wrote.
    """
    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    if config_path != DEFAULT_CONFIG_PATH:
        return
    from deepagents_code.configuration.resolver import get_config_resolver

    try:
        get_config_resolver(refresh_managed=True)
    except Exception as exc:  # noqa: BLE001  # Write committed; see docstring
        logger.warning(
            "Wrote %s but could not refresh the shared config resolver: %s. "
            "This process keeps serving the previous values until it restarts.",
            config_path,
            exc,
        )
