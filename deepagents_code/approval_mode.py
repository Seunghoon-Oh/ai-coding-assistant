"""Approval-mode state shared by the Textual client and agent server."""

# [해설] 모듈 개요: '도구 호출을 사람이 승인해야 하는가'를 정하는 승인 모드(Manual /
# [해설]   Auto / YOLO)의 상태와 영속화를 맡는다.
# [해설] 이 파일 자체는 도구 호출을 가로채지 않는다. 가로채는 쪽은
# [해설]   `deepagents_code/auto_mode.py`의 `AutoModeHITLMiddleware`와
# [해설] `deepagents_code/agent.py`의 승인 미들웨어이고, 여기서는 그들이 읽는 '현재
# [해설]   모드'와 '경고 확인 기록'만 다룬다.
# [해설][설계] 세 모드: MANUAL=승인이 필요한 도구마다 사람에게 묻는다 / AUTO=결정론
# [해설]   규칙+분류기 모델이 먼저 판단하고 애매하면 사람에게 묻는다 / YOLO=묻지 않는다.
# [해설][설계] 모든 읽기 경로는 '못 믿으면 manual'로 닫힌다(fail closed). 값이 깨졌거나
# [해설]   Store가 없으면 가장 보수적인 MANUAL이 된다.
# [해설] 저장 위치가 둘이다. (1) 스레드별 '현재 모드' = 서버의 LangGraph
# [해설]   Store(APPROVAL_MODE_NAMESPACE),
# [해설] (2) 설치 단위 '경고를 이미 확인했는가' = 로컬 `approval.json`(파일 락으로 동시
# [해설]   저장을 직렬화).
# [해설] 모드 전환(Shift+Tab 순환)은 `next_approval_mode`, 서버 쪽 읽기는
# [해설]   `aread_approval_mode_from_store`, 클라이언트 쪽 쓰기는
# [해설]   `awrite_approval_mode`가 담당한다.
from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

APPROVAL_MODE_NAMESPACE: tuple[str, str] = ("deepagents_code", "approval_mode")
"""Store namespace for per-thread approval-mode control records."""

YOLO_ACKNOWLEDGEMENT_POLICY_VERSION = "2026-07-14"
"""Version of the unrestricted-mode warning that must be acknowledged."""

YOLO_WARNING_KEY = "yolo"
"""`[warnings].suppress` key that mutes the recurring "YOLO is active" toast.

Suppression is cosmetic: YOLO still requires an explicit acknowledgement to
enter (a modal in the TUI, a console prompt for `--yolo`), still honors the
`startup.yolo_switcher` setting, and still shows a persistent `YOLO`
status-bar indicator the whole time it is active. Because the acknowledgement
is once per policy version, that indicator is the only remaining in-session
signal for a returning user who has muted the toast.
"""

AUTO_NOTICE_VERSION = "2026-07-24"
"""Version of the first-run Auto mode education notice.

Bump this string whenever the notice copy changes materially enough that
already-shown installs should see it again.
"""


# [해설] 승인 정책 3종. 값 문자열('manual'/'auto'/'yolo')이 그대로 Store와 설정에
# [해설]   저장된다.
class ApprovalMode(StrEnum):
    """Tool-approval policy selected for an interactive thread."""

    MANUAL = "manual"
    AUTO = "auto"
    YOLO = "yolo"


# [해설] Store에 넣는 값의 모양. 모드 문자열 하나뿐이라 서버가 검증하기 쉽다.
class ApprovalModePayload(TypedDict):
    """Stored approval-mode control payload."""

    mode: str


# [해설][설계] 신뢰할 수 없는 입력(설정, 컨텍스트, 저장값)은 여기를 통과시킨다. 문자열이
# [해설]   아니거나 모르는 값이면 MANUAL이라 '실수로 승인이 풀리는' 일이 없다.
def coerce_approval_mode(value: object) -> ApprovalMode:
    """Return a validated mode, failing closed to `manual`.

    Args:
        value: Untrusted mode value from config, context, or storage.

    Returns:
        A validated `ApprovalMode`; invalid values become `ApprovalMode.MANUAL`.
    """
    try:
        return ApprovalMode(value) if isinstance(value, str) else ApprovalMode.MANUAL
    except ValueError:
        return ApprovalMode.MANUAL


# [해설] Shift+Tab 순환 규칙. Manual -> Auto -> YOLO -> Manual이 기본이고, 원격
# [해설]   샌드박스처럼 Auto를 못 쓰면 Auto를 건너뛰며,
# [해설] `startup.yolo_switcher`가 꺼져 있으면 YOLO를 건너뛴다. YOLO에서 나올 때는 항상
# [해설]   Manual로 돌아간다.
def next_approval_mode(
    current: ApprovalMode | str | object,
    *,
    auto_eligible: bool,
    yolo_switcher_enabled: bool,
) -> ApprovalMode | None:
    """Return the next Shift+Tab approval mode for the active session.

    The cycle is Manual → Auto → YOLO → Manual when both Auto and the YOLO
    switcher entry are available. Auto is omitted when `auto_eligible` is false
    (for example a remote sandbox). YOLO is omitted when orgs/users disable
    `startup.yolo_switcher`. Launching with `--yolo` still leaves unrestricted
    mode when the switcher entry is disabled; Shift+Tab only exits YOLO then.

    Args:
        current: Active approval mode (mode enum or raw value).
        auto_eligible: Whether classifier-backed Auto can be selected.
        yolo_switcher_enabled: Whether unrestricted YOLO appears in the cycle.

    Returns:
        The next mode, or `None` when no alternate mode is available.
    """
    mode = (
        current if isinstance(current, ApprovalMode) else coerce_approval_mode(current)
    )
    if mode is ApprovalMode.MANUAL:
        if auto_eligible:
            return ApprovalMode.AUTO
        if yolo_switcher_enabled:
            return ApprovalMode.YOLO
        return None
    if mode is ApprovalMode.AUTO:
        if yolo_switcher_enabled:
            return ApprovalMode.YOLO
        return ApprovalMode.MANUAL
    # Only genuine YOLO reaches here; unknown/invalid values were normalized to
    # Manual above and took that branch. Exiting YOLO always returns to Manual.
    return ApprovalMode.MANUAL


# [해설] 스레드 id를 sha256으로 해시해 Store 키로 쓴다. 원본 thread_id가 Store 키에
# [해설]   노출되지 않는다.
def approval_mode_key(thread_id: str) -> str:
    """Return the store key for a thread's live approval mode.

    Args:
        thread_id: LangGraph thread id for the active session.

    Returns:
        Deterministic store key that does not expose the raw thread id.
    """
    return sha256(thread_id.encode("utf-8")).hexdigest()


# [해설] 모드 문자열을 Store 값으로 포장한다. `auto_approve` Boolean은 옛 API 호환용이며
# [해설]   True는 YOLO, False는 MANUAL로 매핑된다.
# [해설] mode와 auto_approve는 정확히 하나만 줘야 하고, 잘못된 값은 ValueError다(읽기
# [해설]   쪽과 달리 쓰기 쪽은 조용히 MANUAL로 바꾸지 않는다).
def approval_mode_payload(
    *,
    mode: ApprovalMode | str | None = None,
    auto_approve: bool | None = None,
) -> ApprovalModePayload:
    """Return the stored approval-mode payload.

    Args:
        mode: Explicit approval mode.
        auto_approve: Compatibility input for callers using the previous Boolean
            API. `True` maps to unrestricted `yolo`, and `False` maps to `manual`.

    Returns:
        JSON-serializable store value.

    Raises:
        ValueError: If neither or both inputs are supplied, or `mode` is invalid.
    """
    if (mode is None) == (auto_approve is None):
        msg = "Provide exactly one of mode or auto_approve"
        raise ValueError(msg)
    if auto_approve is not None:
        resolved = ApprovalMode.YOLO if auto_approve else ApprovalMode.MANUAL
    else:
        try:
            resolved = ApprovalMode(mode)
        except (TypeError, ValueError) as exc:
            msg = f"Invalid approval mode: {mode!r}"
            raise ValueError(msg) from exc
    return {"mode": resolved.value}


# [해설] Store 항목은 dict일 수도 객체일 수도 있어서(SDK/런타임 차이) 두 모양 모두에서
# [해설]   value를 꺼낸다.
def _item_value(item: object) -> object:
    """Extract a store item's value.

    Args:
        item: SDK or runtime store-item shape.

    Returns:
        The stored value, or `None` when the shape is unrecognized.
    """
    if isinstance(item, Mapping):
        return item.get("value")
    return getattr(item, "value", None)


# [해설] Store 항목 -> ApprovalMode. 누락·손상은 None이며, 호출자는 None을 반드시
# [해설]   manual로 해석해야 한다.
def _approval_mode_from_item(item: object) -> ApprovalMode | None:
    """Extract a validated approval mode from a Store item.

    Args:
        item: SDK or runtime store-item shape.

    Returns:
        The stored mode, or `None` when the item is missing or malformed.
    """
    if item is None:
        logger.debug("Approval-mode store item is missing")
        return None

    value = _item_value(item)
    raw_mode = value.get("mode") if isinstance(value, Mapping) else None
    if isinstance(raw_mode, str):
        try:
            return ApprovalMode(raw_mode)
        except ValueError:
            pass

    logger.warning("Approval-mode store item has invalid contents")
    return None


# [해설] 동기 읽기. Store가 없거나 키가 비었거나 get()이 예외를 내면 모두
# [해설]   None(=manual)이다.
def read_approval_mode_from_store(
    store: object, key: str | None
) -> ApprovalMode | None:
    """Read a live approval mode from the server-side LangGraph Store.

    Args:
        store: `request.runtime.store` from the graph server.
        key: Store key produced by `approval_mode_key`.

    Returns:
        A validated mode, or `None` when the record cannot be trusted. Callers
        must interpret `None` as `manual`.
    """
    if store is None:
        logger.debug("Approval-mode store is unavailable")
        return None
    if not isinstance(key, str) or not key:
        logger.debug("Approval-mode store key is missing or invalid")
        return None

    get = getattr(store, "get", None)
    if get is None:
        logger.debug("Approval-mode store does not expose get()")
        return None

    try:
        item = get(APPROVAL_MODE_NAMESPACE, key)
    except Exception:
        logger.warning("Could not read approval-mode store item", exc_info=True)
        return None
    return _approval_mode_from_item(item)


# [해설][주의] 서버의 Store는 비동기 배치 Store라 이벤트 루프 스레드에서 동기 메서드를
# [해설]   부르면 거부된다. 서버 미들웨어는 반드시 이 비동기 읽기를 쓴다.
# [해설] `AutoModeHITLMiddleware`가 매 모델 호출 때 `_live_mode`를 거쳐 이 함수로 '지금
# [해설]   모드'를 다시 읽으므로, 실행 중 모드를 바꿔도 다음 호출부터 적용된다.
async def aread_approval_mode_from_store(
    store: object, key: str | None
) -> ApprovalMode | None:
    """Asynchronously read a live approval mode from a LangGraph Store.

    The graph server supplies an async batched Store whose synchronous methods
    reject calls from the event-loop thread. Prefer `aget()` for that runtime,
    while retaining a synchronous fallback for lightweight local test stores.

    Args:
        store: `request.runtime.store` from the graph server.
        key: Store key produced by `approval_mode_key`.

    Returns:
        A validated mode, or `None` when the record cannot be trusted. Callers
        must interpret `None` as `manual`.
    """
    if store is None:
        logger.debug("Approval-mode store is unavailable")
        return None
    if not isinstance(key, str) or not key:
        logger.debug("Approval-mode store key is missing or invalid")
        return None

    aget = getattr(store, "aget", None)
    get = getattr(store, "get", None)
    try:
        if callable(aget):
            result = aget(APPROVAL_MODE_NAMESPACE, key)
            item = await result if inspect.isawaitable(result) else result
        elif callable(get):
            item = get(APPROVAL_MODE_NAMESPACE, key)
        else:
            logger.debug("Approval-mode store does not expose get() or aget()")
            return None
    except Exception:
        logger.warning("Could not read approval-mode store item", exc_info=True)
        return None
    return _approval_mode_from_item(item)


# [해설] 클라이언트가 모드를 바꿀 때 원격 에이전트(`aput_store_item`)를 통해 Store에
# [해설]   기록한다. 쓰기 수단이 없는 에이전트면 None을 돌려주고 아무것도 하지 않는다.
async def awrite_approval_mode(
    agent: object,
    thread_id: str,
    *,
    mode: ApprovalMode | str | None = None,
    auto_approve: bool | None = None,
) -> str | None:
    """Persist approval mode through an agent's remote store client.

    Args:
        agent: Agent object. Remote agents expose `aput_store_item`.
        thread_id: LangGraph thread id for the active session.
        mode: Explicit approval mode.
        auto_approve: Compatibility input for the previous Boolean API.

    Returns:
        Store key written, or `None` when the agent has no store writer.
    """
    put = getattr(agent, "aput_store_item", None)
    if put is None:
        return None

    key = approval_mode_key(thread_id)
    await put(
        APPROVAL_MODE_NAMESPACE,
        key,
        approval_mode_payload(mode=mode, auto_approve=auto_approve),
    )
    return key


# [해설] 여기서부터는 설치 단위 파일 `approval.json`(YOLO 경고 확인 / Auto 첫 안내 표시
# [해설]   기록) 처리다.
# [해설][설계] 읽기-병합-쓰기를 두 겹 락으로 보호한다: 프로세스 안 스레드
# [해설]   락(`_approval_state_thread_lock`) + 프로세스 간
# [해설]   `FileLock`(`_approval_state_lock`).
_APPROVAL_STATE_LOCK_TIMEOUT_SECONDS = 5.0
"""Longest a save waits for the shared install-local approval-state lock."""

_APPROVAL_STATE_THREAD_LOCKS: dict[str, threading.Lock] = {}
_APPROVAL_STATE_THREAD_LOCKS_GUARD = threading.Lock()


# [해설] 확인 기록 파일 경로. `DEFAULT_STATE_DIR/approval.json`이며 순환 import를
# [해설]   피하려고 함수 안에서 `model_config`를 import한다.
def yolo_acknowledgement_path() -> Path:
    """Return the installation-local acknowledgement file path.

    Returns:
        Path under the private dcode state directory.
    """
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "approval.json"


# [해설] 락 전용 형제 파일(`approval.json.lock`) 경로. 데이터 파일을 직접 잠그지 않는다.
def _approval_state_lock_path(path: Path) -> Path:
    """Return the sibling lock file path that serializes approval-state saves.

    Args:
        path: Path to `approval.json`.

    Returns:
        Dedicated `.lock` path next to the state file.
    """
    return path.with_name(f"{path.name}.lock")


# [해설] 경로 문자열별로 하나씩 만든 프로세스 내부 락. 락 객체 테이블 자체는
# [해설]   `_APPROVAL_STATE_THREAD_LOCKS_GUARD`로 보호한다.
def _approval_state_thread_lock(path: Path) -> threading.Lock:
    """Return the process-local mutation lock for an approval-state path.

    Each `_approval_state_lock` builds a fresh `FileLock(thread_local=False)`, so
    the cross-process lock alone does not serialize threads within one dcode
    process. This threading lock guarantees only one in-process thread runs the
    read-merge-write at a time, independent of the filelock backend's cross-thread
    behavior.

    Args:
        path: Path to `approval.json`.

    Returns:
        Process-lifetime lock keyed by the normalized path string.
    """
    key = str(path)
    with _APPROVAL_STATE_THREAD_LOCKS_GUARD:
        lock = _APPROVAL_STATE_THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _APPROVAL_STATE_THREAD_LOCKS[key] = lock
        return lock


# [해설] 락 컨텍스트 매니저. 디렉터리를 0700으로 만든 뒤 스레드 락과 파일 락을 함께
# [해설]   잡는다(타임아웃 5초).
@contextmanager
def _approval_state_lock(path: Path) -> Iterator[None]:
    """Serialize read-merge-write updates to install-local approval state.

    Combines a process-local threading lock with a cross-process `FileLock` on a
    sibling `.lock` file so concurrent YOLO and Auto saves cannot drop each
    other's fields.

    Args:
        path: Path to `approval.json`.

    Yields:
        Control while the caller exclusively holds the mutation lock.

    Callers should handle `filelock.Timeout` (lock wait expired) and `OSError`
    (lock directory creation failure).
    """
    # These run during __enter__, before the lock is held. Any OSError from
    # mkdir/chmod propagates out of the `with` to the caller, which must catch
    # it (the only caller, `_merge_approval_state`, does).
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    file_lock = FileLock(
        str(_approval_state_lock_path(path)),
        timeout=_APPROVAL_STATE_LOCK_TIMEOUT_SECONDS,
        thread_local=False,
    )
    with _approval_state_thread_lock(path), file_lock:
        yield


# [해설][주의] 파일이 없으면 조용히 {}, 손상·비객체면 경고 로그를 남기고 {}를 돌려준다.
# [해설]   {}는 '확인 기록 없음'과 같아서 호출자가 보수적으로(다시 묻거나 Auto 복원 안
# [해설]   함) 동작한다.
# [해설] 손상된 파일은 다음 저장 때 덮어써지므로 이 경고 로그가 손상의 유일한 흔적이다.
def _load_approval_state(path: Path) -> dict[str, object]:
    """Load the install-local approval state file, or an empty dict.

    A missing file is the normal first-run case and returns `{}` silently.
    Unreadable, corrupt, or non-object state also returns `{}` but is logged:
    the next save overwrites the file, so this warning is the only surviving
    evidence of the corruption. Callers fail closed on `{}`. Most re-prompt;
    `has_auto_mode_notice` callers may instead decline to restore Auto, so the
    log must not promise a prompt.

    Args:
        path: Path to `approval.json`.

    Returns:
        Parsed object mapping when valid; otherwise `{}`.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except OSError, UnicodeDecodeError, json.JSONDecodeError:
        logger.warning(
            "Ignoring unreadable or corrupt approval state at %s; callers fail "
            "closed and the file will be overwritten on the next save",
            path,
            exc_info=True,
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "Ignoring non-object approval state at %s; callers fail closed and "
            "the file will be overwritten on the next save",
            path,
        )
        return {}
    return data


# [해설] 원자적 쓰기: 같은 디렉터리에 임시 파일을 만들고(`mkstemp`) 0600으로 권한을 준
# [해설]   뒤 `replace`로 교체한다. 쓰는 도중 크래시가 나도 반쯤 쓴 파일이 남지 않는다.
# [해설] 실패는 예외 대신 False로 돌려주고, 임시 파일은 정리한다.
def _write_approval_state(
    path: Path,
    payload: Mapping[str, object],
    *,
    failure_label: str,
) -> bool:
    """Atomically write install-local approval state.

    Callers that load-then-merge must hold `_approval_state_lock` around both
    the load and this write so concurrent savers cannot clobber each other.

    Args:
        path: Destination path under the private dcode state directory.
        payload: JSON-serializable mapping to persist.
        failure_label: Human-readable label for the warning log on OSError.

    Returns:
        `True` when the private atomic write succeeds, otherwise `False`.
    """
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            path.parent.chmod(0o700)
        fd, raw_tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        tmp_path = Path(raw_tmp_path)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
        if os.name != "nt":
            tmp_path.chmod(0o600)
        tmp_path.replace(path)
        if os.name != "nt":
            path.chmod(0o600)
    except OSError:
        logger.warning("Could not persist %s", failure_label, exc_info=True)
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        return False
    return True


# [해설] 락 안에서 '현재 파일 로드 -> updates 덮어쓰기 -> version=1로 저장'을 한 번에
# [해설]   수행한다. YOLO 확인과 Auto 안내 필드가 서로를 지우지 않게 하는 핵심 함수다.
# [해설] 락 대기 Timeout이나 OSError는 경고 로그 후 False를 돌려준다(호출자는 실패해도
# [해설]   UX를 막지 않는 쪽으로 처리).
def _merge_approval_state(
    path: Path,
    updates: Mapping[str, object],
    *,
    failure_label: str,
) -> bool:
    """Load, merge, and write install-local approval state under a lock.

    Args:
        path: Path to `approval.json`.
        updates: Fields to overlay onto the current state mapping.
        failure_label: Human-readable label for warning logs.

    Returns:
        `True` when the locked merge-write succeeds, otherwise `False`.
    """
    try:
        with _approval_state_lock(path):
            payload = {
                **_load_approval_state(path),
                **updates,
                "version": 1,
            }
            return _write_approval_state(
                path,
                payload,
                failure_label=failure_label,
            )
    except Timeout:
        logger.warning(
            "Timed out waiting to persist %s",
            failure_label,
            exc_info=True,
        )
        return False
    except OSError:
        logger.warning(
            "Could not lock approval state for %s", failure_label, exc_info=True
        )
        return False


# [해설] YOLO 경고를 현재 정책 버전(`YOLO_ACKNOWLEDGEMENT_POLICY_VERSION`)으로 이미
# [해설]   확인했는지. 버전이 다르면 다시 확인받는다.
def has_yolo_acknowledgement(path: Path | None = None) -> bool:
    """Return whether the current unrestricted-mode warning was accepted.

    Args:
        path: Alternate acknowledgement path for tests.

    Returns:
        `True` only for a valid record matching the current policy version.
    """
    target = path or yolo_acknowledgement_path()
    data = _load_approval_state(target)
    return (
        data.get("version") == 1
        and data.get("policy_version") == YOLO_ACKNOWLEDGEMENT_POLICY_VERSION
        and data.get("acknowledged") is True
    )


# [해설] YOLO 경고 확인을 저장한다. Auto 안내 필드는 그대로 보존(병합)된다.
def save_yolo_acknowledgement(path: Path | None = None) -> bool:
    """Persist the current unrestricted-mode warning acknowledgement.

    Merges into any existing approval state so first-run Auto notice fields are
    preserved. Concurrent saves are serialized with a cross-process lock.

    Args:
        path: Alternate acknowledgement path for tests.

    Returns:
        `True` when the private atomic write succeeds, otherwise `False`.
    """
    target = path or yolo_acknowledgement_path()
    return _merge_approval_state(
        target,
        {
            "policy_version": YOLO_ACKNOWLEDGEMENT_POLICY_VERSION,
            "acknowledged": True,
        },
        failure_label="YOLO acknowledgement",
    )


# [해설] Auto 첫 사용 안내를 이미 보여줬는지. 최상위 version을 보지 않고
# [해설]   `auto_notice_version`으로 독립 판정해서 YOLO 기록과 따로 진화한다.
def has_auto_mode_notice(path: Path | None = None) -> bool:
    """Return whether the current Auto first-enable notice was already shown.

    The notice self-versions on `auto_notice_version` and deliberately does not
    gate on the top-level `version` (unlike `has_yolo_acknowledgement`), so the
    two records can evolve independently within one `approval.json`.

    Args:
        path: Alternate approval-state path for tests.

    Returns:
        `True` only when a shown notice matches `AUTO_NOTICE_VERSION`.
    """
    target = path or yolo_acknowledgement_path()
    data = _load_approval_state(target)
    return (
        data.get("auto_notice_shown") is True
        and data.get("auto_notice_version") == AUTO_NOTICE_VERSION
    )


# [해설] Auto 안내를 보여줬다고 저장한다. 실패(False)해도 호출자는 막지 않고 진행(fail
# [해설]   open)한다. 안내는 교육용이라 보안 경계가 아니기 때문이다.
def save_auto_mode_notice(path: Path | None = None) -> bool:
    """Persist that the Auto first-enable notice was shown.

    Merges into any existing approval state so YOLO acknowledgement fields are
    preserved. Concurrent saves are serialized with a cross-process lock.
    Callers should fail open when this returns `False`.

    Args:
        path: Alternate approval-state path for tests.

    Returns:
        `True` when the private atomic write succeeds, otherwise `False`.
    """
    target = path or yolo_acknowledgement_path()
    return _merge_approval_state(
        target,
        {
            "auto_notice_version": AUTO_NOTICE_VERSION,
            "auto_notice_shown": True,
        },
        failure_label="Auto mode notice",
    )
