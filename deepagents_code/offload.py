"""Storage paths for offloaded conversation history."""

# [해설] 모듈 역할: 컨텍스트 오프로드(대화 요약/압축) 때 '원문을 어디에 파일로
# [해설]   저장할지'를 정하는 경로 규칙과, 그 파일의 보안 처리(권한 0o700)·보존기간
# [해설]   청소·삭제를 맡는다.
# [해설] 실행 위치: 경로 해석은 에이전트를 조립하는 프로세스에서 일어난다(agent.py 의
# [해설]   create_cli_agent 가 _artifacts_root, _offload_fallback_root 를 호출).
# [해설]   청소/삭제는 세션 관리 쪽(sessions.py 의 스레드 삭제)에서도 호출된다.
# [해설] 주요 심볼: _artifacts_root(큰 도구 결과 저장 위치), _offload_fallback_root(대화
# [해설]   기록 보관 루트), sweep_offloaded_history(보존기간 지난 파일 삭제),
# [해설]   delete_offloaded_history(스레드 삭제 시 정리), CONVERSATION_HISTORY_DIRNAME.
# [해설] 개념: 컨텍스트가 길어지면 오래된 메시지를 요약으로 바꾸고 원문은
# [해설]   {root}/conversation_history/{thread_id}.md 에 남겨, 에이전트가 파일 도구로
# [해설]   다시 읽을 수 있게 한다. 이 파일은 저장 '위치와 수명'만 다루고, 요약 시점과
# [해설]   방법은 offload_middleware.py 와 SDK 의 SummarizationMiddleware 가 정한다.
# [해설][설계] 위치 선택 우선순위: 영구 위치(~/.deepagents) -> 쓸 수 없으면 임시
# [해설]   디렉터리(재시작 시 사라질 수 있음, _EPHEMERAL_OFFLOAD_STORAGE 로 표시). 모두
# [해설]   실패하면 예측 불가능한 mkdtemp 디렉터리.
# [해설] 관련 레인 모듈: agent.py(백엔드 연결), offload_middleware.py,
# [해설]   offload_api.py(별도 모듈, 이 레인 밖).

from __future__ import annotations

import hashlib
import logging
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePath

from deepagents_code._paths import get_deepagents_home

logger = logging.getLogger(__name__)

# [해설] 예측 가능한 임시 폴더를 못 쓸 때 에이전트에게 보여 주는 '가상' 경로 접두사.
# [해설]   실제 저장은 비공개 임시 폴더로 라우팅된다. 가상 루트를 고정해 두면 재시작
# [해설]   후에도 스레드 상태에 저장된 아카이브 경로가 같은 경로 규칙에 계속 맞는다.
_FALLBACK_ARTIFACTS_ROOT = "/dcode-artifacts-fallback"

_SECONDS_PER_DAY = 86_400

CONVERSATION_HISTORY_DIRNAME = "conversation_history"
"""Subdirectory of the offload root that holds per-thread conversation archives.

Lives directly under `~/.deepagents/` in local mode. The `/agent` picker
excludes this reserved name in addition to requiring an `AGENTS.md` marker.
"""


@dataclass(frozen=True)
# [해설] 에이전트에게 보이는 artifacts 루트 경로(root)와, 큰 결과를 따로 보낼 실제
# [해설]   디렉터리(large_results_dir; 폴백일 때만 값 있음)를 묶은 값 객체.
class _ArtifactsStorage:
    """Agent-visible artifacts root and optional routed large-result directory."""

    root: str
    large_results_dir: Path | None = None


# [해설] 호스트의 절대 경로를 SDK 파일시스템 도구가 받는 '/로 시작하는 슬래시 경로'로
# [해설]   바꾼다. Windows 드라이브 경로는 SDK 가 수락하지 않아 //?/ 확장 형식으로
# [해설]   우회한다.
def _filesystem_tool_path(path: PurePath) -> str:
    """Represent an absolute host path in the filesystem tool path format.

    Drive-qualified paths are rejected by the SDK's virtual path validation. The
    Windows extended-length form keeps the drive while starting with the `/`
    required by filesystem tools; `pathlib` and Windows APIs still resolve it to
    the same host directory.

    Args:
        path: Absolute host path to represent.

    Returns:
        A forward-slash path accepted by the filesystem tool contract.
    """
    normalized = path.as_posix()
    if path.drive and not path.drive.startswith("\\\\"):
        return f"//?/{normalized}"
    return normalized


# [해설] 전역 상태: 가장 최근 _offload_fallback_root 호출이 임시 저장소로 떨어졌는지
# [해설]   기록. UI 가 '기록이 유지되지 않을 수 있음'을 알릴 때 쓴다.
_EPHEMERAL_OFFLOAD_STORAGE = False
"""Whether the most recent `_offload_fallback_root` fell back to temp storage."""

_UNIQUE_OFFLOAD_FALLBACK_ROOT: Path | None = None
"""Private random fallback root that cannot be reconstructed on a later call."""


# [해설] 위 플래그를 읽는 공개 함수. 로컬 모드(UI와 같은 프로세스)에서만 의미가 있고,
# [해설]   서버 모드에서는 클라이언트 쪽에서 항상 False 이다.
def offload_storage_is_ephemeral() -> bool:
    """Return whether offload history is routed to non-persistent storage.

    `True` when the persistent `~/.deepagents` location was unwritable and the
    most recent `_offload_fallback_root` fell back to a temporary directory that
    may not survive a restart. Only meaningful in local mode, where
    `_offload_fallback_root` runs in the same process as the UI; in
    server/sandbox mode persistence is owned by the server backend and this flag
    stays `False` client-side.

    Returns:
        `True` if the local offload root is a temporary, non-persistent
        directory; `False` when it is the persistent per-user location (or was
        never resolved in this process).
    """
    return _EPHEMERAL_OFFLOAD_STORAGE


# [해설] 디렉터리를 만들고 현재 사용자만 접근하도록(0o700) 권한을 조인다. 대화 기록과 큰
# [해설]   도구 결과에는 민감한 내용이 있을 수 있기 때문이다.
# [해설][주의] 공유 설정 루트(~/.deepagents) 자체에는 쓰지 않고 전용 하위 디렉터리에만
# [해설]   쓴다. 다른 사용자 소유 디렉터리면 PermissionError 로 거부해 남이 심어 둔
# [해설]   폴더를 쓰지 않게 한다.
def _harden_dir(path: Path) -> None:
    """Create `path` if needed and restrict it to the current user.

    Only ever call this on directories owned by this process's storage (a temp
    dir or a dedicated subdirectory), never on the shared `~/.deepagents` config
    root.

    Args:
        path: Directory to create and harden to `0o700`.

    Raises:
        OSError: If the path exists but is not a directory, or the directory
            cannot be created or its mode changed (e.g. a read-only mount).
        PermissionError: If the existing directory is owned by another local user.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        msg = f"Path is not a directory: {path}"
        raise OSError(msg)
    getuid = getattr(os, "getuid", None)
    if getuid is not None and info.st_uid != getuid():
        msg = f"Directory is owned by another user: {path}"
        raise PermissionError(msg)
    # `mkdir(mode=...)` does not tighten an existing directory. These directories
    # can hold conversation data and offloaded tool results, so they must remain
    # inaccessible to other local accounts regardless of the process umask.
    path.chmod(0o700)


# [해설] 임시 파일을 하나 만들어 실제로 쓸 수 있는지 확인한다. 읽기 전용 마운트는 mkdir
# [해설]   이 성공해도 쓰기에 실패하기 때문이다.
def _probe_writable(path: Path) -> None:
    """Confirm `path` accepts new files (catches read-only mounts).

    Creating the directory is insufficient when it already exists on a read-only
    mount; a temporary file proves writes can succeed.

    Args:
        path: Directory to probe.
    """
    with tempfile.NamedTemporaryFile(dir=path, prefix=".write-test-"):
        pass


# [해설] 큰 도구 결과 등 'artifacts' 저장 위치를 정한다. 기본은
# [해설]   {임시폴더}/dcode-artifacts-{uid} (사용자별 고정 경로, 권한 강화).
# [해설][주의] 고정 경로는 공유 임시폴더에서 다른 사용자가 선점할 수 있어, 소유자/권한
# [해설]   검사에 실패하면 mkdtemp 기반 비공개 폴더 + 가상 루트 방식으로 우회한다.
# [해설]   agent.py 에서 CompositeBackend 의 라우트로 연결된다(자세한 연결은 agent.py 의
# [해설]   create_cli_agent).
def _artifacts_root() -> _ArtifactsStorage:
    """Return storage configuration for offloaded artifacts.

    The normal path is a stable, hardened host directory that filesystem tools
    and shell commands can use directly. If that predictable directory is
    unusable, large results use a private unique directory behind a stable virtual
    root. Keeping the virtual root stable lets conversation archive paths persisted
    in thread state continue matching their dedicated route after a restart.

    Returns:
        The agent-visible artifacts root and an optional directory to which large
        results must be routed.
    """
    getuid = getattr(os, "getuid", None)
    suffix = str(getuid()) if getuid is not None else str(os.getpid())
    temp_root = Path(tempfile.gettempdir())
    root = temp_root / f"dcode-artifacts-{suffix}"
    try:
        _harden_dir(root)
        _probe_writable(root)
    except OSError, RuntimeError:
        logger.warning(
            "Predictable per-user artifacts directory is unavailable; routing "
            "large results from a stable virtual prefix to private temporary storage",
            exc_info=True,
        )
        unique = Path(
            tempfile.mkdtemp(prefix=f"dcode-artifacts-{suffix}-", dir=temp_root)
        )
        _harden_dir(unique)
        _probe_writable(unique)
        return _ArtifactsStorage(
            root=_FALLBACK_ARTIFACTS_ROOT,
            large_results_dir=unique,
        )
    return _ArtifactsStorage(root=_filesystem_tool_path(root))


# [해설] 대화 기록 아카이브의 루트를 돌려준다. 로컬 모드의 conversation_history 백엔드가
# [해설]   이 위치를 쓴다(agent.py).
# [해설][흐름] 1) 이미 mkdtemp 폴백을 만든 적이 있으면 그것을 재사용(다시 유도할 수
# [해설]   없으므로 정리 대상이 같은 파일이 되게) 2) DEEPAGENTS_HOME 아래
# [해설]   conversation_history 를 만들고 쓰기 확인 3) 실패하면 {temp}/deepagents-{uid}
# [해설]   4) 그것도 실패하면 mkdtemp 고유 폴더.
# [해설][주의] 호출만으로도 디렉터리 생성과 probe 파일 쓰기라는 부수효과가 있다. 모듈
# [해설]   전역(_EPHEMERAL_OFFLOAD_STORAGE, _UNIQUE_OFFLOAD_FALLBACK_ROOT)을 바꾸므로
# [해설]   스레드 안전하지 않다(추정: 동시 호출 보호 없음).
def _offload_fallback_root() -> Path:
    """Return a writable base directory for offloaded conversation history.

    Prefers the persistent profile root (`DEEPAGENTS_HOME`, default
    `~/.deepagents`) so offloaded history survives across sessions and is easy
    to locate; falls back to a private temporary directory when that root
    cannot be created or written. The profile root itself is resolved at launch
    and cannot fail here, so an unwritable or unmounted configured root is now
    the only way to reach the fallback -- and history then does not persist,
    which `_EPHEMERAL_OFFLOAD_STORAGE` records so callers can say so. This is
    the live root for the local-mode `conversation_history` backend in
    `agent.py`.

    Archives always live in the `conversation_history` subdirectory of the
    returned root. The `0o700` hardening therefore targets that subdirectory,
    never the shared profile root -- which also houses `config.toml`,
    `hooks.json`, `.env`, and `.state/`, whose permissions this must not
    disturb. A temporary fallback root is created solely for offload, so the
    whole directory is hardened in that case.

    Note: the `S_ISDIR` check below (which uses `lstat`, deliberately not
    following the link) guards the paths it is applied to -- the
    `conversation_history` subdirectory and, in the fallback case, the temp
    root -- not the profile root itself, which is created with a plain `mkdir`.
    So a `conversation_history` (or temp root) that is itself a symlink is
    rejected, whereas a symlinked profile root pointing at a directory the
    current user owns is followed transparently and archives persist normally.
    (A dangling profile-root symlink still falls through to temporary storage,
    but via `mkdir` raising, not via this check.)

    Returns:
        A directory whose `conversation_history` subdirectory is private and
        writable.
    """

    def _prepare_user_dir() -> Path:
        base = get_deepagents_home()
        # Ensure the shared config root exists and is usable, but leave its
        # permissions untouched -- hardening belongs on the archive subdir only.
        base.mkdir(parents=True, exist_ok=True)
        archive_dir = base / CONVERSATION_HISTORY_DIRNAME
        _harden_dir(archive_dir)
        _probe_writable(archive_dir)
        return base

    def _prepare_temp_dir(path: Path) -> Path:
        # A temp dir is created solely for offload and is not shared config, so
        # hardening the whole directory (which protects its archive subdir) is
        # both safe and necessary in world-writable temp locations.
        _harden_dir(path)
        _probe_writable(path)
        return path

    global _EPHEMERAL_OFFLOAD_STORAGE, _UNIQUE_OFFLOAD_FALLBACK_ROOT  # noqa: PLW0603
    if _UNIQUE_OFFLOAD_FALLBACK_ROOT is not None:
        # Unlike the persistent and predictable temp paths, a directory created
        # by `mkdtemp` cannot be derived again. Keep returning the root already
        # used by the archive backend so cleanup reaches the same files.
        _EPHEMERAL_OFFLOAD_STORAGE = True
        return _UNIQUE_OFFLOAD_FALLBACK_ROOT
    try:
        root = _prepare_user_dir()
    except RuntimeError, OSError:
        logger.warning(
            "User data directory is not writable; falling back to temporary "
            "offload storage, which may not persist across restarts",
            exc_info=True,
        )
    else:
        _EPHEMERAL_OFFLOAD_STORAGE = False
        return root
    # Only reached on the fallback path: every root produced below is temporary
    # and may not survive a restart.
    _EPHEMERAL_OFFLOAD_STORAGE = True
    getuid = getattr(os, "getuid", None)
    suffix = str(getuid()) if getuid is not None else str(os.getpid())
    temp_root = Path(tempfile.gettempdir())
    path = temp_root / f"deepagents-{suffix}"
    try:
        return _prepare_temp_dir(path)
    except OSError, RuntimeError:
        logger.warning(
            "Per-user temporary offload directory is unavailable; creating "
            "a private unique directory",
            exc_info=True,
        )
        unique = Path(tempfile.mkdtemp(prefix=f"deepagents-{suffix}-", dir=temp_root))
        _UNIQUE_OFFLOAD_FALLBACK_ROOT = _prepare_temp_dir(unique)
        return _UNIQUE_OFFLOAD_FALLBACK_ROOT


# [해설] 보존기간(일)을 설정 시스템에서 읽는다. 관리형 설정, 환경변수, config.toml 순의
# [해설]   우선순위는 config_manifest 쪽 규칙을 따른다. 0 이면 청소를 끈다.
# [해설][주의] 이 함수는 config_manifest 의 밑줄 접두사 함수(_resolve_option 등)를 직접
# [해설]   가져다 쓴다. 비공개 API 에 의존하므로 그쪽 변경에 깨지기 쉽다.
def _history_retention_days() -> int:
    """Resolve the conversation-history retention window through the manifest.

    Uses the canonical `history.retention_days` option, so managed config, the
    `DEEPAGENTS_CODE_HISTORY_RETENTION_DAYS` env var, and `config.toml` all
    apply with their normal precedence, and `dcode config` reports the same
    value the sweep acts on. A malformed or negative value is rejected by the
    resolver and falls through to the next source, ending at the default.

    Returns:
        The resolved non-negative retention window in days; `0` disables the
        sweep.
    """
    from deepagents_code.config_manifest import (
        HISTORY_RETENTION_DAYS_DEFAULT,
        _emit_ranked_diagnostics,
        _resolve_option,
        get_option,
    )

    option = get_option("history.retention_days")
    if option is None:
        return HISTORY_RETENTION_DAYS_DEFAULT
    resolved = _resolve_option(option)
    _emit_ranked_diagnostics(option, resolved)
    value = resolved.value
    if type(value) is int and value >= 0:
        return value
    return HISTORY_RETENTION_DAYS_DEFAULT


# [해설] 만료된 .md 아카이브 하나를 삭제한다.
# [해설][설계] 요약 미들웨어가 같은 파일을 읽고-덧붙이고-다시 쓰는 중에 삭제가 끼어들면
# [해설]   새 기록이 사라질 수 있다. 그래서 파일을 열어 둔 채 fstat 로 만료를 재확인한
# [해설]   직후 unlink 한다.
def _delete_expired_archive(candidate: Path, archive_dir: Path, cutoff: float) -> bool:
    """Delete one expired regular markdown archive.

    The expiry check and `unlink` are serialized against a concurrent archive
    refresh: the summarization middleware refreshes an archive by path
    (read-append-rewrite), and on POSIX an `unlink` racing that rewrite would
    orphan the open descriptor and silently drop the newly written history.
    Opening the candidate and re-checking expiry with `fstat` on the live
    descriptor immediately before `unlink` closes that window: a rewrite that
    already refreshed the mtime is observed and the archive is kept, and a
    rewrite that starts after the `fstat` still lands in the inode the writer
    has open.

    Args:
        candidate: Direct child of the archive directory to inspect.
        archive_dir: Directory that candidates must remain inside.
        cutoff: Oldest mtime to retain, as Unix seconds.

    Returns:
        Whether the candidate was deleted.
    """
    if candidate.parent != archive_dir or candidate.suffix != ".md":
        return False
    try:
        with candidate.open("rb") as archive_file:
            info = os.fstat(archive_file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mtime >= cutoff:
                return False
            candidate.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning(
            "Failed to sweep offloaded history archive %s", candidate, exc_info=True
        )
        return False
    return True


# [해설] 보존기간이 검증된 뒤 conversation_history 안의 만료 파일을 일괄 삭제하고 삭제
# [해설]   개수를 돌려준다.
def _sweep_offloaded_history(retention_days: int) -> int:
    """Delete expired archives after retention config has been validated.

    Args:
        retention_days: Number of days of archives to retain.

    Returns:
        The number of archives deleted.
    """
    try:
        archive_dir = _offload_fallback_root() / CONVERSATION_HISTORY_DIRNAME
        candidates = list(archive_dir.iterdir())
    except OSError, RuntimeError:
        logger.warning(
            "Could not resolve offloaded history archives for sweep", exc_info=True
        )
        return 0
    cutoff = time.time() - retention_days * _SECONDS_PER_DAY
    return sum(
        _delete_expired_archive(candidate, archive_dir, cutoff)
        for candidate in candidates
    )


# [해설] 공개 청소 함수. 어떤 실패도 예외로 올리지 않고 0 을 돌려준다(시작 시 청소가 앱
# [해설]   기동을 막지 않도록).
def sweep_offloaded_history() -> int:
    """Delete conversation-history archives older than the configured retention.

    Returns:
        The number of archive files deleted. Failures return zero or omit the
        affected file from the count rather than raising.
    """
    try:
        retention_days = _history_retention_days()
        if retention_days == 0:
            return 0
        return _sweep_offloaded_history(retention_days)
    except Exception:
        logger.warning("Unexpected failure sweeping offloaded history", exc_info=True)
        return 0


# [해설] 핸드오프(복구 스냅샷) 파일명 접두사를 만든다. 원본 스레드 ID 를 SHA-256 으로
# [해설]   해시해 파일명에 넣어, 경로 문자나 글롭 패턴이 섞이지 않으면서 소유 관계를
# [해설]   유지한다.
def _handoff_archive_prefix(thread_id: str) -> str:
    """Persist source ownership in filenames without embedding paths or globs.

    Returns:
        The stable filename prefix for this source thread's handoffs.
    """
    owner = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
    return f"handoff_{owner}_"


# [해설] 파일 하나를 삭제하는 최선 노력(best-effort) 함수. 실패는 로그만 남기고 False.
def _delete_archive(archive_path: Path) -> bool:
    """Remove one archive, logging filesystem failures for best-effort cleanup.

    Returns:
        Whether the archive was removed.
    """
    try:
        archive_path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning(
            "Failed to delete offloaded conversation history %s",
            archive_path,
            exc_info=True,
        )
        return False
    logger.debug("Deleted offloaded conversation history %s", archive_path)
    return True


# [해설] 삭제된 스레드가 소유한 모든 handoff_{해시}_*.md 스냅샷을 지운다.
def _delete_handoff_archives(archive_dir: Path, thread_id: str) -> bool:
    """Remove every recovery snapshot owned by the deleted source thread.

    Returns:
        Whether at least one snapshot was removed.
    """
    deleted = False
    try:
        for archive in archive_dir.glob(f"{_handoff_archive_prefix(thread_id)}*.md"):
            deleted = _delete_archive(archive) or deleted
    except OSError:
        logger.warning(
            "Could not list handoff archives for thread %s", thread_id, exc_info=True
        )
    return deleted


# [해설] 스레드 삭제 시 호출된다(sessions.py). 그 스레드의 {thread_id}.md 와 handoff
# [해설]   스냅샷을 지운다. 서버/샌드박스 모드의 아카이브는 샌드박스 쪽에 있으므로
# [해설]   여기서 지울 수 없다.
# [해설][주의] thread_id 가 아카이브 폴더 밖을 가리키면(경로 탈출) 삭제를 거부한다. 정상
# [해설]   ID 는 UUID7 이므로 거부는 비정상 입력이나 버그의 신호다.
def delete_offloaded_history(thread_id: str) -> bool:
    """Remove a thread's compaction archive and handoff recovery snapshots.

    Deletes the per-thread markdown file written by the local-mode
    `conversation_history` backend (`{root}/conversation_history/{thread_id}.md`),
    resolving `root` with `_offload_fallback_root` so the persistent
    `~/.deepagents` location and any temporary fallback are both covered.
    Handoff snapshots encode their source thread's ownership in their filenames
    and are removed with that source, including snapshots from failed handoffs.
    A child summary's recovery link lasts only as long as its source archive.

    Best-effort: filesystem failures are logged and swallowed rather than
    raised, so a failed cleanup never blocks thread deletion. Resolving the
    offload root is not side-effect-free -- it creates (and hardens) the
    `conversation_history` directory and writes a short-lived probe file -- so a
    call for a thread that has no archive still touches the filesystem before
    returning `False`.

    In server/sandbox mode the archive lives on the sandbox backend rather than
    the local `~/.deepagents` directory, so there is no local archive to remove.

    Args:
        thread_id: Thread whose offloaded history should be removed.

    Returns:
        `True` if at least one archive file was removed. `False` otherwise:
        an empty or rejected `thread_id`, an unresolvable offload root, a missing
        archive, or an `unlink` failure.
    """
    if not thread_id:
        return False
    try:
        archive_dir = _offload_fallback_root() / CONVERSATION_HISTORY_DIRNAME
    except OSError, RuntimeError:
        logger.warning(
            "Could not resolve offload root to clean history for thread %s",
            thread_id,
            exc_info=True,
        )
        return False
    archive_path = archive_dir / f"{thread_id}.md"
    # Guard against a crafted thread id escaping the archive directory. Thread
    # ids are system-generated UUID7 strings, so a rejection here means either a
    # crafted input or a bug emitting malformed ids -- both worth a trace, and
    # both distinct from the benign "no archive exists" path below.
    if archive_path.parent != archive_dir:
        logger.warning(
            "Refusing to delete offloaded history for suspicious thread id %r",
            thread_id,
        )
        return False
    deleted = _delete_archive(archive_path)
    return _delete_handoff_archives(archive_dir, thread_id) or deleted
