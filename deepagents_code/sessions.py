"""Thread management using LangGraph's built-in checkpoint persistence."""

# [해설] dcode의 스레드(=세션) 저장소. 별도 세션 테이블을 만들지 않고 LangGraph
# [해설]   체크포인터가 만든 SQLite 파일을 그대로 읽는다.
# [해설] DB 위치: `get_db_path()` = `DEFAULT_STATE_DIR/sessions.db`. 상태 디렉터리는
# [해설]   프로필 루트의 `.state/`다(client/launch/server_manager.py의
# [해설]   `_write_checkpointer`가 이 경로를 환경변수로 서버 프로세스에 넘기므로 서버가
# [해설]   쓰는 체크포인트와 이 모듈이 읽는 DB가 같은 파일이다).
# [해설] 핵심 테이블: LangGraph가 만드는 `checkpoints`(체크포인트마다 한 행, `metadata`
# [해설]   JSON에 agent_name·cwd·updated_at·git_branch)와 `writes`,
# [해설]   그리고 dcode가 덧붙이는 `dcode_thread_names`,
# [해설]   `dcode_thread_message_counts`(메시지 수 캐시), `dcode_btw_prompts`.
# [해설][흐름] 새 스레드: `generate_thread_id`(UUID7) -> 서버가 실행하며 체크포인트 기록
# [해설]   -> `list_threads`가 thread_id로 GROUP BY해서 목록 생성
# [해설]   -> 비대화 재개(`-r <정확한 ID>`)는 client/headless_resume.py의
# [해설]   `resolve_resume_target`이
# [해설]   `thread_exists`/`get_thread_agent`/`get_thread_cwd`로 대상을 확인한다.
# [해설][설계] 목록 속도: `checkpoints` 행은 상태 blob을 함께 담아 크므로, 필요한
# [해설]   metadata 식만 담은 커버링 인덱스(`_THREADS_LIST_INDEX`)를 만들어
# [해설]   인덱스만 훑게 한다. 메시지 수·첫 프롬프트는 비싸서 메모리 캐시 -> 저장된 수
# [해설]   -> writes 재생 순으로 계산한다.
# [해설][주의] 이 파일은 `except A, B:`(괄호 없는 다중 예외, PEP 758) 문법을 쓰므로
# [해설]   Python 3.14 이상에서만 파싱된다.
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, NotRequired, TypedDict, cast

from deepagents_code._constants import DEFAULT_THREAD_LIMIT
from deepagents_code._paths import harden_state_dir
from deepagents_code.goal_state_notice import is_internal_message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    import aiosqlite
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from deepagents_code.output import OutputFormat

logger = logging.getLogger(__name__)

# [해설] 모듈 수준 캐시/플래그 모음. `_message_count_cache`·`_initial_prompt_cache`는
# [해설]   체크포인트 freshness 토큰이 같으면 재사용하고,
# [해설]   `_recent_threads_cache`는 `/threads` 선택창이 첫 화면을 빨리 그리도록 최근
# [해설]   목록(필터 없는 결과만)을 보관한다. 크기 상한이 있다.
_aiosqlite_patched = False
_jsonplus_serializer: JsonPlusSerializer | None = None
_message_count_cache: dict[str, tuple[str | None, int]] = {}
_MAX_MESSAGE_COUNT_CACHE = 4096
_initial_prompt_cache: dict[str, tuple[str | None, str | None]] = {}
_MAX_INITIAL_PROMPT_CACHE = 4096
_recent_threads_cache: dict[tuple[str | None, int], list[ThreadInfo]] = {}
_MAX_RECENT_THREADS_CACHE_KEYS = 16
MAX_THREAD_NAME_LENGTH = 50
_DEFAULT_SQLITE_TIMEOUT = 5.0
"""Seconds to wait out a locked database; matches the `sqlite3` default."""


# [해설] langgraph-checkpoint 2.1 이상이 요구하는 `Connection.is_alive()`가 aiosqlite에
# [해설]   없으면 런타임에 붙인다(원인: langgraph 이슈 6583). 한 번만 적용.
def _patch_aiosqlite() -> None:
    """Patch aiosqlite.Connection with `is_alive()` if missing.

    Required by langgraph-checkpoint>=2.1.0.
    See: https://github.com/langchain-ai/langgraph/issues/6583
    """
    global _aiosqlite_patched  # noqa: PLW0603  # Module-level flag requires global statement
    if _aiosqlite_patched:
        return

    import aiosqlite as _aiosqlite

    if not hasattr(_aiosqlite.Connection, "is_alive"):

        def _is_alive(self: _aiosqlite.Connection) -> bool:
            """Check if the connection is still alive.

            Returns:
                True if connection is alive, False otherwise.
            """
            return bool(self._running and self._connection is not None)

        # Dynamically adding a method to aiosqlite.Connection at runtime.
        # Type checkers can't understand this monkey-patch, so we suppress the
        # "attr-defined" error that would otherwise be raised.
        _aiosqlite.Connection.is_alive = _is_alive  # ty: ignore[unresolved-attribute]

    _aiosqlite_patched = True


# [해설] 연결을 닫은 뒤 aiosqlite 워커 스레드가 실제로 끝나도록 join한다. 닫힌 이벤트
# [해설]   루프에 콜백을 던져 나는 RuntimeError(테스트 경고)를 막는다.
async def _drain_aiosqlite_worker(conn: aiosqlite.Connection) -> None:
    """Join the aiosqlite worker thread after its connection is closed.

    `aiosqlite.Connection` wraps a daemon `Thread` (`conn._thread`) that
    drains its tx queue independently of the caller's event loop. The
    library's `close()` puts a stop sentinel on the queue and awaits the
    sentinel's future, but does not explicitly join the worker thread.

    If the connection is leaked (no explicit close) and the surrounding
    event loop has already shut down, the worker can still pop a queued
    item (typically from `Connection.__del__` calling `stop()`) and call
    `future.get_loop().call_soon_threadsafe(...)` on the closed loop. That
    raises `RuntimeError: Event loop is closed`, which pytest surfaces as
    `PytestUnhandledThreadExceptionWarning` (and GitHub Actions then
    surfaces as a workflow annotation).

    Explicitly joining the worker thread after close guarantees it has
    exited before this coroutine returns, eliminating the race for any
    connection routed through `_connect` / `get_checkpointer`.
    """
    worker = getattr(conn, "_thread", None)
    if worker is None or not worker.is_alive():
        return
    # `RuntimeError` covers the "thread was never started" case; treat as
    # already drained.
    with contextlib.suppress(RuntimeError):
        await asyncio.to_thread(worker.join, 5.0)


# [해설] 앱 종료 시 연결을 여는 중에 태스크가 취소돼도 sqlite 핸들이 새지 않게 하는 방어
# [해설]   코드. aiosqlite 내부 속성을 건드리므로 없으면 조용히 포기한다.
# [해설][주의] 비공개 속성(`_connector`, `_tx`)에 의존한다. aiosqlite가 바뀌면 이 가드만
# [해설]   꺼지고 기능은 유지된다.
def _guard_sqlite_handle(conn: aiosqlite.Connection) -> None:
    """Keep the sqlite handle closable when the opening task is cancelled.

    `aiosqlite` opens the database on its worker thread and delivers the raw
    `sqlite3.Connection` back through a future, recording it on the
    `Connection` only once the awaiting coroutine resumes. Background workers
    are routinely cancelled at app exit, and a cancel landing anywhere in that
    window leaves the handle unreachable from the cleanup that follows:

    - Cancelled while the worker is still opening, the library has no handle
        recorded yet, so the cleanup it queues closes nothing.
    - Cancelled after the handle is delivered but before the coroutine resumes,
        the library clears its own record before that queued cleanup can run, so
        again it closes nothing.

    Either way the garbage collector is left to report `ResourceWarning:
    unclosed database`. Recording the handle from the worker thread covers the
    first case; queueing an explicit close ahead of the library's own cleanup
    covers the second. Both run on the thread that opened the handle, and
    closing twice is a no-op, so neither disturbs a normal shutdown.

    Args:
        conn: A connection that has not been opened yet.
    """
    # No public hooks for any of this, so tolerate it moving: the leak avoided
    # here is a warning at teardown, not something worth failing a query for.
    connector = getattr(conn, "_connector", None)
    queue = getattr(conn, "_tx", None)
    stop = getattr(conn, "stop", None)
    if connector is None or queue is None or stop is None:
        logger.debug("aiosqlite internals moved; cannot guard the sqlite handle")
        return

    def open_and_record() -> sqlite3.Connection:
        handle = connector()
        # The assignment aiosqlite makes once the awaiting coroutine resumes,
        # made early enough that a cancel cannot get in front of it.
        conn._connection = handle
        return handle

    def stop_and_close() -> asyncio.Future[Any] | None:
        # Runs before aiosqlite drops its own reference, so the handle is still
        # here to queue a close for -- ahead of the stop sentinel, which ends
        # the worker loop. A `None` future keeps the worker from reaching for an
        # event loop that may already be gone.
        handle = conn._connection
        if handle is not None:
            queue.put_nowait((None, handle.close))
        return stop()

    conn._connector = open_and_record
    # Shadows the bound method on this one instance; the declared type is the
    # unbound `stop(self)`, which a zero-argument replacement cannot match.
    conn.stop = stop_and_close  # ty: ignore[invalid-assignment]


# [해설] 아직 열지 않은 sessions.db 연결을 만든다(패치 적용 + 핸들 가드).
# [해설]   `get_checkpointer`와 `_connect`가 공용으로 쓴다.
def _new_connection(timeout: float = _DEFAULT_SQLITE_TIMEOUT) -> aiosqlite.Connection:
    """Build an unopened connection to the sessions database.

    Args:
        timeout: Seconds to wait out a locked database before giving up.

    Returns:
        A connection that closes its sqlite handle even when interrupted.
    """
    import aiosqlite as _aiosqlite

    _patch_aiosqlite()

    conn = _aiosqlite.connect(str(get_db_path()), timeout=timeout)
    _guard_sqlite_handle(conn)
    return conn


# [해설] 이 모듈의 모든 DB 함수가 쓰는 연결 컨텍스트 매니저. 30초 대기(락 경합 허용)로
# [해설]   열고, 끝나면 워커 스레드를 정리한다.
@asynccontextmanager
async def _connect() -> AsyncIterator[aiosqlite.Connection]:
    """Import aiosqlite, apply the compatibility patch, and connect.

    Centralizes the deferred import + patch + connect sequence used by every
    database function in this module.

    Yields:
        An open aiosqlite connection to the sessions database.
    """
    conn = _new_connection(timeout=30.0)
    try:
        async with conn as opened:
            yield opened
    finally:
        await _drain_aiosqlite_worker(conn)


# [해설] 스레드 목록의 한 행(TypedDict). 필수 키는 thread_id·agent_name·updated_at,
# [해설]   나머지(`NotRequired`)는 비싼 계산이라 필요할 때만 채운다.
class ThreadInfo(TypedDict):
    """Thread metadata returned by `list_threads`."""

    thread_id: str
    """Unique identifier for the thread."""

    agent_name: str | None
    """Name of the agent that owns the thread."""

    updated_at: str | None
    """ISO timestamp of the last update."""

    created_at: NotRequired[str | None]
    """ISO timestamp of thread creation (earliest checkpoint)."""

    git_branch: NotRequired[str | None]
    """Git branch active when the thread was created."""

    initial_prompt: NotRequired[str | None]
    """First human message in the thread."""

    thread_name: NotRequired[str | None]
    """User-assigned or generated thread name."""

    message_count: NotRequired[int]
    """Number of messages in the thread."""

    latest_checkpoint_id: NotRequired[str | None]
    """Most recent checkpoint ID for cache invalidation."""

    cwd: NotRequired[str | None]
    """Working directory where the thread was last used."""


# [해설] 최신 체크포인트에서 뽑은 요약(메시지 수, 첫 프롬프트). message_count가 None이면
# [해설]   체크포인트가 messages를 담지 않은 것이므로 writes 테이블에서 재구성해야 한다.
class _CheckpointSummary(NamedTuple):
    """Structured data extracted from a thread's latest checkpoint."""

    message_count: int | None
    """Number of messages inlined in the latest checkpoint, or `None`.

    `None` means the latest checkpoint did not inline the `messages` channel
    value, so the count is unknown and must be reconstructed from the `writes`
    table. This happens when `messages` uses a `DeltaChannel` (a LangGraph
    channel the deepagents SDK applies to `messages` as of v0.6) and the latest
    checkpoint falls between periodic snapshots. An `int` (including `0`) is a
    trustworthy count.
    """

    initial_prompt: str | None
    """First human prompt recovered from the latest checkpoint."""


# [해설] 아래 `format_*` 함수들은 목록 출력용 표시 헬퍼(시각, 상대 시각, 경로 축약)이며
# [해설]   저장소 로직과 무관하다.
def format_timestamp(iso_timestamp: str | None) -> str:
    """Format ISO timestamp for display (e.g., 'dec 05, 6:10pm').

    Args:
        iso_timestamp: ISO 8601 timestamp string, or `None`.

    Returns:
        Formatted timestamp string or empty string if invalid.
    """
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp).astimezone()
    except ValueError, TypeError:
        logger.debug(
            "Failed to parse timestamp %r; displaying as blank",
            iso_timestamp,
            exc_info=True,
        )
        return ""
    # `%-I` (12-hour clock, no zero padding) is a glibc/BSD extension. MSVC's
    # CRT rejects it as an invalid formatting code, which CPython surfaces as
    # `ValueError`, so the hour is derived by hand to keep every platform on
    # the same rendering.
    hour_12 = dt.hour % 12 or 12
    return f"{dt:%b %d}, {hour_12}:{dt:%M}{dt:%p}".lower()


def format_relative_timestamp(iso_timestamp: str | None) -> str:
    """Format ISO timestamp as relative time (e.g., '5m ago', '2h ago').

    Args:
        iso_timestamp: ISO 8601 timestamp string, or `None`.

    Returns:
        Relative time string or empty string if invalid.
    """
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp).astimezone()
    except ValueError, TypeError:
        logger.debug(
            "Failed to parse timestamp %r; displaying as blank",
            iso_timestamp,
            exc_info=True,
        )
        return ""

    delta = datetime.now(tz=dt.tzinfo) - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:  # noqa: PLR2004
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:  # noqa: PLR2004
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:  # noqa: PLR2004
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:  # noqa: PLR2004
        return f"{days}d ago"
    if days < 365:  # noqa: PLR2004
        months = days // 30
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"


def format_path(path: str | None) -> str:
    """Format a filesystem path for display.

    Paths under the user's home directory are shown relative to `~`.
    All other paths are returned as-is.

    Args:
        path: Absolute filesystem path, or `None`.

    Returns:
        Formatted path string, or empty string if path is falsy.
    """
    if not path:
        return ""
    try:
        home = str(Path.home())
        if path == home:
            return "~"
        prefix = home + "/"
        if path.startswith(prefix):
            return "~/" + path[len(prefix) :]
    except RuntimeError, KeyError, OSError:
        logger.debug(
            "Could not resolve home directory for path formatting", exc_info=True
        )
        return path
    else:
        return path


_db_path: Path | None = None


# [해설] 전역 sessions.db 경로. 첫 호출 때 상태 디렉터리를 권한 0700으로
# [해설]   만들고(`harden_state_dir`) 결과를 `_db_path`에 캐시한다.
# [해설][주의] `DEFAULT_STATE_DIR` 이름을 직접 import해 쓰는 이유가 코드 주석에 있다:
# [해설]   테스트가 이 이름을 따로 패치하기 때문이다.
def get_db_path() -> Path:
    """Get path to global database.

    The result is cached after the first successful call to avoid repeated
    filesystem operations.

    Returns:
        Path to the SQLite database file.
    """
    global _db_path  # noqa: PLW0603  # Module-level cache requires global statement
    if _db_path is not None:
        return _db_path
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    # Pass the directory rather than letting it default to
    # `PATHS.profile.state_dir`: the database has always been located from
    # `DEFAULT_STATE_DIR`, and tests patch that name on its own.
    harden_state_dir(DEFAULT_STATE_DIR)
    _db_path = DEFAULT_STATE_DIR / "sessions.db"
    return _db_path


# [해설] 새 스레드 ID는 UUID7 전체 문자열. 시간순이라 ID 정렬이 곧 생성 순서이고,
# [해설]   `checkpoint_id`로 최신 여부를 판단하는 쿼리와도 잘 맞는다.
def generate_thread_id() -> str:
    """Generate a new thread ID as a full UUID7 string.

    Returns:
        UUID7 string (time-ordered for natural sort by creation time).
    """
    from uuid_utils import uuid7

    return str(uuid7())


# [해설] 첫 실행 전에는 `checkpoints` 같은 테이블이 아직 없을 수 있어서, 대부분의 쿼리
# [해설]   함수가 시작 전에 이것으로 존재를 확인하고 빈 결과를 돌려준다.
async def _table_exists(conn: aiosqlite.Connection, table: str) -> bool:
    """Check if a table exists in the database.

    Returns:
        True if table exists, False otherwise.
    """
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    async with conn.execute(query, (table,)) as cursor:
        return await cursor.fetchone() is not None


# [해설][설계] 아래 docstring이 설명하는 커버링 인덱스 이름. 쿼리가 읽는
# [해설]   식(`json_extract(metadata, ...)`)과 인덱스 정의를 반드시 같게 유지해야 인덱스
# [해설]   전용 스캔이 된다.
_THREADS_LIST_INDEX = "idx_dcode_threads_list_v2"
"""Covering index that makes the `list_threads` GROUP BY an index-only scan.

LangGraph's `SqliteSaver` stores each checkpoint's full state blob inline in the
`checkpoints` row alongside the small `metadata` field. The thread-list query
only needs `metadata` (per-thread latest `updated_at`, `agent_name`, etc.), but
without a covering index SQLite scans the whole table — dragging every state
blob through I/O. On a large profile (e.g. ~12 GB of blobs) that scan takes
tens of seconds. This index carries exactly the expressions the query reads, so
the planner satisfies the GROUP BY from the index alone and never touches the
blob-bearing rows, turning a ~60 s scan into a sub-second lookup.

The column order (leading `thread_id`) also lets the GROUP BY consume the index
in order. Keep the indexed expressions in sync with the `list_threads` query.
"""


# [해설] `list_threads` 직전에 인덱스를 보장한다(v2 또는 v3가 있으면 재사용). 실패해도
# [해설]   느린 전체 스캔으로 대체되므로 로그만 남기고 계속한다.
async def _ensure_threads_list_index(conn: aiosqlite.Connection) -> None:
    """Create the `list_threads` covering index if it does not already exist.

    Reuse either v2 or its v3 superset from early thread-naming builds. Rebuilding
    it to include names can hold the write lock past workspace binding's timeout.
    Names are loaded separately for the displayed threads instead.

    The one-time build on a pre-existing large database costs a
    single full table scan (seconds to tens of seconds), after which every
    `list_threads` call is a sub-second index-only scan. Runs in the aiosqlite
    worker thread, so it does not block the event loop.

    A failure here is non-fatal: the list query still returns correct results via
    the slower table scan, so we log and continue rather than break `threads
    list` (e.g. on a read-only database or under write-lock contention).
    """
    try:
        async with conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name IN (?, ?)",
            (_THREADS_LIST_INDEX, "idx_dcode_threads_list_v3"),
        ) as cursor:
            if await cursor.fetchone():
                return
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {_THREADS_LIST_INDEX} ON checkpoints("
            "thread_id, "
            "json_extract(metadata, '$.updated_at'), "
            "COALESCE(json_extract(metadata, '$.created_at'), "
            "json_extract(metadata, '$.updated_at')), "
            "checkpoint_id, "
            "json_extract(metadata, '$.agent_name'), "
            "json_extract(metadata, '$.git_branch'), "
            "json_extract(metadata, '$.cwd'))"
        )
        # Replace the older index, which did not cover the creation timestamp.
        await conn.execute("DROP INDEX IF EXISTS idx_dcode_threads_list")
        await conn.commit()
    except Exception:
        logger.warning(
            "Failed to create the %s index; `threads list` will fall back to a "
            "full table scan and may be slow on large databases",
            _THREADS_LIST_INDEX,
            exc_info=True,
        )


# [해설] 이름은 인덱스에 넣지 않고, 표시할 스레드에 대해서만 `dcode_thread_names` 테이블
# [해설]   -> 체크포인트 metadata 순으로 따로 읽는다(인덱스 재빌드가 쓰기 락을 오래 잡는
# [해설]   것을 피함).
async def _populate_thread_names(
    conn: aiosqlite.Connection, threads: list[ThreadInfo]
) -> None:
    """Load names after limiting the list, keeping its covering index unchanged."""
    names: dict[str, str | None] = {}
    if await _table_exists(conn, "dcode_thread_names"):
        async with conn.execute(
            "SELECT thread_id, name FROM dcode_thread_names "
            "WHERE thread_id IN (SELECT value FROM json_each(?))",
            (json.dumps([thread["thread_id"] for thread in threads]),),
        ) as cursor:
            names = {row[0]: row[1] for row in await cursor.fetchall()}
    missing = [
        thread["thread_id"] for thread in threads if thread["thread_id"] not in names
    ]
    if missing:
        names.update(await _checkpoint_thread_names(conn, missing))
    for thread in threads:
        thread["thread_name"] = names.get(thread["thread_id"])


# [해설] 저장된 이름이 없는 스레드를 위한 폴백: 각 스레드의 최신 루트
# [해설]   체크포인트(`checkpoint_ns = ''`)의 `thread_name`을 기본키 탐색으로 읽는다.
async def _checkpoint_thread_names(
    conn: aiosqlite.Connection, thread_ids: list[str]
) -> dict[str, str | None]:
    """Read names from the latest root checkpoints of the requested threads.

    Saved names come from `dcode_thread_names`; this fallback is for checkpoint
    metadata only. Seek via the primary key instead of scanning an unnamed
    thread's entire history looking for a non-null name.

    Returns:
        Names found in the latest checkpoints, keyed by thread ID.
    """
    async with conn.execute(
        "SELECT value, (SELECT json_extract(metadata, '$.thread_name') "
        "FROM checkpoints WHERE thread_id = requested.value "
        "AND checkpoint_ns = '' "
        "ORDER BY checkpoint_id DESC LIMIT 1) FROM json_each(?) AS requested",
        (json.dumps(thread_ids),),
    ) as cursor:
        return {row[0]: row[1] for row in await cursor.fetchall()}


# [해설] `dcode threads list`와 스레드 선택창의 기반 쿼리. checkpoints를 thread_id로
# [해설]   GROUP BY해 에이전트·최근 갱신·생성 시각·브랜치·cwd를 구한다.
# [해설][흐름] 1) 연결 -> 테이블 없으면 빈 목록  2) 인덱스 보장  3)
# [해설]   필터(agent/branch/cwd)와 정렬 열 구성  4) 집계 쿼리  5) 이름 채우기
# [해설]   6) 요청 시 메시지 수 채우기  7) 필터 없는 `updated` 정렬 결과만
# [해설]   `_cache_recent_threads`에 저장.
# [해설][주의] thread_id 외 모든 열을 MAX/MIN으로 집계하는 것은 의도다. 맨 열(bare
# [해설]   column)을 두면 SQLite가 행을 읽어 커버링 인덱스 계획이 깨진다.
# [해설][주의] `cwd` 필터는 정확한 문자열 비교다(정규화·심볼릭 링크 해석·접두 일치
# [해설]   없음). cwd가 없는 옛 행은 제외된다.
async def list_threads(
    agent_name: str | None = None,
    limit: int = DEFAULT_THREAD_LIMIT,
    include_message_count: bool = False,
    sort_by: str = "updated",
    branch: str | None = None,
    cwd: str | None = None,
) -> list[ThreadInfo]:
    """List threads from checkpoints table.

    Args:
        agent_name: Optional filter by agent name.
        limit: Maximum number of threads to return.
        include_message_count: Whether to include message counts.
        sort_by: Sort field — `"updated"` or `"created"`.
        branch: Optional filter by git branch name.
        cwd: Optional filter by working directory. Only threads whose stored
            `cwd` metadata equals this path are returned. Matching is an
            exact string comparison — no path normalization, symlink
            resolution, or prefix matching. Threads without a stored `cwd`
            (older rows) are excluded.

    Returns:
        List of `ThreadInfo` dicts with `thread_id`, `agent_name`,
            `updated_at`, `created_at`, `latest_checkpoint_id`, `git_branch`,
            `cwd`, and optionally `message_count`.

    Raises:
        ValueError: If `sort_by` is not `"updated"` or `"created"`.
    """
    started_at = time.perf_counter()
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            logger.debug(
                "Thread metadata listing completed in %.1f ms: rows=0",
                (time.perf_counter() - started_at) * 1000,
            )
            return []

        # Ensure the covering index exists before the GROUP BY below, so the
        # query is an index-only scan instead of a full scan over the (large,
        # blob-bearing) checkpoints table.
        await _ensure_threads_list_index(conn)

        if sort_by not in {"updated", "created"}:
            msg = f"Invalid sort_by {sort_by!r}; expected 'updated' or 'created'"
            raise ValueError(msg)
        order_col = "created_at" if sort_by == "created" else "updated_at"

        where_clauses: list[str] = []
        params_list: list[str | int] = []

        if agent_name:
            where_clauses.append("json_extract(metadata, '$.agent_name') = ?")
            params_list.append(agent_name)
        if branch:
            where_clauses.append("json_extract(metadata, '$.git_branch') = ?")
            params_list.append(branch)
        if cwd:
            where_clauses.append("json_extract(metadata, '$.cwd') = ?")
            params_list.append(cwd)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Aggregate every column except `thread_id`. A bare column makes SQLite
        # read one table row per group, which drops the covering-index plan.
        query = f"""
            SELECT thread_id,
                   MAX(json_extract(metadata, '$.agent_name')) as agent_name,
                   MAX(json_extract(metadata, '$.updated_at')) as updated_at,
                   MAX(checkpoint_id) as latest_checkpoint_id,
                   MIN(COALESCE(json_extract(metadata, '$.created_at'),
                                json_extract(metadata, '$.updated_at'))) as created_at,
                   MAX(json_extract(metadata, '$.git_branch')) as git_branch,
                   MAX(json_extract(metadata, '$.cwd')) as cwd
            FROM checkpoints
            {where_sql}
            GROUP BY thread_id
            ORDER BY {order_col} DESC
            LIMIT ?
        """  # noqa: S608  # where_sql/order_col derived from controlled internal values; user values use ? placeholders
        params: tuple = (*params_list, limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            threads: list[ThreadInfo] = [
                ThreadInfo(
                    thread_id=r[0],
                    agent_name=r[1],
                    updated_at=r[2],
                    latest_checkpoint_id=r[3],
                    created_at=r[4],
                    git_branch=r[5],
                    cwd=r[6],
                )
                for r in rows
            ]

        if threads:
            await _populate_thread_names(conn, threads)

        # Fetch message counts if requested
        if include_message_count and threads:
            await _populate_message_counts(conn, threads)

        # Only cache unfiltered results so the thread selector modal
        # doesn't receive branch-/cwd-filtered or differently-sorted data.
        if sort_by == "updated" and branch is None and cwd is None:
            _cache_recent_threads(agent_name, limit, threads)
        logger.debug(
            "Thread metadata listing completed in %.1f ms: rows=%d sort=%s",
            (time.perf_counter() - started_at) * 1000,
            len(threads),
            sort_by,
        )
        return threads


# [해설] 이미 만든 목록에 체크포인트 기반 필드(메시지 수, 첫 프롬프트)를 백그라운드에서
# [해설]   한 번에 채운다. `/threads` 창이 쓴다.
async def populate_thread_checkpoint_details(
    threads: list[ThreadInfo],
    *,
    include_message_count: bool = True,
    include_initial_prompt: bool = True,
) -> list[ThreadInfo]:
    """Populate checkpoint-derived fields for an existing thread list.

    This is used by the `/threads` modal to enrich rows in one background pass,
    so the latest checkpoint is fetched and deserialized at most once per row.
    Computed message counts are also saved to the sessions database, best
    effort, so later sessions can reuse them.

    Args:
        threads: Thread rows to enrich in place.
        include_message_count: Whether to populate `message_count`.
        include_initial_prompt: Whether to populate `initial_prompt`.

    Returns:
        The same list object with missing checkpoint-derived fields populated.
    """
    if not threads or (not include_message_count and not include_initial_prompt):
        return threads

    return await _enrich_thread_checkpoint_details(
        threads,
        include_message_count=include_message_count,
        include_initial_prompt=include_initial_prompt,
        source="picker",
    )


async def _enrich_thread_checkpoint_details(
    threads: list[ThreadInfo],
    *,
    include_message_count: bool,
    include_initial_prompt: bool,
    source: str,
) -> list[ThreadInfo]:
    """Enrich threads and record the caller context for diagnostics.

    Returns:
        The enriched thread list.
    """
    started_at = time.perf_counter()
    async with _connect() as conn:
        await _populate_checkpoint_fields(
            conn,
            threads,
            include_message_count=include_message_count,
            include_initial_prompt=include_initial_prompt,
        )
    logger.debug(
        "Thread enrichment completed in %.1f ms: source=%s rows=%d "
        "counts=%s prompts=%s",
        (time.perf_counter() - started_at) * 1000,
        source,
        len(threads),
        include_message_count,
        include_initial_prompt,
    )
    return threads


# [해설] 앱 시작 시(그리고 DB가 바뀐 뒤) 선택창 캐시를 미리 데워 첫 화면 지연을 없앤다.
# [해설]   freshness가 같은 스레드는 다시 읽지 않아 재실행이 싸다.
async def prewarm_thread_message_counts(limit: int | None = None) -> None:
    """Prewarm thread selector cache for faster `/threads` open.

    Fetches a bounded list of recent threads and populates checkpoint-derived
    fields for currently visible columns into the in-memory cache (message
    counts are also saved to the sessions database, best effort). Intended to
    run in a background worker during app startup and again whenever the
    session database has changed (e.g. after a turn writes new checkpoints), so
    the selector's first paint is never missing a thread the user just created.

    Re-running this is cheap: the per-thread message-count and initial-prompt
    caches are keyed on checkpoint freshness, so only threads whose latest
    checkpoint changed are read back from disk.

    Args:
        limit: Maximum threads to prewarm. Uses `get_thread_limit()` when `None`.
    """
    thread_limit = limit if limit is not None else get_thread_limit()
    if thread_limit < 1:
        return

    try:
        from deepagents_code.model_config import load_thread_config

        cfg = load_thread_config()
        threads = await list_threads(limit=thread_limit, include_message_count=False)
        if threads:
            await _enrich_thread_checkpoint_details(
                threads,
                include_message_count=cfg.columns.get("messages", False),
                include_initial_prompt=cfg.columns.get("initial_prompt", False),
                source="startup-prewarm",
            )
        _cache_recent_threads(None, thread_limit, threads)
    except OSError, sqlite3.Error:
        logger.debug("Could not prewarm thread selector cache", exc_info=True)
    except Exception:
        logger.warning(
            "Unexpected error while prewarming thread selector cache",
            exc_info=True,
        )


# [해설] 메모리 캐시에서 최근 스레드 목록을 복사해 돌려준다(없으면 None). DB를 열지
# [해설]   않는다.
def get_cached_threads(
    agent_name: str | None = None,
    limit: int | None = None,
) -> list[ThreadInfo] | None:
    """Get cached recent threads, if available.

    Args:
        agent_name: Optional agent-name filter key.
        limit: Maximum rows requested. Uses `get_thread_limit()` when `None`.

    Returns:
        Copy of cached rows when available, otherwise `None`.
    """

    def _copy_with_cached_counts(rows: list[ThreadInfo]) -> list[ThreadInfo]:
        copied_rows = _copy_threads(rows)
        apply_cached_thread_message_counts(copied_rows)
        apply_cached_thread_initial_prompts(copied_rows)
        return copied_rows

    thread_limit = limit if limit is not None else get_thread_limit()
    if thread_limit < 1:
        return None

    exact = _recent_threads_cache.get((agent_name, thread_limit))
    if exact is not None:
        return _copy_with_cached_counts(exact)

    best_key: tuple[str | None, int] | None = None
    for key in _recent_threads_cache:
        cache_agent, cache_limit = key
        if cache_agent != agent_name or cache_limit < thread_limit:
            continue
        if best_key is None or cache_limit < best_key[1]:
            best_key = key

    if best_key is None:
        return None

    return _copy_with_cached_counts(_recent_threads_cache[best_key][:thread_limit])


def apply_cached_thread_message_counts(threads: list[ThreadInfo]) -> int:
    """Apply cached message counts onto thread rows when freshness matches.

    Args:
        threads: Thread rows to mutate in place.

    Returns:
        Number of rows that were populated from cache.
    """
    populated = 0
    for thread in threads:
        if "message_count" in thread:
            continue
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        cached = _message_count_cache.get(thread_id)
        if cached is None or cached[0] != freshness:
            continue
        thread["message_count"] = cached[1]
        populated += 1
    return populated


def apply_cached_thread_initial_prompts(threads: list[ThreadInfo]) -> int:
    """Apply cached initial prompts onto thread rows when freshness matches.

    Args:
        threads: Thread rows to mutate in place.

    Returns:
        Number of rows that were populated from cache.
    """
    populated = 0
    for thread in threads:
        if "initial_prompt" in thread:
            continue
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        cached = _initial_prompt_cache.get(thread_id)
        if cached is None or cached[0] != freshness:
            continue
        thread["initial_prompt"] = cached[1]
        populated += 1
    return populated


# [해설] 목록에 메시지 수를 채우는 진입점. 실제 계산은 `_populate_checkpoint_fields`가
# [해설]   캐시가 싼 순서로 시도한다.
async def _populate_message_counts(
    conn: aiosqlite.Connection,
    threads: list[ThreadInfo],
) -> None:
    """Fill `message_count` on thread rows with cache-aware lookup."""
    await _populate_checkpoint_fields(
        conn,
        threads,
        include_message_count=True,
        include_initial_prompt=False,
    )


# [해설] 체크포인트 blob을 해독하는 LangGraph `JsonPlusSerializer`를 지연 생성해
# [해설]   재사용한다.
async def _get_jsonplus_serializer() -> JsonPlusSerializer:
    """Return a cached JsonPlus serializer, loading it off the UI loop."""
    global _jsonplus_serializer  # noqa: PLW0603  # Module-level cache requires global statement
    if _jsonplus_serializer is not None:
        return _jsonplus_serializer

    started_at = time.perf_counter()
    loop = asyncio.get_running_loop()
    _jsonplus_serializer = await loop.run_in_executor(None, _create_jsonplus_serializer)
    logger.debug(
        "Thread serializer initialization completed in %.1f ms",
        (time.perf_counter() - started_at) * 1000,
    )
    return _jsonplus_serializer


def _create_jsonplus_serializer() -> JsonPlusSerializer:
    """Import and create a JsonPlus serializer.

    Returns:
        A ready `JsonPlusSerializer` instance.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer()


def _cache_message_count(thread_id: str, freshness: str | None, count: int) -> None:
    """Cache a thread's message count with a freshness token."""
    if len(_message_count_cache) >= _MAX_MESSAGE_COUNT_CACHE and (
        thread_id not in _message_count_cache
    ):
        oldest = next(iter(_message_count_cache))
        _message_count_cache.pop(oldest, None)
    _message_count_cache[thread_id] = (freshness, count)


def _cache_initial_prompt(
    thread_id: str,
    freshness: str | None,
    initial_prompt: str | None,
) -> None:
    """Cache a thread's initial prompt with a freshness token."""
    if len(_initial_prompt_cache) >= _MAX_INITIAL_PROMPT_CACHE and (
        thread_id not in _initial_prompt_cache
    ):
        oldest = next(iter(_initial_prompt_cache))
        _initial_prompt_cache.pop(oldest, None)
    _initial_prompt_cache[thread_id] = (freshness, initial_prompt)


def _thread_freshness(thread: ThreadInfo) -> str | None:
    """Return a cache freshness token for a thread row.

    The token is checkpoint-granular (`latest_checkpoint_id`). The
    writes-reconstructed `message_count` includes pending writes on the latest
    checkpoint, which in principle can change without a new checkpoint ID — so
    this token does not capture intra-checkpoint write churn. In practice that
    is benign: dcode only mutates `messages` through the agent graph, and every
    batch of message writes culminates in a new checkpoint (each superstep,
    `aupdate_state`, interrupt, and cancellation all advance
    `latest_checkpoint_id`). The only window where a cached count can lag is
    opening the `/threads` selector mid-superstep against an actively streaming
    thread; the selector does not live-refresh, so that count stays put until
    the modal is reopened (by then a new checkpoint exists and the cache
    refreshes). Making the key write-sensitive would require probing the
    `writes` table for every row on every `list_threads`, which is not worth it
    for a cosmetic count.
    """
    return thread.get("latest_checkpoint_id") or thread.get("updated_at")


def _cache_recent_threads(
    agent_name: str | None,
    limit: int,
    threads: list[ThreadInfo],
) -> None:
    """Store a copy of recent thread rows for fast selector startup."""
    key = (agent_name, max(1, limit))
    # A fresh, longer window serves every shorter request for the same agent,
    # so drop the shorter keys. Otherwise each thread switcher "Load more"
    # batch adds a key until the cap clears the prewarmed entry.
    shorter = [k for k in _recent_threads_cache if k[0] == agent_name and k[1] < key[1]]
    for old in shorter:
        del _recent_threads_cache[old]
    if len(_recent_threads_cache) >= _MAX_RECENT_THREADS_CACHE_KEYS and (
        key not in _recent_threads_cache
    ):
        _recent_threads_cache.clear()
    _recent_threads_cache[key] = _copy_threads(threads)


def _copy_threads(threads: list[ThreadInfo]) -> list[ThreadInfo]:
    """Return shallow-copied thread rows."""
    return [ThreadInfo(**thread) for thread in threads]


# [해설][설계] 비용이 싼 원천부터: 메모리 캐시 -> `dcode_thread_message_counts`에 저장된
# [해설]   수 -> `writes`/`checkpoints` 재생. 새로 계산한 수는 저장해 다음 세션이
# [해설]   건너뛴다.
async def _populate_checkpoint_fields(
    conn: aiosqlite.Connection,
    threads: list[ThreadInfo],
    *,
    include_message_count: bool,
    include_initial_prompt: bool,
) -> None:
    """Populate checkpoint-derived thread fields, cheapest source first.

    Order: the in-memory caches, then counts an earlier session saved to
    `dcode_thread_message_counts`, then the `writes` and `checkpoints` tables.
    Counts computed here are saved so later sessions skip recomputing them.
    """
    serde = await _get_jsonplus_serializer()
    _apply_memory_caches(
        threads,
        include_message_count=include_message_count,
        include_initial_prompt=include_initial_prompt,
    )
    count_ids = [
        t["thread_id"]
        for t in threads
        if include_message_count and "message_count" not in t
    ]
    if count_ids:
        await _apply_saved_message_counts(conn, threads, count_ids)
        count_ids = [t["thread_id"] for t in threads if "message_count" not in t]
    prompt_ids = [
        t["thread_id"]
        for t in threads
        if include_initial_prompt and "initial_prompt" not in t
    ]
    logger.debug(
        "Thread enrichment cache lookup: rows=%d count_misses=%d prompt_misses=%d",
        len(threads),
        len(count_ids),
        len(prompt_ids),
    )
    if not count_ids and not prompt_ids:
        return

    by_id = {t["thread_id"]: t for t in threads}
    summaries = await _load_prompts_and_summaries(
        conn, by_id, count_ids, prompt_ids, serde
    )
    counts = await _load_message_counts(conn, by_id, count_ids, summaries, serde)
    await _save_message_counts(conn, by_id, counts)


def _apply_memory_caches(
    threads: list[ThreadInfo],
    *,
    include_message_count: bool,
    include_initial_prompt: bool,
) -> None:
    """Fill requested fields from the in-memory caches when still fresh."""
    for thread in threads:
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        cached_count = _message_count_cache.get(thread_id)
        if include_message_count and cached_count and cached_count[0] == freshness:
            thread["message_count"] = cached_count[1]
        cached_prompt = _initial_prompt_cache.get(thread_id)
        if (
            include_initial_prompt
            and "initial_prompt" not in thread
            and cached_prompt
            and cached_prompt[0] == freshness
        ):
            thread["initial_prompt"] = cached_prompt[1]


# [해설] 첫 프롬프트는 최신 체크포인트만으로 복구할 수 없다(`after_model` 미들웨어가
# [해설]   messages 없는 부분 체크포인트를 쓰기 때문). 그래서 가장 이른 write를 따로
# [해설]   읽는다.
async def _load_prompts_and_summaries(
    conn: aiosqlite.Connection,
    by_id: dict[str, ThreadInfo],
    count_ids: list[str],
    prompt_ids: list[str],
    serde: JsonPlusSerializer,
) -> dict[str, _CheckpointSummary]:
    """Fill initial prompts and decode the latest checkpoints that are needed.

    Returns:
        Latest checkpoint summaries, for counts that they inline.
    """
    # `initial_prompt` cannot be recovered from the latest checkpoint alone:
    # `after_model` middleware (e.g., `ResumeStateMiddleware`) writes partial
    # checkpoints whose `channel_values` omit `messages`. Read the very first
    # write to the `messages` channel from the `writes` table instead — that
    # row holds the user's original input.
    started_at = time.perf_counter()
    prompts = (
        await _load_initial_prompts_from_writes_batch(conn, prompt_ids, serde)
        if prompt_ids
        else {}
    )
    _log_phase("Initial prompt loading", started_at, len(prompt_ids), len(prompts))

    # A latest checkpoint is decoded only for a count, or for a prompt whose
    # first `messages` write is missing or unreadable.
    fallback_ids = [tid for tid in prompt_ids if tid not in prompts]
    summary_ids = list(dict.fromkeys([*count_ids, *fallback_ids]))
    started_at = time.perf_counter()
    summaries = (
        await _load_latest_checkpoint_summaries_batch(conn, summary_ids, serde)
        if summary_ids
        else {}
    )
    _log_phase(
        "Latest checkpoint loading and decoding",
        started_at,
        len(summary_ids),
        len(summaries),
    )

    for tid in prompt_ids:
        thread = by_id[tid]
        prompt = (
            prompts[tid]
            if tid in prompts
            else summaries.get(tid, _CheckpointSummary(None, None)).initial_prompt
        )
        thread["initial_prompt"] = prompt
        _cache_initial_prompt(tid, _thread_freshness(thread), prompt)
    return summaries


async def _load_message_counts(
    conn: aiosqlite.Connection,
    by_id: dict[str, ThreadInfo],
    count_ids: list[str],
    summaries: dict[str, _CheckpointSummary],
    serde: JsonPlusSerializer,
) -> dict[str, int]:
    """Fill message counts from checkpoint summaries, else the `writes` table.

    When `messages` uses a `DeltaChannel` (LangGraph channel applied by the
    deepagents SDK as of v0.6) the full list is only snapshotted into
    `channel_values` periodically, so the latest checkpoint usually omits it;
    the count must then be reconstructed from the `writes` table.

    Returns:
        The counts filled in, keyed by thread ID.
    """
    counts: dict[str, int] = {}
    replay_ids: list[str] = []
    for tid in count_ids:
        summary = summaries.get(tid)
        if summary is not None and summary.message_count is not None:
            counts[tid] = summary.message_count
        else:
            replay_ids.append(tid)
    if replay_ids:
        started_at = time.perf_counter()
        replayed = await _load_message_counts_from_writes_batch(conn, replay_ids, serde)
        _log_phase(
            "Message write fetching and count reconstruction",
            started_at,
            len(replay_ids),
            len(replayed),
        )
        counts.update({tid: replayed.get(tid, 0) for tid in replay_ids})
    for tid, count in counts.items():
        by_id[tid]["message_count"] = count
        _cache_message_count(tid, _thread_freshness(by_id[tid]), count)
    return counts


def _log_phase(label: str, started_at: float, requested: int, loaded: int) -> None:
    """Log how long one enrichment phase took."""
    if requested:
        logger.debug(
            "%s completed in %.1f ms: requested=%d loaded=%d",
            label,
            (time.perf_counter() - started_at) * 1000,
            requested,
            loaded,
        )


# [해설] 메시지 수를 스레드·freshness 토큰 단위로 저장하는 dcode 전용 테이블.
# [해설]   `delete_thread`가 함께 지운다.
_MESSAGE_COUNTS_TABLE = "dcode_thread_message_counts"
"""Message counts saved across sessions, keyed by thread and freshness token.

A count from the `writes` replay can take seconds to compute for a batch of
threads on a large database, so it is saved and reused until the thread's
freshness token (see `_thread_freshness`) changes.
"""


async def _apply_saved_message_counts(
    conn: aiosqlite.Connection,
    threads: list[ThreadInfo],
    thread_ids: list[str],
) -> None:
    """Fill message counts saved by an earlier session when still fresh."""
    saved = await _read_saved_message_counts(conn, thread_ids)
    for thread in threads:
        entry = saved.get(thread["thread_id"])
        freshness = _thread_freshness(thread)
        if entry is None or entry[0] != freshness or "message_count" in thread:
            continue
        thread["message_count"] = entry[1]
        _cache_message_count(thread["thread_id"], freshness, entry[1])


_message_count_save_warned: list[bool] = [False]
"""Whether a non-lock failure to save message counts was already logged."""


def _log_message_count_save_failure(exc: sqlite3.Error) -> None:
    """Log a failed save; warn once for errors that will keep recurring.

    A busy or locked database is expected while another process writes. Other
    errors (a read-only file, a bad schema) make every later save fail too, so
    they are worth one warning per process.
    """
    code = getattr(exc, "sqlite_errorcode", None)
    # Extended result codes keep the primary code in the low byte.
    locked = code is not None and (code & 0xFF) in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }
    if locked or _message_count_save_warned[0]:
        logger.debug("Could not save thread message counts", exc_info=exc)
        return
    _message_count_save_warned[0] = True
    logger.warning("Could not save thread message counts", exc_info=exc)


async def _read_saved_message_counts(
    conn: aiosqlite.Connection, thread_ids: list[str]
) -> dict[str, tuple[str, int]]:
    """Return saved `(freshness, count)` pairs by thread ID; best effort.

    The table is only a cache, so an unreadable one (for example, a schema
    from another version) reads as empty and the counts are recomputed.
    """
    try:
        if not await _table_exists(conn, _MESSAGE_COUNTS_TABLE):
            return {}
        async with conn.execute(
            f"SELECT thread_id, freshness, message_count FROM {_MESSAGE_COUNTS_TABLE} "  # noqa: S608  # constant table name
            "WHERE thread_id IN (SELECT value FROM json_each(?))",
            (json.dumps(thread_ids),),
        ) as cursor:
            rows = await cursor.fetchall()
    except sqlite3.Error:
        logger.warning("Could not read saved thread message counts", exc_info=True)
        return {}
    return {row[0]: (row[1], row[2]) for row in rows}


async def _save_message_counts(
    conn: aiosqlite.Connection,
    by_id: dict[str, ThreadInfo],
    counts: dict[str, int],
) -> None:
    """Save computed message counts for later sessions; best effort.

    A listing must never wait on another connection's write lock, so this
    gives up at once when the database is busy and leaves the counts unsaved.
    """
    rows = [
        (tid, freshness, count)
        for tid, count in counts.items()
        if (freshness := _thread_freshness(by_id[tid])) is not None
    ]
    if not rows:
        return
    async with conn.execute("PRAGMA busy_timeout") as cursor:
        busy_timeout = (await cursor.fetchone() or (0,))[0]
    await conn.execute("PRAGMA busy_timeout = 0")
    try:
        await conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_MESSAGE_COUNTS_TABLE} "
            "(thread_id TEXT PRIMARY KEY NOT NULL, freshness TEXT NOT NULL, "
            "message_count INTEGER NOT NULL)"
        )
        await conn.executemany(
            f"INSERT OR REPLACE INTO {_MESSAGE_COUNTS_TABLE} "  # noqa: S608  # constant table name
            "(thread_id, freshness, message_count) VALUES (?, ?, ?)",
            rows,
        )
        await conn.commit()
    except sqlite3.Error as exc:
        _log_message_count_save_failure(exc)
        with contextlib.suppress(sqlite3.Error):
            await conn.rollback()
    finally:
        await conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout)}")


_SQLITE_MAX_VARIABLE_NUMBER = 500
"""Max `?` placeholders per SQL query.

SQLite limits how many `?` parameters a single query can have (default 999,
lower on some builds). If a user accumulates hundreds of threads and the
`/threads` modal fetches them all at once, the `IN (?, ?, ...)` clause could
exceed that limit. We chunk to this size to stay safe.
"""


# [해설] 윈도 함수로 여러 스레드의 최신 체크포인트를 묶음 조회한다(SQLite 변수 개수 제한
# [해설]   때문에 `_SQLITE_MAX_VARIABLE_NUMBER` 단위로 나눠 실행).
async def _load_latest_checkpoint_summaries_batch(
    conn: aiosqlite.Connection,
    thread_ids: list[str],
    serde: JsonPlusSerializer,
) -> dict[str, _CheckpointSummary]:
    """Batch-load the latest checkpoint summary for multiple threads.

    Uses a window function to fetch the latest checkpoint per thread, issuing
    one query per chunk for SQLite variable-limit safety.

    Args:
        conn: Database connection.
        thread_ids: Thread IDs to look up.
        serde: Serializer for decoding checkpoint blobs.

    Returns:
        Dict mapping thread IDs to their checkpoint summaries.
    """
    if not thread_ids:
        return {}

    results: dict[str, _CheckpointSummary] = {}

    for start in range(0, len(thread_ids), _SQLITE_MAX_VARIABLE_NUMBER):
        chunk = thread_ids[start : start + _SQLITE_MAX_VARIABLE_NUMBER]
        placeholders = ",".join("?" * len(chunk))
        query = f"""
            SELECT c.thread_id, c.type, c.checkpoint
            FROM checkpoints AS c
            JOIN (
                SELECT rowid AS rid,
                       ROW_NUMBER() OVER (
                           PARTITION BY thread_id ORDER BY checkpoint_id DESC
                       ) AS rn
                FROM checkpoints
                WHERE thread_id IN ({placeholders})
            ) AS ranked ON c.rowid = ranked.rid
            WHERE ranked.rn = 1
        """  # noqa: S608  # placeholders built from len(chunk); user values use ? params
        async with conn.execute(query, chunk) as cursor:
            rows = await cursor.fetchall()

        loop = asyncio.get_running_loop()
        for row in rows:
            tid, type_str, checkpoint_blob = row
            if not type_str or not checkpoint_blob:
                results[tid] = _CheckpointSummary(
                    message_count=None, initial_prompt=None
                )
                continue
            try:
                data = await loop.run_in_executor(
                    None, serde.loads_typed, (type_str, checkpoint_blob)
                )
                results[tid] = _summarize_checkpoint(data)
            except Exception:
                logger.warning(
                    "Failed to deserialize checkpoint for thread %s; "
                    "message count and initial prompt may be incomplete",
                    tid,
                    exc_info=True,
                )
                results[tid] = _CheckpointSummary(
                    message_count=None, initial_prompt=None
                )

    return results


async def _load_initial_prompts_from_writes_batch(
    conn: aiosqlite.Connection,
    thread_ids: list[str],
    serde: JsonPlusSerializer,
) -> dict[str, str | None]:
    """Batch-load initial prompts from the LangGraph `writes` table.

    For each thread, returns the first human/user message extracted from the
    earliest root-namespace write to the `messages` channel, in primary-key
    order (`checkpoint_id`, `task_id`, `idx`). Each lookup walks that order and
    stops at the first match, which is usually among the thread's first rows,
    rather than ranking every write the thread has.

    Args:
        conn: Database connection.
        thread_ids: Thread IDs to look up.
        serde: Serializer for decoding write blobs.

    Returns:
        Dict mapping thread IDs to their initial prompt text. Threads with no
        readable write to the `messages` channel are absent from the result;
        threads whose first such write decoded but contained no human/user
        entry map to `None`.
    """
    if not thread_ids:
        return {}

    results: dict[str, str | None] = {}
    loop = asyncio.get_running_loop()
    async with conn.execute(
        "SELECT w.thread_id, w.type, w.value FROM json_each(?) AS requested "
        "JOIN writes AS w ON w.rowid = (SELECT rowid FROM writes "
        "WHERE thread_id = requested.value AND checkpoint_ns = '' "
        "AND channel = 'messages' "
        "ORDER BY checkpoint_id, task_id, idx LIMIT 1)",
        (json.dumps(thread_ids),),
    ) as cursor:
        rows = await cursor.fetchall()

    for row in rows:
        tid, type_str, value_blob = row
        if not type_str or not value_blob:
            continue
        try:
            messages = await loop.run_in_executor(
                None, serde.loads_typed, (type_str, value_blob)
            )
        except Exception:
            logger.warning(
                "Failed to deserialize initial messages write for thread %s",
                tid,
                exc_info=True,
            )
            continue
        if not isinstance(messages, list):
            continue
        results[tid] = _initial_prompt_from_messages(cast("list[object]", messages))

    return results


# [해설][SDK] SDK(deepagents >= 0.6)는 `messages`에 LangGraph `DeltaChannel`을 쓴다.
# [해설]   스냅샷 사이의 체크포인트는 messages 전체를 담지 않으므로,
# [해설]   `writes`의 messages 델타를 재생해 개수를 센다. 정확한 reducer 대신
# [해설]   `add_messages`로 대체하는데, ID 중복 제거와 RemoveMessage 처리가 같아 개수가
# [해설]   같다.
async def _load_message_counts_from_writes_batch(
    conn: aiosqlite.Connection,
    thread_ids: list[str],
    serde: JsonPlusSerializer,
) -> dict[str, int]:
    """Reconstruct message counts from the LangGraph `writes` table.

    For threads whose latest checkpoint does not inline the `messages` channel
    value — a `DeltaChannel` between snapshots, where the deepagents SDK (>= 0.6)
    applies LangGraph's `DeltaChannel` to `messages` — the full list is rebuilt
    by replaying every `messages` write, then counted. We replay through
    `add_messages` as a count-equivalent stand-in for the channel's actual
    reducer (`_messages_delta_reducer`): both dedup by ID and honor
    `RemoveMessage` / `REMOVE_ALL_MESSAGES`, so they produce the same final
    message set. An `Overwrite` write resets the accumulator to its value,
    matching the net effect of `DeltaChannel.replay_writes` (where the last
    `Overwrite` is the reset point).

    Reduction runs in a single worker-thread hop per chunk (decode is CPU-bound
    and a long thread can have thousands of writes; dispatching per row both
    serialized the work and added an executor round-trip each time). The common
    append-and-clear history folds in one `add_messages` pass (linear), which is
    why a busy thread no longer takes seconds to count. See
    `_count_messages_from_deltas` for the fold and its exact-fold fallback.

    Only the root namespace (`checkpoint_ns = ''`) is counted, matching both the
    inline path and the conversation the `/threads` selector cares about;
    subgraph (subagent) writes under the same `thread_id` are excluded.

    Folding the *entire* write history (rather than walking the head
    checkpoint's parent chain) is intentional and matches what dcode shows when
    a thread is opened: it reads state via `aget_state` without a
    `checkpoint_id`, which applies pending writes (`apply_pending_writes=True`),
    so the latest checkpoint's not-yet-committed `messages` writes are part of
    the user-visible list and must be counted. dcode only ever appends to the
    latest checkpoint (no time travel, no `checkpoint_id`-targeted
    `aupdate_state`), so histories are linear and the full fold equals the
    head-of-chain reconstruction. A forked/abandoned branch (which dcode does
    not create) is the only case where this could over-count.

    Args:
        conn: Database connection.
        thread_ids: Thread IDs to look up.
        serde: Serializer for decoding write blobs.

    Returns:
        Dict mapping each thread ID with at least one decodable `messages`
            write to its reconstructed message count. Threads with no such
            writes are absent from the result.
    """
    if not thread_ids:
        return {}

    loop = asyncio.get_running_loop()
    results: dict[str, int] = {}
    # Chunks partition by thread, so every write for a given thread lands in the
    # same query; each thread is counted exactly once. Ordering by
    # (checkpoint_id, task_id, idx) replays deltas oldest-to-newest, matching how
    # LangGraph applies them on load.
    for start in range(0, len(thread_ids), _SQLITE_MAX_VARIABLE_NUMBER):
        chunk = thread_ids[start : start + _SQLITE_MAX_VARIABLE_NUMBER]
        placeholders = ",".join("?" * len(chunk))
        query = f"""
            SELECT thread_id, type, value
            FROM writes
            WHERE thread_id IN ({placeholders})
              AND checkpoint_ns = ''
              AND channel = 'messages'
            ORDER BY thread_id, checkpoint_id ASC, task_id ASC, idx ASC
        """  # noqa: S608  # placeholders built from len(chunk); user values use ? params
        async with conn.execute(query, chunk) as cursor:
            rows = await cursor.fetchall()

        chunk_counts = await loop.run_in_executor(
            None, _reduce_message_write_rows, list(rows), serde
        )
        results.update(chunk_counts)

    return results


def _reduce_message_write_rows(
    rows: list[tuple[str, str | None, bytes | None]],
    serde: JsonPlusSerializer,
) -> dict[str, int]:
    """Decode `messages`-channel write rows and count messages per thread.

    Runs synchronously in a worker thread. Rows must be ordered so each thread's
    deltas are oldest-to-newest. Undecodable rows are skipped (logged), matching
    the per-row error handling of the previous implementation.

    Returns:
        Mapping of thread ID to reconstructed message count.
    """
    deltas_by_thread: dict[str, list[Any]] = {}
    for tid, type_str, value_blob in rows:
        if not type_str or not value_blob:
            continue
        try:
            delta = serde.loads_typed((type_str, value_blob))
        except Exception:
            logger.warning(
                "Failed to replay messages write for thread %s; "
                "message count may be inaccurate",
                tid,
                exc_info=True,
            )
            continue
        deltas_by_thread.setdefault(tid, []).append(delta)

    counts: dict[str, int] = {}
    for tid, deltas in deltas_by_thread.items():
        try:
            counts[tid] = _count_messages_from_deltas(deltas)
        except Exception:
            # Keep one malformed thread from failing the whole `threads list`
            # load: skip it (its count is simply absent) rather than propagating.
            logger.warning(
                "Failed to count messages for thread %s; omitting its count",
                tid,
                exc_info=True,
            )
    return counts


def _visible_message_count(messages: list[object]) -> int:
    """Count messages that appear in user-facing thread history.

    Returns:
        Number of messages not classified as hidden application context.
    """
    return sum(not is_internal_message(message) for message in messages)


def _count_messages_from_deltas(deltas: list[Any]) -> int:
    """Count messages from an ordered list of `messages`-channel write deltas.

    Fast path: appends and full-clears (`REMOVE_ALL_MESSAGES`, `Overwrite`) fold
    into one `add_messages` pass — O(n) instead of the O(n^2) incremental fold,
    so threads with thousands of writes count in milliseconds. For these ops the
    single-pass result is count-equivalent to the sequential fold (both dedup by
    ID, and clears collapse to the post-clear tail).

    Slow path: a specific `RemoveMessage` (delete-by-ID) or any reducer error
    falls back to the exact sequential fold as a conservative measure. A
    delete-by-ID concatenated into the single `buffer` can make batch
    `add_messages` raise (the target ID may be absent at that buffer position),
    and we do not rely on unproven count-equivalence of batched removal. In
    practice the two folds still agree on the count for these histories; the
    sequential fold simply guarantees it. Such deletes are rare in linear dcode
    histories, so the common case stays on the fast path.

    Returns:
        Number of messages after reducing the deltas.
    """
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
    from langgraph.types import Overwrite

    buffer: list[Any] = []
    needs_exact_fold = False
    for delta in deltas:
        if isinstance(delta, Overwrite):
            value = delta.value
            buffer = list(value) if isinstance(value, list) else []
            continue
        items = delta if isinstance(delta, list) else [delta]
        for item in items:
            if isinstance(item, RemoveMessage):
                if item.id == REMOVE_ALL_MESSAGES:
                    buffer = []
                else:
                    needs_exact_fold = True
                    break
            else:
                buffer.append(item)
        if needs_exact_fold:
            break

    if not needs_exact_fold:
        try:
            reduced = cast("list[Any]", add_messages([], buffer))
            return _visible_message_count(cast("list[object]", reduced))
        except Exception:
            logger.debug(
                "Batched message-count fold failed; using sequential fold",
                exc_info=True,
            )

    return _incremental_message_count(deltas)


def _incremental_message_count(deltas: list[Any]) -> int:
    """Count messages by folding deltas sequentially through `add_messages`.

    Exact reference reduction: applies one delta at a time, resetting on
    `Overwrite` and skipping any delta the reducer rejects (e.g. a delete for an
    absent ID). Used as the fallback when the batched fast path cannot guarantee
    a matching count.

    Returns:
        Number of messages after the sequential fold.
    """
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    reduced: list[Any] = []
    for delta in deltas:
        if isinstance(delta, Overwrite):
            value = delta.value
            reduced = list(value) if isinstance(value, list) else []
            continue
        try:
            reduced = cast("list[Any]", add_messages(reduced, delta))
        except Exception:
            logger.warning(
                "Failed to replay messages write; message count may be inaccurate",
                exc_info=True,
            )
            continue
    return _visible_message_count(cast("list[object]", reduced))


def _summarize_checkpoint(data: object) -> _CheckpointSummary:
    """Extract message count and initial human prompt from checkpoint data.

    Returns:
        Structured summary for the decoded checkpoint payload.
    """
    messages = _checkpoint_messages(data)
    return _CheckpointSummary(
        message_count=(
            _visible_message_count(messages) if messages is not None else None
        ),
        initial_prompt=_initial_prompt_from_messages(messages or []),
    )


def _checkpoint_messages(data: object) -> list[object] | None:
    """Return inlined checkpoint messages, or `None` when not inlined.

    A `None` return distinguishes a checkpoint that omits the `messages`
    channel entirely (a `DeltaChannel` between snapshots, where the deepagents
    SDK applies LangGraph's `DeltaChannel` to `messages` as of v0.6) from one
    that inlines an empty list. The former requires reconstructing the count
    from the `writes` table; the latter is a genuine zero.
    """
    if not isinstance(data, dict):
        return None

    payload = cast("dict[str, object]", data)
    channel_values = payload.get("channel_values")
    if not isinstance(channel_values, dict):
        return None

    channel_values_dict = cast("dict[str, object]", channel_values)
    messages = channel_values_dict.get("messages")
    if not isinstance(messages, list):
        return None

    return cast("list[object]", messages)


def _initial_prompt_from_messages(messages: list[object]) -> str | None:
    """Return the first non-system human message content from a message list.

    Accepts both LangChain `HumanMessage` objects (with `type == "human"`) and
    plain dicts in OpenAI chat shape (`{"role": "user", "content": ...}`). The
    first write to the `messages` channel is the raw user input passed to the
    agent, which is preserved verbatim as a dict; subsequent writes are
    serialized `BaseMessage` instances produced after the model runs.

    Synthetic `[SYSTEM]`-prefixed human messages (e.g. an interrupt
    cancellation notice) are skipped so they never surface as a thread's prompt.
    """
    for msg in messages:
        if is_internal_message(msg):
            continue
        if getattr(msg, "type", None) == "human":
            prompt = _coerce_prompt_text(getattr(msg, "content", None))
        elif isinstance(msg, dict):
            msg_dict = cast("dict[str, object]", msg)
            role = msg_dict.get("role")
            type_ = msg_dict.get("type")
            if role not in {"user", "human"} and type_ != "human":
                continue
            prompt = _coerce_prompt_text(msg_dict.get("content"))
        else:
            continue
        return prompt
    return None


def _coerce_prompt_text(content: object) -> str | None:
    """Normalize checkpoint message content into displayable text.

    Returns:
        Displayable prompt text, or `None` when the content is empty.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_dict = cast("dict[str, object]", part)
                text = part_dict.get("text")
                parts.append(text if isinstance(text, str) else "")
            else:
                parts.append(str(part))
        joined = " ".join(parts).strip()
        return joined or None
    if content is None:
        return None
    return str(content)


# [해설] 가장 최근 체크포인트의 thread_id(에이전트 필터, 특정 스레드 제외 가능).
# [해설]   TUI(app.py)의 '가장 최근 스레드 재개'가 쓴다. 비대화 `-r`은 정확한 ID를
# [해설]   요구하므로 쓰지 않는다.
async def get_most_recent(
    agent_name: str | None = None,
    *,
    exclude_thread_id: str | None = None,
) -> str | None:
    """Get the most recent thread, optionally agent-filtered and/or excluding a thread.

    Args:
        agent_name: Return only threads created by this agent.
        exclude_thread_id: Ignore this thread when selecting the most recent one.

    Returns:
        Most recent thread ID, or `None` if no matching threads exist.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        if agent_name and exclude_thread_id:
            query = """
                SELECT thread_id FROM checkpoints
                WHERE json_extract(metadata, '$.agent_name') = ?
                  AND thread_id != ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """
            params: tuple[str, ...] = (agent_name, exclude_thread_id)
        elif agent_name:
            query = """
                SELECT thread_id FROM checkpoints
                WHERE json_extract(metadata, '$.agent_name') = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """
            params = (agent_name,)
        elif exclude_thread_id:
            query = """
                SELECT thread_id FROM checkpoints
                WHERE thread_id != ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """
            params = (exclude_thread_id,)
        else:
            query = (
                "SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
            )
            params = ()

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


# [해설] 최신 체크포인트가 새로운 순서의 고유 thread_id 목록(가벼운 조회). TUI(app.py)의
# [해설]   재개 후보 선택에서 쓴다.
async def get_recent_thread_ids(agent_name: str | None = None) -> list[str]:
    """List distinct threads in checkpoint recency order, optionally agent-filtered.

    Returns:
        Matching thread IDs, newest checkpoint first.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []
        query = (
            "SELECT thread_id FROM checkpoints "
            "WHERE (? IS NULL OR json_extract(metadata, '$.agent_name') = ?) "
            "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC"
        )
        async with conn.execute(query, (agent_name, agent_name)) as cursor:
            return [row[0] for row in await cursor.fetchall()]


async def get_thread_updated_at(thread_id: str) -> str | None:
    """Get the latest stored update timestamp for a thread.

    Returns:
        The ISO timestamp, or `None` when none is stored.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        query = """
            SELECT MAX(json_extract(metadata, '$.updated_at'))
            FROM checkpoints
            WHERE thread_id = ?
        """
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            value = row[0] if row else None
            return value if isinstance(value, str) and value else None


# [해설] 원격에서 모델 활동을 읽은 뒤 최신 루트 체크포인트의 `updated_at`만 앞당긴다.
# [해설]   과거 metadata와 더 새로운 서버 기록은 보존한다.
async def refresh_thread_activity(thread_id: str, updated_at: datetime) -> None:
    """Advance local discovery metadata after reading remote model activity.

    Only the latest root checkpoint is updated, preserving historical metadata
    and any newer timestamp already written by a shared server checkpointer.
    Preserve its original timestamp separately so a lone discovery seed keeps
    its creation time as remote activity advances.

    Args:
        thread_id: Thread whose remote state was read.
        updated_at: Verified model-request time from the remote checkpoint.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return
        timestamp = updated_at.isoformat()
        await conn.execute(
            "UPDATE checkpoints SET metadata = json_set(metadata, "
            "'$.created_at', COALESCE(json_extract(metadata, '$.created_at'), "
            "json_extract(metadata, '$.updated_at')), '$.updated_at', ?) "
            "WHERE thread_id = ? AND checkpoint_ns = '' AND checkpoint_id = "
            "(SELECT MAX(checkpoint_id) FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = '') "
            "AND COALESCE(julianday(json_extract(metadata, '$.updated_at')), 0) "
            "< julianday(?)",
            (timestamp, thread_id, thread_id, timestamp),
        )
        await conn.commit()


# [해설] 스레드가 어느 에이전트의 것인지 metadata에서 읽는다. 비대화 재개에서
# [해설]   `ResumeTarget.agent_name`이 되고 `adopt_agent`일 때 실행 에이전트를 바꾼다.
async def get_thread_agent(thread_id: str) -> str | None:
    """Get agent_name for a thread.

    Returns:
        Agent name associated with the thread, or None if not found.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        query = """
            SELECT json_extract(metadata, '$.agent_name')
            FROM checkpoints
            WHERE thread_id = ?
            LIMIT 1
        """
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


# [해설] 스레드가 마지막으로 쓰인 작업 디렉터리. 재개 시 현재 cwd와 비교하는
# [해설]   근거(client/headless_resume.py의 `cwd_conflict`).
async def get_thread_cwd(thread_id: str) -> str | None:
    """Get the most recently stored cwd for a thread.

    Args:
        thread_id: The thread whose stored cwd to look up.

    Returns:
        Most recent cwd for the thread, or None if not found.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        query = """
            SELECT json_extract(metadata, '$.cwd')
            FROM checkpoints
            WHERE thread_id = ? AND json_extract(metadata, '$.cwd') IS NOT NULL
            ORDER BY checkpoint_id DESC
            LIMIT 1
        """
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            value = row[0] if row else None
            return value if isinstance(value, str) and value else None


# [해설] 해당 thread_id의 체크포인트가 하나라도 있는지. 재개 직전
# [해설]   `_reserve_thread`(non_interactive.py)가 삭제 경합을 확인할 때도 쓴다.
async def thread_exists(thread_id: str) -> bool:
    """Check if a thread exists in checkpoints.

    Returns:
        True if thread exists, False otherwise.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False

        query = "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1"
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row is not None


# [해설] 입력한 ID가 없을 때 접두사가 같은 후보를 제안한다. 비대화 재개의
# [해설]   `resolve_resume_target`(client/headless_resume.py)가 'Did you mean' 안내에
# [해설]   쓴다.
async def find_similar_threads(thread_id: str, limit: int = 3) -> list[str]:
    """Find threads whose IDs start with the given prefix.

    Args:
        thread_id: Prefix to match against thread IDs.
        limit: Maximum number of matching threads to return.

    Returns:
        List of thread IDs that begin with the given prefix.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []

        query = """
            SELECT DISTINCT thread_id
            FROM checkpoints
            WHERE thread_id LIKE ?
            ORDER BY thread_id
            LIMIT ?
        """
        prefix = thread_id + "%"
        async with conn.execute(query, (prefix, limit)) as cursor:
            rows = await cursor.fetchall()
            return [r[0] for r in rows]


# [해설] 이름 규칙: 비어 있지 않음, 최대 `MAX_THREAD_NAME_LENGTH`(50)자, 제어 문자 없는
# [해설]   한 줄.
def validate_thread_name(name: str) -> str:
    """Validate a thread name.

    Args:
        name: Proposed name.

    Returns:
        Trimmed name.

    Raises:
        ValueError: If empty, overlong, or containing nonprintable characters.
    """
    if any(not char.isprintable() for char in name):
        msg = "Thread names must be a single line without control characters"
        raise ValueError(msg)
    name = name.strip()
    if not name or len(name) > MAX_THREAD_NAME_LENGTH:
        msg = "Thread names must contain between 1 and 50 characters"
        raise ValueError(msg)
    return name


# [해설] 이름은 체크포인트 개정(revision)과 무관하게 `dcode_thread_names`에 따로
# [해설]   저장되어 체크포인트가 새로 쌓여도 유지된다.
async def get_thread_name(thread_id: str) -> str | None:
    """Read a thread's durable name independently of checkpoint revisions.

    Args:
        thread_id: Thread to look up.

    Returns:
        Saved name, or `None` for unnamed threads.
    """
    async with _connect() as conn:
        if await _table_exists(conn, "dcode_thread_names"):
            async with conn.execute(
                "SELECT name FROM dcode_thread_names WHERE thread_id = ?", (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return row[0]
        if not await _table_exists(conn, "checkpoints"):
            return None
        return (await _checkpoint_thread_names(conn, [thread_id])).get(thread_id)


# [해설] 한 트랜잭션(`BEGIN IMMEDIATE`)에서 이름 테이블과 최신 체크포인트 metadata를
# [해설]   함께 갱신해 대화는 건드리지 않고 이름만 바꾼다. 메모리 캐시도 즉시 맞춘다.
# [해설]   `only_if_unnamed`면 이미 있는 이름(자동 생성 포함)을 덮어쓰지 않는다.
async def rename_thread(
    thread_id: str, name: str, *, only_if_unnamed: bool = False
) -> bool:
    """Atomically name an existing thread without mutating its conversation.

    Args:
        thread_id: Thread to name.
        name: Nonempty, printable, single-line name of at most 50 characters.
        only_if_unnamed: Preserve a previously generated or manually assigned name.

    Returns:
        Whether an existing thread was renamed.
    """
    name = validate_thread_name(name)
    async with _connect() as conn:
        await conn.execute("BEGIN IMMEDIATE")
        if not await _table_exists(conn, "checkpoints"):
            return False
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS dcode_thread_names "
            "(thread_id TEXT PRIMARY KEY NOT NULL, name TEXT NOT NULL)"
        )
        async with conn.execute(
            "INSERT INTO dcode_thread_names (thread_id, name) "
            "SELECT ?, ? WHERE EXISTS "
            "(SELECT 1 FROM checkpoints WHERE thread_id = ?) "
            "AND (? = 0 OR NOT EXISTS (SELECT 1 FROM checkpoints "
            "WHERE thread_id = ? "
            "AND json_extract(metadata, '$.thread_name') IS NOT NULL)) "
            "ON CONFLICT(thread_id) DO UPDATE SET name = excluded.name WHERE ? = 0",
            (thread_id, name, thread_id, only_if_unnamed, thread_id, only_if_unnamed),
        ) as cursor:
            if not cursor.rowcount:
                return False
        await conn.execute(
            "UPDATE checkpoints SET metadata = json_set(metadata, '$.thread_name', ?) "
            "WHERE thread_id = ? AND checkpoint_id = "
            "(SELECT MAX(checkpoint_id) FROM checkpoints WHERE thread_id = ?)",
            (name, thread_id, thread_id),
        )
        await conn.commit()
    for rows in _recent_threads_cache.values():
        for thread in rows:
            if thread["thread_id"] == thread_id:
                thread["thread_name"] = name
    return True


# [해설][흐름] 1) `try_acquire`로 스레드 점유 시도 - 다른 클라이언트가 열고 있으면
# [해설]   `BlockingIOError`  2) 쓰기 가드 안에서 비용·이름·메시지 수·체크포인트·writes
# [해설]   행 삭제
# [해설]   3) 메모리 캐시 정리  4) 오프로드된 대화 기록 파일
# [해설]   삭제(`delete_offloaded_history`, 실패해도 반환값에 영향 없음)  5) 점유 해제.
# [해설][주의] 반환값은 '체크포인트 행이 지워졌는가'만 뜻한다. 삭제에 점유가 필요하므로
# [해설]   재개 경로의 `thread_exists` 확인이 낡지 않는다.
async def delete_thread(thread_id: str) -> bool:
    """Delete thread checkpoints, side-question costs, and offloaded history.

    Removes the thread's checkpoint/write rows, then makes a best-effort attempt
    to remove the per-thread compaction archive and handoff snapshots under
    `~/.deepagents` (local mode) so deletion does not leave orphaned history
    behind. History cleanup failures are logged, not raised, and do not affect
    the return value, which reflects only whether checkpoint rows were removed.

    Returns:
        True if thread checkpoints were deleted, False if not found.

    Raises:
        BlockingIOError: The thread is reserved by a live client.
    """
    from deepagents_code.btw_cost import delete_cost
    from deepagents_code.thread_ownership import (
        _finish_mutation,
        try_acquire,
        writer_guard,
    )

    db_path = get_db_path()
    lease = try_acquire(thread_id, db_path=db_path)
    if lease is None:
        msg = f"Thread {thread_id} is open elsewhere. Close it there before deleting."
        raise BlockingIOError(msg)

    async def remove() -> bool:
        deleted = False
        async with _connect() as conn:
            await delete_cost(conn, thread_id)
            for table in (
                "dcode_btw_prompts",
                "dcode_thread_names",
                _MESSAGE_COUNTS_TABLE,
            ):
                if await _table_exists(conn, table):
                    await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?",  # noqa: S608  # constant table names
                        (thread_id,),
                    )
            if await _table_exists(conn, "checkpoints"):
                cursor = await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
                )
                deleted = cursor.rowcount > 0
                if await _table_exists(conn, "writes"):
                    await conn.execute(
                        "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
                    )
                if deleted:
                    _message_count_cache.pop(thread_id, None)
                    for key, rows in list(_recent_threads_cache.items()):
                        filtered = [
                            row for row in rows if row["thread_id"] != thread_id
                        ]
                        _recent_threads_cache[key] = filtered
            await conn.commit()

        from deepagents_code.offload import delete_offloaded_history

        delete_offloaded_history(thread_id)
        return deleted

    try:
        async with writer_guard(thread_id, db_path=db_path, token=lease.token):
            return await _finish_mutation(remove())
    finally:
        lease.release()


# [해설] sessions.db를 쓰는 `AsyncSqliteSaver` 컨텍스트.
# [해설]   `owned_saver_class`(thread_ownership.py)가 만든 서브클래스라 점유 토큰이
# [해설]   없으면 쓰기를 거부한다.
# [해설]   `from_conn_string` 대신 `_new_connection`을 직접 써서 취소된 연결도
# [해설]   `_guard_sqlite_handle`로 정리할 수 있게 했다.
@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Get a checkpointer requiring live thread ownership for database writes.

    Yields:
        AsyncSqliteSaver enforcing thread reservations and fencing tokens.
    """
    from deepagents_code.thread_ownership import owned_saver_class

    saver_class = owned_saver_class(db_path=get_db_path())

    # Built here rather than through `AsyncSqliteSaver.from_conn_string` so the
    # connection is one this module owns and can clean up after an interrupted
    # connect; see `_guard_sqlite_handle`.
    conn = _new_connection()
    try:
        async with conn as opened:
            yield saver_class(opened)
    finally:
        await _drain_aiosqlite_worker(conn)


# [해설] 원격 핸드오프로 새로 만든 스레드를 로컬 DB에 '발견용 씨앗' 체크포인트로
# [해설]   등록한다. 이미 있으면 아무것도 쓰지 않는다(서버 체크포인트를 덮어쓰지 않음).
# [해설]   이 프로세스가 점유 중이 아니면 저장 동안만 점유했다가 해제한다.
async def save_thread_seed(
    thread_id: str, values: Mapping[str, object], *, agent_name: str, cwd: str
) -> None:
    """Index a remote handoff locally with its initial summary and resume metadata.

    Only seed an absent thread; never replace shared server checkpoints. Thread
    history still loads from the connected agent, so subsequent remote turns
    take precedence over this discovery snapshot. Reserve the local thread for
    the save when this process does not already own it.

    Args:
        thread_id: Newly seeded remote thread identifier.
        values: Initial message and model state saved on the server.
        agent_name: Agent that owns the new thread.
        cwd: Workspace directory used for thread discovery.

    """
    from deepagents_code.thread_ownership import OWNER_KEY, ensure_owned, held_lease

    existing = held_lease(thread_id)
    lease = existing or ensure_owned(thread_id)
    try:
        await _save_thread_seed(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    OWNER_KEY: lease.token,
                },
                "metadata": {"agent_name": agent_name, "cwd": cwd},
            },
            values,
        )
    finally:
        if existing is None:
            lease.release()


# [해설] 점유 상태에서 `empty_checkpoint`에 초기 값을 담아 `aput`한다(step 0,
# [해설]   source=update). 이미 체크포인트가 있으면 반환.
async def _save_thread_seed(
    config: RunnableConfig, values: Mapping[str, object]
) -> None:
    """Write the initial checkpoint while the caller holds its reservation."""
    from langgraph.checkpoint.base import empty_checkpoint

    checkpoint = empty_checkpoint()
    config.setdefault("metadata", {})["updated_at"] = checkpoint["ts"]
    async with get_checkpointer() as checkpointer:
        if await checkpointer.aget_tuple(config) is not None:
            return
        checkpoint["channel_values"] = dict(values)
        version = checkpointer.get_next_version(None, None)
        checkpoint["channel_versions"] = dict.fromkeys(values, version)
        await checkpointer.aput(
            config,
            checkpoint,
            {"source": "update", "step": 0, "parents": {}},
            checkpoint["channel_versions"],
        )


# [해설] 원격 state-update API가 config metadata를 전달하지 않으므로, 저장된
# [해설]   체크포인트에 agent_name·cwd·updated_at을 직접 UPDATE한다. 체크포인트가 없으면
# [해설]   RuntimeError.
async def set_thread_metadata(thread_id: str, *, agent_name: str, cwd: str) -> None:
    """Persist discovery and resume metadata on a newly checkpointed thread.

    The remote state-update API does not forward config metadata. Stamp the
    saved checkpoints before exposing a newly seeded thread to the user.

    Args:
        thread_id: Newly seeded thread identifier.
        agent_name: Agent that owns the new thread.
        cwd: Workspace directory used for thread discovery.

    Raises:
        RuntimeError: If the thread has no saved checkpoint.
    """
    from datetime import UTC

    async with _connect() as conn:
        async with conn.execute(
            "UPDATE checkpoints SET metadata = json_set(metadata, "
            "'$.agent_name', ?, '$.cwd', ?, '$.updated_at', ?) WHERE thread_id = ?",
            (agent_name, cwd, datetime.now(UTC).isoformat(), thread_id),
        ) as cursor:
            if not cursor.rowcount:
                msg = f"Thread {thread_id} has no saved checkpoint"
                raise RuntimeError(msg)
        await conn.commit()


# [해설] 목록 개수 상한: 환경변수 `RECENT_THREADS`(deepagents_code/_env_vars.py), 잘못된
# [해설]   값이면 경고 후 `DEFAULT_THREAD_LIMIT`. `threads list`·`@@` 완성·시작
# [해설]   prewarm이 같은 값을 쓴다.
def get_thread_limit() -> int:
    """Read the thread listing limit from the environment.

    Returns:
        Thread switcher batch size. Also the default row limit for
            `threads list`, `@@` thread completion, and the startup prewarm.
    """
    import os

    from deepagents_code._env_vars import RECENT_THREADS

    raw = os.environ.get(RECENT_THREADS)
    if raw is None:
        return DEFAULT_THREAD_LIMIT
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid %s value %r, using default %d",
            RECENT_THREADS,
            raw,
            DEFAULT_THREAD_LIMIT,
        )
        return DEFAULT_THREAD_LIMIT


# [해설] `dcode threads list` CLI 핸들러. 옵션을 해석해 `list_threads`를 부르고 표(text)
# [해설]   또는 JSON으로 출력한다.
async def list_threads_command(
    agent_name: str | None = None,
    limit: int | None = None,
    sort_by: str | None = None,
    branch: str | None = None,
    cwd: str | None = None,
    verbose: bool = False,
    relative: bool | None = None,
    *,
    output_format: OutputFormat = "text",
) -> None:
    """CLI handler for `deepagents threads list`.

    Fetches and displays a table of recent conversation threads, optionally
    filtered by agent name, git branch, or working directory.

    Args:
        agent_name: Only show threads belonging to this agent.

            When `None`, threads for all agents are shown.
        limit: Maximum number of threads to display.

            When `None`, reads from `DEEPAGENTS_CODE_RECENT_THREADS` or falls
            back to the default.
        sort_by: Sort field — `"updated"` or `"created"`.

            When `None`, reads the merged managed and user config
            (`managed_config.toml` over `~/.deepagents/config.toml`).
        branch: Only show threads from this git branch.
        cwd: Only show threads whose stored `cwd` metadata equals this path
            (exact string match — no normalization or prefix matching). When
            `None`, no cwd filter is applied. Threads without a stored `cwd`
            (older rows) are excluded when this is set.
        verbose: When `True`, show all columns (branch, created, prompt).
        relative: Show timestamps as relative time (e.g., '5m ago').

            When `None`, reads the merged managed and user config
            (`managed_config.toml` over `~/.deepagents/config.toml`).
        output_format: Output format — `'text'` (Rich) or `'json'`.
    """
    from deepagents_code.model_config import (
        load_thread_relative_time,
        load_thread_sort_order,
    )

    if sort_by is None:
        raw = load_thread_sort_order()
        sort_by = "created" if raw == "created_at" else "updated"
    if relative is None:
        relative = load_thread_relative_time()

    fmt_ts = format_relative_timestamp if relative else format_timestamp

    limit = get_thread_limit() if limit is None else max(1, limit)

    threads = await list_threads(
        agent_name,
        limit=limit,
        include_message_count=True,
        sort_by=sort_by,
        branch=branch,
        cwd=cwd,
    )

    if verbose and threads:
        await populate_thread_checkpoint_details(
            threads, include_message_count=False, include_initial_prompt=True
        )

    if output_format == "json":
        from deepagents_code.output import write_json

        write_json("threads list", list(threads))
        return

    from rich.markup import escape as escape_markup
    from rich.table import Table

    from deepagents_code import theme
    from deepagents_code.config import console

    if not threads:
        filters = []
        if agent_name:
            filters.append(f"agent '{escape_markup(agent_name)}'")
        if branch:
            filters.append(f"branch '{escape_markup(branch)}'")
        if cwd:
            filters.append(f"cwd '{escape_markup(cwd)}'")
        if filters:
            console.print(
                f"[yellow]No threads found for {' and '.join(filters)}.[/yellow]"
            )
        else:
            console.print("[yellow]No threads found.[/yellow]")
        if cwd:
            # Older threads predating cwd-metadata storage are silently
            # filtered out by the `json_extract` equality match. Tell the
            # user explicitly when unfiltered rows exist so they don't think
            # their history is gone.
            unfiltered = await list_threads(
                agent_name,
                limit=limit,
                include_message_count=False,
                sort_by=sort_by,
                branch=branch,
            )
            legacy_count = sum(1 for t in unfiltered if not t.get("cwd"))
            if legacy_count:
                console.print(
                    f"[dim]{legacy_count} older thread"
                    f"{'s have' if legacy_count != 1 else ' has'} "
                    "no cwd metadata; run without --cwd to see "
                    f"{'them' if legacy_count != 1 else 'it'}.[/dim]"
                )
        console.print("[dim]Start a conversation with: deepagents[/dim]")
        return

    title_parts = []
    if agent_name:
        title_parts.append(f"agent '{escape_markup(agent_name)}'")
    if branch:
        title_parts.append(f"branch '{escape_markup(branch)}'")
    if cwd:
        title_parts.append(f"cwd '{escape_markup(cwd)}'")

    title_filter = f" for {' and '.join(title_parts)}" if title_parts else ""
    sort_label = "created" if sort_by == "created" else "updated"
    title = f"Recent Threads{title_filter} (last {limit}, by {sort_label})"

    table = Table(title=title, show_header=True, header_style=f"bold {theme.PRIMARY}")
    table.add_column("Thread ID", style="bold")
    table.add_column("Name", max_width=50, no_wrap=True)
    table.add_column("Agent")
    table.add_column("Messages", justify="right")
    if verbose:
        table.add_column("Created")
    table.add_column("Updated" if sort_by == "updated" else "Last Used")
    if verbose:
        table.add_column("Branch")
        table.add_column("Location")
        table.add_column("Prompt", max_width=40, no_wrap=True)

    prompt_max = 40

    for t in threads:
        row: list[str] = [
            t["thread_id"],
            escape_markup(t.get("thread_name") or ""),
            t["agent_name"] or "unknown",
            str(t.get("message_count", 0)),
        ]
        if verbose:
            row.append(fmt_ts(t.get("created_at")))
        row.append(fmt_ts(t.get("updated_at")))
        if verbose:
            prompt = " ".join((t.get("initial_prompt") or "").split())
            if len(prompt) > prompt_max:
                prompt = prompt[: prompt_max - 3] + "..."
            row.extend(
                [
                    t.get("git_branch") or "",
                    format_path(t.get("cwd")),
                    prompt,
                ]
            )
        table.add_row(*row)

    console.print()
    console.print(table)
    if len(threads) >= limit:
        from deepagents_code._env_vars import RECENT_THREADS

        console.print(
            f"[dim]Showing last {limit} threads. "
            f"Override with -n/--limit or {RECENT_THREADS}.[/dim]"
        )
    console.print()


# [해설] `dcode threads delete` CLI 핸들러. `--dry-run`은 존재 여부만 보고하고 변경하지
# [해설]   않는다. 점유 중이면 종료 코드 1.
async def delete_thread_command(
    thread_id: str,
    *,
    dry_run: bool = False,
    output_format: OutputFormat = "text",
) -> None:
    """CLI handler for: deepagents threads delete.

    Args:
        thread_id: ID of the thread to delete.
        dry_run: If `True`, print what would happen without making changes.
        output_format: Output format — `'text'` (Rich) or `'json'`.

    Raises:
        SystemExit: The thread is occupied and cannot be deleted.
    """
    if dry_run:
        exists = await thread_exists(thread_id)
        if output_format == "json":
            from deepagents_code.output import write_json

            write_json(
                "threads delete",
                {"thread_id": thread_id, "exists": exists, "dry_run": True},
            )
            return

        from rich.markup import escape as escape_markup

        from deepagents_code.config import console

        escaped_id = escape_markup(thread_id)
        if exists:
            console.print(f"Would delete thread '{escaped_id}'.")
        else:
            console.print(f"Thread '{escaped_id}' not found. Nothing to delete.")
        console.print("No changes made.", style="dim")
        return

    try:
        deleted = await delete_thread(thread_id)
    except BlockingIOError as exc:
        if output_format == "json":
            from deepagents_code.output import write_json

            write_json(
                "threads delete",
                {"thread_id": thread_id, "deleted": False, "error": str(exc)},
            )
        else:
            from deepagents_code.config import console

            console.print(str(exc), style="red", markup=False, highlight=False)
        raise SystemExit(1) from None

    if output_format == "json":
        from deepagents_code.output import write_json

        write_json("threads delete", {"thread_id": thread_id, "deleted": deleted})
        return

    from rich.markup import escape as escape_markup

    from deepagents_code import theme
    from deepagents_code.config import console

    escaped_id = escape_markup(thread_id)
    if deleted:
        console.print(f"[green]Thread '{escaped_id}' deleted.[/green]")
    else:
        console.print(
            f"Thread '{escaped_id}' not found or already deleted.",
            style=theme.MUTED,
        )
