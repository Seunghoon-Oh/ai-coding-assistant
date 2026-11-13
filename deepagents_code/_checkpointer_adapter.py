"""Forward delta-channel history reads through langgraph-api's saver adapter.

The LangGraph dev server wraps a custom checkpointer in
`langgraph_api._checkpointer._adapter._CustomCheckpointerAdapter`. The adapter
forwards unknown attributes to the inner saver through `__getattr__`, but
`BaseCheckpointSaver` already defines `aget_delta_channel_history`, so the
forwarding never fires. The adapter then runs the base ancestor walk: one
`aget_tuple` and one full checkpoint decode per ancestor. The SQLite saver
overrides the method with two indexed queries, and long threads pay the
difference on every state read and every run start.

This module is a best-effort patch over a private langgraph-api class. It only
patches when the adapter still inherits the base method, and it falls back to
the base walk when the inner saver has no override of its own. Remove it when
the adapter forwards these methods itself.
"""

# [해설] 역할: 서버 프로세스 안에서 "체크포인트(대화 상태) 읽기"를 빠르게 만드는 작은
# [해설]   패치 모듈.
# [해설] 실행 위치: LangGraph 서버 프로세스(langgraph dev). 클라이언트(TUI)
# [해설]   프로세스에서는 쓰이지 않는다.
# [해설] 진입 심볼: install_delta_history_forwarding(). 서버 기동 때 생성된
# [해설]   checkpointer.py의
# [해설]   create_checkpointer()가 호출한다(그 파일은 client/launch/server_manager.py의
# [해설]   _write_checkpointer가 쓴다).
# [해설] 체크포인트 저장소: 프로필 state 디렉터리의 sessions.db(SQLite). 경로는 환경변수
# [해설]   DEEPAGENTS_CODE_SERVER_DB_PATH로 전달되고, 저장기 클래스는
# [해설]   thread_ownership.py의 owned_saver_class다.
# [해설] 큰 그림: langgraph-api가 사용자 저장기를 _CustomCheckpointerAdapter로 감싸는데,
# [해설]   이 어댑터가
# [해설]   저장기의 빠른 경로(SQLite 쿼리 2번)를 가려 버린다. 이 모듈이 그 경로를 다시
# [해설]   이어 준다.
# [해설][주의] langgraph-api의 비공개(private) 클래스를 건드리는 monkey patch다.
# [해설]   업스트림이 고치면 지워도 된다.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import DeltaChannelHistory

logger = logging.getLogger(__name__)

# [해설] 기본 구현(BaseCheckpointSaver)의 원본 메서드를 미리 잡아 둔다. 이후 "어댑터가
# [해설]   아직 기본 구현을
# [해설] 그대로 쓰는가"를 `is` 비교로 판별하고, 내부 저장기가 재정의하지 않았을 때의
# [해설]   되돌림 경로로도 쓴다.
_BASE_ASYNC = BaseCheckpointSaver.aget_delta_channel_history
_BASE_SYNC = BaseCheckpointSaver.get_delta_channel_history


# [해설] 어댑터 객체의 `_inner`(실제 SQLite 저장기)가 해당 메서드를 직접 재정의했을 때만
# [해설]   그 저장기를 돌려준다.
# [해설] 재정의하지 않았으면 None이고, 호출 쪽이 기본 구현(조상 체크포인트를 하나씩
# [해설]   따라가는 방식)으로 되돌아간다.
def _overriding_inner(
    adapter: BaseCheckpointSaver, name: str
) -> BaseCheckpointSaver | None:
    """Return the adapter's inner saver if it overrides the base method `name`."""
    inner = adapter.__dict__.get("_inner")
    if not isinstance(inner, BaseCheckpointSaver):
        return None
    if getattr(type(inner), name) is getattr(BaseCheckpointSaver, name):
        return None
    return inner


# [해설] 어댑터에 끼워 넣을 비동기 버전. 내부 저장기에 빠른 구현이 있으면 위임하고,
# [해설]   없으면 기본 구현을 호출한다.
async def _aget_delta_channel_history(
    self: BaseCheckpointSaver, *, config: RunnableConfig, channels: Sequence[str]
) -> Mapping[str, DeltaChannelHistory]:
    inner = _overriding_inner(self, "aget_delta_channel_history")
    if inner is None:
        return await _BASE_ASYNC(self, config=config, channels=channels)
    return await inner.aget_delta_channel_history(config=config, channels=channels)


# [해설] 위 함수의 동기 버전. 동작은 같고 `get_delta_channel_history`를 대상으로 한다.
def _get_delta_channel_history(
    self: BaseCheckpointSaver, *, config: RunnableConfig, channels: Sequence[str]
) -> Mapping[str, DeltaChannelHistory]:
    inner = _overriding_inner(self, "get_delta_channel_history")
    if inner is None:
        return _BASE_SYNC(self, config=config, channels=channels)
    return inner.get_delta_channel_history(config=config, channels=channels)


# [해설][흐름] 1) 비공개 어댑터 클래스를 import(없으면 경고만 남기고 건너뜀) 2) 비동기
# [해설]   메서드가 아직 기본 구현이면 교체
# [해설][흐름] 3) 동기 메서드도 같은 조건으로 교체. 이미 교체됐거나 업스트림이 고쳤다면
# [해설]   아무것도 하지 않아 멱등이다.
def install_delta_history_forwarding() -> None:
    """Make langgraph-api's checkpointer adapter use the inner saver's fast path.

    Call this in the server process before the adapter serves requests. It is
    idempotent and a no-op when the adapter is missing or already overrides
    the methods.
    """
    try:
        # [해설][주의] 이 import가 실패해도 서버는 계속 뜬다. 다만 긴 스레드에서 상태
        # [해설]   읽기가 느려질 수 있다(성능만 영향).
        # The adapter has no public seam; this patch exists to reach it.
        from langgraph_api._checkpointer._adapter import (  # noqa: PLC2701
            _CustomCheckpointerAdapter,
        )
    except ImportError:
        logger.warning("langgraph-api checkpointer adapter not found; skipping patch")
        return
    adapter = _CustomCheckpointerAdapter
    if adapter.aget_delta_channel_history is _BASE_ASYNC:
        adapter.aget_delta_channel_history = _aget_delta_channel_history
    if adapter.get_delta_channel_history is _BASE_SYNC:
        adapter.get_delta_channel_history = _get_delta_channel_history
