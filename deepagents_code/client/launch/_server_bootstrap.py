"""Run the POSIX server while watching the launching client's lifetime.

The inherited pipe is never written to: EOF means every client-side writer has
closed. Keep this module stdlib-only so the guard starts before server imports.
"""

# [해설] 역할: `langgraph dev`를 직접 띄우기 전에 거치는 "부모 감시 부트스트랩". 서버
# [해설]   자식 프로세스의 첫 코드다.
# [해설] 실행 위치: 서버 프로세스. client/launch/server.py의
# [해설]   ServerProcess._launch_process가
# [해설]   `python -m deepagents_code.client.launch._server_bootstrap <fd목록> <유예초>
# [해설]   dev ...`로 실행한다(POSIX만).
# [해설] 진입 심볼: _main(). 감시 스레드는 _ParentGuard가 맡는다.
# [해설] 동작 원리: 클라이언트가 pipe의 쓰기 쪽을 쥐고 읽기 쪽 fd만 이 프로세스에
# [해설]   넘긴다. 아무도 쓰지 않으므로
# [해설]   read()가 EOF를 돌려주는 순간은 "클라이언트의 모든 쓰기 핸들이
# [해설]   닫혔다(=클라이언트가 죽었다)"는 뜻이다.
# [해설]   SIGKILL로 죽어도 OS가 fd를 닫아 주므로 감지된다.
# [해설] 표준 라이브러리만 import하는 이유: 무거운 서버 import 전에 감시를 먼저
# [해설]   시작하려는 설계(모듈 docstring).
# [해설] 관련 모듈: client/launch/server.py(생성·종료 쪽), _session_lock.py(세션 잠금
# [해설]   fd를 넘겨받음).
from __future__ import annotations

import os
import runpy
import selectors
import signal
import sys
import threading
import time
from pathlib import Path


# [해설][설계] pipe 읽기 fd를 소유하고 "클라이언트 소멸 -> 서버 종료"를 책임지는 감시자.
# [해설] 상태 변수 두 개(_parent_exited, _dispatched)를 락으로 묶어, "감시가 먼저 끝난
# [해설]   경우"와
# [해설] "langgraph_cli 실행을 이미 시작한 경우"를 경쟁 없이 구분한다.
class _ParentGuard:
    """Own the pipe reader and stop the server when its client disappears."""

    def __init__(self, reader: int, timeout: float) -> None:
        # Popen passes this descriptor through one exec, never through the next.
        os.set_inheritable(reader, False)
        self._reader = reader
        self._timeout = timeout
        self._lock = threading.Lock()
        self._parent_exited = False
        self._dispatched = False

    # [해설] 데몬 스레드로 _watch를 시작한다. 서버 import나 CLI 실행 전에 호출해야 그
    # [해설]   사이의 클라이언트 종료도 놓치지 않는다.
    def start(self) -> None:
        """Start watching before importing or dispatching the CLI."""
        threading.Thread(
            target=self._watch, name="server-parent-guard", daemon=True
        ).start()

    # [해설] CLI 실행 직전의 관문. 이미 EOF가 왔거나(selector로 즉시 확인) 감시 스레드가
    # [해설]   종료를 기록했다면
    # [해설] False를 돌려 서버를 아예 시작하지 않는다. True일 때만 _dispatched=True로
    # [해설]   "이제부터는 종료 신호를 보내야 함"을 표시한다.
    def begin_dispatch(self) -> bool:
        """Claim dispatch unless EOF has already arrived.

        Returns:
            Whether the client is still alive at the dispatch boundary.
        """
        with self._lock:
            if self._parent_exited:
                return False
            # Check synchronously too: the watcher may not have run yet. Only
            # the watcher closes the reader, under this same lock, so its number
            # cannot be reused while this check is in progress.
            with selectors.DefaultSelector() as selector:
                selector.register(self._reader, selectors.EVENT_READ)
                if selector.select(timeout=0):
                    return False
            self._dispatched = True
            return True

    # [해설] 감시 스레드 본체. read()가 EOF(빈 바이트)를 돌려줄 때까지 블로킹하고,
    # [해설]   끝나면 fd를 닫는다.
    # [해설] fd 번호가 재사용되지 않도록 닫기와 상태 갱신은 같은 락 안에서 한다. 이미
    # [해설]   CLI가 시작됐으면 _terminate로 넘어간다.
    def _watch(self) -> None:
        try:
            while os.read(self._reader, 1):
                pass
        finally:
            with self._lock:
                self._parent_exited = True
                os.close(self._reader)
                dispatched = self._dispatched
            if dispatched:
                self._terminate()

    # [해설][흐름] 1) 자기 자신(pid)에게 SIGTERM -> uvicorn/LangGraph의 정상 종료 경로
    # [해설]   2) 유예 시간(인자로 받은
    # [해설][흐름] _SHUTDOWN_TIMEOUT 값, server.py) 동안 대기 3) 그래도 살아 있으면
    # [해설]   SIGKILL.
    # [해설][주의] run_module이 반환돼도 이 스레드를 취소하지 않는다. 인터프리터 종료가
    # [해설]   스레드/atexit에서 막혀도 강제 종료가 보장된다.
    def _terminate(self) -> None:
        """Allow ordinary signal shutdown, then enforce the same grace period."""
        pid = os.getpid()
        os.kill(pid, signal.SIGTERM)
        # This runs independently of the main thread's server loop and remains
        # active if the CLI returns but interpreter teardown blocks on a thread
        # or an atexit callback. Do not cancel it when run_module returns.
        time.sleep(self._timeout)
        os.kill(pid, signal.SIGKILL)


# [해설] 부트스트랩 본체. argv 규약: argv[1]="읽기fd,보유fd..." / argv[2]=유예 초 /
# [해설]   argv[3:]=`langgraph_cli`에 넘길 인자(dev --host ... --config ...).
# [해설][흐름] 1) 첫 fd는 pipe 읽기, 나머지는 세션 잠금 fd. 잠금 fd는 이 프로세스에만
# [해설]   남기고 상속 불가로 바꾼다
# [해설][흐름]    (서버가 만든 손자 프로세스가 잠금을 붙들고 설치 업데이트를 막는 일을
# [해설]   방지).
# [해설][흐름] 2) _ParentGuard 시작 -> begin_dispatch가 False면 클라이언트가 이미
# [해설]   사라졌으므로 조용히 반환.
# [해설][흐름] 3) 쓰기 가능하면 sys.prefix 아래 __pycache__에 바이트코드를 쓰도록
# [해설]   되돌린다(부모가 PYTHONDONTWRITEBYTECODE=1을
# [해설]    넣어 두므로 프로젝트 폴더에는 .pyc가 생기지 않고, 서버 import 캐시는 설치
# [해설]   안에만 남는다. test_server_interpreter_caches_bytecode 참고).
# [해설][흐름] 4) sys.argv를 `langgraph_cli dev ...`로 바꾸고 runpy.run_module로 `python
# [해설]   -m langgraph_cli`와 동일하게 실행.
def _main() -> None:
    reader, *held = (int(fd) for fd in sys.argv[1].split(","))
    for fd in held:
        # Keep the session lock in this process only, so a long-lived
        # grandchild cannot hold the install in use after the server exits.
        os.set_inheritable(fd, False)
    guard = _ParentGuard(reader, float(sys.argv[2]))
    guard.start()
    if not guard.begin_dispatch():
        return
    cache_dir = Path(sys.prefix) / "__pycache__"
    if os.access(cache_dir if cache_dir.exists() else sys.prefix, os.W_OK):
        sys.pycache_prefix = str(cache_dir)
        sys.dont_write_bytecode = False
    sys.argv = ["langgraph_cli", *sys.argv[3:]]
    runpy.run_module("langgraph_cli", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    _main()
