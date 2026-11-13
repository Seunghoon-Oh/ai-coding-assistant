"""LangGraph server lifecycle management for the app.

Handles starting/stopping a `langgraph dev` server process and generating the
required `langgraph.json` configuration file.
"""

# [해설] 역할: 클라이언트 프로세스가 LangGraph 서버 프로세스(`langgraph dev`)를 "띄우고,
# [해설]   기다리고, 끄는" 수명 관리 모듈.
# [해설] 프로세스: 이 파일의 코드는 모두 클라이언트(TUI/헤드리스) 프로세스에서 실행된다.
# [해설]   서버 쪽 코드는 server_graph.py다.
# [해설] 진입 심볼: ServerProcess(start/restart/stop) + generate_langgraph_json +
# [해설]   wait_for_server_healthy.
# [해설]   호출자는 server_manager.py의 start_server_and_get_agent다.
# [해설][흐름] 기동 순서: generate_langgraph_json이 langgraph.json 작성 ->
# [해설]   _spawn_process가 포트·환경·명령 준비 ->
# [해설]   _launch_process가 _server_bootstrap 모듈을 자식으로 실행 ->
# [해설]   wait_for_server_healthy가 `/ok` 폴링 ->
# [해설]   wait_for_graph_ready가 그래프를 한 번 조회해 지연 시작 오류를 앞당겨 드러냄.
# [해설] 설정 전달: ServerConfig는 os.environ의 DEEPAGENTS_CODE_SERVER_*에 이미 기록돼
# [해설]   있고, _build_server_env가 이를 복사해 자식에게 준다.
# [해설] 종료: POSIX는 프로세스 그룹에 SIGTERM -> 유예 ->
# [해설]   SIGKILL(_terminate_server_process). 클라이언트가 먼저 죽어도
# [해설]   pipe EOF로 서버가 스스로 종료한다(_server_bootstrap.py).
# [해설] 관련 모듈: server_manager.py, _server_bootstrap.py, server_graph.py,
# [해설]   _startup_error.py, offload_api.py.
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess  # noqa: S404
import sys
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import quote

from deepagents_code._env_vars import INVOKED_AS, SERVER_ENV_PREFIX
from deepagents_code._invocation import invoked_name
from deepagents_code._paths import (
    DEEPAGENTS_HOME_ENV,
    DEFAULT_PROFILE_MARKER_ENV,
    PATHS,
    export_profile_env,
)
from deepagents_code._startup_error import startup_error_from_output
from deepagents_code.config import (
    _INHERITED_PYTHONPATH_ENV,
    _USER_LANGSMITH_ENV_CARRIER,
    _encode_user_langsmith_env,
    _strip_dotenv_loaded_values,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Mapping

logger = logging.getLogger(__name__)

# [해설] 기본 호스트는 루프백(127.0.0.1). 서버는 인증 없이(noop) 로컬 접속만 받는
# [해설]   구조다(아래 _build_server_env의 LANGGRAPH_AUTH_TYPE 참고).
_DEFAULT_HOST = "127.0.0.1"

_EPHEMERAL_PORT = 0
"""Sentinel port meaning "let `start()` pick a free ephemeral port".

The server is internal and ephemeral — callers reach it via `ServerProcess.url`,
never a typed-in address — so it deliberately avoids binding the well-known
`langgraph dev` default (2024). Leaving 2024 free lets users run their own
`langgraph dev` projects alongside `deepagents-code` without a port collision.
"""

# [해설][설계] langgraph.json의 그래프 참조 "모듈:속성" 문자열. 클라이언트와 서버를 잇는
# [해설]   가장 중요한 한 줄이다.
# [해설] 이 값일 때만 offload HTTP 앱이 등록된다(generate_langgraph_json).
_DCODE_GRAPH_REF = "deepagents_code.server_graph:make_graph"
"""Built-in graph reference. Also gates registration of the offload HTTP app:
a custom `graph_ref` gets no `http` block and does not support `/offload`."""

_HEALTH_POLL_INTERVAL_LOCAL = 0.1

_HEALTH_POLL_INTERVAL_REMOTE = 0.3

_HEALTH_TIMEOUT = 60

_SHUTDOWN_TIMEOUT = 5
"""Seconds to wait for a graceful exit before escalating to a hard kill.

The server spends up to two seconds flushing buffered LangSmith traces inside
this window. The remaining margin lets LangGraph finish its own lifespan
teardown before dcode escalates; `TestFlushBudget` guards that margin.
"""

_SIGKILL_TIMEOUT = 2
"""Seconds to wait for the group/process to exit after SIGKILL."""

_ABANDONED_START_GRACE = 0.5
"""Graceful-exit window for a server whose startup was abandoned.

A server that never reached its caller holds no runs, checkpoints, or traces
worth flushing. Uvicorn answers SIGTERM by waiting for in-flight requests, so a
server still building its graph (for example, connecting to remote MCP
servers) would hold the full `_SHUTDOWN_TIMEOUT`. This window still lets an
idle server, and descendants such as stdio MCP servers, exit on SIGTERM.
"""

_WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200
"""Windows creation flag required for targeted console control signals.

A literal because `subprocess.CREATE_NEW_PROCESS_GROUP` exists only on
Windows. The flag also disables Ctrl+C handling for the new group, so
`CTRL_C_EVENT` is inert against it and Ctrl+Break is the only graceful
signal available.
"""

_WINDOWS_CTRL_BREAK_EVENT = 1
"""Windows Ctrl+Break event handled as a graceful SIGBREAK by Uvicorn.

A literal because `signal.CTRL_BREAK_EVENT` exists only on Windows. The
value is load-bearing: `subprocess.Popen.send_signal` dispatches on it
exactly, and 0 would mean `CTRL_C_EVENT` instead.
"""

_PROCESS_GROUP_POLL_INTERVAL = 0.05

_LOG_TAIL_CHARS = 3000
"""Max chars of subprocess log appended to the early-exit `RuntimeError` message.

Enough to carry a Python traceback without flooding the TUI banner when it
surfaces via `ServerStartFailed`.
"""

_STARTUP_ERROR_MARKER = "DEEPAGENTS_STARTUP_ERROR:"
"""Machine-readable prefix emitted by the server subprocess for known startup errors."""

# [해설][주의] 서버 자식 프로세스에 상속시키지 않을 환경변수 목록. PYTHONPATH가
# [해설]   대표적이다: 신뢰하지 않은 프로젝트의 경로가
# [해설] 서버 인터프리터의 sys.path에 오르면 표준/서드파티 모듈을 가로채 승인 단계 전에
# [해설]   코드가 실행될 수 있다.
_SERVER_ENV_DENYLIST = frozenset(
    {
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "GIT_ASKPASS",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NODE_OPTIONS",
        "PYTHONEXECUTABLE",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "SSH_ASKPASS",
    }
)
"""Inherited env keys that can alter subprocess startup behavior.

`PYTHONPATH` is stripped here so an inherited launch value cannot land on the
server interpreter's `sys.path` during startup, where a path inside an untrusted
project could shadow a stdlib/third-party module and run before any approval
gate. A user who launched with `PYTHONPATH` still wants it for their agent
`execute` commands, so `_build_server_env` relays the value via
`config._INHERITED_PYTHONPATH_ENV` and `agent._apply_inherited_pythonpath`
re-applies it only to the approval-gated shell backend.
"""


# [해설] 포트 점검/확보 헬퍼. _port_in_use는 bind 시도로 사용 여부를 확인하고,
# [해설]   _find_free_port는 OS에 임시 포트를 요청한다.
def _port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use.

    Args:
        host: Host to check.
        port: Port to check.

    Returns:
        `True` if the port is in use.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return True
        else:
            return False


def _find_free_port(host: str) -> int:
    """Find a free port on the given host.

    Args:
        host: Host to bind to.

    Returns:
        An available port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def get_server_url(host: str = _DEFAULT_HOST, port: int = _EPHEMERAL_PORT) -> str:
    """Build the server base URL.

    Args:
        host: Server host.
        port: Server port.

    Returns:
        Base URL string.
    """
    return f"http://{host}:{port}"


# [해설] 서버 로그의 마지막 줄부터 거꾸로 훑어 DEEPAGENTS_STARTUP_ERROR: 마커를 찾는다.
# [해설]   마커는 server_graph.py의
# [해설] _print_startup_error가 찍는다. 이 문자열이 두 파일 사이의 암묵적 계약이다.
def _extract_startup_error_marker(output: str) -> str | None:
    """Extract a marked startup error from subprocess output.

    Args:
        output: Combined stdout/stderr captured from the server subprocess.

    Returns:
        The marked startup error message, or `None` if no marker was emitted.
    """
    for line in reversed(output.splitlines()):
        if _STARTUP_ERROR_MARKER in line:
            _, summary = line.rsplit(_STARTUP_ERROR_MARKER, 1)
            return summary.strip() or None
    return None


# [해설][흐름] langgraph.json 구조: 1) graphs.agent = graph_ref 2) 내장 그래프일 때만
# [해설]   http.app = offload_api:app (+ 라우트 인증 옵션)
# [해설][흐름] 3) auth/env는 선택 4) checkpointer.path가 있으면 영속 체크포인터를 쓰도록
# [해설]   지정하고, 소유자 헤더(OWNER_KEY)를
# [해설]    configurable로 전달하게 한다(thread_ownership.py). 체크포인터가 없으면
# [해설]   LangGraph는 메모리 저장기를 쓴다.
def generate_langgraph_json(
    output_dir: str | Path,
    *,
    graph_ref: str = _DCODE_GRAPH_REF,
    env_file: str | None = None,
    checkpointer_path: str | None = None,
    auth_path: str | None = None,
) -> Path:
    """Generate a `langgraph.json` config file for `langgraph dev`.

    Registers the interactive `agent` graph and dcode's custom HTTP operations,
    which opt into LangGraph's route-auth layer (`enable_custom_route_auth`) so a
    deployment that configures auth gates them. Production runs `noop` auth and
    relies on the loopback bind, so that gate is inert there -- see
    `auth_path` below and THREAT_MODEL.md TB10. `/offload` is served by that
    backend boundary rather than exposed as another client-addressable graph.

    Args:
        output_dir: Directory to write the config file.
        graph_ref: Python "module:attribute" reference to the graph, where the
            attribute is a graph factory (e.g. `make_graph`) or a graph object.
            Custom graphs omit the built-in offload service, so `/offload` is
            unsupported when one is supplied.
        env_file: Optional path to an env file.
        checkpointer_path: Import path to an async context manager that yields a
            `BaseCheckpointSaver`. When set, the server persists checkpoint data
            to disk instead of in-memory.
        auth_path: Import path to a LangGraph `Auth` instance, emitted as the
            config's `auth.path`. Production servers run with
            `LANGGRAPH_AUTH_TYPE=noop` and no auth backend; this exists so
            tests can prove `enable_custom_route_auth` gates the operation
            routes when a deployment *does* configure one.

    Returns:
        Path to the generated config file.
    """
    config: dict[str, Any] = {
        "dependencies": ["."],
        "graphs": {"agent": graph_ref},
    }
    if graph_ref == _DCODE_GRAPH_REF:
        config["http"] = {
            "app": "deepagents_code.offload_api:app",
            "enable_custom_route_auth": True,
        }
    if auth_path:
        config["auth"] = {"path": auth_path}
    if env_file:
        config["env"] = env_file
    if checkpointer_path:
        config["checkpointer"] = {"path": checkpointer_path}
        from deepagents_code.thread_ownership import OWNER_KEY

        config.setdefault("http", {})["configurable_headers"] = {"include": [OWNER_KEY]}

    output_path = Path(output_dir) / "langgraph.json"
    output_path.write_text(json.dumps(config, indent=2))
    return output_path


# ---------------------------------------------------------------------------
# Scoped env-var management
# ---------------------------------------------------------------------------


# [해설] os.environ을 임시로 바꾸되 "예외가 났을 때만" 원래대로 되돌리는 컨텍스트
# [해설]   매니저. 정상 종료 시에는 새 값이 남는다(restart가 사용).
@contextlib.contextmanager
def _scoped_env_overrides(
    overrides: dict[str, str],
) -> Iterator[None]:
    """Apply env-var overrides, rolling back only on exception.

    Separates the concern of temporary `os.environ` mutations from subprocess
    management, making both independently testable.

    On normal exit the overrides are left in place (the caller "keeps"
    them). On exception the previous values are restored so the next attempt
    starts from a known-good state.

    Args:
        overrides: Key/value pairs to set in `os.environ`.

    Yields:
        Control to the caller.
    """
    prev: dict[str, str | None] = {}
    for key, val in overrides.items():
        prev[key] = os.environ.get(key)
        os.environ[key] = val
    try:
        yield
    except Exception:
        for key, old_val in prev.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val
        raise


# ---------------------------------------------------------------------------
# Health checking
# ---------------------------------------------------------------------------


# [해설][흐름] `{url}/ok`를 폴링해 서버가 떴는지 확인한다. 1) 자식이 이미 종료했으면
# [해설]   로그에서 마커를 찾아 즉시 실패
# [해설][흐름] 2) 200이면 성공 3) 아니면 폴링 간격만큼 대기 후 반복 4) 제한 시간(기본
# [해설]   60초) 초과 시 RuntimeError.
async def wait_for_server_healthy(
    url: str,
    *,
    timeout: float = _HEALTH_TIMEOUT,  # noqa: ASYNC109
    process: subprocess.Popen | None = None,
    read_log: Callable[[], str] | None = None,
    local: bool = False,
) -> None:
    """Poll a LangGraph server health endpoint until it responds.

    Args:
        url: Server base URL (health endpoint is `{url}/ok`).
        timeout: Max seconds to wait.
        process: Optional subprocess handle; if the process exits early
            we fail fast instead of waiting for the timeout.
        read_log: Optional callable returning log file contents (for
            error messages on early exit).
        local: Use a shorter poll interval for local servers.

    Raises:
        ModelConfigError: If startup reports missing credentials or a provider package.
        RuntimeError: If the server doesn't become healthy in time.
    """  # noqa: DOC502  # recovery error types are reconstructed from subprocess output
    import httpx

    poll_interval = (
        _HEALTH_POLL_INTERVAL_LOCAL if local else _HEALTH_POLL_INTERVAL_REMOTE
    )
    health_url = f"{url}/ok"
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    last_exc: Exception | None = None

    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            if process and process.poll() is not None:
                output = read_log() if read_log else ""
                msg = f"Server process exited with code {process.returncode}"
                if output:
                    summary = _extract_startup_error_marker(output)
                    if summary:
                        msg += f": {summary}"
                    msg += f"\n{output[-_LOG_TAIL_CHARS:]}"
                error = startup_error_from_output(output, msg)
                raise error

            try:
                resp = await client.get(health_url, timeout=2)
                if resp.status_code == 200:  # noqa: PLR2004
                    logger.info("Server is healthy at %s", url)
                    return
                last_status = resp.status_code
                logger.debug("Health check returned status %d", resp.status_code)
            except (httpx.TransportError, OSError) as exc:
                logger.debug("Health check attempt failed: %s", exc)
                last_exc = exc

            await asyncio.sleep(poll_interval)

    msg = f"Server did not become healthy within {timeout}s"
    if last_status is not None:
        msg += f" (last status: {last_status})"
    elif last_exc is not None:
        msg += f" (last error: {last_exc})"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Server command / env construction
# ---------------------------------------------------------------------------


# [해설] 서버 실행 명령 argv. 실제 Popen에서는 `-m langgraph_cli` 자리가
# [해설]   _server_bootstrap으로 치환된다(_launch_process 참고).
# [해설] --no-reload, --no-browser가 붙어 있어 개발 서버지만 파일 감시/브라우저 열기를
# [해설]   하지 않는다.
def _build_server_cmd(config_path: Path, *, host: str, port: int) -> list[str]:
    """Build the `langgraph dev` command line.

    Args:
        config_path: Path to the `langgraph.json` config file.
        host: Host to bind.
        port: Port to bind.

    Returns:
        Command argv list.
    """
    return [
        sys.executable,
        "-m",
        "langgraph_cli",
        "dev",
        "--host",
        host,
        "--port",
        str(port),
        "--no-browser",
        "--no-reload",
        "--config",
        str(config_path),
    ]


# [해설][설계] 서버 자식의 환경 구성. 현재 os.environ을 복사한 뒤 1) dotenv로 주입된 값
# [해설]   제거 2) 프로필 환경 내보내기
# [해설] 3) 인증은 noop, 버전 확인 끄기 4) 위험한 변수 제거 5) 사용자 PYTHONPATH와
# [해설]   LangSmith 환경은 "전달용 변수"에 담아
# [해설] 승인이 필요한 사용자 명령에만 다시 적용되게 한다. DEEPAGENTS_CODE_SERVER_*
# [해설]   설정은 여기서 따로 건드리지 않고 그대로 상속된다.
def _build_server_env() -> dict[str, str]:
    """Build the environment dict for the server subprocess.

    Copies `os.environ`, sets required flags, and strips variables that are not
    needed or can alter subprocess startup behavior.

    A launch-time `PYTHONPATH` is captured into `config._INHERITED_PYTHONPATH_ENV`
    before being stripped, so the value never reaches the server interpreter's
    `sys.path` but can still be re-applied to agent `execute` commands downstream.

    Returns:
        Environment dict for `subprocess.Popen`.
    """
    env = os.environ.copy()
    _strip_dotenv_loaded_values(env)
    export_profile_env(env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LANGGRAPH_AUTH_TYPE"] = "noop"
    # dcode pins `langgraph-api`, so the PyPI upgrade check only costs a network
    # fetch at startup and logs a `pip install -U` hint users cannot act on.
    env["LANGGRAPH_NO_VERSION_CHECK"] = "1"
    env[INVOKED_AS] = invoked_name()

    # Capture a launch-time PYTHONPATH before stripping it. Never trust inherited
    # carrier vars: overwrite them only from bootstrap state in this process.
    env.pop(_INHERITED_PYTHONPATH_ENV, None)
    env.pop(_USER_LANGSMITH_ENV_CARRIER, None)
    inherited_pythonpath = os.environ.get("PYTHONPATH")

    for key in (
        "LANGGRAPH_AUTH",
        "LANGGRAPH_CLOUD_LICENSE_KEY",
        "LANGSMITH_TENANT_ID",
        *_SERVER_ENV_DENYLIST,
    ):
        env.pop(key, None)
    env["LANGSMITH_CONTROL_PLANE_API_KEY"] = ""

    if inherited_pythonpath is not None:
        env[_INHERITED_PYTHONPATH_ENV] = inherited_pythonpath
    env[_USER_LANGSMITH_ENV_CARRIER] = _encode_user_langsmith_env()
    return env


# [해설][주의] 호출자 override로 덮어쓸 수 없는 키 집합. 프로필 선택이나 LangSmith 운반
# [해설]   변수를 바꿀 수 있으면 두 프로세스의 신뢰 기준이 갈라진다.
_IMMUTABLE_SERVER_ENV_KEYS = frozenset(
    {
        DEEPAGENTS_HOME_ENV,
        DEFAULT_PROFILE_MARKER_ENV,
        _INHERITED_PYTHONPATH_ENV,
        _USER_LANGSMITH_ENV_CARRIER,
    }
)
"""Keys `_build_server_env` owns outright, immune to caller overrides.

`persist_env` validates its keys against `SERVER_ENV_PREFIX`, but `update_env`
accepts any key, so without this an `update_env` caller could point the server
at a different profile than the client (splitting the trust root across the two
processes), forge the LangSmith carrier that authenticates approval-gated user
commands, or inject a `PYTHONPATH` into them. Dropping the keys before the
overrides apply keeps that a single list to extend rather than one re-pin per
key, which is how `_INHERITED_PYTHONPATH_ENV` was missed.
"""


# [해설] 최종 자식 환경 = _build_server_env + (영속 override + 한 번 쓰는 override) -
# [해설]   불변 키. 프로필 변경 시도는 경고만 남기고 무시한다.
def _server_env_with_overrides(
    persistent: Mapping[str, str], scoped: Mapping[str, str]
) -> dict[str, str]:
    """Assemble the server environment, ignoring overrides of pinned keys.

    Returns:
        The child environment for the server subprocess.
    """
    env = _build_server_env()
    overrides: dict[str, str] = {**persistent, **scoped}
    # Profile selection is immutable and is not a restart override. Say so when
    # a caller actually tried: silently pointing the server somewhere other
    # than they asked is the kind of thing that gets debugged twice.
    requested = overrides.get(DEEPAGENTS_HOME_ENV)
    if requested is not None and requested != str(PATHS.profile.root):
        logger.warning(
            "Ignoring the %s=%r override for the server subprocess. The "
            "profile is fixed at launch to %s.",
            DEEPAGENTS_HOME_ENV,
            requested,
            PATHS.profile.root,
        )
    for key in _IMMUTABLE_SERVER_ENV_KEYS:
        overrides.pop(key, None)
    env.update(overrides)
    return env


# ---------------------------------------------------------------------------
# Process-group teardown
# ---------------------------------------------------------------------------


# [해설][설계] 아래 종료 헬퍼 묶음: 서버는 새 세션(start_new_session=True)으로 떠서 자기
# [해설]   프로세스 그룹의 리더가 된다.
# [해설] 그 그룹 전체에 신호를 보내면 `langgraph dev`의 자손(예: stdio MCP 서버)까지
# [해설]   정리된다.
# [해설] 방어 로직: dcode 자신의 그룹은 절대 대상이 되지 않는다.
def _server_process_group(pid: int) -> int | None:
    """Return the server's own process group id to signal, or `None`.

    The server is spawned with `start_new_session=True` on POSIX, so it leads
    its own session and process group (its pgid equals its pid). Signaling that
    group reaches the whole `langgraph dev` process tree, so descendants receive
    the same shutdown signals as the root rather than being left running when
    only the root is signaled.

    Returns `None` on Windows (which uses a console process group instead) and
    whenever the server is not the leader of its own dedicated POSIX group. As
    a defensive check, the `pgid == os.getpgid(0)` clause also refuses to return
    dcode's own group, so the group handed back can never be the one whose
    termination would take down the TUI.

    Args:
        pid: Process id of the server subprocess.

    Returns:
        The server's dedicated process group id, or `None` to fall back to
        signaling just the root process.
    """
    if sys.platform == "win32":
        return None
    try:
        pgid = os.getpgid(pid)
        own_pgid = os.getpgid(0)
    except ProcessLookupError:
        # The process already exited; there is no group left to signal.
        return None
    except OSError:
        # Resolving the group failed unexpectedly (getpgid on an owned child
        # should not). Fall back to root-only signaling, but surface it so a
        # silently orphaned descendant tree is diagnosable rather than invisible.
        logger.warning(
            "Could not resolve process group for pid=%d; "
            "falling back to root-only signaling",
            pid,
            exc_info=True,
        )
        return None
    if pgid != pid or pgid == own_pgid:
        return None
    return pgid


# [해설] 리더만이 아니라 그룹 전체가 사라질 때까지 폴링한다(리더의 좀비를 poll로
# [해설]   수거해야 killpg(…,0) 탐지가 맞는다).
def _wait_for_process_group_exit(
    process: subprocess.Popen[Any], pgid: int, timeout: float
) -> bool:
    """Wait until every process in a POSIX process group has exited.

    `Popen.wait()` only observes the group leader. Poll it on every pass so an
    exited leader does not remain a zombie and keep the group probe alive, then
    continue probing because descendants may remain after the leader exits.

    Args:
        process: The group leader, reaped as soon as it exits.
        pgid: Process group id to probe.
        timeout: Maximum seconds to wait for the whole group.

    Returns:
        `True` when the group is gone, or `False` on timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        # `poll()` reaps an exited leader without blocking. Until that happens,
        # its zombie entry keeps `killpg(..., 0)` reporting the group as alive.
        process.poll()
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            process.wait()
            return True
        except PermissionError:
            # The group still exists even if the probe is not permitted.
            pass

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_PROCESS_GROUP_POLL_INTERVAL, remaining))


# [해설] Windows: Ctrl+Break로 정상 종료 요청, 콘솔이 없으면 terminate로 대체.
# [해설]   Windows에서는 부모 감시 pipe(부트스트랩)가 없다.
def _signal_windows_server(process: subprocess.Popen[Any]) -> None:
    """Request graceful Windows shutdown, terminating when no console exists.

    Args:
        process: The owned server process to signal.

    Raises:
        ProcessLookupError: The server exited before the signal landed.
    """
    try:
        process.send_signal(_WINDOWS_CTRL_BREAK_EVENT)
    except ProcessLookupError:
        raise
    except OSError:
        logger.warning(
            "Failed to send Ctrl+Break to server process pid=%d; "
            "falling back to terminate",
            process.pid,
        )
        process.terminate()


# [해설][흐름] 종료 절차: 1) 프로세스 그룹 확인 2) 그룹에 SIGTERM(Windows는 Ctrl+Break)
# [해설]   3) 유예 시간 대기
# [해설][흐름] 4) 남아 있으면 SIGKILL 5) 그래도 안 끝나면 경고. 이미 종료/신호 실패는
# [해설]   각각 로그만 남긴다.
def _terminate_server_process(
    process: subprocess.Popen[Any], *, grace: float | None = None
) -> None:
    """Terminate the `langgraph dev` server and its descendants.

    Signals the server for a graceful exit, waits `grace`, then
    escalates to a hard kill: SIGTERM then SIGKILL on POSIX, Ctrl+Break then
    `TerminateProcess` on Windows.

    On POSIX the whole detached process group is signaled via
    `os.killpg`, and teardown waits for the entire group to exit — not just the
    root — so a child that outlives the `langgraph dev` root is still escalated
    to SIGKILL rather than orphaned. On Windows, the server's console process
    group receives Ctrl+Break, which Uvicorn handles as a graceful SIGBREAK and
    runs the Starlette lifespan shutdown. If Ctrl+Break cannot be delivered,
    such as in a headless session without an attached console, the owned child
    is terminated instead. Note that the Windows escalation reaches only the
    root handle, so a descendant that outlives it is orphaned; the POSIX group
    path is the only one that escalates group-wide.

    If no dedicated group is available on POSIX, only the root process is
    signaled. `_server_process_group` guarantees dcode's own POSIX process
    group is never targeted.

    Args:
        process: The running server subprocess to terminate.
        grace: Seconds to wait for a graceful exit before the hard kill.

            `None` uses `_SHUTDOWN_TIMEOUT`, read at call time.
    """
    timeout = _SHUTDOWN_TIMEOUT if grace is None else grace
    pid = process.pid
    pgid = _server_process_group(pid)
    # Windows resolves no pgid, but Ctrl+Break still reaches the whole console
    # group — while the escalation below only ever kills the root handle. The
    # two scopes are tracked separately so neither log line overstates what
    # actually happened.
    signal_scope = (
        "process group" if pgid is not None or sys.platform == "win32" else "process"
    )
    kill_scope = "process group" if pgid is not None else "process"

    logger.info("Stopping langgraph dev server (pid=%d)", pid)
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
            stopped = _wait_for_process_group_exit(process, pgid, timeout)
        else:
            if sys.platform == "win32":
                _signal_windows_server(process)
            else:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                stopped = False
            else:
                stopped = True
    except ProcessLookupError:
        # The server exited before the graceful signal landed; nothing to reap.
        logger.debug(
            "Server %s pid=%d already exited before the graceful signal",
            signal_scope,
            pid,
        )
        return
    except OSError:
        # The graceful signal could not be delivered (e.g. EPERM). We never
        # reach the hard-kill escalation, so the server is left running —
        # report it with the same fidelity as a failed SIGKILL rather than a
        # bare "error stopping".
        logger.exception(
            "Failed to signal server %s pid=%d; it may be orphaned",
            signal_scope,
            pid,
        )
        return

    if stopped:
        return

    logger.warning("Server did not stop gracefully, killing %s", kill_scope)
    if sys.platform == "win32":
        logger.warning(
            "Windows escalation reaches only the server root; any surviving "
            "`langgraph dev` descendant is left orphaned"
        )
    # Guard escalation explicitly: `ProcessLookupError` means the group exited
    # just before the hard kill, while any other `OSError` means it may be
    # orphaned.
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
            if not _wait_for_process_group_exit(process, pgid, _SIGKILL_TIMEOUT):
                logger.warning(
                    "Server %s pid=%d did not exit after the hard kill", kill_scope, pid
                )
        else:
            process.kill()
            try:
                process.wait(timeout=_SIGKILL_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Server %s pid=%d did not exit after the hard kill", kill_scope, pid
                )
    except ProcessLookupError:
        logger.debug(
            "Server %s pid=%d already exited before the hard kill",
            kill_scope,
            pid,
        )
    except OSError:
        logger.exception(
            "Hard kill failed for server %s pid=%d; it may be orphaned",
            kill_scope,
            pid,
        )


# [해설] 디버그 모드에서 보존된 서버 로그 경로를 모아 두는 전역 큐. TUI가 화면을 소유한
# [해설]   동안의 출력은 사라지므로,
# [해설] 터미널이 복원된 뒤 emit_preserved_log_notices로 한꺼번에 안내한다.
# ---------------------------------------------------------------------------
# Preserved server-log notices
# ---------------------------------------------------------------------------

# Debug-preserved server-log paths awaiting announcement. Recorded by
# `_stop_process_locked` and drained by `emit_preserved_log_notices` once the
# terminal is restored (a notice printed while Textual still owns the alternate
# screen is discarded on exit).
#
# Process-global rather than per-`ServerProcess` on purpose: preserved logs
# accumulate both across `/restart` (which reuses one instance) and across
# throwaway instances that never become the app's tracked process — a server
# whose startup fails, or the previous server replaced by a cwd switch. A
# per-instance slot would strand every path but the tracked instance's last
# one; a single queue lets the terminal teardown announce them all.
_PENDING_PRESERVED_LOGS: list[Path] = []
# Guards the queue independently of any instance's `_state_lock`, since
# `_stop_process_locked` can append from a fallback worker thread.
_PENDING_PRESERVED_LOGS_LOCK = threading.Lock()


def emit_preserved_log_notices() -> None:
    """Print every pending debug-preserved server-log path to stderr, once each.

    Deferred out of `_stop_process_locked` so the interactive TUI can surface
    the paths after Textual restores the terminal. Call this only from terminal
    stop sites that run after the terminal is restored or with no TUI active
    (`run_textual_app` finally, `server_session` finally, `ServerProcess`
    `__aexit__`) — never from the hidden in-session teardown or `/restart`,
    whose prints would be swallowed by the alternate screen.

    Draining is atomic, so the interactive path's overlapping `stop()` calls
    (and any other double teardown) still announce each path exactly once.
    """
    with _PENDING_PRESERVED_LOGS_LOCK:
        paths = list(_PENDING_PRESERVED_LOGS)
        _PENDING_PRESERVED_LOGS.clear()
    for path in paths:
        print(f"Server log preserved at: {path}", file=sys.stderr)  # noqa: T201


# ---------------------------------------------------------------------------
# ServerProcess
# ---------------------------------------------------------------------------


# [해설][설계] ServerProcess: 서버 자식 프로세스 한 대의 수명
# [해설]   관리자(시작/중지/재시작/로그/환경 override).
# [해설] 락 2개: _lifecycle_lock(asyncio, 시작·재시작 직렬화) / _state_lock(threading,
# [해설]   프로세스·로그 상태). 종료는 다른 스레드에서도
# [해설] 호출되므로 둘을 구분해서 쓴다. 세대 번호(_stop_generation)는 "종료가 이긴 뒤
# [해설]   재시작이 서버를 되살리는" 경쟁을 막는다.
class ServerProcess:
    """Manages a `langgraph dev` server subprocess.

    Focuses on subprocess lifecycle (start, stop, restart) and health checking.
    Env-var management for restarts (e.g. configuration changes requiring a full
    restart) is handled by `_scoped_env_overrides`, keeping this class focused
    on process management.

    On POSIX a parent-owned pipe lets the server root detect owner death,
    including SIGKILL. Windows retains the existing explicit-stop behavior.
    """

    def __init__(
        self,
        *,
        host: str = _DEFAULT_HOST,
        port: int = _EPHEMERAL_PORT,
        config_dir: str | Path | None = None,
        owns_config_dir: bool = False,
        scaffold: Callable[[Path], None] | None = None,
    ) -> None:
        """Initialize server process manager.

        Args:
            host: Host to bind the server to.
            port: Initial port to bind the server to. Defaults to
                `_EPHEMERAL_PORT` (0), so `start()` picks a free port and avoids
                squatting the well-known `langgraph dev` default (2024).

                An explicit port is honored, but `start()` still falls back to a
                free port if it is already in use.
            config_dir: Directory containing `langgraph.json`.
            owns_config_dir: When `True`, the server will delete `config_dir`
                on `stop()`.
            scaffold: Optional callable that (re)generates the working
                directory's `langgraph.json` and supporting files. When the
                config is missing at `start()` (e.g. the temp dir was purged
                between the initial boot and a later `/restart`), it is invoked
                to rebuild the workspace instead of failing.
        """
        self.host = host
        self.port = port
        self.config_dir = Path(config_dir) if config_dir else None
        self._owns_config_dir = owns_config_dir
        self._scaffold = scaffold
        self._process: subprocess.Popen | None = None
        self._parent_pipe_writer: int | None = None
        self._parent_pipe_finalizer: weakref.finalize | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._log_file: tempfile.NamedTemporaryFile | None = None  # ty: ignore[invalid-type-form]
        self._env_overrides: dict[str, str] = {}
        self._persistent_env_overrides: dict[str, str] = {}
        # Async lifecycle calls must be serialized by task, not by OS thread:
        # every coroutine on an event loop runs on the same thread, so an
        # RLock would let unrelated tasks enter while another task is awaiting.
        self._lifecycle_lock = asyncio.Lock()
        # Synchronous shutdown can also run from a fallback worker thread. Keep
        # its critical sections short and never hold this lock across an await.
        self._state_lock = threading.Lock()
        self._stopped = False
        self._stop_generation = 0

    @property
    def url(self) -> str:
        """Server base URL."""
        return get_server_url(self.host, self.port)

    @property
    def running(self) -> bool:
        """Whether the server process is running."""
        with self._state_lock:
            return self._running_locked()

    def _running_locked(self) -> bool:
        """Return whether the process is running while `_state_lock` is held."""
        return self._process is not None and self._process.poll() is None

    def _read_log_file(self) -> str:
        """Read the server log file contents.

        Returns:
            Log file contents as a string (may be empty).
        """
        with self._state_lock:
            if self._log_file is None:
                return ""
            try:
                self._log_file.flush()
                return Path(self._log_file.name).read_text(
                    encoding="utf-8", errors="replace"
                )
            # `ValueError` covers a flush/read on a closed handle ("I/O
            # operation on closed file"): re-checking `is None` under
            # `_state_lock` makes a closed-but-non-None handle nearly
            # unreachable, so this is defensive. `read_text(errors="replace")`
            # cannot raise `ValueError` for decoding, so the catch stays narrow
            # in practice.
            except OSError, ValueError:
                logger.warning(
                    "Failed to read server log file %s",
                    self._log_file.name,
                    exc_info=True,
                )
                return ""

    # [해설] 공개 시작 메서드. 락을 잡고 _start에 위임한다.
    async def start(
        self,
        *,
        timeout: float = _HEALTH_TIMEOUT,  # noqa: ASYNC109
        after_spawn: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Start the `langgraph dev` server and wait for it to be healthy.

        Args:
            timeout: Max seconds to wait for the server to become healthy.
            after_spawn: Awaited once the subprocess is spawned, before the
                health check. Lets callers overlap work with server boot that
                must finish before the health check's imports run.

        Raises:
            RuntimeError: If the server fails to start or become healthy.
        """  # noqa: DOC502  # `RuntimeError` propagates from `_start`
        async with self._lifecycle_lock:
            await self._start(timeout=timeout, after_spawn=after_spawn)

    # [해설][흐름] 1) _spawn_process로 자식 생성(이미 실행 중이면 None) 2) after_spawn
    # [해설]   콜백(호출자가 서버 부팅과 겹쳐 할 일)
    # [해설][흐름] 3) wait_for_server_healthy로 `/ok` 대기 4) 성공 못 하면(취소 포함)
    # [해설]   finally에서 짧은 유예로 stop을 스레드에서 호출해 고아 프로세스를 막는다.
    async def _start(
        self,
        *,
        timeout: float,  # noqa: ASYNC109
        expected_stop_generation: int | None = None,
        after_spawn: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Start while the caller owns `_lifecycle_lock`.

        Args:
            timeout: Max seconds to wait for the server to become healthy.
            expected_stop_generation: Generation captured before a restart's
                stop phase. If synchronous terminal shutdown ran meanwhile,
                abort instead of resurrecting the subprocess.
            after_spawn: Awaited once the subprocess is spawned, before the
                health check.
        """
        process = self._spawn_process(
            expected_stop_generation=expected_stop_generation,
        )
        if process is None:
            return
        started = False
        try:
            if after_spawn is not None:
                await after_spawn()
            await wait_for_server_healthy(
                self.url,
                timeout=timeout,
                process=process,
                read_log=self._read_log_file,
                local=True,
            )
            started = True
        finally:
            if not started:
                # Reap the subprocess we just spawned if startup did not
                # complete — including cancellation (e.g. Ctrl+D / SIGINT before
                # the health check returns). A `finally` rather than `except
                # Exception` is deliberate: `asyncio.CancelledError` is a
                # `BaseException`, so an `except Exception` guard would skip this
                # and orphan the process. Offload `stop()` because terminating a
                # subprocess can block for several seconds; restart cancellation
                # runs this path on Textual's event loop. The inner guard stops a
                # `stop()` error from masking the exception already propagating;
                # `stop()` is effectively non-raising today, so if it does fire it
                # signals an unexpected leak — hence `error`, not `warning`.
                # The server has not built its graph yet, so nothing needs the
                # full graceful window.
                try:
                    await asyncio.to_thread(self.stop, grace=_ABANDONED_START_GRACE)
                except Exception:
                    logger.exception(
                        "Error stopping server during startup cleanup",
                    )

    # [해설][흐름] 프로세스 생성 준비(상태 락 안에서 동기 실행): 1) 재시작 세대 확인 2)
    # [해설]   이미 실행 중이면 None
    # [해설][흐름] 3) 작업 디렉터리 결정(임시 디렉터리 또는 config_dir) 4)
    # [해설]   langgraph.json이 없으면 scaffold로 재생성
    # [해설][흐름] 5) 포트 결정(0이면 빈 포트, 사용 중이면 대체) 6) 명령과 환경 구성 7)
    # [해설]   로그 파일 생성 후 _launch_process.
    def _spawn_process(
        self,
        *,
        expected_stop_generation: int | None,
    ) -> subprocess.Popen | None:
        """Synchronously prepare and spawn the subprocess under `_state_lock`.

        Args:
            expected_stop_generation: Optional shutdown generation required by
                a restart.

        Returns:
            The new process, or `None` when one is already running.

        Raises:
            asyncio.CancelledError: If terminal shutdown preempted a restart.
            RuntimeError: If the server workspace cannot be prepared.
        """
        with self._state_lock:
            if (
                expected_stop_generation is not None
                and self._stop_generation != expected_stop_generation
            ):
                raise asyncio.CancelledError
            if self._running_locked():
                return None
            # Retire the exited generation before replacing its log or writer.
            self._stop_process_locked()
            self._stopped = False

            work_dir = self.config_dir
            if work_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory(
                    prefix="deepagents_server_"
                )
                work_dir = Path(self._temp_dir.name)

            config_path = work_dir / "langgraph.json"
            if not config_path.exists() and self._scaffold is not None:
                logger.info("langgraph.json missing in %s; rescaffolding", work_dir)
                try:
                    work_dir.mkdir(parents=True, exist_ok=True)
                    self._scaffold(work_dir)
                except OSError as exc:
                    msg = f"Failed to rescaffold server workspace at {work_dir}: {exc}"
                    raise RuntimeError(msg) from exc
            if not config_path.exists():
                if self._scaffold is not None:
                    contents = sorted(p.name for p in work_dir.iterdir())
                    msg = (
                        f"Rescaffolding {work_dir} did not produce langgraph.json "
                        f"(directory contents: {contents})."
                    )
                else:
                    msg = (
                        f"langgraph.json not found in {work_dir}. "
                        "Call generate_langgraph_json() first."
                    )
                raise RuntimeError(msg)

            if self.port == _EPHEMERAL_PORT:
                self.port = _find_free_port(self.host)
                logger.info(
                    "Using ephemeral port %d for langgraph dev server", self.port
                )
            elif _port_in_use(self.host, self.port):
                self.port = _find_free_port(self.host)
                logger.info("Requested port in use, using port %d instead", self.port)

            cmd = _build_server_cmd(config_path, host=self.host, port=self.port)
            env = _server_env_with_overrides(
                self._persistent_env_overrides, self._env_overrides
            )

            logger.info("Starting langgraph dev server: %s", " ".join(cmd))
            self._log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
                prefix="deepagents_server_log_",
                suffix=".txt",
                delete=False,
                mode="w",
                encoding="utf-8",
            )
            try:
                self._process = self._launch_process(cmd, work_dir, env)
            except BaseException:
                # `_start` has not entered its cleanup block until spawning
                # returns. Failed launches must release their pipe/log here.
                self._stop_process_locked()
                raise
            return self._process

    # [해설][주의] 실제 Popen. POSIX에서는 `python -m langgraph_cli` 대신 `-m
    # [해설]   ...client.launch._server_bootstrap`을 실행하고,
    # [해설]   인자로 상속 fd 목록과 종료 유예 시간을 넘긴다. 클라이언트는 pipe 쓰기
    # [해설]   쪽을 보관하고 읽기 쪽 fd만 자식에게 준다.
    # [해설]   세션 잠금 fd도 함께 상속해 서버가 살아 있는 동안 설치가 "사용 중"으로
    # [해설]   표시되게 한다.
    # [해설] Windows에는 pipe 감시가 없다(분기 참고). cwd는 work_dir이므로 생성된
    # [해설]   checkpointer.py의 상대 경로 ./checkpointer.py가 해석된다.
    def _launch_process(
        self, cmd: list[str], work_dir: Path, env: dict[str, str]
    ) -> subprocess.Popen:
        """Launch one generation while the caller owns `_state_lock`.

        Returns:
            The process owning this generation's server root.
        """
        if sys.platform == "win32":
            return subprocess.Popen(  # noqa: S603  # argv is built internally
                cmd,
                cwd=str(work_dir),
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=False,
                creationflags=_WINDOWS_CREATE_NEW_PROCESS_GROUP,
            )

        from deepagents_code._session_lock import session_lock_fds

        reader, writer = os.pipe()  # Both ends are non-inheritable by default.
        self._parent_pipe_writer = writer
        # Abandoning the manager must not leak its writer. Explicit teardown
        # invokes this same finalizer so a reused descriptor is never closed twice.
        self._parent_pipe_finalizer = weakref.finalize(self, os.close, writer)
        # The session lock keeps the install marked in use while the server
        # runs, so a migration does not rebuild it underneath.
        inherited = (reader, *session_lock_fds())
        try:
            return subprocess.Popen(  # noqa: S603  # argv is built internally
                [
                    *cmd[:2],
                    "deepagents_code.client.launch._server_bootstrap",
                    ",".join(map(str, inherited)),
                    str(_SHUTDOWN_TIMEOUT),
                    *cmd[3:],
                ],
                cwd=str(work_dir),
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                creationflags=0,
                pass_fds=inherited,
            )
        finally:
            os.close(reader)

    # [해설][흐름] 그래프 준비 확인: `/assistants/agent/graph`를 호출해 200을 받을
    # [해설]   때까지 반복한다. 이 요청이 서버의 make_graph를
    # [해설]   처음 실제로 호출시켜 모델/MCP/샌드박스 설정 오류를 첫 사용자 입력 전에
    # [해설]   드러낸다(docstring: lazy startup failures surface early). 중간에 자식이
    # [해설]   죽으면 로그의 마커를 오류로 변환한다.
    async def wait_for_graph_ready(
        self,
        graph_name: str = "agent",
        *,
        timeout: float = _HEALTH_TIMEOUT,  # noqa: ASYNC109
    ) -> None:
        """Resolve the served graph once so lazy startup failures surface early.

        Args:
            graph_name: Registered graph name from `langgraph.json`.
            timeout: Max seconds to wait for the graph readiness request.

        Raises:
            ModelConfigError: If startup reports missing credentials or a
                provider package.
            RuntimeError: If the server process exits or the graph endpoint
                does not return a successful response.
        """  # noqa: DOC502  # recovery errors are reconstructed from subprocess output
        import httpx

        if self._process is None:
            msg = "Server process is not running"
            raise RuntimeError(msg)

        graph_url = f"{self.url}/assistants/{quote(graph_name, safe='')}/graph"
        deadline = time.monotonic() + timeout

        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    msg = f"Server process exited with code {self._process.returncode}"
                    output = self._read_log_file()
                    if output:
                        summary = _extract_startup_error_marker(output)
                        if summary:
                            msg += f": {summary}"
                        msg += f"\n{output[-_LOG_TAIL_CHARS:]}"
                    error = startup_error_from_output(output, msg)
                    raise error

                remaining = max(0.1, deadline - time.monotonic())
                try:
                    resp = await client.get(graph_url, timeout=remaining)
                except (httpx.TransportError, httpx.TimeoutException, OSError) as exc:
                    output = self._read_log_file()
                    summary = _extract_startup_error_marker(output)
                    if self._process.poll() is not None:
                        msg = (
                            f"Server process exited with code "
                            f"{self._process.returncode}"
                        )
                    else:
                        msg = (
                            f"Server graph '{graph_name}' did not initialize within "
                            f"{timeout}s"
                        )
                    if summary:
                        msg += f": {summary}"
                    if output:
                        msg += f"\n{output[-_LOG_TAIL_CHARS:]}"
                    error = startup_error_from_output(output, msg)
                    raise error from exc

                if resp.status_code == 200:  # noqa: PLR2004
                    logger.info("Server graph %s is ready at %s", graph_name, self.url)
                    return

                output = self._read_log_file()
                msg = (
                    f"Server graph '{graph_name}' failed readiness check "
                    f"(status: {resp.status_code})"
                )
                summary = _extract_startup_error_marker(output)
                if summary:
                    msg += f": {summary}"
                if output:
                    msg += f"\n{output[-_LOG_TAIL_CHARS:]}"
                error = startup_error_from_output(output, msg)
                raise error

        msg = f"Server graph '{graph_name}' did not initialize within {timeout}s"
        raise RuntimeError(msg)

    # [해설] 서버 프로세스와 로그만 정리하고 설정 디렉터리는 남겨 같은 설정으로 재시작할
    # [해설]   수 있게 한다(restart가 사용).
    def _stop_process(self) -> None:
        """Stop only the server subprocess and its log file.

        Unlike `stop()`, this does NOT clean up the config directory or temp
        directory, so the server can be restarted with the same config.
        """
        with self._state_lock:
            self._stop_process_locked()

    # [해설] 종료 본체. 프로세스 종료 -> 부모 감시 pipe의 쓰기 쪽을 닫음(자식이 EOF를
    # [해설]   보고 스스로 정리) -> 로그 파일 정리.
    # [해설] pipe는 grace 시간 동안 유지한다(명시적 종료 중에 EOF가 경쟁 종료를 일으키지
    # [해설]   않도록).
    def _stop_process_locked(self, *, grace: float | None = None) -> None:
        """Stop the subprocess while `_state_lock` is held.

        Args:
            grace: Seconds to wait for a graceful exit before the hard kill.

                `None` uses `_SHUTDOWN_TIMEOUT`.
        """
        try:
            if self._process is not None and self._process.poll() is None:
                _terminate_server_process(self._process, grace=grace)
                # Best-effort termination may fail, for example with EPERM.
                if self._process.poll() is None:
                    logger.warning(
                        "Dropping handle to server pid=%d that is still running; "
                        "it may be orphaned",
                        self._process.pid,
                    )
        finally:
            self._process = None
            # Preserve the writer throughout the explicit stop's grace budget;
            # EOF must not start a competing shutdown while that stop is running.
            if self._parent_pipe_finalizer is not None:
                self._parent_pipe_finalizer()
                self._parent_pipe_finalizer = None
            self._parent_pipe_writer = None
            self._close_log_file_locked()

    # [해설] 로그 파일 닫기. DEBUG면 삭제하지 않고 경로를 큐에 넣어 나중에 안내하고,
    # [해설]   아니면 삭제한다.
    def _close_log_file_locked(self) -> None:
        """Release the generation's log even when process creation failed."""
        if self._log_file is not None:
            log_path = Path(self._log_file.name)
            try:
                self._log_file.close()
            except OSError:
                logger.debug("Failed to close log file", exc_info=True)

            from deepagents_code._env_vars import DEBUG, is_env_truthy

            if is_env_truthy(DEBUG):
                # Queue the path rather than printing here: teardown can run
                # while Textual still owns the terminal, so the notice is
                # emitted later via `emit_preserved_log_notices`. Appending
                # (not overwriting) keeps every restart's log announceable.
                with _PENDING_PRESERVED_LOGS_LOCK:
                    _PENDING_PRESERVED_LOGS.append(log_path)
            else:
                try:
                    log_path.unlink()
                except OSError:
                    logger.debug("Failed to clean up log file", exc_info=True)
            self._log_file = None

    # [해설] 완전 종료(멱등, 동시 호출 안전). 세대 번호를 올려 진행 중인 재시작이 서버를
    # [해설]   되살리지 못하게 하고,
    # [해설] 임시/소유 설정 디렉터리까지 정리한다.
    def stop(self, *, grace: float | None = None) -> None:
        """Stop the server process and clean up all resources.

        Idempotent and safe to call concurrently. The synchronous state lock
        prevents process teardown and resource cleanup from interleaving, while
        the generation counter prevents an in-flight async restart from
        spawning a replacement after this terminal stop wins the race.

        Args:
            grace: Seconds to wait for a graceful exit before the hard kill.

                `None` uses `_SHUTDOWN_TIMEOUT`. Only the first call's value
                applies; later calls wait for that stop to finish.
        """
        with self._state_lock:
            if self._stopped:
                return
            self._stopped = True
            self._stop_generation += 1

            self._stop_process_locked(grace=grace)

            if self._temp_dir is not None:
                try:
                    self._temp_dir.cleanup()
                except OSError:
                    # Debug, not warning (unlike the config dir below): a
                    # directory under the OS temp root is eventually reclaimed
                    # by the OS temp reaper (systemd-tmpfiles / tmpwatch / the
                    # macOS periodic cleanup), so a failure here is not the
                    # unrecoverable, never-retried leak the config dir would be.
                    # (`TemporaryDirectory.cleanup()` detaches its finalizer
                    # before removal, so a failed explicit cleanup is not
                    # retried at GC — the OS reaper is what reclaims it.)
                    logger.debug("Failed to clean up temp dir", exc_info=True)
                self._temp_dir = None

            if self._owns_config_dir and self.config_dir is not None:
                import shutil

                try:
                    shutil.rmtree(self.config_dir)
                except OSError:
                    # Warning, not debug: cleanup runs exactly once and is never
                    # retried, and this is a process-owned dir that may hold
                    # session state, so a persistent failure (a real leak) must
                    # be visible to an operator.
                    logger.warning(
                        "Failed to clean up config dir %s",
                        self.config_dir,
                        exc_info=True,
                    )
                self._owns_config_dir = False

    # [해설] 다음 restart 때 적용할 환경 override를 쌓아 둔다(예: 모델 변경 시
    # [해설]   DEEPAGENTS_CODE_SERVER_MODEL).
    def update_env(self, **overrides: str) -> None:
        """Stage env var overrides to apply on the next `restart()`.

        These are applied to `os.environ` immediately before the subprocess
        starts, keeping mutation scoped to the restart call.

        Args:
            **overrides: Key/value env var pairs
                (e.g., `DEEPAGENTS_CODE_SERVER_MODEL="anthropic:claude-sonnet-4-6"`).
        """
        self._env_overrides.update(overrides)

    # [해설][주의] 이후 모든 서버 시작에 적용될 override. 앱 소유
    # [해설]   접두사(DEEPAGENTS_CODE_SERVER_)가 아닌 키는 ValueError로 거부한다.
    def persist_env(self, **overrides: str) -> None:
        """Persist env var overrides for every future subprocess start.

        Args:
            **overrides: Key/value env var pairs that should be passed to all
                future server subprocesses.

        Raises:
            ValueError: If an override is not an app-owned server env var.
        """
        invalid = [key for key in overrides if not key.startswith(SERVER_ENV_PREFIX)]
        if invalid:
            msg = (
                "persistent server env overrides must use the "
                f"{SERVER_ENV_PREFIX!r} prefix"
            )
            raise ValueError(msg)
        self._persistent_env_overrides.update(overrides)

    # [해설][흐름] 재시작: 1) 현재 종료 세대 기록 2) 서버 종료를 스레드에 위임(UI 이벤트
    # [해설]   루프가 얼지 않게, 취소돼도 끝까지 기다림)
    # [해설][흐름] 3) staged override를 환경에 적용하며 _start(실패 시 환경 롤백) 4)
    # [해설]   override 비움. 같은 config_dir를 재사용한다.
    async def restart(self, *, timeout: float = _HEALTH_TIMEOUT) -> None:  # noqa: ASYNC109
        """Restart the server process, reusing the existing config directory.

        Stops the subprocess, then starts a new one. Any env overrides staged
        via `update_env()` are applied within a `_scoped_env_overrides` context
        manager so that failures automatically roll back the environment to the
        last known-good state.

        Args:
            timeout: Max seconds to wait for the server to become healthy.

        Raises:
            asyncio.CancelledError: Either if the restart task is cancelled
                (the blocking subprocess cleanup is awaited to completion
                first), or if a terminal `stop()` bumped the stop generation
                during cleanup — in which case `_start` aborts rather than
                resurrecting the subprocess the terminal stop just tore down.
            RuntimeError: If workspace preparation, process startup, or the
                server health check fails.
        """  # noqa: DOC502  # RuntimeError propagates from _start().
        logger.info("Restarting langgraph dev server")
        async with self._lifecycle_lock:
            with self._state_lock:
                stop_generation = self._stop_generation
            # Offload the synchronous subprocess shutdown (it blocks up to
            # `_SHUTDOWN_TIMEOUT` + SIGKILL grace waiting on `process.wait`) so
            # the caller's event loop — the Textual reactor for `/restart` —
            # keeps processing input instead of freezing the TUI. Shield the
            # thread and await it after cancellation so no later lifecycle call
            # can mutate process state while cleanup is still running.
            stop_task = asyncio.create_task(asyncio.to_thread(self._stop_process))
            try:
                await asyncio.shield(stop_task)
            except asyncio.CancelledError:
                await stop_task
                raise

            with _scoped_env_overrides(self._env_overrides):
                await self._start(
                    timeout=timeout,
                    expected_stop_generation=stop_generation,
                )

            self._env_overrides.clear()

    # [해설] `async with ServerProcess(...)` 지원: 진입 시 start, 종료 시 stop과 보존
    # [해설]   로그 안내.
    async def __aenter__(self) -> Self:
        """Async context manager entry.

        Returns:
            The server process instance.
        """
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        self.stop()
        emit_preserved_log_notices()
