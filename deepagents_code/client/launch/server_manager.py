"""Server lifecycle orchestration for the app.

Provides `start_server_and_get_agent` which handles the full flow of:

1. Building a `ServerConfig` from application arguments
2. Writing config to env vars via `ServerConfig.to_env()`
3. Scaffolding a workspace (langgraph.json, checkpointer, pyproject)
4. Starting the `langgraph dev` server
5. Returning a `RemoteAgent` client

Also provides `server_session`, an async context manager that wraps
server startup and guaranteed cleanup so callers don't need to
duplicate try/finally teardown.
"""

# [해설] 역할: 클라이언트가 서버를 "시작하고 RemoteAgent를 돌려받는" 전체
# [해설]   오케스트레이션. TUI와 헤드리스 모드가 공유하는 관문이다.
# [해설] 프로세스: 모두 클라이언트(TUI/헤드리스) 프로세스에서 실행된다. 서버 프로세스의
# [해설]   코드는 server_graph.py.
# [해설] 진입 심볼: start_server_and_get_agent(핵심), server_session(정리까지 보장하는
# [해설]   컨텍스트 매니저),
# [해설]   PrespawnedServer(앱 로딩과 서버 부팅을 겹치는 선(先)기동).
# [해설][흐름] 클라이언트 -> 서버 순서: 1) ProjectContext 확보 2) MCP 설정 사전 검증 3)
# [해설]   ServerConfig.from_cli_args
# [해설]   4) _apply_server_config(to_env -> os.environ) 5) 임시 작업 디렉터리에
# [해설]   _scaffold_workspace(checkpointer.py,
# [해설]   pyproject.toml, langgraph.json) 6) ServerProcess.start 7)
# [해설]   wait_for_graph_ready 8) RemoteAgent 생성 후 set_workspace.
# [해설] 체크포인트 위치: _write_checkpointer가 DB 경로(sessions.db)를 환경변수로만
# [해설]   전달하는 모듈을 생성한다.
# [해설] 관련 모듈: server.py(프로세스 관리), _server_config.py(직렬화),
# [해설]   server_graph.py(서버측 팩토리), client/remote_client.py.
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from deepagents import FsToolName

    from deepagents_code.client.launch.server import ServerProcess
    from deepagents_code.client.remote_client import RemoteAgent
    from deepagents_code.mcp_tools import MCPSessionManager

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import ServerConfig
from deepagents_code.client.launch.server import (
    _ABANDONED_START_GRACE,
    _EPHEMERAL_PORT,
    emit_preserved_log_notices,
)
from deepagents_code.project_utils import ProjectContext

logger = logging.getLogger(__name__)
_DISTRIBUTION_NAME = "deepagents-code"


# [해설] DEEPAGENTS_CODE_SERVER_<name> 환경변수를 설정하거나(None이면) 삭제한다.
# [해설]   to_env의 "None = 지움" 규약을 구현하는 곳이다.
def _set_or_clear_server_env(name: str, value: str | None) -> None:
    """Set or clear a `DEEPAGENTS_CODE_SERVER_*` environment variable.

    Args:
        name: Suffix after `DEEPAGENTS_CODE_SERVER_`.
        value: String value to set, or `None` to clear the variable.
    """
    key = f"{SERVER_ENV_PREFIX}{name}"
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


# [해설][흐름] ServerConfig.to_env() 결과를 현재 프로세스의 os.environ에 기록한다. 이후
# [해설]   server.py의 _build_server_env가
# [해설]   os.environ을 복사하므로 서버 자식이 이 값들을 상속한다. 설정 전달의 "쓰기"
# [해설]   쪽이며 "읽기"는 ServerConfig.from_env()다.
def _apply_server_config(config: ServerConfig) -> None:
    """Write a `ServerConfig` to `DEEPAGENTS_CODE_SERVER_*` env vars.

    Uses `ServerConfig.to_env()` so that the set of variables and their
    serialization format are defined in one place (the `ServerConfig` dataclass)
    rather than maintained independently here and in the
    reader (`ServerConfig.from_env()`).

    Args:
        config: Fully resolved server configuration.
    """
    for suffix, value in config.to_env().items():
        _set_or_clear_server_env(suffix, value)


# [해설] 사용자의 현재 작업디렉터리와 프로젝트 루트를 캡처한다. cwd를 알 수 없으면
# [해설]   None(이후 단계가 RuntimeError로 막는다).
def _capture_project_context() -> ProjectContext | None:
    """Capture the user's project context for the server subprocess.

    Returns:
        Explicit project context, or `None` when cwd cannot be determined.
    """
    try:
        return ProjectContext.from_user_cwd(Path.cwd())
    except OSError:
        logger.warning("Could not determine working directory for server")
        return None


# ------------------------------------------------------------------
# Workspace scaffolding
# ------------------------------------------------------------------


# [해설][흐름] 서버 작업 디렉터리 채우기: 1) checkpointer.py 2) pyproject.toml 3)
# [해설]   langgraph.json. 재시작 때 임시 디렉터리가 사라졌어도
# [해설]   ServerProcess가 이 함수를 다시 불러 복구한다(scaffold 콜백).
# [해설][주의] graph_ref는 설치된 패키지의 점(.) 경로지만 checkpointer_path는 cwd 기준
# [해설]   상대 경로다. 합치면 안 된다(아래 주석).
def _scaffold_workspace(work_dir: Path) -> None:
    """Prepare the server working directory with all required files.

    Generates the auxiliary files (checkpointer module, `pyproject.toml`,
    `langgraph.json`) that `langgraph dev` needs to boot. The generated
    graph reference imports the installed `deepagents_code` package directly.

    Args:
        work_dir: Temporary directory that will become the server's cwd.
    """
    from deepagents_code.client.launch.server import generate_langgraph_json

    _write_checkpointer(work_dir)
    _write_pyproject(work_dir)

    # `graph_ref` is a dotted import of the installed `deepagents_code` package,
    # but `checkpointer_path` stays cwd-relative: checkpointer.py is generated
    # fresh into work_dir (which `ServerProcess.start()` sets as the subprocess
    # cwd) and is not an importable package module. Don't "unify" these — a
    # dotted ref for the checkpointer would fail to resolve.
    generate_langgraph_json(
        work_dir,
        graph_ref="deepagents_code.server_graph:make_graph",
        checkpointer_path="./checkpointer.py:create_checkpointer",
    )


# [해설][설계] 체크포인터 모듈을 "소스 코드로 생성"한다. DB 경로는 코드에 박지 않고
# [해설]   환경변수(DEEPAGENTS_CODE_SERVER_DB_PATH)로만 전달한다.
# [해설][흐름] 1) sessions.get_db_path()를 os.environ에 기록(= 프로필 state 디렉터리의
# [해설]   sessions.db)
# [해설][흐름] 2) 생성된 create_checkpointer()는 서버 프로세스에서 실행되어
# [해설]   _checkpointer_adapter의 패치를 설치하고
# [해설]    thread_ownership.owned_saver_class(AsyncSqliteSaver 파생)로 같은 DB를 연다.
# [해설]   같은 DB에 workspace.py의 바인딩 테이블도 있다.
# [해설] 아래 f-string 안의 내용은 서버가 읽을 별도 파일(checkpointer.py)의 본문이다(이
# [해설]   모듈 코드가 아니다).
def _write_checkpointer(work_dir: Path) -> None:
    """Write a checkpointer module that reads its DB path from the environment.

    The generated module reads the DB path env var at runtime so the path
    is never baked into generated source. This is consistent with the
    `DEEPAGENTS_CODE_SERVER_*` env-var communication pattern used elsewhere.

    Args:
        work_dir: Server working directory.
    """
    from deepagents_code.sessions import get_db_path

    os.environ[f"{SERVER_ENV_PREFIX}DB_PATH"] = str(get_db_path())

    db_path_var = f"{SERVER_ENV_PREFIX}DB_PATH"
    content = f'''\
"""Persistent SQLite checkpointer for the LangGraph dev server."""

import os
from contextlib import asynccontextmanager


@asynccontextmanager
async def create_checkpointer():
    """Yield an AsyncSqliteSaver connected to the app's sessions DB.

    The database path is read from the `{db_path_var}` env var
    (set by the app before server startup) rather than hard-coded, so
    the checkpointer module works without code generation.
    """
    from pathlib import Path
    from deepagents_code._checkpointer_adapter import (
        install_delta_history_forwarding,
    )
    from deepagents_code.thread_ownership import owned_saver_class

    db_path = os.environ.get("{db_path_var}")
    if not db_path:
        raise RuntimeError(
            "{db_path_var} not set. The app must set this "
            "env var before server startup."
        )
    install_delta_history_forwarding()
    saver_class = owned_saver_class(db_path=Path(db_path))
    async with saver_class.from_conn_string(db_path) as saver:
        yield saver
'''
    (work_dir / "checkpointer.py").write_text(content)


# [해설] 서버 작업 디렉터리의 최소 pyproject.toml. `langgraph dev`가 의존성을 설치할 수
# [해설]   있도록 deepagents-code 자체를 의존성으로 건다.
# [해설] requires-python이 3.14 고정이라 이 코드베이스가 Python 3.14 전용임을 알 수
# [해설]   있다.
def _write_pyproject(work_dir: Path) -> None:
    """Write a minimal pyproject.toml for the server working directory.

    The `langgraph dev` server needs to install the project dependencies.
    We point it at the app package which transitively pulls in the SDK.

    Args:
        work_dir: Server working directory.
    """
    content = f"""[project]
name = "deepagents-server-runtime"
version = "0.0.1"
requires-python = ">=3.14,<3.15"
dependencies = [
    "{_runtime_package_dependency()}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (work_dir / "pyproject.toml").write_text(content)


# [해설] deepagents_code 패키지의 상위 디렉터리(소스 체크아웃이면 프로젝트 루트,
# [해설]   wheel이면 site-packages). 알 수 없으면 None.
def _default_package_project_root() -> Path | None:
    """Return the project root that contains the top-level package.

    Returns:
        The directory above `deepagents_code` — the editable project root in
            source checkouts and `site-packages` for installed wheels — or
            `None` when the package location cannot be determined (e.g. a frozen
            or zipimport build where `__file__` is unset). Returning `None`
            (rather than guessing `Path.cwd()`) lets the caller fall back to
            the installed distribution version instead of pointing at whatever
            unrelated project happens to sit in the launch directory.
    """
    import deepagents_code

    # `getattr` with a default: `__file__` is unset on frozen/zipimport builds
    # and namespace packages, so it is not guaranteed to exist at runtime.
    package_init = getattr(deepagents_code, "__file__", None)
    if package_init is None:
        return None
    return Path(package_init).resolve().parent.parent


# [해설][설계] 서버 런타임이 deepagents-code를 어디서 설치할지 결정: 소스 체크아웃이면
# [해설]   로컬 경로 의존성(작업 트리를 그대로 사용),
# [해설] 아니면 설치된 배포판 버전 고정. 임의의 cwd 프로젝트를 가리키지 않도록 None일 때
# [해설]   추측하지 않는다.
def _runtime_package_dependency(package_root: Path | None = None) -> str:
    """Return the dependency spec for the app package in the server runtime.

    Editable source checkouts can use a local path dependency so the generated
    runtime sees the working tree. Installed wheels cannot: the package parent is
    `site-packages`, which is not an installable project. In that case, depend on
    the installed distribution version instead.

    Args:
        package_root: Optional package project root for tests.

    Returns:
        Requirement string for the generated runtime `pyproject.toml`.
    """
    root = package_root or _default_package_project_root()
    if root is not None and (root / "pyproject.toml").is_file():
        return f"{_DISTRIBUTION_NAME} @ {root.as_uri()}"

    try:
        installed_version = version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _DISTRIBUTION_NAME
    return f"{_DISTRIBUTION_NAME}=={installed_version}"


# ------------------------------------------------------------------
# MCP pre-flight validation
# ------------------------------------------------------------------


# [해설] 서버를 띄우기 전에 부모 프로세스에서 --mcp-config를 검증한다. 서버 안에서
# [해설]   ValueError가 나면 잘린 로그 덩어리로만 보이기 때문이다.
def _preflight_validate_mcp_config(
    *,
    mcp_config_path: str | None,
    no_mcp: bool,
) -> None:
    """Validate the explicit `--mcp-config` path before spawning the server.

    Catches the common failure mode of passing a malformed MCP config: a
    `ValueError` raised inside the server subprocess otherwise surfaces as an
    opaque truncated log dump in `wait_for_server_healthy`. Running the same
    validation in the parent process lets the TUI display a clean, actionable
    message with the offending path and reason.

    Project-level and user-level configs discovered by `resolve_and_load_mcp_tools`
    are not validated here; their errors are already handled leniently via
    `load_mcp_config_with_error` and surface as errored entries in the
    `/mcp` viewer rather than as a fatal startup failure.

    Args:
        mcp_config_path: Explicit path passed via `--mcp-config`, or `None`.
        no_mcp: When `True`, MCP is disabled and validation is skipped.

    Raises:
        MCPConfigError: If the config file is malformed or missing required
            fields. Message includes the offending path for context.
    """
    if no_mcp or not mcp_config_path:
        return

    from deepagents_code.mcp_tools import MCPConfigError, load_mcp_config

    try:
        load_mcp_config(mcp_config_path)
    except MCPConfigError:
        raise
    except FileNotFoundError as exc:
        msg = f"MCP config file not found: {mcp_config_path}"
        raise MCPConfigError(msg) from exc
    except (ValueError, TypeError) as exc:
        # `ValueError` covers `json.JSONDecodeError` (subclass) and the
        # shape/field validators in `_validate_server_config`; `TypeError`
        # covers the wrong-type branches. Bare `RuntimeError` is
        # deliberately NOT caught — it would mask unrelated bugs
        # (recursion, reentrancy, stdlib internals) as config errors.
        msg = f"Invalid MCP config at {mcp_config_path}: {exc}"
        raise MCPConfigError(msg) from exc


# ------------------------------------------------------------------
# Server startup
# ------------------------------------------------------------------


# [해설][주의] 서버 시작의 중심 함수. 반환값은 (RemoteAgent, ServerProcess, None). 인자
# [해설]   대부분은 ServerConfig.from_cli_args로 그대로 전달된다.
# [해설][흐름] 1) 프로젝트 컨텍스트(cwd 인자 또는 현재 디렉터리) 2) MCP 사전 검증 3)
# [해설]   ServerConfig 생성 4) 환경변수로 기록
# [해설][흐름] 5) 임시 디렉터리 + _scaffold_workspace 6) ServerProcess 생성 후 start ->
# [해설]   wait_for_graph_ready("agent")
# [해설][흐름] 7) RemoteAgent 생성, set_workspace로 세션 정책 주장+지문 전달 8)
# [해설]   실패/취소 시 finally에서 서버를 정리(고아 프로세스 방지).
async def start_server_and_get_agent(
    *,
    assistant_id: str,
    model_name: str | None = None,
    summarization_model: str | None = None,
    model_params: dict[str, Any] | None = None,
    cli_max_retries: int | None = None,
    profile_overrides: dict[str, Any] | None = None,
    auto_approve: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    sandbox_snapshot_name: str | None = None,
    sandbox_setup: str | None = None,
    enable_shell: bool = True,
    enable_ask_user: bool = False,
    enable_interpreter: bool | None = None,
    interpreter_ptc: str | list[str] | None = None,
    interpreter_ptc_acknowledge_unsafe: bool = False,
    allow_fs_tools: list[FsToolName] | None = None,
    rubric_model: str | None = None,
    rubric_max_iterations: int | None = None,
    auto_classifier_model: str | None = None,
    recursion_limit: int | None = None,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    trust_project_extensions: bool = False,
    extension_paths: tuple[str, ...] = (),
    interactive: bool = True,
    host: str = "127.0.0.1",
    port: int = _EPHEMERAL_PORT,
    cwd: str | None = None,
    after_spawn: Callable[[], Awaitable[None]] | None = None,
) -> tuple[RemoteAgent, ServerProcess, MCPSessionManager | None]:
    """Start a LangGraph server and return a connected remote agent client.

    Args:
        assistant_id: Agent identifier.
        model_name: Model spec string.
        summarization_model: Model spec used only for context-compaction summaries.
        model_params: Extra model kwargs.
        cli_max_retries: Explicit `--max-retries` value.
        profile_overrides: Model profile metadata overrides.
        auto_approve: Auto-approve all tools.
        interrupt_shell_only: Validate shell commands via middleware instead of HITL.
        shell_allow_list: Restrictive shell allow-list for `ShellAllowListMiddleware`.
        sandbox_type: Sandbox type.
        sandbox_id: Existing sandbox ID to reuse.
        sandbox_snapshot_name: Snapshot (langsmith) or blueprint (runloop) name.
        sandbox_setup: Path to setup script for the sandbox.
        enable_shell: Enable shell execution tools.
        enable_ask_user: Enable ask_user tool.
        enable_interpreter: Enable the JS interpreter (`js_eval`) middleware on
            the main agent. `None` uses the sandbox-aware default.
        interpreter_ptc: Invocation-scoped PTC allowlist override.
        interpreter_ptc_acknowledge_unsafe: Explicit acknowledgement for
            `interpreter_ptc="all"` outside of `auto_approve`.
        allow_fs_tools: Allowlist for `FilesystemMiddleware`'s `tools` param.

            `None` leaves the SDK default (all tools).
        rubric_model: Grader model spec; `None` reuses the main model.
        rubric_max_iterations: Explicit grader iterations per rubric attempt;
            `None` uses the SDK default.
        auto_classifier_model: Auto classifier model spec; `None` resolves from
            env / `config.toml` and then reuses the main model.
        recursion_limit: Explicit main-agent `recursion_limit`; `None` resolves
            from runtime configuration at agent-build time.
        mcp_config_path: Path to MCP config.
        no_mcp: Disable MCP.
        trust_project_mcp: Trust project MCP servers.
        trust_project_extensions: Allow project extension execution.
        extension_paths: Explicit one-run extension files or directories.
        interactive: Whether the agent is interactive.
        host: Server host.
        port: Server port. Defaults to `_EPHEMERAL_PORT` (0), letting the server
            pick a free ephemeral port instead of the well-known `langgraph dev`
            port 2024.
        cwd: Explicit project workspace to bind to new threads.
        after_spawn: Awaited once the server subprocess is spawned, before
            the health check. See `ServerProcess.start`.

    Returns:
        Tuple of `(remote_agent, server_process, mcp_session_manager)`.
            The `mcp_session_manager` is currently always `None` (MCP lifecycle
            is handled server-side).

    Raises:
        MCPConfigError: The explicit `--mcp-config` path is malformed,
            missing, or references contradictory transport fields. Raised
            from the pre-flight validator before any subprocess is spawned.
        RuntimeError: If no explicit workspace can be resolved.
    """  # noqa: DOC502 - `_preflight_validate_mcp_config()` raises indirectly
    from deepagents_code.client.launch.server import ServerProcess

    # [해설][흐름] 1) ProjectContext: 인자 cwd가 있으면 그 경로 기준, 없으면 현재
    # [해설]   디렉터리를 캡처.
    project_context = (
        ProjectContext.from_user_cwd(Path(cwd))
        if cwd is not None
        else _capture_project_context()
    )

    _preflight_validate_mcp_config(
        mcp_config_path=mcp_config_path,
        no_mcp=no_mcp,
    )

    config = ServerConfig.from_cli_args(
        project_context=project_context,
        model_name=model_name,
        summarization_model=summarization_model,
        model_params=model_params,
        cli_max_retries=cli_max_retries,
        profile_overrides=profile_overrides,
        assistant_id=assistant_id,
        auto_approve=auto_approve,
        interrupt_shell_only=interrupt_shell_only,
        shell_allow_list=shell_allow_list,
        sandbox_type=sandbox_type,
        sandbox_id=sandbox_id,
        sandbox_snapshot_name=sandbox_snapshot_name,
        sandbox_setup=sandbox_setup,
        enable_shell=enable_shell,
        enable_ask_user=enable_ask_user,
        enable_interpreter=enable_interpreter,
        interpreter_ptc=interpreter_ptc,
        interpreter_ptc_acknowledge_unsafe=interpreter_ptc_acknowledge_unsafe,
        allow_fs_tools=allow_fs_tools,
        rubric_model=rubric_model,
        rubric_max_iterations=rubric_max_iterations,
        auto_classifier_model=auto_classifier_model,
        recursion_limit=recursion_limit,
        mcp_config_path=mcp_config_path,
        no_mcp=no_mcp,
        trust_project_mcp=trust_project_mcp,
        interactive=interactive,
        trust_project_extensions=trust_project_extensions,
        extension_paths=extension_paths,
    )
    # [해설][흐름] 4) 직렬화 결과를 os.environ에 기록 -> 5) 임시 작업 디렉터리 생성 및
    # [해설]   채우기(이 디렉터리는 서버가 cwd로 사용, 서버가 stop할 때 삭제됨).
    _apply_server_config(config)

    work_dir = Path(tempfile.mkdtemp(prefix="deepagents_server_"))
    _scaffold_workspace(work_dir)

    # [해설][흐름] 6) ServerProcess에 scaffold 콜백을 넘겨 두면 임시 디렉터리가 사라져도
    # [해설]   restart가 복구한다.
    server = ServerProcess(
        host=host,
        port=port,
        config_dir=work_dir,
        owns_config_dir=True,
        scaffold=_scaffold_workspace,
    )
    # [해설][흐름] 7) 서버 시작 -> 그래프 준비 확인 -> 클라이언트 객체 생성. RemoteAgent
    # [해설]   import는 spawn 이후로 미뤄 첫 화면 표시를 앞당긴다.
    started = False
    try:
        await server.start(after_spawn=after_spawn)
        await server.wait_for_graph_ready("agent")
        # Import after the spawn: `PrespawnedServer` runs this before first
        # paint, and only the spawn needs to happen that early.
        from deepagents_code.client.remote_client import RemoteAgent

        agent = RemoteAgent(
            url=server.url,
            graph_name="agent",
        )
        if project_context is None:
            msg = "A workspace is required to start the remote agent."
            raise RuntimeError(msg)
        agent.set_workspace(
            str(project_context.user_cwd),
            config.to_session_workspace_claim(),
            config_fingerprint=config.session_workspace_fingerprint(),
        )
        started = True
        return agent, server, None
    # [해설][주의] 성공하지 못하고 빠져나갈 때(예외/취소)만 서버를 정리한다.
    # [해설]   취소(CancelledError)는 BaseException이라 finally를 써야 한다.
    # [해설] 샌드박스를 쓰는 설정이면 정상 종료 유예 시간을 길게 둔다(서버가 샌드박스를
    # [해설]   삭제할 시간이 필요).
    finally:
        if not started:
            # Startup failed or was cancelled before the server was handed off
            # to the caller (which records the reference only on success). If
            # `start()` itself failed it already reaped its own subprocess, so
            # this `stop()` is then an idempotent no-op; this cleanup is the sole
            # reaper only when `start()` succeeded but `wait_for_graph_ready()`
            # (or `RemoteAgent()`) failed afterward. A `finally` rather than
            # `except Exception` is deliberate: `asyncio.CancelledError` is a
            # `BaseException`, so an `except Exception` guard would skip cleanup
            # and orphan the process. The inner guard stops a `stop()` error
            # from masking the exception already propagating.
            #
            # Offload `stop()`: it blocks for up to `_SHUTDOWN_TIMEOUT` plus the
            # SIGKILL grace while the server finishes its in-flight graph build.
            # Quitting the TUI during startup cancels this coroutine on Textual's
            # event loop, so a blocking call here freezes the screen until the
            # server exits. If a second cancellation interrupts the await, the
            # thread keeps running and `asyncio.run` joins it before the process
            # exits, so the server is still reaped.
            #
            # The server never reached the caller, so it holds no session state
            # and gets only a short graceful window. A sandbox is the exception:
            # the graph build may be creating one, and the server deletes it only
            # when it exits gracefully.
            #
            # `stop()` only *queues* any debug-preserved log path; it is not
            # announced here. This helper is awaited by callers that still own
            # the terminal (the initial TUI startup worker and the in-session
            # cwd-switch flow), where a stderr print would be swallowed by the
            # alternate screen. The queue is process-global, so the outer
            # terminal teardown (`run_textual_app` / `server_session` finally)
            # drains this failed server's path once the terminal is restored.
            grace = None if config.sandbox_type else _ABANDONED_START_GRACE
            try:
                await asyncio.to_thread(server.stop, grace=grace)
            except Exception:
                logger.exception(
                    "Error stopping server during startup cleanup",
                )


# [해설][설계] 앱이 화면을 그리는 동안 서버 프로세스를 먼저 띄워 두는 선기동 래퍼. 구성:
# [해설]   start_server_and_get_agent를 태스크로 실행하되
# [해설] after_spawn 훅(_hold)에서 멈춰 있다가, 앱이 adopt()로 풀어 줄 때 헬스체크를
# [해설]   진행한다. adopt되지 않으면 aclose가 서버를 정리한다.
class PrespawnedServer:
    """A `start_server_and_get_agent` call that spawns before the app loads.

    The subprocess boots while the caller imports and paints the app. The
    start then pauses until `adopt` releases it, so the health check and its
    imports run only after the app hands over its own `after_spawn` gate.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Schedule the server start.

        Args:
            **kwargs: Keyword arguments for `start_server_and_get_agent`,
                except `after_spawn`.
        """
        self._spawned = asyncio.Event()
        self._release = asyncio.Event()
        self._adopted = False
        self._task = asyncio.create_task(
            start_server_and_get_agent(**kwargs, after_spawn=self._hold)
        )

    async def _hold(self) -> None:
        self._spawned.set()
        await self._release.wait()

    async def wait_spawned(self) -> None:
        """Return once the subprocess is spawned or the start has failed."""
        spawned = asyncio.ensure_future(self._spawned.wait())
        try:
            await asyncio.wait({self._task, spawned}, return_when="FIRST_COMPLETED")
        finally:
            spawned.cancel()

    async def adopt(
        self, *, after_spawn: Callable[[], Awaitable[None]]
    ) -> tuple[RemoteAgent, ServerProcess, MCPSessionManager | None]:
        """Release the paused start and take ownership of the server.

        Args:
            after_spawn: Awaited before the health check, as in
                `start_server_and_get_agent`.

        Returns:
            The `start_server_and_get_agent` result.
        """
        await after_spawn()
        self._release.set()
        result = await self._task
        self._adopted = True
        return result

    async def aclose(self) -> None:
        """Cancel and reap a start that no caller adopted.

        Safe to call after `adopt`; an adopted server stays with its owner.
        """
        if not self._task.done():
            self._task.cancel()
        await asyncio.wait({self._task})
        if self._task.cancelled() or self._task.exception() is not None:
            return
        if not self._adopted:
            _, server, _ = self._task.result()
            await asyncio.to_thread(server.stop, grace=_ABANDONED_START_GRACE)


# ------------------------------------------------------------------
# Session context manager
# ------------------------------------------------------------------


# [해설] 서버 시작 + 종료 보장을 묶은 컨텍스트 매니저. 헤드리스
# [해설]   모드(client/non_interactive.py)가 사용한다.
# [해설][흐름] 진입 시 start_server_and_get_agent, 종료 시 MCP 세션 정리(현재는 항상
# [해설]   None) -> server.stop -> 보존 로그 안내를 항상 수행.
@asynccontextmanager
async def server_session(
    *,
    assistant_id: str,
    model_name: str | None = None,
    summarization_model: str | None = None,
    model_params: dict[str, Any] | None = None,
    cli_max_retries: int | None = None,
    profile_overrides: dict[str, Any] | None = None,
    auto_approve: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    sandbox_snapshot_name: str | None = None,
    sandbox_setup: str | None = None,
    enable_shell: bool = True,
    enable_ask_user: bool = False,
    enable_interpreter: bool | None = None,
    interpreter_ptc: str | list[str] | None = None,
    interpreter_ptc_acknowledge_unsafe: bool = False,
    allow_fs_tools: list[FsToolName] | None = None,
    rubric_model: str | None = None,
    rubric_max_iterations: int | None = None,
    auto_classifier_model: str | None = None,
    recursion_limit: int | None = None,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    trust_project_extensions: bool = False,
    extension_paths: tuple[str, ...] = (),
    interactive: bool = True,
    host: str = "127.0.0.1",
    port: int = _EPHEMERAL_PORT,
    cwd: str | None = None,
) -> AsyncIterator[tuple[RemoteAgent, ServerProcess]]:
    """Async context manager that starts a server and guarantees cleanup.

    Wraps `start_server_and_get_agent` so callers don't need to duplicate the
    try/finally pattern for stopping the server.

    Args:
        assistant_id: Agent identifier.
        model_name: Model spec string.
        summarization_model: Model spec used only for context-compaction summaries.
        model_params: Extra model kwargs.
        cli_max_retries: Explicit `--max-retries` value.
        profile_overrides: Model profile metadata overrides.
        auto_approve: Auto-approve all tools.
        interrupt_shell_only: Validate shell commands via middleware instead of HITL.
        shell_allow_list: Restrictive shell allow-list for `ShellAllowListMiddleware`.
        sandbox_type: Sandbox type.
        sandbox_id: Existing sandbox ID to reuse.
        sandbox_snapshot_name: Snapshot (langsmith) or blueprint (runloop) name.
        sandbox_setup: Path to setup script for the sandbox.
        enable_shell: Enable shell execution tools.
        enable_ask_user: Enable ask_user tool.
        enable_interpreter: Enable the JS interpreter (`js_eval`) middleware on
            the main agent. `None` uses the sandbox-aware default.
        interpreter_ptc: Invocation-scoped PTC allowlist override.
        interpreter_ptc_acknowledge_unsafe: Explicit acknowledgement for
            `interpreter_ptc="all"` outside of `auto_approve`.
        allow_fs_tools: Allowlist for `FilesystemMiddleware`'s `tools` param.

            `None` leaves the SDK default (all tools).
        rubric_model: Grader model spec; `None` reuses the main model.
        rubric_max_iterations: Explicit grader iterations per rubric attempt;
            `None` uses the SDK default.
        auto_classifier_model: Auto classifier model spec; `None` resolves from
            env / `config.toml` and then reuses the main model.
        recursion_limit: Explicit main-agent `recursion_limit`; `None` resolves
            from runtime configuration at agent-build time.
        mcp_config_path: Path to MCP config.
        no_mcp: Disable MCP.
        trust_project_mcp: Trust project MCP servers.
        trust_project_extensions: Allow project extension execution.
        extension_paths: Explicit one-run extension files or directories.
        interactive: Whether the agent is interactive.
        host: Server host.
        port: Server port. Defaults to `_EPHEMERAL_PORT` (0), letting the server
            pick a free ephemeral port instead of the well-known `langgraph dev`
            port 2024.
        cwd: Explicit project workspace to bind to new threads.

    Yields:
        Tuple of `(remote_agent, server_process)`.
    """
    server_proc: ServerProcess | None = None
    mcp_session_manager: MCPSessionManager | None = None
    try:
        agent, server_proc, mcp_session_manager = await start_server_and_get_agent(
            assistant_id=assistant_id,
            model_name=model_name,
            summarization_model=summarization_model,
            model_params=model_params,
            cli_max_retries=cli_max_retries,
            profile_overrides=profile_overrides,
            auto_approve=auto_approve,
            interrupt_shell_only=interrupt_shell_only,
            shell_allow_list=shell_allow_list,
            sandbox_type=sandbox_type,
            sandbox_id=sandbox_id,
            sandbox_snapshot_name=sandbox_snapshot_name,
            sandbox_setup=sandbox_setup,
            enable_shell=enable_shell,
            enable_ask_user=enable_ask_user,
            enable_interpreter=enable_interpreter,
            interpreter_ptc=interpreter_ptc,
            interpreter_ptc_acknowledge_unsafe=interpreter_ptc_acknowledge_unsafe,
            allow_fs_tools=allow_fs_tools,
            rubric_model=rubric_model,
            rubric_max_iterations=rubric_max_iterations,
            auto_classifier_model=auto_classifier_model,
            recursion_limit=recursion_limit,
            mcp_config_path=mcp_config_path,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
            trust_project_extensions=trust_project_extensions,
            extension_paths=extension_paths,
            interactive=interactive,
            host=host,
            port=port,
            cwd=cwd,
        )
        yield agent, server_proc
    finally:
        if mcp_session_manager is not None:
            try:
                await mcp_session_manager.cleanup()
            except Exception:
                logger.warning("MCP session cleanup failed", exc_info=True)
        if server_proc is not None:
            server_proc.stop()
        # Drain unconditionally: when startup fails inside
        # `start_server_and_get_agent`, `server_proc` is never assigned here,
        # yet the failed server may have queued a debug-preserved log path.
        # This runs with no TUI active, so the notice reaches the user.
        emit_preserved_log_notices()
