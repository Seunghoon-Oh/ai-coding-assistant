"""Server-side graph entry point for `langgraph dev`.

This module is referenced by the generated `langgraph.json` and exposes a graph
factory that the LangGraph server can load and serve.

The graph is created by `make_graph()`, which reads configuration from
`ServerConfig.from_env()` — the same dataclass the CLI uses to *write* the
configuration via `ServerConfig.to_env()`. This shared schema ensures the two
sides stay in sync.
"""

# [해설] 역할: LangGraph 서버 프로세스 안에서 로드되는 "서버측 그래프 팩토리".
# [해설]   에이전트를 실제로 조립하는 곳이다.
# [해설] 연결 고리: 클라이언트가 만든 langgraph.json(client/launch/server.py의
# [해설]   generate_langgraph_json)의
# [해설]   graphs.agent 값이 "deepagents_code.server_graph:make_graph"이고, `langgraph
# [해설]   dev`가 이 모듈을 import해 make_graph를 부른다.
# [해설] 설정 수신: ServerConfig.from_env()로 환경변수에서 복원(_server_config.py).
# [해설]   쓰기는 클라이언트의 to_env()다.
# [해설] 체크포인트: 이 파일은 checkpointer를 넘기지 않는다. 저장기는 langgraph.json의
# [해설]   checkpointer.path가 가리키는
# [해설]   생성된 checkpointer.py가 제공하고, 파일은
# [해설]   sessions.db(SQLite)다(_checkpointer_adapter.py, server_manager.py 참고).
# [해설] 진입 심볼: make_graph(LangGraph가 호출) -> get_server_runtime /
# [해설]   _workspace_runtime -> _make_graphs ->
# [해설]   _make_graphs_in_environment -> agent.py의 create_cli_agent -> SDK의
# [해설]   create_deep_agent(Day09/deepagents/libs/deepagents).
# [해설] 캐시: 런타임(에이전트+백엔드+오프로드)은 프로세스당/작업공간당 한 번만
# [해설]   만든다(MCP 탐색·샌드박스가 한 번만 일어나야 하므로).
# [해설] 오류 계약: 기동 실패는 stderr에 마커를 찍고 sys.exit(1) ->
# [해설]   부모(client/launch/server.py)가 로그에서 마커를 읽어 TUI에 표시한다.
from __future__ import annotations

import asyncio
import atexit
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

# Imported at runtime rather than under TYPE_CHECKING: the LangGraph server
# classifies `make_graph` by resolving its annotations with
# `typing.get_type_hints` at graph-load time. A name that only type checkers
# can see fails to resolve, and the server then refuses to load the graph.
from langgraph_sdk.runtime import ServerRuntime as LangGraphServerRuntime  # noqa: TC002

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code._server_config import ServerConfig
from deepagents_code._startup_error import (
    STARTUP_ERROR_MARKER as _STARTUP_ERROR_MARKER,
    emit_startup_failure,
)
from deepagents_code.configuration.interpreter import InterpreterConfig
from deepagents_code.configuration.resolver import get_config_resolver
from deepagents_code.model_metadata import ModelMetadata
from deepagents_code.project_utils import ProjectContext, get_server_project_context
from deepagents_code.workspace import (
    PROJECT_POLICY_DRIFT_REASON,
    SERVER_CONFIG_DRIFT_REASON,
    WorkspaceConflictError,
    canonical_fingerprint,
    drifted_project_fields,
    get_snapshot_for_binding,
    resolve_workspace,
)
from deepagents_code.workspace_diagnostics import (
    WorkspaceDiagnostics,
    diff_snapshots,
    snapshot_for_payload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from contextlib import AbstractContextManager

    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.protocol import SandboxBackendProtocol

    EnvironmentContext = Callable[
        [Mapping[str, str] | None], AbstractContextManager[None]
    ]

    from deepagents_code.config import CredentialsSnapshot
    from deepagents_code.extensions.registry import ExtensionRegistry
    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.offload_middleware import OffloadOperation
    from deepagents_code.workspace import WorkspaceBinding

logger = logging.getLogger(__name__)

# [해설][설계] 프로세스 수명 동안 유지되는 모듈 전역: 샌드박스 컨텍스트/백엔드, MCP 세션
# [해설]   관리자, LangSmith 트레이싱 설정 고정값.
# [해설] 서버 프로세스 하나는 샌드박스와 트레이싱 설정을 하나만 가질 수 있어서 전역으로
# [해설]   둔다.
_sandbox_cm: Any = None
_sandbox_backend: Any = None
_mcp_session_manager: Any = None
_server_tracing_settings: tuple[dict[str, str | None], bool] | None = None
_server_tracing_initialized = False


def _close_sandbox(context: AbstractContextManager[Any]) -> None:
    context.__exit__(None, None, None)


# [해설] 샌드박스 생성(블로킹)을 스레드에서 실행하되, 취소돼도 이미 만들어진 샌드박스는
# [해설]   닫아 누수를 막는다(asyncio.shield 사용).
async def _open_sandbox(
    create: Callable[[], AbstractContextManager[Any]],
) -> tuple[AbstractContextManager[Any], Any]:
    def _enter() -> tuple[AbstractContextManager[Any], Any]:
        context = create()
        return context, context.__enter__()  # noqa: PLC2801

    task = asyncio.create_task(asyncio.to_thread(_enter))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            context, _ = await asyncio.shield(task)
        except BaseException:  # Preserve the caller's cancellation
            logger.debug(
                "Sandbox startup did not complete after cancellation", exc_info=True
            )
        else:
            await asyncio.to_thread(_close_sandbox, context)
        raise


# [해설] 트레이싱 설정이 이 서버가 이미 고정한 값과 호환되는지만 확인한다(상태를 바꾸지
# [해설]   않음). 다르면 WorkspaceConflictError.
def _validate_server_tracing(
    environ: Mapping[str, str], *, redact: bool
) -> tuple[dict[str, str | None], bool]:
    """Check tracing compatibility without reserving or configuring the process.

    Returns:
        Compatible tracing settings.
    """
    from deepagents_code.config import _tracing_environment_values

    settings = (_tracing_environment_values(environ), redact)
    if _server_tracing_settings is not None and settings != _server_tracing_settings:
        reason = (
            "its LangSmith tracing settings differ from this server's; "
            "start a separate server for this workspace"
        )
        conflict = WorkspaceConflictError.from_reason(reason)
        raise conflict
    return settings


# [해설][설계] LangSmith 트레이싱/비밀 가림 설정을 서버 수명 동안 한 번만 고정한다.
# [해설]   LangSmith의 환경 캐시와 기본 클라이언트가
# [해설] 프로세스 전역이라, 작업공간마다 다른 설정을 허용하면 이미 돌고 있는 런타임의
# [해설]   추적 경로가 바뀌기 때문이다.
def _configure_server_tracing(environ: Mapping[str, str], *, redact: bool) -> None:
    """Pin tracing for the server lifetime before any runtime can execute.

    LangSmith uses process-wide env caches and a default client. Replacing
    them, even under a build lock, reroutes cached and concurrently executing
    runtimes. Only workspaces with matching tracing settings can share this
    process. Keep the reservation across failed builds and cache eviction.

    Called on the server loop with no suspension between claim and setup.
    """
    from deepagents_code._langsmith_runtime import install_untraced_inline_override
    from deepagents_code.config import (
        configure_langsmith_secret_redaction,
        reconcile_tracing_environment,
    )

    global _server_tracing_settings, _server_tracing_initialized  # noqa: PLW0603  # process-lifetime policy
    _server_tracing_settings = _validate_server_tracing(environ, redact=redact)
    if not _server_tracing_initialized:
        reconcile_tracing_environment(environ)
        # Keep redaction on the server task: its fail-closed disable must
        # reach this task's LangSmith ContextVar, not a worker's copied context.
        configure_langsmith_secret_redaction()
        install_untraced_inline_override()
        _server_tracing_initialized = True


# [해설] 시작 오류를 stderr에 두 번 쓴다(사람용 / STARTUP_ERROR_MARKER 접두사 기계용).
# [해설]   부모 프로세스가 후자를 파싱한다
# [해설] (client/launch/server.py의 _extract_startup_error_marker).
def _print_startup_error(message: str) -> None:
    """Print a startup error for both humans and the parent app process.

    Args:
        message: Concise startup failure to surface in the parent process.
    """
    print(message, file=sys.stderr)  # noqa: T201  # stderr fallback for logs
    print(  # noqa: T201  # machine-readable marker consumed by server.py
        f"{_STARTUP_ERROR_MARKER}{message}",
        file=sys.stderr,
    )


# [해설] 프로세스 전체에서 하나뿐인 MCP 세션 관리자(지연 생성 싱글턴). 세션은 서버
# [해설]   이벤트 루프에 묶인다.
def _get_mcp_session_manager() -> Any:  # noqa: ANN401
    """Return the process-wide MCP session manager singleton.

    Sessions are bound to the langgraph dev server's event loop. Cleanup
    therefore belongs to that loop's normal shutdown path, not `atexit` —
    an atexit handler runs after the loop is already closed and cannot
    await `AsyncExitStack.aclose()` safely. Subprocess handles held by
    stdio transports are released when the Python process exits.
    """
    global _mcp_session_manager  # noqa: PLW0603

    if _mcp_session_manager is None:
        from deepagents_code.mcp_tools import MCPSessionManager

        _mcp_session_manager = MCPSessionManager()

    return _mcp_session_manager


# [해설][흐름] 서버가 쓸 도구 목록 조립: 1) 기본 도구 fetch_url, get_current_thread_id
# [해설]   2) Tavily 키가 있으면 웹 검색 도구 추가
# [해설][흐름] 3) no_mcp가 아니면 플러그인 MCP 설정 탐색(스레드로 오프로드) 후
# [해설]   resolve_and_load_mcp_tools로 MCP 도구 로드.
# [해설] 반환값 4개: 전체 도구, MCP 서버 정보, MCP 도구, "읽기 전용" 내장 도구(기준
# [해설]   작성·채점용으로 노출해도 안전한 것).
# [해설][주의] LangGraph가 팩토리를 이미 실행 중인 이벤트 루프에서 부르므로
# [해설]   asyncio.run이 아니라 await로 탐색해야 한다.
async def _build_tools(
    config: ServerConfig,
    project_context: ProjectContext | None,
    *,
    tavily_api_key: str | None,
) -> tuple[list[Any], list[Any] | None, list[Any], list[Any]]:
    """Assemble the tool list based on server config.

    Loads built-in tools (conditionally including web search when Tavily is
    available) and MCP tools when enabled.

    MCP discovery is awaited on the server's event loop: LangGraph invokes this
    async factory on its running loop, so discovery must use `await` rather than
    `asyncio.run` (which raises inside a running loop). `stateless=True` ensures
    discovery only uses throwaway sessions, while the shared runtime session
    manager binds real sessions lazily inside the server loop on first tool
    invocation. MCP adapter imports are warmed in a worker thread inside
    `_load_tools_from_config` (only when active servers exist) because first
    import can perform blocking package-resource scans.

    Args:
        config: Deserialized server configuration.
        project_context: Resolved project context for MCP discovery.
        tavily_api_key: Workspace Tavily key, or `None` when the workspace
            configures none. An empty string still binds the tool, which then
            reports the key as unconfigured.

    Returns:
        Tuple of `(tools, mcp_server_info, mcp_tools, read_only_builtins)`. The
        last element is the exact built-in tool objects that are safe to expose
        to criteria drafting and rubric grading; read-only-ness is known here,
        at construction, so no consumer has to re-derive it.

    Raises:
        FileNotFoundError: If the MCP config file is not found.
        RuntimeError: If MCP tool loading fails.
    """
    from deepagents_code.tools import (
        create_web_search_tool,
        fetch_url,
        get_current_thread_id,
    )

    tools: list[Any] = [fetch_url, get_current_thread_id]
    read_only_builtins: list[Any] = [fetch_url]
    if tavily_api_key is not None:
        search_tool = create_web_search_tool(tavily_api_key)
        tools.append(search_tool)
        read_only_builtins.append(search_tool)

    mcp_server_info: list[Any] | None = None
    mcp_tools: list[Any] = []
    if not config.no_mcp:
        from deepagents_code.mcp_tools import resolve_and_load_mcp_tools
        from deepagents_code.plugins.adapters.mcp import discover_plugin_mcp_configs

        project_dir = (
            project_context.project_root or project_context.user_cwd
            if project_context is not None
            else None
        )
        # Offload plugin discovery: it does blocking disk IO (`os.mkdir` for
        # per-plugin data dirs, plus state/manifest reads) that `blockbuster`
        # rejects on the server event loop.
        plugin_mcp_configs = await asyncio.to_thread(
            discover_plugin_mcp_configs, project_dir=project_dir
        )
        try:
            mcp_tools, _, mcp_server_info = await resolve_and_load_mcp_tools(
                explicit_config_path=config.mcp_config_path,
                no_mcp=config.no_mcp,
                trust_project_mcp=config.trust_project_mcp,
                project_context=project_context,
                additional_configs=plugin_mcp_configs,
                stateless=True,
                session_manager=_get_mcp_session_manager(),
            )
        except FileNotFoundError:
            logger.exception("MCP config file not found: %s", config.mcp_config_path)
            raise
        except RuntimeError:
            logger.exception(
                "Failed to load MCP tools (config: %s)", config.mcp_config_path
            )
            raise

        tools.extend(mcp_tools)
        if mcp_tools:
            logger.info("Loaded %d MCP tool(s)", len(mcp_tools))

    return tools, mcp_server_info, mcp_tools, read_only_builtins


# [해설] 기준(criteria) 초안/루브릭 채점에 제공할 "읽기 전용" 외부 도구만 골라낸다.
# [해설]   내장은 _build_tools가 표시한 것,
# [해설] MCP 도구는 어노테이션이 명시적으로 읽기 전용일 때만 포함한다.
def _criteria_context_tools(
    tools: list[Any],
    mcp_tools: list[Any],
    read_only_builtins: list[Any],
) -> list[Any]:
    """Select read-only external tools for criteria drafting and rubric grading.

    Args:
        tools: Main agent tools in execution order.
        mcp_tools: Exact tool objects returned by MCP discovery.
        read_only_builtins: Built-in tool objects `_build_tools` created and
            marked read-only.

    Returns:
        External context tools available to criteria generation and grading.
        MCP tools are included only when their protocol annotations explicitly
        declare them read-only.
    """
    allowed_ids = {id(tool) for tool in read_only_builtins}
    allowed_ids.update(
        id(tool) for tool in mcp_tools if _mcp_tool_is_explicitly_read_only(tool)
    )
    return [tool for tool in tools if id(tool) in allowed_ids]


# [해설][설계] 모호하거나 없는 어노테이션은 읽기 전용이 아닌 것으로 본다(fail closed).
# [해설]   판정은 auto_mode.py에 위임한다.
def _mcp_tool_is_explicitly_read_only(tool: Any) -> bool:  # noqa: ANN401
    """Return whether a wrapped MCP tool is unambiguously read-only.

    MCP `ToolAnnotations.readOnlyHint` is serialized by the installed adapter
    into the LangChain tool's metadata as the camel-case `readOnlyHint` key.
    Require the literal boolean `True` and reject a contradictory destructive
    hint so absent, malformed, or ambiguous annotations fail closed.

    Returns:
        `True` only for an explicitly and consistently read-only MCP tool.
    """
    from deepagents_code.auto_mode import mcp_tool_is_coherently_read_only

    return mcp_tool_is_coherently_read_only(tool)


# [해설] 프로세스(또는 작업공간)당 하나인 서버 런타임 묶음. NamedTuple이라 필드가
# [해설]   이름으로만 접근되어 순서 착오를 막는다.
# [해설] agent = 컴파일된 그래프, backend/offload = 에이전트와 같은 백엔드(오프로드 HTTP
# [해설]   라우트가 같은 것을 써야 함),
# [해설] 나머지는 클라이언트(TUI)가 받아 쓰는 메타데이터(MCP 정보, 모델 정보, 작업공간
# [해설]   환경).
class ServerRuntime(NamedTuple):
    """The one-per-process result with named slots to prevent transposition."""

    agent: Any
    """Compiled LangGraph agent graph served as `agent`."""

    backend: CompositeBackend
    """Composite backend the agent and its operations were built with."""

    offload: OffloadOperation
    """Server-owned thread offload operation bound to `backend`."""

    mcp_server_info: list[MCPServerInfo] | None = None
    """Workspace-scoped MCP metadata for the interactive client."""

    model_metadata: ModelMetadata | None = None
    """Resolved startup model properties for the interactive client."""

    model_environment: Mapping[str, str] | None = None
    """Immutable workspace environment shared by metadata resolution and inference."""


# [해설][흐름] 1) 설정 확보: 인자로 받은 override가 없으면 ServerConfig.from_env()로
# [해설]   복원 2) 작업공간 경로 결정
# [해설][흐름] 3) 환경 스냅샷(.env 탐색 등 블로킹 IO)을 스레드로 오프로드 4) 그 환경을
# [해설]   use_environment로 활성화한 채
# [해설]    트레이싱 고정(_configure_server_tracing) 후 _make_graphs_in_environment
# [해설]   호출.
async def _make_graphs(
    *,
    config_override: ServerConfig | None = None,
    project_context_override: ProjectContext | None = None,
    sandbox_backend_override: SandboxBackendProtocol | None = None,
) -> ServerRuntime:
    """Create the agent graph and the backend carrying its shared resources.

    Reads `DEEPAGENTS_CODE_SERVER_*` env vars via `ServerConfig.from_env()`
    (the inverse of `ServerConfig.to_env()` used by the app process), resolves a
    model, assembles tools, and compiles the agent graph.

    Returns:
        The agent graph, its configured composite backend, and the server-owned
            offload operation bound to that backend.
    """
    config = config_override or ServerConfig.from_env()
    workspace_path = (
        project_context_override.user_cwd
        if project_context_override is not None
        else Path(config.cwd)
        if config.cwd is not None
        else None
    )

    # Offload the workspace environment snapshot off the event loop. Dotenv
    # discovery walks parent directories (`Path.resolve()`, `is_file()`) and
    # reads up to three files, and `snapshot_from_environment` adds
    # `find_project_root()` -> `Path.cwd()` — all of which `blockbuster`
    # rejects when invoked directly from the server loop (see issue #5043),
    # for the same reason as the offload in `_make_graphs_in_environment`.
    def _resolve_workspace_environment() -> tuple[
        Mapping[str, str], CredentialsSnapshot, EnvironmentContext, bool
    ]:
        from deepagents_code.config import (
            Credentials,
            _ensure_bootstrap,
            _preview_dotenv_environ,
            is_langsmith_redaction_enabled,
            use_environment,
        )

        # Finish the one-time global credential publication before pinning
        # tracing. A later lazy import of `agent` must not overwrite the pin.
        _ensure_bootstrap()
        environ = MappingProxyType(_preview_dotenv_environ(start_path=workspace_path))
        with use_environment(environ):
            redact = is_langsmith_redaction_enabled()
        return (
            environ,
            Credentials.snapshot_from_environment(
                start_path=workspace_path,
                environ=environ,
            ),
            use_environment,
            redact,
        )

    (
        workspace_env,
        workspace_credentials,
        use_environment,
        redact,
    ) = await asyncio.to_thread(_resolve_workspace_environment)

    with use_environment(workspace_env):
        _configure_server_tracing(workspace_env, redact=redact)
        return await _make_graphs_in_environment(
            config=config,
            project_context_override=project_context_override,
            workspace_env=workspace_env,
            workspace_credentials=workspace_credentials,
            sandbox_backend_override=sandbox_backend_override,
        )


# [해설] 실제 에이전트 조립 본체(약 250줄). 작업공간 환경이 활성화된 상태에서 한 번
# [해설]   호출된다.
# [해설][흐름] 1) 프로젝트 컨텍스트·지연 import 2) 모델 생성 3) 도구/MCP 4) 샌드박스 5)
# [해설]   (실험) 확장 로드
# [해설][흐름] 6) 스레드에서 create_cli_agent로 그래프 컴파일 -> ServerRuntime. 블로킹
# [해설]   IO는 거의 다 asyncio.to_thread로 보낸다.
async def _make_graphs_in_environment(
    *,
    config: ServerConfig,
    project_context_override: ProjectContext | None,
    workspace_env: Mapping[str, str],
    workspace_credentials: CredentialsSnapshot,
    sandbox_backend_override: SandboxBackendProtocol | None = None,
) -> ServerRuntime:
    """Build one runtime while its immutable workspace environment is active.

    Returns:
        Agent graph and its workspace-bound resources.
    """

    # Offload cwd/path resolution and the lazy settings bootstrap off the event
    # loop. On Windows, `Path.resolve()` / `Path.cwd()` call `os.getcwd()`, which
    # `blockbuster` rejects when invoked directly from the server loop (see
    # issue #5043). Importing `deepagents_code.agent` / first `settings` access
    # can also trigger `find_project_root()` -> `Path.cwd()`.
    # [해설][흐름] 1) 프로젝트 컨텍스트와 agent/config 모듈을 스레드에서 import(첫
    # [해설]   settings 접근이 Path.cwd()를 호출해 이벤트 루프에서 막힘).
    def _resolve_project_context_and_settings() -> tuple[
        ProjectContext | None,
        Any,
        Any,
        Any,
        Any,
        Any,
    ]:
        project_context = project_context_override or get_server_project_context()

        from deepagents_code.agent import create_cli_agent, load_async_subagents
        from deepagents_code.config import (
            create_model,
            is_memory_auto_save_enabled,
            resolve_auto_classifier_model_for_provider,
        )

        return (
            project_context,
            create_cli_agent,
            load_async_subagents,
            create_model,
            is_memory_auto_save_enabled,
            resolve_auto_classifier_model_for_provider,
        )

    (
        project_context,
        create_cli_agent,
        load_async_subagents,
        create_model,
        is_memory_auto_save_enabled,
        resolve_auto_classifier_model_for_provider,
    ) = await asyncio.to_thread(_resolve_project_context_and_settings)
    # [해설][흐름] 2) 모델 생성: create_model(config.model, ...)을 스레드에서 실행하고
    # [해설]   결과를 런타임 상태에 반영.
    # Offload to a worker thread: `create_model` does blocking disk IO for some
    # providers (e.g. the `openai_codex` token store currently acquires a file
    # lock via `langchain-openai` that calls `os.mkdir`), which `blockbuster`
    # rejects on the server event loop.
    result = await asyncio.to_thread(
        create_model,
        config.model,
        extra_kwargs=config.model_params,
        profile_overrides=config.profile_overrides,
        cli_max_retries=config.cli_max_retries,
    )
    result.apply_to_runtime_state()

    # [해설][흐름] 3) 도구 조립: _build_tools(MCP 포함)와 기준·채점용 읽기 전용 도구
    # [해설]   선별.
    tools, mcp_server_info, mcp_tools, read_only_builtins = await _build_tools(
        config,
        project_context,
        tavily_api_key=workspace_credentials.tavily_api_key,
    )
    read_only_context_tools = _criteria_context_tools(
        tools, mcp_tools, read_only_builtins
    )

    # [해설][흐름] 4) 샌드박스(선택): config.sandbox_type이 있고 override가 없을 때만
    # [해설]   생성한다. 실패하면 사유를 시작 오류로 찍고 sys.exit(1).
    # Create sandbox backend if a sandbox provider is configured.
    # The context manager is created here in the factory, but its reference is
    # stored in a module-level global (and cleaned up via atexit) so the sandbox
    # lives for the entire server process lifetime. `make_graph` caches the built
    # graph, so this runs once per process despite LangGraph's per-run factory
    # invocation.
    global _sandbox_cm, _sandbox_backend  # noqa: PLW0603
    sandbox_backend = sandbox_backend_override
    if (sandbox_type := config.sandbox_type) and sandbox_backend is None:
        from deepagents_code.integrations.sandbox_factory import create_sandbox

        try:
            context, backend = await _open_sandbox(
                lambda: create_sandbox(
                    sandbox_type,
                    sandbox_id=config.sandbox_id,
                    snapshot_name=config.sandbox_snapshot_name,
                    setup_script_path=config.sandbox_setup,
                )
            )
            _sandbox_cm = context
            _sandbox_backend = backend
            sandbox_backend = backend
            atexit.register(_close_sandbox, context)
        except ImportError:
            logger.exception(
                "Sandbox provider '%s' is not installed", config.sandbox_type
            )
            _print_startup_error(
                f"Sandbox provider '{config.sandbox_type}' is not installed"
            )
            sys.exit(1)
        except NotImplementedError:
            logger.exception("Sandbox type '%s' is not supported", config.sandbox_type)
            _print_startup_error(
                f"Sandbox type '{config.sandbox_type}' is not supported"
            )
            sys.exit(1)
        except ValueError as exc:
            logger.exception(
                "Invalid sandbox configuration for '%s'", config.sandbox_type
            )
            _print_startup_error(f"Invalid sandbox configuration: {exc}")
            sys.exit(1)
        except Exception as exc:
            logger.exception("Sandbox creation failed for '%s'", config.sandbox_type)
            _print_startup_error(
                f"Sandbox creation failed for '{config.sandbox_type}': {exc}"
            )
            sys.exit(1)

    # [해설][설계] 아래 _create_cli_graphs_sync는 "정의"만 여기서 하고, 맨 아래에서
    # [해설]   스레드로 실행된다(create_cli_agent가 동기·블로킹이므로).
    # [해설] 안에서 config의 각 필드가 create_cli_agent 인자로 1:1에 가깝게 전달된다.
    # [해설]   이것이 "ServerConfig가 에이전트 모양을 정한다"의 실체다.
    extension_registry: ExtensionRegistry | None = None

    def _create_cli_graphs_sync() -> ServerRuntime:
        async_subagents = load_async_subagents() or None
        auto_mode_enabled = config.interactive and sandbox_backend is None

        interpreter_config = (
            InterpreterConfig.from_resolver(
                get_config_resolver(),
                ptc=config.interpreter_ptc,
                ptc_acknowledge_unsafe=config.interpreter_ptc_acknowledge_unsafe,
            )
            if config.enable_interpreter
            else None
        )

        agent, composite_backend = create_cli_agent(
            model=result.model,
            assistant_id=config.assistant_id,
            tools=tools,
            mcp_tools=mcp_tools,
            sandbox=sandbox_backend,
            sandbox_type=config.sandbox_type,
            system_prompt=config.system_prompt,
            interactive=config.interactive,
            auto_approve=config.auto_approve,
            auto_mode_enabled=auto_mode_enabled,
            interrupt_shell_only=config.interrupt_shell_only,
            shell_allow_list=config.shell_allow_list,
            fs_tools=config.allow_fs_tools,
            enable_ask_user=config.enable_ask_user,
            enable_memory=config.enable_memory,
            memory_auto_save=is_memory_auto_save_enabled(),
            enable_skills=config.enable_skills,
            enable_shell=config.enable_shell,
            enable_interpreter=config.enable_interpreter,
            interpreter_config=interpreter_config,
            rubric_model=config.rubric_model,
            rubric_max_iterations=config.rubric_max_iterations,
            auto_classifier_model=resolve_auto_classifier_model_for_provider(
                result.provider,
                config.auto_classifier_model,
            ),
            recursion_limit=config.recursion_limit,
            mcp_server_info=mcp_server_info,
            cwd=project_context.user_cwd if project_context is not None else config.cwd,
            project_context=project_context,
            async_subagents=async_subagents,
            goal_criteria_tools=read_only_context_tools,
            rubric_grader_tools=read_only_context_tools,
            model_retries=result.model_retries,
            cli_max_retries=result.cli_max_retries,
            summarization_model=config.summarization_model,
            extension_registry=extension_registry,
            environ=workspace_env,
            credentials_snapshot=workspace_credentials,
            model_result=result,
            profile_overrides=config.profile_overrides,
        )
        # [해설] 서버가 소유하는 /offload 연산을 같은 백엔드에 묶는다. 백엔드가 오프로드
        # [해설]   연산을 노출하지 않으면 오류(RuntimeError).
        from deepagents_code.offload_middleware import offload_operation_from

        offload = offload_operation_from(composite_backend)
        if offload is None:
            msg = (
                "Agent backend did not publish its offload operation; "
                "/offload has no server implementation."
            )
            raise RuntimeError(msg)
        # Bind Langfuse callbacks to the served graph. The agent runs in this
        # server process, so callbacks cannot come from the client's run config.
        from deepagents_code.harness.observability import attach_langfuse

        agent = attach_langfuse(agent, workspace_env)
        return ServerRuntime(
            agent=agent,
            backend=composite_backend,
            offload=offload,
            mcp_server_info=mcp_server_info,
            model_metadata=ModelMetadata(
                result.model_name,
                result.provider,
                result.context_limit,
                result.unsupported_modalities,
            ),
            model_environment=workspace_env,
        )

    # [해설][흐름] 5) 확장(extensions) 로드: 실험 플래그가 켜졌을 때만. 확장 로드 오류는
    # [해설]   경고만 남기고, 활성 확장이 있으면 레지스트리를 만든다.
    from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy

    if is_env_truthy(EXPERIMENTAL, environ=workspace_env):
        from deepagents_code.extensions import ExtensionMode, load_extensions
        from deepagents_code.extensions.runtime import bind_server_extensions

        extension_result = await load_extensions(
            cwd=(
                project_context.user_cwd
                if project_context is not None
                else Path(config.cwd)
                if config.cwd is not None
                else None
            ),
            mode=(
                ExtensionMode.INTERACTIVE
                if config.interactive
                else ExtensionMode.HEADLESS
            ),
            project_root=(
                project_context.project_root or project_context.user_cwd
                if project_context is not None
                else None
            ),
            project_trust_granted=config.trust_project_extensions,
            cli_paths=tuple(Path(path) for path in config.extension_paths),
        )
        for message in extension_result.errors:
            logger.warning("Extension not loaded: %s", message)
        if extension_result.active:
            extension_registry = extension_result.registry
            bind_server_extensions(extension_result)
    # [해설][흐름] 6) create_cli_agent 실행(스레드). 실패하면 이미 연 확장을 정리한 뒤
    # [해설]   예외를 다시 던진다.
    try:
        return await asyncio.to_thread(_create_cli_graphs_sync)
    except BaseException:
        if extension_registry is not None:
            from deepagents_code.extensions.runtime import shutdown_server_extensions

            await shutdown_server_extensions()
        raise


# [해설][주의] 이 캐시는 최적화가 아니라 정확성 요건이다: MCP 탐색, 샌드박스 생성,
# [해설]   atexit 등록이 정확히 한 번만 일어나야 한다.
# [해설] 캐시와 락은 클로저 안에 있어 모듈 import만으로 전역 상태가 생기지 않는다.
# [해설]   실패는 emit_startup_failure + sys.exit(1).
# [해설][흐름] 호출 시: 관리형 설정이 건강한지 확인(require_healthy_managed_config) ->
# [해설]   builder(기본 _make_graphs) 실행 -> 캐시.
def _build_runtime_factory(
    builder: Callable[[], Awaitable[ServerRuntime]] | None = None,
) -> Callable[[], Awaitable[ServerRuntime]]:
    """Build the cached factory for all server-owned runtime resources.

    The cache is load-bearing, not an optimization: MCP discovery, sandbox
    creation, and `atexit` registration each must happen exactly once. Building
    per request would re-discover MCP servers, leak sandbox sessions, and stack
    duplicate `atexit` handlers. Two consumers now share this cache -- the
    interactive graph and the offload HTTP route -- so both must resolve the
    *same* agent, backend, and compaction policy for a server-side archive to be
    readable by the agent.

    The cache and its lock live in this closure rather than in module-level
    globals, so importing this module introduces no shared mutable state; the
    single process-wide instance is created explicitly at the bottom of the
    module.

    Args:
        builder: Optional alternate builder used by unit tests.

    Returns:
        Async runtime factory shared by the graph and custom operation API.
    """
    runtime: ServerRuntime | None = None
    lock = asyncio.Lock()

    async def get_runtime() -> ServerRuntime:
        """Return the cached interactive graph and operation resources."""
        nonlocal runtime
        if runtime is None:
            async with lock:
                if runtime is None:
                    try:
                        from deepagents_code.configuration.service import (
                            require_healthy_managed_config,
                        )

                        await asyncio.to_thread(
                            require_healthy_managed_config,
                            refresh=True,
                        )
                        runtime = await (builder or _make_graphs)()
                    except Exception as exc:  # noqa: BLE001  # startup barrier
                        emit_startup_failure(exc)
                        sys.exit(1)
        return runtime

    return get_runtime


# [해설] 테스트용 그래프 팩토리(운영 경로는 모듈 수준 make_graph를 쓴다). 런타임
# [해설]   팩토리에서 .agent만 꺼내 돌려준다.
def _build_graph_factory(
    builder: Callable[[], Awaitable[ServerRuntime]] | None = None,
) -> Callable[[], Awaitable[Any]]:
    """Build a cached graph factory, for tests.

    `langgraph.json` references the module-level `make_graph`, which delegates to
    `get_server_runtime`; nothing in production calls this. It survives so unit
    tests can inject a builder.

    Args:
        builder: Optional alternate runtime builder used by unit tests.

    Returns:
        Async graph factory for the interactive `agent` graph.
    """
    get_runtime = _build_runtime_factory(builder)

    async def make_graph() -> Any:  # noqa: ANN401
        """Create or return the cached agent graph for `langgraph dev`.

        Returns:
            Compiled LangGraph agent graph.
        """
        return (await get_runtime()).agent

    return make_graph


# [해설] 런타임 캐시 전역: _get_runtime(시작 작업공간용 단일 런타임),
# [해설]   _workspace_runtimes(작업공간별 LRU, 최대 32개),
# [해설] _sandbox_workspace_id(프로세스 전역 샌드박스를 처음 요청한 작업공간).
_get_runtime = _build_runtime_factory()
_MAX_WORKSPACE_RUNTIMES = 32
_workspace_runtimes: OrderedDict[str, ServerRuntime] = OrderedDict()
_workspace_runtime_lock = asyncio.Lock()
_sandbox_workspace_id: str | None = None


# [해설][설계] 캐시 키 = 런타임 지문 + workspace_id. 모델을 바꾸면 키가 바뀌어 런타임만
# [해설]   새로 만들고, 바인딩과 체크포인트는 유지된다.
def _runtime_cache_key(
    binding: WorkspaceBinding, *, current_config_fingerprint: str | None = None
) -> str:
    """Key the runtime cache on full runtime identity, not just the binding.

    The binding's `resource_key` mixes workspace identity with the policy
    fingerprint; the runtime cache must *also* change when the full runtime
    identity (model, model params, prompt, runtime-only fields) changes, so a
    model switch rebuilds the runtime while the durable binding — and its
    checkpoints/history — are preserved.

    Args:
        binding: The durable binding (workspace identity).
        current_config_fingerprint: The runtime fingerprint of the *current*
            resolved config, when the caller has resolved one. Falls back to
            the binding's recorded fingerprint otherwise.

    Returns:
        The cache key for this binding's current runtime identity.
    """
    from deepagents_code.workspace import canonical_fingerprint

    return canonical_fingerprint(
        {
            "runtime_fingerprint": (
                current_config_fingerprint
                or binding.runtime_fingerprint
                or binding.config_fingerprint
            ),
            "workspace_id": binding.workspace_id,
        }
    )


# [해설] 캐시에서 런타임을 찾고, 찾으면 LRU 최신으로 끌어올린다.
def _cached_workspace_runtime(
    binding: WorkspaceBinding, *, current_config_fingerprint: str | None = None
) -> ServerRuntime | None:
    """Return and refresh a cached runtime for one workspace binding."""
    key = _runtime_cache_key(
        binding, current_config_fingerprint=current_config_fingerprint
    )
    cached = _workspace_runtimes.get(key)
    if cached is None:
        return None
    _workspace_runtimes.move_to_end(key)
    return cached


# [해설][주의] 샌드박스는 프로세스 전역 하나뿐이라 먼저 요청한 작업공간이 소유한다. 다른
# [해설]   작업공간이 요청하면 충돌 예외.
def _claim_sandbox_workspace(
    sandbox_type: str | None,
    binding: WorkspaceBinding,
) -> None:
    """Reserve the process-wide sandbox for the first requesting workspace."""
    global _sandbox_workspace_id  # noqa: PLW0603  # process-lifetime ownership
    _validate_sandbox_workspace(sandbox_type, binding)
    if sandbox_type:
        _sandbox_workspace_id = binding.workspace_id


# [해설] 소유권 검사만 하고 예약은 하지 않는다(검증 단계에서 사용).
def _validate_sandbox_workspace(
    sandbox_type: str | None, binding: WorkspaceBinding
) -> None:
    """Check sandbox ownership without reserving the process-wide sandbox."""
    if not sandbox_type or _sandbox_workspace_id in {None, binding.workspace_id}:
        return
    reason = (
        "a runtime for another workspace already exists and the configured "
        "sandbox is process-wide"
    )
    # Built into a local first: `raise X.from_reason(...)` reads as a
    # `from_reason` raise to ruff's DOC501.
    conflict = WorkspaceConflictError.from_reason(
        reason,
        diagnostics=WorkspaceDiagnostics(
            category="policy_drift",
            reason=reason,
            snapshot_status="unavailable",
        ),
    )
    raise conflict


# [해설] 런타임을 캐시에 넣고 최대 개수(32)를 넘으면 가장 오래된 것부터 제거(LRU).
def _remember_workspace_runtime(
    binding: WorkspaceBinding,
    runtime: ServerRuntime,
    *,
    current_config_fingerprint: str | None = None,
) -> None:
    """Cache one workspace runtime and enforce the bounded LRU size."""
    _workspace_runtimes[
        _runtime_cache_key(
            binding, current_config_fingerprint=current_config_fingerprint
        )
    ] = runtime
    if len(_workspace_runtimes) > _MAX_WORKSPACE_RUNTIMES:
        _workspace_runtimes.popitem(last=False)


# [해설] 서버 시작 시점의 작업공간(config.cwd)을 바인딩 형태로 해석한다. cwd가 없으면
# [해설]   None -> 작업공간 개념 없이 단일 런타임.
async def _default_workspace_binding(config: ServerConfig) -> WorkspaceBinding | None:
    """Resolve the launch workspace represented by the server configuration.

    Returns:
        The canonical launch binding, or `None` without a configured workspace.
    """
    if config.cwd is None:
        return None

    def _bind() -> WorkspaceBinding:
        # First pass resolves identity only (cwd plus project root); its
        # fingerprints are digests of an empty policy and are discarded.
        identity = resolve_workspace(config.cwd)
        # The shared policy resolver honors the explicit launch root while
        # keeping the durable identity consistent with workspace validation.
        resolved = config.resolve_workspace(identity.cwd, identity.project_root)
        return resolve_workspace(
            identity.cwd,
            resolved.to_workspace_payload(),
            config_fingerprint=resolved.workspace_fingerprint(),
        )

    return await asyncio.to_thread(_bind)


# [해설][주의] 실행 요청마다 호출되는 드리프트 검사. 바인딩 때와 정책이 달라졌으면
# [해설]   거부하고, 모델/런타임만 바뀌었으면 거부하지 않고 재빌드한다.
# [해설][흐름] 1) 환경에서 ServerConfig 복원 2) 바인딩의 cwd/project_root로 현재 정책
# [해설]   해석
# [해설][흐름] 3) 저장된 스냅샷 조회 4) 확장 신뢰 부여가 사라졌는지(취소 또는 읽기 실패)
# [해설]   검사 -> 거부
# [해설][흐름] 5) 프로젝트 정책 필드 드리프트 -> 거부 6) 접근 정책 지문 비교 -> 거부 7)
# [해설]   런타임 지문만 다르면 로그만 남기고 통과.
async def _resolve_bound_workspace_config(
    binding: WorkspaceBinding,
) -> ServerConfig:
    """Resolve current workspace policy and reject drift from its binding.

    Refusals name the fields that drifted. This runs on every request, and it
    reads the extension trust store each time, so a transient read failure
    reports as a policy change; without the field names that refusal is not
    diagnosable. Only allowlisted, non-path policy values reach the attached
    diagnostics: paths, model parameters, and prompts are never reported.

    Returns:
        The current server configuration resolved for the workspace.
    """
    config = await asyncio.to_thread(ServerConfig.from_env)

    def _resolve_current() -> ServerConfig:
        resolved = config.resolve_workspace(binding.cwd, binding.project_root)
        return resolved.preserve_bound_extension_trust(binding.workspace_config())

    current_config = await asyncio.to_thread(_resolve_current)
    bound_policy = binding.workspace_config()
    snapshot = await get_snapshot_for_binding(binding)
    snapshot_status = "current" if snapshot is not None else "unavailable"
    # Fail closed on a disappeared extension-trust grant. A grant recorded at
    # bind time that the trust store no longer reports is either a genuine
    # revocation or a transient store-read failure (which fails closed to
    # `False`). `preserve_bound_extension_trust` only *adds* privilege, so
    # without this guard a vanished grant would be silently dropped — the policy
    # payload still matches (it omits nothing) but the runtime fingerprint
    # changes and the rebuild would run without the grant. Refuse instead.
    if (
        bound_policy.get("trust_project_extensions") is True
        and current_config.trust_project_extensions is not True
    ):
        reason = (
            "the project's extension trust recorded at binding is no longer "
            "present; re-bind the thread to re-evaluate trust"
        )
        conflict = WorkspaceConflictError.from_reason(
            reason,
            diagnostics=WorkspaceDiagnostics(
                category="policy_drift",
                reason=reason,
                snapshot_status=snapshot_status,
            ),
        )
        logger.warning(
            "Workspace %s extension trust changed since binding", binding.cwd
        )
        raise conflict
    current_snapshot = snapshot_for_payload(current_config.to_workspace_payload())
    drifted = drifted_project_fields(
        bound_policy, current_config.to_project_workspace_policy()
    )
    if drifted:
        fields = ", ".join(drifted)
        changes = diff_snapshots(snapshot, current_snapshot, changed_names=drifted)
        conflict = WorkspaceConflictError.from_reason(
            f"{PROJECT_POLICY_DRIFT_REASON} ({fields})",
            diagnostics=WorkspaceDiagnostics(
                category="policy_drift",
                reason=PROJECT_POLICY_DRIFT_REASON,
                changes=changes,
                snapshot_status=snapshot_status,
            ),
        )
        logger.warning(
            "Workspace %s project policy drifted since binding: %s",
            binding.cwd,
            conflict.diagnostics.log_summary()
            if conflict.diagnostics is not None
            else fields,
        )
        raise conflict
    # Durable access-policy compatibility: only trust/tool/sandbox/approval
    # policy (and workspace identity) invalidate a binding. Cosmetic model
    # settings and runtime-only fields are excluded from the policy payload, so
    # a model switch no longer refuses the thread — it rebuilds the runtime via
    # the runtime-fingerprint cache key instead.
    current_policy = current_config.to_workspace_payload()
    if binding.policy_fingerprint:
        policy_changed = binding.policy_fingerprint != canonical_fingerprint(
            {
                "cwd": binding.cwd,
                "policy": current_policy,
                "project_root": binding.project_root,
            }
        )
    else:
        # Pre-v4 row: no policy fingerprint recorded. Fall back to the strict
        # full-fingerprint comparison (fail closed on any change).
        policy_changed = (
            current_config.workspace_fingerprint() != binding.config_fingerprint
        )
    if policy_changed:
        payload_drift = sorted(
            key
            for key in bound_policy
            if bound_policy.get(key) != current_policy.get(key)
        )
        changes = diff_snapshots(
            snapshot, current_snapshot, changed_names=payload_drift
        )
        conflict = WorkspaceConflictError.from_reason(
            SERVER_CONFIG_DRIFT_REASON,
            diagnostics=WorkspaceDiagnostics(
                category="config_drift",
                reason=SERVER_CONFIG_DRIFT_REASON,
                changes=changes,
                snapshot_status=snapshot_status,
            ),
        )
        logger.warning(
            "Workspace %s access policy changed since binding: %s",
            binding.cwd,
            conflict.diagnostics.log_summary()
            if conflict.diagnostics is not None
            else SERVER_CONFIG_DRIFT_REASON,
        )
        raise conflict
    if current_config.runtime_fingerprint() != (
        binding.runtime_fingerprint or binding.config_fingerprint
    ):
        # Runtime identity (model/params/prompt/runtime fields) changed without
        # any policy drift. Not a refusal: log it and let the cache key rebuild.
        logger.info(
            "Workspace %s runtime identity changed since binding; "
            "rebuilding the runtime (access policy unchanged)",
            binding.cwd,
        )
    return current_config


# [해설] 자원을 만들지 않고 호환성만 점검한다: 바인딩 드리프트, 샌드박스 소유권,
# [해설]   트레이싱 설정.
async def _validate_workspace_runtime(binding: WorkspaceBinding) -> None:
    """Check workspace compatibility without constructing or reserving resources."""
    config = await _resolve_bound_workspace_config(binding)
    _validate_sandbox_workspace(config.sandbox_type, binding)

    def validate_tracing() -> None:
        from deepagents_code.config import (
            _preview_dotenv_environ,
            is_langsmith_redaction_enabled,
            use_environment,
        )

        environ = _preview_dotenv_environ(start_path=Path(binding.cwd))
        with use_environment(environ):
            _validate_server_tracing(environ, redact=is_langsmith_redaction_enabled())

    await asyncio.to_thread(validate_tracing)


# [해설][흐름] 작업공간 바인딩 -> 런타임. 1) 현재 설정 해석 + 런타임 지문 2) 캐시
# [해설]   적중이면 반환 3) 락을 잡고 한 번 더 확인
# [해설][흐름] 4) 샌드박스 예약 5) ProjectContext 구성 6) _make_graphs로 새로 빌드 7)
# [해설]   캐시에 저장.
async def _workspace_runtime(binding: WorkspaceBinding) -> ServerRuntime:
    """Build or reuse a runtime from the persisted workspace resource policy.

    Returns:
        The runtime selected by the binding's workspace identity and the
            current full runtime fingerprint (so a model change rebuilds while
            the binding and its checkpoints are preserved).
    """
    current_config = await _resolve_bound_workspace_config(binding)
    runtime_fp = current_config.runtime_fingerprint()
    cached = _cached_workspace_runtime(binding, current_config_fingerprint=runtime_fp)
    if cached is not None:
        return cached
    async with _workspace_runtime_lock:
        cached = _cached_workspace_runtime(
            binding, current_config_fingerprint=runtime_fp
        )
        if cached is not None:
            return cached
        _claim_sandbox_workspace(current_config.sandbox_type, binding)
        project_context = ProjectContext(
            user_cwd=Path(binding.cwd),
            project_root=(
                Path(current_config.project_root)
                if current_config.project_root
                else None
            ),
        )
        runtime = await _make_graphs(
            config_override=current_config,
            project_context_override=project_context,
            sandbox_backend_override=(
                _sandbox_backend if current_config.sandbox_type else None
            ),
        )
        _remember_workspace_runtime(
            binding, runtime, current_config_fingerprint=runtime_fp
        )
        return runtime


# [해설][주의] "시작 작업공간"용 경로. 요청 범위에서는 SystemExit(BaseException)을 잡아
# [해설]   처리해야 서버가 요청 도중 죽지 않는다.
# [해설] 실패는 마커를 찍고 sys.exit(1)로 끝나며, 부모 프로세스가 이를 읽는다. 바인딩이
# [해설]   없으면 단일 _get_runtime을 쓴다.
async def get_server_runtime() -> ServerRuntime:
    """Return resources shared by the graph and dcode operation routes.

    Builds once and caches. A construction failure is converted into a
    startup-error marker (scraped by the parent app process) before
    `sys.exit(1)`, which is right for the `langgraph.json` graph factory at
    startup. Callers in request scope must contain that exit -- `SystemExit` is a
    `BaseException` -- as `offload_api._execute_offload` does, mapping it to a 503
    rather than killing the server mid-request.

    Returns:
        The cached server runtime.
    """
    # Resolving the launch binding touches the filesystem and can raise, and
    # claiming the sandbox can refuse. Both run before `_get_runtime`, so they
    # sit outside its startup barrier and would exit without the marker the
    # parent app process scrapes. Emit it here instead.
    try:
        config = ServerConfig.from_env()
        binding = await _default_workspace_binding(config)
    except Exception as exc:  # noqa: BLE001  # startup barrier
        emit_startup_failure(exc)
        sys.exit(1)
    async with _workspace_runtime_lock:
        if binding is None:
            return await _get_runtime()
        cached = _cached_workspace_runtime(binding)
        if cached is not None:
            return cached
        _claim_sandbox_workspace(config.sandbox_type, binding)
        runtime = await _get_runtime()
        _remember_workspace_runtime(binding, runtime)
        return runtime


# [해설][주의] langgraph.json이 가리키는 실제 진입점. LangGraph가 (config, runtime) 인자
# [해설]   이름을 보고 호출한다(상단 import 주석 참고).
# [해설][흐름] 1) runtime.execution_runtime이 있으면 = 실제 실행: context.workspace와
# [해설]   thread_id를 필수로 요구
# [해설][흐름] 2) require_thread_workspace로 서버 바인딩 대조(workspace.py) ->
# [해설]   _workspace_runtime -> .agent
# [해설][흐름] 3) execution이 없으면(그래프 구조·스키마 조회, 상태 읽기·갱신 등 실행이
# [해설]   아닌 접근. langgraph_sdk의 execution_runtime은 threads.create_run일 때만 값이
# [해설]   있다) get_server_runtime의 시작 작업공간 런타임을 사용.
async def make_graph(
    config: dict[str, Any] | None = None,
    runtime: LangGraphServerRuntime[CLIContextSchema] | None = None,
) -> Any:  # noqa: ANN401
    """Return the graph after validating execution workspace context.

    Raises:
        ValueError: If execution context is missing or malformed.
    """
    execution = runtime.execution_runtime if runtime is not None else None
    if execution is not None:
        context = CLIContextSchema.from_payload(execution.context)
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if context is None or not isinstance(thread_id, str) or not thread_id:
            msg = "A thread id and workspace context are required for execution."
            raise ValueError(msg)
        from deepagents_code.workspace import require_thread_workspace

        binding = await require_thread_workspace(thread_id, context.workspace)
        return (await _workspace_runtime(binding)).agent
    return (await get_server_runtime()).agent
