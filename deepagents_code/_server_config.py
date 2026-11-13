"""Typed configuration for the app-to-server subprocess communication channel.

The app spawns a `langgraph dev` subprocess and passes configuration via
environment variables prefixed with `DEEPAGENTS_CODE_SERVER_`. This module
provides a single
`ServerConfig` dataclass that both sides share so that the set of variables,
their serialization format, and their default values are defined in one place.
The app writes config with `to_env()` and the server graph reads it back
with `from_env()`.
"""

# [해설] 역할: 클라이언트(TUI/헤드리스)와 LangGraph 서버 사이의 "설정 전달 계약". 두
# [해설]   프로세스는 메모리를 공유하지
# [해설]   않으므로, 설정은 `DEEPAGENTS_CODE_SERVER_*` 환경변수로만 건너간다(접두사는
# [해설]   _env_vars.py의 SERVER_ENV_PREFIX).
# [해설] 방향: 클라이언트 프로세스가 ServerConfig.from_cli_args로 만들고 to_env()로
# [해설]   직렬화한다
# [해설]   (client/launch/server_manager.py의 _apply_server_config가 os.environ에 기록
# [해설]   -> server.py가 자식 env로 복사).
# [해설]   서버 프로세스는 ServerConfig.from_env()로 복원한다(server_graph.py의
# [해설]   _make_graphs 등).
# [해설] 한 클래스가 쓰기(to_env)와 읽기(from_env)를 모두 가지므로 변수
# [해설]   이름/형식/기본값이 한곳에서만 정의된다.
# [해설] 두 번째 역할: 작업공간(workspace) 정책 지문. 필드를 정책/모델/런타임/신원으로
# [해설]   나눠 지문을 계산하고,
# [해설]   workspace.py의 바인딩과 server_graph.py의 드리프트 검사가 이 지문을 사용한다.
# [해설] 읽는 순서 제안: 맨 위 분류 상수 -> ServerConfig 필드 -> to_env/from_env ->
# [해설]   from_cli_args -> 지문 메서드.
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from deepagents_code._constants import DEFAULT_AGENT_NAME as DEFAULT_ASSISTANT_ID
from deepagents_code._env_vars import SERVER_ENV_PREFIX

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents import FsToolName

    from deepagents_code.project_utils import ProjectContext

logger = logging.getLogger(__name__)


# [해설][설계] 필드 분류 상수 5종(SESSION / PROJECT / MODEL_COMPATIBLE / RUNTIME_ONLY /
# [해설]   WORKSPACE_IDENTITY).
# [해설]   정책(to_workspace_payload의 키)은 SESSION과 PROJECT로 "정확히 분할"되어야
# [해설]   한다(아래 docstring 참고).
# [해설]   분류가 곧 보안 경계다: SESSION은 클라이언트가 주장해도 되는 값, PROJECT는
# [해설]   서버가 디렉터리별로 직접 판정할 값이다.
SESSION_WORKSPACE_FIELDS = frozenset(
    {
        "allow_fs_tools",
        "assistant_id",
        "auto_approve",
        "enable_ask_user",
        "enable_interpreter",
        "enable_memory",
        "enable_shell",
        "enable_skills",
        "interactive",
        "interpreter_ptc",
        "interpreter_ptc_acknowledge_unsafe",
        "interrupt_shell_only",
        "no_mcp",
        "recursion_limit",
        "sandbox_id",
        "sandbox_snapshot_name",
        "sandbox_type",
        "shell_allow_list",
    }
)
"""Policy a managed client may claim for its own command invocation.

These come from the client's own CLI flags, so the client already knows them
and claiming them proves only that both sides agree. Together with
`PROJECT_WORKSPACE_FIELDS` this must partition `to_workspace_payload()`
exactly: a payload field in neither set is never verified against a client
claim and never checked for project drift.
`test_workspace_claim_partitions_every_policy_field` pins that.
"""
PROJECT_WORKSPACE_FIELDS = frozenset(
    {
        "extension_paths",
        "mcp_config_path",
        "sandbox_setup",
        "trust_project_extensions",
        "trust_project_mcp",
    }
)
"""Policy the server must resolve per project directory, never accept.

Each of these grants code execution scoped to a checkout -- MCP servers,
sandbox setup commands, Python extensions. A client that could claim them could
execute one directory's configuration against another directory's trust
decision.
"""

MODEL_COMPATIBLE_FIELDS = frozenset(
    {
        "auto_classifier_model",
        "cli_max_retries",
        "model",
        "rubric_model",
        "summarization_model",
    }
)
"""Cosmetic model settings that do not affect durable access-policy compat.

Changing which model serves a bound thread — or the retry budget or auxiliary
grader/classifier models — does not change what the thread is *allowed* to do
(trust, tool, sandbox, and approval policy are unchanged). These fields are
therefore excluded from the policy fingerprint, so a model switch does not
invalidate a durable binding. They remain part of the runtime fingerprint so
the runtime rebuilds. `None` and an empty/unset value are normalized as
equivalent here.
"""

RUNTIME_ONLY_FIELDS = frozenset(
    {
        "assistant_id",
        "model_params",
        "profile_overrides",
        "recursion_limit",
        "rubric_max_iterations",
        "system_prompt",
    }
)
"""Runtime-identity fields that are not durable access policy.

These shape *how* the runtime executes (model kwargs, prompt, step budget,
graph id) but are not the trust/tool/sandbox/approval policy a binding guards.
They are fingerprinted into the runtime identity (so changing them rebuilds the
runtime) yet never persisted or reported — `model_params`, `profile_overrides`,
and `system_prompt` can carry secrets and prompt material.
"""

WORKSPACE_IDENTITY_FIELDS = frozenset({"cwd", "project_root"})
"""Fields that name *which* workspace a binding is for.

Workspace identity is part of durable access-policy compatibility: a binding
is meaningless if it can be replayed against a different directory. These are
compared exactly (canonical paths) and are never normalized away.
"""


# [해설] 모델/런타임/신원 분류 집합의 합집합을 돌려준다. 이 저장소의 deepagents_code
# [해설]   안에서는 호출처를 찾지 못했다(추정: 테스트·보조용).
def _classified_fields() -> frozenset[str]:
    """Return every field the policy/runtime split accounts for.

    Returns:
        The union of all classification sets plus workspace identity.
    """
    return MODEL_COMPATIBLE_FIELDS | RUNTIME_ONLY_FIELDS | WORKSPACE_IDENTITY_FIELDS


# [해설][설계] 두 경로가 같은 프로젝트 디렉터리인지 장치/inode로 비교한다. 비교가
# [해설]   불가능하면 "다르다"로 처리해(fail closed)
# [해설] 프로젝트 정책이 검증 없이 다른 경계로 넘어가지 않게 한다.
def _same_workspace_project(first: str | None, second: str) -> bool:
    """Whether two paths name the same project directory.

    Fails closed: an unset launch root, a missing path, or an undecidable
    comparison counts as *different*, so the caller drops project policy rather
    than carrying it across an unverified boundary. `_same_directory` compares
    by device and inode, so a symlinked or differently cased spelling of one
    directory still compares equal.

    Returns:
        `True` only when both paths name the same directory.
    """
    if first is None:
        return False
    from deepagents_code._paths import DeepAgentsHomeError, _same_directory

    try:
        return _same_directory(Path(first), Path(second))
    except DeepAgentsHomeError:
        logger.warning(
            "Could not compare project directories %s and %s; treating as "
            "separate projects, so project-scoped policy will not apply",
            first,
            second,
            exc_info=True,
        )
        return False


# [해설] 아래 _read_env_* 헬퍼 묶음은 from_env의 부품이다. 모두
# [해설]   `DEEPAGENTS_CODE_SERVER_<suffix>` 한 변수를 읽는다.
# [해설] bool은 "true"(대소문자 무시)만 참이다. 값이 없으면 default를 돌려준다.
def _read_env_bool(suffix: str, *, default: bool = False) -> bool:
    """Read a `DEEPAGENTS_CODE_SERVER_*` boolean from the environment.

    Boolean env vars use the `'true'` / `'false'` convention (case insensitive).
    Missing variables fall back to *default*.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.
        default: Value when the variable is absent.

    Returns:
        Parsed boolean.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return default
    return raw.lower() == "true"


# [해설][주의] JSON이 깨지면 ValueError로 즉시 실패한다(조용히 무시하지 않음). 반면
# [해설]   _read_env_int는 파싱 실패를 default로 삼킨다.
def _read_env_json(suffix: str) -> Any:  # noqa: ANN401
    """Read a JSON-encoded `DEEPAGENTS_CODE_SERVER_*` variable.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.

    Returns:
        Parsed JSON value, or `None` if the variable is absent.

    Raises:
        ValueError: If the variable is present but not valid JSON.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = (
            f"Failed to parse {SERVER_ENV_PREFIX}{suffix} as JSON: {exc}. "
            f"Value was: {raw[:200]!r}"
        )
        raise ValueError(msg) from exc


def _read_env_str_list(suffix: str) -> tuple[str, ...]:
    raw = _read_env_json(suffix)
    if raw is None:
        return ()
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return tuple(raw)
    msg = f"Invalid {SERVER_ENV_PREFIX}{suffix}: expected a JSON string list"
    raise ValueError(msg)


# [해설][설계] 파일시스템 도구 허용 목록은 보안 통제값이다. 서버 쪽에서 환경변수가
# [해설]   변조됐을 가능성까지 고려해,
# [해설] 빈 목록/모르는 도구 이름이면 "제한 없음"으로 흘리지 않고 예외를 던진다(fail
# [해설]   closed).
def _read_env_allow_fs_tools() -> list[FsToolName] | None:
    """Read and shape-validate the `ALLOW_FS_TOOLS` filesystem allowlist.

    The parent writes only an absent variable (unrestricted — `None`, which is
    also what `--allow-fs-tools all` collapses to) or a non-empty JSON list of
    tool names (`main._parse_allow_fs_tools_flag`). This runs in the server
    subprocess, where the variable could be tampered with, so — because the
    value is a security control — any unrecognized shape must fail closed
    (raise) rather than fall through to an unrestricted filesystem.
    (`_read_env_json` already fails closed on malformed JSON.)

    `[]` and unknown tool names are rejected here, not deferred downstream, so
    the returned list genuinely satisfies `list[FsToolName]` and the `cast`
    asserts membership that was actually checked. Importing `deepagents` here is
    fine: the subprocess already imports the SDK to build the agent (this is not
    the arg-parsing hot path guarded in `main`). The `"read_file"` requirement
    is not checked here: `ServerConfig.__post_init__` enforces it when the
    returned value is placed on the config (with `FilesystemMiddleware` as a
    final backstop), so a tampered list without `read_file` still fails closed
    at construction.

    Returns:
        `None` when the variable is absent, or a non-empty list of filesystem
            tool-name strings, each a valid `FsToolName`.

    Raises:
        ValueError: If the present variable parses to anything other than a
            non-empty list of strings, or if any list element is not a
            recognized filesystem tool name.
    """
    env_name = f"{SERVER_ENV_PREFIX}ALLOW_FS_TOOLS"
    if env_name not in os.environ:
        return None

    raw = _read_env_json("ALLOW_FS_TOOLS")
    if isinstance(raw, list) and raw and all(isinstance(name, str) for name in raw):
        from typing import get_args

        from deepagents import FsToolName

        valid_names = frozenset(get_args(FsToolName))
        unknown = [name for name in raw if name not in valid_names]
        if unknown:
            msg = (
                f"Invalid {SERVER_ENV_PREFIX}ALLOW_FS_TOOLS value: unknown "
                f"filesystem tool name(s) {unknown!r}; valid names are "
                f"{sorted(valid_names)}."
            )
            raise ValueError(msg)
        return cast("list[FsToolName]", raw)
    msg = (
        f"Invalid {SERVER_ENV_PREFIX}ALLOW_FS_TOOLS value: {raw!r}; expected "
        "a non-empty list of filesystem tool names."
    )
    raise ValueError(msg)


def _read_env_str(suffix: str) -> str | None:
    """Read an optional `DEEPAGENTS_CODE_SERVER_*` string variable.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.

    Returns:
        The string value, or `None` if absent.
    """
    return os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")


# [해설][주의] 정수 파싱 실패 시 default로 조용히 되돌린다. 위 JSON/allow_fs_tools
# [해설]   헬퍼의 엄격함과 대비해 읽을 것.
def _read_env_int(suffix: str, *, default: int | None) -> int | None:
    """Read a `DEEPAGENTS_CODE_SERVER_*` integer from the environment.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.
        default: Value when the variable is absent or malformed.

    Returns:
        Parsed integer, or the default when absent or parsing fails.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# [해설] 3상태(True/False/None) bool. None이 "지정 안 함 -> 기본 로직 사용"이라는 별도
# [해설]   의미일 때 쓴다(TRUST_PROJECT_MCP).
def _read_env_optional_bool(suffix: str) -> bool | None:
    """Read a tri-state `DEEPAGENTS_CODE_SERVER_*` boolean (`True` / `False` / `None`).

    Used for settings where `None` carries a distinct meaning (e.g. "not
    specified, use default logic").

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.

    Returns:
        `True`, `False`, or `None` when the variable is absent.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return None
    return raw.lower() == "true"


# [해설] enable_interpreter의 3상태 값을 확정한다. 명시값 우선 -> 원격 샌드박스면 False
# [해설]   -> 아니면 설정 resolver의 기본값.
# [해설] 클라이언트 쪽 from_cli_args에서만 호출되므로, 서버가 from_env로 읽는 값은 이미
# [해설]   확정된 bool이다.
def _resolve_enable_interpreter(
    enable_interpreter: bool | None, sandbox_type: str | None
) -> bool:
    """Resolve the interpreter's tri-state caller option to a concrete boolean.

    Args:
        enable_interpreter: Explicit caller preference, or `None` to use the
            sandbox-aware default.
        sandbox_type: Sandbox backend identifier. Any falsy value (`None`, `""`)
            or `"none"` is treated as local execution.

    Returns:
        The explicit `enable_interpreter` value when not `None`; `False` for
            remote-sandbox defaults; otherwise the configured local default
            from `interpreter.enable_interpreter`.

    Raises:
        RuntimeError: If the interpreter option is absent from the manifest.
    """
    if enable_interpreter is not None:
        return enable_interpreter
    if sandbox_type and sandbox_type != "none":
        return False

    from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option
    from deepagents_code.configuration.resolver import get_config_resolver

    option = get_option("interpreter.enable_interpreter")
    if option is None:
        msg = "interpreter.enable_interpreter is missing from the config manifest"
        raise RuntimeError(msg)
    resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, resolved)
    return bool(resolved.value)


# [해설] 샌드박스 때문에 인터프리터가 꺼졌는지(=안내 메시지를 보여 줄지) 판단한다.
# [해설]   사용자가 명시 선택했다면 안내하지 않는다.
def _interpreter_suppressed_by_sandbox(
    *, enable_interpreter: bool | None, sandbox_type: str | None, local_default: bool
) -> bool:
    """Whether a remote sandbox suppressed the otherwise-default interpreter.

    Used to decide whether to surface an advisory: returns `True` only when the
    user made no explicit choice, a remote sandbox is active, and the local
    default would have enabled it — i.e. the sandbox (not an explicit
    `--no-interpreter` opt-out, nor a disabled `[interpreter]` config) is why
    `js_eval` is unavailable.

    Takes the *raw* tri-state caller intent rather than the resolved boolean: a
    sandbox-suppressed default and an explicit `--no-interpreter` both resolve to
    `False`, so the resolved value cannot distinguish them. Any explicit choice
    (`not None`) is the user's own decision and is left unannounced.

    Args:
        enable_interpreter: The raw tri-state caller intent (`--interpreter` →
            `True`, `--no-interpreter` → `False`, unset → `None`).
        sandbox_type: Sandbox backend identifier. Any falsy value (`None`, `""`)
            or `"none"` is treated as local execution.
        local_default: The resolver-backed local-mode default;
            gating on it keeps the advisory quiet for users who disabled the
            interpreter in config.

    Returns:
        `True` when the advisory should be shown, otherwise `False`.
    """
    if enable_interpreter is not None:
        return False
    if not (sandbox_type and sandbox_type != "none"):
        return False
    return local_default


# [해설][설계] ServerConfig: 앱 -> 서버 전달 설정 전체. frozen dataclass라 한 번 만들면
# [해설]   바뀌지 않으며,
# [해설]   변경은 dataclasses.replace로 새 객체를 만든다(resolve_workspace 참고). 필드
# [해설]   하나 = 환경변수 하나(to_env 참고).
# [해설] 아래 필드별 문자열 docstring이 각 필드의 계약이다. 새 필드를 추가하려면
# [해설]   to_env/from_env/from_cli_args와
# [해설] 분류 상수를 모두 맞춰야 classified_config_fields가 RuntimeError를 내지
# [해설]   않는다(tests/unit_tests/test_server_config.py가 호출).
@dataclass(frozen=True)
class ServerConfig:
    """Full configuration payload passed from the app to the server subprocess.

    Serialized to/from `DEEPAGENTS_CODE_SERVER_*` environment variables so
    that the server graph (which runs in a separate Python interpreter)
    can reconstruct the app's intent without sharing memory.
    """

    model: str | None = None
    """Model spec string (e.g. `'anthropic:claude-opus-4-7'`); `None` lets the
    server pick its default."""

    summarization_model: str | None = None
    """Model spec used only for context-compaction summaries.

    `None` reuses the main agent model.
    """

    model_params: dict[str, Any] | None = None
    """Extra kwargs forwarded to the chat model constructor (temperature,
    max_tokens, etc.)."""

    cli_max_retries: int | None = None
    """Explicit `--max-retries` value, separate from provider model kwargs."""

    profile_overrides: dict[str, Any] | None = None
    """Model profile metadata overrides resolved by the client."""

    assistant_id: str = DEFAULT_ASSISTANT_ID
    """Identifier of the agent graph to invoke on the server."""

    system_prompt: str | None = None
    """Override for the agent's system prompt; `None` uses the agent's default."""

    auto_approve: bool = False
    """Auto-approve every tool call without human-in-the-loop interrupts."""

    interrupt_shell_only: bool = False
    """Route only shell tool calls through HITL; validate others via middleware."""

    shell_allow_list: list[str] | None = None
    """Restrictive allow-list of shell commands; `None` disables the allow-list.

    Must be non-empty when set.
    """

    interactive: bool = True
    """Whether the agent runs in an interactive session (vs.
    one-shot/non-interactive)."""

    enable_shell: bool = True
    """Enable the shell execution tool on the server."""

    enable_ask_user: bool = False
    """Enable the `ask_user` tool that lets the agent prompt the user mid-run."""

    enable_memory: bool = True
    """Enable the long-term memory subsystem."""

    enable_skills: bool = True
    """Enable the skills subsystem (SKILL.md loading and skill tools)."""

    enable_interpreter: bool = False
    """Enable `CodeInterpreterMiddleware` (`js_eval` REPL) on the main agent.

    Always the resolved concrete value: `from_cli_args` collapses the tri-state
    caller option via `_resolve_enable_interpreter` before constructing the
    config, so the `bool | None` "defer to default" sentinel never reaches this
    field. The `False` default here is only the bare-constructor/`from_env`
    fallback; the user-facing default (on in local mode) is resolver-backed.

    Local-mode only; the server graph raises if a sandbox is configured and
    this flag is `True`.
    """

    interpreter_ptc: str | list[str] | None = None
    """Invocation-scoped override for `interpreter.ptc`.

    `None` means "fall through to whatever `interpreter.ptc` resolves
    to from `~/.deepagents/config.toml`". A string is one of `"safe"`/`"all"`;
    a list is an explicit allowlist of tool names that may also include the
    `"safe"` preset (expanded at agent-build time); `"all"` is rejected inside
    a list.
    """

    interpreter_ptc_acknowledge_unsafe: bool = False
    """Override for `interpreter.ptc_acknowledge_unsafe` — required when
    `interpreter_ptc="all"` is paired with non-`auto_approve` mode.
    """

    allow_fs_tools: list[FsToolName] | None = None
    """Allowlist for `FilesystemMiddleware`'s `tools` param, from
    `--allow-fs-tools`.

    `None` means "all filesystem tools" and is also what `--allow-fs-tools all`
    parses to: it leaves the SDK's own default `FilesystemMiddleware` in place
    (no replacement). A list is an explicit allowlist of filesystem tool names,
    must include `"read_file"`, and installs a restricted replacement (see
    `create_cli_agent`).
    """

    rubric_model: str | None = None
    """Grader model spec for `RubricMiddleware` (e.g. `'anthropic:...'`).

    `None` reuses the main agent model.
    """

    rubric_max_iterations: int | None = None
    """Explicit grader iterations per rubric attempt; `None` uses the SDK default."""

    auto_classifier_model: str | None = None
    """Classifier model spec for Auto mode (e.g. `'anthropic:claude-haiku-4-5'`).

    `None` falls through to `DEEPAGENTS_CODE_AUTO_CLASSIFIER_MODEL`, then
    `[models].auto_classifier`, and then to the main agent model. An empty value
    round-trips to `None`, so it means "inherit", never "empty spec".
    """

    recursion_limit: int | None = None
    """Explicit main-agent LangGraph `recursion_limit` (graph step budget).

    `None` resolves from runtime configuration. An explicit value from
    `--recursion-limit` wins over the env var and `config.toml`, but managed
    config outranks the flag. Must be a positive integer when set.
    """

    sandbox_type: str | None = None
    """Sandbox backend identifier (e.g. `'daytona'`); `None` runs tools on the
    host. `'none'` is normalized to `None` in `__post_init__`."""

    sandbox_id: str | None = None
    """Existing sandbox ID to attach to; `None` creates a fresh sandbox."""

    sandbox_snapshot_name: str | None = None
    """Sandbox snapshot (langsmith) or blueprint (runloop) name; must be `None`
    when `sandbox_id` is set."""

    sandbox_setup: str | None = None
    """Absolute path to a setup script executed inside the sandbox on first attach."""

    cwd: str | None = None
    """User's original working directory, serialized as an absolute path."""

    project_root: str | None = None
    """Detected project root (e.g. nearest git/uv/npm boundary), or `None` when
    outside a project."""

    mcp_config_path: str | None = None
    """Absolute path to the MCP server config file; `None` disables
    MCP-from-config."""

    no_mcp: bool = False
    """Disable all MCP server connections regardless of other config."""

    trust_project_mcp: bool | None = None
    """Tri-state trust flag for project-scoped MCP servers: `True`/`False`/`None`
    (prompt user)."""

    trust_project_extensions: bool = False
    """Whether the project's Python extensions may execute for this run."""

    extension_paths: tuple[str, ...] = ()
    """Absolute one-run extension files or directories from repeatable CLI flags."""

    # [해설] 서버가 스레드 바인딩에 저장하는 "비밀 없는" 정책 페이로드.
    # [해설]   model_params/system_prompt 같은 민감 값은 의도적으로 빠져 있다.
    # [해설] 이 딕셔너리의 키가 곧 SESSION_WORKSPACE_FIELDS | PROJECT_WORKSPACE_FIELDS와
    # [해설]   같아야 한다.
    def to_workspace_payload(self) -> dict[str, Any]:
        """Return non-secret resource policy for a durable workspace binding."""
        from deepagents_code.configuration.interpreter import InterpreterConfig

        interpreter = InterpreterConfig.from_resolver(
            ptc=self.interpreter_ptc,
            ptc_acknowledge_unsafe=self.interpreter_ptc_acknowledge_unsafe,
        )
        return {
            "assistant_id": self.assistant_id,
            "auto_approve": self.auto_approve,
            "interrupt_shell_only": self.interrupt_shell_only,
            "shell_allow_list": self.shell_allow_list,
            "interactive": self.interactive,
            "enable_shell": self.enable_shell,
            "enable_ask_user": self.enable_ask_user,
            "enable_memory": self.enable_memory,
            "enable_skills": self.enable_skills,
            "enable_interpreter": self.enable_interpreter,
            "interpreter_ptc": interpreter.ptc,
            "interpreter_ptc_acknowledge_unsafe": interpreter.ptc_acknowledge_unsafe,
            "allow_fs_tools": self.allow_fs_tools,
            "recursion_limit": self.recursion_limit,
            "sandbox_type": self.sandbox_type,
            "sandbox_id": self.sandbox_id,
            "sandbox_snapshot_name": self.sandbox_snapshot_name,
            "sandbox_setup": self.sandbox_setup,
            "mcp_config_path": self.mcp_config_path,
            "no_mcp": self.no_mcp,
            "trust_project_mcp": self.trust_project_mcp,
            "trust_project_extensions": self.trust_project_extensions,
            "extension_paths": list(self.extension_paths),
        }

    # [해설] 정책 중 "클라이언트가 자기 명령 호출에 대해 주장해도 되는"
    # [해설]   부분집합(SESSION).
    def to_session_workspace_claim(self) -> dict[str, Any]:
        """Return the command-scoped policy a managed client may claim.

        Returns:
            The session-scoped subset of the workspace policy.
        """
        return {
            key: value
            for key, value in self.to_workspace_payload().items()
            if key in SESSION_WORKSPACE_FIELDS
        }

    # [해설] 정책 중 "서버가 프로젝트 디렉터리마다 직접 해석해야 하는"
    # [해설]   부분집합(PROJECT). 서버는 클라이언트 주장을 받지 않는다.
    def to_project_workspace_policy(self) -> dict[str, Any]:
        """Return policy that must be resolved for each project directory.

        Returns:
            The project-scoped subset of the workspace policy.
        """
        return {
            key: value
            for key, value in self.to_workspace_payload().items()
            if key in PROJECT_WORKSPACE_FIELDS
        }

    # [해설] SESSION 부분집합의 지문. 클라이언트가
    # [해설]   RemoteAgent.set_workspace(config_fingerprint=...)로 서버에 보내
    # [해설] 서버 쪽 값과 대조하는 데 쓴다(server_manager.py의
    # [해설]   start_server_and_get_agent 마지막 부분).
    def session_workspace_fingerprint(self) -> str:
        """Fingerprint the exact client-claimable session policy.

        Returns:
            The canonical SHA-256 fingerprint.
        """
        from deepagents_code.workspace import canonical_fingerprint

        return canonical_fingerprint(self.to_session_workspace_claim())

    # [해설][설계] 서버가 "다른 작업디렉터리"를 호스팅할 때 프로젝트 정책을 어떻게
    # [해설]   다룰지 정한다.
    # [해설][흐름] 1) 시작 프로젝트와 같으면 모든 정책을 그대로 유지 2) 다르면
    # [해설]   MCP·샌드박스 setup·확장 경로를 비우고
    # [해설][흐름]    확장 신뢰는 trust store에서 다시 읽는다. 한 프로젝트의 신뢰 결정이
    # [해설]   다른 체크아웃에 번지지 않게 하는 장치다.
    def resolve_workspace(
        self,
        cwd: str,
        project_root: str | None,
    ) -> ServerConfig:
        """Resolve directory-bound policy for one server workspace.

        Project-scoped policy (`PROJECT_WORKSPACE_FIELDS`) is valid only for
        the directory it was resolved against: it came from the launch-time CLI
        and that project's trust decisions. Reusing it for another directory
        would apply one project's MCP servers, sandbox setup, and extensions to
        a different, possibly untrusted, checkout.

        So the launch project keeps its policy verbatim, and any other project
        starts from nothing: MCP and sandbox setup are *dropped* rather than
        rediscovered, and extension trust is re-read from the trust store for
        that project. `_same_workspace_project` fails closed, so an
        unresolvable path also takes the drop branch.

        Args:
            cwd: Absolute, canonical working directory for the workspace.
            project_root: Canonical project root, or `None` when the workspace
                has none. The launch cwd uses the server's explicit root when
                configured. Otherwise, extension trust is keyed on `cwd` when
                no root exists.

        Returns:
            A config whose session policy is unchanged and whose project policy
            is either the launch project's or empty.
        """
        if self.project_root is not None and _same_workspace_project(self.cwd, cwd):
            project_root = str(Path(self.project_root).expanduser().resolve())
        launch_root = self.project_root or self.cwd
        target_root = project_root or cwd
        if _same_workspace_project(launch_root, target_root):
            return replace(self, cwd=cwd, project_root=project_root)
        from deepagents_code.extensions.trust import is_project_extensions_trusted

        return replace(
            self,
            cwd=cwd,
            project_root=project_root,
            sandbox_setup=None,
            mcp_config_path=None,
            trust_project_mcp=None,
            trust_project_extensions=is_project_extensions_trusted(target_root),
            extension_paths=(),
        )

    # [해설] 이미 묶인 스레드에서 새로 생긴 확장 신뢰 부여는 무시(새 스레드부터
    # [해설]   적용)하고, 취소(True -> False)는 그대로 드러내
    # [해설] 검증 단계가 거부할 수 있게 한다.
    def preserve_bound_extension_trust(
        self, bound_policy: Mapping[str, object]
    ) -> ServerConfig:
        """Keep an existing thread's extension trust when a new grant appears.

        Args:
            bound_policy: Server policy persisted when the thread was bound.

        Returns:
            A config that defers new grants to new threads. Revocations remain
            visible so binding and runtime validation can reject them.
        """
        if bound_policy.get("trust_project_extensions") is False and (
            self.trust_project_extensions is True
        ):
            return replace(self, trust_project_extensions=False)
        return self

    # [해설][설계] 지문 3종 정리. policy_fingerprint = 접근 정책 +
    # [해설]   cwd/project_root(바인딩의 지속 기준, 모델 변경에 안 바뀜),
    # [해설]   runtime_fingerprint = to_env() 전체에서 CWD/PROJECT_ROOT를 뺀
    # [해설]   값(모델/프롬프트까지 포함, 런타임 캐시 키),
    # [해설]   workspace_fingerprint = runtime_fingerprint의 옛 이름(호환용). 모델만
    # [해설]   바꾸면 런타임만 새로 빌드되고 바인딩은 유지된다.
    def policy_fingerprint(self) -> str:
        """Fingerprint durable access policy plus workspace identity.

        Covers the trust/tool/sandbox/approval payload (the
        `to_workspace_payload()` keys) and the workspace's `cwd`/`project_root`.
        Cosmetic model settings (`MODEL_COMPATIBLE_FIELDS`) are excluded so a
        model switch does not invalidate a durable binding; runtime-only fields
        (`RUNTIME_ONLY_FIELDS`) are excluded because they are not persisted.

        Returns:
            The canonical SHA-256 fingerprint.
        """
        from deepagents_code.workspace import canonical_fingerprint

        return canonical_fingerprint(
            {
                "cwd": self.cwd,
                "policy": self.to_workspace_payload(),
                "project_root": self.project_root,
            }
        )

    def runtime_fingerprint(self) -> str:
        """Fingerprint the full runtime identity, including model settings.

        Any change to model, model params, prompt, policy, or runtime fields
        yields a different value, so the runtime cache can rebuild on any of
        them while the binding only guards the policy fingerprint.

        Returns:
            The canonical SHA-256 fingerprint.
        """
        values = self.to_env()
        values.pop("CWD")
        values.pop("PROJECT_ROOT")
        from deepagents_code.workspace import canonical_fingerprint

        return canonical_fingerprint(values)

    def workspace_fingerprint(self) -> str:
        """Fingerprint the resolved runtime config except workspace identity.

        Retained for compatibility; equivalent to `runtime_fingerprint()`.

        Returns:
            The canonical SHA-256 fingerprint.
        """
        return self.runtime_fingerprint()

    # [해설] 불변식 검증. 보안 통제인 allow_fs_tools(비어 있지 않고 read_file 포함)와
    # [해설]   숫자 필드(bool 금지, 양수/음수 범위)를
    # [해설] 생성 시점에 막는다. from_env 경유든 직접 생성이든 이 한 곳을 지난다.
    def __post_init__(self) -> None:
        """Normalize fields and validate invariants.

        Raises:
            TypeError: If `rubric_max_iterations` or `recursion_limit` is a
                boolean.
            ValueError: If `shell_allow_list` is an empty list,
                `allow_fs_tools` is an empty list or omits `"read_file"`, or
                `rubric_max_iterations` / `recursion_limit` is non-positive.
        """
        if self.sandbox_type == "none":
            object.__setattr__(self, "sandbox_type", None)
        if self.shell_allow_list is not None and len(self.shell_allow_list) == 0:
            msg = "shell_allow_list must be None or non-empty"
            raise ValueError(msg)
        # `allow_fs_tools` is a security control: `None` means unrestricted, but
        # an explicit list must be a usable allowlist. Own the non-empty +
        # `read_file`-required invariant here (the single authoritative point
        # for both the env round-trip via `from_env` and direct construction)
        # rather than deferring to `FilesystemMiddleware`, which would only
        # surface the violation a process boundary away. `_parse_allow_fs_tools_flag`
        # still enforces the same rule at the CLI for a friendlier error.
        if self.allow_fs_tools is not None:
            if len(self.allow_fs_tools) == 0:
                msg = "allow_fs_tools must be None or a non-empty list"
                raise ValueError(msg)
            if "read_file" not in self.allow_fs_tools:
                msg = "allow_fs_tools must include 'read_file'"
                raise ValueError(msg)
        if isinstance(self.rubric_max_iterations, bool):
            msg = "rubric_max_iterations must be None or a positive integer"
            raise TypeError(msg)
        if self.rubric_max_iterations is not None and self.rubric_max_iterations <= 0:
            msg = "rubric_max_iterations must be None or a positive integer"
            raise ValueError(msg)
        if isinstance(self.cli_max_retries, bool):
            msg = "cli_max_retries must be None or a non-negative integer"
            raise TypeError(msg)
        if self.cli_max_retries is not None and self.cli_max_retries < 0:
            msg = "cli_max_retries must be None or a non-negative integer"
            raise ValueError(msg)
        if isinstance(self.recursion_limit, bool):
            msg = "recursion_limit must be None or a positive integer"
            raise TypeError(msg)
        if self.recursion_limit is not None and self.recursion_limit <= 0:
            msg = "recursion_limit must be None or a positive integer"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    # [해설][흐름] 클라이언트 -> 서버 방향의 직렬화. 값이 None이면 "변수를 지워라"는
    # [해설]   신호이므로 호출 쪽(server_manager.py의
    # [해설]   _set_or_clear_server_env)이 os.environ에서 pop한다(빈 문자열로 두지
    # [해설]   않음). bool은 "true"/"false" 문자열,
    # [해설]   리스트/딕셔너리는 JSON, SHELL_ALLOW_LIST만 쉼표 구분이다.
    def to_env(self) -> dict[str, str | None]:
        """Serialize this config to a `DEEPAGENTS_CODE_SERVER_*` env-var mapping.

        `None` values signal that the variable should be *cleared* from the
        environment (rather than set to an empty string), so callers can
        iterate and set or clear each variable in `os.environ`.

        Returns:
            Dict mapping env-var suffixes (without the prefix) to their
                string values or `None`.
        """
        return {
            "MODEL": self.model,
            "SUMMARIZATION_MODEL": self.summarization_model,
            "MODEL_PARAMS": (
                json.dumps(self.model_params) if self.model_params is not None else None
            ),
            "MAX_RETRIES": (
                str(self.cli_max_retries) if self.cli_max_retries is not None else None
            ),
            "PROFILE_OVERRIDES": (
                json.dumps(self.profile_overrides)
                if self.profile_overrides is not None
                else None
            ),
            "ASSISTANT_ID": self.assistant_id,
            "SYSTEM_PROMPT": self.system_prompt,
            "AUTO_APPROVE": str(self.auto_approve).lower(),
            "INTERRUPT_SHELL_ONLY": str(self.interrupt_shell_only).lower(),
            "SHELL_ALLOW_LIST": (
                ",".join(self.shell_allow_list)
                if self.shell_allow_list is not None
                else None
            ),
            "INTERACTIVE": str(self.interactive).lower(),
            "ENABLE_SHELL": str(self.enable_shell).lower(),
            "ENABLE_ASK_USER": str(self.enable_ask_user).lower(),
            "ENABLE_MEMORY": str(self.enable_memory).lower(),
            "ENABLE_SKILLS": str(self.enable_skills).lower(),
            "ENABLE_INTERPRETER": str(self.enable_interpreter).lower(),
            "INTERPRETER_PTC": (
                json.dumps(self.interpreter_ptc)
                if self.interpreter_ptc is not None
                else None
            ),
            "INTERPRETER_PTC_ACKNOWLEDGE_UNSAFE": str(
                self.interpreter_ptc_acknowledge_unsafe
            ).lower(),
            "ALLOW_FS_TOOLS": (
                json.dumps(self.allow_fs_tools)
                if self.allow_fs_tools is not None
                else None
            ),
            "RUBRIC_MODEL": self.rubric_model,
            "AUTO_CLASSIFIER_MODEL": self.auto_classifier_model,
            "RUBRIC_MAX_ITERATIONS": (
                str(self.rubric_max_iterations)
                if self.rubric_max_iterations is not None
                else None
            ),
            "RECURSION_LIMIT": (
                str(self.recursion_limit) if self.recursion_limit is not None else None
            ),
            "SANDBOX_TYPE": self.sandbox_type,
            "SANDBOX_ID": self.sandbox_id,
            "SANDBOX_SNAPSHOT_NAME": self.sandbox_snapshot_name,
            "SANDBOX_SETUP": self.sandbox_setup,
            "CWD": self.cwd,
            "PROJECT_ROOT": self.project_root,
            "MCP_CONFIG_PATH": self.mcp_config_path,
            "NO_MCP": str(self.no_mcp).lower(),
            "TRUST_PROJECT_MCP": (
                str(self.trust_project_mcp).lower()
                if self.trust_project_mcp is not None
                else None
            ),
            "TRUST_PROJECT_EXTENSIONS": str(self.trust_project_extensions).lower(),
            "EXTENSION_PATHS": (
                json.dumps(self.extension_paths) if self.extension_paths else None
            ),
        }

    # [해설][흐름] 서버 -> 복원 방향. to_env의 역함수이며 서버 프로세스 안에서
    # [해설]   호출된다(server_graph.py의 _make_graphs,
    # [해설]   _default_workspace_binding, _resolve_bound_workspace_config). 기본값은
    # [해설]   필드 정의와 같게 맞춰져 있다.
    @classmethod
    def from_env(cls) -> ServerConfig:
        """Reconstruct a `ServerConfig` from `DEEPAGENTS_CODE_SERVER_*` env vars.

        This is the inverse of `to_env()` and is called inside the server
        subprocess to recover the app's configuration.

        Returns:
            A `ServerConfig` populated from the environment.
        """
        return cls(
            model=_read_env_str("MODEL"),
            summarization_model=_read_env_str("SUMMARIZATION_MODEL") or None,
            model_params=_read_env_json("MODEL_PARAMS"),
            cli_max_retries=_read_env_int("MAX_RETRIES", default=None),
            profile_overrides=_read_env_json("PROFILE_OVERRIDES"),
            assistant_id=_read_env_str("ASSISTANT_ID") or DEFAULT_ASSISTANT_ID,
            system_prompt=_read_env_str("SYSTEM_PROMPT"),
            auto_approve=_read_env_bool("AUTO_APPROVE"),
            interrupt_shell_only=_read_env_bool("INTERRUPT_SHELL_ONLY"),
            shell_allow_list=(
                [cmd.strip() for cmd in raw.split(",") if cmd.strip()]
                if (raw := _read_env_str("SHELL_ALLOW_LIST"))
                else None
            )
            or None,
            interactive=_read_env_bool("INTERACTIVE", default=True),
            enable_shell=_read_env_bool("ENABLE_SHELL", default=True),
            enable_ask_user=_read_env_bool("ENABLE_ASK_USER"),
            enable_memory=_read_env_bool("ENABLE_MEMORY", default=True),
            enable_skills=_read_env_bool("ENABLE_SKILLS", default=True),
            enable_interpreter=_read_env_bool("ENABLE_INTERPRETER"),
            interpreter_ptc=_read_env_json("INTERPRETER_PTC"),
            interpreter_ptc_acknowledge_unsafe=_read_env_bool(
                "INTERPRETER_PTC_ACKNOWLEDGE_UNSAFE"
            ),
            allow_fs_tools=_read_env_allow_fs_tools(),
            rubric_model=_read_env_str("RUBRIC_MODEL") or None,
            auto_classifier_model=_read_env_str("AUTO_CLASSIFIER_MODEL") or None,
            rubric_max_iterations=_read_env_int("RUBRIC_MAX_ITERATIONS", default=None),
            recursion_limit=_read_env_int("RECURSION_LIMIT", default=None),
            sandbox_type=_read_env_str("SANDBOX_TYPE"),
            sandbox_id=_read_env_str("SANDBOX_ID"),
            sandbox_snapshot_name=_read_env_str("SANDBOX_SNAPSHOT_NAME") or None,
            sandbox_setup=_read_env_str("SANDBOX_SETUP"),
            cwd=_read_env_str("CWD"),
            project_root=_read_env_str("PROJECT_ROOT"),
            mcp_config_path=_read_env_str("MCP_CONFIG_PATH"),
            no_mcp=_read_env_bool("NO_MCP"),
            trust_project_mcp=_read_env_optional_bool("TRUST_PROJECT_MCP"),
            trust_project_extensions=_read_env_bool("TRUST_PROJECT_EXTENSIONS"),
            extension_paths=_read_env_str_list("EXTENSION_PATHS"),
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    # [해설] 클라이언트가 파싱한 CLI 인자 -> ServerConfig. 상대 경로(MCP 설정, 샌드박스
    # [해설]   setup, 확장)는 사용자의 원래 cwd 기준으로
    # [해설] 절대 경로로 바꿔 둔다. 서버 프로세스는 임시 디렉터리를 cwd로 쓰기 때문이다.
    @classmethod
    def from_cli_args(
        cls,
        *,
        project_context: ProjectContext | None,
        model_name: str | None,
        summarization_model: str | None = None,
        model_params: dict[str, Any] | None,
        cli_max_retries: int | None = None,
        profile_overrides: dict[str, Any] | None = None,
        assistant_id: str,
        auto_approve: bool,
        interrupt_shell_only: bool = False,
        shell_allow_list: list[str] | None = None,
        sandbox_type: str = "none",
        sandbox_id: str | None,
        sandbox_snapshot_name: str | None,
        sandbox_setup: str | None,
        enable_shell: bool,
        enable_ask_user: bool,
        enable_interpreter: bool | None = None,
        interpreter_ptc: str | list[str] | None = None,
        interpreter_ptc_acknowledge_unsafe: bool = False,
        allow_fs_tools: list[FsToolName] | None = None,
        rubric_model: str | None = None,
        rubric_max_iterations: int | None = None,
        auto_classifier_model: str | None = None,
        recursion_limit: int | None = None,
        mcp_config_path: str | None,
        no_mcp: bool,
        trust_project_mcp: bool | None,
        interactive: bool,
        trust_project_extensions: bool = False,
        extension_paths: tuple[str, ...] = (),
    ) -> ServerConfig:
        """Build a `ServerConfig` from parsed CLI arguments.

        Handles path normalization (e.g. resolving relative MCP config paths
        against the user's working directory) so that the raw serialized values
        are always absolute and unambiguous.

        Args:
            project_context: Explicit user/project path context.
            model_name: Model spec string.
            summarization_model: Model spec used only for context-compaction
                summaries; `None` reuses the main model.
            model_params: Extra model kwargs.
            cli_max_retries: Explicit `--max-retries` value.
            profile_overrides: Model profile metadata overrides.
            assistant_id: Agent identifier.
            auto_approve: Auto-approve all tools.
            interrupt_shell_only: Validate shell commands via middleware instead
                of HITL.
            shell_allow_list: Restrictive shell allow-list to forward to the
                server subprocess for `ShellAllowListMiddleware`.
            sandbox_type: Sandbox type.
            sandbox_id: Existing sandbox ID to reuse.
            sandbox_snapshot_name: Snapshot (langsmith) or blueprint (runloop)
                name to use or create.
            sandbox_setup: Path to setup script for the sandbox.
            enable_shell: Enable shell execution tools.
            enable_ask_user: Enable ask_user tool.
            enable_interpreter: Enable `CodeInterpreterMiddleware` on the main
                agent. `None` uses the sandbox-aware default.
            interpreter_ptc: Invocation-scoped PTC allowlist override.
            interpreter_ptc_acknowledge_unsafe: Explicit acknowledgement for
                an invocation-scoped `interpreter_ptc="all"`.
            allow_fs_tools: Allowlist for `FilesystemMiddleware`'s `tools`
                param to forward to the server subprocess. `None` leaves the
                SDK default (all tools).
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
            interactive: Whether the agent is interactive.
            trust_project_extensions: Allow project extension execution.
            extension_paths: Explicit one-run extension files or directories.

        Returns:
            A fully resolved `ServerConfig`.
        """
        normalized_mcp = _normalize_path(mcp_config_path, project_context, "MCP config")

        resolved_enable_interpreter = _resolve_enable_interpreter(
            enable_interpreter, sandbox_type
        )

        return cls(
            model=model_name,
            summarization_model=summarization_model,
            model_params=model_params,
            cli_max_retries=cli_max_retries,
            profile_overrides=profile_overrides,
            assistant_id=assistant_id,
            auto_approve=auto_approve,
            interrupt_shell_only=interrupt_shell_only,
            shell_allow_list=shell_allow_list,
            interactive=interactive,
            enable_shell=enable_shell,
            enable_ask_user=enable_ask_user,
            enable_interpreter=resolved_enable_interpreter,
            interpreter_ptc=interpreter_ptc,
            interpreter_ptc_acknowledge_unsafe=interpreter_ptc_acknowledge_unsafe,
            allow_fs_tools=allow_fs_tools,
            rubric_model=rubric_model,
            rubric_max_iterations=rubric_max_iterations,
            auto_classifier_model=auto_classifier_model,
            recursion_limit=recursion_limit,
            sandbox_type=sandbox_type,
            sandbox_id=sandbox_id,
            sandbox_snapshot_name=sandbox_snapshot_name,
            sandbox_setup=_normalize_path(
                sandbox_setup, project_context, "sandbox setup"
            ),
            cwd=(
                str(project_context.user_cwd) if project_context is not None else None
            ),
            project_root=(
                str(project_context.project_root)
                if project_context is not None
                and project_context.project_root is not None
                else None
            ),
            mcp_config_path=normalized_mcp,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
            trust_project_extensions=trust_project_extensions,
            extension_paths=tuple(
                path
                for raw in extension_paths
                if (path := _normalize_path(raw, project_context, "extension"))
            ),
        )


# [해설] 상대 경로를 절대 경로로 변환. 서버의 cwd가 사용자 cwd와 다르므로 직렬화 전에
# [해설]   반드시 해야 하는 단계다.
def _normalize_path(
    raw_path: str | None,
    project_context: ProjectContext | None,
    label: str,
) -> str | None:
    """Resolve a possibly-relative path to absolute.

    The server subprocess runs in a different working directory, so relative
    paths must be resolved against the user's original cwd before serialization.

    Args:
        raw_path: Path from CLI arguments (may be relative).
        project_context: User/project context for path resolution.
        label: Human-readable label for error messages (e.g. "MCP config").

    Returns:
        Absolute path string, or `None` when *raw_path* is `None` or empty.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    if not raw_path:
        return None
    try:
        if project_context is not None:
            return str(project_context.resolve_user_path(raw_path))
        return str(Path(raw_path).expanduser().resolve())
    except OSError as exc:
        msg = (
            f"Could not resolve {label} path {raw_path!r}: {exc}. "
            "Ensure the path exists and is accessible."
        )
        raise ValueError(msg) from exc


# [해설] 모든 필드가 정확히 한 분류(model/runtime/identity/policy)에 속하는지 검사하는
# [해설]   자기 점검 함수(테스트가 호출). 새 필드를 분류하지
# [해설] 않으면 RuntimeError로 실패해, 정책에서 조용히 빠지는 설정이 생기는 것을 막는다.
def classified_config_fields() -> dict[str, str]:
    """Classify every `ServerConfig` field for the policy/runtime split.

    Every dataclass field must land in exactly one bucket so the fingerprint
    split cannot silently drop a setting. A field missing from a bucket is a
    programming error and fails closed (raises) rather than being treated as
    either policy or runtime by default.

    Returns:
        A mapping of field name to one of `"model"`, `"runtime"`,
        `"identity"`, or `"policy"`.

    Raises:
        RuntimeError: If a `ServerConfig` field is not classified.
    """
    import dataclasses

    policy_fields = set(ServerConfig().to_workspace_payload())
    classification: dict[str, str] = {}
    for field_info in dataclasses.fields(ServerConfig):
        name = field_info.name
        if name in MODEL_COMPATIBLE_FIELDS:
            classification[name] = "model"
        elif name in RUNTIME_ONLY_FIELDS:
            classification[name] = "runtime"
        elif name in WORKSPACE_IDENTITY_FIELDS:
            classification[name] = "identity"
        elif name in policy_fields:
            classification[name] = "policy"
        else:
            msg = f"ServerConfig field {name!r} is not classified"
            raise RuntimeError(msg)
    return classification
