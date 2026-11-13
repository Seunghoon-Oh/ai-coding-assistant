"""Middleware for runtime model selection via LangGraph runtime context.

Allows switching the model per invocation by passing a `CLIContext` via
`context=` on `agent.astream()` / `agent.invoke()` without recompiling
the graph.
"""

# [해설] 이 파일은 '실행 중 모델 교체' 미들웨어(ConfigurableModelMiddleware) 한 개와 그
# [해설]   보조 함수들입니다. 그래프를 다시 컴파일하지 않고 호출마다 모델을 바꾸는 것이
# [해설]   목적입니다.
# [해설][흐름] 1) 클라이언트(/model, --model-params)가 agent.astream(...,
# [해설]   context=CLIContext 페이로드)로 model·model_params를 실어 보낸다.
# [해설][흐름] 2) 서버 쪽 wrap_model_call/awrap_model_call이 _get_context로
# [해설]   runtime.context(CLIContextSchema)를 읽는다.
# [해설][흐름] 3) 요청된 spec이 현재 모델과 다르면 config.py의 create_model로 새 모델을
# [해설]   만들어 request.model을 교체한다(_apply_overrides -> _build_overrides).
# [해설][흐름] 4) 교체 후 model_settings 병합, 공급자별 캐시 힌트 주입, Anthropic 전용
# [해설]   설정 제거, 시스템 프롬프트의 Model Identity 갱신을 한다.
# [해설][흐름] 5) 호출이 끝나면 _checkpoint_command가 모델 spec·캐시 상태를
# [해설]   체크포인트(Command update)로 남겨 재개(resume) 때 쓰게 한다.
# [해설][설계] 이 미들웨어는 가장 바깥층에 둔다. Anthropic 프롬프트 캐싱 같은 공급자별
# [해설]   미들웨어가 실행되기 전에 모델을 먼저 확정하기 위해서입니다.
# [해설][주의] 서버 서브프로세스에서 돌기 때문에 프로세스 전역 runtime_state는 /model로
# [해설]   갱신되지 않는다. 그래서 메타데이터는 항상 ModelResult에서 읽는다.
# [해설][SDK] 모델 일치 판정(model_matches_spec)과 식별자 추출(get_model_identifier)은
# [해설]   SDK의 deepagents/_models.py에 있습니다.
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from deepagents._models import (  # noqa: PLC2701
    get_model_identifier,
    model_matches_spec,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    TracePolicy,
    omit_payload,
)
from langchain_core.messages import AIMessage
from langgraph.types import Command

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.cold_cache import (
    CacheActivity,
    cache_identity_params,
    cache_request,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel

    from deepagents_code.config import ModelResult


logger = logging.getLogger(__name__)


# [해설] 오버라이드를 적용한 최종 요청(request)과 '체크포인트에 남길
# [해설]   메타데이터'(model_spec, model_params)를 묶은 불변 값 객체.
# [해설] model_params_known이 False면 '어떤 파라미터가 적용 중인지 모른다'는 뜻이라
# [해설]   체크포인트의 params를 지우지 않는다.
@dataclass(frozen=True)
class _ResolvedModelRequest:
    """Model request plus the checkpoint metadata it should persist."""

    request: ModelRequest
    """Request to pass to the downstream model handler."""

    model_spec: str | None
    """Resolved `provider:model` spec to persist for resume, when known."""

    model_params: dict[str, Any] | None = None
    """Invocation params to persist, or `None` to clear checkpointed params."""

    model_params_known: bool = False
    """Whether `model_params` is known and should be written to the checkpoint."""


# [해설] 프롬프트 캐시가 어느 엔드포인트(base_url)에 속하는지 식별자를 계산한다.
# [해설]   엔드포인트가 바뀌면 캐시가 식어 있을 수 있어 비교용으로 체크포인트에
# [해설]   기록한다.
# [해설][주의] config.toml/자격 증명 저장소를 동기 I/O로 읽으므로, 비동기
# [해설]   경로(awrap_model_call)는 반드시 asyncio.to_thread로 호출한다. 실패해도 예외를
# [해설]   내지 않고 공급자 기본 식별자로 대체한다.
def _cache_endpoint_identity(
    model_spec: str | None, model_params: Mapping[str, Any] | None = None
) -> str:
    """Resolve the endpoint identity used by a model request for checkpointing.

    Performs blocking filesystem reads: `ModelConfig.load()` is process-cached,
    but `get_base_url` falls through to the credential store for any provider
    with no `config.toml` `base_url` and no base-URL env var *set* -- the
    default for `anthropic` and `openai`, whose env vars are registered in
    `PROVIDER_BASE_URL_ENV` but normally unset -- and that store re-reads its
    file every call. Callers on the blockbuster-guarded server loop must
    therefore invoke this via `asyncio.to_thread` (see
    `ConfigurableModelMiddleware.awrap_model_call`).

    A spec with no `provider:` prefix cannot name an endpoint, so it resolves to
    the same identity as the provider default. That is preferable to returning
    nothing: `_last_cache_endpoint` is written unconditionally alongside the
    spec and timestamp it describes, and a skipped write would leave the
    previous turn's endpoint paired with this turn's spec.

    The provider is normalized exactly as the reader normalizes it
    (`app._cold_cache_warning_for`). `get_kwargs`/`get_base_url` are
    exact-key lookups, so a spec the user spelled `Anthropic:claude-opus-5`
    would resolve `default` here while the reader resolved the real endpoint --
    a disagreement that never self-heals, because both sides keep recomputing
    their own answer, and every send would report `identity_changed`.

    Never raises. Cache identity must not prevent a model request. A malformed
    (a non-string `base_url` from a `class_path` provider that ignores it, say)
    endpoint must not turn a diagnostic into a failed model call. Failing to
    the provider default is the same degradation an unreadable checkpointed
    endpoint already gets.

    Returns:
        The normalized endpoint identity, or the provider-default identity when
        it cannot be resolved.
    """
    from deepagents_code.cold_cache import endpoint_cache_identity

    if not model_spec or ":" not in model_spec:
        return endpoint_cache_identity(None)
    from deepagents_code.model_config import ModelConfig

    raw_provider, _, model_name = model_spec.partition(":")
    provider = raw_provider.strip().lower()
    try:
        config = ModelConfig.load()
        kwargs = config.get_effective_kwargs(
            provider,
            model_name=model_name,
            overrides=model_params,
        )
        base_url = (
            kwargs.get("base_url")
            if isinstance(kwargs, dict)
            else config.get_base_url(provider)
        )
    except Exception:
        logger.warning(
            "Could not resolve the cache endpoint for %r; recording the "
            "provider default, so an endpoint change may go undetected for "
            "this turn",
            provider,
            exc_info=True,
        )
        return endpoint_cache_identity(None)
    return endpoint_cache_identity(base_url if isinstance(base_url, str) else None)


# [해설] LangChain 모델의 _get_ls_params()에서 ls_provider(공급자 이름)를 읽는다. 모델
# [해설]   클래스 종류를 import하지 않고도 공급자를 판별하는 방법입니다.
def _get_ls_provider(model: object) -> str | None:
    """Return the LangSmith provider name reported by a chat model.

    Returns:
        The `ls_provider` string when the model reports one, otherwise `None`
            (including when `_get_ls_params` is missing, raises, or yields a
            non-string provider).
    """
    try:
        ls_params = model._get_ls_params()  # ty: ignore[unresolved-attribute]
    except AttributeError, TypeError, RuntimeError, NotImplementedError:
        logger.debug("_get_ls_params raised for %s", type(model).__name__)
        return None
    if isinstance(ls_params, dict):
        provider = ls_params.get("ls_provider")
        if isinstance(provider, str):
            return provider
    return None


# [해설] 교체된 모델이 Anthropic이 아니면 Anthropic 전용 설정(cache_control)을 제거해야
# [해설]   하므로 그 판정에 쓴다.
def _is_anthropic_model(model: object) -> bool:
    """Check whether a resolved model reports `'anthropic'` as its provider.

    Uses `_get_ls_params` from `BaseChatModel` to read the provider name.

    Args:
        model: A model instance to inspect.

            Typed as `object` (rather than `BaseChatModel`) so the caller can
            pass any model without an import-time dependency on a specific
            provider package.

    Returns:
        `True` if the model's `ls_provider` is `'anthropic'`.
    """
    return _get_ls_provider(model) == "anthropic"


def _is_fireworks_model(model: object) -> bool:
    """Check whether a resolved model reports `'fireworks'` as its provider.

    Returns:
        `True` if the model's `ls_provider` is `'fireworks'`.
    """
    return _get_ls_provider(model) == "fireworks"


def _is_openai_model(model: object) -> bool:
    """Check whether a resolved model targets OpenAI's chat/responses API.

    `prompt_cache_key` is an optional, additive OpenAI request field, so it is
    attempted for every model whose LangSmith provider is `'openai'` regardless
    of base URL. `ChatOpenAI` reports `'openai'` for the official API, the
    LangSmith gateway, and other OpenAI-compatible endpoints alike; treating all
    of them as eligible is intentional so the cache-key optimization is not
    silently dropped behind a proxy. Endpoints that reject unknown request
    fields can opt out via the `models.openai_prompt_cache_key` config option.

    Returns:
        `True` if the model reports `'openai'` as its provider.
    """
    return _get_ls_provider(model) == "openai"


# [해설][주의] 교차 공급자 교체 때 지워야 하는 키. 예를 들어 cache_control을 OpenAI
# [해설]   SDK에 넘기면 TypeError가 난다.
_ANTHROPIC_ONLY_SETTINGS: set[str] = {"cache_control"}
"""Keys injected by Anthropic-specific middleware (e.g.
`AnthropicPromptCachingMiddleware`) that are not accepted by other providers and
must be stripped on cross-provider swap."""

_FIREWORKS_SESSION_AFFINITY_HEADER = "x-session-affinity"
"""Fireworks prompt-cache affinity header populated from the active thread ID."""


def _has_header(headers: Mapping[object, object], target: str) -> bool:
    """Return whether a headers mapping already includes `target`.

    Comparison is case-insensitive; `target` must be supplied in lowercase.

    Returns:
        `True` if a string key case-insensitively equal to `target` is present.
    """
    return any(isinstance(key, str) and key.lower() == target for key in headers)


# [해설] Fireworks의 프롬프트 캐시 세션 라우팅을 위해 thread_id를 prompt_cache_key와
# [해설]   x-session-affinity 헤더로 넣는다. 사용자가 이미 넣은 값은 덮지 않는다.
def _with_fireworks_session_settings(
    model_settings: dict[str, Any], thread_id: str
) -> dict[str, Any] | None:
    """Return model settings with Fireworks session settings added if needed.

    Existing settings are preserved and never overwritten. Missing
    `x-session-affinity` headers are populated directly so Fireworks can route
    the conversation to the prompt-cache session for the active thread.

    Returns:
        A new `model_settings` dict with the missing session settings added, or
            `None` when nothing needed adding or `extra_headers` is present but
            not a mapping (leaving the request untouched).
    """
    raw_headers = model_settings.get("extra_headers")
    if raw_headers is None:
        headers: dict[object, object] = {}
    elif isinstance(raw_headers, Mapping):
        headers = dict(raw_headers)
    else:
        logger.warning(
            "Cannot inject Fireworks session settings because extra_headers is %s",
            type(raw_headers).__name__,
        )
        return None

    updated: dict[str, Any] = {}
    has_session_affinity = _has_header(headers, _FIREWORKS_SESSION_AFFINITY_HEADER)
    if "prompt_cache_key" not in model_settings and not has_session_affinity:
        updated["prompt_cache_key"] = thread_id

    if not has_session_affinity:
        headers[_FIREWORKS_SESSION_AFFINITY_HEADER] = thread_id
        updated["extra_headers"] = headers

    if not updated:
        return None
    return {**model_settings, **updated}


# [해설] OpenAI 요청에 thread_id를 prompt_cache_key로 넣는다(스레드별 캐시 적중률 향상).
# [해설]   모델 kwargs나 호출 설정에 이미 있으면 그대로 둔다.
def _with_openai_prompt_cache_key(
    model: object, model_settings: dict[str, Any], thread_id: str
) -> dict[str, Any] | None:
    """Return model settings with an OpenAI `prompt_cache_key` added if needed.

    Adds `thread_id` as a top-level `prompt_cache_key` when the model and the
    current invocation settings do not already carry one. Callers decide
    eligibility (provider check + `models.openai_prompt_cache_key` opt-out)
    before invoking this helper.

    A user-supplied `prompt_cache_key` is always preserved, whether it was
    configured on the model (`model_kwargs`) or supplied for this invocation
    (`model_settings`).

    Returns:
        A new `model_settings` dict with `prompt_cache_key` added, or `None` when
            a key is already present on the model or in the settings (nothing to
            add).
    """
    model_kwargs = getattr(model, "model_kwargs", None)
    if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
        # A non-mapping `model_kwargs` cannot carry a user-supplied key, so it is
        # treated as "no key present" and injection proceeds. Trace the anomaly
        # since a real `ChatOpenAI` always exposes a mapping here.
        logger.debug(
            "Ignoring non-mapping model_kwargs (%s) when checking for a "
            "user-supplied prompt_cache_key",
            type(model_kwargs).__name__,
        )
    if "prompt_cache_key" in model_settings or (
        isinstance(model_kwargs, Mapping) and "prompt_cache_key" in model_kwargs
    ):
        return None
    return {**model_settings, "prompt_cache_key": thread_id}


# [해설] models.openai_prompt_cache_key 옵트아웃(기본 켬)을 읽는다. 미들웨어 생성 때 한
# [해설]   번만 호출한다.
# [해설][주의] 읽기 실패는 '켬'으로 대체하되 BlockingError(이벤트 루프에서 블로킹 I/O)는
# [해설]   숨기지 않고 다시 던진다.
def _resolve_openai_prompt_cache_key_enabled() -> bool:
    """Resolve the `models.openai_prompt_cache_key` opt-out (default on).

    Called once when `ConfigurableModelMiddleware` is constructed. The read is
    kept off the blockbuster-guarded server loop by the caller: on the server
    path `create_cli_agent` runs inside `asyncio.to_thread` (see
    `server_graph._make_graphs`), so the synchronous `config.toml` read happens
    on a worker thread.

    On an unexpected failure this defaults to enabled: breaking agent
    construction over a config hiccup is worse than injecting the key, and the
    ordinary failure modes (a missing or corrupt `config.toml`) are already
    absorbed by `load_config_toml`. The trade-off is real, not cosmetic — a user
    who opted out *because their endpoint 400s on unknown request fields* would
    then see that per-request failure rather than a benign extra key — so the
    fallback logs at `warning` (not `debug`) to leave a breadcrumb.

    `BlockingError` is deliberately excluded from the fail-open: it signals a
    real blocking-I/O-on-the-event-loop regression (construction moved back onto
    the guarded loop), and swallowing it would mask that bug *and* silently
    defeat the opt-out. It is re-raised so the violation surfaces loudly. It is
    matched by class name because `blockbuster` is not a runtime dependency of
    this package (it is supplied by the langgraph runtime), so it cannot be
    imported here for an `isinstance` check.

    Returns:
        `True` when injection is enabled (the default), `False` when the opt-out
            is set.
    """
    try:
        from deepagents_code.config import is_openai_prompt_cache_key_enabled

        return is_openai_prompt_cache_key_enabled()
    except Exception as exc:
        if any(cls.__name__ == "BlockingError" for cls in type(exc).__mro__):
            raise
        logger.warning(
            "Could not resolve models.openai_prompt_cache_key; defaulting to ON "
            "(an opt-out you set may not take effect)",
            exc_info=True,
        )
        return True


# [해설] runtime.context를 CLIContextSchema로 변환한다. 프로세스 안에서는 dataclass,
# [해설]   LangGraph API 서버(RemoteGraph)를 거치면 dict로 오므로 from_payload가 두
# [해설]   모양을 모두 받는다(_cli_context.py).
def _get_context(request: ModelRequest) -> CLIContextSchema | None:
    """Return runtime context when it matches the CLI context shape."""
    runtime = request.runtime
    if runtime is None:
        return None

    return CLIContextSchema.from_payload(runtime.context)


# [해설] 모델 객체에서 재개용 'provider:model' 문자열을 복원한다. ModelResult가 있으면
# [해설]   그것을, 없으면 모델 식별자 + runtime_state / _get_ls_provider 순으로
# [해설]   추정한다.
def _model_spec_from_model(
    model: BaseChatModel, model_result: ModelResult | None = None
) -> str | None:
    """Return a resumable `provider:model` spec for a model object."""
    if model_result is not None:
        return f"{model_result.provider}:{model_result.model_name}"
    model_name = get_model_identifier(model)
    from deepagents_code.config import runtime_state

    settings_provider = runtime_state.model_provider or ""
    settings_model = runtime_state.model_name or ""
    if settings_provider and settings_model and model_name == settings_model:
        return f"{settings_provider}:{settings_model}"
    provider = _get_ls_provider(model)
    if provider and model_name:
        return f"{provider}:{model_name}"
    if settings_provider and settings_model:
        return f"{settings_provider}:{settings_model}"
    return None


# [해설] create_model 결과의 provider:model을 우선 사용하고, 없으면
# [해설]   _model_spec_from_model로 대체한다.
def _model_spec_from_result(
    model_result: ModelResult | None, model: BaseChatModel
) -> str | None:
    """Return the resolved spec from `create_model`, falling back to model metadata."""
    if model_result is not None and model_result.provider and model_result.model_name:
        return f"{model_result.provider}:{model_result.model_name}"
    return _model_spec_from_model(model)


# [해설][설계] 동기·비동기 경로가 공유하는 '모델 객체를 얻은 이후'의 로직. 두 경로의
# [해설]   차이는 model_result를 만드는 방식(직접 호출 vs to_thread)뿐이다.
# [해설][흐름] 1) 새 모델 교체 2) model_params를 model_settings에 얕게 병합 3)
# [해설]   thread_id가 있으면 Fireworks/OpenAI 캐시 힌트 주입 4) Anthropic이 아닌
# [해설]   모델이면 Anthropic 전용 설정 제거 5) 시스템 프롬프트의 Model Identity 갱신 6)
# [해설]   request.override로 새 요청 생성.
def _build_overrides(
    request: ModelRequest,
    ctx: CLIContextSchema,
    model_result: ModelResult | None,
    *,
    openai_prompt_cache_key: bool,
) -> ModelRequest:
    """Build the overridden request from a (possibly resolved) model result.

    Holds the post-construction logic shared by the sync and async override
    paths: applying the model swap, merging `model_params`, stripping
    Anthropic-only settings on a cross-provider swap, and patching the
    `### Model Identity` system-prompt section. The only thing that differs
    between the two callers is how `model_result` is produced (a direct
    `create_model` call vs. an `asyncio.to_thread` offload).

    Args:
        request: The incoming model request from the middleware chain.
        ctx: Runtime CLI context carrying the requested overrides.
        model_result: The resolved model result from `create_model`, or `None`
            when no model swap was requested.
        openai_prompt_cache_key: Whether OpenAI `prompt_cache_key` injection is
            enabled (the resolved `models.openai_prompt_cache_key` opt-out).

    Returns:
        The original request when no overrides apply, otherwise a new request
            with overrides applied via `request.override()`.
    """
    overrides: dict[str, Any] = {}

    new_model = model_result.model if model_result is not None else None
    if new_model is not None:
        overrides["model"] = new_model

    # Param merge
    model_params = ctx.model_params
    if model_params:
        overrides["model_settings"] = {**request.model_settings, **model_params}

    # Inject the provider's prompt-cache routing hint from the active thread.
    # Only one provider path applies per call; both share the fetch/guard/log
    # tail below. `overrides.get` is side-effect-free, so resolving `settings`
    # before the provider check is equivalent to doing it inside each branch.
    effective_model = new_model if new_model is not None else request.model
    if ctx.thread_id:
        settings = overrides.get("model_settings", request.model_settings)
        if _is_fireworks_model(effective_model):
            # Fireworks has no opt-out gate. The classifier is provider-only
            # (like the OpenAI one), so this does not *verify* a fixed endpoint;
            # it rests on the assumption that `ChatFireworks` in practice targets
            # Fireworks' hosted API, where unknown-field rejection is not the
            # concern it is for the broadened, proxy-reachable OpenAI path below.
            updated_settings = _with_fireworks_session_settings(settings, ctx.thread_id)
            injected = "Fireworks session settings"
        elif _is_openai_model(effective_model):
            if openai_prompt_cache_key:
                updated_settings = _with_openai_prompt_cache_key(
                    effective_model, settings, ctx.thread_id
                )
                injected = "OpenAI prompt_cache_key"
            else:
                # Opt-out fired: leave the request untouched but log it so a user
                # verifying `models.openai_prompt_cache_key=false` sees a positive
                # signal rather than having to infer it from an absent log line.
                updated_settings = None
                injected = ""
                logger.debug("Skipped OpenAI prompt_cache_key (opt-out)")
        else:
            updated_settings = None
            injected = ""
        if updated_settings is not None:
            overrides["model_settings"] = updated_settings
            # The thread ID is a sensitive session identifier, so it is kept out
            # of the log line; the line firing at all confirms injection ran.
            logger.debug("Injected %s", injected)

    if not overrides:
        return request

    # When switching away from Anthropic, strip provider-specific settings
    # that would cause errors on other providers (e.g. cache_control passed
    # to the OpenAI SDK raises TypeError).
    if new_model is not None and not _is_anthropic_model(new_model):
        settings = overrides.get("model_settings", request.model_settings)
        dropped = settings.keys() & _ANTHROPIC_ONLY_SETTINGS
        if dropped:
            logger.debug(
                "Stripped Anthropic-only settings %s for non-Anthropic model",
                dropped,
            )
            overrides["model_settings"] = {
                k: v for k, v in settings.items() if k not in dropped
            }

    # Patch the Model Identity section in the system prompt so the new model
    # sees its own name/provider/context-limit, not the original's.
    # Read metadata from `model_result`, not the process-wide runtime state:
    # the middleware runs in the server subprocess, whose state is not updated
    # by `/model`.
    if model_result is not None and request.system_prompt:
        from deepagents_code.agent import (
            MODEL_IDENTITY_RE,
            build_model_identity_section,
        )

        prompt = request.system_prompt
        new_identity = build_model_identity_section(
            model_result.model_name,
            provider=model_result.provider,
            context_limit=model_result.context_limit,
            unsupported_modalities=model_result.unsupported_modalities,
        )
        patched = MODEL_IDENTITY_RE.sub(new_identity, prompt, count=1)
        if patched != prompt:
            overrides["system_prompt"] = patched
        elif "### Model Identity" in prompt:
            logger.warning(
                "System prompt contains '### Model Identity' but regex "
                "did not match; identity section was NOT updated for "
                "model '%s'. The regex may be out of sync with the "
                "prompt template.",
                model_result.model_name,
            )

    return request.override(**overrides)


# [해설] 런타임 모델 교체 때 create_model에 넘길 kwargs를 만든다. --max-retries와
# [해설]   --profile-override를 교체 후에도 유지하기 위한 것이다.
def _model_creation_kwargs(
    ctx: CLIContextSchema, cli_max_retries: int | None
) -> dict[str, Any]:
    """Build constructor kwargs needed for a runtime model switch.

    Returns:
        Keyword arguments for `create_model`.
    """
    kwargs: dict[str, Any] = {}
    if cli_max_retries is not None:
        kwargs["cli_max_retries"] = cli_max_retries
    if ctx.profile_overrides:
        kwargs["profile_overrides"] = ctx.profile_overrides
    return kwargs


# [해설][흐름] 1) context가 없으면 그대로 통과 2) ctx.model이 현재 모델과 다르면
# [해설]   create_model로 새 모델 생성 3) _build_overrides로 요청 변환 4) 체크포인트용
# [해설]   메타데이터와 함께 반환.
# [해설][주의] 정책 차단(ModelNotAllowedError)은 ModelConfigError의 하위 클래스라 반드시
# [해설]   먼저 다시 던진다. 아니면 UI는 교체됐다고 표시하는데 실제로는 이전 모델로
# [해설]   조용히 계속 돌아간다.
# [해설]       strict_model_resolution이 아닐 때 그 외 생성 실패는 로그만 남기고 현재
# [해설]   모델로 계속한다.
def _apply_overrides(
    request: ModelRequest,
    *,
    openai_prompt_cache_key: bool,
    cli_max_retries: int | None,
    strict_model_resolution: bool = False,
    construction_model_result: ModelResult | None = None,
) -> _ResolvedModelRequest:
    """Apply model/param overrides and return checkpoint persistence metadata.

    Reads `'model'` and `'model_params'` from `runtime.context` and, when
    present, swaps the model and/or merges extra settings into the request.
    On a cross-provider swap away from Anthropic, Anthropic-only settings
    (e.g. `cache_control`) are stripped. The `### Model Identity` section
    in the system prompt is also patched to reflect the new model.

    Args:
        request: The incoming model request from the middleware chain.
        openai_prompt_cache_key: The resolved `models.openai_prompt_cache_key`
            opt-out, threaded through to `_build_overrides`.
        cli_max_retries: Explicit CLI retry count retained across model switches.
        strict_model_resolution: Whether model construction failures should propagate.
        construction_model_result: Construction-time workspace model metadata.

    Returns:
        The request to send downstream plus the actual model spec and user-supplied
            model params that should be recorded for resume.

    Raises:
        ModelNotAllowedError: If runtime context requests a blocked model.
        ModelConfigError: If strict resolution is enabled and construction fails.
    """
    # [해설][흐름] 1) 런타임 context 읽기. 없으면 변경 없이 현재 모델의 spec만 기록한다.
    ctx = _get_context(request)
    if ctx is None:
        return _ResolvedModelRequest(
            request, _model_spec_from_model(request.model, construction_model_result)
        )

    model_result = None
    model = ctx.model
    # [해설][흐름] 2) 요청한 spec이 현재 모델과 이미 같으면 새로 만들지
    # [해설]   않는다(model_matches_spec, SDK). 다를 때만 create_model 호출.
    if model and not model_matches_spec(request.model, model):
        from deepagents_code.config import create_model
        from deepagents_code.model_config import ModelConfigError, ModelNotAllowedError

        logger.debug("Overriding model to %s", model)
        model_kwargs = _model_creation_kwargs(ctx, cli_max_retries)
        try:
            model_result = create_model(model, **model_kwargs)
        except ModelNotAllowedError:
            # `ModelNotAllowedError` is a `ModelConfigError`; without this
            # clause the handler below would swallow a policy denial, log it,
            # and silently continue on the *current* model while the UI
            # reported a switch. Not redundant -- do not remove.
            raise
        except ModelConfigError:
            if strict_model_resolution:
                raise
            logger.exception(
                "Failed to resolve runtime model override '%s'; "
                "continuing with current model",
                model,
            )
            # `model_params_known=False` deliberately: the override never
            # reached `_build_overrides`, so which params are in effect is
            # exactly what this path does not know. Writing the default `None`
            # instead would clear the checkpoint's params while the app still
            # holds its override, and the cold-cache identity check would then
            # compare a populated map against `None` on every send -- a
            # permanent, false "the model changed".
            return _ResolvedModelRequest(
                request,
                _model_spec_from_model(request.model, construction_model_result),
                model_params_known=False,
            )

    # [해설][흐름] 3) 새 모델(있다면)을 반영한 요청을 만들고 체크포인트에 남길 spec과
    # [해설]   params를 확정한다.
    updated = _build_overrides(
        request, ctx, model_result, openai_prompt_cache_key=openai_prompt_cache_key
    )
    params = dict(ctx.model_params) if ctx.model_params else None
    return _ResolvedModelRequest(
        updated,
        _model_spec_from_result(model_result, updated.model),
        params,
        model_params_known=True,
    )


# [해설] _apply_overrides의 비동기 버전. 모델 생성(create_model은 파일·네트워크 I/O를 할
# [해설]   수 있음)을 asyncio.to_thread로 넘겨 이벤트 루프를 막지 않는다. 나머지 로직은
# [해설]   동일합니다.
async def _apply_overrides_async(
    request: ModelRequest,
    *,
    openai_prompt_cache_key: bool,
    cli_max_retries: int | None,
    strict_model_resolution: bool = False,
    construction_model_result: ModelResult | None = None,
) -> _ResolvedModelRequest:
    """Async variant of `_apply_overrides` that offloads model construction.

    Args:
        request: The incoming model request from the middleware chain.
        openai_prompt_cache_key: The resolved `models.openai_prompt_cache_key`
            opt-out, threaded through to `_build_overrides`.
        cli_max_retries: Explicit CLI retry count retained across model switches.
        strict_model_resolution: Whether model construction failures should propagate.
        construction_model_result: Construction-time workspace model metadata.

    Returns:
        The request to send downstream plus the actual model spec and user-supplied
            model params that should be recorded for resume.

    Raises:
        ModelNotAllowedError: If runtime context requests a blocked model.
        ModelConfigError: If strict resolution is enabled and construction fails.
    """
    ctx = _get_context(request)
    if ctx is None:
        return _ResolvedModelRequest(
            request, _model_spec_from_model(request.model, construction_model_result)
        )

    model_result = None
    model = ctx.model
    if model and not model_matches_spec(request.model, model):
        from deepagents_code.config import create_model
        from deepagents_code.model_config import ModelConfigError, ModelNotAllowedError

        logger.debug("Overriding model to %s", model)
        model_kwargs = _model_creation_kwargs(ctx, cli_max_retries)
        try:
            model_result = await asyncio.to_thread(
                create_model,
                model,
                **model_kwargs,
            )
        except ModelNotAllowedError:
            # `ModelNotAllowedError` is a `ModelConfigError`; without this
            # clause the handler below would swallow a policy denial, log it,
            # and silently continue on the *current* model while the UI
            # reported a switch. Not redundant -- do not remove.
            raise
        except ModelConfigError:
            if strict_model_resolution:
                raise
            logger.exception(
                "Failed to resolve runtime model override '%s'; "
                "continuing with current model",
                model,
            )
            # `model_params_known=False` deliberately: the override never
            # reached `_build_overrides`, so which params are in effect is
            # exactly what this path does not know. Writing the default `None`
            # instead would clear the checkpoint's params while the app still
            # holds its override, and the cold-cache identity check would then
            # compare a populated map against `None` on every send -- a
            # permanent, false "the model changed".
            return _ResolvedModelRequest(
                request,
                _model_spec_from_model(request.model, construction_model_result),
                model_params_known=False,
            )

    updated = _build_overrides(
        request, ctx, model_result, openai_prompt_cache_key=openai_prompt_cache_key
    )
    params = dict(ctx.model_params) if ctx.model_params else None
    return _ResolvedModelRequest(
        updated,
        _model_spec_from_result(model_result, updated.model),
        params,
        model_params_known=True,
    )


# [해설] 호출 시작 시각을 체크포인트에 저장할 수 있는 UTC ISO 문자열로 돌려준다.
def _utc_now_iso() -> str:
    """Return the current UTC time in checkpoint-safe ISO format."""
    return datetime.now(UTC).isoformat()


# [해설] 캐시 식별에 쓰는 '실제 적용 파라미터'를 계산한다(config.toml 공급자·모델 params
# [해설]   + 런타임 override).
# [해설][설계] 체크포인트에는 런타임 override만 담는 _model_params와 별도로 병합 결과를
# [해설]   _last_cache_params에 담는다. 합쳐 저장하면 오래된 스레드가 옛 설정을 고정해
# [해설]   버리기 때문이다.
def _effective_cache_params(
    model_spec: str | None, runtime_overrides: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Resolve the cache-identity params a request actually runs with.

    Mirrors the app-side comparison: the cold-cache check reads configured
    provider/per-model `params` (via `ModelConfig.get_effective_kwargs`) plus
    runtime overrides. If the checkpoint stores only the runtime overrides, a
    user with a configured `prompt_cache_retention` and no session override
    records `None` here while the reader sees `{"prompt_cache_retention": ...}`,
    so the next turn compares unequal and reports a false `identity_changed`
    every turn. Persisting the effective params makes both sides match.

    The result is projected through `cache_identity_params` and recorded in the
    dedicated `_last_cache_params` channel rather than `_model_params`: the
    latter is read back on resume as per-session runtime overrides, so storing
    the merged config there would pin every configured knob (temperature,
    headers, ...) into old threads and silently override newer config.

    Performs blocking config reads; async callers should offload.

    Args:
        model_spec: `provider:model` spec for the call.
        runtime_overrides: Per-request params (the middleware's `model_params`).

    Returns:
        Cache-identity projection of the effective kwargs (without `base_url`),
        or `None` when no identity keys are present.
    """
    if not model_spec or ":" not in model_spec:
        overrides = dict(runtime_overrides) if runtime_overrides else None
        return cache_identity_params(overrides, model_spec=model_spec) or None
    from deepagents_code.config import _compose_openai_reasoning_effort
    from deepagents_code.model_config import ModelConfig

    _, _, model_name = model_spec.partition(":")
    provider = model_spec.split(":", 1)[0].strip().lower()
    try:
        config = ModelConfig.load()
        kwargs = config.get_effective_kwargs(
            provider,
            model_name=model_name,
            overrides=runtime_overrides,
        )
    except Exception:
        logger.warning(
            "Could not resolve effective cache params for %r; recording only "
            "runtime overrides, so a configured cache param may read as a "
            "spurious identity change until the next turn",
            model_spec,
            exc_info=True,
        )
        overrides = dict(runtime_overrides) if runtime_overrides else None
        return cache_identity_params(overrides, model_spec=model_spec) or None
    if not isinstance(kwargs, dict):
        overrides = dict(runtime_overrides) if runtime_overrides else None
        return cache_identity_params(overrides, model_spec=model_spec) or None
    # Match constructor precedence when config uses native nested reasoning.
    overrides = runtime_overrides or {}
    kwargs = _compose_openai_reasoning_effort(
        provider, kwargs, overrides.get("reasoning_effort"), overrides.get("reasoning")
    )
    # `base_url` is tracked separately as the endpoint identity; keeping it out
    # of the params avoids a double-counted identity change.
    result = cache_identity_params(
        {k: v for k, v in kwargs.items() if k != "base_url"}, model_spec=model_spec
    )
    return result or None


# [해설][흐름] 모델 호출이 '성공한 뒤' 그래프 상태에 반영할 Command(update=...)를
# [해설]   만든다: _last_model_request_at, _last_cache_model_spec, _last_cache_endpoint,
# [해설]   _model_spec, _model_params, _last_cache_params, 캐시 사용·기록.
# [해설][설계] 시각·spec·엔드포인트는 하나의 사실이므로 spec을 알 때만 함께 기록한다.
# [해설]   시각만 남으면 '모델이 바뀜' 경고가 매번 거짓으로 뜨게 된다.
def _checkpoint_command(
    resolved: _ResolvedModelRequest,
    request_started_at: str,
    cache_endpoint: str,
    cache_params: dict[str, Any] | None = None,
    *,
    response: ModelResponse | None = None,
) -> Command[Any]:
    """Build the private resume-state update for a completed model call.

    Args:
        resolved: The request as actually sent, after override resolution.
        request_started_at: UTC ISO timestamp captured before the model call.
            It only reaches a checkpoint because this runs after `handler()`
            returned, which is what makes it a successful-call marker.
        cache_endpoint: Endpoint identity for `resolved.model_spec`, from
            `_cache_endpoint_identity`. Passed in rather than resolved here so
            the async caller can keep its blocking config/credential reads off
            the event loop.
        cache_params: Cache-identity projection of the effective params for
            this call, from `_effective_cache_params`. Passed in for the same
            offloading reason as `cache_endpoint`. When `None` and
            `resolved.model_params_known` is true, falls back to the identity
            projection of the runtime overrides.
        response: This request's response, used to attribute cache activity.

    Returns:
        Command carrying cache timing and effective model metadata.
    """
    update: dict[str, Any] = {}
    # Use the resolved spec, not `_apply_overrides`'s `ctx.model`: when an
    # override fails with `ModelConfigError`, `_apply_overrides` falls back to
    # the original model while `ctx.model` still names the rejected override.
    #
    # The timestamp is written only alongside a known spec. The three are one
    # fact -- when the cache was warmed, for which model, and against which
    # endpoint -- and a timestamp without an identity would read back as a
    # permanent "model changed", warning on every send with copy that names a
    # change that never happened. For the same reason the endpoint is written
    # unconditionally here: a skipped write would leave the *previous* turn's
    # endpoint describing this turn's spec and timestamp.
    if resolved.model_spec:
        update["_last_model_request_at"] = request_started_at
        update["_last_cache_model_spec"] = resolved.model_spec
        update["_last_cache_endpoint"] = cache_endpoint
        update["_model_spec"] = resolved.model_spec
    else:
        # The previous turn's timestamp stays in place, so the next cold-cache
        # age is computed against an older request than the one just made.
        logger.debug(
            "Not recording prompt-cache state: no model spec could be derived from %s",
            type(resolved.request.model).__name__,
        )
    if resolved.model_params_known:
        # `_model_params` stays the *runtime overrides only*: resume reads it
        # back as per-session overrides for `_switch_model`, so storing the
        # merged config there would pin provider defaults (temperature, max
        # retries, headers, ...) into old threads and silently override newer
        # config on resume.
        update["_model_params"] = resolved.model_params
        # The cold-cache identity projection goes to its own channel. It must
        # include configured provider params -- not just the session override
        # above -- or the reader's effective-params comparison reports a false
        # `identity_changed` on every turn for anyone with a configured cache
        # knob.
        update["_last_cache_params"] = (
            cache_params
            if cache_params is not None
            else (
                cache_identity_params(
                    resolved.model_params, model_spec=resolved.model_spec
                )
                or None
            )
        )
    if resolved.model_spec and response is not None:
        activity = CacheActivity(
            requested_at=request_started_at,
            model_spec=resolved.model_spec,
            endpoint=cache_endpoint,
            params=cache_params,
        )
        update.update(_cache_activity_update(response, activity))
    return Command(update=update)


# [해설] 응답의 usage_metadata에서 캐시 쓰기·읽기 토큰을 확인해, 이번 호출이 실제로
# [해설]   보고한 캐시 활동만 _last_cache_write/_last_cache_use로 기록한다.
def _cache_activity_update(
    response: ModelResponse, activity: CacheActivity
) -> dict[str, CacheActivity]:
    """Record only cache activity reported by this specific model call.

    Returns:
        Updates for the last cache write and/or use, when reported.
    """
    from deepagents_code.cost_tracking import cache_token_counts

    update: dict[str, CacheActivity] = {}
    for message in response.result:
        if not isinstance(message, AIMessage) or not message.usage_metadata:
            continue
        reads, writes = cache_token_counts(message.usage_metadata)
        if any(writes):
            update["_last_cache_write"] = activity
        if reads or any(writes):
            update["_last_cache_use"] = activity
    return update


# [해설][설계] runtime.context의 model(provider:model)과 model_params를 읽어 호출마다
# [해설]   모델·설정을 바꾸는 AgentMiddleware. 그래프 재컴파일 없이 /model 전환이 가능한
# [해설]   이유입니다.
# [해설][흐름] wrap_model_call(동기)/awrap_model_call(비동기)이 use_environment로
# [해설]   작업공간 환경에 들어간 뒤 -> 오버라이드 해석 -> 캐시 식별 계산 -> 다음
# [해설]   handler 호출 -> 체크포인트 Command와 함께 ExtendedModelResponse 반환.
class ConfigurableModelMiddleware(AgentMiddleware):
    """Swap the model or per-call settings from `runtime.context`.

    Reads two optional keys from the runtime context dict:

    - `'model'` — a `provider:model` spec (e.g. `"openai:gpt-5"`).
        When present and different from the current model, the request is
        re-routed to the new model.
    - `'model_params'` — a dict of extra model settings (e.g.
        `{"temperature": 0}`) that are shallow-merged into the
        request's `model_settings`.

    This middleware is typically the outermost layer so it intercepts every
    model call before provider-specific middleware (like
    `AnthropicPromptCachingMiddleware`) runs.
    """

    # [해설] LangSmith 트레이스에 훅 입력을 싣지 않는다(민감한 context/요청 내용을
    # [해설]   트레이스에 남기지 않기 위함. 설정으로 재정의 가능).
    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    # [해설][설계] environ과 model_result를 보관한다. 늦은(lazy) 모델 교체는 생성 시점이
    # [해설]   아니라 요청 시점에 일어나므로, 그때 같은 작업공간 환경으로 다시 들어가기
    # [해설]   위해서입니다(wrap_model_call의 use_environment).
    # [해설]       persist_model_state=False는 서브에이전트용이다(부모 스레드의 재개
    # [해설]   상태를 소유하지 않음).
    def __init__(
        self,
        *,
        persist_model_state: bool = True,
        openai_prompt_cache_key: bool | None = None,
        cli_max_retries: int | None = None,
        strict_model_resolution: bool = False,
        environ: Mapping[str, str] | None = None,
        model_result: ModelResult | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            persist_model_state: Whether completed calls should write private
                resume metadata. Subagent instances disable this because they do
                not own the parent thread's resume state.
            openai_prompt_cache_key: Whether to inject a per-thread OpenAI
                `prompt_cache_key`. Left as `None` (the default) it is resolved
                once here from `models.openai_prompt_cache_key` and cached, so no
                per-call read happens. The one-time `config.toml` read assumes
                current callers construct the middleware off the
                blockbuster-guarded server loop (the server path offloads
                `create_cli_agent` via `asyncio.to_thread`); if that assumption
                is ever broken the read would trip `BlockingError`, which
                `_resolve_openai_prompt_cache_key_enabled` re-raises rather than
                masks. Pass an explicit bool to bypass the config read (mainly
                for tests).
            cli_max_retries: Explicit `--max-retries` value to retain across
                runtime model switches.
            strict_model_resolution: Whether invalid runtime model overrides should
                fail the call instead of falling back to the construction-time model.
            environ: Workspace environment retained for lazy model switches.
            model_result: Construction-time workspace model metadata.
        """
        self._persist_model_state = persist_model_state
        self._environ = environ
        self._model_result = model_result
        self._cli_max_retries = cli_max_retries
        self._strict_model_resolution = strict_model_resolution
        self._openai_prompt_cache_key = (
            _resolve_openai_prompt_cache_key_enabled()
            if openai_prompt_cache_key is None
            else openai_prompt_cache_key
        )

    # [해설][흐름] 1) use_environment 진입 2) _apply_overrides로 요청 변환 3)
    # [해설]   엔드포인트·캐시 params 계산 4) cache_request ContextVar에 활동 기록 후
    # [해설]   handler 호출 5) persist 모드면 _checkpoint_command를 붙여 반환.
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Apply runtime overrides and delegate to the next handler.

        Returns:
            The downstream response plus a private resume-state update when the
            completed call has model metadata to checkpoint.
        """
        from deepagents_code.config import use_environment

        with use_environment(self._environ):
            resolved = _apply_overrides(
                request,
                openai_prompt_cache_key=self._openai_prompt_cache_key,
                cli_max_retries=self._cli_max_retries,
                strict_model_resolution=self._strict_model_resolution,
                construction_model_result=self._model_result,
            )
            cache_endpoint = (
                _cache_endpoint_identity(resolved.model_spec, resolved.model_params)
                if resolved.model_params is not None
                else _cache_endpoint_identity(resolved.model_spec)
            )
            cache_params = _effective_cache_params(
                resolved.model_spec, resolved.model_params
            )
            activity = CacheActivity(
                requested_at=_utc_now_iso(),
                model_spec=resolved.model_spec or "",
                endpoint=cache_endpoint,
                params=cache_params,
            )
            token = cache_request.set(activity if self._persist_model_state else None)
            try:
                response = handler(resolved.request)
            finally:
                cache_request.reset(token)
            if not self._persist_model_state:
                return response
            request_started_at = activity["requested_at"]
        command = _checkpoint_command(
            resolved,
            request_started_at,
            cache_endpoint,
            cache_params,
            response=response,
        )
        return ExtendedModelResponse(model_response=response, command=command)

    # [해설] wrap_model_call의 비동기 버전. 블로킹 가능한 계산(_cache_endpoint_identity,
    # [해설]   _effective_cache_params)은 asyncio.to_thread로 오프로드한다.
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Apply runtime overrides and delegate to the next async handler.

        Returns:
            The downstream response plus a private resume-state update when the
            completed call has model metadata to checkpoint.
        """
        from deepagents_code.config import use_environment

        with use_environment(self._environ):
            resolved = await _apply_overrides_async(
                request,
                openai_prompt_cache_key=self._openai_prompt_cache_key,
                cli_max_retries=self._cli_max_retries,
                strict_model_resolution=self._strict_model_resolution,
                construction_model_result=self._model_result,
            )
            cache_endpoint = (
                await asyncio.to_thread(
                    _cache_endpoint_identity,
                    resolved.model_spec,
                    resolved.model_params,
                )
                if resolved.model_params is not None
                else await asyncio.to_thread(
                    _cache_endpoint_identity, resolved.model_spec
                )
            )
            cache_params = await asyncio.to_thread(
                _effective_cache_params,
                resolved.model_spec,
                resolved.model_params,
            )
            activity = CacheActivity(
                requested_at=_utc_now_iso(),
                model_spec=resolved.model_spec or "",
                endpoint=cache_endpoint,
                params=cache_params,
            )
            token = cache_request.set(activity if self._persist_model_state else None)
            try:
                response = await handler(resolved.request)
            finally:
                cache_request.reset(token)
            if not self._persist_model_state:
                return response
            request_started_at = activity["requested_at"]
        command = _checkpoint_command(
            resolved,
            request_started_at,
            cache_endpoint,
            cache_params,
            response=response,
        )
        return ExtendedModelResponse(model_response=response, command=command)
