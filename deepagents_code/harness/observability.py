"""Langfuse monitoring for the agent server.

The agent graph runs in the server process, so tracing is attached there. Each
LangGraph thread becomes one Langfuse session and each user turn becomes one
trace: the session id is the thread id that dcode already stamps on every run.

Monitoring is optional. Without both Langfuse keys every function here is a
no-op, and a failure to set tracing up never stops the agent.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
"""Environment variable holding the Langfuse public key."""

SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"  # noqa: S105  # variable name, not a secret
"""Environment variable holding the Langfuse secret key."""

BASE_URL_ENVS = ("LANGFUSE_BASE_URL", "LANGFUSE_HOST")
"""Environment variables naming the Langfuse server, in lookup order."""

ENABLED_ENV = "LANGFUSE_TRACING_ENABLED"
"""Environment variable that switches tracing off when set to a false value."""

SESSION_METADATA_KEY = "langfuse_session_id"
"""Run metadata key the Langfuse callback reads the session id from."""

USER_METADATA_KEY = "langfuse_user_id"
"""Run metadata key the Langfuse callback reads the user id from."""

INHERITED_SPAN_ATTRIBUTES = ("session.id", "user.id")
"""Span attributes Langfuse reads the session and user of a trace from."""

_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def is_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Report whether Langfuse tracing should be attached.

    Args:
        environ: Environment to read. Defaults to the process environment.

    Returns:
        `True` when both keys are set and tracing is not switched off.
    """
    env = os.environ if environ is None else environ
    if env.get(ENABLED_ENV, "").strip().lower() in _FALSE_VALUES:
        return False
    return bool(env.get(PUBLIC_KEY_ENV)) and bool(env.get(SECRET_KEY_ENV))


def scrub_shell_env(env: dict[str, str]) -> None:
    """Remove the Langfuse secret from an environment handed to agent shells.

    The shell environment is copied from the server environment, so without this
    a command run by the agent could read the secret key.

    Args:
        env: Shell environment, modified in place.
    """
    env.pop(SECRET_KEY_ENV, None)


def session_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Derive Langfuse session attributes from dcode run metadata.

    Args:
        metadata: Metadata of a root run, as built by `build_stream_config`.

    Returns:
        Metadata keys to add. Empty when no thread id is present. Keys that the
            caller already set are left out so an explicit value wins.
    """
    extra: dict[str, str] = {}
    thread_id = metadata.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        extra[SESSION_METADATA_KEY] = thread_id
    user_id = metadata.get("user_id")
    if isinstance(user_id, str) and user_id:
        extra[USER_METADATA_KEY] = user_id
    return {key: value for key, value in extra.items() if key not in metadata}


def _build_session_callback() -> BaseCallbackHandler:
    """Create the callback that maps a thread to a Langfuse session.

    Returns:
        Callback to place before the Langfuse callback handler.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class SessionCallback(BaseCallbackHandler):
        """Stamp Langfuse session attributes on the metadata of root runs."""

        # Inline handlers run first and in list order, so the Langfuse handler
        # sees the session id this callback writes into the shared metadata.
        run_inline = True

        @override
        def on_chain_start(
            self,
            serialized: dict[str, Any] | None,
            inputs: dict[str, Any] | Any,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            """Copy the thread ID into the root run's metadata as the session ID."""
            if parent_run_id is None and metadata is not None:
                metadata.update(session_metadata(metadata))

    return SessionCallback()


def _build_flush_callback(client: Any) -> BaseCallbackHandler:  # noqa: ANN401
    """Create the callback that delivers a turn's spans when it finishes.

    Args:
        client: Langfuse client whose buffer is flushed.

    Returns:
        Callback to place after the Langfuse callback handler.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class FlushCallback(BaseCallbackHandler):
        """Flush buffered spans once a root run has ended."""

        # Not inline on purpose: LangChain runs this handler in a worker thread
        # after the inline Langfuse handler has closed the root span, and waits
        # for it. The blocking flush therefore stays off the event loop, and a
        # short-lived server cannot exit before the turn's trace is delivered.
        run_inline = False

        def __init__(self) -> None:
            """Start with no root runs being tracked."""
            self._roots: set[UUID] = set()

        @override
        def on_chain_start(
            self,
            serialized: dict[str, Any] | None,
            inputs: dict[str, Any] | Any,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            """Remember the run when it is a root run."""
            if parent_run_id is None:
                self._roots.add(run_id)

        @override
        def on_chain_end(
            self,
            outputs: dict[str, Any] | Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            """Flush the client when a root run finishes."""
            self._finish(run_id)

        @override
        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            """Flush the client when a root run fails."""
            self._finish(run_id)

        def _finish(self, run_id: UUID) -> None:
            """Flush pending traces if the run is a tracked root run."""
            if run_id not in self._roots:
                return
            self._roots.discard(run_id)
            try:
                client.flush()
            except Exception:
                logger.warning("Langfuse flush failed", exc_info=True)

    return FlushCallback()


def _build_tracer_provider() -> Any:  # noqa: ANN401  # OpenTelemetry provider
    """Create a tracer provider whose spans inherit the session of their parent.

    Langfuse stamps the session on a span from a context variable set on the
    root run. LangGraph starts child runs in contexts that do not carry that
    variable, so only the root span would get a session. The processor here
    copies the trace attributes from the parent span instead, which does not
    depend on context propagation.

    A dedicated provider also keeps Langfuse from replacing the process-wide
    OpenTelemetry provider.

    Returns:
        Provider to pass to the Langfuse client.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider

    class InheritTraceAttributes(SpanProcessor):
        """Copy session and user attributes from the parent span."""

        @override
        def on_start(self, span: Any, parent_context: Any = None) -> None:
            """Copy the inherited attributes from the parent span onto the new span."""
            parent = trace.get_current_span(parent_context)
            attributes = getattr(parent, "attributes", None)
            if not attributes:
                return
            for key in INHERITED_SPAN_ATTRIBUTES:
                value = attributes.get(key)
                if value:
                    span.set_attribute(key, value)

    provider = TracerProvider()
    provider.add_span_processor(InheritTraceAttributes())
    return provider


def build_callbacks(environ: Mapping[str, str] | None = None) -> list[Any]:
    """Create the callbacks that send graph runs to Langfuse.

    Args:
        environ: Environment holding the Langfuse settings. Defaults to the
            process environment.

    Returns:
        Session callback, Langfuse handler, and flush callback in the order
            they must run, or an empty list when tracing is disabled.
    """
    env = os.environ if environ is None else environ
    if not is_enabled(env):
        return []
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    public_key = env[PUBLIC_KEY_ENV]
    base_url = next((env[name] for name in BASE_URL_ENVS if env.get(name)), None)
    client = Langfuse(
        public_key=public_key,
        secret_key=env[SECRET_KEY_ENV],
        base_url=base_url,
        tracer_provider=_build_tracer_provider(),
    )
    handler = CallbackHandler(public_key=public_key)
    # Inline handlers run in list order and are awaited one by one. That keeps
    # this handler after the session callback, and lets the flush callback run
    # only once this handler has closed the root span.
    handler.run_inline = True
    return [_build_session_callback(), handler, _build_flush_callback(client)]


def attach_langfuse(graph: Any, environ: Mapping[str, str] | None = None) -> Any:  # noqa: ANN401
    """Return `graph` with Langfuse callbacks bound to every run.

    Args:
        graph: Compiled agent graph.
        environ: Environment holding the Langfuse settings. Defaults to the
            process environment.

    Returns:
        A copy of the graph that traces to Langfuse, or `graph` itself when
            tracing is disabled or could not be set up.
    """
    # Monitoring is optional: nothing in here may stop the server from starting.
    try:
        callbacks = build_callbacks(environ)
        if not callbacks:
            return graph
        config = dict(graph.config or {})
        existing = config.get("callbacks")
        if existing is not None and not isinstance(existing, list):
            logger.warning("Graph callbacks are not a list; Langfuse tracing skipped")
            return graph
        # Replace the copied config directly. `with_config` merges through
        # `merge_configs`, which can drop the recursion limit set by the caller.
        config["callbacks"] = [*(existing or []), *callbacks]
        traced = graph.copy({"config": config})
    except Exception:
        logger.warning("Langfuse tracing could not be set up", exc_info=True)
        return graph
    logger.info("Langfuse tracing attached")
    return traced
