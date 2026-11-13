"""Tests for Langfuse monitoring setup. No test here reaches the network."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, START, StateGraph
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel

from deepagents_code.harness import observability

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

KEYS = {
    observability.PUBLIC_KEY_ENV: "pk-test",
    observability.SECRET_KEY_ENV: "sk-test",
}


class _State(BaseModel):
    value: int


def _increment(state: _State) -> dict[str, int]:
    return {"value": state.value + 1}


def _graph() -> CompiledStateGraph[_State, None, _State, _State]:
    builder = StateGraph(_State)
    builder.add_node("increment", _increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile()


class _FakeClient:
    """Stand-in for the Langfuse client that only counts flushes."""

    def __init__(self) -> None:
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1


class _RootMetadataProbe(BaseCallbackHandler):
    """Record the metadata a later handler sees on the root run."""

    def __init__(self) -> None:
        self.root_metadata: dict[str, Any] | None = None

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: object,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized, inputs, run_id, kwargs
        if parent_run_id is None:
            self.root_metadata = dict(metadata or {})


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, False),
        ({observability.PUBLIC_KEY_ENV: "pk-test"}, False),
        ({observability.SECRET_KEY_ENV: "sk-test"}, False),
        (KEYS, True),
        ({**KEYS, observability.ENABLED_ENV: "false"}, False),
        ({**KEYS, observability.ENABLED_ENV: "OFF"}, False),
        ({**KEYS, observability.ENABLED_ENV: "true"}, True),
    ],
)
def test_is_enabled_needs_both_keys_and_no_opt_out(environ, expected) -> None:
    assert observability.is_enabled(environ) is expected


def test_scrub_shell_env_removes_only_the_secret() -> None:
    env = {**KEYS, "PATH": "/usr/bin"}

    observability.scrub_shell_env(env)

    assert observability.SECRET_KEY_ENV not in env
    assert env[observability.PUBLIC_KEY_ENV] == "pk-test"
    assert env["PATH"] == "/usr/bin"


def test_session_metadata_maps_thread_and_user() -> None:
    extra = observability.session_metadata({"thread_id": "t-1", "user_id": "u-1"})

    assert extra == {
        observability.SESSION_METADATA_KEY: "t-1",
        observability.USER_METADATA_KEY: "u-1",
    }


def test_session_metadata_is_empty_without_a_thread() -> None:
    assert observability.session_metadata({"turn_id": "turn-1"}) == {}
    assert observability.session_metadata({"thread_id": ""}) == {}


def test_session_metadata_keeps_an_explicit_session() -> None:
    metadata = {"thread_id": "t-1", observability.SESSION_METADATA_KEY: "custom"}

    assert observability.session_metadata(metadata) == {}


def test_build_callbacks_is_empty_when_disabled() -> None:
    assert observability.build_callbacks({}) == []


def test_attach_langfuse_returns_the_same_graph_when_disabled() -> None:
    graph = _graph()

    assert observability.attach_langfuse(graph, {}) is graph


def test_attach_langfuse_survives_a_setup_failure(monkeypatch) -> None:
    def fail(environ: object) -> list[Any]:
        del environ
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(observability, "build_callbacks", fail)
    graph = _graph()

    assert observability.attach_langfuse(graph, KEYS) is graph


def test_attach_langfuse_survives_a_graph_it_cannot_copy(monkeypatch) -> None:
    def build(environ: object) -> list[Any]:
        del environ
        return [_RootMetadataProbe()]

    monkeypatch.setattr(observability, "build_callbacks", build)
    stub = object()

    assert observability.attach_langfuse(stub, KEYS) is stub


def test_attach_langfuse_binds_callbacks_to_a_copy(monkeypatch) -> None:
    probe = _RootMetadataProbe()

    def build(environ: object) -> list[Any]:
        del environ
        return [probe]

    monkeypatch.setattr(observability, "build_callbacks", build)
    graph = _graph()

    bound = observability.attach_langfuse(graph, KEYS)

    assert bound is not graph
    assert bound.config["callbacks"] == [probe]
    assert not (graph.config or {}).get("callbacks")


async def test_session_callback_runs_before_later_handlers() -> None:
    probe = _RootMetadataProbe()
    callbacks = [observability._build_session_callback(), probe]
    graph = _graph().copy({"config": {"callbacks": callbacks}})
    config: RunnableConfig = {"metadata": {"thread_id": "thread-1"}}

    result = await graph.ainvoke(_State(value=1), config)

    assert result == {"value": 2}
    assert probe.root_metadata is not None
    assert probe.root_metadata[observability.SESSION_METADATA_KEY] == "thread-1"


async def test_flush_callback_flushes_once_per_root_run() -> None:
    client = _FakeClient()
    callbacks = [observability._build_flush_callback(client)]
    graph = _graph().copy({"config": {"callbacks": callbacks}})

    await graph.ainvoke(_State(value=1))
    assert client.flushes == 1

    await graph.ainvoke(_State(value=1))
    assert client.flushes == 2


async def test_flush_failure_does_not_break_the_run() -> None:
    class _BrokenClient:
        def flush(self) -> None:
            msg = "network down"
            raise RuntimeError(msg)

    callbacks = [observability._build_flush_callback(_BrokenClient())]
    graph = _graph().copy({"config": {"callbacks": callbacks}})

    assert await graph.ainvoke(_State(value=1)) == {"value": 2}


def test_child_spans_inherit_the_session_of_their_parent() -> None:
    provider = observability._build_tracer_provider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("root") as root:
        root.set_attribute("session.id", "thread-1")
        root.set_attribute("user.id", "user-1")
        with tracer.start_as_current_span("child"), tracer.start_span("grandchild"):
            pass
    with tracer.start_as_current_span("other-root"):
        pass

    spans = {
        span.name: dict(span.attributes or {}) for span in exporter.get_finished_spans()
    }
    assert spans["child"] == {"session.id": "thread-1", "user.id": "user-1"}
    assert spans["grandchild"] == {"session.id": "thread-1", "user.id": "user-1"}
    assert spans["other-root"] == {}
