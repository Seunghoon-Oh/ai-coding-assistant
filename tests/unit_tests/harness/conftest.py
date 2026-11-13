"""Shared setup for the harness tests."""

from __future__ import annotations

import pytest

from deepagents_code._env_vars import PLAN_STEP_LOG


@pytest.fixture(autouse=True)
def _local_step_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the local step record on; these tests read it back.

    The unit test suite turns it off by default so that agents built without
    a working directory do not write into the source tree. Every harness test
    names a `tmp_path` root, so writing is safe here.
    """
    monkeypatch.setenv(PLAN_STEP_LOG, "1")
