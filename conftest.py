"""pytest configuration — root conftest.py.

Adds project root to sys.path and provides a session-scoped fixture
that loads the pipeline module once (avoids reloading ~6k-line file per test).
"""
import sys
import os
import importlib.util

import pytest

# Make project root importable without sys.path.insert in every test file
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(scope="session")
def pq():
    """Session-scoped pipeline module — loaded once, reused across all tests."""
    _name = "pipeline_quimiometria_14"
    spec = importlib.util.spec_from_file_location(
        _name, os.path.join(os.path.dirname(__file__), f"{_name}.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
