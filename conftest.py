"""pytest configuration — root conftest.py.

Adds ./src to sys.path so the `guaraci` package is importable without an
editable install, and provides a session-scoped fixture that loads the
pipeline module once (avoids reloading the large module per test).
"""
import sys
import os

import pytest

# Torna o pacote `guaraci` (em ./src) importável nos testes sem `pip install -e .`.
_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(scope="session")
def pq():
    """Session-scoped pipeline module — loaded once, reused across all tests."""
    import guaraci.pipeline as mod
    return mod


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
