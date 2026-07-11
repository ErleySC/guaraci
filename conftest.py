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


def achar_pastas_run(pasta_saida_raiz):
    """Localiza as pastas de EXECUCAO (folha) sob pasta_saida_raiz.

    Desde a auditoria jul/2026 (item 4), gerar_nome_saida aninha a saida em
    pasta_saida_raiz/<Amostra>/<Modo>/<execucao>/ em vez de uma pasta direta
    em pasta_saida_raiz/<execucao>/. Esta funcao encontra as pastas-folha
    (identificadas pelo prefixo "PLSDA_OE_") em qualquer profundidade, para
    que os testes nao precisem hardcodar o numero de niveis.
    """
    import os as _os
    achadas = []
    for raiz, dirs, _arqs in _os.walk(str(pasta_saida_raiz)):
        for d in dirs:
            if d.startswith("PLSDA_OE_"):
                achadas.append(_os.path.join(raiz, d))
    return achadas


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
