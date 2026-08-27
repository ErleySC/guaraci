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

# Profile do Hypothesis (Bloco 13d, Frente 2): busca aleatoria SOZINHA (so'
# `max_examples`, sem `@example` fixo) mostrou-se pouco confiavel para achar
# bug conhecido (achado real, Passo 85 -- 80 exemplos passaram raso por cima
# de tres bugs reais de config.yaml sem achar nenhum). A correcao estrutural
# e' dupla: (1) SEMPRE fixar `@example` para o caso adversarial conhecido
# (nao depender so' do profile), (2) rodar com orcamento MAIOR na CI do que
# localmente -- mais exemplos aleatorios aumentam a chance de achar um caso
# novo que ninguem fixou ainda, sem deixar o ciclo local lento.
# `CI=true` e' setado automaticamente pelo GitHub Actions (nenhuma mudanca
# necessaria em test.yml); localmente cai no profile "dev" a nao ser que
# HYPOTHESIS_PROFILE seja setado explicitamente.
try:
    from hypothesis import settings as _hypothesis_settings

    _hypothesis_settings.register_profile("dev", max_examples=50, deadline=None)
    _hypothesis_settings.register_profile("ci", max_examples=300, deadline=None)
    _perfil_padrao = "ci" if os.environ.get("CI") else "dev"
    _hypothesis_settings.load_profile(
        os.environ.get("HYPOTHESIS_PROFILE", _perfil_padrao))
except ImportError:
    pass   # hypothesis e' dependencia de DEV (extra [dev]) -- ausente e' valido


@pytest.fixture(scope="session")
def pq():
    """Session-scoped pipeline module — loaded once, reused across all tests."""
    import guaraci.pipeline as mod
    return mod


def achar_pastas_run(pasta_saida_raiz):
    """Localiza as pastas de EXECUCAO (folha) sob pasta_saida_raiz.

    Desde a auditoria jul/2026 (item 4), generate_output_name aninha a saida em
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
