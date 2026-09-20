"""Testes de `scripts/benchmark_dual_spls_tecator.py` -- so' a logica de
PARSING do formato `.ts` (pura, sem rede/sem baixar o wheel sktime).
`_extrair_tecator_do_sktime`/`rodar_portao`/`rodar_tabela_teste` precisam
de um wheel real baixado via `pip download` e nao sao testados aqui (rede
indisponivel em CI nao deve quebrar a suite principal -- mesmo padrao de
`tests/test_benchmark_tecator.py`) -- rodar manualmente:

    pip download --no-deps -d /tmp/sktime_dl sktime==1.1.0
    python scripts/benchmark_dual_spls_tecator.py /tmp/sktime_dl/sktime-1.1.0-py3-none-any.whl
"""
import os
import sys

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np
import pytest

import benchmark_dual_spls_tecator as bdt  # noqa: E402


def _texto_ts_fake(n_amostras: int, n_canais: int = 4) -> str:
    linhas = [
        "# TECATOR dataset (fake, so' pra' teste de parsing)",
        "@problemName TECATOR",
        "@timeStamps false",
        "@missing false",
        "@univariate true",
        "@equalLength true",
        "@targetLabel true",
        "@data",
    ]
    for i in range(n_amostras):
        valores = ",".join(f"{1.0 + 0.1 * (i + j):.4f}" for j in range(n_canais))
        alvo = 10.0 + i
        linhas.append(f"{valores}:{alvo}")
    return "\n".join(linhas)


def test_parsear_ts_extrai_espectro_e_alvo():
    texto = _texto_ts_fake(5, n_canais=4)
    X, y = bdt._parsear_ts(texto)
    assert X.shape == (5, 4)
    assert y.shape == (5,)
    np.testing.assert_allclose(y, [10.0, 11.0, 12.0, 13.0, 14.0])
    np.testing.assert_allclose(X[0], [1.0, 1.1, 1.2, 1.3], atol=1e-6)


def test_parsear_ts_ignora_linhas_de_cabecalho():
    """Linhas `#`/`@` (comentario/metadado do formato .ts) nao devem virar
    amostras -- so' as linhas de dado (`v1,...,vN:alvo`) contam."""
    texto = _texto_ts_fake(3, n_canais=2)
    n_linhas_cabecalho = sum(
        1 for ln in texto.splitlines() if ln.startswith(("#", "@")))
    assert n_linhas_cabecalho == 8  # ver _texto_ts_fake
    X, _y = bdt._parsear_ts(texto)
    assert X.shape[0] == 3


def test_sha256_e_deterministico():
    assert bdt._sha256(b"abc") == bdt._sha256(b"abc")
    assert bdt._sha256(b"abc") != bdt._sha256(b"abd")


def test_extrair_tecator_do_sktime_rejeita_shape_inesperado(tmp_path, monkeypatch):
    """Se uma versao futura do sktime mudar o dataset empacotado (menos
    amostras, canais diferentes), o script deve falhar alto e claro --
    nao seguir silenciosamente com o formato errado (mesma filosofia dos
    guards de `scripts/benchmark_tecator.py` para o dado real via rede)."""
    import zipfile

    caminho = tmp_path / "fake_sktime.whl"
    texto_pequeno = _texto_ts_fake(5, n_canais=4)  # NAO tem shape (172,100)
    with zipfile.ZipFile(caminho, "w") as z:
        z.writestr(bdt._TS_PATH_TRAIN, texto_pequeno)
        z.writestr(bdt._TS_PATH_TEST, texto_pequeno)

    with pytest.raises(ValueError, match="Formato inesperado"):
        bdt._extrair_tecator_do_sktime(str(caminho))
