"""test_hsi_offline_prova.py -- contra-prova de independencia de terceiro
(Passo 111d, INSTRUCAO_HSI_DADO_PROPRIO.md).

"HSI funciona com dado do proprio usuario, offline, sem dependencia de
dataset publico" e' uma alegacao verificavel, nao uma afirmacao de
prosa -- este teste e' a prova: constroi um cubo hiperespectral
SINTETICO (gerado localmente, ZERO download), desabilita rede de
verdade no nivel de socket (nao so' "nao chamei a funcao de download" --
qualquer tentativa de abrir uma conexao de rede, de QUALQUER lugar do
pipeline, levanta excecao imediatamente) e roda o pipeline HSI completo
(leitura -> quality gate -> segmentacao -> classificacao por pixel ->
mapa espacial -> explicabilidade/confianca por objeto -> validacao)
sobre ele. Se isso passa, "offline e independente de dataset publico"
e' fato medido, nao alegacao.
"""
from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

from guaraci.config import Config
from guaraci.hsi_pipeline import run_hsi_pipeline


class _RedeDesabilitada(Exception):
    """Levantada no lugar de QUALQUER tentativa de abrir socket -- prova
    que nada no pipeline tenta rede, em vez de so' confiar que "nao
    chamamos a funcao de download"."""


@pytest.fixture
def rede_desabilitada(monkeypatch):
    def _socket_bloqueado(*args, **kwargs):
        raise _RedeDesabilitada(
            "tentativa de abrir socket de rede durante o pipeline HSI "
            "offline -- isso NAO deveria acontecer (Passo 111d).")

    monkeypatch.setattr(socket, "socket", _socket_bloqueado)
    monkeypatch.setattr(socket, "create_connection", _socket_bloqueado)
    yield


def _mascara_objeto_suave(n_lin, n_col, raio_frac=0.35, largura_borda=4.0):
    """Transicao suave (nao degrau binario) -- evita que o estimador de
    ruido de Immerkaer (Laplaciano, hsi_quality.py) confunda a borda da
    mascara sintetica com ruido de alta frequencia (ver mesma logica em
    test_hsi_pipeline.py)."""
    yy, xx = np.ogrid[:n_lin, :n_col]
    cy, cx = (n_lin - 1) / 2.0, (n_col - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    raio = raio_frac * min(n_lin, n_col)
    return np.clip((raio - dist) / largura_borda + 0.5, 0.0, 1.0)


def _gravar_cubo(caminho_sem_ext: Path, cubo: np.ndarray, n_bandas: int) -> None:
    n_lin, n_col, _ = cubo.shape
    caminho_sem_ext.parent.mkdir(parents=True, exist_ok=True)
    caminho_sem_ext.with_suffix(".bin").write_bytes(cubo.astype("<f4").tobytes())
    caminho_sem_ext.with_suffix(".hdr").write_text(
        f"ENVI\nsamples = {n_col}\nlines = {n_lin}\nbands = {n_bandas}\n"
        f"header offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
        f"interleave = bip\nbyte order = 0\n", encoding="utf-8")


def _montar_dataset_hsi_sintetico(tmp_path: Path) -> Path:
    """Simula "dado do usuario": 2 classes ('madura'/'verde'), 3 amostras
    fisicas cada (nivel 'high' do Bloco 8 -- 1 subpasta por amostra),
    100% gerado localmente com numpy -- nenhum arquivo baixado."""
    rng = np.random.default_rng(42)
    raiz = tmp_path / "cubo_do_usuario"
    n_lin, n_col, n_bandas = 64, 64, 8
    alpha = _mascara_objeto_suave(n_lin, n_col)[..., None]
    for classe, nivel_objeto in (("madura", 0.85), ("verde", 0.45)):
        for a in range(3):
            fundo = rng.normal(loc=0.05, scale=0.01, size=(n_lin, n_col, n_bandas))
            objeto = rng.normal(loc=nivel_objeto, scale=0.01,
                                size=(n_lin, n_col, n_bandas))
            cubo = alpha * objeto + (1.0 - alpha) * fundo
            _gravar_cubo(raiz / classe / f"amostra{a}" / "vista0", cubo, n_bandas)
    return raiz


def test_pipeline_hsi_completo_offline_com_cubo_sintetico_do_usuario(
        tmp_path, rede_desabilitada):
    """A contra-prova exigida pelo Passo 111d: pipeline HSI ponta-a-ponta
    (leitura -> quality gate -> segmentacao -> classificacao -> mapa ->
    confianca por objeto -> validacao) sobre um cubo 100% sintetico, com
    rede desabilitada de verdade -- nao trava, nao lanca _RedeDesabilitada,
    produz saida real."""
    raiz = _montar_dataset_hsi_sintetico(tmp_path)
    cfg = Config(mode="hsi", hsi_dataset_folder=str(raiz),
                output_root_folder=str(tmp_path / "saida"),
                output_format="png")

    resumo = run_hsi_pipeline(cfg)

    # Prova de que o pipeline rodou de verdade (nao um resultado vazio/forjado).
    assert resumo["n_gravacoes_total"] == 6
    assert resumo["n_gravacoes_aceitas"] == 6
    assert resumo["n_components"] >= 1
    assert resumo["grouping_guarantee"] == "high"
    assert resumo["achados_quimica"] is None  # nao aplicavel sem dataset publico
    assert len(resumo["confianca_por_objeto"]) == 6

    val = resumo["validacao_externa"]
    assert val.n_objetos_teste_interno >= 1
    assert val.n_objetos_teste_externo == 0  # dataset generico: so' interna

    caminho_figura = (Path(cfg.output_folder) / "Graficos" / "hsi" /
                      "hsi_mapa_classificacao_amostra.png")
    assert caminho_figura.is_file()


def test_rede_desabilitada_de_verdade_bloqueia_socket(rede_desabilitada):
    """Contra-prova da contra-prova: confirma que a fixture realmente
    bloqueia rede (senao o teste acima passaria mesmo com uma chamada de
    rede escondida, e a alegacao de 'offline' nao seria verificada)."""
    with pytest.raises(_RedeDesabilitada):
        socket.socket()
