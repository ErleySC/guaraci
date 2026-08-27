# -*- coding: utf-8 -*-
"""Testes de integracao do dossie (Bloco 13d): `append_linearity_
robustness_model_card` (model_card.md + LOD/LOQ + faixa de trabalho +
incerteza + linearidade + robustez, mesmo mecanismo append-only ja usado
para regressao/identificacao/pureza) e validacao contra Corn/Mendeley
reais.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from guaraci.linearity import lack_of_fit_test
from guaraci.resultados_io import append_linearity_robustness_model_card
from guaraci.robustness import (
    RobustnessResult,
    gaussian_noise_variants,
    run_robustness_protocol,
)


def _criar_model_card_vazio(pasta) -> str:
    caminho = pasta / "model_card.md"
    caminho.write_text("# Model Card\n", encoding="utf-8")
    return str(caminho)


def test_append_sem_model_card_nao_lanca_e_nao_cria_arquivo(tmp_path):
    """model_card.md ausente (ex.: generate_model_card falhou antes) -- a
    funcao devolve em silencio, mesmo padrao das outras append_*."""
    append_linearity_robustness_model_card(str(tmp_path), linearidade=None,
                                           robustez=None)
    assert not (tmp_path / "model_card.md").exists()


def test_append_sem_nenhum_resultado_nao_escreve_secao(tmp_path):
    caminho = _criar_model_card_vazio(tmp_path)
    append_linearity_robustness_model_card(str(tmp_path), linearidade=None,
                                           robustez=None)
    conteudo = open(caminho, encoding="utf-8").read()
    assert "Bloco 13d" not in conteudo


def test_append_linearidade_computavel_aparece_no_model_card(tmp_path):
    caminho = _criar_model_card_vazio(tmp_path)
    rng = np.random.default_rng(0)
    y_ref, y_pred, grupos = [], [], []
    for i, nivel in enumerate(np.linspace(0, 10, 6)):
        for _r in range(3):
            y_ref.append(nivel)
            y_pred.append(nivel + rng.normal(scale=0.02))
            grupos.append(f"G{i}")
    resultado = lack_of_fit_test(np.array(y_ref), np.array(y_pred), np.array(grupos))
    assert resultado.computavel

    append_linearity_robustness_model_card(str(tmp_path), linearidade=resultado,
                                           robustez=None)
    conteudo = open(caminho, encoding="utf-8").read()
    assert "Bloco 13d" in conteudo
    assert "lack-of-fit" in conteudo.lower()
    assert "Protocolo de robustez" not in conteudo   # robustez=None -> nao aparece


def test_append_linearidade_nao_computavel_reporta_motivo(tmp_path):
    caminho = _criar_model_card_vazio(tmp_path)
    y_ref = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 2.1])
    grupos = np.array(["G0", "G1"])
    resultado = lack_of_fit_test(y_ref, y_pred, grupos)
    assert not resultado.computavel

    append_linearity_robustness_model_card(str(tmp_path), linearidade=resultado,
                                           robustez=None)
    conteudo = open(caminho, encoding="utf-8").read()
    assert "Nao computavel" in conteudo
    assert resultado.motivo in conteudo


def test_append_robustez_aparece_com_intervalo_no_model_card(tmp_path):
    caminho = _criar_model_card_vazio(tmp_path)
    robustez = {
        "ruido_gaussiano_0.01": RobustnessResult(
            perturbacao="ruido_gaussiano_0.01", baseline=0.5,
            valores=[0.48, 0.52, 0.55], minimo=0.48, maximo=0.55,
            mediana=0.52, variacao_absoluta=0.07, n_replicas=3),
    }
    append_linearity_robustness_model_card(str(tmp_path), linearidade=None,
                                           robustez=robustez)
    conteudo = open(caminho, encoding="utf-8").read()
    assert "Protocolo de robustez" in conteudo
    assert "0.48" in conteudo and "0.55" in conteudo
    # R2: o INTERVALO e' o dado reportado (nao um veredito) -- confere que
    # a linha da tabela e' o intervalo bruto, sem rotulo de aprovado/
    # reprovado colado ao numero (a prosa explicativa PODE citar essas
    # palavras ao dizer o que o relatorio NAO faz -- e' isso, nao a linha
    # de dado, que este teste protege).
    linha_dado = next(l for l in conteudo.splitlines() if "ruido_gaussiano" in l)
    assert "aprovado" not in linha_dado.lower()
    assert "reprovado" not in linha_dado.lower()


# =========================================================================
#  Fim-a-fim contra dados publicos reais (Corn / Mendeley) -- "rodar
#  contra Corn e Mendeley" do checklist do Bloco 13d.
# =========================================================================

def _caminho_corn():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    caminho = Path(raiz) / "corn.mat"
    return caminho if caminho.is_file() else None


requer_corn = pytest.mark.skipif(
    _caminho_corn() is None,
    reason=("dataset publico Corn ausente. Baixe corn.mat de "
            "https://eigenvector.com/data/Corn/ e aponte "
            "GUARACI_DATASETS_DIR para a pasta que o contem."))


@requer_corn
@pytest.mark.slow
def test_linearidade_e_robustez_no_corn_real(tmp_path):
    import scipy.io as sio

    m = sio.loadmat(str(_caminho_corn()))
    X = np.asarray(m["m5spec"]["data"][0, 0], dtype=float)
    Y = np.asarray(m["propvals"]["data"][0, 0], dtype=float)
    proteina = Y[:, 2]
    # Corn nao tem mae_id (cada uma das 80 amostras e' uma medida unica,
    # sem replica fisica registrada) -- "grupo" = indice individual, i.e.
    # NENHUMA replica verdadeira. E' o caso real de "nao computavel" (L2),
    # nao um bug: reportar honestamente, nao forcar.
    grupos_sem_replica = np.array([f"amostra_{i}" for i in range(len(proteina))])
    r_linearidade = lack_of_fit_test(proteina, proteina, grupos_sem_replica)
    assert not r_linearidade.computavel, (
        "Corn nao tem replica fisica -- o teste de linearidade tem que "
        "reportar 'nao computavel' (L2), nunca forcar um resultado")

    from sklearn.cross_decomposition import PLSRegression

    pls = PLSRegression(n_components=7, scale=False)
    idx = np.arange(80)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    idx_cal, idx_teste = idx[:60], idx[60:]
    pls.fit(X[idx_cal], proteina[idx_cal])

    def _avaliar(Xv):
        pred = pls.predict(Xv[idx_teste]).ravel()
        return float(np.sqrt(np.mean((pred - proteina[idx_teste]) ** 2)))

    variantes_dados = gaussian_noise_variants(X, niveis=(0.001, 0.01), n_replicas=3)
    variantes_callables = {
        nome: [lambda Xv=Xv: _avaliar(Xv) for Xv in lista]
        for nome, lista in variantes_dados.items()
    }
    resultados_robustez = run_robustness_protocol(lambda: _avaliar(X), variantes_callables)
    assert resultados_robustez["ruido_gaussiano_0.01"].variacao_absoluta >= \
        resultados_robustez["ruido_gaussiano_0.001"].variacao_absoluta * 0.5, (
        "ruido 10x maior deveria produzir variacao de RMSEP pelo menos "
        "comparavel (nao necessariamente 10x, mas nao menor) no Corn real")

    caminho = tmp_path / "model_card.md"
    caminho.write_text("# Model Card\n", encoding="utf-8")
    append_linearity_robustness_model_card(
        str(tmp_path), linearidade=r_linearidade, robustez=resultados_robustez)
    conteudo = caminho.read_text(encoding="utf-8")
    assert "Bloco 13d" in conteudo
    assert "Nao computavel" in conteudo
    assert "Protocolo de robustez" in conteudo


def _pasta_mendeley():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "mendeley_ctgg7k4m5g"
    return pasta if (pasta / "NIR8mm1A.csv").is_file() else None


requer_mendeley = pytest.mark.skipif(
    _pasta_mendeley() is None,
    reason=("dataset publico Mendeley ctgg7k4m5g ausente. Baixe com "
            "'python scripts/download_datasets/baixar_mendeley_oleos.py' "
            "e aponte GUARACI_DATASETS_DIR para a pasta que contem "
            "mendeley_ctgg7k4m5g/."))


@requer_mendeley
@pytest.mark.slow
def test_linearidade_e_robustez_no_mendeley_real(tmp_path):
    """Complementa o Corn (quantificacao/PLS-R) com o caminho de
    CLASSIFICACAO (PLS-DA/bal.acc), R3 -- os dois casos representativos."""
    import pandas as pd

    from guaraci.robustness import avaliar_bal_acc_plsda

    pasta = _pasta_mendeley()
    df = pd.read_csv(pasta / "NIR8mm1A.csv")
    contagem = df["Class"].value_counts()
    classes_ok = contagem[contagem >= 5].index
    sub = df[df["Class"].isin(classes_ok)]
    X = sub.iloc[:, 2:].to_numpy(dtype=float)
    y = sub["Class"].astype(str).to_numpy()

    # Mendeley (assim como o Corn) nao tem identificador de replica fisica
    # -- cada garrafa foi medida uma vez. "grupo" = amostra individual,
    # ou seja, ZERO replica verdadeira -- outro caso real de "nao
    # computavel" honesto (L2), nao um bug do teste.
    grupos_sem_replica = np.array([f"amostra_{i}" for i in range(len(y))])
    r_linearidade = lack_of_fit_test(
        np.arange(len(y), dtype=float),   # sem y de referencia continuo aqui
        np.arange(len(y), dtype=float), grupos_sem_replica)
    assert not r_linearidade.computavel

    variantes_dados = gaussian_noise_variants(X, niveis=(0.001, 0.01), n_replicas=3)
    variantes_callables = {
        nome: [lambda Xv=Xv: avaliar_bal_acc_plsda(Xv, y, grupos_sem_replica)
               for Xv in lista]
        for nome, lista in variantes_dados.items()
    }
    resultados_robustez = run_robustness_protocol(
        lambda: avaliar_bal_acc_plsda(X, y, grupos_sem_replica), variantes_callables)

    for r in resultados_robustez.values():
        assert 0.0 <= r.baseline <= 1.0
        assert all(0.0 <= v <= 1.0 for v in r.valores)

    caminho = tmp_path / "model_card.md"
    caminho.write_text("# Model Card\n", encoding="utf-8")
    append_linearity_robustness_model_card(
        str(tmp_path), linearidade=r_linearidade, robustez=resultados_robustez)
    conteudo = caminho.read_text(encoding="utf-8")
    assert "Bloco 13d" in conteudo
    assert "Protocolo de robustez" in conteudo
