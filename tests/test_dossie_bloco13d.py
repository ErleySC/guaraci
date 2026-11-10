# -*- coding: utf-8 -*-
"""Testes de integracao do dossie (Bloco 13d): `append_linearity_
robustness_model_card` (model_card.md + LOD/LOQ + faixa de trabalho +
incerteza + linearidade + robustez, mesmo mecanismo append-only ja usado
para regressao/identificacao/pureza).
"""
from __future__ import annotations


import numpy as np

from guaraci.linearity import lack_of_fit_test
from guaraci.resultados_io import append_linearity_robustness_model_card
from guaraci.robustness import (
    RobustnessResult,
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
    assert "Linearidade formal e robustez" not in conteudo


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
    assert "Linearidade formal e robustez" in conteudo
    assert "falta de ajuste" in conteudo.lower()   # relatorio em portugues (padrao)
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




