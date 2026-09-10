# -*- coding: utf-8 -*-
"""O quantificador nao pode ver o rotulo verdadeiro.

REQUISITO DE PROJETO. Quem usa o GUARACI entrega uma amostra desconhecida:
nao sabe a classe, nao sabe se esta adulterada, nao sabe com que. Um R2 de
quantificacao obtido usando a classe VERDADEIRA descreve um cenario que o
usuario final nunca tera' em maos -- e, pior, esconde o erro de
classificacao dentro do numero de quantificacao.

O mode cego (padrao) calibra usando a classe PREDITA. O mode controle usa a
verdadeira, e existe para uma coisa so': isolar erro de quantificacao de
erro de classificacao durante o desenvolvimento.

A prova aqui e' por ENVENENAMENTO: os rotulos verdadeiros sao substituidos
por lixo. Em mode cego o resultado nao pode mudar -- se mudar, e' porque
alguem os leu.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Contrato do seletor ──────────────────────────────────────────────────────

def test_padrao_e_cego(pq):
    """O padrao do software precisa ser o mode que corresponde ao uso real.
    Se um dia alguem inverter o default, este teste falha."""
    assert pq.Config().label_mode == "cego"


def test_modo_cego_devolve_os_preditos(pq):
    verdadeiros = np.array(["A", "A", "B", "B"])
    preditos = np.array(["A", "B", "B", "A"])
    rot, mode = pq.labels_for_quantification(
        pq.Config(label_mode="cego"), verdadeiros, preditos)
    assert mode == "cego"
    assert np.array_equal(rot, preditos)


def test_modo_controle_devolve_os_verdadeiros_e_se_identifica(pq):
    """Controle e' legitimo -- desde que a saida diga que e' controle."""
    verdadeiros = np.array(["A", "A", "B", "B"])
    preditos = np.array(["A", "B", "B", "A"])
    rot, mode = pq.labels_for_quantification(
        pq.Config(label_mode="controle"), verdadeiros, preditos)
    assert mode == "controle"
    assert np.array_equal(rot, verdadeiros)


def test_sem_preditos_o_modo_cego_nao_finge_ser_cego(pq):
    """Sem classificador ajustado, o mode cego nao tem como operar. Cair
    para os verdadeiros e' aceitavel; cair para os verdadeiros DIZENDO que
    e' cego, nao -- seria um resultado de controle disfarcado."""
    verdadeiros = np.array(["A", "B"])
    rot, mode = pq.labels_for_quantification(
        pq.Config(label_mode="cego"), verdadeiros, None)
    assert mode == "controle-forcado"
    assert np.array_equal(rot, verdadeiros)


def test_modo_invalido_falha_alto(pq):
    with pytest.raises(ValueError, match="label_mode"):
        pq.labels_for_quantification(
            pq.Config(label_mode="talvez"), np.array(["A"]), np.array(["A"]))


# ── Prova por envenenamento ──────────────────────────────────────────────────

def _dados_com_teor(pq, semente=3, n_per_class=14):
    """Duas classes espectralmente distintas, teor variando dentro de cada
    uma -- o cenario minimo em que quantificacao por classe faz sentido."""
    rng = np.random.default_rng(semente)
    eixo = np.linspace(4000.0, 10000.0, 160)
    X, y, conc, mae = [], [], [], []
    for k, cls in enumerate(("A", "B")):
        for i in range(n_per_class):
            teor = 10.0 * i / n_per_class
            base = 0.4 + 0.15 * k + 0.03 * np.sin(eixo / (900.0 + 300.0 * k))
            X.append(base + 0.006 * teor + rng.normal(0, 0.001, eixo.size))
            y.append(cls); conc.append(teor); mae.append(f"{cls}{i:02d}")
    return (eixo, np.array(X), np.array(y), np.array(conc, dtype=float),
            np.array(mae))


def test_quantificacao_cega_ignora_rotulo_verdadeiro_envenenado(pq):
    """A prova: com os rotulos verdadeiros trocados por lixo, o resultado da
    quantificacao em mode cego tem que ser IDENTICO.

    Se este teste falhar, o rotulo verdadeiro esta entrando no caminho de
    quantificacao por alguma porta -- que e' exatamente o que o requisito
    proibe.
    """
    _eixo, X, y_true, conc, mae = _dados_com_teor(pq)
    preditos = y_true.copy()          # classificador perfeito, p/ isolar o efeito

    cfg = pq.Config(label_mode="cego", n_splits_cv=2, max_lvs=3)

    rot_ok, modo_ok = pq.labels_for_quantification(cfg, y_true, preditos)
    # Envenena a VERDADE (nao a predicao): rotulos embaralhados de proposito.
    y_lixo = np.array(["Z"] * len(y_true))
    rot_env, modo_env = pq.labels_for_quantification(cfg, y_lixo, preditos)

    assert modo_ok == modo_env == "cego"
    assert np.array_equal(rot_ok, rot_env), (
        "o rotulo verdadeiro influenciou a selecao usada na quantificacao")

    r_ok = pq.r2cv_species_by_adulterant(X, conc, rot_ok, mae, cfg,
                                        min_niveis=3, min_grupos=3)
    r_env = pq.r2cv_species_by_adulterant(X, conc, rot_env, mae, cfg,
                                         min_niveis=3, min_grupos=3)
    if r_ok is None and r_env is None:
        pytest.skip("dataset sintetico sem adulterante nomeado no mae_id")
    assert (r_ok is None) == (r_env is None)
    if r_ok is not None:
        assert r_ok["matriz"].keys() == r_env["matriz"].keys()
        for chave, valor in r_ok["matriz"].items():
            outro = r_env["matriz"][chave]
            assert (np.isnan(valor) and np.isnan(outro)) or valor == outro


def test_modo_controle_de_fato_ve_a_verdade(pq):
    """Contra-prova do teste acima: em mode controle, envenenar a verdade
    MUDA o resultado. Sem isso, o teste anterior poderia estar passando
    porque nenhum dos dois caminhos usa rotulo nenhum."""
    _eixo, _X, y_true, _conc, _mae = _dados_com_teor(pq)
    cfg = pq.Config(label_mode="controle")
    y_lixo = np.array(["Z"] * len(y_true))
    rot_ok, _ = pq.labels_for_quantification(cfg, y_true, y_true.copy())
    rot_env, _ = pq.labels_for_quantification(cfg, y_lixo, y_true.copy())
    assert not np.array_equal(rot_ok, rot_env), (
        "em mode controle a verdade TEM que ser usada -- se envenena-la nao "
        "muda nada, os dois modos sao o mesmo e o teste cego nao prova nada")


def test_erro_de_classificacao_se_propaga_no_modo_cego(pq):
    """No mode cego, um classificador que erra faz a calibracao agrupar
    amostras erradas. Isso NAO e' um defeito a esconder -- e' o que
    aconteceria em producao, e o numero precisa refleti-lo."""
    y_true = np.array(["A", "A", "A", "B", "B", "B"])
    preditos = np.array(["A", "A", "B", "B", "B", "A"])   # 2 erros
    rot, mode = pq.labels_for_quantification(
        pq.Config(label_mode="cego"), y_true, preditos)
    assert mode == "cego"
    assert int(np.sum(rot != y_true)) == 2, (
        "o mode cego deve carregar os erros do classificador para dentro da "
        "quantificacao")
