# -*- coding: utf-8 -*-
"""Testes de duplex_split/spxy_split (Passo 87) -- seguem a mesma
disciplina de agrupamento de test_contrato_validacao_agrupada.py: nenhum
metodo de selecao pode separar replicas do mesmo grupo (mae_id) entre
calibracao e validacao.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from guaraci.dados_io import (
    duplex_split,
    duplex_split_group_aware,
    kennard_stone_split,
    spxy_split,
    spxy_split_group_aware,
)


def _particao_valida(idx_treino, idx_val, n):
    """Confirma que treino+val cobrem TODOS os indices, sem repeticao."""
    assert len(set(idx_treino.tolist()) & set(idx_val.tolist())) == 0
    assert set(idx_treino.tolist()) | set(idx_val.tolist()) == set(range(n))


# =========================================================================
#  DUPLEX
# =========================================================================

def test_duplex_particiona_sem_sobrepor_nem_deixar_de_fora():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(37, 5))
    idx_treino, idx_val = duplex_split(X, frac_treino=0.6)
    _particao_valida(idx_treino, idx_val, len(X))
    assert len(idx_treino) > 0 and len(idx_val) > 0


def test_duplex_frac_treino_aproxima_a_proporcao_pedida():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6))
    idx_treino, idx_val = duplex_split(X, frac_treino=0.75)
    prop = len(idx_treino) / len(X)
    assert 0.70 <= prop <= 0.80, (
        f"proporcao obtida {prop:.3f} longe do pedido 0.75")


def test_duplex_frac_05_e_estritamente_alternado_apos_a_semente():
    """frac_treino=0.5 e' o Duplex classico (Snee, 1977): apos as duas
    sementes (2+2), cada novo ponto alterna estritamente entre os dois
    conjuntos -- com n par, os dois conjuntos saem EXATAMENTE do mesmo
    tamanho."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 4))
    idx_treino, idx_val = duplex_split(X, frac_treino=0.5)
    assert len(idx_treino) == len(idx_val) == 20


@pytest.mark.parametrize("n", [0, 1, 2])
def test_duplex_casos_degenerados_nao_quebram(n):
    X = np.random.default_rng(3).normal(size=(n, 3))
    idx_treino, idx_val = duplex_split(X, frac_treino=0.7)
    _particao_valida(idx_treino, idx_val, n)


def test_duplex_group_aware_nunca_separa_grupo():
    rng = np.random.default_rng(4)
    X_list, mae_list = [], []
    for g in range(10):
        centro = rng.normal(size=4)
        for _ in range(3):
            X_list.append(centro + rng.normal(scale=0.05, size=4))
            mae_list.append(f"G{g}")
    X = np.array(X_list)
    mae = np.array(mae_list)

    idx_treino, idx_val = duplex_split_group_aware(X, mae, frac_cal=0.5)
    _particao_valida(idx_treino, idx_val, len(X))
    assert not (set(mae[idx_treino]) & set(mae[idx_val])), (
        "duplex_split_group_aware separou replicas do mesmo grupo entre "
        "treino e validacao")


# =========================================================================
#  SPXY
# =========================================================================

def test_spxy_particiona_sem_sobrepor_nem_deixar_de_fora():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 5))
    y = rng.normal(size=30)
    idx_treino, idx_val = spxy_split(X, y, frac_treino=0.7)
    _particao_valida(idx_treino, idx_val, len(X))


def test_spxy_aceita_y_1d_e_2d_com_mesmo_resultado():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(25, 4))
    y = rng.normal(size=25)
    idx_1d, _ = spxy_split(X, y, frac_treino=0.6)
    idx_2d, _ = spxy_split(X, y.reshape(-1, 1), frac_treino=0.6)
    assert np.array_equal(idx_1d, idx_2d)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_spxy_casos_degenerados_nao_quebram(n):
    X = np.random.default_rng(7).normal(size=(n, 3))
    y = np.random.default_rng(8).normal(size=n)
    idx_treino, idx_val = spxy_split(X, y, frac_treino=0.7)
    _particao_valida(idx_treino, idx_val, n)


def test_spxy_group_aware_nunca_separa_grupo():
    rng = np.random.default_rng(9)
    X_list, y_list, mae_list = [], [], []
    for g in range(10):
        centro = rng.normal(size=4)
        teor = rng.uniform(0, 10)
        for _ in range(3):
            X_list.append(centro + rng.normal(scale=0.05, size=4))
            y_list.append(teor + rng.normal(scale=0.1))
            mae_list.append(f"G{g}")
    X = np.array(X_list)
    y = np.array(y_list)
    mae = np.array(mae_list)

    idx_treino, idx_val = spxy_split_group_aware(X, y, mae, frac_cal=0.6)
    _particao_valida(idx_treino, idx_val, len(X))
    assert not (set(mae[idx_treino]) & set(mae[idx_val])), (
        "spxy_split_group_aware separou replicas do mesmo grupo entre "
        "treino e validacao")


# =========================================================================
#  Contra-prova: POR QUE SPXY existe -- Kennard-Stone (so' X) pode deixar
#  de fora o extremo do TEOR mesmo cobrindo bem o espectro, porque o
#  extremo de y nao precisa ser um extremo espectral. SPXY (X+y) pega.
# =========================================================================

def test_contraprova_spxy_cobre_extremo_de_y_que_ks_puro_perde():
    rng = np.random.default_rng(10)
    n = 30
    # 29 amostras num cluster espectral compacto (y quase constante) + 1
    # amostra ESPECTRALMENTE afastada (o "outlier" que KS vai pegar
    # primeiro por construcao) tambem com y baixo. O extremo de TEOR fica
    # ESCONDIDO dentro do cluster (indice 5) -- espectralmente comum, mas
    # com o maior y do dataset.
    X = rng.normal(scale=0.1, size=(n, 6))
    X[0] += 20.0                       # outlier espectral, y baixo
    y = rng.normal(scale=0.05, size=n)
    idx_extremo_y = 5
    y[idx_extremo_y] = 100.0           # extremo de TEOR, espectralmente comum

    n_sel = 4
    idx_ks, _ = kennard_stone_split(X, frac_treino=n_sel / n)
    idx_spxy, _ = spxy_split(X, y, frac_treino=n_sel / n)

    assert idx_extremo_y not in idx_ks, (
        "o cenario sintetico precisa ser construido de forma que KS puro "
        "(so' X) DEIXE de fora o extremo de y -- se KS ja' pegou, o teste "
        "nao contrasta nada e precisa ser reconstruido")
    assert idx_extremo_y in idx_spxy, (
        "SPXY deveria incluir o extremo de TEOR no treino (e' exatamente "
        "o problema que a distancia combinada X+y resolve) -- se nao "
        "incluiu, spxy_split nao esta' considerando y de forma efetiva")


# =========================================================================
#  CLI (Bloco 10, tecla [K]) -- `_menu_selecao_amostras`, integrada ao
#  mesmo bloco de planejamento experimental de `_menu_plan`.
# =========================================================================

def _csv_espectros_sinteticos(caminho, n=20, p=6, seed=0, com_alvo=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    cols = {f"canal_{i}": X[:, i] for i in range(p)}
    if com_alvo:
        cols["teor"] = rng.uniform(0, 10, size=n)
    df = pd.DataFrame(cols)
    df.to_csv(caminho, index=False)
    return df


def test_menu_selecao_amostras_cli_kennard_stone(monkeypatch, tmp_path):
    import guaraci.guaraci as guaraci_mod

    caminho_csv = str(tmp_path / "espectros.csv")
    _csv_espectros_sinteticos(caminho_csv, com_alvo=False)
    caminho_saida = str(tmp_path / "saida_ks.csv")

    respostas = iter([
        caminho_csv,   # CSV
        "",            # coluna de referencia: nenhuma
        "1",           # metodo: Kennard-Stone
        "0.6",         # fracao de calibracao
        caminho_saida, # arquivo de saida
        "",            # Enter no _pause() final
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    guaraci_mod._menu_selecao_amostras(guaraci_mod.Config())

    assert os.path.isfile(caminho_saida)
    df_saida = pd.read_csv(caminho_saida)
    assert set(df_saida["conjunto"]) == {"calibracao", "validacao"}
    assert len(df_saida) == 20


def test_menu_selecao_amostras_cli_spxy_com_coluna_alvo(monkeypatch, tmp_path):
    import guaraci.guaraci as guaraci_mod

    caminho_csv = str(tmp_path / "espectros_teor.csv")
    _csv_espectros_sinteticos(caminho_csv, com_alvo=True)
    caminho_saida = str(tmp_path / "saida_spxy.csv")

    respostas = iter([
        caminho_csv,
        "teor",   # coluna de referencia -- habilita SPXY como opcao (3)
        "3",      # metodo: SPXY
        "0.7",
        caminho_saida,
        "",
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    guaraci_mod._menu_selecao_amostras(guaraci_mod.Config())

    assert os.path.isfile(caminho_saida)
    df_saida = pd.read_csv(caminho_saida)
    assert set(df_saida["conjunto"]) == {"calibracao", "validacao"}
    assert "teor" in df_saida.columns


def test_menu_selecao_amostras_cli_arquivo_ausente_nao_gera_saida(monkeypatch, tmp_path):
    """Contra-prova: caminho de CSV inexistente tem que falhar cedo, sem
    gerar arquivo nenhum -- mesmo padrao dos outros menus do Bloco 10."""
    import guaraci.guaraci as guaraci_mod

    respostas = iter([
        str(tmp_path / "isto_nao_existe.csv"),
        "",   # Enter no _pause() apos o erro
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    guaraci_mod._menu_selecao_amostras(guaraci_mod.Config())

    assert list(tmp_path.iterdir()) == []
