# -*- coding: utf-8 -*-
"""Teste de propriedade (Hypothesis) para a condicao de seguranca
NAO-NEGOCIAVEL de `aumento_dados.AumentoVRM` (Parte A do aumento de dados
VRM): toda amostra aumentada herda o `group_id`/`mae_id` da amostra
original, entao NENHUM split via `StableStratifiedGroupKFold` pode separar
uma amostra original de suas variantes aumentadas entre treino e
validacao.

NOTA DE DESENHO DO TESTE (por que nao basta comparar `set(grupos_treino)
& set(grupos_validacao)`): `StableStratifiedGroupKFold` garante, POR
CONTRATO, que um `group_id` e' sempre atomico -- nunca aparece em dois
folds ao mesmo tempo, seja o `group_id` correto ou um inventado. Ou seja,
"nenhum ID de grupo aparece em treino E validacao" seria verdade mesmo
numa implementacao QUEBRADA de `AumentoVRM` que desse um `group_id` NOVO
a cada variante -- o splitter respeitaria esse ID novo tao atomicamente
quanto respeitaria o correto, e o teste passaria por vacuidade. A
propriedade de fato exigida (amostra original e suas variantes NUNCA
separadas) so' e' verificavel rastreando, PARALELAMENTE ao `group_id`
usado pelo split, um `indice_original` que NAO passa pelo splitter (so'
serve para a asserção): para cada amostra original, todas as linhas
derivadas dela (ela mesma + suas `n_aumentos` variantes) precisam cair no
MESMO fold -- isso so' e' garantido pelo splitter SE o `group_id` delas
for de fato o mesmo (heranca de grupo correta). A contra-prova abaixo
constroi deliberadamente um `group_id` novo por variante e mostra que,
sob essa mesma checagem por `indice_original`, o vazamento aparece.
"""
from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from guaraci.aumento_dados import AumentoVRM
from guaraci.validacao_estatistica import StableStratifiedGroupKFold


def _dados_sinteticos(n_grupos: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_grupos, n_features))
    y = rng.normal(size=n_grupos)
    grupos = np.array([f"g{i}" for i in range(n_grupos)])
    return X, y, grupos


def _fold_por_linha(folds, n_total: int) -> np.ndarray:
    """`splitter.split(...)` devolve, por fold, `(idx_treino, idx_val)`;
    os `idx_val` de todos os folds particionam TODAS as linhas exatamente
    uma vez (k-fold padrao) -- converte para um array `fold_id[i]` = em
    qual fold a linha `i` caiu como VALIDACAO, para comparar
    "mesma amostra original -> mesmo fold" com uma unica indexacao."""
    fold_id = np.full(n_total, -1, dtype=int)
    for f, (_idx_tr, idx_va) in enumerate(folds):
        fold_id[idx_va] = f
    assert (fold_id >= 0).all(), "k-fold nao cobriu todas as linhas em validacao"
    return fold_id


@given(
    n_grupos=st.integers(min_value=6, max_value=20),
    n_features=st.integers(min_value=4, max_value=30),
    n_aumentos=st.integers(min_value=1, max_value=4),
    n_splits=st.integers(min_value=2, max_value=4),
    amplitude_multiplicativa=st.floats(min_value=0.0, max_value=0.2),
    amplitude_baseline=st.floats(min_value=0.0, max_value=0.2),
    seed=st.integers(min_value=0, max_value=10_000),
    seed_split=st.integers(min_value=0, max_value=10_000),
)
@settings(deadline=None)
def test_vrm_group_aware_nunca_separa_original_de_variante_aumentada(
        n_grupos, n_features, n_aumentos, n_splits,
        amplitude_multiplicativa, amplitude_baseline, seed, seed_split):
    """Para QUALQUER combinacao de tamanho de dado/hiperparametro de VRM
    dentro de faixas razoaveis, e QUALQUER particao gerada por
    `StableStratifiedGroupKFold` sobre (dado original + dado aumentado),
    a amostra original `i` e TODAS as suas variantes aumentadas caem no
    MESMO fold de validacao -- nunca uma no fold de treino e outra no de
    validacao."""
    X, y, grupos = _dados_sinteticos(n_grupos, n_features, seed)

    av = AumentoVRM(
        n_aumentos=n_aumentos,
        amplitude_multiplicativa=amplitude_multiplicativa,
        amplitude_baseline=amplitude_baseline,
        seed=seed,
    )
    X_full, _y_full, grupos_full = av.aumentar(X, y, grupos)

    # `indice_original`: qual amostra original (0..n_grupos-1) deu origem
    # a cada linha de X_full -- NAO usado no split (so' na asserção). A
    # ordem de `aumentar` e' [originais (n linhas)] + [variantes, em
    # blocos de n_aumentos por amostra original, na mesma ordem] (ver
    # docstring de `AumentoVRM.gerar`/`aumentar`).
    indice_original = np.concatenate([
        np.arange(n_grupos),
        np.repeat(np.arange(n_grupos), n_aumentos),
    ])
    assert len(indice_original) == len(X_full)

    # Grupo unico por construcao (aumento nunca cria grupo novo) -- checagem
    # direta da condicao de seguranca, independente do split.
    assert len(np.unique(grupos_full)) == n_grupos, (
        "aumento criou grupo(s) novo(s) -- contrato de heranca de grupo "
        "quebrado antes mesmo do split")
    for i in range(n_grupos):
        assert set(grupos_full[indice_original == i]) == {grupos[i]}, (
            f"amostra original {i} tem variante(s) com group_id diferente "
            f"do seu proprio -- heranca de grupo quebrada")

    splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed_split)
    alvo_estratificacao = np.zeros(len(X_full))
    folds = list(splitter.split(np.zeros(len(X_full)), alvo_estratificacao,
                                 groups=grupos_full))
    fold_id = _fold_por_linha(folds, len(X_full))

    for i in range(n_grupos):
        folds_da_amostra = set(fold_id[indice_original == i].tolist())
        assert len(folds_da_amostra) == 1, (
            f"amostra original {i} (grupo {grupos[i]!r}) e suas variantes "
            f"aumentadas caíram em folds diferentes ({folds_da_amostra}) -- "
            "vazamento treino->validacao")


def test_contraprova_grupo_novo_por_aumento_de_fato_vaza_entre_folds():
    """Contra-prova: reproduz deliberadamente a quebra de contrato (cada
    variante aumentada recebendo um `group_id` NOVO, como aconteceria se
    alguem comentasse a linha `grupos_novo = np.repeat(grupos, k, axis=0)`
    em `AumentoVRM.gerar` -- verificacao manual descrita no relatorio
    desta tarefa: a quebra foi introduzida de proposito, o teste de
    propriedade acima foi confirmado FALHANDO, e a mudanca foi revertida)
    e confirma que a MESMA checagem por `indice_original` usada acima PEGA
    o vazamento. Sem esta contra-prova, o teste principal poderia estar
    passando so' porque os dados sinteticos/splitter nunca exercitam o
    caminho que vazaria, nao porque a implementacao esta correta."""
    n_grupos, n_aumentos = 8, 3
    X, y, grupos = _dados_sinteticos(n_grupos=n_grupos, n_features=10, seed=0)

    av = AumentoVRM(n_aumentos=n_aumentos, amplitude_multiplicativa=0.05,
                     amplitude_baseline=0.05, seed=0)
    X_novo, _y_novo, _grupos_novo_correto = av.gerar(X, y, grupos)

    # Implementacao QUEBRADA: grupo novo por variante em vez de herdar o
    # da amostra original.
    grupos_novo_quebrado = np.array([f"novo_{i}" for i in range(len(X_novo))])
    X_full = np.vstack([X, X_novo])
    grupos_full_quebrado = np.concatenate([grupos, grupos_novo_quebrado])
    indice_original = np.concatenate([
        np.arange(n_grupos), np.repeat(np.arange(n_grupos), n_aumentos)])

    splitter = StableStratifiedGroupKFold(n_splits=4, seed=0)
    folds = list(splitter.split(np.zeros(len(X_full)), np.zeros(len(X_full)),
                                 groups=grupos_full_quebrado))
    fold_id = _fold_por_linha(folds, len(X_full))

    vazou = any(
        len(set(fold_id[indice_original == i].tolist())) > 1
        for i in range(n_grupos)
    )
    assert vazou, (
        "a implementacao quebrada (grupo novo por variante) deveria ter "
        "separado pelo menos uma amostra original de alguma variante "
        "entre folds -- se nao vazou, o cenario de contra-prova nao "
        "exercita o caminho que deveria, e o teste principal nao estaria "
        "provando nada")
