# -*- coding: utf-8 -*-
"""Validacao de Fluorescencia EEM (excitacao-emissao) REAL contra dataset
publico Zenodo `10.5281/zenodo.19755088` -- Passo 149 (Fase C da
auditoria das 11 tecnicas, 2026-09-04).

Dataset: 330 espectros EEM (35 excitacoes x 270 emissoes) de azeite de
oliva extra-virgem (3 marcas -- Earl/Olivoila/Luhua) misturado com 5
tipos de adulterante (milho, canola, amendoim, soja, noz) em 10 niveis
de fracao de azeite (9,09%-90,91%, passo de 1/11), 3 medicoes
independentes ("rodadas") por combinacao marca/adulterante/fracao.
Licenca **CC BY 4.0** (confirmado via API oficial do Zenodo).

Substitui o dataset EEM cogitado anteriormente (Mendeley `g6y69g8gwm`,
registrado em `eem_multiway.py` como "parser fora de escopo" por
formato irregular de instrumento) -- ESTE dataset tem formato bem mais
regular (ver `eem_io.py`), permitindo pela primeira vez rodar o PARAFAC
generalizado (Passo 144/145, ate aqui so' provado por contra-prova
sintetica) contra EEM real.

ACHADO SOBRE O FORMATO (Passo 149): 16/330 pastas (4,8%) usam o nome de
arquivo `0.dat` em vez do padrao `0_RM.dat` -- mesmo conteudo, variacao
de nomenclatura real do dataset. `eem_io._localizar_arquivo_dat` busca
por glob (`*.dat`) em vez de nome fixo -- as 16 amostras SAO
recuperadas (330/330 carregadas), taxa de descarte de LINHAS dentro dos
arquivos = 0% (formato interno de cada arquivo e' perfeitamente
regular: 270 linhas de dado, 37 colunas, sempre)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

#: Piso de sanidade -- medido em 2026-09-04: correlacao |r|=0,85-0,95
#: (R=2..5 componentes) entre um fator PARAFAC e a fracao real de
#: azeite, RMSEP 3,5-5,6 p.p. na regressao PLS direta (group-aware,
#: n_lv=3..10). Pisos com folga generosa sob o medido.
CORR_PARAFAC_MINIMA = 0.7
R2_REGRESSAO_MINIMO = 0.85
TAXA_DESCARTE_MAXIMA = 0.20   # limiar da propria instrucao (Passo 149)


def _pasta_eem():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "zenodo_19755088_eem_azeite" / "data"
    return pasta if pasta.is_dir() else None


requer_eem_zenodo = pytest.mark.skipif(
    _pasta_eem() is None,
    reason=("dataset Zenodo 19755088 (EEM azeite) ausente. Baixe com "
            "'python scripts/download_datasets/baixar_zenodo_eem_azeite.py' "
            "e aponte GUARACI_DATASETS_DIR para a pasta que contem "
            "zenodo_19755088_eem_azeite/."))


@requer_eem_zenodo
def test_carregar_dataset_eem_le_as_330_amostras_com_baixo_descarte():
    """Contra-prova do parser (regra 9): confirma que o dataset real
    carrega quase por completo e que a taxa de linhas descartadas
    dentro dos arquivos fica abaixo do limiar da instrucao (>20% exigiria
    documentar como achado de formato ruim -- aqui e' 0%, achado de
    formato BOM, tambem documentado, nao escondido)."""
    from guaraci.eem_io import carregar_dataset_eem_azeite

    matrizes, meta, relatorio = carregar_dataset_eem_azeite(_pasta_eem())

    assert relatorio["n_amostras_ok"] == 330, (
        f"esperava 330 amostras (10 fracoes x 11 combinacoes marca/"
        f"adulterante x 3 rodadas), achou {relatorio['n_amostras_ok']} "
        f"-- premissa do dataset mudou.")
    assert relatorio["n_amostras_ignoradas"] == 0
    assert relatorio["taxa_descarte_linhas_media"] <= TAXA_DESCARTE_MAXIMA
    assert set(meta["adulterante"].unique()) == {
        "milho", "canola", "amendoim", "soja", "noz"}
    assert set(meta["marca"].unique()) == {"Earl", "Olivoila", "Luhua"}

    uma = next(iter(matrizes.values()))
    assert uma.shape == (35, 270), (
        f"forma da matriz EEM {uma.shape} != (35 excitacao, 270 emissao) "
        f"esperado -- formato do dataset mudou.")


@requer_eem_zenodo
@pytest.mark.slow
def test_parafac_eem_extrai_fator_correlacionado_com_fracao_real_de_azeite(pq):
    """Passo 149: conecta o PARAFAC generalizado (ate aqui so' provado
    por contra-prova sintetica, `tests/test_eem_multiway.py`) ao dado
    REAL pela primeira vez. Nao ha' rotulo de treino -- PARAFAC e'
    NAO-SUPERVISIONADO -- entao o teste e' se ALGUM dos R fatores
    recuperados correlaciona com a fracao de azeite REAL conhecida
    (nunca vista pelo PARAFAC), a mesma logica de validacao usada na
    contra-prova sintetica do modulo."""
    from guaraci.eem_io import carregar_dataset_eem_azeite
    from guaraci.eem_multiway import construir_tensor_eem, parafac_eem

    matrizes, meta, _relatorio = carregar_dataset_eem_azeite(_pasta_eem())
    tensor, ids = construir_tensor_eem(matrizes)
    y = meta.loc[ids, "fracao_azeite_pct"].to_numpy()

    resultado = parafac_eem(tensor, n_componentes=3, max_iter=200, seed=0)
    assert resultado.erro_reconstrucao_relativo < 0.30, (
        f"erro de reconstrucao {resultado.erro_reconstrucao_relativo:.3f} "
        f"alto demais -- PARAFAC pode nao estar convergindo neste dado real.")

    melhor_corr = max(
        abs(np.corrcoef(resultado.fator_amostra[:, k], y)[0, 1])
        for k in range(resultado.fator_amostra.shape[1]))
    print(f"\n[Passo 149] PARAFAC (R=3): erro_reconstrucao="
          f"{resultado.erro_reconstrucao_relativo:.3f}, melhor |r| com "
          f"fracao de azeite real = {melhor_corr:.3f}")
    assert melhor_corr > CORR_PARAFAC_MINIMA, (
        f"melhor correlacao |r|={melhor_corr:.3f} abaixo do piso "
        f"{CORR_PARAFAC_MINIMA} -- PARAFAC nao estaria capturando o "
        f"gradiente de adulteracao real.")


@requer_eem_zenodo
@pytest.mark.slow
def test_regressao_pls_group_aware_quantifica_adulteracao(pq):
    """Validacao group-aware (regra 5) da quantificacao direta: PLS
    sobre o espectro EEM completo (achatado, 35x270=9450 canais) prevendo
    a fracao real de azeite. Grupo = (marca, adulterante, razao) --
    invariante entre as 3 RODADAS (medicoes independentes da MESMA
    condicao fisica) -- nunca deixa as 3 rodadas de uma condicao se
    separarem entre treino/validacao. `GroupKFold` usado diretamente
    (nao via `pq.executar()`/mode=csv, que nao suporta uma coluna de
    agrupamento arbitraria -- mesma limitacao ja documentada para o
    dataset de esgoto UV-Vis, Passo 147) -- harness limpo, mesmo
    espirito de `selecao_variaveis._avaliar_subset_cv`."""
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from guaraci.eem_io import carregar_dataset_eem_azeite

    matrizes, meta, _relatorio = carregar_dataset_eem_azeite(_pasta_eem())
    ids = list(matrizes.keys())
    X = np.stack([matrizes[i].ravel() for i in ids])
    y = meta.loc[ids, "fracao_azeite_pct"].to_numpy()
    grupos = (meta.loc[ids, "marca"] + "_" + meta.loc[ids, "adulterante"] + "_"
              + meta.loc[ids, "razao_azeite"].astype(str) + "-"
              + meta.loc[ids, "razao_adulterante"].astype(str)).to_numpy()
    n_grupos = len(set(grupos))
    assert n_grupos == 110, (
        f"esperava 110 grupos (11 combinacoes marca/adulterante x 10 "
        f"fracoes), achou {n_grupos}.")

    pipe = Pipeline([
        ("mc", StandardScaler(with_std=False)),
        ("pls", PLSRegression(n_components=8, scale=False)),
    ])
    y_cv = cross_val_predict(pipe, X, y, groups=grupos,
                              cv=GroupKFold(n_splits=5))
    ss_res = float(np.sum((y - y_cv) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rmsep = float(np.sqrt(np.mean((y - y_cv) ** 2)))
    print(f"\n[Passo 149] PLS group-aware (n_lv=8): R2={r2:.4f}, "
          f"RMSEP={rmsep:.2f} p.p. de fracao de azeite")

    assert r2 > R2_REGRESSAO_MINIMO, (
        f"R2={r2:.4f} abaixo do piso {R2_REGRESSAO_MINIMO} -- quantificacao "
        f"de adulteracao a partir do espectro EEM completo nao estaria "
        f"funcionando neste dataset real.")
