# -*- coding: utf-8 -*-
"""Testes de politica_pooled_local.py (Bloco 26) -- reaproveita o portao
de aceite do Bloco 20 (Wilcoxon pareado, 10 seeds), nao um mecanismo
paralelo. Cenarios sinteticos espelham o motivo REAL documentado em
`pipeline.pls_regression_by_species` (variacao inter-especies dominando
o sinal de teor -- por isso o pooled falha e o local funciona)."""
from __future__ import annotations

import numpy as np

from guaraci.politica_pooled_local import (
    MIN_AMOSTRAS_LOCAL_PADRAO,
    decidir_pooled_vs_local,
)


def _dataset(seed, com_offset_especie, especies=("A", "B", "C", "D"),
            n_grupos_por_especie=15, replicas=3, p=30):
    rng = np.random.default_rng(seed)
    eixo = np.arange(p, dtype=float)
    perfil_teor = np.exp(-0.5 * ((eixo - 20) / 3) ** 2)
    offset_especie = ({e: rng.normal(0, 1, size=p) * 2.0 for e in especies}
                      if com_offset_especie else {e: np.zeros(p) for e in especies})

    Xs, ys, rots, grupos = [], [], [], []
    for e in especies:
        for g in range(n_grupos_por_especie):
            teor = rng.uniform(0, 10)
            for _r in range(replicas):
                x = (offset_especie[e] + teor * 0.05 * perfil_teor
                     + rng.normal(0, 0.02, size=p))
                Xs.append(x); ys.append(teor); rots.append(e)
                grupos.append(f"{e}_{g}")
    return (np.array(Xs), np.array(ys), np.array(rots, dtype=str),
            np.array(grupos, dtype=str))


def test_recomenda_local_quando_especie_confunde_o_pooled():
    """Cenario que espelha a razao REAL de pls_regression_by_species
    existir: offset inter-especie domina, pooled falha, local resolve."""
    X, y, rotulos, grupos = _dataset(seed=0, com_offset_especie=True)
    d = decidir_pooled_vs_local(X, y, rotulos, grupos, "A",
                                n_componentes=3, n_seeds=8)
    assert d.dados_suficientes
    assert d.veredito is not None
    assert d.veredito.veredito == "aprovado"
    assert d.recomendacao == "local"


def test_recomenda_pooled_quando_local_nao_ajuda():
    """Sem confundimento inter-especie, o pooled (mais dados de treino)
    nao perde pro local -- recomendacao continua conservadora (pooled)."""
    X, y, rotulos, grupos = _dataset(seed=1, com_offset_especie=False)
    d = decidir_pooled_vs_local(X, y, rotulos, grupos, "A",
                                n_componentes=3, n_seeds=8)
    assert d.recomendacao == "pooled"
    assert d.veredito is not None
    assert d.veredito.veredito != "aprovado"


def test_dados_insuficientes_recomenda_pooled_sem_rodar_portao():
    """min_amostras acima do que a especie tem -- nunca roda o portao (que
    custaria caro), recomendacao fica pooled por definicao."""
    X, y, rotulos, grupos = _dataset(seed=2, com_offset_especie=True,
                                     n_grupos_por_especie=15)
    n_especie = int((rotulos == "A").sum())
    d = decidir_pooled_vs_local(X, y, rotulos, grupos, "A",
                                min_amostras=n_especie + 1)
    assert not d.dados_suficientes
    assert d.veredito is None
    assert d.recomendacao == "pooled"
    assert d.n_amostras == n_especie


def test_usa_o_mesmo_limiar_ja_estabelecido_em_pls_regression_by_species():
    """MIN_AMOSTRAS_LOCAL_PADRAO precisa bater com min_amostras_adult
    (default) de pipeline.pls_regression_by_species -- nunca reinventado."""
    import inspect
    from guaraci.pipeline import pls_regression_by_species
    default_pipeline = inspect.signature(
        pls_regression_by_species).parameters["min_amostras_adult"].default
    assert MIN_AMOSTRAS_LOCAL_PADRAO == default_pipeline


def test_decisao_e_especifica_por_especie_nao_global():
    """Duas especies no MESMO dataset podem receber recomendacoes
    DIFERENTES -- a decisao nunca e' 'local pra tudo' ou 'pooled pra
    tudo' de uma vez so."""
    X, y, rotulos, grupos = _dataset(seed=3, com_offset_especie=True)
    # Espécie "E" tem MUITO menos grupos (abaixo do minimo) -- as outras
    # tem dado suficiente e sinal de confundimento (deveriam ir pra local).
    rng = np.random.default_rng(4)
    p = X.shape[1]
    eixo = np.arange(p, dtype=float)
    perfil_teor = np.exp(-0.5 * ((eixo - 20) / 3) ** 2)
    offset_e = rng.normal(0, 1, size=p) * 2.0
    Xs_e, ys_e, rots_e, grupos_e = [], [], [], []
    for g in range(1):   # so' 1 grupo (3 amostras) -- abaixo do minimo padrao (6)
        teor = rng.uniform(0, 10)
        for _r in range(3):
            x = offset_e + teor * 0.05 * perfil_teor + rng.normal(0, 0.02, size=p)
            Xs_e.append(x); ys_e.append(teor); rots_e.append("E"); grupos_e.append(f"E_{g}")
    X2 = np.vstack([X, np.array(Xs_e)])
    y2 = np.concatenate([y, np.array(ys_e)])
    rotulos2 = np.concatenate([rotulos, np.array(rots_e, dtype=str)])
    grupos2 = np.concatenate([grupos, np.array(grupos_e, dtype=str)])

    d_a = decidir_pooled_vs_local(X2, y2, rotulos2, grupos2, "A", n_componentes=3, n_seeds=8)
    d_e = decidir_pooled_vs_local(X2, y2, rotulos2, grupos2, "E", n_componentes=3, n_seeds=8)
    assert d_a.recomendacao == "local"
    assert d_e.dados_suficientes is False
    assert d_e.recomendacao == "pooled"


# ── Registro no model card (Bloco 26) ─────────────────────────────────────

def test_decisoes_ficam_registradas_no_model_card(tmp_path):
    from guaraci.resultados_io import append_politica_pooled_local_model_card

    caminho = tmp_path / "model_card.md"
    caminho.write_text("# Model Card\n", encoding="utf-8")

    X, y, rotulos, grupos = _dataset(seed=5, com_offset_especie=True,
                                     n_grupos_por_especie=15)
    d_local = decidir_pooled_vs_local(X, y, rotulos, grupos, "A",
                                      n_componentes=3, n_seeds=8)
    d_insuficiente = decidir_pooled_vs_local(
        X, y, rotulos, grupos, "B", min_amostras=10_000)

    append_politica_pooled_local_model_card(str(tmp_path), [d_local, d_insuficiente])
    conteudo = caminho.read_text(encoding="utf-8")
    assert "Bloco 26" in conteudo
    assert "recomendacao=local" in conteudo
    assert "recomendacao=pooled" in conteudo
    assert "portao nao rodado" in conteudo


def test_model_card_sem_arquivo_nao_lanca_excecao(tmp_path):
    from guaraci.resultados_io import append_politica_pooled_local_model_card
    append_politica_pooled_local_model_card(str(tmp_path), [])
    assert not (tmp_path / "model_card.md").exists()
