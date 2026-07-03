"""Rede de regressão das figuras do pipeline.

Roda `executar()` UMA vez (fixture de sessão) com dados sintéticos e todas as
figuras ligadas, e depois faz asserções BARATAS sobre os arquivos gerados.

Objetivo: travar a saída visual ANTES de extrair as ~40 funções fig_* do
monólito (Fase H). Os testes de unidade atuais não pegam uma figura que
some, quebra ou vira uma imagem vazia durante a mudança de código — estes
pegam. Não é comparação pixel-a-pixel (frágil): valida ESTRUTURA (PNG válido,
tamanho não-trivial, dimensões sãs) + presença do conjunto essencial.
"""
import os
import struct

import pytest


# Figuras que DEVEM existir sempre (essenciais + detalhadas robustas que não
# dependem de módulos científicos opcionais). Se alguma sumir, é regressão.
FIGURAS_ESSENCIAIS = {
    "fig1_pca_scores",
    "fig2_plsda_scores",
    "fig3_outliers_T2_Q",
    "fig4_confusao_e_metricas_por_classe",
    "figS1_selecao_lvs",
    "fig_sprint3_sr_vip",
    "fig6_preprocessamento",
    "fig_hca_dendrograma",
    "fig_loadings_pca",
    "fig_roc_auc_multiclasse",
}

# Piso de contagem: o run abaixo gera ~21 figuras; abaixo de 15 algo quebrou
# em massa (ex.: uma etapa inteira deixou de produzir figuras).
MIN_FIGURAS = 15


def _png_dims(caminho):
    """(largura, altura) de um PNG pelos bytes do header, ou None se não-PNG."""
    with open(caminho, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


@pytest.fixture(scope="session")
def figuras_geradas(pq, tmp_path_factory):
    """Roda o pipeline sintético completo uma vez; devolve (figbase, [pngs])."""
    base = tmp_path_factory.mktemp("figregr")
    cfg = pq.Config(
        pasta_entrada=str(base / "dados"),
        pasta_saida_raiz=str(base / "saida"),
        modo="sintetico", n_por_classe=10, n_pontos_sint=60,
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1,
        n_permutacoes=5, n_permutacoes_wold=5,
        n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3,
        max_lvs=5, figuras_detalhadas=True,
    )
    # Liga os módulos científicos que produzem figura (guardado por hasattr
    # para não quebrar se algum atributo for renomeado). Benchmark/MC/SHAP
    # ficam DESLIGADOS de propósito: são lentos e exigem xgboost/shap.
    for attr, val in [
        ("executar_ddsimca", True), ("executar_opls", True),
        ("executar_etapa4", True), ("executar_wold", True),
        ("comparar_pipelines", True), ("executar_cv_anova", True),
        ("executar_benchmark", False), ("executar_monte_carlo", False),
        ("executar_shap", False),
    ]:
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)

    os.makedirs(cfg.pasta_entrada, exist_ok=True)
    pq.executar(cfg)

    runs = [os.path.join(cfg.pasta_saida_raiz, r)
            for r in os.listdir(cfg.pasta_saida_raiz)]
    assert runs, "executar() não criou pasta de saída"
    figbase = os.path.join(runs[0], "figuras")
    assert os.path.isdir(figbase), "pasta figuras/ não foi criada"

    pngs = []
    for dp, _, fs in os.walk(figbase):
        for fn in fs:
            if fn.lower().endswith(".png"):
                pngs.append(os.path.join(dp, fn))
    return figbase, pngs


@pytest.mark.slow
def test_figuras_essenciais_presentes(figuras_geradas):
    """Todas as figuras essenciais foram geradas (nenhuma sumiu)."""
    _figbase, pngs = figuras_geradas
    nomes = {os.path.splitext(os.path.basename(p))[0] for p in pngs}
    faltando = FIGURAS_ESSENCIAIS - nomes
    assert not faltando, f"Figuras essenciais faltando: {sorted(faltando)}"


@pytest.mark.slow
def test_contagem_minima_de_figuras(figuras_geradas):
    """O run gera pelo menos MIN_FIGURAS figuras (nenhuma etapa sumiu em massa)."""
    _figbase, pngs = figuras_geradas
    assert len(pngs) >= MIN_FIGURAS, (
        f"Só {len(pngs)} figuras (esperado >= {MIN_FIGURAS}). "
        f"Alguma etapa parou de gerar figuras?")


@pytest.mark.slow
def test_todos_pngs_validos_e_nao_vazios(figuras_geradas):
    """Cada figura é um PNG válido e não-trivial (não é imagem em branco/corrompida)."""
    _figbase, pngs = figuras_geradas
    problemas = []
    for p in pngs:
        nome = os.path.basename(p)
        tamanho = os.path.getsize(p)
        dims = _png_dims(p)
        if dims is None:
            problemas.append(f"{nome}: não é PNG válido (header)")
        elif tamanho < 5_000:
            problemas.append(f"{nome}: {tamanho}B — pequeno demais (figura vazia?)")
    assert not problemas, "Figuras inválidas/vazias:\n  " + "\n  ".join(problemas)


@pytest.mark.slow
def test_dimensoes_sanas(figuras_geradas):
    """Largura/altura de cada figura ficam numa faixa plausível (não 1x1 nem gigante)."""
    _figbase, pngs = figuras_geradas
    problemas = []
    for p in pngs:
        dims = _png_dims(p)
        if dims is None:
            continue
        w, h = dims
        if not (500 <= w <= 20_000 and 500 <= h <= 12_000):
            problemas.append(f"{os.path.basename(p)}: dimensões {w}x{h} fora da faixa")
    assert not problemas, "Dimensões fora do esperado:\n  " + "\n  ".join(problemas)
