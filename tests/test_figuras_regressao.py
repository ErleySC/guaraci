"""Rede de regressão das figuras do pipeline + contrato de MODO CIENTÍFICO.

Roda `executar()` duas vezes (fixtures de sessão) com dados sintéticos:
  - um run de CLASSIFICAÇÃO (nivel=N2, objetivo derivado=classificacao);
  - um run de QUANTIFICAÇÃO (nivel=N3, objetivo derivado=quantificacao).

Depois faz asserções BARATAS sobre os arquivos gerados. Além de travar a
saída visual (PNG válido, tamanho não-trivial, dimensões sãs), agora também
protege o CONTRATO DE MODO introduzido na auditoria: cada objetivo gera
EXCLUSIVAMENTE as figuras pertinentes ao seu propósito — a Classificação não
emite a figura de regressão, e a Quantificação não emite as figuras
supervisionadas de PLS-DA (confusão/ROC/DD-SIMCA/SR-VIP). Antes desta
mudança, N2 e N3 produziam o mesmo conjunto (defeito auditado).

Não é comparação pixel-a-pixel (frágil): valida ESTRUTURA + presença/ausência
do conjunto pertinente ao modo.
"""
import os
import struct

import pytest


# Figuras de CLASSIFICAÇÃO que devem existir num run de classificação (N2 com
# módulos supervisionados ligados). Se alguma sumir, é regressão.
FIGURAS_CLASSIFICACAO = {
    "fig1_pca_scores",                    # overview (comum a todos os modos)
    "fig2_plsda_scores",
    "fig3_outliers_T2_Q",                 # overview/diagnóstico
    "fig4_confusao_e_metricas_por_classe",
    "figS1_selecao_lvs",
    "fig_sprint3_sr_vip",
    "fig6_preprocessamento",              # exploratória via escotilha figuras_detalhadas
    "fig_hca_dendrograma",
    "fig_loadings_pca",
    "fig_roc_auc_multiclasse",
    "fig_sprint3_ddsimca_acceptance",     # N2 força DD-SIMCA one-class
}

# Figuras de QUANTIFICAÇÃO que devem existir num run N3.
FIGURAS_QUANTIFICACAO = {
    "fig1_pca_scores",       # overview
    "fig3_outliers_T2_Q",    # overview
    "figS2_pls_regressao",   # nome de arquivo de fig7_pls_regressao (só em Quantificação)
}

# Contrato de filtragem: figuras que NÃO podem aparecer no modo Quantificação
# (são supervisionadas de classificação — pertencem a outro objetivo).
FIGURAS_PROIBIDAS_EM_QUANTIFICACAO = {
    "fig2_plsda_scores",
    "fig4_confusao_e_metricas_por_classe",
    "fig_roc_auc_multiclasse",
    "fig_sprint3_ddsimca_acceptance",
    "fig_sprint3_sr_vip",
}

# Contrato de filtragem: a figura de regressão não pode aparecer em Classificação.
FIGURAS_PROIBIDAS_EM_CLASSIFICACAO = {
    "figS2_pls_regressao",
}

# Piso de contagem por modo (abaixo disso algo quebrou em massa).
MIN_FIGURAS_CLASSIFICACAO = 10
MIN_FIGURAS_QUANTIFICACAO = 3


def _png_dims(caminho):
    """(largura, altura) de um PNG pelos bytes do header, ou None se não-PNG."""
    with open(caminho, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def _coletar_pngs(figbase):
    pngs = []
    for dp, _, fs in os.walk(figbase):
        for fn in fs:
            if fn.lower().endswith(".png"):
                pngs.append(os.path.join(dp, fn))
    return pngs


def _cfg_base(pq, base, **overrides):
    cfg = pq.Config(
        pasta_entrada=str(base / "dados"),
        pasta_saida_raiz=str(base / "saida"),
        modo="sintetico", n_por_classe=10, n_pontos_sint=60,
        n_replicas_sint=3,   # replicas fisicas -> DD-SIMCA/figuras de merito treinam de verdade
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1,
        n_permutacoes=5, n_permutacoes_wold=5,
        n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3,
        max_lvs=5, figuras_detalhadas=True,
    )
    # Liga os módulos científicos que produzem figura (guardado por hasattr).
    # Benchmark/MC/SHAP ficam DESLIGADOS: são lentos e exigem xgboost/shap.
    for attr, val in [
        ("executar_ddsimca", True), ("executar_opls", True),
        ("executar_etapa4", True), ("executar_wold", True),
        ("comparar_pipelines", True), ("executar_cv_anova", True),
        ("executar_benchmark", False), ("executar_monte_carlo", False),
        ("executar_shap", False),
    ]:
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    os.makedirs(cfg.pasta_entrada, exist_ok=True)
    return cfg


def _rodar(pq, cfg):
    pq.executar(cfg)
    runs = [os.path.join(cfg.pasta_saida_raiz, r)
            for r in os.listdir(cfg.pasta_saida_raiz)]
    assert runs, "executar() não criou pasta de saída"
    figbase = os.path.join(runs[0], "figuras")
    assert os.path.isdir(figbase), "pasta figuras/ não foi criada"
    return figbase, _coletar_pngs(figbase)


@pytest.fixture(scope="session")
def figuras_classificacao(pq, tmp_path_factory):
    """Run de classificação (nivel=N2 -> objetivo derivado=classificacao)."""
    base = tmp_path_factory.mktemp("figclass")
    cfg = _cfg_base(pq, base, nivel="N2")
    return _rodar(pq, cfg)


@pytest.fixture(scope="session")
def figuras_quantificacao(pq, tmp_path_factory):
    """Run de quantificação (nivel=N3 -> objetivo derivado=quantificacao)."""
    base = tmp_path_factory.mktemp("figquant")
    cfg = _cfg_base(pq, base, nivel="N3")
    return _rodar(pq, cfg)


@pytest.mark.slow
def test_figuras_classificacao_presentes(figuras_classificacao):
    """Todas as figuras essenciais de classificação foram geradas."""
    _figbase, pngs = figuras_classificacao
    nomes = {os.path.splitext(os.path.basename(p))[0] for p in pngs}
    faltando = FIGURAS_CLASSIFICACAO - nomes
    assert not faltando, f"Figuras de classificação faltando: {sorted(faltando)}"


@pytest.mark.slow
def test_classificacao_nao_emite_regressao(figuras_classificacao):
    """Contrato de modo: run de classificação NÃO gera a figura de regressão."""
    _figbase, pngs = figuras_classificacao
    nomes = {os.path.splitext(os.path.basename(p))[0] for p in pngs}
    intrusas = FIGURAS_PROIBIDAS_EM_CLASSIFICACAO & nomes
    assert not intrusas, (
        f"Figuras de quantificação vazaram para o modo Classificação: "
        f"{sorted(intrusas)}")


@pytest.mark.slow
def test_figuras_quantificacao_presentes(figuras_quantificacao):
    """As figuras pertinentes ao modo Quantificação foram geradas."""
    _figbase, pngs = figuras_quantificacao
    nomes = {os.path.splitext(os.path.basename(p))[0] for p in pngs}
    faltando = FIGURAS_QUANTIFICACAO - nomes
    assert not faltando, f"Figuras de quantificação faltando: {sorted(faltando)}"


@pytest.mark.slow
def test_quantificacao_nao_emite_classificacao(figuras_quantificacao):
    """Contrato de modo: run de quantificação NÃO gera figuras supervisionadas
    de classificação (confusão/ROC/DD-SIMCA/SR-VIP/scores PLS-DA)."""
    _figbase, pngs = figuras_quantificacao
    nomes = {os.path.splitext(os.path.basename(p))[0] for p in pngs}
    intrusas = FIGURAS_PROIBIDAS_EM_QUANTIFICACAO & nomes
    assert not intrusas, (
        f"Figuras de classificação vazaram para o modo Quantificação: "
        f"{sorted(intrusas)}")


@pytest.mark.slow
def test_contagem_minima_de_figuras(figuras_classificacao, figuras_quantificacao):
    """Cada modo gera um piso de figuras (nenhuma etapa sumiu em massa)."""
    _fb_c, pngs_c = figuras_classificacao
    _fb_q, pngs_q = figuras_quantificacao
    assert len(pngs_c) >= MIN_FIGURAS_CLASSIFICACAO, (
        f"Classificação só gerou {len(pngs_c)} figuras "
        f"(esperado >= {MIN_FIGURAS_CLASSIFICACAO}).")
    assert len(pngs_q) >= MIN_FIGURAS_QUANTIFICACAO, (
        f"Quantificação só gerou {len(pngs_q)} figuras "
        f"(esperado >= {MIN_FIGURAS_QUANTIFICACAO}).")


@pytest.mark.slow
def test_todos_pngs_validos_e_nao_vazios(figuras_classificacao,
                                          figuras_quantificacao):
    """Cada figura é um PNG válido e não-trivial (não é imagem em branco)."""
    problemas = []
    for _figbase, pngs in (figuras_classificacao, figuras_quantificacao):
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
def test_resumo_persiste_figuras_de_merito(figuras_quantificacao):
    """resumo_modelo.txt guarda o bloco de figuras de merito da regressao
    (LOD/LOQ/SEN/SEL) — antes so saiam no console. Fecha o item do roadmap
    'figuras de merito no relatorio' (o resumo alimenta a aba Relatorios)."""
    figbase, _pngs = figuras_quantificacao
    run_dir = os.path.dirname(figbase)
    resumo = os.path.join(run_dir, "logs", "resumo_modelo.txt")
    assert os.path.isfile(resumo), "resumo_modelo.txt nao foi gerado"
    with open(resumo, encoding="utf-8") as f:
        txt = f.read()
    assert "Analytical Figures of Merit" in txt, "bloco de FOM ausente no resumo"
    assert ("Per-species figures of merit" in txt
            or "single pooled model" in txt), "tabela/bloco de FOM ausente"


@pytest.mark.slow
def test_model_card_gerado_com_addendum_de_regressao(figuras_quantificacao):
    """model_card.md (Mitchell et al. 2019) e' gerado junto com o resumo, e
    ganha o addendum de regressao (secao 9) quando ha quantificacao N3 --
    fecha o item do roadmap 'Model Card automatico'."""
    figbase, _pngs = figuras_quantificacao
    run_dir = os.path.dirname(figbase)
    card = os.path.join(run_dir, "logs", "model_card.md")
    assert os.path.isfile(card), "model_card.md nao foi gerado"
    with open(card, encoding="utf-8") as f:
        txt = f.read()
    for secao in ("# Model Card", "## 1. Detalhes do Modelo",
                  "## 2. Uso Pretendido", "## 3. Fatores Relevantes",
                  "## 4. Metricas de Desempenho",
                  "## 5. Dados de Avaliacao/Treino",
                  "## 6. Analises Quantitativas",
                  "## 7. Consideracoes Eticas",
                  "## 8. Ressalvas e Recomendacoes"):
        assert secao in txt, f"secao ausente no model card: {secao}"
    assert "## 9. Addendum -- Quantificacao" in txt, (
        "addendum de regressao ausente (fixture usa nivel=N3)")
    assert "Wold parsimony criterion" in txt, (
        "notas metodologicas ausentes (deveriam vir de _NOTAS_METODOLOGICAS)")


@pytest.mark.slow
def test_dimensoes_sanas(figuras_classificacao, figuras_quantificacao):
    """Largura/altura de cada figura ficam numa faixa plausível."""
    problemas = []
    for _figbase, pngs in (figuras_classificacao, figuras_quantificacao):
        for p in pngs:
            dims = _png_dims(p)
            if dims is None:
                continue
            w, h = dims
            if not (500 <= w <= 20_000 and 500 <= h <= 12_000):
                problemas.append(
                    f"{os.path.basename(p)}: dimensões {w}x{h} fora da faixa")
    assert not problemas, "Dimensões fora do esperado:\n  " + "\n  ".join(problemas)
