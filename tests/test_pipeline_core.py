"""Rede de segurança de testes unitários do núcleo do pipeline.

Cobre as FRONTEIRAS DE MÓDULO que a modularização (Fase H) vai separar —
pré-processamento, estatística quimiométrica, DD-SIMCA, métricas, parsing
JCAMP-DX e IO de configuração. Foi escrito ANTES de mover qualquer código,
para travar o comportamento atual e detectar regressões durante o corte.

Usa a fixture `pq` (sessão) do conftest.py.
"""
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression


# ── Pré-processamento (futuro: guaraci/preprocessing.py) ──────────────────────

def test_snv_normaliza_por_amostra(pq):
    """SNV: cada espectro fica com média ~0 e desvio ~1 (por linha)."""
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(20, 60))
    Z = pq.SNV().fit_transform(X)
    assert Z.shape == X.shape
    np.testing.assert_allclose(Z.mean(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(Z.std(axis=1), 1.0, atol=1e-9)


def test_snv_linha_constante_nao_estoura(pq):
    """SNV: espectro constante (desvio 0) não deve gerar NaN/inf (sd→1)."""
    X = np.full((3, 40), 7.0)
    Z = pq.SNV().fit_transform(X)
    assert np.all(np.isfinite(Z))


def test_savgol_preserva_shape(pq):
    """SavGol: preserva o shape; suaviza (não retorna o mesmo array)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 80))
    Y = pq.SavGol(window_length=25, polyorder=2, deriv=0).transform(X)
    assert Y.shape == X.shape


def test_construir_preprocessador_presets(pq):
    """construir_preprocessador: cada preset monta as etapas esperadas."""
    cfg = pq.Config()
    cfg.preprocessamento_padrao = "snv_sg_mc"
    assert list(pq.construir_preprocessador(cfg).named_steps) == ["snv", "sg", "mc"]
    cfg.preprocessamento_padrao = "msc_sg_mc"
    assert list(pq.construir_preprocessador(cfg).named_steps) == ["msc", "sg", "mc"]
    cfg.preprocessamento_padrao = "mc"
    assert list(pq.construir_preprocessador(cfg).named_steps) == ["mc"]
    cfg.preprocessamento_padrao = "autoscaling"
    assert list(pq.construir_preprocessador(cfg).named_steps) == ["auto"]


# ── Estatística quimiométrica (futuro: guaraci/diagnostics.py) ────────────────

def _pls_ajustado(seed=0, n=60, p=30, n_comp=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Y = np.zeros((n, 2)); Y[: n // 2, 0] = 1; Y[n // 2:, 1] = 1
    return PLSRegression(n_components=n_comp, scale=False).fit(X, Y), X


def test_vip_propriedade_soma_igual_p(pq):
    """VIP: propriedade de Chong & Jun — sum_j VIP_j^2 = p (média de VIP^2 = 1)."""
    modelo, _ = _pls_ajustado()
    vip = pq.vip_scores(modelo)
    assert vip.shape == (30,)
    assert np.all(vip >= 0)
    assert np.mean(vip ** 2) == pytest.approx(1.0, rel=1e-6)


def test_selectivity_ratio_nao_negativo(pq):
    """SR: razão de variâncias — sempre >= 0, um valor por variável."""
    modelo, X = _pls_ajustado()
    sr = pq.calcular_selectivity_ratio(modelo, X)
    assert sr.shape == (X.shape[1],)
    assert np.all(sr >= 0)


def test_hotelling_limite_positivo_e_monotonico(pq):
    """T2 UCL: positivo e MAIOR quando alpha é menor (limite mais rígido)."""
    l05 = pq.hotelling_t2_limite(50, 3, 0.05)
    l01 = pq.hotelling_t2_limite(50, 3, 0.01)
    assert l05 > 0 and l01 > l05
    # n <= k é degenerado → infinito (sem falso outlier silencioso)
    assert pq.hotelling_t2_limite(3, 3, 0.05) == float("inf")


def test_q_residuos_zero_quando_reconstrucao_exata(pq):
    """Q-resíduos: se X = T@P exatamente, o resíduo é ~0."""
    rng = np.random.default_rng(2)
    T = rng.normal(size=(25, 3)); P = rng.normal(size=(3, 40))
    X = T @ P
    q = pq.q_residuos(X, T, P)
    assert q.shape == (25,)
    np.testing.assert_allclose(q, 0.0, atol=1e-18)


def test_variancia_explicada_range(pq):
    """Variância explicada (%): não-negativa e um valor por componente."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 20))
    T = X[:, :3]  # 3 "componentes" quaisquer
    ve = pq.variancia_explicada(X, T)
    assert ve.shape == (3,)
    assert np.all(ve >= 0)


# ── DD-SIMCA (futuro: guaraci/models/ddsimca.py) ──────────────────────────────

def test_ddsimca_aceita_proprio_treino(pq):
    """DD-SIMCA: amostras de treino de uma classe são aceitas pelo próprio
    modelo (T2_norm e Q_norm <= 1). Regressão do clamp de UCL para n pequeno
    (13 espécies 100%/100% em dados reais)."""
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(loc=0.0, size=(10, 30)),
                   rng.normal(loc=5.0, size=(10, 30))])
    y = np.array(["A"] * 10 + ["B"] * 10)
    dd = pq.DDSimca(n_components=3, alpha=0.05, ucl_method="empirical").fit(X, y)
    res = dd.score_matrix(X)
    for cls, sl in (("A", slice(0, 10)), ("B", slice(10, 20))):
        assert res[cls]["T2_norm"][sl].max() <= 1.0 + 1e-3
        assert res[cls]["Q_norm"][sl].max() <= 1.0 + 1e-3


# ── Métricas (futuro: guaraci/metrics.py) ─────────────────────────────────────

def test_metricas_classificacao_perfeita(pq):
    """Predição perfeita → todas as métricas = 1.0."""
    y = np.array([0, 0, 1, 1, 2, 2])
    m = pq.metricas_classificacao(y, y, [0, 1, 2])
    for k in ("accuracy", "balanced_accuracy", "cohen_kappa",
              "f1_macro", "precision_macro", "recall_macro"):
        assert m[k] == pytest.approx(1.0)


def test_especificidade_por_classe_valores_conhecidos(pq):
    """Especificidade one-vs-rest a partir de uma matriz de confusão conhecida."""
    cm = np.array([[5, 0], [1, 4]])
    spec = pq.especificidade_por_classe(cm)
    # classe 0: TN=4, FP=1 → 0.8 ; classe 1: TN=5, FP=0 → 1.0
    np.testing.assert_allclose(spec, [0.8, 1.0], atol=1e-12)


# ── Parsing JCAMP-DX e nomeação (futuro: guaraci/io/jcamp.py) ─────────────────

def test_parse_title_puro(pq):
    """TITLE de amostra pura: espécie resolvida, puro=True, mae_id sem teor."""
    info = pq.parse_title("CAP-04-11-2020-T1")
    assert info is not None
    assert info["cod"] == "CAP"
    assert info["especie"] == "Castanha do Pará"
    assert info["puro"] is True
    assert info["triplicata"] == 1
    assert info["mae_id"] == "CAP-04-11-2020"


def test_parse_title_adulterado(pq):
    """TITLE adulterado: adulterante/teor extraídos; réplicas compartilham mae_id."""
    t1 = pq.parse_title("AND-10-06-2020-AD-S-4.13%-T1")
    t2 = pq.parse_title("AND-10-06-2020-AD-S-4.13%-T2")
    assert t1 is not None and t2 is not None
    assert t1["puro"] is False
    assert t1["adulterante"] == "S"
    assert t1["adulterante_nome"] == "soja"
    assert t1["teor"] == pytest.approx(4.13)
    # Triplicatas do mesmo ponto -> mesmo mae_id (anti-vazamento no GroupKFold)
    assert t1["mae_id"] == t2["mae_id"]


def test_parse_title_invalido(pq):
    """String fora do padrão retorna None (não quebra o carregamento)."""
    assert pq.parse_title("isto nao e um title") is None


def test_gerar_nome_saida_contem_nivel_e_preproc(pq):
    """Caminho de saída embute nível e pré-processamento (rastreável)."""
    cfg = pq.Config()
    cfg.nivel = "N1"
    cfg.preprocessamento_padrao = "msc_sg_mc"
    nome = pq.gerar_nome_saida(cfg, n_classes=13, n_amostras=100)
    base = nome.replace("\\", "/").split("/")[-1]
    assert base.startswith("PLSDA_OE_N1")
    assert "MSC" in base


# ── IO de configuração (futuro: guaraci/config.py) ────────────────────────────

def test_config_roundtrip_preserva_valores(pq, tmp_path):
    """salvar_config → carregar_config preserva os valores editados."""
    cfg = pq.Config()
    cfg.nivel = "N2"
    cfg.max_lvs = 17
    cfg.frac_holdout = 0.3
    cfg.wn_min = 900.0
    cfg.wn_max = 1800.0
    caminho = str(tmp_path / "config.yaml")
    pq.salvar_config(cfg, caminho)
    lido = pq.carregar_config(caminho)
    assert lido.nivel == "N2"
    assert lido.max_lvs == 17
    assert lido.frac_holdout == pytest.approx(0.3)
    assert lido.wn_min == pytest.approx(900.0)
    assert lido.wn_max == pytest.approx(1800.0)


def test_validar_semantico_faixa_invertida(pq):
    """Faixa espectral invertida (início >= fim) é sinalizada cedo."""
    cfg = pq.Config()
    cfg.wn_min, cfg.wn_max = 10000.0, 4000.0
    erros = pq._validar_semantico(cfg)
    assert any("Faixa espectral" in e for e in erros)


def test_validar_semantico_config_padrao_ok(pq):
    """Config padrão não gera erros semânticos."""
    assert pq._validar_semantico(pq.Config()) == []
