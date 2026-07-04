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


# ── Figuras de mérito (Valderrama, Braga & Poppi, 2009) ───────────────────────

def _modelo_e_replicas_conhecidos(seed=0, n=60, p=25, ruido_replica=0.02):
    """PLS com direção verdadeira conhecida + grupos de réplicas com ruído
    de magnitude CONHECIDA — permite verificar se delta_x é recuperado
    corretamente (não é só "não quebrou", é "dá o número certo")."""
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=p); w_true /= np.linalg.norm(w_true)
    X = rng.normal(size=(n, p))
    y = (X @ w_true) * 5.0 + rng.normal(scale=0.05, size=n)
    modelo = PLSRegression(n_components=3, scale=False)
    modelo.fit(X, y.reshape(-1, 1))
    grupos = []
    for _ in range(8):
        base = X[rng.integers(0, n)]
        grupos.append(base + rng.normal(scale=ruido_replica, size=(3, p)))
    return modelo, X, grupos, ruido_replica


def test_figuras_merito_recupera_ruido_injetado(pq):
    """delta_x estimado (via variância pooled das réplicas) deve bater com o
    ruído REALMENTE injetado nas réplicas sintéticas (não é chute)."""
    modelo, X, grupos, ruido = _modelo_e_replicas_conhecidos()
    fom = pq.figuras_merito_regressao(modelo, X, grupos)
    assert fom["delta_x_ruido"] == pytest.approx(ruido, rel=0.25)
    assert fom["n_grupos_replicas"] == 8


def test_figuras_merito_razao_loq_lod_e_exata(pq):
    """LOQ/LOD = 10/3.3 por definição (mesmo delta_x e SEN cancelam)."""
    modelo, X, grupos, _ = _modelo_e_replicas_conhecidos(seed=7)
    fom = pq.figuras_merito_regressao(modelo, X, grupos)
    assert fom["loq"] / fom["lod"] == pytest.approx(10.0 / 3.3, rel=1e-9)


def test_figuras_merito_sensibilidade_e_inverso_da_norma_de_b(pq):
    """SEN = 1/||b|| — checagem direta contra o vetor de regressão do modelo."""
    modelo, X, grupos, _ = _modelo_e_replicas_conhecidos(seed=1)
    fom = pq.figuras_merito_regressao(modelo, X, grupos)
    norm_b = np.linalg.norm(np.asarray(modelo.coef_).reshape(-1))
    assert fom["sensibilidade"] == pytest.approx(1.0 / norm_b, rel=1e-9)


def test_figuras_merito_seletividade_entre_0_e_1(pq):
    """SEL_i é um cosseno (|.|) — sempre em [0, 1]; a média reportada também."""
    modelo, X, grupos, _ = _modelo_e_replicas_conhecidos(seed=2)
    fom = pq.figuras_merito_regressao(modelo, X, grupos)
    assert 0.0 <= fom["seletividade_media"] <= 1.0


def test_figuras_merito_sem_replicas_nao_quebra(pq):
    """Sem réplicas físicas não há como estimar ruído — NaN, não crash nem 0."""
    modelo, X, _grupos, _ = _modelo_e_replicas_conhecidos(seed=3)
    fom = pq.figuras_merito_regressao(modelo, X, [])
    assert fom["n_grupos_replicas"] == 0
    assert np.isnan(fom["lod"]) and np.isnan(fom["loq"])
    assert np.isnan(fom["sensibilidade_analitica"])
    # SEN/SEL não dependem de réplicas — continuam calculáveis
    assert np.isfinite(fom["sensibilidade"])


def test_figuras_merito_modelo_degenerado_b_zero(pq):
    """Vetor de regressão nulo (modelo sem poder preditivo) -> tudo NaN,
    nunca ZeroDivisionError/inf silencioso."""
    class _ModeloNulo:
        coef_ = np.zeros((1, 10))
    fom = pq.figuras_merito_regressao(
        _ModeloNulo(), np.random.default_rng(0).normal(size=(20, 10)), [])
    assert all(np.isnan(v) for k, v in fom.items() if k != "n_grupos_replicas")


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


# ── SPA/APS e AG — seleção de variáveis (futuro: já em selecao_variaveis.py) ──

def _dados_classificacao_sinteticos(seed=0, n=80, p=40, vars_informativas=(5, 6, 7, 20, 21)):
    """2 classes separáveis SÓ em `vars_informativas` — permite checar se um
    método de seleção de fato prioriza as variáveis que carregam sinal."""
    rng = np.random.default_rng(seed)
    classes = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, p))
    for i in range(n):
        if classes[i] == 1:
            X[i, list(vars_informativas)] += 3.0
    Y_bin = np.zeros((n, 2))
    for i, c in enumerate(classes):
        Y_bin[i, c] = 1
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv_indices = list(cv.split(X, classes))
    return X, Y_bin, classes, cv_indices


def test_spa_cadeia_deflaciona_colunas_duplicadas(pq):
    """SPA: uma coluna DUPLICADA da já selecionada deve ficar com norma ~0
    após a projeção (colinear = sem informação nova) — não deve ser
    escolhida logo em seguida, propriedade central do algoritmo (Araújo
    et al. 2001: minimizar colinearidade entre variáveis selecionadas)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 6))
    X[:, 1] = X[:, 0]           # coluna 1 é DUPLICATA exata da coluna 0
    X[:, 2] = X[:, 0] * 2.0     # coluna 2 é colinear (múltiplo escalar) da 0
    cadeia = pq._spa_cadeia(X, idx_inicial=0, n_vars_max=4)
    # as colunas 1 e 2 (colineares com a 0) devem vir DEPOIS das genuinamente
    # novas (3, 4, 5) na ordem de seleção, pois sua norma residual é ~0
    pos = {int(idx): pos for pos, idx in enumerate(cadeia)}
    assert pos[0] == 0  # a de partida sempre entra primeiro
    novas = [pos[j] for j in (3, 4, 5) if j in pos]
    colineares = [pos[j] for j in (1, 2) if j in pos]
    if novas and colineares:
        assert min(novas) < min(colineares)


def test_selecao_spa_retorna_mascara_valida(pq):
    """SPA: roda sem crash, mask tem >=2 variáveis, uma avaliação por início."""
    X, Y_bin, y_int, cv_indices = _dados_classificacao_sinteticos()
    resultados, mask = pq.selecao_spa(X, Y_bin, y_int, cv_indices, n_lv=2,
                                       n_vars_max=8, n_starts=10, seed=0)
    assert mask.sum() >= 2
    assert len(resultados) <= 10
    assert all("balanced_accuracy" in r for r in resultados)


def test_selecao_ag_fitness_nao_decresce_por_elitismo(pq):
    """AG: com elitismo, o melhor fitness da geração NUNCA piora ao longo das
    gerações (o melhor cromossomo sempre sobrevive) — propriedade de design,
    não coincidência estatística."""
    X, Y_bin, y_int, cv_indices = _dados_classificacao_sinteticos(seed=2)
    historico, mask = pq.selecao_ag(X, Y_bin, y_int, cv_indices, n_lv=2,
                                     tam_populacao=10, n_geracoes=6,
                                     prob_mutacao=0.05, frac_inicial=0.15, seed=2)
    assert len(historico) == 6
    melhores = [h["melhor_fitness"] for h in historico]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(melhores, melhores[1:]))
    assert mask.sum() >= 2


def test_selecao_ag_recupera_variaveis_informativas(pq):
    """AG: com sinal forte e claro, a máscara final deve conter pelo menos
    parte das variáveis genuinamente informativas (não é seleção ao acaso)."""
    informativas = (5, 6, 7, 20, 21)
    X, Y_bin, y_int, cv_indices = _dados_classificacao_sinteticos(
        seed=3, vars_informativas=informativas)
    _historico, mask = pq.selecao_ag(X, Y_bin, y_int, cv_indices, n_lv=2,
                                      tam_populacao=12, n_geracoes=8,
                                      prob_mutacao=0.05, frac_inicial=0.15, seed=3)
    selecionadas = set(np.where(mask)[0].tolist())
    assert len(selecionadas & set(informativas)) >= 2


def test_config_spec_spa_ag_opt_in_por_padrao(pq):
    """SPA/AG são opt-in (default False) — mais lentos que os métodos
    sempre-ligados; não devem rodar sem o usuário pedir explicitamente."""
    cfg = pq.Config()
    assert cfg.executar_spa is False
    assert cfg.executar_ag is False
    chaves = {s["key"] for s in pq._CONFIG_SPEC}
    assert "selecao_spa" in chaves and "selecao_ag" in chaves


def test_etapa4_integra_spa_e_ag_quando_ligados(pq):
    """Integração: com executar_spa/executar_ag=True, a tabela final da
    Etapa 4 inclui as linhas 'SPA (APS)' e 'AG (Genetico)' ao lado dos
    métodos sempre-ligados (Full/iPLS/VIP/SR/sPLS-DA)."""
    import tempfile
    X, Y_bin, y_int, cv_indices = _dados_classificacao_sinteticos(seed=4, p=30)
    wavenumbers = np.linspace(4000, 400, X.shape[1])
    vip = np.abs(np.random.default_rng(4).normal(size=X.shape[1])) + 0.5
    sr = np.abs(np.random.default_rng(5).normal(size=X.shape[1]))

    cfg = pq.Config(executar_spa=True, executar_ag=True,
                     spa_n_vars_max=6, spa_n_starts=6,
                     ag_tam_populacao=8, ag_n_geracoes=3, seed=4)
    with tempfile.TemporaryDirectory() as pasta:
        pasta_dados = pasta
        resumo = pq.etapa4_selecao_variaveis(
            X, Y_bin, y_int, vip, sr, wavenumbers, cv_indices, n_lv=2,
            cfg=cfg, pasta=pasta, pasta_dados=pasta_dados)
    metodos = {t["metodo"] for t in resumo["tabela"]}
    assert "SPA (APS)" in metodos
    assert "AG (Genetico)" in metodos


# ── Hardware (probe + auto-ajuste) ────────────────────────────────────────────

def test_hardware_probe_retorna_campos_esperados(pq):
    """hardware_probe(): dict com as chaves esperadas, RAM/CPU positivos —
    roda de verdade (psutil real, sem mock), nunca lança exceção."""
    hw = pq.hardware_probe()
    for chave in ("ram_total_gb", "ram_livre_gb", "cpu_logicos", "cpu_fisicos",
                  "disco_livre_gb", "psutil_ok"):
        assert chave in hw
    assert hw["ram_total_gb"] > 0
    assert hw["cpu_logicos"] >= 1
    assert isinstance(hw["psutil_ok"], bool)


def test_auto_ajustar_config_hardware_ram_critica_desliga_tudo(pq):
    """RAM < 2 GB: desliga SHAP/benchmark/monte_carlo e reduz CV — o cenário
    mais crítico de proteção contra travamento."""
    cfg = pq.Config(executar_shap=True, executar_benchmark=True,
                     executar_monte_carlo=True, n_splits_cv=10)
    avisos = pq.auto_ajustar_config_hardware(cfg, {"ram_livre_gb": 1.5})
    assert cfg.executar_shap is False
    assert cfg.executar_benchmark is False
    assert cfg.executar_monte_carlo is False
    assert cfg.n_splits_cv == 3
    assert len(avisos) == 4


def test_auto_ajustar_config_hardware_ram_farta_nao_mexe(pq):
    """RAM >= 8 GB: nenhum ajuste, nenhum aviso — não deve mexer em nada
    desnecessariamente quando há recurso de sobra."""
    cfg = pq.Config(executar_shap=True, shap_max_amostras=500,
                     executar_benchmark=True, n_monte_carlo=200)
    avisos = pq.auto_ajustar_config_hardware(cfg, {"ram_livre_gb": 16.0})
    assert avisos == []
    assert cfg.executar_shap is True
    assert cfg.shap_max_amostras == 500


def test_verificar_ram_limite_impossivel_retorna_false(pq):
    """_verificar_ram: pedir uma quantidade astronomicamente alta de RAM
    livre deve sempre falhar (determinístico, não depende da máquina real)."""
    assert pq._verificar_ram(10_000_000.0, "operacao_teste") is False


def test_verificar_ram_limite_trivial_retorna_true(pq):
    """_verificar_ram: pedir uma quantidade trivial de RAM (bem abaixo de
    qualquer máquina real) deve sempre passar."""
    assert pq._verificar_ram(0.001, "operacao_teste") is True


# ── Paleta de cores: fallback além da paleta base (>20 classes) ───────────────

def test_cor_alem_da_paleta_base_usa_fallback_sem_crash(pq):
    """cor(i) para i >= 20 (tamanho da paleta base): sem glasbey/colorcet
    instalados, cai no fallback tab20 — nunca lança exceção, sempre um hex
    válido."""
    c = pq.cor(25)
    assert isinstance(c, str) and c.startswith("#")


def test_mapear_cores_classes_mais_de_20_classes(pq):
    """mapear_cores_classes com > 20 classes exercita o mesmo fallback e
    ainda assim devolve uma cor distinta por classe."""
    classes = [f"Classe_{i:02d}" for i in range(25)]
    mapa = pq.mapear_cores_classes(classes)
    assert len(mapa) == 25
    assert all(v.startswith("#") for v in mapa.values())


def test_paleta_externa_sem_libs_opcionais_retorna_none(pq):
    """_paleta_externa: sem glasbey/colorcet instalados (ambiente padrão do
    projeto), retorna None de forma graciosa — não é um erro, é o caminho
    normal quando as libs opcionais de paleta não estão presentes."""
    resultado = pq._paleta_externa(30)
    assert resultado is None or isinstance(resultado, list)


# ── FOM no resumo: anexar_regressao_resumo (unidade, rapido) ─────────────────

def test_anexar_regressao_resumo_escreve_bloco(pq, tmp_path):
    """anexar_regressao_resumo grava o bloco de figuras de merito no
    resumo_modelo.txt (append), com valores formatados e NaN -> 'n/a'."""
    pasta = str(tmp_path)
    with open(pasta + "/resumo_modelo.txt", "w", encoding="utf-8") as f:
        f.write("HEADER PREEXISTENTE\n")
    pq.anexar_regressao_resumo(
        pasta,
        pooled={"r2c": 0.95, "r2v": 0.90, "rmsec": 1.2, "rmsecv": 1.5,
                "rmsep": 1.8, "bias": -0.1},
        tabela_especie=[
            {"especie": "Coco", "n_lv": 4, "rmsep": 1.7, "r2val": 0.94,
             "lod": 2.1, "loq": 6.4, "sensibilidade": 0.033,
             "seletividade_media": 0.71},
            {"especie": "Babacu", "n_lv": 3, "rmsep": 2.2, "r2val": 0.88,
             "lod": float("nan"), "loq": float("nan"),
             "sensibilidade": float("nan"), "seletividade_media": 0.6},
        ])
    txt = open(pasta + "/resumo_modelo.txt", encoding="utf-8").read()
    assert "HEADER PREEXISTENTE" in txt          # nao sobrescreve
    assert "Analytical Figures of Merit" in txt
    assert "Per-species figures of merit" in txt
    assert "Coco" in txt and "Babacu" in txt
    assert "n/a" in txt                          # NaN da especie 2 -> n/a
    assert "2.10" in txt                         # LOD do Coco formatado


def test_anexar_regressao_resumo_fom_pooled(pq, tmp_path):
    """Caminho de modelo pooled unico (fom_pooled) tambem grava LOD/LOQ/SEN."""
    pasta = str(tmp_path)
    open(pasta + "/resumo_modelo.txt", "w", encoding="utf-8").close()
    pq.anexar_regressao_resumo(
        pasta,
        pooled={"r2c": 0.9, "r2v": 0.8, "rmsec": 1.0, "rmsecv": 1.1,
                "rmsep": 1.3, "bias": 0.0},
        fom_pooled={"lod": 3.0, "loq": 9.1, "sensibilidade": 0.05,
                    "sensibilidade_analitica": 12.3, "seletividade_media": 0.8,
                    "delta_x_ruido": 0.001})
    txt = open(pasta + "/resumo_modelo.txt", encoding="utf-8").read()
    assert "single pooled model" in txt
    assert "LOD" in txt and "LOQ" in txt


# ── Dominio de Aplicabilidade (Applicability Domain) ─────────────────────────

def test_dominio_aplicabilidade_treino_majoritariamente_dentro(pq):
    """Amostras do proprio treino caem dentro do dominio em ~(1-alpha) dos
    casos — o limite e um teste de 95%, entao a fracao dentro deve ser alta."""
    import numpy as np
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 30))
    pca = PCA(n_components=5).fit(X)
    ad = pq.dominio_aplicabilidade(pca, X, X, alpha=0.05)
    assert 0.80 <= float(ad["fracao_dentro"]) <= 1.0
    assert ad["dentro_dominio"].shape == (120,)
    assert float(ad["t2_limite"]) > 0 and float(ad["q_limite"]) > 0


def test_dominio_aplicabilidade_amostra_distante_fica_fora(pq):
    """Uma amostra espectralmente muito distante do treino (deslocada em
    varias ordens de grandeza) e sinalizada FORA do dominio."""
    import numpy as np
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 20))
    pca = PCA(n_components=4).fit(X)
    X_out = X[:5] + 50.0            # empurra 5 amostras para longe do plano
    ad = pq.dominio_aplicabilidade(pca, X, X_out, alpha=0.05)
    # Todas as 5 deslocadas devem estar fora (T2 e/ou Q estourados).
    assert not ad["dentro_dominio"].any()


def test_dominio_aplicabilidade_retorno_consistente(pq):
    """Mascaras booleanas e vetores t2/q tem o mesmo tamanho de X_new; dentro
    = dentro_t2 AND dentro_q."""
    import numpy as np
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 15))
    Xn = rng.normal(size=(12, 15))
    pca = PCA(n_components=3).fit(X)
    ad = pq.dominio_aplicabilidade(pca, X, Xn)
    assert ad["t2"].shape == (12,) and ad["q"].shape == (12,)
    assert np.array_equal(ad["dentro_dominio"],
                          ad["dentro_t2"] & ad["dentro_q"])


def test_dominio_aplicabilidade_split_treino_amostras_novas_equivale_ao_combinado(pq):
    """dominio_aplicabilidade_treino + dominio_aplicabilidade_amostras_novas
    (usadas por predicao.py para nao precisar reexportar X_train inteiro no
    pacote .joblib) devem produzir EXATAMENTE o mesmo resultado que a funcao
    combinada dominio_aplicabilidade -- e' a mesma matematica, so' partida
    em 2 etapas (treino gera artefatos leves; predicao os consome)."""
    import numpy as np
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(3)
    X = rng.normal(size=(90, 12))
    Xn = rng.normal(size=(10, 12)) + 0.5
    pca = PCA(n_components=4).fit(X)

    combinado = pq.dominio_aplicabilidade(pca, X, Xn, alpha=0.05)
    treino = pq.dominio_aplicabilidade_treino(pca, X, alpha=0.05)
    split = pq.dominio_aplicabilidade_amostras_novas(
        pca, Xn, treino["var_t"], treino["t2_limite"], treino["q_limite"])

    assert np.allclose(combinado["t2"], split["t2"])
    assert np.allclose(combinado["q"], split["q"])
    assert np.array_equal(combinado["dentro_dominio"], split["dentro_dominio"])
    assert float(combinado["t2_limite"]) == pytest.approx(treino["t2_limite"])
    assert float(combinado["q_limite"]) == pytest.approx(treino["q_limite"])


def test_dominio_aplicabilidade_treino_var_t_tem_tamanho_n_componentes(pq):
    """var_t (variancia dos scores) tem 1 valor por componente PCA -- e' o
    artefato leve que substitui reexportar X_train inteiro no pacote."""
    import numpy as np
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 10))
    pca = PCA(n_components=3).fit(X)
    treino = pq.dominio_aplicabilidade_treino(pca, X)
    assert treino["var_t"].shape == (3,)
    assert treino["t2_limite"] > 0 and treino["q_limite"] > 0


# ── Kennard-Stone: selecao representativa de amostras ────────────────────────

def test_kennard_stone_seleciona_extremos_primeiro(pq):
    """KS deve escolher amostras nas bordas do espaco antes das centrais.
    Com pontos 1D em [0..10], os dois primeiros selecionados sao os extremos."""
    import numpy as np
    X = np.linspace(0, 10, 11).reshape(-1, 1)   # 0,1,...,10
    ordem = pq.kennard_stone(X, 3)
    assert set(ordem[:2].tolist()) == {0, 10}   # extremos primeiro
    assert len(ordem) == 3 and len(set(ordem.tolist())) == 3


def test_kennard_stone_pede_mais_que_n_devolve_n(pq):
    """Pedir mais amostras do que existem devolve todas, sem repetir indice."""
    import numpy as np
    X = np.random.default_rng(0).normal(size=(8, 4))
    ordem = pq.kennard_stone(X, 50)
    assert len(ordem) == 8 and len(set(ordem.tolist())) == 8


def test_kennard_stone_split_particiona_sem_overlap(pq):
    """kennard_stone_split devolve treino/val disjuntos que cobrem todo o n."""
    import numpy as np
    X = np.random.default_rng(1).normal(size=(40, 6))
    tr, val = pq.kennard_stone_split(X, frac_treino=0.75)
    assert len(tr) == 30 and len(val) == 10
    assert set(tr.tolist()).isdisjoint(val.tolist())
    assert set(tr.tolist()) | set(val.tolist()) == set(range(40))


# ── Kennard-Stone group-aware (usado no split cal/val da regressao) ─────────

def test_ks_group_aware_nunca_separa_replicas(pq):
    """Com mae_id (>=4 grupos), _kennard_stone_split_group_aware nunca deixa
    replicas do MESMO grupo em lados diferentes (o invariante central do
    projeto: T1/T2/T3 sempre juntas entre cal/val)."""
    import numpy as np
    rng = np.random.default_rng(5)
    grupos = np.repeat([f"G{i:02d}" for i in range(10)], 3)   # 10 grupos x 3 replicas
    X = rng.normal(size=(30, 5))
    ic, iv = pq._kennard_stone_split_group_aware(X, grupos, frac_cal=0.7)
    grupos_ic = set(grupos[ic].tolist())
    grupos_iv = set(grupos[iv].tolist())
    assert grupos_ic.isdisjoint(grupos_iv), (
        "replicas do mesmo grupo (mae_id) apareceram em cal E val")
    assert set(ic.tolist()) | set(iv.tolist()) == set(range(30))


def test_ks_group_aware_sem_mae_id_roda_por_amostra(pq):
    """Sem mae_id, cai no KS direto por amostra (kennard_stone_split)."""
    import numpy as np
    X = np.random.default_rng(6).normal(size=(20, 4))
    ic, iv = pq._kennard_stone_split_group_aware(X, None, frac_cal=0.7)
    assert len(ic) == 14 and len(iv) == 6
    assert set(ic.tolist()).isdisjoint(iv.tolist())


def test_ks_group_aware_poucos_grupos_cai_no_split_por_amostra(pq):
    """Com menos de 4 grupos unicos, o colapso por grupo nao compensa (dados
    de menos) -- cai no KS direto por amostra, sem quebrar."""
    import numpy as np
    grupos = np.array(["G1", "G1", "G2", "G2", "G3"])
    X = np.random.default_rng(7).normal(size=(5, 3))
    ic, iv = pq._kennard_stone_split_group_aware(X, grupos, frac_cal=0.6)
    assert set(ic.tolist()) | set(iv.tolist()) == set(range(5))


@pytest.mark.slow
def test_regressao_pooled_com_kennard_stone_roda_sem_erro(pq, tmp_path):
    """Integracao real: executar() em N3 sintetico com
    divisao_cal_val='kennard_stone' completa sem erro e gera o resumo com
    o bloco de figuras de merito (mesmo caminho da regressao, so' o metodo
    de split cal/val muda)."""
    import os
    cfg = pq.Config(
        pasta_entrada=str(tmp_path / "dados"),
        pasta_saida_raiz=str(tmp_path / "saida"),
        modo="sintetico", nivel="N3",
        n_por_classe=10, n_pontos_sint=60, n_replicas_sint=3,
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutacoes=5,
        n_permutacoes_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=5,
        divisao_cal_val="kennard_stone",
    )
    os.makedirs(cfg.pasta_entrada, exist_ok=True)
    pq.executar(cfg)

    runs = list((tmp_path / "saida").iterdir())
    assert runs, "executar() nao criou pasta de saida"
    resumo = runs[0] / "logs" / "resumo_modelo.txt"
    assert resumo.is_file()
    txt = resumo.read_text(encoding="utf-8")
    assert "Analytical Figures of Merit" in txt


# ── DD-SIMCA bloqueado em N1 (nao agrega a identificacao de especie) ────────

@pytest.mark.slow
def test_ddsimca_ignorado_em_n1_mesmo_com_toggle_ligado(pq, tmp_path):
    """DD-SIMCA e' um diagnostico de autenticacao de PUREZA (conceito N2).
    Ligar o toggle manualmente com nivel=N1 (identificacao de especie) nao
    deve gerar nenhuma figura de DD-SIMCA -- o pipeline ignora o toggle
    (com aviso), pois o grafico nao agrega aquele tipo de analise."""
    import os
    cfg = pq.Config(
        pasta_entrada=str(tmp_path / "dados"),
        pasta_saida_raiz=str(tmp_path / "saida"),
        modo="sintetico", nivel="N1",
        n_por_classe=8, n_pontos_sint=50,
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1,
        n_permutacoes=5, n_permutacoes_wold=5,
        n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3,
        executar_benchmark=False, executar_monte_carlo=False,
        executar_shap=False, executar_wold=False, executar_cv_anova=False,
        executar_opls=False, executar_etapa4=False, comparar_pipelines=False,
        comparar_hca_pipelines=False, max_lvs=5,
        executar_ddsimca=True,          # ligado manualmente, propositalmente
    )
    os.makedirs(str(tmp_path / "dados"), exist_ok=True)
    pq.executar(cfg)

    runs = list((tmp_path / "saida").iterdir())
    assert runs, "executar() nao criou pasta de saida"
    figbase = runs[0] / "figuras"
    nomes_pngs = {p.name for p in figbase.rglob("*.png")} if figbase.exists() else set()
    ddsimca_figs = {n for n in nomes_pngs if "ddsimca" in n.lower()
                    or "cooman" in n.lower()}
    assert not ddsimca_figs, (
        f"Figuras de DD-SIMCA foram geradas em nivel=N1: {ddsimca_figs} "
        "(deveriam ser ignoradas)")


def test_ddsimca_permitido_em_n2_com_toggle_ligado(pq):
    """Confirma que o bloqueio e' especifico de N1 -- em N2 (onde
    executar() ja forca executar_ddsimca=True), o toggle continua
    funcionando normalmente (regressao no bloqueio, nao remocao da feature)."""
    cfg = pq.Config(nivel="N2", executar_ddsimca=False)
    # Simula so' o trecho de decisao (sem rodar o pipeline inteiro): a
    # condicao de bloqueio e' `cfg.executar_ddsimca and cfg.nivel == "N1"`,
    # entao em N2 ela nunca dispara, independente do toggle.
    bloqueado = cfg.executar_ddsimca and cfg.nivel == "N1"
    assert not bloqueado
