"""Testes de guaraci.model_registry (item 20: registry de modelos do
Auto-Benchmark / Monte Carlo CV).

Regressão-chave: antes da extração, benchmark_classificadores() e
monte_carlo_cv() tinham a MESMA lista de classificadores hardcoded duas
vezes, e haviam divergido — o Gradient Boosting do Monte Carlo CV não tinha
subsample=0.8 (o do benchmark tinha), apesar da docstring afirmar hiper-
parâmetros idênticos. O teste abaixo trava essa divergência.
"""
from guaraci.model_registry import construir_lista_benchmark, nomes_modelos_benchmark


def test_pls_da_sempre_presente_mesmo_sem_opcionais(pq):
    cfg = pq.Config()
    lista = construir_lista_benchmark(n_opt=5, cfg=cfg, incluir_opcionais=False)
    nomes = [n for n, _ in lista]
    assert nomes == ["PLS-DA"]


def test_lista_completa_inclui_todos_os_modelos_core(pq):
    cfg = pq.Config()
    lista = construir_lista_benchmark(n_opt=5, cfg=cfg, incluir_opcionais=True)
    nomes = [n for n, _ in lista]
    # SVM RBF/Random Forest/Grad. Boost. nao tem dependencia opcional --
    # sempre presentes; XGBoost so' se o pacote estiver instalado.
    assert "PLS-DA" in nomes
    assert "SVM RBF" in nomes
    assert "Random Forest" in nomes
    assert "Grad. Boost." in nomes


def test_gradient_boosting_usa_subsample_08_regressao(pq):
    """Regressao: GB do Monte Carlo CV nao tinha subsample=0.8 (drift vs
    benchmark). Agora vem de UMA fonte so', entao os dois caminhos coincidem."""
    cfg = pq.Config()
    lista = construir_lista_benchmark(n_opt=5, cfg=cfg, incluir_opcionais=True)
    gb = dict(lista)["Grad. Boost."]
    assert gb.subsample == 0.8


def test_construtores_respeitam_seed_e_n_opt(pq):
    cfg = pq.Config()
    cfg.seed = 123
    lista = construir_lista_benchmark(n_opt=7, cfg=cfg, incluir_opcionais=True)
    d = dict(lista)
    assert d["PLS-DA"].n_components == 7
    assert d["Random Forest"].random_state == 123
    assert d["SVM RBF"].random_state == 123


def test_nomes_modelos_benchmark_sem_instanciar():
    assert nomes_modelos_benchmark(incluir_opcionais=False) == ("PLS-DA",)
    completo = nomes_modelos_benchmark(incluir_opcionais=True)
    assert "PLS-DA" in completo and "SVM RBF" in completo


def test_modelo_opcional_ausente_e_pulado_silenciosamente(monkeypatch, pq):
    """Se um pacote opcional (ex.: xgboost) nao estiver instalado, o
    registry pula esse modelo sem lancar excecao (comportamento historico)."""
    import guaraci.model_registry as mr

    def _falha_import(n_opt, cfg):
        raise ImportError("pacote nao instalado (simulado)")

    monkeypatch.setattr(mr, "_construir_xgboost", _falha_import)
    # Reconstroi o registro apontando para o construtor monkeypatched.
    registro_original = mr._REGISTRO
    mr._REGISTRO = [
        (nome, (_falha_import if nome == "XGBoost" else construtor), obrig)
        for nome, construtor, obrig in registro_original
    ]
    try:
        cfg = pq.Config()
        lista = construir_lista_benchmark(n_opt=5, cfg=cfg, incluir_opcionais=True)
        assert "XGBoost" not in dict(lista)
    finally:
        mr._REGISTRO = registro_original
