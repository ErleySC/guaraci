"""Rede de segurança do CONTRATO DE FACHADA do pipeline.

Depois da Fase H, pipeline.py virou uma fachada: grande parte da lógica vive
em módulos dedicados (chemometric_stats, paleta_cores, dados_io,
preprocessamento, classificadores, figuras, validacao_estatistica, hardware,
selecao_variaveis, avaliacao_modelos) e é REEXPORTADA em pipeline.py para que
guaraci.py / cli_assistente.py / app_quimiometria.py — que só conhecem
`import pipeline as pq` — continuem usando `pq.X` sem alteração.

Este teste garante que cada símbolo do contrato público:
  (a) está acessível em pipeline (`pq.X` existe), e
  (b) é EXATAMENTE o objeto do módulo de origem (identidade `is`).

Se alguém remover ou duplicar um reexport no futuro, isto falha na hora —
antes de o app/CLI quebrarem em produção. Também serve de MAPA: onde cada
função realmente vive agora.
"""
import importlib

import pytest

# módulo de origem -> símbolos que pipeline.py deve reexportar (contrato público)
CONTRATO = {
    "chemometric_stats": [
        "vip_scores", "calcular_selectivity_ratio", "hotelling_t2",
        "hotelling_t2_limite", "q_residuos", "q_residuos_limite",
        "variancia_explicada", "figuras_merito_regressao",
    ],
    "paleta_cores": [
        "PALETA", "MARCADORES", "cor", "mapear_cores_classes",
        "mapear_marcadores_classes", "edge_para_cor",
    ],
    "dados_io": [
        "parse_title", "extrair_title_do_dx", "carregar_dados", "carregar_dx",
        "carregar_csv", "gerar_dados_sinteticos", "parse_dx", "parse_spectrum",
        "CODIGO_ESPECIE", "ADULTERANTE_NOME",
    ],
    "dados_imagem": [
        "carregar_imagens", "carregar_imagem_arquivo", "recortar_relativo",
        "extrair_features_cor", "extrair_features_textura",
    ],
    "preprocessamento": [
        "SNV", "SavGol", "MSC", "construir_preprocessador",
    ],
    "classificadores": [
        "DDSimca", "OPLSDAWrapper",
    ],
    "figuras": [
        "salvar", "setup_matplotlib", "especificidade_por_classe",
        "fig1_pca_scores", "fig4_confusao",
    ],
    "validacao_estatistica": [
        "_cv_predict_manual", "bootstrap_bca_ci", "cv_anova_eriksson",
        "teste_wold", "teste_permutacao",
    ],
    "hardware": [
        "hardware_probe", "auto_ajustar_config_hardware", "_verificar_ram",
    ],
    "selecao_variaveis": [
        "selecao_ipls", "sparse_plsda_mask", "etapa4_selecao_variaveis",
        "fig_etapa4_ipls", "fig_etapa4_comparacao",
        "selecao_spa", "selecao_ag", "fig_etapa4_ag_convergencia",
    ],
    "avaliacao_modelos": [
        "PLSDAClassifier", "benchmark_classificadores", "monte_carlo_cv",
        "fig_det_curvas", "fig_shap_benchmark",
    ],
}

_CASOS = [(mod, sym) for mod, syms in CONTRATO.items() for sym in syms]


@pytest.mark.parametrize("modulo,simbolo", _CASOS,
                         ids=[f"{m}.{s}" for m, s in _CASOS])
def test_reexport_identidade(pq, modulo, simbolo):
    """pq.<simbolo> existe e É o mesmo objeto do módulo de origem."""
    origem = importlib.import_module(modulo)
    assert hasattr(pq, simbolo), (
        f"pipeline não reexporta '{simbolo}' (esperado de {modulo}.py) — "
        f"o contrato de fachada foi quebrado; guaraci/cli/app dependem de pq.{simbolo}")
    assert getattr(pq, simbolo) is getattr(origem, simbolo), (
        f"pq.{simbolo} não é {modulo}.{simbolo} — reexport divergente/duplicado")


def test_config_e_spec_no_pipeline(pq):
    """Config e _CONFIG_SPEC continuam no pipeline (núcleo, não extraídos)."""
    assert hasattr(pq, "Config")
    assert hasattr(pq, "_CONFIG_SPEC")
    assert hasattr(pq, "executar")
