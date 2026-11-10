"""technique_registry.py — Catalogo dos metodos cientificos do GUARACI
(Agente 6, item "d": listar/explicar tecnicas disponiveis pelo assistente).

Fonte UNICA da lista que o assistente `G` mostra quando o usuario pergunta
"o que o GUARACI sabe fazer?"/"quais metodos existem para X?". Mesmo padrao
ja usado por `model_registry.py`/`io_registry.py`: uma entrada por metodo,
num so lugar, em vez de uma lista escrita a mao espalhada pela UI (o
problema que `_guaraci_navegar_secoes` tinha antes desta rodada -- ver
docs/projeto/identidade_visual.md, secao do Agente 6).

"Nunca fica desatualizado" nao vem de introspeccao de AST (fragil para
codigo cientifico com muitos helpers internos) e sim de
`tests/test_technique_registry.py`: para os modulos de proposito UNICO
(cada simbolo publico e' de fato uma tecnica, nao um helper interno), o
teste falha se `__all__` tiver algo sem entrada aqui. Adicionar um metodo
novo a um desses modulos sem atualizar este arquivo quebra a suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from guaraci.catalogo_tecnicas import TECNICAS as _TEC

__all__ = [
    "TechniqueEntry",
    "REGISTRY",
    "MODULOS_COBERTURA_TOTAL",
]


@dataclass(frozen=True)
class TechniqueEntry:
    """Uma entrada do catalogo. `referencia` e' `"modulo.simbolo"` (string,
    nunca o objeto importado -- mantem este arquivo sem custo de import de
    numpy/scipy/sklearn so' para listar nomes)."""
    id: str
    categoria: str
    nome: str
    referencia: str
    quando_usar: str
    limitacao: str


REGISTRY: List[TechniqueEntry] = [
    # ---- Classificacao / deteccao (autenticacao pura x adulterada) -------
    TechniqueEntry(
        id="ddsimca",
        categoria="classificacao_deteccao",
        nome="DD-SIMCA",
        referencia="guaraci.classificadores.DDSimca",
        quando_usar=_TEC["ddsimca"]["use"][0],
        limitacao=_TEC["ddsimca"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="conformal_one_class",
        categoria="classificacao_deteccao",
        nome="Predicao conformal (one-class)",
        referencia="guaraci.conformal.ConformalOneClass",
        quando_usar=_TEC["conformal_one_class"]["use"][0],
        limitacao=_TEC["conformal_one_class"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="pls_da",
        categoria="classificacao_deteccao",
        nome="PLS-DA",
        referencia="guaraci.avaliacao_modelos.PLSDAClassifier",
        quando_usar=_TEC["pls_da"]["use"][0],
        limitacao=_TEC["pls_da"]["nao_use"][0],
    ),
    # ---- Quantificacao -----------------------------------------------------
    TechniqueEntry(
        id="pls_r_pooled",
        categoria="quantificacao",
        nome="PLS-R (regressao pooled)",
        referencia="guaraci.pipeline.pls_regressao_pooled",
        quando_usar=_TEC["pls_r_pooled"]["use"][0],
        limitacao=_TEC["pls_r_pooled"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="pls_r_por_especie",
        categoria="quantificacao",
        nome="PLS-R por especie",
        referencia="guaraci.avaliacao_modelos.benchmark_regression_by_species",
        quando_usar=_TEC["pls_r_por_especie"]["use"][0],
        limitacao=_TEC["pls_r_por_especie"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="conformal_regressao",
        categoria="quantificacao",
        nome="Predicao conformal (regressao)",
        referencia="guaraci.conformal.conformal_margin_regression",
        quando_usar=_TEC["conformal_regressao"]["use"][0],
        limitacao=_TEC["conformal_regressao"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="conformal_classificacao",
        categoria="classificacao_deteccao",
        nome="Conjunto de predicao conformal (classificacao)",
        referencia="guaraci.conformal.conformal_margin_classification",
        quando_usar=_TEC["conformal_classificacao"]["use"][0],
        limitacao=_TEC["conformal_classificacao"]["nao_use"][0],
    ),
    # ---- Identificacao de conjunto aberto (Detectar->Identificar->Quantificar) --
    TechniqueEntry(
        id="identificacao_conjunto_aberto",
        categoria="identificacao_conjunto_aberto",
        nome="Identificacao de adulterante (conjunto aberto)",
        referencia="guaraci.identificacao.identify_sample",
        quando_usar=_TEC["identificacao_conjunto_aberto"]["use"][0],
        limitacao=_TEC["identificacao_conjunto_aberto"]["nao_use"][0],
    ),
    # ---- Selecao de amostras (Bloco K) -------------------------------------
    TechniqueEntry(
        id="kennard_stone",
        categoria="selecao_amostras",
        nome="Kennard-Stone",
        referencia="guaraci.dados_io.kennard_stone_split",
        quando_usar=_TEC["kennard_stone"]["use"][0],
        limitacao=_TEC["kennard_stone"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="duplex",
        categoria="selecao_amostras",
        nome="Duplex",
        referencia="guaraci.dados_io.duplex_split",
        quando_usar=_TEC["duplex"]["use"][0],
        limitacao=_TEC["duplex"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="spxy",
        categoria="selecao_amostras",
        nome="SPXY",
        referencia="guaraci.dados_io.spxy_split",
        quando_usar=_TEC["spxy"]["use"][0],
        limitacao=_TEC["spxy"]["nao_use"][0],
    ),
    # ---- Transferencia de calibracao (Bloco 12) ----------------------------
    TechniqueEntry(
        id="direct_standardization",
        categoria="transferencia_calibracao",
        nome="Direct Standardization (DS)",
        referencia="guaraci.transferencia_calibracao.direct_standardization",
        quando_usar=_TEC["direct_standardization"]["use"][0],
        limitacao=_TEC["direct_standardization"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="piecewise_direct_standardization",
        categoria="transferencia_calibracao",
        nome="Piecewise Direct Standardization (PDS)",
        referencia="guaraci.transferencia_calibracao.piecewise_direct_standardization",
        quando_usar=_TEC["piecewise_direct_standardization"]["use"][0],
        limitacao=_TEC["piecewise_direct_standardization"]["nao_use"][0],
    ),
    # ---- Figuras de merito --------------------------------------------------
    TechniqueEntry(
        id="figuras_de_merito_regressao",
        categoria="figuras_de_merito",
        nome="LOD/LOQ/sensibilidade (figuras de merito)",
        referencia="guaraci.chemometric_stats.regression_figures_of_merit",
        quando_usar=_TEC["figuras_de_merito_regressao"]["use"][0],
        limitacao=_TEC["figuras_de_merito_regressao"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="rpd_rer",
        categoria="figuras_de_merito",
        nome="RPD/RER",
        referencia="guaraci.chemometric_stats.rpd_rer",
        quando_usar=_TEC["rpd_rer"]["use"][0],
        limitacao=_TEC["rpd_rer"]["nao_use"][0],
    ),
    # ---- Robustez / linearidade ---------------------------------------------
    TechniqueEntry(
        id="linearidade",
        categoria="robustez_linearidade",
        nome="Teste de falta de ajuste (linearidade)",
        referencia="guaraci.linearity.lack_of_fit_test",
        quando_usar=_TEC["linearidade"]["use"][0],
        limitacao=_TEC["linearidade"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="robustez",
        categoria="robustez_linearidade",
        nome="Protocolo de robustez",
        referencia="guaraci.robustness.run_robustness_protocol",
        quando_usar=_TEC["robustez"]["use"][0],
        limitacao=_TEC["robustez"]["nao_use"][0],
    ),
    # ---- Perfis (ligacao direta com o Agente 5B) ----------------------------
    TechniqueEntry(
        id="perfil_matriz",
        categoria="perfis",
        nome="Perfil de tecnica",
        referencia="guaraci.perfil_matriz.load_profile",
        quando_usar=_TEC["perfil_matriz"]["use"][0],
        limitacao=_TEC["perfil_matriz"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="perfil_tecnica_aquisicao",
        categoria="perfis",
        nome="Perfil de aquisicao de imagem (modo imagem)",
        referencia="guaraci.perfil_matriz.load_profile",
        quando_usar=_TEC["perfil_tecnica_aquisicao"]["use"][0],
        limitacao=_TEC["perfil_tecnica_aquisicao"]["nao_use"][0],
    ),
    # ---- Resolucao de mistura (Bloco 14, 2026-09-04) -----------------------
    TechniqueEntry(
        id="mcr_als",
        categoria="resolucao_mistura",
        nome="MCR-ALS",
        referencia="guaraci.mcr_als.mcr_als",
        quando_usar=_TEC["mcr_als"]["use"][0],
        limitacao=_TEC["mcr_als"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="anova_tukey_unidade",
        categoria="comparacao_grupos",
        nome="ANOVA + Tukey/Games-Howell sobre a unidade independente",
        referencia="guaraci.anova_grupos.anova_uma_via",
        quando_usar=_TEC["anova_tukey_unidade"]["use"][0],
        limitacao=_TEC["anova_tukey_unidade"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="kruskal_dunn_unidade",
        categoria="comparacao_grupos",
        nome="Kruskal-Wallis + Dunn sobre a unidade independente",
        referencia="guaraci.anova_grupos.kruskal_dunn",
        quando_usar=_TEC["kruskal_dunn_unidade"]["use"][0],
        limitacao=_TEC["kruskal_dunn_unidade"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="diagnostico_premissas",
        categoria="comparacao_grupos",
        nome="Diagnostico de premissas do ANOVA e recomendacao de metodo",
        referencia="guaraci.anova_grupos.diagnosticar_premissas",
        quando_usar=_TEC["diagnostico_premissas"]["use"][0],
        limitacao=_TEC["diagnostico_premissas"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="vip_estabilidade",
        categoria="importancia_variaveis",
        nome="VIP do PLS com estabilidade nas dobras agrupadas",
        referencia="guaraci.vip_estabilidade.vip_estabilidade",
        quando_usar=_TEC["vip_estabilidade"]["use"][0],
        limitacao=_TEC["vip_estabilidade"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="perfil_multielementar",
        categoria="perfis",
        nome="Perfil multielementar (ICP-OES / ICP-MS)",
        referencia="guaraci.icp_multielementar.ler_icp_csv",
        quando_usar=_TEC["perfil_multielementar"]["use"][0],
        limitacao=_TEC["perfil_multielementar"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="parafac_eem",
        categoria="resolucao_mistura",
        nome="PARAFAC de fluorescencia EEM",
        referencia="guaraci.tecnicas_avancadas.rodar_parafac_eem",
        quando_usar=_TEC["parafac_eem"]["use"][0],
        limitacao=_TEC["parafac_eem"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="cow_alinhamento",
        categoria="perfis",
        nome="Alinhamento de cromatogramas (COW)",
        referencia="guaraci.tecnicas_avancadas.rodar_cow",
        quando_usar=_TEC["cow_alinhamento"]["use"][0],
        limitacao=_TEC["cow_alinhamento"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="monitoramento_mspc",
        categoria="monitoramento",
        nome="Monitoramento de processo por lote (T2/Q + teste de deriva)",
        referencia="guaraci.monitorar_cli.executar",
        quando_usar=_TEC["monitoramento_mspc"]["use"][0],
        limitacao=_TEC["monitoramento_mspc"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="portao_correcao_sinal",
        categoria="portao_aceite",
        nome="Portao de aceite de correcao de sinal (EMSC/OSC)",
        referencia="guaraci.tecnicas_avancadas.rodar_portao_correcao",
        quando_usar=_TEC["portao_correcao_sinal"]["use"][0],
        limitacao=_TEC["portao_correcao_sinal"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="amostragem_ativa",
        categoria="selecao_amostras",
        nome="Priorizacao por amostragem ativa",
        referencia="guaraci.amostragem_ativa.priorizar_amostragem",
        quando_usar=_TEC["amostragem_ativa"]["use"][0],
        limitacao=_TEC["amostragem_ativa"]["nao_use"][0],
    ),
    TechniqueEntry(
        id="pooled_vs_local",
        categoria="quantificacao",
        nome="Modelo unico x modelo por especie",
        referencia="guaraci.tecnicas_avancadas.rodar_pooled_local",
        quando_usar=_TEC["pooled_vs_local"]["use"][0],
        limitacao=_TEC["pooled_vs_local"]["nao_use"][0],
    ),
]

#: Modulos de proposito UNICO (todo simbolo em __all__ e' de fato uma
#: tecnica) -- tests/test_technique_registry.py exige cobertura completa
#: so' destes, menos o `excecoes` (dataclass/enum/excecao, nao "metodo").
MODULOS_COBERTURA_TOTAL = {
    "guaraci.classificadores": {
        "incluir": {"DDSimca"},
        "excecoes": {"OPLSDAWrapper", "ddsimca_logo_sensitivity",
                     "ddsimca_pcv_sensitivity", "PUREZA_MIN_GRUPOS_LOGO",
                     "ddsimca_alpha_sustentado",
                     # fase 2: PLS-LDA e' variante do classificador PLS-DA (escolhida por `classificador:
                     # plslda`, descrita em catalogo_opcoes), como o OPLSDAWrapper; a deteccao de rampa e'
                     # um diagnostico do pipeline (aviso), nao um metodo a escolher; os LIMIAR_* sao constantes.
                     "PLSLDAClassifier", "detectar_classes_em_rampa", "LIMIAR_RAMPA_COLINEAR",
                     "LIMIAR_RAMPA_QUEDA_SENS", "LIMIAR_RAMPA_SENS_PONTAS"},
    },
    "guaraci.conformal": {
        "incluir": {"ConformalOneClass", "conformal_margin_regression",
                    "conformal_margin_classification"},
        "excecoes": {"achievable_alpha", "n_minimum_for_alpha",
                     "conformal_threshold", "conformal_prediction_set",
                     "conformal_threshold_agrupado"},
    },
    "guaraci.transferencia_calibracao": {
        "incluir": {"direct_standardization", "piecewise_direct_standardization"},
        "excecoes": {"StandardizationTransform", "apply_standardization"},
    },
    "guaraci.linearity": {
        "incluir": {"lack_of_fit_test"},
        "excecoes": {"LackOfFitResult"},
    },
    "guaraci.robustness": {
        "incluir": {"run_robustness_protocol"},
        "excecoes": {"RobustnessResult", "gaussian_noise_variants",
                     "baseline_drift_variants", "preprocessing_config_variants",
                     "avaliar_rmsep_plsr", "avaliar_bal_acc_plsda", "protocolo_padrao"},
    },
}
