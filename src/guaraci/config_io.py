"""config_io.py — Serialização, validação e schema (_CONFIG_SPEC) da Config.

Extraído de pipeline.py (dívida técnica pós-auditoria: pipeline.py era fachada
+ orquestrador + IO de config num arquivo só). Este módulo concentra a
"fonte única de verdade" da configuração — o _CONFIG_SPEC (mapeia campo
amigável <-> atributo da Config, alimenta YAML/menu/app web) e as funções de
ler/gravar/validar/coagir. Depende só de config.Config; nenhuma dependência
do motor (executar) — por isso não há import circular.

pipeline.py reexporta todos estes nomes, então `pipeline._CONFIG_SPEC`,
`pipeline.load_config(...)`, `pq._coagir_valor(...)` etc. seguem
funcionando sem alteração (o menu de terminal legado em pipeline.py e o app
web consomem daqui via a fachada).
"""
from __future__ import annotations

import glob
import os
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple

from guaraci.catalogo_tecnicas import INSTRUMENTOS as _INSTRUMENTOS_CATALOGO
from guaraci.config import Config
from guaraci.perfil_matriz import perfis_disponiveis

__all__ = [
    "save_config",
    "load_config",
    "CONFIG_SPEC",
    "attr_para_yaml",
    "coagir_valor",
    "validar_semantico",
    "validar_fonte_de_dados",
]

_PRE_PROC_FRIENDLY: Dict[str, str] = {
    "MSC+SG+MC":      "msc_sg_mc",
    "SNV+SG+MC":      "snv_sg_mc",
    "Autoscaling":    "autoscaling",
    "Mean-centering": "mc",
    "SNV+MC":         "snv_mc",
    "AirPLS+SG+MC":   "airpls_sg_mc",
    "Personalizado":  "custom",
    # no fim, em ordem de chegada: nao muda o numero das opcoes que ja existiam no seletor da tela
    "AirPLS+SNV+SG+MC": "airpls_snv_sg_mc",
    "PQN+MC":         "pqn_mc",
}
_PRE_PROC_INV: Dict[str, str] = {v: k for k, v in _PRE_PROC_FRIENDLY.items()}

# Cada campo: chave_yaml, atributo da Config, tipo, descricao, opcoes.
#   tipo in {"str","int","float","bool","list","choice","preproc"}
_CONFIG_SPEC: List[Dict[str, Any]] = [
    {"key": "modo_entrada", "attr": "mode", "tipo": "choice",
     "opcoes": ["dx", "csv", "imagem", "hsi", "opus", "spc", "sp", "rmn",
                "hplc", "gcms", "eem", "multielementar", "sintetico"]},
    {"key": "perfil_matriz", "attr": "matrix_profile", "tipo": "choice",
     "opcoes": perfis_disponiveis(apenas="matriz")},
    # Dimensao INDEPENDENTE de perfil_matriz (Agente 5B, 2026-09-01): so'
    # relevante quando modo_entrada="imagem". "" = nao declarado (aceito --
    # mesma convencao de campo string vazio de perfil_matriz/generico, nao
    # usa str_opcional/None porque a UI de escolha exige tipo="choice" nos
    # dois lados, CLI e Streamlit).
    # R6: o que a tela [2.4] grava sobre o eixo dos dados (so' rotulo/orientacao;
    # "" = nao declarado -> vale o perfil, ou o comportamento antigo).
    {"key": "eixo_unidade", "attr": "axis_unit", "tipo": "choice",
     "opcoes": ["", "cm-1", "nm", "raman_shift", "min", "ppm", "ms", "indice"]},
    {"key": "tecnica_analitica", "attr": "technique", "tipo": "choice",
     "opcoes": [""] + list(_INSTRUMENTOS_CATALOGO)},
    {"key": "perfil_tecnica", "attr": "acquisition_profile", "tipo": "choice",
     "opcoes": [""] + perfis_disponiveis(apenas="tecnica")},
    {"key": "pasta_dados", "attr": "input_folder", "tipo": "str",
     "opcoes": None},
    {"key": "hsi_pasta_dataset", "attr": "hsi_dataset_folder", "tipo": "str",
     "opcoes": None},
    {"key": "hsi_calibracao", "attr": "hsi_calibration", "tipo": "choice",
     "opcoes": ["nenhuma", "branco_escuro"]},
    {"key": "hsi_referencia_branco", "attr": "hsi_white_ref", "tipo": "str",
     "opcoes": None},
    {"key": "hsi_referencia_escuro", "attr": "hsi_dark_ref", "tipo": "str",
     "opcoes": None},
    {"key": "hsi_eixo_bandas", "attr": "hsi_band_axis", "tipo": "choice",
     "opcoes": ["recusar", "reamostrar"]},
    {"key": "hsi_validacao", "attr": "hsi_validation", "tipo": "choice",
     "opcoes": ["cv_objetos", "holdout"]},
    {"key": "hsi_cv_dobras", "attr": "hsi_cv_folds", "tipo": "int",
     "opcoes": None, "min": 0, "max": 200},
    {"key": "hplc_detector", "attr": "hplc_detector", "tipo": "choice",
     "opcoes": ["UV", "FID"]},
    {"key": "cromatografia_canal", "attr": "cromatografia_canal", "tipo": "str",
     "opcoes": None},
    {"key": "espectro_max_pontos", "attr": "espectro_max_pontos", "tipo": "int",
     "opcoes": None, "min": 0, "max": 10000000},
    {"key": "cromatograma_max_pontos", "attr": "cromatograma_max_pontos", "tipo": "int",
     "opcoes": None, "min": 0, "max": 10000000},
    {"key": "eem_espalhamento", "attr": "eem_espalhamento", "tipo": "choice",
     "opcoes": ["nenhum", "nan", "zero", "interp"]},
    {"key": "eem_espalhamento_largura_nm", "attr": "eem_espalhamento_largura_nm", "tipo": "list",
     "opcoes": None},
    {"key": "eem_raman", "attr": "eem_raman", "tipo": "bool",
     "opcoes": None},
    {"key": "parafac_nao_negativo", "attr": "parafac_nao_negativo", "tipo": "bool",
     "opcoes": None},
    {"key": "icp_unidade", "attr": "icp_unidade", "tipo": "str_opcional",
     "opcoes": None},
    {"key": "icp_coluna_amostra", "attr": "icp_coluna_amostra", "tipo": "str",
     "opcoes": None},
    {"key": "icp_coluna_tipo", "attr": "icp_coluna_tipo", "tipo": "str_opcional",
     "opcoes": None},
    {"key": "icp_coluna_sessao", "attr": "icp_coluna_sessao", "tipo": "str_opcional",
     "opcoes": None},
    {"key": "icp_padrao_interno", "attr": "icp_padrao_interno", "tipo": "str_opcional",
     "opcoes": None},
    {"key": "icp_politica_censura", "attr": "icp_politica_censura", "tipo": "choice",
     "opcoes": ["substituir", "excluir"]},
    {"key": "icp_limiar_censura", "attr": "icp_limiar_censura", "tipo": "float",
     "opcoes": None, "min": 0.0, "max": 1.0},
    {"key": "icp_log", "attr": "icp_log", "tipo": "bool",
     "opcoes": None},
    {"key": "imagem_incluir_textura", "attr": "include_image_texture", "tipo": "bool",
     "opcoes": None},
    {"key": "arquivo_csv", "attr": "csv_file", "tipo": "str",
     "opcoes": None},
    {"key": "coluna_classe", "attr": "class_column", "tipo": "str",
     "opcoes": None},
    {"key": "coluna_concentracao", "attr": "conc_column", "tipo": "str_opcional",
     "opcoes": None},
    {"key": "coluna_grupo", "attr": "group_column", "tipo": "str_opcional",
     "opcoes": None},
    {"key": "arquivo_grupos", "attr": "group_file", "tipo": "str_opcional",
     "opcoes": None},
    {"key": "unidade_agrupamento", "attr": "group_unit", "tipo": "choice",
     "opcoes": ["auto", "mae_id", "sessao"]},
    {"key": "pasta_saida", "attr": "output_root_folder", "tipo": "str",
     "opcoes": None},
    {"key": "ajuste_automatico", "attr": "auto_adjust", "tipo": "bool",
     "opcoes": None},
    {"key": "nivel", "attr": "level", "tipo": "choice",
     "opcoes": ["N1", "N2", "N3"]},
    {"key": "objetivo", "attr": "objective", "tipo": "choice",
     "opcoes": ["auto", "exploratorio", "classificacao", "quantificacao"]},
    {"key": "pre_processamento", "attr": "default_preprocessing", "tipo": "preproc",
     "opcoes": list(_PRE_PROC_FRIENDLY)},
    # min negativo: deslocamento Raman anti-Stokes (ha' arquivos Raman que comecam abaixo de 0 cm-1)
    {"key": "faixa_min_cm", "attr": "wn_min", "tipo": "float",
     "opcoes": None, "min": -5000.0},
    {"key": "faixa_max_cm", "attr": "wn_max", "tipo": "float",
     "opcoes": None, "min": 0.0},
    # lista de "a-b" na unidade do eixo; formato validado em _coagir_valor (validar="faixas")
    {"key": "faixas_excluir", "attr": "exclude_ranges", "tipo": "list",
     "opcoes": None, "validar": "faixas"},
    {"key": "saturacao", "attr": "saturacao", "tipo": "choice",
     "opcoes": ["auto", "avisar", "nenhuma"]},
    {"key": "normalizacao", "attr": "normalization", "tipo": "choice",
     "opcoes": ["nenhuma", "rip", "area"]},
    {"key": "ims_rip_janela", "attr": "rip_window", "tipo": "str_opcional",
     "opcoes": None, "validar": "janela"},
    {"key": "excluir_classes", "attr": "exclude_classes", "tipo": "list",
     "opcoes": None},
    {"key": "max_lvs", "attr": "max_lvs", "tipo": "int",
     "opcoes": None,
     "min": 1, "max": 200},
    {"key": "selecao_lv_cv_aninhada", "attr": "selecao_lv_cv_aninhada", "tipo": "bool",
     "opcoes": None},
    {"key": "holdout_fracao", "attr": "frac_holdout", "tipo": "float",
     "opcoes": None,
     "min": 0.0, "max": 0.5},
    {"key": "validacao_group_aware", "attr": "group_by_mae_id", "tipo": "bool",
     "opcoes": None},
    {"key": "holdout_estratificado", "attr": "holdout_estratificado", "tipo": "bool",
     "opcoes": None},
    {"key": "aplicar_correcoes_locais", "attr": "apply_local_corrections", "tipo": "bool",
     "opcoes": None},
    {"key": "n_permutacoes", "attr": "n_permutations", "tipo": "int",
     "opcoes": None, "min": 1, "max": 100000},
    {"key": "teste_wold", "attr": "run_wold", "tipo": "bool",
     "opcoes": None},
    {"key": "n_jobs_permutacao", "attr": "n_jobs_permutation", "tipo": "int",
     "opcoes": None, "min": 1, "max": 64},
    {"key": "teste_cv_anova", "attr": "run_cv_anova", "tipo": "bool",
     "opcoes": None},
    {"key": "teste_martens", "attr": "run_martens", "tipo": "bool",
     "opcoes": None},
    {"key": "selecao_variaveis_etapa4", "attr": "executar_etapa4", "tipo": "bool",
     "opcoes": None},
    {"key": "selecao_spa", "attr": "run_spa", "tipo": "bool",
     "opcoes": None},
    {"key": "selecao_ag", "attr": "executar_ag", "tipo": "bool",
     "opcoes": None},
    {"key": "selecao_cars", "attr": "run_cars", "tipo": "bool",
     "opcoes": None},
    {"key": "selecao_uve", "attr": "run_uve", "tipo": "bool",
     "opcoes": None},
    {"key": "ddsimca", "attr": "run_ddsimca", "tipo": "bool",
     "opcoes": None},
    {"key": "modo_ddsimca", "attr": "ddsimca_train_on", "tipo": "choice",
     "opcoes": ["puros", "todos"]},
    {"key": "ddsimca_pcv", "attr": "ddsimca_pcv", "tipo": "bool",
     "opcoes": None},
    {"key": "opls_da", "attr": "run_opls", "tipo": "bool",
     "opcoes": None},
    {"key": "comparar_pre_processamentos", "attr": "comparar_pipelines", "tipo": "bool",
     "opcoes": None},
    {"key": "benchmark", "attr": "run_benchmark", "tipo": "bool",
     "opcoes": None},
    {"key": "benchmark_regressao", "attr": "run_benchmark_regression", "tipo": "bool",
     "opcoes": None},
    {"key": "monte_carlo", "attr": "run_monte_carlo", "tipo": "bool",
     "opcoes": None},
    {"key": "n_monte_carlo", "attr": "n_monte_carlo", "tipo": "int",
     "opcoes": None, "min": 1, "max": 100000},
    {"key": "monte_carlo_incluir_todos", "attr": "monte_carlo_include_all", "tipo": "bool",
     "opcoes": None},
    {"key": "shap_benchmark", "attr": "run_shap", "tipo": "bool",
     "opcoes": None},
    {"key": "shap_max_amostras", "attr": "shap_max_samples", "tipo": "int",
     "opcoes": None,
     "min": 1, "max": 1000000},
    {"key": "figuras_detalhadas", "attr": "detailed_figures", "tipo": "bool",
     "opcoes": None},
    {"key": "figuras_mostrar_marcadores", "attr": "show_class_markers", "tipo": "bool",
     "opcoes": None},
    {"key": "figuras_mostrar_elipses", "attr": "show_group_ellipses", "tipo": "bool",
     "opcoes": None},
    {"key": "formato_figura", "attr": "output_format", "tipo": "choice",
     "opcoes": ["png", "pdf", "svg"]},
    {"key": "dpi", "attr": "save_dpi", "tipo": "int",
     "opcoes": None, "min": 50, "max": 1200},
    {"key": "abrir_figuras_na_tela", "attr": "show_plots", "tipo": "bool",
     "opcoes": None},
    {'key': 'protocolo_robustez', 'attr': 'robustness_protocol', 'tipo': 'bool', 'opcoes': None},
    {'key': 'graficos', 'attr': 'figure_selection', 'tipo': 'str', 'opcoes': None},
    # ---- parametros expostos em 2026-10-03 (antes so' pela API Python)
    {'key': 'etapas_pre_processamento', 'attr': 'etapas_pre_processamento', 'tipo': 'list', 'opcoes': ['airpls', 'snv', 'pqn', 'emsc', 'sg', 'mc', 'osc']},
    {'key': 'sg_janela', 'attr': 'sg_window', 'tipo': 'int', 'opcoes': None, 'min': 3, 'max': 101},
    {'key': 'sg_ordem', 'attr': 'sg_polyorder', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 5},
    {'key': 'sg_derivada', 'attr': 'sg_deriv', 'tipo': 'int', 'opcoes': None, 'min': 0, 'max': 2},
    {'key': 'emsc_ordem', 'attr': 'emsc_ordem_polinomial', 'tipo': 'int', 'opcoes': None, 'min': 0, 'max': 4},
    {'key': 'osc_componentes', 'attr': 'osc_n_componentes', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 5},
    {'key': 'airpls_lambda', 'attr': 'airpls_lam', 'tipo': 'float', 'opcoes': None, 'min': 1.0},
    {'key': 'remover_raios_cosmicos', 'attr': 'apply_despike', 'tipo': 'bool', 'opcoes': None},
    {'key': 'raios_cosmicos_limiar', 'attr': 'despike_limiar', 'tipo': 'float', 'opcoes': None, 'min': 1.0, 'max': 100.0},
    {'key': 'raios_cosmicos_janela', 'attr': 'despike_janela', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 50},
    {'key': 'alinhamento', 'attr': 'alinhamento', 'tipo': 'choice', 'opcoes': ['nenhum', 'cow']},
    {'key': 'alinhamento_segmentos', 'attr': 'alinhamento_segmentos', 'tipo': 'int', 'opcoes': None, 'min': 2, 'max': 2000},
    {'key': 'alinhamento_folga', 'attr': 'alinhamento_folga', 'tipo': 'int', 'opcoes': None, 'min': 0, 'max': 5000},
    {'key': 'classificador', 'attr': 'classificador', 'tipo': 'choice', 'opcoes': ['plsda', 'plslda']},
    {'key': 'quantificacao_rotulo_treino', 'attr': 'quant_rotulo_treino', 'tipo': 'choice', 'opcoes': ['previsto', 'verdadeiro']},
    {'key': 'pca_componentes', 'attr': 'n_pcs_pca', 'tipo': 'int', 'opcoes': None, 'min': 2, 'max': 50},
    {'key': 'hca_componentes', 'attr': 'hca_n_pcs', 'tipo': 'int', 'opcoes': None, 'min': 2, 'max': 200},
    {'key': 'comparar_hca', 'attr': 'compare_hca_pipelines', 'tipo': 'bool', 'opcoes': None},
    {'key': 'ddsimca_componentes', 'attr': 'ddsimca_n_components', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 30},
    {'key': 'ddsimca_limite', 'attr': 'ddsimca_ucl_method', 'tipo': 'choice', 'opcoes': ['empirical', 'theoretical', 'chi2']},
    {'key': 'opls_ortogonais', 'attr': 'n_ortho_opls', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 5},
    {'key': 'ipls_intervalos', 'attr': 'ipls_n_intervalos', 'tipo': 'int', 'opcoes': None, 'min': 2, 'max': 100},
    {'key': 'vip_limiar', 'attr': 'vip_threshold_sel', 'tipo': 'float', 'opcoes': None, 'min': 0.0, 'max': 5.0},
    {'key': 'sr_fracao', 'attr': 'sr_top_frac', 'tipo': 'float', 'opcoes': None, 'min': 0.01, 'max': 1.0},
    {'key': 'splsda_variaveis', 'attr': 'splsda_keep_by_comp', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 5000},
    {'key': 'spa_max_variaveis', 'attr': 'spa_n_vars_max', 'tipo': 'int', 'opcoes': None, 'min': 2, 'max': 200},
    {'key': 'spa_inicios', 'attr': 'spa_n_starts', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 500},
    {'key': 'ag_populacao', 'attr': 'ag_tam_populacao', 'tipo': 'int', 'opcoes': None, 'min': 4, 'max': 500},
    {'key': 'ag_geracoes', 'attr': 'ag_n_geracoes', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 500},
    {'key': 'ag_mutacao', 'attr': 'ag_prob_mutacao', 'tipo': 'float', 'opcoes': None, 'min': 0.0, 'max': 0.5},
    {'key': 'ag_fracao_inicial', 'attr': 'ag_frac_inicial', 'tipo': 'float', 'opcoes': None, 'min': 0.01, 'max': 1.0},
    {'key': 'cars_iteracoes', 'attr': 'cars_n_iteracoes', 'tipo': 'int', 'opcoes': None, 'min': 2, 'max': 1000},
    {'key': 'cars_fracao', 'attr': 'cars_frac_amostragem', 'tipo': 'float', 'opcoes': None, 'min': 0.1, 'max': 1.0},
    {'key': 'uve_repeticoes', 'attr': 'uve_n_repeticoes', 'tipo': 'int', 'opcoes': None, 'min': 2, 'max': 1000},
    {'key': 'uve_fracao', 'attr': 'uve_frac_amostragem', 'tipo': 'float', 'opcoes': None, 'min': 0.1, 'max': 1.0},
    {'key': 'cv_dobras', 'attr': 'n_splits_cv', 'tipo': 'int', 'opcoes': None, 'min': 2, 'max': 20},
    {'key': 'cv_repeticoes', 'attr': 'n_repeats_cv', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 50},
    {'key': 'semente', 'attr': 'seed', 'tipo': 'int', 'opcoes': None, 'min': 0, 'max': 2147483647},
    {'key': 'n_permutacoes_wold', 'attr': 'n_permutations_wold', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 100000},
    {'key': 'n_bootstrap_vip', 'attr': 'n_bootstrap_vip', 'tipo': 'int', 'opcoes': None, 'min': 0, 'max': 10000},
    {'key': 'n_bootstrap_bca', 'attr': 'n_bootstrap_bca', 'tipo': 'int', 'opcoes': None, 'min': 0, 'max': 100000},
    {'key': 'holdout_preservar_puras', 'attr': 'holdout_preserves_pure', 'tipo': 'bool', 'opcoes': None},
    {'key': 'semente_holdout', 'attr': 'seed_holdout', 'tipo': 'int', 'opcoes': None, 'min': 0, 'max': 2147483647},
    {'key': 'fracao_calibracao', 'attr': 'frac_cal', 'tipo': 'float', 'opcoes': None, 'min': 0.3, 'max': 0.95},
    {'key': 'divisao_calibracao', 'attr': 'cal_val_split', 'tipo': 'choice', 'opcoes': ['aleatoria', 'kennard_stone']},
    {'key': 'monte_carlo_fracao_teste', 'attr': 'monte_carlo_test_size', 'tipo': 'float', 'opcoes': None, 'min': 0.05, 'max': 0.5},
    {'key': 'sintetico_n_por_classe', 'attr': 'n_per_class', 'tipo': 'int', 'opcoes': None, 'min': 2, 'max': 1000},
    {'key': 'sintetico_n_pontos', 'attr': 'n_synthetic_points', 'tipo': 'int', 'opcoes': None, 'min': 10, 'max': 20000},
    {'key': 'sintetico_replicas', 'attr': 'n_synthetic_replicates', 'tipo': 'int', 'opcoes': None, 'min': 1, 'max': 10},
    {'key': 'sintetico_adulterantes', 'attr': 'synthetic_adulterants', 'tipo': 'list', 'opcoes': None},
    {'key': 'sintetico_cenario', 'attr': 'synthetic_scenario', 'tipo': 'choice', 'opcoes': ['classico', 'nir', 'raman', 'rmn']},
    {'key': 'dx_ler_titulo', 'attr': 'use_parse_title', 'tipo': 'bool', 'opcoes': None},
    {'key': 'dx_parte_classe', 'attr': 'class_part', 'tipo': 'int', 'opcoes': None, 'min': -10, 'max': 10},
    {'key': 'dx_teor_no_nome', 'attr': 'extract_conc_from_filename', 'tipo': 'bool', 'opcoes': None},
    {'key': 'imagem_recorte', 'attr': 'imagem_recorte', 'tipo': 'recorte', 'opcoes': None},
    {'key': 'imagem_referencia', 'attr': 'imagem_referencia', 'tipo': 'recorte', 'opcoes': None, 'vazio_ok': True},
    {'key': 'imagem_segmentacao', 'attr': 'image_segmentation', 'tipo': 'choice',
     'opcoes': ['nenhuma', 'otsu', 'distancia_fundo']},
    {'key': 'tag', 'attr': 'tag', 'tipo': 'str', 'opcoes': None},
]

#: Campos da Config SEM chave no config.yaml, com o motivo (tests/test_contrato_chaves_config.py).
#: Estado calculado pelo motor ou rota propria; parametro de decisao do usuario nao entra aqui.
CAMPOS_INTERNOS: Dict[str, str] = {
    "correcoes_locais_aplicadas": "estado: registro das correcoes locais aplicadas pelo leitor nesta execucao",
    "multielementar_resumo": "estado: resumo calculado pelo modo multielementar",
    "output_folder": "estado: pasta desta execucao, criada pelo motor (a escolha do usuario e' pasta_saida)",
    "grouping_guarantee": "estado: garantia de agrupamento calculada a partir dos dados",
    "ajustes_hardware": "estado: registro dos ajustes de hardware feitos pelo motor",
    "label_mode": "rota propria: flag --mode=cego|controle da linha de comando",
}

#: Opcoes do catalogo que nao vao para o config.yaml, com o motivo.
CHAVES_FORA_DO_YAML: Dict[str, str] = {
    "nome_execucao": "alias de `tag` no web (mesmo atributo; o config.yaml grava `tag`)",
    "paleta_cores": "preferencia visual do usuario, gravada em ~/.guaraci (tela de visualizacao)",
    "tamanho_fonte": "preferencia visual do usuario, gravada em ~/.guaraci (tela de visualizacao)",
    "grid_major": "preferencia visual do usuario, gravada em ~/.guaraci (tela de visualizacao)",
    "alpha_pontos": "preferencia visual do usuario, gravada em ~/.guaraci (tela de visualizacao)",
    "codificacao_arquivos": "preferencia do usuario, gravada em ~/.guaraci (tela de codificacao dos .dx)",
}

# O texto de cada campo vive no catalogo (guaraci.catalogo_opcoes), nao aqui:
# `desc` = "o que faz", usado como rotulo/ajuda de fallback no CLI e na web.
from guaraci import catalogo as _catalogo  # noqa: E402

for _s in _CONFIG_SPEC:
    # Todo campo do spec tem entrada no catalogo (garantido por test_catalogo); a ausencia seria um defeito do catalogo, e
    # `faz()` a acusa na importacao em vez de cair num texto de reserva.
    _s["desc"] = _catalogo.faz(_s["key"], "PT")


def _attr_para_yaml(spec: Dict[str, Any], cfg: Config) -> Any:
    """Le o atributo da Config e converte para a forma amigavel do YAML."""
    v = getattr(cfg, spec["attr"], getattr(Config(), spec["attr"], None))
    if spec["tipo"] == "preproc":
        return _PRE_PROC_INV.get(str(v).lower(), str(v))
    if spec["tipo"] == "list":
        return list(v) if v is not None else []
    if spec["tipo"] == "recorte":
        return [float(x) for x in v]
    if spec["tipo"] == "str_opcional":
        return "" if v is None else str(v)
    return v


def _checar_faixa(spec: Dict[str, Any], v: float) -> float:
    """Valida um numero contra 'min'/'max' opcionais do spec, com mensagem
    amigavel. Impede que valores impossiveis (fracao negativa, contagem zero)
    passem silenciosamente e so quebrem — ou pior, distorcam o resultado —
    no meio da execucao."""
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and v < lo:
        raise ValueError(
            f"{spec['key']}={v}: valor minimo permitido e {lo}")
    if hi is not None and v > hi:
        raise ValueError(
            f"{spec['key']}={v}: valor maximo permitido e {hi}")
    return v


def _coagir_valor(spec: Dict[str, Any], val: Any) -> Any:
    """Converte um valor do YAML/menu para o tipo correto do atributo Config,
    validando opcoes. Lanca ValueError com mensagem amigavel se invalido."""
    t = spec["tipo"]
    if spec.get("validar") == "faixas":
        from guaraci.preprocessamento import formatar_faixas, interpretar_faixas
        try:
            return formatar_faixas(interpretar_faixas(val))
        except ValueError as e:
            raise ValueError(f"{spec['key']}: {e}") from e
    if spec.get("validar") == "janela":
        from guaraci.preprocessamento import formatar_faixas, interpretar_janela
        try:
            jan = interpretar_janela(val)
        except ValueError as e:
            raise ValueError(f"{spec['key']}: {e}") from e
        return formatar_faixas([jan])[0] if jan else None
    if t == "str_opcional":
        s2 = str(val).strip()
        return s2 if s2 else None
    if t == "bool":
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("true", "sim", "1", "yes", "s", "v")
    if t == "int":
        return int(_checar_faixa(spec, int(val)))
    if t == "float":
        return _checar_faixa(spec, float(val))
    if t == "list":
        if val is None:
            itens: Tuple[str, ...] = ()
        elif isinstance(val, (list, tuple)):
            itens = tuple(str(x).strip() for x in val if str(x).strip())
        else:
            itens = tuple(x.strip() for x in str(val).split(",") if x.strip())
        if spec.get("opcoes"):
            # lista de opcoes fechadas (etapas_pre_processamento): valida aqui e devolve na ordem
            # canonica das opcoes, a mesma que a Config devolve (achado da matriz 3.10/3.11)
            pedidas = {x.lower() for x in itens}
            fora = sorted(pedidas - set(spec["opcoes"]))
            if fora:
                raise ValueError(f"{spec['key']}: valor(es) {fora} invalido(s); use {list(spec['opcoes'])}")
            return tuple(o for o in spec["opcoes"] if o in pedidas)
        return itens
    if t == "recorte":
        # x0, y0, x1, y1 em fracao da foto (achado da matriz 3.10/3.11 de 2026-10-03: como "list"
        # de texto, o tipo aceitava qualquer coisa e o erro so' aparecia ao atribuir na Config)
        if spec.get("vazio_ok") and (val is None or (isinstance(val, (list, tuple)) and not val)
                                     or str(val).strip() in ("", "[]", "None", "none")):
            return ()      # campo opcional (imagem_referencia): vazio = desligado
        partes = val if isinstance(val, (list, tuple)) else str(val).split(",")
        try:
            v = tuple(float(str(x).strip()) for x in partes)
        except ValueError:
            v = ()
        if len(v) != 4 or not all(0.0 <= x <= 1.0 for x in v) or not (v[0] < v[2] and v[1] < v[3]):
            raise ValueError(f"{spec['key']}: use 4 valores entre 0 e 1 (x0, y0, x1, y1) com x0 < x1 e y0 < y1")
        return v
    if t == "choice":
        sv = str(val)
        # perfil_matriz/perfil_tecnica: a lista de `opcoes` e' so' um atalho
        # de UI (picker numerado) para os perfis EMBUTIDOS -- o campo tambem
        # aceita o caminho de um YAML proprio (load_profile() ja documenta e
        # valida isso). Travar aqui quebraria qualquer config.yaml existente
        # com um perfil de usuario -- a checagem real acontece so' quando o
        # perfil e' de fato carregado, nao no parse do config.
        if (spec["opcoes"] and sv not in spec["opcoes"]
                and spec["key"] not in ("perfil_matriz", "perfil_tecnica")):
            raise ValueError(f"valor '{sv}' invalido; use {spec['opcoes']}")
        return sv
    if t == "preproc":
        sv = str(val)
        if sv in _PRE_PROC_FRIENDLY:
            return _PRE_PROC_FRIENDLY[sv]
        if sv.lower() in _PRE_PROC_INV:
            return sv.lower()
        raise ValueError(
            f"pre-processamento '{sv}' invalido; use {list(_PRE_PROC_FRIENDLY)}")
    return str(val)


def _validar_semantico(cfg: "Config") -> List[str]:
    """Validacoes CRUZADAS entre campos, que 'min'/'max' isolados nao pegam.

    Retorna lista de mensagens amigaveis (vazia = tudo certo). Deve ser chamada
    ANTES de rodar (app e CLI) para falhar cedo, com mensagem clara, em vez de
    quebrar so no meio da execucao (depois de minutos carregando/processando).
    """
    erros: List[str] = []

    # Selecao de graficos: padrao, todos ou nomes do catalogo separados por virgula.
    sel = str(getattr(cfg, "figure_selection", "padrao") or "padrao").strip().lower()
    if sel not in ("padrao", "todos"):
        from guaraci.catalogo_tecnicas import GRAFICOS
        desconhecidos = sorted({c.strip() for c in sel.split(",") if c.strip()} - set(GRAFICOS))
        if desconhecidos:
            erros.append(f"graficos: nome(s) desconhecido(s) {desconhecidos}; use padrao, todos ou nomes de "
                         f"`guaraci ajuda graficos`.")

    # Savitzky-Golay: janela impar e maior que a ordem (senao o filtro nao e' definido).
    try:
        jan, ordem = int(cfg.sg_window), int(cfg.sg_polyorder)
        if jan % 2 == 0 or jan <= ordem:
            erros.append(f"Savitzky-Golay invalido: a janela ({jan}) deve ser impar e maior que a ordem ({ordem}).")
    except (TypeError, ValueError):
        pass

    # Faixa espectral: o inicio precisa ser menor que o fim. Se invertido, o
    # filtro [wn_min, wn_max] remove TODAS as variaveis e o pipeline so
    # quebraria depois de carregar os dados (crash tardio, confuso).
    try:
        if float(cfg.wn_min) >= float(cfg.wn_max):
            erros.append(
                f"Faixa espectral invalida: o inicio ({cfg.wn_min:.0f}) deve "
                f"ser MENOR que o fim ({cfg.wn_max:.0f}) cm-1.")
    except (TypeError, ValueError):
        pass

    # EEM: larguras de espalhamento em nm, numeros > 0 (um por ordem ou um para todas).
    try:
        from guaraci.eem_io import larguras_do_config
        larguras_do_config(getattr(cfg, "eem_espalhamento_largura_nm", ()))
    except ValueError as e:
        erros.append(str(e))
    # Faixas excluidas e normalizacao por RIP: formato (a<b) e coerencia entre as chaves. O cruzamento com
    # o EIXO dos dados so' e' possivel depois de ler o arquivo (pipeline e `guaraci checar --dados`).
    try:
        from guaraci.preprocessamento import interpretar_faixas, interpretar_janela
        interpretar_faixas(getattr(cfg, "exclude_ranges", ()))
        jan_rip = interpretar_janela(getattr(cfg, "rip_window", None))
        norm = str(getattr(cfg, "normalization", "nenhuma") or "nenhuma")
        if norm == "rip" and jan_rip is None:
            erros.append("normalizacao: rip exige ims_rip_janela (ex.: '7-9', na unidade do eixo, "
                         "cobrindo o pico do RIP).")
        if norm == "rip" and jan_rip is not None and (jan_rip[1] < float(cfg.wn_min) or jan_rip[0] > float(cfg.wn_max)):
            erros.append(f"ims_rip_janela {jan_rip[0]:g}-{jan_rip[1]:g} esta fora da faixa usada "
                         f"({cfg.wn_min:g} a {cfg.wn_max:g}): a normalizacao e' feita sobre o espectro ja cortado.")
    except ValueError as e:
        erros.append(f"faixas_excluir/ims_rip_janela invalido: {e}")

    # Holdout: precisa sobrar treino suficiente. 'max'=0.5 ja barra pelo widget,
    # mas um config.yaml editado a mao pode trazer valor fora da faixa — aqui e
    # o backstop para esse caminho (Config cru, sem passar por _coagir_valor).
    try:
        fh = float(cfg.frac_holdout)
        if fh < 0.0 or fh > 0.5:
            erros.append(
                f"Fracao de holdout invalida ({fh}): use um valor entre 0 e 0.5 "
                "(0 = sem teste externo).")
    except (TypeError, ValueError):
        pass

    return erros


def _yaml_reinterpretaria(v: str) -> bool:
    """True se PyYAML leria `v` (sem aspas) como algo DIFERENTE da propria
    string. Nao e' so' os 6 nomes reservados (true/false/null/yes/no/~) --
    e' qualquer escalar YAML implicito: int (inclusive octal -- '010' virava
    8 em silencio, achado por teste de propriedade do Hypothesis em
    2026-08-27, Passo 85), float com zeros a direita ('1.50' virava 1.5,
    perdendo digito), etc. Usa o PROPRIO parser como oraculo em vez de
    reimplementar a regex do resolver -- garante que bate exatamente com o
    que `load_config` de fato faz, sem risco de as duas regras divergirem."""
    try:
        import yaml
    except ImportError:
        return True   # sem PyYAML p/ conferir: aspeia por seguranca
    try:
        return yaml.safe_load(v) != v
    except yaml.YAMLError:
        return True   # nao parseia nem como escalar simples: aspeia por seguranca


def _fmt_yaml(v: Any) -> str:
    """Formata um valor Python como YAML simples (para o arquivo comentado).
    Usa ASPAS SIMPLES quando precisa citar: em YAML, dentro de aspas simples a
    barra invertida e literal — essencial p/ caminhos do Windows (C:\\Users\\alguem).
    Em aspas duplas, '\\U' / '\\D' seriam sequencias de escape e quebrariam a leitura."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_yaml(x) for x in v) + "]"
    if isinstance(v, str):
        # O check de pontuacao cobre o caso em que v vira ITEM de uma lista
        # flow "[a, b]" (um "?" ou virgula/colchete dentro do item quebraria
        # a lista mesmo que v sozinho parseasse limpo como escalar -- "?" e'
        # so' perigoso em CONTEXTO de fluxo, achado por Hypothesis testando
        # o campo "list" `excluir_classes` no Passo 85: item "0?" nem
        # parseava, item "?0" virava mapa {0: None} em silencio); o oraculo
        # YAML cobre o caso em que v sozinho ja' parseia para outra coisa
        # (int/float/bool/null/strip). Precisa dos dois.
        precisa = (any(c in v for c in ':#,?[]{}&*!|>%@`"\'')
                   or _yaml_reinterpretaria(v))
        if precisa:
            return "'" + v.replace("'", "''") + "'"
        return v
    return str(v)


def save_config(cfg: Config, caminho: str) -> None:
    """Escreve config.yaml em linguagem simples, com um comentario explicativo
    acima de cada campo. Regenera os comentarios a cada salvamento."""
    linhas = [
        "# " + "=" * 60,
        "#  CONFIGURACAO DO PIPELINE QUIMIOMETRICO",
        "#  Edite os valores abaixo. Cada campo tem uma explicacao no",
        "#  comentario logo acima. Voce NAO precisa abrir o codigo.",
        "#  Para rodar:   python pipeline.py --rodar",
        "#  Assistente:   python guaraci.py",
        "# " + "=" * 60,
        "",
    ]
    for s in _CONFIG_SPEC:
        linhas.append(f"# {s['desc']}")
        if s["opcoes"]:
            linhas.append(f"#   opcoes: {' | '.join(map(str, s['opcoes']))}")
        linhas.append(f"{s['key']}: {_fmt_yaml(_attr_para_yaml(s, cfg))}")
        linhas.append("")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


def load_config(caminho: str, base: Optional[Config] = None) -> Config:
    """Le config.yaml e devolve uma Config. Mantem os defaults para chaves
    ausentes; reune erros (valor invalido OU chave desconhecida) numa
    mensagem clara.

    CHAVE DESCONHECIDA E' ERRO, NAO E' IGNORADA EM SILENCIO. Ate
    2026-08-20 este metodo pulava silenciosamente qualquer chave que nao
    reconhecesse -- um campo digitado errado, ou de uma versao diferente
    do software, fazia o pipeline rodar com o DEFAULT daquele campo sem
    aviso nenhum. Para um pipeline cientifico isso e' pior que um crash:
    produz resultado plausivel e errado. Achado na verificacao
    independente do Passo 24c; corrigido aqui reaproveitando o mesmo
    agregador de erros que ja' existia para valor invalido, em vez de
    introduzir um mecanismo novo so' de aviso.
    """
    try:
        import yaml
    except ImportError as _e_yaml:
        # `from _e_yaml`: preserva a cadeia de excecoes. Sem isso o traceback
        # mostra "During handling of the above exception, another exception
        # occurred", que sugere falha DENTRO do tratamento de erro em vez de
        # dependencia ausente -- confunde o diagnostico do usuario.
        raise RuntimeError(
            "PyYAML nao instalado. Rode: pip install pyyaml") from _e_yaml
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Config nao encontrado: {caminho}. Gere um pelo assistente "
            f"(opcao [S] salvar) antes de usar --rodar.")
    with open(caminho, "r", encoding="utf-8") as f:
        dados = yaml.safe_load(f) or {}
    cfg = base if base is not None else Config()
    spec_por_key = {s["key"]: s for s in _CONFIG_SPEC}
    chaves_conhecidas = sorted(spec_por_key)
    erros: List[str] = []
    for key, val in dados.items():
        s = spec_por_key.get(key)
        if s is None:
            sugestao = get_close_matches(key, chaves_conhecidas, n=1)
            dica = f" -- voce quis dizer '{sugestao[0]}'?" if sugestao else ""
            erros.append(f"  - '{key}': chave desconhecida{dica}")
            continue
        try:
            setattr(cfg, s["attr"], _coagir_valor(s, val))
        except Exception as e:  # noqa: BLE001 -- agregador DELIBERADO: junta
            # o erro de CADA chave num relatorio unico (raise ValueError
            # abaixo) em vez de parar na 1a falha; _coagir_valor lanca
            # majoritariamente ValueError, mas um TypeError isolado (ex.:
            # tipo incompativel no YAML) nao deve impedir reportar as
            # demais chaves invalidas.
            erros.append(f"  - '{key}': {e}")
    if erros:
        raise ValueError("Problemas no config.yaml:\n" + "\n".join(erros))
    return cfg


def _validar_pasta_dados(cfg: Config) -> Tuple[bool, str]:
    """Checagem amigavel da fonte de dados, ciente do mode de entrada.
    Generico: serve para .dx (FT-NIR), CSV (qualquer dado tabular) ou
    dados sinteticos de teste."""
    mode = getattr(cfg, "mode", "dx")
    if mode == "sintetico":
        return True, "OK — modo sintetico (dados gerados em memoria, sem arquivo)"
    if mode == "csv":
        cam = cfg.csv_file
        if not cam or not os.path.isfile(cam):
            return False, f"CSV nao encontrado: '{cam}' (confira o caminho)"
        # R5-03: "OK" verde para arquivo lixo/vazio era falso. Le so' o comeco.
        try:
            from guaraci.csv_robusto import CsvDecimalAmbiguoError, ler_csv
            amostra = ler_csv(cam, nrows=5)
        except CsvDecimalAmbiguoError as e:
            return False, f"'{os.path.basename(cam)}': {e}"
        except Exception as e:  # noqa: BLE001 -- qualquer falha de leitura = nao e' CSV utilizavel
            return False, (f"'{os.path.basename(cam)}' nao parece um CSV de "
                           f"espectros (vazio ou nao e' texto tabular: {type(e).__name__})")
        col_cls = getattr(cfg, "class_column", "") or ""
        nomes = [str(c) for c in amostra.columns]
        def _num(c):
            try:
                float(c.strip().replace(",", "."))
                return True
            except ValueError:
                return False
        n_esp = sum(1 for c in nomes if c != col_cls and _num(c))
        if n_esp < 2:
            return False, (f"'{os.path.basename(cam)}' nao tem colunas espectrais "
                           f"(cabecalhos numericos, ex.: 4000.5). Colunas: "
                           f"{', '.join(nomes[:6])}")
        if col_cls and col_cls not in nomes:
            return False, (f"a coluna de classe '{col_cls}' nao existe em "
                           f"'{os.path.basename(cam)}'. Colunas nao numericas: "
                           f"{', '.join(c for c in nomes if not _num(c)) or 'nenhuma'}")
        return True, f"OK — CSV: {os.path.basename(cam)}"
    if mode == "multielementar":
        cam = cfg.csv_file
        if not cam or not os.path.isfile(cam):
            return False, f"arquivo ICP nao encontrado: '{cam}' (campo arquivo_csv)"
        if not str(getattr(cfg, "icp_unidade", "") or "").strip():
            return False, "informe icp_unidade (ex.: mg/kg) -- obrigatoria no mode multielementar"
        if not getattr(cfg, "group_column", None):
            return False, ("informe coluna_grupo (a amostra fisica/lote independente) "
                           "-- obrigatoria no mode multielementar")
        return True, f"OK — ICP: {os.path.basename(cam)}"
    if mode == "imagem":
        p_img = cfg.input_folder
        if not p_img or not os.path.isdir(p_img):
            return False, f"pasta nao encontrada: '{p_img}' (confira o caminho)"
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        n_img = sum(len(glob.glob(os.path.join(p_img, "**", f"*{e}"), recursive=True))
                    for e in exts)
        if n_img == 0:
            return False, f"nenhuma imagem em '{p_img}' (nem nas subpastas)"
        return True, f"OK — {n_img} imagens encontradas"
    if mode in ("spc", "sp", "gcms"):
        p = cfg.input_folder
        if not p or not os.path.isdir(p):
            return False, f"pasta nao encontrada: '{p}' (confira o caminho)"
        ext = {"spc": ".spc", "sp": ".sp", "gcms": ".cdf"}[mode]
        # em sistema de arquivos que ignora maiuscula (Windows/macOS) os dois padroes acham os MESMOS arquivos: conta distintos
        n = len({os.path.normcase(f) for e in (ext, ext.upper())
                 for f in glob.glob(os.path.join(p, "**", f"*{e}"), recursive=True)})
        if n == 0:
            return False, f"nenhum arquivo {ext} em '{p}' (nem nas subpastas)"
        return True, f"OK — {n} arquivos {ext} encontrados"
    if mode == "opus":
        p = cfg.input_folder
        if not p or not os.path.isdir(p):
            return False, f"pasta nao encontrada: '{p}' (confira o caminho)"
        if not os.listdir(p):
            return False, f"pasta '{p}' vazia"
        return True, (f"OK — pasta '{os.path.basename(p) or p}' com arquivos "
                       f"(OPUS nao tem extensao fixa filtravel; validado no carregamento)")
    if mode == "rmn":
        p = cfg.input_folder
        if not p or not os.path.isdir(p):
            return False, f"pasta nao encontrada: '{p}' (confira o caminho)"
        n = len(glob.glob(os.path.join(p, "**", "pdata"), recursive=True))
        if n == 0:
            return False, f"nenhum diretorio 'pdata' em '{p}' (esperado por amostra RMN)"
        return True, f"OK — {n} pasta(s)-amostra RMN (com 'pdata') encontrada(s)"
    if mode == "hplc":
        p = cfg.input_folder
        if not p or not os.path.isdir(p):
            return False, f"pasta nao encontrada: '{p}' (confira o caminho)"
        n = len([d for d in glob.glob(os.path.join(p, "**", "*"), recursive=True)
                 if os.path.isdir(d) and d.lower().endswith((".d", ".raw"))])
        if n == 0:
            return False, f"nenhuma pasta '.D'/'.raw' em '{p}' (esperado por amostra HPLC/GC)"
        return True, f"OK — {n} pasta(s)-amostra HPLC/GC encontrada(s)"
    if mode == "eem":
        p = cfg.input_folder
        if not p or not os.path.isdir(p):
            return False, f"pasta nao encontrada: '{p}' (confira o caminho)"
        # recursivo: o layout documentado e' 1 subpasta por classe (o leitor
        # `leitores_avancados._leitor_eem` tambem busca em subpastas).
        n = len(glob.glob(os.path.join(p, "**", "*.dat"), recursive=True))
        if n == 0:
            return False, f"nenhum arquivo .dat em '{p}' (nem nas subpastas)"
        return True, f"OK — {n} arquivos .dat (EEM) encontrados"
    # mode dx (padrao)
    p = cfg.input_folder
    if not p or not os.path.isdir(p):
        return False, f"pasta nao encontrada: '{p}' (confira o caminho)"
    n_dx = len(glob.glob(os.path.join(p, "**", "*.dx"), recursive=True))
    if n_dx == 0:
        return False, f"nenhum arquivo .dx em '{p}' (nem nas subpastas)"
    return True, f"OK — {n_dx} arquivos .dx encontrados"

# m-22: nomes PUBLICOS para o que as interfaces (CLI e web)
# usam. Os nomes com "_" continuam (testes e codigo antigo), mas a UI nao deve
# depender de simbolo privado.
CONFIG_SPEC = _CONFIG_SPEC
attr_para_yaml = _attr_para_yaml
coagir_valor = _coagir_valor
validar_semantico = _validar_semantico
validar_fonte_de_dados = _validar_pasta_dados
