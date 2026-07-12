"""cli_assistente.py — Modulo de dados/i18n compartilhado do GUARACI.

ATENCAO: este modulo NAO e mais um ponto de entrada executavel. A CLI
interativa completa (menus, wizard, execucao do pipeline) foi UNIFICADA em
guaraci.py (item 16 da auditoria: existiam DUAS implementacoes paralelas do
mesmo fluxo -- menu_projeto/menu_dados/.../wizard_inicial/main aqui E em
guaraci.py). guaraci.py absorveu a interface; este arquivo agora contem
apenas os DADOS que guaraci.py consome (via a funcao `_try()` em guaraci.py,
que resolve simbolos deste modulo por nome):

    dicionarios de i18n/rotulos (FIELD_NAMES, HELP_DB), classificacao de
    risco (RISK_CLASS), perfis prontos (PROFILES/PROFILE_DESC/
    PROFILE_KEY_SUMMARY), paletas/fontes (PALETAS_COR/FONT_PRESETS),
    referencias cientificas (REFERENCIAS_GUARACI), tecnicas analiticas
    (TECNICAS), layout de menu (MENU_FIELDS), rotulos de DD-SIMCA
    (_DDSIMCA_DISPLAY/_DDSIMCA_INPUT), e os re-exports de pipeline.py
    (_CONFIG_SPEC, _SPEC_BY_KEY, _attr_para_yaml, _coagir_valor, _fmt_yaml,
    carregar_config, salvar_config) que guaraci.py tambem resolve por aqui.

Uso:
    python -m guaraci.guaraci   (unico ponto de entrada da CLI interativa)

Requer: pipeline.py no mesmo pacote.
"""

from __future__ import annotations


from typing import Any, Dict

import guaraci.pipeline as pq

_CONFIG_SPEC = pq._CONFIG_SPEC

salvar_config = pq.salvar_config

carregar_config = pq.carregar_config

_attr_para_yaml = pq._attr_para_yaml

_fmt_yaml = pq._fmt_yaml

_coagir_valor = pq._coagir_valor


RISK_CLASS: Dict[str, str] = {
    # VISUAL
    "dpi": "VISUAL", "formato_figura": "VISUAL", "figuras_detalhadas": "VISUAL",
    "figuras_mostrar_marcadores": "VISUAL", "figuras_mostrar_elipses": "VISUAL",
    "abrir_figuras_na_tela": "VISUAL", "tag": "VISUAL", "nome_execucao": "VISUAL",
    # ANALITICO
    "pre_processamento": "ANALITICO", "max_lvs": "ANALITICO",
    "n_permutacoes": "ANALITICO", "n_jobs_permutacao": "ANALITICO",
    "holdout_fracao": "ANALITICO",
    "nivel": "ANALITICO", "objetivo": "ANALITICO", "excluir_classes": "ANALITICO",
    "faixa_min_cm": "ANALITICO", "faixa_max_cm": "ANALITICO",
    "modo_ddsimca": "ANALITICO", "ddsimca": "ANALITICO",
    "opls_da": "ANALITICO", "selecao_variaveis_etapa4": "ANALITICO",
    "selecao_spa": "ANALITICO", "selecao_ag": "ANALITICO",
    "comparar_pre_processamentos": "ANALITICO",
    "validacao_group_aware": "ANALITICO", "teste_wold": "ANALITICO",
    "teste_cv_anova": "ANALITICO", "teste_martens": "ANALITICO",
    "pasta_dados": "ANALITICO",
    "pasta_saida": "ANALITICO", "modo_entrada": "ANALITICO",
    "arquivo_csv": "ANALITICO", "coluna_classe": "ANALITICO",
    "coluna_concentracao": "ANALITICO", "imagem_incluir_textura": "ANALITICO",
    # AVANCADO
    "benchmark": "AVANCADO", "monte_carlo": "AVANCADO",
    "shap_benchmark": "AVANCADO", "n_monte_carlo": "AVANCADO",
    "shap_max_amostras": "AVANCADO", "monte_carlo_incluir_todos": "AVANCADO",
    "benchmark_regressao": "AVANCADO",
}

FIELD_NAMES: Dict[str, Dict[str, str]] = {
    "pasta_dados":                  {"PT": "Pasta de entrada",        "EN": "Input folder"},
    "pasta_saida":                  {"PT": "Pasta de saida",          "EN": "Output folder"},
    "tag":                          {"PT": "Sufixo da pasta saida",   "EN": "Output folder tag"},
    "modo_entrada":                 {"PT": "Modo de entrada",         "EN": "Input mode"},
    "arquivo_csv":                  {"PT": "Arquivo CSV",             "EN": "CSV file"},
    "coluna_classe":                {"PT": "Coluna de classe",        "EN": "Class column"},
    "coluna_concentracao":          {"PT": "Coluna concentracao",     "EN": "Concentration column"},
    "imagem_incluir_textura":       {"PT": "Textura (imagem)",        "EN": "Texture (image)"},
    "faixa_min_cm":                 {"PT": "Faixa minima (cm-1)",     "EN": "Min range (cm-1)"},
    "faixa_max_cm":                 {"PT": "Faixa maxima (cm-1)",     "EN": "Max range (cm-1)"},
    "excluir_classes":              {"PT": "Excluir classes",         "EN": "Exclude classes"},
    "pre_processamento":            {"PT": "Pre-processamento",       "EN": "Preprocessing"},
    "comparar_pre_processamentos":  {"PT": "Comparar pre-proc.",      "EN": "Compare preproc."},
    "nivel":                        {"PT": "Nivel de analise",        "EN": "Analysis level"},
    "objetivo":                     {"PT": "Objetivo cientifico",     "EN": "Scientific objective"},
    "max_lvs":                      {"PT": "Maximo de LVs",           "EN": "Max LVs"},
    "opls_da":                      {"PT": "OPLS-DA",                 "EN": "OPLS-DA"},
    "ddsimca":                      {"PT": "DD-SIMCA",                "EN": "DD-SIMCA"},
    "modo_ddsimca":                 {"PT": "Modo de treino (DD-SIMCA)", "EN": "Training mode (DD-SIMCA)"},
    "selecao_variaveis_etapa4":     {"PT": "Selecao de variaveis",    "EN": "Variable selection"},
    "selecao_spa":                  {"PT": "SPA (APS)",               "EN": "SPA (successive proj.)"},
    "selecao_ag":                   {"PT": "AG (Genetico)",           "EN": "GA (genetic algorithm)"},
    "holdout_fracao":               {"PT": "Fracao holdout",          "EN": "Holdout fraction"},
    "validacao_group_aware":        {"PT": "Validacao group-aware",   "EN": "Group-aware CV"},
    "n_permutacoes":                {"PT": "N. permutacoes",          "EN": "N permutations"},
    "n_jobs_permutacao":            {"PT": "Processos paralelos",     "EN": "Parallel processes"},
    "teste_wold":                   {"PT": "Teste de Wold",           "EN": "Wold test"},
    "teste_cv_anova":               {"PT": "CV-ANOVA",                "EN": "CV-ANOVA"},
    "teste_martens":                {"PT": "Teste de Martens",        "EN": "Martens test"},
    "benchmark":                    {"PT": "Benchmark",               "EN": "Benchmark"},
    "benchmark_regressao":          {"PT": "Benchmark de regressao",  "EN": "Regression benchmark"},
    "monte_carlo":                  {"PT": "Monte Carlo CV",          "EN": "Monte Carlo CV"},
    "n_monte_carlo":                {"PT": "N. repeticoes MC",        "EN": "N MC repetitions"},
    "monte_carlo_incluir_todos":    {"PT": "MC incluir todos",        "EN": "MC include all"},
    "shap_benchmark":               {"PT": "SHAP values",             "EN": "SHAP values"},
    "shap_max_amostras":            {"PT": "SHAP max. amostras",      "EN": "SHAP max samples"},
    "figuras_detalhadas":           {"PT": "Figuras detalhadas",      "EN": "Detailed figures"},
    "figuras_mostrar_marcadores":   {"PT": "Marcadores por classe",   "EN": "Class markers"},
    "figuras_mostrar_elipses":      {"PT": "Elipses de confianca",    "EN": "Confidence ellipses"},
    "formato_figura":               {"PT": "Formato das figuras",     "EN": "Figure format"},
    "dpi":                          {"PT": "Resolucao (DPI)",         "EN": "Resolution (DPI)"},
    "abrir_figuras_na_tela":        {"PT": "Abrir figuras na tela",   "EN": "Open figures on screen"},
    "nome_execucao":                {"PT": "Nome da execucao",         "EN": "Run name"},
}

_DDSIMCA_DISPLAY: Dict[str, Dict[str, str]] = {
    "PT": {"puros": "somente puras (autenticacao)",
           "todos": "todas as amostras (exploratorio)"},
    "EN": {"puros": "only pure (authentication)",
           "todos": "all samples (exploratory)"},
}

_DDSIMCA_INPUT: Dict[str, str] = {
    "somente puras (autenticacao)": "puros", "only pure (authentication)": "puros",
    "autenticacao": "puros", "authentication": "puros", "puros": "puros",
    "todas as amostras (exploratorio)": "todos", "all samples (exploratory)": "todos",
    "exploratorio": "todos", "exploratory": "todos", "todos": "todos",
}

PALETAS_COR: Dict[str, Dict[str, Any]] = {
    "qualitativo": {
        "PT": {
            "nome": "Qualitativo (padrao matplotlib)",
            "desc": "Paleta padrao do matplotlib. Boa para apresentacoes gerais.",
        },
        "EN": {
            "nome": "Qualitative (matplotlib default)",
            "desc": "Default matplotlib palette. Good for general presentations.",
        },
        "cores": None,
    },
    "daltonismo_safe": {
        "PT": {
            "nome": "Seguro para Daltonismo (Wong 2011)",
            "desc": "8 cores distinguiveis por pessoas com deuteranopia, protanopia e tritanopia.",
        },
        "EN": {
            "nome": "Colorblind Safe (Wong 2011)",
            "desc": "8 colors distinguishable for people with deuteranopia, protanopia and tritanopia.",
        },
        "cores": ["#000000", "#E69F00", "#56B4E9", "#009E73",
                  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
    },
    "cinza": {
        "PT": {
            "nome": "Escala de Cinza",
            "desc": "Para publicacoes em preto e branco. Distingue por intensidade.",
        },
        "EN": {
            "nome": "Grayscale",
            "desc": "For black and white publications. Distinguishes by intensity.",
        },
        "cores": ["#000000", "#333333", "#555555", "#777777",
                  "#999999", "#bbbbbb", "#dddddd", "#eeeeee"],
    },
    "viridis": {
        "PT": {
            "nome": "Viridis (perceptualmente uniforme)",
            "desc": "Paleta sequencial perceptualmente uniforme. Robusta para daltonismo.",
        },
        "EN": {
            "nome": "Viridis (perceptually uniform)",
            "desc": "Perceptually uniform sequential palette. Robust for colorblindness.",
        },
        "cores": None,
        "cmap": "viridis",
    },
    "publicacao": {
        "PT": {
            "nome": "Publicacao Cientifica (Nature/Elsevier)",
            "desc": "Cores usadas em revistas como Nature, Food Chemistry e Talanta.",
        },
        "EN": {
            "nome": "Scientific Publication (Nature/Elsevier)",
            "desc": "Colors used in journals like Nature, Food Chemistry and Talanta.",
        },
        "cores": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    },
    "escuro": {
        "PT": {
            "nome": "Tema Escuro (fundo preto)",
            "desc": "Otimizado para fundos escuros. Ideal para apresentacoes em tela.",
        },
        "EN": {
            "nome": "Dark Theme (black background)",
            "desc": "Optimized for dark backgrounds. Ideal for screen presentations.",
        },
        "cores": ["#00bfff", "#ff6347", "#7fff00", "#ffd700",
                  "#da70d6", "#ff8c00", "#40e0d0", "#ff69b4"],
        "style": "dark_background",
    },
}

FONT_PRESETS: Dict[str, Dict[str, Any]] = {
    "xs": {"font.size": 8,  "axes.titlesize": 9,  "axes.labelsize": 8,
           "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7},
    "s":  {"font.size": 9,  "axes.titlesize": 10, "axes.labelsize": 9,
           "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8},
    "m":  {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
           "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9},
    "l":  {"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
           "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10},
    "xl": {"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
           "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11},
}

HELP_DB: Dict[str, Dict[str, Any]] = {
    "dpi": {
        "PT": {
            "desc": "Resolucao das figuras em pontos por polegada.",
            "impacto": "VISUAL — nao altera resultados analiticos.",
            "exemplos": {"300": "Apresentacoes/slides", "600": "Artigos cientificos", "1200": "Impressao profissional"},
        },
        "EN": {
            "desc": "Figure resolution in dots per inch.",
            "impacto": "VISUAL — does not affect analytical results.",
            "exemplos": {"300": "Presentations/slides", "600": "Scientific articles", "1200": "Professional printing"},
        },
        "default": 600, "range": "150-1200",
    },
    "max_lvs": {
        "PT": {
            "desc": "Numero maximo de Variaveis Latentes testadas na selecao automatica.",
            "impacto": "ANALITICO — afeta diretamente o modelo PLS-DA.",
            "exemplos": {"10": "Datasets pequenos", "40": "Recomendado", "80": "Datasets grandes"},
        },
        "EN": {
            "desc": "Maximum number of Latent Variables tested in automatic selection.",
            "impacto": "ANALYTICAL — directly affects the PLS-DA model.",
            "exemplos": {"10": "Small datasets", "40": "Recommended", "80": "Large datasets"},
        },
        "default": 40, "range": "5-80",
    },
    "n_permutacoes": {
        "PT": {
            "desc": "Numero de permutacoes para o teste Y-randomization.",
            "impacto": "ANALITICO — afeta o p-value do teste de permutacao.",
            "exemplos": {"50": "Diagnostico rapido", "200": "Publicacao", "1000": "Alta precisao"},
        },
        "EN": {
            "desc": "Number of permutations for the Y-randomization test.",
            "impacto": "ANALYTICAL — affects the permutation test p-value.",
            "exemplos": {"50": "Quick diagnosis", "200": "Publication", "1000": "High precision"},
        },
        "default": 200, "range": "50-1000",
    },
    "pre_processamento": {
        "PT": {
            "desc": "Pipeline de pre-processamento espectral aplicado antes da modelagem quimiometrica.",
            "impacto": "ANALITICO — o pre-processamento correto e essencial para resultados validos em FT-NIR.",
            "exemplos": {
                "msc_sg_mc": "MSC + Savitzky-Golay + Mean-Centering (recomendado para oleos vegetais)",
                "snv_sg_mc": "SNV + Savitzky-Golay + Mean-Centering (alternativa robusta)",
                "autoscaling": "Autoscaling — linha de base simples, nao recomendado para FT-NIR",
            },
        },
        "EN": {
            "desc": "Spectral preprocessing pipeline applied before chemometric modelling.",
            "impacto": "ANALYTICAL — correct preprocessing is essential for valid FT-NIR results.",
            "exemplos": {
                "msc_sg_mc": "MSC + Savitzky-Golay + Mean-Centering (recommended for vegetable oils)",
                "snv_sg_mc": "SNV + Savitzky-Golay + Mean-Centering (robust alternative)",
                "autoscaling": "Autoscaling — simple baseline, not recommended for FT-NIR",
            },
        },
        "default": "MSC+SG+MC", "range": "Escolha na lista / Choose from list",
    },
    "modo_ddsimca": {
        "PT": {
            "desc": "Define COMO o DD-SIMCA treina: 'somente puras' usa so amostras de referencia "
                    "pura (o resto da classe conta como contaminante/adulterado -- autenticacao de "
                    "verdade); 'todas as amostras' treina com a classe inteira (exploratorio, mais "
                    "robusto quando ha poucas puras, porem menos rigoroso).",
            "impacto": "ANALITICO — muda o que o modelo aprende como 'normal' para cada classe.",
            "exemplos": {"somente puras (autenticacao)": "Validacao de autenticidade (recomendado para publicacao)",
                        "todas as amostras (exploratorio)": "Analise exploratoria inicial, poucas amostras puras"},
        },
        "EN": {
            "desc": "Defines HOW DD-SIMCA trains: 'only pure' uses only pure reference samples "
                    "(the rest of the class counts as contaminant/adulterated -- true authentication); "
                    "'all samples' trains on the whole class (exploratory, more robust with few pure "
                    "samples, but less rigorous).",
            "impacto": "ANALYTICAL — changes what the model learns as 'normal' for each class.",
            "exemplos": {"only pure (authentication)": "Authenticity validation (recommended for publication)",
                        "all samples (exploratory)": "Initial exploratory analysis, few pure samples"},
        },
        "default": "somente puras (autenticacao)",
        "range": "somente puras | todas as amostras  (EN: only pure | all samples)",
    },
    "nivel": {
        "PT": {
            "desc": "Modo de analise: Classificacao por especie (identifica a "
                    "especie; codigo interno N1), Discriminacao puro/adulterado "
                    "(autenticacao com DD-SIMCA; codigo interno N2) ou "
                    "Quantificacao (estima o teor de adulterante, PLS-R; "
                    "codigo interno N3).",
            "impacto": "ANALITICO — define quais modulos sao executados.",
            "exemplos": {
                "N1": "Classificacao por especie (PLS-DA/OPLS-DA)",
                "N2": "Puro vs. adulterado (PLS-DA + DD-SIMCA one-class)",
                "N3": "Quantificacao do teor de adulterante (PLS-R)"},
        },
        "EN": {
            "desc": "Analysis mode: Species classification (identifies the "
                    "species; internal code N1), Pure/adulterated "
                    "discrimination (authentication with DD-SIMCA; internal "
                    "code N2), or Quantification (estimates adulterant "
                    "content, PLS-R; internal code N3).",
            "impacto": "ANALYTICAL — defines which modules are executed.",
            "exemplos": {
                "N1": "Species classification (PLS-DA/OPLS-DA)",
                "N2": "Pure vs. adulterated (PLS-DA + DD-SIMCA one-class)",
                "N3": "Adulterant content quantification (PLS-R)"},
        },
        "default": "N2", "range": "N1 | N2 | N3",
    },
    "objetivo": {
        "PT": {
            "desc": "Objetivo cientifico do run — FILTRA quais figuras e "
                    "relatorios sao gerados, para que cada modo produza apenas "
                    "o pertinente. 'auto' deriva do modo de analise "
                    "(Classificacao/Discriminacao -> Classificacao, "
                    "Quantificacao -> Quantificacao) e preserva o "
                    "comportamento historico.",
            "impacto": "ANALITICO — controla as figuras/relatorios de saida.",
            "exemplos": {
                "auto": "Deriva do nivel (recomendado)",
                "exploratorio": "So PCA/HCA/loadings/pre-proc (sem PLS-DA)",
                "classificacao": "PLS-DA, confusao, ROC, VIP, DD-SIMCA",
                "quantificacao": "Regressao PLS + figuras de merito analiticas"},
        },
        "EN": {
            "desc": "Scientific objective of the run — FILTERS which figures "
                    "and reports are generated, so each mode produces only what "
                    "is pertinent. 'auto' derives from the analysis mode "
                    "(Classification/Discrimination -> Classification, "
                    "Quantification -> Quantification), preserving historical "
                    "behavior.",
            "impacto": "ANALYTICAL — controls the output figures/reports.",
            "exemplos": {
                "auto": "Derive from level (recommended)",
                "exploratorio": "Only PCA/HCA/loadings/preproc (no PLS-DA)",
                "classificacao": "PLS-DA, confusion, ROC, VIP, DD-SIMCA",
                "quantificacao": "PLS regression + analytical figures of merit"},
        },
        "default": "auto",
        "range": "auto | exploratorio | classificacao | quantificacao",
    },
    "holdout_fracao": {
        "PT": {
            "desc": "Fracao do dataset reservada para validacao externa independente (holdout).",
            "impacto": "ANALITICO — afeta a estimativa de generalizacao do modelo.",
            "exemplos": {"0.1": "Poucos dados disponiveis", "0.2": "Recomendado (padrao)", "0.3": "Dataset grande"},
        },
        "EN": {
            "desc": "Fraction of the dataset held out for independent external validation.",
            "impacto": "ANALYTICAL — affects the model generalization estimate.",
            "exemplos": {"0.1": "Limited data available", "0.2": "Recommended (default)", "0.3": "Large dataset"},
        },
        "default": 0.2, "range": "0.1-0.3",
    },
    "ddsimca": {
        "PT": {
            "desc": "Ativa o DD-SIMCA para autenticacao e deteccao de amostras desconhecidas por classe.",
            "impacto": "ANALITICO — adiciona figuras de Cooman's Plot e metricas de autenticacao.",
            "exemplos": {"true": "Recomendado para publicacao", "false": "Economiza tempo em exploracao"},
        },
        "EN": {
            "desc": "Enables DD-SIMCA for per-class sample authentication and unknown detection.",
            "impacto": "ANALYTICAL — adds Cooman's Plot figures and authentication metrics.",
            "exemplos": {"true": "Recommended for publication", "false": "Saves time in exploration"},
        },
        "default": True, "range": "true | false",
    },
    "benchmark": {
        "PT": {
            "desc": "Compara PLS-DA contra SVM RBF, Random Forest e XGBoost com mesma CV group-aware.",
            "impacto": "AVANCADO — adiciona ~30-60 minutos de processamento.",
            "exemplos": {"false": "TCC/producao (recomendado)", "true": "Artigo com comparacao de classificadores"},
        },
        "EN": {
            "desc": "Compares PLS-DA against SVM RBF, Random Forest and XGBoost with same group-aware CV.",
            "impacto": "ADVANCED — adds ~30-60 minutes of processing time.",
            "exemplos": {"false": "TCC/production (recommended)", "true": "Article with classifier comparison"},
        },
        "default": False, "range": "true | false",
    },
    "benchmark_regressao": {
        "PT": {
            "desc": "Compara PLS-R (ja calibrado por especie) contra Ridge, Lasso, Elastic Net, "
                    "SVR e Random Forest -- mesmo split cal/val e pre-processamento, por especie.",
            "impacto": "AVANCADO — so' roda em Discriminacao/Quantificacao (N2/N3) com regressao multi-especie ja calculada.",
            "exemplos": {"false": "TCC/producao (recomendado)", "true": "Artigo com comparacao de regressores"},
        },
        "EN": {
            "desc": "Compares PLS-R (already calibrated per species) against Ridge, Lasso, Elastic Net, "
                    "SVR and Random Forest -- same cal/val split and preprocessing, per species.",
            "impacto": "ADVANCED — only runs on Discrimination/Quantification (N2/N3) with multi-species regression already computed.",
            "exemplos": {"false": "TCC/production (recommended)", "true": "Article with regressor comparison"},
        },
        "default": False, "range": "true | false",
    },
    "formato_figura": {
        "PT": {
            "desc": "Formato de saida das figuras geradas pelo pipeline.",
            "impacto": "VISUAL — nao altera analise ou resultados.",
            "exemplos": {"png": "Apresentacoes/TCC (menor tamanho)", "pdf": "Artigos (vetorial, editavel)", "svg": "Edicao pos-processamento"},
        },
        "EN": {
            "desc": "Output format for pipeline-generated figures.",
            "impacto": "VISUAL — does not affect analysis or results.",
            "exemplos": {"png": "Presentations/TCC (smaller size)", "pdf": "Articles (vector, editable)", "svg": "Post-processing editing"},
        },
        "default": "png", "range": "png | pdf | svg",
    },
    "monte_carlo": {
        "PT": {
            "desc": "Ativa Monte Carlo CV: calcula IC95% das metricas por reamostragem estratificada por grupo.",
            "impacto": "AVANCADO — aumenta significativamente o tempo de processamento.",
            "exemplos": {"false": "Producao/TCC (recomendado)", "true": "Dissertacao/Tese"},
        },
        "EN": {
            "desc": "Enables Monte Carlo CV: computes 95% CI for metrics via stratified group resampling.",
            "impacto": "ADVANCED — significantly increases processing time.",
            "exemplos": {"false": "Production/TCC (recommended)", "true": "Dissertation/Thesis"},
        },
        "default": False, "range": "true | false",
    },
    "n_monte_carlo": {
        "PT": {
            "desc": "Numero de repeticoes do Monte Carlo CV.",
            "impacto": "AVANCADO — mais repeticoes = IC mais estreito, mas mais tempo.",
            "exemplos": {"50": "Teste rapido", "100": "TCC", "200": "Dissertacao/Tese"},
        },
        "EN": {
            "desc": "Number of Monte Carlo CV repetitions.",
            "impacto": "ADVANCED — more repetitions = narrower CI, but more time.",
            "exemplos": {"50": "Quick test", "100": "TCC", "200": "Dissertation/Thesis"},
        },
        "default": 100, "range": "50-500",
    },
    "shap_benchmark": {
        "PT": {
            "desc": "Calcula SHAP values (TreeExplainer) para RF/XGBoost — interpretabilidade espectral.",
            "impacto": "AVANCADO — requer benchmark=true; adiciona ~10-20 minutos.",
            "exemplos": {"false": "Sem SHAP (producao)", "true": "Artigo com interpretabilidade espectral"},
        },
        "EN": {
            "desc": "Computes SHAP values (TreeExplainer) for RF/XGBoost — spectral interpretability.",
            "impacto": "ADVANCED — requires benchmark=true; adds ~10-20 minutes.",
            "exemplos": {"false": "No SHAP (production)", "true": "Article with spectral interpretability"},
        },
        "default": False, "range": "true | false",
    },
    "shap_max_amostras": {
        "PT": {
            "desc": "Limite de amostras para calculo de SHAP (controla uso de memoria RAM).",
            "impacto": "AVANCADO — valores maiores aumentam tempo e uso de RAM.",
            "exemplos": {"200": "RAM limitada (<8 GB)", "500": "Recomendado (padrao)", "1000": "RAM abundante (>32 GB)"},
        },
        "EN": {
            "desc": "Sample limit for SHAP computation (controls RAM usage).",
            "impacto": "ADVANCED — larger values increase time and RAM usage.",
            "exemplos": {"200": "Limited RAM (<8 GB)", "500": "Recommended (default)", "1000": "Abundant RAM (>32 GB)"},
        },
        "default": 500, "range": "100-1000",
    },
    "validacao_group_aware": {
        "PT": {
            "desc": "Manter replicas (T1/T2/T3) sempre juntas nos folds de validacao cruzada.",
            "impacto": "ANALITICO — false pode inflar artificialmente as metricas (data leakage).",
            "exemplos": {"true": "Obrigatorio para dados com replicas fisicas", "false": "Apenas para amostras 100% independentes"},
        },
        "EN": {
            "desc": "Keep replicates (T1/T2/T3) together in all cross-validation folds.",
            "impacto": "ANALYTICAL — false may artificially inflate metrics (data leakage).",
            "exemplos": {"true": "Mandatory for data with physical replicates", "false": "Only for fully independent samples"},
        },
        "default": True, "range": "true | false",
    },
    "opls_da": {
        "PT": {
            "desc": "Executa OPLS-DA (Analise Discriminante por Minimos Quadrados Parciais Ortogonais).",
            "impacto": "ANALITICO — gera S-Plot e separa variacao ortogonal da preditiva.",
            "exemplos": {"true": "Recomendado para publicacao", "false": "Analise exploratoria rapida"},
        },
        "EN": {
            "desc": "Runs OPLS-DA (Orthogonal Partial Least Squares Discriminant Analysis).",
            "impacto": "ANALYTICAL — generates S-Plot and separates orthogonal from predictive variation.",
            "exemplos": {"true": "Recommended for publication", "false": "Quick exploratory analysis"},
        },
        "default": True, "range": "true | false",
    },
    "selecao_variaveis_etapa4": {
        "PT": {
            "desc": "Executa selecao de variaveis: iPLS, VIP>=1, SR top 20% e sPLS-DA.",
            "impacto": "ANALITICO — identifica regioes espectrais mais relevantes para o modelo.",
            "exemplos": {"true": "Publicacao (recomendado)", "false": "Exploracao rapida"},
        },
        "EN": {
            "desc": "Runs variable selection: iPLS, VIP>=1, SR top 20% and sPLS-DA.",
            "impacto": "ANALYTICAL — identifies the most relevant spectral regions for the model.",
            "exemplos": {"true": "Publication (recommended)", "false": "Quick exploration"},
        },
        "default": True, "range": "true | false",
    },
    "selecao_spa": {
        "PT": {
            "desc": "Alem dos metodos acima, roda tambem SPA/APS (Algoritmo das "
                    "Projecoes Sucessivas, Araujo et al. 2001) — constroi cadeias "
                    "de variaveis com baixa colinearidade a partir de varios "
                    "pontos de partida do espectro.",
            "impacto": "ANALITICO — mais lento que iPLS/VIP/SR/sPLS-DA (varias "
                       "avaliacoes de CV por ponto de partida).",
            "exemplos": {"false": "Exploracao rapida (recomendado)", "true": "Publicacao"},
        },
        "EN": {
            "desc": "In addition to the methods above, also runs SPA (Successive "
                    "Projections Algorithm, Araujo et al. 2001) — builds "
                    "low-collinearity variable chains from several spectral "
                    "starting points.",
            "impacto": "ANALYTICAL — slower than iPLS/VIP/SR/sPLS-DA (several CV "
                       "evaluations per starting point).",
            "exemplos": {"false": "Quick exploration (recommended)", "true": "Publication"},
        },
        "default": False, "range": "true | false",
    },
    "selecao_ag": {
        "PT": {
            "desc": "Alem dos metodos acima, roda tambem AG (Algoritmo Genetico, "
                    "GA-PLS) — populacao de subconjuntos de variaveis evoluida "
                    "por selecao/crossover/mutacao, fitness = acuracia via CV.",
            "impacto": "ANALITICO — o mais lento dos metodos de selecao "
                       "(populacao x geracoes avaliacoes de CV).",
            "exemplos": {"false": "Exploracao rapida (recomendado)", "true": "Publicacao"},
        },
        "EN": {
            "desc": "In addition to the methods above, also runs GA (Genetic "
                    "Algorithm, GA-PLS) — a population of variable subsets "
                    "evolved via selection/crossover/mutation, fitness = CV "
                    "accuracy.",
            "impacto": "ANALYTICAL — the slowest selection method (population x "
                       "generations CV evaluations).",
            "exemplos": {"false": "Quick exploration (recommended)", "true": "Publication"},
        },
        "default": False, "range": "true | false",
    },
    "comparar_pre_processamentos": {
        "PT": {
            "desc": "Compara automaticamente varios pipelines de pre-processamento e reporta o melhor.",
            "impacto": "ANALITICO — aumenta o tempo de execucao; util para otimizar o pipeline.",
            "exemplos": {"false": "Usar pre-processamento padrao", "true": "Otimizacao do pipeline espectral"},
        },
        "EN": {
            "desc": "Automatically compares several preprocessing pipelines and reports the best.",
            "impacto": "ANALYTICAL — increases execution time; useful to optimize the pipeline.",
            "exemplos": {"false": "Use default preprocessing", "true": "Spectral pipeline optimization"},
        },
        "default": False, "range": "true | false",
    },
    "pasta_dados": {
        "PT": {
            "desc": "Pasta com arquivos .dx (JCAMP-DX). Suporta FT-NIR, NIR, MIR, Raman.",
            "impacto": "ANALITICO — define os dados de entrada do pipeline.",
            "exemplos": {"dados": "Pasta padrao do projeto", r"C:\meus_dados\oleos": "Caminho absoluto personalizado"},
        },
        "EN": {
            "desc": "Folder with .dx (JCAMP-DX) files. Supports FT-NIR, NIR, MIR, Raman.",
            "impacto": "ANALYTICAL — defines the pipeline input data.",
            "exemplos": {"dados": "Default project folder", r"C:\meus_dados\oleos": "Custom absolute path"},
        },
        "default": "dados", "range": "Valid system path",
    },
    "pasta_saida": {
        "PT": {
            "desc": "Pasta onde os resultados (figuras, metricas, relatorios) serao gravados.",
            "impacto": "ANALITICO — define onde as saidas do pipeline sao salvas.",
            "exemplos": {"resultados": "Pasta padrao", r"C:\experimento_01": "Experimento com nome especifico"},
        },
        "EN": {
            "desc": "Folder where results (figures, metrics, reports) will be saved.",
            "impacto": "ANALYTICAL — defines where pipeline outputs are saved.",
            "exemplos": {"resultados": "Default folder", r"C:\experimento_01": "Specifically named experiment"},
        },
        "default": "resultados", "range": "Valid system path",
    },
    "modo_entrada": {
        "PT": {
            "desc": "Origem dos dados de entrada: dx (espectros JCAMP-DX) | csv | "
                    "imagem (colorimetria digital, prototipo) | sintetico (testes).",
            "impacto": "ANALITICO — define o formato de leitura e parsing dos dados.",
            "exemplos": {"dx": "Espectros JCAMP-DX (FT-NIR, Raman, MIR)", "csv": "Tabela generica com colunas espectrais",
                         "imagem": "Fotos (1 subpasta por classe) -> features de cor RGB/HSV/Lab",
                         "sintetico": "Dados simulados para teste do pipeline"},
        },
        "EN": {
            "desc": "Input data source: dx (JCAMP-DX spectra) | csv | "
                    "imagem (digital colorimetry, prototype) | synthetic (for testing).",
            "impacto": "ANALYTICAL — defines the data reading and parsing format.",
            "exemplos": {"dx": "JCAMP-DX spectra (FT-NIR, Raman, MIR)", "csv": "Generic table with spectral columns",
                         "imagem": "Photos (1 subfolder per class) -> RGB/HSV/Lab color features",
                         "sintetico": "Simulated data for pipeline testing"},
        },
        "default": "dx", "range": "dx | csv | imagem | sintetico",
    },
    "imagem_incluir_textura": {
        "PT": {
            "desc": "Modo imagem: alem das features de cor (media/desvio RGB+HSV+Lab), "
                    "inclui features de textura via GLCM (contraste, homogeneidade, "
                    "energia, correlacao). Requer 'pip install scikit-image'.",
            "impacto": "ANALITICO — mais variaveis, so tem efeito no modo_entrada='imagem'.",
            "exemplos": {"false": "So cor (recomendado, sem dependencia extra)", "true": "Cor + textura"},
        },
        "EN": {
            "desc": "Image mode: besides color features (RGB+HSV+Lab mean/std), "
                    "also includes GLCM texture features (contrast, homogeneity, "
                    "energy, correlation). Requires 'pip install scikit-image'.",
            "impacto": "ANALYTICAL — more variables, only affects modo_entrada='imagem'.",
            "exemplos": {"false": "Color only (recommended, no extra dependency)", "true": "Color + texture"},
        },
        "default": False, "range": "true | false",
    },
    "faixa_min_cm": {
        "PT": {
            "desc": "Limite inferior da faixa espectral analisada (numero de onda minimo, cm-1).",
            "impacto": "ANALITICO — regioes abaixo deste valor sao descartadas.",
            "exemplos": {"4000": "Padrao NIR (recomendado)", "5500": "Regiao de combinacoes apenas", "6000": "Sobretons superiores"},
        },
        "EN": {
            "desc": "Lower bound of the analyzed spectral range (minimum wavenumber, cm-1).",
            "impacto": "ANALYTICAL — regions below this value are discarded.",
            "exemplos": {"4000": "Standard NIR (recommended)", "5500": "Combination region only", "6000": "Upper overtones"},
        },
        "default": 4000.0, "range": "400-12000",
    },
    "faixa_max_cm": {
        "PT": {
            "desc": "Limite superior da faixa espectral analisada (numero de onda maximo, cm-1).",
            "impacto": "ANALITICO — regioes acima deste valor sao descartadas.",
            "exemplos": {"10000": "Padrao NIR (recomendado)", "7500": "NIR curto", "12000": "NIR completo"},
        },
        "EN": {
            "desc": "Upper bound of the analyzed spectral range (maximum wavenumber, cm-1).",
            "impacto": "ANALYTICAL — regions above this value are discarded.",
            "exemplos": {"10000": "Standard NIR (recommended)", "7500": "Short NIR", "12000": "Full NIR"},
        },
        "default": 10000.0, "range": "400-12000",
    },
    "excluir_classes": {
        "PT": {
            "desc": "Lista de especies/classes a remover da analise (por nome exato).",
            "impacto": "ANALITICO — altera o conjunto de treinamento e o numero de classes.",
            "exemplos": {"[]": "Usar todas as classes disponiveis", "[Copaiba]": "Remover Copaiba (outlier composicional)", "[Copaiba, Palmiste]": "Remover multiplas classes"},
        },
        "EN": {
            "desc": "List of species/classes to remove from analysis (by exact name).",
            "impacto": "ANALYTICAL — changes the training set and number of classes.",
            "exemplos": {"[]": "Use all available classes", "[Copaiba]": "Remove Copaiba (compositional outlier)", "[Copaiba, Palmiste]": "Remove multiple classes"},
        },
        "default": [], "range": "List of class names",
    },
    "teste_wold": {
        "PT": {
            "desc": "Executa o criterio de parcimonia de Wold para selecao automatica do numero otimo de LVs.",
            "impacto": "ANALITICO — define o numero de LVs usando tolerancia de 2% no RMSECV.",
            "exemplos": {"true": "Recomendado (evita overfitting)", "false": "Usar max_lvs diretamente"},
        },
        "EN": {
            "desc": "Runs Wold's parsimony criterion for automatic optimal number of LVs selection.",
            "impacto": "ANALYTICAL — defines LV count using 2% RMSECV tolerance.",
            "exemplos": {"true": "Recommended (prevents overfitting)", "false": "Use max_lvs directly"},
        },
        "default": True, "range": "true | false",
    },
    "teste_cv_anova": {
        "PT": {
            "desc": "Executa CV-ANOVA (Eriksson et al. 2008) para testar a significancia estatistica do modelo.",
            "impacto": "ANALITICO — gera F-statistic e p-value de significancia.",
            "exemplos": {"true": "Publicacao (obrigatorio)", "false": "Exploracao rapida"},
        },
        "EN": {
            "desc": "Runs CV-ANOVA (Eriksson et al. 2008) to test statistical significance of the model.",
            "impacto": "ANALYTICAL — generates F-statistic and significance p-value.",
            "exemplos": {"true": "Publication (mandatory)", "false": "Quick exploration"},
        },
        "default": True, "range": "true | false",
    },
    "teste_martens": {
        "PT": {
            "desc": "Jackknifing group-aware dos coeficientes PLS (Martens & Martens, 2000) -- "
                    "teste de hipotese formal (p-valor) de significancia por variavel, mais "
                    "rigoroso que VIP/Selectivity Ratio (medidas de magnitude).",
            "impacto": "ANALITICO — gera teste_martens.csv com t/p-valor por comprimento de onda.",
            "exemplos": {"true": "Artigo com selecao de variaveis rigorosa", "false": "Exploracao rapida"},
        },
        "EN": {
            "desc": "Group-aware jackknifing of PLS coefficients (Martens & Martens, 2000) -- "
                    "formal hypothesis test (p-value) of per-variable significance, more "
                    "rigorous than VIP/Selectivity Ratio (magnitude measures).",
            "impacto": "ANALYTICAL — generates teste_martens.csv with t/p-value per wavenumber.",
            "exemplos": {"true": "Article with rigorous variable selection", "false": "Quick exploration"},
        },
        "default": False, "range": "true | false",
    },
    "figuras_detalhadas": {
        "PT": {
            "desc": "Gerar tambem as figuras exploratorias/detalhadas (HCA, loadings PCA, "
                    "pre-processamento, contribuicao de score, DD-SIMCA por classe, Cooman).",
            "impacto": "VISUAL — mais arquivos e tempo de execucao; nao altera o modelo.",
            "exemplos": {"false": "So o essencial (mais rapido, recomendado)", "true": "Analise exploratoria completa"},
        },
        "EN": {
            "desc": "Also generate exploratory/detailed figures (HCA, PCA loadings, "
                    "preprocessing, score contribution, per-class DD-SIMCA, Cooman's plot).",
            "impacto": "VISUAL — more files and runtime; does not change the model.",
            "exemplos": {"false": "Essentials only (faster, recommended)", "true": "Full exploratory analysis"},
        },
        "default": False, "range": "true | false",
    },
    "figuras_mostrar_marcadores": {
        "PT": {
            "desc": "Usar formas diferentes (circulo, triangulo, quadrado) por classe nos graficos de score.",
            "impacto": "VISUAL — nao altera resultados analiticos.",
            "exemplos": {"true": "Graficos mais informativos e acessiveis", "false": "Graficos mais limpos e simples"},
        },
        "EN": {
            "desc": "Use different shapes (circle, triangle, square) per class in score plots.",
            "impacto": "VISUAL — does not affect analytical results.",
            "exemplos": {"true": "More informative and accessible plots", "false": "Cleaner, simpler plots"},
        },
        "default": True, "range": "true | false",
    },
    "figuras_mostrar_elipses": {
        "PT": {
            "desc": "Desenhar elipses de confianca de Hotelling T2 por grupo nos graficos de score.",
            "impacto": "VISUAL — nao altera resultados analiticos.",
            "exemplos": {"true": "Padrao para publicacao cientifica", "false": "Graficos mais simples"},
        },
        "EN": {
            "desc": "Draw Hotelling T2 confidence ellipses per group in score plots.",
            "impacto": "VISUAL — does not affect analytical results.",
            "exemplos": {"true": "Standard for scientific publication", "false": "Simpler plots"},
        },
        "default": True, "range": "true | false",
    },
    "abrir_figuras_na_tela": {
        "PT": {
            "desc": "Abrir cada figura na tela ao gerar (alem de salvar automaticamente em arquivo).",
            "impacto": "VISUAL — nao altera analise.",
            "exemplos": {"false": "Execucao automatizada sem interrupcoes", "true": "Revisao interativa das figuras"},
        },
        "EN": {
            "desc": "Open each figure on screen when generated (in addition to saving to file).",
            "impacto": "VISUAL — does not affect analysis.",
            "exemplos": {"false": "Automated execution without interruptions", "true": "Interactive figure review"},
        },
        "default": False, "range": "true | false",
    },
    "arquivo_csv": {
        "PT": {
            "desc": "Caminho do arquivo CSV (no modo csv): deve conter colunas espectrais + 1 coluna de classe.",
            "impacto": "ANALITICO — define os dados de entrada no modo CSV.",
            "exemplos": {"dados.csv": "Arquivo na pasta atual", r"C:\dados\amostras.csv": "Caminho absoluto"},
        },
        "EN": {
            "desc": "CSV file path (csv mode): must contain spectral columns + 1 class column.",
            "impacto": "ANALYTICAL — defines input data in CSV mode.",
            "exemplos": {"dados.csv": "File in current folder", r"C:\dados\amostras.csv": "Absolute path"},
        },
        "default": "", "range": "Valid .csv path",
    },
    "coluna_classe": {
        "PT": {
            "desc": "Nome exato da coluna de classe/rotulo no arquivo CSV.",
            "impacto": "ANALITICO — define o alvo de classificacao.",
            "exemplos": {"classe": "Nome padrao", "especie": "Nome alternativo", "label": "Nome em ingles"},
        },
        "EN": {
            "desc": "Exact name of the class/label column in the CSV file.",
            "impacto": "ANALYTICAL — defines the classification target.",
            "exemplos": {"classe": "Default name (Portuguese)", "especie": "Alternative name", "label": "English name"},
        },
        "default": "classe", "range": "Existing column name in CSV",
    },
    "coluna_concentracao": {
        "PT": {
            "desc": "Nome da coluna de concentracao no CSV (deixe vazio se nao houver).",
            "impacto": "ANALITICO — usado na Quantificacao (N3, regressao de adulterantes).",
            "exemplos": {"": "Sem concentracao — use para Classificacao/Discriminacao (N1/N2)",
                        "conc": "Com concentracao — use para Quantificacao (N3)"},
        },
        "EN": {
            "desc": "Name of the concentration column in CSV (leave empty if not present).",
            "impacto": "ANALYTICAL — used for Quantification (N3, adulterant regression).",
            "exemplos": {"": "No concentration — use for Classification/Discrimination (N1/N2)",
                        "conc": "With concentration — use for Quantification (N3)"},
        },
        "default": "", "range": "Column name or empty",
    },
    "monte_carlo_incluir_todos": {
        "PT": {
            "desc": "MC CV: incluir SVM RBF / RF / XGBoost alem do PLS-DA (muito mais lento).",
            "impacto": "AVANCADO — aumenta consideravelmente o tempo do Monte Carlo CV.",
            "exemplos": {"false": "Apenas PLS-DA (recomendado)", "true": "Comparacao completa de classificadores"},
        },
        "EN": {
            "desc": "MC CV: include SVM RBF / RF / XGBoost in addition to PLS-DA (much slower).",
            "impacto": "ADVANCED — considerably increases Monte Carlo CV time.",
            "exemplos": {"false": "PLS-DA only (recommended)", "true": "Full classifier comparison"},
        },
        "default": False, "range": "true | false",
    },
    "tag": {
        "PT": {
            "desc": "Sufixo personalizado para o nome da pasta de resultados.",
            "impacto": "VISUAL — nao altera analise, apenas organiza as saidas.",
            "exemplos": {
                "": "Pasta automatica: PLSDA_N2_MSC-SG1-MC_YYYYMMDD_HHMMSS",
                "experimento01": "Pasta: PLSDA_N2_..._experimento01",
                "artigo_final": "Pasta: PLSDA_N2_..._artigo_final",
            },
        },
        "EN": {
            "desc": "Custom suffix for the results folder name.",
            "impacto": "VISUAL — does not affect analysis, only organizes outputs.",
            "exemplos": {
                "": "Auto folder: PLSDA_N2_MSC-SG1-MC_YYYYMMDD_HHMMSS",
                "experiment01": "Folder: PLSDA_N2_..._experiment01",
                "final_paper": "Folder: PLSDA_N2_..._final_paper",
            },
        },
        "default": "", "range": "Texto livre (sem espacos recomendado)",
    },
    "nome_execucao": {
        "PT": {
            "desc": "Nome personalizado para a execucao atual. Aparece no nome da pasta de resultados.",
            "impacto": "VISUAL — nao afeta calculos. Organiza diferentes rodadas.",
            "exemplos": {
                "": "Nome automatico: PLSDA_N2_MSC-SG1-MC_YYYYMMDD_HHMMSS",
                "artigo_2026": "Pasta: ...artigo_2026",
                "dataset_completo": "Pasta: ...dataset_completo",
            },
        },
        "EN": {
            "desc": "Custom name for the current run. Appears in the results folder name.",
            "impacto": "VISUAL — does not affect calculations. Organizes different runs.",
            "exemplos": {
                "": "Auto name: PLSDA_N2_MSC-SG1-MC_YYYYMMDD_HHMMSS",
                "paper_2026": "Folder: ...paper_2026",
                "full_dataset": "Folder: ...full_dataset",
            },
        },
        "default": "", "range": "Texto livre (sem espacos)",
    },
    "paleta_cores": {
        "PT": {
            "desc": "Paleta de cores para as figuras geradas pelo pipeline.",
            "impacto": "VISUAL — nao afeta calculos. Aplica antes de rodar.",
            "exemplos": {
                "qualitativo": "Cores matplotlib padrao (multiuso)",
                "daltonismo_safe": "Seguro para daltonismo (Wong 2011)",
                "publicacao": "Cores de revistas cientificas",
                "escuro": "Tema escuro para apresentacoes",
            },
        },
        "EN": {
            "desc": "Color palette for pipeline-generated figures.",
            "impacto": "VISUAL — does not affect calculations. Applied before running.",
            "exemplos": {
                "qualitativo": "Default matplotlib colors (multipurpose)",
                "daltonismo_safe": "Colorblind safe (Wong 2011)",
                "publicacao": "Scientific journal colors",
                "escuro": "Dark theme for presentations",
            },
        },
        "default": "qualitativo", "range": "qualitativo | daltonismo_safe | cinza | viridis | publicacao | escuro",
    },
    "codificacao_arquivos": {
        "PT": {
            "desc": "Formato de nomenclatura dos arquivos DX para leitura pelo pipeline.",
            "impacto": "ANALITICO — arquivos com nomenclatura incorreta serao ignorados.",
            "exemplos": {
                "AND-10-06-2020_T1.dx": "Andiroba pura, triplicata 1",
                "BCB-03-03-2020_AD-S-20_T1.dx": "Bacaba adulterada com 20% de soja",
            },
        },
        "EN": {
            "desc": "DX file naming format for pipeline reading.",
            "impacto": "ANALYTICAL — files with incorrect naming will be ignored.",
            "exemplos": {
                "AND-10-06-2020_T1.dx": "Pure Andiroba, replicate 1",
                "BCB-03-03-2020_AD-S-20_T1.dx": "Bacaba adulterated with 20% soy",
            },
        },
        "default": "COD-DD-MM-AAAA_Tn.dx", "range": "Ver menu Codificacao",
    },
    "tamanho_fonte": {
        "PT": {
            "desc": "Tamanho das fontes nas figuras geradas (titulo, eixos, legenda).",
            "impacto": "VISUAL — nao afeta calculos. Essencial para posters e apresentacoes.",
            "exemplos": {"xs": "Muito pequeno", "m": "Medio (padrao)", "xl": "Muito grande"},
        },
        "EN": {
            "desc": "Font size in generated figures (title, axes, legend).",
            "impacto": "VISUAL — no effect on calculations. Essential for posters and presentations.",
            "exemplos": {"xs": "Very small", "m": "Medium (default)", "xl": "Very large"},
        },
        "default": "m", "range": "xs | s | m | l | xl",
    },
    "grid_major": {
        "PT": {
            "desc": "Ativa grid principal nas figuras. Melhora leitura de valores.",
            "impacto": "VISUAL — padrao cientifico para publicacoes.",
            "exemplos": {"true": "Grid principal ativado (recomendado)", "false": "Sem grid"},
        },
        "EN": {
            "desc": "Enable major grid in figures. Improves value reading.",
            "impacto": "VISUAL — scientific standard for publications.",
            "exemplos": {"true": "Major grid enabled (recommended)", "false": "No grid"},
        },
        "default": "true", "range": "true | false",
    },
    "alpha_pontos": {
        "PT": {
            "desc": "Transparencia dos pontos nos graficos de dispersao (scatter).",
            "impacto": "VISUAL — alfa alto revela densidade de pontos sobrepostos.",
            "exemplos": {"baixo": "0.9 — pontos opacos", "medio": "0.65 — equilibrado", "alto": "0.35 — translucido"},
        },
        "EN": {
            "desc": "Transparency of points in scatter plots.",
            "impacto": "VISUAL — high alpha reveals density of overlapping points.",
            "exemplos": {"baixo": "0.9 — opaque points", "medio": "0.65 — balanced", "alto": "0.35 — translucent"},
        },
        "default": "medio", "range": "baixo | medio | alto",
    },
    "pca_biplot": {
        "PT": {
            "desc": "Gera PCA Biplot 2D com elipse de confianca 95% (Hotelling T2) por classe.",
            "impacto": "ANALITICO — mostra separacao espectral e regioes de sobreposicao entre classes.",
            "exemplos": {
                "PC1 vs PC2": "Componentes com maior variancia explicada",
                "Elipse 95%": "Regiao de confianca por Hotelling T2 (chi2 2 graus)",
                "Setas loadings": "Top-8 variaveis com maior contribuicao nos PCs",
            },
        },
        "EN": {
            "desc": "Generates 2D PCA Biplot with 95% confidence ellipse (Hotelling T2) per class.",
            "impacto": "ANALYTICAL — shows spectral separation and overlap regions between classes.",
            "exemplos": {
                "PC1 vs PC2": "Components with highest explained variance",
                "95% ellipse": "Confidence region by Hotelling T2 (chi2 2 degrees)",
                "Loading arrows": "Top-8 variables with highest PC contribution",
            },
        },
        "default": "N/A", "range": "menu [7] > [B]",
    },
    "wavelength_importance": {
        "PT": {
            "desc": "Importancia espectral por loadings PCA ponderados pela variancia explicada.",
            "impacto": "ANALITICO — identifica regioes espectrais informativas antes da modelagem.",
            "exemplos": {
                "Pico em 5200 cm-1": "Alta importancia = regiao discriminante para as classes",
                "Top-10 regioes": "Marcados em vermelho no grafico",
            },
        },
        "EN": {
            "desc": "Spectral importance via PCA loadings weighted by explained variance.",
            "impacto": "ANALYTICAL — identifies informative spectral regions before modelling.",
            "exemplos": {
                "Peak at 5200 cm-1": "High importance = discriminating region for classes",
                "Top-10 regions": "Marked in red on the figure",
            },
        },
        "default": "N/A", "range": "menu [7] > [V]",
    },
}

PROFILES: Dict[str, Dict[str, Any]] = {
    # ── Presets por OBJETIVO CIENTIFICO (CLAUDE.md secao 6 / auditoria
    #    2026-07-12: "3 presets Autenticar/Explorar/Quantificar" p/ reduzir a
    #    barreira dos 84 campos de Config para quem so' quer "rodar a analise
    #    certa" sem entender nivel/objetivo primeiro). Diferem dos perfis
    #    abaixo (que sao niveis de RIGOR: rapido -> tese) — estes escolhem O
    #    QUE analisar, nao QUAO A FUNDO. Nomes sem sufixo N1/N2/N3 (P8).
    "Explorar Dados": {
        "nivel": "N1", "objetivo": "exploratorio",
        "max_lvs": 15, "n_permutacoes": 50,
        "ddsimca": False, "opls_da": False,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": False, "comparar_pre_processamentos": False,
        "dpi": 200,
    },
    "Autenticar Pureza": {
        "nivel": "N2", "objetivo": "auto",
        "ddsimca": True, "modo_ddsimca": "puros",
        "max_lvs": 15, "n_permutacoes": 100,
        "opls_da": False, "benchmark": False, "monte_carlo": False,
        "shap_benchmark": False, "selecao_variaveis_etapa4": False,
        "comparar_pre_processamentos": False, "dpi": 300,
    },
    "Quantificar Teor": {
        "nivel": "N3", "objetivo": "auto",
        "max_lvs": 30, "n_permutacoes": 100,
        "ddsimca": False, "opls_da": False,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": False, "comparar_pre_processamentos": False,
        "dpi": 300, "figuras_mostrar_elipses": True,
    },
    # 1 — primeiro contato com os dados, sem esperar
    "Exploracao Rapida": {
        "max_lvs": 20, "n_permutacoes": 50,
        "ddsimca": False, "opls_da": False,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": False, "comparar_pre_processamentos": False,
        "dpi": 150,
    },
    # 2 — análise completa balanceada, neutro, para uso geral
    "Analise Padrao": {
        "max_lvs": 30, "n_permutacoes": 100,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": True, "comparar_pre_processamentos": False,
        "dpi": 300, "figuras_mostrar_elipses": True,
        "_paleta": "publicacao",
    },
    # 3 — validação estatística mais rigorosa para trabalho científico
    "Pesquisa Academica": {
        "max_lvs": 40, "n_permutacoes": 200,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": True, "comparar_pre_processamentos": False,
        "dpi": 300, "figuras_mostrar_elipses": True,
        "_paleta": "publicacao",
    },
    # 4 — pronto para submissão em periódico indexado
    "Publicacao Cientifica": {
        "max_lvs": 40, "n_permutacoes": 200,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": True, "monte_carlo": False, "shap_benchmark": True,
        "selecao_variaveis_etapa4": True, "comparar_pre_processamentos": False,
        "dpi": 600, "figuras_mostrar_elipses": True,
        "_paleta": "publicacao",
    },
    # 5 — máxima rigorosidade estatística (dissertação / tese)
    "Alta Rigorosidade": {
        "max_lvs": 40, "n_permutacoes": 500,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": True, "monte_carlo": True, "n_monte_carlo": 200,
        "shap_benchmark": True, "selecao_variaveis_etapa4": True,
        "comparar_pre_processamentos": False,
        "dpi": 600, "formato_figura": "pdf",
        "_paleta": "publicacao",
    },
    # 6 — foco exclusivo em comparar pipelines de pré-processamento
    "Benchmark Preprocessamento": {
        "max_lvs": 30, "n_permutacoes": 100,
        "ddsimca": False, "opls_da": False,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": False, "comparar_pre_processamentos": True,
        "dpi": 300,
    },
    # 7 — acessibilidade para daltonismo (único com daltonismo_safe)
    "Acessibilidade": {
        "max_lvs": 30, "n_permutacoes": 100,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": True, "comparar_pre_processamentos": False,
        "dpi": 300, "figuras_mostrar_elipses": True,
        "figuras_mostrar_marcadores": True,
        "_paleta": "daltonismo_safe",
    },
}

PROFILE_DESC: Dict[str, Dict[str, str]] = {
    "Explorar Dados": {
        "PT": ("Primeiro olhar nos dados: PCA, HCA, espectros medios — sem forcar\n"
               "    classificacao. Use quando ainda nao sabe o que procurar.\n"
               "    Ideal para: checar agrupamentos, outliers, qualidade do dado bruto.\n"
               "    Tempo estimado: ~5-10 min."),
        "EN": ("First look at the data: PCA, HCA, mean spectra — no forced\n"
               "    classification. Use when you don't know what to look for yet.\n"
               "    Ideal for: checking clusters, outliers, raw data quality.\n"
               "    Estimated time: ~5-10 min."),
    },
    "Autenticar Pureza": {
        "PT": ("Puro vs. adulterado, por especie, via DD-SIMCA one-class.\n"
               "    Sensibilidade honesta por leave-one-group-out (LOGO) — nunca\n"
               "    re-substituicao. Use quando o objetivo e' checar autenticidade.\n"
               "    Tempo estimado: ~10-20 min."),
        "EN": ("Pure vs. adulterated, per species, via one-class DD-SIMCA.\n"
               "    Honest leave-one-group-out (LOGO) sensitivity — never\n"
               "    re-substitution. Use when the goal is authenticity checking.\n"
               "    Estimated time: ~10-20 min."),
    },
    "Quantificar Teor": {
        "PT": ("Estima o teor (%) de adulterante por regressao PLS calibrada por\n"
               "    especie, com figuras de merito (LOD/LOQ/seletividade) e o mapa\n"
               "    de calor espécie x adulterante (mostra o que NAO e' quantificavel).\n"
               "    Tempo estimado: ~15-30 min."),
        "EN": ("Estimates adulterant content (%) via species-calibrated PLS\n"
               "    regression, with figures of merit (LOD/LOQ/selectivity) and the\n"
               "    species x adulterant heatmap (shows what is NOT quantifiable).\n"
               "    Estimated time: ~15-30 min."),
    },
    "Exploracao Rapida": {
        "PT": ("Primeiro contato com os dados. Somente PLS-DA basico, sem modulos pesados.\n"
               "    Ideal para: verificar se os dados carregam, testar o pipeline.\n"
               "    Tempo estimado: ~5 min."),
        "EN": ("First contact with data. Basic PLS-DA only, no heavy modules.\n"
               "    Ideal for: verifying data loads, testing the pipeline.\n"
               "    Estimated time: ~5 min."),
    },
    "Analise Padrao": {
        "PT": ("Analise completa equilibrada. Uso geral, neutro, sem exageros.\n"
               "    PLS-DA + OPLS-DA + DD-SIMCA + selecao de variaveis. DPI 300.\n"
               "    Paleta Nature/Elsevier. Sem benchmark nem Monte Carlo.\n"
               "    Tempo estimado: ~15-30 min."),
        "EN": ("Balanced complete analysis. General use, neutral, no excess.\n"
               "    PLS-DA + OPLS-DA + DD-SIMCA + variable selection. DPI 300.\n"
               "    Nature/Elsevier palette. No benchmark or Monte Carlo.\n"
               "    Estimated time: ~15-30 min."),
    },
    "Pesquisa Academica": {
        "PT": ("Validacao estatistica mais rigorosa para trabalho cientifico.\n"
               "    Analise completa com 200 permutacoes. Sem benchmark (economiza tempo).\n"
               "    Recomendado para relatorios de pesquisa e comunicacoes em eventos.\n"
               "    Tempo estimado: ~30-45 min."),
        "EN": ("Stricter statistical validation for scientific work.\n"
               "    Full analysis with 200 permutations. No benchmark (saves time).\n"
               "    Recommended for research reports and conference communications.\n"
               "    Estimated time: ~30-45 min."),
    },
    "Publicacao Cientifica": {
        "PT": ("Pronto para submissao em revista indexada. Foco: comparacao de modelos.\n"
               "    Benchmark SVM/RF/XGBoost vs PLS-DA + SHAP + 200 perm + DPI 600.\n"
               "    Recomendado: Talanta, Food Chemistry, J. Chemometrics.\n"
               "    Tempo estimado: ~60-120 min."),
        "EN": ("Ready for indexed journal submission. Focus: model comparison.\n"
               "    Benchmark SVM/RF/XGBoost vs PLS-DA + SHAP + 200 perm + DPI 600.\n"
               "    Recommended: Talanta, Food Chemistry, J. Chemometrics.\n"
               "    Estimated time: ~60-120 min."),
    },
    "Alta Rigorosidade": {
        "PT": ("Maxima rigorosidade estatistica. Foco: incerteza e validacao multipla.\n"
               "    Monte Carlo CV IC95% (200 rep.) + benchmark + SHAP + 500 perm.\n"
               "    Figuras PDF vetorial. Para dissertacao, tese ou revisao externa.\n"
               "    Tempo estimado: ~3-6 horas."),
        "EN": ("Maximum statistical rigor. Focus: uncertainty and multiple validation.\n"
               "    Monte Carlo CV CI95% (200 rep.) + benchmark + SHAP + 500 perm.\n"
               "    Vector PDF figures. For dissertation, thesis or external review.\n"
               "    Estimated time: ~3-6 hours."),
    },
    "Benchmark Preprocessamento": {
        "PT": ("Foco exclusivo: qual pipeline de pre-processamento e o melhor?\n"
               "    Compara raw, MSC, SNV, SG1, SG2, MSC+SG, SNV+SG com metricas.\n"
               "    Sem analise completa — apenas comparacao de pipelines espectrais.\n"
               "    Tempo estimado: ~20-40 min."),
        "EN": ("Exclusive focus: which pre-processing pipeline is best?\n"
               "    Compares raw, MSC, SNV, SG1, SG2, MSC+SG, SNV+SG with metrics.\n"
               "    No full analysis — only spectral pipeline comparison.\n"
               "    Estimated time: ~20-40 min."),
    },
    "Acessibilidade": {
        "PT": ("Foco: figuras acessiveis para daltonismo. Analise padrao.\n"
               "    Paleta Wong 2011 (daltonismo_safe): 8 cores para deuteranopia/protanopia.\n"
               "    Marcadores de forma por classe alem da cor. DPI 300.\n"
               "    Tempo estimado: ~15-30 min."),
        "EN": ("Focus: accessible figures for color blindness. Standard analysis.\n"
               "    Wong 2011 palette (colorblind_safe): 8 colors for deuteranopia/protanopia.\n"
               "    Shape markers per class in addition to color. DPI 300.\n"
               "    Estimated time: ~15-30 min."),
    },
}

PROFILE_KEY_SUMMARY: Dict[str, Dict[str, str]] = {
    "Explorar Dados": {
        "PT": "objetivo=exploratorio | LVs=15 | PCA/HCA | sem PLS-DA/DD-SIMCA",
        "EN": "objective=exploratory | LVs=15 | PCA/HCA | no PLS-DA/DD-SIMCA",
    },
    "Autenticar Pureza": {
        "PT": "DD-SIMCA=ON (puros) | LVs=15 | sensibilidade LOGO | sem benchmark",
        "EN": "DD-SIMCA=ON (pure) | LVs=15 | LOGO sensitivity | no benchmark",
    },
    "Quantificar Teor": {
        "PT": "regressao PLS por especie | heatmap especie x adulterante | LVs=30",
        "EN": "species-wise PLS regression | species x adulterant heatmap | LVs=30",
    },
    "Exploracao Rapida": {
        "PT": "LVs=20 | perm=50 | DD-SIMCA=OFF | OPLS=OFF | DPI=150",
        "EN": "LVs=20 | perm=50 | DD-SIMCA=OFF | OPLS=OFF | DPI=150",
    },
    "Analise Padrao": {
        "PT": "LVs=30 | perm=100 | OPLS=ON | DD-SIMCA=ON | DPI=300 | paleta=publicacao",
        "EN": "LVs=30 | perm=100 | OPLS=ON | DD-SIMCA=ON | DPI=300 | palette=publication",
    },
    "Pesquisa Academica": {
        "PT": "LVs=40 | perm=200 | OPLS=ON | DD-SIMCA=ON | DPI=300 | paleta=publicacao",
        "EN": "LVs=40 | perm=200 | OPLS=ON | DD-SIMCA=ON | DPI=300 | palette=publication",
    },
    "Publicacao Cientifica": {
        "PT": "LVs=40 | perm=200 | Benchmark=ON | SHAP=ON | DPI=600 | paleta=publicacao",
        "EN": "LVs=40 | perm=200 | Benchmark=ON | SHAP=ON | DPI=600 | palette=publication",
    },
    "Alta Rigorosidade": {
        "PT": "perm=500 | MonteCarlo=200rep | Benchmark=ON | SHAP=ON | PDF | paleta=publicacao",
        "EN": "perm=500 | MonteCarlo=200rep | Benchmark=ON | SHAP=ON | PDF | palette=publication",
    },
    "Benchmark Preprocessamento": {
        "PT": "comparar_pipelines=ON | LVs=30 | perm=100 | OPLS=OFF | DD-SIMCA=OFF",
        "EN": "compare_pipelines=ON | LVs=30 | perm=100 | OPLS=OFF | DD-SIMCA=OFF",
    },
    "Acessibilidade": {
        "PT": "LVs=30 | perm=100 | OPLS=ON | DD-SIMCA=ON | marcadores=ON | paleta=daltonismo_safe",
        "EN": "LVs=30 | perm=100 | OPLS=ON | DD-SIMCA=ON | markers=ON | palette=colorblind_safe",
    },
}

REFERENCIAS_GUARACI: Dict[str, Dict[str, str]] = {
    "pls_da_brereton": {
        "cit": ("Brereton, R. G.; Lloyd, G. R. (2014). Partial least squares discriminant "
                "analysis: taking the magic away. Journal of Chemometrics, 28(4), 213-225. "
                "doi:10.1002/cem.2609"),
        "contexto": "PLS-DA — fundamentos e armadilhas comuns",
    },
    "pls_geladi_1986": {
        "cit": ("Geladi, P.; Kowalski, B. R. (1986). Partial least-squares regression: "
                "a tutorial. Analytica Chimica Acta, 185, 1-17. "
                "doi:10.1016/0003-2670(85)85121-2"),
        "contexto": "PLS — referencia fundamental",
    },
    "opls_da_trygg_2002": {
        "cit": ("Trygg, J.; Wold, S. (2002). Orthogonal projections to latent structures (O-PLS). "
                "Journal of Chemometrics, 16(3), 119-128. doi:10.1002/cem.695"),
        "contexto": "OPLS-DA — metodo base",
    },
    "dd_simca_pomerantsev": {
        "cit": ("Pomerantsev, A. L.; Rodionova, O. Y. (2014). Concept and role of extreme objects "
                "in PCA/SIMCA. Journal of Chemometrics, 28(5), 429-438. doi:10.1002/cem.2506"),
        "contexto": "DD-SIMCA — fundamentos",
    },
    "msc_geladi_1985": {
        "cit": ("Geladi, P. et al. (1985). Linearization and scatter-correction for "
                "near-infrared reflectance spectra of meat. Applied Spectroscopy, 39(3), 491-500. "
                "doi:10.1366/0003702854248684"),
        "contexto": "MSC — artigo original",
    },
    "snv_barnes_1989": {
        "cit": ("Barnes, R. J. et al. (1989). Standard Normal Variate Transformation and "
                "De-trending of Near-Infrared Diffuse Reflectance Spectra. "
                "Applied Spectroscopy, 43(5), 772-777. doi:10.1366/0003702894202201"),
        "contexto": "SNV — artigo original",
    },
    "savitzky_golay_1964": {
        "cit": ("Savitzky, A.; Golay, M. J. E. (1964). Smoothing and Differentiation of Data "
                "by Simplified Least Squares Procedures. Analytical Chemistry, 36(8), 1627-1639. "
                "doi:10.1021/ac60214a047"),
        "contexto": "Savitzky-Golay — artigo original",
    },
    "oleos_vegetais_nir_review": {
        "cit": ("Sherazi, S. T. H. et al. (2023). Application of near-infrared spectroscopy "
                "for authentication of edible oils: A review. Food Chemistry, 408, 135181. "
                "doi:10.1016/j.foodchem.2022.135181"),
        "contexto": "Revisao NIR para oleos vegetais — atual",
    },
    "shap_lundberg_2017": {
        "cit": ("Lundberg, S. M.; Lee, S.-I. (2017). A Unified Approach to Interpreting "
                "Model Predictions. Advances in Neural Information Processing Systems (NeurIPS), "
                "30. arXiv:1705.07874"),
        "contexto": "SHAP values — metodo base",
    },
    "monte_carlo_cv_xu": {
        "cit": ("Xu, Q.-S.; Liang, Y.-Z. (2001). Monte Carlo cross validation. "
                "Chemometrics and Intelligent Laboratory Systems, 56(1), 1-11. "
                "doi:10.1016/S0169-7439(00)00122-2"),
        "contexto": "Monte Carlo CV — metodo",
    },
}

MENU_FIELDS: Dict[str, list] = {
    "projeto": ["pasta_dados", "pasta_saida", "nome_execucao"],
    "dados": ["modo_entrada", "arquivo_csv", "coluna_classe", "coluna_concentracao",
              "faixa_min_cm", "faixa_max_cm", "excluir_classes",
              "imagem_incluir_textura"],
    "preproc": ["pre_processamento", "comparar_pre_processamentos"],
    "modelo": ["nivel", "objetivo", "max_lvs", "opls_da", "ddsimca", "modo_ddsimca",
               "selecao_variaveis_etapa4", "selecao_spa", "selecao_ag"],
    "validacao": ["holdout_fracao", "validacao_group_aware", "n_permutacoes",
                  "teste_wold", "teste_cv_anova", "teste_martens", "n_jobs_permutacao"],
    "avancado": ["benchmark", "benchmark_regressao", "monte_carlo", "n_monte_carlo",
                 "monte_carlo_incluir_todos", "shap_benchmark", "shap_max_amostras"],
    "visualizacao": ["figuras_detalhadas", "figuras_mostrar_marcadores",
                     "figuras_mostrar_elipses",
                     "formato_figura", "dpi", "abrir_figuras_na_tela"],
}

_SPEC_BY_KEY: Dict[str, Dict[str, Any]] = {s["key"]: s for s in _CONFIG_SPEC}

_SPEC_EXTRAS: Dict[str, Dict[str, Any]] = {
    "tag": {"key": "tag", "attr": "tag", "tipo": "str", "desc": "Sufixo da pasta de saida", "opcoes": None},
    "nome_execucao": {"key": "nome_execucao", "attr": "tag", "tipo": "str",
                      "desc": "Nome da execucao atual (alias de tag)", "opcoes": None},
}

_SPEC_BY_KEY.update(_SPEC_EXTRAS)

TECNICAS: Dict[str, Dict[str, Any]] = {
    "ft-nir": {
        "PT": {
            "nome": "FT-NIR (Infravermelho Proximo por Transformada de Fourier)",
            "desc": "Espectroscopia NIR de alta resolucao. Ideal para oleos, alimentos, farmacos.",
            "preproc_rec": "MSC+SG+MC",
            "faixa": "4000-10000 cm-1 (tipico) ou 4000-12000 cm-1",
        },
        "EN": {
            "nome": "FT-NIR (Fourier Transform Near Infrared Spectroscopy)",
            "desc": "High-resolution NIR spectroscopy. Ideal for oils, food, pharmaceuticals.",
            "preproc_rec": "MSC+SG+MC",
            "faixa": "4000-10000 cm-1 (typical) or 4000-12000 cm-1",
        },
        "faixa_min": 4000.0, "faixa_max": 10000.0,
        "preproc": "msc_sg_mc", "modo": "dx",
    },
    "nir": {
        "PT": {
            "nome": "NIR Dispersivo (Infravermelho Proximo)",
            "desc": "NIR convencional com detector dispersivo. Faixa tipica 700-2500 nm.",
            "preproc_rec": "SNV+SG+MC ou MSC+SG+MC",
            "faixa": "4000-14000 cm-1 (700-2500 nm)",
        },
        "EN": {
            "nome": "Dispersive NIR (Near Infrared Spectroscopy)",
            "desc": "Conventional NIR with dispersive detector. Typical range 700-2500 nm.",
            "preproc_rec": "SNV+SG+MC or MSC+SG+MC",
            "faixa": "4000-14000 cm-1 (700-2500 nm)",
        },
        "faixa_min": 4000.0, "faixa_max": 14000.0,
        "preproc": "snv_sg_mc", "modo": "dx",
    },
    "mir": {
        "PT": {
            "nome": "MIR/FTIR (Infravermelho Medio)",
            "desc": "Infravermelho medio (4000-400 cm-1). Bandas fundamentais de absorcao molecular.",
            "preproc_rec": "SNV+MC ou SG+MC (MSC menos comum em MIR)",
            "faixa": "400-4000 cm-1",
        },
        "EN": {
            "nome": "MIR/FTIR (Mid-Infrared Spectroscopy)",
            "desc": "Mid-infrared (4000-400 cm-1). Fundamental molecular absorption bands.",
            "preproc_rec": "SNV+MC or SG+MC (MSC less common in MIR)",
            "faixa": "400-4000 cm-1",
        },
        "faixa_min": 400.0, "faixa_max": 4000.0,
        "preproc": "snv_sg_mc", "modo": "dx",
    },
    "raman": {
        "PT": {
            "nome": "Raman (Espectroscopia Raman)",
            "desc": "Espectroscopia vibracional por espalhamento Raman. Complementar ao IR.",
            "preproc_rec": "SG+MC (sem MSC — baseline Raman diferente do NIR)",
            "faixa": "50-4000 cm-1 (Raman shift)",
        },
        "EN": {
            "nome": "Raman Spectroscopy",
            "desc": "Vibrational spectroscopy by Raman scattering. Complementary to IR.",
            "preproc_rec": "SG+MC (no MSC — Raman baseline differs from NIR)",
            "faixa": "50-4000 cm-1 (Raman shift)",
        },
        "faixa_min": 50.0, "faixa_max": 4000.0,
        "preproc": "sg_mc", "modo": "dx",
    },
    "uv-vis": {
        "PT": {
            "nome": "UV-Vis (Ultravioleta-Visivel)",
            "desc": "Espectroscopia de absorbancia UV-Vis. Use modo CSV com colunas de comprimento de onda.",
            "preproc_rec": "SNV+MC ou Mean-centering (dados UV geralmente ja normalizados)",
            "faixa": "190-900 nm (use CSV — wavelength em nm como colunas)",
        },
        "EN": {
            "nome": "UV-Vis (Ultraviolet-Visible Spectroscopy)",
            "desc": "UV-Vis absorbance spectroscopy. Use CSV mode with wavelength columns.",
            "preproc_rec": "SNV+MC or Mean-centering (UV data is often already normalized)",
            "faixa": "190-900 nm (use CSV — wavelength in nm as columns)",
        },
        "faixa_min": 190.0, "faixa_max": 900.0,
        "preproc": "snv_mc", "modo": "csv",
    },
    "fluorescencia": {
        "PT": {
            "nome": "Fluorescencia Molecular",
            "desc": "Espectrofluorimetria. Dados tipicamente em formato CSV (comprimento de onda em nm).",
            "preproc_rec": "SNV+MC ou apenas MC (fluorescencia varia muito entre instrumentos)",
            "faixa": "200-800 nm (use CSV)",
        },
        "EN": {
            "nome": "Molecular Fluorescence",
            "desc": "Spectrofluorimetry. Data typically in CSV format (wavelength in nm).",
            "preproc_rec": "SNV+MC or MC only (fluorescence varies widely between instruments)",
            "faixa": "200-800 nm (use CSV)",
        },
        "faixa_min": 200.0, "faixa_max": 800.0,
        "preproc": "snv_mc", "modo": "csv",
    },
    "hplc": {
        "PT": {
            "nome": "HPLC (Cromatografia Liquida de Alta Performance)",
            "desc": "Dados cromatograficos em formato CSV. Colunas = tempo de retencao ou compostos.",
            "preproc_rec": "MC ou autoscaling (dados HPLC sao areas/alturas de pico)",
            "faixa": "Nao aplicavel — use CSV com colunas de compostos",
        },
        "EN": {
            "nome": "HPLC (High Performance Liquid Chromatography)",
            "desc": "Chromatographic data in CSV format. Columns = retention times or compounds.",
            "preproc_rec": "MC or autoscaling (HPLC data are peak areas/heights)",
            "faixa": "Not applicable — use CSV with compound columns",
        },
        "faixa_min": 0.0, "faixa_max": 60.0,
        "preproc": "autoscaling", "modo": "csv",
    },
    "gc-ms": {
        "PT": {
            "nome": "GC-MS (Cromatografia Gasosa / Massas)",
            "desc": "Compostos volateis. Dados em CSV: TIC ou tabela de fragmentos m/z.",
            "preproc_rec": "MC ou autoscaling (apos alinhamento de picos)",
            "faixa": "Nao aplicavel — use CSV (m/z ou tempo de retencao)",
        },
        "EN": {
            "nome": "GC-MS (Gas Chromatography / Mass Spectrometry)",
            "desc": "Volatile compounds. CSV data: TIC or m/z fragment table.",
            "preproc_rec": "MC or autoscaling (after peak alignment)",
            "faixa": "Not applicable — use CSV (m/z or retention time)",
        },
        "faixa_min": 0.0, "faixa_max": 90.0,
        "preproc": "autoscaling", "modo": "csv",
    },
    "nmr": {
        "PT": {
            "nome": "RMN / NMR (Ressonancia Magnetica Nuclear)",
            "desc": "Fingerprint molecular. Bucketing por janelas de ppm antes de PCA/PLS.",
            "preproc_rec": "SNV ou PQN + MC (apos binning)",
            "faixa": "0-12 ppm (deslocamento quimico)",
        },
        "EN": {
            "nome": "NMR (Nuclear Magnetic Resonance)",
            "desc": "Molecular fingerprint. Bucketing by ppm windows before PCA/PLS.",
            "preproc_rec": "SNV or PQN + MC (after binning)",
            "faixa": "0-12 ppm (chemical shift)",
        },
        "faixa_min": 0.0, "faixa_max": 12.0,
        "preproc": "snv_mc", "modo": "csv",
    },
    "ims": {
        "PT": {
            "nome": "IMS (Espectrometria de Mobilidade Ionica)",
            "desc": "Separacao de ions por mobilidade. Normalizar pelo pico do solvente (RIP).",
            "preproc_rec": "SNV+SG+MC (apos alinhamento por tempo de deriva)",
            "faixa": "5-50 ms (tempo de deriva)",
        },
        "EN": {
            "nome": "IMS (Ion Mobility Spectrometry)",
            "desc": "Ion separation by mobility. Normalize by solvent peak (RIP).",
            "preproc_rec": "SNV+SG+MC (after drift-time alignment)",
            "faixa": "5-50 ms (drift time)",
        },
        "faixa_min": 5.0, "faixa_max": 50.0,
        "preproc": "snv_sg_mc", "modo": "csv",
    },
    "generico": {
        "PT": {
            "nome": "Generico / Personalizado",
            "desc": "Qualquer tipo de dado multivariado. Configure faixa e pre-processamento manualmente.",
            "preproc_rec": "Depende dos dados — configure manualmente no menu Pre-processamento",
            "faixa": "Configurar manualmente no menu Dados",
        },
        "EN": {
            "nome": "Generic / Custom",
            "desc": "Any type of multivariate data. Configure range and preprocessing manually.",
            "preproc_rec": "Depends on data — configure manually in Preprocessing menu",
            "faixa": "Configure manually in Data menu",
        },
        "faixa_min": 0.0, "faixa_max": 999999.0,
        "preproc": "msc_sg_mc", "modo": "dx",
    },
}
