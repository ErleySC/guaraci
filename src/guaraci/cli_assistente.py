"""cli_assistente.py — Modulo de dados/i18n compartilhado do GUARACI.

ATENCAO: este modulo NAO e mais um ponto de entrada executavel. A CLI
interativa completa (menus, wizard, execucao do pipeline) foi UNIFICADA em
guaraci.py (item 16 da auditoria: existiam DUAS implementacoes paralelas do
mesmo fluxo -- menu_project/menu_data/.../wizard_inicial/main aqui E em
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
    load_config, save_config) que guaraci.py tambem resolve por aqui.

Uso:
    python -m guaraci.guaraci   (unico ponto de entrada da CLI interativa)

Requer: pipeline.py no mesmo pacote.
"""

from __future__ import annotations


from typing import Any, Dict

from guaraci import catalogo as _cat
# Config e IO de configuracao vem direto de modulos leves: o motor (`guaraci.pipeline`,
# com sklearn/scipy/pandas) so' e' importado quando uma execucao realmente comeca.
from guaraci import config_io as _cio

_CONFIG_SPEC = _cio._CONFIG_SPEC

_save_config = _cio.save_config

_load_config = _cio.load_config

_attr_para_yaml = _cio._attr_para_yaml

_fmt_yaml = _cio._fmt_yaml

_coagir_valor = _cio._coagir_valor

__all__ = [
    "RISK_CLASS",
    "FIELD_NAMES",
    "PALETAS_COR",
    "FONT_PRESETS",
    "HELP_DB",
    "REFERENCIAS_GUARACI",
    "MENU_FIELDS",
    "TECNICAS",
]


RISK_CLASS: Dict[str, str] = _cat.risk_class()  # derivado do catalogo

FIELD_NAMES: Dict[str, Dict[str, str]] = _cat.field_names()  # derivado do catalogo

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
            "nome": "Qualitativo (maxima distincao)",
            "desc": "Paleta de maxima distincao do GUARACI (Trubetskoy/Glasbey, 20 cores). NAO e' segura para daltonismo: a forma dos marcadores (ligada por padrao) garante a identidade da classe.",
        },
        "EN": {
            "nome": "Qualitative (maximum distinctness)",
            "desc": "GUARACI maximum-distinctness palette (Trubetskoy/Glasbey, 20 colors). NOT colorblind-safe: marker shapes (on by default) keep class identity.",
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
    "alto_contraste": {
        "PT": {
            "nome": "Alto Contraste",
            "desc": "8 cores escuras, todas com contraste >= 4,9:1 sobre branco e bem separadas entre si. Para projecao, impressao fraca e baixa visao.",
        },
        "EN": {
            "nome": "High Contrast",
            "desc": "8 dark colors, all >= 4.9:1 contrast on white and well separated. For projectors, poor printing and low vision.",
        },
        # Derivada dos design tokens do projeto (text/primary de design_tokens.py)
        # mais preto puro -- nao e' uma paleta nova inventada, e' a paleta de
        # identidade levada ao extremo de contraste. Adicionada em 2026-09-08
        # para atender o esquema "Alto contraste" da tela Visualizacao.
        "cores": ["#000000", "#C24A00", "#0B5394", "#2F7A48",
                  "#7B2D8E", "#8A6400", "#666666", "#B0005A"],
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
            "nome": "Publicacao (tab10 do matplotlib)",
            "desc": "A paleta tab10 do matplotlib, a mais comum em artigos. NAO e' segura para daltonismo (vermelho/verde).",
        },
        "EN": {
            "nome": "Publication (matplotlib tab10)",
            "desc": "matplotlib's tab10, the most common palette in papers. NOT colorblind-safe (red/green).",
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

def apply_palette(nome: str) -> None:
    """Ativa a paleta `nome` de `PALETAS_COR` para as figuras da proxima
    execucao. Implementacao UNICA, usada pela CLI (`guaraci.py`, antes de
    `executar()`) e pelo app web (aba Modelo) -- antes so' existia inline na
    CLI.

    Faz duas coisas, porque o projeto tem dois caminhos de cor:

    1. `paleta_cores.set_active_palette` -- e' o que muda a cor das figuras
       de fato. Toda figura do pipeline passa `color=color(i)` /
       `map_class_colors()` explicitamente; nenhuma usa o ciclo padrao do
       matplotlib. Ate' 2026-09-08 a escolha de paleta so' mexia no item 2
       abaixo, ou seja: nao mudava cor nenhuma nas figuras gravadas.
    2. `rcParams` (prop_cycle/cmap/style) -- para o que porventura NAO passe
       cor explicita, e para o `style`/`cmap` (ex.: tema escuro, viridis).

    Nome desconhecido = no-op (volta ao padrao), nunca excecao: escolha
    cosmetica nao derruba uma analise.
    """
    import matplotlib.pyplot as plt

    from guaraci.paleta_cores import set_active_palette

    paleta = PALETAS_COR.get(nome) or {}
    cores = paleta.get("cores")
    set_active_palette(cores)
    try:
        plt.style.use(paleta.get("style", "default"))
    except OSError:
        pass   # nome de estilo matplotlib desconhecido -- mantem o default
    if cores:
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cores)
    cmap = paleta.get("cmap")
    if cmap:
        plt.rcParams["image.cmap"] = cmap


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

HELP_DB: Dict[str, Dict[str, Any]] = _cat.help_db()  # derivado do catalogo (guaraci.catalogo_opcoes)

# Perfis (objetivo x rigor): guaraci.catalogo_dados.PERFIS_OBJETIVO / PERFIS_RIGOR.

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
                "doi:10.1016/0003-2670(86)80028-9"),
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
                "doi:10.1366/0003702854248656"),
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
    "oleos_vegetais_ir_review": {
        "cit": ("Rohman, A. (2017). The use of infrared spectroscopy in combination "
                "with chemometrics for quality control and authentication of edible "
                "fats and oils: a review. Applied Spectroscopy Reviews, 52(7), 589-604. "
                "doi:10.1080/05704928.2016.1266493"),
        "contexto": "Revisao de IV + quimiometria para oleos comestiveis",
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

MENU_FIELDS: Dict[str, list] = _cat.menu_fields()  # derivado do catalogo (ordem = numeracao dos menus)

_SPEC_BY_KEY: Dict[str, Dict[str, Any]] = {s["key"]: s for s in _CONFIG_SPEC}

_SPEC_EXTRAS: Dict[str, Dict[str, Any]] = {
    "nome_execucao": {"key": "nome_execucao", "attr": "tag", "tipo": "str",
                      "desc": "Nome da execucao atual (alias de tag)", "opcoes": None},
}

_SPEC_BY_KEY.update(_SPEC_EXTRAS)

# Tecnicas analiticas da tela [2.4]: visao no formato historico, derivada do catalogo
# (guaraci.catalogo_tecnicas.INSTRUMENTOS) -- nao ha texto proprio aqui.
TECNICAS: Dict[str, Dict[str, Any]] = {
    k: {"PT": {"nome": v["nome"][0], "desc": v["faz"][0], "preproc_rec": v["preproc_rec"][0], "faixa": v["faixa_txt"][0],
               "faixa_curta": v.get("faixa_curta", v["faixa_txt"])[0]},
        "EN": {"nome": v["nome"][1], "desc": v["faz"][1], "preproc_rec": v["preproc_rec"][1], "faixa": v["faixa_txt"][1],
               "faixa_curta": v.get("faixa_curta", v["faixa_txt"])[1]},
        "faixa_min": v["faixa_min"], "faixa_max": v["faixa_max"], "preproc": v["preproc"], "mode": v["mode"],
        "eixo_unidade": v.get("eixo_unidade", ""), "raios_cosmicos": bool(v.get("raios_cosmicos", False)),
        "faixas_excluir": tuple(v.get("faixas_excluir", ()))}
    for k, v in _cat.INSTRUMENTOS.items()
}
