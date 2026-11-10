"""grupos_figuras.py -- subpastas de Graficos/ por grupo de analise.

Tabela prefixo do nome da figura -> subpasta, usada por `figuras.save()`
quando `figuras.ORGANIZAR_POR_GRUPO` esta ligado: figuras que se corroboram
(ex.: scores PLS-DA, matriz de confusao e ROC) ficam lado a lado na mesma
pasta. Modulo leve (so' `os`), para que quem LE as figuras de volta
(`app_logic.list_figures`, relatorios) use a mesma tabela sem importar
matplotlib.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

#: Ligada pela CLI (`guaraci.main`). Mora aqui, e nao so' em `figuras`, porque
#: este modulo so' importa `os`: ligar a chave nao carrega o matplotlib e os
#: comandos leves (`--help`, `ajuda`...) continuam sem carregar o motor.
ORGANIZAR = False

__all__ = ["GRUPOS_FIGURAS", "DESCRICAO_GRUPOS", "SUBPASTAS_REMAPEADAS", "SERIES_FIGURAS",
           "subpasta_do_grupo", "base_da_serie", "subpasta_da_serie", "nomes_da_serie",
           "nomes_do_grupo", "pasta_local", "GRUPOS_EN", "DESCRICAO_GRUPOS_EN", "chave_ordem_figura", "texto_leiame",
           "gravar_leiame"]

#: (prefixo do nome, subpasta). Primeiro prefixo que casa vence.
GRUPOS_FIGURAS: Tuple[Tuple[str, str], ...] = (
    ("fig0_espectros_medios", "01_dados_preprocessamento"),
    ("fig6_preprocessing", "01_dados_preprocessamento"),
    ("fig_extra_comparacao_pipelines", "01_dados_preprocessamento"),
    ("fig_hca_comparacao_pipelines", "01_dados_preprocessamento"),
    ("fig1_pca_scores", "02_exploracao_PCA_HCA"),
    ("fig1b_pca_scree", "02_exploracao_PCA_HCA"),
    ("fig_loadings_pca", "02_exploracao_PCA_HCA"),
    ("fig_biplot_pca", "02_exploracao_PCA_HCA"),
    ("fig_hca_dendrogram", "02_exploracao_PCA_HCA"),
    ("fig2_plsda_scores", "03_PLS-DA"),
    ("figS1_selecao_lvs", "03_PLS-DA"),
    ("fig4_confusao", "03_PLS-DA"),
    ("fig_roc_auc", "03_PLS-DA"),
    ("fig3_outliers", "04_diagnostico_outliers"),
    ("fig5b_vip", "05_VIP_selecao_variaveis"),
    ("fig_selectivity_ratio_vip", "05_VIP_selecao_variaveis"),
    ("fig_score_contribution", "05_VIP_selecao_variaveis"),
    ("fig_etapa4_", "05_VIP_selecao_variaveis"),
    ("fig_extra_holdout", "06_validacao"),
    ("fig_extra_wold", "06_validacao"),
    ("fig_monte_carlo", "06_validacao"),
    ("fig_det_curvas", "06_validacao"),
    ("fig_ddsimca", "07_DD-SIMCA"),
    ("ddsimca_", "07_DD-SIMCA"),
    ("fig_cooman_ddsimca", "07_DD-SIMCA"),
    ("fig_opls_scores", "08_OPLS-DA"),
    ("fig_splot_opls", "08_OPLS-DA"),
    ("fig_benchmark_", "09_benchmark_SHAP"),
    ("fig_shap_", "09_benchmark_SHAP"),
    ("figS2_", "10_quantificacao"),
    ("figS2b_", "10_quantificacao"),
    ("figS2c_", "10_quantificacao"),
    ("figS2d_", "10_quantificacao"),
    ("figS3_", "10_quantificacao"),
    ("fig_heatmap_species_by_adulterant", "10_quantificacao"),
    ("fig_multielementar", "11_multielementar"),
    ("fig_mcr_als", "12_resolucao_MCR"),
    ("fig_parafac", "13_multivias_PARAFAC"),
    ("fig_asca", "14_ASCA"),
    ("fig_transferencia", "15_transferencia"),
)

#: subpasta pedida pelo chamador -> grupo, com a chave ligada
#: ("ddsimca" era a subpasta das figuras individuais e do Cooman).
SUBPASTAS_REMAPEADAS: Dict[str, str] = {"ddsimca": "07_DD-SIMCA"}

#: uma linha (PT) sobre o que as figuras de cada pasta mostram juntas.
DESCRICAO_GRUPOS: Dict[str, str] = {
    "01_dados_preprocessamento": "espectros medios brutos e o efeito do pre-processamento; "
                                 "comparacoes entre pre-processamentos.",
    "02_exploracao_PCA_HCA": "estrutura nao supervisionada: scores, scree, loadings e biplot da "
                             "PCA e dendrograma HCA devem contar a mesma historia de agrupamento.",
    "03_PLS-DA": "modelo PLS-DA: escolha de VLs, scores, matriz de confusao e ROC "
                 "(a separacao nos scores deve bater com os acertos da CV).",
    "04_diagnostico_outliers": "T2 de Hotelling e Q por amostra: quem esta fora do modelo.",
    "05_VIP_selecao_variaveis": "variaveis importantes: VIP (bootstrap), razao de seletividade, "
                                "contribuicao de score e selecao de variaveis (Etapa 4).",
    "06_validacao": "validacao: holdout externo, permutacao de Wold, Monte Carlo e curvas DET.",
    "07_DD-SIMCA": "autenticacao one-class: aceitacao por classe e grafico de Cooman.",
    "08_OPLS-DA": "OPLS-DA: scores preditivo x ortogonal e S-plot das variaveis discriminantes.",
    "09_benchmark_SHAP": "outros classificadores/regressores na mesma CV e explicacao SHAP.",
    "10_quantificacao": "regressao PLS: predito x referencia (com faixa +-RMSEP), residuos, "
                        "coeficientes b, grafico de Williams e figuras de merito.",
    "11_multielementar": "medias por elemento (ICP multielementar).",
    "12_resolucao_MCR": "MCR-ALS: perfis de concentracao C e espectros puros S resolvidos.",
    "13_multivias_PARAFAC": "PARAFAC da EEM: loadings de excitacao, de emissao e scores das amostras.",
    "14_ASCA": "ASCA: scores e loadings do efeito de cada fator do delineamento.",
    "15_transferencia": "transferencia de calibracao: mestre x escravo antes/depois de DS/PDS "
                        "e erro por amostra.",
}

#: nome da pasta do grupo em ingles (sessao EN); grupos sem entrada tem o mesmo nome nos dois idiomas.
GRUPOS_EN: Dict[str, str] = {
    "01_dados_preprocessamento": "01_data_preprocessing",
    "02_exploracao_PCA_HCA": "02_exploration_PCA_HCA",
    "04_diagnostico_outliers": "04_outlier_diagnostics",
    "05_VIP_selecao_variaveis": "05_VIP_variable_selection",
    "06_validacao": "06_validation",
    "10_quantificacao": "10_quantification",
    "11_multielementar": "11_multielement",
    "12_resolucao_MCR": "12_MCR_resolution",
    "13_multivias_PARAFAC": "13_multiway_PARAFAC",
    "15_transferencia": "15_transfer",
}

#: uma linha (EN) por grupo, para o README.txt de uma sessao em ingles.
DESCRICAO_GRUPOS_EN: Dict[str, str] = {
    "01_dados_preprocessamento": "raw mean spectra and the effect of pre-processing; comparisons between "
                                 "pre-processing pipelines.",
    "02_exploracao_PCA_HCA": "unsupervised structure: PCA scores, scree, loadings and biplot and the HCA "
                             "dendrogram should tell the same grouping story.",
    "03_PLS-DA": "PLS-DA model: choice of LVs, scores, confusion matrix and ROC (separation in the scores "
                 "should match the CV hits).",
    "04_diagnostico_outliers": "Hotelling T2 and Q per sample: who is outside the model.",
    "05_VIP_selecao_variaveis": "important variables: VIP (bootstrap), selectivity ratio, score contribution "
                                "and variable selection (Step 4).",
    "06_validacao": "validation: external holdout, Wold permutation, Monte Carlo and DET curves.",
    "07_DD-SIMCA": "one-class authentication: acceptance per class and Cooman's plot.",
    "08_OPLS-DA": "OPLS-DA: predictive x orthogonal scores and S-plot of the discriminant variables.",
    "09_benchmark_SHAP": "other classifiers/regressors in the same CV and SHAP explanation.",
    "10_quantificacao": "PLS regression: predicted x reference (with +-RMSEP band), residuals, b coefficients, "
                        "Williams plot and figures of merit.",
    "11_multielementar": "means per element (multi-element ICP).",
    "12_resolucao_MCR": "MCR-ALS: resolved concentration profiles C and pure spectra S.",
    "13_multivias_PARAFAC": "EEM PARAFAC: excitation and emission loadings and sample scores.",
    "14_ASCA": "ASCA: scores and loadings of the effect of each design factor.",
    "15_transferencia": "calibration transfer: master x slave before/after DS/PDS and error per sample.",
}


def nomes_do_grupo(grupo: str) -> Tuple[str, str]:
    """(PT, EN) da pasta do grupo `grupo` (nome canonico em PT)."""
    return grupo, GRUPOS_EN.get(grupo, grupo)


def pasta_local(base: str, pt: str, en: str, lang: Optional[str] = None) -> str:
    """`base/pt` ou `base/en`: a que ja' existe vence (rodada antiga ou em outro idioma continua na
    mesma pasta); se nenhuma existe, a do idioma da sessao."""
    for nome in (pt, en):
        if os.path.isdir(os.path.join(base, nome)):
            return os.path.join(base, nome)
    from guaraci.idioma_relatorio import idioma
    return os.path.join(base, en if idioma(lang) == "en" else pt)


_ARQUIVO_LEIAME = "README.txt"


#: Series de figuras (um arquivo por especie/classe/elemento...: nome_01, nome_02...) vao para uma
#: subpasta propria dentro do grupo, para nao misturar dezenas de arquivos com as figuras unicas.
#: base do nome -> (pasta em portugues, pasta em ingles). Serie sem entrada usa a propria base.
SERIES_FIGURAS: Dict[str, Tuple[str, str]] = {
    "figS2d_williams": ("williams_por_especie", "williams_per_species"),
    "figS2c_coeficientes_regressao": ("coeficientes_por_especie", "coefficients_per_species"),
    "fig_ddsimca_aceitacao": ("aceitacao_por_classe", "acceptance_per_class"),
    "fig_cooman_ddsimca": ("cooman_por_par_de_classes", "cooman_per_class_pair"),
    "fig_loadings_pca": ("loadings_por_componente", "loadings_per_component"),
    "fig0_espectros_medios_classe": ("media_por_classe", "mean_per_class"),
    "fig_multielementar_medias": ("medias_por_elemento", "means_per_element"),
    "fig_hca_comparacao_pipelines": ("hca_por_preprocessamento", "hca_per_preprocessing"),
}

_SUFIXO_SERIE = re.compile(r"^(?P<base>.+)_(?P<indice>\d{2})$")


def base_da_serie(nome: str) -> str:
    """Base de uma figura de serie (`figS2d_williams_03` -> `figS2d_williams`); "" se nao e' serie.

    So' `figuras.nome_pagina` cria o sufixo `_NN`, e so' quando ha' mais de um arquivo."""
    m = _SUFIXO_SERIE.match(nome)
    return m.group("base") if m else ""


def nomes_da_serie(nome: str) -> Tuple[str, str]:
    """(PT, EN) da subpasta da serie da figura `nome`; ("", "") para figura unica."""
    base = base_da_serie(nome)
    if not base:
        return "", ""
    return SERIES_FIGURAS.get(base, (base, base))


def subpasta_da_serie(nome: str, lang: Optional[str] = None) -> str:
    """Subpasta da serie da figura `nome`, no idioma da sessao; "" para figura unica."""
    pt, en = nomes_da_serie(nome)
    if not pt:
        return ""
    from guaraci.idioma_relatorio import idioma
    return en if idioma(lang) == "en" else pt


def _e_pasta_de_serie(pasta: str, arquivo: str) -> bool:
    base = base_da_serie(os.path.splitext(arquivo)[0])
    return bool(base) and pasta in (base, *SERIES_FIGURAS.get(base, ()))


def subpasta_do_grupo(nome: str) -> str:
    """Subpasta do grupo da figura `nome` (sem extensao); "" se nao mapeada."""
    for prefixo, pasta in GRUPOS_FIGURAS:
        if nome.startswith(prefixo):
            return pasta
    return ""


def chave_ordem_figura(caminho: str) -> str:
    """Caminho como seria SEM as subpastas de grupo (o layout plano antigo).

    Ordenar por esta chave deixa a selecao das N primeiras figuras dos
    relatorios igual com a chave ligada ou desligada."""
    partes = os.path.normpath(caminho).split(os.sep)
    grupos = set(DESCRICAO_GRUPOS) | set(GRUPOS_EN.values())
    saida = []
    for i, p in enumerate(partes):
        if i == len(partes) - 2 and _e_pasta_de_serie(p, partes[-1]):
            continue                      # subpasta da serie: a ordem e' a do layout plano
        if p in grupos and i < len(partes) - 1:
            # DD-SIMCA individual/Cooman ficavam em Graficos/ddsimca/
            if (p == "07_DD-SIMCA" and i == len(partes) - 2
                    and (partes[-1].startswith("ddsimca_")
                         or partes[-1].startswith("fig_cooman_ddsimca"))):
                saida.append("ddsimca")
            continue
        saida.append(p)
    return os.sep.join(saida)


def texto_leiame(lang: Optional[str] = None) -> str:
    """Conteudo do README.txt gravado em Graficos/ com a chave ligada, no idioma da sessao."""
    from guaraci.idioma_relatorio import idioma
    if idioma(lang) == "en":
        linhas = ["Figures organised by analysis group.",
                  "Figures in the same folder corroborate each other: read them together.",
                  "A series with one file per species/class/element (name_01, name_02...) has its own "
                  "subfolder inside the group.", ""]
        for pasta, descr in DESCRICAO_GRUPOS_EN.items():
            linhas.append(f"{GRUPOS_EN.get(pasta, pasta)}/: {descr}")
        linhas += ["", "Figures without a known group stay loose in this folder; "
                   "hsi/ holds the hyperspectral map."]
    else:
        linhas = ["Graficos organizados por grupo de analise.",
                  "Figuras na mesma pasta se corroboram: leia-as juntas.",
                  "Uma serie com um arquivo por especie/classe/elemento (nome_01, nome_02...) tem "
                  "subpasta propria dentro do grupo.", ""]
        for pasta, descr in DESCRICAO_GRUPOS.items():
            linhas.append(f"{pasta}/: {descr}")
        linhas += ["", "Figuras sem grupo conhecido ficam soltas nesta pasta; "
                   "hsi/ guarda o mapa hiperespectral."]
    return "\n".join(linhas) + "\n"


def gravar_leiame(base: str) -> None:
    """Grava README.txt em `base` uma unica vez (nao sobrescreve)."""
    caminho = os.path.join(base, _ARQUIVO_LEIAME)
    if os.path.exists(caminho):
        return
    try:
        with open(caminho, "w", encoding="utf-8") as fh:
            fh.write(texto_leiame())
    except OSError:
        pass   # README e' so' ajuda; nao derruba o salvamento da figura
