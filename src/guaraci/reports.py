"""reports.py — Geração de relatórios do app web (PDF, Word, Excel, LaTeX,
PowerPoint).

Extraído de app_quimiometria.py (item 18 da auditoria: quebrar o monólito da
UI web por serviço/aba). Estas funções são PURAS quanto a I/O de terminal —
não importam `streamlit`, recebem (pasta, projeto) e devolvem bytes/BytesIO.
A UI (app_quimiometria.py) apenas as chama e envolve o resultado em
`st.cache_data` + `st.download_button`.
"""
from __future__ import annotations

import io
import logging
import os
import re
import time
from typing import Any, Dict, List, Mapping, Tuple, cast

import pandas as pd

import guaraci.pipeline as _pq
from guaraci.app_logic import (
    load_summary as _ler_resumo,
    list_figures as _list_figures,
)
from guaraci import idioma_relatorio as _i18n
from guaraci.config import NOME_TABELAS, pasta_execucao
# Parsing do resumo_modelo.txt centralizado (item 19): _ex e o dicionario de
# metricas eram duplicados nos 5 geradores; agora vem de resumo_parse.
from guaraci.resumo_parse import (MODO_QUANTIFICACAO, avisos_do_modelo, extract_metric, modo_do_resumo,
                                  parse_model_metrics, parse_regressao)

# Fonte unica de versao (mesmo padrao de app_quimiometria.py: pipeline.__version__).
_APP_VERSION = f"v{getattr(_pq, '__version__', '?')}"

# Texto unico do carimbo de prototipo (achado B4-1 da auditoria de
# 2026-08-16). Fonte unica para os 4 geradores nao divergirem.
_AVISO_PROTOTIPO_TITULO = "PROTOTYPE OUTPUT - NO GROUPING GUARANTEE"
_AVISO_PROTOTIPO_CORPO = (
    "This run has no source of physical-sample grouping (no per-sample "
    "subfolder structure, no association CSV) - group-aware validation was "
    "therefore DISABLED and the pipeline fell back to plain StratifiedKFold. "
    "Results should be treated as exploratory, not suitable for publication."
)
_AVISO_PROTOTIPO_TITULO_PT = "RESULTADO DE PROTÓTIPO - SEM GARANTIA DE AGRUPAMENTO"
_AVISO_PROTOTIPO_CORPO_PT = (
    "Esta execução não tem fonte de agrupamento por amostra física (nem "
    "subpastas por amostra, nem CSV de associação) - a validação agrupada foi "
    "DESLIGADA e o pipeline usou StratifiedKFold simples. Trate os resultados "
    "como exploratórios, não publicáveis."
)


def _T(pt: str, en: str) -> str:
    """Frase no idioma dos relatorios (GUARACI_IDIOMA; padrao portugues)."""
    return _i18n.t(pt, en)


def _aviso_prototipo() -> tuple:
    """(titulo, corpo) do carimbo de prototipo no idioma dos relatorios."""
    return (_T(_AVISO_PROTOTIPO_TITULO_PT, _AVISO_PROTOTIPO_TITULO),
            _T(_AVISO_PROTOTIPO_CORPO_PT, _AVISO_PROTOTIPO_CORPO))


#: chave de `resumo_parse.parse_model_metrics` -> (rotulo PT, rotulo EN).
_ROTULOS_METRICA = {
    "Balanced Accuracy (CV)": ("Acurácia balanceada (VC)", "Balanced accuracy (CV)"),
    "AUC macro OvR": ("AUC macro (um contra todos)", "AUC macro (one-vs-rest)"),
    "R2Y": ("R2Y", "R2Y"), "Q2Y": ("Q2Y", "Q2Y"), "R2X": ("R2X", "R2X"),
    "Optimal LVs": ("VLs ótimas", "Optimal LVs"),
    "p-value (permutation)": ("p-valor (permutação)", "p-value (permutation)"),
    "Preprocessing": ("Pré-processamento", "Preprocessing"),
    "Hotelling T2 UCL (95%)": ("Limite T2 de Hotelling (95%)", "Hotelling T2 UCL (95%)"),
    "Q-residual UCL (95%)": ("Limite do resíduo Q (95%)", "Q-residual UCL (95%)"),
    "n samples (training)": ("Amostras (espectros)", "Samples (spectra)"),
    "n classes": ("Classes", "Classes"),
}


def _rotulo_metrica(chave: str) -> str:
    pt, en = _ROTULOS_METRICA.get(chave, (chave, chave))
    return _T(pt, en)


def _criterios_qualidade() -> List[str]:
    """Criterios de qualidade da literatura (PDF e Word), no idioma dos relatorios."""
    return [
        _T("R2Y > 0,60 e Q2Y > 0,50 → modelo preditivo robusto (regra prática)",
           "R2Y > 0.60 and Q2Y > 0.50 → robust predictive model (rule of thumb)"),
        _T("Q2Y / R2Y > 0,70 → baixo risco de sobreajuste",
           "Q2Y / R2Y > 0.70 → low overfitting risk"),
        _T("p-valor da permutação < 0,05 → resultado não aleatório (permutação de Y)",
           "permutation p-value < 0.05 → non-random result (Y-randomization)"),
        _T("Intercepto de R2Y < 0,40 e de Q2Y < 0,05 → validação geométrica (regra prática; "
           "Eriksson et al. 2006, livro)",
           "Intercept R2Y < 0.40 and intercept Q2Y < 0.05 → geometric validation (rule of thumb; "
           "Eriksson et al. 2006, book)"),
        _T("IC BCa 95% da acurácia balanceada não inclui 1/n_classes → discriminação significativa "
           "[Efron 1993]",
           "BCa 95% CI of balanced accuracy does not include 1/n_classes → significant "
           "discrimination [Efron 1993]"),
    ]


def _criterios_quantificacao() -> List[str]:
    """Criterios de leitura de uma quantificacao PLS (PDF/Word/PPTX), no idioma dos relatorios."""
    return [
        _T("O RMSEP do holdout externo (amostras que o modelo nunca viu) é o número publicável; RMSECV e "
           "RMSEP interno (divisão 70/30 do treino) são otimistas, sobretudo com n pequeno.",
           "The external-holdout RMSEP (samples the model never saw) is the publishable number; RMSECV and "
           "the internal RMSEP (70/30 split of the training set) are optimistic, especially with small n."),
        _T("R2cv e RMSECV só valem com validação agrupada pela amostra física; sem isso, réplicas da mesma "
           "amostra em treino e teste inflam os dois.",
           "R2cv and RMSECV only hold with validation grouped by physical sample; otherwise replicates of the "
           "same sample in training and test inflate both."),
        _T("LOD/LOQ calculados só com o ruído de réplica podem ficar abaixo do RMSEP: nesse caso use o RMSEP "
           "como piso do que o modelo distingue (Valderrama, Braga & Poppi, 2009; Allegrini & Olivieri, 2014).",
           "LOD/LOQ computed from replicate noise alone can fall below the RMSEP: in that case use the RMSEP "
           "as the floor of what the model can tell apart (Valderrama, Braga & Poppi, 2009; "
           "Allegrini & Olivieri, 2014)."),
        _T("RPD, quando presente, só vale com a faixa de uso ao lado, e essa faixa é convenção de NIR agrícola, "
           "não regra geral; intervalo de predição só com grupos de validação suficientes (mínimo 19 para "
           "alpha = 0,05).",
           "RPD, when present, only means something with its range of use beside it, and that range is an "
           "agricultural-NIR convention, not a general rule; prediction interval only with enough validation "
           "groups (minimum 19 for alpha = 0.05)."),
    ]


#: referencias citadas pelo sumario de quantificacao (sem DOI quando nao ha certeza do numero).
_REFS_MULTIELEMENTAR_EXTRA = [
    "Benjamini, Y. & Hochberg, Y. (1995) Controlling the false discovery rate: a practical and powerful approach to "
    "multiple testing. J. R. Stat. Soc. B 57(1):289-300.",
    "Games, P.A. & Howell, J.F. (1976) Pairwise multiple comparison procedures with unequal n's and/or variances: a "
    "Monte Carlo study. J. Educ. Stat. 1(2):113-125.",
    "Welch, B.L. (1951) On the comparison of several mean values: an alternative approach. Biometrika 38(3-4):330-336.",
]

_REFS_QUANTIFICACAO = [
    "Allegrini, F. & Olivieri, A.C. (2014) IUPAC-consistent approach to the limit of detection in partial "
    "least-squares calibration. Anal. Chem. 86(15):7858-7866.",
    "Rinnan, A. et al. (2009) Review of the most common pre-processing techniques for near-infrared spectra. "
    "TrAC Trends Anal. Chem. 28(10):1201-1222. doi:10.1016/j.trac.2009.07.007",
    "Valderrama, P., Braga, J.W.B. & Poppi, R.J. (2009) Estado da arte de figuras de merito em calibracao "
    "multivariada. Quim. Nova 32(5):1278-1287.",
    "Wold, S. (1978) Cross-validatory estimation of the number of components in factor and principal "
    "components models. Technometrics 20(4):397-405. doi:10.1080/00401706.1978.10489693",
]


_FAIXA_RPD = {
    "nao utilizavel": ("não utilizável", "not usable"),
    "triagem grosseira": ("triagem grosseira", "rough screening"),
    "triagem": ("triagem", "screening"),
    "controle de qualidade": ("controle de qualidade", "quality control"),
    "controle de processo / quantificacao": ("controle de processo / quantificação", "process control / quantification"),
}


def _fmt_num(v: Any, nd: int = 3) -> str:
    """Numero do resumo para a tabela ('-' quando ausente ou n/a)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if f != f else f"{f:.{nd}f}"


def _pi_texto(e: Mapping[str, Any]) -> str:
    if e.get("pi_validado") is True:
        return _T(f"intervalo de predição ±{_fmt_num(e.get('pi_margem'), 2)} (conforme 95%, "
                  f"{e.get('pi_n_grupos')} grupos)",
                  f"prediction interval ±{_fmt_num(e.get('pi_margem'), 2)} (conformal 95%, "
                  f"{e.get('pi_n_grupos')} groups)")
    if e.get("pi_validado") is False:
        return _T(f"intervalo de predição não validado ({e.get('pi_n_grupos')} grupo(s); mínimo 19 para alpha = 0,05)",
                  f"prediction interval not validated ({e.get('pi_n_grupos')} group(s); minimum 19 for alpha = 0.05)")
    return ""


def _sumario_executivo(resumo_raw: str, default: str = "-") -> Dict[str, Any]:
    """Conteudo do Resumo Executivo conforme o objetivo da execucao (fonte unica de PDF/Word/Excel/PPTX/LaTeX).

    Quantificacao: RMSEP do holdout externo, R2 de predicao, vies, RMSECV, R2cv, RMSEC, R2cal, RMSEP interno,
    R2val, SEP/RPD/RER, DModY, LOD/LOQ e a tabela por especie -- e nenhuma metrica de classificacao sem
    sentido (R2Y/Q2Y/R2X do PLS-DA, AUC, permutacao). A acuracia da identificacao de especie aparece num
    bloco proprio so' quando ha 2+ classes. Classificacao: as metricas de sempre; com 1 classe elas nao se
    aplicam e o sumario diz isso em vez de mostrar R2Y=0.

    Devolve: modo, linhas [(rotulo, valor, base)], preproc, tabela_especie (cabecalho, linhas) ou None, notas,
    criterios, titulo_criterios, tem_base, cabecalho_valor, n_amostras, n_classes, avisos."""
    modo = modo_do_resumo(resumo_raw)
    pm = parse_model_metrics(resumo_raw, default)
    preproc = pm["Preprocessing"]
    if preproc != default:
        preproc = _i18n.preprocessamento(preproc)
    n_amostras, n_classes = pm["n samples (training)"], pm["n classes"]
    uma_classe = n_classes.strip() == "1"
    out: Dict[str, Any] = {
        "modo": modo, "preproc": preproc, "n_amostras": n_amostras, "n_classes": n_classes,
        "linhas": [], "tabela_especie": None, "notas": [], "avisos": [], "tem_base": False,
        "cabecalho_valor": _T("Valor (validação cruzada interna)", "Value (internal CV validation)"),
        "criterios": _criterios_qualidade(),
        "titulo_criterios": _T("Critérios de qualidade (PLS-DA — literatura):",
                               "Quality criteria (PLS-DA — literature):"),
    }
    L: List[tuple] = out["linhas"]
    if modo == MODO_QUANTIFICACAO:
        reg = parse_regressao(resumo_raw)
        po, ho, esps = reg["pooled"], reg["holdout"], reg["especies"]
        out["tem_base"] = True
        out["cabecalho_valor"] = _T("Valor", "Value")
        out["criterios"] = _criterios_quantificacao()
        out["titulo_criterios"] = _T("Como ler a quantificação (PLS-R — literatura):",
                                     "How to read the quantification (PLS-R — literature):")
        b_ho = _T("holdout externo (número publicável)", "external holdout (publishable number)")
        b_cv = _T("validação cruzada interna", "internal cross-validation")
        b_cal = _T("calibração (no treino, otimista)", "calibration (on training data, optimistic)")
        b_int = _T("validação interna 70/30 (n pequeno)", "internal 70/30 validation (small n)")
        if ho:
            L.append((_T(f"RMSEP (holdout externo, n={ho['n']})", f"RMSEP (external holdout, n={ho['n']})"),
                      _fmt_num(ho["rmsep"]), b_ho))
            L.append((_T("R2 de predição (holdout externo)", "Prediction R2 (external holdout)"),
                      _fmt_num(ho["r2"]), b_ho))
            L.append((_T("Viés (holdout externo)", "Bias (external holdout)"), _fmt_num(ho["bias"]), b_ho))
        else:
            out["avisos"].append(_T("Não há holdout externo nesta execução: o RMSEP abaixo vem da validação "
                                    "interna e não é o número publicável.",
                                    "There is no external holdout in this run: the RMSEP below comes from the "
                                    "internal validation and is not the publishable number."))
        L.append(("RMSECV", _fmt_num(po.get("rmsecv")), b_cv))
        L.append(("R2cv (Q2)", _fmt_num(po.get("r2cv"), 4), b_cv))
        L.append(("RMSEC", _fmt_num(po.get("rmsec")), b_cal))
        L.append(("R2cal", _fmt_num(po.get("r2cal"), 4), b_cal))
        L.append((_T("RMSEP (validação interna)", "RMSEP (internal validation)"), _fmt_num(po.get("rmsep")), b_int))
        L.append(("R2val", _fmt_num(po.get("r2val"), 4), b_int))
        L.append((_T("Viés (validação interna)", "Bias (internal validation)"), _fmt_num(po.get("bias"), 4), b_int))
        for chave, rot, nd in (("sep", "SEP", 3), ("rer", "RER", 1)):
            if po.get(chave) is not None:
                L.append((rot, _fmt_num(po[chave], nd), b_int))
        if po.get("rpd") is not None:
            L.append(("RPD", _fmt_num(po["rpd"], 2), b_int))
            faixa = _FAIXA_RPD.get(str(reg.get("rpd_faixa_txt") or ""))
            if faixa:
                L.append((_T("Faixa de uso do RPD (convenção NIR agrícola)", "RPD range of use (agricultural-NIR convention)"),
                          _T(*faixa), ""))
        if po.get("dmody_crit") is not None:
            L.append((_T("DModY crítico (SIMCA)", "Critical DModY (SIMCA)"), _fmt_num(po["dmody_crit"]), ""))
            L.append((_T("Amostras fora do DModY", "Samples outside DModY"),
                      str(int(po.get("n_fora_dmody") or 0)), ""))
        fom = reg.get("fom_pooled") or {}
        for chave, rot in (("lod", "LOD"), ("loq", "LOQ")):
            if fom.get(chave) is not None:
                L.append((_T(f"{rot} (modelo único)", f"{rot} (single pooled model)"),
                          _fmt_num(fom[chave], 2) + " %", ""))
        _rm_ref = ho["rmsep"] if ho else po.get("rmsep")
        _lod_p = fom.get("lod")
        if _lod_p is not None and _rm_ref is not None and _lod_p < _rm_ref:
            out["avisos"].append(_T(
                f"LOD do modelo único ({_fmt_num(_lod_p, 2)}) abaixo do RMSEP ({_fmt_num(_rm_ref, 2)}): o LOD só com "
                "ruído de réplica é irreal; não o cite como limite de detecção, use o RMSEP como piso.",
                f"Pooled-model LOD ({_fmt_num(_lod_p, 2)}) is below the RMSEP ({_fmt_num(_rm_ref, 2)}): an LOD from "
                "replicate noise alone is unrealistic; do not cite it as a detection limit, use the RMSEP as the floor."))
        hm = reg.get("heatmap")
        if hm:
            L.append((_T(f"Combinações espécie x adulterante com R2cv >= {hm['limiar']:.2f}",
                         f"Species x adulterant combinations with R2cv >= {hm['limiar']:.2f}"),
                      f"{hm['n_ok']} / {hm['n_total']}", b_cv))
        L.append((_T("Amostras (espectros)", "Samples (spectra)"), n_amostras, ""))
        if not uma_classe and n_classes != default:
            L.append((_T("Espécies/classes", "Species/classes"), n_classes, ""))
            ba = pm["Balanced Accuracy (CV)"]
            L.append((_T("Identificação da espécie (PLS-DA): acurácia balanceada",
                         "Species identification (PLS-DA): balanced accuracy"), ba, b_cv))
            L.append((_T("Identificação da espécie (PLS-DA): VLs ótimas", "Species identification (PLS-DA): optimal LVs"),
                      pm["Optimal LVs"], ""))
        if esps:
            cab = [_T("Espécie", "Species"), _T("VLs", "LVs"), "RMSEP", "R2val", "LOD", "LOQ", "SEN", "SEL"]
            lin = [[str(e["especie"]), str(e["lvs"]), _fmt_num(e["rmsep"], 2), _fmt_num(e["r2val"], 3),
                    _fmt_num(e["lod"], 2), _fmt_num(e["loq"], 2), _fmt_num(e["sen"], 3), _fmt_num(e["sel"], 3)]
                   for e in esps]
            out["tabela_especie"] = (cab, lin)
            for e in esps:
                partes = []
                if e.get("lod_ic"):
                    partes.append(_T(f"IC {e['ic_conf']}% do LOD [{e['lod_ic']}], do LOQ [{e['loq_ic']}]",
                                     f"{e['ic_conf']}% CI of LOD [{e['lod_ic']}], of LOQ [{e['loq_ic']}]"))
                    partes.append(_T(f"faixa de validação [{e['faixa_validacao']}] (DP {e['dp_validacao']})",
                                     f"validation range [{e['faixa_validacao']}] (SD {e['dp_validacao']})"))
                pi = _pi_texto(e)
                if pi:
                    partes.append(pi)
                if partes:
                    out["notas"].append(f"{e['especie']}: " + "; ".join(partes) + ".")
                if e.get("lod") is not None and e.get("rmsep") is not None and e["lod"] < e["rmsep"]:
                    out["avisos"].append(_T(
                        f"{e['especie']}: LOD ({_fmt_num(e['lod'], 2)}) abaixo do RMSEP ({_fmt_num(e['rmsep'], 2)}): "
                        "o LOD só com ruído de réplica é irreal; não o cite como limite de detecção.",
                        f"{e['especie']}: LOD ({_fmt_num(e['lod'], 2)}) is below the RMSEP "
                        f"({_fmt_num(e['rmsep'], 2)}): an LOD from replicate noise alone is unrealistic; do not "
                        "cite it as a detection limit."))
            out["notas"].append(_T("LOD, LOQ e RMSEP estão na unidade da concentração de referência dos dados.",
                                   "LOD, LOQ and RMSEP are in the unit of the reference concentration in the data."))
        if not po and not ho:
            out["avisos"].append(_T("O resumo desta execução não traz as métricas de regressão.",
                                    "This run's summary has no regression metrics."))
    else:
        for k, v in pm.items():
            if k == "Preprocessing":
                continue
            if uma_classe and k in ("Balanced Accuracy (CV)", "AUC macro OvR", "R2Y", "Q2Y", "R2X", "Optimal LVs",
                                    "p-value (permutation)"):
                continue
            L.append((_rotulo_metrica(k), v, ""))
        if uma_classe:
            out["avisos"].append(_T("Há uma única classe nos dados: as métricas de classificação (acurácia, "
                                    "AUC, R2Y, Q2Y, R2X, permutação) não se aplicam e foram omitidas.",
                                    "The data have a single class: the classification metrics (accuracy, "
                                    "AUC, R2Y, Q2Y, R2X, permutation) do not apply and were omitted."))
    return out


#: palavras que o texto PT do .tex grava sem acento -> com acento (o artigo vai para revista).
_ACENTOS_PT = {
    "metodos": "métodos", "Metodos": "Métodos", "quimiometricos": "quimiométricos",
    "variaveis": "variáveis", "pre-processamento": "pré-processamento",
    "Pre-processamento": "Pré-processamento", "acuracia": "acurácia", "Acuracia": "Acurácia",
    "permutacao": "permutação", "Permutacao": "Permutação", "permutacoes": "permutações",
    "Introducao": "Introdução", "paragrafos": "parágrafos", "analitico": "analítico",
    "tecnica": "técnica", "tecnicas": "técnicas", "rapida": "rápida",
    "classificacao": "classificação", "autenticacao": "autenticação", "aquisicao": "aquisição",
    "resolucao": "resolução", "numero": "número", "comparacao": "comparação",
    "validacao": "validação", "Validacao": "Validação", "estatistica": "estatística",
    "Discussao": "Discussão", "metricas": "métricas", "Metricas": "Métricas",
    "Metrica": "Métrica", "interpretacao": "interpretação", "quimica": "química",
    "Conclusao": "Conclusão", "implicacoes": "implicações", "satisfatorio": "satisfatório",
    "estao": "estão", "nao": "não", "NAO": "NÃO", "execucao": "execução", "Atencao": "Atenção",
    "replicas": "réplicas", "fisico": "físico", "fisicas": "físicas", "Analise": "Análise",
    "Referencias": "Referências", "otimas": "ótimas", "possivel": "possível",
    "configuracao": "configuração", "Quimiometrica": "Quimiométrica", "Quimiometria": "Quimiometria",
}
_ACENTOS_RE = re.compile(r"(?<![:/\\\w])(" + "|".join(sorted(_ACENTOS_PT, key=len, reverse=True)) + r")(?!\w)")


def _acentuar(texto: str) -> str:
    """Acentua o texto PT do .tex (fora de rotulos `tab:...`, comandos e chaves)."""
    return _ACENTOS_RE.sub(lambda m: _ACENTOS_PT[m.group(1)], texto)


def _legenda_figura(caminho: str) -> str:
    """Legenda de uma figura no idioma dos relatorios (o nome do arquivo e' um id em PT)."""
    return _i18n.titulo_figura(caminho)

__all__ = [
    "generate_pdf_report",
    "generate_word_report",
    "generate_excel_report",
    "generate_latex_template",
    "generate_pptx_report",
]


def _avisos_modelo(resumo_raw: str) -> List[str]:
    """Avisos do modelo gravados no resumo (classes em rampa, roteamento fraco da quantificacao...) no
    idioma dos relatorios. Vazio quando a execucao nao teve nenhum."""
    return [_T(pt, en) for _cod, pt, en in avisos_do_modelo(resumo_raw)]


def _e_modo_prototipo(resumo_raw: str) -> bool:
    """True quando a execucao nao tem nenhuma garantia de agrupamento
    group-aware (achado B4-1, redesenhado no Bloco 8 de 2026-08-25).

    Ate' 2026-08-25 isto testava `mode == "imagem"` diretamente -- todo
    modo imagem carimbava, mesmo quando o usuario organizou a pasta por
    amostra fisica (nivel "high") ou forneceu o CSV de associacao (nivel
    "medium"), os dois com a MESMA garantia que dx/sintetico tem. Agora le
    o campo "Grouping guarantee" que `pipeline.executar()` grava no
    resumo_modelo.txt -- carimba SO' quando o nivel e' "none", nao mais
    todo o modo de entrada.
    """
    return extract_metric(
        resumo_raw, r"Grouping guarantee\s*[:=]\s*(\w+)", ""
    ).strip().lower() == "none"


# ──────────────────────────────────────────────────────────────────────────
# Execucao HSI (imageamento hiperespectral): nao ha' resumo_modelo.txt, so' Relatorios/hsi_relatorio.txt
# ──────────────────────────────────────────────────────────────────────────
def _e_execucao_hsi(pasta: str) -> bool:
    """True para a pasta de uma execucao HSI (tem hsi_relatorio.txt e nao tem resumo_modelo.txt)."""
    from guaraci.config import NOME_RELATORIOS
    rel = pasta_execucao(pasta, NOME_RELATORIOS)
    return (os.path.isfile(os.path.join(rel, "hsi_relatorio.txt"))
            and not os.path.isfile(os.path.join(rel, "resumo_modelo.txt")))


_NOTAS_HSI_EN = [
    (r"^Faixa: todas as (\d+) bandas \((.+)\)$", r"Range: all \1 bands (\2)"),
    (r"^Faixa: (\d+) bandas \((.+)\)$", r"Range: \1 bands (\2)"),
    (r"^Faixa \[(.+)\] IGNORADA: nao cobre o eixo do sensor \((.+) nm\); usadas todas as bandas$",
     r"Range [\1] IGNORED: it does not cover the sensor axis (\2 nm); all bands used"),
    (r"^Pre-processamento: SNV por pixel \+ centragem na media \(PLS\)$",
     "Pre-processing: per-pixel SNV + mean-centring (PLS)"),
    (r"^Pre-processamento: so' centragem na media \(PLS\)$", "Pre-processing: mean-centring only (PLS)"),
    (r"^Pre-processamento: so' centragem na media \(PLS\); preset '(.+)' nao se aplica ao modo hsi \(use snv_mc\)$",
     r"Pre-processing: mean-centring only (PLS); preset '\1' does not apply to the HSI mode (use snv_mc)"),
]


def _nota_hsi(nota: str) -> str:
    if _i18n.idioma() != "en":
        return nota
    for rx, rep in _NOTAS_HSI_EN:
        if re.match(rx, nota):
            return re.sub(rx, rep, nota)
    return nota


# ──────────────────────────────────────────────────────────────────────────
# Execucao multielementar (ICP-OES / ICP-MS): ANOVA por elemento, censura e QC
# ──────────────────────────────────────────────────────────────────────────
def _json_multielementar(pasta: str) -> Dict[str, Any]:
    """Relatorios/multielementar.json da execucao ({} se nao for uma execucao multielementar)."""
    import json
    from guaraci.config import NOME_RELATORIOS
    try:
        with open(os.path.join(pasta_execucao(pasta, NOME_RELATORIOS), "multielementar.json"), encoding="utf-8") as fh:
            dados = json.load(fh)
        return dados if isinstance(dados, dict) else {}
    except (OSError, ValueError):
        return {}


def _e_execucao_multielementar(resumo_raw: str, pasta: str) -> bool:
    return (extract_metric(resumo_raw, r"Input mode\s*[:=]\s*(\w+)", "").lower() == "multielementar"
            and bool(_json_multielementar(pasta)))


def _num_csv(v: Any, nd: int = 3, sci: bool = True) -> str:
    """Numero de um CSV do motor (virgula decimal) para a tabela; notacao cientifica so' para p muito pequeno."""
    try:
        f = float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return "-"
    if f != f:
        return "-"
    return f"{f:.2e}" if (sci and abs(f) < 10 ** -nd and f != 0) else f"{f:.{nd}f}"


_VERIF_QC = {
    "recuperacao_crm": {
        "sem_valor_certificado": ("NÃO calculada: o arquivo não tem a linha Tipo=CRM_CERT com os valores certificados.",
                                  "NOT computed: the file has no Tipo=CRM_CERT row with the certified values."),
        "sem_linhas_crm": ("NÃO calculada: há valores certificados mas nenhuma linha Tipo=CRM medida.",
                           "NOT computed: certified values exist but there is no measured Tipo=CRM row."),
    },
    "brancos_vs_lod": {
        "sem_brancos": ("NÃO verificada: o arquivo não tem linhas de branco (Tipo=branco).",
                        "NOT checked: the file has no blank rows (Tipo=branco)."),
        "sem_lod": ("NÃO verificada: o arquivo não tem a linha Tipo=LD com o limite de detecção de cada elemento.",
                    "NOT checked: the file has no Tipo=LD row with each element's detection limit."),
    },
    "padrao_interno": {
        "sem_padrao_interno": ("NÃO verificada: nenhum padrão interno foi informado (icp_padrao_interno).",
                               "NOT checked: no internal standard was given (icp_padrao_interno)."),
        "poucas_linhas": ("NÃO verificada: menos de 3 leituras do padrão interno.",
                          "NOT checked: fewer than 3 internal-standard readings."),
    },
}
_NOME_VERIF = {"recuperacao_crm": ("Recuperação do material de referência", "Reference-material recovery"),
               "brancos_vs_lod": ("Brancos contra o LD", "Blanks against the LD"),
               "padrao_interno": ("Deriva do padrão interno", "Internal-standard drift")}


def _blocos_multielementar(pasta: str, numero: int = 2) -> List[tuple]:
    """Secao multielementar do relatorio como blocos neutros: aviso de maturidade, censura, ANOVA por elemento
    (com a correcao de multiplas comparacoes que o motor aplicou), comparacoes par a par, sensibilidade a censura e QC."""
    import csv
    from guaraci.config import NOME_TABELAS
    d = _json_multielementar(pasta)
    if not d:
        return []
    anova, sens, qc = d.get("anova", {}), d.get("sensibilidade_censura", {}), d.get("qc", {})
    alpha = anova.get("alpha", 0.05)
    B: List[tuple] = [("h1", f"{numero}. " + _T("Perfil multielementar (ICP-OES / ICP-MS)", "Multi-element profile (ICP-OES / ICP-MS)"))]
    if d.get("selo"):
        B.append(("aviso", _T("EM VALIDAÇÃO: o mecanismo foi verificado em dados sintéticos; os padrões são ponto de "
                              "partida, com poucas unidades independentes por grupo.",
                              "UNDER VALIDATION: the mechanism was verified on synthetic data; defaults are a starting "
                              "point, with few independent units per group.")))
    excl = d.get("elementos_excluidos") or []
    pol = {"substituir": _T("substituir por LOD/sqrt(2)", "replace by LOD/sqrt(2)"),
           "excluir": _T("excluir a linha com valor censurado", "drop the row with a censored value")}.get(
        str(d.get("politica_censura")), str(d.get("politica_censura")))
    B += [("h2", _T("Dados e tratamento", "Data and treatment")),
          ("table", [_T("Item", "Item"), _T("Valor", "Value")], [
              [_T("Unidade", "Unit"), str(d.get("unidade", "-"))],
              [_T("Elementos modelados", "Modelled elements"), ", ".join(d.get("elementos_modelados", [])) or "-"],
              [_T("Elementos retirados por censura", "Elements removed by censoring"), ", ".join(excl) or _T("nenhum", "none")],
              [_T("Política de censura", "Censoring policy"),
               f"{pol} ({_T('limiar', 'threshold')} {float(d.get('limiar_censura', 0)):.0%})"],
              [_T("Transformação", "Transformation"),
               str(d.get("transformacao", "-")) if _i18n.idioma() != "en"
               else str(d.get("transformacao", "-")).replace("autoescala", "autoscaling")],
              [_T("Linhas descartadas (censura / ausência)", "Rows dropped (censoring / missing)"),
               f"{d.get('linhas_descartadas_pela_censura', 0)} / {d.get('linhas_descartadas_por_ausencia', 0)}"],
          ])]
    frac = {k: v for k, v in (d.get("fracao_censurada") or {}).items() if v}
    if frac:
        B.append(("p", _T("Fração censurada por elemento (só os que têm valores < LD): ",
                          "Censored fraction per element (only those with values < LD): ")
                  + "; ".join(f"{k} {v:.0%}" for k, v in frac.items()) + "."))
    # ANOVA por elemento
    tab_dir = pasta_execucao(pasta, NOME_TABELAS)
    linhas_anova: List[List[str]] = []
    metodos: set = set()
    try:
        with open(os.path.join(tab_dir, "multielementar_anova_por_elemento.csv"), encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter=";"):
                metodos.add(f"{r.get('metodo', '')} + {r.get('pos_teste', '')}".strip(" +"))
                try:
                    dif = float(str(r.get("p_fdr", "")).replace(",", ".")) < alpha
                except ValueError:
                    dif = False
                linhas_anova.append([r.get("elemento", ""), r.get("n_unidades", ""), _num_csv(r.get("F"), 2),
                                     _num_csv(r.get("p"), 4), _num_csv(r.get("p_fdr"), 4),
                                     f"{_num_csv(r.get('eta2'), 2, False)} [{_num_csv(r.get('eta2_ic95_inf'), 2, False)}; "
                                     f"{_num_csv(r.get('eta2_ic95_sup'), 2, False)}]",
                                     _T("sim", "yes") if dif else _T("não", "no")])
    except OSError:
        pass
    B.append(("h2", _T("ANOVA por elemento (unidade independente)", "ANOVA per element (independent unit)")))
    if linhas_anova:
        B.append(("table", [_T("Elemento", "Element"), _T("n unid.", "n units"), "F", "p", _T("p (FDR)", "p (FDR)"),
                            "eta2 [IC95%]", _T(f"Difere (FDR<{alpha})", f"Differs (FDR<{alpha})")], linhas_anova))
        B.append(("p", _T(f"Método: {', '.join(sorted(metodos)) or 'ANOVA'} sobre a unidade independente (as réplicas "
                          f"são agregadas na unidade). Correção de múltiplas comparações: Games-Howell dentro de cada "
                          f"elemento e Benjamini-Hochberg (FDR, alpha = {alpha}) entre os {len(linhas_anova)} elementos. "
                          f"Elementos com diferença: {', '.join(anova.get('elementos_com_diferenca_entre_grupos', [])) or 'nenhum'}.",
                          f"Method: {', '.join(sorted(metodos)) or 'ANOVA'} on the independent unit (replicates are "
                          f"aggregated into the unit). Multiple-comparison correction: Games-Howell within each element "
                          f"and Benjamini-Hochberg (FDR, alpha = {alpha}) across the {len(linhas_anova)} elements. "
                          f"Elements that differ: {', '.join(anova.get('elementos_com_diferenca_entre_grupos', [])) or 'none'}.")))
    else:
        B.append(("p", _T("Tabela da ANOVA não encontrada (Tabelas/multielementar_anova_por_elemento.csv).",
                          "ANOVA table not found (Tables/multielementar_anova_por_elemento.csv).")))
    if anova.get("recusados"):
        B.append(("p", _T("Elementos não testados: ", "Elements not tested: ")
                  + "; ".join(f"{k} ({v})" for k, v in anova["recusados"].items())))
    # comparacoes par a par que passam do alpha (ja ajustadas)
    pares: List[List[str]] = []
    try:
        with open(os.path.join(tab_dir, "multielementar_comparacoes.csv"), encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter=";"):
                try:
                    pa = float(str(r.get("p_ajustado", "")).replace(",", "."))
                except ValueError:
                    continue
                if pa < alpha:
                    pares.append([r.get("elemento", ""), f"{r.get('grupo_a', '')} x {r.get('grupo_b', '')}",
                                  _num_csv(r.get("diferenca"), 3),
                                  f"[{_num_csv(r.get('ic95_inf'), 3)}; {_num_csv(r.get('ic95_sup'), 3)}]", _num_csv(pa, 4)])
    except OSError:
        pass
    if pares:
        B += [("h3", _T("Comparações par a par significativas (p ajustado < alpha)", "Significant pairwise comparisons (adjusted p < alpha)")),
              ("table", [_T("Elemento", "Element"), _T("Grupos", "Groups"), _T("Diferença", "Difference"), "IC95% (simult.)",
                         _T("p ajustado", "adjusted p")], pares[:60])]
        if len(pares) > 60:
            B.append(("p", _T(f"(mostradas 60 de {len(pares)}; todas em Tabelas/multielementar_comparacoes.csv)",
                              f"(60 of {len(pares)} shown; all in Tables/multielementar_comparacoes.csv)")))
    # sensibilidade a censura
    B.append(("h2", _T("Sensibilidade à censura (< LD)", "Censoring sensitivity (< LD)")))
    if sens.get("muda") is None:
        B.append(("aviso", _T("Uma das duas políticas não pôde ser aplicada: ", "One of the two policies could not be applied: ")
                  + "; ".join(f"{k}: {v}" for k, v in (sens.get("motivo") or {}).items())))
    else:
        B.append(("p", _T("A análise foi refeita com as duas políticas (substituir por LOD/sqrt(2) e excluir). ",
                          "The analysis was redone with both policies (replace by LOD/sqrt(2) and drop). ")
                  + (_T("A conclusão MUDA entre elas: ", "The conclusion CHANGES between them: ") if sens.get("muda")
                     else _T("A conclusão NÃO muda entre elas.", "The conclusion does NOT change between them."))
                  + ("; ".join(f"{k}: substituir={v.get('substituir')}, excluir={v.get('excluir')}"
                               for k, v in (sens.get("diferencas") or {}).items()) if sens.get("muda") else "")))
        B.append(("p", _T("Elementos com diferença — substituir: ", "Elements that differ — replace: ")
                  + (", ".join(sens.get("significativos_substituir", [])) or _T("nenhum", "none"))
                  + _T("; excluir: ", "; drop: ") + (", ".join(sens.get("significativos_excluir", [])) or _T("nenhum", "none")) + "."))
    # QC
    B.append(("h2", _T("Controle de qualidade (QC)", "Quality control (QC)")))
    alertas = qc.get("alertas") or []
    if alertas:
        for a in alertas:
            B.append(("aviso", a))
    else:
        B.append(("p", _T("Nenhum alerta nas verificações que puderam ser feitas (abaixo).",
                          "No alert in the checks that could be run (below).")))
    ver = qc.get("verificacoes") or {}
    nao = [[_T(*_NOME_VERIF[k]), _T(*_VERIF_QC[k][v])] for k, v in ver.items() if k in _VERIF_QC and v in _VERIF_QC[k]]
    if nao:
        B.append(("table", [_T("Verificação", "Check"), _T("Situação", "Status")], nao))
    rec = qc.get("recuperacao_crm_pct")
    if rec:
        B.append(("table", [_T("Elemento", "Element"), _T("Recuperação do MRC (%)", "RM recovery (%)")],
                  [[k, "-" if v is None else f"{v:.1f}"] for k, v in rec.items()]))
        B.append(("p", _T("Recuperação = média medida / valor certificado (alerta fora de 80-120%; é alerta, não conformidade).",
                          "Recovery = mean measured / certified value (alert outside 80-120%; an alert, not a conformity claim).")))
    pi = qc.get("padrao_interno")
    if isinstance(pi, dict):
        B.append(("p", _T(f"Padrão interno {pi.get('elemento')}: deriva {pi.get('deriva_pct_por_linha', 0):+.2f}% por linha, "
                          f"amplitude {pi.get('amplitude_pct', 0):.1f}%.",
                          f"Internal standard {pi.get('elemento')}: drift {pi.get('deriva_pct_por_linha', 0):+.2f}% per row, "
                          f"range {pi.get('amplitude_pct', 0):.1f}%.")))
    ses = d.get("efeito_de_sessao") or {}
    if ses.get("testado"):
        B.append(("p", _T("Efeito de sessão (linhas de QC): elementos com efeito — ", "Session effect (QC rows): elements with an effect — ")
                  + (", ".join(ses.get("elementos_com_efeito_de_sessao", [])) or _T("nenhum", "none")) + "."))
    elif ses.get("motivo"):
        B.append(("p", _T("Efeito de sessão não testado (sem coluna de sessão ou menos de 4 linhas de QC em 2+ sessões).",
                          "Session effect not tested (no session column or fewer than 4 QC rows in 2+ sessions).")))
    return B


def _legenda_hsi(caminho: str) -> str:
    """Legenda de figura HSI (o nome do arquivo e' um id em PT; as de HSI nao estao no catalogo de titulos)."""
    nome = os.path.splitext(os.path.basename(caminho))[0]
    if nome.startswith("hsi_mapa_classificacao"):
        return _T("Mapa de classificação por pixel de uma gravação (predição fora da amostra)",
                  "Per-pixel classification map of one recording (out-of-sample prediction)")
    return _legenda_figura(caminho)


def _blocos_hsi(pasta: str, img_base: str = "") -> List[tuple]:
    """Conteudo do relatorio HSI como blocos neutros (relatorio_blocos): resumo do hsi_relatorio.json (ou .txt), tipo de validacao
    (CV por objeto ou holdout), calibracao, eixo de bandas, aviso de n pequeno de objetos avaliados, sens./espec. com IC por classe, ficha anti-vazamento (ramo HSI, no
    idioma do relatorio), figuras e referencias. `img_base` != "" grava os caminhos das figuras relativos a ele."""
    from types import SimpleNamespace
    from guaraci import assistente_vazamento as _av
    from guaraci.config import NOME_RELATORIOS
    from guaraci.resumo_parse import N_TESTE_HSI_INDICATIVO, ler_hsi_pasta
    rel = pasta_execucao(pasta, NOME_RELATORIOS)
    h = ler_hsi_pasta(rel)
    tipo = str(h.get("tipo") or "holdout")
    cv = tipo == "cv_objetos"
    n_av = _av._n_avaliados(h)
    n_tot = h.get("n_objetos_total")
    nd = h.get("n_dobras")
    forma = (_T("leave-one-object-out", "leave-one-object-out") if isinstance(nd, int) and isinstance(n_tot, int) and nd >= n_tot
             else _T(f"{nd} dobras", f"{nd} folds"))
    B: List[tuple] = [("h1", _T("Relatório do imageamento hiperespectral (HSI)", "Hyperspectral imaging (HSI) report"))]
    if cv:
        B.append(("p", _T("Classificação por pixel (PLS-DA) com decisão por objeto físico. Validação: cruzada por objeto "
                          "(todos os objetos são previstos por um modelo ajustado sem eles; as variáveis latentes são "
                          "reescolhidas dentro de cada treino). Não há validação externa por dia/lote/equipamento.",
                          "Per-pixel classification (PLS-DA) with a decision per physical object. Validation: per-object "
                          "cross-validation (every object is predicted by a model fitted without it; the latent variables "
                          "are re-chosen inside each training set). There is no external validation by day/batch/instrument.")))
    else:
        B.append(("p", _T("Classificação por pixel (PLS-DA) com decisão por objeto físico. Validação: um único holdout de "
                          "objetos (hsi_validation: holdout); não há validação externa por dia/lote e a métrica vem de poucos "
                          "objetos.",
                          "Per-pixel classification (PLS-DA) with a decision per physical object. Validation: a single object "
                          "holdout (hsi_validation: holdout); there is no external validation by day/batch and the metric "
                          "comes from few objects.")))
    garantia = {"high": _T("alta (subpasta por objeto)", "high (one subfolder per object)"),
                "medium": _T("média (CSV de associação)", "medium (association CSV)"),
                "none": _T("nenhuma (cada gravação é um objeto)", "none (each recording is an object)")}.get(
        str(h.get("garantia") or ""), str(h.get("garantia") or "-"))
    cal = h.get("calibracao") if isinstance(h.get("calibracao"), dict) else {}
    if cal.get("modo") == "branco_escuro":
        cal_txt = _T(f"branco/escuro, R = (I - D)/(W - D), {cal.get('n_cubos_calibrados', '-')} cubo(s)",
                     f"white/dark, R = (I - D)/(W - D), {cal.get('n_cubos_calibrados', '-')} cube(s)")
    else:
        cal_txt = _T("nenhuma (cubos usados como estão; presume-se reflectância)",
                     "none (cubes used as they are; reflectance is assumed)")
    eixo = h.get("eixo_bandas") if isinstance(h.get("eixo_bandas"), dict) else {}
    eixo_txt = ""
    if eixo:
        fx = eixo.get("faixa_nm")
        eixo_txt = (f"{eixo.get('n_bandas')} " + _T("bandas", "bands")
                    + (f", {fx[0]:g}-{fx[1]:g} nm" if isinstance(fx, (list, tuple)) and len(fx) == 2 else "")
                    + "; " + _T(f"cubos reamostrados: {eixo.get('n_reamostrados', 0)} (modo {eixo.get('modo')})",
                                f"resampled cubes: {eixo.get('n_reamostrados', 0)} (mode {eixo.get('modo')})"))
    fora = h.get("fora_da_amostra")
    linhas = [
        [_T("Gravações aceitas", "Recordings accepted"),
         f"{h.get('aceitas', '-')}/{h.get('total', '-')} " + _T(f"(rejeitadas no controle de qualidade: {h.get('rejeitadas')})",
                                                                  f"(rejected by quality control: {h.get('rejeitadas')})")],
        [_T("Variáveis latentes (regra de parcimônia, CV por objeto)", "Latent variables (parsimony rule, per-object CV)"),
         str(h.get("vls", "-"))],
        [_T("Garantia de agrupamento por objeto", "Object grouping guarantee"), garantia],
        [_T("Calibração radiométrica", "Radiometric calibration"), cal_txt],
    ] + ([[_T("Eixo de bandas", "Band axis"), eixo_txt]] if eixo_txt else []) + [
        [_T("Validação", "Validation"),
         (_T(f"CV por objeto ({forma}), estratificada por classe", f"per-object CV ({forma}), stratified by class") if cv
          else _T("holdout único de objetos", "single object holdout"))],
        [_T("Objetos avaliados fora da amostra", "Objects evaluated out of sample"),
         f"{n_av if n_av is not None else '-'}" + (f" / {n_tot}" if n_tot is not None else "")],
        [_T("Validação externa", "External validation"),
         _T(f"{h.get('n_teste_externo', 0) or 0} objetos (não há validação externa)",
            f"{h.get('n_teste_externo', 0) or 0} objects (no external validation)")],
    ] + ([[_T("Confiança por objeto", "Per-object confidence"),
           _T("fora da amostra (modelo que não viu o objeto)", "out of sample (model that did not see the object)") if fora
           else _T("dentro da amostra (otimista)", "in-sample (optimistic)")]] if fora is not None else []) + (
        [[_T("Acurácia por objeto (IC95%)", "Accuracy per object (95% CI)"),
          f"{h['acuracia']:.2f} [{h['acuracia_ic'][0]:.2f}; {h['acuracia_ic'][1]:.2f}]"]]
        if isinstance(h.get("acuracia"), (int, float)) and isinstance(h.get("acuracia_ic"), (list, tuple))
        and len(h["acuracia_ic"]) == 2 and all(isinstance(v, (int, float)) for v in h["acuracia_ic"]) else []
    ) + [[_T("Faixa espectral", "Spectral range") if n.startswith("Faixa") else _T("Pré-processamento", "Pre-processing"),
          _nota_hsi(n).split(": ", 1)[-1] if n.startswith(("Pre-processamento", "Faixa:")) else _nota_hsi(n)]
         for n in h.get("notas", [])]
    B += [("h2", _T("1. Resumo da execução", "1. Run summary")), ("table", [_T("Item", "Item"), _T("Valor", "Value")], linhas)]
    if isinstance(n_av, int) and n_av < N_TESTE_HSI_INDICATIVO:
        B.append(("aviso", _T(f"Atenção: só {n_av} objeto(s) avaliado(s) fora da amostra (menos de {N_TESTE_HSI_INDICATIVO}). "
                              f"Sensibilidade e especificidade abaixo são MÉTRICA INDICATIVA, não estimativa de desempenho: "
                              f"um objeto a mais ou a menos move a métrica em 1/{max(n_av, 1)}.",
                              f"Warning: only {n_av} object(s) evaluated out of sample (fewer than {N_TESTE_HSI_INDICATIVO}). "
                              f"Sensitivity and specificity below are an INDICATIVE METRIC, not a performance estimate: one "
                              f"object more or less moves the metric by 1/{max(n_av, 1)}.")))
    for a in h.get("avisos") or []:
        B.append(("aviso", str(a)))
    if h.get("exploratorio") or str(h.get("garantia") or "") == "none":
        B.append(("aviso", _T("Sem garantia de agrupamento por objeto: resultado exploratório.",
                              "No object grouping guarantee: exploratory result.")))
    classes = h.get("classes") or []
    if classes:
        def _f(v: Any, ic: Any = None) -> str:
            if v is None:
                return "n/d"
            t = f"{v:.2f}"
            if isinstance(ic, (list, tuple)) and len(ic) == 2 and all(isinstance(x, (int, float)) for x in ic):
                t += f" [{ic[0]:.2f}; {ic[1]:.2f}]"
            return t
        B += [("h2", _T("2. Sensibilidade e especificidade por classe (" + ("CV por objeto" if cv else "holdout interno de objetos") + ")",
                        "2. Sensitivity and specificity per class (" + ("per-object CV" if cv else "internal object holdout") + ")")),
              ("table", [_T("Classe", "Class"), _T("n objetos", "n objects"), _T("Sensibilidade [IC95%]", "Sensitivity [95% CI]"),
                         _T("Especificidade [IC95%]", "Specificity [95% CI]")],
               [[str(c["classe"]), "-" if c.get("n") is None else str(c["n"]), _f(c["sens"], c.get("sens_ic")),
                 _f(c["espec"], c.get("espec_ic"))] for c in classes]),
              ("p", _T("IC de Clopper-Pearson a 95%; cada objeto é uma tentativa independente"
                       + (" (aproximado na CV: as dobras compartilham treino)" if cv else "") + ". n/d = nenhuma amostra desta "
                       "classe entre os objetos avaliados (não é erro do modelo).",
                       "Clopper-Pearson 95% CI; each object is an independent trial"
                       + (" (approximate in CV: folds share training objects)" if cv else "") + ". n/a = no sample of this "
                       "class among the evaluated objects (not a model error)."))]
    # ficha anti-vazamento do ramo HSI, no idioma deste relatorio
    notas_txt = " ".join(h.get("notas", []))
    mpre = re.search(r"preset '([^']+)'", notas_txt)
    preset = "snv_mc" if "SNV por pixel" in notas_txt else (mpre.group(1) if mpre else "mc")
    cfg_h = SimpleNamespace(mode="hsi", output_folder=pasta, default_preprocessing=preset)
    its = _av.itens_hsi(cfg_h, None, "EN" if _i18n.idioma() == "en" else "PT")
    simb = {"ok": "ok", "atencao": _T("atenção", "attention"), "declarar": _T("declare", "declare")}
    B += [("h2", _T("3. Ficha anti-vazamento (fluxo HSI)", "3. Leakage sheet (HSI flow)")),
          ("p", _T("O que esta execução fez em cada fonte de vazamento (Kapoor & Narayanan, 2023). 'declare' é pergunta ao "
                   "autor: o programa não tem como saber.",
                   "What this run did for each leakage source (Kapoor & Narayanan, 2023). 'declare' is a question for the "
                   "author: the program cannot know.")),
          ("table", [_T("Estado", "State"), _T("Item", "Item"), _T("O que foi feito", "What was done")],
           [[simb.get(i.estado, i.estado), i.titulo, i.detalhe] for i in its])]
    imgs = _list_figures(pasta)
    if imgs:
        B.append(("h2", _T("4. Figuras", "4. Figures")))
        for im in imgs[:14]:
            B.append(("img", os.path.relpath(im, img_base).replace("\\", "/") if img_base else im, _legenda_hsi(im)))
    B += [("h2", _T("Referências", "References")),
          ("p", "Barnes, R.J., Dhanoa, M.S. & Lister, S.J. (1989) Standard normal variate transformation and de-trending "
                "of near-infrared diffuse reflectance spectra. Appl. Spectrosc. 43(5):772-777. doi:10.1366/0003702894202201"),
          ("p", "Kapoor, S. & Narayanan, A. (2023) Leakage and the reproducibility crisis in machine-learning-based "
                "science. Patterns 4:100804. doi:10.1016/j.patter.2023.100804"),
          ("p", "Otsu, N. (1979) A threshold selection method from gray-level histograms. IEEE Trans. Syst. Man Cybern. "
                "9(1):62-66.")]
    return B


def _pdf_blocos(pdf: Any, blocos: List[tuple], _a: Any) -> None:
    """Desenha blocos neutros (h1/h2/h3, p, aviso, table, img) num FPDF ja' com pagina aberta."""
    for b in blocos:
        tipo = b[0]
        if tipo in ("h1", "h2", "h3"):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", {"h1": 14, "h2": 12, "h3": 11}[tipo])
            pdf.set_text_color(30, 80, 140)
            pdf.multi_cell(0, 8, _a(b[1]), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
        elif tipo == "p":
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _a(b[1]), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif tipo == "aviso":
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(150, 25, 25)
            pdf.set_fill_color(253, 240, 240)
            pdf.multi_cell(0, 5, _a(b[1]), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        elif tipo == "table":
            cab, lin = b[1], b[2]
            n = len(cab)
            larg = [175 / n] * n
            if n == 3 and cab[0] in ("Estado", "State"):
                larg = [20, 55, 100]
            elif n == 2:
                larg = [75, 100]
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(30, 80, 140)
            pdf.set_text_color(255, 255, 255)
            for j, c in enumerate(cab):
                pdf.cell(larg[j], 6, _a(c), border=1, fill=True)
            pdf.ln(6)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 8)
            for i, row in enumerate(lin):
                pdf.set_fill_color(*((240, 245, 255) if i % 2 == 0 else (252, 252, 252)))
                x0, y0 = pdf.get_x(), pdf.get_y()
                alturas = []
                for j, c in enumerate(row):          # altura da linha = a maior celula quebrada
                    linhas_c = pdf.multi_cell(larg[j], 4.5, _a(c), border=0, align="L", dry_run=True, output="LINES")
                    alturas.append(max(1, len(linhas_c)) * 4.5)
                h_lin = max(alturas)
                if y0 + h_lin > pdf.page_break_trigger:
                    pdf.add_page()
                    x0, y0 = pdf.get_x(), pdf.get_y()
                for j, c in enumerate(row):
                    pdf.set_xy(x0 + sum(larg[:j]), y0)
                    pdf.rect(pdf.get_x(), y0, larg[j], h_lin, style="DF")
                    pdf.multi_cell(larg[j], 4.5, _a(c), border=0, align="L", new_x="RIGHT", new_y="TOP")
                pdf.set_xy(x0, y0 + h_lin)
            pdf.ln(2)
        elif tipo == "img":
            try:
                if pdf.get_y() + 95 > pdf.page_break_trigger:
                    pdf.add_page()
                y0 = pdf.get_y()
                pdf.image(b[1], x=15, y=y0, w=170, h=90)
                pdf.set_xy(15, y0 + 91)
                pdf.set_font("Helvetica", "I", 7)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(170, 5, _a(str(b[2])[:90]), align="C", new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
            except OSError as e:
                logging.getLogger(__name__).warning("PDF HSI: figura nao embutida (%s): %s", b[1], e)


def _pdf_hsi(pasta: str, projeto: Dict) -> io.BytesIO:
    """PDF da execucao HSI (blocos de `_blocos_hsi`, mesma capa e rodape dos outros PDFs)."""
    import unicodedata
    from fpdf import FPDF

    def _a(txt: str) -> str:
        t = str(txt)
        for de, para in (("—", "-"), ("–", "-"), ("→", "->"), ("≥", ">="), ("≤", "<="), ("’", "'"), ("“", '"'),
                         ("”", '"'), ("•", "-")):
            t = t.replace(de, para)
        return unicodedata.normalize("NFC", t).encode("latin-1", "ignore").decode("latin-1")

    blocos = _blocos_hsi(pasta)
    nome = projeto.get("nome") or os.path.basename(os.path.normpath(pasta))

    class _PDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(100, 100, 100)
            self.cell(130, 6, _a(nome)[:60])
            self.cell(0, 6, f"{_T('Gerado em', 'Generated')}: {time.strftime('%Y-%m-%d')}", align="R",
                      new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 200, 200)
            self.line(15, self.get_y(), 195, self.get_y())
            self.ln(2)
            self.set_text_color(0, 0, 0)

        def footer(self):
            if self.page_no() == 1:
                return
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(140, 140, 140)
            self.cell(0, 6, f"{_T('Página', 'Page')} {self.page_no()} / {{nb}}", align="C")

    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 20, 15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_fill_color(30, 80, 140)
    pdf.rect(0, 0, 210, 55, style="F")
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 12)
    pdf.cell(0, 12, _a(_T("Plataforma de Quimiometria", "Chemometrics Platform")), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_xy(15, 30)
    pdf.cell(0, 8, _a(_T("Imageamento hiperespectral (HSI)", "Hyperspectral imaging (HSI)")), align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(15, 65)
    pdf.set_font("Helvetica", "B", 17)
    pdf.multi_cell(180, 9, _a(nome), align="C")
    pdf.ln(8)
    pdf.set_fill_color(240, 244, 250)
    for label, valor in ((_T("Autor(es)", "Author(s)"), projeto.get("autor") or "-"),
                         (_T("Instituição", "Institution"), projeto.get("inst") or "-"),
                         (_T("Tipo de estudo", "Study type"), projeto.get("tipo") or "-"),
                         (_T("Data", "Date"), time.strftime("%Y-%m-%d")),
                         (_T("Pasta", "Folder"), os.path.basename(os.path.normpath(pasta)))):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(55, 7, _a(f"{label}:"), border="B", fill=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(125, 7, _a(str(valor))[:80], border="B", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(-30)
    pdf.set_fill_color(30, 80, 140)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "I", 8)
    pdf.rect(0, pdf.get_y(), 210, 30, style="F")
    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.cell(0, 5, f"{_T('Gerado por', 'Generated by')}: GUARACI {_APP_VERSION}", align="C")
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)

    _pdf_blocos(pdf, blocos, _a)
    buf = io.BytesIO()
    buf.write(pdf.output())
    buf.seek(0)
    return buf


def _docx_blocos(doc: Any, blocos: List[tuple]) -> None:
    """Escreve blocos neutros (h1/h2/h3, p, aviso, table, img) num Document do python-docx."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.image.exceptions import UnrecognizedImageError
    from docx.shared import Inches, Pt, RGBColor
    for b in blocos:
        tipo = b[0]
        if tipo in ("h1", "h2", "h3"):
            doc.add_heading(b[1], int(tipo[1]))
        elif tipo == "p":
            doc.add_paragraph(b[1])
        elif tipo == "aviso":
            r = doc.add_paragraph().add_run(b[1])
            r.bold = True
            r.font.color.rgb = RGBColor(150, 25, 25)
        elif tipo == "table":
            t = doc.add_table(rows=1 + len(b[2]), cols=len(b[1]))
            t.style = "Table Grid"
            for j, c in enumerate(b[1]):
                t.cell(0, j).text = str(c)
                t.cell(0, j).paragraphs[0].runs[0].bold = True
            for i, row in enumerate(b[2], 1):
                for j, c in enumerate(row):
                    t.cell(i, j).text = str(c)
            doc.add_paragraph()
        elif tipo == "img":
            try:
                doc.add_picture(b[1], width=Inches(5.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                leg = doc.add_paragraph(str(b[2]))
                leg.alignment = WD_ALIGN_PARAGRAPH.CENTER
                leg.runs[0].italic = True
                leg.runs[0].font.size = Pt(9)
            except (OSError, UnrecognizedImageError):
                doc.add_paragraph(f"[{b[2]} — {_T('imagem indisponível', 'image not available')}]")


def _docx_hsi(pasta: str, projeto: Dict) -> io.BytesIO:
    """Word da execucao HSI (blocos de `_blocos_hsi`)."""
    from docx import Document
    from docx.shared import Pt, RGBColor

    blocos = _blocos_hsi(pasta)
    doc = Document()
    for i, size in enumerate((20, 16, 13)):
        st = cast(Any, doc.styles[f"Heading {i + 1}"])
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(30, 80, 140)
        st.font.bold = True
    doc.add_heading(_T("Plataforma de Quimiometria", "Chemometrics Platform"), 0)
    doc.add_heading(projeto.get("nome") or os.path.basename(os.path.normpath(pasta)), 1)
    t_capa = doc.add_table(rows=5, cols=2)
    t_capa.style = "Table Grid"
    for i, (label, valor) in enumerate(((_T("Autor(es)", "Author(s)"), projeto.get("autor") or "-"),
                                        (_T("Instituição", "Institution"), projeto.get("inst") or "-"),
                                        (_T("Tipo de estudo", "Study type"), projeto.get("tipo") or "-"),
                                        (_T("Data", "Date"), time.strftime("%Y-%m-%d %H:%M")),
                                        (_T("Pasta", "Folder"), os.path.basename(os.path.normpath(pasta))))):
        t_capa.cell(i, 0).text = label
        t_capa.cell(i, 1).text = str(valor)
        t_capa.cell(i, 0).paragraphs[0].runs[0].bold = True
    doc.add_page_break()
    _docx_blocos(doc, blocos)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def _xlsx_hsi(pasta: str) -> io.BytesIO:
    """Excel da execucao HSI: Resumo (+ classes), Ficha anti-vazamento e o hsi_relatorio.txt bruto."""
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from guaraci.config import NOME_RELATORIOS
    blocos = _blocos_hsi(pasta)
    rel = pasta_execucao(pasta, NOME_RELATORIOS)
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = _T("Resumo", "Summary")
    hdr = PatternFill("solid", fgColor="1E508C")

    def _tabela(w, r0: int, cab, linhas) -> int:
        for j, c in enumerate(cab, 1):
            cell = w.cell(r0, j, c)
            cell.font = Font(bold=True, color="FFFFFF", size=10)
            cell.fill = hdr
        for i, row in enumerate(linhas, r0 + 1):
            for j, c in enumerate(row, 1):
                cell = w.cell(i, j, c)
                cell.font = Font(size=9)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        return r0 + len(linhas) + 2

    r = 1
    ficha_tab = None
    for b in blocos:
        if b[0] == "table" and b[1][0] in ("Estado", "State"):
            ficha_tab = b
            continue
        if b[0] in ("h1", "h2"):
            ws.cell(r, 1, b[1]).font = Font(bold=True, size=12 if b[0] == "h1" else 11)
            r += 1
        elif b[0] in ("p", "aviso") and not str(b[1]).startswith(("Barnes", "Kapoor", "Otsu")):
            c = ws.cell(r, 1, b[1])
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.font = Font(size=9, bold=b[0] == "aviso", color="962222" if b[0] == "aviso" else "000000")
            r += 1
        elif b[0] == "table":
            r = _tabela(ws, r, b[1], b[2])
    ws.column_dimensions["A"].width = 55
    ws.column_dimensions["B"].width = 60
    ws.column_dimensions["C"].width = 18
    if ficha_tab:
        w2 = wb.create_sheet(_T("Ficha_vazamento", "Leakage_sheet"))
        _tabela(w2, 1, ficha_tab[1], ficha_tab[2])
        w2.column_dimensions["A"].width = 12
        w2.column_dimensions["B"].width = 45
        w2.column_dimensions["C"].width = 110
    w3 = wb.create_sheet(_T("Relatorio_bruto", "Raw_report"))
    w3.cell(1, 1, "hsi_relatorio.txt").font = Font(bold=True)
    with open(os.path.join(rel, "hsi_relatorio.txt"), encoding="utf-8", errors="replace") as fh:
        for i, linha in enumerate(fh.read().splitlines(), 2):
            w3.cell(i, 1, linha).font = Font(name="Courier New", size=9)
    w3.column_dimensions["A"].width = 100
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def _tex_hsi(pasta: str) -> bytes:
    """LaTeX da execucao HSI (relatorio_blocos.renderizar_tex; figuras relativas a Relatorios/)."""
    from guaraci.config import NOME_RELATORIOS
    from guaraci.relatorio_blocos import renderizar_tex
    rel = pasta_execucao(pasta, NOME_RELATORIOS)
    return renderizar_tex(_blocos_hsi(pasta, img_base=rel)).encode("utf-8")


def generate_pdf_report(pasta: str, projeto: Dict,
                          max_figuras: int = 14) -> io.BytesIO:
    """
    Generates a complete PDF report with fpdf2.
    Structure: Cover | Metrics | Figures (2/page) | References.
    Returns BytesIO ready for st.download_button.
    """
    if _e_execucao_hsi(pasta):
        return _pdf_hsi(pasta, projeto)
    import unicodedata
    from fpdf import FPDF

    def _a(txt: str) -> str:
        """Texto para as fontes Latin-1 do fpdf2: mantem acentos do portugues (Latin-1) e troca
        a pontuacao fora dele (travessao, setas, >=) por equivalentes."""
        t = str(txt)
        for de, para in (("—", "-"), ("–", "-"), ("→", "->"), ("≥", ">="), ("≤", "<="),
                         ("’", "'"), ("“", '"'), ("”", '"'), ("•", "-")):
            t = t.replace(de, para)
        t = unicodedata.normalize("NFC", t)
        return t.encode("latin-1", "ignore").decode("latin-1")

    # ── Parse resumo_modelo.txt ───────────────────────────────────────
    resumo_raw = _ler_resumo(pasta) or ""

    sumario = _sumario_executivo(resumo_raw)
    # B4-1: carimbo de prototipo quando a execucao veio do mode imagem
    # (colorimetria digital) -- nao validado e sem mae_id, logo sem
    # validacao group-aware. Ver `_e_modo_prototipo`.
    prototipo = _e_modo_prototipo(resumo_raw)

    imgs = _list_figures(pasta)[:max_figuras]

    # ── PDF class ──────────────────────────────────────────────────────
    class RelatorioPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(100, 100, 100)
            proj_nome = _a(projeto.get("nome") or _T("Plataforma de Quimiometria", "Chemometrics Platform"))
            self.cell(130, 6, proj_nome[:60], border=0, align="L")
            self.cell(0, 6, f"{_T('Gerado em', 'Generated')}: {time.strftime('%Y-%m-%d')}",
                      border=0, align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 200, 200)
            self.line(15, self.get_y(), 195, self.get_y())
            self.ln(2)
            self.set_text_color(0, 0, 0)

        def footer(self):
            if self.page_no() == 1:
                return
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(140, 140, 140)
            self.cell(0, 6, f"{_T('Página', 'Page')} {self.page_no()} / {{nb}}", align="C")

    pdf = RelatorioPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 20, 15)
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── COVER ────────────────────────────────────────────────────────────
    pdf.add_page()
    # Blue top band
    pdf.set_fill_color(30, 80, 140)
    pdf.rect(0, 0, 210, 55, style="F")
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 12)
    pdf.cell(0, 12, _a(_T("Plataforma de Quimiometria", "Chemometrics Platform")), align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_xy(15, 30)
    pdf.cell(0, 8, "PLS-R  |  PCA" if sumario["modo"] == MODO_QUANTIFICACAO
             else "PLS-DA  |  PCA  |  OPLS-DA  |  DD-SIMCA", align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(15, 65)
    pdf.set_font("Helvetica", "B", 17)
    pdf.multi_cell(180, 9,
                   _a(projeto.get("nome") or _T("Relatório de Análise Quimiométrica",
                                                "Chemometric Analysis Report")),
                   align="C")

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_fill_color(240, 244, 250)
    campos_capa = [
        (_T("Autor(es)", "Author(s)"),          projeto.get("autor") or "-"),
        (_T("Instituição", "Institution"),      projeto.get("inst") or "-"),
        (_T("Tipo de estudo", "Study type"),    projeto.get("tipo") or "-"),
        (_T("Data", "Date"),                    time.strftime("%Y-%m-%d")),
        (_T("Pasta", "Folder"),                 os.path.basename(pasta)),
    ]
    for label, valor in campos_capa:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(55, 7, _a(f"{label}:"), border="B", fill=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(125, 7, _a(str(valor))[:80], border="B", fill=True,
                 new_x="LMARGIN", new_y="NEXT")

    obj = projeto.get("objetivo", "")
    if obj:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _T("Objetivo:", "Objective:"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(180, 5, _a(obj))

    # B4-1: carimbo de prototipo na CAPA (nao em nota de rodape) -- e' o
    # primeiro elemento que alguem ve ao abrir o PDF.
    if prototipo:
        pdf.ln(6)
        pdf.set_draw_color(170, 40, 40)
        pdf.set_fill_color(253, 240, 240)
        pdf.set_text_color(150, 25, 25)
        pdf.set_font("Helvetica", "B", 12)
        _tit_prot, _corpo_prot = _aviso_prototipo()
        pdf.multi_cell(180, 8, _a(_tit_prot),
                       border=1, align="C", fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(180, 5, _a(_corpo_prot),
                       border="LRB", align="C", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_draw_color(200, 200, 200)

    # Cover footer
    pdf.set_y(-30)
    pdf.set_fill_color(30, 80, 140)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "I", 8)
    pdf.rect(0, pdf.get_y(), 210, 30, style="F")
    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.cell(0, 5,
             f"{_T('Gerado por', 'Generated by')}: GUARACI {_APP_VERSION}",
             align="C")

    # ── SECTION 1: METRICS ───────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 80, 140)
    pdf.cell(0, 9, _a(_T("1. Resumo executivo", "1. Executive Summary")), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 80, 140)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(4)

    # Avisos do modelo (antes dos numeros: quem cita a tabela abaixo precisa ve-los primeiro)
    _avisos = _avisos_modelo(resumo_raw)
    if _avisos:
        pdf.set_draw_color(170, 40, 40)
        pdf.set_fill_color(253, 240, 240)
        pdf.set_text_color(150, 25, 25)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(180, 7, _a(_T("AVISOS DO MODELO - leia antes de citar os números",
                               "MODEL WARNINGS - read before quoting the numbers")),
                 border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        for _av in _avisos:
            pdf.multi_cell(180, 4.5, _a(_av), border="LRB", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_draw_color(200, 200, 200)
        pdf.ln(4)

    # Metrics table (o conteudo depende do objetivo: ver _sumario_executivo)
    pdf.set_text_color(0, 0, 0)

    def _celula(w, h, txt, fill=True, size=9, bold=False):
        """Uma celula que nunca vaza da coluna: reduz a fonte (ate 6,5 pt) e so' entao corta."""
        txt = _a(str(txt))
        while size > 6.5 and pdf.get_string_width(txt) > w - 2:
            size -= 0.5
            pdf.set_font("Helvetica", "B" if bold else "", size)
        while len(txt) > 3 and pdf.get_string_width(txt) > w - 2:
            txt = txt[:-2]
        pdf.cell(w, h, txt, border=1, fill=fill)
        pdf.set_font("Helvetica", "B" if bold else "", 9)

    larg: Tuple[int, ...] = (85, 30, 60) if sumario["tem_base"] else (100, 75)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(30, 80, 140)
    pdf.set_text_color(255, 255, 255)
    cab_tab = [_T("Métrica", "Metric"), sumario["cabecalho_valor"]] + ([_T("Base", "Basis")] if sumario["tem_base"] else [])
    for j, c in enumerate(cab_tab):
        _celula(larg[j], 7, c, bold=True)
    pdf.ln(7)
    pdf.set_text_color(0, 0, 0)
    for i, (rot, val, base) in enumerate(sumario["linhas"]):
        pdf.set_fill_color(240, 245, 255) if i % 2 == 0 else pdf.set_fill_color(252, 252, 252)
        pdf.set_font("Helvetica", "", 9)
        _celula(larg[0], 6, rot)
        _celula(larg[1], 6, val)
        if sumario["tem_base"]:
            _celula(larg[2], 6, base, size=8)
        pdf.ln(6)
    # pre-processamento completo (a cadeia inteira, ex.: SNV -> centragem na media), em linha propria
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 245, 255)
    pdf.multi_cell(sum(larg), 6, _a(f"{_rotulo_metrica('Preprocessing')}: {sumario['preproc']}"),
                   border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)

    for av in sumario["avisos"]:
        pdf.ln(2)
        pdf.set_text_color(150, 25, 25)
        pdf.multi_cell(175, 5, _a("(!) " + av), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    if sumario["tabela_especie"]:
        cab_e, lin_e = sumario["tabela_especie"]
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 80, 140)
        pdf.cell(0, 8, _a(_T("Figuras de mérito por espécie", "Figures of merit per species")),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        w_e = [55, 12, 18, 18, 18, 18, 18, 18]
        pdf.set_fill_color(30, 80, 140)
        pdf.set_text_color(255, 255, 255)
        for j, c in enumerate(cab_e):
            _celula(w_e[j], 6, c, bold=True)
        pdf.ln(6)
        pdf.set_text_color(0, 0, 0)
        for i, row in enumerate(lin_e):
            pdf.set_fill_color(240, 245, 255) if i % 2 == 0 else pdf.set_fill_color(252, 252, 252)
            for j, c in enumerate(row):
                _celula(w_e[j], 6, c)
            pdf.ln(6)
        pdf.set_font("Helvetica", "I", 8)
        for nt in sumario["notas"]:
            pdf.multi_cell(175, 4.5, _a(nt), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)

    # Quality criteria
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 80, 140)
    pdf.cell(0, 8, _a(sumario["titulo_criterios"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)
    for c in sumario["criterios"]:
        pdf.set_x(18)
        pdf.cell(4, 5, "-")
        pdf.multi_cell(173, 5, _a(c))

    # ── Multielementar (ICP): ANOVA por elemento, censura e QC ───────────
    icp = _e_execucao_multielementar(resumo_raw, pasta)
    n_fig = 3 if icp else 2
    if icp:
        pdf.add_page()
        pdf.set_text_color(0, 0, 0)
        _pdf_blocos(pdf, _blocos_multielementar(pasta, 2), _a)

    # ── SECTION 2: FIGURES ────────────────────────────────────────────────
    if imgs:
        # 2 figures per page, stacked vertically
        fig_w = 175
        fig_h = 100   # max height per figure
        cap_h = 6     # caption height

        for par in range(0, len(imgs), 2):
            pdf.add_page()
            if par == 0:
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(30, 80, 140)
                pdf.cell(0, 9, _T(f"{n_fig}. Figuras", f"{n_fig}. Figures"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_draw_color(30, 80, 140)
                pdf.set_line_width(0.5)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.set_line_width(0.2)
                pdf.ln(3)

            # Top figure
            y_top = pdf.get_y()
            nome1 = _legenda_figura(imgs[par])
            try:
                pdf.image(imgs[par], x=15, y=y_top, w=fig_w, h=fig_h)
            except OSError as _e_img:   # arquivo ausente/corrompido/formato invalido
                logging.getLogger(__name__).warning(
                    "PDF: figura nao embutida (%s): %s", imgs[par], _e_img)
            pdf.set_xy(15, y_top + fig_h + 1)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(fig_w, cap_h,
                     f"{_T('Fig.', 'Fig.')} {par + 1}: {_a(nome1[:70])}",
                     align="C", new_x="LMARGIN", new_y="NEXT")

            # Bottom figure (if it exists)
            if par + 1 < len(imgs):
                pdf.ln(3)
                y_bot = pdf.get_y()
                nome2 = _legenda_figura(imgs[par + 1])
                try:
                    pdf.image(imgs[par + 1], x=15, y=y_bot, w=fig_w, h=fig_h)
                except OSError as _e_img:
                    logging.getLogger(__name__).warning(
                        "PDF: figura nao embutida (%s): %s",
                        imgs[par + 1], _e_img)
                pdf.set_xy(15, y_bot + fig_h + 1)
                pdf.set_font("Helvetica", "I", 7)
                pdf.cell(fig_w, cap_h,
                         f"Fig. {par + 2}: {_a(nome2[:70])}",
                         align="C")

            pdf.set_text_color(0, 0, 0)

    # ── SECTION 3: REFERENCES ────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 80, 140)
    pdf.cell(0, 9, _a(_T(f"{n_fig + 1}. Referências", f"{n_fig + 1}. References")), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 80, 140)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)

    refs = [
        "Bylesjo, M. et al. (2006) OPLS discriminant analysis: combining the strengths of PLS-DA and SIMCA. J. Chemom. 20(8-10):341-351. doi:10.1002/cem.1006",
        "Chong, I.G. & Jun, C.H. (2005) Performance of some variable selection methods when multicollinearity is present. Chemom. Intell. Lab. Syst. 78(1-2):103-112. doi:10.1016/j.chemolab.2004.12.011",
        "Efron, B. & Tibshirani, R.J. (1993) An Introduction to the Bootstrap. CRC Press, Boca Raton.",
        "Eriksson, L. et al. (2008) CV-ANOVA for significance testing of PLS and OPLS models. J. Chemom. 22(11-12):594-600. doi:10.1002/cem.1187",
        "Pomerantsev, A.L. & Rodionova, O.Y. (2020) Popular decision rules in SIMCA: critical review. J. Chemom. 34(8):e3250. doi:10.1002/cem.3250",
        "Rajalahti, T. et al. (2009) Discriminating variable test and selectivity ratio plot: quantitative tools for interpretation and variable (biomarker) selection in complex spectral or chromatographic profiles. Anal. Chem. 81(7):2581-2590. doi:10.1021/ac802514y",
        "Rinnan, A. et al. (2009) Review of the most common pre-processing techniques for near-infrared spectra. TrAC Trends Anal. Chem. 28(10):1201-1222. doi:10.1016/j.trac.2009.07.007",
        "Tracy, N.D., Young, J.C. & Mason, R.L. (1992) Multivariate control charts for individual observations. J. Qual. Technol. 24(2):88-95. doi:10.1080/00224065.1992.12015232",
        "Wold, S. (1978) Cross-validatory estimation of the number of components in factor and principal components models. Technometrics 20(4):397-405. doi:10.1080/00401706.1978.10489693",
    ]
    if sumario["modo"] == MODO_QUANTIFICACAO:
        refs = _REFS_QUANTIFICACAO
    if icp:   # autoscaling de uma tabela de elementos nao e' o pre-processamento NIR de Rinnan
        refs = [r for r in refs if not r.startswith("Rinnan")] + _REFS_MULTIELEMENTAR_EXTRA
    for i, ref in enumerate(refs, 1):
        pdf.multi_cell(0, 5, f"[{i}] {_a(ref)}")
        pdf.ln(1)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, _a(_T(
                   f"Relatório gerado automaticamente pelo GUARACI {_APP_VERSION}.",
                   f"Report automatically generated by GUARACI {_APP_VERSION}.")))

    buf = io.BytesIO()
    buf.write(pdf.output())
    buf.seek(0)
    return buf


def generate_word_report(pasta: str, projeto: Dict,
                           max_figuras: int = 14) -> io.BytesIO:
    """
    Generates an editable Word report (.docx) with python-docx.
    Same structure as the PDF: cover, metrics, figures, references.
    """
    if _e_execucao_hsi(pasta):
        return _docx_hsi(pasta, projeto)
    from docx import Document
    from docx.image.exceptions import UnrecognizedImageError
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    resumo_raw = _ler_resumo(pasta) or ""

    sumario = _sumario_executivo(resumo_raw)
    imgs = _list_figures(pasta)[:max_figuras]

    doc = Document()

    # ── Heading styles ──
    for i, (level, size) in enumerate([(0, 20), (1, 16), (2, 13)]):
        style = cast(Any, doc.styles[f"Heading {i + 1}"])
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(30, 80, 140)
        style.font.bold = True

    # ── COVER ──────────────────────────────────────────────────────────────
    doc.add_heading(_T("Plataforma de Quimiometria", "Chemometrics Platform"), 0)
    doc.add_heading(projeto.get("nome") or _T("Relatório de Análise Quimiométrica",
                                              "Chemometric Analysis Report"), 1)

    t_capa = doc.add_table(rows=5, cols=2)
    t_capa.style = "Table Grid"
    campos_capa = [
        (_T("Autor(es)", "Author(s)"),          projeto.get("autor") or "-"),
        (_T("Instituição", "Institution"),      projeto.get("inst") or "-"),
        (_T("Tipo de estudo", "Study type"),    projeto.get("tipo") or "-"),
        (_T("Data", "Date"),                    time.strftime("%Y-%m-%d %H:%M")),
        (_T("Pasta", "Folder"),                 os.path.basename(pasta)),
    ]
    for i, (label, valor) in enumerate(campos_capa):
        c0 = t_capa.cell(i, 0)
        c1 = t_capa.cell(i, 1)
        c0.text = label
        c1.text = str(valor)
        c0.paragraphs[0].runs[0].bold = True

    obj = projeto.get("objetivo", "")
    if obj:
        doc.add_heading(_T("Objetivo", "Objective"), 2)
        doc.add_paragraph(obj)

    # B4-1: carimbo de prototipo na capa, antes da quebra de pagina.
    if _e_modo_prototipo(resumo_raw):
        p_av = doc.add_paragraph()
        p_av.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _tit_prot, _corpo_prot = _aviso_prototipo()
        r_tit = p_av.add_run(_tit_prot)
        r_tit.bold = True
        r_tit.font.size = Pt(13)
        r_tit.font.color.rgb = RGBColor(150, 25, 25)
        p_corpo = doc.add_paragraph()
        p_corpo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_corpo = p_corpo.add_run(_corpo_prot)
        r_corpo.font.size = Pt(9)
        r_corpo.font.color.rgb = RGBColor(150, 25, 25)

    doc.add_page_break()

    # ── SECTION 1: METRICS ─────────────────────────────────────────────────
    doc.add_heading(_T("1. Resumo executivo — métricas do modelo",
                       "1. Executive Summary — Model Metrics"), 1)

    _avisos = _avisos_modelo(resumo_raw)
    if _avisos:
        p_hav = doc.add_paragraph()
        r_hav = p_hav.add_run(_T("AVISOS DO MODELO - leia antes de citar os números",
                                 "MODEL WARNINGS - read before quoting the numbers"))
        r_hav.bold = True
        r_hav.font.color.rgb = RGBColor(150, 25, 25)
        for _av in _avisos:
            p_av2 = doc.add_paragraph(_av, style="List Bullet")
            for r in p_av2.runs:
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(150, 25, 25)

    ncol = 3 if sumario["tem_base"] else 2
    t_met = doc.add_table(rows=1 + len(sumario["linhas"]), cols=ncol)
    t_met.style = "Table Grid"
    hdr = t_met.rows[0].cells
    hdr[0].text = _T("Métrica", "Metric")
    hdr[1].text = sumario["cabecalho_valor"]
    if ncol == 3:
        hdr[2].text = _T("Base", "Basis")
    for cell in hdr:
        cell.paragraphs[0].runs[0].bold = True       # o cabecalho acabou de receber texto: tem uma corrida
        cell.paragraphs[0].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for i, (rot, val, base) in enumerate(sumario["linhas"], 1):
        t_met.cell(i, 0).text = rot
        t_met.cell(i, 1).text = str(val)
        if ncol == 3:
            t_met.cell(i, 2).text = base

    p_pp = doc.add_paragraph()
    r_pp = p_pp.add_run(f"{_rotulo_metrica('Preprocessing')}: ")
    r_pp.bold = True
    p_pp.add_run(sumario["preproc"])

    for av in sumario["avisos"]:
        p_av2 = doc.add_paragraph()
        r_av2 = p_av2.add_run("(!) " + av)
        r_av2.font.color.rgb = RGBColor(150, 25, 25)

    if sumario["tabela_especie"]:
        cab_e, lin_e = sumario["tabela_especie"]
        doc.add_heading(_T("Figuras de mérito por espécie", "Figures of merit per species"), 2)
        t_esp = doc.add_table(rows=1 + len(lin_e), cols=len(cab_e))
        t_esp.style = "Table Grid"
        for j, c in enumerate(cab_e):
            t_esp.cell(0, j).text = c
            t_esp.cell(0, j).paragraphs[0].runs[0].bold = True
        for i, row in enumerate(lin_e, 1):
            for j, c in enumerate(row):
                t_esp.cell(i, j).text = c
        for nt in sumario["notas"]:
            doc.add_paragraph(nt).runs[0].font.size = Pt(9)

    doc.add_heading(_T("Critérios de qualidade", "Quality criteria") if sumario["modo"] != MODO_QUANTIFICACAO
                    else _T("Como ler a quantificação", "How to read the quantification"), 2)
    for c in sumario["criterios"]:
        doc.add_paragraph(c, style="List Bullet")

    icp = _e_execucao_multielementar(resumo_raw, pasta)
    n_fig = 3 if icp else 2
    if icp:
        doc.add_page_break()
        _docx_blocos(doc, _blocos_multielementar(pasta, 2))

    # ── SECTION 2: FIGURES ──────────────────────────────────────────────────
    if imgs:
        doc.add_page_break()
        doc.add_heading(_T(f"{n_fig}. Figuras", f"{n_fig}. Figures"), 1)
        for i, img in enumerate(imgs, 1):
            nome_fig = _legenda_figura(img)
            try:
                doc.add_picture(img, width=Inches(5.5))
                p = doc.paragraphs[-1]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                leg = doc.add_paragraph(f"Fig. {i}: {nome_fig}")
                leg.alignment = WD_ALIGN_PARAGRAPH.CENTER
                leg.runs[0].italic = True
                leg.runs[0].font.size = Pt(9)
            except (OSError, UnrecognizedImageError):   # arquivo ausente/corrompido/formato invalido
                doc.add_paragraph(f"[Fig. {i}: {nome_fig} — "
                                  f"{_T('imagem indisponível', 'image not available')}]")
            if i % 2 == 0 and i < len(imgs):
                doc.add_page_break()

    # ── SECTION 3: REFERENCES ─────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading(_T(f"{n_fig + 1}. Referências", f"{n_fig + 1}. References"), 1)
    refs = [
        "Bylesjo, M. et al. (2006) OPLS discriminant analysis. J. Chemom. 20(8-10):341-351. doi:10.1002/cem.1006",
        "Chong, I.G. & Jun, C.H. (2005) VIP scores. Chemom. Intell. Lab. Syst. 78(1-2):103-112. doi:10.1016/j.chemolab.2004.12.011",
        "Efron, B. & Tibshirani, R.J. (1993) An Introduction to the Bootstrap. CRC Press.",
        "Eriksson, L. et al. (2008) CV-ANOVA for PLS/OPLS significance. J. Chemom. 22(11-12):594-600. doi:10.1002/cem.1187",
        "Pomerantsev, A.L. & Rodionova, O.Y. (2020) Popular decision rules in SIMCA. J. Chemom. 34(8):e3250.",
        "Rajalahti, T. et al. (2009) Selectivity Ratio. Anal. Chem. 81(7):2581-2590. doi:10.1021/ac802514y",
        "Rinnan, A. et al. (2009) NIR pre-processing review. TrAC 28(10):1201-1222.",
        "Tracy, N.D. et al. (1992) Multivariate control charts. J. Qual. Technol. 24(2):88-95.",
        "Wold, S. (1978) Cross-validatory estimation. Technometrics 20(4):397-405. doi:10.1080/00401706.1978.10489693",
    ]
    if sumario["modo"] == MODO_QUANTIFICACAO:
        refs = _REFS_QUANTIFICACAO
    if icp:
        refs = [r for r in refs if not r.startswith("Rinnan")] + _REFS_MULTIELEMENTAR_EXTRA
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph(f"[{i}] {ref}")
        p.paragraph_format.space_after = Pt(4)

    doc.add_paragraph()
    nota = doc.add_paragraph(_T(f"Relatório gerado pelo GUARACI {_APP_VERSION}.",
                                f"Report generated by GUARACI {_APP_VERSION}."))
    nota.runs[0].italic = True
    nota.runs[0].font.size = Pt(8)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def generate_excel_report(pasta: str) -> io.BytesIO:
    """
    Generates an Excel report with 4 sheets via openpyxl:
      - Metrics: metrics extracted from the summary
      - Identifiers: samples with T2, Q, class (pipeline CSV)
      - VIP_Selection: VIP/SR scores (pipeline CSV, if present)
      - Raw_Summary: full text of resumo_modelo.txt
    """
    if _e_execucao_hsi(pasta):
        return _xlsx_hsi(pasta)
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    resumo_raw = _ler_resumo(pasta) or ""

    sumario = _sumario_executivo(resumo_raw)

    _AZUL      = PatternFill("solid", fgColor="1E508C")
    _AZUL_CLAR = PatternFill("solid", fgColor="DCE6F5")
    _BRANCO    = PatternFill("solid", fgColor="FFFFFF")
    _HDR_FONT  = Font(bold=True, color="FFFFFF", size=10)
    _BODY_FONT = Font(size=9)
    _THIN      = Side(style="thin", color="AAAAAA")
    _BORDER    = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

    def _cabecalho(ws, colunas: List[str]):
        for j, col in enumerate(colunas, 1):
            cell = ws.cell(1, j, col)
            cell.font = _HDR_FONT
            cell.fill = _AZUL
            cell.alignment = Alignment(horizontal="center")
            cell.border = _BORDER

    def _preencher_df(ws, df: pd.DataFrame):
        _cabecalho(ws, list(df.columns))
        for i, row in enumerate(df.itertuples(index=False), 2):
            fill = _AZUL_CLAR if i % 2 == 0 else _BRANCO
            for j, val in enumerate(row, 1):
                c = ws.cell(i, j, val)
                c.font = _BODY_FONT
                c.fill = fill
                c.border = _BORDER

    def _preencher_df_ws(ws, df: pd.DataFrame, row_start: int = 1):
        """Variant of _preencher_df with configurable row_start."""
        cols = list(df.columns)
        for j, col in enumerate(cols, 1):
            cell = ws.cell(row_start, j, col)
            cell.font = _HDR_FONT; cell.fill = _AZUL
            cell.alignment = Alignment(horizontal="center"); cell.border = _BORDER
        for i, row in enumerate(df.itertuples(index=False), row_start + 1):
            fill = _AZUL_CLAR if i % 2 == 0 else _BRANCO
            for j, val in enumerate(row, 1):
                c = ws.cell(i, j, val)
                c.font = _BODY_FONT; c.fill = fill; c.border = _BORDER

    def _auto_width(ws):
        for col in ws.columns:
            max_len = max((len(str(c.value)) for c in col if c.value), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 40)

    wb = openpyxl.Workbook()

    # ── SHEET 1: Metrics ───────────────────────────────────────────────────
    ws1 = wb.active
    assert ws1 is not None
    ws1.title = _T("Métricas", "Metrics")
    tem_base = sumario["tem_base"]
    _cabecalho(ws1, [_T("Métrica", "Metric"),
                     _T("Valor", "Value") if tem_base else _T("Valor (VC interna)", "Value (internal CV)")]
               + ([_T("Base", "Basis")] if tem_base else []))
    linhas_x = list(sumario["linhas"]) + [(_rotulo_metrica("Preprocessing"), sumario["preproc"], "")]
    for i, (rot, val, base) in enumerate(linhas_x, 2):
        fill = _AZUL_CLAR if i % 2 == 0 else _BRANCO
        for j, val_c in enumerate([rot, val] + ([base] if tem_base else []), 1):
            c = ws1.cell(i, j, val_c)
            c.font = _BODY_FONT
            c.fill = fill
            c.border = _BORDER
    for k, av in enumerate(sumario["avisos"], len(linhas_x) + 3):
        ws1.cell(k, 1, "(!) " + av).font = Font(size=9, color="962222")
    _auto_width(ws1)
    _avisos = _avisos_modelo(resumo_raw)
    if _avisos:
        _linha_av = len(linhas_x) + len(sumario["avisos"]) + 4
        ws1.cell(_linha_av, 1, _T("AVISOS DO MODELO - leia antes de citar os números",
                                  "MODEL WARNINGS - read before quoting the numbers")).font = Font(
            bold=True, color="962222")
        for _k, _av in enumerate(_avisos, 1):
            c = ws1.cell(_linha_av + _k, 1, _av)
            c.font = Font(size=9, color="962222")
            c.alignment = Alignment(wrap_text=True, vertical="top")
            ws1.merge_cells(start_row=_linha_av + _k, start_column=1, end_row=_linha_av + _k, end_column=2)
            ws1.row_dimensions[_linha_av + _k].height = max(30, 13 * (len(_av) // 55 + 1))

    # ── SHEET 2: Identifiers ────────────────────────────────────────────────
    ws2 = wb.create_sheet(_T("Identificadores", "Identifiers"))
    id_csv = os.path.join(pasta_execucao(pasta, NOME_TABELAS), "amostras_identificadores.csv")
    if os.path.exists(id_csv):
        try:
            df_id = pd.read_csv(id_csv, sep=";", decimal=",")
            _preencher_df(ws2, df_id)
            _auto_width(ws2)
        except (pd.errors.ParserError, OSError, UnicodeDecodeError):
            ws2.cell(1, 1, _T("Erro ao ler amostras_identificadores.csv",
                              "Error reading amostras_identificadores.csv"))
    else:
        ws2.cell(1, 1, _T("Arquivo não encontrado (rode a análise antes).",
                          "File not found (run the analysis first)."))

    # ── SHEET 3: VIP_Selection ────────────────────────────────────────────────
    ws3 = wb.create_sheet(_T("Seleção_VIP", "VIP_Selection"))
    vip_csv = os.path.join(pasta_execucao(pasta, NOME_TABELAS), "etapa4_selecao_variaveis.csv")
    if os.path.exists(vip_csv):
        try:
            df_vip = pd.read_csv(vip_csv, sep=";", decimal=",")
            _preencher_df(ws3, df_vip)
            _auto_width(ws3)
        except (pd.errors.ParserError, OSError, UnicodeDecodeError):
            ws3.cell(1, 1, _T("Erro ao ler etapa4_selecao_variaveis.csv",
                              "Error reading etapa4_selecao_variaveis.csv"))
    else:
        ws3.cell(1, 1, _T("A etapa 4 (seleção de variáveis) não foi executada.",
                          "Step 4 (variable selection) was not executed."))

    # ── SHEET 4: Raw_Summary ───────────────────────────────────────────────
    ws4 = wb.create_sheet(_T("Resumo_bruto", "Raw_Summary"))
    ws4.cell(1, 1, _T("resumo_modelo.txt — conteúdo completo (registro técnico da execução)",
                      "resumo_modelo.txt — full content (technical run log)")).font = Font(bold=True)
    from guaraci.app_logic import load_summary_text   # o .txt no idioma da execucao, nao o canonico
    for i, linha in enumerate((load_summary_text(pasta) or resumo_raw).splitlines(), 2):
        ws4.cell(i, 1, linha).font = Font(name="Courier New", size=9)
    ws4.column_dimensions["A"].width = 80

    if _e_execucao_multielementar(resumo_raw, pasta):
        ws_m = wb.create_sheet(_T("Multielementar", "Multi_element"))
        r_m = 1
        for b in _blocos_multielementar(pasta, 2):
            if b[0] in ("h1", "h2", "h3"):
                ws_m.cell(r_m, 1, b[1]).font = Font(bold=True, size=12)
                r_m += 1
            elif b[0] in ("p", "aviso"):
                ws_m.cell(r_m, 1, b[1]).font = Font(size=9, bold=b[0] == "aviso",
                                                    color="962222" if b[0] == "aviso" else "000000")
                r_m += 1
            elif b[0] == "table":
                _preencher_df_ws(ws_m, pd.DataFrame(b[2], columns=b[1]), row_start=r_m)
                r_m += len(b[2]) + 2
        _auto_width(ws_m)

    if sumario["tabela_especie"]:
        cab_e, lin_e = sumario["tabela_especie"]
        ws_e = wb.create_sheet(_T("Mérito_por_espécie", "Merit_per_species"))
        _preencher_df(ws_e, pd.DataFrame(lin_e, columns=cab_e))
        for k, nt in enumerate(sumario["notas"], len(lin_e) + 3):
            ws_e.cell(k, 1, nt).font = Font(size=9, italic=True)
        _auto_width(ws_e)

    # ── SHEET 5: Benchmark (if it exists) ─────────────────────────────────
    bench_csv = os.path.join(pasta_execucao(pasta, NOME_TABELAS), "benchmark_classifiers.csv")
    mc_csv    = os.path.join(pasta_execucao(pasta, NOME_TABELAS), "monte_carlo_cv.csv")
    if os.path.exists(bench_csv) or os.path.exists(mc_csv):
        ws5 = wb.create_sheet(_T("Comparação", "Benchmark"))
        row_cursor = 1
        if os.path.exists(bench_csv):
            try:
                df_bench = pd.read_csv(bench_csv, sep=";", decimal=",")
                ws5.cell(row_cursor, 1,
                         _T("Comparação automática — acurácia balanceada por classificador (GroupKFold)",
                            "Auto-benchmark — balanced accuracy by classifier (GroupKFold)")).font = Font(bold=True)
                row_cursor += 1
                _preencher_df_ws(ws5, df_bench, row_start=row_cursor)
                row_cursor += len(df_bench) + 3
                _auto_width(ws5)
            except (pd.errors.ParserError, OSError, UnicodeDecodeError):
                ws5.cell(row_cursor, 1, _T("Erro ao ler benchmark_classifiers.csv",
                                           "Error reading benchmark_classifiers.csv"))
                row_cursor += 2
        if os.path.exists(mc_csv):
            try:
                df_mc = pd.read_csv(mc_csv, sep=";", decimal=",")
                ws5.cell(row_cursor, 1,
                         _T("VC Monte Carlo — IC 95% por percentil",
                            "Monte Carlo CV — 95% CI by percentile")).font = Font(bold=True)
                row_cursor += 1
                _preencher_df_ws(ws5, df_mc, row_start=row_cursor)
                _auto_width(ws5)
            except (pd.errors.ParserError, OSError, UnicodeDecodeError):
                ws5.cell(row_cursor, 1, _T("Erro ao ler monte_carlo_cv.csv",
                                           "Error reading monte_carlo_cv.csv"))

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def generate_latex_template(pasta: str, projeto: Dict) -> bytes:
    """
    Generates a LaTeX template for a journal article (Elsevier-style class;
    adapt it to the target journal). Includes auto-filled metrics,
    \\includegraphics blocks for figures, and complete bibliography. The
    technique and the spectral range come from the run (profile and
    configuration), never from a fixed instrument or matrix.
    Texto no idioma dos relatorios (GUARACI_IDIOMA): artigo em portugues
    (babel brazil) ou em ingles.
    Returns UTF-8 bytes (.tex file).
    """
    if _e_execucao_hsi(pasta):
        return _tex_hsi(pasta)
    T = _T
    en = _i18n.idioma() == "en"

    resumo_raw = _ler_resumo(pasta) or ""

    def _ex(padrao: str, default: str = "-") -> str:
        return extract_metric(resumo_raw, padrao, default)

    def _esc(txt: str) -> str:
        """Escape LaTeX special characters."""
        for old, new in [
            ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
            ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
            ("{", r"\{"), ("}", r"\}"),
            ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
        ]:
            txt = txt.replace(old, new)
        return txt

    _pm = parse_model_metrics(resumo_raw)
    met = {
        "bal_acc":   _ex(r"[Bb]alanced.*?[:=]\s*([\d.]+)"),
        "auc":       _ex(r"ROC AUC macro.*?[:=]\s*([\d.]+)"),
        "r2y":       _ex(r"\bR2Y\b.*?[:=]\s*([\d.]+)"),
        "q2y":       _ex(r"\bQ2\b.*?[:=]\s*([\d.E+-]+)"),
        "r2x":       _ex(r"\bR2X\b.*?[:=]\s*([\d.]+)"),
        "lvs":       _ex(r"Optimal LVs.*?[:=]\s*(\d+)"),
        "perm_p":    _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
        "preproc":   _pm["Preprocessing"],
        "n_train":   _pm["n samples (training)"],
        "n_classes": _pm["n classes"],
        "cv_label":    _ex(r"Validation\s*[:=]\s*(.+)"),
        "group_aware": _ex(r"Group-aware \(mae_id\)\s*[:=]\s*(\w+)"),
        "faixa_min":   _ex(r"Faixa espectral[^:=\n]*[:=]\s*\[\s*([\d.]+)"),
        "faixa_max":   _ex(r"Faixa espectral[^:=\n]*[:=]\s*\[\s*[\d.]+\s*,\s*([\d.]+)"),
        "faixa_unid":  _ex(r"Faixa espectral \(([^)]*)\)\s*[:=]\s*\["),
        "tecnica":     _ex(r"Tecnica \(perfil\)\s*[:=]\s*(.+)"),
        "n_perm":      _ex(r"Permutation n_validos\s*[:=]\s*(\d+)"),
    }

    sumario = _sumario_executivo(resumo_raw)
    quant = sumario["modo"] == MODO_QUANTIFICACAO
    met["preproc"] = sumario["preproc"]
    icp = _e_execucao_multielementar(resumo_raw, pasta)      # a cadeia inteira (ex.: SNV -> centragem na media), nao so' a 1a palavra

    # B4-1: mode="imagem" (colorimetria digital) e' PROTOTIPO nao validado e
    # nao produz mae_id -- a validacao group-aware fica desligada. Sem este
    # carimbo, o .tex gerado e' tipograficamente identico ao de uma analise
    # analise espectral validada.
    _tit_prot, _corpo_prot = _aviso_prototipo()
    carimbo_prototipo = (
        "\n\\begin{center}\n"
        "\\fbox{\\parbox{0.9\\linewidth}{\\centering\\bfseries\n"
        f"{_tit_prot}\\\\[2pt]\n"
        f"\\normalfont {_corpo_prot}}}}}\n"
        "\\end{center}\n"
        if _e_modo_prototipo(resumo_raw) else "")
    # Faixa e unidade vem da execucao. Sem unidade declarada no perfil o texto
    # nao presume uma (\numrange); sem faixa no resumo, nao inventa numero.
    _unid_tex = {"cm⁻¹": "\\per\\centi\\meter", "cm-1": "\\per\\centi\\meter",
                 "nm": "\\nano\\meter", "min": "\\minute",
                 "ms": "\\milli\\second"}.get(met["faixa_unid"].strip())
    if met["faixa_min"] == "-" or met["faixa_max"] == "-":
        frase_faixa = T("% TODO: informe a faixa espectral (nao encontrada no resumo da execucao).\n"
                        "Os espectros foram adquiridos na faixa declarada na configuracao da execucao.",
                        "% TODO: report the spectral range (not found in the run summary).\n"
                        "Spectra were acquired over the range stated in the run configuration.")
    elif _unid_tex:
        frase_faixa = T(f"Os espectros foram adquiridos na faixa de "
                        f"\\SIrange{{{met['faixa_min']}}}{{{met['faixa_max']}}}{{{_unid_tex}}}.",
                        f"Spectra were acquired in the range "
                        f"\\SIrange{{{met['faixa_min']}}}{{{met['faixa_max']}}}{{{_unid_tex}}}.")
    elif met["faixa_unid"].strip() == "ppm":     # sem macro de siunitx portavel entre versoes
        frase_faixa = T(f"Os espectros foram adquiridos na faixa de "
                        f"\\numrange{{{met['faixa_min']}}}{{{met['faixa_max']}}}~ppm.",
                        f"Spectra were acquired in the range "
                        f"\\numrange{{{met['faixa_min']}}}{{{met['faixa_max']}}}~ppm.")
    else:
        frase_faixa = T(f"Os espectros foram adquiridos na faixa de "
                        f"\\numrange{{{met['faixa_min']}}}{{{met['faixa_max']}}} "
                        f"(unidade do eixo nao declarada no perfil).",
                        f"Spectra were acquired in the range "
                        f"\\numrange{{{met['faixa_min']}}}{{{met['faixa_max']}}} "
                        f"(axis unit not declared in the profile).")
    if icp:   # tabela de elementos: nao ha' faixa espectral nem espectro
        _und = _esc(str(_json_multielementar(pasta).get("unidade", "")))
        frase_faixa = T("% TODO: descreva o equipamento (ICP-OES ou ICP-MS), o preparo da amostra e as linhas/massas.\n"
                        f"As concentracoes dos elementos foram determinadas por ICP (unidade: {_und}).",
                        "% TODO: describe the instrument (ICP-OES or ICP-MS), sample preparation and lines/masses.\n"
                        f"Element concentrations were determined by ICP (unit: {_und}).")
    _tec = met["tecnica"].strip()
    tecnica_tex = f" ({_esc(_tec)})" if _tec and _tec != "-" else ""
    # a tabela de elementos do ICP nao e' espectroscopia: o modelo de artigo nao pode dizer que e'
    tec_en = "multi-element determination by ICP (ICP-OES/ICP-MS)" if icp else f"spectroscopy{tecnica_tex}"
    tec_pt = "determinacao multielementar por ICP (ICP-OES/ICP-MS)" if icp else f"espectroscopia{tecnica_tex}"
    tec_pt_art = "a determinacao multielementar por ICP (ICP-OES/ICP-MS)" if icp else f"a espectroscopia{tecnica_tex}"

    # B3-1 (auditoria 2026-08): NAO afirmar "group-aware cross-validation"
    # cravado no texto -- o pipeline pode cair para StratifiedKFold quando
    # mae_id esta indisponivel (ex.: modo_entrada="imagem", ver B4-1), e o
    # manuscrito nao pode alegar uma protecao que nao foi de fato aplicada
    # naquela execucao. `met['group_aware']`/`met['cv_label']` vem do MESMO
    # resumo_modelo.txt que grava cv_label (pipeline.py, ~linha 1957),
    # entao a frase abaixo reflete o que realmente rodou.
    if met["group_aware"].strip().lower() == "sim":
        frase_validacao = T(
            f"selecionadas por validacao cruzada agrupada "
            f"({_esc(met['cv_label'])}, agrupando as replicas tecnicas pelo "
            f"ponto de amostragem fisico para evitar vazamento de dados) "
            f"\\citep{{chong2005}}",
            f"selected by group-aware cross-validation "
            f"({_esc(met['cv_label'])}, grouping technical replicates by "
            f"physical sampling point to prevent data leakage) "
            f"\\citep{{chong2005}}")
    elif met["group_aware"].strip().lower() == "nao":
        frase_validacao = T(
            f"selecionadas por {_esc(met['cv_label'])}. "
            f"\\textbf{{Atencao: a validacao agrupada NAO foi aplicada nesta "
            f"execucao}} (sem identificador de grupo de replicas) -- os "
            f"resultados nao devem ser descritos como protegidos contra "
            f"vazamento entre replicas fisicas",
            f"selected by {_esc(met['cv_label'])}. "
            f"\\textbf{{Warning: group-aware validation was NOT applied in "
            f"this run}} (no replicate-group identifier available) -- "
            f"results should not be described as protected against "
            f"physical-replicate leakage")
    else:
        frase_validacao = T(
            f"selecionadas por {_esc(met['cv_label'])} "
            f"(nao foi possivel saber pelo resumo da execucao se a validacao "
            f"foi agrupada -- confira antes de submeter)",
            f"selected by {_esc(met['cv_label'])} "
            f"(group-aware status could not be determined from the run "
            f"summary -- verify before submission)")

    imgs = _list_figures(pasta)[:8]
    nome_proj = _esc(projeto.get("nome") or T("Analise Quimiometrica", "Chemometric Analysis"))
    autor     = _esc(projeto.get("autor") or T("Sobrenome, N.", "Surname, N."))
    inst      = _esc(projeto.get("inst", ""))

    # Metrics table
    linhas_met = [
        "        \\textbf{" + T("Metrica", "Metric") + "} & \\textbf{"
        + T("Valor (VC interna)", "Value (internal CV)") + "} \\\\",
        r"        \midrule",
        f"        {T('Acuracia balanceada', 'Balanced Accuracy')} & {met['bal_acc']} \\\\",
        f"        {T('AUC macro (um contra todos)', 'AUC macro (OvR)')} & {met['auc']} \\\\",
        f"        $R^2Y$            & {met['r2y']} \\\\",
        f"        $Q^2Y$            & {met['q2y']} \\\\",
        f"        $R^2X$            & {met['r2x']} \\\\",
        f"        {T('VLs otimas', 'Optimal LVs')} & {met['lvs']} \\\\",
        f"        $p$ ({T('permutacao', 'permutation')}) & {met['perm_p']} \\\\",
        f"        {T('Pre-processamento', 'Preprocessing')} & {_esc(met['preproc'])} \\\\",
        f"        $n$ {T('amostras', 'samples')} & {met['n_train']} \\\\",
        f"        $n$ {T('classes', 'classes')} & {met['n_classes']} \\\\",
    ]
    tabela = "\n".join(linhas_met)
    # Barra invertida nao pode ficar dentro de expressao de f-string no Python 3.11.
    cite_rinnan_en = "" if icp else " according to \\citet{rinnan2009}"
    cite_rinnan_pt = "" if icp else ", conforme \\citet{rinnan2009}"
    extra_icp = ""
    if icp:
        from guaraci.relatorio_blocos import renderizar_tex_corpo
        extra_icp = "\n" + renderizar_tex_corpo(_blocos_multielementar(pasta, 0)).replace(
            "\\section*{0. ", "\\subsection*{").replace("\\subsection*{", "\\subsection*{", 1) + "\n"
    colunas_tab = "ll"
    extra_quant = ""
    if quant:
        colunas_tab = "llp{0.34\\linewidth}"
        linhas_q = [
            "        \\textbf{" + T("Metrica", "Metric") + "} & \\textbf{" + T("Valor", "Value") + "} & \\textbf{"
            + T("Base", "Basis") + "} \\\\",
            r"        \midrule",
        ] + [f"        {_esc(rot)} & {_esc(str(val))} & {_esc(base)} \\\\" for rot, val, base in sumario["linhas"]
        ] + [f"        {T('Pre-processamento', 'Preprocessing')} & \\multicolumn{{2}}{{l}}{{{_esc(met['preproc'])}}} \\\\"]
        tabela = "\n".join(linhas_q)
        if sumario["tabela_especie"]:
            cab_e, lin_e = sumario["tabela_especie"]
            extra_quant = (
                "\n\\begin{table}[htbp]\n    \\centering\n    \\caption{"
                + T("Figuras de merito por especie (mesma unidade da concentracao de referencia).",
                    "Figures of merit per species (same unit as the reference concentration).")
                + "}\n    \\label{tab:merito}\n    \\begin{tabular}{l" + "r" * (len(cab_e) - 1) + "}\n        \\toprule\n"
                + "        " + " & ".join("\\textbf{" + _esc(c) + "}" for c in cab_e) + " \\\\\n        \\midrule\n"
                + "\n".join("        " + " & ".join(_esc(c) for c in row) + " \\\\" for row in lin_e)
                + "\n        \\bottomrule\n    \\end{tabular}\n\\end{table}\n")
            if sumario["notas"]:
                extra_quant += "\n" + " ".join(_esc(nt) for nt in sumario["notas"]) + "\n"
        for av in sumario["avisos"]:
            extra_quant += "\n\\textbf{" + T("Atencao: ", "Warning: ") + "}" + _esc(av) + "\n"

    # Figure blocks
    blocos_fig = []
    for i, img in enumerate(imgs, 1):
        nome_f = os.path.splitext(os.path.basename(img))[0]
        label  = nome_f.replace(" ", "_").replace("-", "_")
        # Caminho RELATIVO a pasta da execucao (barras normais): o absoluto levava o caminho da
        # maquina e o usuario do sistema para um arquivo feito para ser compartilhado (achado de
        # 2026-10-05, tests/test_saida_sem_caminho_da_maquina.py). \graphicspath resolve a partir
        # de Relatorios/ (onde `guaraci relatorio` grava) ou da propria pasta da execucao.
        img_tex = os.path.relpath(img, pasta).replace("\\", "/")
        blocos_fig.append(f"""
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.85\\linewidth]{{{img_tex}}}
    % TODO: {T('acrescente a interpretacao quimica.', 'add chemical interpretation.')}
    \\caption{{{_esc(_legenda_figura(img))}.}}
    \\label{{fig:{label}}}
\\end{{figure}}""")
    figs_block = "\n".join(blocos_fig)

    _babel = "english" if en else "brazil"
    if quant:
        _reg = parse_regressao(resumo_raw)
        _ho, _po, _esps = _reg["holdout"], _reg["pooled"], _reg["especies"]
        _lvs_q = ", ".join(f"{_esc(str(e['especie']))}: {e['lvs']}" for e in _esps) or "-"
        _rmsep_txt = (T(f"RMSEP de {_fmt_num(_ho['rmsep'])} no holdout externo (n = {_ho['n']}; "
                        f"$R^2$ de predicao = {_fmt_num(_ho['r2'])})",
                        f"RMSEP of {_fmt_num(_ho['rmsep'])} on the external holdout (n = {_ho['n']}; "
                        f"prediction $R^2$ = {_fmt_num(_ho['r2'])})")
                       if _ho else
                       T(f"RMSEP de {_fmt_num(_po.get('rmsep'))} na validacao interna (sem holdout externo)",
                         f"RMSEP of {_fmt_num(_po.get('rmsep'))} in internal validation (no external holdout)"))
        _cv_txt = (f"RMSECV = {_fmt_num(_po.get('rmsecv'))}; $R^2_{{cv}}$ = {_fmt_num(_po.get('r2cv'), 4)}")
        if en:
            corpo = f"""%% ── Abstract ─────────────────────────────────────────────────────────────
\\begin{{abstract}}
The analyte content was quantified by spectroscopy{tecnica_tex} combined with
partial least squares (PLS) regression. With {_esc(met['preproc'])} preprocessing
the models achieved {_rmsep_txt}; {_cv_txt}.
% TODO: respect the abstract word limit of the target journal.
\\end{{abstract}}

\\textbf{{Keywords:}} spectroscopy; PLS regression; chemometrics; quantification.

%% ── Introduction ──────────────────────────────────────────────────────────
\\section{{Introduction}}
% TODO: contextualize, justify the method and state the objective in ~3 paragraphs.
% TODO: state the analytical problem, the analyte and the sample type studied.

%% ── Material and Methods ──────────────────────────────────────────────────
\\section{{Material and Methods}}

\\subsection{{Samples and spectral acquisition}}
% TODO: describe number of samples, instrument (make and model),
% resolution, number of scans, temperature, reference method for the content.
{frase_faixa}

\\subsection{{Spectral preprocessing}}
Preprocessing: {_esc(met['preproc'])} \\citep{{rinnan2009}}.

\\subsection{{PLS regression}}
PLS regression models were calibrated per species (latent variables: {_lvs_q}), {frase_validacao}.

\\subsection{{Validation and figures of merit}}
Performance was assessed by:
\\begin{{itemize}}
    \\item \\textbf{{External holdout}}: samples never used in calibration or in choosing the number of latent variables (RMSEP, prediction $R^2$ and bias);
    \\item \\textbf{{Cross-validation}}: RMSECV and $R^2_{{cv}}$ \\citep{{wold1978}};
    \\item \\textbf{{Figures of merit}}: LOD, LOQ, sensitivity and selectivity \\citep{{valderrama2009}}, with confidence intervals for LOD/LOQ \\citep{{allegrini2014}}.
\\end{{itemize}}

%% ── Results and Discussion ──────────────────────────────────────────────
\\section{{Results and Discussion}}

\\subsection{{Model performance}}
Performance metrics are presented in Table~\\ref{{tab:metricas}}.

\\begin{{table}}[htbp]
    \\centering
    \\caption{{PLS regression performance metrics.}}
    \\label{{tab:metricas}}
    \\begin{{tabular}}{{{colunas_tab}}}
        \\toprule
{tabela}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
{extra_quant}
\\subsection{{Figures}}
% Figures generated by the pipeline. Insert chemical interpretation in each caption.

{figs_block}

%% ── Conclusion ───────────────────────────────────────────────────────────
\\section{{Conclusion}}
% TODO: summarise results and implications in 1-2 paragraphs.
% Report the external-holdout RMSEP as the performance of the method; the cross-validation
% figures are optimistic.
"""
        else:
            corpo = f"""%% ── Resumo ───────────────────────────────────────────────────────────────
\\begin{{abstract}}
O teor do analito foi quantificado por espectroscopia{tecnica_tex} combinada a
regressao por minimos quadrados parciais (PLS). Com pre-processamento {_esc(met['preproc'])}
os modelos obtiveram {_rmsep_txt}; {_cv_txt}.
% TODO: respeite o limite de palavras do resumo da revista-alvo.
\\end{{abstract}}

\\textbf{{Palavras-chave:}} espectroscopia; regressao PLS; quimiometria; quantificacao.

%% ── Introducao ────────────────────────────────────────────────────────────
\\section{{Introducao}}
% TODO: contextualize, justifique o metodo e declare o objetivo em ~3 paragrafos.
% TODO: descreva o problema analitico, o analito e o tipo de amostra estudado.

%% ── Material e Metodos ─────────────────────────────────────────────────────
\\section{{Material e Metodos}}

\\subsection{{Amostras e aquisicao espectral}}
% TODO: descreva numero de amostras, instrumento (marca e modelo),
% resolucao, numero de varreduras, temperatura, metodo de referencia do teor.
{frase_faixa}

\\subsection{{Pre-processamento espectral}}
Pre-processamento: {_esc(met['preproc'])} \\citep{{rinnan2009}}.

\\subsection{{Regressao PLS}}
Os modelos de regressao PLS foram calibrados por especie (variaveis latentes: {_lvs_q}), {frase_validacao}.

\\subsection{{Validacao e figuras de merito}}
O desempenho foi avaliado por:
\\begin{{itemize}}
    \\item \\textbf{{Holdout externo}}: amostras nunca usadas na calibracao nem na escolha do numero de variaveis latentes (RMSEP, $R^2$ de predicao e vies);
    \\item \\textbf{{Validacao cruzada}}: RMSECV e $R^2_{{cv}}$ \\citep{{wold1978}};
    \\item \\textbf{{Figuras de merito}}: LOD, LOQ, sensibilidade e seletividade \\citep{{valderrama2009}}, com intervalos de confianca para LOD/LOQ \\citep{{allegrini2014}}.
\\end{{itemize}}

%% ── Resultados e Discussao ──────────────────────────────────────────────
\\section{{Resultados e Discussao}}

\\subsection{{Desempenho do modelo}}
As metricas de desempenho estao na Tabela~\\ref{{tab:metricas}}.

\\begin{{table}}[htbp]
    \\centering
    \\caption{{Metricas de desempenho da regressao PLS.}}
    \\label{{tab:metricas}}
    \\begin{{tabular}}{{{colunas_tab}}}
        \\toprule
{tabela}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
{extra_quant}
\\subsection{{Figuras}}
% Figuras geradas pelo pipeline. Acrescente a interpretacao quimica em cada legenda.

{figs_block}

%% ── Conclusao ────────────────────────────────────────────────────────────
\\section{{Conclusao}}
% TODO: resuma os resultados e as implicacoes em 1-2 paragrafos.
% Relate o RMSEP do holdout externo como o desempenho do metodo; os numeros da validacao
% cruzada sao otimistas.
"""
    elif en:
        corpo = f"""%% ── Abstract ─────────────────────────────────────────────────────────────
\\begin{{abstract}}
The sample classes were investigated by {tec_en}
combined with chemometric methods (PLS-DA, PCA, OPLS-DA and DD-SIMCA).
The PLS-DA model with {_esc(met['lvs'])} latent variables and {_esc(met['preproc'])}
preprocessing achieved a balanced accuracy of {met['bal_acc']}
($R^2Y = {met['r2y']}$; $Q^2Y = {met['q2y']}$; $p = {met['perm_p']}$,
Y-randomization), evidencing significant discriminatory power.
% TODO: respect the abstract word limit of the target journal.
\\end{{abstract}}

\\textbf{{Keywords:}} spectroscopy; PLS-DA; chemometrics; classification.

%% ── Introduction ──────────────────────────────────────────────────────────
\\section{{Introduction}}
% TODO: contextualize, justify the method and state the objective in ~3 paragraphs.
% TODO: state the analytical problem and the sample type studied.
Spectroscopy combined with chemometrics has shown potential as a rapid
technique for sample classification and authentication
\\citep{{bylesjo2006, eriksson2008}}.

%% ── Material and Methods ──────────────────────────────────────────────────
\\section{{Material and Methods}}

\\subsection{{Samples and spectral acquisition}}
% TODO: describe number of samples, instrument (make and model),
% resolution, number of scans, temperature.
{frase_faixa}

\\subsection{{{'Preprocessing' if icp else 'Spectral preprocessing'}}}
The {_esc(met['preproc'])} preprocessing was selected based on pipeline comparison
(balanced accuracy in CV validation){cite_rinnan_en}.

\\subsection{{PLS-DA modelling}}
The PLS-DA model was calibrated with {met['lvs']} latent variables, {frase_validacao}.

\\subsection{{Statistical validation}}
Robustness was assessed by:
\\begin{{itemize}}
    \\item \\textbf{{Y-randomization}} ({_esc(met['n_perm'])} permutations) \\citep{{eriksson2008}};
    \\item \\textbf{{Wold's test}} (intercepts $R^2Y < 0.40$ and $Q^2Y < 0.05$);
    \\item \\textbf{{CV-ANOVA}} by Eriksson \\citep{{eriksson2008}};
    \\item \\textbf{{BCa Bootstrap}} 95\\% for balanced accuracy \\citep{{efron1993}}.
\\end{{itemize}}

%% ── Results and Discussion ──────────────────────────────────────────────
\\section{{Results and Discussion}}

\\subsection{{Model performance}}
Performance metrics are presented in Table~\\ref{{tab:metricas}}.

\\begin{{table}}[htbp]
    \\centering
    \\caption{{PLS-DA model performance metrics.}}
    \\label{{tab:metricas}}
    \\begin{{tabular}}{{ll}}
        \\toprule
{tabela}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}

{extra_icp}
\\subsection{{Figures}}
% Figures generated by the pipeline. Insert chemical interpretation in each caption.

{figs_block}

%% ── Conclusion ───────────────────────────────────────────────────────────
\\section{{Conclusion}}
% TODO: summarise results and implications in 1-2 paragraphs.
The PLS-DA model with {_esc(met['preproc'])} preprocessing achieved satisfactory
performance ($Q^2Y = {met['q2y']}$; $p = {met['perm_p']}$), demonstrating the
viability of {tec_en} for rapid classification of the studied samples.
"""
    else:
        corpo = f"""%% ── Resumo ───────────────────────────────────────────────────────────────
\\begin{{abstract}}
As classes de amostras foram investigadas por {tec_pt}
combinada a metodos quimiometricos (PLS-DA, PCA, OPLS-DA e DD-SIMCA).
O modelo PLS-DA com {_esc(met['lvs'])} variaveis latentes e pre-processamento
{_esc(met['preproc'])} alcancou acuracia balanceada de {met['bal_acc']}
($R^2Y = {met['r2y']}$; $Q^2Y = {met['q2y']}$; $p = {met['perm_p']}$,
permutacao de Y), evidenciando poder discriminante significativo.
% TODO: respeite o limite de palavras do resumo da revista-alvo.
\\end{{abstract}}

\\textbf{{Palavras-chave:}} espectroscopia; PLS-DA; quimiometria; classificacao.

%% ── Introducao ────────────────────────────────────────────────────────────
\\section{{Introducao}}
% TODO: contextualize, justifique o metodo e declare o objetivo em ~3 paragrafos.
% TODO: descreva o problema analitico e o tipo de amostra estudado.
A espectroscopia combinada a quimiometria tem mostrado potencial como tecnica
rapida para classificacao e autenticacao de amostras
\\citep{{bylesjo2006, eriksson2008}}.

%% ── Material e Metodos ─────────────────────────────────────────────────────
\\section{{Material e Metodos}}

\\subsection{{Amostras e aquisicao espectral}}
% TODO: descreva numero de amostras, instrumento (marca e modelo),
% resolucao, numero de varreduras, temperatura.
{frase_faixa}

\\subsection{{{'Pre-processamento' if icp else 'Pre-processamento espectral'}}}
O pre-processamento {_esc(met['preproc'])} foi escolhido pela comparacao de pipelines
(acuracia balanceada na validacao cruzada){cite_rinnan_pt}.

\\subsection{{Modelagem PLS-DA}}
O modelo PLS-DA foi calibrado com {met['lvs']} variaveis latentes, {frase_validacao}.

\\subsection{{Validacao estatistica}}
A robustez foi avaliada por:
\\begin{{itemize}}
    \\item \\textbf{{Permutacao de Y}} ({_esc(met['n_perm'])} permutacoes) \\citep{{eriksson2008}};
    \\item \\textbf{{Teste de Wold}} (interceptos $R^2Y < 0{{,}}40$ e $Q^2Y < 0{{,}}05$);
    \\item \\textbf{{CV-ANOVA}} de Eriksson \\citep{{eriksson2008}};
    \\item \\textbf{{Bootstrap BCa}} 95\\% para a acuracia balanceada \\citep{{efron1993}}.
\\end{{itemize}}

%% ── Resultados e Discussao ──────────────────────────────────────────────
\\section{{Resultados e Discussao}}

\\subsection{{Desempenho do modelo}}
As metricas de desempenho estao na Tabela~\\ref{{tab:metricas}}.

\\begin{{table}}[htbp]
    \\centering
    \\caption{{Metricas de desempenho do modelo PLS-DA.}}
    \\label{{tab:metricas}}
    \\begin{{tabular}}{{ll}}
        \\toprule
{tabela}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}

{extra_icp}
\\subsection{{Figuras}}
% Figuras geradas pelo pipeline. Acrescente a interpretacao quimica em cada legenda.

{figs_block}

%% ── Conclusao ────────────────────────────────────────────────────────────
\\section{{Conclusao}}
% TODO: resuma os resultados e as implicacoes em 1-2 paragrafos.
O modelo PLS-DA com pre-processamento {_esc(met['preproc'])} teve desempenho
satisfatorio ($Q^2Y = {met['q2y']}$; $p = {met['perm_p']}$), mostrando a
viabilidade d{tec_pt_art} para a classificacao rapida das amostras estudadas.
"""

    bib_quant = (
        "\\bibitem{valderrama2009}\nValderrama,~P., Braga,~J.W.B. \\& Poppi,~R.J. (2009).\n"
        "Estado da arte de figuras de merito em calibracao multivariada.\n"
        "\\textit{Quimica Nova}, 32(5), 1278--1287.\n\n"
        "\\bibitem{allegrini2014}\nAllegrini,~F. \\& Olivieri,~A.C. (2014).\n"
        "IUPAC-consistent approach to the limit of detection in partial least-squares calibration.\n"
        "\\textit{Analytical Chemistry}, 86(15), 7858--7866.\n\n"
        if quant else "")

    tex = f"""% ================================================================
% {T('Modelo LaTeX', 'LaTeX Template')} — {T('Plataforma de Quimiometria', 'Chemometrics Platform')} {_APP_VERSION}
% {T('Classe de artigo no estilo Elsevier: adapte a classe e o estilo a revista-alvo',
     'Elsevier-style article class: adapt the class and style to the target journal')}
% {T('Gerado em', 'Generated')}: {time.strftime('%Y-%m-%d %H:%M')}
% ================================================================

\\documentclass[12pt,a4paper]{{article}}

%% Packages
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[{_babel}]{{babel}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{graphicx}}
\\graphicspath{{{{../}}{{./}}}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage[colorlinks=true,citecolor=blue,linkcolor=blue]{{hyperref}}
\\usepackage{{geometry}}
\\usepackage{{caption}}
\\usepackage{{float}}
\\usepackage{{siunitx}}
\\usepackage{{natbib}}

\\geometry{{a4paper, left=2.5cm, right=2.5cm, top=3cm, bottom=3cm}}
\\captionsetup{{font=small, labelfont=bf}}
\\setlength{{\\parindent}}{{0.5cm}}

%% Metadata
\\title{{{nome_proj}}}
\\author{{{autor} \\\\
         \\small {inst}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle
{carimbo_prototipo}

{corpo}
%% {T('(sem secao de agradecimentos: o modelo nao afirma apoio, financiamento nem vinculo de ninguem; o autor acrescenta a sua)', '(no thanks section on purpose: the template claims no support or affiliation for anyone; the author adds their own)')}

%% ── {T('Referencias', 'References')} ─────────────────────────────────────────────────────────
\\bibliographystyle{{elsarticle-num}}  %% Elsevier numeric style
%% \\bibliographystyle{{apalike}}       %% {T('Alternativa APA', 'APA alternative')}

\\begin{{thebibliography}}{{99}}

\\bibitem{{bylesjo2006}}
Bylesjo,~M. et~al. (2006).
OPLS discriminant analysis: combining the strengths of PLS-DA and SIMCA.
\\textit{{Journal of Chemometrics}}, 20(8--10), 341--351.

\\bibitem{{chong2005}}
Chong,~I.G. \\& Jun,~C.H. (2005).
Performance of some variable selection methods when multicollinearity is present.
\\textit{{Chemometrics and Intelligent Laboratory Systems}}, 78(1--2), 103--112.

\\bibitem{{efron1993}}
Efron,~B. \\& Tibshirani,~R.J. (1993).
\\textit{{An Introduction to the Bootstrap}}. CRC Press.

\\bibitem{{eriksson2008}}
Eriksson,~L., Trygg,~J. \\& Wold,~S. (2008).
CV-ANOVA for significance testing of PLS and OPLS models.
\\textit{{Journal of Chemometrics}}, 22(11--12), 594--600.

\\bibitem{{pomerantsev2020}}
Pomerantsev,~A.L. \\& Rodionova,~O.Y. (2020).
Popular decision rules in SIMCA: critical review.
\\textit{{Journal of Chemometrics}}, 34(8), e3250. doi:10.1002/cem.3250.

\\bibitem{{rajalahti2009}}
Rajalahti,~T. et~al. (2009).
Discriminating variable test and selectivity ratio plot.
\\textit{{Analytical Chemistry}}, 81(7), 2581--2590.

\\bibitem{{rinnan2009}}
Rinnan,~\\AA., van~den~Berg,~F. \\& Engelsen,~S.B. (2009).
Review of the most common pre-processing techniques for near-infrared spectra.
\\textit{{TrAC Trends in Analytical Chemistry}}, 28(10), 1201--1222.

\\bibitem{{tracy1992}}
Tracy,~N.D., Young,~J.C. \\& Mason,~R.L. (1992).
Multivariate control charts for individual observations.
\\textit{{Journal of Quality Technology}}, 24(2), 88--95.

{bib_quant}\\bibitem{{wold1978}}
Wold,~S. (1978).
Cross-validatory estimation of the number of components.
\\textit{{Technometrics}}, 20(4), 397--405.

\\end{{thebibliography}}

\\end{{document}}
"""
    if not en:
        tex = _acentuar(tex)
    return tex.encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────
# Execution with live feedback (background thread + progress bar)
# ──────────────────────────────────────────────────────────────────────────


def generate_pptx_report(pasta: str, projeto: Dict,
                           max_figuras: int = 12) -> io.BytesIO:
    """
    Generates a PowerPoint presentation with professional scientific design.
    Palette: Navy #0F172A + Accent #0369A1 (UI Pro Max — B2B Professional).
    Structure: Cover | Methodology | Metrics | Figures | Benchmark | Conclusions
    """
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    # ── UI Pro Max — B2B Professional palette ──────────────────────────────
    _NAVY   = RGBColor(0x0F, 0x17, 0x2A)
    _SLATE  = RGBColor(0x33, 0x41, 0x55)
    _ACCENT = RGBColor(0x03, 0x69, 0xA1)
    _LIGHT  = RGBColor(0xF8, 0xFA, 0xFC)
    _WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
    _MUTED  = RGBColor(0x64, 0x74, 0x8B)

    resumo_raw = _ler_resumo(pasta) or ""

    def _ex(padrao: str, default: str = "—") -> str:
        return extract_metric(resumo_raw, padrao, default)

    sumario = _sumario_executivo(resumo_raw, default="—")
    quant = sumario["modo"] == MODO_QUANTIFICACAO
    _pm = parse_model_metrics(resumo_raw, default="—")
    metricas = {
        "Balanced Accuracy (CV)": _pm["Balanced Accuracy (CV)"],
        "AUC macro OvR":          _pm["AUC macro OvR"],
        "R2Y":                    _pm["R2Y"],
        "Q2Y":                    _pm["Q2Y"],
        "Optimal LVs":            _pm["Optimal LVs"],
        "Preprocessing":          sumario["preproc"],
        "N samples":              _pm["n samples (training)"],
        "N classes":              _pm["n classes"],
        "p permutation":          _pm["p-value (permutation)"],
    }
    _agrupada = _ex(r"Group-aware \(mae_id\)\s*[:=]\s*(\w+)", "").strip().lower() == "sim"

    # ── Layout helpers ─────────────────────────────────────────────────────
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)
    W = prs.slide_width   # Emu — assigned above, never None
    H = prs.slide_height  # Emu — assigned above, never None
    assert W is not None and H is not None  # satisfies Pylance Optional[Emu] stub

    def _rect(slide, l, t, w, h, fill_rgb: RGBColor):
        from pptx.util import Emu
        shape = slide.shapes.add_shape(1, Emu(l), Emu(t), Emu(w), Emu(h))
        shape.fill.solid(); shape.fill.fore_color.rgb = fill_rgb
        shape.line.fill.background()
        return shape

    def _txt(slide, texto: str, l, t, w, h,
             bold=False, size=18, color=_WHITE,
             align=PP_ALIGN.LEFT, wrap=True):
        tb = slide.shapes.add_textbox(Emu(l), Emu(t), Emu(w), Emu(h))
        tf = tb.text_frame; tf.word_wrap = wrap
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run(); run.text = texto
        run.font.bold = bold; run.font.size = Pt(size)
        run.font.color.rgb = color
        return tb

    def _slide_novo(layout_idx=6):
        layout = prs.slide_layouts[layout_idx]
        return prs.slides.add_slide(layout)

    def _fundo(slide, cor=_LIGHT):
        bg = slide.background; fill = bg.fill
        fill.solid(); fill.fore_color.rgb = cor

    def _barra_topo(slide, titulo: str):
        _rect(slide, 0, 0, int(W), int(Inches(1.0)), _NAVY)
        _rect(slide, 0, int(Inches(1.0)), int(W), int(Inches(0.06)), _ACCENT)
        _txt(slide, titulo, int(Inches(0.4)), int(Inches(0.15)),
             int(W - Inches(0.8)), int(Inches(0.75)),
             bold=True, size=22, color=_WHITE)

    def _rodape(slide):
        _rect(slide, 0, int(H - Inches(0.4)), int(W), int(Inches(0.4)), _SLATE)
        data_str = time.strftime("%Y-%m-%d")
        inst = projeto.get("inst", "")
        _txt(slide,
             f"{inst}  •  {_T('Plataforma de Quimiometria', 'Chemometrics Platform')}  •  {data_str}",
             int(Inches(0.3)), int(H - Inches(0.35)),
             int(W - Inches(0.6)), int(Inches(0.35)),
             size=9, color=_LIGHT, align=PP_ALIGN.CENTER)

    # ── SLIDE 1: Cover ─────────────────────────────────────────────────────
    slide1 = _slide_novo()
    _fundo(slide1, _NAVY)
    # Horizontal accent band
    _rect(slide1, 0, int(Inches(3.5)), int(W), int(Inches(0.12)), _ACCENT)
    # Main title
    _txt(slide1, projeto.get("nome") or _T("Plataforma de Quimiometria", "Chemometrics Platform"),
         int(Inches(1.0)), int(Inches(1.2)),
         int(W - Inches(2.0)), int(Inches(1.5)),
         bold=True, size=36, color=_WHITE)
    # Subtitle
    _txt(slide1, projeto.get("tipo") or _T("Análise quimiométrica", "Chemometric Analysis"),
         int(Inches(1.0)), int(Inches(2.8)),
         int(W - Inches(2.0)), int(Inches(0.8)),
         size=20, color=RGBColor(0xCB, 0xD5, 0xE1))
    # Metadata
    autor  = projeto.get("autor", "")
    inst   = projeto.get("inst", "")
    data_s = time.strftime("%Y-%m-%d")
    _txt(slide1, f"{autor}\n{inst}\n{data_s}",
         int(Inches(1.0)), int(Inches(4.0)),
         int(Inches(6.0)), int(Inches(2.0)),
         size=14, color=RGBColor(0x94, 0xA3, 0xB8))

    # ── SLIDE 2: Methodology ──────────────────────────────────────────────
    slide2 = _slide_novo()
    _fundo(slide2)
    _barra_topo(slide2, _T("Metodologia", "Methodology"))
    _rodape(slide2)

    _val_cv = (_T("Validação cruzada: agrupada por amostra física (mae_id), sem vazamento entre réplicas",
                  "Cross-validation: grouped by physical sample (mae_id), no leakage between replicates")
               if _agrupada else
               _T("Validação cruzada: SEM agrupamento por amostra física (ver model_card.md)",
                  "Cross-validation: NOT grouped by physical sample (see model_card.md)"))
    _uma = metricas["N classes"].strip() == "1"
    _cls_pt = "" if _uma else f"  |  {metricas['N classes']} classes"
    _cls_en = "" if _uma else f"  |  {metricas['N classes']} classes"
    if quant:
        _reg = parse_regressao(resumo_raw)
        _ho = _reg["holdout"]
        _lvs_esp = ", ".join(f"{e['especie']}: {e['lvs']}" for e in _reg["especies"])
        metod_items = [
            _T(f"Pré-processamento: {metricas['Preprocessing']}", f"Preprocessing: {metricas['Preprocessing']}"),
            (_T(f"Modelo: regressão PLS por espécie (VLs: {_lvs_esp})",
                f"Model: PLS regression per species (LVs: {_lvs_esp})") if _lvs_esp else
             _T("Modelo: regressão PLS", "Model: PLS regression")),
            _val_cv,
            (_T(f"Validação externa: holdout com n = {_ho['n']} (RMSEP do holdout é o número publicável)",
                f"External validation: holdout with n = {_ho['n']} (the holdout RMSEP is the publishable number)")
             if _ho else
             _T("Sem holdout externo nesta execução: o RMSEP vem da validação interna",
                "No external holdout in this run: the RMSEP comes from the internal validation")),
            _T("Figuras de mérito: LOD/LOQ com IC, sensibilidade e seletividade (Valderrama et al., 2009)",
               "Figures of merit: LOD/LOQ with CI, sensitivity and selectivity (Valderrama et al., 2009)"),
            _T(f"Dados: {metricas['N samples']} amostras{_cls_pt}",
               f"Data: {metricas['N samples']} samples{_cls_en}"),
        ]
    else:
        metod_items = [
            _T(f"Pré-processamento: {metricas['Preprocessing']}",
               f"Preprocessing: {metricas['Preprocessing']}"),
            _T(f"Classificador principal: PLS-DA ({metricas['Optimal LVs']} VLs ótimas)",
               f"Main classifier: PLS-DA ({metricas['Optimal LVs']} optimal LVs)"),
            _val_cv,
            _T("Testes estatísticos: permutação de Y, Wold (R2Y/Q2Y), CV-ANOVA",
               "Statistical tests: Y-permutation, Wold (R2Y/Q2Y), CV-ANOVA"),
            _T("Seleção de variáveis: iPLS, VIP >= 1, SR 20% maiores, sPLS-DA",
               "Variable selection: iPLS, VIP >= 1, SR top-20%, sPLS-DA"),
            _T("Validação externa: holdout estratificado (amostras puras sempre no treino)",
               "External validation: stratified holdout (pure samples always in training)"),
            _T("Comparação: PLS-DA x SVM RBF x RF x GBM x XGBoost",
               "Benchmark: PLS-DA vs SVM RBF vs RF vs GBM vs XGBoost"),
            _T(f"Dados: {metricas['N samples']} amostras{_cls_pt}",
               f"Data: {metricas['N samples']} samples{_cls_en}"),
        ]
    for i, item in enumerate(metod_items):
        _txt(slide2, f"• {item}",
             int(Inches(0.7)), int(Inches(1.3) + i * Inches(0.62)),
             int(W - Inches(1.4)), int(Inches(0.6)),
             size=13, color=_SLATE)

    # ── SLIDE 3: Performance Metrics ───────────────────────────────────────
    slide3 = _slide_novo()
    _fundo(slide3)
    _barra_topo(slide3, _T("Resultados — métricas de desempenho", "Results — Performance Metrics"))
    _rodape(slide3)

    if quant:
        _cores = [_ACCENT, _NAVY, _SLATE, _SLATE, _MUTED, _MUTED]
        met_exibir = [(rot, val, _cores[k]) for k, (rot, val, _b) in enumerate(sumario["linhas"][:6])]
    else:
        met_exibir = [
            (_rotulo_metrica("Balanced Accuracy (CV)"), metricas["Balanced Accuracy (CV)"], _ACCENT),
            (_rotulo_metrica("AUC macro OvR"),          metricas["AUC macro OvR"],          _NAVY),
            ("R2Y",                                     metricas["R2Y"],                    _SLATE),
            ("Q2Y",                                     metricas["Q2Y"],                    _SLATE),
            (_rotulo_metrica("Optimal LVs"),            metricas["Optimal LVs"],            _MUTED),
            (_rotulo_metrica("p-value (permutation)"),  metricas["p permutation"],          _MUTED),
        ]
    cols = 3; w_box = int((W - Inches(1.0)) / cols)
    h_box = int(Inches(1.6))
    for idx, (lbl, val, cor) in enumerate(met_exibir):
        col = idx % cols; row = idx // cols
        x = int(Inches(0.5) + col * w_box)
        y = int(Inches(1.3)  + row * Inches(2.0))
        _rect(slide3, x, y, w_box - int(Inches(0.15)), h_box, cor)
        _txt(slide3, val, x + int(Inches(0.2)), y + int(Inches(0.2)),
             w_box - int(Inches(0.4)), int(Inches(0.8)),
             bold=True, size=28, color=_WHITE, align=PP_ALIGN.CENTER)
        _txt(slide3, lbl, x + int(Inches(0.1)), y + int(Inches(0.95)),
             w_box - int(Inches(0.2)), int(Inches(0.55)),
             size=11, color=_LIGHT, align=PP_ALIGN.CENTER)

    # ── Quantificacao: tabela por especie (RMSEP, R2val, LOD, LOQ, SEN, SEL) ──────────
    if quant and sumario["tabela_especie"]:
        cab_e, lin_e = sumario["tabela_especie"]
        slide_e = _slide_novo()
        _fundo(slide_e)
        _barra_topo(slide_e, _T("Resultados — figuras de mérito por espécie", "Results — figures of merit per species"))
        _rodape(slide_e)
        n_lin = min(len(lin_e), 10)
        tbl = slide_e.shapes.add_table(n_lin + 1, len(cab_e), Emu(int(Inches(0.5))), Emu(int(Inches(1.4))),
                                       Emu(int(W - Inches(1.0))), Emu(int(Inches(0.45) * (n_lin + 1)))).table
        for j, c in enumerate(cab_e):
            tbl.cell(0, j).text = str(c)
        for i, row in enumerate(lin_e[:n_lin], 1):
            for j, c in enumerate(row):
                tbl.cell(i, j).text = str(c)
        for k, nt in enumerate(sumario["notas"][:3]):
            _txt(slide_e, nt, int(Inches(0.5)), int(Inches(1.6) + Inches(0.45) * (n_lin + 1) + k * Inches(0.5)),
                 int(W - Inches(1.0)), int(Inches(0.5)), size=10, color=_MUTED)

    # ── SLIDES 4+: Figures ────────────────────────────────────────────────
    imgs = _list_figures(pasta)
    # Prioritize relevant figures
    prioridade = ["scores","confus","vip","pca","outlier","splot","cooman",
                  "roc","hca","opls","ddsimca","benchmark","shap","monte_carlo"]
    def _prio(p: str) -> int:
        nome_l = os.path.basename(p).lower()
        for i, t in enumerate(prioridade):
            if t in nome_l: return i
        return 99
    imgs_sorted = sorted(imgs, key=_prio)[:max_figuras]

    for i in range(0, len(imgs_sorted), 2):
        slide_f = _slide_novo()
        _fundo(slide_f)
        _barra_topo(slide_f, _T("Resultados — figuras", "Results — Figures"))
        _rodape(slide_f)
        for j, img_path in enumerate(imgs_sorted[i:i+2]):
            x = int(Inches(0.3) + j * Inches(6.4))
            try:
                slide_f.shapes.add_picture(img_path,
                    Emu(x), Emu(int(Inches(1.15))),
                    width=Emu(int(Inches(6.1))),
                    height=Emu(int(Inches(5.6))))
            except OSError:   # arquivo ausente/corrompido/formato invalido
                _txt(slide_f, f"[{_T('Figura', 'Figure')}: {os.path.basename(img_path)}]",
                     x, int(Inches(2.0)), int(Inches(6.0)), int(Inches(1.0)),
                     size=11, color=_MUTED)
            cap = _legenda_figura(img_path)
            _txt(slide_f, cap, x, int(H - Inches(0.85)),
                 int(Inches(6.0)), int(Inches(0.45)),
                 size=9, color=_MUTED, align=PP_ALIGN.CENTER)

    # ── SLIDE Benchmark (if CSV exists) ──────────────────────────────────
    bench_csv = os.path.join(pasta_execucao(pasta, NOME_TABELAS), "benchmark_classifiers.csv")
    if os.path.exists(bench_csv):
        try:
            df_b = pd.read_csv(bench_csv, sep=";", decimal=",")
            slide_b = _slide_novo()
            _fundo(slide_b)
            _barra_topo(slide_b, _T("Comparação de classificadores", "Classifier Benchmark"))
            _rodape(slide_b)
            # Table
            cols_b = list(df_b.columns)
            n_rows = min(len(df_b) + 1, 8)
            col_w  = int((W - Inches(1.0)) / len(cols_b))
            row_h  = int(Inches(0.52))
            # Header
            for ci, col_n in enumerate(cols_b):
                _rect(slide_b, int(Inches(0.5) + ci * col_w),
                      int(Inches(1.3)), col_w - 2, row_h, _NAVY)
                _txt(slide_b, str(col_n),
                     int(Inches(0.5) + ci * col_w + Inches(0.05)),
                     int(Inches(1.35)), col_w - int(Inches(0.1)),
                     int(Inches(0.45)), bold=True, size=10, color=_WHITE)
            # Data rows
            fill_alts = [_LIGHT, RGBColor(0xE8, 0xEC, 0xF1)]
            for ri, row_data in enumerate(df_b.itertuples(index=False)):
                if ri >= 7: break
                fy = int(Inches(1.3) + (ri + 1) * row_h)
                for ci, val in enumerate(row_data):
                    _rect(slide_b, int(Inches(0.5) + ci * col_w),
                          fy, col_w - 2, row_h, fill_alts[ri % 2])
                    _txt(slide_b, str(val),
                         int(Inches(0.5) + ci * col_w + Inches(0.05)),
                         fy + int(Inches(0.05)),
                         col_w - int(Inches(0.1)), int(Inches(0.45)),
                         size=10, color=_SLATE)
        except Exception as _e_slide:  # noqa: BLE001 -- slide opcional
            # (leitura CSV + N formas desenhadas); erro logado, o restante
            # do PPTX (capa, figuras, conclusoes) continua sendo gerado.
            logging.getLogger(__name__).warning(
                "PPTX: slide de benchmark nao gerado: %s", _e_slide)

    # ── Final SLIDE: Conclusions ───────────────────────────────────────────
    slide_c = _slide_novo()
    _fundo(slide_c, _NAVY)
    _rect(slide_c, 0, int(Inches(3.2)), int(W), int(Inches(0.1)), _ACCENT)
    _txt(slide_c, _T("Conclusões", "Conclusions"),
         int(Inches(1.0)), int(Inches(1.0)),
         int(W - Inches(2.0)), int(Inches(0.9)),
         bold=True, size=30, color=_WHITE)
    ba   = metricas.get("Balanced Accuracy (CV)", "—")
    auc  = metricas.get("AUC macro OvR", "—")
    r2y  = metricas.get("R2Y", "—"); q2y = metricas.get("Q2Y", "—")
    _pp = metricas["Preprocessing"]
    if quant:
        _l = {r[0]: r[1] for r in sumario["linhas"]}
        _ho_rot = next((k for k in _l if k.startswith("RMSEP (") and ("holdout" in k)), None)
        conc = [
            _T(f"• Pré-processamento {_pp}", f"• Preprocessing {_pp}"),
            (_T(f"• {_ho_rot} = {_l[_ho_rot]}; R2 de predição = {_l.get(_T('R2 de predição (holdout externo)', 'Prediction R2 (external holdout)'), '—')}",
                f"• {_ho_rot} = {_l[_ho_rot]}; prediction R2 = {_l.get(_T('R2 de predição (holdout externo)', 'Prediction R2 (external holdout)'), '—')}")
             if _ho_rot else
             _T("• Sem holdout externo: o RMSEP interno não é o número publicável",
                "• No external holdout: the internal RMSEP is not the publishable number")),
            _T(f"• RMSECV = {_l.get('RMSECV', '—')}  |  R2cv = {_l.get('R2cv (Q2)', '—')}  |  RMSEC = {_l.get('RMSEC', '—')}",
               f"• RMSECV = {_l.get('RMSECV', '—')}  |  R2cv = {_l.get('R2cv (Q2)', '—')}  |  RMSEC = {_l.get('RMSEC', '—')}"),
            _T(f"• {metricas['N samples']} amostras" + ("" if _uma else f", {metricas['N classes']} classes"),
               f"• {metricas['N samples']} samples" + ("" if _uma else f", {metricas['N classes']} classes")),
            _T("• O modelo fica salvo (.joblib) para prever amostras novas",
               "• The model is saved (.joblib) to predict new samples"),
        ]
    else:
        conc = [
            _T(f"• Pré-processamento {_pp}: acurácia balanceada = {ba} (validação cruzada"
               f"{' agrupada' if _agrupada else ' SEM agrupamento'})",
               f"• {_pp} preprocessing: balanced accuracy = {ba} ("
               f"{'grouped' if _agrupada else 'NOT grouped'} cross-validation)"),
            _T(f"• AUC macro = {auc}  |  R2Y = {r2y}  |  Q2Y = {q2y}",
               f"• AUC macro OvR = {auc}  |  R2Y = {r2y}  |  Q2Y = {q2y}"),
            _T(f"• {metricas['N samples']} amostras, {metricas['N classes']} classes",
               f"• {metricas['N samples']} samples, {metricas['N classes']} classes"),
            _T(f"• Validação: permutação de Y (p = {metricas['p permutation']}), "
               f"Wold, CV-ANOVA e holdout externo",
               f"• Validation: Y-permutation (p = {metricas['p permutation']}), "
               f"Wold, CV-ANOVA and external holdout"),
            _T("• O modelo fica salvo (.joblib) para prever amostras novas",
               "• The model is saved (.joblib) to predict new samples"),
        ]
    for i, c in enumerate(conc):
        _txt(slide_c, c,
             int(Inches(1.0)), int(Inches(2.0) + i * Inches(0.78)),
             int(W - Inches(2.0)), int(Inches(0.75)),
             size=14, color=RGBColor(0xCB, 0xD5, 0xE1))

    _rodape(slide_c)

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf
