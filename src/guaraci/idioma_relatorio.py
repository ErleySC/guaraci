"""idioma_relatorio.py -- um idioma so' por relatorio (model_card.md, PDF, Word, Excel, PPTX, LaTeX).

Antes de 2026-11-06 o model card saia sempre em portugues com rotulos em ingles no meio ("Total
samples", "Accuracy (CV)", "N samples outside DModX") e os relatorios PDF/Word/PPTX saiam em ingles
com pedacos em portugues. Agora o texto fixo dos relatorios passa por aqui e sai INTEIRO no idioma
da sessao:

- `idioma()` -- "pt" ou "en": a variavel GUARACI_IDIOMA (que o menu grava ao trocar o idioma com
  [I]); sem ela, portugues.
- `t(pt, en)` -- escolhe a frase.
- `rotulo_resumo(chave)` -- rotulo das chaves do resumo_modelo.txt (que continuam as mesmas: sao
  lidas por regex por outros modulos e por execucoes antigas).
- `valor(texto)` -- valores curtos gravados em portugues no resumo ("sim", "nao", "nenhuma").
- `achado(nome, mensagem)` -- mensagem da auditoria de delineamento no idioma (o nucleo grava em PT).
- `contrair(texto)` -- "de o teor" -> "do teor", "em a matriz" -> "na matriz" (frases montadas
  com o vocabulario do perfil).
- `vocabulario(voc)` -- o vocabulario dos perfis embutidos em ingles; texto proprio do usuario
  passa como esta' (e' dado dele).

Modulo de apresentacao: nao produz numero nenhum (fora do nucleo congelado).
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["idioma", "t", "rotulo_resumo", "valor", "achado", "contrair", "vocabulario", "preprocessamento",
           "titulo_figura", "TITULOS_FIGURA",
           "ROTULOS_RESUMO", "PALAVRAS_SO_PT", "PALAVRAS_SO_EN", "vazamentos"]


def idioma_escolhido() -> Optional[str]:
    """Idioma gravado com [I] no menu (~/.guaraci/.cli_wizard_done: "PT" ou "EN"), ou None."""
    try:
        bruto = (Path.home() / ".guaraci" / ".cli_wizard_done").read_text(encoding="utf-8").strip()
    except (OSError, RuntimeError):
        return None
    return {"EN": "en", "PT": "pt"}.get(bruto.upper())


def idioma(lang: Optional[str] = None) -> str:
    """'pt' ou 'en', o mesmo para menu, mensagens, relatorios e figuras.

    `lang` explicito vence; senao GUARACI_IDIOMA; senao o idioma escolhido com [I] no menu; senao
    ingles -- o mesmo padrao do menu, para quem nunca escolheu nao ver menu num idioma e relatorio
    em outro."""
    bruto = lang if lang is not None else os.environ.get("GUARACI_IDIOMA", "")
    if not str(bruto or "").strip():
        bruto = idioma_escolhido() or "en"
    return "en" if str(bruto).strip().lower().startswith("en") else "pt"


def t(pt: str, en: str, lang: Optional[str] = None) -> str:
    """A frase no idioma do relatorio."""
    return en if idioma(lang) == "en" else pt


#: chave do resumo (como o pipeline grava) -> (rotulo PT, rotulo EN).
ROTULOS_RESUMO: Dict[str, Tuple[str, str]] = {
    "Total samples": ("Total de amostras (espectros)", "Total samples (spectra)"),
    "Total variables": ("Total de variaveis", "Total variables"),
    "Total classes": ("Total de classes", "Total classes"),
    "NaN integrity": ("Amostras com NaN removidas", "Samples with NaN removed"),
    "Integridade Inf": ("Amostras com Inf removidas", "Samples with Inf removed"),
    "Constant variables": ("Variaveis constantes removidas", "Constant variables removed"),
    "Exact duplicates": ("Duplicatas exatas", "Exact duplicates"),
    "Approximate duplicates": ("Duplicatas aproximadas", "Approximate duplicates"),
    "Accuracy (CV)": ("Acuracia (VC)", "Accuracy (CV)"),
    "Balanced accuracy": ("Acuracia balanceada", "Balanced accuracy"),
    "F1 (macro)": ("F1 (macro)", "F1 (macro)"),
    "Cohen's kappa": ("Kappa de Cohen", "Cohen's kappa"),
    "R2X": ("R2X", "R2X"),
    "R2Y": ("R2Y", "R2Y"),
    "Q2": ("Q2", "Q2"),
    "Permutation p-value": ("p-valor da permutacao", "Permutation p-value"),
    "Hotelling T2 (95%)": ("Limite T2 de Hotelling (95%)", "Hotelling T2 limit (95%)"),
    "Q-residual (95%)": ("Limite do residuo Q (95%)", "Q-residual limit (95%)"),
    "ROC AUC macro (OvR)": ("AUC ROC macro (um contra todos)", "ROC AUC macro (one-vs-rest)"),
    "BCa Accuracy": ("IC BCa da acuracia", "BCa CI of accuracy"),
    "BCa Balanced acc.": ("IC BCa da acuracia balanceada", "BCa CI of balanced accuracy"),
    "BCa F1 (macro)": ("IC BCa do F1 (macro)", "BCa CI of F1 (macro)"),
    "BCa Cohen's kappa": ("IC BCa do kappa de Cohen", "BCa CI of Cohen's kappa"),
    "Martens n_significativas": ("Martens: variaveis significativas", "Martens: significant variables"),
    "Martens n_folds_validos": ("Martens: folds validos", "Martens: valid folds"),
    "DModX critical (SIMCA)": ("DModX critico (SIMCA)", "Critical DModX (SIMCA)"),
    "N samples outside DModX": ("Amostras fora do DModX", "Samples outside DModX"),
    "Group-aware (mae_id)": ("Validacao agrupada (mae_id)", "Grouped validation (mae_id)"),
    "N grupos mae_id": ("Grupos mae_id", "mae_id groups"),
    "Imbalance ratio": ("Razao de desbalanceamento", "Imbalance ratio"),
    "Holdout n": ("n", "n"),
    "Holdout accuracy": ("Acuracia", "Accuracy"),
    "Holdout balanced acc": ("Acuracia balanceada", "Balanced accuracy"),
    "Holdout F1 (macro)": ("F1 (macro)", "F1 (macro)"),
    "Holdout Cohen's kappa": ("Kappa de Cohen", "Cohen's kappa"),
    "Holdout classes ausentes": ("Classes ausentes do holdout", "Classes missing from the holdout"),
}


def rotulo_resumo(chave: str, lang: Optional[str] = None) -> str:
    """Rotulo de uma chave do resumo. "BCa Holdout X" e "Holdout X" desconhecidos caem num rotulo
    montado; chave fora da tabela volta como esta' (e o teste de vazamento acusa)."""
    k = str(chave)
    if k in ROTULOS_RESUMO:
        pt, en = ROTULOS_RESUMO[k]
        return t(pt, en, lang)
    if k.startswith("BCa Holdout "):
        resto = k[len("BCa Holdout "):]
        base = {k.lower(): v for k, v in ROTULOS_RESUMO.items()}.get(
            f"holdout {resto}".lower(), _HOLDOUT_BCA.get(resto, (resto, resto)))
        return t(f"IC BCa: {base[0].lower()}", f"BCa CI: {base[1].lower()}", lang)
    return k


#: metricas do holdout que so' aparecem com prefixo "BCa Holdout".
_HOLDOUT_BCA: Dict[str, Tuple[str, str]] = {"Bal.acc": ("Acuracia balanceada", "Balanced accuracy"),
                                            "Accuracy": ("Acuracia", "Accuracy")}

_VALORES: Dict[str, str] = {"sim": "yes", "nao": "no", "não": "no", "nenhuma": "none",
                            "nenhum": "none", "faixa toda util": "whole range usable"}
_VALORES_PT: Dict[str, str] = {"yes": "sim", "no": "nao", "none": "nenhuma",
                               "true": "sim", "false": "nao"}


#: valores em frase que o pipeline (nucleo) grava em PT no resumo -> EN; e os termos em ingles
#: que aparecem no meio deles -> PT.
_FRASES_VALOR_EN: List[Tuple[str, str]] = [
    ("[CI nao estimavel: sem variacao entre reamostragens -- grupos insuficientes por classe]",
     "[CI not estimable: no variation between resamples -- too few groups per class]"),
    ("grupos: este IC tende a ser estreito demais)", "groups: this CI tends to be too narrow)"),
    ("(so' ", "(only "),
    ("n/a (uma unica classe)", "n/a (a single class)"),
]
_FRASES_VALOR_PT: List[Tuple[str, str]] = [("[CI nao estimavel", "[IC nao estimavel")]


def valor(texto: Any, lang: Optional[str] = None) -> str:
    """Valor curto do resumo no idioma ("sim" -> "yes"; True -> "sim"/"yes")."""
    if isinstance(texto, bool):
        return t("sim" if texto else "nao", "yes" if texto else "no", lang)
    s = str(texto)
    if idioma(lang) == "en":
        if s.lower().startswith("sim - aumente"):
            return "YES - increase max_lvs"
        for de, para in _FRASES_VALOR_EN:
            s = s.replace(de, para)
        return _VALORES.get(s.strip().lower(), s)
    for de, para in _FRASES_VALOR_PT:
        s = s.replace(de, para)
    return _VALORES_PT.get(s.strip().lower(), s)


# ---------------------------------------------------------------------------
#  Contracoes do portugues em frases montadas com o vocabulario do perfil
# ---------------------------------------------------------------------------
_CONTRACOES: List[Tuple[str, str]] = [
    (r"\bde o\b", "do"), (r"\bde a\b", "da"), (r"\bde os\b", "dos"), (r"\bde as\b", "das"),
    (r"\bem o\b", "no"), (r"\bem a\b", "na"), (r"\bem os\b", "nos"), (r"\bem as\b", "nas"),
    (r"\bpor o\b", "pelo"), (r"\bpor a\b", "pela"), (r"\bpor os\b", "pelos"), (r"\bpor as\b", "pelas"),
    (r"\bDe o\b", "Do"), (r"\bDe a\b", "Da"), (r"\bEm o\b", "No"), (r"\bEm a\b", "Na"),
]


def contrair(texto: str) -> str:
    """Aplica as contracoes obrigatorias (de+o=do, em+a=na, por+o=pelo...)."""
    for padrao, troca in _CONTRACOES:
        texto = re.sub(padrao, troca, texto)
    return texto


# ---------------------------------------------------------------------------
#  Vocabulario dos perfis embutidos (guaraci/perfis_matriz/*.yaml) em ingles
# ---------------------------------------------------------------------------
_VOC_EN: Dict[str, str] = {
    "classe": "class", "classes": "classes",
    "a matriz analisada": "the analysed matrix", "a amostra": "the sample",
    "a amostra fotografada": "the photographed sample", "a amostra escaneada": "the scanned sample",
    "a amostra (nir, 8 mm)": "the sample (NIR, 8 mm)",
    "o teor do analito de interesse": "the content of the analyte of interest",
    "conforme": "compliant", "nao conforme": "non-compliant",
    "dentro da especificacao": "within specification", "fora da especificacao": "out of specification",
    "dentro do esperado": "as expected", "fora do esperado": "not as expected",
    "especie": "species", "especies": "species",
}


def vocabulario(voc: Any, lang: Optional[str] = None) -> Dict[str, str]:
    """{classe, classe_plural, matriz, alvo, conforme, nao_conforme} no idioma. Em ingles, os textos
    dos perfis embutidos sao traduzidos; texto proprio do usuario fica como ele escreveu."""
    campos = ("classe", "classe_plural", "matriz", "alvo", "conforme", "nao_conforme")
    bruto = {c: str(getattr(voc, c, "") or "") for c in campos}
    if idioma(lang) != "en":
        return bruto
    return {c: _VOC_EN.get(v.strip().lower(), v) for c, v in bruto.items()}


# ---------------------------------------------------------------------------
#  Auditoria de delineamento: o nucleo (auditoria_delineamento.py) grava em PT
# ---------------------------------------------------------------------------
#: (padrao da mensagem PT, modelo EN com os grupos). Coberto por tests/test_idioma_relatorios.py,
#: que roda cada ramo das checagens e exige traducao.
_ACHADOS_EN: List[Tuple[str, str]] = [
    (r"Garantia de agrupamento: high \(mae_id confiavel para toda amostra\)\.",
     "Grouping guarantee: high (reliable mae_id for every sample)."),
    (r"Garantia de agrupamento: medium \(CSV de associacao, nao estrutura de pasta -- confira a fonte\)\.",
     "Grouping guarantee: medium (association CSV, not folder structure -- check the source)."),
    (r"Garantia de agrupamento: (.+?) -- validacao pode ter caido em StratifiedKFold sem protecao contra "
     r"vazamento de replica\. Trate metricas como exploratorias\. Para proteger as replicas: informe "
     r"arquivo_grupos \(CSV arquivo,grupo\), coluna_grupo \(CSV\) ou organize uma subpasta por amostra fisica\.",
     "Grouping guarantee: \\1 -- validation may have fallen back to StratifiedKFold without protection "
     "against replicate leakage. Treat metrics as exploratory. To protect the replicates: give arquivo_grupos "
     "(CSV file,group), coluna_grupo (CSV) or organize one subfolder per physical sample."),
    (r"Garantia de agrupamento: (.+?) -- validacao pode ter caido em StratifiedKFold sem protecao contra "
     r"vazamento de replica\. Trate metricas como exploratorias\.",
     "Grouping guarantee: \\1 -- validation may have fallen back to StratifiedKFold without protection "
     "against replicate leakage. Treat metrics as exploratory."),
    (r"Sem chave de grupo: nada a checar\.", "No group key: nothing to check."),
    (r"Cada grupo tem espectros de uma unica classe\.", "Every group has spectra of a single class."),
    (r"(\d+) grupo\(s\) com espectros de MAIS de uma classe \((.*?)\)\. Replicas da mesma amostra deveriam "
     r"ter a mesma classe: confira o rotulo ou o grupo\.",
     "\\1 group(s) with spectra of MORE than one class (\\2). Replicates of the same sample should have "
     "the same class: check the label or the group."),
    (r"mae_id ausente -- nao e' possivel checar confundimento classe x sessao sem identificador de amostra "
     r"fisica\.",
     "mae_id missing -- class x session confounding cannot be checked without a physical-sample identifier."),
    (r"Todo o dataset veio de 1 UNICA sessao de coleta -- nenhuma classe pode ser distinguida de deriva "
     r"instrumental/temporal\. Isso nao e' um problema por classe, e' estrutural do dataset inteiro\.",
     "The whole dataset came from 1 SINGLE collection session -- no class can be told apart from "
     "instrumental/temporal drift. This is not a per-class problem; it is structural to the whole dataset."),
    (r"Nenhuma classe confinada a 1 unica sessao \((\d+) sessoes no total\)\.",
     "No class confined to a single session (\\1 sessions in total)."),
    (r"(\d+) classe\(s\) confinada\(s\) a 1 unica sessao de coleta \(de (\d+) sessoes no total\): (.*?)\. "
     r"Deriva instrumental daquela sessao fica indistinguivel de efeito de classe para essas classes\.",
     "\\1 class(es) confined to a single collection session (out of \\2 sessions in total): \\3. "
     "Instrumental drift of that session cannot be told apart from the class effect for these classes."),
    (r"mae_id ausente -- sem chave de grupo nao ha' unidade independente declarada \(ver garantia de "
     r"agrupamento\)\.",
     "mae_id missing -- without a group key there is no declared independent unit (see grouping guarantee)."),
    (r"Agrupando por sessao restam so' (\d+) unidade\(s\): a validacao agrupada fica degenerada "
     r"\(precisa de >=3\)\.",
     "Grouping by session leaves only \\1 unit(s): grouped validation degenerates (needs >= 3)."),
    (r"Validacao agrupada por SESSAO de coleta \((\d+) unidades independentes; (\d+) mae_id distintos\)\.",
     "Validation grouped by collection SESSION (\\1 independent units; \\2 distinct mae_id)."),
    (r"NAO VERIFICAVEL: (\d+) grupos declarados por voce, mas os ids nao seguem o padrao de sessao, entao "
     r"nao da' para checar se replicas ou diluicoes da MESMA sessao/lote receberam grupos diferentes\. Se "
     r"receberam, a validacao vaza por sessao e o desempenho e' OTIMISTA \(teste nulo: balanced accuracy "
     r"0,72 CV / 0,82 holdout com acaso 0,33\)\. Agrupe pelo nivel mais grosso que e' de fato independente "
     r"\(sessao/lote/dia\)\.",
     "NOT VERIFIABLE: \\1 groups declared by you, but the ids do not follow the session pattern, so it "
     "cannot be checked whether replicates or dilutions of the SAME session/batch got different groups. If "
     "they did, validation leaks through the session and performance is OPTIMISTIC (null test: balanced "
     "accuracy 0.72 CV / 0.82 holdout against chance 0.33). Group by the coarsest level that is really "
     "independent (session/batch/day)."),
    (r"mae_id e sessao coincidem \((\d+) unidades\): agrupar por mae_id equivale a agrupar por sessao\.",
     "mae_id and session coincide (\\1 units): grouping by mae_id is the same as grouping by session."),
    (r"O mae_id \((\d+) valores\) e' mais fino que a sessao de coleta \((\d+) sessoes\): niveis de "
     r"teor/diluicoes da MESMA sessao podem cair em folds diferentes\. Alem disso cada sessao tem UMA classe "
     r"so': o modelo pode estar distinguindo a SESSAO, nao a classe \(medido em teste nulo: bal\.acc "
     r"0,90-0,98 contra acaso 0,25\)\. Use unidade_agrupamento=sessao e trate os numeros por mae_id como "
     r"otimistas\.",
     "The mae_id (\\1 values) is finer than the collection session (\\2 sessions): content levels/dilutions "
     "of the SAME session may fall into different folds. Moreover each session has ONE class only: the "
     "model may be telling SESSIONS apart, not classes (measured in a null test: bal.acc 0.90-0.98 against "
     "chance 0.25). Use `unidade_agrupamento=sessao` and treat the per-mae_id numbers as optimistic."),
    (r"O mae_id \((\d+) valores\) e' mais fino que a sessao de coleta \((\d+) sessoes\): niveis de "
     r"teor/diluicoes da MESMA sessao podem cair em folds diferentes\. Se a sessao e' a unidade independente "
     r"do seu delineamento, use unidade_agrupamento=sessao\.",
     "The mae_id (\\1 values) is finer than the collection session (\\2 sessions): content levels/dilutions "
     "of the SAME session may fall into different folds. If the session is the independent unit of your "
     "design, use `unidade_agrupamento=sessao`."),
    (r"(\d+) classes presentes\.", "\\1 classes present."),
    (r"Uma unica classe \(quantificacao de um so' material\): as metricas de classificacao nao se aplicam; "
     r"valem as de regressao\.",
     "A single class (quantification of one material): classification metrics do not apply; regression "
     "metrics do."),
    (r"Apenas UMA classe presente: accuracy, balanced accuracy, kappa e F1 sao degeneradas \(1,0 por "
     r"construcao\) e NAO medem desempenho\. Nao cite esses numeros; use outro nivel/objetivo ou traga mais "
     r"classes\.",
     "Only ONE class present: accuracy, balanced accuracy, kappa and F1 are degenerate (1.0 by "
     "construction) and do NOT measure performance. Do not cite these numbers; use another level/objective "
     "or bring more classes."),
    (r"Nenhuma duplicata exata ou aproximada detectada\.", "No exact or approximate duplicate detected."),
    (r"(\d+) duplicata\(s\) EXATA\(s\), (\d+) amostra\(s\) com alta correlacao \(possiveis replicas nao "
     r"identificadas como tal\) -- risco de vazamento treino/teste se cairem em folds diferentes\.",
     "\\1 EXACT duplicate(s), \\2 sample(s) with high correlation (possible replicates not identified as "
     "such) -- risk of train/test leakage if they fall into different folds."),
    (r"mae_id ausente -- nao e' possivel contar sessoes independentes por classe\.",
     "mae_id missing -- independent sessions per class cannot be counted."),
    (r"Toda classe tem >= (\d+) sessoes independentes \(minimo p/ alpha=([\d.]+) conformal\)\.",
     "Every class has >= \\1 independent sessions (minimum for conformal alpha=\\2)."),
    (r"(\d+) classe\(s\) com menos de (\d+) sessoes independentes \(minimo p/ alpha=([\d.]+) conformal\): "
     r"(.*?)\. Gates estatisticos para essas classes ficam nao-validados \(sao necessarias mais sessoes de "
     r"coleta independentes\) -- nao e' erro, e' o regime honesto do dataset atual\.",
     "\\1 class(es) with fewer than \\2 independent sessions (minimum for conformal alpha=\\3): \\4. "
     "Statistical gates for these classes stay non-validated (more independent collection sessions are "
     "needed) -- not an error, it is the honest regime of the current dataset."),
    (r"Sem dados de concentracao -- nao e' possivel comparar faixa de calibracao com faixa de trabalho "
     r"declarada\.",
     "No concentration data -- the calibration range cannot be compared with the declared working range."),
    (r"Nenhum valor de concentracao valido para comparar\.", "No valid concentration value to compare."),
    (r"Perfil de tecnica nao declara faixa de trabalho -- nada a comparar \(ausencia de declaracao nao e' "
     r"garantia de que a faixa calibrada cobre o uso pretendido\)\.",
     "The technique profile declares no working range -- nothing to compare (no declaration is no "
     "guarantee that the calibrated range covers the intended use)."),
    (r"Faixa calibrada \[(.*?)\] dentro da faixa de trabalho declarada (.*?)\.",
     "Calibrated range [\\1] within the declared working range \\2."),
    (r"Faixa calibrada \[(.*?)\] NAO cobre totalmente a faixa de trabalho declarada (.*?) -- predicoes em "
     r"amostras dentro da faixa de trabalho mas fora da faixa calibrada seriam extrapolacao\.",
     "Calibrated range [\\1] does NOT fully cover the declared working range \\2 -- predictions on samples "
     "inside the working range but outside the calibrated range would be extrapolation."),
    (r"Os padroes do motor sao ponto de partida, nao universais, e nao validam o seu estudo: confirme as "
     r"conclusoes com validacao externa propria \(amostras independentes\)\.",
     "The engine defaults are a starting point, not universal, and do not validate your study: confirm the "
     "conclusions with your own external validation (independent samples)."),
    # texto da versao anterior (execucoes antigas relidas)
    (r"Os padroes do motor foram calibrados em dados de referencia, o que nao valida o seu estudo: confirme "
     r"as conclusoes com validacao externa propria \(amostras independentes\)\. DD-SIMCA/OPLS-DA ainda sem "
     r"essa calibracao\.",
     "The engine defaults are a starting point, not universal, and do not validate your study: confirm the "
     "conclusions with your own external validation (independent samples)."),
]
_ACHADOS_EN_RE = [(re.compile(p, re.DOTALL), en) for p, en in _ACHADOS_EN]
_SILENCIADO = re.compile(r"^(.*) \[SILENCIADO: (.*)\]$", re.DOTALL)


#: termos em ingles que o nucleo grava no meio das mensagens em PT.
_PT_AJUSTES: List[Tuple[str, str]] = [
    ("balanced accuracy", "acuracia balanceada"), ("bal.acc", "acur. bal."),
    ("Gates estatisticos", "Portoes estatisticos"), ("accuracy, acuracia balanceada", "acuracia, acuracia balanceada"),
    ("accuracy,", "acuracia,"), ("mean-centering", "centragem na media"),
    ("Garantia de agrupamento: high", "Garantia de agrupamento: alta"),
    ("Garantia de agrupamento: medium", "Garantia de agrupamento: media"),
    ("Garantia de agrupamento: none", "Garantia de agrupamento: nenhuma"),
    ("Garantia de agrupamento: unknown", "Garantia de agrupamento: desconhecida"),
]


def _ajustar_pt(texto: str) -> str:
    for de, para in _PT_AJUSTES:
        texto = texto.replace(de, para)
    return texto


def preprocessamento(texto: Any, lang: Optional[str] = None) -> str:
    """Descricao do pre-processamento gravada pelo pipeline ("MSC -> SG(...) -> mean-centering")."""
    s = str(texto)
    return s if idioma(lang) == "en" else s.replace("mean-centering", "centragem na media")


def achado(mensagem: str, lang: Optional[str] = None) -> str:
    """Mensagem da auditoria no idioma. Em PT corrige os termos em ingles que o nucleo grava; em EN,
    traduz pela tabela (sem par na tabela volta em PT -- o teste de vazamento acusa)."""
    if idioma(lang) != "en":
        return _ajustar_pt(str(mensagem))
    msg = str(mensagem)
    sufixo = ""
    m = _SILENCIADO.match(msg)
    if m:
        msg, sufixo = m.group(1), f" [SILENCED: {m.group(2)}]"
    for padrao, en in _ACHADOS_EN_RE:
        mm = padrao.fullmatch(msg.strip())
        if mm:
            return mm.expand(en) + sufixo
    return msg + sufixo


# ---------------------------------------------------------------------------
#  Deteccao de vazamento (usada pelos testes; exposta para checagem manual)
# ---------------------------------------------------------------------------
#: palavras (inteiras) que so' existem em portugues -- forma sem acento, como o projeto grava.
PALAVRAS_SO_PT = ("amostras", "amostra", "variaveis", "sessao", "sessoes", "nao", "validacao",
                  "acuracia", "especie", "especies", "teor", "matriz", "dados", "duplicatas",
                  "nenhuma", "nenhum", "faixa", "limite", "agrupada", "ressalvas", "metricas",
                  "unica", "quantificacao", "identificacao", "deriva", "replicas", "grupos",
                  "intervalo", "veredito", "recomendacao", "portao", "sem", "com", "da",
                  "em", "para", "pelo", "pela", "classe", "figuras", "pasta", "autor",
                  "instituicao", "referencias", "conclusoes", "resultados", "metodologia",
                  "resumo", "criterios", "gerado", "pagina", "valor")
#: palavras (inteiras) que so' existem em ingles.
PALAVRAS_SO_EN = ("samples", "sample", "variables", "the", "and", "of", "with", "without",
                  "outside", "missing", "groups", "species", "range", "limit", "figures", "figure",
                  "page", "generated", "report", "value", "metric", "metrics", "author",
                  "institution", "folder", "date", "references", "conclusions", "results",
                  "methodology", "summary", "quality", "criteria", "intercept", "not", "random",
                  "overfitting", "duplicates", "accuracy", "balanced", "present", "none", "yes",
                  "validation", "session", "sessions", "warning", "critical", "check")


def vazamentos(texto: str, lang: str, ignorar: Tuple[str, ...] = ()) -> List[str]:
    """Palavras inteiras do OUTRO idioma em `texto`. Trechos entre crases (`identificadores`) e os
    trechos de `ignorar` (nomes de classe, citacoes) sao removidos antes da busca."""
    alvo = re.sub(r"`[^`]*`", " ", f" {texto} ")   # `identificadores` (chaves de config, ids)
    for trecho in ignorar:
        alvo = re.sub(re.escape(trecho), " ", alvo, flags=re.IGNORECASE)
    import unicodedata
    sem_acento = unicodedata.normalize("NFKD", alvo.lower()).encode("ascii", "ignore").decode("ascii")
    palavras = set(re.findall(r"[a-z_]+", sem_acento))   # com "_" = identificador, nao palavra
    lista = PALAVRAS_SO_EN if idioma(lang) == "pt" else PALAVRAS_SO_PT
    return [p for p in lista if p in palavras]


# ---------------------------------------------------------------------------
#  Titulo das figuras nas legendas dos relatorios (o nome do arquivo e' um id em PT)
# ---------------------------------------------------------------------------
#: (prefixo do nome do arquivo, titulo PT, titulo EN). Primeiro prefixo que casa vence.
TITULOS_FIGURA: Tuple[Tuple[str, str, str], ...] = (
    ("fig0_espectros_medios", "Espectros médios por classe", "Mean spectra per class"),
    ("fig6_preprocessing", "Efeito do pré-processamento", "Effect of pre-processing"),
    ("fig_extra_comparacao_pipelines", "Comparação de pré-processamentos", "Pre-processing comparison"),
    ("fig_hca_comparacao_pipelines", "HCA por pré-processamento", "HCA by pre-processing"),
    ("fig1b_pca_scree", "PCA: variância explicada", "PCA: explained variance"),
    ("fig1_pca_scores", "PCA: scores", "PCA: scores"),
    ("fig_loadings_pca", "PCA: loadings", "PCA: loadings"),
    ("fig_biplot_pca", "PCA: biplot", "PCA: biplot"),
    ("fig_hca_dendrogram", "HCA: dendrograma", "HCA: dendrogram"),
    ("fig2_plsda_scores", "PLS-DA: scores", "PLS-DA: scores"),
    ("figS1_selecao_lvs", "Escolha do número de VLs", "Choice of the number of LVs"),
    ("fig4_confusao", "Matriz de confusão e métricas por classe", "Confusion matrix and per-class metrics"),
    ("fig_roc_auc", "Curvas ROC", "ROC curves"),
    ("fig3_outliers", "T² de Hotelling e Q por amostra", "Hotelling T² and Q per sample"),
    ("fig5b_vip", "VIP com bootstrap", "Bootstrapped VIP"),
    ("fig_selectivity_ratio_vip", "Razão de seletividade e VIP", "Selectivity ratio and VIP"),
    ("fig_score_contribution", "Contribuição ao score", "Score contribution"),
    ("fig_etapa4_", "Seleção de variáveis (etapa 4)", "Variable selection (step 4)"),
    ("fig_extra_holdout", "Holdout externo x validação cruzada", "External holdout vs cross-validation"),
    ("fig_extra_wold", "Teste de permutação (Wold)", "Permutation test (Wold)"),
    ("fig_monte_carlo", "Validação cruzada Monte Carlo", "Monte Carlo cross-validation"),
    ("fig_det_curvas", "Curvas DET", "DET curves"),
    ("fig_ddsimca_aceitacao", "DD-SIMCA: aceitação por modelo de classe", "DD-SIMCA: acceptance per class model"),
    ("fig_cooman_ddsimca", "DD-SIMCA: gráfico de Cooman", "DD-SIMCA: Cooman's plot"),
    ("fig_ddsimca", "DD-SIMCA", "DD-SIMCA"),
    ("ddsimca_", "DD-SIMCA: modelo da classe", "DD-SIMCA: model of class"),
    ("fig_opls_scores", "OPLS-DA: scores", "OPLS-DA: scores"),
    ("fig_splot_opls", "OPLS-DA: S-plot", "OPLS-DA: S-plot"),
    ("fig_benchmark_", "Comparação de modelos", "Model benchmark"),
    ("fig_shap_", "Explicação SHAP:", "SHAP explanation:"),
    ("figS2b_", "Predito x referência com faixa ±RMSEP", "Predicted vs reference with ±RMSEP band"),
    ("figS2c_", "Coeficientes de regressão (b)", "Regression coefficients (b)"),
    ("figS2d_", "Gráfico de Williams", "Williams plot"),
    ("figS2_", "Regressão PLS", "PLS regression"),
    ("figS3_", "Figuras de mérito da regressão", "Regression figures of merit"),
    ("fig_heatmap_species_by_adulterant", "R² por espécie x adulterante", "R² per species x adulterant"),
    ("fig_multielementar", "Médias por elemento", "Means per element"),
    ("fig_mcr_als", "MCR-ALS: perfis C e S", "MCR-ALS: C and S profiles"),
    ("fig_parafac", "PARAFAC: loadings", "PARAFAC: loadings"),
    ("fig_asca", "ASCA: efeitos", "ASCA: effects"),
    ("fig_transferencia", "Transferência de calibração", "Calibration transfer"),
    ("carta_t2_q", "Carta T² e Q", "T² and Q chart"),
    ("contribuicao_q", "Contribuição ao Q", "Q contribution"),
    ("tendencia_por_lote", "Tendência por lote", "Trend per batch"),
)


#: prefixos cujo resto do nome identifica o item (classe do DD-SIMCA, tag do SHAP).
_PREFIXOS_COM_ITEM = {"ddsimca_", "fig_shap_"}


def titulo_figura(caminho: str, lang: Optional[str] = None) -> str:
    """Legenda de uma figura no idioma: titulo pela tabela + `(parte N)` quando a figura foi
    dividida em paginas (`nome_02`) + o resto do nome quando ele identifica algo (classe, tag)."""
    nome = os.path.splitext(os.path.basename(str(caminho)))[0]
    parte = ""
    m = re.match(r"^(.*)_(\d{2})$", nome)
    if m:
        nome, parte = m.group(1), t(f" (parte {int(m.group(2))})", f" (part {int(m.group(2))})", lang)
    for prefixo, pt, en in TITULOS_FIGURA:
        if nome.startswith(prefixo):
            resto = nome[len(prefixo):].strip("_")
            # sufixo que identifica o item (classe do DD-SIMCA, tag do SHAP); o resto do nome fixo
            # do arquivo (ex.: "_e_metricas_por_classe") nao e' repetido
            extra = f" {resto.replace('_', ' ')}" if prefixo in _PREFIXOS_COM_ITEM and resto else ""
            return f"{t(pt, en, lang)}{extra}{parte}"
    return nome.replace("_", " ") + parte


# ---------------------------------------------------------------------------
#  resumo_modelo.txt legivel num idioma so'
# ---------------------------------------------------------------------------
# O pipeline (nucleo) monta o resumo com chaves fixas, metade em ingles e metade em portugues, e
# outros modulos as leem por regex. Essas chaves continuam as mesmas no resumo_modelo.json (texto
# canonico, so' para maquina); o resumo_modelo.txt, que a pessoa le, passa por aqui.

#: rotulos das chaves que so' aparecem no resumo (o model card tem os seus em ROTULOS_RESUMO).
_ROTULOS_SO_RESUMO: Dict[str, Tuple[str, str]] = {
    "Input mode": ("Modo de entrada", "Input mode"),
    "Grouping guarantee": ("Garantia de agrupamento", "Grouping guarantee"),
    "Method": ("Metodo", "Method"),
    "Pre-processamento": ("Pre-processamento", "Preprocessing"),
    "Tecnica (perfil)": ("Tecnica (perfil)", "Technique (profile)"),
    "Faixas excluidas": ("Faixas excluidas", "Excluded ranges"),
    "Normalizacao por espectro": ("Normalizacao por espectro", "Per-spectrum normalisation"),
    "Saturacao do detector": ("Saturacao do detector", "Detector saturation"),
    "Variaveis com sinal (SNR>=3)": ("Variaveis com sinal (SNR>=3)", "Variables with signal (SNR>=3)"),
    "Faixa com sinal": ("Faixa com sinal", "Range with signal"),
    "Optimal LVs": ("VLs otimas", "Optimal LVs"),
    "LVs no teto (max_lvs)": ("VLs no teto (max_lvs)", "LVs at the ceiling (max_lvs)"),
    "Validation": ("Validacao", "Validation"),
    "Unidade de agrupamento": ("Unidade de agrupamento", "Grouping unit"),
    "N unidades independentes": ("Unidades independentes", "Independent units"),
    "Nivel": ("Nivel", "Level"),
    "Tag": ("Etiqueta", "Tag"),
    "N outliers T2": ("Outliers T2", "T2 outliers"),
    "N outliers Q": ("Outliers Q", "Q outliers"),
    "Largest class": ("Maior classe", "Largest class"),
    "Smallest class": ("Menor classe", "Smallest class"),
    "Permutation n_validos": ("Permutacoes validas", "Valid permutations"),
    "Permutation n_falhos": ("Permutacoes com falha", "Failed permutations"),
    "Permutation failure_rate": ("Taxa de falha das permutacoes", "Permutation failure rate"),
    "Holdout n": ("Holdout: n", "Holdout: n"),
    "Holdout accuracy": ("Holdout: acuracia", "Holdout: accuracy"),
    "Holdout balanced acc": ("Holdout: acuracia balanceada", "Holdout: balanced accuracy"),
    "Holdout F1 (macro)": ("Holdout: F1 (macro)", "Holdout: F1 (macro)"),
    "Holdout Cohen kappa": ("Holdout: kappa de Cohen", "Holdout: Cohen's kappa"),
    "Holdout Cohen's kappa": ("Holdout: kappa de Cohen", "Holdout: Cohen's kappa"),
    "BCa Holdout Accuracy": ("Holdout: IC BCa da acuracia", "Holdout: BCa CI of accuracy"),
    "BCa Holdout Bal.acc": ("Holdout: IC BCa da acuracia balanceada", "Holdout: BCa CI of balanced accuracy"),
    "Conformal classif. alcancavel": ("Conformal (classificacao) alcancavel", "Conformal (classification) reachable"),
    "Conformal classif. cobertura": ("Conformal (classificacao): cobertura", "Conformal (classification): coverage"),
    "Conformal classif. n_grupos": ("Conformal (classificacao): grupos", "Conformal (classification): groups"),
    "DD-SIMCA n_components": ("DD-SIMCA: componentes", "DD-SIMCA: components"),
    "DD-SIMCA n_desconhecidos": ("DD-SIMCA: desconhecidas", "DD-SIMCA: unknown"),
    "DD-SIMCA n_ambiguos": ("DD-SIMCA: ambiguas", "DD-SIMCA: ambiguous"),
    "DD-SIMCA mode treino": ("DD-SIMCA: modo de treino", "DD-SIMCA: training mode"),
    "OPLS-DA n_ortho": ("OPLS-DA: componentes ortogonais", "OPLS-DA: orthogonal components"),
    "CV-ANOVA F": ("CV-ANOVA: F", "CV-ANOVA: F"),
    "CV-ANOVA p": ("CV-ANOVA: p", "CV-ANOVA: p"),
    "Wold R2Y intercept": ("Wold: intercepto do R2Y", "Wold: R2Y intercept"),
    "Wold Q2Y intercept": ("Wold: intercepto do Q2Y", "Wold: Q2Y intercept"),
    "Wold R2Y valido (<.40)": ("Wold: R2Y valido (<0,40)", "Wold: R2Y valid (<0.40)"),
    "Wold Q2Y valido (<.05)": ("Wold: Q2Y valido (<0,05)", "Wold: Q2Y valid (<0.05)"),
    "AD aviso n pequeno": ("Dominio de aplicabilidade: aviso de n pequeno", "Applicability domain: small-n warning"),
    "AD taxa de rejeicao em CV (referencia do sentinela)": (
        "Dominio de aplicabilidade: rejeicao na VC (referencia do sentinela)",
        "Applicability domain: rejection rate in CV (sentinel reference)"),
    "AD taxa esperada de alarmes em amostras legitimas": (
        "Dominio de aplicabilidade: alarmes esperados em amostras legitimas",
        "Applicability domain: expected alarm rate on legitimate samples"),
    "Identificacao n_combinacoes": ("Identificacao: combinacoes", "Identification: combinations"),
    "Identificacao n_validado": ("Identificacao: validadas", "Identification: validated"),
    "Pureza n_confiavel": ("Pureza: confiaveis", "Purity: reliable"),
    "Pureza n_especies": ("Pureza: especies", "Purity: species"),
    "anova": ("ANOVA", "ANOVA"),
    "  >> Mais parcimonioso": ("  >> Mais parcimonioso", "  >> Most parsimonious"),
    "--- ROC AUC por classe ---": ("--- AUC ROC por classe ---", "--- ROC AUC per class ---"),
    "--- Outliers T2 por classe ---": ("--- Outliers T2 por classe ---", "--- T2 outliers per class ---"),
    "--- Accuracy por classe ---": ("--- Acuracia por classe ---", "--- Accuracy per class ---"),
    "--- Etapa 4: selecao de variaveis ---": ("--- Etapa 4: selecao de variaveis ---",
                                              "--- Stage 4: variable selection ---"),
    "parametros_efetivos": ("Parametros efetivos (chaves de configuracao)", "Effective parameters (config keys)"),
}

#: chaves montadas com o nome da classe: (regex, PT, EN) com \1 = classe.
_ROTULOS_DINAMICOS: List[Tuple[str, str, str]] = [
    (r"^(\s*)AUC (.+)$", r"\1AUC \2", r"\1AUC \2"),
    (r"^(\s*)Acc (.+)$", r"\1Acuracia \2", r"\1Accuracy \2"),
    (r"^(\s*)T2-outliers (.+)$", r"\1Outliers T2 \2", r"\1T2 outliers \2"),
    (r"^DD-SIMCA (.+) sens\(LOGO\)/esp$", r"DD-SIMCA \1: sens. (LOGO) / esp.", r"DD-SIMCA \1: sens. (LOGO) / spec."),
    (r"^DD-SIMCA (.+) sens\(PCV, exploratorio\)$", r"DD-SIMCA \1: sens. (PCV, exploratoria)",
     r"DD-SIMCA \1: sens. (PCV, exploratory)"),
    (r"^DD-SIMCA (.+) AVISO PCV$", r"DD-SIMCA \1: aviso do PCV", r"DD-SIMCA \1: PCV warning"),
    (r"^DD-SIMCA (.+) AVISO treino$", r"DD-SIMCA \1: aviso de treino", r"DD-SIMCA \1: training warning"),
    (r"^DD-SIMCA (.+) AVISO$", r"DD-SIMCA \1: aviso", r"DD-SIMCA \1: warning"),
    (r"^Faixa espectral(.*)$", r"Faixa espectral\1", r"Spectral range\1"),
]


def rotulo_linha_resumo(chave: str, lang: Optional[str] = None) -> str:
    """Rotulo de uma chave do resumo_modelo.txt (mantem o recuo da chave)."""
    k = str(chave)
    if k in _ROTULOS_SO_RESUMO:
        return t(*_ROTULOS_SO_RESUMO[k], lang=lang)
    if k in ROTULOS_RESUMO:
        return rotulo_resumo(k, lang)
    en = idioma(lang) == "en"
    for padrao, pt, ing in _ROTULOS_DINAMICOS:
        if re.match(padrao, k):
            return re.sub(padrao, ing if en else pt, k)
    return rotulo_resumo(k, lang)


#: valores em frase (regex) gravados pelo nucleo: (regex, PT, EN). PT tambem corrige grafia.
_VALORES_RESUMO: List[Tuple[str, str, str]] = [
    (r"\(grupos_LOGO=(\d+), puros=(\d+), adult=(\d+)\)",
     r"(grupos LOGO=\1, puros=\2, adulteradas=\3)", r"(LOGO groups=\1, pure=\2, adulterated=\3)"),
    (r"^puros \(one-class\)$", "puros (uma classe)", "pure (one-class)"),
    (r"^todos \(in-sample; sens/esp NAO sao autenticacao\)$",
     "todos (na propria amostra; sens/esp NAO sao autenticacao)",
     "all (in-sample; sens/spec are NOT authentication)"),
    (r"n/a \(nao validado\)", "n/a (nao validado)", "n/a (not validated)"),
    (r"<-- ANOMALO \(>30%\)", "<-- ANOMALO (>30%)", "<-- ANOMALOUS (>30%)"),
    (r"Sensibilidade estimada por LOGO com apenas (\d+) grupos de replica\. Incerteza alta; IC bootstrap "
     r"nao e confiavel neste regime\. Interpretar como exploratoria\.",
     r"\g<0>",
     r"Sensitivity estimated by LOGO with only \1 replicate groups. High uncertainty; a bootstrap CI is "
     r"not reliable in this regime. Read it as exploratory."),
    (r"Sensibilidade LOGO inconclusiva: (\d+) dobra\(s\) valida\(s\) de (\d+) grupos \(puros por grupo "
     r"insuficientes para ajustar o modelo\)\. Interpretar como nao-validado\.",
     r"\g<0>",
     r"LOGO sensitivity inconclusive: \1 valid fold(s) out of \2 groups (too few pure samples per group to "
     r"fit the model). Read it as not validated."),
    (r"Sensibilidade nao estimavel por LOGO: apenas (\d+) grupo\(s\) de replica pura\. Sem replicacao "
     r"independente nao ha validacao possivel; re-substituicao seria enganosa\.",
     r"\g<0>",
     r"Sensitivity not estimable by LOGO: only \1 pure replicate group(s). Without independent replication "
     r"no validation is possible; resubstitution would be misleading."),
    (r"Sensibilidade nao estimavel: mae_id ausente \(sem grupos de replica para LOGO\)\.", r"\g<0>",
     "Sensitivity not estimable: mae_id missing (no replicate groups for LOGO)."),
    (r"PCV com (\d+) grupos de replica\. Diagnostico exploratorio, complementar ao LOGO -- nao o substitui\.",
     r"\g<0>",
     r"PCV with \1 replicate groups. Exploratory diagnostic, complementary to LOGO -- it does not replace it."),
    (r"PCV com um unico grupo de replica pura: .*", r"\g<0>",
     "PCV with a single pure replicate group: the PV-set reproduces only MEASUREMENT noise (T1/T2/T3 of the "
     "same sample), not variation between different physical samples. It is not evidence of generalisation "
     "-- read it as robustness to instrumental noise, never as validated authentication."),
    (r"Modelo nao ajustavel com estes puros \(ver LOGO\)\.", r"\g<0>",
     "Model cannot be fitted with these pure samples (see LOGO)."),
    (r"Amostras insuficientes \(n=(\d+)\) para gerar PV-set\.", r"\g<0>",
     r"Too few samples (n=\1) to build the PV-set."),
    (r"(\d+) replica\(s\) de treino atipica\(s\) \(indices (\[[^\]]*\]), deteccao robusta mediana/MAD\) -- "
     r"nao removidas automaticamente, considere investigar\.", r"\g<0>",
     r"\1 atypical training replicate(s) (indices \2, robust median/MAD detection) -- not removed "
     r"automatically, consider investigating."),
    (r"Dominio de aplicabilidade calibrado com poucos (amostras|grupos \(amostras fisicas\)) por classe "
     r"\(minimo (\d+); (\d+) classe\(s\) abaixo de (\d+)\)\. (.*)$", r"\g<0>",
     r"Applicability domain calibrated with few \1 per class (minimum \2; \3 class(es) below \4). In this "
     r"regime it tends to reject many legitimate samples: the expected alarm rate is unknown and can be much "
     r"higher than the nominal 5%. Use the rejection rate in cross-validation as the estimate and do not "
     r"treat 'outside the domain' as proof of a problem with the sample."),
    (r"Applicability domain calibrated with few amostras per", "Applicability domain calibrated with few samples per",
     "Applicability domain calibrated with few samples per"),
    (r"Applicability domain calibrated with few grupos \(amostras fisicas\) per",
     "Applicability domain calibrated with few groups (physical samples) per",
     "Applicability domain calibrated with few groups (physical samples) per"),
    (r"^(N\d) \((.+)\)$", r"\1 (\2)", r"\1 (\2)"),   # nivel: o nome vem de _NIVEL_NOME (tratado abaixo)
    (r"(\d+) vars \| (\d+) LVs", r"\1 variaveis | \2 VLs", r"\1 vars | \2 LVs"),
    (r"\((\d+) vars, ", r"(\1 variaveis, ", r"(\1 vars, "),
]
_VALORES_RESUMO_RE = [(re.compile(p), pt, en) for p, pt, en in _VALORES_RESUMO]

#: nome dos niveis (pipeline._NIVEL_NOME, em PT) -> EN.
_NIVEIS_EN: Dict[str, str] = {
    "Classificacao por especie": "Species classification",
    "Identificacao de especie": "Species identification",
    "Discriminacao puro/adulterado": "Pure/adulterated discrimination",
    "Quantificacao de teor": "Content quantification",
    "Exploratorio": "Exploratory",
}


def valor_resumo(v: Any, lang: Optional[str] = None) -> str:
    """Valor de uma linha do resumo_modelo.txt no idioma."""
    if isinstance(v, bool):
        return valor(v, lang)
    if isinstance(v, float):
        s = f"{v:.4f}"
    else:
        s = str(v)
    en = idioma(lang) == "en"
    if " -> " in s and ("mean-centering" in s or "centraliz" in s or "MSC" in s or "SNV" in s):
        return preprocessamento(s, lang)
    for rx, pt, ing in _VALORES_RESUMO_RE:
        s = rx.sub(ing if en else pt, s)
    if en:
        for pt_nome, en_nome in _NIVEIS_EN.items():
            s = s.replace(f"({pt_nome})", f"({en_nome})")
    return valor(s, lang)


#: linhas fixas dos blocos anexados ao resumo (regressao, heatmap): (regex da linha inteira, PT, EN).
_LINHAS_RESUMO: List[Tuple[str, str, str]] = [
    (r"^(\s*)PLS Regression — Analytical Figures of Merit$",
     r"\1Regressao PLS — figuras de merito analiticas", r"\g<0>"),
    (r"^(\s*)Pooled metrics:$", r"\1Metricas do modelo unico (todas as especies):", r"\1Pooled metrics:"),
    (r"^(\s*)\(RMSEP acima = validacao INTERNA: split 70/30 dos dados de$",
     r"\1(RMSEP acima = validacao INTERNA: divisao 70/30 dos dados de",
     r"\1(RMSEP above = INTERNAL validation: 70/30 split of the training"),
    (r"^(\s*)treino, n pequeno; nao e' o numero publicavel\.\)$",
     r"\1treino, n pequeno; nao e' o numero publicavel.)", r"\1data, small n; not the publishable number.)"),
    (r"^(\s*)RMSEP \(holdout externo\) : (.*)<- numero publicavel$",
     r"\1RMSEP (holdout externo) : \2<- numero publicavel", r"\1RMSEP (external holdout) : \2<- publishable number"),
    (r"^(\s*)\((\d+) amostra\(s\) do holdout sem modelo de regressao para a classe atribuida\)$",
     r"\g<0>", r"\1(\2 holdout sample(s) without a regression model for the assigned class)"),
    (r"^(\s*)DModY critico \(SIMCA\): (.*)$", r"\g<0>", r"\1Critical DModY (SIMCA): \2"),
    (r"^(\s*)Amostras fora do DModY: (.*)$", r"\g<0>", r"\1Samples outside DModY: \2"),
    (r"^(\s*)Per-species figures of merit:$", r"\1Figuras de merito por especie:", r"\1Per-species figures of merit:"),
    (r"^(\s*)Species(\s+)LVs(.*)$", r"\1Especie\2VLs\3", r"\g<0>"),
    (r"^(\s*)IC\((.*?)\): LOD (.*)\|  validacao: (.*) DP=(.*)$", r"\g<0>",
     r"\1CI(\2): LOD \3|  validation: \4 SD=\5"),
    (r"^(\s*)Intervalo de predicao \(conforme, 95%\): (.*) \((\S+) grupos de validacao\)$", r"\g<0>",
     r"\1Prediction interval (conformal, 95%): \2 (\3 validation groups)"),
    (r"^(\s*)Intervalo de predicao: NAO_VALIDADO \((\S+) grupo\(s\) de validacao, minimo 19 p/ alpha=0\.05\)$",
     r"\g<0>", r"\1Prediction interval: NOT_VALIDATED (\2 validation group(s), minimum 19 for alpha=0.05)"),
    (r"^(\s*)Figures of merit \(single pooled model\):$", r"\1Figuras de merito (modelo unico):",
     r"\1Figures of merit (single pooled model):"),
    (r"^(\s*)LOD, LOQ, RMSEP and the validation range are in the unit of the reference concentration\.$",
     r"\1LOD, LOQ, RMSEP e a faixa de validacao estao na unidade do teor de referencia.", r"\g<0>"),
    (r"^(\s*)Note: LOD/LOQ/SEN require physical replicates \(mae_id\) to$",
     r"\1Nota: LOD/LOQ/SEN exigem replicas fisicas (mae_id) para", r"\g<0>"),
    (r"^(\s*)estimate instrumental noise; 'n/a' means none available\.$",
     r"\1estimar o ruido instrumental; 'n/a' significa que nao ha.", r"\g<0>"),
    (r"^(\s*)Quantificacao por especie x adulterante \(R2cv, group-aware\)$",
     r"\1Quantificacao por especie x adulterante (R2cv, validacao agrupada)",
     r"\1Quantification per species x adulterant (R2cv, grouped validation)"),
    (r"^(\s*)Limiar de aceite: (.*)$", r"\g<0>", r"\1Acceptance threshold: \2"),
    (r"^(\s*)Combinacoes avaliadas: (\S+) \| aprovadas: (\S+) \| abaixo do limiar: (\S+) \| sem dados \(n/a\): (\S+)$",
     r"\g<0>", r"\1Combinations evaluated: \2 | passed: \3 | below threshold: \4 | no data (n/a): \5"),
    (r"^(\s*)>> (\S+) combinacoes abaixo de (.*)$", r"\g<0>", r"\1>> \2 combinations below \3"),
    (r"^(\s*)Combinacoes abaixo do limiar \(nao quantificaveis\):$", r"\g<0>",
     r"\1Combinations below the threshold (not quantifiable):"),
    (r"^(\s*)\(nenhuma — todas as combinacoes com dados passaram\)$", r"\g<0>",
     r"\1(none — every combination with data passed)"),
]
_LINHAS_RESUMO_RE = [(re.compile(p), pt, en) for p, pt, en in _LINHAS_RESUMO]


def linha_resumo(linha: str, lang: Optional[str] = None) -> str:
    """Uma linha dos blocos anexados ao resumo (regressao, heatmap) no idioma."""
    en = idioma(lang) == "en"
    for rx, pt, ing in _LINHAS_RESUMO_RE:
        if rx.match(linha):
            return rx.sub(ing if en else pt, linha)
    m = re.match(r"^(\s+)(\S[^:]*?)(\s*\.+)?: (.*)$", linha)   # "    RPD .......: 2.1  (faixa)"
    if m and not en:
        return f"{m.group(1)}{m.group(2)}{m.group(3) or ''}: {valor_resumo(m.group(4), lang)}"
    return linha


__all__ += ["rotulo_linha_resumo", "valor_resumo", "linha_resumo", "idioma_escolhido"]
