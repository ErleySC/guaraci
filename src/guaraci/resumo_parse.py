"""resumo_parse.py — Parsing PURO do resumo_modelo.txt (item 19 da auditoria:
lógica testável extraída da UI).

O pipeline grava um resumo de texto livre (logs/resumo_modelo.txt); tanto a
aba Validation quanto os 5 geradores de relatório (guaraci.reports) extraíam
métricas dele por regex — cada gerador com sua PRÓPRIA cópia do helper `_ex`
(5 cópias idênticas) e PDF/Word repetindo o MESMO dicionário de 12 métricas.
Este módulo centraliza esse parsing: nada de I/O, só texto → dados, testável
em isolamento (ver tests/test_resumo_parse.py).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TypedDict

# Métricas padrão exibidas no cabeçalho de relatório (PDF/Word). Cada valor é
# (padrão_regex) aplicado ao resumo com IGNORECASE|MULTILINE; group(1) é o valor.
_PADROES_METRICAS = {
    "Balanced Accuracy (CV)":   r"[Bb]alanced[_ ]?[Aa]ccuracy.*?[:=]\s*([\d.]+)",
    "AUC macro OvR":            r"ROC AUC macro.*?[:=]\s*([\d.]+)",
    "R2Y":                      r"\bR2Y\b.*?[:=]\s*([\d.]+)",
    "Q2Y":                      r"\bQ2\b.*?[:=]\s*([\d.E+-]+)",
    "R2X":                      r"\bR2X\b.*?[:=]\s*([\d.]+)",
    "Optimal LVs":              r"Optimal LVs.*?[:=]\s*(\d+)",
    "p-value (permutation)":    r"p.?value.*?[:=]\s*([\d.E+-]+)",
    # a linha INTEIRA ("SNV -> mean-centering"): com [A-Za-z0-9_+]+ o relatorio mostrava so' "SNV".
    "Preprocessing":            r"[Pp]re.?[Pp]rocessamento[^:=\n]*[:=][ \t]*(\S[^\n]*)",
    "Hotelling T2 UCL (95%)":   r"[Hh]otelling.*?[:=]\s*([\d.]+)",
    "Q-residual UCL (95%)":     r"Q.residual.*?[:=]\s*([\d.E+-]+)",
    # "Total samples"/"Total classes" e' o que o pipeline grava hoje; "N treino"/"N. Classes" de
    # execucoes antigas (sem isto os relatorios mostravam "-" para o total de amostras).
    "n samples (training)":     r"(?:\bN\s+treino|^\s*Total (?:de )?(?:samples|amostras)).*?[:=]\s*(\d+)",
    "n classes":                r"(?:\bN\.?\s*Classes|^\s*Total (?:de )?classes)\s*[:=]\s*(\d+)",
}


# Contagens do conjunto de dados da execucao. As chaves do resumo mudaram de
# PT para EN ao longo do projeto ("Total de amostras" -> "Total samples"), e
# runs antigos continuam em disco -- os dois nomes sao aceitos, senao o painel
# de status mostraria "-" para uma execucao que tem o numero gravado.
_PADROES_CONTAGEM = {
    "amostras":  r"^\s*Total (?:de )?(?:samples|amostras)\s*[:=]\s*(\d+)",
    "variaveis": r"^\s*Total (?:de )?(?:variables|vari[aá]veis)\s*[:=]\s*(\d+)",
    "classes":   r"^\s*Total (?:de )?classes\s*[:=]\s*(\d+)",
    # "Amostra fisica" = grupo mae_id (todas as replicas do mesmo ponto de
    # coleta). E' o numero que importa para dizer se a validacao esta'
    # protegida contra vazamento -- por isso aparece no painel de status ao
    # lado do total de espectros.
    "amostras_fisicas": r"^\s*N grupos mae_id\s*[:=]\s*(\d+)",
}


def parse_dataset_counts(resumo: str) -> Dict[str, Optional[int]]:
    """Contagens reais da execucao (amostras/variaveis/classes) lidas do
    resumo_modelo.txt. Valor `None` quando a linha nao existe -- nunca zero,
    que se leria como "nenhuma amostra" em vez de "nao informado"."""
    contagens: Dict[str, Optional[int]] = {}
    for nome, padrao in _PADROES_CONTAGEM.items():
        bruto = extract_metric(resumo, padrao, default="")
        contagens[nome] = int(bruto) if bruto.isdigit() else None
    return contagens


def extract_metric(resumo: str, padrao: str, default: str = "-") -> str:
    """Extrai group(1) do primeiro casamento de `padrao` no resumo.

    IGNORECASE|MULTILINE, `.strip()` no valor; devolve `default` se nada casar.
    É o núcleo único do antigo `_ex` (que existia em 5 cópias nos geradores).
    """
    m = re.search(padrao, resumo or "", re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else default


def parse_model_metrics(resumo: str, default: str = "-") -> Dict[str, str]:
    """Dicionário padrão de 12 métricas (Balanced Accuracy, R2Y/Q2Y, LVs,
    p-valor, etc.) para o cabeçalho dos relatórios PDF/Word."""
    return {nome: extract_metric(resumo, padrao, default)
            for nome, padrao in _PADROES_METRICAS.items()}


def parse_accuracy_by_class(resumo: str) -> Dict[str, float]:
    """Extrai a acurácia (recall) por classe das linhas 'Acc <classe>: <val>'
    do resumo. Usado pela aba Validation para a tabela colorida por classe.
    Retorna {classe: valor_float}; dict vazio se não houver nenhuma linha."""
    acc: Dict[str, float] = {}
    for linha in (resumo or "").splitlines():
        m = re.match(r"\s*Acc\s+(.+?)\s*[:=]\s*([\d.]+)", linha)
        if m:
            acc[m.group(1).strip()] = float(m.group(2))
    return acc


__all__ = ["extract_metric", "parse_model_metrics", "parse_accuracy_by_class",
           "parse_dataset_counts", "modo_do_resumo", "parse_regressao", "parse_hsi_relatorio", "ler_hsi_pasta",
           "N_TESTE_HSI_INDICATIVO", "avisos_do_modelo"]


def ler_resumo_canonico(caminho_txt: str) -> Optional[str]:
    """Texto do resumo com as chaves fixas do pipeline (o que os padroes acima conhecem).

    Unica funcao com I/O deste modulo. Desde que o resumo_modelo.txt sai no idioma da sessao, o texto
    canonico mora ao lado, em resumo_modelo.json; execucoes antigas so' tem o .txt, ja' canonico.
    `None` se nenhum dos dois existir."""
    import json
    import os
    pasta = os.path.dirname(os.path.abspath(caminho_txt))
    try:
        with open(os.path.join(pasta, "resumo_modelo.json"), encoding="utf-8") as f:
            dados = json.load(f)
        if isinstance(dados, dict) and isinstance(dados.get("texto"), str):
            return dados["texto"]
    except (OSError, ValueError):
        pass
    try:
        with open(caminho_txt, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


# --------------------------------------------------------------------------
# Modo da execucao e metricas de regressao (quantificacao) -- usados pelos
# relatorios para mostrar so' as metricas que fazem sentido para o objetivo.
# --------------------------------------------------------------------------
MODO_QUANTIFICACAO = "quantificacao"
MODO_CLASSIFICACAO = "classificacao"
MODO_DESCONHECIDO = "geral"


def modo_do_resumo(resumo: str) -> str:
    """'quantificacao' | 'classificacao' | 'geral', pelo que o resumo contem.

    O bloco "PLS Regression ... Figures of Merit" so' e' gravado quando o objetivo
    resolvido e' quantificacao, e a permutacao/BCa so' quando e' classificacao
    (pipeline.py); sem nenhum dos dois, o nivel (N3 = teor; N1/N2 = classes) decide.
    'geral' = nao da' para saber (resumo antigo/parcial): o relatorio mostra o
    sumario de classificacao de sempre."""
    txt = resumo or ""
    if re.search(r"PLS Regression\s*[—-]\s*Analytical Figures of Merit", txt):
        return MODO_QUANTIFICACAO
    if re.search(r"^\s*Permutation p-value\s*[:=]", txt, re.MULTILINE):
        return MODO_CLASSIFICACAO
    nivel = extract_metric(txt, r"^\s*(?:Nivel|Level)\s*[:=]\s*(N\d)", "")
    if nivel == "N3":
        return MODO_QUANTIFICACAO
    if nivel in ("N1", "N2"):
        return MODO_CLASSIFICACAO
    return MODO_DESCONHECIDO


_NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf|n/a)"


def _float_ou_none(txt: Optional[str]) -> Optional[float]:
    try:
        v = float(str(txt).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


class HoldoutRegressao(TypedDict):
    """`RMSEP (holdout externo)` do resumo."""
    rmsep: Optional[float]
    n: int
    r2: Optional[float]
    bias: Optional[float]


class EspecieRegressao(TypedDict, total=False):
    """Linha da tabela por especie; as chaves de IC e de intervalo de predicao so' existem se o resumo as traz."""
    especie: str
    lvs: int
    rmsep: Optional[float]
    r2val: Optional[float]
    lod: Optional[float]
    loq: Optional[float]
    sen: Optional[float]
    sel: Optional[float]
    ic_conf: str
    lod_ic: str
    loq_ic: str
    faixa_validacao: str
    dp_validacao: str
    pi_validado: bool
    pi_margem: Optional[float]
    pi_n_grupos: int


class MapaCalorRegressao(TypedDict):
    limiar: float
    n_total: int
    n_ok: int
    n_abaixo: int
    n_na: int


class Regressao(TypedDict, total=False):
    """Retorno de `parse_regressao` (chaves sempre presentes, salvo `rpd_faixa_txt`)."""
    pooled: Dict[str, Optional[float]]
    holdout: Optional[HoldoutRegressao]
    especies: List[EspecieRegressao]
    fom_pooled: Dict[str, Optional[float]]
    heatmap: Optional[MapaCalorRegressao]
    rpd_faixa_txt: str


def parse_regressao(resumo: str) -> Regressao:
    """Metricas do bloco de regressao PLS do resumo (vazio se nao houver o bloco).

    Chaves: `pooled` (dict r2cal, r2cv, r2val, rmsec, rmsecv, rmsep, bias, sep, rpd, rer, rpd_faixa,
    dmody_crit, n_fora_dmody; valor None = ausente ou n/a), `holdout` (rmsep, n, r2, bias) ou None,
    `especies` (lista de dicts: especie, lvs, rmsep, r2val, lod, loq, sen, sel, lod_ic, loq_ic,
    faixa_validacao, dp_validacao, ic_conf, pi_validado, pi_margem, pi_n_grupos) e `fom_pooled` (lod, loq na unidade do teor de referencia),
    `heatmap` (n_total, n_ok, n_abaixo, n_na, limiar) ou None. Nada e' calculado aqui: so' lido."""
    txt = resumo or ""
    i0 = re.search(r"PLS Regression\s*[—-]\s*Analytical Figures of Merit", txt)
    saida: Regressao = {"pooled": {}, "holdout": None, "especies": [], "fom_pooled": {}, "heatmap": None}
    if not i0:
        return saida
    bloco = txt[i0.start():]
    pooled: Dict[str, Optional[float]] = {}
    nomes = {"R2cal": "r2cal", "R2cv": "r2cv", "R2val": "r2val", "RMSEC": "rmsec", "RMSECV": "rmsecv",
             "RMSEP": "rmsep", "Bias": "bias", "SEP": "sep", "RPD": "rpd", "RER": "rer"}
    for rot, chave in nomes.items():
        m = re.search(rf"^\s*{rot}\s*\.+\s*:\s*({_NUM})(?:[ \t]+\(([^)\n]*)\))?", bloco, re.MULTILINE | re.IGNORECASE)
        if m:
            pooled[chave] = _float_ou_none(m.group(1))
            if chave == "rpd" and m.group(2):
                saida["rpd_faixa_txt"] = m.group(2).strip()
    m = re.search(r"DModY critico \(SIMCA\)\s*:\s*(" + _NUM + ")", bloco)
    if m:
        pooled["dmody_crit"] = _float_ou_none(m.group(1))
    m = re.search(r"Amostras fora do DModY\s*:\s*(\d+)", bloco)
    if m:
        pooled["n_fora_dmody"] = float(m.group(1))
    saida["pooled"] = pooled
    m = re.search(r"RMSEP \(holdout externo\)\s*:\s*(" + _NUM + r")\s+n=(\d+)\s+R2=(" + _NUM + r")\s+bias=(" + _NUM + ")", bloco)
    if m:
        saida["holdout"] = {"rmsep": _float_ou_none(m.group(1)), "n": int(m.group(2)),
                            "r2": _float_ou_none(m.group(3)), "bias": _float_ou_none(m.group(4))}
    # tabela por especie: cabecalho "Species LVs RMSEP ...", linha de tracos, linhas de dados
    linhas = bloco.splitlines()
    especies: List[EspecieRegressao] = []
    for i, ln in enumerate(linhas):
        if re.match(r"^\s*Species\s+LVs\s+RMSEP", ln):
            j = i + 2
            while j < len(linhas):
                mm = re.match(r"^\s+(\S.*?)\s+(\d+)\s+(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s+("
                              + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*$", linhas[j])
                if not mm:
                    break
                esp: EspecieRegressao = {
                    "especie": mm.group(1), "lvs": int(mm.group(2)), "rmsep": _float_ou_none(mm.group(3)),
                    "r2val": _float_ou_none(mm.group(4)), "lod": _float_ou_none(mm.group(5)),
                    "loq": _float_ou_none(mm.group(6)), "sen": _float_ou_none(mm.group(7)),
                    "sel": _float_ou_none(mm.group(8))}
                j += 1
                while j < len(linhas) and linhas[j].startswith("      "):
                    sub = linhas[j].strip()
                    ic = re.match(r"IC\((\d+)%\):\s*LOD\s*\[([^\]]+)\]\s*LOQ\s*\[([^\]]+)\]\s*\|\s*validacao:\s*"
                                  r"\[([^\]]+)\]%?\s*DP=([^\s%]+)%?", sub)
                    if ic:
                        esp["ic_conf"], esp["lod_ic"], esp["loq_ic"] = ic.group(1), ic.group(2), ic.group(3)
                        esp["faixa_validacao"], esp["dp_validacao"] = ic.group(4), ic.group(5)
                    elif sub.startswith("Intervalo de predicao"):
                        nv = re.search(r"NAO_VALIDADO \((\d+) grupo", sub)
                        cf = re.search(r"\+/-(" + _NUM + r")%?\s*\((\d+) grupos", sub)
                        if nv:
                            esp["pi_validado"], esp["pi_n_grupos"] = False, int(nv.group(1))
                        elif cf:
                            esp["pi_validado"], esp["pi_margem"] = True, _float_ou_none(cf.group(1))
                            esp["pi_n_grupos"] = int(cf.group(2))
                    j += 1
                especies.append(esp)
            break
    saida["especies"] = especies
    fom: Dict[str, Optional[float]] = {}
    ini = bloco.find("Figures of merit (single pooled model)")
    if ini >= 0:
        for rot in ("LOD", "LOQ"):
            m = re.search(rf"^\s*{rot}\s*\.+\s*:\s*({_NUM})%?", bloco[ini:], re.MULTILINE)
            if m:
                fom[rot.lower()] = _float_ou_none(m.group(1))
    saida["fom_pooled"] = fom
    m = re.search(r"Limiar de aceite: R2cv >= ([\d.]+)\s*\n\s*Combinacoes avaliadas: (\d+) \| aprovadas: (\d+) \| "
                  r"abaixo do limiar: (\d+) \| sem dados \(n/a\): (\d+)", txt)
    if m:
        saida["heatmap"] = {"limiar": float(m.group(1)), "n_total": int(m.group(2)), "n_ok": int(m.group(3)),
                            "n_abaixo": int(m.group(4)), "n_na": int(m.group(5))}
    return saida


# --------------------------------------------------------------------------
# hsi_relatorio.txt (fluxo HSI: nao ha' resumo_modelo.txt)
# --------------------------------------------------------------------------
#: abaixo deste numero de OBJETOS de teste a sensibilidade/especificidade do HSI e' so' indicativa:
#: um objeto a mais ou a menos move a metrica em 1/n (limite de apresentacao, nao um teste estatistico).
N_TESTE_HSI_INDICATIVO = 10


def parse_hsi_relatorio(texto: str) -> Dict[str, Any]:
    """Le o `hsi_relatorio.txt` (hsi_pipeline.gravar_relatorio_hsi) num dicionario.

    Chaves: aceitas, total, rejeitadas, vls, garantia, notas (linhas de faixa/pre-processamento),
    n_teste_interno, n_teste_externo, classes (lista de dicts classe/sens/espec; None = n/d) e
    exploratorio (True se o relatorio diz que nao ha' garantia de agrupamento). Chave ausente = None/[]."""
    out: Dict[str, Any] = {"aceitas": None, "total": None, "rejeitadas": None, "vls": None, "garantia": None,
                              "notas": [], "n_teste_interno": None, "n_teste_externo": None,
                              "classes": [], "exploratorio": False}
    notas: List[str] = []
    for ln in (texto or "").splitlines():
        t = ln.strip()
        if not t:
            continue
        m = re.match(r"Gravacoes aceitas:\s*(\d+)/(\d+)\s*\(rejeitadas pelo controle de qualidade:\s*(\d+)\)", t)
        if m:
            out.update(aceitas=int(m.group(1)), total=int(m.group(2)), rejeitadas=int(m.group(3)))
            continue
        m = re.match(r"Variaveis latentes \(Wold\):\s*(\d+)", t)
        if m:
            out["vls"] = int(m.group(1))
            continue
        m = re.match(r"Garantia de agrupamento por objeto:\s*(\w+)", t)
        if m:
            out["garantia"] = m.group(1).lower()
            continue
        m = re.match(r"Objetos de teste:\s*interno\s*(\d+),\s*externo\s*(\d+)", t)
        if m:
            out.update(n_teste_interno=int(m.group(1)), n_teste_externo=int(m.group(2)))
            continue
        m = re.match(r"(.+?):\s*sensibilidade interna\s*(\S+)(?: \(IC95% ([\d.]+)-([\d.]+)\))?,\s*"
                     r"especificidade interna\s*(\S+)(?: \(IC95% ([\d.]+)-([\d.]+)\))?(?:;\s*n objetos = (\d+))?\s*$", t)
        if m:
            cl: List[Dict[str, Any]] = out["classes"]
            cl.append({"classe": m.group(1).strip(), "sens": _float_ou_none(m.group(2)),
                       "espec": _float_ou_none(m.group(5)),
                       "sens_ic": ([float(m.group(3)), float(m.group(4))] if m.group(3) else None),
                       "espec_ic": ([float(m.group(6)), float(m.group(7))] if m.group(6) else None),
                       "n": (int(m.group(8)) if m.group(8) else None)})
            continue
        m = re.match(r"Validacao:\s*CV por objeto,\s*(.+?);\s*(\d+) de (\d+) objetos", t)
        if m:
            nd = re.match(r"(\d+) dobras", m.group(1))
            out.update(tipo="cv_objetos", n_dobras=(int(nd.group(1)) if nd else 0),
                       n_objetos_avaliados=int(m.group(2)), n_objetos_total=int(m.group(3)))
            continue
        m = re.match(r"Validacao:\s*holdout unico de (\d+) objeto\(s\) de (\d+)", t)
        if m:
            out.update(tipo="holdout", n_dobras=0, n_objetos_avaliados=int(m.group(1)), n_objetos_total=int(m.group(2)))
            continue
        if t.startswith("Confianca por objeto:"):
            out["fora_da_amostra"] = "fora da amostra" in t.split(":", 1)[1][:25]
            continue
        m = re.match(r"Calibracao radiometrica:\s*(branco/escuro|nenhuma)", t)
        if m:
            out["calibracao"] = {"modo": "branco_escuro" if m.group(1) == "branco/escuro" else "nenhuma"}
            continue
        if t.startswith("AVISO:"):
            av: List[str] = out.setdefault("avisos", [])
            av.append(t[6:].strip())
            continue
        if t.startswith("Aviso: so'"):
            out["aviso_n_pequeno"] = t
            continue
        if t.startswith("Sem garantia de agrupamento"):
            out["exploratorio"] = True
            continue
        if t.startswith(("Faixa", "Pre-processamento")):
            notas.append(t)
    out["notas"] = notas
    return out


_RE_AVISO_MODELO = re.compile(r"^\s*\[AVISO-MODELO\]\s+(\S+)\s+PT=(.*?)\s+EN=(.*)$", re.MULTILINE)


def avisos_do_modelo(resumo: str) -> list:
    """Avisos do modelo gravados no resumo canonico: [(codigo, texto_pt, texto_en)], na ordem em
    que aparecem. Lista vazia se nao houver (execucoes antigas ou sem aviso)."""
    return [(m.group(1), m.group(2).strip(), m.group(3).strip())
            for m in _RE_AVISO_MODELO.finditer(resumo or "")]


def ler_hsi_pasta(rel: str) -> Dict[str, Any]:
    """Dados de uma execucao HSI a partir de `Relatorios/` (`rel`): `hsi_relatorio.json` quando existe (a fonte
    estruturada, gravada por hsi_pipeline.gravar_relatorio_hsi), senao o `.txt` (execucoes antigas, tratadas como
    holdout). Chaves: as de `parse_hsi_relatorio` mais tipo ("cv_objetos" | "holdout" | "externa_por_dia" | None),
    n_dobras, n_objetos_total, n_objetos_avaliados, n_objetos_por_classe, fora_da_amostra (True/False/None),
    calibracao, eixo_bandas, acuracia, acuracia_ic, avisos. Vazio se nao houver nenhum dos dois arquivos."""
    import json
    import os
    out: Dict[str, Any] = {}
    try:
        with open(os.path.join(rel, "hsi_relatorio.txt"), encoding="utf-8", errors="replace") as fh:
            out = dict(parse_hsi_relatorio(fh.read()))
    except OSError:
        pass
    try:
        with open(os.path.join(rel, "hsi_relatorio.json"), encoding="utf-8") as fh:
            j = json.load(fh)
    except (OSError, ValueError):
        return out
    if not isinstance(j, dict):
        return out
    v = j.get("validacao") if isinstance(j.get("validacao"), dict) else {}
    out.update(
        aceitas=j.get("n_gravacoes_aceitas"), total=j.get("n_gravacoes_total"),
        rejeitadas=j.get("n_gravacoes_rejeitadas"), vls=j.get("n_components"), garantia=j.get("grouping_guarantee"),
        calibracao=j.get("calibracao") or out.get("calibracao"), eixo_bandas=j.get("eixo_bandas") or {},
        tipo=v.get("tipo"), n_dobras=v.get("n_dobras"), n_objetos_total=v.get("n_objetos_total"),
        n_objetos_avaliados=v.get("n_objetos_avaliados"), n_objetos_por_classe=v.get("n_objetos_por_classe") or {},
        fora_da_amostra=v.get("confianca_por_objeto_fora_da_amostra"), acuracia=v.get("acuracia_objetos"),
        acuracia_ic=v.get("acuracia_objetos_ic95"), avisos=list(v.get("avisos") or []))
    if not out.get("notas") and j.get("pre_processamento"):
        out["notas"] = [str(n) for n in j["pre_processamento"]]
    pc = v.get("por_classe") or {}
    if pc:
        out["classes"] = [{"classe": c, "sens": d.get("sensibilidade"), "espec": d.get("especificidade"),
                           "sens_ic": d.get("sensibilidade_ic95"), "espec_ic": d.get("especificidade_ic95"),
                           "n": d.get("n_objetos")} for c, d in pc.items()]
    return out
