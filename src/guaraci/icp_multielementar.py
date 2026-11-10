"""Perfil multielementar (ICP-OES / ICP-MS): tabela amostra x elemento.

Nao e' espectro: cada coluna e' um elemento (com massa ou linha de emissao),
cada linha uma determinacao (amostra, branco, QC ou material de referencia).
ICP-OES e ICP-MS diferem so' em metadados e limites; o perfil e' um so'.

Entrada: CSV exportado do instrumento (separador e decimal detectados). Sem
formato proprietario. XLSX nao e' lido aqui (exigiria dependencia nova).

Regras (fail-closed):
- unidade declarada obrigatoria (mg/kg, ug/L, ..., ou 'mista'); sem unidade -> recusa;
- nome de elemento normalizado para "Simbolo massa" ou "Simbolo linha"
  (`Cd 111`, `Cd 228.802`); dois cabecalhos que viram o mesmo nome -> erro,
  nunca adivinhacao;
- valores censurados (`<LOD`, `<LOQ`, `ND`, `BDL`, `<0,05`) ficam marcados,
  com o limite quando o token o traz; a fracao censurada por elemento e'
  registrada;
- duas politicas de censura e analise de sensibilidade obrigatoria
  (`sensibilidade_censura`): a conclusao e' refeita com as duas e o relatorio
  diz se muda. Nenhuma e' escolhida pelo resultado.
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

__all__ = [
    "SIMBOLOS",
    "UNIDADES_ACEITAS",
    "TabelaMultielementar",
    "normalizar_elemento",
    "ler_icp_csv",
    "aplicar_censura",
    "sensibilidade_censura",
    "relatorio_qc",
    "relatorio_multielementar",
    "SELO_MATURIDADE",
]

#: Simbolos quimicos (1-118). Usado para reconhecer coluna de elemento.
SIMBOLOS = frozenset("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe
Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba
La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv
Ts Og""".split())

UNIDADES_ACEITAS = frozenset({"mg/kg", "ug/kg", "µg/kg", "ng/g", "ug/g", "µg/g", "mg/L",
                              "ug/L", "µg/L", "ng/L", "ng/mL", "ppm", "ppb", "%", "mista"})
# "mista": colunas em unidades diferentes (ou a fonte nao declara). So' vale para
# analises invariantes a escala de cada elemento (ANOVA, autoescala); nao
# permite comparar magnitudes entre elementos nem recuperacao de CRM.

TIPOS_LINHA = {"amostra": "amostra", "sample": "amostra", "branco": "branco",
               "blank": "branco", "qc": "qc", "crm": "crm", "mrc": "crm",
               "srm": "crm", "referencia": "crm",
               # linha com o limite de deteccao de cada elemento (muitos laudos de ICP
               # trazem "<LD" sem numero e o LD numa linha propria do arquivo)
               "ld": "lod", "lod": "lod", "dl": "lod", "limite_deteccao": "lod",
               # linha com o valor CERTIFICADO de cada elemento do material de referencia (as linhas
               # `crm` sao as medidas dele): sem ela a recuperacao do MRC nao tem contra o que comparar
               "crm_cert": "crm_cert", "crm_certificado": "crm_cert", "certificado": "crm_cert",
               "cert": "crm_cert", "valor_certificado": "crm_cert"}

_MAX_BYTES = 50 * 1024 * 1024
_TOKEN_CENSURA = re.compile(r"^\s*(?:<\s*(?:LOD|LOQ|LD|LQ|DL|QL)|ND|N\.D\.|BDL|<\s*([0-9]+(?:[.,][0-9]+)?))\s*$",
                            re.IGNORECASE)
# simbolo + massa (Cd111, Cd 111, 111Cd) ou simbolo + linha (Cd 228.802, Cd228,802)
_RX_ELEM = [
    re.compile(r"^(?P<s>[A-Z][a-z]?)\s*[-_ ]?\s*(?P<m>\d{1,3})(?![\d.,])"),
    re.compile(r"^(?P<m>\d{1,3})\s*(?P<s>[A-Z][a-z]?)(?![a-z])"),
    re.compile(r"^(?P<s>[A-Z][a-z]?)\s*[-_ ]?\s*(?P<l>\d{3}[.,]\d{1,4})"),
    re.compile(r"^(?P<s>[A-Z][a-z]?)(?![A-Za-z])"),
]


@dataclass
class TabelaMultielementar:
    amostras: List[str]
    tipos: List[str]                      # amostra | branco | qc | crm
    grupos: Optional[List[str]]
    unidades_fisicas: Optional[List[str]]
    sessoes: Optional[List[str]]
    elementos: List[str]
    valores: np.ndarray                   # NaN onde censurado/ausente
    censurado: np.ndarray                 # bool
    limite_censura: np.ndarray            # limite do token ("<0,05") ou NaN
    unidade: str
    padrao_interno: Optional[str] = None
    colunas_ignoradas: List[str] = field(default_factory=list)
    lod: Dict[str, float] = field(default_factory=dict)   # da linha Tipo=LD do arquivo
    crm_certificado: Dict[str, float] = field(default_factory=dict)   # da linha Tipo=CRM_CERT do arquivo

    def fracao_censurada(self, so_amostras: bool = True) -> Dict[str, float]:
        m = np.array([t == "amostra" for t in self.tipos]) if so_amostras else np.ones(len(self.tipos), bool)
        if not m.any():
            return {e: float("nan") for e in self.elementos}
        return {e: float(self.censurado[m, j].mean()) for j, e in enumerate(self.elementos)}


def normalizar_elemento(cabecalho: str) -> Optional[str]:
    """'Cd 111', 'Cd111', '111Cd', 'Cd 228.802 (mg/kg)' -> nome canonico;
    None se o cabecalho nao for elemento."""
    texto = re.sub(r"\(.*?\)|\[.*?\]", "", str(cabecalho)).strip()
    for rx in _RX_ELEM:
        m = rx.match(texto)
        if not m:
            continue
        s = m.group("s")
        if s not in SIMBOLOS:
            return None
        d = m.groupdict()
        if d.get("m"):
            return f"{s} {int(d['m'])}"
        if d.get("l"):
            return f"{s} {d['l'].replace(',', '.')}"
        resto = texto[m.end():].strip()
        return s if not resto else None
    return None


def _detectar_formato(amostra_texto: str) -> Tuple[str, str]:
    try:
        sep = csv.Sniffer().sniff(amostra_texto, delimiters=";,\t").delimiter
    except csv.Error:
        sep = ";" if amostra_texto.count(";") > amostra_texto.count(",") else ","
    decimal = "," if sep in (";", "\t") and re.search(r"\d,\d", amostra_texto) else "."
    return sep, decimal


def _para_float(v: Any, decimal: str) -> Tuple[float, bool, float]:
    """(valor, censurado, limite)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan"), False, float("nan")
    s = str(v).strip()
    if s == "":
        return float("nan"), False, float("nan")
    m = _TOKEN_CENSURA.match(s)
    if m:
        lim = float(m.group(1).replace(",", ".")) if m.group(1) else float("nan")
        return float("nan"), True, lim
    if decimal == ",":
        s = s.replace(".", "").replace(",", ".") if re.search(r"\d\.\d{3},", s) else s.replace(",", ".")
    try:
        return float(s), False, float("nan")
    except ValueError:
        raise ValueError(f"valor nao numerico e nao reconhecido como censura: {v!r}")


def ler_icp_csv(caminho: str, *, unidade: str, coluna_amostra: str,
                coluna_tipo: Optional[str] = None, coluna_grupo: Optional[str] = None,
                coluna_unidade_fisica: Optional[str] = None,
                coluna_sessao: Optional[str] = None,
                padrao_interno: Optional[str] = None) -> TabelaMultielementar:
    """Le o CSV do instrumento. `unidade` e' obrigatoria (ex.: 'mg/kg')."""
    if not unidade or unidade.strip() not in UNIDADES_ACEITAS:
        raise ValueError(f"unidade obrigatoria e reconhecida: recebido {unidade!r}; "
                         f"aceitas: {sorted(UNIDADES_ACEITAS)}")
    p = Path(caminho)
    tamanho = p.stat().st_size
    if tamanho > _MAX_BYTES:
        raise ValueError(f"arquivo com {tamanho} bytes excede o limite de {_MAX_BYTES} bytes")
    bruto = p.read_bytes()
    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = bruto.decode("latin-1")
    sep, decimal = _detectar_formato(texto[:20000])
    df = pd.read_csv(io.StringIO(texto), sep=sep, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for obrig in [coluna_amostra] + [c for c in (coluna_tipo, coluna_grupo,
                                                  coluna_unidade_fisica, coluna_sessao) if c]:
        if obrig not in df.columns:
            raise ValueError(f"coluna {obrig!r} nao encontrada; colunas: {list(df.columns)}")
    meta = {coluna_amostra, coluna_tipo, coluna_grupo, coluna_unidade_fisica, coluna_sessao}
    elementos: List[str] = []
    origem: Dict[str, str] = {}
    ignoradas: List[str] = []
    for c in df.columns:
        if c in meta:
            continue
        nome = normalizar_elemento(c)
        if nome is None:
            ignoradas.append(c)
            continue
        if nome in origem:
            raise ValueError(f"cabecalhos ambiguos: {origem[nome]!r} e {c!r} viram o mesmo "
                             f"elemento {nome!r}; renomeie um deles (massa ou linha)")
        origem[nome] = c
        elementos.append(nome)
    if not elementos:
        raise ValueError("nenhuma coluna reconhecida como elemento (ex.: 'Cd 111', 'Pb 220.353')")
    n, k = len(df), len(elementos)
    vals = np.full((n, k), np.nan)
    cens = np.zeros((n, k), dtype=bool)
    lim = np.full((n, k), np.nan)
    for j, e in enumerate(elementos):
        for i, v in enumerate(df[origem[e]].tolist()):
            vals[i, j], cens[i, j], lim[i, j] = _para_float(v, decimal)
    if coluna_tipo:
        tipos = []
        for t in df[coluna_tipo].tolist():
            chave = str(t).strip().lower()
            if chave not in TIPOS_LINHA:
                raise ValueError(f"tipo de linha desconhecido {t!r}; use {sorted(set(TIPOS_LINHA.values()))}")
            tipos.append(TIPOS_LINHA[chave])
    else:
        tipos = ["amostra"] * n
    if padrao_interno is not None:
        pi = normalizar_elemento(padrao_interno) or padrao_interno
        if pi not in elementos:
            raise ValueError(f"padrao interno {padrao_interno!r} nao esta entre os elementos lidos")
        padrao_interno = pi
    col = lambda c: [str(x).strip() for x in df[c].tolist()] if c else None  # noqa: E731
    lod_arquivo: Dict[str, float] = {}
    for i in [i for i, t in enumerate(tipos) if t == "lod"]:
        for j, e in enumerate(elementos):
            if np.isfinite(vals[i, j]) and vals[i, j] > 0:
                lod_arquivo[e] = float(vals[i, j])
    cert_arquivo: Dict[str, float] = {}
    linhas_cert = [i for i, t in enumerate(tipos) if t == "crm_cert"]
    if len(linhas_cert) > 1:
        raise ValueError(f"{len(linhas_cert)} linhas Tipo=CRM_CERT: use uma so' (valores certificados de um unico "
                         "material de referencia); para mais de um material, rode separado")
    for i in linhas_cert:
        for j, e in enumerate(elementos):
            if np.isfinite(vals[i, j]) and vals[i, j] > 0:
                cert_arquivo[e] = float(vals[i, j])
    return TabelaMultielementar(
        amostras=col(coluna_amostra) or [], tipos=tipos, grupos=col(coluna_grupo),
        unidades_fisicas=col(coluna_unidade_fisica), sessoes=col(coluna_sessao),
        elementos=elementos, valores=vals, censurado=cens, limite_censura=lim,
        unidade=unidade.strip(), padrao_interno=padrao_interno,
        colunas_ignoradas=ignoradas, lod=lod_arquivo, crm_certificado=cert_arquivo)


def aplicar_censura(tab: TabelaMultielementar, *, politica: str, limiar: float = 0.5,
                    lod: Optional[Dict[str, float]] = None, divisor: float = math.sqrt(2),
                    por_elemento: bool = False
                    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Matriz das linhas de AMOSTRA pronta para modelagem.

    politica='substituir': elementos com fracao censurada > limiar saem; nos
      demais, celula censurada vira LOD/divisor (divisor sqrt(2) ou 2,
      declarado). O LOD vem do token ('<0,05') ou de `lod[elemento]`; sem
      nenhum dos dois -> erro (nao inventa limite).
    politica='excluir': elementos com fracao censurada > limiar saem; nos
      demais, a linha com valor censurado sai.
    Celula VAZIA (ausente, nao censurada) nao e' censura: a linha sai nas duas
      politicas, contada a parte (`linhas_descartadas_por_ausencia`).
    O LOD sem numero no token ('<LD') vem de `lod` ou da linha Tipo=LD do arquivo.
    O padrao interno (`tab.padrao_interno`) nunca entra na matriz: e' adicionado
      em concentracao constante e so' carrega a deriva do instrumento.
    por_elemento=True (so' para analises por elemento, ex.: ANOVA): na politica
      'excluir' a celula censurada fica NaN e sai so' daquele elemento, em vez de
      derrubar a linha inteira (o que tiraria poder de elementos sem censura).
    Devolve (X, elementos_mantidos, registro)."""
    if politica not in ("substituir", "excluir"):
        raise ValueError("politica deve ser 'substituir' ou 'excluir'")
    if not 0.0 <= limiar <= 1.0:
        raise ValueError("limiar deve estar entre 0 e 1")
    lod = {**tab.lod, **(lod or {})}
    m = np.array([t == "amostra" for t in tab.tipos])
    frac = tab.fracao_censurada()
    fora = [e for e in tab.elementos if frac[e] > limiar and e != tab.padrao_interno]
    mant = [e for e in tab.elementos if e not in fora and e != tab.padrao_interno]
    idx = [tab.elementos.index(e) for e in mant]
    X = tab.valores[m][:, idx].copy()
    C = tab.censurado[m][:, idx]
    L = tab.limite_censura[m][:, idx]
    # Ausencia (celula vazia, NaN NAO censurada) NAO e' censura: nao ha limite
    # de deteccao a substituir. A linha com ausencia sai nas DUAS politicas e a
    # contagem fica separada da censura no registro.
    ausente = np.isnan(X) & ~C
    com_ausencia = ausente.any(axis=1)
    if politica == "substituir":
        for j, e in enumerate(mant):
            for i in np.where(C[:, j])[0]:
                limite = L[i, j] if np.isfinite(L[i, j]) else lod.get(e, float("nan"))
                if not np.isfinite(limite):
                    raise ValueError(f"celula censurada sem limite em {e!r} (linha {i}): "
                                     f"escreva o limite no token ('<0,005'), ou inclua no CSV uma "
                                     f"linha com Tipo=LD e o LD de cada elemento, ou use "
                                     f"politica='excluir'")
                X[i, j] = limite / divisor
        linhas_mantidas = ~com_ausencia
    elif por_elemento:
        linhas_mantidas = ~com_ausencia
    else:
        linhas_mantidas = ~np.isnan(X).any(axis=1)
    n_aus = int(com_ausencia.sum())
    n_cens = int((~linhas_mantidas).sum()) - n_aus
    X = X[linhas_mantidas]
    registro = {"politica": politica, "limiar": limiar, "divisor": divisor if politica == "substituir" else None,
                "fracao_censurada": frac, "elementos_excluidos": fora,
                "padrao_interno_fora_do_modelo": tab.padrao_interno,
                "celulas_ausentes": int(ausente.sum()),
                "linhas_descartadas_por_ausencia": n_aus,
                "linhas_descartadas_por_censura": n_cens,
                "linhas_descartadas": int((~linhas_mantidas).sum()),
                "indices_amostra": np.where(m)[0][linhas_mantidas].tolist()}
    return X, mant, registro


def sensibilidade_censura(tab: TabelaMultielementar,
                          conclusao: Callable[[np.ndarray, List[str], Dict[str, Any]], Dict[str, Any]],
                          *, limiar: float = 0.5, lod: Optional[Dict[str, float]] = None
                          ) -> Dict[str, Any]:
    """Roda `conclusao(X, elementos, registro)` com as DUAS politicas e diz se
    a conclusao muda. `conclusao` devolve um dict de resultados comparaveis
    (ex.: {elemento: significativo?}). Obrigatorio no relatorio."""
    out: Dict[str, Any] = {}
    for pol in ("substituir", "excluir"):
        try:
            X, els, reg = aplicar_censura(tab, politica=pol, limiar=limiar, lod=lod)
            out[pol] = {"resultado": conclusao(X, els, reg), "registro": reg}
        except ValueError as e:
            out[pol] = {"erro": str(e)}
    a, b = out["substituir"].get("resultado"), out["excluir"].get("resultado")
    if a is None or b is None:
        out["muda"] = None
        out["diferencas"] = "uma das politicas nao pode ser aplicada (ver 'erro')"
    else:
        chaves = sorted(set(a) | set(b))
        dif = {k: (a.get(k), b.get(k)) for k in chaves if a.get(k) != b.get(k)}
        out["muda"] = bool(dif)
        out["diferencas"] = dif
    return out


def relatorio_qc(tab: TabelaMultielementar, *,
                 crm_certificado: Optional[Dict[str, float]] = None,
                 faixa_recuperacao: Tuple[float, float] = (80.0, 120.0),
                 lod: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Recuperacao do material de referencia por elemento (limites declarados
    pelo usuario -> alerta, nao bloqueio), brancos acima do LOD, deriva do
    padrao interno ao longo da sequencia. Nenhuma alegacao de conformidade."""
    lod = {**tab.lod, **(lod or {})}
    crm_certificado = {**tab.crm_certificado, **(crm_certificado or {})}
    tipos = np.array(tab.tipos)
    out: Dict[str, Any] = {"unidade": tab.unidade, "alertas": []}
    # o que foi (ou nao) verificavel: o relatorio nunca deve calar uma checagem que nao rodou
    ver: Dict[str, str] = {
        "recuperacao_crm": ("ok" if crm_certificado and (tipos == "crm").any()
                            else "sem_linhas_crm" if crm_certificado else "sem_valor_certificado"),
        "brancos_vs_lod": ("sem_brancos" if not (tipos == "branco").any()
                           else "ok" if lod else "sem_lod"),
        "padrao_interno": "ok" if tab.padrao_interno else "sem_padrao_interno",
    }
    out["verificacoes"] = ver
    if crm_certificado:
        rec: Dict[str, Optional[float]] = {}
        m = tipos == "crm"
        for e, cert in crm_certificado.items():
            en = normalizar_elemento(e) or e
            if en not in tab.elementos or not m.any() or cert <= 0:
                rec[en] = None
                continue
            obs = tab.valores[m, tab.elementos.index(en)]
            obs = obs[np.isfinite(obs)]
            rec[en] = float(100 * obs.mean() / cert) if len(obs) else None
            if rec[en] is not None and not (faixa_recuperacao[0] <= rec[en] <= faixa_recuperacao[1]):
                out["alertas"].append(f"recuperacao de {en} no material de referencia = {rec[en]:.1f}% "
                                      f"(fora de {faixa_recuperacao[0]:g}-{faixa_recuperacao[1]:g}%)")
        out["recuperacao_crm_pct"] = rec
    mb = tipos == "branco"
    acima = []
    if mb.any():
        for j, e in enumerate(tab.elementos):
            limite = lod.get(e)
            if limite is None:
                continue
            v = tab.valores[mb, j]
            if np.nanmax(np.where(np.isfinite(v), v, -np.inf)) > limite:
                acima.append(e)
        if acima:
            out["alertas"].append(f"elemento(s) acima do LOD no branco: {acima}")
    out["brancos_acima_lod"] = acima
    if tab.padrao_interno:
        j = tab.elementos.index(tab.padrao_interno)
        v = tab.valores[:, j]
        ok = np.isfinite(v)
        if ok.sum() < 3:
            ver["padrao_interno"] = "poucas_linhas"
        if ok.sum() >= 3:
            ordem = np.arange(len(v))[ok]
            rel = 100 * v[ok] / np.median(v[ok])
            inclinacao = float(np.polyfit(ordem, rel, 1)[0])
            out["padrao_interno"] = {"elemento": tab.padrao_interno,
                                     "recuperacao_relativa_pct": rel.tolist(),
                                     "deriva_pct_por_linha": inclinacao,
                                     "amplitude_pct": float(np.ptp(rel))}
            if abs(inclinacao) * len(v) > 20:
                out["alertas"].append(f"deriva do padrao interno {tab.padrao_interno}: "
                                      f"{inclinacao:+.2f}% por linha")
    return out


# ---------------------------------------------------------------------------
#  Integracao com o pipeline (mode="multielementar")
# ---------------------------------------------------------------------------

#: Selo de maturidade (Partes C/D da instrucao ICP): a validacao e' de mecanismo
#: (dados sinteticos); padroes sao ponto de partida.
SELO_MATURIDADE = ("EM VALIDACAO -- mecanismo verificado em dados sinteticos; padroes sao ponto "
                   "de partida, com poucas unidades independentes por grupo; "
                   "sem comparacao com resultados publicados")
PREPROC_PERMITIDOS = ("autoscaling", "mc")
#: Politica de variancias da ANOVA por elemento: 'desiguais' = Welch ANOVA + Games-Howell (ver _anova_por_elemento).
POLITICA_VARIANCIAS_ANOVA = "desiguais"


def _transformar(X: np.ndarray, log: bool) -> np.ndarray:
    """Ordem fixa: censura -> log10(1+x) opcional -> (autoescala no pipeline,
    ajustada so' no treino de cada dobra). O log e' sem estado: nao vaza."""
    if not log:
        return X
    if np.nanmin(X) < 0:
        raise ValueError("log10(1+x) recusado: ha' concentracao negativa no arquivo")
    return np.log10(1.0 + X)


def _contexto_de_cfg(cfg: Any) -> Dict[str, Any]:
    return {
        "unidade": str(getattr(cfg, "icp_unidade", "") or "").strip(),
        "politica": str(getattr(cfg, "icp_politica_censura", "substituir") or "substituir"),
        "limiar": float(getattr(cfg, "icp_limiar_censura", 0.5)),
        "log": bool(getattr(cfg, "icp_log", False)),
        "divisor": "raiz(2)",
        # so' para a figura (dpi/formato configurados; nao entram em JSON/CSV)
        "fig_dpi": int(getattr(cfg, "save_dpi", 600) or 600),
        "fig_formato": str(getattr(cfg, "output_format", "png") or "png"),
    }


def _leitor_multielementar(cfg: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                              Optional[np.ndarray], Optional[np.ndarray],
                                              Optional[pd.DataFrame]]:
    """Leitor do io_registry. Coluna de classe (o grupo a comparar) e coluna
    de grupo (a unidade independente) sao obrigatorias: sem a unidade, a
    validacao e a ANOVA tratariam replicas como independentes."""
    if not getattr(cfg, "class_column", None):
        raise ValueError("mode multielementar: informe coluna_classe (o grupo a comparar)")
    if not getattr(cfg, "group_column", None):
        raise ValueError("mode multielementar: informe coluna_grupo (a amostra fisica, "
                         "lote ou produtor independente). Sem ela a classificacao e a "
                         "ANOVA sao recusadas (pseudo-replicacao).")
    preset = str(getattr(cfg, "default_preprocessing", "") or "").lower()
    if preset not in PREPROC_PERMITIDOS:
        raise ValueError(f"mode multielementar: pre_processamento {cfg.default_preprocessing!r} "
                         "e' espectral (MSC/SNV/Savitzky-Golay nao se aplicam a uma tabela "
                         "de elementos). Use Autoscaling (recomendado) ou Mean-centering.")
    ctx = _contexto_de_cfg(cfg)
    tab = ler_icp_csv(cfg.csv_file, unidade=ctx["unidade"],
                      coluna_amostra=str(getattr(cfg, "icp_coluna_amostra", "Amostra") or "Amostra"),
                      coluna_tipo=getattr(cfg, "icp_coluna_tipo", "") or None,
                      coluna_grupo=cfg.class_column,
                      coluna_unidade_fisica=cfg.group_column,
                      coluna_sessao=getattr(cfg, "icp_coluna_sessao", "") or None,
                      padrao_interno=getattr(cfg, "icp_padrao_interno", "") or None)
    X, els, reg = aplicar_censura(tab, politica=ctx["politica"], limiar=ctx["limiar"])
    if not els:
        raise ValueError("todos os elementos excederam o limiar de censura; nada para modelar")
    X = _transformar(X, ctx["log"])
    idx = reg["indices_amostra"]
    assert tab.grupos is not None and tab.unidades_fisicas is not None
    rotulos = np.asarray([tab.grupos[i] for i in idx], dtype=str)
    unidades = np.asarray([tab.unidades_fisicas[i] for i in idx], dtype=str)
    eixo = np.arange(1, len(els) + 1, dtype=float)
    # nomes de coluna que `dados_io.sanitizar_metadados` ja' trata: o codigo
    # da amostra ('arquivo') nunca vai para disco e a unidade ('mae_id') vira
    # 'grupo_replica' anonimo.
    meta = pd.DataFrame({"arquivo": [tab.amostras[i] for i in idx],
                         "grupo": rotulos, "mae_id": unidades})
    meta.attrs["multielementar"] = {"tabela": tab, "contexto": ctx, "elementos": els,
                                    "registro_censura": reg}
    print(f"[INFO] Multielementar: {len(idx)} amostra(s), {len(els)} elemento(s) "
          f"({tab.unidade}); {len(reg['elementos_excluidos'])} excluido(s) por censura "
          f"acima de {ctx['limiar']:.0%}: {reg['elementos_excluidos'] or 'nenhum'}. "
          f"{SELO_MATURIDADE}.")
    return eixo, X, rotulos, None, unidades, meta


def _anova_por_elemento(tab: TabelaMultielementar, politica: str, limiar: float, log: bool,
                        alpha: float, n_boot: int) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    from guaraci.anova_grupos import anova_varias_respostas
    X, els, reg = aplicar_censura(tab, politica=politica, limiar=limiar, por_elemento=True)
    X = _transformar(X, log)
    idx = reg["indices_amostra"]
    assert tab.grupos is not None and tab.unidades_fisicas is not None
    grupos = [tab.grupos[i] for i in idx]
    unidades = [tab.unidades_fisicas[i] for i in idx]
    # Politica fixa Welch + Games-Howell (= padrao do `guaraci anova`). O `auto` (Brown-Forsythe decide
    # Tukey x Games-Howell) levava o erro por familia a ~0,25 sob H0 quando o grupo pequeno tem a maior
    # variancia (simulacao com 200-500 repeticoes; o Games-Howell fixo ficou em ~0,07).
    out = anova_varias_respostas({e: X[:, j] for j, e in enumerate(els)}, grupos, unidades,
                                 alpha=alpha, n_boot=n_boot, variancias=POLITICA_VARIANCIAS_ANOVA)
    return out, els, reg


def _significativos(r: Any, alpha: float) -> Dict[str, bool]:
    if isinstance(r, Exception):
        return {}
    return {e: bool(res.p_fdr is not None and res.p_fdr < alpha)
            for e, res in r[0]["resultados"].items()}


def relatorio_multielementar(contexto: Dict[str, Any], pasta_tabelas: str,
                             pasta_relatorios: str, pasta_graficos: str, *,
                             alpha: float = 0.05, n_boot: int = 500) -> Dict[str, Any]:
    """ANOVA + pos-teste por elemento (FDR entre elementos), analise de
    sensibilidade da censura (as duas politicas), QC e efeito de sessao.
    Grava tabelas/JSON/figura e devolve o resumo para o model card."""
    import json
    import os

    tab: TabelaMultielementar = contexto["tabela"]
    ctx = contexto["contexto"]
    resultados: Dict[str, Any] = {}
    for pol in ("substituir", "excluir"):
        try:
            resultados[pol] = _anova_por_elemento(tab, pol, ctx["limiar"], ctx["log"], alpha, n_boot)
        except ValueError as erro:
            resultados[pol] = erro
    principal = resultados[ctx["politica"]]
    if isinstance(principal, Exception):
        raise ValueError(f"ANOVA por elemento impossivel com a politica '{ctx['politica']}': {principal}")
    out, els, reg = principal

    sig = {pol: _significativos(r, alpha) for pol, r in resultados.items()}
    if any(isinstance(r, Exception) for r in resultados.values()):
        sens: Dict[str, Any] = {"muda": None, "motivo": {p: str(r) for p, r in resultados.items()
                                                       if isinstance(r, Exception)}}
    else:
        chaves = sorted(set(sig["substituir"]) | set(sig["excluir"]))
        dif = {k: {"substituir": sig["substituir"].get(k), "excluir": sig["excluir"].get(k)}
               for k in chaves if sig["substituir"].get(k) != sig["excluir"].get(k)}
        sens = {"muda": bool(dif), "diferencas": dif,
                "significativos_substituir": sorted(k for k, v in sig["substituir"].items() if v),
                "significativos_excluir": sorted(k for k, v in sig["excluir"].items() if v)}

    linhas, comps = [], []
    for el, r in out["resultados"].items():
        linhas.append({"elemento": el, "metodo": r.metodo, "pos_teste": r.pos_teste,
                       "n_unidades": r.n_unidades, "F": r.F, "p": r.p, "p_fdr": r.p_fdr,
                       "eta2": r.eta2, "eta2_ic95_inf": r.eta2_ic95[0],
                       "eta2_ic95_sup": r.eta2_ic95[1], "omega2": r.omega2,
                       "levene_p": r.levene_p, "avisos": " | ".join(r.avisos)})
        for c in r.comparacoes:
            comps.append({"elemento": el, "grupo_a": c["a"], "grupo_b": c["b"],
                          "diferenca": c["diferenca"], "ic95_inf": c["ic95"][0],
                          "ic95_sup": c["ic95"][1], "p_ajustado": c["p_ajustado"]})
    pd.DataFrame(linhas).to_csv(os.path.join(pasta_tabelas, "multielementar_anova_por_elemento.csv"),
                                index=False, sep=";", decimal=",")
    pd.DataFrame(comps).to_csv(os.path.join(pasta_tabelas, "multielementar_comparacoes.csv"),
                               index=False, sep=";", decimal=",")

    qc = relatorio_qc(tab)
    sessao: Dict[str, Any] = {"testado": False}
    tipos = np.array(tab.tipos)
    m_qc = tipos == "qc"
    if (tab.sessoes is not None and int(m_qc.sum()) >= 4
            and len(set(np.array(tab.sessoes)[m_qc])) >= 2):
        from guaraci.anova_grupos import anova_varias_respostas
        sess = list(np.array(tab.sessoes)[m_qc])
        ids = [f"qc{i}" for i in np.where(m_qc)[0]]
        resp = {e: tab.valores[m_qc, j] for j, e in enumerate(tab.elementos)}
        o = anova_varias_respostas(resp, sess, ids, alpha=alpha, n_boot=0,
                                   calcular_pos_teste=False)
        sessao = {"testado": True,
                  "unidade": "cada injecao de QC (mesmo material medido em sessoes diferentes)",
                  "elementos_com_efeito_de_sessao": sorted(
                      e for e, r in o["resultados"].items() if r.p_fdr is not None and r.p_fdr < alpha),
                  "nao_testados": o["recusadas"]}
    else:
        sessao["motivo"] = "sem coluna de sessao ou menos de 4 linhas de QC em >= 2 sessoes"

    fig = _figura_medias(out["resultados"], tab.unidade,
                         os.path.join(pasta_graficos, "fig_multielementar_medias.png"), alpha,
                         valores=_valores_por_unidade(tab, ctx), ctx=ctx)
    resumo = {
        "selo": SELO_MATURIDADE,
        "unidade": tab.unidade, "politica_censura": ctx["politica"],
        "limiar_censura": ctx["limiar"], "divisor_lod": ctx["divisor"],
        "transformacao": "log10(1+x) -> autoescala" if ctx["log"] else "autoescala",
        "elementos_modelados": els, "elementos_excluidos": reg["elementos_excluidos"],
        "fracao_censurada": {k: round(v, 4) for k, v in reg["fracao_censurada"].items()},
        "linhas_descartadas_pela_censura": reg["linhas_descartadas_por_censura"],
        "linhas_descartadas_por_ausencia": reg["linhas_descartadas_por_ausencia"],
        "celulas_ausentes": reg["celulas_ausentes"],
        "anova": {"correcao": out["correcao"], "alpha": alpha,
                  "elementos_com_diferenca_entre_grupos": sorted(
                      k for k, v in sig[ctx["politica"]].items() if v),
                  "recusados": out["recusadas"]},
        "sensibilidade_censura": sens, "qc": qc, "efeito_de_sessao": sessao, "figura": fig,
        "colunas_ignoradas": tab.colunas_ignoradas,
    }
    with open(os.path.join(pasta_relatorios, "multielementar.json"), "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, ensure_ascii=False, indent=2, default=str)
    return resumo


def _valores_por_unidade(tab: "TabelaMultielementar", ctx: Dict[str, Any]
                         ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Valor de cada UNIDADE independente por elemento (mesma censura e
    transformacao da ANOVA principal) -- so' para a figura (pontos e DP)."""
    from guaraci.anova_grupos import agregar_por_unidade
    try:
        X, els, reg = aplicar_censura(tab, politica=ctx["politica"], limiar=ctx["limiar"],
                                      por_elemento=True)
        X = _transformar(X, ctx["log"])
    except (ValueError, KeyError):
        return {}
    idx = reg["indices_amostra"]
    if tab.grupos is None or tab.unidades_fisicas is None:
        return {}
    grupos = [tab.grupos[i] for i in idx]
    unidades = [tab.unidades_fisicas[i] for i in idx]
    saida: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for j, e in enumerate(els):
        try:
            y_u, g_u, _, _ = agregar_por_unidade(cast(Sequence[float], X[:, j]), grupos, unidades)
        except ValueError:
            continue
        saida[e] = (np.asarray(y_u, dtype=float), np.asarray(g_u).astype(str))
    return saida


def _figura_medias(resultados: Dict[str, Any], unidade: str, caminho: str,
                   alpha: float, valores: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                   ctx: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Ate' 12 elementos com diferenca, uma imagem por elemento (teto
    `figuras.MAX_PAINEIS_POR_FIGURA`): `fig_multielementar_medias_01`, `_02`...
    Devolve o caminho da primeira imagem (None se nenhum elemento passou)."""
    import os
    from guaraci.figuras import GRAFICOS_POR_ITEM, nome_pagina, paginas
    sel = [e for e, r in resultados.items() if r.p_fdr is not None and r.p_fdr < alpha][:12]
    if not sel:
        return None
    blocos = paginas(sel, GRAFICOS_POR_ITEM)
    base, ext = os.path.splitext(caminho)
    saidas = [_figura_medias_pagina(resultados, unidade,
                                    nome_pagina(base, i, len(blocos)) + ext, alpha,
                                    valores, ctx, bloco)
              for i, bloco in enumerate(blocos, 1)]
    return saidas[0]


def _figura_medias_pagina(resultados: Dict[str, Any], unidade: str, caminho: str,
                          alpha: float, valores: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
                          ctx: Optional[Dict[str, Any]], sel: List[str]) -> Optional[str]:
    """Media por grupo de cada elemento com diferenca (FDR < alpha), ate' 12
    elementos: barra = media das unidades independentes, barra de erro = DP
    entre unidades (ddof=1), pontos = cada unidade. Cores da paleta de
    classes. Sem letras de agrupamento (so' fariam sentido com desenho
    balanceado declarado). Grava via `figuras.save` (dpi/formato
    configurados, subpasta de grupo quando ligada)."""
    import os
    import types
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from guaraci.paleta_cores import map_class_colors

    ctx = ctx or {}
    valores = valores or {}
    log = bool(ctx.get("log", False))
    rot_y = f"log10(1 + x)  [{unidade}]" if log else unidade
    ncol = min(3, len(sel))
    nlin = int(math.ceil(len(sel) / ncol))
    fig, axs = plt.subplots(nlin, ncol, figsize=(4.2 * ncol, 3.4 * nlin), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, e in zip(axs.flat, sel):
        r = resultados[e]
        cores = map_class_colors(list(r.grupos))
        medias = [r.media_por_grupo[g] for g in r.grupos]
        y_u, g_u = valores.get(e, (np.array([]), np.array([], dtype=str)))
        pts = [y_u[g_u == str(g)] for g in r.grupos]
        dps = [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in pts]
        pos = np.arange(len(r.grupos))
        ax.bar(pos, medias, color=[cores[g] for g in r.grupos], alpha=0.55,
               edgecolor="0.25", lw=0.6, zorder=2)
        if valores:
            ax.errorbar(pos, medias, yerr=dps, fmt="none", ecolor="black",
                        elinewidth=1.0, capsize=4, zorder=4)
            for k, (g, v) in enumerate(zip(r.grupos, pts)):
                ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=14,
                           color=cores[g], edgecolors="black", linewidths=0.4,
                           zorder=5)
        ax.set_xticks(pos)
        ax.set_xticklabels(r.grupos, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{e}  (p FDR = {r.p_fdr:.3g})", fontsize=9)
        ax.set_ylabel(rot_y, fontsize=8)
    for ax in list(axs.flat)[len(sel):]:
        ax.axis("off")
    fig.suptitle("Mean per group ± SD between independent units (points = units)"
                 if valores else "Mean per group (independent units)", fontsize=10)
    fig.tight_layout()

    from guaraci import figuras
    nome = os.path.splitext(os.path.basename(caminho))[0]
    fmt = str(ctx.get("fig_formato", "png") or "png")
    dpi = int(ctx.get("fig_dpi", 600) or 600)
    pasta_g = os.path.dirname(os.path.abspath(caminho))
    from guaraci.config import nomes_pasta_execucao
    if os.path.basename(pasta_g) not in nomes_pasta_execucao(figuras.NOME_GRAFICOS):
        # chamada direta com outra pasta: grava exatamente onde pedido
        final = os.path.splitext(caminho)[0] + f".{fmt}"
        os.makedirs(pasta_g, exist_ok=True)
        from guaraci.idioma_figuras import traduzir_figura   # texto da figura no idioma da sessao
        traduzir_figura(fig)
        fig.savefig(final, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return final
    pasta = os.path.dirname(pasta_g)
    cfg_fig = types.SimpleNamespace(output_format=fmt, show_plots=False, mode="multielementar")
    with mpl.rc_context({"savefig.dpi": dpi, "savefig.bbox": "tight"}):
        final = figuras.save(fig, nome, pasta, cast(Any, cfg_fig))   # caminho real (grupo e serie)
    plt.close(fig)
    return final if os.path.exists(final) else None


def _registrar() -> None:
    from guaraci.io_registry import register_reader
    register_reader("multielementar", _leitor_multielementar)


_registrar()
