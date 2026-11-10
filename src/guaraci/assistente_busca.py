"""assistente_busca.py -- busca BM25 com sinonimos sobre o catalogo (offline, deterministica, so' stdlib).

BM25 (Robertson & Zaragoza 2009, Found. Trends Inf. Retr. 4(1-2):1-174;
DOI 10.1561/1500000019) ranqueia os documentos do catalogo por uma consulta em linguagem
livre. Duas diferencas em relacao ao ranqueador por palavra-chave de `assistente.buscar`:

  * o peso de cada palavra cai com a frequencia dela no catalogo (idf): "amostras" quase nao decide,
    "outlier" decide;
  * a consulta e' expandida com sinonimos do vocabulario da area (SINONIMOS), com peso menor que a
    palavra digitada, e a resposta traz a COBERTURA: quantos conceitos da consulta o documento
    realmente contem. Consulta sem cobertura (futebol, receita de bolo) nao vira resposta.

Nao ha' modelo treinado nem rede: o resultado e' funcao so' do texto do catalogo e desta tabela.
O limiar de resposta direta e' conservador e esta' em `decidir`; quem usa decide o que fazer
com o resultado (ver `assistente.responder_estruturado`).
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = ["Documento", "Resultado", "ranquear", "decidir", "SINONIMOS", "radical", "conceitos"]

_K1, _B = 1.5, 0.75
_PESO_SINONIMO = 0.6
_TAM_RADICAL = 5

_STOP = frozenset((
    "o a os as um uma uns umas de do da dos das no na nos nas em e ou que qual quais como para por com sem se ao "
    "aos eu me meu minha meus minhas isso isto este esta estes estas quero fazer preciso saber significa explique "
    "explica ter tem tenho tinha sao ser esta estao foi vale pena pode posso devo deve qualquer algum alguma "
    "the of how to an is are was do does did my i me it its this that these those for on in at by with can should "
    "what which when where why who a and or not mas porem entao ja so aqui ali muito muita muitos muitas bem "
    "mesmo assim depois antes quando onde porque por que").split())

#: Grupos de palavras equivalentes no uso da area. Cada grupo e' uma lista de palavras; a consulta que contem
#: uma delas tambem busca as demais, com peso menor. Vocabulario de quimiometria, nao de uma pergunta.
SINONIMOS: Tuple[Tuple[str, ...], ...] = (
    ("outlier", "anomalo", "anomala", "atipico", "estranho", "discrepante", "anomalous", "atypical", "abnormal"),
    ("lote", "batch", "sessao", "session", "dia", "dias", "day", "days", "data", "date", "semana", "week"),
    ("equipamento", "espectrometro", "instrumento", "aparelho", "maquina", "spectrometer", "instrument", "device",
     "transferencia", "transfer"),
    ("citar", "citacao", "cite", "citation", "referenciar"),
    ("publicar", "publicacao", "artigo", "paper", "submeter", "revista", "publish", "publication", "journal"),
    ("latente", "latentes", "fator", "fatores", "factor", "factors", "componente", "componentes", "component",
     "components", "lv", "latent"),
    ("acuracia", "acerto", "accuracy", "perfeito", "perfeita", "perfect"),
    ("rejeita", "rejeitada", "rejeitado", "rejeitadas", "rejeitados", "rejected", "reject", "rejects"),
    ("erro", "error", "rmsep", "rmsecv", "rmsec", "previsao", "predicao", "prediction"),
    ("abrir", "abre", "carregar", "carrega", "ler", "le", "load", "open", "read", "importar", "import"),
    ("desbalanceada", "desbalanceadas", "desbalanceamento", "desequilibrada", "desequilibrio", "imbalance",
     "imbalanced", "unbalanced"),
    ("janela", "window", "savitzky", "golay", "suavizacao", "smoothing"),
    ("faixa", "range", "regiao", "region", "intervalo", "onda", "wavenumber"),
    ("replica", "replicas", "replicate", "replicates", "repetida", "repetidas", "repeated", "repetidas"),
    ("preprocessamento", "pretratamento", "preprocessing", "pretreatment", "derivada", "derivative", "snv", "msc"),
    ("vazamento", "leakage", "contaminacao"),
    ("poucas", "pouca", "few", "pequeno", "pequena", "small"),
    ("normalidade", "normal", "gaussiana", "shapiro"),
    ("variancia", "variancias", "variance", "heterocedasticidade", "levene"),
)


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "")
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def radical(palavra: str) -> str:
    """Radical por truncamento: 'variaveis' e 'variavel' -> 'varia'. Palavras curtas ficam inteiras."""
    return palavra[:_TAM_RADICAL] if len(palavra) > _TAM_RADICAL else palavra


def _palavras(texto: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9]+", _norm(texto)) if len(w) > 1 and w not in _STOP]


def _radicais(texto: str) -> List[str]:
    return [radical(w) for w in _palavras(texto)]


_GRUPO: Dict[str, int] = {}
for _i, _g in enumerate(SINONIMOS):
    for _w in _g:
        _GRUPO.setdefault(radical(_norm(_w)), _i)
_MEMBROS: List[List[str]] = [sorted({radical(_norm(w)) for w in g}) for g in SINONIMOS]


@dataclass(frozen=True)
class Documento:
    tipo: str
    id: str
    texto: str          # tudo que descreve o documento (ambos os idiomas); a busca olha so' isto


@dataclass(frozen=True)
class Resultado:
    tipo: str
    id: str
    score: float        # BM25 bruto
    relativo: float     # score / maximo possivel para esta consulta (0 a 1)
    cobertura: float    # fracao dos conceitos da consulta presentes no documento (0 a 1)


def conceitos(consulta: str) -> List[List[Tuple[str, float]]]:
    """Cada palavra util da consulta vira um conceito: o radical digitado (peso 1) e os radicais dos
    sinonimos (peso `_PESO_SINONIMO`). Dois conceitos iguais contam uma vez."""
    vistos, out = set(), []
    for r in _radicais(consulta):
        if r in vistos:
            continue
        vistos.add(r)
        termos = [(r, 1.0)]
        g = _GRUPO.get(r)
        if g is not None:
            termos += [(m, _PESO_SINONIMO) for m in _MEMBROS[g] if m != r]
        out.append(termos)
    return out


class _Indice:
    def __init__(self, docs: Sequence[Documento]):
        self.docs = list(docs)
        self.tf: List[Dict[str, int]] = []
        self.len: List[int] = []
        df: Dict[str, int] = {}
        for d in self.docs:
            rs = _radicais(d.texto)
            tf: Dict[str, int] = {}
            for r in rs:
                tf[r] = tf.get(r, 0) + 1
            self.tf.append(tf)
            self.len.append(len(rs))
            for r in tf:
                df[r] = df.get(r, 0) + 1
        n = max(1, len(self.docs))
        self.avg = sum(self.len) / n
        self.idf = {r: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for r, c in df.items()}
        self.n = n

    def idf_de(self, r: str) -> float:
        # radical que nao existe no catalogo: nao ajuda a achar nada, mas conta como conceito da consulta
        return self.idf.get(r, math.log(1.0 + (self.n + 0.5) / 0.5))

    def ranquear(self, consulta: str) -> List[Resultado]:
        cons = conceitos(consulta)
        if not cons:
            return []
        maximo = sum(max(w * self.idf_de(r) for r, w in c) * (_K1 + 1.0) for c in cons) or 1.0
        out: List[Resultado] = []
        for d, tf, ln in zip(self.docs, self.tf, self.len):
            s, achou = 0.0, 0
            for c in cons:
                melhor = 0.0
                for r, w in c:
                    f = tf.get(r, 0)
                    if f:
                        v = w * self.idf_de(r) * f * (_K1 + 1.0) / (f + _K1 * (1.0 - _B + _B * ln / (self.avg or 1.0)))
                        melhor = max(melhor, v)
                if melhor > 0:
                    achou += 1
                    s += melhor
            if achou:
                out.append(Resultado(d.tipo, d.id, s, min(1.0, s / maximo), achou / len(cons)))
        out.sort(key=lambda x: (-x.score, x.tipo, x.id))
        return out


_CACHE: Dict[Tuple[int, ...], _Indice] = {}


def ranquear(consulta: str, docs: Iterable[Documento]) -> List[Resultado]:
    """Documentos que contem algum conceito da consulta, do mais ao menos relevante."""
    docs = list(docs)
    chave = (len(docs), hash(tuple((d.tipo, d.id, len(d.texto)) for d in docs)))
    ix = _CACHE.get(chave)
    if ix is None:
        _CACHE.clear()
        ix = _CACHE[chave] = _Indice(docs)
    return ix.ranquear(consulta)


#: Resposta direta so' quando o primeiro resultado e' claramente melhor que o segundo e cobre a consulta.
LIMIAR_RELATIVO = 0.50
LIMIAR_COBERTURA = 0.60
MARGEM = 1.25
#: Abaixo disto a consulta nao tem relacao com o catalogo: vale "nao sei", nem lista.
COBERTURA_MINIMA = 0.34
RELATIVO_MINIMO = 0.20


def decidir(rk: Sequence[Resultado]) -> str:
    """'direta' (o primeiro resultado responde), 'lista' (mostrar resultados) ou 'nada'."""
    if not rk or rk[0].cobertura < COBERTURA_MINIMA or rk[0].relativo < RELATIVO_MINIMO:
        return "nada"
    a = rk[0]
    b = rk[1].score if len(rk) > 1 else 0.0
    if a.relativo >= LIMIAR_RELATIVO and a.cobertura >= LIMIAR_COBERTURA and a.score >= MARGEM * b:
        return "direta"
    return "lista"
