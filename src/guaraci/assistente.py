"""assistente.py -- Assistente Guaraci: deterministico, offline, sem LLM e sem rede.

Responde a partir do catalogo (`guaraci.catalogo`, `catalogo_ajuda`): nunca inventa
opcao, numero ou tecnica. Cinco usos:

  rota        "O que quero fazer?": objetivo + tipo de dado -> telas, config e exemplo
  explicar    opcao, tecnica, grafico, termo, tela ou comando, no gabarito
  sugerir     le a configuracao atual e propoe passos (nunca aplica sozinho)
  diagnosticar aviso ou erro da ultima execucao -> o que significa e o que fazer
  buscar      palavra-chave (difflib, stdlib) sobre todo o catalogo

  checar      configuracao + resumo dos dados + resultado -> achados com causa, acao e fonte
             (registro unico em `assistente_regras`; erros de execucao em `assistente_erros`)

Nenhum dado sai desta maquina. Resposta nunca relata validacao: o selo e a evidencia
ficam no assistente de confianca (`guaraci confianca`).
"""
from __future__ import annotations

import copy
import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from guaraci import assistente_busca as _bm25
from guaraci import assistente_erros as _erros
from guaraci import assistente_regras as _regras
from guaraci import catalogo as cat
from guaraci.catalogo_ajuda import DIAGNOSTICOS, ROTAS, TERMOS
from guaraci.catalogo_duvidas import DUVIDAS

__all__ = ["Sugestao", "rotas", "rota", "explicar", "buscar", "sugerir", "diagnosticar", "responder",
           "formatar_rota", "formatar_explicacao", "duvida", "formatar_duvida", "duvidas",
           "responder_estruturado", "checar", "ficha_vazamento"]


def _L(lang: str) -> int:
    return 1 if str(lang).upper() == "EN" else 0


_STOP = frozenset("o a os as um uma uns umas de do da dos das no na nos nas em e ou que qual quais como para por com "
                  "sem se ao aos eu me meu minha isso isto este esta quero fazer preciso saber significa "
                  "explique explica what is the of how to a an".split())


def _sem_stop(t: str) -> List[str]:
    return [w for w in re.findall(r"[\w-]+", _norm(t)) if w not in _STOP]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "")
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()


# Nome do termo sem stopwords -> nome no catalogo ("criterio wold" -> "criterio de wold")
_TERMOS_SEM_STOP = {" ".join(_sem_stop(k)): k for k in TERMOS}

# Rotulos fixos das respostas (PT, EN)
_R = {
    "rota": ("Rota recomendada:", "Recommended route:"), "config": ("Configuracao", "Configuration"),
    "ver": ("Para ver funcionando", "To see it working"), "tec": ("Tecnicas", "Techniques"),
    "selo": ("veja o selo: guaraci confianca METODO", "see the seal: guaraci confianca METHOD"),
    "porque": ("Por que", "Why"), "faz": ("Faz", "Does"), "use": ("Use quando", "Use when"),
    "nao": ("Nao use quando", "Do not use when"), "entra": ("Entra", "Input"), "sai": ("Sai", "Output"),
    "prox": ("Proximo passo", "Next step"), "ex": ("Exemplo", "Example"), "efeito": ("Efeito", "Effect"),
    "padrao": ("Padrao", "Default"), "faixa": ("Faixa", "Range"), "mostra": ("Mostra", "Shows"),
    "util": ("Util quando", "Useful when"), "req": ("Requisito", "Requires"),
    "precisa": ("Voce vai precisar de", "You will need"), "cmd": ("Tambem e um comando", "Also a command"), "tecla": ("tecla", "key"),
    "fazer": ("O que fazer", "What to do"),
    "sig": ("Isto significa", "This means"), "pq": ("Por que acontece", "Why it happens"),
    "agora": ("O que fazer agora", "What to do now"), "onde": ("Onde ver na CLI", "Where to look in the CLI"),
    "fora": ("Nao sei responder isso com seguranca. Eu cubro estas duvidas:", "I cannot answer that reliably. I cover these questions:"),
    "fora2": ("alem de termos, opcoes e avisos do programa. Veja a ajuda da tela com `?` ou o manual (`guaraci manual 5.11` e `guaraci manual 9`).",
              "plus terms, options and program warnings. See the screen help with `?` or the manual (`guaraci manual 5.11` and `guaraci manual 9`)."),
    "nada": ("Nao encontrei nada no catalogo para essa pergunta. Tente outra palavra, ou `guaraci ajuda`.",
             "Nothing in the catalogue matches that question. Try another word, or `guaraci ajuda`."),
    "result": ("Resultados:", "Results:"), "graf": ("Grafico", "Plot"), "det": ("Para detalhes: guaraci ajuda ID", "For details: guaraci ajuda ID"),
}


def _r(chave: str, lang: str) -> str:
    return _R[chave][_L(lang)]


# ------------------------------------------------------------------ rota
def rotas() -> List[Tuple[str, str]]:
    return list(ROTAS)


def rota(objetivo: str, dado: str) -> Optional[Dict[str, Any]]:
    """Rota recomendada para (objetivo, tipo de dado) ou None se nao ha rota."""
    r = ROTAS.get((objetivo, dado))
    if r is None:
        return None
    return dict(objetivo=objetivo, dado=dado, **r)


def formatar_rota(r: Dict[str, Any], lang: str = "PT") -> str:
    L = [_r("rota", lang)]
    L += [f"  {i}. {cat.passo_texto(p, lang)}" for i, p in enumerate(r["passos"], 1)]
    if r["config"]:
        L.append(f"{_r('config', lang)}: " + ", ".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}"
                                                       for k, v in r["config"].items()))
    if r["comando"]:
        L.append(f"{_r('ver', lang)}: {r['comando']}")
    if r["tecnicas"]:
        L.append(f"{_r('tec', lang)}: " + ", ".join(r["tecnicas"]) + f"  ({_r('selo', lang)})")
    L.append(f"{_r('porque', lang)}: {cat.tx(r['motivo'], lang)}")
    return "\n".join(L)


# --------------------------------------------------------------- explicar
def _gabarito_tecnica(t: Dict[str, Any], lang: str) -> str:
    return (f"{t['nome']}\n{_r('faz', lang)}: {t['faz']}\n{_r('use', lang)} {t['use']}\n{_r('nao', lang)} {t['nao_use']}\n"
            f"{_r('entra', lang)}: {t['entradas']}\n{_r('sai', lang)}: {t['saidas']}\n{_r('prox', lang)}: {t['proximo']}\n"
            f"{_r('ex', lang)}: {t['exemplo']}")


def explicar(termo: str, lang: str = "PT") -> Optional[Dict[str, Any]]:
    """Resposta no gabarito para `termo` (id ou nome exato, sem acento/caixa) ou None.
    Se o termo tambem e' um comando (ex.: `vip`), a resposta cita o comando."""
    r = _explicar_item(termo, lang)
    if r is None or r["tipo"] == "comando":
        return r
    n = _norm(termo)
    for c in cat.comandos():
        if n in (_norm(c["id"]), _norm(c["uso"].split()[0])):
            r["texto"] += f"\n{_r('cmd', lang)}: guaraci {c['uso']}"
            break
    return r


def _explicar_item(termo: str, lang: str) -> Optional[Dict[str, Any]]:
    n = _norm(termo)
    if not n:
        return None
    L = _L(lang)
    for k, o in cat.OPCOES.items():
        if n in (_norm(k), _norm(o["nome"][0]), _norm(o["nome"][1])):
            texto = (f"{o['nome'][L]}\n{_r('faz', lang)}: {o['faz'][L]}\n{_r('use', lang)} {o['use'][L]}\n"
                     f"{_r('nao', lang)} {o['nao_use'][L]}\n{_r('efeito', lang)}: {o['impacto'][L]}\n"
                     f"{_r('padrao', lang)}: {o['default']!r}   {_r('faixa', lang)}: {o['faixa']}")
            return dict(tipo="opcao", id=k, titulo=o["nome"][L], texto=texto)
    for t in cat.tecnicas(lang):
        if n in (_norm(t["id"]), _norm(t["nome"])):
            return dict(tipo="tecnica", id=t["id"], titulo=t["nome"], texto=_gabarito_tecnica(t, lang))
    # termo antes de grafico: "o que e VIP" pede a definicao, nao a figura (a figura vai citada)
    if n not in TERMOS:
        n = _TERMOS_SEM_STOP.get(" ".join(_sem_stop(n)), n)
    if n in TERMOS:
        d, e = TERMOS[n]
        texto = f"{n}: {d[L]}\n{_r('ex', lang)}: {e[L]}"
        for g in cat.graficos(lang):
            if n in (_norm(g["id"]), _norm(g["rotulo"])):
                texto += f"\n{_r('graf', lang)}: {g['rotulo']} (guaraci ajuda {g['id']})"
                break
        return dict(tipo="termo", id=n, titulo=n, texto=texto)
    cod = _erros.eh_codigo(termo)
    if cod and n == _norm(cod):
        return dict(tipo="erro", id=cod, titulo=cod, texto=_erros.explicar(cod, lang))
    for g in cat.graficos(lang):
        if n in (_norm(g["id"]), _norm(g["rotulo"])):
            return dict(tipo="grafico", id=g["id"], titulo=g["rotulo"],
                        texto=f"{g['rotulo']}\n{_r('mostra', lang)}: {g['mostra']}\n{_r('util', lang)} {g['util']}\n"
                              f"{_r('req', lang)}: {g['req']}")
    for tid, t in cat.TELAS.items():
        if n in (_norm(tid), _norm(t["nome"][0]), _norm(t["nome"][1])):
            extra = f"\n{_r('precisa', lang)}: {', '.join(p[L] for p in t['precisa'])}" if t["precisa"] else ""
            return dict(tipo="tela", id=tid, titulo=t["nome"][L],
                        texto=f"{t['nome'][L]} ({_r('tecla', lang)} {t['tecla']})\n{_r('faz', lang)}: {t['faz'][L]}{extra}\n"
                              f"{cat.trilha(tid, lang)}")
    for c in cat.comandos():
        if n in (_norm(c["id"]), _norm(c["uso"].split()[0])):
            return dict(tipo="comando", id=c["id"], titulo=f"guaraci {c['uso']}",
                        texto=f"guaraci {c['uso']}\n{_r('faz', lang)}: {c['faz'][L]}")
    return None


def formatar_explicacao(e: Dict[str, Any]) -> str:
    return e["texto"]


# ------------------------------------------------------------------ buscar
def _indice(lang: str) -> List[Tuple[str, str, str, str, str]]:
    """(tipo, id, titulo, texto de busca, trecho) de todo o catalogo. O texto de busca
    junta os dois idiomas (acha 'batch' ou 'lote'); o trecho sai no idioma pedido."""
    L = _L(lang)
    out = []
    for k, o in cat.OPCOES.items():
        out.append(("opcao", k, o["nome"][L], f"{k} {o['nome'][0]} {o['nome'][1]} {o['faz'][0]} {o['faz'][1]} "
                                             f"{o['use'][0]} {o['use'][1]}", o["faz"][L]))
    for t, te in zip(cat.tecnicas("PT"), cat.tecnicas("EN")):
        out.append(("tecnica", t["id"], t["nome"], f"{t['id']} {t['nome']} {t['faz']} {te['faz']} {t['use']} {te['use']}",
                    (te if L else t)["faz"]))
    for g, ge in zip(cat.graficos("PT"), cat.graficos("EN")):
        out.append(("grafico", g["id"], (ge if L else g)["rotulo"], f"{g['id']} {g['rotulo']} {g['mostra']} {ge['mostra']} {g['util']}",
                    (ge if L else g)["mostra"]))
    for tid, t in cat.TELAS.items():
        out.append(("tela", tid, t["nome"][L], f"{tid} {t['nome'][0]} {t['nome'][1]} {t['faz'][0]} {t['faz'][1]}", t["faz"][L]))
    for c in cat.comandos():
        out.append(("comando", c["id"], f"guaraci {c['uso']}", f"{c['id']} {c['uso']} {c['faz'][0]} {c['faz'][1]}", c["faz"][L]))
    for k, (defn, exemplo) in TERMOS.items():
        out.append(("termo", k, k, f"{k} {defn[0]} {defn[1]} {exemplo[0]}", defn[L]))
    for k, d in DIAGNOSTICOS.items():
        out.append(("diagnostico", k, k, f"{k} {d['significa'][0]} {d['significa'][1]} {d['fazer'][0]}", d["significa"][L]))
    for k, d in DUVIDAS.items():
        out.append(("duvida", k, d["nome"][L], f"{k} {d['nome'][0]} {d['nome'][1]} {d['significa'][0]} {d['significa'][1]}", d["significa"][L]))
    return out


def buscar(consulta: str, lang: str = "PT", limite: int = 8) -> List[Dict[str, Any]]:
    """Busca por palavra-chave em todo o catalogo (sem rede). Ordena por relevancia."""
    toks = [t for w in _sem_stop(consulta) for t in re.split(r"[^a-z0-9]+", w) if len(t) > 1]
    if not toks:
        return []
    achados = []
    q = _norm(consulta)
    for tipo, ident, titulo, texto, trecho in _indice(lang):
        nt, ni, ntexto = _norm(titulo), _norm(ident), _norm(texto)
        palavras = set(re.split(r"[^a-z0-9]+", ntexto))
        # o nome ou o id inteiro digitado vence qualquer soma de palavras soltas
        score = 50.0 if q in (nt, ni, ni.replace("_", " ")) else 0.0
        for t in toks:
            if t == ni or t == nt:
                score += 10
            elif t in ni or t in nt:
                score += 5
            elif t in palavras:
                score += 2
            elif t in ntexto:
                score += 1
            else:
                near = difflib.get_close_matches(t, list(palavras), n=1, cutoff=0.8)
                if near:
                    score += 1.2
        if score > 0:
            achados.append((score, tipo, ident, titulo, trecho))
    achados.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [dict(score=s, tipo=t, id=i, titulo=ti, trecho=tr) for s, t, i, ti, tr in achados[:limite]]


# ----------------------------------------------------------------- sugerir
@dataclass
class Sugestao:
    texto: str
    campo: Optional[str] = None       # chave de configuracao a mudar (None = acao do usuario)
    valor: Any = None                 # valor proposto (padrao visivel); so' aplica se o usuario aceitar


_S = {
    "corrigir": ("Corrigir antes de rodar", "Fix before running"),
    "nada": ("Nada a sugerir para a configuracao atual.", "Nothing to suggest for the current configuration."),
}


def sugerir(cfg: Any, lang: str = "PT", dados: Any = None) -> List[Sugestao]:
    """Passos propostos para a configuracao atual. NUNCA aplica nada: quem aplica e'
    o usuario, uma sugestao por vez (`campo`/`valor`)."""
    from guaraci import contexto
    L = _L(lang)
    copia = copy.deepcopy(cfg)
    r = contexto.aplicar(copia, auto=False, lang=lang)
    out: List[Sugestao] = []
    for e in r.erros:
        out.append(Sugestao(f"{_S['corrigir'][L]}: {e}"))
    for s in r.sugestoes:
        out.append(Sugestao(s))
    # regras do registro unico (as mesmas do pre-voo e da revisao): config, e dados se houver resumo
    momentos = ("config", "dados") if dados is not None else ("config",)
    for a in _regras.avaliar(cfg, dados, None, lang, momentos=momentos):
        txt = a.mensagem if a.severidade != "critico" else f"{_S['corrigir'][L]}: {a.mensagem}"
        out.append(Sugestao(txt, a.campo, a.valor))
    if not out:
        out.append(Sugestao(_S["nada"][L]))
    return out


# ------------------------------------------------------------- diagnosticar
def diagnosticar(texto: str, lang: str = "PT") -> List[Dict[str, Any]]:
    """Mapeia aviso/erro (ou o id do codigo) para o que significa e o que fazer."""
    n = _norm(texto)
    L = _L(lang)
    out = []
    for k, d in DIAGNOSTICOS.items():
        if n == k or any(_norm(p) in n for p in d["padroes"]):
            out.append(dict(id=k, significa=d["significa"][L], fazer=d["fazer"][L]))
    vistos = {d["id"] for d in out}
    for c in _erros.codigos(texto):
        if c not in vistos:
            ex = _erros.CODIGOS[c]
            out.append(dict(id=c, tipo="erro", significa=ex["causa"][L], fazer=ex["fazer"][L],
                            texto=_erros.explicar(c, lang)))
    return out


# ----------------------------------------------------------------- duvidas
_FORTE = 6.0  # abaixo disto uma busca por palavra solta nao e' resposta: vale o "nao sei"
_PERGUNTA_DE_DEFINICAO = re.compile(r"\b(o que (e|significa|quer dizer)|what (is|are|does)|defina|define|meaning of|significado|quando usar|when to use|como usar|how to use)\b")
_MAX_PALAVRAS_SOLTAS = 3
_MAX_CONCEITOS_CURTA = 2   # consulta curta: o BM25 so' atua se a duvida se destacar das vizinhas  # acima disto, uma palavra isolada nao decide a resposta


#: Palavras que negam o termo seguinte ("meu Q2 NAO esta ruim").
_NEGACAO = frozenset(("nao", "not", "nem", "no", "never", "nunca", "isn't", "isnt"))


def _negada(palavras: List[str], i: int) -> bool:
    """A palavra na posicao i tem negacao ate duas posicoes antes."""
    return any(w in _NEGACAO for w in palavras[max(0, i - 2):i])


def _palavra_casa(alt: str, palavras: List[str]) -> bool:
    """`=x` casa so' a palavra inteira; `~x` nao casa se estiver negada ("nao ... ruim");
    alternativa de 4+ letras casa por prefixo."""
    sensivel = alt.startswith("~")
    alt = alt.lstrip("~")
    exato = alt.startswith("=")
    alt = alt.lstrip("=")
    for i, w in enumerate(palavras):
        casa = (w == alt) if (exato or len(alt) < 4) else w.startswith(alt)
        if casa and not (sensivel and _negada(palavras, i)):
            return True
    return False


def _gatilho_casa(gatilho: str, palavras: List[str]) -> int:
    """Numero de palavras do gatilho se TODAS casam (cada uma por uma de suas alternativas), senao 0.
    Uma negacao nunca e' gatilho por si: ela so' cancela alternativas marcadas com `~`."""
    partes = gatilho.split()
    return len(partes) if all(any(_palavra_casa(a, palavras) for a in p.split("|")) for p in partes) else 0


def duvida(pergunta: str, lang: str = "PT") -> Optional[Dict[str, Any]]:
    """A duvida de quimiometria que a pergunta descreve (regra declarativa), ou None."""
    palavras = re.findall(r"[\w-]+", _norm(pergunta))
    melhor: Optional[Tuple[int, str]] = None
    for k, d in DUVIDAS.items():
        n = max((_gatilho_casa(g, palavras) for g in d["gatilhos"]), default=0)
        if n and (melhor is None or n > melhor[0]):
            melhor = (n, k)
    if melhor is None:
        return None
    return duvida_por_id(melhor[1], lang)


def _onde(tipo: str, ident: str, lang: str) -> str:
    if tipo == "tela":
        t = cat.TELAS[ident]
        return f"{_r('tecla', lang)} {t['tecla']}: {cat.trilha(ident, lang)}"
    if tipo == "comando":
        return f"guaraci {cat.COMANDOS[ident]['uso']}"
    if tipo == "grafico":
        return f"guaraci ajuda {ident}"
    return f"guaraci manual {ident}"


def formatar_duvida(d: Dict[str, Any], lang: str = "PT") -> str:
    return "\n".join([d["nome"], f"{_r('sig', lang)}: {d['significa']}", f"{_r('pq', lang)}: {d['por_que']}",
                      f"{_r('agora', lang)}: {d['fazer']}", f"{_r('onde', lang)}: " + "; ".join(d["onde"])])


def duvidas(lang: str = "PT") -> List[str]:
    """Nomes das duvidas que o assistente sabe responder."""
    return [d["nome"][_L(lang)] for d in DUVIDAS.values()]


# ---------------------------------------------------------------- responder
_DOCS: Dict[str, List[Any]] = {}


def _docs_bm25(tipos: Tuple[str, ...]) -> List[Any]:
    """Documentos do catalogo para o BM25, nos dois idiomas. Para duvida entram tambem as
    perguntas-exemplo e as palavras dos gatilhos (sem os marcadores `=`, `~`, `|`)."""
    chave = ",".join(tipos)
    if chave not in _DOCS:
        docs = []
        for tipo, ident, _titulo, texto, _trecho in _indice("PT"):
            if tipo not in tipos:
                continue
            if tipo == "duvida":
                d = DUVIDAS[ident]
                ex = " ".join(" ".join(x) for x in d.get("exemplos", ()))
                gat = " ".join(re.sub(r"[=~|]", " ", g) for g in d["gatilhos"])
                texto = f"{texto} {d['por_que'][0]} {d['por_que'][1]} {d['fazer'][0]} {d['fazer'][1]} {ex} {gat}"
            docs.append(_bm25.Documento(tipo, ident, texto))
        _DOCS[chave] = docs
    return _DOCS[chave]


def _tem_negacao(pergunta: str) -> bool:
    return any(w in _NEGACAO for w in re.findall(r"[\w'-]+", _norm(pergunta)))


def _duvida_por_bm25(pergunta: str, lang: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """(duvida mais provavel, resposta direta?). Direta so' sem negacao na pergunta: o BM25 nao le
    "nao esta baixo" como o contrario de "baixo"."""
    rk = _bm25.ranquear(pergunta, _docs_bm25(("duvida",)))
    if not rk or rk[0].cobertura < _bm25.COBERTURA_MINIMA or rk[0].relativo < _bm25.RELATIVO_MINIMO:
        return None, False
    if len(_bm25.conceitos(pergunta)) <= _MAX_CONCEITOS_CURTA and (
            rk[0].relativo < _bm25.LIMIAR_RELATIVO or (len(rk) > 1 and rk[0].score < _bm25.MARGEM * rk[1].score)):
        return None, False    # consulta de 1 ou 2 palavras: indicio fraco, so' vale se destacar-se das vizinhas
    d = duvida_por_id(rk[0].id, lang)
    return d, _bm25.decidir(rk) == "direta" and not _tem_negacao(pergunta)


#: A regra declarativa e o BM25 discordam: o BM25 so' vence com folga (a regra casou palavras soltas
#: de uma duvida vizinha, como "poucas amostras" numa pergunta sobre acerto de 100%).
_FOLGA_ARBITRAGEM = 1.4


def _arbitrar_regra(id_regra: str, pergunta: str, lang: str) -> Optional[Dict[str, Any]]:
    if _tem_negacao(pergunta):
        return None
    rk = _bm25.ranquear(pergunta, _docs_bm25(("duvida",)))
    if not rk or rk[0].id == id_regra or rk[0].cobertura < _bm25.LIMIAR_COBERTURA or rk[0].relativo < 0.30:
        return None
    da_regra = next((r.score for r in rk if r.id == id_regra), 0.0)
    if rk[0].score >= _FOLGA_ARBITRAGEM * da_regra:
        return duvida_por_id(rk[0].id, lang)
    return None


def duvida_por_id(k: str, lang: str = "PT") -> Dict[str, Any]:
    L = _L(lang)
    d = DUVIDAS[k]
    return dict(id=k, nome=d["nome"][L], significa=d["significa"][L], por_que=d["por_que"][L], fazer=d["fazer"][L],
                onde=[_onde(tipo, ident, lang) for tipo, ident in d["onde"]])


def responder_estruturado(pergunta: str, lang: str = "PT") -> Dict[str, Any]:
    """Resposta com tipo e id (para teste e avaliacao). Ordem: aviso do programa ou codigo de erro,
    duvida de quimiometria (regra), termo ou opcao exata, busca forte. Fora disso: tipo `nao_sei`.
    `busca` traz o id do primeiro resultado; nao e' resposta direta."""
    diag = diagnosticar(pergunta, lang)
    if diag:
        texto = "\n\n".join(d.get("texto") or f"{d['id']}: {d['significa']}\n{_r('fazer', lang)}: {d['fazer']}" for d in diag)
        return dict(tipo=diag[0].get("tipo", "diagnostico"), id=diag[0]["id"], texto=texto)
    dv = duvida(pergunta, lang)
    if dv:
        melhor = _arbitrar_regra(dv["id"], pergunta, lang)
        if melhor is not None:
            dv = melhor
        return dict(tipo="duvida", id=dv["id"], texto=formatar_duvida(dv, lang))
    palavras = _sem_stop(pergunta)
    # uma palavra solta so' vale como resposta em pergunta curta ("o que e VIP"); numa pergunta longa
    # ("SNV antes ou depois da derivada?") ela e' indicio fraco e a decisao fica com a busca
    soltas = palavras if (len(palavras) <= 2 or (len(palavras) <= _MAX_PALAVRAS_SOLTAS and _PERGUNTA_DE_DEFINICAO.search(_norm(pergunta)))) else []
    for cand in (pergunta, " ".join(palavras), *soltas):
        e = explicar(cand, lang) if cand else None
        if e:
            return dict(tipo=e["tipo"], id=e["id"], texto=formatar_explicacao(e))
    dbm, direta = _duvida_por_bm25(pergunta, lang)
    if dbm is not None and direta:
        return dict(tipo="duvida", id=dbm["id"], texto=formatar_duvida(dbm, lang))
    rk_todos = _bm25.ranquear(pergunta, _docs_bm25(("opcao", "tecnica", "grafico", "tela", "comando", "termo", "diagnostico", "duvida")))
    if _bm25.decidir(rk_todos) == "nada" and len(_bm25.conceitos(pergunta)) > _MAX_CONCEITOS_CURTA:
        return dict(tipo="nao_sei", id="", texto=_fora_de_escopo(lang))
    achados = [a for a in buscar(pergunta, lang, 5) if a["score"] >= _FORTE]
    if dbm is not None:   # a duvida mais provavel abre a lista, mesmo sem ser resposta direta
        achados = [dict(score=99.0, tipo="duvida", id=dbm["id"], titulo=dbm["nome"], trecho=dbm["significa"])] + \
                  [a for a in achados if not (a["tipo"] == "duvida" and a["id"] == dbm["id"])][:4]
    if not achados:
        return dict(tipo="nao_sei", id="", texto=_fora_de_escopo(lang))
    return dict(tipo="busca", id=f"{achados[0]['tipo']}:{achados[0]['id']}",
                texto=_r("result", lang) + "\n" + "\n".join(f"  [{a['tipo']}] {a['titulo']} ({a['id']}): {a['trecho']}"
                                                            for a in achados) + "\n" + _r("det", lang))


def responder(pergunta: str, lang: str = "PT") -> str:
    """Resposta a uma pergunta livre (texto). Ver `responder_estruturado`."""
    return responder_estruturado(pergunta, lang)["texto"]


# ------------------------------------------------------------------ checar
def checar(cfg: Any = None, dados: Any = None, resultado: Any = None, lang: str = "PT") -> str:
    """Achados do registro unico de regras para o que houver: configuracao, resumo dos dados
    (`assistente_regras.resumir_dados`) e resultado (dict ou texto do resumo_modelo.txt)."""
    if isinstance(resultado, str):
        resultado = _regras.resultado_de_resumo(resultado)
    return _regras.formatar_achados(_regras.avaliar(cfg, dados, resultado, lang), lang)


def ficha_vazamento(cfg: Any = None, dados: Any = None, resultado: Any = None, lang: str = "PT") -> str:
    """Ficha anti-vazamento da execucao (Markdown). Ver `assistente_vazamento`."""
    from guaraci import assistente_vazamento as _vz
    if isinstance(resultado, str):
        resultado = _regras.resultado_de_resumo(resultado)
    return _vz.ficha(cfg, dados, resultado, lang)


def _fora_de_escopo(lang: str) -> str:
    return f"{_r('fora', lang)} " + "; ".join(duvidas(lang)) + f", {_r('fora2', lang)}"
