"""confianca.py -- Assistente de Confianca (texto puro, offline).

Mostra, para uma tecnica ou opcao, o selo, o que foi verificado, os limites e
como reproduzir. Todo o texto e' GERADO de `guaraci.catalogo` (registro de
evidencias e selos): nao ha prosa de validacao escrita aqui. Opcao sem selo
proprio diz isso. O selo se apoia so' em evidencia que vai junto com o pacote;
sobre os valores padrao vai uma frase generica (decisao D8, variante (b), provisoria).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from guaraci import catalogo as cat
from guaraci.catalogo_validacao import ROTULO_ESTADO, SIMBOLO_ESTADO

__all__ = ["entradas", "pagina", "lista", "buscar_entrada"]

_TIPO_EVID = {"simulacao": ("simulacao", "simulation"), "testes": ("testes automatizados", "automated tests")}


def _L(lang: str) -> int:
    return 1 if str(lang).upper() == "EN" else 0


def _selo_txt(estado: str, lang: str, ascii_only: bool) -> str:
    return f"{SIMBOLO_ESTADO[estado][1 if ascii_only else 0]} {ROTULO_ESTADO[estado][_L(lang)]}"


def entradas(tid: Optional[str] = None, *, opcoes: bool = True, lang: str = "PT") -> List[Tuple[str, str, str, str]]:
    """(tipo, id, nome, estado) com selo proprio. `tid` restringe a opcoes da tela e
    as tecnicas do grupo da tela; None lista tudo. Opcao so' informativa nao entra
    na lista (nao ha o que validar); `opcoes=False` lista so' as tecnicas."""
    out: List[Tuple[str, str, str, str]] = []
    vistos: set = set()   # um id aparece uma vez so' (a tecnica e a opcao de mesmo id contam como uma)
    grupo = cat.tela(tid)["grupo"] if tid else None
    modelagem = ("dados", "modelo", "validacao", "avancado", "tecnicas_avancadas")
    for t in cat.tecnicas(lang):
        if grupo is None or t["grupo"] == grupo or tid in modelagem:
            vistos.add(t["id"])
            out.append(("tecnica", t["id"], t["nome"], cat.SELOS[t["selo"]]["estado"]))
    if opcoes:
        for oid in cat.opcoes_com_selo():
            o = cat.opcao(oid)
            estado = cat.SELOS[cat.OPCAO_SELO[oid]]["estado"]
            if estado != "nao_aplicavel" and (tid is None or o["tela"] == tid) and oid not in vistos:
                vistos.add(oid)
                out.append(("opcao", oid, o["nome"][_L(lang)], estado))
    return out


def buscar_entrada(termo: str) -> Optional[Tuple[str, str]]:
    """Acha uma tecnica ou opcao pelo id ou por parte do nome (sem acento/caixa)."""
    t = termo.strip().lower()
    if not t:
        return None
    for tipo, ident, nome, _ in entradas():
        if t == ident.lower():
            return tipo, ident
    for tipo, ident, nome, _ in entradas():
        if t in ident.lower() or t in nome.lower():
            return tipo, ident
    return None


def lista(tid: Optional[str] = None, lang: str = "PT", ascii_only: bool = False, *, opcoes: bool = True) -> str:
    linhas = []
    for i, (tipo, ident, nome, estado) in enumerate(entradas(tid, opcoes=opcoes, lang=lang), 1):
        linhas.append(f"[{i}] {nome}  {_selo_txt(estado, lang, ascii_only)}  ({ident})")
    return "\n".join(linhas)


def pagina(tipo: str, ident: str, lang: str = "PT", ascii_only: bool = False) -> str:
    c = cat.confianca(tipo, ident, lang)
    L = _L(lang)
    if tipo == "tecnica":
        nome = next(t["nome"] for t in cat.tecnicas(lang) if t["id"] == ident)
        faz = cat.tx(cat.TECNICAS[ident]["faz"], lang)
    else:
        nome = cat.opcao(ident)["nome"][L]
        faz = cat.faz(ident, lang)
    if c["selo"] is None:
        return f"{nome}\n{faz}\n\n" + ("Sem selo proprio: opcao de configuracao, sem calculo proprio a validar."
                                       if L == 0 else "No seal of its own: a configuration option with no calculation to validate.")
    r: List[str] = [f"{nome}   {_selo_txt(c['estado'], lang, ascii_only)}", faz, ""]
    if c["limites"]:
        r.append("Limites:" if L == 0 else "Limits:")
        r.extend(f"  - {x}" for x in c["limites"])
        r.append("")
    r.append("Evidencia:" if L == 0 else "Evidence:")
    for e in c["evidencias"]:
        r.append(f"  - [{_TIPO_EVID[e['tipo']][L]}] {e['rotulo']} ({e['ref']}, {e['data']})")
        r.append(f"    {'Resultado' if L == 0 else 'Result'}: {e['resumo']}")
        if e["ressalva"]:
            r.append(f"    {'Ressalva' if L == 0 else 'Caveat'}: {e['ressalva']}")
        if e["reproduzir"]:
            r.append(f"    {'Reproduzir' if L == 0 else 'Reproduce'}: " + "; ".join(e["reproduzir"]))
        if e["nao_executado"]:
            r.append(f"    {'Nao executado nesta maquina' if L == 0 else 'Not run on this machine'}: {e['nao_executado']}")
        if e["licenca"]:
            r.append(f"    {'Licenca' if L == 0 else 'Licence'}: {e['licenca']}")
    if c.get("calibracao"):
        r.append("")
        r.append(c["calibracao"])
    return "\n".join(r)
