"""catalogo.py -- catalogo declarativo unico do GUARACI.

Uma fonte alimenta menus, `--help`, ajuda, assistentes, schema do config e
documentacao: `catalogo_opcoes` (opcoes de configuracao), `catalogo_dados`
(grupos e telas), `catalogo_tecnicas` (tecnicas e graficos) e
`catalogo_validacao` (evidencias e selos). Este modulo so' le esses dados e
oferece as visoes derivadas; nao importa numpy nem sklearn, entao carrega em
milissegundos.

Gabarito de texto de toda entrada: Faz (verbo + resultado) | Use quando |
Nao use quando | Entra / Sai | Proximo passo. Descricao nao relata validacao;
a evidencia fica no selo (`selo_de`) e no assistente de confianca. Todo texto
mostrado ao usuario e' um par (PT, EN); `tx(valor, lang)` escolhe o idioma.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from guaraci.catalogo_ajuda import DADOS, DIAGNOSTICOS, OBJETIVOS, ROTAS, TERMOS
from guaraci.catalogo_duvidas import DUVIDAS
from guaraci.catalogo_dados import (ACOES_MENU, COMANDOS, GRUPOS, GRUPOS_CLI, PERFIS_OBJETIVO, PERFIS_RIGOR, SECOES_TELA, TECLAS_ANTIGAS,
                                     TELAS, TEMPO_PERFIL, TEMPO_PERFIL_MEDIDO_EM)
from guaraci.catalogo_opcoes import OPCOES
from guaraci.catalogo_tecnicas import GRAFICOS, INSTRUMENTOS, TECNICAS
from guaraci.catalogo_validacao import (
    CALIBRACAO_PADROES, ESTADOS, EVIDENCIAS, OPCAO_SELO, ROTULO_ESTADO, SELOS, SIMBOLO_ESTADO, TIPOS_EVIDENCIA)

__all__ = [
    "GRUPOS", "TELAS", "COMANDOS", "OPCOES", "INSTRUMENTOS", "PERFIS_OBJETIVO", "PERFIS_RIGOR", "TEMPO_PERFIL",
    "TEMPO_PERFIL_MEDIDO_EM", "ROTAS", "TERMOS", "DIAGNOSTICOS", "OBJETIVOS", "DADOS", "TECNICAS", "GRAFICOS", "EVIDENCIAS", "SELOS",
    "grupos", "grupo", "telas", "comandos", "tela", "tela_por_tecla", "opcao", "faz",
    "campos_da_tela", "campos_avancados", "campos_restantes_web", "ajuda_texto", "ajuda_telas", "guia_tela", "trilha", "help_db", "menu_fields",
    "field_names", "risk_class", "tecnicas", "graficos", "selo_de", "tx", "passo_texto",
    "simbolo_selo", "opcoes_com_selo", "OPCAO_SELO", "confianca", "texto_proibido", "validar_catalogo",
    "RODAPE_TELA", "RODAPE_MENU", "GRUPOS_CLI", "SECOES_TELA", "TECLAS_ANTIGAS",
    "grupos_cli", "telas_do_grupo_cli", "secoes_da_tela",
]

# Rodape padrao de toda tela (molde unico). O menu principal tem o proprio.
# As teclas globais sao as mesmas nos dois; so' a ultima muda: dentro de uma
# tela V volta, no menu principal Q sai (e so' ali se sai do programa).
RODAPE_TELA = {"PT": "? ajuda | G Guaraci | I idioma | C confianca | R rodar | V voltar",
               "EN": "? help | G Guaraci | I language | C trust | R run | V back"}
RODAPE_MENU = {"PT": "? ajuda | G Guaraci | I idioma | C confianca | M modo | R rodar | Q sair",
               "EN": "? help | G Guaraci | I language | C trust | M mode | R run | Q quit"}


def _i(lang: str) -> int:
    return 1 if str(lang).upper() == "EN" else 0


def tx(valor: Any, lang: str = "PT") -> Any:
    """Texto no idioma: par (PT, EN) -> o item do idioma; qualquer outro valor passa direto."""
    if isinstance(valor, (tuple, list)) and len(valor) == 2 and all(isinstance(x, str) for x in valor):
        return valor[_i(lang)]
    return valor


def passo_texto(passo: Any, lang: str = "PT") -> str:
    """Passo de rota: ("cmd", comando, comando) ou (tela, instrucao PT, EN) -> "[tecla] Nome: instrucao"."""
    tid, pt, en = passo
    if tid == "cmd":
        return pt
    t = TELAS[tid]
    instr = en if _i(lang) else pt
    return f"[{t['tecla']}] {t['nome'][_i(lang)]}" + (f": {instr}" if instr else "")


# ------------------------------------------------------------------ grupos
def grupos() -> List[Dict[str, Any]]:
    return [dict(id=g[0], nome=(g[1], g[2]), faz=(g[3], g[4])) for g in GRUPOS]


def grupo(gid: str) -> Dict[str, Any]:
    for g in grupos():
        if g["id"] == gid:
            return g
    raise KeyError(gid)


# ------------------------------------------------------------------- telas
def telas(gid: Optional[str] = None) -> List[Dict[str, Any]]:
    out = [dict(id=k, **v) for k, v in TELAS.items()]
    return [t for t in out if gid is None or t["grupo"] == gid]


def tela(tid: str) -> Dict[str, Any]:
    return dict(id=tid, **TELAS[tid])


def comandos(gid: Optional[str] = None) -> List[Dict[str, Any]]:
    return [dict(id=k, **v) for k, v in COMANDOS.items() if gid is None or v["grupo"] == gid]


def tela_por_tecla() -> Dict[str, str]:
    """tecla (maiuscula) -> id da tela; so' as teclas de tela, nao as acoes."""
    return {v["tecla"].upper(): k for k, v in TELAS.items()}


# ------------------------------------------------------------------ opcoes
def opcao(oid: str) -> Dict[str, Any]:
    return OPCOES[oid]


def faz(oid: str, lang: str = "PT") -> str:
    return OPCOES[oid]["faz"][_i(lang)]


def campos_da_tela(tid: str, *, avancado: bool = True, canal: str = "cli",
                   basicos_primeiro: bool = False) -> List[str]:
    """Chaves de opcao da tela, na ordem do catalogo (a numeracao dos menus
    depende dela). `avancado=False` oculta as de nivel avancado (Modo Basico).
    `canal` filtra onde a opcao aparece: "cli", "web" ou "extra" (sub-handlers).
    `basicos_primeiro=True` poe as basicas antes das avancadas, mantendo a
    ordem do catalogo dentro de cada nivel: assim o numero de uma opcao
    basica e' o mesmo nos modos Basico e Avancado (a CLI numera assim)."""
    out = [k for k, v in OPCOES.items()
           if v["tela"] == tid and canal in v["canais"] and (avancado or v["nivel"] == "basico")]
    if basicos_primeiro:
        out.sort(key=lambda k: OPCOES[k]["nivel"] != "basico")
    return out


def campos_restantes_web(explicitas: Any, validas: Any) -> List[str]:
    """Opcoes do canal web, com chave valida no config, que nenhuma aba lista a mao
    (`explicitas`): o web as mostra num bloco generico de parametros avancados,
    na ordem do catalogo. Assim nenhuma chave nova fica sem widget."""
    ja, ok = set(explicitas), set(validas)
    return [k for k, v in OPCOES.items() if "web" in v["canais"] and k in ok and k not in ja]


def campos_avancados(tid: str, *, canal: str = "cli") -> frozenset:
    return frozenset(k for k, v in OPCOES.items()
                     if v["tela"] == tid and canal in v["canais"] and v["nivel"] == "avancado")


def ajuda_texto(oid: str, lang: str = "PT") -> str:
    """Texto de ajuda completo de uma opcao no gabarito (Faz / Use quando / Nao use quando)."""
    o = OPCOES[oid]
    if _i(lang):
        return f"{o['faz'][1]}  Use when {o['use'][1]}  Do not use when {o['nao_use'][1]}"
    return f"{o['faz'][0]}  Use quando {o['use'][0]}  Nao use quando {o['nao_use'][0]}"


def ajuda_telas() -> Dict[str, Dict[str, str]]:
    """Ajuda das telas de fluxo por tecla ("?" dentro de um prompt)."""
    out: Dict[str, Dict[str, str]] = {}
    for tid, t in TELAS.items():
        if t["tipo"] == "config" and campos_da_tela(tid):   # com campos: "?" pede a ajuda do CAMPO
            continue
        pt = f"{t['nome'][0]}: {t['faz'][0]}"
        en = f"{t['nome'][1]}: {t['faz'][1]}"
        if t["precisa"]:
            pt += " Voce vai precisar de: " + ", ".join(p[0] for p in t["precisa"]) + "."
            en += " You will need: " + ", ".join(p[1] for p in t["precisa"]) + "."
        out[t["tecla"].upper()] = {"PT": pt + " Enter vazio volta ao menu.", "EN": en + " Empty Enter goes back."}
    return out


def guia_tela(tid: str, lang: str = "PT") -> Dict[str, str]:
    """Ajuda curta da tela no gabarito: faz, use (quando usar) e proximo passo."""
    t = TELAS[tid]
    return {c: t[c][_i(lang)] for c in ("faz", "use", "proximo")}


def trilha(tid: str, lang: str = "PT") -> str:
    """Trilha de navegacao da CLI: Menu > [n] Grupo > [n.m] Tela."""
    t = TELAS[tid]
    gnum = t["tecla"].split(".")[0]
    g = next((x for x in GRUPOS_CLI if x[0] == gnum), None)
    if g is None:                       # acao de uma letra (R): sem grupo
        return f"Menu › {t['nome'][_i(lang)]}"
    return f"Menu › {g[0]} {g[1 + _i(lang)]} › {t['tecla']} {t['nome'][_i(lang)]}"


# ---------------------------------------------------------------- menu da CLI
def grupos_cli() -> List[Dict[str, Any]]:
    """Os 6 grupos numerados da tela inicial da CLI."""
    return [dict(num=g[0], nome=(g[1], g[2]), resumo=(g[3], g[4])) for g in GRUPOS_CLI]


def telas_do_grupo_cli(num: str) -> List[Dict[str, Any]]:
    """Telas do grupo `num` na ordem do item (tecla "num.item"); basicas antes
    das avancadas, entao o numero de cada uma e' o mesmo nos dois modos."""
    out = [tela(k) for k, v in TELAS.items() if v["tecla"].split(".")[0] == num and "." in v["tecla"]]
    return sorted(out, key=lambda t: int(t["tecla"].split(".")[1]))


def secoes_da_tela(tid: str) -> List[Dict[str, Any]]:
    """Subpastas das opcoes avancadas da tela (vazio = lista unica)."""
    return [dict(id=s[0], nome=(s[1], s[2]), campos=list(s[3]), resumo=(s[4], s[5])) for s in SECOES_TELA.get(tid, ())]


def help_db() -> Dict[str, Dict[str, Any]]:
    """Visao no formato historico do `HELP_DB` (PT/EN: desc, impacto, exemplos)."""
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in OPCOES.items():
        out[k] = {
            "PT": {"desc": v["faz"][0], "impacto": v["impacto"][0], "exemplos": v["exemplos"][0],
                   "use": v["use"][0], "nao_use": v["nao_use"][0]},
            "EN": {"desc": v["faz"][1], "impacto": v["impacto"][1], "exemplos": v["exemplos"][1],
                   "use": v["use"][1], "nao_use": v["nao_use"][1]},
            "default": v["default"], "range": v["faixa"],
        }
    return out


def menu_fields() -> Dict[str, List[str]]:
    """Visao `MENU_FIELDS` (canal web): tela de configuracao -> campos."""
    out: Dict[str, List[str]] = {}
    for tid, t in TELAS.items():
        if t["tipo"] == "config":
            campos = campos_da_tela(tid, canal="web")
            if campos:
                out[tid] = campos
    return out


def field_names() -> Dict[str, Dict[str, str]]:
    return {k: {"PT": v["nome"][0], "EN": v["nome"][1]} for k, v in OPCOES.items()}


def risk_class() -> Dict[str, str]:
    return {k: v["risco"] for k, v in OPCOES.items() if v["risco"]}


# --------------------------------------------------------------- tecnicas
# Nome das tecnicas em ingles quando difere do registro (que e' a fonte em portugues).
NOME_TECNICA_EN = {
    "conformal_one_class": "Conformal prediction (one-class)",
    "pls_r_pooled": "PLS-R (pooled regression)",
    "pls_r_por_especie": "PLS-R per species",
    "conformal_regressao": "Conformal prediction (regression)",
    "conformal_classificacao": "Conformal prediction set (classification)",
    "identificacao_conjunto_aberto": "Adulterant identification (open set)",
    "figuras_de_merito_regressao": "LOD/LOQ/sensitivity (figures of merit)",
    "linearidade": "Lack-of-fit test (linearity)",
    "robustez": "Robustness protocol",
    "perfil_matriz": "Technique profile",
    "perfil_tecnica_aquisicao": "Acquisition-technique profile (image mode)",
    "anova_tukey_unidade": "ANOVA + Tukey/Games-Howell on the independent unit",
    "kruskal_dunn_unidade": "Kruskal-Wallis + Dunn on the independent unit",
    "diagnostico_premissas": "ANOVA assumption diagnostics and method recommendation",
    "vip_estabilidade": "PLS VIP with stability across grouped folds",
    "perfil_multielementar": "Multi-element profile (ICP-OES / ICP-MS)",
    "parafac_eem": "PARAFAC of EEM fluorescence",
    "cow_alinhamento": "Chromatogram alignment (COW)",
    "monitoramento_mspc": "Batch process monitoring (T2/Q + drift test)",
    "portao_correcao_sinal": "Signal-correction acceptance gate (EMSC/OSC)",
    "amostragem_ativa": "Active-sampling prioritisation",
    "pooled_vs_local": "Pooled model vs. per-species model",
}


def tecnicas(lang: str = "PT") -> List[Dict[str, Any]]:
    """Tecnicas = identidade (technique_registry) + gabarito (catalogo_tecnicas), no idioma pedido."""
    from guaraci.technique_registry import REGISTRY
    textos = ("faz", "use", "nao_use", "entradas", "saidas", "proximo")
    return [dict(id=e.id, categoria=e.categoria, nome=NOME_TECNICA_EN.get(e.id, e.nome) if lang == "EN" else e.nome,
                 referencia=e.referencia,
                 **{k: (tx(v, lang) if k in textos else v) for k, v in TECNICAS[e.id].items()})
            for e in REGISTRY if e.id in TECNICAS]


# ---------------------------------------------------------------- graficos
def graficos(lang: str = "PT") -> List[Dict[str, Any]]:
    """Graficos = catalogo_tecnicas.GRAFICOS + objetivos e rotulo de modos_analise, no idioma pedido."""
    from guaraci import modos_analise as ma
    out = []
    for k, v in GRAFICOS.items():
        vv = {c: (tx(x, lang) if c in ("mostra", "util", "req") else x) for c, x in v.items()}
        # rotulo: o de modos_analise (PT) ou, em EN e nos graficos de comando, o inicio de "mostra"
        resumo = re.split(r"[:,.]", vv["mostra"])[0].strip()
        rotulo = resumo if (_i(lang) or k not in ma._FIG_DESCRICAO) else ma._FIG_DESCRICAO[k]
        out.append(dict(id=k, rotulo=rotulo, objetivos=sorted(ma._FIG_OBJETIVOS.get(k, ())), **vv))
    return out


# ------------------------------------------------------------ selo/confianca
def selo_de(selo_id: str) -> Dict[str, Any]:
    s = SELOS[selo_id]
    return dict(id=selo_id, estado=s["estado"], nota=s["nota"], evid=s["evid"])


def opcoes_com_selo() -> List[str]:
    return [k for k in OPCOES if k in OPCAO_SELO]


def simbolo_selo(estado: str, *, ascii_only: bool = False) -> str:
    return SIMBOLO_ESTADO[estado][1 if ascii_only else 0]


def _selo_da_entrada(tipo: str, ident: str) -> Optional[str]:
    if tipo == "opcao":
        return OPCAO_SELO.get(ident)
    if tipo == "tecnica":
        return TECNICAS[ident]["selo"] if ident in TECNICAS else None
    return None


def confianca(tipo: str, ident: str, lang: str = "PT") -> Dict[str, Any]:
    """Pagina de confianca de uma opcao ou tecnica, gerada so' do registro:
    selo, evidencias, limites (nao use quando), e como reproduzir (testes)."""
    sid = _selo_da_entrada(tipo, ident)
    if sid is None:
        return dict(tipo=tipo, id=ident, selo=None, estado=None,
                    texto="sem selo proprio: opcao de configuracao sem calculo proprio")
    sel = SELOS[sid]
    evid = []
    for eid in sel["evid"]:
        e = EVIDENCIAS[eid]
        evid.append(dict(id=eid, tipo=e["tipo"], rotulo=tx(e["rotulo"], lang), ref=e["ref"], data=e["data"],
                         resumo=tx(e["resumo"], lang), ressalva=tx(e["ressalva"], lang),
                         reproduzir=[f"pytest tests/{t}" for t in e["testes"]], nao_executado="", licenca=""))
    limites = []
    if tipo == "tecnica":
        limites.append(tx(TECNICAS[ident]["nao_use"], lang))
    elif tipo == "opcao":
        limites.append(tx(OPCOES[ident]["nao_use"], lang))
    if tx(sel["nota"], lang):
        limites.append(tx(sel["nota"], lang))
    return dict(tipo=tipo, id=ident, selo=sid, estado=sel["estado"],
                rotulo_estado=ROTULO_ESTADO[sel["estado"]][_i(lang)], evidencias=evid,
                limites=limites, calibracao=tx(CALIBRACAO_PADROES, lang))


# ------------------------------------------------------------ texto proibido
# Padroes de relato de validacao proibidos em descricoes (faz, use, nao_use,
# entradas, saidas, proximo, impacto, mostra, util).
_PROIBIDO = [
    r"nos testes", r"obtivemos", r"validad[oa]s? (em|com|contra)", r"medid[oa]s?:", r"medid[oa]s? em ",
    r"\d+[,.]?\d*\s?%\s*(de )?(acuracia|acerto|precisao|exatidao)", r"bal\.?\s?acc", r"\bCV\b", r"\bIC\s?95", r"\bIC\b",
    r"\bdatasets?\b", r"\bdados? p[uú]blicos?\b", r"10\.(17632|6084|25678|5281|5061)/",
    r"VALIDACAO_PUBLICA", r"validacao_dados_publicos", r"achado #", r"Passo \d+", r"Bloco \d+", r"\bP[I]BIC\b", r"\bTCC\b",
]
# o mesmo em ingles (relato de validacao nas descricoes EN)
_PROIBIDO += [r"in (our|the) tests", r"we (obtained|measured|found)", r"validated (on|with|against)",
              r"\d+[.]?\d*\s?%\s*(of )?(accuracy|precision)", r"\bCI\s?95", r"95% CI"]
_PROIBIDO_RE = re.compile("|".join(_PROIBIDO), re.I)
_HEDGES = re.compile(r"\b(pode ajudar|tenta |talvez|quem sabe|may help|tries to|maybe|perhaps)\b", re.I)


def texto_proibido(txt: str) -> Optional[str]:
    """Devolve o trecho proibido (relato de validacao ou hedge) ou None."""
    # nomes de metodo que contem a sigla CV nao sao relato de validacao
    limpo = (txt or "").replace("CV-ANOVA", "").replace("Monte Carlo CV", "Monte Carlo")
    m = _PROIBIDO_RE.search(limpo) or _HEDGES.search(limpo)
    return m.group(0) if m else None


# ---------------------------------------------------------------- validacao
_LIMITES_CAMPO = {"faz": 170, "use": 150, "nao_use": 200, "entradas": 130, "saidas": 170, "proximo": 130,
                  "mostra": 110, "util": 110}


def _checar_par(problemas: List[str], where: str, campo: str, par: Any) -> None:
    """Campo bilingue: tem de ser (PT, EN), os dois preenchidos e sem texto proibido."""
    if not (isinstance(par, (tuple, list)) and len(par) == 2):
        problemas.append(f"{where}.{campo}: nao e um par (PT, EN)")
        return
    for lang, txt in zip(("PT", "EN"), par):
        _checar_texto(problemas, f"{where}[{lang}]", campo, txt)


def _checar_texto(problemas: List[str], where: str, campo: str, txt: str) -> None:
    if not txt or not txt.strip():
        problemas.append(f"{where}: campo '{campo}' vazio")
        return
    lim = _LIMITES_CAMPO.get(campo)
    if lim and len(txt) > lim:
        problemas.append(f"{where}.{campo}: {len(txt)} caracteres (limite {lim})")
    bad = texto_proibido(txt)
    if bad:
        problemas.append(f"{where}.{campo}: texto proibido ({bad!r})")


def validar_catalogo() -> List[str]:
    """Lista de problemas do catalogo (vazia = contrato cumprido)."""
    p: List[str] = []
    gids = {g[0] for g in GRUPOS}
    if len(gids) != len(GRUPOS):
        p.append("ids de grupo repetidos")
    teclas: Dict[str, str] = {}
    for tid, t in TELAS.items():
        if t["grupo"] not in gids:
            p.append(f"tela {tid}: grupo inexistente {t['grupo']!r}")
        k = t["tecla"].upper()
        if k in teclas:
            p.append(f"tecla {k!r} repetida em {tid} e {teclas[k]}")
        teclas[k] = tid
        if k in ACOES_MENU:
            p.append(f"tecla {k!r} de {tid} colide com acao do menu")
        for lang in (0, 1):
            _checar_texto(p, f"tela {tid}", "faz", t["faz"][lang])
        if "curta" not in t:
            p.append(f"tela {tid}: sem descricao curta")
        else:
            for lang in (0, 1):
                if len(t["curta"][lang]) > 44 or "..." in t["curta"][lang]:
                    p.append(f"tela {tid}: descricao curta longa ou cortada")
        for campo in ("use", "proximo"):
            if campo not in t:
                p.append(f"tela {tid}: sem '{campo}' (ajuda da tela)")
            else:
                _checar_par(p, f"tela {tid}", campo, t[campo])
    nums = {g[0] for g in GRUPOS_CLI}
    for tid, t in TELAS.items():
        k = t["tecla"]
        if k != "R" and (k.count(".") != 1 or k.split(".")[0] not in nums or not k.split(".")[1].isdigit()):
            p.append(f"tela {tid}: tecla {k!r} fora do formato grupo.item da CLI")
    for num in nums:
        itens = telas_do_grupo_cli(num)
        niveis = [t["nivel"] for t in itens]
        if niveis != sorted(niveis, key=lambda n: n != "basico"):
            p.append(f"grupo {num}: tela avancada antes de basica (a numeracao mudaria entre os modos)")
    for tid, secs in SECOES_TELA.items():
        for s_ in secs:
            if len(s_[1]) > 22 or len(s_[2]) > 22:
                p.append(f"secao {tid}.{s_[0]}: nome com mais de 22 caracteres (nao cabe alinhado)")
        vistos = [c for s in secs for c in s[3]]
        if len(vistos) != len(set(vistos)):
            p.append(f"secoes de {tid}: opcao repetida")
        if set(vistos) != set(campos_avancados(tid)):
            p.append(f"secoes de {tid}: nao cobrem exatamente as opcoes avancadas "
                     f"(faltam {sorted(set(campos_avancados(tid)) - set(vistos))}, sobram {sorted(set(vistos) - set(campos_avancados(tid)))})")
    for cid, c in COMANDOS.items():
        if c["grupo"] not in gids:
            p.append(f"comando {cid}: grupo inexistente")
        for lang in (0, 1):
            _checar_texto(p, f"comando {cid}", "faz", c["faz"][lang])
    for oid, o in OPCOES.items():
        if o["tela"] not in TELAS:
            p.append(f"opcao {oid}: tela inexistente {o['tela']!r}")
        if o["nivel"] not in ("basico", "avancado"):
            p.append(f"opcao {oid}: nivel invalido")
        for lang in (0, 1):
            _checar_texto(p, f"opcao {oid}", "faz", o["faz"][lang])
        _checar_par(p, f"opcao {oid}", "use", o["use"])
        _checar_par(p, f"opcao {oid}", "nao_use", o["nao_use"])
        for lang in (0, 1):
            bad = texto_proibido(o["impacto"][lang])
            if bad:
                p.append(f"opcao {oid}.impacto: texto proibido ({bad!r})")
    for tid, t in TECNICAS.items():
        for campo in ("faz", "use", "nao_use", "entradas", "saidas", "proximo"):
            _checar_par(p, f"tecnica {tid}", campo, t[campo])
        if t["grupo"] not in gids:
            p.append(f"tecnica {tid}: grupo inexistente")
        if t["selo"] not in SELOS:
            p.append(f"tecnica {tid}: selo inexistente {t['selo']!r}")
    for gid_, g in GRAFICOS.items():
        _checar_par(p, f"grafico {gid_}", "mostra", g["mostra"])
        _checar_par(p, f"grafico {gid_}", "util", g["util"])
        _checar_par(p, f"grafico {gid_}", "req", g["req"])
    for sid, s in SELOS.items():
        if s["estado"] not in ESTADOS:
            p.append(f"selo {sid}: estado invalido")
        if not s["evid"]:
            p.append(f"selo {sid}: sem evidencia")
        for e in s["evid"]:
            if e not in EVIDENCIAS:
                p.append(f"selo {sid}: evidencia inexistente {e!r}")
        if s["estado"] == "validado":
            p.append(f"selo {sid}: 'validado' exige evidencia de dado real que va' junto com o pacote (nenhuma hoje)")
    # ---- assistente: rotas, glossario, diagnosticos
    from guaraci.exemplos import EXEMPLOS
    obj_ids = {o[0] for o in OBJETIVOS}
    dado_ids = {d[0] for d in DADOS}
    for (obj, dado), r in ROTAS.items():
        w = f"rota ({obj}, {dado})"
        if obj not in obj_ids or dado not in dado_ids:
            p.append(f"{w}: objetivo ou dado inexistente")
        for k in r["config"]:
            if k not in OPCOES:
                p.append(f"{w}: opcao inexistente {k!r}")
        for tid in r["tecnicas"]:
            if tid not in TECNICAS:
                p.append(f"{w}: tecnica inexistente {tid!r}")
        if r["exemplo"] and r["exemplo"] not in EXEMPLOS:
            p.append(f"{w}: exemplo inexistente {r['exemplo']!r}")
        if r["comando"]:
            cmd = r["comando"].split()
            if len(cmd) < 2 or cmd[0] != "guaraci" or (cmd[1] not in COMANDOS):
                p.append(f"{w}: comando fora do catalogo {r['comando']!r}")
        for passo in r["passos"]:
            tid_p, pt, en = passo
            if tid_p == "cmd":
                cmd = pt.split()
                if len(cmd) < 2 or cmd[0] != "guaraci" or cmd[1] not in COMANDOS:
                    p.append(f"{w}: comando fora do catalogo {pt!r}")
            elif tid_p not in TELAS:
                p.append(f"{w}: tela inexistente {tid_p!r}")
            elif bool(pt) != bool(en):
                p.append(f"{w}: passo sem os dois idiomas {passo!r}")
            for txt in (pt, en):
                bad = texto_proibido(txt)
                if bad:
                    p.append(f"{w}: texto proibido ({bad!r})")
        _checar_par(p, w, "motivo", r["motivo"])
    for k, (defn, exemplo) in TERMOS.items():
        _checar_par(p, f"termo {k}", "definicao", defn)
        _checar_par(p, f"termo {k}", "exemplo", exemplo)
    for k, d in DIAGNOSTICOS.items():
        for campo in ("significa", "fazer"):
            _checar_par(p, f"diagnostico {k}", campo, d[campo])
    for k, d in DUVIDAS.items():
        _checar_par(p, f"duvida {k}", "nome", d["nome"])
        for campo in ("significa", "por_que", "fazer"):
            _checar_par(p, f"duvida {k}", campo, d[campo])
        if not d["gatilhos"] or not all(isinstance(g, str) and g.split() for g in d["gatilhos"]):
            p.append(f"duvida {k}: gatilhos vazios")
        if not d["onde"]:
            p.append(f"duvida {k}: sem 'onde ver na CLI'")
        for tipo, ident in d["onde"]:
            existe = {"tela": TELAS, "comando": COMANDOS, "grafico": GRAFICOS}.get(tipo)
            if tipo == "manual":
                if not re.fullmatch(r"\d+(\.\d+)?", ident):
                    p.append(f"duvida {k}: secao do manual invalida {ident!r}")
            elif existe is None or ident not in existe:
                p.append(f"duvida {k}: 'onde' inexistente ({tipo}, {ident})")
    for tid, t in TELAS.items():
        for item in t["precisa"]:
            _checar_par(p, f"tela {tid}", "precisa", item)
    for eid, e in EVIDENCIAS.items():
        if e["tipo"] not in TIPOS_EVIDENCIA:
            p.append(f"evidencia {eid}: tipo {e['tipo']!r} nao vai junto com o pacote")
        for campo in ("rotulo", "resumo"):
            if not (isinstance(e[campo], tuple) and all(e[campo])):
                p.append(f"evidencia {eid}.{campo}: nao e um par (PT, EN) preenchido")
        if not (isinstance(e["ressalva"], tuple) and bool(e["ressalva"][0]) == bool(e["ressalva"][1])):
            p.append(f"evidencia {eid}.ressalva: idiomas desiguais")
    from guaraci.config_io import _CONFIG_SPEC, _PRE_PROC_INV
    for iid, ins in INSTRUMENTOS.items():
        for campo in ("nome", "faz", "preproc_rec", "faixa_txt"):
            if not (isinstance(ins[campo], tuple) and all(ins[campo])):
                p.append(f"instrumento {iid}.{campo}: nao e um par (PT, EN) preenchido")
        for txt in ins["faz"]:
            bad = texto_proibido(txt)
            if bad:
                p.append(f"instrumento {iid}.faz: texto proibido ({bad!r})")
        if ins["preproc"] not in _PRE_PROC_INV:
            p.append(f"instrumento {iid}: preproc {ins['preproc']!r} nao e um preset do motor")
        modos = next(s_["opcoes"] for s_ in _CONFIG_SPEC if s_["key"] == "modo_entrada")
        fmts = ins.get("formatos") or ()
        if fmts and fmts[0][0] != ins["mode"]:
            p.append(f"instrumento {iid}: o primeiro formato deve ser o modo padrao {ins['mode']!r}")
        for m, rot in fmts:
            if m not in modos:
                p.append(f"instrumento {iid}: formato {m!r} nao e um modo_entrada do motor")
            if not (isinstance(rot, tuple) and all(rot)):
                p.append(f"instrumento {iid}: rotulo do formato {m!r} nao e um par (PT, EN)")
    for sid, s in SELOS.items():
        if not (isinstance(s["nota"], tuple) and bool(s["nota"][0]) == bool(s["nota"][1])):
            p.append(f"selo {sid}.nota: idiomas desiguais")
    for oid, sid in OPCAO_SELO.items():
        if oid not in OPCOES or sid not in SELOS:
            p.append(f"OPCAO_SELO {oid}->{sid}: referencia inexistente")
    return p
