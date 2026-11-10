"""cli_logic.py — Lógica PURA da CLI de terminal (guaraci.py), sem dependência
de Rich/console/input.

Extraído de guaraci.py (item 19 da auditoria, mesmo padrão de app_logic.py):
funções aqui não fazem I/O de terminal nem leem estado global de i18n — dados
como idioma são sempre parâmetros explícitos, nunca lidos de uma função tipo
`_lang()`. Isso as torna testáveis em isolamento (ver tests/test_cli_logic.py).
guaraci.py mantém wrappers finos que buscam o estado (idioma ativo, Config)
e chamam estas funções puras.
"""
from __future__ import annotations

import os
import time


def trunc(s: str, n: int) -> str:
    """Trunca uma string em n chars sem partir palavra (reticências se cortar)."""
    s = str(s)
    if len(s) <= n:
        return s
    corte = s[:n - 1]
    if " " in corte:
        corte = corte[:corte.rfind(" ")]
    return corte.rstrip() + "…"


def truncate_desc_by_sentence(desc: str, max_c: int) -> str:
    """Núcleo de truncamento de descrições curtas: prefere cortar na primeira
    frase (se couber em max_c); senão, corta em borda de palavra (nunca no
    meio) com reticências. Usado por `_desc_curta` (guaraci.py) e por
    `menu_advanced`/`_print_submenu_compact` indiretamente.
    """
    desc = (desc or "").strip()
    if not desc:
        return ""
    if "." in desc[:max_c]:
        return desc[:desc.index(".") + 1]
    if len(desc) <= max_c:
        return desc
    corte = desc[:max_c - 1]
    if " " in corte:
        corte = corte[:corte.rfind(" ")]
    return corte.rstrip() + "…"


def fmt_bool(v, lang: str) -> str:
    """Formata um valor booleano como markup Rich Sim/Não (PT) ou Yes/No (EN).

    Valores não-booleanos são devolvidos como string (escape fica a cargo do
    chamador, que tem acesso a `rich.markup.escape`).
    """
    if isinstance(v, bool):
        if lang == "PT":
            return "[g]Sim[/g]" if v else "[m]Nao[/m]"
        return "[g]Yes[/g]" if v else "[m]No[/m]"
    return str(v)


def validate_ranges(faixa_min: float, faixa_max: float) -> list:
    """Retorna lista de avisos se faixa_min >= faixa_max (intervalo inválido)."""
    avisos = []
    if faixa_min >= faixa_max:
        avisos.append(
            f"ERRO: faixa_min ({faixa_min}) >= faixa_max ({faixa_max}) — intervalo invalido.")
    return avisos


_CACHE_DX: dict = {}      # pasta -> (instante, mtime_ns da raiz, contagem)
_VALIDADE_DX = 3.0        # segundos: o menu redesenha varias vezes por tela e cada contagem varre a pasta


def _contar_dx(pasta: str) -> int:
    n = 0
    with os.scandir(pasta) as it:
        subs = []
        for e in it:
            if e.is_file():
                n += e.name.lower().endswith(".dx")
            elif e.is_dir():
                subs.append(e.path)
    if n > 0:
        return n
    for sub in subs:
        try:
            with os.scandir(sub) as it:
                n += sum(1 for e in it if e.is_file() and e.name.lower().endswith(".dx"))
        except OSError:
            continue
    return n


def count_dx(pasta: str) -> int:
    """Conta arquivos .dx em `pasta` — checa a raiz E subpastas imediatas
    (suporta tanto layout plano quanto uma-subpasta-por-classe).

    O resultado vale por `_VALIDADE_DX` segundos enquanto a pasta nao muda: o menu principal
    chama esta funcao tres vezes por redesenho, e em pasta sincronizada com a nuvem ou sob
    antivirus cada varredura custa centenas de milissegundos."""
    try:
        chave = os.path.abspath(str(pasta))
        mtime = os.stat(chave).st_mtime_ns
        agora = time.monotonic()
        visto = _CACHE_DX.get(chave)
        if visto and visto[1] == mtime and agora - visto[0] < _VALIDADE_DX:
            return visto[2]
        n = _contar_dx(chave)
        _CACHE_DX[chave] = (agora, mtime, n)
        return n
    except OSError:   # pasta inacessivel (permissao/race condition) ou inexistente
        return 0


__all__ = ["trunc", "truncate_desc_by_sentence", "fmt_bool", "validate_ranges", "count_dx"]
