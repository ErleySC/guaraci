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

from pathlib import Path


def trunc(s: str, n: int) -> str:
    """Trunca uma string em n chars sem partir palavra (reticências se cortar)."""
    s = str(s)
    if len(s) <= n:
        return s
    corte = s[:n - 1]
    if " " in corte:
        corte = corte[:corte.rfind(" ")]
    return corte.rstrip() + "…"


def truncar_desc_por_frase(desc: str, max_c: int) -> str:
    """Núcleo de truncamento de descrições curtas: prefere cortar na primeira
    frase (se couber em max_c); senão, corta em borda de palavra (nunca no
    meio) com reticências. Usado por `_desc_curta` (guaraci.py) e por
    `menu_avancado`/`_print_submenu_compact` indiretamente.
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


def validar_faixas(faixa_min: float, faixa_max: float) -> list:
    """Retorna lista de avisos se faixa_min >= faixa_max (intervalo inválido)."""
    avisos = []
    if faixa_min >= faixa_max:
        avisos.append(
            f"ERRO: faixa_min ({faixa_min}) >= faixa_max ({faixa_max}) — intervalo invalido.")
    return avisos


def contar_dx(pasta: str) -> int:
    """Conta arquivos .dx em `pasta` — checa a raiz E subpastas imediatas
    (suporta tanto layout plano quanto uma-subpasta-por-classe)."""
    try:
        p = Path(pasta)
        if not p.is_dir():
            return 0
        n = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".dx")
        if n > 0:
            return n
        for sub in p.iterdir():
            if sub.is_dir():
                n += sum(1 for f in sub.iterdir()
                         if f.is_file() and f.suffix.lower() == ".dx")
        return n
    except Exception:
        return 0


__all__ = ["trunc", "truncar_desc_por_frase", "fmt_bool", "validar_faixas", "contar_dx"]
