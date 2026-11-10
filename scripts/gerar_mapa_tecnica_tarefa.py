#!/usr/bin/env python3
"""gerar_mapa_tecnica_tarefa.py -- mapa tecnica x tarefa x comando x exemplo sintetico x teste, gerado do catalogo.

    python scripts/gerar_mapa_tecnica_tarefa.py            # grava docs/qualidade/mapa_tecnica_tarefa.md e os blocos dos READMEs
    python scripts/gerar_mapa_tecnica_tarefa.py --verificar  # sai 1 se algum arquivo gerado estiver desatualizado

Fonte unica: `guaraci.catalogo` (instrumentos, tecnicas, comandos) e `guaraci.exemplos` (cenarios sinteticos). Nada e escrito a mao:
a lista de tecnicas e de tarefas dos READMEs e o mapa de `docs/qualidade/` saem daqui.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from guaraci import catalogo as cat  # noqa: E402
from guaraci import exemplos as ex  # noqa: E402

SAIDA = RAIZ / "docs" / "qualidade" / "mapa_tecnica_tarefa.md"
INICIO, FIM = "<!-- guaraci:tecnicas:inicio -->", "<!-- guaraci:tecnicas:fim -->"
LEITURAS = {"README.md": 1, "README.pt-BR.md": 0}          # arquivo -> indice do idioma em tuplas (pt, en)

#: Cenario sintetico -> teste que o executa. O teste e' parametrizado sobre `exemplos.listar()`, entao todo cenario esta coberto.
TESTE_DOS_EXEMPLOS = "tests/test_exemplos_e_comandos.py"


def _instrumentos(lang: int) -> List[str]:
    return [f"**{cat.INSTRUMENTOS[k]['nome'][lang].split(' (')[0]}**" for k in cat.INSTRUMENTOS if k != "generico"] + \
           [("outras (perfil `generico`)" if lang == 0 else "others (`generico` profile)")]


def bloco_readme(lang: int) -> str:
    """Lista de instrumentos e de tarefas, no idioma `lang` (0 = PT, 1 = EN). Gerada, nunca editada a mao."""
    linhas = [INICIO]
    linhas.append(("**Técnicas de medida:** " if lang == 0 else "**Measurement techniques:** ") + ", ".join(_instrumentos(lang)) + ".")
    linhas.append("")
    linhas.append(("**Tarefas:**" if lang == 0 else "**Tasks:**"))
    for _id, pt, en in cat.OBJETIVOS:
        linhas.append(f"- {pt if lang == 0 else en}")
    linhas.append("")
    linhas.append(("**Cenários sintéticos prontos** (`guaraci exemplo <cenário>`): " if lang == 0 else "**Ready synthetic scenarios** (`guaraci exemplo <scenario>`): ")
                  + ", ".join(f"`{k}`" for k in ex.listar()) + ".")
    linhas.append(FIM)
    return "\n".join(linhas)


def mapa_md() -> str:
    L: List[str] = []
    L.append("# Mapa técnica × tarefa × comando × exemplo × teste\n")
    L.append("**Público:** quem avalia o que o programa cobre · **Estado:** gerado por `scripts/gerar_mapa_tecnica_tarefa.py` a partir do catálogo "
             "(não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`\n")
    L.append("## 1. Cenários sintéticos por tarefa\n")
    L.append("Cada cenário gera dados sintéticos por código próprio e a configuração pronta; nenhum dado de terceiros.\n")
    L.append("| tarefa | cenário | o que faz | como rodar | teste |")
    L.append("|---|---|---|---|---|")
    for k in ex.listar():
        e = ex.EXEMPLOS[k]
        L.append(f"| {e['grupo']} | `{k}` | {e['faz'][0]} | `guaraci exemplo {k}` e `{e['comando']}` | `{TESTE_DOS_EXEMPLOS}` |")
    L.append("\n## 2. Técnicas de análise do catálogo\n")
    L.append("| técnica | tarefa | selo | evidência (tipos) | exemplo de uso |")
    L.append("|---|---|---|---|---|")
    for k, t in cat.TECNICAS.items():
        selo = cat.SELOS.get(t["selo"], {}) if isinstance(t.get("selo"), str) else {}
        evid = selo.get("evid", ())
        tipos: Dict[str, int] = {}
        for ev in evid:
            tipos[cat.EVIDENCIAS[ev]["tipo"]] = tipos.get(cat.EVIDENCIAS[ev]["tipo"], 0) + 1
        resumo = ", ".join(f"{n} {tp}" for tp, n in sorted(tipos.items())) or "-"
        L.append(f"| `{k}` | {t['grupo']} | {selo.get('estado', '-')} | {resumo} | {t.get('exemplo', '')} |")
    L.append("\n## 3. Técnicas de medida\n")
    L.append("| técnica | faixa típica | pré-processamento recomendado | modo de entrada |")
    L.append("|---|---|---|---|")
    for k, i in cat.INSTRUMENTOS.items():
        L.append(f"| `{k}` | {i['faixa_txt'][0]} | {i['preproc_rec'][0]} | {i.get('mode', '-')} |")
    L.append("\n## 4. Comandos\n")
    L.append("| comando | para que serve |")
    L.append("|---|---|")
    for k, c in cat.COMANDOS.items():
        L.append(f"| `guaraci {c['uso']}` | {c['faz'][0]} |")
    return "\n".join(L) + "\n"


def _trocar_bloco(texto: str, novo: str) -> str:
    rx = re.compile(re.escape(INICIO) + r".*?" + re.escape(FIM), re.S)
    return rx.sub(lambda _m: novo, texto) if rx.search(texto) else texto


def _ler(p: Path) -> str:
    return p.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _gravar(p: Path, s: str) -> None:
    """Grava preservando o fim de linha que o arquivo ja tinha (LF para arquivo novo)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    crlf = p.exists() and b"\r\n" in p.read_bytes()
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    esperado: Dict[Path, str] = {SAIDA: mapa_md()}
    for nome, lang in LEITURAS.items():
        p = RAIZ / nome
        if p.exists():
            esperado[p] = _trocar_bloco(_ler(p), bloco_readme(lang))
    defasados = [p for p, s in esperado.items() if not p.exists() or _ler(p) != s]
    if a.verificar:
        for p in defasados:
            print(f"desatualizado: {p.relative_to(RAIZ).as_posix()}")
        return 1 if defasados else 0
    for p in defasados:
        _gravar(p, esperado[p])
        print(f"gravado: {p.relative_to(RAIZ).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
