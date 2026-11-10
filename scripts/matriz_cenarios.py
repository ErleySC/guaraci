#!/usr/bin/env python3
"""matriz_cenarios.py -- executa cada cenario do catalogo PELO CLI (`python -m guaraci ...`).

    python scripts/matriz_cenarios.py [--saida docs/qualidade/matriz_cenarios.md] [--so ID ...]

Cada passo e' um subprocesso (HOME isolado, rede bloqueada): um comando (`guaraci exemplo`, `guaraci run`, `guaraci anova`,
`guaraci vip`, `guaraci monitorar`) ou um roteiro de teclas enviado as telas (tecnicas especiais e fusao). TODOS os cenarios usam
dado SINTETICO gerado por `guaraci exemplo`; a matriz nunca le dado de terceiros (a validacao do mantenedor fica em `validacao/`).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))


def _env(home: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "USERPROFILE": str(home),
            "HOME": str(home), "HTTP_PROXY": "http://0.0.0.0:9", "HTTPS_PROXY": "http://0.0.0.0:9",
            "MPLBACKEND": "Agg"}


def _proc(args, cwd: Path, home: Path, entrada: str = "", timeout=1800):
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "guaraci", *args], input=entrada, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(cwd), env=_env(home), timeout=timeout)
    return r.returncode, time.perf_counter() - t0, r.stdout + r.stderr


# (id, grupo, dataset, passos, achado). Passo: ("cli", [args]) | ("menu", [teclas]).
# Em ("cli"/"menu"), "{ex}" vira a pasta do exemplo do cenario. "menu" roda com cwd = {ex} (carrega config.yaml).
CENARIOS = [
    ("explorar_sintetico", "explorar", "sintetico", [("cli", ["exemplo", "explorar", "{ex}"]), ("cli", ["run", "{ex}/config.yaml"])], "resumo_modelo.txt"),
    ("classificar_sintetico", "classificar", "sintetico", [("cli", ["exemplo", "classificar", "{ex}"]), ("cli", ["run", "{ex}/config.yaml"])], "resumo_modelo.txt"),
    ("quantificar_sintetico", "quantificar", "sintetico", [("cli", ["exemplo", "quantificar", "{ex}"]), ("cli", ["run", "{ex}/config.yaml"])], "resumo_modelo.txt"),
    ("multielementar_sintetico", "comparar", "sintetico", [("cli", ["exemplo", "multielementar", "{ex}"]), ("cli", ["run", "{ex}/config.yaml"])], "multielementar.json"),
    ("comparar_sintetico", "comparar", "sintetico", [("cli", ["exemplo", "comparar", "{ex}"]),
        ("cli", ["anova", "{ex}/tabela.csv", "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra", "--saida", "{ex}/res", "--sem-bootstrap"])],
     "anova_por_resposta.csv"),
    ("anova_nao_parametrico", "comparar", "sintetico", [("cli", ["exemplo", "comparar", "{ex}"]),
        ("cli", ["anova", "{ex}/tabela.csv", "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra", "--metodo",
                 "kruskal-dunn", "--saida", "{ex}/res", "--sem-bootstrap"])],
     "anova_relatorio.md"),
    ("pls_vip", "modelar", "sintetico", [("cli", ["exemplo", "vip", "{ex}"]),
        ("cli", ["vip", "{ex}/tabela.csv", "--classe", "classe", "--grupo", "amostra", "--saida", "{ex}/res"])],
     "vip_estabilidade.csv"),
    ("fluorescencia_eem", "classificar", "sintetico", [("cli", ["exemplo", "fluorescencia_eem", "{ex}"]), ("cli", ["run", "{ex}/config.yaml"])],
     "resumo_modelo.txt"),
    ("parafac_eem", "explorar", "sintetico", [("cli", ["exemplo", "fluorescencia_eem", "{ex}"]),
                                              ("menu", ["E", "fluorescencia_eem", "{ex}", "s", "", "v", "T", "5", "3", "", "q"])], "parafac_fator_amostra.csv"),
    ("cromatograma_gcms", "classificar", "sintetico", [("cli", ["exemplo", "cromatograma_gcms", "{ex}"]), ("cli", ["run", "{ex}/config.yaml"])],
     "resumo_modelo.txt"),
    ("alinhamento_cromatogramas", "explorar", "sintetico", [("cli", ["exemplo", "cromatograma_gcms", "{ex}"]),
                                                            ("menu", ["E", "cromatograma_gcms", "{ex}", "s", "", "v", "T", "6", "", "", "", "q"])], "cromatogramas_alinhados.csv"),
    ("imagens_hsi", "imagens", "sintetico", [("cli", ["exemplo", "hsi", "{ex}"]), ("cli", ["run", "{ex}/config.yaml"])],
     "hsi_relatorio.txt"),
    ("monitorar_processos", "monitorar", "sintetico", [("cli", ["exemplo", "monitorar", "{ex}", "--rodar"])], "monitor_lotes.csv"),
    ("fusao_multibloco", "modelar", "sintetico", [("cli", ["exemplo", "quantificar", "{ex}"]),
                                                  ("menu", ["T", "4", "NIR", "csv", "{ex}/dados.csv", "classe", "conc",
                                                            "MIR", "csv", "{ex}/dados.csv", "classe", "conc", "", "q"])],
     "stdout:Fusao multibloco"),
]


def _rodar_cenario(ident, passos, achado, base: Path, home: Path):
    ex = base / ident / "ex"
    (base / ident).mkdir(parents=True)
    tempo, rc_total, saidas = 0.0, 0, []
    subs = {"{ex}": str(ex)}

    def sub(a):
        for k, v in subs.items():
            a = a.replace(k, v)
        return a

    for tipo, arg in passos:
        if tipo == "cli":
            rc, t, out = _proc([sub(a) for a in arg], base / ident, home)
            tempo, rc_total = tempo + t, rc_total or rc
            saidas.append(out)
        else:  # menu
            cwd = ex if ex.exists() else base / ident
            rc, t, out = _proc([], cwd, home, entrada="\n".join(sub(a) for a in arg) + "\n")
            tempo, rc_total = tempo + t, rc_total or rc
            saidas.append(out)
    if achado.startswith("stdout:"):
        ok = rc_total == 0 and achado[7:] in "".join(saidas)
    else:
        ok = rc_total == 0 and bool(list((base / ident).rglob(achado)))
    return ("passou" if ok else "FALHOU"), tempo, f"rc={rc_total}; {'achou' if ok else 'NAO achou'} {achado}"


def _via(passos) -> str:
    partes = []
    for t, a in passos:
        if t == "cli":
            partes.append("guaraci " + " ".join(a[:2]))
        else:
            partes.append("teclas " + " ".join(a[:2]))
    return " + ".join(partes)


def executar(so=None) -> list:
    saida = []
    base = Path(tempfile.mkdtemp(prefix="gm"))
    home = base / "home"
    home.mkdir()
    try:
        for ident, grupo, dataset, passos, achado in CENARIOS:
            if so and ident not in so:
                continue
            try:
                res, tempo, nota = _rodar_cenario(ident, passos, achado, base, home)
            except Exception as e:  # noqa: BLE001 -- a matriz registra a falha em vez de parar
                res, tempo, nota = "FALHOU", 0.0, f"erro: {type(e).__name__}: {e}"
            saida.append((ident, grupo, dataset, _via(passos), res, f"{tempo:.1f} s; {nota}"))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return saida


def tabela(linhas: list) -> str:
    t = ["| cenario | grupo | dataset | caminho no CLI | resultado | tempo / nota |", "|---|---|---|---|---|---|"]
    for ident, grupo, dataset, via, res, nota in linhas:
        t.append(f"| {ident} | {grupo} | {dataset} | {via} | {res} | {nota} |")
    return "\n".join(t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="")
    ap.add_argument("--so", nargs="*")
    a = ap.parse_args()
    md = tabela(executar(a.so))
    print(md)
    if a.saida:
        Path(a.saida).write_text(md + "\n", encoding="utf-8")
