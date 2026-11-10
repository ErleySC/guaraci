#!/usr/bin/env python3
"""checar_design.py -- a conferencia de uma rodada de design (meta: ate 10 minutos no Windows). NAO e' a verificacao final.

    python scripts/checar_design.py            # confere tudo
    python scripts/checar_design.py --rapido   # sem os testes de interface e sem os valores de referencia
    python scripts/checar_design.py --base     # regrava o instantaneo de comparacao (so' depois de aprovar a mudanca visual)
    python scripts/checar_design.py --base --com-wheel   # idem, e mede o tamanho do wheel (constroi o wheel: mais lento)

Etapas (BLOQUEIA = faz o comando sair com codigo != 0):
  ruff              estilo e erros estaticos                                   BLOQUEIA
  nucleo            o manifesto do nucleo (docs/qualidade/nucleo_congelado.json) bate com os arquivos   BLOQUEIA (codigo 3)
  valores           numeros de referencia e contrato da API                   BLOQUEIA
  varreduras        e-mail, caminho de usuario, termos sensiveis, termos de matriz   BLOQUEIA
  testes_interface  menus, telas, paleta, vocabulario, CLI e ajuda            BLOQUEIA
  instantaneo       --help, --version, telas e figuras do demo contra ~/.guaraci_local/design_base/   so' LISTA as diferencas

O instantaneo fica FORA do repositorio (`~/.guaraci_local/design_base/`). Detalhes: docs/projeto/FASE_DESIGN.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

RAIZ = Path(__file__).resolve().parents[1]
BASE = Path.home() / ".guaraci_local" / "design_base"
PYTHON = sys.executable

#: arquivos de teste (nomes ou padroes de `tests/`) que cobrem menus, telas, paleta, vocabulario, CLI e ajuda
TESTES_INTERFACE = (
    "test_app_nav.py", "test_molde_telas.py", "test_vocabulario_interface.py", "test_paleta_cores.py", "test_seletor_paleta_web.py",
    "test_interfaces_configuraveis.py", "test_menu_hsi.py", "test_menu_objetivo.py", "test_lacunas_guaraci_ajuda_checklist.py",
    "test_lacunas_guaraci_menus_a.py", "test_guaraci_cli.py", "test_cli_logic.py", "test_cli_opcoes_desconhecidas.py",
    "test_cli_status_e_despedida.py", "test_entradas_ruins_cli.py", "test_catalogo.py", "test_lacunas_catalogo.py",
    "test_assistente_cli.py", "test_tempo_ate_o_menu.py", "test_paridade_predicao_cli_web.py",
)
TESTES_VALORES = ("test_golden_valores.py", "test_contrato_api_publica.py", "test_contrato_saida_tabular.py")

#: so' os subcomandos cuja `--help` e' so' texto (`run`, `doctor` e `web` executam trabalho: o `--help` geral os descreve)
SUBCOMANDOS_AJUDA = ("anova", "vip", "monitorar", "transferir", "relatorio", "exemplo", "confianca", "assistente", "ajuda", "manual",
                     "buscar")


# ---------------------------------------------------------------------------
# Funcoes puras (testadas em tests/test_checar_design.py)
# ---------------------------------------------------------------------------
def normalizar(texto: str, raizes: Sequence[str] = ()) -> str:
    """Tira o que varia de maquina para maquina: caminhos conhecidos, fim de linha, espaco no fim, tempos."""
    t = texto.replace("\r\n", "\n").replace("\r", "\n")
    for r in sorted({x for x in raizes if x}, key=len, reverse=True):
        for forma in (r, r.replace("\\", "/"), r.replace("/", "\\")):
            t = t.replace(forma, "<PASTA>")
    t = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", "<USUARIO>", t)
    t = re.sub(r"/(?:home|Users)/[^/\s]+", "<USUARIO>", t)
    t = re.sub(r"\b\d+(?:[.,]\d+)?\s?(?:s|ms|seg)\b", "<TEMPO>", t)
    t = re.sub(r"\d{8}_\d{6}", "<CARIMBO>", t)
    t = re.sub(r"\d+(?:[.,]\d+)?\s?GB\b", "<GB>", t)            # memoria livre/total mudam de uma execucao para outra
    t = re.sub(r"\bCPU:\s*\S+", "CPU: <N>", t)
    return "\n".join(l.rstrip() for l in t.split("\n")).strip() + "\n"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def comparar_textos(base: Dict[str, str], atual: Dict[str, str]) -> List[str]:
    """Linhas legiveis com o que mudou entre dois conjuntos de textos (nunca imprime o texto inteiro)."""
    saida = []
    for nome in sorted(set(base) | set(atual)):
        if nome not in atual:
            saida.append(f"texto removido: {nome}")
        elif nome not in base:
            saida.append(f"texto novo: {nome}")
        elif base[nome] != atual[nome]:
            a, b = base[nome].split("\n"), atual[nome].split("\n")
            novas = sum(1 for l in b if l not in set(a))
            sumiram = sum(1 for l in a if l not in set(b))
            saida.append(f"texto mudou: {nome} (+{novas} linha(s), -{sumiram} linha(s))")
    return saida


def comparar_figuras(base: Dict[str, Dict], atual: Dict[str, Dict]) -> List[str]:
    saida = []
    for nome in sorted(set(base) | set(atual)):
        if nome not in atual:
            saida.append(f"figura removida: {nome}")
        elif nome not in base:
            saida.append(f"figura nova: {nome}")
        elif base[nome].get("sha") != atual[nome].get("sha"):
            b, a = base[nome], atual[nome]
            dim = f", tamanho {b.get('dim')} -> {a.get('dim')}" if b.get("dim") != a.get("dim") else ""
            saida.append(f"figura mudou: {nome}{dim}")
    return saida


def comparar_tempos(base: Dict[str, float], atual: Dict[str, float], fator: float = 1.5) -> List[str]:
    saida = []
    for nome in sorted(set(base) & set(atual)):
        b, a = base[nome], atual[nome]
        if b > 0 and a > b * fator and a - b > 0.3:
            saida.append(f"mais lento: {nome} {b:.2f}s -> {a:.2f}s")
    return saida


def figuras_png(pasta: Path) -> Dict[str, Dict]:
    """Nome relativo -> {sha, bytes, dim} de cada PNG da pasta."""
    out: Dict[str, Dict] = {}
    for p in sorted(pasta.rglob("*.png")):
        b = p.read_bytes()
        dim = None
        if b[:8] == b"\x89PNG\r\n\x1a\n" and len(b) >= 24:
            dim = [int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")]
        nome = re.sub(r"\d{8}_\d{6}", "<CARIMBO>", str(p.relative_to(pasta)).replace("\\", "/"))
        out[nome] = {"sha": sha(b), "bytes": len(b), "dim": dim}
    return out


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def _env() -> Dict[str, str]:
    e = dict(os.environ)
    e.update({"NO_COLOR": "1", "COLUMNS": "100", "LINES": "40", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
              "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
              "GUARACI_PULADOS_AMBIENTE": e.get("GUARACI_PULADOS_AMBIENTE", "")})
    if not e["GUARACI_PULADOS_AMBIENTE"]:
        e.pop("GUARACI_PULADOS_AMBIENTE")
    e["PYTHONPATH"] = str(RAIZ / "src") + os.pathsep + e.get("PYTHONPATH", "")
    return e


def _rodar(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: int = 900, entrada: Optional[str] = "") -> Tuple[int, str]:
    try:
        r = subprocess.run(list(cmd), cwd=str(cwd or RAIZ), env=_env(), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, input=entrada)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"tempo esgotado ({timeout}s): {' '.join(cmd[:4])}"


def _tests(nomes: Sequence[str]) -> List[str]:
    return [f"tests/{n}" for n in nomes if (RAIZ / "tests" / n).exists()]


def _pytest(nomes: Sequence[str], workers: int) -> Tuple[int, str]:
    alvos = _tests(nomes)
    if not alvos:
        return 2, "nenhum arquivo de teste encontrado"
    cmd = [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "-m", "not slow", *alvos]
    if workers > 1:
        cmd += ["-n", str(workers)]
    return _rodar(cmd, timeout=1500)


def etapa_ruff() -> Tuple[bool, str]:
    rc, out = _rodar([PYTHON, "-m", "ruff", "check", "."], timeout=300)
    return rc == 0, "ruff check . sem achados" if rc == 0 else out.strip()[-600:]


def etapa_nucleo() -> Tuple[bool, str]:
    rc, out = _rodar([PYTHON, "scripts/congelar_nucleo.py", "--verificar"], timeout=120)
    return rc == 0, "manifesto do nucleo bate com os arquivos" if rc == 0 else "NUCLEO MUDOU: " + out.strip()[-500:]


def etapa_valores(workers: int) -> Tuple[bool, str]:
    rc, out = _pytest(TESTES_VALORES, workers)
    return rc == 0, _ultima_linha(out)


def etapa_varreduras() -> Tuple[bool, str]:
    partes, ok = [], True
    for nome, cmd in (("exposicao", [PYTHON, "scripts/auditar_exposicao.py"]), ("termos_de_matriz", [PYTHON, "scripts/auditar_termos_matriz.py"])):
        rc, out = _rodar(cmd, timeout=300)
        partes.append(f"{nome}: {'0 ocorrencias' if rc == 0 else 'ACHOU (rc=' + str(rc) + ') ' + out.strip()[-200:]}")
        ok = ok and rc == 0
    return ok, "; ".join(partes)


def etapa_testes_interface(workers: int) -> Tuple[bool, str]:
    rc, out = _pytest(TESTES_INTERFACE, workers)
    return rc == 0, _ultima_linha(out) if rc == 0 else out.strip()[-800:]


def _ultima_linha(out: str) -> str:
    linhas = [l for l in out.strip().split("\n") if l.strip()]
    return linhas[-1] if linhas else "(sem saida)"


def capturar_instantaneo(com_wheel: bool = False) -> Dict:
    """Textos de `--help`/`--version`/subcomandos, saida do demo, figuras do demo, tempos de abertura (e, opcional, tamanho do wheel)."""
    textos: Dict[str, str] = {}
    tmp = Path(tempfile.mkdtemp(prefix="gd_"))
    raizes = [str(RAIZ), str(tmp), str(Path(PYTHON).parent), str(Path.home())]
    try:
        for nome, args in [("--help", ["--help"]), ("--version", ["--version"])] + [(f"{s} --help", [s, "--help"]) for s in SUBCOMANDOS_AJUDA]:
            rc, out = _rodar([PYTHON, "-m", "guaraci", *args], cwd=tmp, timeout=120)
            textos[nome] = normalizar(out, raizes)
        for nome, args in (("exemplo (lista)", ["exemplo"]), ("manual (lista)", ["manual"]), ("confianca", ["confianca"])):
            rc, out = _rodar([PYTHON, "-m", "guaraci", *args], cwd=tmp, timeout=180)
            textos[nome] = normalizar(out, raizes)
        anova = tmp / "demo_anova"
        rc, out = _rodar([PYTHON, "-m", "guaraci", "exemplo", "comparar", str(anova), "--rodar"], cwd=tmp, timeout=600)
        textos["exemplo comparar (saida)"] = normalizar(out, raizes)
        corrida = tmp / "demo_corrida"
        corrida.mkdir()
        rc, out = _rodar([PYTHON, "-m", "guaraci", "demo"], cwd=corrida, timeout=900)
        textos["demo (saida)"] = normalizar(out, raizes)
        figuras = {**{f"anova/{k}": v for k, v in figuras_png(anova).items()}, **{f"demo/{k}": v for k, v in figuras_png(corrida).items()}}
        tempos = {}
        for nome, cmd in (("abrir --version", [PYTHON, "-m", "guaraci", "--version"]),
                          ("importar guaraci.guaraci", [PYTHON, "-c", "import guaraci.guaraci"])):
            medidas = []
            for _ in range(3):
                t0 = time.perf_counter()
                _rodar(cmd, cwd=tmp, timeout=120)
                medidas.append(time.perf_counter() - t0)
            tempos[nome] = round(statistics.median(medidas), 3)
        inst = {"textos": textos, "figuras": figuras, "tempos": tempos}
        if com_wheel:
            inst["wheel_bytes"] = _tamanho_do_wheel(tmp)
        return inst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _tamanho_do_wheel(tmp: Path) -> Optional[int]:
    saida = tmp / "wheel"
    rc, out = _rodar([PYTHON, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(saida), "-q"], cwd=RAIZ, timeout=900)
    rodas = list(saida.glob("*.whl")) if saida.exists() else []
    return rodas[0].stat().st_size if rodas else None


def gravar_base(inst: Dict, destino: Path = BASE) -> None:
    if destino.exists():
        shutil.rmtree(destino)
    (destino / "textos").mkdir(parents=True)
    for nome, t in inst["textos"].items():
        (destino / "textos" / (re.sub(r"[^A-Za-z0-9_.-]+", "_", nome) + ".txt")).write_text(t, encoding="utf-8")
    meta = {"figuras": inst["figuras"], "tempos": inst["tempos"], "wheel_bytes": inst.get("wheel_bytes"),
            "nomes_dos_textos": {re.sub(r"[^A-Za-z0-9_.-]+", "_", n) + ".txt": n for n in inst["textos"]}}
    (destino / "base.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def ler_base(origem: Path = BASE) -> Optional[Dict]:
    arq = origem / "base.json"
    if not arq.exists():
        return None
    meta = json.loads(arq.read_text(encoding="utf-8"))
    textos = {}
    for arquivo, nome in meta["nomes_dos_textos"].items():
        p = origem / "textos" / arquivo
        if p.exists():
            textos[nome] = p.read_text(encoding="utf-8")
    return {"textos": textos, "figuras": meta["figuras"], "tempos": meta["tempos"], "wheel_bytes": meta.get("wheel_bytes")}


def etapa_instantaneo() -> Tuple[bool, List[str]]:
    base = ler_base()
    if base is None:
        return True, ["sem instantaneo salvo: rode `python scripts/checar_design.py --base` na versao boa antes de comecar o design"]
    atual = capturar_instantaneo()
    dif = comparar_textos(base["textos"], atual["textos"]) + comparar_figuras(base["figuras"], atual["figuras"]) \
        + comparar_tempos(base["tempos"], atual["tempos"])
    return True, dif or ["nenhuma diferenca em relacao ao instantaneo"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Conferencia de uma rodada de design (nao e' a verificacao final).")
    ap.add_argument("--rapido", action="store_true", help="sem testes de interface e sem valores de referencia")
    ap.add_argument("--base", action="store_true", help="regrava o instantaneo de comparacao e sai")
    ap.add_argument("--com-wheel", action="store_true", help="com --base: mede tambem o tamanho do wheel")
    ap.add_argument("--workers", type=int, default=2, help="processos do pytest (padrao 2)")
    a = ap.parse_args(argv)

    if a.base:
        inst = capturar_instantaneo(com_wheel=a.com_wheel)
        gravar_base(inst)
        print(f"instantaneo gravado em {BASE.name}/: {len(inst['textos'])} texto(s), {len(inst['figuras'])} figura(s), tempos {inst['tempos']}"
              + (f", wheel {inst.get('wheel_bytes')} bytes" if a.com_wheel else ""))
        return 0

    etapas: List[Tuple[str, object]] = [("ruff", etapa_ruff), ("nucleo", etapa_nucleo)]
    if not a.rapido:
        etapas.append(("valores", lambda: etapa_valores(a.workers)))
    etapas.append(("varreduras", etapa_varreduras))
    if not a.rapido:
        etapas.append(("testes_interface", lambda: etapa_testes_interface(a.workers)))

    t0 = time.perf_counter()
    falhas, nucleo_mudou = [], False
    for nome, fn in etapas:
        ti = time.perf_counter()
        ok, det = fn()                                     # type: ignore[operator]
        print(f"[{'ok' if ok else 'FALHOU':6}] {nome:<17} {time.perf_counter() - ti:6.1f}s  {det}")
        if not ok:
            falhas.append(nome)
            nucleo_mudou = nucleo_mudou or nome == "nucleo"
    ti = time.perf_counter()
    _, difs = etapa_instantaneo()
    print(f"[info  ] {'instantaneo':<17} {time.perf_counter() - ti:6.1f}s  {len(difs)} linha(s) (nao bloqueia)")
    for d in difs:
        print(f"           - {d}")
    total = time.perf_counter() - t0
    print(f"\ntotal {total / 60:.1f} min; " + ("FALHAS: " + ", ".join(falhas) if falhas else "tudo verde (as diferencas do instantaneo sao so' informacao)"))
    if nucleo_mudou:
        print("O NUCLEO MUDOU: isto nao e' mudanca de design. Desfaca, ou trate como mudanca cientifica (ver docs/projeto/FASE_DESIGN.md).")
        return 3
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
