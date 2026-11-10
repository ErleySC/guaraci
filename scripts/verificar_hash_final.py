#!/usr/bin/env python3
"""verificar_hash_final.py -- UMA verificacao completa de UM hash, com resultado gravado so' se o hash nao mudou.

    python scripts/verificar_hash_final.py [--historico ORIGEM..HEAD] [--pular ETAPA ...] [--so ETAPA ...] [--sem-gravar] [--sem-reaproveitar]

Etapas (cada uma `ok`, `falhou`, `nao_se_aplica` com motivo, `pendente_do_autor` ou `nao_executada`; `nao_executada` CONTA COMO FALHA --
nunca e' aprovacao; `pendente_do_autor` nao e' falha mas aparece separada no relatorio):
  arvore_limpa, suite_completa (pytest com slow, xdist, junit + cobertura de ramos), cobertura_minima (nucleo 100%, global >= 95%),
  cobertura_lacunas (docs/qualidade/cobertura_lacunas.md em dia com a cobertura medida), ruff, mypy,
  contrato_catalogo, links, exposicao (arvore e, com --historico, o intervalo), r5_termos_locais (`pendente_do_autor` com a lista vazia),
  termos_de_matriz, referencias, docs_gerados_em_dia,
  matriz_cenarios (14 cenarios sinteticos pela CLI), pacote_construido (wheel+sdist, venv limpo, ajuda do wheel instalado),
  rodada_local_formatos (leitores contra arquivos reais de fabricante, registro sem arquivos),
  vault (contagens; sem GUARACI_VAULT fica `pendente_do_autor`),
  e, no WSL (`scripts/verificar_wsl.sh`, num clone no hash): bandit, semgrep, pip_audit_pisos, pip_audit_lock,
  instalacao_limpa_3.10/3.13 e pytest_3.10/3.11/3.13 (uma versao por vez).

INCREMENTAL: cada etapa pesada declara seus caminhos de entrada (`verificacao_incremental.ENTRADAS`); se o hash desses caminhos e' igual
ao de uma execucao anterior em que a etapa foi `ok`, ela e' reaproveitada ("reaproveitado de <hash>"). Mudar so' `docs/qualidade/`
(gerado) nao reexecuta a suite, mas links, estado e auditores sempre rodam. `--sem-reaproveitar` forca tudo.

O hash verificado e' o HEAD do inicio; se o HEAD mudar durante a execucao o resultado e' DESCARTADO (rc 2). Arvore suja reprova.
Grava `docs/qualidade/resultado_<hash>.json` (sem caminho de maquina). Uma "falha" e' etapa `falhou` ou `nao_executada`.
Codigo de saida: 0 so' com 0 falhas e HEAD estavel.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import verificacao_incremental as vi  # noqa: E402

ESTADOS = ("ok", "falhou", "nao_se_aplica", "pendente_do_autor", "nao_executada")
PYTHON = sys.executable
DISTRO = os.environ.get("GUARACI_WSL_DISTRO", "Ubuntu")
VERSOES_WSL = ("3.10", "3.11", "3.13")
COBERTURA_NUCLEO_MIN = 100.0
COBERTURA_GLOBAL_MIN = 95.0
Resultado = Tuple[str, str]          # (estado, detalhe)
METRICAS: Dict[str, Any] = {}        # dados extras de uma etapa executada nesta rodada (ex.: cobertura da suite)


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=str(RAIZ), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip()


def head() -> str:
    return _git("rev-parse", "HEAD")


def pasta_de_registros(hash_: str) -> Path:
    """Onde ficam junit, cobertura e log completo de cada execucao: FORA do repositorio e nunca apagada pelo verificador (uma rodada
    reprovada sem o junit nao diz qual teste falhou). `GUARACI_REGISTROS_VERIFICACAO` troca a base (padrao `~/.guaraci_local/verificacoes`)."""
    base = Path(os.environ.get("GUARACI_REGISTROS_VERIFICACAO") or Path.home() / ".guaraci_local" / "verificacoes")
    pasta = base / f"{hash_[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def _rodar(cmd: List[str], timeout: int = 7200, env: Optional[Dict[str, str]] = None, log: Optional[Path] = None) -> Tuple[int, str]:
    """(codigo de saida, ultima linha). Com `log`, a saida COMPLETA fica gravada nele (o detalhe da etapa so' guarda a ultima linha)."""
    try:
        r = subprocess.run(cmd, cwd=str(RAIZ), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           env={**os.environ, **(env or {})})
    except FileNotFoundError as e:
        return 127, f"comando ausente: {e.filename}"
    except subprocess.TimeoutExpired:
        return 124, f"estourou {timeout}s"
    if log is not None:
        log.write_text(r.stdout + r.stderr, encoding="utf-8")
    saida = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode, (saida[-1] if saida else "")[:300]


def testes_que_falharam(junit: Path) -> List[str]:
    """`modulo::teste` de cada caso com <failure> ou <error> no junit (o detalhe da etapa precisa dizer QUAL teste falhou)."""
    import xml.etree.ElementTree as ET
    return [f"{tc.get('classname', '')}::{tc.get('name', '')}" for tc in ET.parse(junit).getroot().iter("testcase")
            if tc.find("failure") is not None or tc.find("error") is not None]


def _com_nomes(detalhe: str, nomes: List[str], limite: int = 20) -> str:
    if not nomes:
        return detalhe
    mais = f" (+{len(nomes) - limite})" if len(nomes) > limite else ""
    return f"{detalhe}; falharam: {', '.join(nomes[:limite])}{mais}"


def _por_rc(rc: int, detalhe: str, ok: str = "ok") -> Resultado:
    return ("ok", ok) if rc == 0 else ("falhou", detalhe or f"rc={rc}")


def trabalhadores() -> int:
    """Processos do xdist SO' no Windows: 2 por padrao (com mais, a CPU saturada ja derrubou o processo com 0xc000070a);
    `GUARACI_TESTES_WORKERS=N` escolhe outro, com teto de 4 e de metade dos nucleos. O WSL tem a propria variavel
    (`GUARACI_TESTES_WORKERS_WSL`, ver `_limite_xdist_wsl`)."""
    teto = max(1, min(4, (os.cpu_count() or 2) // 2))
    try:
        pedido = int(os.environ.get("GUARACI_TESTES_WORKERS", ""))
    except ValueError:
        return min(2, teto)
    return max(1, min(pedido, teto))


def pasta_temporaria_curta() -> str:
    """Base curta para os temporarios do pytest: o limite de 260 caracteres do Windows estoura com o nome longo do usuario + `popen-gwN`."""
    base = Path(os.environ.get("SystemDrive", "C:") + "/") / "gt" if os.name == "nt" else Path("/tmp/gt")
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


# ── etapas deste lado (Windows/Linux nativo) ─────────────────────────────────
def e_arvore_limpa() -> Resultado:
    sujo = _git("status", "--porcelain")
    return ("ok", "arvore limpa") if not sujo else ("falhou", f"{len(sujo.splitlines())} arquivo(s) modificado(s) ou nao rastreado(s)")


def e_suite_completa() -> Resultado:
    import congelar_nucleo as cn
    import estado_do_projeto as ep
    d = pasta_de_registros(head())
    junit, cob = d / "suite_completa_junit.xml", d / "suite_completa_cobertura.xml"
    rc, ult = _rodar([PYTHON, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-n", str(trabalhadores()), "-rf",
                      f"--basetemp={pasta_temporaria_curta()}", f"--junitxml={junit}", "--cov=src/guaraci", "--cov-branch",
                      f"--cov-report=xml:{cob}"],
                     env={"GUARACI_PULADOS_AMBIENTE": "ci", "OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"},
                     log=d / "suite_completa.log")
    if not junit.exists():
        return "falhou", f"sem junit: {ult}"
    j = ep.ler_junit(junit)
    prob = ep.pulados_fora_do_previsto(j["motivos"], "ci")
    camadas = ep.cobertura_por_camada(cob, cn.arquivos_do_nucleo()) if cob.exists() else None
    lacunas = _lacunas_em_dia(cob) if cob.exists() else None
    METRICAS["suite_completa"] = {"cobertura": camadas, "cobertura_lacunas": lacunas}
    cobtxt = (f"global linhas {camadas['global']['linhas']}% ramos {camadas['global']['ramos']}%, "
              f"nucleo linhas {camadas['nucleo']['linhas']}% ramos {camadas['nucleo']['ramos']}%") if camadas else "sem cobertura"
    detalhe = f"{j['passaram']} passaram, {j['pulados']} pulados, {j['falharam']} falharam, {j['erros']} erros; {cobtxt}"
    if j["falharam"] or j["erros"] or rc not in (0, 1) or prob:
        return "falhou", _com_nomes(detalhe, testes_que_falharam(junit)) + (f"; pulados fora do previsto: {prob}" if prob else "")
    return ("ok", detalhe) if rc == 0 else ("falhou", detalhe + f"; rc={rc}")


def _lacunas_em_dia(cob: Path) -> Resultado:
    """`docs/qualidade/cobertura_lacunas.md` e' o gerado a partir da cobertura DESTA execucao?"""
    import gerar_cobertura_lacunas as gcl
    try:
        atual = gcl.SAIDA.read_bytes().decode("utf-8").replace("\r\n", "\n") if gcl.SAIDA.exists() else ""
        if gcl.gerar(cob) == atual:
            return "ok", "docs/qualidade/cobertura_lacunas.md em dia com a cobertura medida"
        return "falhou", "docs/qualidade/cobertura_lacunas.md desatualizado: rode scripts/gerar_cobertura_lacunas.py com o coverage.xml da suite"
    except Exception as e:  # noqa: BLE001 -- documento que nao gera e' etapa que falhou
        return "falhou", f"cobertura_lacunas: {type(e).__name__}: {e}"


def e_cobertura_lacunas() -> Resultado:
    return (METRICAS.get("suite_completa") or {}).get("cobertura_lacunas") or ("nao_executada", "a suite nao produziu cobertura")


def avaliar_cobertura(camadas: Optional[Dict[str, Dict[str, float]]]) -> Resultado:
    """Nucleo cientifico 100% de linhas e de ramos; global combinada >= 95%."""
    if not camadas:
        return "nao_executada", "a suite nao produziu cobertura"
    n, g = camadas["nucleo"], camadas["global"]
    ruins = []
    if n["linhas"] < COBERTURA_NUCLEO_MIN or n["ramos"] < COBERTURA_NUCLEO_MIN:
        ruins.append(f"nucleo linhas {n['linhas']}% ramos {n['ramos']}% (exigido {COBERTURA_NUCLEO_MIN:g}%)")
    if g["combinada"] < COBERTURA_GLOBAL_MIN:
        ruins.append(f"global {g['combinada']}% (exigido {COBERTURA_GLOBAL_MIN:g}%)")
    texto = f"nucleo {n['linhas']}%/{n['ramos']}% (linhas/ramos), global {g['combinada']}% (combinada)"
    return ("falhou", "; ".join(ruins)) if ruins else ("ok", texto)


def e_cobertura_minima() -> Resultado:
    return avaliar_cobertura((METRICAS.get("suite_completa") or {}).get("cobertura"))


def e_ruff() -> Resultado:
    return _por_rc(*_rodar([PYTHON, "-m", "ruff", "check", "."]), ok="ruff limpo")


def e_mypy() -> Resultado:
    return _por_rc(*_rodar([PYTHON, "-m", "mypy", "src/guaraci/", "app_quimiometria.py"]), ok="mypy limpo")


def e_contrato_catalogo() -> Resultado:
    import estado_do_projeto as ep
    v = ep.violacoes_do_catalogo()
    return ("ok", "0 violacoes") if not v else ("falhou", f"{len(v)} violacao(oes)")


def e_links() -> Resultado:
    import verificar_links_docs as vl
    q = vl.quebrados_na_arvore(RAIZ)
    return ("ok", "0 links quebrados") if not q else ("falhou", f"{sum(len(x) for x in q.values())} link(s) quebrado(s) em {len(q)} documento(s)")


def e_exposicao(historico: Optional[str] = None) -> Resultado:
    import auditar_exposicao as ae
    locais = ae.termos_locais()
    arvore = sum(len(v) for v in ae.varrer_arvore(RAIZ).values())
    hist = sum(ae.varrer_historico(historico).values()) if historico else 0
    ok = arvore == 0 and hist == 0
    d = f"arvore {arvore}" + (f", historico {historico} {hist}" if historico else ", historico nao varrido")
    if not locais:
        d += "; R5 (termos locais) NAO executado: lista vazia"
    return ("ok", d) if ok else ("falhou", d)


def e_r5_termos_locais() -> Resultado:
    """R5 (termos sensiveis do autor) so' vale se a lista local tem termos: lista vazia = NAO EXECUTADO, nunca '0 ocorrencias'."""
    import auditar_exposicao as ae
    n = len(ae.termos_locais())
    if n == 0:
        return ("pendente_do_autor", "R5 NAO EXECUTADO: a lista local de termos sensiveis esta vazia (preencha termos_sensiveis.txt, "
                                     "um termo por linha); as demais regras rodaram")
    return "ok", f"R5 executado com {n} termo(s) locais"


def e_termos_de_matriz() -> Resultado:
    import estado_do_projeto as ep
    n = ep.ocorrencias_termos_de_matriz()
    return ("ok", "0 ocorrencias fora da validacao") if n == 0 else ("falhou", f"{n} ocorrencia(s) fora da validacao")


def e_referencias() -> Resultado:
    return _por_rc(*_rodar([PYTHON, "scripts/gerar_tabela_referencias.py", "--verificar"]), ok="0 nao conferidas, tabela em dia")


def e_docs_gerados_em_dia() -> Resultado:
    for s in ("gerar_mapa_tecnica_tarefa.py", "gerar_indice_docs.py", "gerar_docs_pacote.py"):
        rc, d = _rodar([PYTHON, f"scripts/{s}", "--verificar"])
        if rc != 0:
            return "falhou", f"{s}: {d}"
    rc, d = _rodar([PYTHON, "scripts/referencias_docs.py", "--verificar"])
    if rc != 0:
        return "falhou", f"referencias_docs.py: {d}"
    return "ok", "mapa, indice, copias do pacote e referencias de documentos em dia"


def e_matriz_cenarios() -> Resultado:
    d = pasta_de_registros(head())
    saida = d / "matriz_cenarios.md"
    rc, ult = _rodar([PYTHON, "scripts/matriz_cenarios.py", "--saida", str(saida)], timeout=3600, log=d / "matriz_cenarios.log")
    if not saida.exists():
        return "falhou", ult or f"rc={rc}"
    import estado_do_projeto as ep
    c = ep.contar_cenarios(saida.read_text(encoding="utf-8"))
    total = sum(c.values())
    return ("ok", f"{c.get('passou', 0)}/{total} passaram") if c.get("passou", 0) == total and total else ("falhou", f"{c}")


def e_pacote_construido() -> Resultado:
    """Constroi wheel+sdist, confere o conteudo e percorre a ajuda do wheel instalado em venv limpo (Python 3.13 por padrao)."""
    py = os.environ.get("GUARACI_PY_WHEEL") or str(Path.home() / ".venvs" / "py313" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    if not Path(py).is_file():
        return "nao_executada", "sem interprete 3.10-3.13 para o venv limpo (defina GUARACI_PY_WHEEL)"
    rc, ult = _rodar([PYTHON, "scripts/auditar_pacote_construido.py", "--instalar", py], timeout=1800)
    return _por_rc(rc, ult, ok="wheel e sdist sem arquivo indevido; ajuda do wheel instalado percorrida sem 'nao encontrada'")


def e_rodada_local_formatos() -> Resultado:
    """A conformidade dos leitores com arquivos REAIS de fabricante foi rodada (no checkout do mantenedor) para o codigo atual dos leitores?"""
    import registrar_rodada_local as rrl
    motivo = rrl.registro_atual_cobre_o_codigo()
    return ("ok", "rodada local registrada e em dia com o codigo dos leitores") if motivo is None else ("falhou", motivo)


def e_vault() -> Resultado:
    """Roda `medir_vault.py --gerar` se GUARACI_VAULT existe; sem a variavel nao e' aprovacao: e' pendencia do autor."""
    if not os.environ.get("GUARACI_VAULT"):
        return "pendente_do_autor", "nao executado: defina GUARACI_VAULT (e GUARACI_VAULT_GERADOR, GUARACI_LIVRO_RAZAO) para medir o vault"
    args = ["--gerar"] if os.environ.get("GUARACI_VAULT_GERADOR") else []
    rc, ult = _rodar([PYTHON, "scripts/medir_vault.py", *args], timeout=1800)
    if rc == 3:
        return "pendente_do_autor", ult
    return _por_rc(rc, ult, ok=ult)


# ── lado WSL ─────────────────────────────────────────────────────────────────
def _caminho_wsl(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    return f"/mnt/{m.group(1).lower()}/{m.group(2)}" if m else s


def parse_resumo(texto: str) -> Dict[str, Tuple[int, str]]:
    """Linhas `RESUMO nome rc=N [resumo="..."]` -> {nome: (rc, resumo)}; a ultima ocorrencia de cada nome vale.
    Linhas `FALHOU nome teste` (o teste que falhou numa etapa pytest do WSL) entram no resumo daquela etapa."""
    out: Dict[str, Tuple[int, str]] = {}
    falhas: Dict[str, List[str]] = {}
    for ln in texto.splitlines():
        m = re.match(r'^RESUMO (\w[\w.]*)(?: rc=(\d+))?(?: rc=(\d+))?(?: resumo="(.*)")?', ln.strip())
        if m:
            rc = m.group(2) or m.group(3)
            out[m.group(1)] = (int(rc) if rc is not None else 99, m.group(4) or "")
        f = re.match(r"^FALHOU (\w[\w.]*) (\S+)", ln.strip())
        if f:
            falhas.setdefault(f.group(1), []).append(f.group(2))
    return {n: (rc, _com_nomes(txt, falhas.get(n, []))) for n, (rc, txt) in out.items()}


def nomes_wsl() -> List[str]:
    return (["bandit", "semgrep", "pip_audit_pisos", "pip_audit_lock"] + [f"instalacao_limpa_{v}" for v in ("3.10", "3.13")]
            + [f"pytest_{v}" for v in VERSOES_WSL])


def _limite_xdist_wsl() -> str:
    """`export GUARACI_TESTES_WORKERS_WSL=N && ` quando essa variavel foi pedida (inteiro >= 1): o WSL nao herda o ambiente do Windows.
    A variavel do Windows (`GUARACI_TESTES_WORKERS`) NAO e' repassada; o teto e o padrao (4) ficam no `verificar_wsl.sh`."""
    try:
        n = int(os.environ.get("GUARACI_TESTES_WORKERS_WSL", ""))
    except ValueError:
        return ""
    return f"export GUARACI_TESTES_WORKERS_WSL={n} && " if n >= 1 else ""


def rodar_wsl(hash_: str, etapas: Optional[List[str]] = None) -> Tuple[Dict[str, Tuple[int, str]], str]:
    """Roda no WSL so' as `etapas` pedidas (todas se None). Os pytest rodam UMA versao por vez."""
    sh = _caminho_wsl(RAIZ / "scripts" / "verificar_wsl.sh")
    so = ",".join(etapas) if etapas else ""
    cmd = ["wsl", "-d", DISTRO, "--", "bash", "-c",
           f"{_limite_xdist_wsl()}tr -d '\\r' < '{sh}' > /tmp/verificar_wsl.sh && "
           f"bash /tmp/verificar_wsl.sh {hash_[:12]} '{_caminho_wsl(RAIZ)}' '' '{so}'"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=6 * 3600)
        texto = (r.stdout or "").replace("\x00", "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {}, f"WSL indisponivel: {type(e).__name__}"
    return parse_resumo(texto), ""


def etapas_wsl(resumo: Dict[str, Tuple[int, str]], motivo: str) -> List[Tuple[str, Resultado]]:
    out: List[Tuple[str, Resultado]] = []
    for n in nomes_wsl():
        if n not in resumo:
            out.append((n, ("nao_executada", motivo or "o script do WSL nao reportou esta etapa")))
        else:
            rc, txt = resumo[n]
            out.append((n, ("ok", txt or "rc=0") if rc == 0 else ("falhou", f"rc={rc} {txt}".strip())))
    return out


# ── nucleo testavel ──────────────────────────────────────────────────────────
def contar_falhas(etapas: List[dict]) -> int:
    """Falha = etapa `falhou` ou `nao_executada`. `ok`, `nao_se_aplica` (com motivo) e `pendente_do_autor` nao contam."""
    return sum(1 for e in etapas if e["estado"] in ("falhou", "nao_executada"))


def versoes_python_ok(etapas: List[dict], local: str) -> List[str]:
    """Versoes de Python cuja suite completa passou: a do ambiente local e as do WSL."""
    por = {e["nome"]: e["estado"] for e in etapas}
    v = [local] if por.get("suite_completa") == "ok" else []
    v += [x for x in VERSOES_WSL if por.get(f"pytest_{x}") == "ok"]
    return sorted(set(v), key=lambda s: tuple(int(p) for p in s.split(".")))


def montar_resultado(hash_inicial: str, hash_final: str, arvore: str, ramo: str, etapas: List[dict], local: str,
                     inicio: str, fim: str) -> dict:
    return {"hash": hash_inicial, "head_estavel": hash_inicial == hash_final, "arvore": arvore, "ramo": ramo,
            "iniciado_em": inicio, "concluido_em": fim, "versoes_python": versoes_python_ok(etapas, local),
            "etapas": etapas, "falhas": contar_falhas(etapas),
            "pendentes_do_autor": [e["nome"] for e in etapas if e["estado"] == "pendente_do_autor"],
            "reaproveitadas": [e["nome"] for e in etapas if e.get("origem") == "reaproveitada"],
            "executadas": [e["nome"] for e in etapas if e.get("origem", "executada") == "executada"]}


def veredito(res: dict) -> int:
    """0 so' com 0 falhas e HEAD estavel; 2 se o HEAD mudou (resultado invalido); 1 se houve falha."""
    if not res["head_estavel"]:
        return 2
    return 0 if res["falhas"] == 0 else 1


def decidir_reaproveitamento(nome: str, anteriores: List[dict], permitido: bool, raiz: Path = RAIZ,
                             git: Optional[Callable[..., str]] = None) -> Tuple[Optional[str], Optional[Tuple[str, dict]]]:
    """(hash das entradas da etapa, (hash de origem, etapa anterior) se da' para reaproveitar). Sem reaproveitamento: (hash, None)."""
    h = vi.hash_de_entradas(raiz, nome, git)
    if not permitido:
        return h, None
    return h, vi.reaproveitar(nome, h, anteriores)


ETAPAS_LOCAIS: List[Tuple[str, Callable[[], Resultado]]] = [
    ("arvore_limpa", e_arvore_limpa), ("suite_completa", e_suite_completa), ("cobertura_minima", e_cobertura_minima),
    ("cobertura_lacunas", e_cobertura_lacunas), ("ruff", e_ruff), ("mypy", e_mypy),
    ("contrato_catalogo", e_contrato_catalogo), ("links", e_links), ("termos_de_matriz", e_termos_de_matriz),
    ("referencias", e_referencias), ("docs_gerados_em_dia", e_docs_gerados_em_dia), ("matriz_cenarios", e_matriz_cenarios),
    ("pacote_construido", e_pacote_construido), ("rodada_local_formatos", e_rodada_local_formatos), ("vault", e_vault),
]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--historico", metavar="INTERVALO", help="intervalo git a varrer no auditor de exposicao (ex.: origin/master..HEAD)")
    ap.add_argument("--pular", nargs="*", default=[], help="etapas a nao executar (viram `nao_executada`: contam como falha)")
    ap.add_argument("--so", nargs="*", default=[], help="so' estas etapas (para diagnostico; o resultado NAO e' gravado)")
    ap.add_argument("--sem-gravar", action="store_true")
    ap.add_argument("--sem-reaproveitar", action="store_true", help="executa tudo, mesmo o que ja passou com as mesmas entradas")
    a = ap.parse_args(argv)
    h0 = head()
    inicio = datetime.now(timezone.utc).isoformat(timespec="seconds")
    etapas: List[dict] = []
    anteriores = vi.carregar_anteriores(RAIZ / "docs" / "qualidade")

    def registrar(nome: str, fn: Callable[[], Resultado]) -> None:
        if a.so and nome not in a.so:
            return
        if nome in a.pular:
            etapas.append({"nome": nome, "estado": "nao_executada", "detalhe": "pulada por --pular", "duracao_s": 0.0})
            return
        h, ant = decidir_reaproveitamento(nome, anteriores, not a.sem_reaproveitar)
        if ant is not None and nome == "suite_completa" and not (ant[1].get("metricas") or {}).get("cobertura"):
            ant = None                                       # registro antigo, sem a cobertura que a etapa `cobertura_minima` precisa
        if ant is not None:
            origem, e_ant = ant
            if nome == "suite_completa":
                METRICAS[nome] = e_ant["metricas"]
            etapas.append({"nome": nome, "estado": "ok", "detalhe": f"{e_ant.get('detalhe', '')} (reaproveitado de {origem})", "duracao_s": 0.0,
                           "origem": "reaproveitada", "reaproveitada_de": origem, "entradas": h, "metricas": e_ant.get("metricas")})
            print(f"[{'reaproveitada':>17}] {nome}: de {origem}", flush=True)
            return
        t0 = time.time()
        try:
            estado, det = fn()
        except Exception as e:  # noqa: BLE001 -- uma etapa que explode e' uma etapa que falhou, nao um erro do verificador
            estado, det = "falhou", f"{type(e).__name__}: {e}"
        reg: Dict[str, Any] = {"nome": nome, "estado": estado, "detalhe": det, "duracao_s": round(time.time() - t0, 1), "origem": "executada"}
        if h:
            reg["entradas"] = h
        if nome == "suite_completa":
            reg["metricas"] = METRICAS.get(nome)
        etapas.append(reg)
        print(f"[{estado:>17}] {nome}: {det}", flush=True)

    registrar("arvore_limpa", e_arvore_limpa)
    registrar("exposicao", lambda: e_exposicao(a.historico))
    registrar("r5_termos_locais", e_r5_termos_locais)
    for nome, fn in ETAPAS_LOCAIS[1:]:
        registrar(nome, fn)
    if not a.so or any(x.startswith(("bandit", "semgrep", "pip_audit", "instalacao", "pytest_")) for x in a.so):
        a_rodar = [n for n in nomes_wsl() if (not a.so or n in a.so) and n not in a.pular
                   and decidir_reaproveitamento(n, anteriores, not a.sem_reaproveitar)[1] is None]
        resumo, motivo = rodar_wsl(h0, a_rodar) if a_rodar else ({}, "")
        for nome, res in etapas_wsl(resumo, motivo):
            registrar(nome, lambda r=res: r)
    h1 = head()
    res = montar_resultado(h0, h1, _git("rev-parse", "HEAD^{tree}"), _git("rev-parse", "--abbrev-ref", "HEAD"), etapas,
                           f"{sys.version_info.major}.{sys.version_info.minor}", inicio, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    rc = veredito(res)
    print(f"\nHASH {h0[:12]}  falhas {res['falhas']}  pendentes do autor {len(res['pendentes_do_autor'])}  "
          f"HEAD {'estavel' if res['head_estavel'] else 'MUDOU (resultado descartado)'}  "
          f"Python verde: {', '.join(res['versoes_python']) or '-'}  executadas {len(res['executadas'])}  reaproveitadas {len(res['reaproveitadas'])}")
    if rc != 2 and not a.sem_gravar and not a.so:
        destino = RAIZ / "docs" / "qualidade" / f"resultado_{h0[:12]}.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"gravado: {destino.relative_to(RAIZ).as_posix()}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
