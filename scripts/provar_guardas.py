#!/usr/bin/env python3
"""provar_guardas.py -- as guardas das guardas: injeta UMA violacao controlada numa copia descartavel e prova que a guarda falha.

    python scripts/provar_guardas.py [--so NOME[,NOME]] [--saida ARQUIVO.md]

Como funciona: copia os arquivos RASTREADOS para uma pasta temporaria (nunca o `.git` do projeto) e faz nela um repositorio descartavel
(`git init`, identidade sintetica), porque varias guardas leem `git ls-files`. Para cada caso: injeta a violacao, roda os comandos da guarda
(script ou teste) e confere que sai com codigo != 0, depois restaura a copia. Antes, cada comando roda na copia LIMPA: uma guarda que ja
falha sem violacao e' reportada como tal (nao conta como prova). O projeto real nao e' tocado e nada e' enviado a lugar nenhum.

As violacoes sao sinteticas e montadas em tempo de execucao (nenhum literal sensivel neste arquivo, para nao disparar as proprias guardas).
Codigo de saida 1 se alguma guarda NAO falhou diante da violacao. Tabela: guarda | violacao injetada | falhou? | teste que cobre.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

RAIZ = Path(__file__).resolve().parents[1]
PY = sys.executable
BARRA = chr(92)
ARROBA = chr(64)


def _pytest(*alvos: str, k: str = "") -> List[str]:
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", *alvos]
    return cmd + (["-k", k] if k else [])


def _anexar(rel: str, texto: str) -> Callable[[Path], None]:
    def f(raiz: Path) -> None:
        with open(raiz / rel, "a", encoding="utf-8", newline="") as fh:
            fh.write("\n" + texto + "\n")
    return f


def _novo(rel: str, texto: str) -> Callable[[Path], None]:
    def f(raiz: Path) -> None:
        p = raiz / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texto, encoding="utf-8")
    return f


def _trocar(rel: str, velho: str, novo: str) -> Callable[[Path], None]:
    def f(raiz: Path) -> None:
        p = raiz / rel
        s = p.read_bytes().decode("utf-8")
        assert velho in s, f"{rel}: trecho a trocar nao encontrado"
        p.write_bytes(s.replace(velho, novo, 1).encode("utf-8"))
    return f


@dataclass
class Caso:
    guarda: str
    violacao: str
    injetar: Callable[[Path], None]
    comandos: List[Tuple[str, List[str]]]
    teste_que_cobre: str
    #: versao BENIGNA da injecao (mesmo arquivo, sem a violacao), usada na linha de base quando o comando precisa do arquivo
    benigno: Optional[Callable[[Path], None]] = field(default=None)


def casos(tmp: Path) -> List[Caso]:
    termos = tmp / "termos_sinteticos.txt"
    termos.write_text("palavra_sintetica_" + "xyz\n", encoding="utf-8")
    vazio = tmp / "termos_vazio.txt"
    vazio.write_text("# vazio\n", encoding="utf-8")
    exposicao = [PY, "scripts/auditar_exposicao.py"]
    email = "contato maria.teste" + ARROBA + "inst.edu.br"
    caminho = "veja C:" + BARRA + "Users" + BARRA + "maria_teste" + BARRA + "dados"
    return [
        Caso("dados publicos nao citados", "frase 'dados publicos' no README", _anexar("README.md", "Usamos dados " + "públicos de referencia."),
             [("teste", _pytest("tests/test_guarda_dados_de_terceiros.py"))], "tests/test_guarda_dados_de_terceiros.py"),
        Caso("termos de matriz", "nome de matriz no README", _anexar("README.md", "Aplicado a amostras de mi" + "lho."),
             [("script", [PY, "scripts/auditar_termos_matriz.py"]), ("teste", _pytest("tests/test_auditar_termos_matriz.py"))],
             "tests/test_auditar_termos_matriz.py"),
        Caso("termos sensiveis (lista privada)", "termo sintetico da lista, no README", _anexar("README.md", "ver palavra_sintetica_" + "xyz aqui"),
             [("script", exposicao + ["--local", str(termos)])], "tests/test_auditar_exposicao.py::test_lista_local_do_autor_vira_regra"),
        Caso("e-mail", "e-mail sintetico no README", _anexar("README.md", email),
             [("script", exposicao + ["--local", str(vazio)]),
              ("teste", _pytest("tests/test_identidade_e_contato.py", k="email or generico"))],
             "tests/test_identidade_e_contato.py::test_nenhum_arquivo_versionado_tem_email_nem_padrao_generico"),
        Caso("caminho de usuario", "caminho Windows de usuario sintetico no MANUAL", _anexar("MANUAL.md", caminho),
             [("script", exposicao + ["--local", str(vazio)])], "tests/test_auditar_exposicao.py (regra caminho_usuario_windows)"),
        Caso("identificador de amostra real", "codigo de amostra no formato COD-DD-MM-AAAA no MANUAL", _anexar("MANUAL.md", "amostra ES" + "C-03-01-2020 do lote"),
             [("teste", _pytest("tests/test_sem_identificador_real.py"))], "tests/test_sem_identificador_real.py"),
        Caso("links", "link quebrado em docs/index.md", _anexar("docs/index.md", "[some](docs/NAO_EXISTE_XYZ.md)"),
             [("script", [PY, "scripts/verificar_links_docs.py"]), ("teste", _pytest("tests/test_links_docs.py"))], "tests/test_links_docs.py"),
        Caso("excecoes de cobertura", "novo `# pragma: no cover` sem registro", _anexar("src/guaraci/log.py", "_x_injetado = 1  # pragma: " + "no cover"),
             [("teste", _pytest("tests/test_cobertura_excecoes.py"))], "tests/test_cobertura_excecoes.py"),
        Caso("manifesto do nucleo", "arquivo do nucleo alterado sem regenerar o manifesto", _anexar("src/guaraci/classificadores.py", "# alteracao injetada"),
             [("script", [PY, "scripts/congelar_nucleo.py", "--verificar"]), ("teste", _pytest("tests/test_nucleo_congelado.py"))],
             "tests/test_nucleo_congelado.py"),
        Caso("sincronia de versao", "versao do CITATION.cff diferente da do codigo", _trocar("CITATION.cff", 'version: "', 'version: "9.9.9-injetada'),
             [("teste", _pytest("tests/test_versao_coerente.py"))], "tests/test_versao_coerente.py"),
        Caso("documentos gerados (copias no pacote)", "MANUAL.md alterado sem regenerar a copia do pacote", _anexar("MANUAL.md", "linha injetada"),
             [("script", [PY, "scripts/gerar_docs_pacote.py", "--verificar"]), ("teste", _pytest("tests/test_documentos_no_pacote.py"))],
             "tests/test_documentos_no_pacote.py"),
        Caso("documentos gerados (indice)", "documento novo sem descricao no indice", _novo("docs/projeto/injetado.md", "# Injetado\n\n**Público:** quem testa\n"),
             [("script", [PY, "scripts/gerar_indice_docs.py", "--verificar"])], "tests/test_estado_do_projeto.py / indice gerado"),
        Caso("dependencia de maquina nos testes", "teste com relogio sem decisao registrada", _novo("tests/test_zz_injetado.py", "import time\n\n\ndef test_x():\n    time.sleep(0)\n"),
             [("script", [PY, "scripts/varrer_dependencias_de_maquina.py", "--verificar"])], "tests/test_dependencias_de_maquina.py"),
        Caso("pulados esperados", "teste que pula com motivo nao previsto", _novo("tests/test_zz_injetado.py", "import pytest\n\n\ndef test_x():\n    pytest.skip('motivo novo e imprevisto')\n"),
             [("teste", _pytest("tests/test_zz_injetado.py"))], "tests/test_pulados_esperados.py",
             benigno=_novo("tests/test_zz_injetado.py", "def test_x():\n    assert True\n")),
        Caso("remissao a script inexistente", "documento cita um script que nao existe", _anexar("docs/projeto/ROADMAP.md", "rode `scripts/" + "nao_existe_xyz.py`"),
             [("teste", _pytest("tests/test_referencias_a_scripts.py"))], "tests/test_referencias_a_scripts.py"),
        Caso("licenca unica", "README oferece termos diferentes da GPL", _anexar("README.md", "For terms other than the GPLv3, ask the copyright holder."),
             [("teste", _pytest("tests/test_autoria_e_licencas.py", k="outros_termos"))], "tests/test_autoria_e_licencas.py::test_nenhuma_oferta_de_outros_termos_de_licenca"),
        Caso("ferramental de reescrita fora do git", "script de reescrita de historico volta para scripts/", _novo("scripts/reconstruir_historico_" + "publicavel.py","# injetado\n"),
             [("teste", _pytest("tests/test_identidade_e_contato.py", k="ferramental"))], "tests/test_identidade_e_contato.py::test_ferramental_de_reescrita_do_historico_nao_e_versionado"),
    ]


def _env(home: Path, raiz: Path) -> Dict[str, str]:
    e = dict(os.environ)
    e.update({"USERPROFILE": str(home), "HOME": str(home), "PYTHONPATH": str(raiz / "src"), "PYTHONUTF8": "1", "MPLBACKEND": "Agg",
              "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"})
    return e


def _rodar(cmd: Sequence[str], raiz: Path, home: Path, timeout: int = 900) -> int:
    try:
        return subprocess.run(list(cmd), cwd=str(raiz), env=_env(home, raiz), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124


def _git(raiz: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=teste", "-c", "user.email=teste" + ARROBA + "example.org", *args], cwd=str(raiz), check=True,
                   capture_output=True)


def copia_descartavel(destino: Path) -> None:
    """Copia os arquivos rastreados (sem o `.git` do projeto) e cria um repositorio descartavel por cima."""
    nomes = subprocess.run(["git", "ls-files", "-z"], cwd=str(RAIZ), capture_output=True).stdout.decode("utf-8").split("\0")
    for n in nomes:
        if not n:
            continue
        origem = RAIZ / n
        if origem.is_file():
            (destino / n).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origem, destino / n)
    _git(destino, "init", "-q")
    _git(destino, "add", "-A")
    _git(destino, "commit", "-q", "-m", "copia descartavel")


def restaurar(raiz: Path) -> None:
    _git(raiz, "checkout", "-q", "--", ".")
    _git(raiz, "clean", "-fdq")
    _git(raiz, "reset", "-q", "--hard")


def provar(selecionados: Optional[Sequence[str]] = None) -> List[Dict]:
    resultados: List[Dict] = []
    with tempfile.TemporaryDirectory(prefix="pg_") as t:
        tmp = Path(t)
        raiz, home = tmp / "copia", tmp / "home"
        raiz.mkdir()
        home.mkdir()
        copia_descartavel(raiz)
        base_cache: Dict[Tuple[str, ...], int] = {}
        for c in casos(tmp):
            if selecionados and not any(s.lower() in c.guarda.lower() for s in selecionados):
                continue
            linhas = []
            if c.benigno:
                c.benigno(raiz)
                _git(raiz, "add", "-A")
            for rotulo, cmd in c.comandos:
                chave = tuple(cmd)
                if chave not in base_cache:
                    base_cache[chave] = _rodar(cmd, raiz, home)
            if c.benigno:
                restaurar(raiz)
            c.injetar(raiz)
            _git(raiz, "add", "-A")
            for rotulo, cmd in c.comandos:
                rc = _rodar(cmd, raiz, home)
                base = base_cache[tuple(cmd)]
                linhas.append({"comando": rotulo, "rc_limpa": base, "rc_injetada": rc, "prova": base == 0 and rc != 0})
            restaurar(raiz)
            resultados.append({"guarda": c.guarda, "violacao": c.violacao, "teste": c.teste_que_cobre, "comandos": linhas,
                               "falhou": any(l["prova"] for l in linhas), "base_ja_falhava": any(l["rc_limpa"] != 0 for l in linhas)})
            print(f"[{'falhou' if resultados[-1]['falhou'] else 'NAO FALHOU'}] {c.guarda}: "
                  + ", ".join(f"{l['comando']} limpa={l['rc_limpa']} injetada={l['rc_injetada']}" for l in linhas), flush=True)
    return resultados


def tabela(resultados: Sequence[Dict]) -> str:
    linhas = ["| guarda | violacao injetada | falhou? | teste que cobre |", "|---|---|---|---|"]
    for r in resultados:
        det = "; ".join(f"{l['comando']}: limpa={l['rc_limpa']}, injetada={l['rc_injetada']}" for l in r["comandos"])
        linhas.append(f"| {r['guarda']} | {r['violacao']} | {'SIM' if r['falhou'] else 'NAO'} ({det}) | `{r['teste']}` |")
    return "\n".join(linhas) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--so", default="", help="so' os casos cujo nome de guarda contem um destes trechos (separados por virgula)")
    ap.add_argument("--saida", default="", help="grava a tabela em Markdown neste arquivo")
    a = ap.parse_args(argv)
    res = provar([s for s in a.so.split(",") if s] or None)
    tab = tabela(res)
    print("\n" + tab)
    if a.saida:
        Path(a.saida).write_text(tab, encoding="utf-8")
    return 0 if res and all(r["falhou"] for r in res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
