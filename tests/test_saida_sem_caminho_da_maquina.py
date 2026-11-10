"""A pasta de resultados de `guaraci run` (e os relatorios de `guaraci relatorio`) nao carrega o
caminho absoluto da maquina nem usuario de sistema.

POR QUE: os resultados saem da maquina (relatorio para colegas, material suplementar, modelo
portatil). O mecanismo que protege isso esta espalhado -- `contexto.texto_efetiva` grava so' o
nome final dos caminhos, `sanitizar_metadados` tira identificadores, as tabelas usam id anonimo --
e so' a config efetiva tinha teste (test_contexto). As funcoes de varredura de diretorio do
Passo 169 nunca foram ligadas a nada (removidas em 2026-10-04); esta e' a varredura que faltava,
sobre a saida inteira. No Windows a pasta temporaria do pytest fica dentro da pasta do usuario, entao
um vazamento aparece como o proprio caminho do teste ou como caminho de usuario do sistema.
"""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_SCRIPTS = str(RAIZ / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from privacidade_amostras import CATEGORIAS_REAIS, PADRAO_CAMINHO_ABSOLUTO, classificar_ocorrencia  # noqa: E402

_TEXTO = {".csv", ".txt", ".md", ".json", ".yaml", ".yml", ".tex", ".html", ".log", ".svg"}
_ZIP = {".xlsx", ".docx", ".pptx"}


def _conteudos(raiz: Path):
    """(arquivo, texto) de cada arquivo de texto, e de cada parte XML dentro de xlsx/docx/pptx."""
    for p in sorted(raiz.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(raiz))
        if p.suffix.lower() in _TEXTO:
            yield rel, p.read_text(encoding="utf-8", errors="ignore")
        elif p.suffix.lower() in _ZIP:
            with zipfile.ZipFile(p) as z:
                for nome in z.namelist():
                    if nome.endswith((".xml", ".rels")):
                        yield f"{rel}!{nome}", z.read(nome).decode("utf-8", "ignore")


def _vazamentos(raiz: Path, proibidos):
    achados = []
    for rel, texto in _conteudos(raiz):
        for trecho in proibidos:
            if trecho and trecho in texto:
                achados.append((rel, "caminho da pasta de dados"))
        for m in PADRAO_CAMINHO_ABSOLUTO.finditer(texto):
            if classificar_ocorrencia(m.group(0)) in CATEGORIAS_REAIS:
                achados.append((rel, "caminho de usuario"))
    return achados


def test_varredura_acusa_caminho_plantado(tmp_path):
    """Contraprova: a varredura acha o caminho quando ele esta la (texto e dentro de xlsx)."""
    (tmp_path / "a.txt").write_text("dados em " + str(tmp_path / "dados"), encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "r.xlsx", "w") as z:
        z.writestr("xl/sharedStrings.xml", "<t>" + "C:" + "\\" + "Users" + "\\" + "fulano" + "\\x</t>")
    achados = _vazamentos(tmp_path, [str(tmp_path / "dados")])
    assert ("a.txt", "caminho da pasta de dados") in achados
    assert any(r.startswith("r.xlsx!") and tipo == "caminho de usuario" for r, tipo in achados)


@pytest.mark.slow
def test_resultados_e_relatorios_sem_caminho_da_maquina(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "USERPROFILE": str(tmp_path / "home"), "HOME": str(tmp_path / "home")}
    (tmp_path / "home").mkdir()
    dados = tmp_path / "projeto"
    for args in (["exemplo", "classificar", str(dados)], ["run", str(dados / "config.yaml")]):
        r = subprocess.run([sys.executable, "-m", "guaraci", *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env, cwd=str(tmp_path), timeout=600)
        assert r.returncode == 0, r.stdout[-800:]
    execucao = next(dados.rglob("resumo_modelo.txt")).parents[1]
    r = subprocess.run([sys.executable, "-m", "guaraci", "relatorio", str(execucao), "--formato", "xlsx,docx,tex"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                       cwd=str(tmp_path), timeout=600)
    assert r.returncode == 0, r.stdout[-800:]
    proibidos = {str(dados), str(dados).replace("\\", "/"), str(tmp_path), str(tmp_path).replace("\\", "/")}
    achados = _vazamentos(execucao, proibidos)
    assert not achados, f"{len(achados)} vazamento(s): " + "; ".join(f"{a} ({t})" for a, t in achados[:10])
    assert any(p.suffix == ".xlsx" for p in execucao.rglob("*"))      # o relatorio foi de fato varrido
