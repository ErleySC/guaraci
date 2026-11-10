"""scripts/executar_docs.py: todo comando dos documentos de entrada e' classificado, e os que sao do produto rodam (dado sintetico)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import executar_docs as ed  # noqa: E402

BLOCO = """texto
```bash
pip install -e .        # instala
guaraci demo            # roda
guaraci assistente "uma pergunta"   # com aspas
$ guaraci perfis
```
```python
guaraci nao_e_shell
```
fora do bloco: guaraci doctor
"""


def test_extrai_so_linhas_de_comando_de_blocos_de_shell_e_tira_o_comentario():
    cmds = ed.extrair_comandos(BLOCO, "x.md")
    assert [c[2] for c in cmds] == ["pip install -e .", "guaraci demo", "guaraci assistente 'uma pergunta'", "guaraci perfis"]
    assert [c[1] for c in cmds] == [3, 4, 5, 6]


@pytest.mark.parametrize("cmd, acao", [
    ("guaraci", "excluir"), ("guaraci web", "excluir"), ("guaraci --perfil=generico", "excluir"), ("guaraci --modo=controle", "excluir"),
    ("pip install x", "excluir"), ("python -m venv .venv", "excluir"), ("pytest tests/", "excluir"), ("git archive HEAD", "excluir"),
    ("guaraci demo", "executar"), ("guaraci --version", "executar"), ("guaraci exemplo classificar pasta", "executar"),
    ("python -m guaraci.pipeline --rodar", "executar"),
    ("ferramenta_desconhecida --x", "nao_classificado"), ("python script_qualquer.py", "nao_classificado"),
])
def test_classificacao_dos_comandos(cmd, acao):
    assert ed.classificar(cmd, True)["acao"] == acao


def test_run_sem_config_prepara_um_exemplo_e_com_config_nao():
    assert ed.classificar("guaraci run", False)["preparo"] == [["exemplo", "classificar", "."]]
    assert ed.classificar("guaraci run", True)["preparo"] == []
    assert ed.classificar("guaraci run pasta/config.yaml", False)["preparo"] == []


def test_todo_excluido_traz_motivo():
    for cmd in ("guaraci", "guaraci web", "pip install x", "git status"):
        assert ed.classificar(cmd, False)["motivo"]


def test_todo_comando_dos_documentos_de_entrada_esta_classificado():
    """Comando novo num README, no manual ou num tutorial sem regra aqui reprova: classifique (rodar ou excluir com motivo)."""
    ruins = []
    for arq, lista in ed.comandos_dos_documentos().items():
        for _a, n, cmd in lista:
            if ed.classificar(cmd, True)["acao"] == "nao_classificado":
                ruins.append(f"{arq}:{n}: {cmd}")
    assert not ruins, ruins


def test_os_documentos_de_entrada_trazem_comandos_a_executar():
    todos = [c for lista in ed.comandos_dos_documentos().values() for c in lista]
    executaveis = [c for c in todos if ed.classificar(c[2], True)["acao"] == "executar"]
    assert len(executaveis) >= 20, len(executaveis)
    assert {"README.md", "README.pt-BR.md", "MANUAL.md"} <= set(ed.comandos_dos_documentos())


@pytest.mark.slow
@pytest.mark.parametrize("arquivo", ["README.md", "README.pt-BR.md", "MANUAL.md", "docs/usuario/roteiro_teste_usabilidade.md"])
def test_comandos_do_documento_rodam_com_dado_sintetico(arquivo):
    cmds = ed.comandos_dos_documentos([arquivo])[arquivo]
    res = ed.executar_documento(arquivo, cmds)
    ruins = [r for r in res if r["estado"] in ("falhou", "nao_classificado")]
    assert not ruins, ruins
    assert any(r["estado"] == "ok" for r in res)
