"""Documentos vivos só citam arquivos versionados, e o índice do projeto lista todo documento de docs/."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import verificar_links_docs as vl  # noqa: E402

PASTAS_DE_DOCS = ("usuario", "cientifico", "contribuir", "projeto", "qualidade", "api")


def _versionados() -> set[str]:
    return vl.versionados(RAIZ)             # sem .git o conftest pula este arquivo inteiro (checkout git exigido)


def test_documentos_vivos_so_citam_arquivos_versionados():
    problemas = vl.quebrados_na_arvore(RAIZ, _versionados())
    assert not problemas, problemas


def test_o_detector_acha_um_caminho_inexistente_e_aceita_o_que_existe():
    v = {"MANUAL.md", "scripts/estado_do_projeto.py"}
    assert vl.caminhos_quebrados("veja `docs/FANTASMA.md` e `MANUAL.md`", "README.md", v) == ["docs/FANTASMA.md"]
    assert vl.caminhos_quebrados("[x](MANUAL.md) [y](docs/NAO_EXISTE.md) [z](https://exemplo.org/a.md)", "README.md", v) == ["docs/NAO_EXISTE.md"]
    assert vl.caminhos_quebrados("`scripts/estado_do_projeto.py`, `docs/SECURITY.md`, `tests/test_` e `docs/qualidade/X_<data>.md`", "README.md", v) == []


def test_o_detector_resolve_link_relativo_com_subida_de_pasta():
    v = {"docs/cientifico/a.md", "validacao/medicoes/b.py"}
    assert vl.caminhos_quebrados("[b](../../validacao/medicoes/b.py)", "docs/cientifico/a.md", v) == []
    assert vl.caminhos_quebrados("[b](../validacao/medicoes/b.py)", "docs/cientifico/a.md", v) == ["docs/validacao/medicoes/b.py"]


def test_indice_do_projeto_lista_todo_documento_de_docs():
    v = _versionados()
    indice = (RAIZ / "docs/projeto/INDICE.md").read_text(encoding="utf-8")
    faltam = [f for f in sorted(v) if f.startswith("docs/") and f.endswith((".md", ".yaml", ".json")) and f not in indice
              and not (f.startswith("docs/qualidade/resultado_") and f.endswith(".json"))
              and f"({'../' + f[len('docs/'):]})" not in indice]
    assert not faltam, f"documentos de docs/ que o INDICE nao cita: {faltam}"


def test_indice_gerado_esta_em_dia():
    r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "gerar_indice_docs.py"), "--verificar"], cwd=RAIZ, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_tabela_de_referencias_esta_em_dia():
    # antes so' a etapa `referencias` do verificador final via a tabela desatualizada (e reprovava apos ~3 h)
    r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "gerar_tabela_referencias.py"), "--verificar"], cwd=RAIZ,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr


def test_so_docs_de_destino_conhecido_sao_versionados():
    """docs/ so' tem as pastas publicas (e docs/qualidade/ so' com saida gerada); `docs/auditoria/` e' local e fica fora do git."""
    fora = []
    for f in sorted(_versionados()):
        if not f.startswith("docs/"):
            continue
        partes = f.split("/")
        if len(partes) == 2 and partes[1] in ("index.md", "POLITICA_DE_DOCUMENTACAO.md"):
            continue
        if len(partes) >= 3 and partes[1] in PASTAS_DE_DOCS:
            continue
        fora.append(f)
    assert not fora, f"arquivo versionado fora das pastas publicas de docs/: {fora}"


def test_so_o_mapa_de_migracao_pode_citar_caminho_antigo():
    assert vl.DOCS_QUE_CITAM_CAMINHO_ANTIGO == {"docs/projeto/MAPA_DE_MIGRACAO.md"}


def test_verificador_falha_quando_nao_acha_documento_nenhum(tmp_path, monkeypatch, capsys):
    """Sem git (ou fora de um clone) a varredura nao via nada e o relatorio dizia "0 links quebrados": verde falso."""
    with pytest.raises(vl.SemDocumentos):
        vl.quebrados_na_arvore(tmp_path, set())
    assert vl.versionados(tmp_path) == set()                       # pasta sem repositorio git
    monkeypatch.setattr(vl, "versionados", lambda raiz=RAIZ: set())
    assert vl.main([]) == 2
    assert vl.main(["--json"]) == 2
    assert "nenhum documento" in capsys.readouterr().err


def test_verificador_diz_quantos_documentos_olhou(capsys):
    assert vl.main([]) == 0
    saida = capsys.readouterr().out
    assert "documento(s) verificado(s)" in saida and not saida.startswith("0 ")
