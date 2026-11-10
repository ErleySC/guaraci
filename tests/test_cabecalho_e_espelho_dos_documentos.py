"""Padrao dos documentos: cabecalho (publico, estado, atualizado em) e README em PT e EN espelhados."""
from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import executar_docs as ed  # noqa: E402
import padronizar_cabecalhos as pc  # noqa: E402


def test_todo_documento_versionado_tem_cabecalho():
    faltam = [d for d, _ in pc.sem_cabecalho(RAIZ)]
    assert not faltam, f"sem cabecalho (rode scripts/padronizar_cabecalhos.py): {faltam}"


def test_cabecalho_e_inserido_logo_depois_do_titulo_e_so_uma_vez():
    t = "# Titulo\n\ncorpo\n"
    novo = pc.inserir(t, "docs/cientifico/x.md")
    assert novo is not None and novo.startswith("# Titulo\n\n**Público:** quem avalia os métodos e os números · **Estado:** manual")
    assert "corpo" in novo and pc.inserir(novo, "docs/cientifico/x.md") is None
    assert pc.tem_cabecalho(novo) and not pc.tem_cabecalho(t)


def test_documento_gerado_diz_que_e_gerado_e_sem_titulo_nao_e_alterado():
    assert "gerado por script" in pc.cabecalho_padrao("docs/qualidade/x.md")
    assert "manual" in pc.cabecalho_padrao("docs/usuario/x.md")
    assert pc.inserir("sem titulo\n", "docs/usuario/x.md") is None


def _comandos(readme: str) -> list:
    """Comandos do README com o texto entre aspas neutralizado (a pergunta de exemplo muda de idioma)."""
    return [re.sub(r"'[^']*'", "'...'", c) for _a, _n, c in ed.extrair_comandos((RAIZ / readme).read_text(encoding="utf-8"), readme)]


def test_readmes_em_pt_e_en_tem_os_mesmos_comandos_e_a_mesma_estrutura():
    assert _comandos("README.md") == _comandos("README.pt-BR.md")
    h2 = lambda f: len(re.findall(r"^## ", (RAIZ / f).read_text(encoding="utf-8"), re.M))  # noqa: E731
    assert h2("README.md") == h2("README.pt-BR.md")
    en, pt = (RAIZ / "README.md").read_text(encoding="utf-8"), (RAIZ / "README.pt-BR.md").read_text(encoding="utf-8")
    assert re.search(r"8 shipped", en) and re.search(r"8 distribuídos", pt)
    assert "README.pt-BR.md" in en and "README.md" in pt        # cada um aponta para o espelho
