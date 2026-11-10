"""Referencias da tela Sobre (6.3): todo DOI no registro conferido e toda tecnica coberta."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml

from guaraci import catalogo_referencias as cr
from guaraci.catalogo_tecnicas import TECNICAS

RAIZ = Path(__file__).resolve().parents[1]
REGISTRO = RAIZ / "docs" / "cientifico" / "referencias_verificadas.yaml"


def _norm(s: str) -> str:
    s = str(s).replace("ø", "o").replace("Ø", "O")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    toks = re.sub(r"[^a-z0-9]+", " ", s).split()
    return " ".join(t for t in toks if len(t) > 1)


def _registro():
    refs = yaml.safe_load(REGISTRO.read_text(encoding="utf-8"))["referencias"]
    return {r["doi"].lower(): r for r in refs if r.get("conferido") and not r.get("retirado")}


def test_todo_doi_esta_no_registro_com_autor_ano_e_titulo():
    reg = _registro()
    erros = []
    for k, r in cr.REFS.items():
        e = reg.get(r.doi.lower())
        if e is None:
            erros.append(f"{k}: DOI fora do registro conferido ({r.doi})")
            continue
        if _norm(e["autor1"]) not in _norm(r.autores):
            erros.append(f"{k}: primeiro autor {e['autor1']!r} != {r.autores!r}")
        if int(e["ano"]) != r.ano:
            erros.append(f"{k}: ano {e['ano']} != {r.ano}")
        if _norm(e["titulo"]) not in _norm(r.titulo):
            erros.append(f"{k}: titulo nao confere com o registro ({e['titulo']!r})")
    assert not erros, "\n".join(erros)


def test_toda_tecnica_do_catalogo_aparece_nas_referencias():
    cobertas = {it.tecnica for _t, _p, _e, itens in cr.TEMAS for it in itens if it.tecnica}
    faltam = sorted(set(TECNICAS) - cobertas)
    assert not faltam, f"tecnicas sem entrada em catalogo_referencias.TEMAS: {faltam}"
    desconhecidas = sorted(cobertas - set(TECNICAS))
    assert not desconhecidas, desconhecidas


def test_chaves_existem_e_nenhuma_ref_sobra():
    usadas = set(cr.chaves_usadas())
    assert usadas <= set(cr.REFS), sorted(usadas - set(cr.REFS))
    assert set(cr.REFS) <= usadas, f"referencias sem uso: {sorted(set(cr.REFS) - usadas)}"


def test_nenhum_metodo_de_terceiros_fica_sem_referencia():
    assert cr.lacunas() == [], f"metodos sem fonte conferida: {cr.lacunas()}"
    for _t, _p, _e, itens in cr.TEMAS:
        for it in itens:
            if not it.refs:
                assert it.proprio and it.nota.strip(), f"{it.metodo}: sem referencia e sem dizer que e' do GUARACI"


def test_painel_referencias_mostra_todas(monkeypatch):
    from rich.console import Console
    from guaraci import guaraci as g

    con = Console(width=100, record=True, force_terminal=False, color_system=None)
    monkeypatch.setattr(g, "console", con)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    entradas = iter(["T", "V"])
    monkeypatch.setattr(g, "_input", lambda *a, **k: next(entradas))
    g._referencias_tecnicas("PT")
    def _cola(x: str) -> str:   # DOI longo quebra em linhas: compara sem espacos/bordas
        return re.sub(r"[^a-z0-9]", "", x.lower())
    bruto = con.export_text()
    txt = _cola(bruto)
    for k in cr.REFS:
        assert _cola(cr.REFS[k].doi) in txt, k
    palavras = set(_norm(bruto).split())
    assert "lacuna" not in bruto.lower()
    sem_ref = [it.metodo for _t, _p, _e, itens in cr.TEMAS for it in itens if not it.refs]
    for metodo in sem_ref:   # nome pode quebrar na coluna estreita
        assert set(_norm(metodo).split()) <= palavras, metodo
