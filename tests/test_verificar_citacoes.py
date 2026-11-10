"""Guarda permanente de citacoes (B6/m-01 do parecer de 2026-09-25).

Parte OFFLINE (roda sempre): todo DOI do repositorio esta no registro
`docs/cientifico/referencias_verificadas.yaml`, e nenhum DOI conhecido como incorreto
aparece em codigo ou texto de interface. A conferencia contra o Crossref/
DataCite e' feita por `python scripts/verificar_citacoes.py` (CI, com rede);
aqui so' a logica de comparacao e' testada, com metadados simulados.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
PREFIXO_DEPOSITO = "10.5281/" "zeno" "do"     # partido: o extrator de DOI do repositorio nao toma o prefixo por citacao


def _modulo():
    spec = importlib.util.spec_from_file_location(
        "verificar_citacoes", _RAIZ / "scripts" / "verificar_citacoes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verificar_citacoes"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_todo_doi_do_repositorio_esta_no_registro():
    vc = _modulo()
    registro, incorretos = vc.carregar_registro()
    achados = vc.extrair_dois()
    fora = sorted(d for d in achados if d not in registro and d not in incorretos)
    assert not fora, (
        "DOI(s) fora de docs/cientifico/referencias_verificadas.yaml (rode "
        f"`python scripts/verificar_citacoes.py --sugerir`): {fora}")


def test_doi_incorreto_conhecido_nao_aparece_em_codigo_nem_interface():
    vc = _modulo()
    _registro, incorretos = vc.carregar_registro()
    achados = vc.extrair_dois()
    proibidos = [f"{d} em {arq}:{lin}" for d in incorretos
                 for arq, lin, _t in achados.get(d, [])
                 if not arq.startswith("docs/") or arq.startswith("docs/MANUAL")]
    assert not proibidos, proibidos


def test_a_referencia_sherazi_inexistente_nao_esta_mais_na_interface():
    from guaraci.cli_assistente import REFERENCIAS_GUARACI
    texto = " ".join(r["cit"] for r in REFERENCIAS_GUARACI.values())
    assert "Sherazi" not in texto and "135181" not in texto


def test_extracao_de_doi_com_parenteses_e_sici():
    vc = _modulo()
    m = vc._RX_DOI.search("doi:10.1016/S0169-7439(00)00122-2 e mais")
    assert vc._limpar_doi(m.group(0)) == "10.1016/S0169-7439(00)00122-2"
    m = vc._RX_DOI.search("(veja 10.1080/00224065.1992.12015232).")
    assert vc._limpar_doi(m.group(0)) == "10.1080/00224065.1992.12015232"
    m = vc._RX_DOI.search("DOI: 10.1002/(SICI)1099-128X(199601)10:1<47::AID-CEM400>3.0.CO;2-C.")
    limpo = vc._limpar_doi(m.group(0))
    assert "1099-128X(199601)10:1<47" in limpo and limpo.startswith("10.1002/")


def test_comparacao_detecta_autor_ano_e_titulo_divergentes():
    vc = _modulo()
    meta = {"titulo": "Sensor surface chemistry for SPR", "autores": ["Altintas"],
            "ano": 2012}
    esperado = {"autor1": "Zwanenburg", "ano": 2005,
                "titulo": "ANOVA principal component analysis"}
    erros = vc.comparar("10.x/y", esperado, meta)
    assert len(erros) == 3
    assert vc.comparar("10.x/y", {"autor1": "Altintas", "ano": 2012}, meta) == []
    assert vc.comparar("10.x/y", esperado, None)[0].endswith("registro oficial")


def test_verificador_falha_alto_sem_rede(monkeypatch, capsys):
    """Sem rede a verificacao NAO passa em silencio: sai com codigo 2."""
    vc = _modulo()

    def _sem_rede(*_a, **_k):
        raise ConnectionError("sem rede simulada")
    monkeypatch.setattr(vc, "_get_json", _sem_rede)
    monkeypatch.setattr(vc, "_carregar_cache", lambda: {})
    monkeypatch.setattr(vc, "_salvar_cache", lambda c: None)
    assert vc.main([]) == 2


def test_resolucao_http_distingue_referencia_viva_da_retirada():
    """`--rede` (etapa com rede, desligada por padrao); aqui so' a decisao, com codigos simulados e sem rede."""
    vc = _modulo()
    assert vc.avaliar_resolucao("10.1/x", {}, 200) is None
    assert vc.avaliar_resolucao("10.1/x", {}, 302) is None
    assert "nao resolve" in vc.avaliar_resolucao("10.1/x", {}, 404)
    assert vc.avaliar_resolucao("10.1/x", {"retirado": True}, 410) is None
    assert "retirado, mas resolve" in vc.avaliar_resolucao("10.1/x", {"retirado": True}, 200)
    assert "sem resposta" in vc.avaliar_resolucao("10.1/x", {}, None)


def test_o_deposito_antigo_do_proprio_software_nao_e_referencia():
    vc = _modulo()
    registro, _ = vc.carregar_registro()
    antigo = PREFIXO_DEPOSITO + ".2131" "1867"
    assert not [e for e in registro.values() if str(e["doi"]) == antigo]
    tabela = (_RAIZ / "docs" / "cientifico" / "referencias.md").read_text(encoding="utf-8")
    assert antigo not in tabela


def test_a_flag_rede_existe_e_nao_e_padrao():
    import inspect
    vc = _modulo()
    fonte = inspect.getsource(vc.main)
    assert '"--rede"' in fonte and "args.rede" in fonte
