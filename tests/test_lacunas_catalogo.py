"""Catalogo (`guaraci.catalogo`): o validador aponta cada tipo de defeito que o contrato proibe.

O catalogo real e' consistente (a lista de problemas vem vazia); aqui cada teste estraga UMA copia do catalogo de um jeito e confere que o
validador diz qual foi o defeito. Sem isso, um validador que nunca reclama passaria despercebido.
"""
from __future__ import annotations

import copy

import pytest

import guaraci.catalogo as cat


def _estragar(monkeypatch, nome, mutar):
    copia = copy.deepcopy(getattr(cat, nome))
    mutar(copia)
    monkeypatch.setattr(cat, nome, copia)
    return cat.validar_catalogo()


def _primeiro(d):
    return next(iter(d))


def test_o_catalogo_real_nao_tem_problema():
    assert cat.validar_catalogo() == []


def test_grupos_repetidos(monkeypatch):
    monkeypatch.setattr(cat, "GRUPOS", cat.GRUPOS + cat.GRUPOS[:1])
    assert "ids de grupo repetidos" in cat.validar_catalogo()


def test_telas_com_defeitos(monkeypatch):
    def mutar(t):
        a, b = list(t)[:2]
        t[a]["grupo"] = "grupo_que_nao_existe"
        t[b]["tecla"] = t[a]["tecla"]
        outra = list(t)[2]
        t[outra]["tecla"] = "M"
        t[_primeiro(t)]["faz"] = ("", t[_primeiro(t)]["faz"][1])
    p = _estragar(monkeypatch, "TELAS", mutar)
    texto = "\n".join(p)
    for trecho in ("grupo inexistente", "repetida em", "colide com acao do menu", "campo 'faz' vazio"):
        assert trecho in texto, trecho


def test_comandos_opcoes_tecnicas_e_graficos_com_defeitos(monkeypatch):
    def comando(c):
        c[_primeiro(c)]["grupo"] = "zz"
    assert any("comando" in x and "grupo inexistente" in x for x in _estragar(monkeypatch, "COMANDOS", comando))

    def opcao(o):
        k = _primeiro(o)
        o[k]["tela"] = "tela_que_nao_existe"
        o[k]["nivel"] = "intermediario"
        o[k]["impacto"] = ("talvez ajude", "ok")
        o[k]["faz"] = ("x" * 400, o[k]["faz"][1])
        o[k]["use"] = "nao e um par"
    texto = "\n".join(_estragar(monkeypatch, "OPCOES", opcao))
    for trecho in ("tela inexistente", "nivel invalido", "impacto: texto proibido", "limite", "nao e um par"):
        assert trecho in texto, trecho

    def tecnica(t):
        k = _primeiro(t)
        t[k]["grupo"] = "zz"
        t[k]["selo"] = "selo_que_nao_existe"
    texto = "\n".join(_estragar(monkeypatch, "TECNICAS", tecnica))
    assert "tecnica" in texto and "selo inexistente" in texto


def test_selos_com_defeitos(monkeypatch):
    def mutar(s):
        ks = list(s)
        s[ks[0]]["estado"] = "estado_que_nao_existe"
        s[ks[1]]["evid"] = ()
        s[ks[2]]["evid"] = ("evidencia_que_nao_existe",)
        s[ks[3]]["estado"] = "validado"
        s[ks[3]]["evid"] = tuple(e for e in s[ks[3]]["evid"] if cat.EVIDENCIAS[e]["tipo"] != "dataset_publico") or ()
        s[ks[4]]["nota"] = ("so em portugues", "")
    texto = "\n".join(_estragar(monkeypatch, "SELOS", mutar))
    for trecho in ("estado invalido", "sem evidencia", "evidencia inexistente", "idiomas desiguais"):
        assert trecho in texto, trecho


def test_rotas_com_defeitos(monkeypatch):
    def mutar(r):
        ks = list(r)
        r[ks[0]]["config"] = {"opcao_que_nao_existe": 1}
        r[ks[0]]["tecnicas"] = ("tecnica_que_nao_existe",)
        r[ks[0]]["exemplo"] = "exemplo_que_nao_existe"
        r[ks[0]]["comando"] = "outro programa"
        r[ks[0]]["passos"] = (("cmd", "outro comando qualquer", "x"), ("tela_que_nao_existe", "a", "b"), ("modelo", "so pt", ""),
                              ("modelo", "talvez", "perhaps"))
        r[(("objetivo_que_nao_existe"), "dado_que_nao_existe")] = copy.deepcopy(r[ks[1]])
    texto = "\n".join(_estragar(monkeypatch, "ROTAS", mutar))
    for trecho in ("objetivo ou dado inexistente", "opcao inexistente", "tecnica inexistente", "exemplo inexistente", "comando fora do catalogo",
                   "tela inexistente", "passo sem os dois idiomas", "texto proibido"):
        assert trecho in texto, trecho


def test_duvidas_com_defeitos(monkeypatch):
    def mutar(d):
        ks = list(d)
        d[ks[0]]["gatilhos"] = ()
        d[ks[1]]["onde"] = ()
        d[ks[2]]["onde"] = (("manual", "abc"),)
        d[ks[3]]["onde"] = (("tela", "tela_que_nao_existe"),)
    texto = "\n".join(_estragar(monkeypatch, "DUVIDAS", mutar))
    for trecho in ("gatilhos vazios", "sem 'onde ver na CLI'", "secao do manual invalida", "'onde' inexistente"):
        assert trecho in texto, trecho


def test_evidencias_instrumentos_e_selos_de_opcao_com_defeitos(monkeypatch):
    def evidencia(e):
        k = _primeiro(e)
        e[k]["rotulo"] = ("", "x")
        e[k]["ressalva"] = ("so pt", "")
    texto = "\n".join(_estragar(monkeypatch, "EVIDENCIAS", evidencia))
    assert "nao e um par (PT, EN) preenchido" in texto and "idiomas desiguais" in texto

    def instrumento(i):
        k = _primeiro(i)
        i[k]["nome"] = ("", "")
        i[k]["faz"] = ("talvez", "perhaps")
        i[k]["preproc"] = "preset_que_nao_existe"
    texto = "\n".join(_estragar(monkeypatch, "INSTRUMENTOS", instrumento))
    assert "nao e um preset do motor" in texto and "texto proibido" in texto

    def opcao_selo(o):
        o["opcao_que_nao_existe"] = "selo_que_nao_existe"
    assert any("referencia inexistente" in x for x in _estragar(monkeypatch, "OPCAO_SELO", opcao_selo))


def test_consultas_do_catalogo():
    assert cat.grupo("comecar")["id"] == "comecar"
    with pytest.raises(KeyError):
        cat.grupo("grupo_que_nao_existe")
    selo = cat.selo_de(next(iter(cat.SELOS)))
    assert selo["estado"] in cat.ESTADOS
    assert cat.simbolo_selo("validado") == "●" and cat.simbolo_selo("validado", ascii_only=True) == "[V]"
    assert cat._selo_da_entrada("tecnica", "tecnica_que_nao_existe") is None
    assert cat._selo_da_entrada("outro_tipo", "x") is None
    opcao = next(iter(cat.OPCAO_SELO))
    assert cat._selo_da_entrada("opcao", opcao) == cat.OPCAO_SELO[opcao]


def test_confianca_de_opcao_com_selo_traz_o_limite_de_uso_e_sem_selo_explica():
    com_selo = cat.confianca("opcao", "ddsimca", "PT")
    assert com_selo["selo"] and cat.OPCOES["ddsimca"]["nao_use"][0] in com_selo["limites"]
    sem_selo = cat.confianca("opcao", "pasta_dados", "PT")
    assert sem_selo["selo"] is None and "sem selo proprio" in sem_selo["texto"]


def test_pares_e_textos_malformados_sao_apontados():
    problemas = []
    cat._checar_par(problemas, "x", "faz", "texto solto")
    cat._checar_texto(problemas, "y", "faz", "   ")
    cat._checar_texto(problemas, "z", "faz", "a" * 500)
    assert len(problemas) == 3
