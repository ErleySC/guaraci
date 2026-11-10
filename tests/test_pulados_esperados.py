"""Testes da guarda de pulados (tests/_pulados.py)."""
import re

import pytest

from tests._pulados import (
    ARQUIVO_ESPERADOS, SEM_MOTIVO, avaliar, carregar_esperados, classificar, escolher_ambiente)

_ESP = {"dataset X ausente": {"padrao": "dataset X ausente", "max": 2}}


def test_motivo_previsto_dentro_do_teto_nao_gera_problema():
    contagem, problemas = avaliar(["dataset X ausente. Baixe..."] * 2, _ESP)
    assert contagem == {"dataset X ausente": 2}
    assert problemas == []


def test_acima_do_teto_reprova():
    _c, problemas = avaliar(["dataset X ausente"] * 3, _ESP)
    assert problemas == ["dataset X ausente: 3 pulados, teto 2"]


def test_motivo_novo_reprova_mesmo_uma_vez():
    _c, problemas = avaliar(["algo totalmente novo"], _ESP)
    assert problemas == ["motivo de pulo NAO previsto (1x): algo totalmente novo"]


def test_skip_sem_motivo_reprova():
    assert classificar("", _ESP) == SEM_MOTIVO
    assert classificar("   ", _ESP) == SEM_MOTIVO
    _c, problemas = avaliar([""], _ESP)
    assert problemas == [f"motivo de pulo NAO previsto (1x): {SEM_MOTIVO}"]


def test_nenhum_pulado_nenhum_problema():
    assert avaliar([], _ESP) == ({}, [])


def test_mistura_conta_cada_motivo_separado():
    contagem, problemas = avaliar(
        ["dataset X ausente", "novo A", "dataset X ausente", "novo A"], _ESP)
    assert contagem == {"dataset X ausente": 2, "novo A": 2}
    assert problemas == ["motivo de pulo NAO previsto (2x): novo A"]


_AMBIENTES = ("ci", "local-completo", "local-sem-dados", "sem-checkout-git")

# trechos REAIS de motivos medidos (rodadas locais, WSL e -rs) dos perfis da suite padrao
_REAIS = {
    "ci": [
        "could not import 'glasbey': No module named 'glasbey'",
        "MAX_PATH e' do Windows",
        "clone raso -- checagem de historico completo nao se aplica",
    ],
    "sem-checkout-git": ["requer um checkout git (sem .git aqui: zip/sdist)",
                         "bash utilizavel ausente: C:\\WINDOWS\\system32\\wsl.exe -d Ubuntu -- bash devolveu 1"],
    "local-sem-dados": ["bash utilizavel ausente: nem wsl nem bash no PATH"],
}


def _amostra_do_padrao(padrao: str) -> str:
    """Um texto que o padrao casa: sem ancoras, `.*` vira `x` e as barras de escape saem."""
    return re.sub(chr(92) * 2 + "(.)", chr(92) + "1", padrao.replace("^", "").replace("$", "").replace(".*", "x"))


@pytest.mark.parametrize("ambiente", _AMBIENTES)
def test_arquivo_versionado_e_valido_por_ambiente(ambiente):
    esperados = carregar_esperados(ambiente)
    for nome, regra in esperados.items():
        re.compile(regra["padrao"])
        assert isinstance(regra["max"], int) and regra["max"] >= 1, nome
        assert classificar(_amostra_do_padrao(regra["padrao"]), esperados) == nome, (ambiente, nome)
    for m in _REAIS.get(ambiente, []):
        assert classificar(m, esperados) in esperados, (ambiente, m)


def test_perfis_nao_aceitam_o_que_so_outro_perfil_aceita():
    """Sem motivo generico permissivo: nenhum perfil da suite padrao (o arquivo versionado) aceita pulo por conjunto de dados,
    e o perfil local completo nao aceita o que so' o `ci`/zip aceitam."""
    for ambiente in _AMBIENTES:
        esperados = carregar_esperados(ambiente, ARQUIVO_ESPERADOS)
        for nome, regra in esperados.items():
            assert "dataset" not in nome.lower() and "dataset" not in regra["padrao"].lower(), (ambiente, nome)
            assert classificar("dataset publico X ausente", esperados) not in esperados, (ambiente, nome)
    completo = carregar_esperados("local-completo", ARQUIVO_ESPERADOS)
    for m in _REAIS["ci"] + _REAIS["sem-checkout-git"]:
        assert classificar(m, completo) not in completo, m


def test_nenhum_perfil_aceita_pulo_sem_motivo_ou_motivo_novo():
    for ambiente in _AMBIENTES:
        esperados = carregar_esperados(ambiente)
        for texto in ("", "   ", "um motivo que ninguem previu"):
            _c, problemas = avaliar([texto], esperados)
            assert problemas, (ambiente, texto)


def test_nenhum_padrao_casa_qualquer_texto():
    for ambiente in _AMBIENTES:
        for nome, regra in carregar_esperados(ambiente).items():
            for texto in ("x", "ausente", "Skipped", "could not import 'qualquer'"):
                assert not re.search(regra["padrao"], texto), (ambiente, nome, texto)


def test_ambiente_desconhecido_e_erro_explicito():
    with pytest.raises(KeyError, match="ambiente de pulados desconhecido"):
        carregar_esperados("marte")


def test_escolha_do_ambiente():
    assert escolher_ambiente({}) == "local-sem-dados"
    assert escolher_ambiente({"GUARACI_DATASETS_DIR": "/x"}) == "local-completo"
    assert escolher_ambiente({"CI": "true"}) == "ci"
    assert escolher_ambiente({"CI": "true", "GUARACI_DATASETS_DIR": "/x"}) == "ci"
    assert escolher_ambiente({"CI": "true", "GUARACI_PULADOS_AMBIENTE": "local-completo"}) == "local-completo"
    assert escolher_ambiente({"GUARACI_PULADOS_AMBIENTE": "  ci "}) == "ci"
    assert escolher_ambiente({"GUARACI_PULADOS_AMBIENTE": "   "}) == "local-sem-dados"


def test_abaixo_do_teto_nao_reprova():
    """Teto e' MAXIMO, nao contagem exata (mutante `>` -> `!=` sobrevivia):
    menos pulados (ex.: dataset baixado) nunca reprova."""
    _c, problemas = avaliar(["dataset X ausente"], _ESP)
    assert problemas == []


def test_sem_o_arquivo_da_validacao_os_perfis_da_suite_padrao_continuam_valendo(monkeypatch, tmp_path):
    """Sem `validacao/` (o normal: ela fica fora do repositorio), todos os perfis da suite padrao carregam do arquivo versionado
    e um perfil desconhecido vira erro explicito (mata o mutante que trata arquivo ausente como se existisse)."""
    import tests._pulados as pul
    monkeypatch.setattr(pul, "ARQUIVO_ESPERADOS_VALIDACAO", tmp_path / "nao_existe.json")
    for ambiente in _AMBIENTES:
        assert pul.carregar_esperados(ambiente) == pul.carregar_esperados(ambiente, pul.ARQUIVO_ESPERADOS)
    with pytest.raises(KeyError) as e:
        pul.carregar_esperados("so-da-validacao")
    assert "so-da-validacao" in str(e.value)


def test_validacao_local_do_mantenedor_tem_precedencia_no_mesmo_perfil(monkeypatch, tmp_path):
    """Quando o mantenedor poe a validacao local em `validacao/`, o perfil dela (que conhece os conjuntos) vale no lugar do da
    suite padrao; perfis que ela nao define continuam vindo do arquivo versionado."""
    import json

    import tests._pulados as pul
    arq = tmp_path / "pulados_esperados.json"
    arq.write_text(json.dumps({"ambientes": {"local-sem-dados": {"motivos": {
        "conjunto X ausente": {"padrao": "^conjunto X ausente$", "max": 2}}}}}), encoding="utf-8")
    monkeypatch.setattr(pul, "ARQUIVO_ESPERADOS_VALIDACAO", arq)
    assert list(pul.carregar_esperados("local-sem-dados")) == ["conjunto X ausente"]
    assert pul.carregar_esperados("ci") == pul.carregar_esperados("ci", pul.ARQUIVO_ESPERADOS)
