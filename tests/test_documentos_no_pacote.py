"""Tudo que a CLI, o assistente e a ajuda citam existe DENTRO do pacote instalado (sem ponteiro para `docs/` do repositorio)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import gerar_docs_pacote as gdp  # noqa: E402
import referencias_docs as rd  # noqa: E402

from guaraci import cli_comandos as cmd  # noqa: E402
from guaraci import docs_pacote as dp  # noqa: E402
from guaraci.catalogo_duvidas import DUVIDAS  # noqa: E402


def test_nenhuma_referencia_de_execucao_aponta_para_fora_do_pacote():
    assert rd.quebradas() == []


def test_copias_do_pacote_estao_em_dia_com_as_fontes():
    assert gdp.defasados() == []


def test_topicos_e_fontes_do_gerador_sao_os_mesmos_arquivos():
    assert {arq for arq, _ in dp.TOPICOS.values()} == set(gdp.FONTES)


def test_o_gerador_aponta_defasagem_e_regrava(tmp_path, monkeypatch):
    monkeypatch.setattr(gdp, "DESTINO", tmp_path)
    assert sorted(gdp.defasados()) == sorted(gdp.FONTES)           # pasta vazia: tudo ausente
    assert sorted(gdp.gravar()) == sorted(gdp.FONTES)
    assert gdp.defasados() == []
    (tmp_path / "manual.md").write_text("alterado", encoding="utf-8")
    assert gdp.defasados() == ["manual.md"]
    assert gdp.main(["--verificar"]) == 1
    assert gdp.main([]) == 0 and gdp.main(["--verificar"]) == 0


def test_o_detector_separa_docstring_de_texto_de_execucao(tmp_path):
    f = tmp_path / "m.py"
    f.write_text('"""ver MANUAL.md"""\n\ndef g():\n    """ver docs/x.md"""\n    return "veja docs/y.md"\n', encoding="utf-8")
    achados = [t for _, t in rd._strings_de_execucao(f)]
    assert achados == ["veja docs/y.md"]


def test_referencia_a_arquivo_do_repositorio_em_texto_de_execucao_e_apontada(tmp_path, monkeypatch):
    pacote = tmp_path / "src" / "guaraci"
    pacote.mkdir(parents=True)
    (pacote / "m.py").write_text('MSG = "ver docs/cientifico/limitacoes.md e guaraci manual limitacoes e guaraci manual 99"\n', encoding="utf-8")
    monkeypatch.setattr(rd, "RAIZ", tmp_path)
    quebradas = {a for _, a, _, _ in rd.quebradas()}
    assert quebradas == {"docs/cientifico/limitacoes.md", "guaraci manual 99"}


@pytest.mark.parametrize("topico", list(dp.TOPICOS))
def test_cada_topico_tem_texto_nao_vazio(topico):
    assert len(dp.ler(topico)) > 500


def test_topico_inexistente_levanta_keyerror():
    with pytest.raises(KeyError):
        dp.ler("nao_existe")


def test_secao_vai_do_titulo_ao_proximo_titulo_de_mesmo_nivel_ou_maior():
    texto = dp.secao_do_manual("5.8")
    assert texto.startswith("### 5.8 ")
    assert "### 5.9" not in texto
    assert dp.secao_do_manual("9").startswith("## 9 ")
    assert dp.secao_do_manual("99") is None


def test_toda_secao_do_manual_citada_pelo_catalogo_de_duvidas_existe():
    citadas = {ident for d in DUVIDAS.values() for tipo, ident in d["onde"] if tipo == "manual"}
    assert citadas
    assert [s for s in citadas if dp.secao_do_manual(s) is None] == []


def test_manual_sem_argumento_lista_os_topicos(capsys):
    assert cmd.subcomando("manual", []) == 0
    saida = capsys.readouterr().out
    assert all(t in saida for t in dp.TOPICOS)


def test_manual_com_topico_e_com_secao_imprime_o_texto(capsys):
    assert cmd.subcomando("manual", ["limitacoes"]) == 0
    assert "Limitações conhecidas" in capsys.readouterr().out
    assert cmd.subcomando("manual", ["5.8"]) == 0
    assert "### 5.8 " in capsys.readouterr().out


def test_manual_com_alvo_inexistente_da_erro_de_uso(capsys):
    assert cmd.subcomando("manual", ["zzz"]) == 2
    assert "guaraci manual" in capsys.readouterr().err


def test_tabela_de_validacao_nao_vai_mais_no_pacote():
    assert "validacao" not in dp.TOPICOS and "benchmark" not in dp.TOPICOS
