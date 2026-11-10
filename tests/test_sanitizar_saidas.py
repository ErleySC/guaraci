"""scripts/sanitizar_saidas.py: troca caminho de maquina por marcador neutro
em saidas de reproducao, sem lista de nomes de usuario no codigo."""
import re
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import sanitizar_saidas as ss  # noqa: E402

BS = chr(92)


def _win(usuario: str, resto: str = "AppData" + BS + "x.txt") -> str:
    return "C:" + BS + "Users" + BS + usuario + BS + resto


def test_qualquer_usuario_vira_sentinela():
    for usuario in ("fulano", "Maria.Silva", "x"):
        assert ss.sanitizar_texto("log: " + _win(usuario)) == "log: " + _win("alguem")
        barra = "C:/Users/" + usuario + "/tmp"
        assert ss.sanitizar_texto(barra) == "C:/Users/alguem/tmp"


def test_sentinela_nao_e_alterada_e_prefixo_dela_e():
    assert ss.sanitizar_texto(_win("alguem")) == _win("alguem")
    assert ss.sanitizar_texto(_win("alguem2")) == _win("alguem")


def test_forma_achatada_vira_marcador():
    texto = "tmp/claude/C" + "--" + "Users-fulano-One" + "Drive-Doc-code-py/sessao/x"
    assert ss.sanitizar_texto(texto) == "tmp/claude/<pasta-temporaria-do-projeto>/sessao/x"


def test_texto_sem_caminho_fica_igual():
    for t in ("RMSEP 0,144", "opcao --Users-foo", "Users/fulano sem unidade"):
        assert ss.sanitizar_texto(t) == t


def test_outra_unidade_separador_repetido_e_posix():
    assert ss.sanitizar_texto("D:" + BS + BS + "Users" + BS + BS + "fulano" + BS + BS + "x") == \
        "D:" + BS + BS + "Users" + BS + BS + "alguem" + BS + BS + "x"
    for base in ("/home/", "/Users/"):
        assert ss.sanitizar_texto("em " + base + "Maria.Silva/dados") == "em " + base + "alguem/dados"
    assert ss.sanitizar_texto("/home/alguem/x") == "/home/alguem/x"


def test_caminho_em_pasta_sincronizada_sai_sem_o_usuario():
    """Caminho tipico de projeto em pasta sincronizada (usuario sintetico): nada do usuario sobra."""
    t = "Loading CSV: " + _win("beltrano", "One" + "Drive" + BS + "Documentos" + BS + "proj" + BS + "a.csv")
    assert "beltrano" not in ss.sanitizar_texto(t) and "One" + "Drive" in ss.sanitizar_texto(t)


def test_padroes_locais_ausente_silencioso_malformado_erro_claro(tmp_path):
    assert ss.carregar_padroes_locais(tmp_path / "nao_existe.txt") == []
    ok = tmp_path / "padroes.txt"
    ok.write_text("# comentario\nInstituto Ficticio\nGrupo-[0-9]+\n", encoding="utf-8")
    padroes = ss.carregar_padroes_locais(ok)
    assert ss.sanitizar_texto("pasta Instituto Ficticio/Grupo-42/x", padroes) == "pasta <pessoal>/<pessoal>/x"
    ruim = tmp_path / "ruim.txt"
    ruim.write_text("ok\n(sem fechar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="linha 2"):
        ss.carregar_padroes_locais(ruim)
    binario = tmp_path / "bin.txt"
    binario.write_bytes(b"\xff\xfe\x00x")
    with pytest.raises(ValueError, match="UTF-8"):
        ss.carregar_padroes_locais(binario)


@pytest.mark.parametrize("regra, entrada", [
    ("_RX_WIN", "C:" + BS + "Users" + BS + "fulano" + BS + "x"),
    ("_RX_POSIX", "/" + "home" + "/fulano/x"),
    ("_RX_ACHATADA", "C" + "--Users-fulano-proj/x"),
])
def test_contraprova_sem_a_regra_o_usuario_vaza(monkeypatch, regra, entrada):
    """Contraprova por copia: com a regra trocada por uma que nunca casa, o usuario
    sintetico sobrevive -- cada regra e' necessaria para o teste acima passar."""
    assert "fulano" not in ss.sanitizar_texto(entrada)
    monkeypatch.setattr(ss, regra, re.compile(r"(?!x)x"))
    assert "fulano" in ss.sanitizar_texto(entrada)


def test_main_reescreve_so_arquivos_de_texto(tmp_path, capsys):
    (tmp_path / "a.txt").write_text(_win("fulano"), encoding="utf-8")
    (tmp_path / "b.bin").write_text(_win("fulano"), encoding="utf-8")
    ss.main([str(tmp_path)], padroes_locais=[])   # nunca le o arquivo pessoal real no teste
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == _win("alguem")
    assert (tmp_path / "b.bin").read_text(encoding="utf-8") == _win("fulano")
    assert "1 arquivo(s) sanitizado(s)" in capsys.readouterr().out
