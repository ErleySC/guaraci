# -*- coding: utf-8 -*-
"""scripts/checar_design.py: as funcoes puras (comparacao de instantaneos) e o codigo de saida quando o nucleo muda."""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import checar_design as cd  # noqa: E402


def _png(w: int, h: int) -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_normalizar_tira_caminho_usuario_fim_de_linha_e_tempo():
    t = "pasta C:\\Users\\alguem\\x\\y  \r\nlevou 12.5s e 300 ms\n"
    n = cd.normalizar(t, raizes=["C:\\Users\\alguem\\x"])
    assert "alguem" not in n and "<PASTA>" in n and "<TEMPO>" in n and "\r" not in n
    assert cd.normalizar("/home/alguem/abc") == "<USUARIO>/abc\n"
    assert cd.normalizar("igual") == cd.normalizar("igual   ")
    assert cd.normalizar("pasta HSI_20261014_004031") == "pasta HSI_<CARIMBO>\n"      # carimbo de data e hora das corridas
    assert cd.normalizar("RAM total: 15.8 GB  livre: 7.0 GB  CPU: 8f/16l") == "RAM total: <GB>  livre: <GB>  CPU: <N>\n"


def test_comparar_textos_lista_novo_removido_e_mudado_sem_imprimir_o_texto():
    base = {"a": "um\ndois\n", "b": "x\n", "c": "y\n"}
    atual = {"a": "um\ntres\nquatro\n", "b": "x\n", "d": "z\n"}
    dif = cd.comparar_textos(base, atual)
    assert "texto mudou: a (+2 linha(s), -1 linha(s))" in dif
    assert "texto removido: c" in dif and "texto novo: d" in dif
    assert not any("tres" in d for d in dif)
    assert cd.comparar_textos(base, base) == []


def test_figuras_png_le_dimensao_e_comparar_figuras_acusa_mudanca(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.png").write_bytes(_png(4, 3))
    (tmp_path / "sub" / "b.png").write_bytes(_png(5, 5))
    f1 = cd.figuras_png(tmp_path)
    assert f1["a.png"]["dim"] == [4, 3] and set(f1) == {"a.png", "sub/b.png"}
    (tmp_path / "a.png").write_bytes(_png(8, 3))
    (tmp_path / "c.png").write_bytes(_png(1, 1))
    f2 = cd.figuras_png(tmp_path)
    dif = cd.comparar_figuras(f1, f2)
    assert "figura mudou: a.png, tamanho [4, 3] -> [8, 3]" in dif and "figura nova: c.png" in dif
    assert cd.comparar_figuras(f1, f1) == []


def test_comparar_tempos_so_acusa_lentidao_relevante():
    assert cd.comparar_tempos({"x": 1.0}, {"x": 1.2}) == []
    assert cd.comparar_tempos({"x": 0.1}, {"x": 0.3}) == []          # mais que 1,5x, mas so' 0,2 s de diferenca
    assert cd.comparar_tempos({"x": 1.0}, {"x": 2.0}) == ["mais lento: x 1.00s -> 2.00s"]


def test_base_grava_e_le_de_volta_igual(tmp_path):
    inst = {"textos": {"--help": "ajuda\n", "anova --help": "anova\n"}, "figuras": {"a.png": {"sha": "1", "bytes": 2, "dim": [1, 1]}},
            "tempos": {"abrir": 0.5}}
    cd.gravar_base(inst, tmp_path / "base")
    lido = cd.ler_base(tmp_path / "base")
    assert lido["textos"] == inst["textos"] and lido["figuras"] == inst["figuras"] and lido["tempos"] == inst["tempos"]
    assert cd.ler_base(tmp_path / "nao_existe") is None


def test_nucleo_mudou_faz_o_comando_sair_com_codigo_3(monkeypatch, capsys):
    monkeypatch.setattr(cd, "etapa_ruff", lambda: (True, "ok"))
    monkeypatch.setattr(cd, "etapa_nucleo", lambda: (False, "NUCLEO MUDOU: x"))
    monkeypatch.setattr(cd, "etapa_varreduras", lambda: (True, "ok"))
    monkeypatch.setattr(cd, "etapa_instantaneo", lambda: (True, ["nenhuma diferenca"]))
    assert cd.main(["--rapido"]) == 3
    assert "O NUCLEO MUDOU" in capsys.readouterr().out


def test_tudo_verde_sai_0_e_falha_comum_sai_1(monkeypatch):
    for nome in ("etapa_ruff", "etapa_nucleo", "etapa_varreduras"):
        monkeypatch.setattr(cd, nome, lambda: (True, "ok"))
    monkeypatch.setattr(cd, "etapa_instantaneo", lambda: (True, ["diferenca so' informativa"]))
    assert cd.main(["--rapido"]) == 0                                    # diferenca de instantaneo nao bloqueia
    monkeypatch.setattr(cd, "etapa_ruff", lambda: (False, "erro"))
    assert cd.main(["--rapido"]) == 1


def test_o_manifesto_real_confere_pelo_comando_do_script():
    ok, det = cd.etapa_nucleo()
    assert ok, det


def test_ha_testes_de_interface_e_de_valores_que_existem():
    assert len(cd._tests(cd.TESTES_INTERFACE)) >= 15
    assert len(cd._tests(cd.TESTES_VALORES)) == 3
