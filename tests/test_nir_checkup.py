# -*- coding: utf-8 -*-
"""Checkup NIR (FT-NIR e NIR dispersivo): faixa do catalogo em nm, JCAMP AFFN/PAC,
arquivo_grupos no modo dx e pacote opcional ausente."""
import numpy as np
import pytest

from guaraci import dados_io
from guaraci.catalogo_tecnicas import INSTRUMENTOS
from guaraci.config import Config


def test_nir_dispersivo_tem_faixa_em_nm_que_cobre_os_instrumentos_reais():
    """NIR dispersivo exporta em nm: a faixa que a tela [2.4] grava nao pode ser em cm-1
    (4000-14000 nao cruzava 1100-2498 nm e o pre-flight recusava toda execucao)."""
    ins = INSTRUMENTOS["nir"]
    for lo, hi in ((1100.0, 2498.0), (950.0, 1650.0), (900.0, 1700.0)):
        assert ins["faixa_min"] <= lo and hi <= ins["faixa_max"]
    assert "cm-1" not in ins["faixa_txt"][0]


def _jcamp(caminho, x, yi, linhas_dados, xfactor=1.0):
    cab = ["##TITLE=t", "##JCAMP-DX=4.24", f"##XFACTOR={xfactor}", "##YFACTOR=0.001",
           f"##FIRSTX={x[0]}", f"##LASTX={x[-1]}", f"##NPOINTS={len(x)}", "##XYDATA=(X++(Y..Y))"]
    caminho.write_text("\n".join(cab + linhas_dados + ["##END="]) + "\n")


def test_parse_dx_le_linhas_affn_e_pac(tmp_path):
    x = np.array([12500.0, 12496.0, 12492.0, 12488.0, 12484.0, 12480.0])
    yi = [523, 524, -12, 7, 600, 601]
    affn = tmp_path / "affn.dx"
    _jcamp(affn, x, yi, ["12500.0 523 524 -12", "12488.0 7 600 601"])
    pac = tmp_path / "pac.dx"
    _jcamp(pac, x, yi, ["12500+523+524-12", "12488+7+600+601"])
    for arq in (affn, pac):
        X, Y = dados_io.parse_dx(str(arq))
        assert not np.isnan(Y).any()
        np.testing.assert_allclose(Y, np.array(yi) * 0.001)
        np.testing.assert_allclose(X, x)


def test_parse_dx_asdf_continua_igual():
    SQZ = {"@": 0, **{c: i for i, c in enumerate("ABCDEFGHI", 1)}, **{c: -i for i, c in enumerate("abcdefghi", 1)}}
    DIF = {"%": 0, **{c: i for i, c in enumerate("JKLMNOPQR", 1)}, **{c: -i for i, c in enumerate("jklmnopqr", 1)}}
    DUP = {c: i for i, c in enumerate("STUVWXYZs", 1)}
    assert dados_io._decodificar_linha_asdf("4000E12J", SQZ, DIF, DUP) == (4000.0, [512, 513])


def test_modo_dx_usa_arquivo_grupos(tmp_path):
    x = np.linspace(1100.0, 2498.0, 50)
    nomes = []
    for cls in ("A", "B"):
        (tmp_path / "d" / cls).mkdir(parents=True)
        for lote in (1, 2, 3):
            for rep in (1, 2):
                n = f"{cls}{lote}_R{rep}"
                nomes.append((n, f"{cls}{lote}"))
                y = np.round(1000 * (0.3 + 0.01 * lote + 0.001 * rep * np.sin(x / 50))).astype(int)
                linhas = [f"{x[i]:.4f} " + " ".join(str(v) for v in y[i:i + 10]) for i in range(0, 50, 10)]
                _jcamp(tmp_path / "d" / cls / f"{n}.dx", x, y, linhas)
    gr = tmp_path / "grupos.csv"
    gr.write_text("amostra,lote\n" + "\n".join(f"{n},{g}" for n, g in nomes) + "\n")
    cfg = Config(mode="dx", input_folder=str(tmp_path / "d"), use_parse_title=False, group_file=str(gr))
    assert dados_io.garantia_agrupamento_prevista(cfg) == "medium"
    res = dados_io.load_data(cfg)
    mae = res[4]
    assert mae is not None and len(np.unique(mae)) == 6
    assert cfg.grouping_guarantee == "medium"
    # sem arquivo de grupos: continua sem mae_id (comportamento anterior)
    cfg2 = Config(mode="dx", input_folder=str(tmp_path / "d"), use_parse_title=False)
    assert dados_io.load_data(cfg2)[4] is None


def test_pacote_opcional_ausente_falha_no_primeiro_arquivo(tmp_path):
    from guaraci.leitores_avancados import load_generic_file_dataset
    for i in range(3):
        (tmp_path / f"a{i}.0").write_bytes(b"\n\n\xfe\xfe")
    chamadas = []

    def parser(_c):
        chamadas.append(_c)
        raise ImportError("Pacote opcional 'brukeropus' nao instalado")

    with pytest.raises(ImportError, match="brukeropus"):
        load_generic_file_dataset(str(tmp_path), parser, "OPUS")
    assert len(chamadas) == 1
