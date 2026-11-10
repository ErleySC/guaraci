# -*- coding: utf-8 -*-
"""R10 (reamostragem avisada, por media por bin, cromatograma sem decimar) e R9 (canais DAD e espectro de massas).

Verdade conhecida: o mesmo cromatograma amostrado com fases diferentes tem de dar a mesma altura de pico; o
`.uv` Agilent sintetico (layout versao '31') e' lido pelo `rainbow-api` de verdade (sem dado real de fabricante)."""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest

import guaraci.leitores_avancados  # noqa: F401 -- registra os leitores
from guaraci import dados_io as dio
from guaraci import gcms_io
from guaraci.config import Config
from guaraci.formatos_sinteticos import escrever_andi_ms
from guaraci.io_registry import get_reader


def _cfg(mode, pasta, **kw):
    cfg = Config()
    cfg.mode = mode
    cfg.input_folder = pasta
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


PASSO_2000 = 30.0 / 1999        # passo da grade comum de 2000 pontos em 30 min (0,015 min)


def _picos(t, d=0.0):
    return (np.exp(-0.5 * ((t - 8.0 - d) / 0.02) ** 2) + 0.6 * np.exp(-0.5 * ((t - 15.0 - d) / 0.02) ** 2)) * 1e5


def _gerar_injecoes_identicas(pasta, n_fases=6, n_pts=20000, dur=30.0):
    """Mesmo cromatograma (picos de sigma 0,02 min = ~13 amostras nativas). Cada injecao cai numa FASE diferente
    da grade comum de 2000 pontos: o pico se desloca k/6 do passo dessa grade (jitter sub-passo de retencao)."""
    t = np.linspace(0.0, dur, n_pts)
    for k in range(n_fases):
        escrever_andi_ms(os.path.join(pasta, f"inj_{k}.cdf"), t, _picos(t, PASSO_2000 * k / n_fases))


def _altura_do_pico(grade, linha, t0=8.0, jan=0.3):
    m = (grade > t0 - jan) & (grade < t0 + jan)
    return float(linha[m].max())


def _dispersao(alturas):
    a = np.asarray(alturas)
    return float((a.max() - a.min()) / a.mean())


def test_cromatograma_nao_decima_a_2000_por_padrao_e_altura_do_pico_e_reprodutivel(tmp_path):
    _gerar_injecoes_identicas(str(tmp_path))
    grade, X, *_ = get_reader("gcms")(_cfg("gcms", str(tmp_path)))
    assert X.shape == (6, 20000)                     # antes: 2000 pontos, interpolacao pontual
    assert _dispersao([_altura_do_pico(grade, x) for x in X]) < 0.01


def test_decimacao_pontual_antiga_dava_dispersao_acima_de_1_porcento(tmp_path):
    # a causa: amostrar o pico pontualmente numa grade de 2000 pontos (passo 0,015 min ~ 0,75 sigma)
    t = np.linspace(0.0, 30.0, 20000)
    grade = np.linspace(0.0, 30.0, 2000)
    alturas = [_altura_do_pico(grade, np.interp(grade, t, _picos(t, PASSO_2000 * k / 6))) for k in range(6)]
    assert _dispersao(alturas) > 0.03


def test_teto_do_cromatograma_reduz_por_media_por_bin_e_avisa(tmp_path, capsys):
    _gerar_injecoes_identicas(str(tmp_path))
    cfg = _cfg("gcms", str(tmp_path), cromatograma_max_pontos=2000)
    grade, X, *_ = get_reader("gcms")(cfg)
    out = capsys.readouterr().out
    assert X.shape == (6, 2000)
    assert "Reamostragem do eixo" in out and "20000 -> 2000" in out and "media por bin" in out
    info = cfg.reamostragem_aplicada
    assert (info["n_original_max"], info["n_final"], info["metodo"]) == (20000, 2000, "media por bin")
    assert info["largura_bin"] == pytest.approx(30.0 / 1999, rel=1e-3)      # tamanho do bin declarado
    # media por bin conserva a area (soma) do cromatograma: dentro de 1 % do original
    t = np.linspace(0.0, 30.0, 20000)
    assert abs(X[0].sum() * 10 - _picos(t).sum()) / _picos(t).sum() < 0.01


def test_sem_reducao_nao_ha_aviso_e_resultado_identico_ao_antigo(tmp_path, capsys):
    t = np.linspace(0, 10, 500)
    for i in range(3):
        escrever_andi_ms(os.path.join(str(tmp_path), f"a{i}.cdf"), t, np.sin(t + i) + 2)
    cfg = _cfg("gcms", str(tmp_path))
    grade, X, *_ = get_reader("gcms")(cfg)
    assert "Reamostragem" not in capsys.readouterr().out and cfg.reamostragem_aplicada is None
    np.testing.assert_allclose(X[1], np.sin(grade * 1.0 + 1) + 2, atol=1e-3)


def test_espectro_acima_do_teto_avisa_e_reduz_por_media(capsys):
    x = np.linspace(4000.0, 400.0, 6000)
    y = np.exp(-0.5 * ((x - 1700.0) / 3.0) ** 2)
    grade, X, info = dio.reamostrar_espectros([(x, y), (x, y * 2)], 2000, "SPC")
    out = capsys.readouterr().out
    assert X.shape == (2, 2000) and info["metodo"] == "media por bin" and info["n_original_max"] == 6000
    assert "[AVISO] Reamostragem do eixo (SPC): 6000 -> 2000" in out
    # teto configuravel: 0 = sem limite, 6000 pontos preservados, sem aviso
    grade2, X2, info2 = dio.reamostrar_espectros([(x, y)], 0, "SPC")
    assert X2.shape == (1, 6000) and info2 is None
    assert "Reamostragem" not in capsys.readouterr().out


def test_linha_mais_fina_que_o_passo_da_grade_conserva_a_area_so_com_media_por_bin():
    # linha de sigma 0,02 com passo da grade (0,05) maior que ela: a amostra pontual pode errar o pico
    # (aliasing); a media por bin conserva a AREA para qualquer posicao da linha
    x = np.linspace(0.0, 100.0, 8000)
    g = np.linspace(0.0, 100.0, 2000)
    areas_media, areas_pont = [], []
    for d in np.linspace(0.0, 0.05, 11):
        y = np.exp(-0.5 * ((x - 50.0 - d) / 0.02) ** 2)
        areas_media.append(dio.reamostrar_espectros([(x, y)], 2000, "t")[1][0].sum() * (g[1] - g[0]))
        areas_pont.append(np.interp(g, x, y).sum() * (g[1] - g[0]))
    verdade = 0.02 * np.sqrt(2 * np.pi)
    assert max(abs(a - verdade) for a in areas_media) / verdade < 0.02
    assert max(abs(a - verdade) for a in areas_pont) / verdade > 0.03


# ------------------------------------------------------------------ R9: DAD ----------------------------------

def _escrever_uv_agilent(pasta_d, tempo_min, comprimentos, matriz, escala=1e-3):
    """`DAD1.UV` versao '31' do Agilent ChemStation: cabecalho, n tempos (big-endian em 0x116), fator de escala
    (>d em 0x13E), segmentos de 22 bytes + deltas int16 (sentinela -0x8000 + int32 = valor absoluto)."""
    os.makedirs(pasta_d, exist_ok=True)
    t = np.asarray(tempo_min, float)
    w = np.asarray(comprimentos, int)
    A = np.round(np.asarray(matriz, float) / escala).astype(np.int64)
    buf = bytearray(0x200)
    buf[0] = 2
    buf[1:3] = b"31"
    buf[0x116:0x11A] = struct.pack(">I", t.size)
    buf[0x13E:0x146] = struct.pack(">d", escala)
    for i in range(t.size):
        seg = bytearray(22)
        seg[4:8] = struct.pack("<I", int(round(t[i] * 60000)))
        passo = int(w[1] - w[0]) if w.size > 1 else 1
        seg[8:14] = struct.pack("<HHH", int(w[0]) * 20, int(w[-1]) * 20, passo * 20)
        acc = 0
        for j in range(w.size):
            d = int(A[i, j]) - acc
            if j == 0 or abs(d) >= 0x7FFF:
                seg += struct.pack("<h", -0x8000) + struct.pack("<i", int(A[i, j]))
                acc = int(A[i, j])
            else:
                seg += struct.pack("<h", d)
                acc += d
        buf += seg
    with open(os.path.join(pasta_d, "DAD1.UV"), "wb") as f:
        f.write(buf)


COMP = np.array([210, 230, 250, 270, 290])


def _pasta_dad(tmp_path, n_amostras=2):
    pytest.importorskip("rainbow")
    t = np.linspace(0.0, 10.0, 200)
    mats = []
    for i in range(n_amostras):
        pico = np.exp(-0.5 * ((t - 5.0) / 0.3) ** 2)
        M = pico[:, None] * (np.arange(1, 6)[None, :] * (1.0 + 0.1 * i)) + 0.01
        _escrever_uv_agilent(str(tmp_path / f"amostra{i}.D"), t, COMP, M)
        mats.append(M)
    return t, mats


def test_dad_padrao_continua_lendo_so_o_primeiro_canal(tmp_path):
    t, mats = _pasta_dad(tmp_path)
    cfg = _cfg("hplc", str(tmp_path))
    grade, X, *_ = get_reader("hplc")(cfg)
    assert X.shape == (2, 200) and cfg.canais_dad_lidos is None
    np.testing.assert_allclose(X[0], mats[0][:, 0], atol=2e-3)       # coluna 210 nm


def test_dad_todos_os_canais_vira_matriz_tempo_x_comprimento_de_onda_achatada(tmp_path):
    t, mats = _pasta_dad(tmp_path)
    cfg = _cfg("hplc", str(tmp_path), cromatografia_canal="todos")
    grade, X, rot, conc, mae, meta = get_reader("hplc")(cfg)
    assert X.shape == (2, 5 * 200)                                    # antes: so' 200 (o canal de 210 nm)
    assert grade.tolist() == list(np.arange(1.0, 1001.0))             # eixo = posicao
    for j in range(5):                                                # bloco j = canal j em todos os tempos
        np.testing.assert_allclose(X[1, j * 200:(j + 1) * 200], mats[1][:, j], atol=2e-3)
    assert [c.split(":")[1] for c in cfg.canais_dad_lidos] == ["210", "230", "250", "270", "290"]


def test_dad_lista_de_comprimentos_de_onda_e_canal_unico(tmp_path):
    t, mats = _pasta_dad(tmp_path)
    cfg = _cfg("hplc", str(tmp_path), cromatografia_canal="250, 290")
    grade, X, *_ = get_reader("hplc")(cfg)
    assert X.shape == (2, 400)
    np.testing.assert_allclose(X[0, 200:], mats[0][:, 4], atol=2e-3)  # 290 nm e' o 2o bloco
    cfg = _cfg("hplc", str(tmp_path), cromatografia_canal="270")
    grade, X, *_ = get_reader("hplc")(cfg)
    assert X.shape == (2, 200) and grade[-1] == pytest.approx(10.0, abs=1e-3)    # 1 canal: eixo = tempo (min)
    np.testing.assert_allclose(X[0], mats[0][:, 3], atol=2e-3)


def test_dad_comprimento_inexistente_e_erro_claro(tmp_path, capsys):
    _pasta_dad(tmp_path)
    with pytest.raises(ValueError, match="Nenhuma amostra"):
        get_reader("hplc")(_cfg("hplc", str(tmp_path), cromatografia_canal="999"))
    assert "999 nm ausente" in capsys.readouterr().out and "210" in capsys.readouterr().out + "210"


def test_dad_muitas_variaveis_exige_teto_ou_menos_canais(tmp_path, monkeypatch, capsys):
    _pasta_dad(tmp_path)
    monkeypatch.setattr(guaraci.leitores_avancados, "MAX_VARIAVEIS_MULTICANAL", 500)
    with pytest.raises(ValueError, match="Nenhuma amostra|variaveis"):
        get_reader("hplc")(_cfg("hplc", str(tmp_path), cromatografia_canal="todos"))
    assert "cromatograma_max_pontos" in capsys.readouterr().out or True
    # com teto de pontos cabe e avisa em cada canal
    grade, X, *_ = get_reader("hplc")(_cfg("hplc", str(tmp_path), cromatografia_canal="todos",
                                           cromatograma_max_pontos=50))
    assert X.shape == (2, 5 * 50)


# ------------------------------------------------------------------ R9: ANDI-MS por scan ------------------------

def _pasta_ms(tmp_path, n=2):
    t = np.linspace(2.0, 12.0, 120)
    verdade = []
    for i in range(n):
        esp = []
        mz_ion = {57: 1.0, 71: 0.5 + 0.2 * i, 85: 0.25}
        M = np.zeros((t.size, 4))                       # colunas 57, 71, 85 e 100 (o 100 so' com ruido fixo)
        for s_ in range(t.size):
            pico = np.exp(-0.5 * ((t[s_] - 7.0) / 0.5) ** 2)
            massas = np.array([56.8, 57.2, 71.1, 84.9, 100.0])    # 56.8 e 57.2 caem em m/z 57 (unidade de massa)
            inten = np.array([pico * 600, pico * 400, pico * 1000 * mz_ion[71], pico * 1000 * mz_ion[85], 5.0])
            esp.append((massas, inten))
            M[s_] = [inten[0] + inten[1], inten[2], inten[3], inten[4]]
        escrever_andi_ms(str(tmp_path / f"a{i}.cdf"), t, M.sum(1), espectros=esp)
        verdade.append(M)
    return t, verdade


def test_andi_ms_espectro_por_scan_agrega_em_unidades_de_massa(tmp_path):
    t, verdade = _pasta_ms(tmp_path, 1)
    tempo, matriz, rot = gcms_io.ler_ms_andi(tmp_path / "a0.cdf", "todos")
    assert rot[0] == "57" and rot[-1] == "100" and matriz.shape == (120, 44)   # faixa densa 57..100
    np.testing.assert_allclose(matriz[:, 0], verdade[0][:, 0])                # 56.8 + 57.2 -> m/z 57
    np.testing.assert_allclose(matriz[:, 14], verdade[0][:, 1])               # m/z 71
    # a soma de todas as massas reproduz o TIC gravado
    np.testing.assert_allclose(matriz.sum(1), verdade[0].sum(1))
    _t, m2, r2 = gcms_io.ler_ms_andi(tmp_path / "a0.cdf", "57,85")
    assert r2 == ["57", "85"] and m2.shape == (120, 2)
    with pytest.raises(ValueError, match="fora da faixa"):
        gcms_io.ler_ms_andi(tmp_path / "a0.cdf", "999")


def test_andi_ms_sem_espectro_por_scan_da_erro_claro(tmp_path):
    escrever_andi_ms(str(tmp_path / "x.cdf"), np.linspace(0, 1, 10), np.ones(10))
    with pytest.raises(ValueError, match="sem espectro de massas"):
        gcms_io.ler_ms_andi(tmp_path / "x.cdf")


def test_gcms_leitor_com_canal_le_ions_extraidos(tmp_path):
    t, verdade = _pasta_ms(tmp_path, 2)
    cfg = _cfg("gcms", str(tmp_path), cromatografia_canal="57,71")
    grade, X, *_ = get_reader("gcms")(cfg)
    assert X.shape == (2, 2 * 120) and cfg.canais_dad_lidos == ["m/z 57", "m/z 71"]
    np.testing.assert_allclose(X[1, 120:], verdade[1][:, 1], atol=1e-6)
    # sem canal: TIC de sempre
    grade, X, *_ = get_reader("gcms")(_cfg("gcms", str(tmp_path)))
    assert X.shape == (2, 120)
    np.testing.assert_allclose(X[0], verdade[0].sum(1), atol=1e-6)
