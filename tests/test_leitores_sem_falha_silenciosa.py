# -*- coding: utf-8 -*-
"""Os leitores nunca falham em silencio: ou leem o arquivo, ou dizem onde e por que nao leram.

Bateria de 2026-11-20 (arquivos gerados na hora, nada de dados externos):
- a MESMA tabela em CSV (virgula/ponto, ponto e virgula/virgula decimal, TAB, latin-1, UTF-8 com BOM,
  UTF-16 do Excel), Excel e .mat v5 vira o mesmo X, eixo, classes e grupos;
- linha totalmente vazia (sobra de planilha) sai com aviso, em vez de acusar a coluna de grupo;
- NaN/Inf param o pre-voo dizendo a amostra e a variavel;
- .dx truncado e' recusado (antes virava um espectro completado com valor constante, inventado);
- arquivo pulado na leitura de pasta fica em ARQUIVOS_IGNORADOS.txt e no model card, nao so' no log.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from guaraci import dados_io
from guaraci import assistente_regras as regras

EIXO = np.linspace(4000.0, 10000.0, 20)


def _tabela() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    linhas = []
    for k, c in enumerate(["Acai", "Cupuacu"]):
        for g in range(3):
            for r in range(2):
                y = np.exp(-((EIXO - (5000 + 2000 * k)) / 800) ** 2) + 0.01 * rng.standard_normal(EIXO.size)
                linhas.append([c, f"{c}_L{g}", 1.0 * g + 0.1 * r] + list(np.round(y, 6)))
    return pd.DataFrame(linhas, columns=["classe", "lote", "teor"] + [f"{w:.2f}" for w in EIXO])


def _ler(caminho):
    return dados_io._carregar_csv_completo(str(caminho), "classe", "teor", col_grupo="lote")


def _gravar_variantes(df: pd.DataFrame, pasta) -> list:
    caminhos = []

    def txt(nome, conteudo, enc="utf-8"):
        p = pasta / nome
        p.write_bytes(conteudo.encode(enc))
        caminhos.append(p)

    txt("virgula.csv", df.to_csv(index=False))
    txt("pv_decimal_virgula.csv", df.to_csv(index=False, sep=";", decimal=","))
    txt("tab.txt", df.to_csv(index=False, sep="\t"))
    txt("latin1.csv", df.to_csv(index=False, sep=";", decimal=","), enc="latin-1")
    txt("bom.csv", "﻿" + df.to_csv(index=False))
    txt("utf16.txt", df.to_csv(index=False, sep="\t"), enc="utf-16")   # "Texto Unicode" do Excel
    pytest.importorskip("openpyxl")
    df.to_excel(pasta / "planilha.xlsx", index=False)
    caminhos.append(pasta / "planilha.xlsx")
    import scipy.io as sio
    sio.savemat(pasta / "dados.mat", {"X": df.iloc[:, 3:].to_numpy(), "eixo": EIXO,
                                      "classe": df.classe.to_numpy(dtype=object),
                                      "lote": df.lote.to_numpy(dtype=object), "teor": df.teor.to_numpy()})
    caminhos.append(pasta / "dados.mat")
    return caminhos


def test_mesma_tabela_em_todos_os_formatos_da_o_mesmo_resultado(tmp_path):
    df = _tabela()
    ref = _ler(_gravar_variantes(df, tmp_path)[0])
    for p in _gravar_variantes(df, tmp_path)[1:]:
        eixo, X, rot, conc, grupo, _ = _ler(p)
        assert X.shape == ref[1].shape, p.name
        np.testing.assert_allclose(eixo, ref[0], rtol=1e-6, err_msg=p.name)
        np.testing.assert_allclose(X, ref[1], rtol=1e-9, err_msg=p.name)
        assert list(rot) == list(ref[2]), p.name
        np.testing.assert_allclose(conc, ref[3], err_msg=p.name)
        assert list(grupo) == list(ref[4]), p.name


def test_utf16_do_excel_e_lido(tmp_path):
    p = tmp_path / "unicode.txt"
    p.write_bytes(_tabela().to_csv(index=False, sep="\t").encode("utf-16"))
    _, X, rot, *_ = _ler(p)
    assert X.shape == (12, 20) and set(rot) == {"Acai", "Cupuacu"}


def test_linha_totalmente_vazia_sai_com_aviso(tmp_path, capsys):
    df = _tabela()
    linhas = df.to_csv(index=False, sep=";").splitlines()
    linhas.insert(4, ";" * (df.shape[1] - 1))
    p = tmp_path / "com_vazia.csv"
    p.write_text("\n".join(linhas) + "\n;;;\n", encoding="utf-8")
    _, X, *_ = _ler(p)
    assert X.shape == (12, 20)
    assert "linha(s) totalmente vazia(s)" in capsys.readouterr().out


def test_nan_e_inf_dizem_amostra_e_variavel():
    X = np.ones((5, 4))
    X[1, 2] = np.nan
    X[3, 0] = np.inf
    r = regras.resumir_dados(X, eixo=[10.0, 20.0, 30.0, 40.0])
    assert r.n_nan == 2
    assert r.nan_amostras == "2, 4" and r.nan_variaveis == "10, 30"


def _dx(caminho, y, eixo, titulo="x"):
    linhas = ["##TITLE=" + titulo, "##JCAMP-DX=4.24", "##XUNITS=1/CM", "##YUNITS=ABSORBANCE",
              f"##FIRSTX={eixo[0]}", f"##LASTX={eixo[-1]}", "##XFACTOR=1", "##YFACTOR=1",
              f"##NPOINTS={len(eixo)}", "##XYDATA=(X++(Y..Y))"]
    for i in range(0, len(eixo), 10):
        linhas.append(f"{eixo[i]:.4f} " + " ".join(f"{v:.6f}" for v in y[i:i + 10]))
    linhas.append("##END=")
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def test_dx_truncado_e_recusado_em_vez_de_completado(tmp_path):
    eixo = np.linspace(4000, 10000, 300)
    p = tmp_path / "a.dx"
    _dx(p, np.sin(eixo / 500), eixo)
    texto = p.read_text()
    p.write_text(texto[: len(texto) // 3])
    with pytest.raises(ValueError, match="truncado"):
        dados_io.parse_dx(str(p))


def test_dx_com_um_ponto_faltando_ainda_e_lido_com_aviso(tmp_path, capsys):
    eixo = np.linspace(4000, 10000, 300)
    p = tmp_path / "a.dx"
    _dx(p, np.sin(eixo / 500), eixo)
    linhas = p.read_text().splitlines()
    i = next(k for k, s in enumerate(linhas) if s.startswith("##XYDATA")) + 5
    partes = linhas[i].split()
    linhas[i] = " ".join(partes[:-1])           # tira o ultimo valor de uma linha do meio
    p.write_text("\n".join(linhas) + "\n")
    x, y = dados_io.parse_dx(str(p))
    assert len(y) == 300 and np.all(np.isfinite(y))
    assert "preenchido(s) por interpolacao" in capsys.readouterr().out


def test_arquivo_pulado_na_pasta_vai_para_arquivos_ignorados(tmp_path):
    from guaraci import pipeline
    eixo = np.linspace(4000, 10000, 100)
    dados = tmp_path / "dados"
    for c in ("A", "B"):
        (dados / c).mkdir(parents=True)
        for i in range(3):
            _dx(dados / c / f"{c}_{i}.dx", np.cos(eixo / (300 + 50 * i)), eixo, titulo=f"{c}_{i}")
    (dados / "A" / "vazio.dx").write_text("")
    bom = (dados / "B" / "B_0.dx").read_text()
    (dados / "B" / "cortado.dx").write_text(bom[: len(bom) // 3])
    dados_io.load_dx(str(dados), use_parse_title=False)
    ignorados, total = dados_io.obter_arquivos_ignorados()
    nomes = sorted(os.path.basename(a) for a, _ in ignorados)
    assert nomes == ["cortado.dx", "vazio.dx"] and total == 8
    saida = tmp_path / "saida"
    logs = saida / "Reports"
    logs.mkdir(parents=True)
    (logs / "model_card.md").write_text("# card\n", encoding="utf-8")
    assert pipeline._gravar_arquivos_ignorados(str(saida), str(logs)) == 2
    texto = (saida / pipeline.NOME_ARQUIVOS_IGNORADOS).read_text(encoding="utf-8")
    assert "2 de 8" in texto and "cortado.dx" in texto and "truncado" in texto
    assert "ARQUIVOS IGNORADOS" in (logs / "model_card.md").read_text(encoding="utf-8")
    dados_io.resetar_arquivos_ignorados()
    assert pipeline._gravar_arquivos_ignorados(str(saida), str(logs)) == 0


def test_holdout_de_regressao_muito_pior_que_a_cv_gera_alerta():
    from guaraci.resultados_io import alerta_holdout_regressao as alerta
    # caso real (conjunto publico de NIR): RMSECV 0,81, holdout RMSEP 3,38 e R2 = -5,1
    assert "NAO GENERALIZOU" in alerta({"r2": -5.1, "rmsep": 3.38}, 0.81, "pt")
    assert "DID NOT GENERALIZE" in alerta({"r2": -5.1, "rmsep": 3.38}, 0.81, "en")
    assert "2x RMSECV" in alerta({"r2": 0.6, "rmsep": 2.5}, 1.0, "pt")
    # R2 do holdout ainda alto: diz que a CV esta' otimista, nao que o modelo "nao generalizou"
    so_razao = alerta({"r2": 0.973, "rmsep": 0.829}, 0.408, "pt")
    assert "BEM PIOR QUE A VALIDACAO CRUZADA" in so_razao and "NAO GENERALIZOU" not in so_razao
    assert alerta({"r2": 0.9, "rmsep": 1.1}, 1.0, "pt") == ""
    assert alerta(None, 1.0) == "" and alerta({"r2": float("nan"), "rmsep": 1.0}, None) == ""
