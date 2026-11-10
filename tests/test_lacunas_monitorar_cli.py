"""guaraci monitorar: o caminho completo (tabelas, relatorio, figuras) e as recusas de uso, com tabelas sinteticas minusculas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guaraci import monitorar_cli as mc


def _tabela(tmp_path, nome, lotes, n_por_lote=8, deslocamento=None, seed=0, colunas=("v1", "v2", "v3", "v4", "v5")):
    rng = np.random.default_rng(seed)
    linhas = []
    for lote in lotes:
        desloc = (deslocamento or {}).get(lote, 0.0)
        n = n_por_lote if not isinstance(n_por_lote, dict) else n_por_lote[lote]
        for i in range(n):
            base = rng.normal(size=len(colunas)) + desloc
            linhas.append({"id": f"{lote}-{i}", "lote": lote, **dict(zip(colunas, base))})
    caminho = tmp_path / nome
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    return str(caminho)


def _ref(tmp_path):
    return _tabela(tmp_path, "ref.csv", [f"R{i}" for i in range(6)], seed=1)


def test_monitoramento_completo_alarma_o_lote_deslocado_e_marca_o_que_nao_tem_poder(tmp_path, capsys):
    novos = _tabela(tmp_path, "novos.csv", ["ok", "desviado", "pequeno"], seed=2, deslocamento={"desviado": 6.0},
                    n_por_lote={"ok": 60, "desviado": 60, "pequeno": 2})
    saida = tmp_path / "saida"
    rc = mc.executar(["--referencia", _ref(tmp_path), "--novos", novos, "--lote", "lote", "--id", "id", "--mais-graficos",
                      "--saida", str(saida)])
    texto = capsys.readouterr().out
    assert rc == 0
    assert "lote desviado" in texto and "ALARME" in texto and "sem poder para testar" in texto and "em controle" in texto
    for arq in ("tabelas/monitor_amostras.csv", "tabelas/monitor_lotes.csv", "relatorio/monitor_resumo.txt", "config_efetiva.yaml"):
        assert (saida / arq).is_file(), arq
    amostras = pd.read_csv(saida / "tabelas" / "monitor_amostras.csv", sep=";", decimal=",")
    assert list(amostras.columns[:2]) == ["id", "lote"] and len(amostras) == 122
    figuras = list((saida / "Graficos").glob("*"))
    assert {f.stem for f in figuras} >= {"carta_t2_q", "tendencia_por_lote", "contribuicao_q"}


def test_monitoramento_sem_coluna_de_lote_sugere_informar_e_trata_os_novos_como_um_lote(tmp_path, capsys):
    ref = _tabela(tmp_path, "ref.csv", [f"R{i}" for i in range(6)], seed=1)
    novos = _tabela(tmp_path, "novos.csv", ["a"], seed=3)
    rc = mc.executar(["--referencia", ref, "--novos", novos, "--min-lotes", "1", "--saida", str(tmp_path / "s2")])
    assert rc == 0 and "informe --lote COL" in capsys.readouterr().out


@pytest.mark.parametrize("argv_extra, trecho", [
    (["--ajuda"], "Uso: guaraci monitorar"),
])
def test_ajuda_imprime_o_uso(capsys, argv_extra, trecho):
    assert mc.executar(argv_extra) == 0 and trecho in capsys.readouterr().out


def test_recusas_de_uso_devolvem_codigo_2_com_mensagem(tmp_path, capsys):
    ref = _ref(tmp_path)
    novos = _tabela(tmp_path, "n.csv", ["a"], seed=2)
    casos = [
        (["--referencia", ref], "informe --referencia e --novos"),
        (["--referencia", ref, "--novos", novos, "--alpha", "0.9"], "--alpha deve estar entre"),
        (["--referencia", ref, "--novos", novos, "--alpha", "abc"], "argumento invalido"),
        (["--referencia", ref, "--novos", str(tmp_path / "nao_existe.csv"), "--lote", "lote"], ""),
        (["--referencia", ref, "--novos", novos, "--lote", "coluna_que_nao_existe"], ""),
        (["--referencia", ref, "--novos", novos, "--lote", "lote", "--min-lotes", "20"], "monitoramento recusado"),
    ]
    for argv, trecho in casos:
        assert mc.executar([*argv, "--saida", str(tmp_path / "x")]) == 2, argv
        assert trecho in capsys.readouterr().err


def test_tabelas_sem_variavel_numerica_suficiente_com_celula_vazia_ou_com_colunas_diferentes_sao_recusadas(tmp_path, capsys):
    ref = _ref(tmp_path)
    uma_coluna = _tabela(tmp_path, "uma.csv", ["a"], colunas=("v1",))
    assert mc.executar(["--referencia", uma_coluna, "--novos", uma_coluna, "--lote", "lote"]) == 2
    assert "menos de 2 colunas numericas" in capsys.readouterr().err
    com_nan = pd.read_csv(ref)
    com_nan.loc[0, "v2"] = np.nan
    caminho_nan = tmp_path / "nan.csv"
    com_nan.to_csv(caminho_nan, index=False)
    assert mc.executar(["--referencia", str(caminho_nan), "--novos", ref, "--lote", "lote"]) == 2
    assert "celulas vazias ou nao numericas" in capsys.readouterr().err
    outras = _tabela(tmp_path, "outras.csv", ["a"], colunas=("v1", "v2", "v3", "v4", "w9"))
    assert mc.executar(["--referencia", ref, "--novos", outras, "--lote", "lote"]) == 2
    assert "diferem" in capsys.readouterr().err


def test_figuras_sem_limite_de_dominio_e_sem_taxa_de_referencia_omitem_as_linhas_de_apoio(tmp_path):
    from sklearn.decomposition import PCA
    Z = np.random.default_rng(0).normal(size=(6, 4))
    res = {"dentro_dominio": [True, True, False, True, True, True], "t2": np.arange(6.0), "q": np.arange(6.0)[::-1],
           "t2_limite": None, "q_limite": float("nan"),
           "por_lote": [{"lote": "a", "taxa_fora": 0.1, "alarme": False, "taxa_referencia": None},
                        {"lote": "b", "taxa_fora": 0.4, "alarme": True, "taxa_referencia": None}]}
    feitos = mc._figuras(str(tmp_path), res, ["a", "a", "a", "b", "b", "b"], True, Z, PCA(n_components=2).fit(Z))
    assert len(feitos) == 3 and all(f.endswith(".png") for f in feitos)
