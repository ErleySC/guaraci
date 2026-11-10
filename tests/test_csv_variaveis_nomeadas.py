# -*- coding: utf-8 -*-
"""CSV generico (nao espectral): cabecalhos NOMEADOS ou MISTOS viram variaveis (eixo = posicao 1..k),
metadado fica fora, e o erro de CSV sem variaveis nao e' diagnosticado como corte de faixa (GUA-E007)."""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from guaraci import assistente_erros as E
from guaraci import dados_io as dio

NOMES9 = ["var_a", "var_b", "var_c", "var_d", "var_e", "var_f", "var_g", "var_h", "var_i"]


def _csv(tmp_path, cab, linhas):
    p = tmp_path / "t.csv"
    p.write_text(",".join(cab) + "\n" + "\n".join(",".join(str(v) for v in l) for l in linhas) + "\n")
    return str(p)


def test_cabecalho_misto_texto_e_numero_vira_variaveis_sem_corromper_o_eixo(tmp_path):
    p = _csv(tmp_path, ["classe", "ph", "5", "brix", "12", "cinzas"],
             [["A", 1, 2, 3, 4, 5], ["B", 2, 3, 4, 5, 6]])
    eixo, X, rot, conc, mae, nomes = dio._carregar_csv_completo(p, "classe", "")
    assert X.shape == (2, 5)                         # antes: so' as colunas "5" e "12" (eixo 5, 12)
    assert eixo.tolist() == [1, 2, 3, 4, 5]
    assert nomes == ["ph", "5", "brix", "12", "cinzas"]
    assert X[0].tolist() == [1, 2, 3, 4, 5]


def test_so_duas_colunas_nomeadas_e_uma_numerica_e_ok(tmp_path):
    p = _csv(tmp_path, ["classe", "ph", "brix"], [["A", 1, 2], ["B", 2, 3]])
    eixo, X, *_ = dio.load_csv_com_grupos(p, "classe", "")
    assert X.shape == (2, 2) and eixo.tolist() == [1, 2]


def test_espectro_com_coluna_extra_nomeada_continua_espectro(tmp_path):
    cab = ["classe"] + [str(i) for i in range(1, 9)] + ["temperatura"]
    p = _csv(tmp_path, cab, [["A"] + list(range(8)) + [25], ["B"] + list(range(8)) + [26]])
    eixo, X, rot, conc, mae, nomes = dio._carregar_csv_completo(p, "classe", "")
    assert nomes is None and X.shape == (2, 8) and eixo.tolist() == list(range(1, 9))


def test_metadado_numerico_nao_vira_variavel_na_tabela_mista(tmp_path):
    p = _csv(tmp_path, ["classe", "lote", "x1", "7", "x3"], [["A", 1, 1, 2, 3], ["B", 2, 4, 5, 6]])
    _eixo, X, *_r, nomes = dio._carregar_csv_completo(p, "classe", "")
    assert nomes == ["x1", "7", "x3"] and X.shape == (2, 3)


def test_erro_sem_variaveis_nao_e_diagnosticado_como_corte_de_faixa(tmp_path):
    p = _csv(tmp_path, ["classe", "ph"], [["A", 1], ["B", 2]])
    with pytest.raises(ValueError) as ei:
        dio.load_csv_com_grupos(p, "classe", "")
    cods = E.codigos(ei.value)
    assert "GUA-E007" not in cods and cods == ["GUA-E014"], (cods, str(ei.value))
    # E007 continua para o corte de faixa de verdade, sem mandar "cabecalho tem de ser numero"
    assert "GUA-E007" in E.codigos("ValueError: Found array with 0 feature(s)")
    assert "cabecalhos das colunas espectrais devem ser numeros" not in E.explicar("GUA-E007")


def _tabela_9(path, com_conc):
    rng = np.random.default_rng(1)
    linhas = ["classe,lote,teor," + ",".join(NOMES9)]
    for c, base in (("c1", 0.0), ("c2", 1.5), ("c3", 3.0)):
        for lote in range(6):
            v = base + rng.normal(0, 0.4, 9) + np.arange(9) * 0.1
            teor = 10 * v[2] + rng.normal(0, 0.3)
            for _r in range(2):
                w = v + rng.normal(0, 0.05, 9)
                linhas.append(f"{c},{c}_{lote},{teor:.3f}," + ",".join(f"{x:.4f}" for x in w))
    Path(path).write_text("\n".join(linhas) + "\n")


@pytest.mark.slow
@pytest.mark.parametrize("objetivo", ["classificacao", "quantificacao"])
def test_tabela_de_9_colunas_nomeadas_roda_o_pipeline_completo(objetivo):
    from guaraci import guaraci as g
    pasta = tempfile.mkdtemp(prefix="gn")
    try:
        csv = os.path.join(pasta, "tab.csv")
        _tabela_9(csv, True)
        cfg = os.path.join(pasta, "c.yaml")
        Path(cfg).write_text(
            f"modo_entrada: csv\nfaixa_min_cm: 0.0\nfaixa_max_cm: 999999.0\npre_processamento: autoscaling\n"
            f"arquivo_csv: {csv}\npasta_dados: {pasta}\ncoluna_classe: classe\ncoluna_grupo: lote\n"
            f"coluna_concentracao: {'teor' if objetivo == 'quantificacao' else chr(39) * 2}\n"
            f"abrir_figuras_na_tela: false\npasta_saida: {pasta}/saida\nn_permutacoes: 20\nmax_lvs: 6\n"
            f"objetivo: {objetivo}\n")
        g.main(["run", cfg])
        resumo = next(Path(pasta).rglob("resumo_modelo.txt")).read_text()
        assert "var_a, var_b" in resumo and resumo.count("var_") >= 9
        assert "var_a, var_b" in resumo and "var_i" in resumo   # nomes no resumo
        figs = list(Path(pasta).rglob("*.png"))
        assert figs and all(f.stat().st_size > 5000 for f in figs)
    finally:
        shutil.rmtree(pasta, ignore_errors=True)


def test_figura_de_loadings_mostra_nomes_e_ordem_do_arquivo(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from guaraci import figuras as F
    from guaraci.pipeline import Config
    cfg = Config(mode="csv")
    cfg.nomes_variaveis = ["x1", "x2", "x3"]
    assert F.eixo_x_decrescente(cfg) is False        # nomes: nunca eixo decrescente
    fig, ax = plt.subplots()
    ax.bar([1, 2, 3], [1, 2, 3])
    ax.set_xlabel("Spectral variable")
    F._nomes_de_variaveis_no_eixo(fig, cfg)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["x1", "x2", "x3"]
    assert ax.get_xlabel() == "Variable"
    plt.close(fig)
