"""`guaraci anova`: ANOVA sobre a unidade independente, fail-closed."""
import csv
import os

import numpy as np
import pytest
from scipy import stats

import guaraci.guaraci as g
from guaraci import anova_cli


def _tabela(caminho, *, replicas=3, seed=1, sep=","):
    rng = np.random.default_rng(seed)
    linhas = [["amostra", "grupo", "replica", "teor"]]
    for gi, (grp, media) in enumerate((("A", 10.0), ("B", 11.5), ("C", 14.0))):
        for a in range(8):
            ef = rng.normal(0, 0.8)
            for r in range(replicas):
                linhas.append([f"{grp}{a:02d}", grp, r + 1, f"{media + ef + rng.normal(0, 0.2):.4f}"])
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, delimiter=sep).writerows(linhas)
    return linhas


def _run(args):
    return anova_cli.executar(args)


def test_recusa_sem_unidade_e_explica_pseudo_replicacao(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t)
    assert _run([str(t), "--resposta", "teor", "--grupo", "grupo"]) == 2
    err = capsys.readouterr().err
    assert "--unidade" in err and "pseudo-replicacao" in err and "replicas" in err.lower()
    assert not (tmp_path / "resultados_anova").exists()          # nada gravado


@pytest.mark.parametrize("args, trecho", [
    (["--resposta", "teor", "--unidade", "amostra"], "--grupo"),
    (["--grupo", "grupo", "--unidade", "amostra"], "--resposta"),
    (["--resposta", "teor", "--grupo", "nao_existe", "--unidade", "amostra"], "nao existe"),
    (["--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra", "--alpha", "2"], "--alpha"),
])
def test_erros_de_uso_dizem_a_causa(tmp_path, capsys, args, trecho):
    t = tmp_path / "t.csv"
    _tabela(t)
    assert _run([str(t)] + args) == 2
    assert trecho in capsys.readouterr().err


def test_valor_nao_numerico_aponta_coluna_e_linha(tmp_path, capsys):
    t = tmp_path / "t.csv"
    linhas = _tabela(t)
    linhas[5][3] = "abc"
    with open(t, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)
    assert _run([str(t), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra"]) == 2
    err = capsys.readouterr().err
    assert "'teor'" in err and "linha 6" in err and "'abc'" in err


def test_resultado_confere_com_oraculo_scipy_sobre_a_media_por_unidade(tmp_path, capsys):
    t = tmp_path / "t.csv"
    linhas = _tabela(t)
    saida = tmp_path / "res"
    assert _run([str(t), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra", "--metodo", "tukey",     # o oraculo e' o F classico
                 "--saida", str(saida), "--sem-bootstrap"]) == 0
    por_unidade = {}
    for amostra, grupo, _, y in linhas[1:]:
        por_unidade.setdefault((grupo, amostra), []).append(float(y))
    grupos = {}
    for (grupo, _), ys in por_unidade.items():
        grupos.setdefault(grupo, []).append(np.mean(ys))
    F, p = stats.f_oneway(*grupos.values())
    tab = (saida / "tabelas" / "anova_por_resposta.csv").read_text(encoding="utf-8").splitlines()
    cab, lin = tab[0].split(";"), tab[1].split(";")
    linha = dict(zip(cab, lin))
    assert float(linha["F"].replace(",", ".")) == pytest.approx(F, rel=1e-6)
    assert float(linha["p"].replace(",", ".")) == pytest.approx(p, rel=1e-4)
    assert int(linha["n_unidades"]) == 24
    assert (saida / "tabelas" / "anova_comparacoes.csv").is_file()
    assert "unidade: amostra" in (saida / "config_efetiva.yaml").read_text(encoding="utf-8")
    assert "Leia o tamanho de efeito" in capsys.readouterr().out


def test_duplicar_replicas_nao_muda_o_resultado(tmp_path):
    """Invariancia: a agregacao por unidade torna o resultado independente do
    numero de replicas por amostra fisica (se as replicas forem identicas)."""
    base = []
    rng = np.random.default_rng(3)
    for grp, media in (("A", 5.0), ("B", 6.0), ("C", 8.0)):
        for a in range(6):
            base.append((f"{grp}{a}", grp, float(media + rng.normal(0, 0.7))))
    res = []
    for n_rep in (1, 4):
        t = tmp_path / f"t{n_rep}.csv"
        with open(t, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["amostra", "grupo", "teor"])
            for a, grp, y in base:
                for _ in range(n_rep):
                    w.writerow([a, grp, f"{y:.6f}"])
        saida = tmp_path / f"r{n_rep}"
        assert _run([str(t), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra",
                     "--saida", str(saida), "--sem-bootstrap"]) == 0
        linha = (saida / "tabelas" / "anova_por_resposta.csv").read_text(encoding="utf-8").splitlines()[1].split(";")
        res.append(linha[:6])
    assert res[0] == res[1]


def test_pseudo_replicacao_infla_a_significancia_se_a_unidade_for_a_linha():
    """Sob H0 (sem diferenca), declarar a LINHA como unidade (id unico por linha)
    da p bem menor que declarar a amostra fisica. Em memoria (sem CSV nem CLI): o
    mesmo calculo que o `guaraci anova` faz (Welch/Games-Howell, sem bootstrap, sem pos-teste: so' o p da ANOVA interessa), 30 repeticoes
    com os mesmos dados de antes (y arredondado em 5 casas, como na tabela). A fiacao da
    coluna `--unidade` na CLI e' coberta pelos testes de uso acima."""
    from guaraci.anova_grupos import anova_uma_via
    rng = np.random.default_rng(11)
    p_linha, p_fisica = [], []
    for _ in range(30):
        grupo, amostra, linha, y = [], [], [], []
        k = 0
        for grp in ("A", "B", "C"):
            for a in range(5):
                ef = rng.normal(0, 1.0)
                for _r in range(6):
                    grupo.append(grp)
                    amostra.append(f"{grp}{a}")
                    linha.append(f"L{k}")
                    y.append(round(ef + rng.normal(0, 0.1), 5))
                    k += 1
        y = np.asarray(y)
        for unidade, alvo in ((amostra, p_fisica), (linha, p_linha)):
            r = anova_uma_via(y, grupo, unidade, nome_resposta="teor", bloco=None, alpha=0.05,
                              n_boot=0, variancias="desiguais", calcular_pos_teste=False)
            alvo.append(float(r.p))
    assert np.mean(np.array(p_linha) < 0.05) > 3 * max(np.mean(np.array(p_fisica) < 0.05), 0.03)


def test_main_encaminha_o_subcomando_e_sai_com_codigo_2(tmp_path):
    t = tmp_path / "t.csv"
    _tabela(t)
    with pytest.raises(SystemExit) as e:
        g.main(["anova", str(t), "--resposta", "teor", "--grupo", "grupo"])
    assert e.value.code == 2


def test_virgula_decimal_e_ponto_e_virgula_sao_aceitos(tmp_path):
    t = tmp_path / "t.csv"
    linhas = _tabela(t, sep=";")
    texto = t.read_text(encoding="utf-8").replace(".", ",")
    t.write_text(texto, encoding="utf-8")
    assert _run([str(t), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra",
                 "--saida", str(tmp_path / "r"), "--sem-bootstrap"]) == 0
    assert os.path.isfile(tmp_path / "r" / "relatorio" / "anova_resumo.txt")
    assert len(linhas) > 10
