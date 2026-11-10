"""Lacunas de ramo do ANOVA, do Kruskal e do VIP: cada ramo raro tem um teste."""
import csv

import numpy as np
import pytest

from guaraci import anova_cli, vip_cli, vip_estabilidade as ve
from guaraci.anova_grupos import anova_uma_via, anova_varias_respostas


def _csv(caminho, linhas):
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)


def _tab_anova(caminho, n=8, extra=None, seed=2):
    rng = np.random.default_rng(seed)
    cab = ["amostra", "grupo", "teor", "bloco"] + list(extra or [])
    linhas = [cab]
    for g, m in zip("ABC", (10, 11.5, 14)):
        for a in range(n):
            linha = [f"{g}{a}", g, f"{rng.normal(m, 0.8):.4f}", f"b{a % 2}"]
            for e in (extra or []):
                linha.append(f"{rng.normal(5, 1):.4f}" if e != "constante" else "5")
            linhas.append(linha)
    _csv(caminho, linhas)


# ------------------------------------------------------------------ anova_grupos

def test_bootstrap_descarta_reamostragem_degenerada_e_ainda_da_ic():
    y = [1.0, 2.0, 5.0, 6.0]
    r = anova_uma_via(y, ["a", "a", "b", "b"], ["u0", "u1", "u2", "u3"], n_boot=300, seed=1)
    assert np.isfinite(r.eta2_ic95[0]) and np.isfinite(r.eta2_ic95[1])


def test_shapiro_e_pulado_com_mais_de_5000_unidades():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 3000), rng.normal(0.1, 1, 3000)])
    g = ["a"] * 3000 + ["b"] * 3000
    r = anova_uma_via(y, g, [f"u{i}" for i in range(6000)], n_boot=0, calcular_pos_teste=False)
    assert r.shapiro_p is None


def test_unidade_em_dois_blocos_e_erro():
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    g = ["a"] * 4 + ["b"] * 4
    u = [f"u{i}" for i in range(8)]
    b = ["x", "y"] * 4
    ub = list(u)
    ub[1] = ub[0]            # a mesma unidade em linhas de blocos diferentes
    g2 = list(g)
    with pytest.raises(ValueError, match="mais de um bloco"):
        anova_uma_via(y, g2, ub, bloco=b, n_boot=0)


def test_varias_respostas_todas_recusadas_nao_aplica_fdr():
    g = ["a"] * 4 + ["b"] * 4
    u = [f"u{i}" for i in range(8)]
    out = anova_varias_respostas({"c1": [5.0] * 8, "c2": [3.0] * 8}, g, u, n_boot=0)
    assert out["resultados"] == {} and set(out["recusadas"]) == {"c1", "c2"}


# ------------------------------------------------------------------ anova_cli

def test_erro_do_nucleo_desconhecido_em_ingles_e_marcado_como_portugues():
    anova_cli._ESTADO["lang"] = "en"
    try:
        assert "message in Portuguese" in str(anova_cli._erro_nucleo(ValueError("algo novo")))
    finally:
        anova_cli._ESTADO["lang"] = "pt"


def test_arquivo_vazio_e_celula_vazia(tmp_path, capsys):
    vazio = tmp_path / "v.csv"
    vazio.write_text("", encoding="utf-8")
    assert anova_cli.executar([str(vazio), "--resposta", "x", "--grupo", "g", "--unidade", "u"]) == 2
    assert "nao consegui ler" in capsys.readouterr().err
    t = tmp_path / "t.csv"
    _tab_anova(t)
    linhas = list(csv.reader(open(t, encoding="utf-8")))
    linhas[3][2] = ""
    _csv(t, linhas)
    assert anova_cli.executar([str(t), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra",
                               "--sem-bootstrap", "--sem-figuras", "--saida", str(tmp_path / "r")]) == 0
    assert "descartada" in capsys.readouterr().out


def test_recomendar_com_bloco_nao_aplica_a_recomendacao(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab_anova(t, n=12)
    assert anova_cli.executar([str(t), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra",
                               "--bloco", "bloco", "--metodo", "recomendar", "--sem-bootstrap", "--sem-figuras",
                               "--saida", str(tmp_path / "r")]) == 0
    assert "recomendacao nao se aplica" in capsys.readouterr().out


def test_recomendar_com_um_grupo_so_cai_no_padrao_e_recusa(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _csv(t, [["amostra", "grupo", "teor"]] + [[f"a{i}", "A", str(i + 1.5)] for i in range(6)])
    assert anova_cli.executar([str(t), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra",
                               "--metodo", "recomendar"]) == 2
    assert "menos de 2 grupos" in capsys.readouterr().err


def test_resposta_recusada_aparece_como_nao_testada_no_resumo_e_no_relatorio(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab_anova(t, extra=["constante"])
    s = tmp_path / "r"
    assert anova_cli.executar([str(t), "--resposta", "teor", "--resposta", "constante", "--grupo", "grupo",
                               "--unidade", "amostra", "--sem-bootstrap", "--sem-figuras", "--saida", str(s)]) == 0
    assert "NAO TESTADA" in capsys.readouterr().out
    assert "NAO TESTADA" in (s / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")


def test_mais_de_30_respostas_limita_as_figuras(tmp_path, capsys):
    t = tmp_path / "t.csv"
    extra = [f"e{i:02d}" for i in range(31)]
    _tab_anova(t, n=4, extra=extra)
    s = tmp_path / "r"
    assert anova_cli.executar([str(t), *sum((["--resposta", e] for e in extra), []), "--grupo", "grupo",
                               "--unidade", "amostra", "--sem-bootstrap", "--saida", str(s)]) == 0
    assert "so' para as 30 primeiras" in capsys.readouterr().out
    assert len(list((s / "figuras").glob("*.png"))) == 30


# ------------------------------------------------------------------ vip_estabilidade

def _d(seed=0, n=24, rep=2, p=6):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n), rep).astype(str)
    X = rng.normal(0, 1, (n, p))[np.repeat(np.arange(n), rep)] + rng.normal(0, 0.05, (n * rep, p))
    y = X[:, 0] + rng.normal(0, 0.3, n * rep)
    return X, y, g


def test_sem_autoescala_e_coluna_constante_com_autoescala():
    X, y, g = _d()
    r = ve.vip_estabilidade(X, y, 1, g, tarefa="regressao", n_repeticoes=1, autoescalar=False)
    assert r["soma_vip2"] == pytest.approx(6, rel=1e-8)
    X2 = X.copy()
    X2[:, 3] = 7.0
    r2 = ve.vip_estabilidade(X2, y, 1, g, tarefa="regressao", n_repeticoes=1)
    assert np.all(np.isfinite(r2["media"]))


def test_ajustes_de_dobra_que_falham_ficam_de_fora_com_aviso(monkeypatch):
    X, y, g = _d()
    original = ve.vip_scores
    chamadas = {"n": 0}

    def instavel(modelo):
        chamadas["n"] += 1
        if chamadas["n"] % 3 == 0:
            raise ValueError("falha simulada")
        v = original(modelo)
        if chamadas["n"] % 3 == 1 and chamadas["n"] > 1:
            v = v * np.nan
        return v

    monkeypatch.setattr(ve, "vip_scores", instavel)
    r = ve.vip_estabilidade(X, y, 1, g, tarefa="regressao", n_repeticoes=2)
    assert r["n_falhos"] > 0 and any("falharam" in a for a in r["avisos"])


def test_nenhum_ajuste_de_dobra_converge_e_erro(monkeypatch):
    X, y, g = _d()
    original = ve.vip_scores
    estado = {"primeira": True}

    def so_a_primeira(modelo):          # o ajuste completo passa; todos os das dobras falham
        if estado["primeira"]:
            estado["primeira"] = False
            return original(modelo)
        raise np.linalg.LinAlgError("falha simulada")

    monkeypatch.setattr(ve, "vip_scores", so_a_primeira)
    with pytest.raises(ValueError, match="nenhum ajuste"):
        ve.vip_estabilidade(X, y, 1, g, tarefa="regressao", n_repeticoes=1)


def test_tarefa_regressao_com_inf_na_resposta_e_recusada():
    X, y, g = _d()
    y[0] = np.inf
    with pytest.raises(ValueError, match="NaN/inf"):
        ve.vip_estabilidade(X, y, 1, g, tarefa="regressao")


# ------------------------------------------------------------------ vip_cli

def test_vip_cli_arquivo_vazio_argumento_ruim_e_sem_variaveis(tmp_path, capsys):
    vazio = tmp_path / "v.csv"
    vazio.write_text("", encoding="utf-8")
    assert vip_cli.executar([str(vazio), "--classe", "c"]) == 2
    assert "nao consegui ler" in capsys.readouterr().err
    t = tmp_path / "t.csv"
    _csv(t, [["amostra", "classe"]] + [[f"a{i}", "AB"[i % 2]] for i in range(8)])
    assert vip_cli.executar([str(t), "--classe", "classe", "--grupo", "amostra"]) == 2
    assert "nenhuma coluna de variavel" in capsys.readouterr().err
    assert vip_cli.executar([str(t), "--classe", "classe", "--componentes", "abc"]) == 2


def test_vip_cli_mais_de_20_variaveis_e_idioma_por_ambiente(tmp_path, capsys, monkeypatch):
    rng = np.random.default_rng(0)
    cab = ["amostra", "classe"] + [f"v{i:02d}" for i in range(25)]
    linhas = [cab]
    for c, d in (("A", -2.0), ("B", 2.0)):
        for a in range(10):
            base = rng.normal(0, 1, 25)
            base[0] += d
            for _ in range(2):
                linhas.append([f"{c}{a}", c] + [f"{x:.4f}" for x in base + rng.normal(0, 0.1, 25)])
    t = tmp_path / "t.csv"
    _csv(t, linhas)
    monkeypatch.setenv("GUARACI_IDIOMA", "en")
    assert vip_cli.executar([str(t), "--classe", "classe", "--grupo", "amostra", "--repeticoes", "2",
                             "--saida", str(tmp_path / "r")]) == 0
    out = capsys.readouterr().out
    assert "first 20 of 25" in (tmp_path / "r" / "relatorio" / "vip_relatorio.md").read_text(encoding="utf-8")
    assert "VIP with stability" in out
    vip_cli._ESTADO["lang"] = "pt"


def test_vip_cli_erro_do_nucleo_desconhecido_em_ingles():
    vip_cli._ESTADO["lang"] = "en"
    try:
        assert "refused by the calculation" in str(vip_cli._erro_nucleo(ValueError("algo novo")))
    finally:
        vip_cli._ESTADO["lang"] = "pt"
