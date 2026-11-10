"""`guaraci anova` de ponta a ponta: CSV -> relatorio -> figura, metodos, erros PT/EN."""
import csv

import numpy as np
import pytest
from scipy import stats

from guaraci import anova_cli
from guaraci.anova_cli import letras_de_grupos


def _tabela(caminho, *, desvios=(0.8, 0.8, 0.8), medias=(10.0, 11.5, 14.0), n=8, seed=1, lognormal=False):
    rng = np.random.default_rng(seed)
    linhas = [["amostra", "grupo", "teor", "outro"]]
    for (grp, m), sd in zip(zip("ABC", medias), desvios):
        for a in range(n):
            y = rng.lognormal(np.log(m) / 3, sd) if lognormal else rng.normal(m, sd)
            linhas.append([f"{grp}{a:02d}", grp, f"{y:.4f}", f"{rng.normal(5, 1):.4f}"])
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)
    return linhas


def _run(*args):
    return anova_cli.executar([str(a) for a in args])


BASE = ["--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra", "--sem-bootstrap"]


@pytest.mark.parametrize("metodo, pos", [("tukey", "Tukey"), ("games-howell", "Games-Howell"),
                                          ("kruskal-dunn", "Dunn"), ("auto", "Tukey"), ("recomendar", "")])
def test_cada_metodo_gera_tabelas_relatorios_e_figura(tmp_path, capsys, metodo, pos):
    t = tmp_path / "t.csv"
    _tabela(t)
    s = tmp_path / "r"
    assert _run(t, *BASE, "--metodo", metodo, "--saida", s) == 0
    out = capsys.readouterr().out
    assert pos in out
    for f in ("tabelas/anova_por_resposta.csv", "tabelas/anova_comparacoes.csv", "relatorio/anova_resumo.txt",
              "relatorio/anova_relatorio.md", "relatorio/anova_relatorio.html", "relatorio/anova_relatorio.tex",
              "figuras/anova_teor.png", "config_efetiva.yaml"):
        assert (s / f).is_file(), f
    md = (s / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    assert "![" in md and "(../figuras/anova_teor.png)" in md             # caminho relativo
    assert "| grupo | n |" in md and "porteiro" in md
    assert "../figuras/anova_teor.png" in (s / "relatorio" / "anova_relatorio.html").read_text(encoding="utf-8")
    tex = (s / "relatorio" / "anova_relatorio.tex").read_text(encoding="utf-8")
    assert "\\includegraphics" in tex and "\\begin{tabular}" in tex and "\\end{document}" in tex
    assert (s / "figuras" / "anova_teor.png").read_bytes()[:4] == b"\x89PNG"
    cfg = (s / "config_efetiva.yaml").read_text(encoding="utf-8")
    assert f"metodo: {metodo}" in cfg and "troca_automatica_de_metodo: nenhuma" in cfg


def test_kruskal_traz_a_frase_obrigatoria_a_regra_do_ajuste_e_confere_com_scipy(tmp_path, capsys):
    t = tmp_path / "t.csv"
    linhas = _tabela(t)
    s = tmp_path / "r"
    assert _run(t, *BASE, "--metodo", "kruskal-dunn", "--ajuste", "bh", "--saida", s) == 0
    out = capsys.readouterr().out
    assert "dominancia estocastica" in out and "medianas" in out
    md = (s / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    assert "Benjamini-Hochberg" in md and "postos globais" in md and "dominancia estocastica" in md
    grupos = {}
    for amostra, grupo, y, _ in linhas[1:]:
        grupos.setdefault(grupo, []).append(float(y))
    ref = stats.kruskal(*grupos.values())
    with open(s / "tabelas" / "anova_por_resposta.csv", encoding="utf-8", newline="") as fh:
        linha = next(csv.DictReader(fh, delimiter=";"))
    assert float(linha["H"].replace(",", ".")) == pytest.approx(ref.statistic, rel=1e-6)
    assert linha["pos_teste"] == "Dunn (bh)"


def test_sem_figuras_nao_gera_png_e_o_relatorio_nao_cita_figura(tmp_path):
    t = tmp_path / "t.csv"
    _tabela(t)
    s = tmp_path / "r"
    assert _run(t, *BASE, "--sem-figuras", "--saida", s) == 0
    assert not (s / "figuras").exists()
    assert "![" not in (s / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    assert "figuras: false" in (s / "config_efetiva.yaml").read_text(encoding="utf-8")


def test_recomendar_com_variancias_1_para_10_diz_a_razao_e_roda_games_howell(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t, desvios=(0.3, 3.0, 1.0), n=12)
    assert _run(t, *BASE, "--metodo", "recomendar", "--saida", tmp_path / "r") == 0
    out = capsys.readouterr().out
    assert "recomendacao: variancias muito diferentes" in out
    assert "Welch ANOVA" in out and "[Games-Howell]" in out
    assert "recomendacao nao aplicada" not in out                 # a sugestao coincide com o padrao


def test_recomendar_nao_troca_o_padrao_e_diz_o_que_foi_sugerido(tmp_path, capsys):
    """A recomendacao perdeu para o padrao na simulacao: ela informa, nao decide."""
    t = tmp_path / "t.csv"
    _tabela(t, n=15, seed=1)                                       # variancias parecidas: o diagnostico sugere Tukey
    s = tmp_path / "r"
    assert _run(t, *BASE, "--metodo", "recomendar", "--saida", s) == 0
    out = capsys.readouterr().out
    assert "recomendacao nao aplicada: sugerido ANOVA/Tukey; rodado o padrao ANOVA/Games-Howell" in out
    assert "--metodo tukey" in out
    assert "games-howell" in (s / "config_efetiva.yaml").read_text(encoding="utf-8")


def test_sem_metodo_o_padrao_e_games_howell_e_o_config_diz_que_nao_foi_pedido(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t, n=15, seed=1)                                       # variancias parecidas: o auto escolheria Tukey
    s = tmp_path / "r"
    assert _run(t, *BASE, "--saida", s) == 0
    out = capsys.readouterr().out
    assert "[Games-Howell]" in out and "Tukey-Kramer" not in out
    cfg = (s / "config_efetiva.yaml").read_text(encoding="utf-8")
    assert "metodo: games-howell" in cfg and "metodo_pedido_pelo_usuario: nao (padrao)" in cfg


def test_tukey_so_roda_a_pedido(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t)
    s = tmp_path / "r"
    assert _run(t, *BASE, "--metodo", "tukey", "--saida", s) == 0
    assert "Tukey" in capsys.readouterr().out
    assert "metodo_pedido_pelo_usuario: sim" in (s / "config_efetiva.yaml").read_text(encoding="utf-8")


def test_com_bloco_o_padrao_e_auto(tmp_path):
    from guaraci.anova_cli import metodo_padrao
    assert metodo_padrao(False) == "games-howell" and metodo_padrao(True) == "auto"


def test_metodo_pedido_diferente_do_recomendado_e_mantido_e_avisado(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t, desvios=(0.3, 3.0, 1.0), n=12)
    assert _run(t, *BASE, "--metodo", "tukey", "--saida", tmp_path / "r") == 0
    out = capsys.readouterr().out
    assert "mantido o pedido" in out and "ANOVA/Games-Howell" in out
    assert "[Tukey" in out or "Tukey-Kramer" in out


def test_auto_declara_a_troca_de_levene_na_saida(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t, desvios=(0.1, 3.0, 8.0), n=12)
    assert _run(t, *BASE, "--metodo", "auto", "--saida", tmp_path / "r") == 0
    assert "Welch" in capsys.readouterr().out


def test_varias_respostas_aplicam_fdr_entre_elas_tambem_no_kruskal(tmp_path):
    t = tmp_path / "t.csv"
    _tabela(t)
    s = tmp_path / "r"
    assert _run(t, "--resposta", "teor", "--resposta", "outro", "--grupo", "grupo", "--unidade", "amostra",
                "--metodo", "kruskal-dunn", "--saida", s) == 0
    with open(s / "tabelas" / "anova_por_resposta.csv", encoding="utf-8", newline="") as fh:
        linhas = list(csv.DictReader(fh, delimiter=";"))
    ps = [float(x["p"].replace(",", ".")) for x in linhas]
    pf = [float(x["p_fdr"].replace(",", ".")) for x in linhas]
    ref = stats.false_discovery_control(ps, method="bh")
    assert pf == pytest.approx(list(ref), rel=1e-5)
    assert (s / "figuras" / "anova_outro.png").is_file()


# ------------------------------------------------------------- erros PT / EN

def _tabela_um_grupo(caminho):
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["amostra", "grupo", "teor"])
        for i in range(6):
            w.writerow([f"a{i}", "A", f"{i + 1.5}"])


@pytest.mark.parametrize("idioma, trecho_um_grupo, trecho_ausente, trecho_texto, prefixo", [
    ("pt", "menos de 2 grupos", "nao existe na tabela", "valor nao numerico", "Erro:"),
    ("en", "fewer than 2 groups", "does not exist in the table", "non-numeric value", "Error:"),
])
def test_erros_de_entrada_em_pt_e_en_com_a_acao_sugerida(tmp_path, capsys, idioma, trecho_um_grupo,
                                                         trecho_ausente, trecho_texto, prefixo):
    um = tmp_path / "um.csv"
    _tabela_um_grupo(um)
    assert _run(um, *BASE, "--idioma", idioma, "--saida", tmp_path / "r") == 2
    err = capsys.readouterr().err
    assert err.startswith(prefixo) and trecho_um_grupo in err
    assert not (tmp_path / "r").exists()
    t = tmp_path / "t.csv"
    linhas = _tabela(t)
    assert _run(t, "--resposta", "teor", "--grupo", "inexistente", "--unidade", "amostra", "--idioma", idioma) == 2
    assert trecho_ausente in capsys.readouterr().err
    linhas[3][2] = "abc"
    with open(t, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)
    assert _run(t, *BASE, "--idioma", idioma) == 2
    err = capsys.readouterr().err
    assert trecho_texto in err and "'abc'" in err and "'teor'" in err


def test_sem_unidade_em_ingles_explica_pseudo_replicacao(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t)
    assert _run(t, "--resposta", "teor", "--grupo", "grupo", "--idioma", "en") == 2
    err = capsys.readouterr().err
    assert "pseudo-replication" in err and "--unidade" in err


@pytest.mark.parametrize("extra, trecho", [
    (["--metodo", "bonferroni"], "--metodo"),
    (["--ajuste", "xx"], "--ajuste"),
    (["--bloco", "outro", "--metodo", "games-howell"], "--bloco"),
    (["--bloco", "outro", "--metodo", "kruskal-dunn"], "--bloco"),
])
def test_opcoes_invalidas_dizem_o_que_fazer(tmp_path, capsys, extra, trecho):
    t = tmp_path / "t.csv"
    _tabela(t)
    assert _run(t, *BASE, *extra) == 2
    assert trecho in capsys.readouterr().err


def test_ajuda_e_tabela_ausente_nos_dois_idiomas(tmp_path, capsys):
    assert _run("--ajuda") == 0 and "kruskal-dunn" in capsys.readouterr().out
    assert _run("--ajuda", "--idioma", "en") == 0 and "Usage" in capsys.readouterr().out
    assert _run("--idioma=en") == 2 and "provide the CSV table" in capsys.readouterr().err
    assert _run("nao_existe.csv", *BASE, "--idioma", "en") == 2 and "file not found" in capsys.readouterr().err
    assert _run("--alpha", "x", "--idioma", "en") == 2 and "invalid argument" in capsys.readouterr().err


def test_variancia_zero_em_ingles_traduz_a_recusa_do_nucleo(tmp_path, capsys):
    t = tmp_path / "t.csv"
    with open(t, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["amostra", "grupo", "teor"])
        for g in "AB":
            for i in range(4):
                w.writerow([f"{g}{i}", g, "5"])
    assert _run(t, *BASE, "--idioma", "en") == 2
    assert "within-group variance is zero" in capsys.readouterr().err


def test_idioma_pela_variavel_de_ambiente(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GUARACI_IDIOMA", "EN")
    assert _run("--grupo", "g") == 2
    assert "provide the CSV table" in capsys.readouterr().err


# ---------------------------------------------------------------- letras (CLD)

def _c(a, b, p):
    return {"a": a, "b": b, "p_ajustado": p}


def test_letras_sem_diferenca_todos_a():
    r = letras_de_grupos(["x", "y", "z"], {"x": 1, "y": 2, "z": 3},
                         [_c("x", "y", .5), _c("x", "z", .5), _c("y", "z", .5)], 0.05)
    assert set(r.values()) == {"a"}


def test_letras_tres_grupos_separados():
    r = letras_de_grupos(["x", "y", "z"], {"x": 1, "y": 2, "z": 3},
                         [_c("x", "y", .01), _c("x", "z", .01), _c("y", "z", .01)], 0.05)
    assert r == {"z": "a", "y": "b", "x": "c"}


def test_letras_sobrepostas_x_z_diferem_mas_y_e_ponte():
    r = letras_de_grupos(["x", "y", "z"], {"x": 1, "y": 2, "z": 3},
                         [_c("x", "y", .5), _c("x", "z", .01), _c("y", "z", .5)], 0.05)
    assert r == {"z": "a", "y": "ab", "x": "b"}
    # regra: quem compartilha letra nao difere; quem nao compartilha difere
    assert not set(r["x"]) & set(r["z"]) and set(r["x"]) & set(r["y"]) and set(r["y"]) & set(r["z"])


def test_letras_p_igual_a_alpha_nao_e_significativo():
    r = letras_de_grupos(["x", "y"], {"x": 1, "y": 2}, [_c("x", "y", 0.05)], 0.05)
    assert r["x"] == r["y"] == "a"


def test_menu_comparar_pergunta_o_metodo_e_o_repassa_ao_comando(tmp_path, monkeypatch, capsys):
    import guaraci.guaraci as g
    t = tmp_path / "t.csv"
    _tabela(t)
    respostas = iter([str(t), "teor", "grupo", "amostra", "Kruskal-Dunn"])
    chamadas = []
    monkeypatch.setattr(g, "_perguntar", lambda *a, **k: next(respostas))
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")   # opcoes extras: Enter = nenhuma
    original = anova_cli.executar
    monkeypatch.setattr(anova_cli, "executar", lambda args: chamadas.append(list(args)) or original(args))
    g._menu_comparar(g.Config())
    assert "--metodo" in chamadas[0] and chamadas[0][chamadas[0].index("--metodo") + 1] == "kruskal-dunn"
    assert (tmp_path / "resultados_anova" / "relatorio" / "anova_relatorio.md").is_file()
    assert "Dunn" in capsys.readouterr().out
