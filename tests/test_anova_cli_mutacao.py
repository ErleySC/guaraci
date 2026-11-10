"""Testes escritos para matar sobreviventes da mutacao de `anova_cli.py`: conteudo da saida, da figura e das bordas."""
import csv
import sys

import numpy as np
import pytest

from guaraci import anova_cli
from guaraci.anova_cli import letras_de_grupos
from guaraci.anova_grupos import TEXTO_KRUSKAL, anova_uma_via

BASE = ["--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra"]


def _tab(caminho, medias=(10.0, 10.2, 15.0), dp=(0.5, 0.5, 0.5), n=8, seed=1, extra_nulo=False, grupos="ABC"):
    rng = np.random.default_rng(seed)
    linhas = [["amostra", "grupo", "teor", "nulo"]]
    for g, m, s in zip(grupos, medias, dp):
        for a in range(n):
            linhas.append([f"{g}{a}", g, f"{rng.normal(m, s):.4f}", f"{rng.normal(5, 1):.4f}"])
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)
    return linhas


def _run(*a):
    return anova_cli.executar([str(x) for x in a])


def _lin(caminho):
    with open(caminho, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def _num(s):
    return float(s.replace(",", "."))


# ---------------------------------------------------------------- letras (CLD): propriedade

@pytest.mark.parametrize("seed", range(40))
def test_letras_compartilhadas_se_e_so_se_o_par_nao_difere(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(3, 8))
    nomes = [f"g{i}" for i in range(k)]
    medias = {n: float(rng.normal()) for n in nomes}
    comps = [{"a": nomes[i], "b": nomes[j], "p_ajustado": float(rng.choice([0.001, 0.5]))}
             for i in range(k) for j in range(i + 1, k)]
    letras = letras_de_grupos(nomes, medias, comps, 0.05)
    for c in comps:
        compartilha = bool(set(letras[c["a"]]) & set(letras[c["b"]]))
        assert compartilha == (c["p_ajustado"] >= 0.05), (seed, c, letras)
    por_letra = {}
    for n, ls in letras.items():
        for ch in ls:
            por_letra.setdefault(ch, set()).add(n)
    grupos = list(por_letra.values())
    for i, a in enumerate(grupos):                     # nenhuma letra e' subconjunto de outra (cobertura minima)
        assert not any(a < b for b in grupos), (seed, grupos)
        assert not any(a == b for j, b in enumerate(grupos) if j != i), (seed, grupos)


def test_letras_alem_de_26_colunas_voltam_ao_a():
    nomes = [f"g{i:02d}" for i in range(27)]
    comps = [{"a": nomes[i], "b": nomes[j], "p_ajustado": 0.001} for i in range(27) for j in range(i + 1, 27)]
    letras = letras_de_grupos(nomes, {n: float(-i) for i, n in enumerate(nomes)}, comps, 0.05)
    assert letras["g00"] == "a" and letras["g25"] == "z" and letras["g26"] == "a"


# ---------------------------------------------------------------- saida tabular e textual

def test_csv_tem_colunas_sem_indice_metodo_e_ics_exatos_e_o_bootstrap_padrao_e_500(tmp_path):
    t = tmp_path / "t.csv"
    linhas = _tab(t)
    s = tmp_path / "r"
    assert _run(t, *BASE, "--metodo", "auto", "--saida", s, "--sem-figuras") == 0
    assert _run(t, *BASE, "--metodo", "auto", "--saida", s, "--sem-figuras") == 0   # segunda execucao na mesma pasta nao quebra
    cab = (s / "tabelas" / "anova_por_resposta.csv").read_text(encoding="utf-8").split("\n")[0].split(";")
    assert cab[0] == "resposta" and "Unnamed" not in cab[0]
    cab2 = (s / "tabelas" / "anova_comparacoes.csv").read_text(encoding="utf-8").split("\n")[0].split(";")
    assert cab2[0] == "resposta"
    l1 = _lin(s / "tabelas" / "anova_por_resposta.csv")[0]
    assert l1["metodo"] == "ANOVA classica" and l1["pos_teste"] == "Tukey"
    g = {}
    for _, grp, y, _n in linhas[1:]:
        g.setdefault(grp, []).append(float(y))
    y, gr, u = zip(*[(float(r[2]), r[1], r[0]) for r in linhas[1:]])
    r = anova_uma_via(y, gr, u, n_boot=500, seed=0)
    assert _num(l1["eta2_ic95_inf"]) == pytest.approx(r.eta2_ic95[0], rel=1e-5)
    assert _num(l1["eta2_ic95_sup"]) == pytest.approx(r.eta2_ic95[1], rel=1e-5)
    c0 = _lin(s / "tabelas" / "anova_comparacoes.csv")[0]
    esperado = next(c for c in r.comparacoes if (c["a"], c["b"]) == (c0["grupo_a"], c0["grupo_b"]))
    assert _num(c0["ic95_inf"]) == pytest.approx(esperado["ic95"][0], rel=1e-5)
    assert _num(c0["ic95_sup"]) == pytest.approx(esperado["ic95"][1], rel=1e-5)
    # sem bootstrap: IC de eta2 indefinido
    s2 = tmp_path / "r2"
    assert _run(t, *BASE, "--saida", s2, "--sem-figuras", "--sem-bootstrap") == 0
    assert _lin(s2 / "tabelas" / "anova_por_resposta.csv")[0]["eta2_ic95_inf"] == ""


def test_kruskal_grava_metodo_e_nao_tem_ic_e_o_resumo_so_cita_a_frase_do_kruskal_quando_usa(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab(t)
    s = tmp_path / "k"
    assert _run(t, *BASE, "--metodo", "kruskal-dunn", "--saida", s, "--sem-figuras") == 0
    assert TEXTO_KRUSKAL in capsys.readouterr().out
    assert _lin(s / "tabelas" / "anova_por_resposta.csv")[0]["metodo"] == "Kruskal-Wallis"
    assert _lin(s / "tabelas" / "anova_comparacoes.csv")[0]["ic95_inf"] == ""
    assert TEXTO_KRUSKAL in (s / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    s2 = tmp_path / "a"
    assert _run(t, *BASE, "--metodo", "tukey", "--saida", s2, "--sem-figuras") == 0
    assert TEXTO_KRUSKAL not in capsys.readouterr().out
    assert TEXTO_KRUSKAL not in (s2 / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    assert TEXTO_KRUSKAL not in (s2 / "relatorio" / "anova_resumo.txt").read_text(encoding="utf-8")


def test_resumo_marca_diferenca_so_quando_ha_e_lista_so_os_pares_significativos(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab(t)
    assert _run(t, "--resposta", "teor", "--resposta", "nulo", "--grupo", "grupo", "--unidade", "amostra",
                "--sem-figuras", "--sem-bootstrap", "--saida", tmp_path / "r") == 0
    out = capsys.readouterr().out
    lt = next(x for x in out.split("\n") if x.startswith("- teor:"))
    ln = next(x for x in out.split("\n") if x.startswith("- nulo:"))
    assert lt.endswith("diferenca entre grupos") and ln.endswith("sem diferenca detectavel")
    assert "A x C:" in out and "B x C:" in out and "A x B:" not in out      # A (10) e B (10,2) nao diferem
    # Kruskal/Dunn: mesma regra de listagem dos pares
    assert _run(t, *BASE, "--metodo", "kruskal-dunn", "--sem-figuras", "--saida", tmp_path / "k") == 0
    out2 = capsys.readouterr().out
    assert "A x C:" in out2 and "A x B:" not in out2


def test_aviso_de_variancias_desiguais_so_aparece_quando_o_levene_rejeita(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab(t, dp=(0.1, 3.0, 8.0), n=12)
    assert _run(t, *BASE, "--sem-figuras", "--sem-bootstrap", "--saida", tmp_path / "h") == 0
    assert "variancias desiguais (Brown-Forsythe" in capsys.readouterr().out
    t2 = tmp_path / "t2.csv"
    _tab(t2, dp=(1.0, 1.0, 1.0), n=12, seed=2)
    assert _run(t2, *BASE, "--sem-figuras", "--sem-bootstrap", "--saida", tmp_path / "i") == 0
    assert "variancias desiguais (Brown-Forsythe" not in capsys.readouterr().out


def test_correcao_declarada_nos_dois_idiomas(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab(t)
    assert _run(t, *BASE, "--sem-figuras", "--sem-bootstrap", "--saida", tmp_path / "p") == 0
    assert "Benjamini-Hochberg (FDR) entre respostas" in capsys.readouterr().out
    assert _run(t, *BASE, "--sem-figuras", "--sem-bootstrap", "--idioma", "en", "--saida", tmp_path / "e") == 0
    assert "Benjamini-Hochberg (FDR) across responses" in capsys.readouterr().out


def test_linha_de_premissas_mostra_shapiro_ou_na():
    d = {"shapiro_p": None, "razao_variancias": 2.0, "brown_forsythe_p": 0.3, "assimetria": 0.1, "curtose_excesso": 0.2}
    assert "Shapiro p = n/a" in anova_cli._linha_premissas(d)
    d["shapiro_p"] = 0.0123
    assert "Shapiro p = 0.0123" in anova_cli._linha_premissas(d)


# ---------------------------------------------------------------- relatorio

def test_relatorio_md_traz_letras_comparacoes_premissas_e_recomendacao(tmp_path):
    t = tmp_path / "t.csv"
    _tab(t)
    s = tmp_path / "r"
    assert _run(t, *BASE, "--sem-bootstrap", "--saida", s) == 0
    md = (s / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    assert "| grupo | n | media | letra |" in md
    assert "| A | 8 |" in md and "Grupos com a mesma letra nao diferem" in md
    assert "### Comparacoes par a par" in md and "| grupo A | grupo B | diferenca | p ajustado | significativo |" in md
    assert "| sim |" in md and "| nao |" in md                              # A x B nao difere; A x C difere
    assert "premissas: razao de variancias" in md and "recomendacao:" in md
    k = tmp_path / "k"
    assert _run(t, *BASE, "--metodo", "kruskal-dunn", "--saida", k) == 0
    mk = (k / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    assert "| grupo | n | posto medio | letra |" in mk and "diferenca de postos" in mk
    assert "Dunn com postos globais" in mk


# ---------------------------------------------------------------- figura (inspecao dos artistas)

@pytest.fixture
def figura_capturada(monkeypatch):
    from matplotlib.figure import Figure
    guardada = {}
    original = Figure.savefig
    monkeypatch.setitem(__import__('matplotlib').rcParams, 'savefig.bbox', 'standard')   # estado global de outros testes nao entra na medida

    def capturar(self, *a, **k):
        guardada["fig"] = self
        return original(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", capturar)
    return guardada


def test_figura_tem_dois_paineis_pontos_nas_posicoes_certas_letras_acima_do_maximo_e_tamanho_150dpi(
        tmp_path, figura_capturada):
    t = tmp_path / "t.csv"
    linhas = _tab(t)
    s = tmp_path / "r"
    assert _run(t, *BASE, "--sem-bootstrap", "--saida", s) == 0
    fig = figura_capturada["fig"]
    assert len(fig.axes) == 2
    assert fig.axes[0].get_subplotspec().get_geometry()[:2] == (1, 2)
    assert fig.axes[0].get_subplotspec().num1 == 0 and fig.axes[1].get_subplotspec().num1 == 1
    ax = fig.axes[0]
    por = {g: [float(r[2]) for r in linhas[1:] if r[1] == g] for g in "ABC"}
    medias = {g: float(np.mean(v)) for g, v in por.items()}
    # pontos: a unidade e' a amostra (uma linha cada); x dentro de +-0.12 da posicao do grupo; y = valores
    pts = np.vstack([np.asarray(c.get_offsets()) for c in ax.collections])
    assert len(pts) == 24
    for i, g in enumerate("ABC", start=1):
        sel = pts[(np.abs(pts[:, 0] - i) <= 0.1200001)]
        assert sorted(np.round(sel[:, 1], 4)) == sorted(np.round(por[g], 4))
    assert len(set(np.round(pts[:, 0] - np.round(pts[:, 0]), 6))) > 1          # ha' jitter (nao colapsa)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]
    # letras: uma por grupo, acima do maximo do grupo, deslocadas em 6% da amplitude geral
    topo = max(max(v) for v in por.values())
    base = min(min(v) for v in por.values())
    esperadas = letras_de_grupos(list("ABC"), medias,
                                 [{"a": "A", "b": "B", "p_ajustado": 0.5}, {"a": "A", "b": "C", "p_ajustado": 0.001},
                                  {"a": "B", "b": "C", "p_ajustado": 0.001}], 0.05)
    textos = sorted(ax.texts, key=lambda x: x.get_position()[0])
    assert [x.get_text() for x in textos] == [esperadas[g] for g in "ABC"]
    for x, g in zip(textos, "ABC"):
        assert x.get_position()[1] == pytest.approx(max(por[g]) + 0.06 * (topo - base))
    qq = fig.axes[1]
    assert len(qq.collections[0].get_offsets()) == 24
    png = (s / "figuras" / "anova_teor.png").read_bytes()
    largura, altura = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    assert (largura, altura) == (1350, 570)                                  # 9 x 3,8 polegadas a 150 dpi


def test_boxplot_usa_os_dados_de_cada_grupo(tmp_path, figura_capturada):
    t = tmp_path / "t.csv"
    linhas = _tab(t)
    assert _run(t, *BASE, "--sem-bootstrap", "--saida", tmp_path / "r") == 0
    ax = figura_capturada["fig"].axes[0]
    medianas = sorted(round(float(ln.get_ydata()[0]), 4) for ln in ax.lines if len(ln.get_ydata()) == 2
                      and ln.get_ydata()[0] == ln.get_ydata()[1] and ln.get_xdata()[1] - ln.get_xdata()[0] < 0.6
                      and ln.get_color() in ("orange", "C1", "#ff7f0e"))
    esperado = sorted(round(float(np.median([float(r[2]) for r in linhas[1:] if r[1] == g])), 4) for g in "ABC")
    assert medianas == esperado


def test_figura_sem_comparacoes_nao_tem_letras(tmp_path, figura_capturada):
    """`_figura` com resultado sem comparacoes (calcular_pos_teste=False) nao escreve letras e sem diagnostico nao tem o Q-Q."""
    y = [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 7.0, 6.0]
    g = np.array(["a"] * 4 + ["b"] * 4, dtype=object)
    r = anova_uma_via(y, list(g), [f"u{i}" for i in range(8)], n_boot=0, calcular_pos_teste=False)
    caminho = str(tmp_path / "f.png")
    anova_cli._figura("x", np.array(y), g, r, None, caminho, 0.05)
    fig = figura_capturada["fig"]
    assert len(fig.axes) == 1 and len(fig.axes[0].texts) == 0


# ---------------------------------------------------------------- entradas e limites

def test_rotulo_de_grupo_NA_nao_vira_valor_ausente(tmp_path):
    t = tmp_path / "t.csv"
    _tab(t, grupos=("NA", "B", "C"))
    s = tmp_path / "r"
    assert _run(t, *BASE, "--sem-bootstrap", "--sem-figuras", "--saida", s) == 0
    assert "C x NA:" in (s / "relatorio" / "anova_resumo.txt").read_text(encoding="utf-8")


def test_exatamente_30_respostas_gera_30_figuras_sem_aviso_de_limite(tmp_path, capsys):
    t = tmp_path / "t.csv"
    rng = np.random.default_rng(0)
    nomes = [f"e{i:02d}" for i in range(30)]
    linhas = [["amostra", "grupo"] + nomes]
    for g in "AB":
        for a in range(4):
            linhas.append([f"{g}{a}", g] + [f"{rng.normal(0, 1):.3f}" for _ in nomes])
    with open(t, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)
    s = tmp_path / "r"
    assert _run(t, *sum((["--resposta", n] for n in nomes), []), "--grupo", "grupo", "--unidade", "amostra",
                "--sem-bootstrap", "--saida", s) == 0
    assert "so' para as" not in capsys.readouterr().out and len(list((s / "figuras").glob("*.png"))) == 30


@pytest.mark.parametrize("alpha", ["0", "1", "1.5", "-0.1"])
def test_alpha_fora_de_0_1_e_recusado(tmp_path, capsys, alpha):
    t = tmp_path / "t.csv"
    _tab(t)
    assert _run(t, *BASE, "--alpha", alpha) == 2
    assert "--alpha deve estar entre 0 e 1" in capsys.readouterr().err


def test_idioma_sem_valor_no_fim_nao_quebra_e_idioma_igual_funciona(capsys):
    assert _run("--idioma") == 2
    assert capsys.readouterr().err
    assert _run("--idioma=en") == 2 and "provide the CSV table" in capsys.readouterr().err


def test_sem_argv_explicito_le_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["guaraci", "--ajuda"])
    assert anova_cli.executar() == 0
    assert "kruskal-dunn" in capsys.readouterr().out


def test_tabela_ausente_mostra_o_uso_no_idioma_certo(capsys):
    assert _run("--grupo", "g") == 2
    assert "Uso: guaraci anova" in capsys.readouterr().err
    assert _run("--grupo", "g", "--idioma", "en") == 2
    assert "Usage: guaraci anova" in capsys.readouterr().err


# ---------------------------------------------------------------- borda exata: alfa igual ao p observado

def _core(linhas, **kw):
    from guaraci.anova_grupos import kruskal_dunn
    y, g, u = zip(*[(float(r[2]), r[1], r[0]) for r in linhas[1:]])
    return anova_uma_via(y, g, u, n_boot=0, **kw), kruskal_dunn(y, g, u)


def _col_csv(caminho, coluna, par=None):
    """Valor (texto, precisao total) de uma coluna do CSV gravado pelo proprio relatorio; `par` escolhe a linha (grupo_a, grupo_b)."""
    cab, *corpo = caminho.read_text(encoding="utf-8").strip().split("\n")
    col = cab.split(";")
    reg = next(l.split(";") for l in corpo if par is None or tuple(l.split(";")[1:3]) == par)
    return reg[col.index(coluna)].replace(",", ".")


def test_alfa_igual_ao_p_observado_nao_e_significativo(tmp_path, capsys):
    """A regra e' estritamente `p < alfa`: com alfa == p (valor exato, calculado pelo nucleo e passado com repr)
    o par/resposta NAO e' significativo."""
    t = tmp_path / "t.csv"
    linhas = _tab(t, dp=(0.5, 0.6, 0.5))
    anova, dunn = _core(linhas, variancias="iguais")
    # 1) p ajustado de um par == alfa (p minusculo: o Levene segue aceitando variancias iguais)
    # o p usado e' o que o PROPRIO relatorio gravou (precisao total no CSV): recalcular no nucleo pode diferir nos ultimos bits e deixar a
    # fronteira `p == alfa` sem ser exercida na tabela do relatorio
    s1 = tmp_path / "r1"
    assert _run(t, *BASE, "--sem-figuras", "--sem-bootstrap", "--saida", s1) == 0
    capsys.readouterr()
    p_bc = _col_csv(s1 / "tabelas" / "anova_comparacoes.csv", "p_ajustado", ("B", "C"))      # o de maior p entre os significativos
    s2 = tmp_path / "r2"
    assert _run(t, *BASE, "--sem-figuras", "--sem-bootstrap", "--alpha", p_bc, "--saida", s2) == 0
    out = capsys.readouterr().out
    assert "B x C:" not in out and "A x C:" in out
    md = (s2 / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    linha_ac = next(l for l in md.split("\n") if l.startswith("| B | C |"))
    assert linha_ac.rstrip().endswith("| nao |")
    assert next(l for l in md.split("\n") if l.startswith("| A | C |")).rstrip().endswith("| sim |")
    # 2) p(FDR) da resposta (uma so: igual ao p) == alfa: "sem diferenca detectavel"
    p_resp = _col_csv(s1 / "tabelas" / "anova_por_resposta.csv", "p_fdr")
    assert _run(t, *BASE, "--sem-figuras", "--sem-bootstrap", "--alpha", p_resp, "--saida", tmp_path / "r3") == 0
    assert next(x for x in capsys.readouterr().out.split("\n") if x.startswith("- teor:")).endswith("sem diferenca detectavel")
    # 3) Levene p == alfa: sem a mensagem de variancias desiguais
    t2 = tmp_path / "t2.csv"
    linhas2 = _tab(t2, dp=(0.1, 3.0, 8.0), n=12)
    assert _run(t2, *BASE, "--sem-figuras", "--sem-bootstrap", "--saida", tmp_path / "h") == 0
    assert "variancias desiguais (Brown-Forsythe" in capsys.readouterr().out
    lev = _core(linhas2)[0].levene_p
    assert _run(t2, *BASE, "--sem-figuras", "--sem-bootstrap", "--alpha", repr(lev), "--saida", tmp_path / "h2") == 0
    assert "variancias desiguais (Brown-Forsythe" not in capsys.readouterr().out
    # 4) Kruskal/Dunn: par com p ajustado == alfa nao e listado
    sk = tmp_path / "k1"
    assert _run(t, *BASE, "--metodo", "kruskal-dunn", "--sem-figuras", "--saida", sk) == 0
    capsys.readouterr()
    pk = _col_csv(sk / "tabelas" / "anova_comparacoes.csv", "p_ajustado", ("A", "C"))
    assert _run(t, *BASE, "--metodo", "kruskal-dunn", "--sem-figuras", "--alpha", pk, "--saida", tmp_path / "k2") == 0
    assert "A x C:" not in capsys.readouterr().out


def test_figura_segunda_execucao_na_mesma_pasta_e_layout_dos_dois_paineis(tmp_path, figura_capturada):
    t = tmp_path / "t.csv"
    _tab(t)
    s = tmp_path / "r"
    assert _run(t, *BASE, "--sem-bootstrap", "--saida", s) == 0
    assert _run(t, *BASE, "--sem-bootstrap", "--saida", s) == 0              # figuras/ ja existe
    fig = figura_capturada["fig"]
    assert [a.get_subplotspec().get_geometry()[:2] for a in fig.axes] == [(1, 2), (1, 2)]
    assert [a.get_subplotspec().num1 for a in fig.axes] == [0, 1]


def test_tabela_de_grupos_do_relatorio_tem_a_letra_na_ultima_coluna(tmp_path):
    t = tmp_path / "t.csv"
    _tab(t)
    s = tmp_path / "r"
    assert _run(t, *BASE, "--sem-bootstrap", "--sem-figuras", "--saida", s) == 0
    md = (s / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    for g, letra in (("A", "b"), ("B", "b"), ("C", "a")):
        linha = next(l for l in md.split("\n") if l.startswith(f"| {g} | 8 |"))
        celulas = [c.strip() for c in linha.strip("|").split("|")]
        assert len(celulas) == 4 and celulas[3] == letra


def test_idioma_de_aceita_as_duas_formas_a_variavel_de_ambiente_e_o_valor_com_igual(monkeypatch, tmp_path):
    monkeypatch.delenv("GUARACI_IDIOMA", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert anova_cli._idioma_de(["--idioma=en"]) == "en" and anova_cli._idioma_de(["--idioma=pt"]) == "pt"
    assert anova_cli._idioma_de(["--idioma=EN-us"]) == "en" and anova_cli._idioma_de(["--idioma=e=n"]) == "pt"
    assert anova_cli._idioma_de(["--idioma", "en"]) == "en" and anova_cli._idioma_de(["--idioma", "pt"]) == "pt"
    # sem flag, variavel nem escolha gravada: o mesmo padrao do menu (ingles)
    assert anova_cli._idioma_de(["--idioma"]) == "en" and anova_cli._idioma_de([]) == "en"
    (tmp_path / ".guaraci").mkdir()
    (tmp_path / ".guaraci" / ".cli_wizard_done").write_text("PT", encoding="utf-8")
    assert anova_cli._idioma_de([]) == "pt"   # escolha gravada com [I] no menu
    monkeypatch.setenv("GUARACI_IDIOMA", "en_US")
    assert anova_cli._idioma_de([]) == "en" and anova_cli._idioma_de(["--idioma=pt"]) == "pt"
