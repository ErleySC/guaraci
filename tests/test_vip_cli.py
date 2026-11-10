"""`guaraci vip` de ponta a ponta: CSV -> tabela, figura e relatorio; erros em PT e EN."""
import csv

import numpy as np
import pytest

from guaraci import cli_comandos as cmd
from guaraci import exemplos as ex
from guaraci import vip_cli


def _tabela(caminho, n=8, rep=3, seed=7, alvo="classe"):
    rng = np.random.default_rng(seed)
    cab = ["amostra", "classe", "conc"] + [f"v{i:02d}" for i in range(8)]
    linhas = [cab]
    for c, d in (("A", -2.0), ("B", 0.0), ("C", 2.0)):
        for a in range(n):
            base = rng.normal(0, 1, 8)
            base[:2] += d
            for _ in range(rep):
                x = base + rng.normal(0, 0.1, 8)
                linhas.append([f"{c}{a}", c, f"{d + rng.normal(0, 0.1):.4f}"] + [f"{v:.4f}" for v in x])
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)


def _run(*a):
    return vip_cli.executar([str(x) for x in a])


def test_classificacao_gera_tabela_figura_relatorios_e_texto_honesto(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t)
    s = tmp_path / "r"
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", ",".join(f"v{i:02d}" for i in range(8)),
                "--repeticoes", "3", "--saida", s) == 0
    out = capsys.readouterr().out
    assert "regra pratica" in out and "soma de VIP^2 = 8.0000" in out and "estavel acima" in out
    for f in ("tabelas/vip_estabilidade.csv", "figuras/vip_estabilidade.png", "relatorio/vip_resumo.txt",
              "relatorio/vip_relatorio.md", "relatorio/vip_relatorio.html", "relatorio/vip_relatorio.tex",
              "config_efetiva.yaml"):
        assert (s / f).is_file(), f
    with open(s / "tabelas" / "vip_estabilidade.csv", encoding="utf-8", newline="") as fh:
        linhas = list(csv.DictReader(fh, delimiter=";"))
    por_var = {x["variavel"]: float(x["frequencia_vip_maior_limiar"].replace(",", ".")) for x in linhas}
    assert por_var["v00"] >= 0.8 and por_var["v01"] >= 0.8
    md = (s / "relatorio" / "vip_relatorio.md").read_text(encoding="utf-8")
    assert "(../figuras/vip_estabilidade.png)" in md and "aninhada" in md
    assert (s / "figuras" / "vip_estabilidade.png").read_bytes()[:4] == b"\x89PNG"


def test_regressao_e_sem_figuras_e_sem_grupo_avisa(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t)
    s = tmp_path / "r"
    assert _run(t, "--resposta", "conc", "--variaveis", ",".join(f"v{i:02d}" for i in range(8)),
                "--repeticoes", "2", "--sem-figuras", "--saida", s) == 0
    out = capsys.readouterr().out
    assert "sem coluna de grupo" in out and not (s / "figuras").exists()
    assert "figuras: false" in (s / "config_efetiva.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize("idioma, p1, p2, p3, pref", [
    ("pt", "informe --classe", "nao existe na tabela", "nao numerico", "Erro:"),
    ("en", "provide --classe", "does not exist in the table", "non-numeric", "Error:"),
])
def test_erros_em_pt_e_en(tmp_path, capsys, idioma, p1, p2, p3, pref):
    t = tmp_path / "t.csv"
    _tabela(t)
    assert _run(t, "--grupo", "amostra", "--idioma", idioma) == 2
    err = capsys.readouterr().err
    assert err.startswith(pref) and p1 in err
    assert _run(t, "--classe", "naoexiste", "--grupo", "amostra", "--idioma", idioma) == 2
    assert p2 in capsys.readouterr().err
    linhas = list(csv.reader(open(t, encoding="utf-8")))
    linhas[4][5] = "abc"
    with open(t, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", "v02,v03,v04", "--idioma", idioma) == 2
    assert p3 in capsys.readouterr().err


@pytest.mark.parametrize("extra, trecho_pt, trecho_en", [
    (["--classe", "classe", "--resposta", "conc"], "nao os dois", "not both"),
    (["--classe", "classe", "--componentes", "0"], "inteiros positivos", "positive integers"),
    (["--classe", "classe", "--componentes", "99"], "n_componentes|fora", "range"),
])
def test_opcoes_invalidas(tmp_path, capsys, extra, trecho_pt, trecho_en):
    t = tmp_path / "t.csv"
    _tabela(t)
    for idioma, trecho in (("pt", trecho_pt), ("en", trecho_en)):
        assert _run(t, "--grupo", "amostra", *extra, "--idioma", idioma) == 2
        err = capsys.readouterr().err
        assert any(x in err for x in trecho.split("|"))


def test_um_grupo_so_e_recusado_nos_dois_idiomas(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tabela(t)
    linhas = list(csv.reader(open(t, encoding="utf-8")))
    for i in range(1, len(linhas)):
        linhas[i][0] = "mesmo"
    with open(t, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", "v00,v01,v02") == 2
    assert "menos de 2 grupos" in capsys.readouterr().err
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", "v00,v01,v02", "--idioma", "en") == 2
    assert "fewer than 2 independent groups" in capsys.readouterr().err


def test_ajuda_arquivo_ausente_e_despacho_pelo_main(tmp_path, capsys):
    assert _run("--ajuda") == 0 and "regra pratica" in capsys.readouterr().out
    assert _run("--ajuda", "--idioma", "en") == 0 and "rule of thumb" in capsys.readouterr().out
    assert _run("--idioma", "en") == 2 and "provide the CSV table" in capsys.readouterr().err
    assert _run("nao_existe.csv", "--classe", "c", "--idioma", "en") == 2 and "file not found" in capsys.readouterr().err
    assert _run("x.csv") == 2 and "informe --classe" in capsys.readouterr().err
    assert "vip" in cmd.SUBCOMANDOS
    assert cmd.subcomando("vip", ["--ajuda"]) == 0


def test_exemplo_vip_gera_tabela_e_roda_pelo_executor(tmp_path, capsys):
    r = ex.gerar("vip", str(tmp_path / "e"))
    assert r["arquivos"] == ["tabela.csv"]
    assert cmd._rodar_vip_exemplo(r["pasta"]) == 0
    assert (tmp_path / "e" / "resultados_vip" / "tabelas" / "vip_estabilidade.csv").is_file()
    assert cmd.EXECUTORES_SEM_CONFIG["vip"] is cmd._rodar_vip_exemplo
    assert cmd._cmd_exemplo(["vip", str(tmp_path / "e2"), "--rodar"]) == 0
    assert "guaraci vip tabela.csv" in capsys.readouterr().out
