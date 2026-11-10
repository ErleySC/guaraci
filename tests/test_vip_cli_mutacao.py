"""Testes escritos para matar sobreviventes da mutacao de `vip_cli.py`: saida exata, bordas e padroes."""
import csv
import re
import sys

import numpy as np
import pytest

from guaraci import vip_cli
from guaraci.vip_estabilidade import vip_estabilidade


def _tab(caminho, p=8, informativas=2, n=8, rep=2, seed=3, classes=("A", "B", "C"), alvo="classe"):
    rng = np.random.default_rng(seed)
    cab = ["amostra", "classe", "conc"] + [f"v{i:02d}" for i in range(p)]
    linhas = [cab]
    for k, c in enumerate(classes):
        for a in range(n):
            base = rng.normal(0, 1, p)
            base[:informativas] += (k - 1) * 3.0
            for _ in range(rep):
                x = base + rng.normal(0, 0.1, p)
                linhas.append([f"{c}{a}", c, f"{(k - 1) * 3 + rng.normal(0, 0.1):.4f}"] + [f"{v:.4f}" for v in x])
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)
    return linhas


def _run(*a):
    return vip_cli.executar([str(x) for x in a])


def _vars(p):
    return ",".join(f"v{i:02d}" for i in range(p))


def _lin(caminho):
    with open(caminho, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def test_valores_padrao_e_tabela_alinhada_com_a_chamada_direta(tmp_path, capsys):
    t = tmp_path / "t.csv"
    linhas = _tab(t)
    s = tmp_path / "r"
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", _vars(8), "--saida", s) == 0
    cfg = (s / "config_efetiva.yaml").read_text(encoding="utf-8")
    for k, v in (("componentes", 2), ("divisoes", 5), ("repeticoes", 10), ("semente", 0), ("limiar", 1.0)):
        assert f"{k}: {v}" in cfg
    X = np.array([[float(x) for x in r[3:]] for r in linhas[1:]])
    res = vip_estabilidade(X, [r[1] for r in linhas[1:]], 2, [r[0] for r in linhas[1:]], tarefa="classificacao",
                           n_divisoes=5, n_repeticoes=10, seed=0)
    tab = _lin(s / "tabelas" / "vip_estabilidade.csv")
    assert [float(r["vip"].replace(",", ".")) for r in tab] == pytest.approx(list(res["vip"]), rel=1e-5)
    assert [float(r["frequencia_vip_maior_limiar"].replace(",", ".")) for r in tab] == pytest.approx(list(res["frequencia"]))
    cab = (s / "tabelas" / "vip_estabilidade.csv").read_text(encoding="utf-8").split("\n")[0].split(";")
    assert cab[0] == "variavel" and "Unnamed" not in cab[0]
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", _vars(8), "--saida", s) == 0   # mesma pasta


def test_resumo_nomeia_o_modelo_e_lista_as_estaveis_em_ordem_decrescente_de_vip(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab(t)
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", _vars(8), "--repeticoes", "3",
                "--saida", tmp_path / "c") == 0
    out = capsys.readouterr().out
    assert "(PLS-DA)" in out
    linha = next(x for x in out.split("\n") if "estavel acima" in x)
    assert "2 variavel(is) com VIP > 1" in linha and ("v00, v01" in linha or "v01, v00" in linha)
    assert "v02" not in linha
    assert _run(t, "--resposta", "conc", "--grupo", "amostra", "--variaveis", _vars(8), "--repeticoes", "3",
                "--saida", tmp_path / "r") == 0
    assert "(PLS)" in capsys.readouterr().out
    # a tabela do relatorio comeca pelo maior VIP medio
    md = (tmp_path / "c" / "relatorio" / "vip_relatorio.md").read_text(encoding="utf-8")
    primeira = [l for l in md.split("\n") if re.match(r"\| v\d\d ", l)][0]
    medias = {r["variavel"]: float(r["vip_medio_dobras"].replace(",", "."))
              for r in _lin(tmp_path / "c" / "tabelas" / "vip_estabilidade.csv")}
    assert primeira.split("|")[1].strip() == max(medias, key=medias.get)


@pytest.mark.parametrize("n_estaveis, termina_com_reticencias", [(15, False), (16, True)])
def test_lista_de_estaveis_corta_em_15_com_reticencias(n_estaveis, termina_com_reticencias):
    import argparse
    p = n_estaveis + 3
    res = {"tarefa": "classificacao", "n_variaveis": p, "n_grupos": 10, "n_ajustes": 50, "n_divisoes": 5, "n_repeticoes": 10,
           "soma_vip2": float(p), "limiar": 1.0, "avisos": [],
           "classe": np.array(["estavel_acima"] * n_estaveis + ["instavel"] * 2 + ["estavel_abaixo"]),
           "media": np.linspace(2.0, 0.1, p)}
    nomes = [f"v{i:02d}" for i in range(p)]
    linhas = vip_cli._resumo(res, nomes, argparse.Namespace(componentes=2), 30)
    linha = next(x for x in linhas if "estavel acima" in x)
    assert linha.endswith(" ...") is termina_com_reticencias
    assert f"{n_estaveis} variavel(is)" in linha and nomes[0] in linha and nomes[14] in linha
    assert (nomes[15] in linha) is False


@pytest.mark.parametrize("p, ha_nota", [(20, False), (21, True)])
def test_tabela_do_relatorio_mostra_20_linhas_e_avisa_quando_ha_mais(tmp_path, p, ha_nota):
    t = tmp_path / "t.csv"
    _tab(t, p=p, informativas=3, n=6)
    s = tmp_path / "r"
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", _vars(p), "--repeticoes", "1",
                "--saida", s) == 0
    md = (s / "relatorio" / "vip_relatorio.md").read_text(encoding="utf-8")
    assert sum(1 for l in md.split("\n") if re.match(r"\| v\d\d ", l)) == 20
    assert (f"20 primeiras de {p}" in md) is ha_nota


@pytest.mark.parametrize("p, corta", [(50, False), (51, True)])
def test_config_efetiva_lista_ate_50_variaveis(tmp_path, p, corta):
    t = tmp_path / "t.csv"
    _tab(t, p=p, informativas=3, n=6)
    s = tmp_path / "r"
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", _vars(p), "--repeticoes", "1",
                "--sem-figuras", "--saida", s) == 0
    linha = next(l for l in (s / "config_efetiva.yaml").read_text(encoding="utf-8").split("\n") if l.startswith("variaveis:"))
    assert linha.endswith(", ...]") is corta and (f"v{p - 1:02d}" in linha) is (not corta)


# ------------------------------------------------------------------ entradas

def test_idioma_de():
    assert vip_cli._idioma_de(["--idioma=en"]) == "en" and vip_cli._idioma_de(["--idioma=pt"]) == "pt"
    assert vip_cli._idioma_de(["--idioma", "en"]) == "en" and vip_cli._idioma_de(["--idioma", "pt"]) == "pt"
    assert vip_cli._idioma_de(["--idioma"]) == "pt" and vip_cli._idioma_de([]) == "pt"
    assert vip_cli._idioma_de(["--idioma=EN-us"]) == "en"


def test_idioma_no_fim_sem_valor_nao_quebra(capsys):
    assert _run("--idioma") == 2 and capsys.readouterr().err


def test_argumento_desconhecido_e_tipo_invalido_dao_mensagem(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab(t)
    assert _run(t, "--classe", "classe", "--xyz") == 2
    assert "argumento invalido" in capsys.readouterr().err
    assert _run(t, "--classe", "classe", "--componentes", "abc") == 2
    assert "invalid int value" in capsys.readouterr().err


def test_sem_argv_explicito_le_sys_argv_e_o_uso_vem_no_idioma_certo(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["guaraci", "--ajuda"])
    assert vip_cli.executar() == 0 and "Uso: guaraci vip" in capsys.readouterr().out
    assert _run("--classe", "c") == 2 and "Uso: guaraci vip" in capsys.readouterr().err
    assert _run("--classe", "c", "--idioma", "en") == 2 and "Usage: guaraci vip" in capsys.readouterr().err


def test_limites_numericos_1_aceito_e_0_recusado(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab(t)
    base = [t, "--classe", "classe", "--grupo", "amostra", "--variaveis", _vars(8), "--sem-figuras"]
    assert _run(*base, "--componentes", "1", "--divisoes", "2", "--repeticoes", "1", "--saida", tmp_path / "ok") == 0
    capsys.readouterr()
    for extra in (["--componentes", "0"], ["--divisoes", "0"], ["--repeticoes", "0"], ["--limiar", "0"],
                  ["--limiar", "-1"]):
        assert _run(*base, *extra) == 2
        assert "inteiros positivos" in capsys.readouterr().err
    assert _run(*base, "--limiar", "0.5", "--repeticoes", "1", "--saida", tmp_path / "l") == 0


def test_celula_vazia_na_variavel_e_erro_e_rotulos_de_coluna_dizem_o_papel(tmp_path, capsys):
    t = tmp_path / "t.csv"
    linhas = _tab(t)
    linhas[3][5] = ""
    with open(t, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(linhas)
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", "v00,v01,v02") == 2
    assert "nao numerico ou vazio" in capsys.readouterr().err
    _tab(t)
    assert _run(t, "--resposta", "naoexiste", "--grupo", "amostra") == 2
    assert "coluna resposta 'naoexiste'" in capsys.readouterr().err
    assert _run(t, "--classe", "naoexiste", "--grupo", "amostra") == 2
    assert "coluna classe 'naoexiste'" in capsys.readouterr().err
    assert _run(t, "--classe", "classe", "--grupo", "naoexiste") == 2
    assert "coluna grupo 'naoexiste'" in capsys.readouterr().err
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", "v00,zzz") == 2
    assert "coluna variavel 'zzz'" in capsys.readouterr().err
    assert _run(t, "--classe", "classe", "--grupo", "amostra", "--variaveis", "v00,zzz", "--idioma", "en") == 2
    assert "variable column 'zzz'" in capsys.readouterr().err


def test_tipo_invalido_sai_com_a_mensagem_do_programa_e_sem_o_uso_do_argparse(tmp_path, capsys):
    t = tmp_path / "t.csv"
    _tab(t)
    assert _run(t, "--classe", "classe", "--componentes", "abc") == 2
    err = capsys.readouterr().err
    assert err.startswith("Erro: argumento invalido:") and "usage:" not in err


def test_parser_que_sai_por_conta_propria_vira_erro_de_uso_em_qualquer_versao_do_python(monkeypatch, capsys):
    """No Python 3.10/3.11 o argparse chama `parser.error` (SystemExit 2) mesmo com `exit_on_error=False`; no 3.12 isso nao acontece.
    Um parser que sai sozinho prova o ramo de compatibilidade em qualquer versao."""
    class _Sai:
        def parse_args(self, _args):
            raise SystemExit(2)
    monkeypatch.setattr(vip_cli, "_parser", lambda: _Sai())
    assert vip_cli.executar(["t.csv", "--classe", "c"]) == 2
    assert "--componentes/--divisoes/--repeticoes/--semente/--limiar" in capsys.readouterr().err
