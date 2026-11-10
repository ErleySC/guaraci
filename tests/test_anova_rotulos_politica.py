"""ANOVA: a saida descreve o metodo que de fato rodou, e a politica padrao do multielementar controla o erro por familia.

Achados da vistoria 2026-10 (V20, V64):
  - a CLI escrevia "Welch + Tukey" quando rodava a ANOVA classica (pedido de Tukey ou `--bloco`) e "como solicitado"
    sem pedido; chamava o Brown-Forsythe de "Levene"; rotulava "IC95" com qualquer alpha e mostrava "IC95 nan a nan"
    com `--sem-bootstrap`; a nota das letras dizia "nao diferem";
  - o ICP multielementar usava a politica `auto` (Brown-Forsythe decide Tukey x Games-Howell), que leva o erro por
    familia a ~0,25 sob H0 quando o grupo pequeno tem a maior variancia; agora usa Games-Howell fixo.
Dados fixos e semente fixa: o cenario "S2" (variancias muito desiguais, unidade com 3 replicas) e' o da vistoria.
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci import anova_cli
from guaraci.anova_grupos import anova_uma_via


def _tabela_s2(caminho) -> None:
    """Cenario S2 da vistoria: 3 grupos x 8 unidades x 3 replicas; desvio entre unidades 0,2 / 5,0 / 0,2."""
    rng = np.random.default_rng(7)
    linhas = ["grupo,unidade,sessao,y"]
    for g, sd, mu in (("a", 0.2, 0.0), ("b", 5.0, 0.0), ("c", 0.2, 1.0)):
        for u in range(8):
            base = rng.normal(mu, sd)
            for _ in range(3):
                linhas.append(f"{g},{g}{u},S{u % 4},{base + rng.normal(0, 0.05)!r}")
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


BASE = ("--resposta", "y", "--grupo", "grupo", "--unidade", "unidade", "--sem-figuras")


@pytest.fixture()
def s2(tmp_path):
    t = tmp_path / "s2.csv"
    _tabela_s2(t)
    return t


def _rodar(capsys, *args) -> str:
    assert anova_cli.executar([str(a) for a in args]) == 0
    return capsys.readouterr().out


def _linha_resposta(out: str) -> str:
    return next(x for x in out.split("\n") if x.startswith("- y:"))


# ------------------------------------------------------------------ V20: rotulos

def test_tukey_pedido_com_variancias_desiguais_nao_diz_welch(s2, tmp_path, capsys):
    out = _rodar(capsys, s2, *BASE, "--metodo", "tukey", "--sem-bootstrap", "--saida", tmp_path / "t")
    assert "ANOVA classica" in _linha_resposta(out)
    assert "Welch" not in out
    assert "rodou ANOVA classica + Tukey, que supoe variancias iguais (mantido a pedido: --metodo tukey)" in out


def test_bloco_nao_diz_welch_e_avisa_que_o_pos_teste_supoe_variancias_iguais(s2, tmp_path, capsys):
    for extra in ((), ("--metodo", "tukey"), ("--metodo", "auto")):
        out = _rodar(capsys, s2, *BASE, "--bloco", "sessao", "--sem-bootstrap", *extra, "--saida", tmp_path / "b")
        assert "ANOVA com bloco" in _linha_resposta(out)
        assert "Welch" not in out, extra
        assert "o pos-teste com bloco supoe variancias iguais" in out, extra


def test_sem_metodo_pedido_nao_diz_como_solicitado_e_declara_o_padrao(s2, tmp_path, capsys):
    out = _rodar(capsys, s2, *BASE, "--sem-bootstrap", "--saida", tmp_path / "p")
    assert "como solicitado" not in out
    assert "padrao do programa (Welch + Games-Howell); para Tukey use --metodo tukey" in out
    assert "Welch ANOVA" in _linha_resposta(out)
    # pedido explicito: nao aparece a frase do padrao
    out2 = _rodar(capsys, s2, *BASE, "--metodo", "games-howell", "--sem-bootstrap", "--saida", tmp_path / "q")
    assert "padrao do programa" not in out2 and "como solicitado" not in out2


def test_valor_de_variancias_e_chamado_brown_forsythe_nunca_levene(s2, tmp_path, capsys):
    for metodo in ("games-howell", "tukey", "auto"):
        out = _rodar(capsys, s2, *BASE, "--metodo", metodo, "--sem-bootstrap", "--saida", tmp_path / metodo)
        assert "Levene p" not in out and "Levene rejeitou" not in out, metodo
        assert "Brown-Forsythe" in out, metodo


def test_sem_bootstrap_omite_o_ic_de_eta2_e_alpha_muda_o_rotulo_do_ic_par_a_par(s2, tmp_path, capsys):
    out = _rodar(capsys, s2, *BASE, "--sem-bootstrap", "--saida", tmp_path / "s")
    assert "nan a nan" not in out and "IC95" not in _linha_resposta(out)
    out = _rodar(capsys, s2, *BASE, "--alpha", "0.1", "--saida", tmp_path / "a")
    pares = [x for x in out.split("\n") if " x " in x and "[Games-Howell]" in x]
    assert pares and all("(IC90 " in x for x in pares), pares
    assert "IC95 bootstrap" in _linha_resposta(out)          # eta2: bootstrap percentil 2,5-97,5, sempre 95%


def test_nota_das_letras_nao_afirma_igualdade(s2, tmp_path, capsys):
    s = tmp_path / "l"
    _rodar(capsys, s2, *BASE, "--sem-bootstrap", "--saida", s)
    md = (s / "relatorio" / "anova_relatorio.md").read_text(encoding="utf-8")
    assert "nao diferem significativamente ao nivel alpha = 0.05" in md and "nao prova igualdade" in md


# ------------------------------------------------------------------ V64: politica

def _dados_s2_nucleo():
    rng = np.random.default_rng(7)
    y, g, u = [], [], []
    for gi, (sd, mu) in enumerate(((0.2, 0.0), (5.0, 0.0), (0.2, 1.0))):
        for k in range(8):
            y.append(rng.normal(mu, sd))
            g.append(f"g{gi}")
            u.append(f"g{gi}u{k}")
    sess = [f"S{int(x[-1]) % 4}" for x in u]
    return np.asarray(y), g, u, sess


def test_aviso_do_bloco_no_nucleo_aparece_com_variancias_desiguais_e_nao_com_iguais():
    y, g, u, sess = _dados_s2_nucleo()
    r = anova_uma_via(y, g, u, bloco=sess, n_boot=0)
    assert any("o pos-teste com bloco supoe variancias iguais" in a for a in r.avisos), r.avisos
    padrao = np.array([-1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5])          # mesma dispersao nos 3 grupos (razao 1)
    y_ig = np.concatenate([padrao + d for d in (0.0, 0.3, 0.6)])
    r2 = anova_uma_via(y_ig, g, u, bloco=sess, n_boot=0)
    assert r2.levene_p >= 0.05
    assert not any("o pos-teste com bloco" in a for a in r2.avisos), r2.avisos


def _tab_icp(tmp_path, dp_por_grupo):
    from guaraci.icp_multielementar import ler_icp_csv
    rng = np.random.default_rng(11)
    linhas = ["Amostra;Tipo;Grupo;Unidade;Cd 111;Pb 208"]
    for grp, dp in dp_por_grupo.items():
        for j in range(8):
            cd = f"{rng.normal(1.0, dp):.4f}".replace(".", ",")
            pb = f"{rng.normal(3.0, dp):.4f}".replace(".", ",")
            linhas.append(f"{grp}{j};amostra;{grp};{grp}{j};{cd};{pb}")
    p = tmp_path / "icp.csv"
    p.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return ler_icp_csv(str(p), unidade="mg/kg", coluna_amostra="Amostra", coluna_tipo="Tipo",
                       coluna_grupo="Grupo", coluna_unidade_fisica="Unidade")


def test_icp_multielementar_usa_welch_e_games_howell_mesmo_com_variancias_iguais(tmp_path):
    """Com variancias iguais o `auto` escolheria Tukey; a politica do multielementar e' Games-Howell fixo."""
    from guaraci.icp_multielementar import POLITICA_VARIANCIAS_ANOVA, _anova_por_elemento
    assert POLITICA_VARIANCIAS_ANOVA == "desiguais"
    tab = _tab_icp(tmp_path, {"sul": 0.3, "norte": 0.3, "leste": 0.3})
    out, els, _ = _anova_por_elemento(tab, "substituir", 0.5, False, 0.05, 0)
    assert set(els) == {"Cd 111", "Pb 208"}
    for r in out["resultados"].values():
        assert r.levene_p >= 0.05                        # o `auto` ficaria com a ANOVA classica aqui
        assert r.metodo == "Welch ANOVA" and r.pos_teste == "Games-Howell"


def test_politica_do_multielementar_nao_tem_o_falso_positivo_que_o_auto_produz():
    """Caso deterministico curto que fixa a politica (a simulacao longa fica em scripts/simular_politicas_anova.py e nos
    testes `slow`). H0 verdadeira (todas as medias 0), grupo pequeno com a maior variancia: n = (3, 8, 8, 8),
    dp = (3, 1, 1, 1), semente 5. O Brown-Forsythe NAO rejeita, o `auto` cai no Tukey e acusa pares que nao diferem;
    o Games-Howell nao. Medido na vistoria (500 rep., mesmo desenho): erro por familia Games-Howell 0,066, auto 0,258."""
    from guaraci.icp_multielementar import POLITICA_VARIANCIAS_ANOVA
    rng = np.random.default_rng(5)
    ns, sds = (3, 8, 8, 8), (3.0, 1.0, 1.0, 1.0)
    y = np.concatenate([rng.normal(0, s, m) for m, s in zip(ns, sds)])
    g = np.concatenate([[f"g{i}"] * m for i, m in enumerate(ns)])
    u = [f"u{i}" for i in range(len(g))]
    auto = anova_uma_via(y, g, u, n_boot=0, variancias="auto")
    assert auto.levene_p > 0.5 and auto.pos_teste == "Tukey-Kramer"
    assert min(c["p_ajustado"] for c in auto.comparacoes) < 0.01          # falso positivo do `auto`
    pol = anova_uma_via(y, g, u, n_boot=0, variancias=POLITICA_VARIANCIAS_ANOVA)
    assert pol.pos_teste == "Games-Howell"
    assert min(c["p_ajustado"] for c in pol.comparacoes) > 0.10           # a politica escolhida nao acusa nada
