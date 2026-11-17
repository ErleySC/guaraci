# -*- coding: utf-8 -*-
"""As analises do trabalho de campo percorridas pelo menu interativo com teclas reais, sem comando avulso.

Um motorzinho responde ao `input()` do menu olhando a tela que acabou de ser desenhada: acha o numero de cada opcao pelo
rotulo (como a pessoa faz), digita o valor e confirma. No fim aperta R e o teste confere a configuracao que chegou ao motor."""
from __future__ import annotations

import builtins
import re

import pytest

import guaraci.guaraci as g
from guaraci import catalogo as cat


def _numero(saida: str, rotulo: str) -> str:
    """Numero que a tela mostra ao lado do rotulo, na ultima vez que ele foi desenhado. O rotulo quebra de linha e e' cortado:
    usa o maior prefixo que aparece."""
    for n in (len(rotulo), 32, 24, 18, 12):
        achados = re.findall(r"\[(\d+)\][^\n]*?" + re.escape(rotulo[:n]), saida)
        if achados:
            return achados[-1]
    raise AssertionError(f"'{rotulo}' nao aparece na tela: {saida[-700:]}")


class Motor:
    def __init__(self, cli, monkeypatch, real=False):
        self.cli, self.passos, self.log, self.cfg_rodada = cli, [], [], None
        g._STATE["modo_usuario"] = "avancado"
        monkeypatch.setattr(builtins, "input", self._entrada)
        monkeypatch.setattr(g, "_pause", lambda *a, **k: None)      # "Enter para continuar" nao consome a digitacao
        if not real:
            monkeypatch.setattr(g, "_rodar_pipeline", self._capturar)

    def _capturar(self, cfg, retomar=False):
        self.cfg_rodada = cfg

    def _entrada(self, prompt=""):
        self.cli.prompts.append(prompt)
        if not self.passos:
            raise EOFError
        p = self.passos.pop(0)
        if isinstance(p, tuple):
            p = _numero(self.cli.saida(), p[1])
        self.log.append((prompt.strip()[:60], p))
        return p

    def editar(self, chave: str, *valor: str) -> None:
        """Caminho de teclas de `chave`: tela, pasta (se a opcao estiver numa), opcao, valor."""
        o = cat.opcao(chave)
        tela = cat.TELAS[o["tela"]]["tecla"]
        rotulo = o["nome"][0]
        sec = next((s for s in cat.secoes_da_tela(o["tela"]) if chave in s["campos"]), None)
        self.passos += [tela]
        if sec is not None:
            self.passos += [("num", f"+{sec['nome'][0]}")]
        confirma = ["s"] if o.get("risco") not in (None, "VISUAL") else []     # ◆ e ▲ pedem confirmacao
        self.passos += [("num", rotulo), *valor, *confirma, "V", "V"]

    def rodar(self) -> "Motor":
        self.passos += ["R", "Q"]
        g.main([])
        return self


# (opcao, o que se digita, atributo do Config que o motor le, valor esperado): as escolhas do protocolo do trabalho
CASOS = [
    ("faixa_min_cm", ["4540"], "wn_min", 4540.0),
    ("faixa_max_cm", ["9500"], "wn_max", 9500.0),
    ("faixas_excluir", ["5647-5875"], "exclude_ranges", ("5647-5875",)),
    ("saturacao", ["2"], "saturacao", "avisar"),
    ("sg_janela", ["25"], "sg_window", 25),
    ("sg_ordem", ["2"], "sg_polyorder", 2),
    ("sg_derivada", ["1"], "sg_deriv", 1),
    ("nivel", ["3"], "level", "N3"),
    ("objetivo", ["4"], "objective", "quantificacao"),
    ("divisao_calibracao", ["3"], "cal_val_split", "spxy"),
    ("validacao_group_aware", ["2"], "group_by_mae_id", False),
    ("n_permutacoes", ["200"], "n_permutations", 200),
    ("n_permutacoes_wold", ["200"], "n_permutations_wold", 200),
    ("n_bootstrap_bca", ["1000"], "n_bootstrap_bca", 1000),
    ("n_bootstrap_vip", ["100"], "n_bootstrap_vip", 100),
    ("teste_wold", ["1"], "run_wold", True),
    ("pca_componentes", ["6"], "n_pcs_pca", 6),
    ("hca_componentes", ["10"], "hca_n_pcs", 10),
    ("comparar_hca", ["1"], "compare_hca_pipelines", True),
    ("selecao_variaveis_etapa4", ["1"], "executar_etapa4", True),
    ("ipls_intervalos", ["30"], "ipls_n_intervalos", 30),
    ("vip_limiar", ["1.2"], "vip_threshold_sel", 1.2),
    ("splsda_variaveis", ["40"], "splsda_keep_by_comp", 40),
    ("graficos", ["todos"], "figure_selection", "todos"),
    ("figuras_detalhadas", ["1"], "detailed_figures", True),
    ("holdout_fracao", ["0.25"], "frac_holdout", 0.25),
]


@pytest.mark.parametrize("chave,digitado,attr,esperado", CASOS, ids=[c[0] for c in CASOS])
def test_cada_opcao_do_protocolo_chega_ao_motor_pelo_menu(cli, monkeypatch, chave, digitado, attr, esperado):
    m = Motor(cli, monkeypatch)
    m.editar(chave, *digitado)
    m.rodar()
    assert m.cfg_rodada is not None, m.log
    assert getattr(m.cfg_rodada, attr) == esperado, m.log


def _achar(raiz, nome):
    import os
    return [os.path.join(r, f) for r, _, fs in os.walk(raiz) for f in fs if f == nome]


def test_classificacao_completa_pelo_menu_e_predicao_do_modelo_gerado(cli, monkeypatch, tmp_path):
    """PCA, HCA, PLS-DA, Wold, permutacao, VIP/SR e selecao de variaveis (iPLS, sPLS-DA): configurados e rodados so' pelo menu;
    depois o modelo que saiu e' aplicado pela tela 3.3."""
    import numpy as np
    import pandas as pd
    import joblib
    saida = tmp_path / "saida"
    m = Motor(cli, monkeypatch, real=True)
    m.editar("pasta_saida", str(saida))
    m.editar("modo_entrada", "13")                    # dados sinteticos
    m.editar("sintetico_n_por_classe", "10")
    m.editar("sintetico_n_pontos", "60")
    m.editar("faixa_min_cm", "400")
    m.editar("faixa_max_cm", "4001")
    m.editar("nivel", "1")
    m.editar("max_lvs", "5")
    m.editar("n_permutacoes", "5")
    m.editar("n_permutacoes_wold", "5")
    m.editar("n_bootstrap_vip", "3")
    m.editar("n_bootstrap_bca", "20")
    m.editar("teste_wold", "1")
    m.editar("comparar_hca", "2")
    m.editar("selecao_variaveis_etapa4", "1")
    m.editar("figuras_detalhadas", "1")
    m.editar("graficos", "todos")
    m.passos += ["R", "", "s", "s", "s", "", ""]
    g.main([])                                          # sem passos sobrando: o fim da digitacao e' EOF = sair
    for nome in ("fig1_pca_scores.png", "fig_hca_dendrogram.png", "fig2_plsda_scores.png", "fig_extra_wold_permutacao.png",
                 "fig_selectivity_ratio_vip.png", "fig_etapa4_ipls_intervalos.png", "fig_etapa4_comparacao_metodos.png",
                 "model_card.md", "modelo_plsda.joblib"):
        assert _achar(saida, nome), f"{nome} nao saiu da corrida pelo menu"
    assert _achar(saida, "etapa4_selecao_variaveis.csv")

    cam_modelo = _achar(saida, "modelo_plsda.joblib")[0]
    wn = np.asarray(joblib.load(cam_modelo)["wavenumbers"], dtype=float)
    novos = tmp_path / "novos.csv"
    pd.DataFrame(np.random.default_rng(1).normal(0.5, 0.05, (4, wn.size)), columns=[f"{v:.4f}" for v in wn]).to_csv(novos, index=False, sep=";")
    m2 = Motor(cli, monkeypatch, real=True)
    m2.passos += ["3.3", cam_modelo, "s", str(novos), ""]
    g.main([])
    assert (tmp_path / "novos_predicao.csv").is_file()
    assert len(pd.read_csv(tmp_path / "novos_predicao.csv", sep=";", decimal=",")) == 4


def test_quantificacao_por_par_pelo_menu(cli, monkeypatch, tmp_path):
    """Tela [3.7]: um modelo N3 por par, tabela de merito e VIP/SR, tudo digitado no menu."""
    import csv
    from tests.test_quantificacao_pares import _raiz
    raiz = _raiz(tmp_path)
    saida = tmp_path / "pares"
    m = Motor(cli, monkeypatch, real=True)
    m.editar("faixa_min_cm", "4000")
    m.editar("faixa_max_cm", "10000")
    m.editar("n_permutacoes", "3")
    m.editar("n_bootstrap_vip", "3")
    m.editar("n_bootstrap_bca", "10")
    m.passos += ["3.7", raiz, str(saida), "", ""]
    g.main([])
    with open(saida / "tabela_pares.csv", encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))
    assert len(linhas) == 2 and all(not r["erro"] for r in linhas) and all(float(r["lod"]) > 0 for r in linhas)
    assert len(_achar(saida, "regressao_vip_sr.csv")) == 2 and len(_achar(saida, "regressao_figuras_de_merito.csv")) == 2
