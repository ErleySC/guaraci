"""M-19(a) do parecer de 2026-09-25: toda linha de tabela do model card tem o
mesmo numero de celulas do cabecalho (o status "nao validado" nao some)."""
import re

import pytest

from guaraci.resultados_io import MARCADOR_LEIA_PRIMEIRO, _md_tabela

BS = chr(92)


def _celulas(linha):
    # separa por "|" NAO escapado
    return [c for c in re.split(r"(?<!" + re.escape(BS) + r")\|", linha)[1:-1]]


def test_pipe_interno_e_escapado_e_a_linha_tem_2_celulas():
    t = _md_tabela([("Combinacao AND x adulterante S",
                     "cobertura=nao_validado_n1 | n_grupos=1 | alpha=nan")])
    linhas = t.splitlines()
    assert all(len(_celulas(l)) == 2 for l in linhas)
    assert "nao_validado_n1" in linhas[2]        # o status sobrevive
    assert BS + "|" in linhas[2]


@pytest.fixture(scope="module")
def card_de_uma_execucao_real(tmp_path_factory):
    """UMA execucao sintetica (~30 s) compartilhada pelos testes do model card (antes cada teste
    rodava a sua, 2x ~30 s). Config do N2 com DD-SIMCA ligado: o card cobre as tabelas de todas as secoes."""
    import guaraci.pipeline as pq
    raiz = tmp_path_factory.mktemp("card")
    cfg = pq.Config(mode="sintetico", output_root_folder=str(raiz), show_plots=False,
                    n_permutations=3, run_benchmark=False, run_monte_carlo=False,
                    run_shap=False, run_wold=False, run_cv_anova=False, run_opls=False,
                    executar_etapa4=False, max_lvs=3, selecao_lv_cv_aninhada=False,
                    frac_holdout=0.2, level="N2")
    pq.executar(cfg)
    return list(raiz.rglob("model_card.md"))[0].read_text(encoding="utf-8")


def test_model_card_de_uma_execucao_real_tem_tabelas_consistentes(card_de_uma_execucao_real):
    for linha in card_de_uma_execucao_real.splitlines():
        if linha.startswith("|") and not set(linha) <= set("|-: "):
            assert len(_celulas(linha)) == 2, linha


def test_classe_unica_e_critica_so_nos_niveis_que_classificam():
    import guaraci.pipeline as pq
    from guaraci.auditoria_delineamento import check_classe_unica
    import numpy as np
    assert check_classe_unica(np.array(["A"] * 5), pq.Config(level="N1")).severidade == "critico"
    assert check_classe_unica(np.array(["A"] * 5), pq.Config(level="N3")).severidade == "ok"
    assert check_classe_unica(np.array(["A", "B"]), pq.Config(level="N1")).severidade == "ok"


def test_model_card_traz_holdout_resumo_no_topo_e_acuracia_por_classe(card_de_uma_execucao_real):
    card = card_de_uma_execucao_real
    assert "## Holdout externo" in card
    # R5-11: resumo no topo; ressalvas so' dos modulos que rodaram
    assert card.index(MARCADOR_LEIA_PRIMEIRO) < card.index("## 1. Detalhes do Modelo")
    assert "**SHAP:**" not in card and "benchmark:**" not in card
    assert "## 6. Acuracia por classe" in card
