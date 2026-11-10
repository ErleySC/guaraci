"""Graficos padrao e "mais graficos" honrados pelo motor (Fase 5)."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from guaraci import catalogo as cat
from guaraci import modos_analise as ma
from guaraci.config import Config

RAIZ = Path(__file__).resolve().parents[1]
OBJETIVOS = ("exploratorio", "classificacao", "quantificacao")


def _chaves_usadas_no_motor():
    chaves = set()
    for nome in ("pipeline.py", "avaliacao_modelos.py"):
        fonte = (RAIZ / "src" / "guaraci" / nome).read_text(encoding="utf-8")
        chaves |= set(re.findall(r'should_generate\(cfg, "([a-z_0-9]+)"\)', fonte))
    return chaves


def test_todo_grafico_do_motor_esta_no_catalogo_e_vice_versa():
    do_pipeline = {k for k, v in cat.GRAFICOS.items() if "comando" not in v}
    assert _chaves_usadas_no_motor() == do_pipeline == set(ma._FIG_OBJETIVOS) == set(ma._FIG_DESCRICAO)


def test_conjunto_padrao_de_cada_objetivo_tem_3_a_5_graficos():
    for obj in OBJETIVOS:
        padrao = [k for k, v in cat.GRAFICOS.items() if obj in v["padrao"] and not v.get("opt_in")
                  and obj in ma._FIG_OBJETIVOS.get(k, ())]
        assert 3 <= len(padrao) <= 5, (obj, padrao)


def test_selecao_padrao_todos_e_lista():
    cfg = Config(objective="exploratorio")
    assert ma.should_generate(cfg, "loadings") and not ma.should_generate(cfg, "hca")
    cfg.figure_selection = "todos"
    assert ma.should_generate(cfg, "hca") and ma.should_generate(cfg, "outliers_t2q")
    cfg.figure_selection = "hca, outliers_t2q"
    assert ma.should_generate(cfg, "hca") and not ma.should_generate(cfg, "loadings")
    assert not ma.should_generate(cfg, "confusao")          # fora do objetivo, mesmo nomeado


def test_grafico_de_etapa_ligada_sai_mesmo_no_padrao():
    cfg = Config(objective="classificacao", level="N1", run_benchmark=True)
    assert ma.should_generate(cfg, "benchmark") and not ma.should_generate(cfg, "vip")


def test_figuras_detalhadas_na_classificacao_trazem_as_exploratorias():
    """Regressao de f70e4ae achada na rodada 1/2: o gate novo exigia objetivo exploratorio."""
    cfg = Config(level="N2", detailed_figures=True)
    for chave in ("hca", "loadings", "biplot", "preprocessamento"):
        assert ma.should_generate(cfg, chave) and chave in ma.figure_plan(cfg)
    cfg.detailed_figures = False
    assert not ma.should_generate(cfg, "hca") and "hca" not in ma.figure_plan(cfg)


def test_lista_de_graficos_nao_desliga_etapa_opt_in_pedida():
    """O mesmo gate decide o calculo da etapa: uma lista sem `benchmark_regressao` nao pode
    pular em silencio o benchmark que o usuario ligou (achado de 2026-10-03)."""
    cfg = Config(objective="quantificacao", level="N3", run_benchmark_regression=True, figure_selection="regressao")
    assert ma.should_generate(cfg, "benchmark_regressao") and ma.should_generate(cfg, "regressao")
    assert not ma.should_generate(cfg, "pca_scores")


def test_valor_invalido_de_graficos_e_erro_de_configuracao():
    from guaraci.config_io import _validar_semantico
    assert any("graficos" in e for e in _validar_semantico(Config(figure_selection="loadings, inexistente")))
    assert not any("graficos" in e for e in _validar_semantico(Config(figure_selection="todos")))


@pytest.mark.slow
def test_padrao_grava_os_essenciais_e_mais_graficos_grava_mais(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}

    def rodar(pasta, *extra):
        subprocess.run([sys.executable, "-m", "guaraci", "exemplo", "explorar", str(pasta)], env=env,
                       capture_output=True, check=True, timeout=600)
        r = subprocess.run([sys.executable, "-m", "guaraci", "run", str(pasta / "config.yaml"), *extra], env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        assert r.returncode == 0, r.stdout[-800:]
        return {p.stem for p in pasta.rglob("Graficos/**/*.png")}   # pela CLI as figuras ficam em subpastas por grupo

    padrao = rodar(tmp_path / "a")
    todos = rodar(tmp_path / "b", "--mais-graficos")
    assert {"fig_mean_spectra_by_class", "fig1_pca_scores"} & padrao or padrao, padrao
    assert len(todos) > len(padrao) and padrao < todos, (sorted(padrao), sorted(todos))
