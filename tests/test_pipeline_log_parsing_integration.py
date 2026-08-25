"""Teste de integracao: o stdout REAL de `executar()` (nao uma string
sintetica) deve continuar casando com os regex de app_logic.py que
alimentam o painel de progresso ao vivo (CLI) e a barra de progresso do
app web.

Existe porque a migracao de `print()` para `logging` em pipeline.py
(CLAUDE.md P6) e' arriscada por construcao: os dois paineis fazem PARSING
DE TEXTO do stdout capturado (`_RE_ETAPA`, `_RE_ARQUIVO_SALVO`,
`_RE_AVISO` em app_logic.py), nao consomem um objeto estruturado. Um
`print()` -> `log.info()` que mude a string (ou pare de passar pelo mesmo
canal capturado) quebra os paineis silenciosamente -- sem este teste,
nenhuma outra parte da suite pegaria essa regressao."""
import contextlib
import io
import os

import pytest

from guaraci.app_logic import log_progress, figures_completed


@pytest.mark.slow
def test_stdout_real_do_executar_ainda_casa_com_regex_do_painel(pq, tmp_path):
    cfg = pq.Config(
        input_folder=str(tmp_path / "dados"),
        output_root_folder=str(tmp_path / "saida"),
        modo="sintetico",
        n_por_classe=8,
        n_pontos_sint=50,
        wn_min=400.0,
        wn_max=4001.0,
        n_splits_cv=2,
        n_repeats_cv=1,
        n_permutacoes=5,
        n_permutacoes_wold=5,
        n_bootstrap_vip=3,
        n_bootstrap_bca=20,
        n_monte_carlo=3,
        run_benchmark=False,
        run_monte_carlo=False,
        run_shap=False,
        run_wold=False,
        run_cv_anova=False,
        run_opls=False,
        executar_etapa4=False,
        comparar_pipelines=False,
        comparar_hca_pipelines=False,
        max_lvs=5,
    )
    os.makedirs(str(tmp_path / "dados"), exist_ok=True)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pq.executar(cfg)
    texto = buf.getvalue()

    # 1) marcadores de etapa "[N/7]" -- sempre presentes em qualquer run
    frac, _nome = log_progress(texto)
    assert frac > 0.0, (
        "Nenhum marcador '[N/7]' encontrado no stdout real -- o painel de "
        "progresso (CLI e app web) ficaria travado em 'Starting...' para "
        "sempre. Verificar se print()->logging preservou o texto exato.")

    # 2) linha "-> <arquivo>.png" apos salvar figura -- sempre ha' >=1 figura
    figs = figures_completed(texto)
    assert len(figs) > 0, (
        "Nenhuma figura detectada no stdout real via '-> arquivo.png' -- "
        "o painel de figuras concluidas ficaria sempre vazio.")
