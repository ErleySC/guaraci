"""Golden test de VALORES numéricos do pipeline (CLAUDE.md P9, passo 1).

Por que este arquivo existe
---------------------------
Os testes end-to-end existentes verificam que o pipeline *roda* e que os PNGs
*são gerados*. Nenhum deles verifica que os **números não mudaram**. Essa é a
lacuna que torna perigoso refatorar `executar()`: uma mudança pode preservar
todos os arquivos de saída e ainda assim alterar silenciosamente um R², um
limite de Hotelling ou uma especificidade DD-SIMCA — o pior tipo de bug em
software científico, porque não trava, não avisa, e o número errado acaba numa
monografia.

Este teste roda o pipeline sintético completo e compara os valores gravados em
`resumo_modelo.txt` contra um arquivo de referência versionado
(`tests/golden/pipeline_n2_sintetico.json`). Se algum valor mudar, o teste
falha e nomeia a métrica.

**O teste falhar não significa necessariamente que há um bug.** Significa que
os números mudaram e isso precisa de justificativa explícita. Se a mudança for
correta (ex.: uma correção científica como a do DD-SIMCA em 2026-07-19), o
procedimento é: entender POR QUE mudou, documentar no CHANGELOG, e regravar o
golden com

    GUARACI_REGRAVAR_GOLDEN=1 pytest tests/test_golden_valores.py

Regravar sem entender a causa anula o propósito do teste.

Determinismo
------------
Verificado empiricamente antes de escrever este teste: duas execuções
independentes com a mesma config (seed=42) produzem `resumo_modelo.txt`
byte-idêntico. A tolerância `_RTOL` existe apenas para diferenças de BLAS
entre plataformas (a CI roda Linux/Windows/macOS), não para instabilidade do
próprio pipeline — é frouxa o suficiente para não dar falso positivo entre
sistemas e apertada o suficiente para pegar qualquer regressão real, que
sempre move os valores muito mais que isso.
"""
from __future__ import annotations

import io
import json
import os
import re
import contextlib
from pathlib import Path

import pytest

from conftest import achar_pastas_run

# Tolerância relativa: ver docstring do módulo.
_RTOL = 1e-6

_GOLDEN = Path(__file__).parent / "golden" / "pipeline_n2_sintetico.json"

# Rótulos do resumo_modelo.txt cujo valor é travado. Escolhidos por serem
# cientificamente significativos E sensíveis a mudança de cálculo: métricas de
# ajuste (R2X/R2Y/Q2), limites de diagnóstico (Hotelling/Q/DModX), e contagens
# estruturais (LVs, grupos) que denunciam mudança na validação group-aware.
_METRICAS_TRAVADAS = [
    "Total de amostras",
    "Total de variaveis",
    "Total de classes",
    "LVs otimas",
    "N grupos mae_id",
    "Accuracy (CV)",
    "Balanced accuracy",
    "F1 (macro)",
    "Cohen's kappa",
    "R2X",
    "R2Y",
    "Q2",
    "Hotelling T2 (95%)",
    "Q-residual (95%)",
    "N outliers T2",
    "N outliers Q",
    "DModX critico (SIMCA)",
    "N amostras fora do DModX",
    "ROC AUC macro (OvR)",
    "Holdout accuracy",
    "Holdout balanced acc",
    "DD-SIMCA n_components",
    "DD-SIMCA n_desconhecidos",
]

_RE_NUM = re.compile(r"^\s*(.+?)\s*:\s*([-\d.]+(?:[eE][-+]?\d+)?)\s*$")
# Linha de DD-SIMCA por espécie:
#   "DD-SIMCA Esp_A sens(LOGO)/esp : n/a (nao validado) / 61.9% (grupos_LOGO=1, ...)"
# A especificidade é a métrica DD-SIMCA legitimamente validável com o desenho
# de dados atual (a sensibilidade é n/a — ver CLAUDE.md P1), então é ela que
# precisa estar travada contra regressão.
_RE_DDSIMCA = re.compile(
    r"^\s*DD-SIMCA\s+(\S+)\s+sens\(LOGO\)/esp\s*:.*?/\s*([\d.]+)%")


def _extrair_valores(resumo: str) -> dict:
    """resumo_modelo.txt -> {rotulo: float} apenas com as métricas travadas.

    Parser deliberadamente estrito: só casa linhas `rotulo : numero` cujo
    rótulo esteja em `_METRICAS_TRAVADAS`, mais as linhas de especificidade
    DD-SIMCA por espécie. Se o pipeline parar de emitir uma métrica travada,
    ela some do dict e o teste acusa a ausência — o que é o comportamento
    desejado (uma métrica que desaparece é uma regressão silenciosa).
    """
    valores: dict = {}
    for linha in resumo.splitlines():
        m_dd = _RE_DDSIMCA.match(linha)
        if m_dd:
            valores[f"DD-SIMCA especificidade {m_dd.group(1)}"] = float(m_dd.group(2))
            continue
        m = _RE_NUM.match(linha)
        if m and m.group(1) in _METRICAS_TRAVADAS:
            valores[m.group(1)] = float(m.group(2))
    return valores


def _cfg_golden(pq, base: Path):
    """Config FIXA do run golden. Não mudar sem regravar o golden.

    Valores pequenos de propósito (o teste roda na CI em 10 combinações de
    SO/versão), mas com `n_replicas_sint=3` para que a validação group-aware
    por `mae_id` e o DD-SIMCA tenham o que exercitar de verdade.
    """
    cfg = pq.Config(
        pasta_entrada=str(base / "in"),
        pasta_saida_raiz=str(base / "saida"),
        modo="sintetico", n_por_classe=10, n_pontos_sint=60,
        n_replicas_sint=3, wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1,
        n_permutacoes=5, n_permutacoes_wold=5,
        n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3,
        max_lvs=5, nivel="N2", figuras_detalhadas=False,
    )
    # DD-SIMCA ligado (é o diferencial científico do projeto e o cálculo mais
    # sensível); módulos pesados/opcionais desligados para manter o teste
    # rápido e sem depender de xgboost/shap.
    for attr, val in [
        ("executar_ddsimca", True), ("executar_opls", False),
        ("executar_etapa4", False), ("executar_wold", False),
        ("comparar_pipelines", False), ("executar_cv_anova", False),
        ("executar_benchmark", False), ("executar_monte_carlo", False),
        ("executar_shap", False),
    ]:
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)
    os.makedirs(cfg.pasta_entrada, exist_ok=True)
    return cfg


@pytest.fixture(scope="session")
def valores_run(pq, tmp_path_factory) -> dict:
    """Roda o pipeline UMA vez por sessão e devolve os valores extraídos."""
    base = tmp_path_factory.mktemp("golden")
    cfg = _cfg_golden(pq, base)
    with contextlib.redirect_stdout(io.StringIO()):
        pq.executar(cfg)
    runs = achar_pastas_run(cfg.pasta_saida_raiz)
    assert runs, "executar() nao criou pasta de saida"
    resumo_path = Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt"
    assert resumo_path.is_file(), f"resumo_modelo.txt nao encontrado em {runs[0]}"
    valores = _extrair_valores(resumo_path.read_text(encoding="utf-8"))
    assert valores, "nenhuma metrica extraida do resumo — parser ou formato mudou"
    return valores


@pytest.mark.slow
def test_valores_numericos_nao_mudaram(valores_run):
    """Golden test: nenhum valor numérico do pipeline mudou sem justificativa.

    Ver docstring do módulo para o procedimento quando este teste falha.
    """
    if os.environ.get("GUARACI_REGRAVAR_GOLDEN") == "1":
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(
            json.dumps(valores_run, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
        pytest.skip(f"golden regravado em {_GOLDEN} (GUARACI_REGRAVAR_GOLDEN=1)")

    assert _GOLDEN.is_file(), (
        f"arquivo golden ausente: {_GOLDEN}\n"
        "Gere com: GUARACI_REGRAVAR_GOLDEN=1 pytest tests/test_golden_valores.py")
    esperado = json.loads(_GOLDEN.read_text(encoding="utf-8"))

    faltando = sorted(set(esperado) - set(valores_run))
    assert not faltando, (
        f"metricas SUMIRAM da saida do pipeline: {faltando}\n"
        "Uma metrica que deixa de ser emitida e' uma regressao silenciosa.")

    divergentes = []
    for chave, ref in sorted(esperado.items()):
        obtido = valores_run[chave]
        if ref == 0:
            ok = obtido == 0
        else:
            ok = abs(obtido - ref) <= _RTOL * abs(ref)
        if not ok:
            divergentes.append(f"  {chave}: golden={ref!r} -> obtido={obtido!r}")
    assert not divergentes, (
        "REGRESSAO NUMERICA — os valores abaixo mudaram:\n"
        + "\n".join(divergentes)
        + "\n\nInvestigue a CAUSA antes de regravar o golden. Se a mudanca for"
          " correta, documente no CHANGELOG e rode:\n"
          "  GUARACI_REGRAVAR_GOLDEN=1 pytest tests/test_golden_valores.py")


@pytest.mark.slow
def test_golden_cobre_o_diferencial_do_projeto(valores_run):
    """O golden precisa travar a especificidade DD-SIMCA por espécie.

    Contrato explícito: a validação group-aware + DD-SIMCA é o argumento
    central do projeto (CLAUDE.md secao 1). Se uma refatoracao fizer essas
    linhas sumirem do resumo, o teste acima (que so' compara o que existe nos
    DOIS lados) poderia passar despercebido caso o golden fosse regravado sem
    atencao. Este teste falha explicitamente nesse cenario.
    """
    esp = [k for k in valores_run if k.startswith("DD-SIMCA especificidade")]
    assert esp, ("nenhuma especificidade DD-SIMCA por especie no resumo — "
                 "o diferencial cientifico do projeto parou de ser reportado")
    for chave in esp:
        assert 0.0 <= valores_run[chave] <= 100.0, (
            f"{chave} fora do intervalo valido de porcentagem: {valores_run[chave]}")
