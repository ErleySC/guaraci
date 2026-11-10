"""O veredito de linearidade nao afirma curvatura quando o erro puro vem so' de replicas tecnicas.

Achado da auditoria de 2026-11-10: `guaraci exemplo quantificar` (sinal linear por construcao)
saia no model card com "FALTA DE AJUSTE SIGNIFICATIVA (curvatura detectada)".
"""
import numpy as np

from guaraci import resultados_io as rio
from guaraci.linearity import lack_of_fit_test


def _curva_reta_com_replicas_tecnicas():
    rng = np.random.default_rng(0)
    niveis = np.repeat(np.arange(6, dtype=float), 3)
    y_ref = niveis * 2.0
    # erro de preparo por nivel (maior que o ruido de replica): o erro puro subestima o real
    y_pred = y_ref + np.repeat(rng.normal(0, 0.5, 6), 3) + rng.normal(0, 0.01, 18)
    grupos = np.repeat([f"g{i}" for i in range(6)], 3)
    return y_ref, y_pred, grupos


def test_model_card_diz_nao_conclusivo_e_traz_a_ressalva(tmp_path):
    y_ref, y_pred, grupos = _curva_reta_com_replicas_tecnicas()
    lin = lack_of_fit_test(y_ref, y_pred, grupos, nivel_por="grupo")
    assert lin.computavel and not lin.linear
    (tmp_path / "model_card.md").write_text("# card\n", encoding="utf-8")
    rio.append_linearity_robustness_model_card(str(tmp_path), linearidade=lin)
    texto = (tmp_path / "model_card.md").read_text(encoding="utf-8")
    assert "curvatura detectada" not in texto and "curvature detected" not in texto
    assert "NAO CONCLUSIVO" in texto or "INCONCLUSIVE" in texto
    assert "replicas tecnicas" in texto or "technical replicates" in texto


def test_com_niveis_por_concentracao_a_rejeicao_continua_afirmativa(tmp_path):
    x = np.repeat(np.arange(6, dtype=float), 3)
    y_pred = x ** 2 + np.random.default_rng(1).normal(0, 0.01, 18)       # curva de verdade
    lin = lack_of_fit_test(x, y_pred, x, nivel_por="conc")
    assert lin.computavel and not lin.linear
    (tmp_path / "model_card.md").write_text("# card\n", encoding="utf-8")
    rio.append_linearity_robustness_model_card(str(tmp_path), linearidade=lin)
    texto = (tmp_path / "model_card.md").read_text(encoding="utf-8")
    assert "curvatura detectada" in texto or "curvature detected" in texto
