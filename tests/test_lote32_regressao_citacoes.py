"""Lote 3.2 da vistoria: RMSECV pooled, ajuda de `teste_wold` e citacoes (V73, V74, V80, V96)."""
from __future__ import annotations

import inspect

import numpy as np

from guaraci import catalogo_opcoes, chemometric_stats, pipeline as P, reports, resultados_io
from guaraci.config import Config


def _cenario_duas_classes():
    """Cenario de A6-01: a classe B tem teor com ruido 4x maior (semente fixa)."""
    rng = np.random.default_rng(0)
    eixo = np.linspace(0, 1, 120)
    X, y, rot, mae = [], [], [], []
    for e, (amp, sc, n) in enumerate([(1.0, 1.0, 40), (0.5, 4.0, 20)]):
        for g in range(n // 2):
            c = rng.uniform(0, 20)
            for _ in range(2):
                s = (amp * np.exp(-((eixo - 0.3 - 0.2 * e) / 0.1) ** 2)
                     + 0.01 * c * np.exp(-((eixo - 0.7) / 0.05) ** 2))
                s = s + rng.normal(0, 0.003 * sc, eixo.size)
                X.append(s); y.append(c); rot.append(f"E{e}"); mae.append(f"E{e}_{g}")
    return np.array(X), np.array(y), np.array(rot), np.array(mae)


def test_rmsecv_relatado_e_pooled_e_coerente_com_as_classes(tmp_path):
    X, y, rot, mae = _cenario_duas_classes()
    cfg = Config()
    cfg.default_preprocessing = "mc"
    cfg.max_lvs = 8
    r = P.pls_regression_by_species(X, y, rot, mae, np.unique(rot), cfg, str(tmp_path), 5)
    por_classe = {t["especie"]: t["rmsecv"] for t in r["tabela_especie"]}
    assert set(por_classe) == {"E0", "E1"}
    # A classe ruidosa tem RMSECV bem maior; o relatado nao pode ser o da classe boa.
    assert por_classe["E1"] > 2 * por_classe["E0"]
    # Pooled: entre o menor e o maior RMSECV por classe (media quadratica ponderada).
    assert min(por_classe.values()) < r["rmsecv"] < max(por_classe.values())
    assert r["rmsecv"] > 3 * por_classe["E0"]  # antes: so' a classe maior (E0), 0,112
    # Mesma populacao do RMSEP pooled: mesma ordem de grandeza (antes: 0,112 x 0,405).
    assert 0.5 < r["rmsecv"] / r["rmsep"] < 2.0
    assert r["rmsecv"] > r["rmsec"]


def test_ajuda_teste_wold_descreve_a_permutacao_e_o_padrao_real():
    e = catalogo_opcoes.OPCOES["teste_wold"]
    texto = " ".join(str(x) for x in (e["nome"], e["faz"], e["impacto"])).lower()
    assert "permutacao" in texto and "r2y" in texto and "q2y" in texto
    assert "dentro de 2% do melhor" not in texto
    assert e["default"] is False
    assert e["default"] == Config().run_wold


def test_regra_dos_2pct_nao_e_atribuida_a_wold():
    for texto in [t for _, t in resultados_io._NOTAS_METODOLOGICAS] + [t for _, t, _ in resultados_io._NOTAS_MODEL_CARD]:
        assert "criterio de parcimonia de wold" not in texto.lower()
        assert "wold parsimony criterion" not in texto.lower()
    src = inspect.getsource(P)
    assert "Wold parsimony criterion (Wold 1978)" not in src


def test_reports_sem_citacoes_sem_fonte():
    src = inspect.getsource(reports)
    assert "Wold 1984" not in src
    assert "[Eriksson 2008]" not in src
    assert "Eriksson et al. 2006" in src


def test_rpd_traz_ressalva_de_convencao_e_critica_publicada():
    doc = chemometric_stats.rpd_rer.__doc__
    assert "convencao de NIR agricola" in doc
    assert "10.1255/nirn.1462" in doc
    assert "convencao de NIR agricola" in chemometric_stats.interpret_rpd.__doc__


def test_rpd_nao_atribui_os_cortes_a_williams_nem_a_aacc():
    """Auditoria A-06: os cortes nao foram conferidos no texto original; o codigo nao pode afirmar a fonte."""
    for doc in (chemometric_stats.rpd_rer.__doc__, chemometric_stats.interpret_rpd.__doc__):
        assert "esta pendente" not in doc and "ainda por conferir" not in doc
    assert "SEM fonte unica verificada" in chemometric_stats.rpd_rer.__doc__
    assert "NAO atribuir os cortes" in chemometric_stats.rpd_rer.__doc__
