"""M-08 do parecer de 2026-09-25: os dois mutantes que sobreviviam a suite inteira.

m3 -- remover `mae_id = mae_id[keep]` (sincronizacao na remocao de NaN)
      desloca os grupos e vaza replicas SEM erro nem aviso.
m4 -- desligar a divisao de holdout por grupo (a alegacao central "incluindo o
      holdout externo").
Cada teste abaixo FALHA se a respectiva linha for mutada.
"""
import numpy as np

import guaraci.pipeline as pq
from guaraci.io_registry import register_reader


def _dados(nan_na_linha0=False):
    rng = np.random.default_rng(0)
    p = 120
    wn = np.linspace(4000, 10000, p)
    X, y, g = [], [], []
    for k in range(3):
        base = np.exp(-((wn - 5000 - k * 600) / 300) ** 2)
        for s in range(8):
            a = 1 + 0.2 * rng.standard_normal()
            for _ in range(3):
                X.append(1 + base * a + 0.01 * rng.standard_normal(p))
                y.append(f"C{k}")
                g.append(f"C{k}_S{s}")
    X = np.array(X)
    if nan_na_linha0:
        X[0, 5] = np.nan
    return wn, X, np.array(y), np.array(g)


def test_m3_remocao_de_nan_mantem_grupos_alinhados_as_amostras():
    wn, X, y, g = _dados(nan_na_linha0=True)
    X2, _wn2, y2, _conc2, g2, _rel = pq.validate_input(X, wn, y, None, g)
    assert len(g2) == len(y2) == len(X2) == len(X) - 1
    # cada grupo restante tem que continuar dentro de UMA classe: se os grupos
    # ficaram deslocados uma posicao, um grupo passa a misturar classes
    for grupo in np.unique(g2):
        assert len(set(y2[g2 == grupo])) == 1, f"grupo {grupo} misturou classes (deslocamento)"


def test_m4_holdout_principal_e_dividido_por_grupo(tmp_path, monkeypatch):
    wn, X, y, g = _dados()
    import guaraci.io_registry as _ior
    monkeypatch.setattr(_ior, "_LEITORES", dict(_ior._LEITORES))   # nao vaza o leitor "mut"
    register_reader("mut", lambda cfg: (wn, X, y, None, g, None))
    chamadas = []
    original = pq.holdout_agrupado_estratificado

    def espiao(rot, grupos, frac, semente):
        tr, ho = original(rot, grupos, frac, semente)
        chamadas.append((np.asarray(grupos), tr, ho))
        return tr, ho

    monkeypatch.setattr(pq, "holdout_agrupado_estratificado", espiao)
    cfg = pq.Config(mode="mut", level="N1", objective="classificacao",
                    output_root_folder=str(tmp_path), show_plots=False,
                    run_benchmark=False, run_monte_carlo=False, run_shap=False,
                    run_wold=False, run_cv_anova=False, run_opls=False,
                    run_ddsimca=False, executar_etapa4=False, n_permutations=3,
                    frac_holdout=0.25, seed=0, max_lvs=4, wn_min=4000, wn_max=10000)
    pq.executar(cfg)
    assert chamadas, "o holdout NAO passou pela divisao agrupada (mutante m4)"
    grupos, tr, ho = chamadas[0]
    assert not (set(grupos[tr]) & set(grupos[ho])), "replica do mesmo grupo dos dois lados do holdout"
