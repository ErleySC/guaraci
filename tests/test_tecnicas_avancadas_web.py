# -*- coding: utf-8 -*-
"""Testes web (Streamlit AppTest) de app_tabs/tecnicas.py -- equivalente da
aba [T] Tecnicas Avancadas do CLI (tests/test_tecnicas_avancadas_cli.py),
confirmando paridade CLI/web real (nao so' alegada) p/ ASCA/EPO-GLSW/
MCR-ALS/fusao multibloco.

`AppTest.from_function` so' enxerga o CORPO da funcao passada (extraido e
reescrito como script isolado) -- nenhum helper de modulo e' visivel de
dentro dela (mesmo padrao ja usado em test_fluxo_cego_bloco9b.py), entao
cada `_app()` abaixo e' auto-contida, sem chamar nada definido fora dela."""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_aba_tecnicas_carrega_sem_excecao():
    def _app() -> None:
        import types
        import numpy as np
        import pandas as pd
        from guaraci.app_tabs import tecnicas
        from guaraci.config import Config

        rng = np.random.default_rng(0)
        classes = np.repeat(["Copaiba", "Andiroba", "Buriti"], 8)
        n = len(classes)
        base = {"Copaiba": 1.0, "Andiroba": 2.0, "Buriti": 3.0}
        X = np.array([base[c] + rng.normal(0, 0.1, 30) for c in classes])
        wn = np.linspace(4000, 10000, 30)
        mae_id = np.array([f"G{i // 2}" for i in range(n)])
        conc = np.where(classes == "Andiroba", rng.uniform(1, 10, n), np.nan)
        sessao = np.tile(["S1", "S2"], n // 2 + 1)[:n]
        metadados = pd.DataFrame({"especie": classes, "sessao": sessao})

        pq = types.SimpleNamespace()
        pq.Config = Config
        pq.load_data = lambda cfg: (wn, X, classes, conc, mae_id, metadados)
        pq.validate_input = lambda X, wn, rot, conc, mae: (X, wn, rot, conc, mae, {})
        tecnicas.render(pq, pq.Config(), lambda s: s)

    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]


def test_asca_end_to_end_web():
    def _app() -> None:
        import types
        import numpy as np
        import pandas as pd
        from guaraci.app_tabs import tecnicas
        from guaraci.config import Config

        rng = np.random.default_rng(0)
        classes = np.repeat(["Copaiba", "Andiroba", "Buriti"], 8)
        n = len(classes)
        base = {"Copaiba": 1.0, "Andiroba": 2.0, "Buriti": 3.0}
        X = np.array([base[c] + rng.normal(0, 0.1, 30) for c in classes])
        wn = np.linspace(4000, 10000, 30)
        mae_id = np.array([f"G{i // 2}" for i in range(n)])
        conc = np.where(classes == "Andiroba", rng.uniform(1, 10, n), np.nan)
        sessao = np.tile(["S1", "S2"], n // 2 + 1)[:n]
        metadados = pd.DataFrame({"especie": classes, "sessao": sessao})

        pq = types.SimpleNamespace()
        pq.Config = Config
        pq.load_data = lambda cfg: (wn, X, classes, conc, mae_id, metadados)
        pq.validate_input = lambda X, wn, rot, conc, mae: (X, wn, rot, conc, mae, {})
        tecnicas.render(pq, pq.Config(), lambda s: s)

    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    at.button(key="btn_tec_asca_load").click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    at.multiselect(key="tec_asca_fatores").set_value(["especie"])
    at.run()
    at.button(key="btn_tec_asca_run").click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    saida = "\n".join(str(el.value) for el in at.markdown) + \
        "\n".join(str(el.value) for el in at.success)
    assert "variance" in saida.lower() or "especie" in saida.lower()


def test_epo_glsw_end_to_end_web():
    def _app() -> None:
        import types
        import numpy as np
        import pandas as pd
        from guaraci.app_tabs import tecnicas
        from guaraci.config import Config

        rng = np.random.default_rng(0)
        classes = np.repeat(["Copaiba", "Andiroba", "Buriti"], 8)
        n = len(classes)
        base = {"Copaiba": 1.0, "Andiroba": 2.0, "Buriti": 3.0}
        X = np.array([base[c] + rng.normal(0, 0.1, 30) for c in classes])
        wn = np.linspace(4000, 10000, 30)
        mae_id = np.array([f"G{i // 2}" for i in range(n)])
        conc = np.where(classes == "Andiroba", rng.uniform(1, 10, n), np.nan)
        sessao = np.tile(["S1", "S2"], n // 2 + 1)[:n]
        metadados = pd.DataFrame({"especie": classes, "sessao": sessao})

        pq = types.SimpleNamespace()
        pq.Config = Config
        pq.load_data = lambda cfg: (wn, X, classes, conc, mae_id, metadados)
        pq.validate_input = lambda X, wn, rot, conc, mae: (X, wn, rot, conc, mae, {})
        tecnicas.render(pq, pq.Config(), lambda s: s)

    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    at.button(key="btn_tec_epo_load").click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    at.selectbox(key="tec_epo_f1").set_value("especie")
    at.selectbox(key="tec_epo_f2").set_value("sessao")
    at.run()
    at.button(key="btn_tec_epo_run").click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    saida = "\n".join(str(el.value) for el in at.success)
    assert "EPO" in saida


def test_mcr_als_end_to_end_web():
    def _app() -> None:
        import types
        import numpy as np
        import pandas as pd
        from guaraci.app_tabs import tecnicas
        from guaraci.config import Config

        rng = np.random.default_rng(0)
        classes = np.repeat(["Copaiba", "Andiroba", "Buriti"], 8)
        n = len(classes)
        base = {"Copaiba": 1.0, "Andiroba": 2.0, "Buriti": 3.0}
        X = np.abs(np.array([base[c] + rng.normal(0, 0.1, 30) for c in classes]))
        wn = np.linspace(4000, 10000, 30)
        mae_id = np.array([f"G{i // 2}" for i in range(n)])
        sessao = np.tile(["S1", "S2"], n // 2 + 1)[:n]
        metadados = pd.DataFrame({"especie": classes, "sessao": sessao})

        pq = types.SimpleNamespace()
        pq.Config = Config
        pq.load_data = lambda cfg: (wn, X, classes, None, mae_id, metadados)
        pq.validate_input = lambda X, wn, rot, conc, mae: (X, wn, rot, conc, mae, {})
        tecnicas.render(pq, pq.Config(), lambda s: s)

    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    at.button(key="btn_tec_mcr_load").click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    at.button(key="btn_tec_mcr_run").click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    saida = "\n".join(str(el.value) for el in at.success)
    assert "iteration" in saida.lower()


def test_fusao_multibloco_end_to_end_web():
    def _app() -> None:
        import types
        import numpy as np
        from guaraci.app_tabs import tecnicas
        from guaraci.config import Config

        rng = np.random.default_rng(7)
        n, p1, p2 = 30, 250, 200
        y = rng.uniform(0, 10, n)
        bloco_a = y[:, None] * 0.1 + rng.normal(0, 0.05, (n, p1))
        bloco_b = y[:, None] * 0.2 + rng.normal(0, 0.05, (n, p2))
        rot = np.array(["oleo"] * n)
        mae = np.array([f"G{i}" for i in range(n)])
        chamadas = {"n": 0}

        def _load_data(cfg):
            chamadas["n"] += 1
            if chamadas["n"] == 1:
                return np.linspace(4000, 10000, p1), bloco_a, rot, y, mae, None
            return np.linspace(700, 4000, p2), bloco_b, rot, None, mae, None

        pq = types.SimpleNamespace()
        pq.Config = Config
        pq.load_data = _load_data
        pq.validate_input = lambda X, wn, rot, conc, mae: (X, wn, rot, conc, mae, {})
        tecnicas.render(pq, pq.Config(), lambda s: s)

    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    at.text_input(key="tec_fusao_caminho_1").set_value("pasta_a")
    at.text_input(key="tec_fusao_caminho_2").set_value("pasta_b")
    at.run()
    at.button(key="btn_tec_fusao_run").click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    saida = "\n".join(str(el.value) for el in at.success)
    assert "RMSEP" in saida
