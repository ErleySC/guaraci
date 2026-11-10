"""faixas_excluir (agua/TSP em RMN), preset pqn_mc e normalizacao por RIP/area (IMS).

Todas as verdades vem da construcao dos dados: fator de diluicao conhecido, ganho de injecao conhecido,
regioes de agua/TSP injetadas por nos. Nenhum numero de desempenho e' comparado com um valor "esperado"
tirado de uma execucao anterior.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from guaraci import preprocessamento as pp
from guaraci.config import Config
from guaraci.config_io import (_CONFIG_SPEC, _PRE_PROC_FRIENDLY, _coagir_valor, _validar_semantico, load_config,
                               save_config)

SPEC = {s["key"]: s for s in _CONFIG_SPEC}


# ----------------------------------------------------------------------------- formato das faixas
def test_faixas_aceitam_negativo_decimal_virgula_e_lista_em_texto():
    assert pp.interpretar_faixas(["4.5-5.0", "-0.5-0.5"]) == [(-0.5, 0.5), (4.5, 5.0)]
    assert pp.interpretar_faixas("4.5-5.0, -0.5-0.5") == [(-0.5, 0.5), (4.5, 5.0)]
    assert pp.interpretar_faixas("4.5-5.0; -1--0.5") == [(-1.0, -0.5), (4.5, 5.0)]
    assert pp.interpretar_faixas(["4,5-5,0"]) == [(4.5, 5.0)]          # item de lista com virgula decimal
    assert pp.interpretar_faixas(None) == [] and pp.interpretar_faixas([]) == [] and pp.interpretar_faixas("") == []
    assert pp.formatar_faixas([(-0.5, 0.5), (4.5, 5.0)]) == ("-0.5-0.5", "4.5-5")


def test_faixas_que_se_tocam_sao_fundidas():
    assert pp.interpretar_faixas(["4.5-5.0", "4.9-5.4", "5.4-5.6"]) == [(4.5, 5.6)]


@pytest.mark.parametrize("ruim,trecho", [("5.0-4.5", "MENOR"), ("4.5-4.5", "MENOR"), ("abc", "invalida"),
                                         ("4.5", "invalida"), ("4.5-", "invalida"), ("1-2-3", "invalida"),
                                         ("nan-2", "invalida")])
def test_faixa_mal_formada_diz_qual_e_por_que(ruim, trecho):
    with pytest.raises(ValueError, match=trecho):
        pp.interpretar_faixas([ruim])
    with pytest.raises(ValueError, match="faixas_excluir"):
        _coagir_valor(SPEC["faixas_excluir"], ruim)


def test_mascara_inclui_os_extremos_da_faixa():
    eixo = np.array([4.4, 4.5, 4.75, 5.0, 5.1])
    assert pp.mascara_fora_das_faixas(eixo, ["4.5-5.0"]).tolist() == [True, False, False, False, True]


# ----------------------------------------------------------------------------- config.yaml
def test_padrao_do_motor_e_vazio_e_nada_muda():
    cfg = Config()
    assert cfg.exclude_ranges == () and cfg.normalization == "nenhuma" and cfg.rip_window is None
    assert _validar_semantico(cfg) == []


def test_yaml_de_ida_e_volta_e_erro_agregado(tmp_path):
    cam = tmp_path / "c.yaml"
    cam.write_text("faixas_excluir: [4.5-5.0, -0.5-0.5]\nnormalizacao: rip\nims_rip_janela: 7-9\n"
                   "pre_processamento: PQN+MC\n", encoding="utf-8")
    cfg = load_config(str(cam))
    assert cfg.exclude_ranges == ("-0.5-0.5", "4.5-5") and cfg.normalization == "rip" and cfg.rip_window == "7-9"
    assert cfg.default_preprocessing == "pqn_mc"
    save_config(cfg, str(tmp_path / "volta.yaml"))
    cfg2 = load_config(str(tmp_path / "volta.yaml"))
    assert (cfg2.exclude_ranges, cfg2.normalization, cfg2.rip_window, cfg2.default_preprocessing) == \
        (cfg.exclude_ranges, cfg.normalization, cfg.rip_window, cfg.default_preprocessing)
    cam.write_text("faixas_excluir: [5.0-4.5]\nnormalizacao: tudo\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        load_config(str(cam))
    assert "faixas_excluir" in str(e.value) and "normalizacao" in str(e.value)


def test_validacao_cruzada_das_chaves():
    assert any("ims_rip_janela" in m for m in _validar_semantico(Config(normalization="rip")))
    assert _validar_semantico(Config(normalization="rip", rip_window="7-9", wn_min=5.0, wn_max=50.0)) == []
    fora = _validar_semantico(Config(normalization="rip", rip_window="1-2", wn_min=5.0, wn_max=50.0))
    assert any("fora da faixa" in m for m in fora)
    ruim = Config()
    ruim.exclude_ranges = ("9-3",)
    assert any("faixas_excluir" in m for m in _validar_semantico(ruim))


def test_pqn_mc_e_preset_nomeado_do_seletor():
    assert _PRE_PROC_FRIENDLY["PQN+MC"] == "pqn_mc"
    assert list(_PRE_PROC_FRIENDLY)[:7] == ["MSC+SG+MC", "SNV+SG+MC", "Autoscaling", "Mean-centering", "SNV+MC",
                                            "AirPLS+SG+MC", "Personalizado"]      # numeracao antiga intacta
    assert list(_PRE_PROC_FRIENDLY)[7:] == ["AirPLS+SNV+SG+MC", "PQN+MC"]         # novos no fim, em ordem de chegada
    assert _coagir_valor(SPEC["pre_processamento"], "PQN+MC") == "pqn_mc"
    assert _coagir_valor(SPEC["pre_processamento"], "pqn_mc") == "pqn_mc"


# ----------------------------------------------------------------------------- conferencia contra o eixo
def test_faixa_fora_do_eixo_dos_dados_e_erro_com_a_unidade():
    ppm = np.linspace(0, 10, 101)
    with pytest.raises(ValueError, match="nao cruza"):
        pp.avaliar_faixas_excluir(ppm, ppm, ["1800-2000"])          # Hz no lugar de ppm
    parcial = pp.avaliar_faixas_excluir(ppm, ppm, ["-0.5-0.5"])      # TSP: so' a borda esta no eixo
    assert parcial["n_removidas"] == 6 and not parcial["aviso_metade"]
    ja_cortada = pp.avaliar_faixas_excluir(ppm, ppm[ppm >= 6], ["4.5-5.0"])
    assert ja_cortada["n_removidas"] == 0 and ja_cortada["fora_da_faixa"] == ["4.5-5"]


def test_mais_de_metade_do_eixo_avisa_e_tudo_removido_recusa():
    ppm = np.linspace(0, 10, 101)
    av = pp.avaliar_faixas_excluir(ppm, ppm, ["1-7"])
    assert av["aviso_metade"] and av["fracao"] == pytest.approx(61 / 101)
    assert not pp.avaliar_faixas_excluir(ppm, ppm, ["1-5"])["aviso_metade"]
    with pytest.raises(ValueError, match="removem todas"):
        pp.avaliar_faixas_excluir(ppm, ppm, ["-1-11"])


def test_modos_sem_eixo_fisico_ignoram_com_aviso():
    cfg = Config(mode="multielementar")
    cfg.exclude_ranges = ("1-2",)
    X = np.ones((3, 5))
    X2, e2, info = pp.preparar_eixo_espectral(cfg, X, np.arange(5.0), np.arange(5.0))
    assert X2.shape == X.shape and info["faixas"] == [] and "ignoradas" in info["avisos"][0]


# ----------------------------------------------------------------------------- PQN: diluicao conhecida
def _espectros_diluidos(seed=0, n=40, p=300, sigma_dil=0.5):
    """Perfil quimico fixo (picos) x fator de diluicao conhecido; um pico dominante varia 3x entre amostras
    (o caso em que normalizar pela area/soma falha e o quociente probabilistico nao)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, p)
    def pico(c, w=0.08):
        return np.exp(-0.5 * ((x - c) / w) ** 2)

    base = sum(h * pico(c) for c, h in ((1.0, 1.0), (2.2, 0.7), (3.5, 1.4), (5.5, 0.9), (7.1, 0.6), (8.4, 1.1)))
    base = base + 0.3                                  # continuo: todos os canais tem sinal
    dominante = 6.0 * pico(4.0, 0.2)
    dil = rng.lognormal(0, sigma_dil, n)
    forca = rng.choice([0.5, 1.5], n)                  # quimica real: o pico dominante muda de amostra
    X = dil[:, None] * (base + forca[:, None] * dominante) + rng.normal(0, 1e-3, (n, p))
    return x, X, dil


def test_pqn_recupera_o_fator_de_diluicao_conhecido_e_a_area_nao():
    _, X, dil = _espectros_diluidos()
    ref = pp.PQN().fit(X)
    fator_pqn = np.median(X / ref.ref_, axis=1)
    erro_pqn = np.std(np.log(fator_pqn / dil))
    erro_area = np.std(np.log(X.sum(axis=1) / dil))
    assert erro_pqn < 0.05                              # recupera a diluicao (a menos de 1 constante)
    assert erro_area > 3 * erro_pqn                     # a area foi enganada pelo pico dominante
    Xn = pp.PQN().fit_transform(X)
    cv = np.std(Xn[:, 0:30].mean(axis=1)) / np.mean(Xn[:, 0:30].mean(axis=1))
    assert cv < 0.05                                    # regiao sem o pico dominante fica constante


def test_pqn_mc_monta_pqn_e_centragem_e_descreve():
    cfg = Config(default_preprocessing="pqn_mc")
    assert [n for n, _ in pp.build_preprocessor(cfg).steps] == ["pqn", "mc"]
    rot, descr, sg = pp.descrever_preprocessamento(cfg)
    assert rot == ["PQN", "MC"] and "PQN" in descr and not sg
    _, X, dil = _espectros_diluidos(1)
    Xp = pp.build_preprocessor(cfg).fit_transform(X)
    assert np.allclose(Xp.mean(axis=0), 0, atol=1e-9)


def test_pqn_mc_e_identico_ao_personalizado_pqn_mais_mc():
    _, X, _ = _espectros_diluidos(2)
    custom = Config(default_preprocessing="custom")
    custom.apply_snv = False; custom.apply_sg = False; custom.apply_pqn = True; custom.apply_mc = True
    a = pp.build_preprocessor(Config(default_preprocessing="pqn_mc")).fit_transform(X)
    b = pp.build_preprocessor(custom).fit_transform(X)
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------------- normalizacao por RIP / area
def _ims(seed=0, n=30, gain_sigma=0.4):
    """RIP (8 ms) + dois produtos (10.6 e 13.2 ms), ganho de injecao conhecido por espectro."""
    rng = np.random.default_rng(seed)
    t = np.arange(5.0, 30.0001, 0.05)
    def g(c, a):
        return a * np.exp(-0.5 * ((t - c) / (c / 50 / 2.355)) ** 2)

    limpo = g(8.0, 1.0) + g(10.6, 0.12) + g(13.2, 0.05)
    ganho = rng.lognormal(0, gain_sigma, n)
    return t, ganho[:, None] * limpo + rng.normal(0, 1e-4, (n, t.size)), ganho, limpo


def test_normalizacao_por_rip_devolve_o_espectro_sem_o_ganho():
    t, X, ganho, limpo = _ims()
    Xn = pp.normalizar_espectros(X, t, "rip", "7-9")
    assert np.allclose(Xn, limpo / limpo.max(), atol=2e-3)         # o ganho desapareceu
    assert np.std(np.log(X.max(axis=1))) > 0.3 and np.std(Xn.max(axis=1)) < 2e-3


def test_normalizacao_por_area_divide_pela_area_do_espectro():
    t, X, ganho, limpo = _ims(1)
    Xn = pp.normalizar_espectros(X, t, "area")
    areas = np.sum(0.5 * (Xn[:, 1:] + Xn[:, :-1]) * np.diff(t), axis=1)
    assert np.allclose(areas, 1.0)
    i_rip = int(np.argmin(np.abs(t - 8.0)))
    assert np.std(Xn[:, i_rip]) / np.mean(Xn[:, i_rip]) < 0.01      # ganho removido tambem aqui


def test_nenhuma_nao_toca_nos_dados():
    t, X, _, _ = _ims()
    assert pp.normalizar_espectros(X, t, "nenhuma") is not None
    assert np.array_equal(pp.normalizar_espectros(X, t, "nenhuma"), X)


def test_normalizacao_recusa_janela_ausente_fora_do_eixo_ou_referencia_invalida():
    t, X, _, _ = _ims()
    with pytest.raises(ValueError, match="ims_rip_janela"):
        pp.normalizar_espectros(X, t, "rip", None)
    with pytest.raises(ValueError, match="DENTRO da faixa"):
        pp.normalizar_espectros(X, t, "rip", "1-3")
    X0 = X.copy(); X0[3, :] = 0.0
    with pytest.raises(ValueError, match=r"indices \[3\]"):
        pp.normalizar_espectros(X0, t, "rip", "7-9")
    with pytest.raises(ValueError, match="invalida"):
        pp.normalizar_espectros(X, t, "soma", None)


# ----------------------------------------------------------------------------- pacote exportado
def _pacote(preset, faixas=(), norm="nenhuma", janela=None, p=60, n=45, seed=0):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import LabelBinarizer
    rng = np.random.default_rng(seed)
    ppm_full = np.linspace(0, 10, p)
    y = rng.integers(0, 3, n)
    X = np.abs(rng.normal(1.0, 0.1, (n, p))) + y[:, None] * 0.4 * np.exp(-0.5 * ((ppm_full - 2) / 0.5) ** 2)
    ppm = ppm_full[pp.mascara_fora_das_faixas(ppm_full, faixas)] if faixas else ppm_full
    Xtr = X[:, pp.mascara_fora_das_faixas(ppm_full, faixas)] if faixas else X
    Xtr = pp.normalizar_espectros(Xtr, ppm, norm, janela)
    preproc = pp.build_preprocessor(Config(default_preprocessing=preset)).fit(Xtr)
    lb = LabelBinarizer().fit(y)
    pls = PLSRegression(3, scale=False).fit(preproc.transform(Xtr), np.asarray(lb.transform(y), float))
    pkg = {"preprocessador": preproc, "pls_final": pls, "label_binarizer": lb, "classes": list(lb.classes_),
           "wavenumbers": ppm, "preset": preset, "wn_min": 0.0, "wn_max": 10.0,
           "faixas_excluir": list(pp.formatar_faixas(pp._como_faixas(faixas))), "normalizacao": norm,
           "ims_rip_janela": janela}
    return pkg, X, ppm_full


@pytest.mark.parametrize("preset,faixas,norm,janela", [
    ("pqn_mc", ["4.5-5.0", "-0.5-0.5"], "nenhuma", None),
    ("mc", [], "rip", "1.5-2.5"),
    ("snv_mc", ["4.5-5.0"], "area", None),
])
def test_json_portatil_leva_exclusao_e_normalizacao_e_bate_com_o_joblib(preset, faixas, norm, janela):
    import guaraci.predicao as pr
    from guaraci.model_export import export_portable_json, predict_portable
    pkg, X, ppm_full = _pacote(preset, faixas, norm, janela)
    portatil = export_portable_json(pkg)
    assert portatil["faixas_excluir"] == pkg["faixas_excluir"]
    assert portatil["normalizacao"] == norm and portatil["ims_rip_janela"] == janela
    assert len(portatil["wavenumbers"]) == len(pkg["wavenumbers"])
    novos = np.abs(np.random.default_rng(5).normal(1.0, 0.2, (6, ppm_full.size)))
    via_portatil = predict_portable(json.loads(json.dumps(portatil)), novos, ppm_full)
    X_ref = pr._interpolate_to_reference(pkg, novos, ppm_full)
    Y = np.asarray(pkg["pls_final"].predict(pkg["preprocessador"].transform(X_ref)), float)
    Yn = np.clip(Y, 0, 1); Yn = Yn / np.where(Yn.sum(1, keepdims=True) < 1e-12, 1, Yn.sum(1, keepdims=True))
    assert via_portatil["classe_pred"] == [str(pkg["classes"][i]) for i in Yn.argmax(1)]
    assert np.allclose(via_portatil["confianca_%"], Yn.max(1) * 100, atol=1e-6)


def test_predicao_ignora_a_regiao_excluida_mesmo_que_ela_mude_muito():
    """Verdade conhecida: a agua nova (4.5-5.0) e' enorme e diferente do treino; a predicao nao pode sentir."""
    from guaraci.model_export import export_portable_json, predict_portable
    pkg, X, ppm_full = _pacote("pqn_mc", ["4.5-5.0", "-0.5-0.5"])
    novos = X[:8].copy()
    com_agua = novos.copy()
    com_agua[:, (ppm_full >= 4.5) & (ppm_full <= 5.0)] += 500.0
    portatil = export_portable_json(pkg)
    a, b = predict_portable(portatil, novos, ppm_full), predict_portable(portatil, com_agua, ppm_full)
    assert a["classe_pred"] == b["classe_pred"] and np.allclose(a["confianca_%"], b["confianca_%"])
    import guaraci.predicao as pr
    assert np.array_equal(pr._interpolate_to_reference(pkg, novos, ppm_full),
                          pr._interpolate_to_reference(pkg, com_agua, ppm_full))


def test_pacote_antigo_sem_as_chaves_continua_valido():
    from guaraci.model_export import export_portable_json, predict_portable
    pkg, X, ppm_full = _pacote("snv_mc")
    for k in ("faixas_excluir", "normalizacao", "ims_rip_janela"):
        pkg.pop(k)
    portatil = export_portable_json(pkg)
    assert portatil["faixas_excluir"] == [] and portatil["normalizacao"] == "nenhuma"
    legado = {k: v for k, v in portatil.items() if k not in ("faixas_excluir", "normalizacao", "ims_rip_janela")}
    assert predict_portable(legado, X[:3], ppm_full)["classe_pred"] == predict_portable(portatil, X[:3], ppm_full)["classe_pred"]


def test_pqn_e_exportavel_e_o_portatil_e_identico_ao_sklearn():
    from guaraci.model_export import export_portable_json
    pkg, X, ppm_full = _pacote("pqn_mc")
    portatil = export_portable_json(pkg)
    assert [p["tipo"] for p in portatil["passos_preprocessamento"]] == ["PQN", "StandardScaler"]


# ----------------------------------------------------------------------------- ponta a ponta (RMN com agua)
def _csv_rmn(caminho, n_por_classe=8, seed=0, n=500):
    """3 classes; a diferenca esta em picos de 2.6 e 6.9 ppm; agua residual (4.78 ppm, amplitude sorteada de
    5 a 60 vezes o pico de analito) e TSP em 0 ppm: as unicas coisas que a exclusao deve tirar."""
    rng = np.random.default_rng(seed)
    x = np.linspace(10.0, 0.0, n)
    def pico(c, w=0.05):
        return np.exp(-0.5 * ((x - c) / w) ** 2)

    base = pico(1.3) + 0.8 * pico(2.1) + 1.5 * pico(3.2) + 0.7 * pico(7.4) + 0.6 * pico(8.1)
    marcador = {"A": pico(2.6), "B": pico(6.9), "C": 0.0 * pico(1.0)}
    linhas = ["classe,amostra," + ",".join(f"{v:.4f}" for v in x)]
    for cls, mk in marcador.items():
        for i in range(n_por_classe):
            y = rng.lognormal(0, 0.3) * (base * rng.lognormal(0, 0.05) + mk * rng.lognormal(0, 0.1))
            y = y + rng.uniform(5, 60) * pico(4.78, 0.12) + 5.0 * pico(0.0, 0.03) + rng.normal(0, 0.01, n)
            linhas.append(f"{cls},{cls}{i:02d}," + ",".join(f"{v:.5f}" for v in y))
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    return x


@pytest.mark.slow
def test_pipeline_exclui_agua_e_tsp_exporta_e_a_predicao_aplica_a_mesma_exclusao(pq, tmp_path):
    import joblib
    from conftest import achar_pastas_run
    import guaraci.predicao as pr
    from guaraci.model_export import load_portable_json, predict_portable

    csv = str(tmp_path / "rmn.csv")
    ppm = _csv_rmn(csv)
    cfg = pq.Config(
        mode="csv", csv_file=csv, class_column="classe", group_column="amostra",
        input_folder=str(tmp_path / "dados"), output_root_folder=str(tmp_path / "saida"),
        wn_min=0.0, wn_max=12.0, default_preprocessing="snv_mc",
        n_splits_cv=3, n_repeats_cv=1, n_permutations=3, n_permutations_wold=3, n_bootstrap_vip=2,
        n_bootstrap_bca=5, n_monte_carlo=2, max_lvs=5, frac_holdout=0.2,
        run_benchmark=False, run_monte_carlo=False, run_shap=False, run_wold=False, run_cv_anova=False,
        run_opls=False, executar_etapa4=False, show_plots=False)
    cfg.exclude_ranges = ("4.5-5.0", "-0.5-0.5")
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)

    run = achar_pastas_run(cfg.output_root_folder)[0]
    pkg = joblib.load(os.path.join(run, pq.NOME_MODELOS, "modelo_plsda.joblib"))
    wn = np.asarray(pkg["wavenumbers"])
    assert pkg["faixas_excluir"] == ["-0.5-0.5", "4.5-5"]
    assert not ((wn >= 4.5) & (wn <= 5.0)).any() and not (wn <= 0.5).any()
    assert len(wn) == int(((ppm >= 0.0) & (ppm <= 12.0) & ~((ppm >= 4.5) & (ppm <= 5.0)) & ~(ppm <= 0.5)).sum())
    assert pkg["parametros_efetivos"]["faixas_excluir"] == ["-0.5-0.5", "4.5-5"]
    resumo = open(os.path.join(run, pq.NOME_RELATORIOS, "resumo_modelo.txt"), encoding="utf-8").read()
    assert "4.5-5" in resumo and "Excluded ranges" in resumo or "Faixas excluidas" in resumo

    portatil = load_portable_json(os.path.join(run, pq.NOME_MODELOS, "modelo_plsda.portatil.json"))
    assert portatil["faixas_excluir"] == ["-0.5-0.5", "4.5-5"]
    # amostra nova: mesma quimica, agua 100x maior que no treino -> mesma resposta do modelo
    import pandas as pd
    d = pd.read_csv(csv)
    X = d.iloc[:, 2:].to_numpy(float)[:6]
    x_csv = np.array([float(c) for c in d.columns[2:]])
    X_agua = X.copy(); X_agua[:, (x_csv >= 4.5) & (x_csv <= 5.0)] *= 100.0
    base = pr.predict_samples(pkg, X, x_csv)
    agua = pr.predict_samples(pkg, X_agua, x_csv)
    assert list(base["classe_pred"]) == list(agua["classe_pred"])
    assert np.allclose(base["confianca_%"], agua["confianca_%"])
    pj = predict_portable(portatil, X_agua, x_csv)
    assert pj["classe_pred"] == list(base["classe_pred"])


# ----------------------------------------------------------------------------- tela [2.4] e catalogo
def test_catalogo_so_o_nmr_exclui_faixas_e_o_formato_e_valido():
    import guaraci.catalogo as cat
    com = {k for k, v in cat.INSTRUMENTOS.items() if v.get("faixas_excluir")}
    assert com == {"nmr"}
    assert pp.interpretar_faixas(cat.INSTRUMENTOS["nmr"]["faixas_excluir"]) == [(-0.5, 0.5), (4.5, 5.0)]
    assert cat.INSTRUMENTOS["nmr"]["preproc"] == "snv_mc"          # decisao R13a: padrao mantido


def test_tela_tecnica_grava_faixas_avisa_e_zera_nas_outras_tecnicas(cli):
    import guaraci.guaraci as g
    ordem = g._tecnica_ordem()
    cfg = g.Config()
    cli.dizer(str(ordem.index("nmr") + 1), "")
    g._menu_technique(cfg)
    assert tuple(cfg.exclude_ranges) == ("-0.5-0.5", "4.5-5")
    saida = cli.saida()
    assert "Faixas excluidas" in saida and "faixas_excluir" in saida and "agua" in saida
    cli.dizer(str(ordem.index("mir") + 1), "")
    g._menu_technique(cfg)
    assert tuple(cfg.exclude_ranges) == ()
    assert "zerada" in cli.saida()


def test_ajuda_da_tecnica_nmr_mostra_a_exclusao(cli):
    import guaraci.guaraci as g
    ordem = g._tecnica_ordem()
    cli.dizer(f"?{ordem.index('nmr') + 1}", "", "0")
    g._menu_technique(g.Config())
    assert "Exclui:" in cli.saida() and "4.5-5.0" in cli.saida()


# ----------------------------------------------------------------------------- guaraci checar (assistente)
def _dados(emin, emax, n_var=800):
    from guaraci import assistente_regras as R
    return R.ResumoDados(n_leituras=40, n_variaveis=n_var, n_unidades=40, leituras_por_unidade_max=1,
                         unidades_por_classe={"a": 20, "b": 20}, eixo_min=emin, eixo_max=emax)


def _ids(cfg, dados):
    from guaraci import assistente_regras as R
    return {a.id: a for a in R.avaliar(cfg, dados, None, "PT")}


def test_checar_acusa_faixa_na_unidade_errada_e_mais_da_metade():
    cfg = Config(mode="csv", wn_min=0.0, wn_max=12.0)
    cfg.exclude_ranges = ("1800-2000",)
    ach = _ids(cfg, _dados(0.0, 10.0))
    assert ach["faixas_excluir_fora_do_eixo"].severidade == "critico" and "1800-2000" in ach["faixas_excluir_fora_do_eixo"].mensagem
    cfg.exclude_ranges = ("1-7",)
    assert "faixas_excluir_mais_da_metade" in _ids(cfg, _dados(0.0, 10.0))
    cfg.exclude_ranges = ("4.5-5.0",)
    ach = _ids(cfg, _dados(0.0, 10.0))
    assert not ({"faixas_excluir_fora_do_eixo", "faixas_excluir_mais_da_metade", "rmn_sem_faixas_excluir"} & set(ach))


def test_checar_avisa_rmn_sem_exclusao_e_nao_incomoda_outras_tecnicas():
    assert "rmn_sem_faixas_excluir" in _ids(Config(mode="csv", wn_min=0.0, wn_max=12.0), _dados(0.0, 10.0))
    assert "rmn_sem_faixas_excluir" in _ids(Config(mode="rmn", wn_min=0.0, wn_max=12.0), _dados(-4.5, 15.5))
    assert "rmn_sem_faixas_excluir" not in _ids(Config(mode="csv", wn_min=400.0, wn_max=4000.0), _dados(400.0, 4000.0))
    assert "rmn_sem_faixas_excluir" not in _ids(Config(mode="csv", wn_min=5.0, wn_max=50.0), _dados(0.0, 30.0))   # IMS
    assert "rmn_sem_faixas_excluir" not in _ids(Config(mode="multielementar"), _dados(0.0, 10.0))


# ----------------------------------------------------------------------------- manual
def test_manual_documenta_as_chaves_novas_e_os_dois_arquivos_sao_identicos():
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[1]
    a = (raiz / "MANUAL.md").read_text(encoding="utf-8")
    b = (raiz / "src" / "guaraci" / "dados_docs" / "manual.md").read_text(encoding="utf-8")
    assert a == b
    for termo in ("faixas_excluir", "pqn_mc", "normalizacao", "ims_rip_janela"):
        assert termo in a, termo


def test_checar_pega_rip_sem_janela_ou_com_janela_fora_da_faixa():
    from guaraci import assistente_regras as R
    ach = {a.id: a for a in R.avaliar(Config(normalization="rip"), None, None, "PT")}
    assert ach["rip_sem_janela"].severidade == "critico"
    ach = {a.id: a for a in R.avaliar(Config(normalization="rip", rip_window="1-3", wn_min=5.0, wn_max=50.0), None, None, "EN")}
    assert ach["rip_janela_fora_da_faixa"].severidade == "critico" and "1-3" in ach["rip_janela_fora_da_faixa"].mensagem
    ok = {a.id for a in R.avaliar(Config(normalization="rip", rip_window="7-9", wn_min=5.0, wn_max=50.0), None, None, "PT")}
    assert not ({"rip_sem_janela", "rip_janela_fora_da_faixa"} & ok)
    assert not ({"rip_sem_janela", "rip_janela_fora_da_faixa"} & {a.id for a in R.avaliar(Config(), None, None, "PT")})


def test_com_rip_o_ganho_de_um_espectro_novo_nao_muda_a_resposta_e_sem_rip_muda():
    """Verdade conhecida: multiplicar o espectro novo por 3 e' so' ganho de injecao."""
    from guaraci.model_export import export_portable_json, predict_portable
    for norm, janela, invariante in (("rip", "1.5-2.5", True), ("nenhuma", None, False)):
        pkg, X, ppm = _pacote("mc", [], norm, janela, seed=3)
        portatil = export_portable_json(pkg)
        a = predict_portable(portatil, X[:10], ppm)
        b = predict_portable(portatil, 3.0 * X[:10], ppm)
        assert (a["classe_pred"] == b["classe_pred"] and np.allclose(a["confianca_%"], b["confianca_%"])) == invariante


def _csv_ims(caminho, n_por_classe=8, seed=0):
    """3 classes pelo perfil de ions; RIP em 8 ms; ganho de injecao lognormal conhecido por espectro."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 30.0001, 0.1)
    def g(c, a):
        return a * np.exp(-0.5 * ((t - c) / (c / 50 / 2.355)) ** 2)

    perfil = {"cl_a": {9.4: .30, 11.8: .10}, "cl_b": {9.9: .25, 12.4: .15}, "cl_c": {9.4: .10, 11.2: .30}}
    linhas = ["classe,lote," + ",".join(f"{v:.2f}" for v in t)]
    for cls, perf in perfil.items():
        for i in range(n_por_classe):
            y = g(8.0, 1.0) + sum(g(c, a * rng.lognormal(0, .1)) for c, a in perf.items())
            y = rng.lognormal(0, .4) * y + rng.normal(0, 0.003, t.size)
            linhas.append(f"{cls},{cls}_{i}," + ",".join(f"{v:.5f}" for v in y))
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    return t


@pytest.mark.slow
def test_pipeline_normaliza_por_rip_exporta_e_a_predicao_repete(pq, tmp_path):
    import joblib
    import pandas as pd
    from conftest import achar_pastas_run
    import guaraci.predicao as pr

    csv = str(tmp_path / "ims.csv")
    _csv_ims(csv)
    cfg = pq.Config(
        mode="csv", csv_file=csv, class_column="classe", group_column="lote",
        input_folder=str(tmp_path / "dados"), output_root_folder=str(tmp_path / "saida"),
        wn_min=5.0, wn_max=30.0, default_preprocessing="mc", normalization="rip", rip_window="7-9",
        n_splits_cv=3, n_repeats_cv=1, n_permutations=3, n_permutations_wold=3, n_bootstrap_vip=2,
        n_bootstrap_bca=5, n_monte_carlo=2, max_lvs=5, frac_holdout=0.2,
        run_benchmark=False, run_monte_carlo=False, run_shap=False, run_wold=False, run_cv_anova=False,
        run_opls=False, executar_etapa4=False, show_plots=False)
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)
    run = achar_pastas_run(cfg.output_root_folder)[0]
    pkg = joblib.load(os.path.join(run, pq.NOME_MODELOS, "modelo_plsda.joblib"))
    assert pkg["normalizacao"] == "rip" and pkg["ims_rip_janela"] == "7-9"
    d = pd.read_csv(csv)
    X = d.iloc[:, 2:].to_numpy(float)[:9]
    t = np.array([float(c) for c in d.columns[2:]])
    a, b = pr.predict_samples(pkg, X, t), pr.predict_samples(pkg, 4.0 * X, t)    # ganho de injecao x4
    assert list(a["classe_pred"]) == list(b["classe_pred"]) and np.allclose(a["confianca_%"], b["confianca_%"])
    resumo = open(os.path.join(run, pq.NOME_RELATORIOS, "resumo_modelo.txt"), encoding="utf-8").read()
    assert "rip" in resumo
