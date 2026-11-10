"""B4a/B4b do parecer de 2026-09-25: a validacao agrupada (o diferencial do
projeto) tem que valer para CSV e para os leitores de instrumento, e a
interface nunca pode afirmar uma protecao que nao existe.

Reproduz, pelo ponto de entrada real, o achado R5-08: uma execucao via CSV sem
chave de grupo NUNCA exibe "Anti-leakage ativo".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import guaraci.pipeline as pq
from guaraci import dados_io, leitores_avancados as la
from tests._resumo import canonico


def _csv(tmp_path, com_grupo=True, sep=",", decimal="."):
    rng = np.random.default_rng(0)
    wn = np.linspace(4000, 10000, 60)
    linhas = []
    for k, cls in enumerate(["A", "B", "C"]):
        base = np.exp(-((wn - 5000 - 1200 * k) / 400) ** 2)
        for g in range(5):                      # 5 amostras fisicas por classe
            for r in range(3):                  # 3 replicas
                linhas.append([f"{cls}{g}", f"{cls}{g}_{r}", cls]
                              + list(1 + base * (1 + 0.1 * g) + 0.01 * rng.normal(size=60)))
    df = pd.DataFrame(linhas, columns=["grupo", "amostra", "classe"] + [f"{w:.1f}" for w in wn])
    caminho = tmp_path / "d.csv"
    df.to_csv(caminho, index=False, sep=sep, decimal=decimal)
    return caminho, df


def test_csv_com_coluna_de_grupo_devolve_mae_id_e_exclui_a_coluna_do_espectro(tmp_path):
    caminho, df = _csv(tmp_path)
    wn, X, rot, conc, mae = dados_io.load_csv_com_grupos(
        caminho, "classe", None, col_grupo="grupo")
    assert mae is not None and len(mae) == len(df)
    assert X.shape == (45, 60) and len(wn) == 60      # 'amostra' (texto) ignorada
    assert set(mae) == set(df["grupo"])


def test_csv_com_arquivo_de_grupos_usa_a_primeira_coluna_como_id(tmp_path):
    caminho, df = _csv(tmp_path)
    mapa = pd.DataFrame({"amostra": df["amostra"], "grupo_fisico": df["grupo"]})
    arq = tmp_path / "grupos.csv"
    mapa.to_csv(arq, index=False)
    *_, mae = dados_io.load_csv_com_grupos(caminho, "classe", None, arquivo_grupos=str(arq))
    assert np.array_equal(mae, df["grupo"].astype(str).values)


def test_csv_sem_fonte_de_grupo_devolve_none(tmp_path):
    caminho, _ = _csv(tmp_path)
    *_, mae = dados_io.load_csv_com_grupos(caminho, "classe", None)
    assert mae is None


def test_csv_ponto_e_virgula_pt_br_e_aceito_no_treino(tmp_path):
    """m-28: o CSV de treino detecta separador e virgula decimal."""
    caminho, df = _csv(tmp_path, sep=";", decimal=",")
    wn, X, rot, conc, mae = dados_io.load_csv_com_grupos(
        caminho, "classe", None, col_grupo="grupo")
    assert X.shape == (45, 60)
    assert np.isfinite(X).all()


@pytest.mark.parametrize("kw,trecho", [
    ({"col_grupo": "nao_existe"}, "coluna de grupo 'nao_existe' nao existe"),
    ({"col_grupo": "grupo", "arquivo_grupos": "x.csv"}, "so' UMA fonte"),
])
def test_erros_de_grupo_sao_acionaveis(tmp_path, kw, trecho):
    caminho, _ = _csv(tmp_path)
    with pytest.raises(ValueError, match=trecho):
        dados_io.load_csv_com_grupos(caminho, "classe", None, **kw)


def test_arquivo_de_grupos_incompleto_lista_as_amostras_sem_grupo(tmp_path):
    caminho, df = _csv(tmp_path)
    parcial = pd.DataFrame({"amostra": df["amostra"][:-3], "g": df["grupo"][:-3]})
    arq = tmp_path / "g.csv"
    parcial.to_csv(arq, index=False)
    with pytest.raises(ValueError, match="3 de 45 amostras nao tem grupo"):
        dados_io.load_csv_com_grupos(caminho, "classe", None, arquivo_grupos=str(arq))


def test_coluna_de_classe_inexistente_lista_as_colunas_do_csv(tmp_path):
    caminho, _ = _csv(tmp_path)
    with pytest.raises(ValueError, match="Colunas encontradas"):
        dados_io.load_csv_com_grupos(caminho, "classe_errada", None)


def test_garantia_prevista_por_modo_e_fonte_de_grupo():
    P = dados_io.garantia_agrupamento_prevista
    assert P(pq.Config(mode="dx")) == "high"
    assert P(pq.Config(mode="sintetico")) == "high"
    assert P(pq.Config(mode="csv")) == "none"
    assert P(pq.Config(mode="csv", group_column="g")) == "medium"
    assert P(pq.Config(mode="opus", group_file="g.csv")) == "medium"
    assert P(pq.Config(mode="spc")) == "none"
    assert P(pq.Config(mode="csv", group_column="g", group_by_mae_id=False)) == "none"


def test_load_data_declara_garantia_medium_e_none_e_nao_vaza_entre_execucoes(tmp_path):
    caminho, _ = _csv(tmp_path)
    cfg = pq.Config(mode="csv", csv_file=str(caminho), class_column="classe",
                    group_column="grupo")
    dados_io.load_data(cfg)
    assert cfg.grouping_guarantee == "medium"
    cfg.group_column = None
    dados_io.load_data(cfg)
    assert cfg.grouping_guarantee == "none"
    cfg.mode = "sintetico"
    dados_io.load_data(cfg)                      # recalculado: sem "none" de sobra
    assert cfg.grouping_guarantee == "high"


def test_checklist_da_cli_nao_afirma_anti_leakage_sem_grupo(tmp_path, monkeypatch):
    """R5-08 (revisor5): CSV sem `mae_id` nunca mostra 'Anti-leakage ativo'."""
    import guaraci.guaraci as g
    caminho, _ = _csv(tmp_path)
    cfg = pq.Config(mode="csv", csv_file=str(caminho), class_column="classe")
    _pode, _erros, checks = g._checklist(cfg)
    textos = " | ".join(t for _ok, t in checks)
    assert "Anti-leakage ativo" not in textos and "anti-vazamento ATIVO" not in textos
    assert "SEM validação agrupada" in textos or "NO grouped validation" in textos
    # Com grupo declarado, a garantia e' MEDIA, nao "ativo"
    cfg.group_column = "grupo"
    _pode, _erros, checks = g._checklist(cfg)
    textos = " | ".join(t for _ok, t in checks)
    assert "MÉDIA" in textos or "MEDIUM" in textos
    assert "anti-vazamento ATIVO" not in textos


def test_execucao_csv_com_grupo_e_agrupada_e_sem_grupo_declara_a_ausencia(tmp_path, capsys):
    """Fluxo real (`executar`): com grupo -> validacao agrupada; sem grupo ->
    StratifiedKFold declarado no resumo e ressalva final no log."""
    caminho, _ = _csv(tmp_path)
    base = dict(mode="csv", csv_file=str(caminho), class_column="classe",
                matrix_profile="generico", level="N1", objective="classificacao",
                show_plots=False, n_permutations=5, frac_holdout=0.0,
                run_benchmark=False, run_monte_carlo=False, run_shap=False,
                run_wold=False, run_cv_anova=False, run_opls=False,
                run_ddsimca=False, executar_etapa4=False, max_lvs=4,
                selecao_lv_cv_aninhada=False, seed=0,
                wn_min=4000, wn_max=10000)
    cfg_g = pq.Config(output_root_folder=str(tmp_path / "com"),
                      group_column="grupo", **base)
    pq.executar(cfg_g)
    saida_g = capsys.readouterr().out
    assert "GroupKFold" in saida_g or "group-aware ATIVA" in saida_g or "group-aware" in saida_g
    resumos = list((tmp_path / "com").rglob("resumo_modelo.txt"))
    assert resumos and "medium" in canonico(resumos[0]).lower()

    cfg_s = pq.Config(output_root_folder=str(tmp_path / "sem"), **base)
    pq.executar(cfg_s)
    saida_s = capsys.readouterr().out
    assert "VALIDACAO NAO AGRUPADA" in saida_s
    r = canonico(list((tmp_path / "sem").rglob("resumo_modelo.txt"))[0])
    assert "Grouping guarantee" in r and "none" in r.lower()


# ── Leitores de instrumento (B4a) ────────────────────────────────────────
def _parser_txt(caminho):
    dados = np.loadtxt(caminho)
    return dados[:, 0], dados[:, 1]


def _pasta_instrumento(tmp_path):
    pasta = tmp_path / "spc"
    pasta.mkdir()
    x = np.linspace(4000, 7000, 50)
    for cls in ("A", "B"):
        (pasta / cls).mkdir()
        for i in range(4):
            y = np.exp(-((x - (5000 if cls == "A" else 6000)) / 300) ** 2)
            np.savetxt(pasta / cls / f"{cls}_{i}.txt", np.column_stack([x, y]))
    return pasta


def test_leitor_de_instrumento_sem_grupo_devolve_none_e_com_arquivo_devolve_mae_id(tmp_path):
    pasta = _pasta_instrumento(tmp_path)
    *_, mae0, _meta = la.load_generic_file_dataset(
        str(pasta), _parser_txt, "TXT", extensoes=(".txt",))
    assert mae0 is None
    mapa = pd.DataFrame({"amostra": [f"{c}_{i}" for c in "AB" for i in range(4)],
                         "grupo": [f"{c}{i // 2}" for c in "AB" for i in range(4)]})
    arq = tmp_path / "g.csv"
    mapa.to_csv(arq, index=False)
    *_, mae1, _meta = la.load_generic_file_dataset(
        str(pasta), _parser_txt, "TXT", extensoes=(".txt",),
        arquivo_grupos=str(arq))
    assert mae1 is not None and len(set(mae1)) == 4


def test_leitor_registrado_repassa_group_file_da_config(tmp_path, monkeypatch):
    """O adaptador registrado (`_leitor_spc`) le `cfg.group_file` -- o caminho
    real que o usuario alcanca via config/menu."""
    pasta = tmp_path / "spc_real"
    pasta.mkdir()
    x = np.linspace(4000, 7000, 50)
    for cls in ("A", "B"):
        for i in range(4):
            y = np.exp(-((x - (5000 if cls == "A" else 6000)) / 300) ** 2)
            np.savetxt(pasta / f"{cls}_{i}.spc", np.column_stack([x, y]))
    mapa = pd.DataFrame({"amostra": [f"{c}_{i}" for c in "AB" for i in range(4)],
                         "grupo": [f"{c}{i // 2}" for c in "AB" for i in range(4)]})
    arq = tmp_path / "g.csv"
    mapa.to_csv(arq, index=False)
    monkeypatch.setattr(la, "parse_spc", _parser_txt)   # sem .spc binario aqui
    cfg = pq.Config(mode="spc", input_folder=str(pasta), group_file=str(arq))
    *_, mae, _meta = la._leitor_spc(cfg)
    assert mae is not None and len(set(mae)) == 4
    cfg2 = pq.Config(mode="spc", input_folder=str(pasta))
    *_, mae2, _meta2 = la._leitor_spc(cfg2)
    assert mae2 is None
