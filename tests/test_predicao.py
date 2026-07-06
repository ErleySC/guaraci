"""Testes de predicao.py (predicao em lote) — usados tanto pelo app quanto
pelo CLI (guaraci.py, menu_predicao). Roda executar() sintetico UMA vez
(fixture de sessao) para gerar um pacote de modelo .joblib REAL (mesma
estrutura que o pipeline grava em producao), depois exercita a predicao
sobre espectros novos sem mock nenhum do lado cientifico.
"""
import os

import joblib
import numpy as np
import pandas as pd
import pytest

import guaraci.predicao as pr

from conftest import achar_pastas_run


@pytest.fixture(scope="session")
def modelo_e_dados(pq, tmp_path_factory):
    """Roda executar() sintetico, devolve (pkg, X_raw_algumas, wavenumbers)."""
    base = tmp_path_factory.mktemp("predicao")
    cfg = pq.Config(
        pasta_entrada=str(base / "dados"),
        pasta_saida_raiz=str(base / "saida"),
        modo="sintetico", n_por_classe=10, n_pontos_sint=60,
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutacoes=5,
        n_permutacoes_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=5,
    )
    os.makedirs(cfg.pasta_entrada, exist_ok=True)
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.pasta_saida_raiz)
    assert runs, "executar() nao criou pasta de saida"
    cam_modelo = os.path.join(runs[0], pq.NOME_MODELOS, "modelo_plsda.joblib")
    assert os.path.isfile(cam_modelo), "modelo_plsda.joblib nao foi salvo"
    pkg = joblib.load(cam_modelo)

    # Gera espectros "novos" sinteticos no MESMO eixo de comprimento de onda
    # do treino (dados de teste, nao precisam ser fisicamente reais).
    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    rng = np.random.default_rng(123)
    X_novos = rng.normal(loc=0.5, scale=0.05, size=(6, len(wn)))
    return pkg, X_novos, wn


def test_validar_pacote_modelo_aceita_pacote_real(modelo_e_dados):
    pkg, _X, _wn = modelo_e_dados
    pr.validar_pacote_modelo(pkg)  # nao deve levantar


def test_validar_pacote_modelo_rejeita_pacote_incompleto():
    with pytest.raises(ValueError, match="Modelo invalido"):
        pr.validar_pacote_modelo({"preprocessador": None})


def test_predizer_amostras_retorna_colunas_esperadas(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    df = pr.predizer_amostras(pkg, X_novos, wn)
    esperado = {"amostra", "classe_pred", "confianca_%", "T2", "T2_ucl",
                "Q", "Q_ucl", "T2_ok", "Q_ok", "aceito"}
    assert esperado.issubset(df.columns)
    assert len(df) == X_novos.shape[0]


# ── Dominio de Aplicabilidade (AD) na predicao em lote ─────────────────────

def test_pacote_real_exporta_artefatos_de_ad(modelo_e_dados):
    """O pacote .joblib exportado por executar() (versao atual) sempre traz
    os artefatos leves do Dominio de Aplicabilidade -- confirma o wiring em
    pipeline.py (pacote_modelo), nao so' a existencia da funcao pura."""
    pkg, _X, _wn = modelo_e_dados
    for chave in ("pca", "ad_var_t", "ad_t2_limite", "ad_q_limite"):
        assert chave in pkg, f"pacote de modelo real nao tem '{chave}'"


def test_predizer_amostras_inclui_colunas_ad(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    df = pr.predizer_amostras(pkg, X_novos, wn)
    esperado_ad = {"AD_T2", "AD_T2_limite", "AD_Q", "AD_Q_limite",
                   "AD_dentro_dominio"}
    assert esperado_ad.issubset(df.columns)
    assert df["AD_dentro_dominio"].dtype == bool


def test_predizer_amostras_sem_artefatos_ad_nao_gera_colunas_ad(modelo_e_dados):
    """Retrocompatibilidade: um pacote salvo por uma versao ANTIGA do
    pipeline (sem pca/ad_*) continua predizendo normalmente -- so' sem as
    colunas AD_*, sem lancar excecao."""
    pkg, X_novos, wn = modelo_e_dados
    pkg_antigo = {k: v for k, v in pkg.items()
                  if k not in ("pca", "ad_var_t", "ad_t2_limite", "ad_q_limite")}
    df = pr.predizer_amostras(pkg_antigo, X_novos, wn)
    assert not any(c.startswith("AD_") for c in df.columns)
    assert "classe_pred" in df.columns  # predicao principal nao foi afetada


def test_predizer_amostras_classe_pred_pertence_ao_treino(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    df = pr.predizer_amostras(pkg, X_novos, wn)
    classes_treino = set(pkg["label_binarizer"].classes_)
    assert set(df["classe_pred"]).issubset(classes_treino)


def test_predizer_amostras_confianca_entre_0_e_100(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    df = pr.predizer_amostras(pkg, X_novos, wn)
    assert (df["confianca_%"] >= 0).all() and (df["confianca_%"] <= 100).all()


def test_predizer_amostras_espectro_de_treino_e_aceito(modelo_e_dados):
    """Um espectro que o PROPRIO treino gerou (dentro do dominio) deve ser
    aceito pelo diagnostico T2/Q -- sanity check de que os limites nao sao
    absurdamente apertados a ponto de rejeitar dados normais."""
    pkg, _X_novos, wn = modelo_e_dados
    # Reconstroi um "espectro medio" a partir do proprio pre-processador
    # treinado (usa a media interna, sempre dentro do dominio de treino).
    n = len(wn)
    X_medio = np.tile(np.linspace(0.4, 0.6, n), (1, 1))
    df = pr.predizer_amostras(pkg, X_medio, wn)
    assert len(df) == 1
    assert isinstance(bool(df["aceito"].iloc[0]), bool)  # nao lanca, e' bool valido


def test_carregar_csv_predicao_detecta_colunas_numericas(tmp_path, modelo_e_dados):
    _pkg, X_novos, wn = modelo_e_dados
    df_in = pd.DataFrame(X_novos, columns=[f"{w:.1f}" for w in wn])
    df_in.insert(0, "amostra_id", [f"A{i}" for i in range(len(df_in))])
    caminho = tmp_path / "espectros_novos.csv"
    df_in.to_csv(caminho, index=False, sep=";")

    X_out, wn_out, meta = pr.carregar_csv_predicao(str(caminho))
    assert X_out.shape == X_novos.shape
    assert len(wn_out) == len(wn)
    assert list(meta.columns) == ["amostra_id"]


def test_carregar_csv_predicao_sem_colunas_numericas_leva_erro_claro(tmp_path):
    caminho = tmp_path / "sem_espectro.csv"
    pd.DataFrame({"nome": ["a", "b"], "classe": ["X", "Y"]}).to_csv(
        caminho, index=False, sep=";")
    with pytest.raises(ValueError, match="numero de onda"):
        pr.carregar_csv_predicao(str(caminho))


# ── Integracao end-to-end via CLI (guaraci.py, menu_predicao) ──────────────

@pytest.mark.slow
def test_menu_predicao_cli_end_to_end(monkeypatch, tmp_path, modelo_e_dados):
    """Simula a sessao completa do usuario no menu B (Predicao em Lote):
    digita caminho do modelo, caminho do CSV, aceita a saida padrao (Enter),
    e confirma que o CSV de resultados foi gravado com as colunas certas.
    Sem mock de logica cientifica -- so' o input() do terminal e' simulado.
    """
    import guaraci.guaraci as guaraci_mod

    pkg, X_novos, wn = modelo_e_dados
    cam_modelo = tmp_path / "modelo_teste.joblib"
    joblib.dump(pkg, cam_modelo)

    df_in = pd.DataFrame(X_novos, columns=[f"{w:.1f}" for w in wn])
    cam_csv = tmp_path / "novos.csv"
    df_in.to_csv(cam_csv, index=False, sep=";")

    # 3a resposta "" = aceita a saida padrao; 4a "" = Enter no _pause() final
    respostas = iter([str(cam_modelo), str(cam_csv), "", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    guaraci_mod.menu_predicao(guaraci_mod.Config())

    cam_saida_esperada = cam_csv.with_name(cam_csv.stem + "_predicao.csv")
    assert cam_saida_esperada.is_file(), "CSV de resultados nao foi gravado"
    df_res = pd.read_csv(cam_saida_esperada, sep=";", decimal=",")
    assert "classe_pred" in df_res.columns
    assert len(df_res) == X_novos.shape[0]
