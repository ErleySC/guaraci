# -*- coding: utf-8 -*-
"""R14 (fase 2): PLS-DA com classes ORDENADAS (niveis de teor, claro/medio/escuro) perde a classe do
meio (mascaramento da regressao linear sobre indicadores; Hastie, Tibshirani & Friedman, ESL, sec. 4.2).

Verdade conhecida: 3 classes = 3 niveis de UMA propriedade (banda de intensidade proporcional ao nivel),
ruido pequeno; uma LDA separa as 3 quase sem erro, o PLS-DA com 2 VLs erra a classe do meio. Provas:
(a) PLS-LDA resolve o mascaramento; (b) a deteccao acusa rampa so' quando ha' estrutura E efeito (nao
acusa classes sem ordem, nem rampa sem mascaramento); (c) aviso no resumo/model card/relatorios;
(d) `classificador: plslda` e' opt-in, e o padrao nao muda; (e) o JSON portatil reproduz o PLS-LDA."""
from __future__ import annotations

import re

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import guaraci.pipeline as pq
from guaraci.classificadores import PLSLDAClassifier, detectar_classes_em_rampa
from guaraci.io_registry import register_reader
from guaraci.resumo_parse import avisos_do_modelo, ler_resumo_canonico


def ordinal(n=30, seed=0, ruido=0.02, k=3):
    rng = np.random.default_rng(seed)
    wn = np.arange(200)
    banda = np.exp(-0.5 * ((wn - 80) / 12) ** 2)
    base = np.exp(-0.5 * ((wn - 140) / 30) ** 2)
    X, y = [], []
    for c in range(k):
        for _ in range(n):
            nivel = (c + 1) * 2.0 + rng.normal(0, 0.6)
            X.append(base + 0.05 * nivel * banda + ruido * rng.standard_normal(len(wn)) + rng.normal(0, 0.02))
            y.append(c)
    return wn, np.array(X), np.array(y)


def sem_ordem(n=30, seed=0, ruido=0.02):
    """3 classes sem ordem: cada uma com a sua banda."""
    rng = np.random.default_rng(seed)
    wn = np.arange(200)
    X, y = [], []
    for c, centro in enumerate((40, 100, 160)):
        for _ in range(n):
            X.append(np.exp(-0.5 * ((wn - centro) / 10) ** 2) * (1 + 0.1 * rng.standard_normal())
                     + ruido * rng.standard_normal(len(wn)))
            y.append(c)
    return wn, np.array(X), np.array(y)


def _cv(cls, X, y, k=2):
    Y = np.eye(3)[y]
    pred = np.zeros(len(y), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler(with_std=False).fit(X[tr])
        m = cls(n_components=k, scale=False).fit(sc.transform(X[tr]), Y[tr])
        pred[te] = np.argmax(m.predict(sc.transform(X[te])), axis=1)
    return pred


@pytest.mark.parametrize("semente", [0, 1, 2])
def test_plslda_resolve_o_mascaramento_da_classe_do_meio(semente):
    _wn, X, y = ordinal(seed=semente)
    da = _cv(PLSRegression, X, y)
    lda = _cv(PLSLDAClassifier, X, y)
    rec_da = recall_score(y, da, average=None)
    rec_lda = recall_score(y, lda, average=None)
    assert rec_da[1] < 0.5                       # PLS-DA: a classe do meio some
    assert balanced_accuracy_score(y, da) < 0.8
    assert rec_lda[1] > 0.75                     # PLS-LDA: volta
    assert balanced_accuracy_score(y, lda) > 0.85
    assert balanced_accuracy_score(y, lda) - balanced_accuracy_score(y, da) > 0.10


def test_deteccao_acusa_rampa_com_estrutura_e_efeito():
    _wn, X, y = ordinal(seed=0)
    Y = np.eye(3)[y]
    Xc = StandardScaler(with_std=False).fit_transform(X)
    T = PLSRegression(2, scale=False).fit(Xc, Y).x_scores_
    pred = _cv(PLSRegression, X, y)
    d = detectar_classes_em_rampa(T, y.astype(str), pred.astype(str))
    assert d["aplicavel"] and d["estrutura"] and d["mascaramento"] and d["detectado"]
    assert d["frac_colinear"] > 0.95
    assert d["ordem"] in (["0", "1", "2"], ["2", "1", "0"])
    assert d["classe_mascarada"] == "1" and d["sens_meio"] <= d["sens_pontas"] - 0.2


def test_deteccao_nao_acusa_se_o_modelo_acerta_a_classe_do_meio():
    """Rampa (estrutura) mas sem mascaramento: com o PLS-LDA nao ha' o que avisar."""
    _wn, X, y = ordinal(seed=0)
    Xc = StandardScaler(with_std=False).fit_transform(X)
    T = PLSRegression(2, scale=False).fit(Xc, np.eye(3)[y]).x_scores_
    pred = _cv(PLSLDAClassifier, X, y)
    d = detectar_classes_em_rampa(T, y.astype(str), pred.astype(str))
    assert d["estrutura"] and not d["mascaramento"] and not d["detectado"]


def test_deteccao_nao_acusa_classes_sem_ordem_nem_duas_classes():
    _wn, X, y = sem_ordem()
    Xc = StandardScaler(with_std=False).fit_transform(X)
    T = PLSRegression(2, scale=False).fit(Xc, np.eye(3)[y]).x_scores_
    pred = _cv(PLSRegression, X, y)
    d = detectar_classes_em_rampa(T, y.astype(str), pred.astype(str))
    assert d["aplicavel"] and not d["estrutura"] and not d["detectado"]
    assert d["frac_colinear"] < 0.95
    # mesmo que o modelo erre muito a "classe do meio", sem estrutura de rampa nao se acusa
    pred_ruim = pred.copy()
    pred_ruim[y == 1] = 0
    assert not detectar_classes_em_rampa(T, y.astype(str), pred_ruim.astype(str))["detectado"]
    # duas classes: nao ha' classe do meio
    m = y < 2
    d2 = detectar_classes_em_rampa(T[m], y[m].astype(str), pred[m].astype(str))
    assert not d2["aplicavel"] and not d2["detectado"]


def test_deteccao_nao_acusa_classes_ao_acaso():
    """Sem sinal nenhum as medias de 3 classes saem quase colineares por sorte e a do meio perde por sorte:
    exigir pontas bem reconhecidas evita o falso alarme (aconteceu antes desse criterio)."""
    acusou = 0
    for semente in range(40):
        rng = np.random.default_rng(semente)
        X = rng.standard_normal((36, 60))
        y = np.repeat([0, 1, 2], 12)
        Xc = X - X.mean(axis=0)
        T = PLSRegression(2, scale=False).fit(Xc, np.eye(3)[y]).x_scores_
        pred = _cv(PLSRegression, X, y)
        acusou += detectar_classes_em_rampa(T, y.astype(str), pred.astype(str))["detectado"]
    assert acusou == 0


def test_plslda_predicoes_ate_k_igual_a_um_ajuste_por_k():
    """O atalho de CV (um ajuste com K VLs, LDA nos k primeiros escores) bate com ajustar PLS-LDA(k)."""
    _wn, X, y = ordinal(n=20, seed=3)
    Xc = StandardScaler(with_std=False).fit_transform(X)
    Y = np.eye(3)[y]
    tr, te = np.arange(0, 45), np.arange(45, 60)
    cheio = PLSLDAClassifier(n_components=5, scale=False).fit(Xc[tr], Y[tr])
    atalho = cheio.predicoes_ate_k(Xc[tr], Y[tr], Xc[te])
    for k in (1, 2, 4, 5):
        direto = PLSLDAClassifier(n_components=k, scale=False).fit(Xc[tr], Y[tr]).predict(Xc[te])
        assert np.allclose(atalho[k - 1], direto, atol=1e-6), k
    # probabilidades: linhas somam 1
    assert np.allclose(cheio.predict(Xc[te]).sum(axis=1), 1.0)


def _rodar(tmp_path, nome, classificador="plsda", dados=ordinal, **extra):
    wn, X, y = dados(n=24, seed=0)
    rotulos = np.array([f"nivel_{c}" for c in y])
    grupos = np.array([f"{r}_g{i // 2}" for i, r in enumerate(rotulos)])   # 2 replicas por amostra fisica
    register_reader("rampa_teste", lambda cfg: (wn.astype(float), X, rotulos, None, grupos, None))
    cfg = pq.Config(mode="rampa_teste", level="N1", objective="classificacao",
                    output_root_folder=str(tmp_path / nome), show_plots=False, run_benchmark=False,
                    run_monte_carlo=False, run_shap=False, run_wold=False, run_cv_anova=False, run_opls=False,
                    run_ddsimca=False, executar_etapa4=False, n_permutations=5, frac_holdout=0.0, seed=0,
                    max_lvs=4, wn_min=0.0, wn_max=199.0, default_preprocessing="mc",
                    selecao_lv_cv_aninhada=False, classificador=classificador, **extra)
    pq.executar(cfg)
    pasta = next((tmp_path / nome).rglob("resumo_modelo.txt"))
    return ler_resumo_canonico(str(pasta)), pasta.parent


def _ba(resumo):
    return float(re.search(r"Balanced accuracy.*?:\s*([\d.]+)", resumo, re.I).group(1))


@pytest.fixture(scope="module")
def execucoes(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("rampa")
    return {"raiz": raiz, "da": _rodar(raiz, "da", "plsda"), "lda": _rodar(raiz, "lda", "plslda")}


def test_pipeline_avisa_classes_em_rampa_e_o_plslda_resolve(execucoes):
    resumo_da, pasta_da = execucoes["da"]
    resumo_lda, pasta_lda = execucoes["lda"]
    avisos = avisos_do_modelo(resumo_da)
    assert [a[0] for a in avisos] == ["classes_em_rampa"]
    _cod, pt, en = avisos[0]
    assert "mascaramento do PLS-DA" in pt and "classificador: plslda" in pt
    assert "masking" in en and "classificador: plslda" in en
    assert _ba(resumo_da) < 0.8                                   # antes
    assert _ba(resumo_lda) > 0.9                                  # depois
    assert avisos_do_modelo(resumo_lda) == []                     # resolvido: sem aviso
    # o aviso traz a comparacao na MESMA validacao cruzada
    assert re.search(r"PLS-LDA da acuracia balanceada 0\.\d\d contra 0\.\d\d do PLS-DA", pt)
    # resumo legivel (idioma da sessao: pt nos testes), model card e relatorios
    assert "AVISOS DO MODELO: CLASSIFICACAO" in (pasta_da / "resumo_modelo.txt").read_text(encoding="utf-8")
    assert "Classes em rampa" in (pasta_da / "model_card.md").read_text(encoding="utf-8")
    assert "Classes em rampa" not in (pasta_lda / "model_card.md").read_text(encoding="utf-8")


def test_relatorios_pdf_docx_xlsx_mostram_o_aviso(execucoes):
    from guaraci.reports import generate_excel_report, generate_pdf_report, generate_word_report
    _r, pasta = execucoes["da"]
    projeto = {"nome": "teste", "autor": "x", "inst": "y", "tipo": "z", "objetivo": ""}
    pdf = generate_pdf_report(str(pasta.parent), projeto).getvalue()
    from io import BytesIO
    from pypdf import PdfReader
    texto_pdf = " ".join(p.extract_text() for p in PdfReader(BytesIO(pdf)).pages)
    assert "AVISOS DO MODELO" in texto_pdf and "mascaramento do PLS-DA" in texto_pdf.replace("\n", " ")
    import docx
    d = docx.Document(BytesIO(generate_word_report(str(pasta.parent), projeto).getvalue()))
    assert any("mascaramento do PLS-DA" in p.text for p in d.paragraphs)
    import openpyxl
    ws = openpyxl.load_workbook(BytesIO(generate_excel_report(str(pasta.parent)).getvalue())).worksheets[0]
    assert any("mascaramento do PLS-DA" in str(c.value) for row in ws.iter_rows() for c in row if c.value)
    _r2, pasta2 = execucoes["lda"]
    d2 = docx.Document(BytesIO(generate_word_report(str(pasta2.parent), projeto).getvalue()))
    assert not any("mascaramento" in p.text for p in d2.paragraphs)


def test_json_portatil_reproduz_o_plslda(execucoes):
    import joblib
    from guaraci.model_export import export_portable_json, predict_portable
    pkg = joblib.load(next((execucoes["raiz"] / "lda").rglob("modelo_plsda.joblib")))
    assert isinstance(pkg["pls_final"], PLSLDAClassifier)
    portatil = export_portable_json(pkg)
    assert portatil["versao_formato"] == 3 and "plslda" in portatil
    wn, Xn, yn = ordinal(n=20, seed=11)
    saida = predict_portable(portatil, Xn, wn.astype(float))
    Xp = pkg["preprocessador"].transform(Xn)
    Y = pkg["pls_final"].predict(Xp)
    esperado = [pkg["label_binarizer"].classes_[i] for i in np.argmax(Y, axis=1)]
    assert saida["classe_pred"] == esperado
    assert np.mean(np.array(saida["classe_pred"]) == np.array([f"nivel_{c}" for c in yn])) > 0.75
    # o padrao (PLS-DA) segue no formato 2 e sem o campo
    pkg_da = joblib.load(next((execucoes["raiz"] / "da").rglob("modelo_plsda.joblib")))
    assert export_portable_json(pkg_da)["versao_formato"] == 2


def test_padrao_continua_plsda_e_chave_invalida_e_recusada(tmp_path):
    from guaraci.config_io import load_config
    cfg = pq.Config()
    assert cfg.classificador == "plsda"
    assert type(pq._novo_pls_classificador(cfg, 2)) is PLSRegression
    cfg.classificador = "plslda"
    assert isinstance(pq._novo_pls_classificador(cfg, 2), PLSLDAClassifier)
    cfg.classificador = "svm"
    with pytest.raises(ValueError):
        pq._novo_pls_classificador(cfg, 2)
    p = tmp_path / "c.yaml"
    p.write_text("classificador: plslda\n", encoding="utf-8")
    assert load_config(str(p)).classificador == "plslda"
    p.write_text("classificador: lda\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_sem_ordem_o_pipeline_nao_emite_aviso_de_rampa(tmp_path):
    resumo, _ = _rodar(tmp_path, "livre", "plsda", dados=sem_ordem)
    assert avisos_do_modelo(resumo) == []
