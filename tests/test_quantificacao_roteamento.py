# -*- coding: utf-8 -*-
"""P2 (fase 2): na quantificacao com mais de uma classe, o modelo de teor e' escolhido pela classe
PREVISTA. Com a classificacao fraca o R2 de validacao do teor pode ficar negativo (-0,57 contra 0,88 com
uma classe so', no checkup de cromatografia; aqui, -2,2 contra 1,00).

Obs.: a degradacao depende da semente (3 de 15 execucoes numa varredura); seed=3 a reproduz.

Verdade conhecida: 3 especies com matrizes quase iguais (classificacao ao acaso), teor 0-10 conhecido e
INDEPENDENTE da especie (banda do analito identica). Provas: (a) o aviso aparece quando a acuracia
balanceada do roteamento e' baixa e some quando e' alta; (b) `quantificacao_rotulo_treino: verdadeiro`
(opt-in) treina com a classe verdadeira e valida roteando pela prevista -- o R2 agregado volta ao do
modelo de classe unica; (c) o padrao nao muda: com o roteamento ja' correto os dois modos dao o mesmo
resultado e sem a chave o resultado e' o da v1.0."""
from __future__ import annotations

import re

import numpy as np
import pytest

import guaraci.pipeline as pq
from guaraci.io_registry import register_reader
from guaraci.resumo_parse import avisos_do_modelo, ler_resumo_canonico


def dados_p2(n_lotes=8, n_rep=2, seed=3, ruido=0.01, uma_classe=False):
    """3 especies (matriz comum + variacao de lote), analito em banda propria, teor aleatorio por lote."""
    rng = np.random.default_rng(seed)
    wn = np.arange(4000, 10000, 30.0)
    g = lambda c, w: np.exp(-0.5 * ((wn - c) / w) ** 2)  # noqa: E731
    base = 0.5 * g(5200, 300) + 0.4 * g(6900, 400) + 0.3 * g(8300, 500)
    analito = g(6000, 120)
    X, y, grupos, conc = [], [], [], []
    for esp in ("especie_A", "especie_B", "especie_C"):
        for lote in range(n_lotes):
            teor = rng.uniform(0, 10)
            matriz = 1.0 + 0.05 * rng.standard_normal()
            for _ in range(n_rep):
                esp_sp = base * matriz + 0.0 * rng.standard_normal() * g(4600, 150)  # mesma sequencia do gerador
                X.append(esp_sp + 0.05 * teor * analito + ruido * rng.standard_normal(len(wn))
                         + rng.normal(0, 0.01))
                y.append("unica" if uma_classe else esp)
                grupos.append(f"{esp}_L{lote:02d}")
                conc.append(teor)
    return wn, np.array(X), np.array(y), np.array(conc), np.array(grupos)


def _rodar(tmp_path, nome, rotulo_treino="previsto", uma_classe=False, **extra):
    wn, X, y, conc, g = dados_p2(uma_classe=uma_classe)
    register_reader("quant_roteamento_teste", lambda cfg: (wn, X, y, conc, g, None))
    cfg = pq.Config(mode="quant_roteamento_teste", level="N3", objective="quantificacao",
                    output_root_folder=str(tmp_path / nome), show_plots=False, run_benchmark=False,
                    run_monte_carlo=False, run_shap=False, run_wold=False, run_cv_anova=False, run_opls=False,
                    run_ddsimca=False, executar_etapa4=False, n_permutations=5, max_lvs=8,
                    wn_min=3900.0, wn_max=10100.0, default_preprocessing="mc",
                    quant_rotulo_treino=rotulo_treino, **extra)
    pq.executar(cfg)
    pasta = next((tmp_path / nome).rglob("resumo_modelo.txt"))
    return ler_resumo_canonico(str(pasta)), pasta.parent


def _r2_val(resumo):
    return float(re.search(r"R2val \.+:\s*(-?[\d.]+)", resumo).group(1))


def _r2_holdout(resumo):
    return float(re.search(r"holdout externo\)\s*:.*?R2=(-?[\d.]+)", resumo).group(1))


@pytest.fixture(scope="module")
def execucoes(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("quant_rot")
    return {"previsto": _rodar(raiz, "previsto"), "verdadeiro": _rodar(raiz, "verdadeiro", "verdadeiro"),
            "uma": _rodar(raiz, "uma", uma_classe=True)}


def test_antes_r2_negativo_com_3_classes_e_aviso_depois_r2_alto_com_classe_verdadeira(execucoes):
    prev, pasta = execucoes["previsto"]
    verd, pasta_v = execucoes["verdadeiro"]
    uma, _ = execucoes["uma"]
    ba = float(re.search(r"Balanced accuracy.*?:\s*([\d.]+)", prev, re.I).group(1))
    assert ba < 0.6                                   # classificacao proxima do acaso (3 classes: 0,33)
    r2_prev, r2_verd, r2_uma = _r2_val(prev), _r2_val(verd), _r2_val(uma)
    assert r2_prev < 0.0                              # ANTES: negativo, com 3 classes
    assert r2_uma > 0.95                              # a mesma coisa com 1 classe (o contorno atual)
    assert r2_verd > 0.95                             # DEPOIS (opt-in): volta ao nivel do modelo unico
    assert r2_verd - r2_prev > 1.0
    assert _r2_holdout(verd) > 0.95 > _r2_holdout(prev)    # o holdout externo tambem melhora


def test_aviso_de_roteamento_fraco_no_resumo_no_model_card_e_nos_relatorios(execucoes):
    prev, pasta = execucoes["previsto"]
    avisos = avisos_do_modelo(prev)
    assert [a[0] for a in avisos] == ["roteamento_fraco"]
    _cod, pt, en = avisos[0]
    assert "acuracia balanceada" in pt and "0.80" in pt and "coluna_classe" in pt
    assert "quantificacao_rotulo_treino: verdadeiro" in pt
    assert "balanced accuracy" in en and "constant coluna_classe" in en
    assert "AVISOS DO MODELO: QUANTIFICACAO" in (pasta / "resumo_modelo.txt").read_text(encoding="utf-8")
    assert "Quantificacao com 3 classes" in (pasta / "model_card.md").read_text(encoding="utf-8")
    from io import BytesIO
    import docx
    from guaraci.reports import generate_word_report
    d = docx.Document(BytesIO(generate_word_report(
        str(pasta.parent), {"nome": "t", "autor": "", "inst": "", "tipo": "", "objetivo": ""}).getvalue()))
    assert any("Quantificacao com 3 classes" in p.text for p in d.paragraphs)


def test_modo_verdadeiro_e_registrado_e_o_aviso_muda_de_texto(execucoes):
    verd, pasta = execucoes["verdadeiro"]
    txt = (pasta / "resumo_modelo.txt").read_text(encoding="utf-8")
    assert "Teor treinado com a classe VERDADEIRA" in txt
    assert "ja inclui o roteamento pela classe prevista" in avisos_do_modelo(verd)[0][1]
    assert "pessimistas" not in avisos_do_modelo(verd)[0][1]
    assert "Teor treinado com a classe VERDADEIRA" in (pasta / "model_card.md").read_text(encoding="utf-8")


def test_uma_classe_nao_tem_aviso_de_roteamento(execucoes):
    uma, _ = execucoes["uma"]
    assert avisos_do_modelo(uma) == []


def _dados_funcao():
    wn, X, y, conc, g = dados_p2(n_lotes=10, seed=7)
    return X, conc, y, g


def test_classificacao_perfeita_os_dois_modos_dao_o_mesmo_resultado():
    """Quando o roteamento ja' e' o verdadeiro nada muda: o modo novo reproduz o da v1.0 exatamente."""
    X, conc, y, g = _dados_funcao()
    cfg = pq.Config(default_preprocessing="mc", max_lvs=6, seed=0)
    classes = np.unique(y)
    a = pq.pls_regression_by_species(X, conc, y, g, classes, cfg, "", 4)
    b = pq.pls_regression_by_species(X, conc, y, g, classes, cfg, "", 4, rotulos_treino=y)
    for k in ("r2c", "r2v", "rmsec", "rmsecv", "rmsep", "bias"):
        assert a[k] == pytest.approx(b[k], abs=1e-12), k
    assert a["roteamento"] is None and b["roteamento"]["acuracia_roteamento"] == 1.0


def test_roteamento_errado_so_o_modo_verdadeiro_corrige_o_treino():
    """30% das amostras roteadas para a especie errada: com o rotulo previsto no treino cada modelo
    mistura amostras de outras especies; com o verdadeiro treina limpo e a validacao inclui o erro."""
    X, conc, y, g = _dados_funcao()
    rng = np.random.default_rng(0)
    prev = y.copy()
    # troca a classe de lotes inteiros (as replicas de um lote vao juntas)
    lotes = np.unique(g)
    for lote in rng.choice(lotes, size=int(0.3 * len(lotes)), replace=False):
        m = g == lote
        prev[m] = rng.choice([c for c in np.unique(y) if c != y[m][0]])
    cfg = pq.Config(default_preprocessing="mc", max_lvs=6, seed=0)
    classes = np.unique(y)
    a = pq.pls_regression_by_species(X, conc, prev, g, classes, cfg, "", 4)
    b = pq.pls_regression_by_species(X, conc, prev, g, classes, cfg, "", 4, rotulos_treino=y)
    assert b["roteamento"]["acuracia_roteamento"] < 1.0
    assert b["roteamento"]["n_sem_modelo"] == 0
    assert b["rmsep"] <= a["rmsep"] + 1e-9
    # os modelos finais (os que vao para o pacote) foram treinados com a classe verdadeira
    for cls, info in b["pipelines_especie"].items():
        n_treino_verd = int(np.sum(y == cls))
        assert info["pipeline"].named_steps["pls"].x_scores_.shape[0] <= n_treino_verd


def test_chave_do_config_e_padrao():
    from guaraci.config_io import load_config
    assert pq.Config().quant_rotulo_treino == "previsto"
    with pytest.raises(ValueError):
        pq._quant_treina_com_classe_verdadeira(pq.Config(quant_rotulo_treino="xyz"), "cego")
    assert pq._quant_treina_com_classe_verdadeira(pq.Config(quant_rotulo_treino="verdadeiro"), "cego")
    # no modo controle a classe ja' e' a verdadeira
    assert not pq._quant_treina_com_classe_verdadeira(pq.Config(quant_rotulo_treino="verdadeiro"), "controle")
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "c.yaml"
        p.write_text("quantificacao_rotulo_treino: verdadeiro\n", encoding="utf-8")
        assert load_config(str(p)).quant_rotulo_treino == "verdadeiro"
        p.write_text("quantificacao_rotulo_treino: ambos\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(str(p))
