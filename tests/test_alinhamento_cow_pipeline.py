# -*- coding: utf-8 -*-
"""R9 (fase 2): alinhamento de retencao COW como etapa do pipeline (`alinhamento: cow`).

Verdade conhecida: cromatogramas sinteticos cujas classes diferem pela altura de 3 picos marcadores e
em que cada injecao tem um deslocamento de retencao proprio (maior que a largura do pico). Sem
alinhamento o modelo pontual nao encontra os picos nas mesmas colunas; com COW o acerto sobe (na fase
1 do checkup, em GC: 0,25 -> 0,90 com COW + MC). Tambem prova que (a) o COW vetorizado e' identico ao
laco original, (b) a referencia e' escolhida so' no treino (nao vaza), em cada dobra, (c) o modelo
exportado alinha amostras novas como o treinado, (d) o padrao (sem alinhamento) nao muda nada."""
from __future__ import annotations

import re

import numpy as np
import pytest

import guaraci.pipeline as pq
from guaraci.alinhamento_retencao import AlinhadorCOW, alinhar_com_referencia, cow
from guaraci.io_registry import register_reader
from guaraci.resumo_parse import ler_resumo_canonico


def cromatogramas(n_grupos=6, rep=2, n_pts=300, desloc=10.0, seed=0, classes=3):
    """8 picos; as classes diferem pela altura dos picos 2-4; deslocamento por injecao."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_pts, dtype=float)
    centros = np.linspace(0.08, 0.92, 8) * n_pts
    fator = {0: (1.0, 1.0, 1.0), 1: (1.8, 0.5, 1.0), 2: (0.5, 1.8, 1.6)}
    X, y, g = [], [], []
    for c in range(classes):
        for s in range(n_grupos):
            base = rng.uniform(-desloc, desloc)
            amp = 1 + 0.05 * rng.standard_normal(8)
            for _ in range(rep):
                sh = base + rng.normal(0, 1.0)
                alt = amp.copy()
                alt[1:4] *= np.array(fator[c % 3])
                sinal = sum(alt[k] * np.exp(-0.5 * ((t - ck - sh) / 3.0) ** 2) for k, ck in enumerate(centros))
                X.append(sinal + 0.01 * rng.standard_normal(n_pts))
                y.append(f"C{c}")
                g.append(f"C{c}_G{s}")
    return t, np.array(X), np.array(y), np.array(g)


def _cow_em_laco(referencia, amostra, n_segmentos, slack):
    """A implementacao original (duplo laco), copiada de `alinhamento_retencao` da v1.0, como verdade."""
    def corr(a, b):
        if a.std() < 1e-300 or b.std() < 1e-300:
            return 0.0
        r = np.corrcoef(a, b)[0, 1]
        return float(r) if np.isfinite(r) else 0.0

    def interp(seg, n):
        return seg if len(seg) == n else np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(seg)), seg)

    n_ref, n_am = len(referencia), len(amostra)
    nos_ref = np.round(np.linspace(0, n_ref - 1, n_segmentos + 1)).astype(int)
    nom = np.round(np.linspace(0, n_am - 1, n_segmentos + 1)).astype(int)
    jan = [np.array([0])]
    for i in range(1, n_segmentos):
        jan.append(np.arange(max(1, nom[i] - slack), min(n_am - 2, nom[i] + slack) + 1))
    jan.append(np.array([n_am - 1]))
    dp, voltar = [np.zeros(1)], [np.zeros(1, dtype=int)]
    for i in range(1, n_segmentos + 1):
        tam = nos_ref[i] - nos_ref[i - 1] + 1
        seg_ref = referencia[nos_ref[i - 1]:nos_ref[i] + 1]
        atual = np.full(len(jan[i]), -np.inf)
        orig = np.zeros(len(jan[i]), dtype=int)
        for ka, pa in enumerate(jan[i]):
            mv, mk = -np.inf, 0
            for kp, pp in enumerate(jan[i - 1]):
                if pa - pp < 1:
                    continue
                v = dp[i - 1][kp] + corr(interp(amostra[pp:pa + 1], tam), seg_ref)
                if v > mv:
                    mv, mk = v, kp
            atual[ka], orig[ka] = mv, mk
        dp.append(atual)
        voltar.append(orig)
    nos = [0] * (n_segmentos + 1)
    nos[-1] = jan[-1][0]
    k = 0
    for i in range(n_segmentos, 0, -1):
        k = voltar[i][k]
        nos[i - 1] = jan[i - 1][k]
    return np.array(nos)


def test_cow_vetorizado_escolhe_os_mesmos_limites_que_o_laco_original():
    rng = np.random.default_rng(1)
    for n, ns, sl in [(300, 10, 6), (260, 5, 0), (220, 12, 15)]:
        t, X, _y, _g = cromatogramas(n_grupos=1, rep=2, n_pts=n, seed=n)
        ref, am = X[0], X[3]
        esperado = _cow_em_laco(ref, am, ns, sl)
        obtido = cow(ref, am, ns, sl).nos_amostra
        assert np.array_equal(esperado, obtido), (n, ns, sl)
    assert rng is not None


def test_alinhador_sobe_a_correlacao_e_mantem_o_numero_de_pontos():
    _t, X, _y, _g = cromatogramas(n_grupos=4, rep=2)
    al = AlinhadorCOW(n_segmentos=15, folga=12)
    Xa = al.fit_transform(X)
    assert Xa.shape == X.shape
    assert al.correlacao_antes_ < 0.5 and al.correlacao_depois_ > 0.8
    # a referencia passa sem mudanca
    assert np.array_equal(Xa[al.indice_referencia_], X[al.indice_referencia_])
    # transform repete exatamente o mesmo resultado (sem estado aleatorio)
    assert np.array_equal(al.transform(X), Xa)


def test_referencia_escolhida_so_no_treino_e_nao_muda_se_o_teste_muda():
    """Nao vazamento: `fit` ve so' as linhas de treino; trocar/alterar o TESTE nao muda a referencia."""
    _t, X, _y, _g = cromatogramas(n_grupos=6, rep=2)
    tr, te = np.arange(0, 24), np.arange(24, len(X))
    a1 = AlinhadorCOW(n_segmentos=15, folga=12).fit(X[tr])
    a2 = AlinhadorCOW(n_segmentos=15, folga=12).fit(X[tr])
    X_alt = X.copy()
    X_alt[te] = X_alt[te][::-1] * 3.0 + 1.0           # outro teste, qualquer
    a3 = AlinhadorCOW(n_segmentos=15, folga=12).fit(X_alt[tr])
    assert np.array_equal(a1.referencia_, a2.referencia_) and np.array_equal(a1.referencia_, a3.referencia_)
    assert a1.indice_referencia_ < len(tr)             # a referencia e' uma linha do treino
    # e a saida do teste depende so' da referencia do treino: mesmo transform com teste igual
    assert np.array_equal(a1.transform(X[te]), a3.transform(X[te]))
    # contra-prova: mudar o TREINO muda a referencia (a escolha depende dos dados do fit)
    sem_a_referencia = np.delete(X, a1.indice_referencia_, axis=0)
    outra = AlinhadorCOW(n_segmentos=15, folga=12).fit(sem_a_referencia)
    assert not np.array_equal(outra.referencia_, a1.referencia_)


@pytest.fixture(scope="module")
def execucoes(tmp_path_factory):
    """Duas execucoes reais (sem alinhamento e com COW) sobre os mesmos cromatogramas deslocados; o
    espiao registra quantas amostras cada `fit` do alinhador recebeu (uma execucao so', para o teste ser rapido)."""
    raiz = tmp_path_factory.mktemp("cow")
    n_vistos = []
    original = AlinhadorCOW.fit

    def espiao(self, X, y=None):
        n_vistos.append(int(np.asarray(X).shape[0]))
        return original(self, X, y)

    mp = pytest.MonkeyPatch()
    mp.setattr(AlinhadorCOW, "fit", espiao)
    try:
        sem = _rodar("nenhum", raiz)
        com = _rodar("cow", raiz)
    finally:
        mp.undo()
    return {"raiz": raiz, "sem": sem, "com": com, "n_vistos": n_vistos}


def _rodar(alinhamento, raiz, n_grupos=6, **extra):
    t, X, y, g = cromatogramas(n_grupos=n_grupos)
    register_reader("croma_cow_teste", lambda cfg: (t, X, y, None, g, None))
    cfg = pq.Config(mode="croma_cow_teste", level="N1", objective="classificacao",
                    output_root_folder=str(raiz / alinhamento), show_plots=False,
                    run_benchmark=False, run_monte_carlo=False, run_shap=False, run_wold=False,
                    run_cv_anova=False, run_opls=False, run_ddsimca=False, executar_etapa4=False,
                    n_permutations=5, frac_holdout=0.0, seed=0, max_lvs=8, wn_min=float(t[0]),
                    wn_max=float(t[-1]), default_preprocessing="mc", selecao_lv_cv_aninhada=False,
                    alinhamento=alinhamento, alinhamento_segmentos=15, alinhamento_folga=12, **extra)
    pq.executar(cfg)
    pasta = next((raiz / alinhamento).rglob("resumo_modelo.txt"))
    return ler_resumo_canonico(str(pasta)), pasta.parent


def _ba(resumo):
    return float(re.search(r"Balanced accuracy.*?:\s*([\d.]+)", resumo, re.I).group(1))


def test_cada_dobra_do_pipeline_ajusta_o_alinhador_so_com_o_treino_da_dobra(execucoes):
    """Pelo fluxo real (`executar`): toda chamada de `fit` do alinhador recebe menos amostras do que o
    conjunto inteiro, exceto o modelo final (e o refit dele para exportar): nenhuma dobra ve o teste."""
    n_vistos = execucoes["n_vistos"]
    n_total = 3 * 6 * 2
    cheio = [n for n in n_vistos if n == n_total]
    parciais = [n for n in n_vistos if n < n_total]
    assert 1 <= len(cheio) <= 3
    assert len(parciais) >= 5 and max(n_vistos) == n_total
    assert all(n <= n_total * 0.9 for n in parciais)


def test_cow_no_pipeline_sobe_o_acerto_com_deslocamento_de_retencao(execucoes):
    sem, _ = execucoes["sem"]
    com, pasta = execucoes["com"]
    assert _ba(sem) < 0.65, "sem alinhamento o deslocamento maior que o pico deve atrapalhar"
    assert _ba(com) >= 0.85
    assert _ba(com) - _ba(sem) > 0.25
    # registro no resumo canonico, no resumo legivel e no model card
    assert "Alinhamento de retencao" in com
    assert "COW(" in com
    assert "COW" in (pasta / "model_card.md").read_text(encoding="utf-8")
    assert "COW" in (pasta / "resumo_modelo.txt").read_text(encoding="utf-8")
    assert not any("COW" in ln for ln in sem.splitlines())


def test_padrao_sem_alinhamento_nao_adiciona_etapa(tmp_path):
    from guaraci.preprocessamento import build_preprocessor
    cfg = pq.Config()
    assert cfg.alinhamento == "nenhum"
    assert "cow" not in dict(build_preprocessor(cfg).steps)
    cfg.alinhamento = "cow"
    assert [n for n, _ in build_preprocessor(cfg).steps][0] == "cow"
    cfg.apply_despike = True
    assert [n for n, _ in build_preprocessor(cfg).steps][:2] == ["despike", "cow"]


def test_chaves_do_config_yaml_do_alinhamento(tmp_path):
    from guaraci.config_io import load_config
    p = tmp_path / "c.yaml"
    p.write_text("alinhamento: cow\nalinhamento_segmentos: 12\nalinhamento_folga: 7\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert (cfg.alinhamento, cfg.alinhamento_segmentos, cfg.alinhamento_folga) == ("cow", 12, 7)
    p.write_text("alinhamento: ptw\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_modelo_exportado_alinha_amostra_nova_como_o_treinado(execucoes):
    """Pacote .joblib e JSON portatil levam a referencia: uma amostra nova deslocada e' alinhada igual,
    e as duas previsoes concordam com a do pipeline ajustado."""
    import joblib
    from guaraci.model_export import export_portable_json, predict_portable
    pkg = joblib.load(next((execucoes["raiz"] / "cow").rglob("modelo_plsda.joblib")))
    assert "cow" in pkg["preprocessador"].named_steps
    portatil = export_portable_json(pkg)
    assert any(p["tipo"] == "AlinhadorCOW" for p in portatil["passos_preprocessamento"])
    # o arquivo .portatil.json gravado pela execucao tambem leva a referencia
    gravado = next((execucoes["raiz"] / "cow").rglob("modelo_plsda.portatil.json")).read_text(encoding="utf-8")
    assert "AlinhadorCOW" in gravado
    # amostras NOVAS (outra semente), com deslocamento proprio
    t, Xn, yn, _g = cromatogramas(n_grupos=3, seed=99)
    pred = predict_portable(portatil, Xn, t)["classe_pred"]
    Xp = pkg["preprocessador"].transform(Xn)
    esperado = [pkg["label_binarizer"].classes_[i] for i in np.argmax(pkg["pls_final"].predict(Xp), axis=1)]
    assert pred == esperado
    assert np.mean(np.array(pred) == yn) > 0.8
    # sem o passo de alinhamento a mesma amostra nova e' pior classificada (o passo faz diferenca)
    from sklearn.pipeline import Pipeline
    sem_cow = Pipeline(pkg["preprocessador"].steps[1:])
    sem = [pkg["label_binarizer"].classes_[i]
           for i in np.argmax(pkg["pls_final"].predict(sem_cow.transform(Xn)), axis=1)]
    assert np.mean(np.array(sem) == yn) < np.mean(np.array(pred) == yn)


def test_alinhar_com_referencia_e_identidade_para_a_propria_referencia():
    _t, X, _y, _g = cromatogramas(n_grupos=2, rep=1)
    assert np.array_equal(alinhar_com_referencia(X[0], X[:1], 20, 10), X[:1])
