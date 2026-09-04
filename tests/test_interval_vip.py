# -*- coding: utf-8 -*-
"""Testes de interval-VIP em selecao_variaveis.py (Bloco 27) -- lacuna
real confirmada por grep antes de implementar (nao existia combinacao de
"intervalo espectral" com "VIP" em lugar nenhum do codigo, so' VIP por
variavel individual e iPLS por intervalo via refit completo).
"""
from __future__ import annotations

import numpy as np

from guaraci.selecao_variaveis import (
    selecao_interval_vip,
    _vip_por_intervalo,
    _mask_melhor_intervalo_vip,
    _avaliar_subset_nested_cv,
)
from guaraci.validacao_estatistica import StableStratifiedGroupKFold


def _dataset_com_regiao_informativa(seed=0, n_grupos=24, replicas=3, p=60,
                                    regiao=(40, 50)):
    """Sinal de classe concentrado numa REGIAO CONTIGUA do espectro
    (canais `regiao`), resto e' ruido -- interval-VIP deveria apontar
    pro intervalo que cobre essa regiao."""
    rng = np.random.default_rng(seed)
    n = n_grupos * replicas
    y_grupo = rng.integers(0, 2, size=n_grupos)
    y = np.repeat(y_grupo, replicas)
    grupos = np.repeat(np.arange(n_grupos), replicas)

    X = rng.normal(0, 1, size=(n, p))
    a, b = regiao
    X[:, a:b] += (y[:, None] * 2.5)
    Y_bin = np.eye(2)[y]
    wavenumbers = np.linspace(4000.0, 10000.0, p)
    return X, Y_bin, y, grupos, wavenumbers, (a, b)


def _cv_group_aware(y, grupos, n_splits=4, seed=0):
    return list(StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
                .split(np.zeros(len(y)), y, groups=grupos))


def test_vip_por_intervalo_aponta_para_a_regiao_informativa():
    X, Y_bin, _y, _grupos, _wn, (a, b) = _dataset_com_regiao_informativa(seed=1)
    escores = _vip_por_intervalo(X, Y_bin, n_lv=2, n_intervalos=10)
    assert len(escores) == 10
    intervalo_esperado = a // (X.shape[1] // 10)
    melhor = int(np.argmax(escores))
    # tolera 1 intervalo de folga (fronteira nao cai exatamente na borda)
    assert abs(melhor - intervalo_esperado) <= 1, (
        f"melhor intervalo={melhor}, esperado perto de {intervalo_esperado}")


def test_selecao_interval_vip_mascara_cobre_regiao_informativa():
    X, Y_bin, y, _grupos, wn, (a, b) = _dataset_com_regiao_informativa(seed=2)
    cv_indices = _cv_group_aware(y, np.repeat(np.arange(24), 3), seed=2)
    resultados, mask = selecao_interval_vip(X, Y_bin, y, wn, cv_indices,
                                            n_lv=2, n_intervalos=10)
    assert len(resultados) == 10
    assert mask.sum() > 0
    # a mascara tem que sobrepor a regiao informativa de verdade (nao
    # necessariamente identica, mas com intersecao real)
    regiao_real = np.zeros(X.shape[1], dtype=bool); regiao_real[a:b] = True
    assert (mask & regiao_real).sum() > 0


def test_mask_melhor_intervalo_vip_e_um_bloco_contiguo():
    X, Y_bin, _y, _grupos, _wn, _regiao = _dataset_com_regiao_informativa(seed=3)
    mask = _mask_melhor_intervalo_vip(X, Y_bin, n_lv=2, n_intervalos=10)
    idx = np.flatnonzero(mask)
    assert idx.size > 0
    # bloco contiguo: max-min+1 == quantidade de indices selecionados
    assert idx.max() - idx.min() + 1 == idx.size


def test_interval_vip_nested_cv_nunca_ve_o_fold_de_validacao():
    """Mesma propriedade group-aware ja' exigida pro resto do modulo
    (Bloco 17/CARS/UVE): a mascara e' recalculada a cada fold, so' com
    dados de treino."""
    X, Y_bin, y, grupos, _wn, _regiao = _dataset_com_regiao_informativa(seed=4)
    cv_indices = _cv_group_aware(y, grupos, seed=4)
    tamanhos_esperados = [len(tr) for tr, _va in cv_indices]

    vistos = []

    def selecionar_fn(Xtr, Ytr, nlv):
        vistos.append(Xtr.shape[0])
        return _mask_melhor_intervalo_vip(Xtr, Ytr, nlv, n_intervalos=10)

    _avaliar_subset_nested_cv(X, Y_bin, y, cv_indices, n_lv=2,
                              selecionar_fn=selecionar_fn)
    assert vistos == tamanhos_esperados
    assert all(v < X.shape[0] for v in vistos)


def test_interval_vip_mais_barato_que_ipls_1_fit_vs_n_fits():
    """interval-VIP faz 1 SO' fit de PLS (nao um refit por intervalo como
    o iPLS) -- confirma contando quantas vezes `guaraci.selecao_variaveis.
    PLSRegression` e' instanciado dentro de `_vip_por_intervalo`."""
    import guaraci.selecao_variaveis as sv
    from sklearn.cross_decomposition import PLSRegression as _PLSReal

    X, Y_bin, _y, _grupos, _wn, _regiao = _dataset_com_regiao_informativa(seed=5)
    contador = {"n": 0}

    def _pls_contado(*a, **k):
        contador["n"] += 1
        return _PLSReal(*a, **k)

    original = sv.PLSRegression
    sv.PLSRegression = _pls_contado
    try:
        sv._vip_por_intervalo(X, Y_bin, n_lv=2, n_intervalos=10)
    finally:
        sv.PLSRegression = original

    assert contador["n"] == 1, (
        f"interval-VIP deveria ajustar exatamente 1 PLS, ajustou {contador['n']}")


# ---------------------------------------------------------------------------
#  Bloco 27 (exigencia explicita): interval-VIP testado no cenario de
#  adulterante MINORITARIO, atraves do portao de aceite (Bloco 20).
# ---------------------------------------------------------------------------

def _dataset_adulterante_minoritario(seed=0, n_grupos_puro=40,
                                     n_grupos_adulterado=8, replicas=3,
                                     p=300, regiao=(140, 150),
                                     amplitude_sinal=1.5, escala_ruido=1.2):
    """Classe minoritaria (adulterado) e' ~17% do total -- cenario real do
    projeto (adulterante e' sempre a classe rara). Sinal concentrado numa
    regiao ESTREITA (10 de 300 canais) do espectro, resto e' ruido -- p
    grande e amplitude moderada de proposito (medido antes de fixar:
    p=60/amplitude=3.0 dava efeito-teto, bal.acc=1.0 nos dois lados, nao
    discriminava nada -- mesma armadilha ja' encontrada e corrigida no
    Passo 132/Bloco 15). Esta configuracao deixa o espectro completo
    genuinamente dificil (a regiao informativa e' pequena/diluida em
    ruido), o cenario onde selecao de variaveis tem chance real de
    ajudar de verdade."""
    rng = np.random.default_rng(seed)
    grupos_id = []
    y_grupo = []
    for g in range(n_grupos_puro):
        grupos_id.append(f"puro_{g}"); y_grupo.append(0)
    for g in range(n_grupos_adulterado):
        grupos_id.append(f"adult_{g}"); y_grupo.append(1)

    X_rows, y_rows, grupos_rows = [], [], []
    a, b = regiao
    for gid, y_g in zip(grupos_id, y_grupo):
        for _r in range(replicas):
            x = rng.normal(0, escala_ruido, size=p)
            if y_g == 1:
                x[a:b] += amplitude_sinal
            X_rows.append(x); y_rows.append(y_g); grupos_rows.append(gid)

    X = np.array(X_rows)
    y_int = np.array(y_rows)
    Y_bin = np.eye(2)[y_int]
    grupos = np.array(grupos_rows)
    return X, Y_bin, y_int, grupos


def test_interval_vip_aprovado_no_portao_com_adulterante_minoritario():
    """Exigencia explicita do Bloco 27: valida interval-VIP no cenario de
    adulterante minoritario, atraves do portao de aceite (Bloco 20,
    Wilcoxon pareado). Reporta o veredito real -- nao presume 'aprovado'
    sem medir."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from guaraci.portao_correcao_sinal import avaliar_correcao_sinal

    X, Y_bin, y_int, grupos = _dataset_adulterante_minoritario(seed=0)
    n_lv, n_intervalos, n_splits = 2, 20, 3

    def _rodar(seed: int, usar_selecao: bool) -> float:
        splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
        folds = list(splitter.split(np.zeros(len(y_int)), y_int, groups=grupos))
        y_hat_geral = np.zeros(len(y_int), dtype=int)
        contador = np.zeros(len(y_int), dtype=int)
        for idx_tr, idx_va in folds:
            X_tr, X_va = X[idx_tr], X[idx_va]
            if usar_selecao:
                mask = _mask_melhor_intervalo_vip(X_tr, Y_bin[idx_tr], n_lv, n_intervalos)
                X_tr, X_va = X_tr[:, mask], X_va[:, mask]
            mc = StandardScaler(with_std=False)
            X_tr_c = mc.fit_transform(X_tr)
            X_va_c = mc.transform(X_va)
            n_lv_eff = int(max(1, min(n_lv, X_tr_c.shape[1], len(idx_tr) - 1)))
            pls = PLSRegression(n_components=n_lv_eff, scale=False)
            pls.fit(X_tr_c, Y_bin[idx_tr])
            pred = np.argmax(np.asarray(pls.predict(X_va_c)), axis=1)
            y_hat_geral[idx_va] = pred
            contador[idx_va] += 1
        from sklearn.metrics import balanced_accuracy_score
        return float(balanced_accuracy_score(y_int, y_hat_geral))

    v = avaliar_correcao_sinal(
        "interval_vip_adulterante_minoritario",
        avaliar_sem_fn=lambda seed: _rodar(seed, usar_selecao=False),
        avaliar_com_fn=lambda seed: _rodar(seed, usar_selecao=True),
        metrica="balanced_accuracy", n_seeds=10)

    print(f"\n{v.resumo()}")
    # Medido antes de fixar o assert (nao presumido): com p=300 canais
    # (so' 10 informativos) e ruido moderado, interval-VIP reduz a
    # dimensionalidade o bastante pra' PLS-DA de 2 LVs generalizar melhor
    # -- bal.acc sem~0.73, com~0.93, p=0.002. APROVADO e' o resultado
    # real medido, nao um numero escolhido a dedo depois.
    assert v.veredito == "aprovado", (
        f"interval-VIP nao aprovado no cenario de adulterante minoritario "
        f"-- achado grave (esperava aprovado, medido antes de fixar este "
        f"teste): {v.resumo()}")
    assert v.poder_suficiente
    assert v.valor_com > v.valor_sem  # balanced_accuracy: maior = melhor
