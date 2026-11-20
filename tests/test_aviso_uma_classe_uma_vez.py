# -*- coding: utf-8 -*-
"""O aviso de classe unica sai uma vez por execucao, nao uma vez por dobra de CV."""
from guaraci import pipeline as P


def test_aviso_de_uma_classe_so_uma_vez(monkeypatch):
    avisos = []
    monkeypatch.setattr(P.log, "info", lambda m, *a, **k: avisos.append(m))
    P._AVISO_UMA_CLASSE_JA_DADO[0] = False
    for _ in range(5):
        m = P.classification_metrics(["a", "a", "a"], ["a", "a", "a"], ["a"])
        assert m["degenerada_uma_classe"] == 1.0          # a chave segue em toda chamada
    assert sum("UMA classe" in a for a in avisos) == 1


def test_aviso_rearma_na_proxima_execucao(monkeypatch):
    avisos = []
    monkeypatch.setattr(P.log, "info", lambda m, *a, **k: avisos.append(m))
    P._AVISO_UMA_CLASSE_JA_DADO[0] = False
    P.classification_metrics(["a", "a"], ["a", "a"], ["a"])
    P._AVISO_UMA_CLASSE_JA_DADO[0] = False                # o que _executar_impl faz ao comecar
    P.classification_metrics(["a", "a"], ["a", "a"], ["a"])
    assert sum("UMA classe" in a for a in avisos) == 2


def test_predicoes_fora_da_faixa_pega_a_extrapolacao_e_ignora_o_resto():
    import numpy as np
    ycal = np.array([0.2, 1.0, 2.0, 8.8])
    fora = P.predicoes_fora_da_faixa([1.0, 9.0, 45.8, -0.1], ycal)
    assert list(fora) == [45.8]                       # 9,0 e -0,1 cabem na folga de 50% da amplitude
    assert P.predicoes_fora_da_faixa([5.0], np.array([3.0, 3.0])).size == 0   # sem amplitude, nada a dizer
