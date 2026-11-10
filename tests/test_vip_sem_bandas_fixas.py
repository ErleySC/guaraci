# -*- coding: utf-8 -*-
"""V76: o grafico de VIP nao carrega atribuicoes de banda NIR fixas (duas
estavam erradas: 4325 e 5800 sao C-H, nao O-H nem C=O)."""
import matplotlib
matplotlib.use("Agg")

from guaraci import figuras as fg


def test_grafico_de_vip_nao_anota_bandas_quimicas_fixas():
    assert not hasattr(fg, "_BANDAS_NIR")
    assert not hasattr(fg, "_anotar_bandas_vip")
    fonte = open(fg.__file__, encoding="utf-8").read()
    assert "ac. graxos" not in fonte and "1o overt" not in fonte
