"""Evidência reproduzível de itens do parecer de 2026-09-25 que estavam 'corrigidos' sem teste citado."""
import numpy as np
import pytest

from guaraci import dados_io


def _espectros(n_amostras: int, replicas: int = 3, p: int = 12):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_amostras * replicas, p))
    grupos = np.repeat([f"G{i}" for i in range(n_amostras)], replicas)
    return X, grupos


@pytest.mark.parametrize("funcao", ["kennard_stone_split_group_aware", "duplex_split_group_aware"])
def test_m03_menos_de_4_grupos_avisa_que_o_split_cai_em_nivel_de_amostra(funcao, capsys):
    X, grupos = _espectros(3)
    getattr(dados_io, funcao)(X, grupos, 0.7)
    saida = capsys.readouterr().out
    assert "[AVISO]" in saida and "so' 3 grupo(s)" in saida and "vazamento" in saida


@pytest.mark.parametrize("funcao", ["kennard_stone_split_group_aware", "duplex_split_group_aware"])
def test_m03_com_4_grupos_ou_mais_nao_avisa(funcao, capsys):
    X, grupos = _espectros(6)
    getattr(dados_io, funcao)(X, grupos, 0.7)
    assert "[AVISO]" not in capsys.readouterr().out


def test_m03_split_com_grupos_nunca_separa_replicas_da_mesma_amostra():
    X, grupos = _espectros(8)
    cal, val = dados_io.kennard_stone_split_group_aware(X, grupos, 0.7)
    assert not (set(grupos[cal]) & set(grupos[val]))
