"""Testes de guaraci.io_registry (item 20: registry de leitores de dados).

Verifica o registry em si (registrar/obter/listar) e que dados_io.py
registrou corretamente os 4 leitores built-in (dx, csv, imagem, sintetico),
sem regressão no comportamento de carregar_dados().
"""
import pytest

from guaraci.io_registry import registrar_leitor, obter_leitor, modos_registrados


def test_leitores_builtin_registrados():
    modos = modos_registrados()
    assert set(modos) == {"csv", "dx", "imagem", "sintetico"}


def test_obter_leitor_modo_desconhecido_da_erro_com_lista():
    with pytest.raises(ValueError, match="Modo de entrada desconhecido"):
        obter_leitor("formato_que_nao_existe")


def test_obter_leitor_retorna_callable_para_cada_modo():
    for modo in modos_registrados():
        leitor = obter_leitor(modo)
        assert callable(leitor)


def test_registrar_novo_leitor_fica_disponivel():
    """Simula extensao externa: registra um formato novo sem tocar em
    dados_io.py, exatamente o caso de uso que o registry existe para servir."""
    chamadas = []

    def _leitor_customizado(cfg):
        chamadas.append(cfg)
        import numpy as np
        wn = np.array([1.0, 2.0, 3.0])
        X = np.zeros((2, 3))
        rot = np.array(["a", "b"])
        return wn, X, rot, None, None, None

    registrar_leitor("meu_formato_de_teste", _leitor_customizado)
    try:
        assert "meu_formato_de_teste" in modos_registrados()
        leitor = obter_leitor("meu_formato_de_teste")
        resultado = leitor(cfg="config_falso")
        assert len(resultado) == 6
        assert chamadas == ["config_falso"]
    finally:
        # limpeza: nao deixar o modo de teste vazando para outros testes
        from guaraci.io_registry import _LEITORES
        _LEITORES.pop("meu_formato_de_teste", None)


def test_carregar_dados_sintetico_usa_o_registry(pq):
    """carregar_dados(cfg) com modo='sintetico' deve funcionar via registry,
    devolvendo a tupla de 6 elementos com os tipos esperados."""
    cfg = pq.Config(modo="sintetico")
    wn, X, rot, conc, mae, meta = pq.carregar_dados(cfg)
    assert wn.ndim == 1
    assert X.ndim == 2
    assert len(rot) == X.shape[0]
