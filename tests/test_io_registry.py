"""Testes de guaraci.io_registry (item 20: registry de leitores de dados).

Verifica o registry em si (registrar/obter/listar) e que dados_io.py
registrou corretamente os 4 leitores built-in (dx, csv, imagem, sintetico),
sem regressão no comportamento de load_data().
"""
import pytest

from guaraci.io_registry import register_reader, get_reader, registered_modes


def test_leitores_builtin_registrados():
    modos = registered_modes()
    assert set(modos) == {"csv", "dx", "imagem", "sintetico"}


def test_obter_leitor_modo_desconhecido_da_erro_com_lista():
    with pytest.raises(ValueError, match="Modo de entrada desconhecido"):
        get_reader("formato_que_nao_existe")


def test_obter_leitor_retorna_callable_para_cada_modo():
    for mode in registered_modes():
        leitor = get_reader(mode)
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

    register_reader("meu_formato_de_teste", _leitor_customizado)
    try:
        assert "meu_formato_de_teste" in registered_modes()
        leitor = get_reader("meu_formato_de_teste")
        resultado = leitor(cfg="config_falso")
        assert len(resultado) == 6
        assert chamadas == ["config_falso"]
    finally:
        # limpeza: nao deixar o mode de teste vazando para outros testes
        from guaraci.io_registry import _LEITORES
        _LEITORES.pop("meu_formato_de_teste", None)


def test_carregar_dados_sintetico_usa_o_registry(pq):
    """load_data(cfg) com mode='sintetico' deve funcionar via registry,
    devolvendo a tupla de 6 elementos com os tipos esperados."""
    cfg = pq.Config(mode="sintetico")
    wn, X, rot, conc, mae, meta = pq.load_data(cfg)
    assert wn.ndim == 1
    assert X.ndim == 2
    assert len(rot) == X.shape[0]
