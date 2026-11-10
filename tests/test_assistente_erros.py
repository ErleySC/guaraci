"""Codigos de erro do assistente: cada padrao casa com a mensagem REAL (gerada aqui pelo scikit-learn,
pelo Python ou pelo leitor do Guaraci), nunca com um texto inventado no teste."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci import assistente_erros as E


def _excecao(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return e
    raise AssertionError("nao levantou")


def _sk_nan():
    from sklearn.cross_decomposition import PLSRegression
    X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
    PLSRegression(1).fit(X, [0, 1, 0])


def _sk_grupos():
    from sklearn.model_selection import GroupKFold
    list(GroupKFold(5).split(np.zeros((6, 2)), groups=[0, 0, 1, 1, 2, 2]))


def _sk_componentes():
    from sklearn.cross_decomposition import PLSRegression
    PLSRegression(5).fit(np.random.default_rng(0).normal(size=(4, 3)), [0, 1, 0, 1])


def _sk_zero_variaveis():
    from sklearn.preprocessing import StandardScaler
    StandardScaler().fit(np.zeros((4, 0)))


def _sk_uma_classe():
    from sklearn.svm import SVC
    SVC().fit(np.zeros((4, 2)), [1, 1, 1, 1])


@pytest.mark.parametrize("gerar,codigo", [
    (_sk_nan, "GUA-E001"),
    (lambda: float("0,123"), "GUA-E002"),
    (lambda: b"\xff\xfe\x00a".decode("utf-8"), "GUA-E003"),
    (_sk_grupos, "GUA-E005"),
    (_sk_componentes, "GUA-E006"),
    (_sk_zero_variaveis, "GUA-E007"),
    (lambda: open("/caminho/que/nao/existe.csv"), None),
    (_sk_uma_classe, "GUA-E013"),
])
def test_codigo_casa_com_a_mensagem_real(gerar, codigo):
    e = _excecao(gerar)
    if codigo is None:      # FileNotFoundError cru do Python nao e' do leitor do Guaraci: sem codigo
        assert E.codigos(e) == []
    else:
        assert codigo in E.codigos(e), f"{type(e).__name__}: {e}"


def test_memoria_texto_do_numpy():
    assert E.codigos("numpy.core._exceptions._ArrayMemoryError: Unable to allocate 12.0 GiB for an array") == ["GUA-E004"]
    assert E.codigos(MemoryError()) == ["GUA-E004"]


def test_leitor_csv_do_guaraci(tmp_path):
    from guaraci import dados_io
    e = _excecao(lambda: dados_io.load_csv_com_grupos(str(tmp_path / "nao.csv"), "classe", ""))
    assert "GUA-E008" in E.codigos(e), str(e)

    csv = tmp_path / "d.csv"
    csv.write_text("amostra,1000,1002\na,0.1,0.2\nb,0.3,0.4\n", encoding="utf-8")
    e = _excecao(lambda: dados_io.load_csv_com_grupos(str(csv), "classe", ""))
    assert "GUA-E011" in E.codigos(e), str(e)

    csv.write_text("classe,1000,1002\na,abc,0.2\nb,0.3,0.4\n", encoding="utf-8")   # virgula decimal o leitor ja aceita
    e = _excecao(lambda: dados_io.load_csv_com_grupos(str(csv), "classe", ""))
    assert "GUA-E002" in E.codigos(e), str(e)


def test_leitor_de_pasta_do_guaraci(tmp_path):
    from guaraci import dados_io
    e = _excecao(lambda: dados_io.load_dx(str(tmp_path / "nao_existe")))
    assert "GUA-E008" in E.codigos(e), str(e)
    e = _excecao(lambda: dados_io.load_dx(str(tmp_path)))
    assert "GUA-E008" in E.codigos(e), str(e)


def test_mensagens_internas_existem_no_codigo():
    """Padroes escritos para mensagens do proprio Guaraci: a mensagem tem de existir no codigo-fonte."""
    import inspect
    from guaraci import dados_io, leitores_avancados, pipeline
    fonte = "\n".join(inspect.getsource(m) for m in (dados_io, leitores_avancados, pipeline)).lower()
    for trecho in ("proprietary binary format", "unrecognized binary file", "abb bomem horizon mb",
                   "passaria de 260", "validacao cruzada impossivel", "deixou apenas", "pasta nao existe",
                   "nenhum arquivo candidato", "valores espectrais nao numericos", "menos de 2 colunas numericas de variaveis"):
        assert trecho in fonte, trecho


def test_resumo_de_auditoria_ok_nao_vira_erro():
    # texto 'ok' da auditoria nao pode disparar codigo (antes 'uma unica classe' disparava E013)
    assert E.codigos("Cada grupo tem espectros de uma unica classe.") == []


def test_diagnostico_preserva_mensagem_original():
    e = _excecao(_sk_nan)
    t = E.diagnosticar_excecao(e)
    assert str(e) in t and "GUA-E001" in t and "O que fazer" in t
    t = E.diagnosticar_excecao(RuntimeError("algo novo e estranho"), "EN")
    assert t.startswith("RuntimeError: algo novo e estranho") and "no known code" in t


def test_codigo_digitado_e_explicar():
    assert E.eh_codigo("o que e' gua-e005?") == "GUA-E005"
    assert E.eh_codigo("GUA-E999") is None
    assert E.codigos("GUA-E006") == ["GUA-E006"]
    for c in E.CODIGOS:
        for lang in ("PT", "EN"):
            t = E.explicar(c, lang)
            assert t and t.startswith(c) and len(t.splitlines()) == 3
    assert E.explicar("GUA-E999") is None
    assert len(E.lista("EN")) == len(E.CODIGOS)


def test_catalogo_de_codigos_bem_formado():
    from guaraci.catalogo import texto_proibido
    for c, d in E.CODIGOS.items():
        assert set(d) == {"titulo", "padroes", "causa", "fazer"}, c
        for k in ("titulo", "causa", "fazer"):
            assert len(d[k]) == 2 and all(d[k]), (c, k)
            for t in d[k]:
                assert not texto_proibido(t), (c, k, texto_proibido(t))
