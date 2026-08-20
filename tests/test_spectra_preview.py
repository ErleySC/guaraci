"""Testes de spectra_preview.py (0% de cobertura, achado da auditoria
metodologica de 2026-08-07 — ver AUDITORIA_METODOLOGICA_2026-08-07.md,
secao "Dívida de engenharia observada"). Usado pelas abas Data/Preprocessing
do app web para a prévia de espectros; funções puras de leitura/parsing (o
único acoplamento a Streamlit é o decorator `st.cache_data`, que funciona
normalmente fora de uma sessão real do Streamlit — só cai para cache em
memória, sem lançar exceção).
"""
import numpy as np
import pandas as pd

from guaraci.spectra_preview import (
    preview_espectros_dx, preview_espectros_csv, plot_espectros_media,
)


def _sqz(v: int) -> str:
    """Codigo ASDF 'squeeze' de um digito (-9..9) — mesmo alfabeto usado em
    test_dados_io_jcamp.py, duplicado aqui p/ manter os testes deste arquivo
    autocontidos (sem import cruzado entre modulos de teste)."""
    if v == 0:
        return "@"
    if 1 <= v <= 9:
        return "ABCDEFGHI"[v - 1]
    if -9 <= v <= -1:
        return "abcdefghi"[-v - 1]
    raise ValueError("fora do alfabeto SQZ simples (-9..9)")


def _escrever_dx(caminho: str, title: str, firstx: float, lastx: float,
                  y_ints) -> None:
    """Grava um .dx minimo, valido, com um digito SQZ por ponto."""
    npoints = len(y_ints)
    xs = np.linspace(firstx, lastx, npoints)
    linhas = [
        "##TITLE=" + title,
        "##XFACTOR=1",
        "##YFACTOR=1",
        f"##FIRSTX={firstx}",
        f"##LASTX={lastx}",
        f"##NPOINTS={npoints}",
        "##XYDATA=(X++(Y..Y))",
    ]
    for x, y in zip(xs, y_ints):
        linhas.append(f"{int(round(x))}{_sqz(int(y))}")
    linhas.append("##END=")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


# ── preview_espectros_dx ─────────────────────────────────────────────────

def test_preview_dx_estrutura_multi_pasta(tmp_path):
    """Pasta com subpastas (uma por classe), cada uma com .dx -- retorna
    wn/specs/labs com uma linha por arquivo, rotulada pelo nome da subpasta."""
    base = tmp_path / "dados"
    for classe, n_arqs in (("Andiroba", 2), ("Copaiba", 3)):
        d = base / classe
        d.mkdir(parents=True)
        for i in range(n_arqs):
            _escrever_dx(str(d / f"amostra_{i}.dx"), f"{classe}-T{i}",
                        firstx=100, lastx=109, y_ints=[1, 2, 3, 4, 5, 6, 7, 8, 9, 1])

    wn, specs, labs = preview_espectros_dx(str(base), wn_min=0, wn_max=1000)
    assert wn is not None
    assert specs.shape == (5, 10)   # 2 + 3 arquivos, 10 pontos
    assert sorted(np.unique(labs).tolist()) == ["Andiroba", "Copaiba"]
    assert list(labs).count("Andiroba") == 2
    assert list(labs).count("Copaiba") == 3


def test_preview_dx_respeita_max_por_classe(tmp_path):
    """max_por_classe limita quantos arquivos de CADA subpasta entram."""
    base = tmp_path / "dados"
    d = base / "Andiroba"
    d.mkdir(parents=True)
    for i in range(5):
        _escrever_dx(str(d / f"a_{i}.dx"), f"T{i}", firstx=100, lastx=104,
                    y_ints=[1, 2, 3, 4, 5])

    _wn, specs, labs = preview_espectros_dx(str(base), wn_min=0, wn_max=1000,
                                            max_por_classe=2)
    assert specs.shape[0] == 2
    assert len(labs) == 2


def test_preview_dx_pasta_sem_dx_retorna_none(tmp_path):
    """Pasta vazia (sem subpastas com .dx, sem .dx na raiz) -- contrato
    documentado (None, None, None), nao excecao."""
    base = tmp_path / "vazia"
    base.mkdir()
    wn, specs, labs = preview_espectros_dx(str(base), wn_min=0, wn_max=1000)
    assert wn is None and specs is None and labs is None


def test_preview_dx_pasta_plana_usa_o_proprio_nome_como_rotulo(tmp_path):
    """Sem subpastas, mas com .dx direto na raiz -- usa a propria pasta como
    'classe' unica (fallback documentado em preview_espectros_dx)."""
    base = tmp_path / "MinhaAmostra"
    base.mkdir()
    _escrever_dx(str(base / "a.dx"), "T1", firstx=100, lastx=104,
                y_ints=[1, 2, 3, 4, 5])
    wn, specs, labs = preview_espectros_dx(str(base), wn_min=0, wn_max=1000)
    assert wn is not None
    assert specs.shape == (1, 5)
    assert labs[0] == "MinhaAmostra"


def test_preview_dx_arquivo_corrompido_e_excluido_sem_derrubar_os_outros(tmp_path):
    """Achado P10 (CLAUDE.md): 1 arquivo ruim numa PREVIA nao pode
    interromper os demais -- e' best-effort por design, nao a analise real."""
    base = tmp_path / "Andiroba"
    base.mkdir()
    _escrever_dx(str(base / "bom.dx"), "T1", firstx=100, lastx=104,
                y_ints=[1, 2, 3, 4, 5])
    with open(base / "corrompido.dx", "w", encoding="utf-8") as f:
        f.write("isso nao e' um JCAMP-DX valido\n")

    wn, specs, labs = preview_espectros_dx(str(base), wn_min=0, wn_max=1000)
    assert wn is not None
    assert specs.shape[0] == 1   # so' o arquivo bom entrou
    assert labs[0] == "Andiroba"


def test_preview_dx_reamostra_arquivo_com_grade_diferente_da_referencia(tmp_path):
    """2o arquivo com FIRSTX/LASTX/NPOINTS diferentes do 1o (grade de
    referencia) -- interpolado via np.interp para a MESMA grade. Achado P10:
    np.interp exige eixo crescente; aqui a funcao ja ordena (_ord) antes de
    interpolar -- este teste trava que a reamostragem da' valores plausiveis
    (nao lixo por eixo desordenado)."""
    base = tmp_path / "Andiroba"
    base.mkdir()
    # referencia: rampa linear 0..9 em x=100..109
    _escrever_dx(str(base / "a_ref.dx"), "T1", firstx=100, lastx=109,
                y_ints=list(range(10)))
    # 2o arquivo: MESMA rampa, mas grade mais fina (20 pontos) -- ao ser
    # reamostrado para a grade de 10 pontos do 1o, deve reconstruir a mesma
    # rampa (a menos de erro de interpolacao pequeno).
    _escrever_dx(str(base / "b_fino.dx"), "T2", firstx=100, lastx=109,
                y_ints=[round(i) for i in np.linspace(0, 9, 20)])

    wn, specs, _labs = preview_espectros_dx(str(base), wn_min=0, wn_max=1000)
    assert specs.shape == (2, 10)
    # a rampa original E a reamostrada devem concordar (mesma funcao linear)
    np.testing.assert_allclose(specs[0], specs[1], atol=1.0)


# ── preview_espectros_csv ────────────────────────────────────────────────

def test_preview_csv_le_colunas_numericas_e_classe(tmp_path):
    caminho = tmp_path / "espectros.csv"
    df = pd.DataFrame({
        "classe": ["A", "A", "B"],
        "100.0": [1.0, 2.0, 3.0],
        "200.0": [4.0, 5.0, 6.0],
        "300.0": [7.0, 8.0, 9.0],
    })
    df.to_csv(caminho, index=False, sep=";")

    wn, X, labs = preview_espectros_csv(str(caminho), col_cls="classe",
                                        wn_min=0, wn_max=1000)
    assert wn is not None
    assert X.shape == (3, 3)
    assert list(labs) == ["A", "A", "B"]


def test_preview_csv_filtra_por_faixa_wn(tmp_path):
    caminho = tmp_path / "espectros.csv"
    df = pd.DataFrame({
        "classe": ["A", "B"],
        "100.0": [1.0, 2.0],
        "200.0": [3.0, 4.0],
        "300.0": [5.0, 6.0],
    })
    df.to_csv(caminho, index=False, sep=";")

    wn, X, _labs = preview_espectros_csv(str(caminho), col_cls="classe",
                                         wn_min=150, wn_max=250)
    assert list(wn) == [200.0]
    assert X.shape == (2, 1)


def test_preview_csv_colunas_nao_numericas_retorna_none(tmp_path):
    """Cabecalho sem numeros de onda (arquivo com formato errado) -- contrato
    documentado (None, None, None), nao ValueError propagado."""
    caminho = tmp_path / "espectros_ruim.csv"
    pd.DataFrame({"classe": ["A"], "nome_amostra": ["x1"]}).to_csv(
        caminho, index=False, sep=";")
    wn, X, labs = preview_espectros_csv(str(caminho), col_cls="classe",
                                        wn_min=0, wn_max=1000)
    assert wn is None and X is None and labs is None


def test_preview_csv_sem_coluna_classe_usa_rotulo_curinga(tmp_path):
    """col_cls nao presente no CSV -- fallback documentado: rotulo "?" para
    todas as linhas, em vez de KeyError."""
    caminho = tmp_path / "sem_classe.csv"
    pd.DataFrame({"100.0": [1.0, 2.0], "200.0": [3.0, 4.0]}).to_csv(
        caminho, index=False, sep=";")
    _wn, _X, labs = preview_espectros_csv(str(caminho), col_cls="classe_ausente",
                                          wn_min=0, wn_max=1000)
    assert list(labs) == ["?", "?"]


# ── plot_espectros_media ─────────────────────────────────────────────────

def test_plot_espectros_media_uma_linha_por_classe():
    rng = np.random.default_rng(0)
    wn = np.linspace(400, 4000, 50)
    X = rng.normal(size=(12, 50))
    rotulos = np.array(["A"] * 4 + ["B"] * 5 + ["C"] * 3)

    fig = plot_espectros_media(wn, X, rotulos, titulo="teste")
    ax = fig.axes[0]
    assert len(ax.lines) == 3   # 1 linha (media) por classe


def test_plot_espectros_media_inverte_eixo_com_wn_decrescente():
    """FT-NIR costuma gravar wavenumber decrescente -- o grafico deve
    inverter o eixo X para exibir na convencao espectroscopica usual."""
    rng = np.random.default_rng(1)
    wn = np.linspace(4000, 400, 30)   # decrescente
    X = rng.normal(size=(6, 30))
    rotulos = np.array(["A"] * 3 + ["B"] * 3)

    fig = plot_espectros_media(wn, X, rotulos)
    ax = fig.axes[0]
    assert ax.xaxis_inverted()
