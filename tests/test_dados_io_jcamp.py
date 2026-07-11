"""Testes de dados_io.py com arquivos JCAMP-DX (.dx) REAIS gravados em disco
(não só parse_title isolado, já coberto em test_pipeline_core.py) — exercita
parse_dx (decodificação ASDF) e carregar_dx (estrutura de pastas, mae_id, CSV
de metadados), que respondiam por boa parte dos 81% não cobertos do módulo.
"""
import numpy as np


def _sqz(v: int) -> str:
    """Codigo ASDF 'squeeze' de um digito (-9..9), mesmo alfabeto de parse_dx."""
    if v == 0:
        return "@"
    if 1 <= v <= 9:
        return "ABCDEFGHI"[v - 1]
    if -9 <= v <= -1:
        return "abcdefghi"[-v - 1]
    raise ValueError("fora do alfabeto SQZ simples (-9..9)")


def _escrever_dx(caminho: str, title: str, firstx: float, lastx: float,
                  y_ints) -> None:
    """Grava um .dx minimo, valido, com um digito SQZ por ponto (sem
    DIF/DUP) — simples de decodificar e de conferir manualmente."""
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


def test_parse_dx_reconstroi_grade_e_valores(pq, tmp_path):
    """parse_dx: FIRSTX/LASTX/NPOINTS reconstroem a grade certa, e os valores
    Y decodificados batem com os inteiros gravados (YFACTOR=1)."""
    y_ints = [1, 2, 3, 4, 5, -1, -2, 0, 3, 9]
    caminho = str(tmp_path / "amostra.dx")
    _escrever_dx(caminho, "AND-04-11-2020-T1", firstx=100, lastx=109,
                 y_ints=y_ints)

    x, y = pq.parse_dx(caminho)
    assert len(x) == len(y_ints)
    np.testing.assert_allclose(x, np.linspace(100, 109, len(y_ints)))
    np.testing.assert_allclose(y, y_ints, atol=1e-9)


def test_extrair_title_do_dx_le_sem_carregar_espectro(pq, tmp_path):
    """extrair_title_do_dx: le só a linha ##TITLE=, sem decodificar os dados."""
    caminho = str(tmp_path / "amostra.dx")
    _escrever_dx(caminho, "CAP-04-11-2020-AD-S-4.13%-T2", firstx=100,
                 lastx=105, y_ints=[1, 2, 3, 4, 5, 6])
    title = pq.extrair_title_do_dx(caminho)
    assert title == "CAP-04-11-2020-AD-S-4.13%-T2"


def test_carregar_dx_estrutura_multi_pasta_com_replicas(pq, tmp_path):
    """carregar_dx: estrutura real (1 subpasta por espécie), TITLE com
    réplicas T1/T2/T3 do mesmo ponto — confirma classe, mae_id compartilhado
    entre réplicas, e teor de adulteração extraído do TITLE."""
    raiz = tmp_path / "dados"
    (raiz / "Andiroba").mkdir(parents=True)
    (raiz / "CastanhaDoPara").mkdir(parents=True)

    y_base = [1, 2, 3, 4, 2, 1, 0, -1, -2, 3]

    # Andiroba: 1 ponto puro com 3 replicas (T1/T2/T3, mesmo mae_id)
    for t in (1, 2, 3):
        _escrever_dx(str(raiz / "Andiroba" / f"and_puro_T{t}.dx"),
                     f"AND-04-11-2020-T{t}", 100, 109, y_base)

    # Castanha do Para: 1 ponto adulterado (teor 4.13%), 2 replicas
    for t in (1, 2):
        _escrever_dx(str(raiz / "CastanhaDoPara" / f"cap_adult_T{t}.dx"),
                     f"CAP-05-11-2020-AD-S-4.13%-T{t}", 100, 109, y_base)

    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.carregar_dx(str(raiz))

    assert X.shape[0] == 5  # 3 replicas Andiroba + 2 replicas Castanha
    assert set(rotulos) == {"Andiroba", "Castanha do Pará"}
    # as 3 replicas de Andiroba compartilham o MESMO mae_id (mesmo ponto fisico)
    mae_andiroba = mae_id[rotulos == "Andiroba"]
    assert len(set(mae_andiroba)) == 1
    # Castanha adulterada: teor 4.13% extraido do TITLE (puros ficam 0.0 por convencao)
    conc_castanha = conc[rotulos == "Castanha do Pará"]
    np.testing.assert_allclose(conc_castanha, 4.13, atol=1e-6)
    conc_andiroba = conc[rotulos == "Andiroba"]
    np.testing.assert_allclose(conc_andiroba, 0.0, atol=1e-6)
    assert meta_df is not None and len(meta_df) == 5


def test_carregar_dados_modo_dx_delega_para_carregar_dx(pq, tmp_path):
    """carregar_dados(cfg) com modo='dx' delega corretamente para carregar_dx
    (mesmo caminho que o pipeline real usa a partir de Config)."""
    raiz = tmp_path / "dados"
    (raiz / "Andiroba").mkdir(parents=True)
    _escrever_dx(str(raiz / "Andiroba" / "and_T1.dx"), "AND-04-11-2020-T1",
                 100, 105, [1, 2, 3, 4, 5, 6])

    cfg = pq.Config(modo="dx", pasta_entrada=str(raiz))
    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.carregar_dados(cfg)
    assert X.shape[0] == 1
    assert rotulos[0] == "Andiroba"
