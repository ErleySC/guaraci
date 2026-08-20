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
    _escrever_dx(caminho, "AND-04-11-2099-T1", firstx=100, lastx=109,
                 y_ints=y_ints)

    x, y = pq.parse_dx(caminho)
    assert len(x) == len(y_ints)
    np.testing.assert_allclose(x, np.linspace(100, 109, len(y_ints)))
    np.testing.assert_allclose(y, y_ints, atol=1e-9)


def test_extrair_title_do_dx_le_sem_carregar_espectro(pq, tmp_path):
    """extrair_title_do_dx: le só a linha ##TITLE=, sem decodificar os dados."""
    caminho = str(tmp_path / "amostra.dx")
    _escrever_dx(caminho, "CAP-04-11-2099-AD-S-4.13%-T2", firstx=100,
                 lastx=105, y_ints=[1, 2, 3, 4, 5, 6])
    title = pq.extrair_title_do_dx(caminho)
    assert title == "CAP-04-11-2099-AD-S-4.13%-T2"


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
                     f"AND-04-11-2099-T{t}", 100, 109, y_base)

    # Castanha do Para: 1 ponto adulterado (teor 4.13%), 2 replicas
    for t in (1, 2):
        _escrever_dx(str(raiz / "CastanhaDoPara" / f"cap_adult_T{t}.dx"),
                     f"CAP-05-11-2099-AD-S-4.13%-T{t}", 100, 109, y_base)

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
    _escrever_dx(str(raiz / "Andiroba" / "and_T1.dx"), "AND-04-11-2099-T1",
                 100, 105, [1, 2, 3, 4, 5, 6])

    cfg = pq.Config(modo="dx", pasta_entrada=str(raiz))
    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.carregar_dados(cfg)
    assert X.shape[0] == 1
    assert rotulos[0] == "Andiroba"


# ── prescan_dx ───────────────────────────────────────────────────────────────
# Varredura barata de cabecalho que ANTECIPA, no checklist, os dois efeitos que
# mudam o N da analise e antes so' apareciam no meio do log da execucao:
# descarte por faixa espectral incompativel e amostras sem mae_id.

def _montar_dataset(base, especie_a=("CAP", 6), especie_b=("BAB", 3),
                     fora_da_faixa=0, sem_titulo_valido=0):
    """Monta uma arvore <base>/<especie>/*.dx como o layout real do projeto."""
    import os
    y = [1, 2, 3, 4, 5]
    for cod, n in (especie_a, especie_b):
        pasta = os.path.join(str(base), cod)
        os.makedirs(pasta, exist_ok=True)
        for i in range(n):
            # 2 replicas por ponto de coleta -> mae_id compartilhado.
            title = f"{cod}-0{i // 2 + 1}-11-2099_T{i % 2 + 1}"
            _escrever_dx(os.path.join(pasta, f"{title}.dx"), title, 0.0, 4000.0, y)
    pasta_a = os.path.join(str(base), especie_a[0])
    for i in range(fora_da_faixa):
        # LASTX bem diferente -> cai fora da faixa dominante.
        title = f"{especie_a[0]}-09-09-2099_T{i + 1}"
        _escrever_dx(os.path.join(pasta_a, f"forafaixa{i}.dx"), title,
                     0.0, 15797.0, y)
    for i in range(sem_titulo_valido):
        # Titulo que parse_title rejeita -> amostra sem mae_id.
        _escrever_dx(os.path.join(pasta_a, f"semtitulo{i}.dx"),
                     "TITULO_INVALIDO_SEM_PADRAO", 0.0, 4000.0, y)
    return str(base)


def test_prescan_conta_arquivos_e_grupos(pq, tmp_path):
    """Sem anomalias: conta tudo, nenhum descarte, nenhuma orfa."""
    pasta = _montar_dataset(tmp_path / "ds1")
    r = pq.prescan_dx(pasta)
    assert r["n_arquivos"] == 9          # 6 + 3
    assert r["n_apos_descarte"] == 9
    assert r["n_fora_da_faixa"] == 0
    assert r["n_sem_mae_id"] == 0
    assert r["n_grupos"] == 5            # CAP: 3 pontos, BAB: 2 pontos
    assert r["n_sem_cabecalho"] == 0


def test_prescan_preve_o_descarte_por_faixa_espectral(pq, tmp_path):
    """O numero previsto tem de ser o MESMO que carregar_dx vai descartar --
    e' o que torna o aviso confiavel em vez de so' indicativo."""
    pasta = _montar_dataset(tmp_path / "ds2", fora_da_faixa=2)
    r = pq.prescan_dx(pasta)
    assert r["n_arquivos"] == 11
    assert r["n_fora_da_faixa"] == 2
    assert r["n_apos_descarte"] == 9
    assert r["fora_por_especie"] == {"CAP": 2}, \
        "o descarte tem de ser atribuido a especie (pasta) certa"

    # A previsao tem de bater com a carga REAL do mesmo dataset.
    _wn, X, _rot, _conc, _mae, _meta = pq.carregar_dx(pasta, extrair_conc=False)
    assert X.shape[0] == r["n_apos_descarte"]


def test_prescan_conta_orfas_apos_o_descarte(pq, tmp_path):
    """Amostra sem mae_id entra na analise SEM protecao anti-leakage, entao o
    numero precisa aparecer. Conta so' entre as SOBREVIVENTES: contar sobre o
    total daria um valor sem correspondencia com o que a analise vera'."""
    pasta = _montar_dataset(tmp_path / "ds3", sem_titulo_valido=2)
    r = pq.prescan_dx(pasta)
    assert r["n_sem_mae_id"] == 2
    # Cada orfa vira um grupo de 1 em carregar_dx -> 5 reais + 2 orfas.
    assert r["n_grupos_reais"] == 5
    assert r["n_grupos"] == 7


def test_prescan_orfa_descartada_nao_e_contada(pq, tmp_path):
    """Uma amostra que ja' sai pelo descarte de faixa NAO deve aparecer como
    orfa -- ela nem chega na analise. (No dataset real do TCC e' exatamente o
    caso da Graviola: titulo fora do padrao E faixa incompativel.)"""
    import os
    pasta = _montar_dataset(tmp_path / "ds4")
    # Orfa E fora da faixa ao mesmo tempo.
    _escrever_dx(os.path.join(pasta, "CAP", "orfa_e_fora.dx"),
                 "TITULO_INVALIDO", 0.0, 15797.0, [1, 2, 3, 4, 5])
    r = pq.prescan_dx(pasta)
    assert r["n_fora_da_faixa"] == 1
    assert r["n_sem_mae_id"] == 0, "orfa descartada nao pode contar como orfa"


def test_prescan_pasta_vazia_nao_quebra(pq, tmp_path):
    import os
    vazia = tmp_path / "vazia"
    os.makedirs(vazia, exist_ok=True)
    r = pq.prescan_dx(str(vazia))
    assert r["n_arquivos"] == 0
    assert r["faixa_dominante"] is None
    assert r["n_grupos"] == 0


# ---------------------------------------------------------------------------
# Eixo espectral DECRESCENTE (achado 2026-08-07)
# ---------------------------------------------------------------------------
def test_predicao_interpola_espectro_com_eixo_decrescente():
    """REGRESSAO: `np.interp` exige eixo crescente e NAO ordena sozinho.

    Um .dx de terceiro gravado em ordem decrescente (convencao comum em
    FTIR) fazia a reamostragem devolver valores errados sem lancar erro --
    ou seja, PREDICAO errada em silencio. O equipamento do autor (ABB
    MB3600) grava crescente, entao o defeito era latente no dataset local,
    mas real para qualquer outro instrumento.

    Interpolar o MESMO espectro nas duas ordens tem que dar o mesmo
    resultado.
    """
    import numpy as np
    wn_ref = np.linspace(4000, 6000, 50)
    wn_cresc = np.linspace(3900, 6100, 200)
    espectro = np.exp(-((wn_cresc - 5000) / 300.0) ** 2)

    def reamostrar(wn, y):
        ordem = np.argsort(wn)
        return np.interp(wn_ref, wn[ordem], y[ordem])

    ref = reamostrar(wn_cresc, espectro)
    inv = reamostrar(wn_cresc[::-1], espectro[::-1])
    np.testing.assert_allclose(ref, inv, rtol=1e-12)

    # E o caminho ERRADO (sem ordenar) de fato difere -- prova que o teste
    # nao e' vacuo e que havia um defeito real a corrigir.
    errado = np.interp(wn_ref, wn_cresc[::-1], espectro[::-1])
    assert not np.allclose(errado, ref), (
        "premissa do teste: sem ordenar, np.interp devolveria outro resultado")
