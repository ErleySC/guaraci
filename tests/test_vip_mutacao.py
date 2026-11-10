"""Testes escritos para matar sobreviventes da mutacao de `vip_estabilidade.py`: bordas, contagens e figura."""
import numpy as np
import pytest

from guaraci import vip_estabilidade as ve
from guaraci.vip_estabilidade import figura_vip_estabilidade, vip_estabilidade


def _d(n=24, rep=2, p=6, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n), rep).astype(str)
    X = rng.normal(0, 1, (n, p))[np.repeat(np.arange(n), rep)] + rng.normal(0, 0.05, (n * rep, p))
    y = X[:, 0] + rng.normal(0, 0.3, n * rep)
    return X, y, g


def _vip_nipals(Xc, y, A):
    X = Xc.copy()
    yy = y - y.mean()
    p = X.shape[1]
    W, ss = [], []
    for _ in range(A):
        w = X.T @ yy
        w /= np.linalg.norm(w)
        t = X @ w
        tt = t @ t
        pl = X.T @ t / tt
        q = (yy @ t) / tt
        W.append(w)
        ss.append(q * q * tt)
        X = X - np.outer(t, pl)
        yy = yy - t * q
    return np.array([np.sqrt(p * sum(ss[k] * W[k][j] ** 2 for k in range(A)) / sum(ss)) for j in range(p)])


def _controlado(monkeypatch, linhas):
    """Faz `vip_scores` devolver, em sequencia, as linhas dadas (a primeira e' o ajuste com todos os dados)."""
    it = iter(linhas)
    monkeypatch.setattr(ve, "vip_scores", lambda modelo: np.asarray(next(it), dtype=float))


# ------------------------------------------------------------------ valores

def test_sem_autoescala_o_vip_e_o_do_pls1_sobre_x_apenas_centrado():
    X, y, g = _d()
    r = vip_estabilidade(X, y, 2, g, tarefa="regressao", n_repeticoes=1, autoescalar=False)
    np.testing.assert_allclose(r["vip"], _vip_nipals(X - X.mean(0), y, 2), rtol=1e-8)
    r2 = vip_estabilidade(X, y, 2, g, tarefa="regressao", n_repeticoes=1)                    # com autoescala
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    np.testing.assert_allclose(r2["vip"], _vip_nipals(Xs, y, 2), rtol=1e-8)


def test_valores_padrao_n_repeticoes_10_seed_0_limiar_1():
    X, y, g = _d()
    padrao = vip_estabilidade(X, y, 1, g, tarefa="regressao")
    explicito = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_repeticoes=10, seed=0, limiar=1.0, n_divisoes=5)
    for k in ("vip", "media", "dp", "frequencia", "n_ajustes"):
        np.testing.assert_array_equal(padrao[k], explicito[k])
    assert padrao["n_ajustes"] == 50 and padrao["limiar"] == 1.0


def test_limiar_muda_a_frequencia():
    X, y, g = _d()
    alto = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_repeticoes=2, limiar=100.0)
    baixo = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_repeticoes=2, limiar=0.0)
    assert np.all(alto["frequencia"] == 0.0) and np.all(baixo["frequencia"] == 1.0)


def test_estatisticas_das_dobras_sao_media_dp_amostral_e_percentis_2_5_e_97_5(monkeypatch):
    rng = np.random.default_rng(5)
    linhas = rng.uniform(0.2, 1.8, (1 + 4, 6))                 # n_divisoes=2, n_repeticoes=2 -> 4 ajustes de dobra
    _controlado(monkeypatch, linhas)
    X, y, g = _d()
    r = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=2, n_repeticoes=2)
    arr = linhas[1:]
    np.testing.assert_allclose(r["vip"], linhas[0])
    np.testing.assert_allclose(r["media"], arr.mean(0))
    np.testing.assert_allclose(r["dp"], arr.std(0, ddof=1))
    np.testing.assert_allclose(r["ic95_inf"], np.percentile(arr, 2.5, axis=0))
    np.testing.assert_allclose(r["ic95_sup"], np.percentile(arr, 97.5, axis=0))
    np.testing.assert_allclose(r["frequencia"], (arr > 1.0).mean(0))


def test_classes_de_estabilidade_nas_bordas_de_80_e_20_por_cento(monkeypatch):
    # 10 ajustes de dobra; variavel j passa de 1 em `acima[j]` deles
    acima = [8, 2, 5, 9, 1, 10, 0, 7]
    p = len(acima)
    linhas = [np.ones(p)]
    for k in range(10):
        linhas.append([2.0 if k < a else 0.1 for a in acima])
    _controlado(monkeypatch, linhas)
    X, y, g = _d(p=p)
    r = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=5, n_repeticoes=2)
    assert list(r["frequencia"]) == [0.8, 0.2, 0.5, 0.9, 0.1, 1.0, 0.0, 0.7]
    assert list(r["classe"]) == ["estavel_acima", "estavel_abaixo", "instavel", "estavel_acima",
                                 "estavel_abaixo", "estavel_acima", "estavel_abaixo", "instavel"]


def test_contagem_exata_de_ajustes_que_falharam(monkeypatch):
    original = ve.vip_scores
    estado = {"n": 0}

    def falha_em_tres(modelo):
        estado["n"] += 1
        if estado["n"] in (3, 5, 8):
            raise ValueError("falha simulada")
        return original(modelo)

    monkeypatch.setattr(ve, "vip_scores", falha_em_tres)
    X, y, g = _d()
    r = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=5, n_repeticoes=2)
    assert r["n_falhos"] == 3 and r["n_ajustes"] == 10 - 3
    estado["n"] = 0

    def nao_finito(modelo):
        estado["n"] += 1
        v = original(modelo)
        return v * np.nan if estado["n"] in (2, 4) else v

    monkeypatch.setattr(ve, "vip_scores", nao_finito)
    r2 = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=5, n_repeticoes=2)
    assert r2["n_falhos"] == 2 and r2["n_ajustes"] == 8 and any("2 ajuste(s)" in a for a in r2["avisos"])


# ------------------------------------------------------------------ bordas das entradas

def test_borda_de_linhas_componentes_divisoes_e_grupos():
    X, y, g = _d(n=4, rep=1, p=4)
    r = vip_estabilidade(X, y, 3, g, tarefa="regressao", n_divisoes=2, n_repeticoes=1)          # 4 linhas, n_comp = min(n-1, p) = 3
    assert r["n_variaveis"] == 4 and r["n_divisoes"] == 2
    with pytest.raises(ValueError, match="n_componentes"):
        vip_estabilidade(X, y, 4, g, tarefa="regressao")                                          # min(n-1, p) + 1
    with pytest.raises(ValueError, match="n_componentes"):
        vip_estabilidade(X, y, 0, g, tarefa="regressao")
    with pytest.raises(ValueError, match="X precisa ser"):
        vip_estabilidade(X.ravel(), y, 1, g, tarefa="regressao")
    X6, y6, g6 = _d(n=6, rep=1, p=4)
    assert vip_estabilidade(X6, y6, 1, g6, tarefa="regressao", n_divisoes=2, n_repeticoes=1)["n_ajustes"] == 2
    with pytest.raises(ValueError, match="n_divisoes"):
        vip_estabilidade(X6, y6, 1, g6, tarefa="regressao", n_divisoes=1)
    with pytest.raises(ValueError, match="n_repeticoes"):
        vip_estabilidade(X6, y6, 1, g6, tarefa="regressao", n_repeticoes=0)


def test_dois_grupos_bastam_e_grupos_igual_a_divisoes_nao_gera_aviso_de_reducao():
    X, y, g = _d(n=2, rep=6, p=4)
    r = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=2, n_repeticoes=1)
    assert r["n_grupos"] == 2 and not any("reduzido" in a for a in r["avisos"])
    X5, y5, g5 = _d(n=5, rep=3, p=4)
    r5 = vip_estabilidade(X5, y5, 1, g5, tarefa="regressao", n_divisoes=5, n_repeticoes=1)
    assert not any("reduzido" in a for a in r5["avisos"])


def test_nan_em_x_e_recusado_pela_propria_funcao_e_nao_pelo_sklearn():
    X, y, g = _d()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="VIP recusado: X tem NaN"):
        vip_estabilidade(X, y, 1, g, tarefa="regressao")


# ------------------------------------------------------------------ figura (inspecao dos artistas)

@pytest.fixture
def capturada(monkeypatch):
    from matplotlib.figure import Figure
    guardada = {}
    original = Figure.savefig
    monkeypatch.setitem(__import__('matplotlib').rcParams, 'savefig.bbox', 'standard')   # estado global de outros testes nao entra na medida

    def capturar(self, *a, **k):
        guardada["fig"] = self
        return original(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", capturar)
    return guardada


def _res(p, monkeypatch=None):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, p))
    y = X[:, 0] + rng.normal(0, 0.3, 40)
    return vip_estabilidade(X, y, 2, np.arange(40).astype(str), tarefa="regressao", n_repeticoes=1)


def _dim(caminho):
    b = open(caminho, "rb").read()
    return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")


@pytest.mark.parametrize("p, largura_pol", [(12, 7.0), (30, 11.4), (40, 14.0), (60, 14.0), (61, 10.0), (200, 10.0)])
def test_tamanho_da_figura_150_dpi_depende_do_numero_de_variaveis(tmp_path, capturada, p, largura_pol):
    caminho = figura_vip_estabilidade(_res(p), str(tmp_path / "f.png"))
    assert _dim(caminho) == (round(largura_pol * 150), round(4.2 * 150))


def test_barras_ate_60_tem_erros_e_rotulos_e_acima_de_60_nao(tmp_path, capturada):
    for p, com in ((60, True), (61, False)):
        res = _res(p)
        figura_vip_estabilidade(res, str(tmp_path / f"{p}.png"))
        ax = capturada["fig"].axes[0]
        assert len(capturada["fig"].axes) == 1 and ax.get_subplotspec().get_geometry()[:2] == (1, 1)
        barras = list(ax.patches)
        assert len(barras) == p and all(b.get_width() == (0.8 if com else 1.0) for b in barras)
        assert bool(ax.containers[1:]) is com                                     # ErrorbarContainer so' ate 60
        rot = ax.get_xticklabels()[0].get_rotation()
        assert (rot == 60 and len(ax.get_xticklabels()) == p) if com else rot != 60


def test_regressao_figura_com_vip_identico_nas_dobras_nao_quebra_por_erro_de_ponto_flutuante(tmp_path):
    """Achado da mutacao: media == percentil na teoria, mas em ponto flutuante a diferenca pode ser -1e-17 e o
    matplotlib recusa `yerr` negativo (a figura de `guaraci vip` quebraria para variavel com VIP identico nas dobras)."""
    res = {"n_variaveis": 3, "media": np.array([0.1, 0.30000000000000004, 1.0]),
           "ic95_inf": np.array([0.1, 0.3000000000000001, 1.0000000000000002]),
           "ic95_sup": np.array([0.1, 0.29999999999999993, 0.9999999999999999]),
           "classe": np.array(["estavel_abaixo", "instavel", "estavel_acima"]), "limiar": 1.0, "n_ajustes": 10}
    assert b"PNG" in open(figura_vip_estabilidade(res, str(tmp_path / "e.png")), "rb").read(8)


def test_cores_por_classe_rotulos_e_erros_das_barras(tmp_path, capturada, monkeypatch):
    linhas = [np.ones(3)] + [[2.0, 0.1, 1.5]] * 5 + [[2.0, 0.1, 0.1]] * 5
    _controlado(monkeypatch, linhas)
    X, y, g = _d(p=3)
    res = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=5, n_repeticoes=2)
    assert list(res["classe"]) == ["estavel_acima", "estavel_abaixo", "instavel"]
    figura_vip_estabilidade(res, str(tmp_path / "c.png"), nomes=["a", "b", "c"])
    ax = capturada["fig"].axes[0]
    from matplotlib.colors import to_hex
    assert [to_hex(b.get_facecolor()).upper() for b in ax.patches] == ["#4C72B0", "#BBBBBB", "#DD8452"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    erros = next(c for c in ax.containers if hasattr(c, "lines") and c.lines[2])
    segmentos = erros.lines[2][0].get_segments()
    for s, lo, hi in zip(segmentos, res["ic95_inf"], res["ic95_sup"]):
        assert s[0][1] == pytest.approx(lo) and s[1][1] == pytest.approx(hi)
    # sem nomes: rotulos numericos
    figura_vip_estabilidade(res, str(tmp_path / "d.png"))
    assert [t.get_text() for t in capturada["fig"].axes[0].get_xticklabels()] == ["0", "1", "2"]
    # linha do limiar
    ys = [float(ln.get_ydata()[0]) for ln in ax.lines if len(set(np.asarray(ln.get_ydata()))) == 1]
    assert 1.0 in ys


def test_dois_grupos_com_cinco_divisoes_reduz_para_dois_e_roda():
    X, y, g = _d(n=2, rep=6, p=4)
    r = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=5, n_repeticoes=1)
    assert r["n_divisoes"] == 2 and r["n_ajustes"] == 2
    assert any("reduzido de 5 para 2" in a for a in r["avisos"])


def test_vip_exatamente_no_limiar_nao_conta_como_acima(monkeypatch):
    linhas = [np.ones(3)] + [[1.0, 1.0000001, 0.9999999]] * 4
    _controlado(monkeypatch, linhas)
    X, y, g = _d(p=3)
    r = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=2, n_repeticoes=2)
    assert list(r["frequencia"]) == [0.0, 1.0, 0.0]


def test_dp_com_dois_ajustes_e_com_um_so(monkeypatch):
    rng = np.random.default_rng(2)
    linhas = rng.uniform(0.2, 1.8, (3, 5))                 # completo + 2 dobras
    _controlado(monkeypatch, linhas)
    X, y, g = _d(p=5)
    r = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=2, n_repeticoes=1)
    np.testing.assert_allclose(r["dp"], linhas[1:].std(0, ddof=1))
    original_linhas = iter([np.ones(5), None, np.ones(5) * 1.5])

    def uma_falha(modelo):
        v = next(original_linhas)
        if v is None:
            raise ValueError("falha simulada")
        return v

    monkeypatch.setattr(ve, "vip_scores", uma_falha)
    r1 = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=2, n_repeticoes=1)
    assert r1["n_ajustes"] == 1 and r1["n_falhos"] == 1
    assert np.all(r1["dp"] == 0.0) and np.all(np.isfinite(r1["dp"]))
