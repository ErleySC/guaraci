"""EEM: tratamento do espalhamento Rayleigh/Raman (1a e 2a ordem), PARAFAC com dados faltantes, aviso de EEM bruta, e a nota do
PARAFAC nao-negativo (HALS) / ALS livre. Verdade conhecida: 2 fluoroforos sinteticos + espalhamento gaussiano forte."""
import json
import os

import numpy as np
import pytest

from guaraci.config import Config
from guaraci.eem_io import (forca_do_espalhamento, larguras_do_config, mascara_espalhamento, raman_agua_emissao,
                            tratar_espalhamento)
from guaraci.formatos_sinteticos import escrever_eem_dat

EX = np.arange(240.0, 401.0, 10.0)
EM = np.arange(250.0, 560.0, 5.0)


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _perfis():
    return [(_g(EX, 260, 15) + 0.4 * _g(EX, 340, 20), _g(EM, 450, 40)), (_g(EX, 280, 10), _g(EM, 340, 20))]


def _eem(c, com_espalhamento, rng):
    E, M = np.meshgrid(EX, EM)
    m = sum(1000.0 * ci * np.outer(pe, px) for ci, (px, pe) in zip(c, _perfis()))
    m = m * (M >= E - 10)
    if com_espalhamento:
        t = rng.uniform(0.5, 2.0)
        m = m + 4000 * t * _g(M, E, 3.0) + 800 * t * _g(M, 2 * E, 4.0)
    return m + rng.normal(0, 3.0, m.shape)


def _pasta(tmp_path, nome="bruta", com_espalhamento=True, n=8, nan_no_arquivo=False):
    rng = np.random.default_rng(3)
    conc = []
    for cl in ("a", "b"):
        for i in range(n):
            c = np.array([1.0, 0.3] if cl == "a" else [0.4, 1.2]) * rng.uniform(0.6, 1.4, 2)
            conc.append(c)
            m = _eem(c, com_espalhamento, rng)
            if nan_no_arquivo:
                E, M = np.meshgrid(EX, EM)
                m = np.where((M < E + 15) | (np.abs(M - 2 * E) < 15), np.nan, m)
            escrever_eem_dat(str(tmp_path / nome / cl / f"{cl}{i}.dat"), EX, EM, m)
    return str(tmp_path / nome), np.array(conc)


def _cfg(pasta, modo="nenhum", **kw):
    c = Config()
    c.mode = "eem"
    c.input_folder = pasta
    c.eem_espalhamento = modo
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _carregar(cfg, manter_constantes=False):
    from guaraci import pipeline as pq
    wn, X, rot, conc, mae, meta = pq.load_data(cfg)
    return pq.validate_input(X, wn, rot, conc, mae, **({"tol_const": -1.0} if manter_constantes else {})), meta


def test_geometria_das_bandas_rayleigh_e_raman():
    assert raman_agua_emissao(350.0) == pytest.approx(397.3, abs=0.1)        # Raman da agua com 3400 cm-1
    bandas, abaixo = mascara_espalhamento([350.0], [300.0, 350.0, 370.0, 397.0, 700.0, 720.0], [10, 10, 10, 10], raman=True)
    assert bandas[:, 0].tolist() == [False, True, False, True, True, False]  # 350 (1a), 397 (Raman), 700 (2a)
    assert abaixo[0, 0] and not abaixo[1, 0]                                 # em < ex: regiao sem sinal (Stokes)
    sem_raman, _ = mascara_espalhamento([350.0], [350.0, 397.0], [10], raman=False)
    assert sem_raman[:, 0].tolist() == [True, False]
    assert larguras_do_config(()) == (30.0,) * 4 and larguras_do_config(("20", "10")) == (20.0, 10.0, 10.0, 10.0)
    with pytest.raises(ValueError):
        larguras_do_config(("-5",))


def test_modos_do_tratamento_e_interpolacao_recupera_superficie_suave():
    E, M = np.meshgrid(EX, EM)
    suave = 100.0 + 0.2 * E + 0.1 * M                       # plano: a interpolacao linear 2D o reproduz
    sujo = suave + 5000 * _g(M, E, 3.0) + 900 * _g(M, 2 * E, 4.0)
    nenhum, rem0, aus0 = tratar_espalhamento(sujo, EX, EM, "nenhum")
    assert np.array_equal(nenhum, sujo) and not rem0.any()
    bandas, abaixo = mascara_espalhamento(EX, EM)
    for modo in ("nan", "zero"):
        m, rem, _ = tratar_espalhamento(sujo, EX, EM, modo)
        assert rem.sum() == (bandas | abaixo).sum() and (m[rem] == 0).all()
    m, rem, _ = tratar_espalhamento(sujo, EX, EM, "interp")
    interior = bandas & ~abaixo & (M > E) & (M < 2 * E - 40) & (M > E + 20) | (bandas & (M > 2 * E - 10) & (M < 2 * E + 10))
    # nas bandas com vizinhos validos dos dois lados a interpolacao fica perto do plano (e longe do dado sujo)
    erro_interp = np.abs(m - suave)[interior & (EM[:, None] < 540)]
    assert np.median(erro_interp) < 10 and np.median(np.abs(sujo - suave)[interior]) > 100
    assert (m[abaixo] == 0).all()
    with pytest.raises(ValueError, match="invalido"):
        tratar_espalhamento(sujo, EX, EM, "xyz")


def test_forca_do_espalhamento_distingue_bruta_de_limpa():
    rng = np.random.default_rng(0)
    c = np.array([1.0, 0.5])
    assert forca_do_espalhamento(_eem(c, True, rng), EX, EM) > 10
    assert forca_do_espalhamento(_eem(c, False, rng), EX, EM) < 3
    assert forca_do_espalhamento(np.zeros((EM.size, EX.size)), EX, EM) == 0.0


def test_leitor_avisa_se_a_eem_bruta_tem_espalhamento_forte_e_nada_foi_aplicado(tmp_path, capsys):
    from guaraci.leitores_avancados import _leitor_eem
    bruta, _ = _pasta(tmp_path, "bruta", True)
    limpa, _ = _pasta(tmp_path, "limpa", False)
    *_, meta = _leitor_eem(_cfg(bruta))
    assert "espalhamento" in meta.attrs["aviso_espalhamento"] and "[CAUTION]" in capsys.readouterr().out
    *_, meta = _leitor_eem(_cfg(limpa))
    assert "aviso_espalhamento" not in meta.attrs and "[CAUTION]" not in capsys.readouterr().out
    *_, meta = _leitor_eem(_cfg(bruta, "nan"))                       # tratada: sem aviso
    assert "aviso_espalhamento" not in meta.attrs


def test_classificador_ve_a_mesma_regiao_no_ajuste_e_na_predicao(tmp_path):
    """Regiao removida = 0 em todas as amostras: colunas constantes, descartadas por validate_input; nenhuma amostra some; o que
    sobra nao depende do espalhamento que estava na regiao removida (ajuste e predicao leem o mesmo vetor)."""
    bruta, _ = _pasta(tmp_path, "bruta", True)
    nanf, _ = _pasta(tmp_path, "nanf", True, nan_no_arquivo=True)
    (X, wn, rot, *_), meta = _carregar(_cfg(bruta, "nan"))
    assert X.shape[0] == 16
    mask = meta.attrs["mascara_eem"]
    assert wn.size == mask.size - mask.sum() and not mask[wn.astype(int)].any()
    assert np.isfinite(X).all()
    (Xn, wnn, *_), metan = _carregar(_cfg(nanf, "nenhum"))            # NaN vindos do arquivo (antes: todas as amostras sumiam)
    assert Xn.shape[0] == 16 and np.isfinite(Xn).all()
    assert metan.attrs["mascara_eem_faltante"].sum() > 0
    # os dois classificam: PLS no vetor mantido separa as classes
    from sklearn.cross_decomposition import PLSRegression
    y = np.array([0.0] * 8 + [1.0] * 8)
    for XX in (X, Xn):
        pred = PLSRegression(2).fit(XX - XX.mean(0), y).predict(XX - XX.mean(0)).ravel() > 0.5
        assert (pred == (y > 0.5)).all()


def test_parafac_com_espalhamento_tratado_recupera_os_perfis_e_sem_tratar_nao(tmp_path):
    pytest.importorskip("tensorly")
    from guaraci.tecnicas_avancadas import rodar_parafac_eem

    bruta, conc = _pasta(tmp_path, "bruta", True)

    def rodar(modo, **kw):
        (X, *_), meta = _carregar(_cfg(bruta, modo), manter_constantes=True)    # como a tela [3.5]
        return rodar_parafac_eem(X, list(meta.attrs["grade_excitacao"]), list(meta.attrs["grade_emissao"]), 2,
                                 mascara_faltantes=meta.attrs["mascara_eem_faltante"], **kw)

    def avaliar(r):
        out = []
        for k, (_px, pe) in enumerate(_perfis()):
            cs = [abs(r.fator_emissao[:, j] @ pe) / np.linalg.norm(r.fator_emissao[:, j]) / np.linalg.norm(pe) for j in range(2)]
            j = int(np.argmax(cs))
            out.append((cs[j], abs(np.corrcoef(r.fator_amostra[:, j], conc[:, k])[0, 1])))
        return out

    for modo in ("nan", "interp"):
        for tucker, corr in avaliar(rodar(modo)):
            assert tucker >= 0.95 and corr >= 0.95, (modo, tucker, corr)
    r = rodar("nan")
    assert r.n_faltantes > 0 and r.n_iter_em >= 1 and r.consistencia_nucleo > 80
    # antes (bruta, sem tratamento): perfis e escores errados
    pior = avaliar(rodar("nenhum"))
    assert min(min(t, c) for t, c in pior) < 0.9


def test_parafac_aceita_nan_no_tensor_e_mascara_de_forma_curta():
    pytest.importorskip("tensorly")
    from guaraci.eem_multiway import parafac_eem
    rng = np.random.default_rng(0)
    A = rng.uniform(0.2, 1.0, (10, 2))
    B = np.stack([_g(EX, 280, 15), _g(EX, 340, 25)], 1)
    C = np.stack([_g(EM, 350, 30), _g(EM, 450, 35)], 1)
    T = np.einsum("ir,jr,kr->ijk", A, B, C)
    mask = np.zeros(T.shape[1:], dtype=bool)
    mask[:, 20:26] = True
    Tn = T.copy()
    Tn[:, mask] = np.nan
    r = parafac_eem(Tn, 2)
    assert r.n_faltantes == 10 * mask.sum() and r.erro_reconstrucao_relativo < 0.01
    r2 = parafac_eem(T, 2, mascara_faltantes=mask)           # mascara (n_ex, n_em) vale para todas as amostras
    assert r2.n_faltantes == r.n_faltantes
    with pytest.raises(ValueError, match="mascara_faltantes"):
        parafac_eem(T, 2, mascara_faltantes=np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="faltantes"):
        parafac_eem(T, 2, mascara_faltantes=np.ones_like(T, dtype=bool))


def _menu(monkeypatch, cfg, is_pt=True):
    from guaraci import guaraci as g
    saida = []
    monkeypatch.setattr(g, "_ask", lambda *a, **k: "2")
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    monkeypatch.setattr(g.console, "print", lambda *a, **k: saida.append(" ".join(str(x) for x in a)))
    g._menu_tec_parafac(cfg, is_pt)
    return " ".join(" ".join(saida).split())      # junta quebras de linha


@pytest.mark.parametrize("nao_neg,restricao,algoritmo,nota_pt,nota_en", [
    (True, "nao-negativo", "HALS", "restricao de nao-negatividade (HALS) desde a v1.0.0", "non-negativity constraint (HALS) since v1.0.0"),
    (False, "nenhuma", "ALS", "PARAFAC sem restricao (ALS livre)", "PARAFAC without constraint (free ALS)")])
def test_parafac_roda_nos_dois_modos_e_a_nota_aparece_na_tela_e_nos_arquivos(
        monkeypatch, tmp_path, nao_neg, restricao, algoritmo, nota_pt, nota_en):
    pytest.importorskip("tensorly")
    bruta, _ = _pasta(tmp_path, "bruta", True)
    cfg = _cfg(bruta, "nan", parafac_nao_negativo=nao_neg, output_root_folder=str(tmp_path / "saida"))
    pt = _menu(monkeypatch, cfg, True)
    assert nota_pt in pt and "CORCONDIA" in pt and restricao in pt
    en = _menu(monkeypatch, cfg, False)
    assert nota_en in en
    pasta = tmp_path / "saida" / "TecnicasEspeciais" / "parafac"
    resumo = json.loads((pasta / "parafac_resumo.json").read_text(encoding="utf-8"))
    assert resumo["restricao"] == restricao and algoritmo in resumo["algoritmo"]
    assert resumo["parafac_nao_negativo"] is nao_neg and nota_pt in resumo["nota_pt"] and nota_en in resumo["nota_en"]
    assert "consistencia_nucleo_percent" in resumo and resumo["espalhamento"]["modo"] == "nan"
    txt = (pasta / "parafac_resumo.txt").read_text(encoding="utf-8")
    assert nota_pt in txt and nota_en in txt
    assert os.path.isfile(pasta / "parafac_fator_amostra.csv")


def test_parafac_alsc_livre_normaliza_sinal_e_nao_negativo_e_o_padrao():
    pytest.importorskip("tensorly")
    from guaraci.eem_multiway import nota_parafac, parafac_eem
    assert Config().parafac_nao_negativo is True and Config().eem_espalhamento == "nenhum"
    rng = np.random.default_rng(1)
    A = rng.uniform(0.2, 1.0, (10, 2))
    B = np.stack([_g(EX, 280, 15), _g(EX, 340, 25)], 1)
    C = np.stack([_g(EM, 350, 30), _g(EM, 450, 35)], 1)
    T = np.einsum("ir,jr,kr->ijk", A, B, C)
    nn, livre = parafac_eem(T, 2), parafac_eem(T, 2, nao_negativo=False)
    assert (nn.restricao, livre.restricao) == ("nao-negativo", "nenhuma") and "HALS" in nn.algoritmo and "ALS" in livre.algoritmo
    assert livre.erro_reconstrucao_relativo < 0.01 and livre.fator_emissao.sum(axis=0).min() > 0
    assert "HALS" in nota_parafac("nao-negativo", "PT") and "HALS" in nota_parafac("nao-negativo", "EN")


def test_config_yaml_aceita_as_chaves_novas_e_recusa_largura_invalida(tmp_path):
    from guaraci.config_io import _validar_semantico, load_config, save_config
    cfg = Config()
    cfg.eem_espalhamento = "interp"
    cfg.eem_espalhamento_largura_nm = ("20", "30")
    cfg.eem_raman = True
    cfg.parafac_nao_negativo = False
    p = str(tmp_path / "c.yaml")
    save_config(cfg, p)
    c2 = load_config(p)
    assert (c2.eem_espalhamento, c2.eem_raman, c2.parafac_nao_negativo) == ("interp", True, False)
    assert tuple(c2.eem_espalhamento_largura_nm) == ("20", "30")
    cfg.eem_espalhamento_largura_nm = ("abc",)
    assert any("eem_espalhamento_largura_nm" in e for e in _validar_semantico(cfg))
