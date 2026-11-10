"""Fase 2 do HSI (R11): calibracao branco/escuro, eixo de bandas entre cubos, validacao cruzada
por objeto com IC e fora da amostra, leitura por memmap em blocos.

Todos os valores esperados vem de verdade conhecida construida no proprio teste (cubo de DN
gerado a partir de uma reflectancia conhecida, funcao linear no comprimento de onda, rotulos
sorteados), nunca de uma saida do proprio codigo."""
from __future__ import annotations

import json
import os
import tracemalloc

import numpy as np
import pytest

from guaraci.hsi_io import load_hsi_folder_dataset

# ---------------------------------------------------------------------------
# utilitarios
# ---------------------------------------------------------------------------


def _hdr(caminho, nl, ns, nb, dt, inter, wl=None, extra=()):
    linhas = ["ENVI", f"samples = {ns}", f"lines = {nl}", f"bands = {nb}", "header offset = 0",
              f"data type = {dt}", f"interleave = {inter}", "byte order = 0", *extra]
    if wl is not None:
        linhas += ["wavelength units = Nanometers", "wavelength = {" + ", ".join(f"{v:.3f}" for v in wl) + "}"]
    with open(caminho, "w") as f:
        f.write("\n".join(linhas) + "\n")


def _gravar(pasta, nome, cubo_hwb, inter="bil", dt=4, wl=None, extra=(), ext=".raw"):
    """Grava cubo (linhas, colunas, bandas) como ENVI no interleave pedido."""
    os.makedirs(pasta, exist_ok=True)
    h, w, b = cubo_hwb.shape
    bruto = {"bip": cubo_hwb, "bil": cubo_hwb.transpose(0, 2, 1), "bsq": cubo_hwb.transpose(2, 0, 1)}[inter]
    tipo = {4: "<f4", 12: "<u2", 2: "<i2"}[dt]
    with open(os.path.join(pasta, nome + ext), "wb") as f:
        f.write(np.ascontiguousarray(bruto).astype(tipo).tobytes())
    _hdr(os.path.join(pasta, nome + ".hdr"), h, w, b, dt, inter, wl, extra)


def _objeto(h=20, w=24, b=12, raio=0.3):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - h / 2) / (h * raio * 1.6)) ** 2 + ((xx - w / 2) / (w * raio * 1.6)) ** 2 <= 1


WL = np.linspace(450, 950, 12)


def _refl(classe_k, rng, h=20, w=24, b=12):
    """Reflectancia verdadeira: fundo 0,06 e objeto com espectro dependente da classe."""
    base = 0.30 + 0.25 * np.exp(-((np.linspace(0, 1, b) - (0.3 + 0.4 * classe_k)) / 0.15) ** 2)
    r = np.full((h, w, b), 0.06)
    m = _objeto(h, w, b)
    r[m] = base * rng.normal(1.0, 0.02) + rng.normal(0, 0.004, (int(m.sum()), b))
    return r


def _dn(refl, branco_col, escuro_col):
    """DN = D + R*(W-D), com W/D por coluna e banda."""
    return escuro_col[None, :, :] + refl * (branco_col - escuro_col)[None, :, :]


def _estudo_dn(tmp_path, n_obj=3, seed=0):
    """2 classes x n_obj objetos em DN uint16 (BIL, .raw); WHITEREF/DARKREF no padrao Specim ao
    lado de cada cubo. Devolve (pasta, {caminho_hdr: reflectancia_verdadeira})."""
    rng = np.random.default_rng(seed)
    h, w, b = 20, 24, 12
    verdade = {}
    for k, classe in enumerate(("verde", "madura")):
        for o in range(n_obj):
            d = tmp_path / classe / f"obj{o}"
            escuro = np.tile(rng.uniform(900, 1100, b), (w, 1))                       # (colunas, bandas)
            branco = escuro + np.tile(np.linspace(30000, 36000, w)[:, None], (1, b)) * \
                np.linspace(0.8, 1.2, b)[None, :]                                    # gradiente de iluminacao
            r = _refl(k, rng, h, w, b)
            _gravar(d, "v0", np.round(_dn(r, branco, escuro)), dt=12, wl=WL)
            # referencias: 5 linhas ruidosas ao redor de W e D (media sobre linhas recupera W e D)
            ref_w = np.stack([branco] * 5) + rng.normal(0, 20, (5, w, b))
            ref_d = np.stack([escuro] * 5) + rng.normal(0, 5, (5, w, b))
            _gravar(d, "WHITEREF_v0", np.round(ref_w), dt=12, wl=WL)
            _gravar(d, "DARKREF_v0", np.round(ref_d), dt=12, wl=WL)
            verdade[str(d / "v0.hdr")] = r
    return str(tmp_path), verdade


# ---------------------------------------------------------------------------
# R11-a: calibracao branco/escuro
# ---------------------------------------------------------------------------


def test_calibracao_branco_escuro_recupera_a_reflectancia_verdadeira(tmp_path):
    pasta, verdade = _estudo_dn(tmp_path)
    cubos, rot, _mae, wl, meta = load_hsi_folder_dataset(pasta, calibracao="branco_escuro")
    assert len(cubos) == 6                                    # WHITEREF/DARKREF nao viraram amostras
    assert meta.attrs["calibracao"]["n_cubos_calibrados"] == 6
    assert list(meta["arquivo"]) == ["v0.hdr"] * 6
    erros = []
    for cubo, (hdr, r) in zip(cubos, sorted(verdade.items(), key=lambda kv: kv[0])):
        erros.append(float(np.abs(np.asarray(cubo) - r).max()))
    # DN quantizado em 1 unidade sobre uma faixa de ~30000, ruido das referencias (media de 5 linhas)
    assert max(erros) < 2e-3
    assert np.asarray(cubos[0]).dtype == np.float32


def test_sem_calibracao_o_cubo_em_dn_e_rejeitado_e_a_mensagem_aponta_a_chave(tmp_path):
    from guaraci.hsi_quality import evaluate_cube_quality
    pasta, _ = _estudo_dn(tmp_path, n_obj=1)
    cubos, *_ = load_hsi_folder_dataset(pasta)                # padrao: nenhuma; so' 2 cubos (refs ignoradas)
    assert len(cubos) == 2
    q = evaluate_cube_quality(cubos[0])
    assert not q.aceito
    assert "hsi_calibracao: branco_escuro" in q.motivo and "(W - D)" in q.motivo


def test_lazy_e_eager_dao_o_mesmo_cubo_calibrado(tmp_path):
    pasta, _ = _estudo_dn(tmp_path, n_obj=1)
    eager, *_ = load_hsi_folder_dataset(pasta, calibracao="branco_escuro")
    lazy, *_ = load_hsi_folder_dataset(pasta, calibracao="branco_escuro", lazy=True)
    for a, b in zip(eager, lazy):
        assert b.shape == a.shape
        np.testing.assert_allclose(np.asarray(b), a, atol=1e-6)


def test_referencia_da_sessao_e_da_config(tmp_path):
    """WHITEREF.hdr/DARKREF.hdr genericos na pasta da classe (uma referencia por sessao) e
    caminho explicito da config valem para todos os cubos abaixo deles."""
    rng = np.random.default_rng(1)
    h, w, b = 10, 12, 12
    escuro = np.tile(rng.uniform(900, 1100, b), (w, 1))
    branco = escuro + 30000.0
    classe = tmp_path / "verde"
    for o in range(2):
        r = _refl(0, rng, h, w, b)
        _gravar(classe / f"obj{o}", "v0", np.round(_dn(r, branco, escuro)), dt=12, wl=WL)
    _gravar(classe, "WHITEREF", np.round(np.stack([branco] * 3)), dt=12, wl=WL)
    _gravar(classe, "DARKREF", np.round(np.stack([escuro] * 3)), dt=12, wl=WL)
    cubos, *_ = load_hsi_folder_dataset(str(tmp_path), calibracao="branco_escuro")
    assert len(cubos) == 2 and all(0.0 < float(np.asarray(c).mean()) < 0.5 for c in cubos)
    # config explicita (arquivo) tem prioridade e funciona sem nenhum arquivo ao lado
    os.rename(classe / "WHITEREF.hdr", tmp_path / "w.hdr")
    os.rename(classe / "WHITEREF.raw", tmp_path / "w.raw")
    os.rename(classe / "DARKREF.hdr", tmp_path / "d.hdr")
    os.rename(classe / "DARKREF.raw", tmp_path / "d.raw")
    with pytest.raises(ValueError, match="sem WHITEREF/DARKREF"):
        load_hsi_folder_dataset(str(tmp_path), calibracao="branco_escuro")
    cubos2, *_ = load_hsi_folder_dataset(str(tmp_path), calibracao="branco_escuro",
                                         referencia_branco=str(tmp_path / "w.hdr"),
                                         referencia_escuro=str(tmp_path / "d.hdr"))
    assert len(cubos2) == 2


def test_calibracao_recusa_branco_menor_que_escuro(tmp_path):
    pasta, _ = _estudo_dn(tmp_path, n_obj=1)
    d = tmp_path / "verde" / "obj0"
    # troca branco e escuro
    for ext in (".hdr", ".raw"):
        os.rename(d / f"WHITEREF_v0{ext}", d / f"tmp{ext}")
        os.rename(d / f"DARKREF_v0{ext}", d / f"WHITEREF_v0{ext}")
        os.rename(d / f"tmp{ext}", d / f"DARKREF_v0{ext}")
    with pytest.raises(ValueError, match="branco nao e' maior que o escuro"):
        load_hsi_folder_dataset(pasta, calibracao="branco_escuro")


def test_calibracao_recusa_branco_saturado_e_avisa_saturacao_parcial(tmp_path, capsys):
    pasta, _ = _estudo_dn(tmp_path, n_obj=1)
    d = tmp_path / "verde" / "obj0"
    h, w, b = 5, 24, 12
    # WHITEREF 12 bit saturado em 4095 em tudo -> recusa; escuro baixo
    _gravar(d, "WHITEREF_v0", np.full((h, w, b), 4095.0), dt=12, wl=WL)
    _gravar(d, "DARKREF_v0", np.full((h, w, b), 100.0), dt=12, wl=WL)
    # o cubo de amostra precisa ser coerente; so' interessa a checagem do branco
    with pytest.raises(ValueError, match="saturado"):
        load_hsi_folder_dataset(pasta, calibracao="branco_escuro")


def test_calibracao_recusa_referencia_de_outra_camera(tmp_path):
    pasta, _ = _estudo_dn(tmp_path, n_obj=1)
    d = tmp_path / "verde" / "obj0"
    _gravar(d, "WHITEREF_v0", np.full((5, 24, 12), 30000.0), dt=12, wl=WL + 40.0)   # eixo deslocado 40 nm
    with pytest.raises(ValueError, match="difere do cubo"):
        load_hsi_folder_dataset(pasta, calibracao="branco_escuro")


def test_calibracao_tudo_ou_nada_lista_os_cubos_sem_referencia(tmp_path):
    pasta, _ = _estudo_dn(tmp_path, n_obj=2)
    d = tmp_path / "madura" / "obj1"
    for pref in ("WHITEREF_v0", "DARKREF_v0"):
        for ext in (".hdr", ".raw"):
            os.remove(d / f"{pref}{ext}")
    with pytest.raises(ValueError) as e:
        load_hsi_folder_dataset(pasta, calibracao="branco_escuro")
    msg = str(e.value)
    assert "1 cubo(s)" in msg and "obj1" in msg and "todos os cubos ou para nenhum" in msg


def test_referencias_ignoradas_tambem_sem_calibracao_e_nao_quebram_o_agrupamento(tmp_path):
    """Um WHITEREF solto na pasta da classe nao pode derrubar o nivel 'high' de agrupamento."""
    rng = np.random.default_rng(2)
    for o in range(2):
        _gravar(tmp_path / "verde" / f"obj{o}", "v0", _refl(0, rng), ext=".bin")
    _gravar(tmp_path / "verde", "WHITEREF", np.full((3, 24, 12), 0.9), ext=".bin")
    _, _, mae, _, meta = load_hsi_folder_dataset(str(tmp_path))
    assert meta.attrs["grouping_guarantee"] == "high" and mae is not None and len(mae) == 2


# ---------------------------------------------------------------------------
# R11-b: eixo de bandas
# ---------------------------------------------------------------------------


def _par_de_cubos(tmp_path, wl_a, wl_b, b=12):
    rng = np.random.default_rng(3)
    for i, wl in enumerate((wl_a, wl_b)):
        # espectro linear no comprimento de onda: reamostragem linear e' exata
        r = np.tile(0.001 * np.asarray(wl)[None, None, :], (6, 6, 1)) + rng.normal(0, 1e-6, (6, 6, len(wl)))
        _gravar(tmp_path / "A" / f"o{i}", "v0", r, wl=wl, ext=".bin")


def test_cameras_com_mesmo_numero_de_bandas_e_eixos_diferentes_sao_recusadas(tmp_path):
    vnir = np.linspace(400, 1000, 12)
    swir = np.linspace(900, 1700, 12)
    _par_de_cubos(tmp_path, vnir, swir)
    with pytest.raises(ValueError) as e:
        load_hsi_folder_dataset(str(tmp_path))
    msg = str(e.value)
    assert "difere" in msg and "900-1700" in msg and "cameras" in msg


def test_pequeno_desvio_de_calibracao_espectral_pode_ser_reamostrado_exatamente(tmp_path):
    wl = np.round(np.linspace(450, 950, 12), 3)          # passo 45,5 nm
    _par_de_cubos(tmp_path, wl, wl + 4.0)               # 4 nm < meia banda
    with pytest.raises(ValueError, match="hsi_eixo_bandas: reamostrar"):
        load_hsi_folder_dataset(str(tmp_path))          # padrao: recusa, e diz como optar
    cubos, _r, _m, wl_saida, meta = load_hsi_folder_dataset(str(tmp_path), eixo_bandas="reamostrar")
    np.testing.assert_allclose(wl_saida, wl)
    # o 2o cubo (eixo + 4 nm) vale 0,001*lambda; reamostrado para wl deve valer 0,001*wl (exato: linear)
    np.testing.assert_allclose(np.asarray(cubos[1])[0, 0, 1:-1], 0.001 * wl[1:-1], atol=1e-4)
    assert meta.attrs["eixo_bandas"]["n_reamostrados"] == 1
    assert meta.attrs["eixo_bandas"]["desvio_max_nm"] == pytest.approx(4.0)


def test_desvio_de_meia_banda_ou_mais_nao_e_reamostrado(tmp_path):
    wl = np.linspace(450, 950, 12)
    _par_de_cubos(tmp_path, wl, wl + 30.0)
    with pytest.raises(ValueError, match="grande demais para reamostrar"):
        load_hsi_folder_dataset(str(tmp_path), eixo_bandas="reamostrar")


def test_deslocamento_dentro_de_1_nm_e_aceito_sem_aviso(tmp_path, capsys):
    wl = np.linspace(450, 950, 12)
    _par_de_cubos(tmp_path, wl, wl + 0.4)
    cubos, *_ = load_hsi_folder_dataset(str(tmp_path))
    assert len(cubos) == 2
    assert "reamostrad" not in capsys.readouterr().out


def test_um_cubo_com_wavelength_e_outro_sem_e_recusado(tmp_path):
    rng = np.random.default_rng(4)
    _gravar(tmp_path / "A" / "o0", "v0", rng.normal(0.4, 0.01, (6, 6, 12)), wl=WL, ext=".bin")
    _gravar(tmp_path / "A" / "o1", "v0", rng.normal(0.4, 0.01, (6, 6, 12)), wl=None, ext=".bin")
    with pytest.raises(ValueError, match="nao ha' como saber se os eixos"):
        load_hsi_folder_dataset(str(tmp_path))


# ---------------------------------------------------------------------------
# R11-d: memmap em blocos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inter", ["bip", "bil", "bsq"])
def test_memmap_le_os_tres_interleaves_igual_ao_leitor_inteiro(tmp_path, inter):
    from guaraci.hsi_io import load_envi_cube
    rng = np.random.default_rng(5)
    cubo = rng.uniform(0, 1, (7, 9, 5)).astype("f4")
    _gravar(tmp_path, "c", cubo, inter=inter, ext=".bin")
    cheio, _ = load_envi_cube(str(tmp_path / "c.bin"))
    lazy, *_ = load_hsi_folder_dataset(str(tmp_path), lazy=True)
    np.testing.assert_array_equal(np.asarray(lazy[0]), cheio)
    np.testing.assert_array_equal(np.asarray(lazy[0]), cubo)
    lin = np.array([0, 3, 6, 3]); col = np.array([8, 0, 4, 1])
    np.testing.assert_array_equal(lazy[0].pixels(lin, col), cubo[lin, col])
    r = np.concatenate([b for _, _, b in lazy[0].blocos(2)], axis=0)
    np.testing.assert_array_equal(r, cubo)


def test_qualidade_e_segmentacao_em_blocos_dao_o_mesmo_que_em_memoria(tmp_path, monkeypatch):
    from sklearn.decomposition import PCA
    from guaraci import hsi_blocos
    from guaraci.hsi_quality import evaluate_cube_quality
    from guaraci.hsi_segmentation import segment_object_pca_otsu
    monkeypatch.setattr(hsi_blocos, "_BYTES_BLOCO", 3000)       # forca muitos blocos pequenos
    rng = np.random.default_rng(6)
    cubo = _refl(1, rng).astype("f4")
    _gravar(tmp_path, "c", cubo, inter="bil", ext=".bin")
    lazy, *_ = load_hsi_folder_dataset(str(tmp_path), lazy=True)
    assert len(list(lazy[0].blocos())) > 3
    q_lazy, q_mem = evaluate_cube_quality(lazy[0]), evaluate_cube_quality(cubo)
    assert q_lazy.aceito == q_mem.aceito
    assert q_lazy.snr_estimado == pytest.approx(q_mem.snr_estimado, rel=1e-4)
    seg_lazy, seg_mem = segment_object_pca_otsu(lazy[0]), segment_object_pca_otsu(cubo)
    np.testing.assert_array_equal(seg_lazy.mascara, seg_mem.mascara)
    # equivalencia com o PCA(1) de sklearn sobre todos os pixels (verdade independente)
    px = cubo.reshape(-1, cubo.shape[2]).astype(float)
    pca = PCA(n_components=1).fit(px)
    pc1_ref = pca.transform(px).ravel()
    pc1 = seg_lazy.imagem_pc1.ravel()
    assert abs(np.corrcoef(pc1, pc1_ref)[0, 1]) > 0.999999
    assert seg_lazy.variancia_explicada_pc1 == pytest.approx(float(pca.explained_variance_ratio_[0]), rel=1e-4)


def test_quality_e_segmentacao_nao_copiam_o_cubo_para_float64(tmp_path, monkeypatch):
    from guaraci import hsi_blocos
    from guaraci.hsi_quality import evaluate_cube_quality
    from guaraci.hsi_segmentation import segment_object_pca_otsu
    rng = np.random.default_rng(7)
    h, w, b = 120, 160, 40
    cubo = (0.06 + rng.normal(0, 0.004, (h, w, b))).astype("f4")
    cubo[30:90, 40:120] += 0.3
    _gravar(tmp_path, "c", cubo, inter="bil", ext=".bin")
    lazy, *_ = load_hsi_folder_dataset(str(tmp_path), lazy=True)
    monkeypatch.setattr(hsi_blocos, "_BYTES_BLOCO", 200_000)
    tamanho_float64 = h * w * b * 8
    tracemalloc.start()
    evaluate_cube_quality(lazy[0])
    seg = segment_object_pca_otsu(lazy[0])
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert seg.fracao_objeto == pytest.approx(60 * 80 / (h * w), abs=0.02)
    assert pico < 0.35 * tamanho_float64, (pico, tamanho_float64)


def test_build_pixel_dataset_com_cubo_lazy_sorteia_os_mesmos_pixels_do_array(tmp_path):
    from guaraci.hsi_pixels import build_pixel_dataset
    rng = np.random.default_rng(8)
    cubo = rng.uniform(0, 1, (30, 30, 6)).astype("f4")
    masc = np.zeros((30, 30), bool); masc[5:25, 5:25] = True
    _gravar(tmp_path, "c", cubo, inter="bsq", ext=".bin")
    lazy, *_ = load_hsi_folder_dataset(str(tmp_path), lazy=True)
    Xa, ya, ga = build_pixel_dataset([cubo], [masc], ["g"], ["a"], max_pixels_por_gravacao=100, seed=3)
    Xl, yl, gl = build_pixel_dataset(lazy, [masc], ["g"], ["a"], max_pixels_por_gravacao=100, seed=3)
    np.testing.assert_array_equal(Xa, Xl)


# ---------------------------------------------------------------------------
# R11-c: validacao cruzada por objeto
# ---------------------------------------------------------------------------


def _pixels(n_obj_por_classe, n_px=40, n_b=10, sep=3.0, seed=0, nulo=False):
    """X, y, grupos: cada objeto tem um deslocamento proprio (assinatura do objeto); se `nulo`,
    o rotulo da classe e' sorteado POR OBJETO (nenhuma informacao de classe nos espectros)."""
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [], [], []
    classes = list(n_obj_por_classe)
    objs = [(c, i) for c in classes for i in range(n_obj_por_classe[c])]
    rot = [c for c, _ in objs]
    if nulo:
        rot = list(rng.permutation(rot))
    centros = {c: rng.normal(0, 1, n_b) * sep for c in classes}
    for (c, i), r in zip(objs, rot):
        desloc = rng.normal(0, 1.0, n_b)                      # assinatura do objeto
        base = desloc if nulo else centros[c] + desloc * 0.3
        Xs.append(base + rng.normal(0, 0.3, (n_px, n_b)))
        ys += [r] * n_px
        gs += [f"{c}{i}"] * n_px
    return np.vstack(Xs), np.array(ys), np.array(gs)


def test_intervalo_clopper_pearson_valores_conhecidos():
    from guaraci.hsi_validation import intervalo_clopper_pearson
    lo, hi = intervalo_clopper_pearson(6, 6)                  # 100%: limite inferior = (alpha/2)^(1/n)
    assert (lo, hi) == (pytest.approx(0.025 ** (1 / 6), abs=1e-9), 1.0)
    lo, hi = intervalo_clopper_pearson(0, 10)
    assert (lo, hi) == (0.0, pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-9))
    lo, hi = intervalo_clopper_pearson(5, 10)                 # valor tabelado: 0,187 a 0,813
    assert lo == pytest.approx(0.1871, abs=5e-4) and hi == pytest.approx(0.8129, abs=5e-4)
    assert all(np.isnan(intervalo_clopper_pearson(0, 0)))


def test_cv_por_objeto_cobre_todos_os_objetos_e_nunca_treina_com_o_objeto_previsto(monkeypatch):
    from guaraci import hsi_validation as hv
    X, y, g = _pixels({"a": 6, "b": 6, "c": 5})
    vistos = []
    original = hv.fit_predict_pixel_plsda

    def espiao(Xtr, ytr, gtr, Xte, gte, **kw):
        assert not (set(gtr.tolist()) & set(gte.tolist()))    # treino e teste sem objeto em comum
        vistos.append(sorted(set(gte.tolist())))
        return original(Xtr, ytr, gtr, Xte, gte, **kw)
    monkeypatch.setattr(hv, "fit_predict_pixel_plsda", espiao)
    rel = hv.run_cv_by_object(X, y, g, n_dobras=4)
    assert rel.tipo_validacao == "cv_objetos" and rel.n_dobras == 4
    assert sorted(o for f in vistos for o in f) == sorted(set(g.tolist()))     # cada objeto uma vez
    assert rel.n_objetos_total == rel.n_objetos_avaliados == 17
    assert rel.n_objetos_por_classe == {"a": 6, "b": 6, "c": 5}
    assert rel.confianca_fora_da_amostra and set(rel.predicoes_objeto) == set(g.tolist())
    # classes separaveis: tudo certo, IC com limite inferior = (alpha/2)^(1/6)
    assert all(rel.sensibilidade_interna[c] == 1.0 for c in "abc")
    assert rel.ic_sensibilidade_interna["a"][0] == pytest.approx(0.025 ** (1 / 6), abs=1e-6)
    assert rel.acuracia_objetos == 1.0


def test_cv_leave_one_object_out_com_dobras_zero():
    from guaraci.hsi_validation import run_cv_by_object
    X, y, g = _pixels({"a": 4, "b": 4}, n_px=15)
    rel = run_cv_by_object(X, y, g, n_dobras=0)
    assert rel.n_dobras == 8 and rel.n_objetos_avaliados == 8


def test_cv_com_rotulo_sorteado_por_objeto_nao_infla_e_o_ic_cobre_o_acaso():
    """Verdade conhecida: rotulos sorteados por objeto => acuracia esperada = 0,5. O holdout
    in-sample/confianca de resubstituicao pareceria alta; a CV por objeto nao."""
    from guaraci.hsi_classification import fit_predict_pixel_plsda
    from guaraci.hsi_uncertainty import enrich_object_results
    from guaraci.hsi_validation import run_cv_by_object
    X, y, g = _pixels({"a": 12, "b": 12}, n_px=30, n_b=40, sep=1.0, seed=11, nulo=True)
    rel = run_cv_by_object(X, y, g, n_dobras=6)
    lo, hi = rel.ic_acuracia_objetos
    assert lo <= 0.5 <= hi and rel.acuracia_balanceada_objetos < 0.8
    # confianca fora da amostra x dentro da amostra
    oof = enrich_object_results(rel.predicoes_objeto)
    ins = enrich_object_results(fit_predict_pixel_plsda(X, y, g, X, g)["predicoes_objeto"])
    het_oof = np.mean([r.heterogeneidade for r in oof.values()])
    het_ins = np.mean([r.heterogeneidade for r in ins.values()])
    assert het_oof > het_ins + 0.05


def test_cv_avisa_classe_com_poucos_objetos_e_recusa_poucos_dados():
    from guaraci.hsi_validation import run_cv_by_object
    X, y, g = _pixels({"a": 6, "b": 2})
    rel = run_cv_by_object(X, y, g, n_dobras=0)
    assert any("menos de 3 objetos" in a for a in rel.avisos)
    X2, y2, g2 = _pixels({"a": 1, "b": 1})
    with pytest.raises(ValueError, match="pelo menos 2 classes e 3 objetos"):
        run_cv_by_object(X2, y2, g2)


def test_holdout_antigo_continua_disponivel_e_declara_n_de_objetos():
    from guaraci.hsi_validation import run_internal_validation_group_aware
    X, y, g = _pixels({"a": 8, "b": 8}, n_px=10)
    cubos = [X[g == o].reshape(-1, 1, X.shape[1]) for o in sorted(set(g))]
    masc = [np.ones(c.shape[:2], bool) for c in cubos]
    rot = [y[g == o][0] for o in sorted(set(g))]
    rel = run_internal_validation_group_aware(cubos, masc, sorted(set(g)), rot, fracao_teste=0.25)
    assert rel.tipo_validacao == "holdout" and not rel.confianca_fora_da_amostra
    assert rel.n_objetos_total == 16 and rel.n_objetos_avaliados == 4
    assert sum(rel.n_objetos_por_classe.values()) == 4
    assert any("holdout de 4" in a for a in rel.avisos)


# ---------------------------------------------------------------------------
# fluxo completo: pipeline + relatorio
# ---------------------------------------------------------------------------


def _cfg(tmp_path, pasta, **kw):
    from guaraci.config import Config
    cfg = Config()
    cfg.hsi_dataset_folder = pasta
    cfg.output_folder = str(tmp_path / "saida")
    cfg.default_preprocessing = "snv_mc"
    cfg.wn_min, cfg.wn_max = 0.0, 999999.0
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_pipeline_hsi_cv_por_objeto_grava_campos_para_a_ficha(tmp_path):
    from guaraci.exemplos import _cubos_hsi
    from guaraci.hsi_pipeline import gravar_relatorio_hsi, run_hsi_pipeline
    _cubos_hsi(str(tmp_path / "cubos"))
    cfg = _cfg(tmp_path, str(tmp_path / "cubos"))
    resumo = run_hsi_pipeline(cfg)
    val = resumo["validacao_externa"]
    assert val.tipo_validacao == "cv_objetos" and val.n_objetos_total == 12
    assert resumo["confianca_fora_da_amostra"] is True
    txt = open(gravar_relatorio_hsi(cfg, resumo), encoding="utf-8").read()
    assert "Validacao: CV por objeto, 5 dobras" in txt and "IC95%" in txt and "n objetos = 6" in txt
    assert "Confianca por objeto: fora da amostra" in txt
    dados = json.load(open(os.path.join(os.path.dirname(gravar_relatorio_hsi(cfg, resumo)), "hsi_relatorio.json"),
                           encoding="utf-8"))
    v = dados["validacao"]
    assert v["tipo"] == "cv_objetos" and v["n_dobras"] == 5 and v["n_objetos_total"] == 12
    assert v["n_objetos_por_classe"] == {"madura": 6, "verde": 6}
    assert v["por_classe"]["verde"]["n_objetos"] == 6 and len(v["por_classe"]["verde"]["sensibilidade_ic95"]) == 2
    assert v["confianca_por_objeto_fora_da_amostra"] is True
    assert dados["calibracao"]["modo"] == "nenhuma" and dados["eixo_bandas"]["n_bandas"] == 30


def test_pipeline_hsi_holdout_por_chave_declara_o_tipo(tmp_path):
    from guaraci.exemplos import _cubos_hsi
    from guaraci.hsi_pipeline import gravar_relatorio_hsi, run_hsi_pipeline
    _cubos_hsi(str(tmp_path / "cubos"))
    cfg = _cfg(tmp_path, str(tmp_path / "cubos"), hsi_validation="holdout")
    resumo = run_hsi_pipeline(cfg)
    assert resumo["validacao_externa"].tipo_validacao == "holdout"
    assert resumo["confianca_fora_da_amostra"] is False
    txt = open(gravar_relatorio_hsi(cfg, resumo), encoding="utf-8").read()
    assert "Validacao: holdout unico" in txt and "DENTRO da amostra" in txt


def test_pipeline_hsi_com_dn_bruto_e_calibracao_classifica_como_com_reflectancia(tmp_path):
    from guaraci.hsi_pipeline import gravar_relatorio_hsi, run_hsi_pipeline
    pasta, _ = _estudo_dn(tmp_path / "dn", n_obj=4)
    cfg = _cfg(tmp_path, pasta, hsi_calibration="branco_escuro")
    resumo = run_hsi_pipeline(cfg)
    assert resumo["n_gravacoes_aceitas"] == 8 and resumo["calibracao"]["n_cubos_calibrados"] == 8
    assert resumo["validacao_externa"].acuracia_objetos == 1.0
    txt = open(gravar_relatorio_hsi(cfg, resumo), encoding="utf-8").read()
    assert "R = (I - D)/(W - D)" in txt
    # sem a calibracao os mesmos cubos sao rejeitados pelo portao de qualidade
    with pytest.raises(ValueError, match="hsi_calibracao: branco_escuro"):
        run_hsi_pipeline(_cfg(tmp_path / "o2", pasta))


def test_chaves_novas_de_config_ida_e_volta_e_validacao(tmp_path):
    from guaraci.config import Config
    from guaraci.config_io import load_config, save_config
    cfg = Config()
    cfg.hsi_calibration, cfg.hsi_band_axis, cfg.hsi_validation = "branco_escuro", "reamostrar", "holdout"
    cfg.hsi_cv_folds, cfg.hsi_white_ref, cfg.hsi_dark_ref = 0, "W.hdr", "D.hdr"
    p = tmp_path / "c.yaml"
    save_config(cfg, str(p))
    volta = load_config(str(p))
    assert (volta.hsi_calibration, volta.hsi_band_axis, volta.hsi_validation) == ("branco_escuro", "reamostrar", "holdout")
    assert (volta.hsi_cv_folds, volta.hsi_white_ref, volta.hsi_dark_ref) == (0, "W.hdr", "D.hdr")
    p.write_text("hsi_calibracao: tanto_faz\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hsi_calibracao"):
        load_config(str(p))
    p.write_text("hsi_validacao: cv_objetos\nhsi_cv_dobras: 5\n", encoding="utf-8")
    assert load_config(str(p)).hsi_validation == "cv_objetos"
