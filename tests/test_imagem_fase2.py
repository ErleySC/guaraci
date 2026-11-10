"""Fase 2 da imagem digital (R12): normalizacao por cartao cinza/branco, segmentacao do objeto,
aviso de fundo dominante, TIFF/PNG 16 bit e matiz circular.

Verdade conhecida: fotos sinteticas em que o ganho multiplicativo por canal de cada sessao e o
fundo de cada classe sao escolhidos no teste."""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from guaraci.dados_imagem import (extract_color_features, extrair_features_foto, load_image_file,
                                  load_images, normalizar_por_referencia, segmentar_objeto)

CARTAO = (0.02, 0.02, 0.18, 0.18)          # x0, y0, x1, y1 (canto superior esquerdo)
H, W = 80, 100
OBJ = (0.4, 0.4, 0.6, 0.6)                  # recorte dentro do objeto (so' a cor do objeto)


def _foto(cor_obj, fundo, ganho=(1.0, 1.0, 1.0), cartao=(128, 128, 128), rng=None, ruido=2.0, raio=0.28):
    """Cena: fundo uniforme, elipse central (objeto), cartao cinza no canto. O ganho por canal
    multiplica a CENA INTEIRA (iluminacao + balanco de branco da sessao), cartao incluido."""
    rng = rng or np.random.default_rng(0)
    img = np.empty((H, W, 3), dtype=np.float64)
    img[:] = np.asarray(fundo, dtype=float)
    yy, xx = np.mgrid[0:H, 0:W]
    obj = ((yy - H / 2) / (H * raio)) ** 2 + ((xx - W / 2) / (W * raio)) ** 2 <= 1
    img[obj] = np.asarray(cor_obj, dtype=float)
    y0, y1, x0, x1 = int(H * CARTAO[1]), int(H * CARTAO[3]), int(W * CARTAO[0]), int(W * CARTAO[2])
    img[y0:y1, x0:x1] = np.asarray(cartao, dtype=float)
    img = img * np.asarray(ganho)[None, None, :] + rng.normal(0, ruido, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _salvar(pasta, classe, amostra, nome, img):
    d = os.path.join(pasta, classe, amostra)
    os.makedirs(d, exist_ok=True)
    Image.fromarray(img).save(os.path.join(d, nome))


# ---------------------------------------------------------------------------
# (a) normalizacao por cartao
# ---------------------------------------------------------------------------


def test_normalizacao_por_cartao_cancela_o_ganho_por_canal():
    obj = (190, 90, 60)
    fundo = (60, 60, 60)
    a = _foto(obj, fundo, ganho=(1.0, 1.0, 1.0), rng=np.random.default_rng(1))
    b = _foto(obj, fundo, ganho=(0.6, 0.8, 1.1), rng=np.random.default_rng(2))
    fa, _ = extrair_features_foto(a, recorte=OBJ)
    fb, _ = extrair_features_foto(b, recorte=OBJ)
    assert abs(fa["R_media"] - fb["R_media"]) > 30                      # sem correcao a sessao muda a cor
    na, _ = extrair_features_foto(a, recorte=OBJ, referencia=CARTAO)
    nb, _ = extrair_features_foto(b, recorte=OBJ, referencia=CARTAO)
    for canal in "RGB":
        assert abs(na[f"{canal}_media"] - nb[f"{canal}_media"]) < 2.5   # com cartao, igual a menos do ruido
    # e a cor corrigida vale (cor do objeto / cartao) * nivel de referencia
    assert na["G_media"] == pytest.approx(90 / 128 * 190, abs=3.0)


def test_cartao_escuro_demais_e_recusado_com_mensagem():
    img = _foto((190, 90, 60), (60, 60, 60), cartao=(5, 5, 5))
    with pytest.raises(ValueError, match="escura demais"):
        normalizar_por_referencia(img, CARTAO)


def test_aviso_de_cartao_saturado_e_de_regiao_nao_uniforme(tmp_path, capsys):
    for i in range(4):
        _salvar(str(tmp_path), "A", f"s{i}", "f.png",
                _foto((190, 90, 60), (60, 60, 60), cartao=(254, 254, 254), rng=np.random.default_rng(i)))
    load_images(str(tmp_path), referencia=CARTAO)
    saida = capsys.readouterr().out
    assert "esta saturado" in saida and "4/4" in saida
    # regiao de referencia sobre o objeto/fundo (nao uniforme)
    load_images(str(tmp_path), referencia=(0.1, 0.1, 0.6, 0.6))
    assert "nao e' uniforme" in capsys.readouterr().out


def _sessoes(tmp_path, ganhos, cor_a=(180, 100, 60), cor_b=(180, 140, 60), n=6, fundo=(70, 70, 70)):
    """Pasta classe/amostra/foto; cada foto pertence a uma sessao (prefixo s0_/s1_ no nome)."""
    for s, ganho in enumerate(ganhos):
        for classe, cor in (("A", cor_a), ("B", cor_b)):
            for i in range(n):
                _salvar(str(tmp_path), classe, f"{classe}_{s}_{i}", f"s{s}_{i}.png",
                        _foto(cor, fundo, ganho=ganho, rng=np.random.default_rng(100 * s + i + (7 if classe == "B" else 0))))


def _acc_entre_sessoes(tmp_path, **kw):
    """Treina na sessao 0 e testa na sessao 1 (acuracia cruzada por sessao)."""
    _, X, rot, _c, _mae, meta = load_images(str(tmp_path), OBJ, **kw)
    sess = np.array([a.split("_")[0] for a in meta["arquivo"]])
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000))
    modelo.fit(X[sess == "s0"], rot[sess == "s0"])
    return float(np.mean(modelo.predict(X[sess == "s1"]) == rot[sess == "s1"]))


def test_iluminacao_por_sessao_derruba_a_acuracia_cruzada_e_o_cartao_recupera(tmp_path):
    # sessao 1 tem o canal G a 70%: a classe B (G=140) da sessao 1 fica com a cor da classe A da sessao 0
    _sessoes(tmp_path, ganhos=[(1.0, 1.0, 1.0), (1.0, 0.7, 1.0)])
    sem = _acc_entre_sessoes(tmp_path)
    com = _acc_entre_sessoes(tmp_path, referencia=CARTAO)
    assert sem <= 0.6, sem            # o ganho da sessao se confunde com a classe
    assert com >= 0.95, com           # normalizado pelo cartao: classes recuperadas
    assert com - sem >= 0.35


# ---------------------------------------------------------------------------
# (b) segmentacao
# ---------------------------------------------------------------------------


def test_segmentacao_otsu_e_distancia_ao_fundo_acham_o_objeto():
    img = _foto((200, 60, 60), (30, 30, 30))
    for metodo in ("otsu", "distancia_fundo"):
        m, info = segmentar_objeto(img, metodo)
        assert m is not None
        yy, xx = np.mgrid[0:H, 0:W]
        verdade = ((yy - H / 2) / (H * 0.28)) ** 2 + ((xx - W / 2) / (W * 0.28)) ** 2 <= 1
        iou = (m & verdade).sum() / (m | verdade).sum()
        assert iou > 0.9, (metodo, iou)
    # objeto maior que o fundo: o lado que domina a borda continua sendo o fundo
    grande = _foto((200, 60, 60), (30, 30, 30), raio=0.45)
    m, _ = segmentar_objeto(grande, "otsu")
    assert m is not None and m.mean() > 0.5
    # sem contraste: devolve None e o motivo
    plano = np.full((H, W, 3), 100, np.uint8)
    assert segmentar_objeto(plano, "otsu")[0] is None


def _fundos_por_classe(tmp_path, cor_a, cor_b, fundo_a, fundo_b, n=10):
    for classe, cor, fundo in (("A", cor_a, fundo_a), ("B", cor_b, fundo_b)):
        for i in range(n):
            _salvar(str(tmp_path), classe, f"{classe}{i}", "f.png",
                    _foto(cor, fundo, cartao=fundo, rng=np.random.default_rng(i + (50 if classe == "B" else 0))))


def _acc_cv(tmp_path, **kw):
    _, X, rot, _c, mae, _m = load_images(str(tmp_path), **kw)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000))
    return float(np.mean(cross_val_score(modelo, X, rot, groups=mae, cv=GroupKFold(5))))


def test_sem_segmentacao_o_modelo_aprende_o_fundo_com_segmentacao_nao(tmp_path):
    """Objeto IDENTICO nas duas classes; so' o fundo difere (A escuro, B claro). Sinal real = 0:
    a acuracia deve ficar no acaso (0,5) quando so' o objeto e' medido."""
    _fundos_por_classe(tmp_path, (190, 90, 60), (190, 90, 60), (25, 25, 25), (225, 225, 225))
    sem = _acc_cv(tmp_path)
    com_otsu = _acc_cv(tmp_path, segmentacao="otsu")
    com_dist = _acc_cv(tmp_path, segmentacao="distancia_fundo")
    assert sem >= 0.95, sem                       # aprendeu a mesa
    assert com_otsu <= 0.7 and com_dist <= 0.7, (com_otsu, com_dist)


def test_com_segmentacao_o_sinal_real_do_objeto_aparece_mesmo_com_fundo_contrario(tmp_path):
    """Objetos diferem (B mais claro no G), mas o fundo de B e' MAIS ESCURO que o de A, o que
    atrapalha uma leitura da foto inteira com poucos pixels de objeto; medido so' no objeto, 1,0."""
    _fundos_por_classe(tmp_path, (190, 90, 60), (190, 140, 60), (200, 200, 200), (20, 20, 20))
    assert _acc_cv(tmp_path, segmentacao="otsu") >= 0.95


def test_segmentacao_que_falha_usa_a_foto_inteira_e_avisa(tmp_path, capsys):
    for i in range(3):
        _salvar(str(tmp_path), "A", f"s{i}", "f.png", np.full((H, W, 3), 100, np.uint8))
    load_images(str(tmp_path), segmentacao="otsu")
    assert "segmentacao 'otsu' falhou em 3/3" in capsys.readouterr().out


def test_mascara_e_usada_nas_estatisticas_e_o_matiz_continua_circular():
    import colorsys
    rng = np.random.default_rng(3)
    img = np.full((40, 40, 3), 255, np.uint8)                        # fundo branco
    m = np.zeros((40, 40), bool); m[10:30, 10:30] = True
    # objeto vermelho com matiz dos dois lados de 0 (359 e 1 graus)
    hs = np.where(rng.random((20, 20)) < 0.5, -0.005, 0.005)
    rgb = np.array([[colorsys.hsv_to_rgb(h % 1.0, 0.9, 0.7) for h in lin] for lin in hs])
    img[m] = (rgb.reshape(-1, 3) * 255).round().astype(np.uint8)
    f = extract_color_features(img, m)
    assert min(f["H_media"], 1 - f["H_media"]) < 0.01 and f["H_dp"] < 0.02
    f_inteira = extract_color_features(img)
    assert f_inteira["V_media"] > f["V_media"] + 0.2                  # sem mascara o fundo domina


# ---------------------------------------------------------------------------
# (c) aviso de fundo dominante / fundo que separa classes
# ---------------------------------------------------------------------------


def test_aviso_de_fundo_dominante_sem_segmentacao_nem_recorte(tmp_path, capsys):
    _fundos_por_classe(tmp_path, (190, 90, 60), (190, 140, 60), (60, 60, 60), (60, 60, 60), n=4)
    meta = load_images(str(tmp_path))[5]
    saida = capsys.readouterr().out
    assert "ocupa mais de 50%" in saida and "imagem_segmentacao" in saida
    assert meta.attrs["avisos"]
    assert not any("cor do fundo" in a for a in meta.attrs["avisos"])      # mesmo fundo nas duas classes
    # com segmentacao ou recorte o aviso some
    load_images(str(tmp_path), segmentacao="otsu")
    assert "ocupa mais de 50%" not in capsys.readouterr().out
    load_images(str(tmp_path), caixa_recorte=OBJ)
    assert "ocupa mais de 50%" not in capsys.readouterr().out


def test_aviso_quando_a_cor_do_fundo_separa_as_classes(tmp_path, capsys):
    _fundos_por_classe(tmp_path, (190, 90, 60), (190, 90, 60), (25, 25, 25), (225, 225, 225), n=4)
    load_images(str(tmp_path))
    assert "o modelo pode estar aprendendo o fundo" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 16 bit
# ---------------------------------------------------------------------------


def test_png_cinza_16bit_nao_satura(tmp_path):
    g = np.full((8, 8), 20000, dtype=np.uint16)
    Image.fromarray(g).save(tmp_path / "f.png")
    img = load_image_file(str(tmp_path / "f.png"))
    assert img.shape == (8, 8, 3) and int(img[0, 0, 0]) == pytest.approx(20000 / 257, abs=1)


def test_tiff_rgb_16bit_da_as_mesmas_features_que_o_8bit(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(4)
    base8 = rng.integers(40, 220, (30, 30, 3)).astype(np.uint8)
    tifffile.imwrite(tmp_path / "a16.tif", (base8.astype(np.uint16) * 257), photometric="rgb")
    Image.fromarray(base8).save(tmp_path / "a8.png")
    f16 = extract_color_features(load_image_file(str(tmp_path / "a16.tif")))
    f8 = extract_color_features(load_image_file(str(tmp_path / "a8.png")))
    for k in f8:
        assert f16[k] == pytest.approx(f8[k], abs=0.6 if not k.startswith("H") else 0.01)


def test_normalizacao_por_cartao_funciona_com_tiff_16bit(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    obj, fundo = (190, 90, 60), (60, 60, 60)
    for i, ganho in enumerate(((1.0, 1.0, 1.0), (0.6, 0.8, 1.1))):
        img = _foto(obj, fundo, ganho=ganho, rng=np.random.default_rng(i))
        d = tmp_path / "A" / f"s{i}"
        d.mkdir(parents=True)
        tifffile.imwrite(d / "f.tif", img.astype(np.uint16) * 257, photometric="rgb")
    _, X, *_ = load_images(str(tmp_path), OBJ, referencia=CARTAO)
    assert np.abs(X[0, :3] - X[1, :3]).max() < 2.5


# ---------------------------------------------------------------------------
# config, exportacao e fluxo completo
# ---------------------------------------------------------------------------


def test_chaves_imagem_referencia_e_segmentacao_ida_e_volta(tmp_path):
    from guaraci.config import Config
    from guaraci.config_io import load_config, save_config
    cfg = Config()
    assert cfg.imagem_referencia == () and cfg.image_segmentation == "nenhuma"
    cfg.imagem_referencia = (0.02, 0.02, 0.18, 0.18)
    cfg.image_segmentation = "otsu"
    p = tmp_path / "c.yaml"
    save_config(cfg, str(p))
    volta = load_config(str(p))
    assert volta.imagem_referencia == (0.02, 0.02, 0.18, 0.18) and volta.image_segmentation == "otsu"
    p.write_text("imagem_referencia: []\nimagem_segmentacao: nenhuma\n", encoding="utf-8")
    assert load_config(str(p)).imagem_referencia == ()
    p.write_text("imagem_referencia: [0.5, 0, 0.2, 1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="imagem_referencia"):
        load_config(str(p))
    p.write_text("imagem_segmentacao: watershed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_fluxo_completo_grava_os_parametros_no_modelo_exportado(tmp_path):
    from guaraci.config import Config
    from guaraci.model_export import load_portable_json
    from guaraci.pipeline import executar
    _sessoes(tmp_path / "fotos", ganhos=[(1.0, 1.0, 1.0), (1.0, 0.7, 1.0)], n=5)
    cfg = Config()
    cfg.mode = "imagem"
    cfg.input_folder = str(tmp_path / "fotos")
    cfg.output_root_folder = str(tmp_path / "res")
    cfg.default_preprocessing = "autoscaling"
    cfg.wn_min, cfg.wn_max = -1.0, 100.0
    cfg.imagem_recorte = OBJ
    cfg.imagem_referencia = CARTAO
    cfg.image_segmentation = "nenhuma"
    cfg.frac_holdout = 0.0
    cfg.n_permutations = 0
    cfg.run_benchmark = False
    executar(cfg)
    achados = [os.path.join(r, f) for r, _d, fs in os.walk(tmp_path / "res") for f in fs if f.endswith(".portatil.json")]
    assert achados, "o modelo portatil nao foi gravado"
    dados = load_portable_json(achados[0])
    assert dados["imagem"]["referencia"] == list(CARTAO)
    assert dados["imagem"]["recorte"] == list(OBJ)
    assert dados["imagem"]["segmentacao"] == "nenhuma" and dados["imagem"]["nivel_referencia"] == 190.0
    # a predicao reaplica igual: as mesmas features saem do mesmo caminho de codigo
    foto = load_image_file(os.path.join(str(tmp_path / "fotos"), "A", "A_0_0", "s0_0.png"))
    f, _ = extrair_features_foto(foto, **dados["imagem"])
    _, X, *_ = load_images(str(tmp_path / "fotos"), OBJ, referencia=CARTAO)
    assert f["R_media"] == pytest.approx(X[0, 0], abs=1e-9)
