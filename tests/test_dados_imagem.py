"""Testes de dados_imagem.py — colorimetria digital (protótipo genérico):
conversões de cor, recorte, extração de features e carregamento fim-a-fim
com imagens REAIS gravadas em disco (mesmo padrão de test_dados_io_jcamp.py).
"""
import colorsys
import numpy as np
import pytest
from PIL import Image

from conftest import achar_pastas_run


# ── Conversões de cor (validadas contra referências conhecidas) ──────────────

def test_rgb_para_hsv_bate_com_colorsys(pq):
    """RGB->HSV deve bater com colorsys.rgb_to_hsv (referência da stdlib)."""
    rng = np.random.default_rng(0)
    amostras = rng.random((50, 3))
    from guaraci.dados_imagem import _rgb_para_hsv
    hsv = _rgb_para_hsv(amostras)
    for i in range(50):
        h_ref, s_ref, v_ref = colorsys.rgb_to_hsv(*amostras[i])
        assert hsv[i, 0] == pytest.approx(h_ref, abs=1e-6)
        assert hsv[i, 1] == pytest.approx(s_ref, abs=1e-6)
        assert hsv[i, 2] == pytest.approx(v_ref, abs=1e-6)


def test_rgb_para_lab_branco_e_preto(pq):
    """Branco puro -> L=100,a=0,b=0; preto puro -> L=0,a=0,b=0 (referência
    conhecida de colorimetria, D65)."""
    from guaraci.dados_imagem import _rgb_para_lab
    branco = np.array([[[1.0, 1.0, 1.0]]])
    preto = np.array([[[0.0, 0.0, 0.0]]])
    lab_branco = _rgb_para_lab(branco)[0, 0]
    lab_preto = _rgb_para_lab(preto)[0, 0]
    assert lab_branco[0] == pytest.approx(100.0, abs=0.1)
    assert lab_branco[1] == pytest.approx(0.0, abs=0.1)
    assert lab_branco[2] == pytest.approx(0.0, abs=0.1)
    np.testing.assert_allclose(lab_preto, 0.0, atol=0.1)


# ── Recorte relativo ───────────────────────────────────────────────────────

def test_recortar_relativo_metade_central(pq):
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    recorte = pq.recortar_relativo(img, (0.25, 0.25, 0.75, 0.75))
    assert recorte.shape == (20, 20, 3)


def test_recortar_relativo_caixa_degenerada_devolve_imagem_inteira(pq):
    """Caixa invertida/vazia (ex.: direita <= esquerda) é tratada com
    segurança — devolve a imagem inteira em vez de um array vazio/erro."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    recorte = pq.recortar_relativo(img, (0.5, 0.5, 0.5, 0.5))
    assert recorte.shape == img.shape


# ── Extração de features de cor ───────────────────────────────────────────

def test_extrair_features_cor_retorna_18_features_finitas(pq):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(30, 30, 3), dtype=np.uint8)
    feats = pq.extrair_features_cor(img)
    assert len(feats) == 18
    assert all(np.isfinite(v) for v in feats.values())


def test_extrair_features_cor_distingue_cores_diferentes(pq):
    """Duas imagens de cores bem diferentes devem produzir R_media bem
    diferente — checagem de sanidade (não é so ruído numérico)."""
    amarelo = np.zeros((20, 20, 3), dtype=np.uint8)
    amarelo[..., 0] = 220; amarelo[..., 1] = 200; amarelo[..., 2] = 40
    azul = np.zeros((20, 20, 3), dtype=np.uint8)
    azul[..., 0] = 30; azul[..., 1] = 40; azul[..., 2] = 200

    f_amarelo = pq.extrair_features_cor(amarelo)
    f_azul = pq.extrair_features_cor(azul)
    assert f_amarelo["R_media"] > f_azul["R_media"]
    assert f_azul["B_media"] > f_amarelo["B_media"]


def test_extrair_features_cor_aceita_imagem_2d_tons_de_cinza(pq):
    """Imagem 2D (H, W) sem canal de cor -- e' replicada em 3 canais R=G=B
    automaticamente (linha de compatibilidade pouco exercitada)."""
    cinza = np.full((20, 20), 128, dtype=np.uint8)
    feats = pq.extrair_features_cor(cinza)
    assert len(feats) == 18
    assert feats["R_media"] == pytest.approx(feats["G_media"])
    assert feats["G_media"] == pytest.approx(feats["B_media"])


def test_extrair_features_textura_sem_scikit_image_devolve_vazio(pq, monkeypatch, capsys):
    """scikit-image e' dependencia OPCIONAL (extra [imagem]) -- sem ela,
    extrair_features_textura devolve dict vazio com aviso, nunca lanca
    ImportError pro chamador. Forca o ImportError via monkeypatch (nao
    depende de scikit-image estar ou nao instalado no ambiente de teste)."""
    import builtins
    _import_real = builtins.__import__

    def _import_bloqueado(nome, *args, **kwargs):
        if nome.startswith("skimage"):
            raise ImportError(f"simulado: {nome} indisponivel")
        return _import_real(nome, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_bloqueado)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    feats = pq.extrair_features_textura(img)
    assert feats == {}
    assert "scikit-image" in capsys.readouterr().out


# ── Carregamento fim-a-fim (arquivos reais em disco) ──────────────────────

def _salvar_imagem_solida(caminho, rgb, tamanho=30, ruido=3, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)
    for c in range(3):
        arr[..., c] = np.clip(
            rgb[c] + rng.integers(-ruido, ruido + 1, size=(tamanho, tamanho)),
            0, 255)
    Image.fromarray(arr, "RGB").save(caminho)


def test_detectar_subpastas_imagem_raiz_inexistente(pq):
    from guaraci.dados_imagem import _detectar_subpastas_imagem
    assert _detectar_subpastas_imagem("/caminho/que/nao/existe") == []


def test_carregar_imagens_estrutura_multi_pasta(pq, tmp_path):
    """carregar_imagens: 1 subpasta por classe, sem duplicar arquivos (guarda
    contra o bug de busca case-insensitive de extensão em Windows/macOS)."""
    raiz = tmp_path / "dados_img"
    (raiz / "Puro").mkdir(parents=True)
    (raiz / "Adulterado").mkdir(parents=True)

    for i in range(4):
        _salvar_imagem_solida(str(raiz / "Puro" / f"p{i}.png"),
                               (200, 180, 50), seed=i)
    for i in range(3):
        _salvar_imagem_solida(str(raiz / "Adulterado" / f"a{i}.png"),
                               (150, 60, 40), seed=i + 10)

    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.carregar_imagens(str(raiz))

    assert X.shape == (7, 18)  # 4 + 3 imagens, NUNCA duplicadas
    assert set(rotulos) == {"Puro", "Adulterado"}
    assert conc is None and mae_id is None  # prototipo generico: sem metadado
    assert meta_df is not None and len(meta_df) == 7
    # a classe "Puro" (mais amarela) deve ter R_media maior que "Adulterado"
    assert X[rotulos == "Puro", 0].mean() > X[rotulos == "Adulterado", 0].mean()


def test_carregar_imagens_pasta_inexistente_levanta_filenotfound(pq, tmp_path):
    with pytest.raises(FileNotFoundError, match="Pasta nao existe"):
        pq.carregar_imagens(str(tmp_path / "nao_existe"))


def test_carregar_imagens_pasta_vazia_levanta_filenotfound(pq, tmp_path):
    (tmp_path / "vazia").mkdir()
    with pytest.raises(FileNotFoundError, match="nao contem imagens"):
        pq.carregar_imagens(str(tmp_path / "vazia"))


def test_carregar_imagens_modo_flat_usa_nome_do_arquivo_como_classe(pq, tmp_path):
    """Sem subpastas (arquivos soltos na raiz), cada imagem vira sua PROPRIA
    classe (nome do arquivo sem extensao) -- fallback documentado."""
    raiz = tmp_path / "flat"
    raiz.mkdir()
    _salvar_imagem_solida(str(raiz / "amostra1.png"), (100, 150, 200))
    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.carregar_imagens(str(raiz))
    assert X.shape[0] == 1
    assert rotulos[0] == "amostra1"


def test_carregar_imagens_arquivo_corrompido_e_pulado_com_aviso(pq, tmp_path, capsys):
    """Uma imagem corrompida no meio do lote nao derruba o carregamento --
    e' contada como falha, avisada, e as demais seguem normalmente."""
    raiz = tmp_path / "com_corrompida"
    (raiz / "Classe").mkdir(parents=True)
    _salvar_imagem_solida(str(raiz / "Classe" / "boa1.png"), (100, 150, 200))
    (raiz / "Classe" / "corrompida.png").write_bytes(b"nao e uma imagem valida")

    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.carregar_imagens(str(raiz))
    assert X.shape[0] == 1   # so' a imagem boa foi carregada
    assert "ERROR" in capsys.readouterr().out


def test_carregar_imagens_todas_corrompidas_levanta_valueerror(pq, tmp_path):
    """Se NENHUMA imagem do lote carrega com sucesso, levanta ValueError
    explicito (em vez de devolver um X vazio silenciosamente)."""
    raiz = tmp_path / "todas_corrompidas"
    (raiz / "Classe").mkdir(parents=True)
    (raiz / "Classe" / "ruim.png").write_bytes(b"lixo binario, nao e imagem")
    with pytest.raises(ValueError, match="Nenhuma imagem valida"):
        pq.carregar_imagens(str(raiz))


def test_carregar_dados_modo_imagem_delega_corretamente(pq, tmp_path):
    """carregar_dados(cfg) com modo='imagem' delega para carregar_imagens."""
    raiz = tmp_path / "dados_img"
    (raiz / "ClasseA").mkdir(parents=True)
    _salvar_imagem_solida(str(raiz / "ClasseA" / "img1.png"), (100, 150, 200))

    cfg = pq.Config(modo="imagem", pasta_entrada=str(raiz))
    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.carregar_dados(cfg)
    assert X.shape[0] == 1
    assert rotulos[0] == "ClasseA"


def test_validar_pasta_dados_modo_imagem(pq, tmp_path):
    """_validar_pasta_dados reconhece o modo 'imagem' (pasta vazia -> False,
    pasta com imagens -> True)."""
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    cfg_vazio = pq.Config(modo="imagem", pasta_entrada=str(vazio))
    ok, _msg = pq._validar_pasta_dados(cfg_vazio)
    assert ok is False

    com_imagem = tmp_path / "com_imagem"
    com_imagem.mkdir()
    _salvar_imagem_solida(str(com_imagem / "x.png"), (100, 100, 100))
    cfg_ok = pq.Config(modo="imagem", pasta_entrada=str(com_imagem))
    ok2, _msg2 = pq._validar_pasta_dados(cfg_ok)
    assert ok2 is True


@pytest.mark.slow
def test_executar_pipeline_completo_modo_imagem(pq, tmp_path):
    """Integração completa: executar() com modo='imagem' de ponta a ponta.

    Cuidado necessário: `carregar_imagens` devolve um eixo de variaveis
    simbolico (np.arange(n_features), NÃO um numero de onda real) — por isso
    wn_min/wn_max (que por padrao truncam a faixa espectral em cm-1) precisam
    ser ajustados pra cobrir esse intervalo pequeno, senao TODAS as variaveis
    seriam descartadas pelo filtro espectral."""
    raiz = tmp_path / "dados_img"
    for cls, rgb in [("Esp_A", (210, 190, 40)), ("Esp_B", (60, 130, 200)),
                     ("Esp_C", (180, 50, 90))]:
        (raiz / cls).mkdir(parents=True)
        for i in range(10):
            _salvar_imagem_solida(str(raiz / cls / f"{cls}_{i}.png"), rgb,
                                   seed=hash((cls, i)) % 1000)

    cfg = pq.Config(
        modo="imagem", pasta_entrada=str(raiz),
        pasta_saida_raiz=str(tmp_path / "saida"),
        wn_min=-1.0, wn_max=100.0,  # cobre o eixo simbolico 0..17
        # "autoscaling" (nao "msc_sg_mc"): MSC/Savitzky-Golay pressupoem um
        # sinal espectral continuo (eixo de comprimento de onda) — nao fazem
        # sentido cientifico p/ um vetor curto de estatisticas de cor
        # discretas e heterogeneas (H em [0,1], Lab em dezenas/centenas).
        preprocessamento_padrao="autoscaling",
        n_splits_cv=2, n_repeats_cv=1, max_lvs=3,
        n_permutacoes=3, n_permutacoes_wold=3,
        n_bootstrap_vip=2, n_bootstrap_bca=10, n_monte_carlo=2,
        executar_ddsimca=False, executar_opls=False, executar_etapa4=False,
        executar_wold=False, comparar_pipelines=False,
        executar_cv_anova=False, executar_benchmark=False,
        executar_monte_carlo=False, executar_shap=False,
    )
    pq.executar(cfg)

    from pathlib import Path
    runs = achar_pastas_run(tmp_path / "saida")
    assert runs, "executar() nao criou pasta de saida p/ modo imagem"
    resumo = Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt"
    assert resumo.exists(), "resumo_modelo.txt nao gerado p/ modo imagem"
