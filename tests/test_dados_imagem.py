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
    feats = pq.extract_color_features(img)
    assert len(feats) == 18
    assert all(np.isfinite(v) for v in feats.values())


def test_extrair_features_cor_distingue_cores_diferentes(pq):
    """Duas imagens de cores bem diferentes devem produzir R_media bem
    diferente — checagem de sanidade (não é so ruído numérico)."""
    amarelo = np.zeros((20, 20, 3), dtype=np.uint8)
    amarelo[..., 0] = 220; amarelo[..., 1] = 200; amarelo[..., 2] = 40
    azul = np.zeros((20, 20, 3), dtype=np.uint8)
    azul[..., 0] = 30; azul[..., 1] = 40; azul[..., 2] = 200

    f_amarelo = pq.extract_color_features(amarelo)
    f_azul = pq.extract_color_features(azul)
    assert f_amarelo["R_media"] > f_azul["R_media"]
    assert f_azul["B_media"] > f_amarelo["B_media"]


def test_extrair_features_cor_aceita_imagem_2d_tons_de_cinza(pq):
    """Imagem 2D (H, W) sem canal de cor -- e' replicada em 3 canais R=G=B
    automaticamente (linha de compatibilidade pouco exercitada)."""
    cinza = np.full((20, 20), 128, dtype=np.uint8)
    feats = pq.extract_color_features(cinza)
    assert len(feats) == 18
    assert feats["R_media"] == pytest.approx(feats["G_media"])
    assert feats["G_media"] == pytest.approx(feats["B_media"])


def test_extrair_features_textura_sem_scikit_image_devolve_vazio(pq, monkeypatch, capsys):
    """scikit-image e' dependencia OPCIONAL (extra [imagem]) -- sem ela,
    extract_texture_features devolve dict vazio com aviso, nunca lanca
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
    feats = pq.extract_texture_features(img)
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
    """load_images: 1 subpasta por classe, sem duplicar arquivos (guarda
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

    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.load_images(str(raiz))

    assert X.shape == (7, 18)  # 4 + 3 imagens, NUNCA duplicadas
    assert set(rotulos) == {"Puro", "Adulterado"}
    assert conc is None and mae_id is None  # prototipo generico: sem metadado
    assert meta_df is not None and len(meta_df) == 7
    # a classe "Puro" (mais amarela) deve ter R_media maior que "Adulterado"
    assert X[rotulos == "Puro", 0].mean() > X[rotulos == "Adulterado", 0].mean()


def test_carregar_imagens_pasta_inexistente_levanta_filenotfound(pq, tmp_path):
    with pytest.raises(FileNotFoundError, match="Pasta nao existe"):
        pq.load_images(str(tmp_path / "nao_existe"))


def test_carregar_imagens_pasta_vazia_levanta_filenotfound(pq, tmp_path):
    (tmp_path / "vazia").mkdir()
    with pytest.raises(FileNotFoundError, match="nao contem imagens"):
        pq.load_images(str(tmp_path / "vazia"))


def test_carregar_imagens_modo_flat_usa_nome_do_arquivo_como_classe(pq, tmp_path):
    """Sem subpastas (arquivos soltos na raiz), cada imagem vira sua PROPRIA
    classe (nome do arquivo sem extensao) -- fallback documentado."""
    raiz = tmp_path / "flat"
    raiz.mkdir()
    _salvar_imagem_solida(str(raiz / "amostra1.png"), (100, 150, 200))
    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.load_images(str(raiz))
    assert X.shape[0] == 1
    assert rotulos[0] == "amostra1"


def test_carregar_imagens_arquivo_corrompido_e_pulado_com_aviso(pq, tmp_path, capsys):
    """Uma imagem corrompida no meio do lote nao derruba o carregamento --
    e' contada como falha, avisada, e as demais seguem normalmente."""
    raiz = tmp_path / "com_corrompida"
    (raiz / "Classe").mkdir(parents=True)
    _salvar_imagem_solida(str(raiz / "Classe" / "boa1.png"), (100, 150, 200))
    (raiz / "Classe" / "corrompida.png").write_bytes(b"nao e uma imagem valida")

    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.load_images(str(raiz))
    assert X.shape[0] == 1   # so' a imagem boa foi carregada
    assert "ERROR" in capsys.readouterr().out


def test_carregar_imagens_todas_corrompidas_levanta_valueerror(pq, tmp_path):
    """Se NENHUMA imagem do lote carrega com sucesso, levanta ValueError
    explicito (em vez de devolver um X vazio silenciosamente)."""
    raiz = tmp_path / "todas_corrompidas"
    (raiz / "Classe").mkdir(parents=True)
    (raiz / "Classe" / "ruim.png").write_bytes(b"lixo binario, nao e imagem")
    with pytest.raises(ValueError, match="Nenhuma imagem valida"):
        pq.load_images(str(raiz))


# ── Bloco 8 (2026-08-25): 3 niveis de garantia de agrupamento ──────────────

def test_carregar_imagens_nivel_none_sem_estrutura_nem_csv(pq, tmp_path):
    """Sem subpasta por amostra nem CSV: nivel 'none', mae_id=None (mesmo
    comportamento de antes do Bloco 8) -- mas agora com o nivel declarado
    explicitamente em metadados_df.attrs, nao so' silenciosamente None."""
    raiz = tmp_path / "sem_estrutura"
    (raiz / "Classe").mkdir(parents=True)
    for i in range(3):
        _salvar_imagem_solida(str(raiz / "Classe" / f"f{i}.png"),
                               (100, 150, 200), seed=i)
    _, _, _, conc, mae_id, meta_df = pq.load_images(str(raiz))
    assert conc is None and mae_id is None
    assert meta_df.attrs["grouping_guarantee"] == "none"


def test_carregar_imagens_nivel_high_subpasta_por_amostra(pq, tmp_path):
    """Classe/Amostra/*.jpg (2 niveis): nivel 'high', mae_id real -- cada
    amostra fisica (subpasta) vira 1 grupo, replicas dentro dela
    compartilham o grupo."""
    raiz = tmp_path / "por_amostra"
    for classe, amostras in (("Puro", 2), ("Adulterado", 2)):
        for a in range(amostras):
            pasta_amostra = raiz / classe / f"amostra{a}"
            pasta_amostra.mkdir(parents=True)
            for r in range(3):  # 3 replicas (fotos) da mesma amostra
                _salvar_imagem_solida(
                    str(pasta_amostra / f"foto{r}.png"),
                    (200, 180, 50) if classe == "Puro" else (150, 60, 40),
                    seed=a * 10 + r)

    _, X, rotulos, conc, mae_id, meta_df = pq.load_images(str(raiz))
    assert X.shape[0] == 12  # 2 classes x 2 amostras x 3 replicas
    assert conc is None
    assert mae_id is not None
    assert meta_df.attrs["grouping_guarantee"] == "high"
    # 4 grupos no total (2 amostras x 2 classes), 3 replicas cada
    grupos, contagens = np.unique(mae_id, return_counts=True)
    assert len(grupos) == 4
    assert set(contagens) == {3}
    # grupos de classes diferentes nunca colidem (qualificados por classe)
    assert len({g for g in grupos if g.startswith("Puro/")}) == 2
    assert len({g for g in grupos if g.startswith("Adulterado/")}) == 2


def test_carregar_imagens_nivel_medium_csv_associacao(pq, tmp_path):
    """Estrutura flat (sem subpasta por amostra) + `amostras.csv` cobrindo
    TODO arquivo: nivel 'medium', mae_id vem do CSV."""
    raiz = tmp_path / "com_csv"
    (raiz / "Classe").mkdir(parents=True)
    for i in range(4):
        _salvar_imagem_solida(str(raiz / "Classe" / f"f{i}.png"),
                               (100, 150, 200), seed=i)
    # f0,f1 sao a mesma amostra fisica (2 fotos); f2,f3 sao outra amostra.
    (raiz / "amostras.csv").write_text(
        "arquivo,id_amostra\n"
        "Classe/f0.png,S1\nClasse/f1.png,S1\n"
        "Classe/f2.png,S2\nClasse/f3.png,S2\n",
        encoding="utf-8",
    )
    _, X, rotulos, conc, mae_id, meta_df = pq.load_images(str(raiz))
    assert X.shape[0] == 4
    assert meta_df.attrs["grouping_guarantee"] == "medium"
    grupos, contagens = np.unique(mae_id, return_counts=True)
    assert sorted(grupos) == ["S1", "S2"]
    assert set(contagens) == {2}


def test_carregar_imagens_nivel_medium_csv_incompleto_levanta_valueerror(
        pq, tmp_path):
    """CSV presente mas nao cobre TODA imagem do dataset: erro explicito
    listando os arquivos faltantes -- nunca processamento parcial em
    silencio."""
    raiz = tmp_path / "csv_incompleto"
    (raiz / "Classe").mkdir(parents=True)
    for i in range(3):
        _salvar_imagem_solida(str(raiz / "Classe" / f"f{i}.png"),
                               (100, 150, 200), seed=i)
    (raiz / "amostras.csv").write_text(
        "arquivo,id_amostra\nClasse/f0.png,S1\nClasse/f1.png,S1\n",
        encoding="utf-8",
    )  # f2.png ausente do CSV de proposito
    with pytest.raises(ValueError, match="f2.png"):
        pq.load_images(str(raiz))


def test_carregar_imagens_nivel_high_tem_prioridade_sobre_medium(pq, tmp_path):
    """Quando subpasta-por-amostra E CSV estao presentes ao mesmo tempo,
    'high' vence (mais confiavel) -- o CSV e' ignorado, mesmo declarando
    grupos diferentes dos da estrutura de pastas."""
    raiz = tmp_path / "ambos_niveis"
    for a in range(2):
        pasta_amostra = raiz / "Classe" / f"amostra{a}"
        pasta_amostra.mkdir(parents=True)
        for r in range(2):
            _salvar_imagem_solida(str(pasta_amostra / f"foto{r}.png"),
                                   (100, 150, 200), seed=a * 10 + r)
    # CSV deliberadamente CONTRADIZ a estrutura de pastas (grupos por foto,
    # nao por amostra) -- se 'high' nao tiver prioridade, o teste pega isso.
    (raiz / "amostras.csv").write_text(
        "arquivo,id_amostra\n"
        "Classe/amostra0/foto0.png,X1\nClasse/amostra0/foto1.png,X2\n"
        "Classe/amostra1/foto0.png,X3\nClasse/amostra1/foto1.png,X4\n",
        encoding="utf-8",
    )
    _, _, _, _, mae_id, meta_df = pq.load_images(str(raiz))
    assert meta_df.attrs["grouping_guarantee"] == "high"
    grupos = set(mae_id)
    assert grupos == {"Classe/amostra0", "Classe/amostra1"}  # nao {X1..X4}


def test_carregar_dados_modo_imagem_delega_corretamente(pq, tmp_path):
    """load_data(cfg) com mode='imagem' delega para load_images."""
    raiz = tmp_path / "dados_img"
    (raiz / "ClasseA").mkdir(parents=True)
    _salvar_imagem_solida(str(raiz / "ClasseA" / "img1.png"), (100, 150, 200))

    cfg = pq.Config(mode="imagem", input_folder=str(raiz))
    wavenumbers, X, rotulos, conc, mae_id, meta_df = pq.load_data(cfg)
    assert X.shape[0] == 1
    assert rotulos[0] == "ClasseA"
    # Bloco 8: load_data (via _leitor_imagem) copia o nivel detectado para
    # cfg.grouping_guarantee -- e' o que pipeline.executar() le depois p/
    # declarar a limitacao no log/resumo/manifesto. Config() comeca em
    # "high" (default cobre dx/sintetico); 1 classe sem estrutura/CSV deve
    # ter rebaixado para "none".
    assert cfg.grouping_guarantee == "none"


def test_executar_pipeline_modo_imagem_nivel_high_ativa_group_aware(
        pq, tmp_path):
    """Bloco 8c: quando a pasta de imagens tem estrutura por amostra fisica
    (nivel 'high'), o pipeline entra no MESMO caminho group-aware que
    dx/sintetico usa (`if mae_id is not None`, pipeline.py) -- nao existe
    um caminho de validacao separado p/ imagem. Confirma isso end-to-end,
    lendo `Group-aware (mae_id)` do resumo real (nao so' checando
    mae_id em memoria, que os testes de load_images ja cobrem)."""
    raiz = tmp_path / "dados_img_agrupado"
    for cls, rgb in [("Esp_A", (210, 190, 40)), ("Esp_B", (60, 130, 200))]:
        for a in range(5):  # 5 amostras fisicas por classe
            pasta_amostra = raiz / cls / f"amostra{a}"
            pasta_amostra.mkdir(parents=True)
            for r in range(3):  # 3 replicas (fotos) por amostra
                _salvar_imagem_solida(str(pasta_amostra / f"foto{r}.png"),
                                       rgb, seed=hash((cls, a, r)) % 1000)

    cfg = pq.Config(
        mode="imagem", input_folder=str(raiz),
        output_root_folder=str(tmp_path / "saida"),
        wn_min=-1.0, wn_max=100.0, default_preprocessing="autoscaling",
        n_splits_cv=2, n_repeats_cv=1, max_lvs=3,
        n_permutations=3, n_permutations_wold=3,
        n_bootstrap_vip=2, n_bootstrap_bca=10, n_monte_carlo=2,
        run_ddsimca=False, run_opls=False, executar_etapa4=False,
        run_wold=False, comparar_pipelines=False,
        run_cv_anova=False, run_benchmark=False,
        run_monte_carlo=False, run_shap=False,
    )
    pq.executar(cfg)

    from pathlib import Path
    runs = achar_pastas_run(tmp_path / "saida")
    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS
              / "resumo_modelo.txt").read_text(encoding="utf-8")
    assert "Grouping guarantee" in resumo
    assert "high" in resumo.lower()
    assert "Group-aware (mae_id)" in resumo
    assert "sim" in resumo.split("Group-aware (mae_id)")[1][:20].lower()

    model_card = (Path(runs[0]) / pq.NOME_RELATORIOS
                  / "model_card.md").read_text(encoding="utf-8")
    assert "GROUPING GUARANTEE" not in model_card  # so' carimba p/ "none"


def test_validar_pasta_dados_modo_imagem(pq, tmp_path):
    """_validar_pasta_dados reconhece o mode 'imagem' (pasta vazia -> False,
    pasta com imagens -> True)."""
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    cfg_vazio = pq.Config(mode="imagem", input_folder=str(vazio))
    ok, _msg = pq._validar_pasta_dados(cfg_vazio)
    assert ok is False

    com_imagem = tmp_path / "com_imagem"
    com_imagem.mkdir()
    _salvar_imagem_solida(str(com_imagem / "x.png"), (100, 100, 100))
    cfg_ok = pq.Config(mode="imagem", input_folder=str(com_imagem))
    ok2, _msg2 = pq._validar_pasta_dados(cfg_ok)
    assert ok2 is True


@pytest.mark.slow
def test_executar_pipeline_completo_modo_imagem(pq, tmp_path):
    """Integração completa: executar() com mode='imagem' de ponta a ponta.

    Cuidado necessário: `load_images` devolve um eixo de variaveis
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
        mode="imagem", input_folder=str(raiz),
        output_root_folder=str(tmp_path / "saida"),
        wn_min=-1.0, wn_max=100.0,  # cobre o eixo simbolico 0..17
        # "autoscaling" (nao "msc_sg_mc"): MSC/Savitzky-Golay pressupoem um
        # sinal espectral continuo (eixo de comprimento de onda) — nao fazem
        # sentido cientifico p/ um vetor curto de estatisticas de cor
        # discretas e heterogeneas (H em [0,1], Lab em dezenas/centenas).
        default_preprocessing="autoscaling",
        n_splits_cv=2, n_repeats_cv=1, max_lvs=3,
        n_permutations=3, n_permutations_wold=3,
        n_bootstrap_vip=2, n_bootstrap_bca=10, n_monte_carlo=2,
        run_ddsimca=False, run_opls=False, executar_etapa4=False,
        run_wold=False, comparar_pipelines=False,
        run_cv_anova=False, run_benchmark=False,
        run_monte_carlo=False, run_shap=False,
    )
    pq.executar(cfg)

    from pathlib import Path
    runs = achar_pastas_run(tmp_path / "saida")
    assert runs, "executar() nao criou pasta de saida p/ mode imagem"
    resumo = Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt"
    assert resumo.exists(), "resumo_modelo.txt nao gerado p/ mode imagem"

    # Bloco 8c: esta pasta de dados e' flat (sem subpasta por amostra nem
    # CSV) -- nivel "none" esperado. A limitacao precisa aparecer nas 3
    # saidas (nao so' em docstring/comentario interno).
    texto_resumo = resumo.read_text(encoding="utf-8")
    assert "Grouping guarantee" in texto_resumo
    assert "none" in texto_resumo.lower()

    model_card = Path(runs[0]) / pq.NOME_RELATORIOS / "model_card.md"
    texto_card = model_card.read_text(encoding="utf-8")
    assert "GROUPING GUARANTEE" in texto_card
    assert "NONE" in texto_card

    import json
    manifesto = next((Path(runs[0]) / pq.NOME_MODELOS).glob("*.manifest.json"))
    dados_manifesto = json.loads(manifesto.read_text(encoding="utf-8"))
    assert dados_manifesto.get("grouping_guarantee") == "none"
