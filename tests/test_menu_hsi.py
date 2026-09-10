"""Testes de _menu_hsi (Passo 102): reachability real do modo `hsi` via
CLI -- "nenhuma funcao implementada sem que o usuario consiga usar",
mesmo requisito ja aplicado ao resto desta auditoria."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import guaraci.guaraci as guaraci_mod


def test_menu_hsi_cancelar_com_0_nao_lanca_excecao(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "0")
    guaraci_mod._menu_hsi(guaraci_mod.Config())


# ── Passo 103 (INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md): texto/UI da tela ──

def _renderizar_tela_hsi(monkeypatch, cfg, lang="PT"):
    """Renderiza a tela HSI de verdade (cabecalho + intro) e devolve o
    texto sem codigos ANSI -- mesmo padrao de `_render` em
    test_guaraci_cli.py, adaptado para uma funcao que pede input()
    (cancela com '0' logo apos o cabecalho renderizar)."""
    import contextlib
    import io
    import re as _re

    monkeypatch.setattr("builtins.input", lambda *a, **k: "0")
    lang_antes = guaraci_mod._STATE["lang"]
    guaraci_mod._STATE["lang"] = lang
    buf = io.StringIO()
    file_antes = guaraci_mod.console._file
    try:
        guaraci_mod.console.file = buf
        with contextlib.redirect_stdout(io.StringIO()):
            guaraci_mod._menu_hsi(cfg)
    finally:
        guaraci_mod.console._file = file_antes
        guaraci_mod._STATE["lang"] = lang_antes
    return _re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


def test_menu_hsi_nao_mostra_frase_solta_de_prototipo(monkeypatch):
    """Achado do Passo 103: "Prototipo 'minimo viavel'" era uma frase
    solta e generica -- a tela agora descreve a limitacao real e
    especifica (fonte unica em _AVISO_MATURIDADE_HSI_PT/EN)."""
    saida = _renderizar_tela_hsi(monkeypatch, guaraci_mod.Config(), lang="PT")
    assert "minimo viavel" not in saida.lower()
    assert "prototipo" not in saida.lower()
    assert "Kaki" in saida and "VIS" in saida  # limitacao especifica citada


def test_menu_hsi_cabecalho_nao_mostra_tecnica_ft_nir(monkeypatch):
    """Achado do Passo 103: o cabecalho fixo herdava 'Tecnica: FT-NIR' do
    template generico -- FT-NIR nao faz sentido nenhum para um cubo
    hiperespectral VIS/NIR de imagem. Agora mostra 'Tecnica: HSI'."""
    saida = _renderizar_tela_hsi(monkeypatch, guaraci_mod.Config(), lang="PT")
    assert "FT-NIR" not in saida
    assert "Tecnica: HSI" in saida


def test_menu_hsi_cabecalho_ingles_tambem_correto(monkeypatch):
    saida = _renderizar_tela_hsi(monkeypatch, guaraci_mod.Config(), lang="EN")
    assert "FT-NIR" not in saida
    assert "Technique: HSI" in saida


def test_rotulo_tecnica_efetivo_modo_imagem_tambem_corrigido():
    """Mesma classe de bug (Passo 103 pede varredura das demais telas):
    mode='imagem' tambem herdava 'FT-NIR' -- corrigido pela MESMA fonte
    unica (_rotulo_tecnica_efetivo), nao um patch so' para HSI."""
    cfg_img = guaraci_mod.Config(mode="imagem")
    rotulo = guaraci_mod._rotulo_tecnica_efetivo(cfg_img)
    assert rotulo != "FT-NIR"
    assert "colorimetria" in rotulo.lower() or "colorimetry" in rotulo.lower()


def test_rotulo_tecnica_efetivo_modo_dx_preserva_tecnica_selecionada():
    """Contra-prova: modes espectrais de verdade (dx/csv/sintetico) devem
    CONTINUAR mostrando a tecnica escolhida em [8] -- a correcao e' so'
    para hsi/imagem, nao uma regressao no caso que ja funcionava."""
    cfg_dx = guaraci_mod.Config(mode="dx")
    assert guaraci_mod._rotulo_tecnica_efetivo(cfg_dx) == \
        guaraci_mod._TECNICA_SELECIONADA.get("nome", "FT-NIR")


def test_menu_hsi_pasta_invalida_reporta_erro_e_nao_lanca(monkeypatch, tmp_path):
    pasta_sem_manifest = str(tmp_path)
    respostas = iter([pasta_sem_manifest, ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    guaraci_mod._menu_hsi(guaraci_mod.Config())


def _pasta_deephs_kaki():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "deephs_kaki_vis"
    return pasta if (pasta / "manifest.json").is_file() else None


requer_deephs_kaki = pytest.mark.skipif(
    _pasta_deephs_kaki() is None,
    reason=("dataset publico DeepHS Fruit/Kaki/VIS ausente. Baixe com "
            "'python scripts/download_datasets/baixar_deephs_kaki.py' e "
            "aponte GUARACI_DATASETS_DIR."))


def _mascara_objeto_suave(n_lin, n_col, raio_frac=0.35, largura_borda=4.0):
    """Mesma logica de test_hsi_pipeline.py -- transicao suave evita que
    o estimador de ruido (Laplaciano, hsi_quality.py) confunda um degrau
    binario de mascara com ruido de alta frequencia."""
    import numpy as np
    yy, xx = np.ogrid[:n_lin, :n_col]
    cy, cx = (n_lin - 1) / 2.0, (n_col - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    raio = raio_frac * min(n_lin, n_col)
    return np.clip((raio - dist) / largura_borda + 0.5, 0.0, 1.0)


def _montar_pasta_hsi_generica_cli(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    raiz = tmp_path / "meu_dataset_hsi"
    n_lin, n_col, n_bandas = 64, 64, 6
    alpha = _mascara_objeto_suave(n_lin, n_col)[..., None]
    for classe, nivel_objeto in (("madura", 0.85), ("verde", 0.45)):
        for a in range(3):
            pasta_amostra = raiz / classe / f"amostra{a}"
            pasta_amostra.mkdir(parents=True)
            fundo = rng.normal(loc=0.05, scale=0.01, size=(n_lin, n_col, n_bandas))
            objeto = rng.normal(loc=nivel_objeto, scale=0.01,
                                size=(n_lin, n_col, n_bandas))
            cubo = alpha * objeto + (1.0 - alpha) * fundo
            (pasta_amostra / "vista0.bin").write_bytes(cubo.astype("<f4").tobytes())
            (pasta_amostra / "vista0.hdr").write_text(
                f"ENVI\nsamples = {n_col}\nlines = {n_lin}\nbands = {n_bandas}\n"
                f"header offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
                f"interleave = bip\nbyte order = 0\n", encoding="utf-8")
    return raiz


def test_menu_hsi_roda_pipeline_generico_via_cli_sem_dataset_publico(
        monkeypatch, tmp_path):
    """Passo 111: a prova de reachability p/ dado do PROPRIO usuario --
    nao depende de GUARACI_DATASETS_DIR/dataset publico nenhum, ao
    contrario de `test_menu_hsi_roda_pipeline_completo_via_cli` abaixo."""
    pasta_dataset = str(_montar_pasta_hsi_generica_cli(tmp_path))
    respostas = iter([pasta_dataset, ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    cfg = guaraci_mod.Config(output_root_folder=str(tmp_path / "saida"))
    guaraci_mod._menu_hsi(cfg)

    assert cfg.mode == "hsi"
    assert cfg.hsi_dataset_folder == pasta_dataset
    assert cfg.output_folder != ""
    caminho_figura = (Path(cfg.output_folder) / "Graficos" / "hsi" /
                      "hsi_mapa_classificacao_amostra.png")
    assert caminho_figura.is_file()


@requer_deephs_kaki
def test_menu_hsi_roda_pipeline_completo_via_cli(monkeypatch, tmp_path):
    """A prova final de reachability: o usuario aciona o pipeline HSI
    inteiro digitando so' o caminho da pasta na tela do menu -- sem
    chamar hsi_pipeline diretamente."""
    pasta_dataset = str(_pasta_deephs_kaki())
    respostas = iter([pasta_dataset, ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    cfg = guaraci_mod.Config(output_root_folder=str(tmp_path))
    guaraci_mod._menu_hsi(cfg)

    assert cfg.mode == "hsi"
    assert cfg.hsi_dataset_folder == pasta_dataset
    assert cfg.output_folder != ""
    caminho_figura = (Path(cfg.output_folder) / "Graficos" / "hsi" /
                      "hsi_mapa_classificacao_amostra.png")
    assert caminho_figura.is_file()
