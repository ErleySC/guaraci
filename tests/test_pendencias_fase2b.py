"""Pendencias pequenas que a fase 2 deixou abertas (sem mudar nenhum padrao):

- `checar --resultado PASTA_HSI` quebrava por falta do resumo_modelo.txt (e o erro de pasta inexistente era um traceback);
- faixas_excluir / ims_rip_janela fora do eixo saiam como "Erro sem codigo conhecido" (agora GUA-E015);
- `exemplo hsi` nao gravava snv_mc nem a faixa aberta (a execucao avisava que o preset nao se aplicava);
- a previa da web ignorava faixas_excluir;
- os extras do HSI liam os cubos sem a calibracao e sem a checagem de eixo do modelo principal.
"""
import json

import numpy as np
import pytest

from guaraci import assistente_checar as C
from guaraci import assistente_erros as E


# ---------------------------------------------------------------- checar --resultado PASTA_HSI
def _pasta_hsi(tmp_path, sub="Relatorios"):
    rel = tmp_path / "exec" / sub
    rel.mkdir(parents=True)
    (rel / "hsi_relatorio.json").write_text(json.dumps({
        "n_gravacoes_aceitas": 6, "n_gravacoes_total": 6, "n_components": 3, "grouping_guarantee": "objeto",
        "validacao": {"tipo": "cv_objetos", "n_dobras": 5, "n_objetos_total": 6, "n_objetos_avaliados": 6,
                      "confianca_por_objeto_fora_da_amostra": True, "acuracia_objetos": 1.0},
    }), encoding="utf-8")
    return tmp_path / "exec"


@pytest.mark.parametrize("sub", ["Relatorios", "Reports"])
def test_ler_resultado_aceita_pasta_hsi_sem_resumo_modelo(tmp_path, sub):
    r = C.ler_resultado(str(_pasta_hsi(tmp_path, sub)))
    assert r["hsi_tipo"] == "cv_objetos" and r["hsi_n_objetos_avaliados"] == 6


def test_checar_resultado_hsi_imprime_a_ficha_do_hsi(tmp_path, capsys):
    pasta = _pasta_hsi(tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"modo_entrada: hsi\nhsi_pasta_dataset: {tmp_path}\npasta_saida: {pasta}\n", encoding="utf-8")
    assert C.main([str(cfg), "--resultado", str(pasta), "--ficha"]) in (0, 1)
    saida = capsys.readouterr().out
    assert "Ficha anti-vazamento" in saida and "Separacao por objeto fisico" in saida
    assert "MSC" not in saida.split("Pre-processamento sem parametros")[0]


def test_checar_resultado_inexistente_nao_e_traceback(tmp_path, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("modo_entrada: csv\n", encoding="utf-8")
    assert C.main([str(cfg), "--resultado", str(tmp_path / "nada")]) == 2
    assert "Traceback" not in capsys.readouterr().err


# ---------------------------------------------------------------- GUA-E015
def _mensagens_reais():
    from guaraci.preprocessamento import avaliar_faixas_excluir, normalizar_espectros
    eixo = np.linspace(0, 10, 50)
    out = []
    for fn in (lambda: avaliar_faixas_excluir(eixo, eixo, ["20-30"]),
               lambda: avaliar_faixas_excluir(eixo, eixo, ["-1-11"]),
               lambda: normalizar_espectros(np.ones((3, 50)), eixo, "rip", "20-30")):
        with pytest.raises(ValueError) as ei:
            fn()
        out.append(ei.value)
    return out


def test_faixa_fora_do_eixo_ganha_codigo():
    for exc in _mensagens_reais():
        assert E.codigos(exc) == ["GUA-E015"], str(exc)
        assert "sem codigo conhecido" not in E.diagnosticar_excecao(exc, "PT")


def test_gua_e007_continua_so_para_faixa_de_corte():
    assert E.codigos("ValueError: Found array with 0 feature(s)") == ["GUA-E007"]


# ---------------------------------------------------------------- exemplo hsi
def test_exemplo_hsi_grava_snv_mc_e_faixa_aberta(tmp_path):
    from guaraci import exemplos
    exemplos.gerar("hsi", str(tmp_path))
    from guaraci.config_io import load_config
    cfg = load_config(str(tmp_path / "config.yaml"))
    assert cfg.default_preprocessing == "snv_mc"
    assert cfg.wn_min == 0 and cfg.wn_max >= 999999


# ---------------------------------------------------------------- previa da web
def test_previa_remove_as_faixas_excluidas():
    from guaraci.spectra_preview import remover_faixas_excluidas
    wn = np.linspace(0, 10, 101)
    X = np.tile(wn, (3, 1))
    wn2, X2, n = remover_faixas_excluidas(wn, X, ["4.5-5.0", "-0.5-0.5"])
    esperado = int((((wn >= 4.5) & (wn <= 5.0)) | ((wn >= -0.5) & (wn <= 0.5))).sum())
    assert n == esperado and wn2.size == wn.size - esperado
    assert X2.shape == (3, wn2.size) and not ((wn2 >= 4.5) & (wn2 <= 5.0)).any()


@pytest.mark.parametrize("faixas", [None, [], ["20-30"], ["lixo"], ["-1-11"]])
def test_previa_nunca_levanta_e_nao_apaga_tudo(faixas):
    from guaraci.spectra_preview import remover_faixas_excluidas
    wn = np.linspace(0, 10, 101)
    X = np.ones((2, 101))
    wn2, X2, n = remover_faixas_excluidas(wn, X, faixas)
    assert n == 0 and wn2 is wn and X2 is X


# ---------------------------------------------------------------- extras do HSI
def test_preparar_dados_hsi_repassa_calibracao_e_eixo(monkeypatch):
    from guaraci import hsi_extras
    chamado = {}

    def falso(pasta, **kw):
        chamado.update(kw)
        raise RuntimeError("parou aqui")
    monkeypatch.setattr(hsi_extras, "load_hsi_folder_dataset", falso)
    with pytest.raises(RuntimeError):
        hsi_extras.preparar_dados_hsi("x")
    assert chamado == {"calibracao": "nenhuma", "referencia_branco": "", "referencia_escuro": "", "eixo_bandas": "recusar"}
    with pytest.raises(RuntimeError):
        hsi_extras.preparar_dados_hsi("x", calibracao="branco_escuro", referencia_branco="b", referencia_escuro="d",
                                      eixo_bandas="reamostrar")
    assert chamado["calibracao"] == "branco_escuro" and chamado["eixo_bandas"] == "reamostrar"


def test_menu_dos_extras_hsi_passa_a_calibracao_do_cfg(monkeypatch, tmp_path):
    """O menu [X] carrega os cubos com a calibracao e o eixo do config da execucao (antes: sempre bruto)."""
    from guaraci import guaraci as g
    from guaraci import hsi_extras as hx
    from guaraci.config import Config
    chamado = {}

    def falso(pasta, **kw):
        chamado.update(kw)
        raise OSError("para aqui")
    monkeypatch.setattr(hx, "preparar_dados_hsi", falso)
    respostas = iter(["1", ""])
    monkeypatch.setattr(g, "_ask", lambda *a, **k: next(respostas))
    cfg = Config(mode="hsi", hsi_calibration="branco_escuro", hsi_white_ref="b.hdr", hsi_band_axis="reamostrar")
    g._menu_hsi_extras(str(tmp_path), True, cfg)
    assert chamado["calibracao"] == "branco_escuro" and chamado["referencia_branco"] == "b.hdr"
    assert chamado["eixo_bandas"] == "reamostrar"
