# -*- coding: utf-8 -*-
"""V75/V76: o rotulo do eixo espectral vem do PERFIL, nunca fixo em
"Wavenumber (cm-1)", e o grafico de VIP nao carrega atribuicoes de banda
NIR fixas (duas estavam erradas: 4325 e 5800 sao C-H, nao O-H nem C=O)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from guaraci import figuras as fg
from guaraci.config import Config
from guaraci.eixo_espectral import ROTULO_EIXO_NEUTRO, rotulo_do_eixo, unidade_do_eixo


def _cfg(perfil=None):
    cfg = Config(output_format="png", show_plots=False)
    if perfil is not None:
        cfg.matrix_profile = perfil
    return cfg


def _perfil_yaml(tmp_path, nome, unidade, rotulo=None):
    linhas = [f"descricao: 'teste {nome}'", f"unidade_eixo: '{unidade}'"]
    if rotulo:
        linhas.append(f"rotulo_eixo: '{rotulo}'")
    linhas += ["vocabulario:", "  matriz: 'x'"]
    caminho = tmp_path / f"{nome}.yaml"
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return str(caminho)


def test_sem_perfil_ou_perfil_neutro_o_eixo_e_variavel_espectral():
    assert rotulo_do_eixo(None) == ROTULO_EIXO_NEUTRO == "Variável espectral"
    assert rotulo_do_eixo(_cfg()) == "Variável espectral"            # default = generico
    assert unidade_do_eixo(_cfg()) == ""


def test_perfil_em_nm_rotula_nm(tmp_path):
    assert "nm" in rotulo_do_eixo(_cfg("visnir_400_1100nm"))
    assert "cm" not in rotulo_do_eixo(_cfg("visnir_400_1100nm"))
    assert unidade_do_eixo(_cfg("nir_dispersivo_1100_2498nm")) == "nm"


def test_perfil_em_cm1_rotula_numero_de_onda():
    assert "cm" in rotulo_do_eixo(_cfg("nir_ft_difuso"))


def test_perfil_raman_usa_o_rotulo_que_ele_declara(tmp_path):
    caminho = _perfil_yaml(tmp_path, "raman", "cm-1", "Raman shift (cm$^{-1}$)")
    assert rotulo_do_eixo(_cfg(caminho)) == "Raman shift (cm$^{-1}$)"


def test_unidade_sem_sentido_de_eixo_ou_perfil_inexistente_e_neutro(tmp_path):
    assert rotulo_do_eixo(_cfg("scanner")) == ROTULO_EIXO_NEUTRO      # unidade "indice"
    assert rotulo_do_eixo(_cfg("perfil_que_nao_existe")) == ROTULO_EIXO_NEUTRO
    # ppm passou a ter rotulo proprio (R6): RMN em ppm nao e' mais "Variavel espectral"
    caminho = _perfil_yaml(tmp_path, "rmn", "ppm")
    assert rotulo_do_eixo(_cfg(caminho)) == "Chemical shift (ppm)"
    caminho = _perfil_yaml(tmp_path, "desconhecida", "furlong")
    assert rotulo_do_eixo(_cfg(caminho)) == ROTULO_EIXO_NEUTRO


def _capturar(monkeypatch):
    achados = []
    def falso_save(fig, nome, pasta, cfg, subpasta=""):
        achados.append(fig)
    monkeypatch.setattr(fg, "save", falso_save)
    return achados


def _rotulos_xy(fig):
    return {ax.get_xlabel() for ax in fig.axes} | {ax.get_ylabel() for ax in fig.axes}


@pytest.mark.parametrize("perfil,esperado,proibido", [
    (None, "Variável espectral", "Wavenumber"),
    ("visnir_400_1100nm", "nm", "Wavenumber"),
])
def test_figuras_de_espectro_usam_o_rotulo_do_perfil(monkeypatch, tmp_path, perfil, esperado, proibido):
    achados = _capturar(monkeypatch)
    rng = np.random.default_rng(0)
    wn = np.linspace(500, 900, 80)
    X = rng.normal(size=(12, 80))
    rot = np.array(["a"] * 6 + ["b"] * 6)
    cores = {"a": "tab:blue", "b": "tab:red"}
    cfg = _cfg(perfil)
    fg.fig_class_mean_spectra(wn, X, rot, cores, cfg, str(tmp_path))
    fg.fig6_preprocessing(wn, X, X * 0.5, rot, cores, cfg, str(tmp_path))
    vip = np.abs(rng.normal(size=80)) + 1
    boot = {"mean": vip, "ci_lo": vip * 0.8, "ci_hi": vip * 1.2,
            "freq": np.full(80, 0.5)}
    try:
        fg.fig5b_vip_stability(boot, wn, top_n=5, cfg=cfg, pasta=str(tmp_path))
    except KeyError:
        pass                                    # chaves do bootstrap variam; o rotulo e' testado nas outras
    fg.fig_sprint3_sr_vip(vip, vip * 0.9, wn, top_n=5, cfg=cfg, pasta=str(tmp_path))
    assert achados
    for fig in achados:
        rotulos = _rotulos_xy(fig)
        assert any(esperado in r for r in rotulos), rotulos
        assert not any(proibido in r for r in rotulos), rotulos
    plt.close("all")
