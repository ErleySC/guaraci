# -*- coding: utf-8 -*-
"""`classificador: auto` e a faixa padrao que se abre sozinha (e so' quando falharia)."""
import numpy as np
import pytest

from guaraci import contexto as ctx
from guaraci import pipeline as P
from guaraci.config import CLASSIFICADOR_AUTO_MIN_UNIDADES, Config
from guaraci.config_io import load_config


def _cfg(**kw):
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_padrao_e_auto_e_o_yaml_aceita_os_tres(tmp_path):
    assert Config().classificador == "auto"
    for nome in ("auto", "plsda", "plslda"):
        p = tmp_path / f"{nome}.yaml"
        p.write_text(f"classificador: {nome}\n", encoding="utf-8")
        assert load_config(str(p)).classificador == nome
    p = tmp_path / "x.yaml"
    p.write_text("classificador: svm\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(p))


@pytest.mark.parametrize("n,esperado", [(30, "plslda"), (CLASSIFICADOR_AUTO_MIN_UNIDADES - 1, "plslda"),
                                        (CLASSIFICADOR_AUTO_MIN_UNIDADES, "plsda"), (500, "plsda")])
def test_auto_escolhe_pelo_numero_de_unidades(n, esperado):
    cfg = _cfg(level="N1")
    P._resolver_classificador_auto(cfg, n, None)
    assert cfg.classificador == esperado
    ctx.desfazer(cfg, manter_registro=True)
    assert cfg.classificador == "auto"                       # o cfg do usuario volta a 'auto'
    assert cfg.ajustes_automaticos[-1]["campo"] == "classificador" or esperado == "plsda"


def test_auto_conta_grupos_e_nao_replicas():
    cfg = _cfg(level="N1")
    mae = np.repeat(np.arange(40), 5)                        # 200 espectros, 40 amostras fisicas
    P._resolver_classificador_auto(cfg, 200, mae)
    assert cfg.classificador == "plslda"


def test_escolha_manual_nao_e_tocada_nem_registrada():
    for nome in ("plsda", "plslda"):
        cfg = _cfg(level="N1", classificador=nome)
        P._resolver_classificador_auto(cfg, 10, None)
        assert cfg.classificador == nome and not getattr(cfg, "ajustes_automaticos", [])


def test_auto_nao_age_em_quantificacao():
    cfg = _cfg(level="N3")
    P._resolver_classificador_auto(cfg, 10, None)
    assert cfg.classificador == "auto"


def test_auto_desligado_pelo_ajuste_automatico():
    cfg = _cfg(level="N1", auto_adjust=False)
    P._resolver_classificador_auto(cfg, 10, None)
    assert cfg.classificador == "auto"


def test_faixa_padrao_fora_do_eixo_abre_para_o_eixo_e_desfaz():
    cfg = _cfg()
    P._abrir_faixa_padrao_se_fora_do_eixo(cfg, (900.0, 1700.0))
    assert (cfg.wn_min, cfg.wn_max) == (900.0, 1700.0)
    ctx.desfazer(cfg, manter_registro=True)
    assert (cfg.wn_min, cfg.wn_max) == (4000.0, 10000.0)


def test_faixa_padrao_que_cruza_o_eixo_nao_muda():
    cfg = _cfg()
    P._abrir_faixa_padrao_se_fora_do_eixo(cfg, (3800.0, 12000.0))   # dado NIR: resultado publicado intacto
    assert (cfg.wn_min, cfg.wn_max) == (4000.0, 10000.0)


def test_faixa_escolhida_pelo_usuario_nao_e_tocada():
    cfg = _cfg(wn_min=100.0, wn_max=200.0)
    P._abrir_faixa_padrao_se_fora_do_eixo(cfg, (900.0, 1700.0))
    assert (cfg.wn_min, cfg.wn_max) == (100.0, 200.0)               # o erro de faixa segue sendo dela


def test_auto_nao_escolhe_plslda_com_menos_de_2_amostras_por_classe():
    import guaraci.pipeline as P
    cfg = P.Config(classificador="auto")
    P._resolver_classificador_auto(cfg, 8, None, n_classes=6)
    assert cfg.classificador == "auto" and not getattr(cfg, "ajustes_automaticos", [])
