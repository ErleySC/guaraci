"""Ajuste automatico por contexto: lista branca, sugestoes, recusas, idempotencia."""
import copy

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from guaraci import contexto as ctx
from guaraci.config import Config

LISTA_BRANCA = {a[2] for a in ctx.AJUSTES}
# Escolhas que MUDAM o resultado cientifico: nunca automaticas.
NUNCA_AUTOMATICO = {"validacao_group_aware", "n_permutacoes", "holdout_fracao", "holdout_estratificado",
                    "icp_politica_censura", "icp_limiar_censura", "unidade_agrupamento", "coluna_grupo",
                    "selecao_spa", "selecao_ag", "selecao_cars", "selecao_uve", "nivel", "objetivo",
                    "selecao_lv_cv_aninhada", "max_lvs", "modo_ddsimca"}


def test_lista_branca_nao_contem_escolha_cientifica():
    assert not LISTA_BRANCA & NUNCA_AUTOMATICO, sorted(LISTA_BRANCA & NUNCA_AUTOMATICO)


def test_toda_regra_de_ajuste_tem_motivo_e_campo_do_config():
    from guaraci.config_io import _CONFIG_SPEC
    chaves = {s["key"] for s in _CONFIG_SPEC}
    for rid, _q, campo, _alvo, motivo in ctx.AJUSTES:
        assert campo in chaves, rid
        assert all(len(m) > 15 for m in motivo), rid      # (PT, EN)


def test_tabular_com_preproc_espectral_vira_autoescala_com_linha_de_aviso():
    cfg = Config()
    cfg.mode, cfg.default_preprocessing = "multielementar", "msc_sg_mc"
    r = ctx.aplicar(cfg)
    assert cfg.default_preprocessing == "autoscaling"
    a = [x for x in r.ajustes if x.regra == "tabular_preproc"][0]
    linha = a.linha()
    assert linha.startswith("Ajustado: pre_processamento msc_sg_mc -> autoscaling. Motivo:")
    assert "Desfazer:" in linha and "--sem-ajuste-automatico" in linha
    assert cfg.ajustes_automaticos and cfg.ajustes_automaticos[0]["campo"] == "pre_processamento"


def test_n1_desliga_ddsimca_e_n2_liga():
    cfg = Config()
    cfg.level, cfg.run_ddsimca = "N1", True
    ctx.aplicar(cfg)
    assert cfg.run_ddsimca is False
    cfg2 = Config()
    cfg2.level, cfg2.run_ddsimca = "N2", False
    ctx.aplicar(cfg2)
    assert cfg2.run_ddsimca is True


def test_objetivo_quantificacao_desliga_etapas_de_classificacao():
    cfg = Config()
    cfg.level, cfg.run_benchmark, cfg.run_monte_carlo, cfg.run_opls = "N3", True, True, True
    ctx.aplicar(cfg)
    assert not (cfg.run_benchmark or cfg.run_monte_carlo or cfg.run_opls)


def test_desligado_nao_altera_nada():
    cfg = Config()
    cfg.mode, cfg.default_preprocessing = "multielementar", "msc_sg_mc"
    ctx.aplicar(cfg, auto=False)
    assert cfg.default_preprocessing == "msc_sg_mc" and cfg.ajustes_automaticos == []
    cfg.auto_adjust = False
    ctx.aplicar(cfg)
    assert cfg.default_preprocessing == "msc_sg_mc"


def test_desfazer_restaura_os_valores_originais():
    cfg = Config()
    cfg.level, cfg.run_ddsimca, cfg.run_benchmark = "N1", True, True
    antes = copy.deepcopy(vars(cfg))
    ctx.aplicar(cfg)
    assert vars(cfg) != antes
    assert ctx.desfazer(cfg) > 0
    registro = ("ajustes_automaticos", "sugestoes_contexto")
    assert {k: v for k, v in vars(cfg).items() if k not in registro} == \
           {k: v for k, v in antes.items() if k not in registro}


def test_recusa_n3_csv_sem_concentracao_com_causa_e_acao():
    cfg = Config()
    cfg.mode, cfg.level, cfg.conc_column = "csv", "N3", None
    r = ctx.aplicar(cfg)
    assert r.erros and "coluna_concentracao" in r.erros[0] and "N1" in r.erros[0]


def test_sugestao_nao_aplica_a_escolha_de_validacao():
    cfg = Config()
    cfg.mode, cfg.group_column, cfg.group_by_mae_id = "csv", "lote", False
    r = ctx.aplicar(cfg)
    assert any("validacao_group_aware: true" in s for s in r.sugestoes)
    assert cfg.group_by_mae_id is False             # a sugestao NAO foi aplicada


def test_campos_visiveis_oculta_o_que_nao_se_aplica_ao_modo():
    cfg = Config()
    cfg.mode = "dx"
    campos = ["pasta_dados", "arquivo_csv", "icp_unidade", "hplc_detector", "faixa_min_cm", "dpi"]
    assert ctx.campos_visiveis(cfg, campos) == ["pasta_dados", "faixa_min_cm", "dpi"]
    cfg.mode = "multielementar"
    assert ctx.campos_visiveis(cfg, campos) == ["pasta_dados", "arquivo_csv", "icp_unidade", "dpi"]


@settings(max_examples=60, deadline=None)
@given(modo=st.sampled_from(["dx", "csv", "multielementar", "imagem", "hsi", "sintetico"]),
       nivel=st.sampled_from(["N1", "N2", "N3"]),
       objetivo=st.sampled_from(["auto", "exploratorio", "classificacao", "quantificacao"]),
       pre=st.sampled_from(["msc_sg_mc", "snv_sg_mc", "autoscaling", "mc"]),
       bench=st.booleans(), dds=st.booleans(), opls=st.booleans())
@example(modo="multielementar", nivel="N1", objetivo="auto", pre="msc_sg_mc", bench=True, dds=True, opls=True)
@example(modo="csv", nivel="N3", objetivo="auto", pre="autoscaling", bench=True, dds=False, opls=True)
def test_propriedades_idempotencia_lista_branca_e_determinismo(modo, nivel, objetivo, pre, bench, dds, opls):
    def novo():
        c = Config()
        c.mode, c.level, c.objective, c.default_preprocessing = modo, nivel, objetivo, pre
        c.run_benchmark, c.run_ddsimca, c.run_opls = bench, dds, opls
        c.conc_column = "conc" if nivel == "N3" else None
        return c
    a, b = novo(), novo()
    r1 = ctx.aplicar(a)
    r2 = ctx.aplicar(b)
    assert [x.linha() for x in r1.ajustes] == [x.linha() for x in r2.ajustes]       # mesma entrada, mesma saida
    assert {x.campo for x in r1.ajustes} <= LISTA_BRANCA                            # so' lista branca
    r3 = ctx.aplicar(a)                                                              # idempotente
    assert r3.ajustes == []
    # configuracao efetiva sempre valida para o modo tabular
    if modo in ctx.MODOS_TABULARES:
        assert a.default_preprocessing in ctx.PREPROC_TABULAR


# ---------------------------------------------------------------- integracao
def _csv(caminho):
    import numpy as np
    rng = np.random.default_rng(0)
    linhas = ["Amostra;Tipo;Origem;Lote;Mn 55;Zn 66;Sr 88"]
    for o in range(3):
        for a in range(6):
            base = rng.lognormal(0, 0.2, 3) * (1 + 0.8 * o)
            for r in range(2):
                v = base * rng.lognormal(0, 0.04, 3)
                linhas.append(f"O{o}A{a}R{r};amostra;origem{o};O{o}L{a};" + ";".join(f"{x:.4f}".replace(".", ",") for x in v))
    caminho.write_text("\n".join(linhas), encoding="utf-8")


def _cfg_icp(pq, csv, saida, **kw):
    cfg = pq.Config()
    cfg.mode, cfg.csv_file, cfg.class_column, cfg.group_column = "multielementar", str(csv), "Origem", "Lote"
    cfg.icp_unidade, cfg.icp_coluna_amostra, cfg.icp_coluna_tipo = "mg/kg", "Amostra", "Tipo"
    cfg.output_root_folder, cfg.show_plots, cfg.n_permutations, cfg.max_lvs = saida, False, 10, 4
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.slow
def test_executar_ajusta_grava_efetiva_e_devolve_o_cfg_do_usuario(pq, tmp_path):
    import json
    import shutil
    import tempfile
    from pathlib import Path
    from conftest import achar_pastas_run
    csv = tmp_path / "icp.csv"
    _csv(csv)
    saida = tempfile.mkdtemp(prefix="gc")
    try:
        cfg = _cfg_icp(pq, csv, saida, default_preprocessing="msc_sg_mc")      # incompativel com tabela de elementos
        pq.executar(cfg)
        assert cfg.default_preprocessing == "msc_sg_mc"                         # o cfg do usuario nao mudou
        assert [a["campo"] for a in cfg.ajustes_automaticos] == ["pre_processamento"]
        run = Path(achar_pastas_run(saida)[0])
        efetiva = (run / "config_efetiva.yaml").read_text(encoding="utf-8")
        assert "pre_processamento: Autoscaling" in efetiva or "pre_processamento: autoscaling" in efetiva
        assert "# Ajustado: pre_processamento" in efetiva
        assert str(tmp_path) not in efetiva                                     # sem caminho da maquina
        j = json.loads((run / pq.NOME_RELATORIOS / "ajustes_automaticos.json").read_text(encoding="utf-8"))
        assert j["ajuste_automatico"] is True and j["ajustes"][0]["campo"] == "pre_processamento"
    finally:
        shutil.rmtree(saida, ignore_errors=True)


def test_sem_ajuste_automatico_a_combinacao_incompativel_falha_com_mensagem(pq, tmp_path):
    csv = tmp_path / "icp.csv"
    _csv(csv)
    cfg = _cfg_icp(pq, csv, str(tmp_path / "s"), default_preprocessing="msc_sg_mc", auto_adjust=False)
    with pytest.raises(ValueError, match="espectral"):
        pq.executar(cfg)
    assert cfg.default_preprocessing == "msc_sg_mc" and cfg.ajustes_automaticos == []


def test_executar_recusa_n3_csv_sem_concentracao_antes_de_ler_dado(pq, tmp_path):
    cfg = pq.Config()
    cfg.mode, cfg.level, cfg.conc_column, cfg.csv_file = "csv", "N3", None, str(tmp_path / "nao_existe.csv")
    with pytest.raises(ValueError, match="coluna_concentracao"):
        pq.executar(cfg)


def test_ajuste_automatico_e_opcao_do_config_e_aceita_no_yaml(pq, tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("ajuste_automatico: false\n", encoding="utf-8")
    assert pq.load_config(str(y)).auto_adjust is False
