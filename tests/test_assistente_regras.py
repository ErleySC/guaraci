"""Registro unico de regras do assistente: cada regra dispara no caso que descreve e fica calada no caso
vizinho que nao descreve. A configuracao vem de `Config()` real (defaults do programa), nunca de um
objeto inventado com atributos que o programa nao tem."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from guaraci import assistente_regras as R
from guaraci.config import Config


def cfg(**kw):
    c = Config()
    for k, v in kw.items():
        assert hasattr(c, k), f"Config nao tem o atributo {k}"
        setattr(c, k, v)
    return c


def dados(**kw):
    base = dict(n_leituras=60, n_variaveis=500, n_unidades=60, leituras_por_unidade_max=1,
                unidades_por_classe={"a": 30, "b": 30}, eixo_min=600.0, eixo_max=4000.0)
    base.update(kw)
    return R.ResumoDados(**base)


def ids(**kw):
    return {a.id for a in R.avaliar(**kw)}


# ------------------------------------------------------------ registro bem formado
def test_registro_bem_formado():
    from guaraci.catalogo import texto_proibido
    vistos = set()
    for r in R.REGRAS:
        assert r.id not in vistos, r.id
        vistos.add(r.id)
        assert r.momento in R.MOMENTOS and r.severidade in R.SEVERIDADES, r.id
        for ref in r.refs:
            assert ref in R.REFERENCIAS, (r.id, ref)
        for par in (r.mensagem, r.causa, r.fazer):
            assert len(par) == 2 and all(par), r.id
            for t in par:
                assert not texto_proibido(t), (r.id, texto_proibido(t))
    for c, u in R.REFERENCIAS.values():
        assert u.startswith("https://doi.org/10."), u


def test_campo_e_chave_yaml_real():
    """`campo` e' o que o usuario escreve no config.yaml: tem de existir no esquema do config_io."""
    from guaraci import config_io
    chaves = {d["key"] for d in config_io._CONFIG_SPEC} if hasattr(config_io, "_CONFIG_SPEC") else None
    if chaves is None:
        import inspect
        fonte = inspect.getsource(config_io)
        for r in R.REGRAS:
            if r.campo:
                assert f'"key": "{r.campo}"' in fonte or f"'key': '{r.campo}'" in fonte, r.campo
    else:
        for r in R.REGRAS:
            if r.campo:
                assert r.campo in chaves, r.campo


def test_config_padrao_nao_tem_achado_critico():
    assert not [a for a in R.avaliar(cfg()) if a.severidade == "critico"]
    assert not [a for a in R.avaliar(cfg(), dados()) if a.severidade == "critico"]


# ------------------------------------------------------------ config
@pytest.mark.parametrize("kw,regra,dispara", [
    (dict(wn_min=4000.0, wn_max=600.0), "faixa_invertida", True),
    (dict(wn_min=600.0, wn_max=4000.0), "faixa_invertida", False),
    (dict(mode="multielementar", wn_min=4000.0, wn_max=600.0), "faixa_invertida", False),
    (dict(n_permutations=10), "permutacoes_sem_resolucao", True),      # 1/11 > 0,05
    (dict(n_permutations=19), "permutacoes_sem_resolucao", False),     # 1/20 = 0,05
    (dict(n_permutations=0), "permutacoes_sem_resolucao", False),      # desligado
    (dict(n_permutations=50), "permutacoes_poucas", True),
    (dict(n_permutations=10), "permutacoes_poucas", False),            # ja coberto pela critica
    (dict(n_permutations=200), "permutacoes_poucas", False),
    (dict(mode="csv", group_column=""), "grupo_nao_declarado", True),
    (dict(mode="csv", group_column="lote"), "grupo_nao_declarado", False),
    (dict(mode="multielementar", default_preprocessing="msc_sg_mc"), "preproc_espectral_em_tabela", True),
    (dict(mode="multielementar", default_preprocessing="autoscaling"), "preproc_espectral_em_tabela", False),
    (dict(default_preprocessing="autoscaling"), "autoescala_em_espectro", True),
    (dict(mode="multielementar", default_preprocessing="autoscaling"), "autoescala_em_espectro", False),
    (dict(frac_holdout=0.0), "sem_holdout", True),
    (dict(frac_holdout=0.2), "sem_holdout", False),
    (dict(frac_holdout=0.1), "holdout_fora_da_faixa", True),
    (dict(frac_holdout=0.4), "holdout_fora_da_faixa", True),
    (dict(frac_holdout=0.2), "holdout_fora_da_faixa", False),
    (dict(frac_holdout=0.0), "holdout_fora_da_faixa", False),
    (dict(default_preprocessing="raw"), "preproc_bruto_em_espectro", True),
    (dict(default_preprocessing="msc_sg_mc"), "preproc_bruto_em_espectro", False),
    (dict(max_lvs=41), "max_lvs_alto", True),
    (dict(max_lvs=40), "max_lvs_alto", False),
])
def test_regras_de_config(kw, regra, dispara):
    assert (regra in ids(cfg=cfg(**kw))) is dispara


def test_permutacoes_sem_resolucao_mensagem():
    a = next(x for x in R.avaliar(cfg(n_permutations=10)) if x.id == "permutacoes_sem_resolucao")
    assert "1/11 = 0.091" in a.mensagem and "19" in a.fazer and a.campo == "n_permutacoes" and a.valor == 200


# ------------------------------------------------------------ dados
@pytest.mark.parametrize("ckw,dkw,regra,dispara", [
    ({}, dict(n_nan=3), "dados_com_nan", True),
    ({}, {}, "dados_com_nan", False),
    (dict(wn_min=10000.0, wn_max=12000.0), {}, "faixa_fora_do_eixo", True),
    (dict(wn_min=900.0, wn_max=1800.0), {}, "faixa_fora_do_eixo", False),
    (dict(group_by_mae_id=False), dict(leituras_por_unidade_max=3, n_unidades=20), "replicas_sem_validacao_agrupada", True),
    (dict(group_by_mae_id=True), dict(leituras_por_unidade_max=3, n_unidades=20), "replicas_sem_validacao_agrupada", False),
    (dict(group_by_mae_id=False), {}, "replicas_sem_validacao_agrupada", False),
    ({}, dict(unidades_por_classe={"a": 30, "b": 1}), "classe_com_uma_unidade", True),
    ({}, dict(unidades_por_classe={"a": 30, "b": 2}), "classe_com_uma_unidade", False),
    # o pipeline ja limita as variaveis latentes ao tamanho do treino de cada dobra (min(max_lvs, n_treino-1)):
    # max_lvs alto com poucas unidades nao e' achado (antes avisava em toda execucao com o padrao 40)
    (dict(max_lvs=40), dict(n_unidades=12), "max_lvs_vs_unidades", False),
    (dict(n_splits_cv=5), dict(unidades_por_classe={"a": 3, "b": 30}), "dobras_maior_que_grupos", True),
    (dict(n_splits_cv=5), {}, "dobras_maior_que_grupos", False),
    (dict(frac_holdout=0.2), dict(n_unidades=40), "holdout_pequeno", True),      # 8 unidades
    (dict(frac_holdout=0.2), dict(n_unidades=200), "holdout_pequeno", False),    # 40 unidades
    (dict(frac_holdout=0.0), dict(n_unidades=40), "holdout_pequeno", False),
    ({}, dict(unidades_por_classe={"a": 30, "b": 9}), "classes_desbalanceadas", True),
    ({}, dict(unidades_por_classe={"a": 30, "b": 11}), "classes_desbalanceadas", False),
    ({}, dict(n_constantes=4), "variaveis_constantes", True),
    ({}, {}, "variaveis_constantes", False),
])
def test_regras_de_dados(ckw, dkw, regra, dispara):
    assert (regra in ids(cfg=cfg(**ckw), dados=dados(**dkw))) is dispara


def test_holdout_pequeno_meia_largura():
    a = next(x for x in R.avaliar(cfg(frac_holdout=0.2), dados(n_unidades=40)) if x.id == "holdout_pequeno")
    # 8 unidades, p=0,9: Wilson 95% = [0,556; 0,985] (Brown, Cai & DasGupta 2001, eq. 4)
    assert "8 unidades" in a.mensagem and "56% a 98%" in a.mensagem
    lo, hi = R._wilson(0.9, 8)
    assert abs(lo - 0.5557) < 1e-3 and abs(hi - 0.9848) < 1e-3
    assert R._wilson(1.0, 3)[1] == 1.0 and R._wilson(0.5, 0) == (0.0, 1.0)


def test_regras_de_dados_so_com_dados():
    assert not ids(cfg=cfg(), momentos=["dados"])


# ------------------------------------------------------------ resultado
@pytest.mark.parametrize("res,regra,dispara", [
    ({"_auditoria_critica": ["confundimento classe x sessao"]}, "auditoria_critica", True),
    ({"_auditoria_critica": []}, "auditoria_critica", False),
    ({"Permutation p-value": 0.12}, "permutacao_nao_significativa", True),
    ({"Permutation p-value": 0.005}, "permutacao_nao_significativa", False),
    ({"Accuracy (CV)": 1.0, "N unidades independentes": 12}, "acuracia_perfeita", True),
    ({"Accuracy (CV)": 1.0, "N unidades independentes": 80}, "acuracia_perfeita", False),
    ({"Accuracy (CV)": 0.95, "N unidades independentes": 12}, "acuracia_perfeita", False),
    ({"R2Y": 0.95, "Q2": 0.40}, "r2_muito_acima_de_q2", True),
    ({"R2Y": 0.95, "Q2": 0.80}, "r2_muito_acima_de_q2", False),
    ({"R2Y": 0.95}, "r2_muito_acima_de_q2", False),
    ({"Balanced accuracy": 0.95, "Holdout balanced acc": 0.70}, "cv_muito_acima_do_holdout", True),
    ({"Balanced accuracy": 0.95, "Holdout balanced acc": 0.90}, "cv_muito_acima_do_holdout", False),
    ({"LVs no teto (max_lvs)": "True"}, "lvs_no_teto", True),
    ({"LVs no teto (max_lvs)": "False"}, "lvs_no_teto", False),
])
def test_regras_de_resultado(res, regra, dispara):
    assert (regra in ids(resultado=res)) is dispara


def test_valor_de_tipo_errado_nao_quebra():
    # texto onde se espera numero: a regra nao se pronuncia, nao levanta
    assert ids(resultado={"Permutation p-value": "n/a", "R2Y": "?", "Q2": None}) == set()


def test_ordem_por_severidade():
    a = R.avaliar(cfg(n_permutations=10, frac_holdout=0.0, wn_min=4000.0, wn_max=600.0),
                  dados(n_constantes=2))
    sev = [x.severidade for x in a]
    assert sev == sorted(sev, key=R.SEVERIDADES.index)


# ------------------------------------------------------------ resumo_modelo.txt
_RESUMO = """GUARACI -- resumo do modelo
  Accuracy (CV)             : 1.0000
  Balanced accuracy         : 0.9500
  Permutation p-value       : 0.0050
  N unidades independentes  : 12
  LVs no teto (max_lvs)     : True
  Pre-processing            : msc_sg_mc (Savitzky-Golay 11)
  Accuracy (CV)             : 0.5000

Auditoria de delineamento
  [CRITICO] confundimento_classe_sessao: classe a medida so' no dia 1
  [ok] classe_unica: 2 classes presentes.
"""


def test_resultado_de_resumo():
    r = R.resultado_de_resumo(_RESUMO)
    assert r["Accuracy (CV)"] == 1.0             # primeira ocorrencia vale
    assert r["N unidades independentes"] == 12.0
    assert r["Pre-processing"] == "msc_sg_mc (Savitzky-Golay 11)"
    assert r["LVs no teto (max_lvs)"] == "True"
    assert r["_auditoria_critica"] == ["confundimento_classe_sessao"]
    assert {"auditoria_critica", "acuracia_perfeita", "lvs_no_teto"} <= ids(resultado=r)
    assert R.resultado_de_resumo("")["_auditoria_critica"] == []


def test_chaves_lidas_sao_gravadas_pelo_pipeline():
    """As chaves que as regras de resultado leem tem de aparecer no codigo que grava o resumo."""
    import inspect
    from guaraci import pipeline
    fonte = inspect.getsource(pipeline)
    for k in ("Permutation p-value", "Accuracy (CV)", "Holdout accuracy", "N unidades independentes", "R2Y", "Q2",
              "Balanced accuracy", "Holdout balanced acc", "LVs no teto (max_lvs)", "[CRITICO]"):
        assert k in fonte, k


# ------------------------------------------------------------ resumir_dados e formatar
def test_resumir_dados():
    X = np.array([[1.0, 2.0, 5.0], [1.0, np.nan, 6.0], [1.0, 4.0, 7.0], [1.0, 5.0, 8.0]])
    d = R.resumir_dados(X, eixo=[1000, 1002, 1004], rotulos=["a", "a", "b", "b"], mae_id=["m1", "m1", "m2", "m3"])
    assert (d.n_leituras, d.n_variaveis, d.n_unidades, d.leituras_por_unidade_max) == (4, 3, 3, 2)
    assert d.unidades_por_classe == {"a": 1, "b": 2}
    assert d.n_nan == 1 and d.n_constantes == 1
    assert (d.eixo_min, d.eixo_max) == (1000.0, 1004.0)
    assert R.resumir_dados(X).n_unidades == 4
    with pytest.raises(ValueError):
        R.resumir_dados(np.zeros(3))


def test_formatar_achados():
    a = R.avaliar(cfg(frac_holdout=0.0, n_permutations=10))
    t = R.formatar_achados(a)
    assert t.startswith("[CRITICO]") and "Aceitar: n_permutacoes: 200" in t and "https://doi.org/" in t
    assert "Fonte:" not in R.formatar_achados(a, refs=False)
    assert "Why:" in R.formatar_achados(a, "EN")
    assert R.formatar_achados([]).startswith("Nenhum problema")
    b = R.avaliar(cfg(group_by_mae_id=False), dados(leituras_por_unidade_max=2))
    assert "validacao_group_aware: true" in R.formatar_achados(b)


def test_achado_e_dataclass_simples():
    a = R.avaliar(cfg(frac_holdout=0.0))[0]
    assert dataclasses.asdict(a)["id"]


@pytest.mark.parametrize("res,regra", [
    ({"Accuracy (CV)": 1.0, "N unidades independentes": 12}, "acuracia_perfeita"),
    ({"Permutation p-value": 0.12}, "permutacao_nao_significativa"),
    ({"R2Y": 0.95, "Q2": 0.40}, "r2_muito_acima_de_q2"),
    ({"Balanced accuracy": 0.95, "Holdout balanced acc": 0.70}, "cv_muito_acima_do_holdout"),
])
def test_metricas_de_classe_nao_avisam_numa_quantificacao(res, regra):
    """Auditoria A-08: numa quantificacao essas metricas vem do modelo auxiliar; o aviso confundia o usuario."""
    assert regra in ids(cfg=cfg(objective="classificacao"), resultado=res)
    assert regra not in ids(cfg=cfg(objective="quantificacao"), resultado=res)
