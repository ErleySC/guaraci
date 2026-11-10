"""unidade_agrupamento=auto e plano de coleta a partir dos dados (pedido de 2026-11-10).

Origem: no model card de 2026-11-05 todas as sessoes tinham UMA classe so' e a
validacao por mae_id deu bal.acc 0,90-0,98 contra acaso 0,25 (o modelo aprende a
sessao). "auto" passa a validar por sessao quando isso e' possivel.
"""
from __future__ import annotations

import numpy as np

from guaraci.auditoria_delineamento import resolver_unidade_agrupamento, run_audit
from guaraci.config import Config
from guaraci.plano_coleta import lacunas_de_sessoes, texto_lacunas


def _dados(sessoes_por_classe: dict, diluicoes: int = 3):
    """mae_id no padrao ESP-DD-MM-AAAA-<token de teor>: varias diluicoes por sessao."""
    rot, mae = [], []
    for classe, dias in sessoes_por_classe.items():
        for d in dias:
            for k in range(diluicoes):
                rot.append(classe)
                mae.append(f"{classe}-{d:02d}-01-2026-A{k + 1}.0{k + 1}")
    return np.array(rot), np.array(mae)


def test_explicito_passa_direto_e_nunca_e_trocado():
    rot, mae = _dados({"A": [1, 2], "B": [3, 4]})
    assert resolver_unidade_agrupamento(rot, mae, "mae_id") == ("mae_id", "")
    assert resolver_unidade_agrupamento(rot, mae, "sessao") == ("sessao", "")


def test_auto_vira_sessao_quando_cada_sessao_tem_uma_classe_e_da_para_validar():
    rot, mae = _dados({"A": [1, 2], "B": [3, 4]})
    un, motivo = resolver_unidade_agrupamento(rot, mae, "auto")
    assert un == "sessao" and "SESSAO" in motivo


def test_auto_fica_em_mae_id_se_alguma_classe_tem_uma_unica_sessao():
    rot, mae = _dados({"A": [1, 2], "B": [3, 4], "C": [5]})
    un, motivo = resolver_unidade_agrupamento(rot, mae, "auto")
    assert un == "mae_id" and "C" in motivo and "OTIMISTAS" in motivo


def test_auto_fica_em_mae_id_com_menos_de_3_sessoes():
    rot, mae = _dados({"A": [1, 2]})
    un, motivo = resolver_unidade_agrupamento(rot, mae, "auto")
    assert un == "mae_id" and motivo


def test_auto_sem_mae_id_ou_com_grupo_declarado_nao_decide():
    rot, mae = _dados({"A": [1, 2], "B": [3, 4]})
    assert resolver_unidade_agrupamento(rot, None, "auto") == ("mae_id", "")
    assert resolver_unidade_agrupamento(rot, mae, "auto", "medium") == ("mae_id", "")


def test_auto_com_classes_misturadas_na_mesma_sessao_fica_em_mae_id():
    rot = np.array(["A", "B", "A", "B"])
    mae = np.array(["X-01-01-2026-A1.01", "X-01-01-2026-A2.02", "X-02-01-2026-A1.01", "X-02-01-2026-A2.02"])
    assert resolver_unidade_agrupamento(rot, mae, "auto") == ("mae_id", "")


def test_auto_com_sessao_igual_ao_mae_id_fica_em_mae_id():
    rot = np.array(["A", "A", "B", "B"])
    mae = np.array(["a1", "a2", "b1", "b2"])
    assert resolver_unidade_agrupamento(rot, mae, "auto") == ("mae_id", "")


def test_run_audit_resolve_o_auto_sozinho_e_aceita_a_unidade_do_pipeline():
    rot, mae = _dados({"A": [1, 2], "B": [3, 4]})
    X = np.random.default_rng(0).normal(size=(len(rot), 6))
    wn = np.arange(6, dtype=float)
    cfg = Config()
    assert cfg.group_unit == "auto"
    achado = {a.nome: a for a in run_audit(X, wn, rot, cfg, None, mae)}["unidade_independente"]
    assert achado.severidade == "ok" and "SESSAO" in achado.mensagem          # auto -> sessao
    achado2 = {a.nome: a for a in run_audit(X, wn, rot, cfg, None, mae, unidade="mae_id")}["unidade_independente"]
    assert achado2.severidade == "critico"


def test_lacunas_contam_sessoes_e_o_que_falta():
    rot, mae = _dados({"A": [1, 2, 3], "B": [4]})
    lac = lacunas_de_sessoes(rot, mae)
    por = {x["classe"]: x for x in lac["classes"]}
    assert por["A"]["sessoes"] == 3 and por["A"]["faltam_honesto"] == 2 and por["A"]["faltam_conformal"] == 16
    assert por["B"]["sessoes"] == 1 and por["B"]["faltam_honesto"] == 4
    assert lac["n_sessoes_total"] == 4 and len(lac["confundidas"]) == 4
    assert lac["minimo_honesto"] == 5 and lac["minimo_conformal"] == 19


def test_lacunas_sem_mae_id_nao_inventam_nada():
    lac = lacunas_de_sessoes(["A", "B"], None)
    assert lac["classes"] == [] and texto_lacunas(lac) == []


def test_texto_lacunas_em_pt_e_en():
    rot, mae = _dados({"A": [1, 2, 3], "B": [4]})
    pt = " ".join(texto_lacunas(lacunas_de_sessoes(rot, mae), "PT"))
    en = " ".join(texto_lacunas(lacunas_de_sessoes(rot, mae), "EN"))
    assert "UMA classe so'" in pt and "conformal" in pt and "Replicas" in pt
    assert "SINGLE class" in en and "conformal" in en and "Replicates" in en


def test_texto_lacunas_ramos_misto_e_completo():
    rot = np.array(["A", "B", "A", "B", "A"])
    mae = np.array(["X-01-01-2026-A1.01", "X-01-01-2026-A2.02", "X-02-01-2026-A1.01", "X-02-01-2026-A2.02", "X-03-01-2026-A1.01"])
    misto = " ".join(texto_lacunas(lacunas_de_sessoes(rot, mae), "PT"))
    assert "de 3 sessoes tem uma classe so'" in misto
    en = " ".join(texto_lacunas(lacunas_de_sessoes(rot, mae), "EN"))
    assert "of 3 sessions hold a single class" in en
    grande = []
    maes = []
    for classe in "AB":
        for d in range(1, 21):
            grande.append(classe)
            maes.append(f"{classe}{d}-{d:02d}-01-2026")
            grande.append("B" if classe == "A" else "A")
            maes.append(f"{classe}{d}-{d:02d}-01-2026-x")
    cheio = " ".join(texto_lacunas(lacunas_de_sessoes(np.array(grande), np.array(maes)), "PT"))
    assert "resultado honesto possivel" in cheio and "garantia conformal" not in cheio
    cheio_en = " ".join(texto_lacunas(lacunas_de_sessoes(np.array(grande), np.array(maes)), "EN"))
    assert "an honest result is possible" in cheio_en


def test_pipeline_com_auto_valida_por_sessao_no_delineamento_aninhado(tmp_path):
    """Roda o `executar()` real com o padrao (auto): o teste nulo (rotulo por sessao) tem que
    cair no acaso, como em `unidade_agrupamento=sessao`, e o relatorio diz 'sessao'."""
    from tests.test_unidade_independente_nulo_sessao import _rodar
    bal, txt = _rodar(tmp_path, "auto", "por_auto")
    assert bal < 0.45, f"auto deveria virar sessao e ficar no acaso; bal.acc={bal:.3f}"
    assert "Unidade de agrupamento" in txt and "sessao" in txt


def test_texto_lacunas_sem_sessao_confundida_nao_fala_de_mistura():
    rot = np.array(["A", "B", "A", "B"])
    mae = np.array(["X-01-01-2026-A1.01", "X-01-01-2026-A2.02", "X-02-01-2026-A1.01", "X-02-01-2026-A2.02"])
    txt = " ".join(texto_lacunas(lacunas_de_sessoes(rot, mae), "PT"))
    assert "uma classe so'" not in txt and "faltam" in txt
