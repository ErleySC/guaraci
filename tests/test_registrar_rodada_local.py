"""scripts/registrar_rodada_local.py: o registro guarda so' desfechos (nunca arquivo ou caminho) e acusa quando o codigo testado mudou."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import registrar_rodada_local as rrl  # noqa: E402

JUNIT = """<?xml version="1.0"?><testsuites><testsuite>
<testcase name="test_b" classname="x"/><testcase name="test_a" classname="x"><failure message="falhou em C:\\Users\\alguem\\arq.spc"/></testcase>
<testcase name="test_c" classname="x"><skipped message="arquivo ausente em /home/alguem/f"/></testcase></testsuite></testsuites>"""


def test_resultados_ordenados_sem_mensagem_de_falha_nem_motivo_de_pulo(tmp_path):
    j = tmp_path / "r.xml"
    j.write_text(JUNIT, encoding="utf-8")
    r = rrl.resultados_do_junit(j)
    assert r == [{"teste": "test_a", "desfecho": "falhou"}, {"teste": "test_b", "desfecho": "passou"}, {"teste": "test_c", "desfecho": "pulado"}]
    assert "alguem" not in json.dumps(r)


def test_registro_so_e_aprovado_sem_falhas_sem_pulos_e_com_ao_menos_um_teste():
    ok = [{"teste": "a", "desfecho": "passou"}]
    assert rrl.montar_registro(ok, "h")["aprovada"] is True
    assert rrl.montar_registro([], "h")["aprovada"] is False                                            # nada rodou
    assert rrl.montar_registro(ok + [{"teste": "b", "desfecho": "pulado"}], "h")["aprovada"] is False     # sem os arquivos: nao e' aprovacao
    assert rrl.montar_registro(ok + [{"teste": "b", "desfecho": "falhou"}], "h")["aprovada"] is False


def test_hash_do_resultado_muda_quando_um_desfecho_muda_e_nao_depende_da_ordem():
    a = [{"teste": "a", "desfecho": "passou"}, {"teste": "b", "desfecho": "passou"}]
    b = [{"teste": "a", "desfecho": "passou"}, {"teste": "b", "desfecho": "falhou"}]
    assert rrl.montar_registro(a, "h")["hash_do_resultado"] != rrl.montar_registro(b, "h")["hash_do_resultado"]
    assert rrl.montar_registro(a, "h")["hash_do_resultado"] == rrl.montar_registro(a, "outro codigo")["hash_do_resultado"]


def _arvore_falsa(tmp_path, conteudo="x = 1\n"):
    for rel in rrl.LEITORES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(conteudo.encode("utf-8"))
    return tmp_path


def test_hash_do_codigo_ignora_fim_de_linha_e_muda_com_o_conteudo(tmp_path):
    a = rrl.hash_do_codigo(_arvore_falsa(tmp_path / "a", "x = 1\n"))
    assert a == rrl.hash_do_codigo(_arvore_falsa(tmp_path / "b", "x = 1\n".replace("\n", "\r\n")))
    assert a != rrl.hash_do_codigo(_arvore_falsa(tmp_path / "c", "x = 2\n"))


@pytest.mark.parametrize("estado, trecho", [("ausente", "nao ha registro"), ("reprovado", "nao foi aprovada"), ("velho", "leitores mudaram"),
                                            ("em_dia", None)])
def test_o_registro_acusa_ausencia_reprovacao_e_codigo_que_mudou(tmp_path, estado, trecho):
    raiz = _arvore_falsa(tmp_path)
    if estado != "ausente":
        (raiz / "docs" / "qualidade").mkdir(parents=True)
        reg = rrl.montar_registro([{"teste": "a", "desfecho": "passou" if estado != "reprovado" else "falhou"}], rrl.hash_do_codigo(raiz))
        if estado == "velho":
            reg["hash_do_codigo_testado"] = "0" * 64
        (raiz / "docs" / "qualidade" / "rodada_local_formatos_reais.json").write_text(json.dumps(reg), encoding="utf-8")
    motivo = rrl.registro_atual_cobre_o_codigo(raiz)
    assert (motivo is None) if trecho is None else (trecho in motivo)
