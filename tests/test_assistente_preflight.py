"""Ficha anti-vazamento, pre-voo por linha de comando e a ligacao das regras com assistente.py."""
from __future__ import annotations

import pytest

from guaraci import assistente as ass
from guaraci import assistente_checar as chk
from guaraci import assistente_vazamento as V
from guaraci.assistente_regras import ResumoDados
from guaraci.config import Config


def cfg(**kw):
    c = Config()
    for k, v in kw.items():
        assert hasattr(c, k), k
        setattr(c, k, v)
    return c


def _estados(**kw):
    return {i.id: i.estado for i in V.itens(**kw)}


# ------------------------------------------------------------ ficha
def test_ficha_padrao():
    e = _estados(cfg=cfg())
    assert e["separacao_por_unidade"] == V.OK and e["preproc_nas_dobras"] == V.OK
    assert e["selecao_lvs"] == V.OK and e["holdout_externo"] == V.OK
    # o que o programa nao sabe vira pergunta, nunca "ok"
    assert e["holdout_intocado"] == V.DECLARAR and e["populacao"] == V.DECLARAR
    assert e["replicas"] == V.DECLARAR and e["confundimento"] == V.DECLARAR


def test_ficha_marca_atencao():
    d = ResumoDados(n_leituras=60, n_variaveis=10, n_unidades=20, leituras_por_unidade_max=3)
    e = _estados(cfg=cfg(group_by_mae_id=False, frac_holdout=0.0, selecao_lv_cv_aninhada=False), dados=d,
                 resultado={"_auditoria_critica": ["confundimento_classe_sessao"]})
    for k in ("separacao_por_unidade", "replicas", "selecao_lvs", "holdout_externo", "confundimento"):
        assert e[k] == V.ATENCAO, k


def test_ficha_resultado_tem_precedencia_sobre_config():
    e = _estados(cfg=cfg(group_by_mae_id=True), resultado={"Group-aware (mae_id)": "False", "_auditoria_critica": []})
    assert e["separacao_por_unidade"] == V.ATENCAO and e["confundimento"] == V.OK
    e = _estados(cfg=cfg(), resultado={"Grouping guarantee": "medium"})
    assert e["separacao_por_unidade"] == V.ATENCAO


def test_ficha_markdown():
    t = V.ficha(cfg(frac_holdout=0.0))
    assert t.startswith("# Ficha anti-vazamento") and "10.1016/j.patter.2023.100804" in t
    assert "[!]" in t and "[?]" in t and t.rstrip().endswith("a declarar.")
    assert V.ficha(cfg(), lang="EN").startswith("# Leakage sheet")


def test_ficha_sem_alegacao_de_validacao():
    from guaraci.catalogo import texto_proibido
    for lang in ("PT", "EN"):
        for i in V.itens(cfg(frac_holdout=0.0, group_by_mae_id=False), lang=lang):
            assert not texto_proibido(i.titulo + " " + i.detalhe), (i.id, texto_proibido(i.detalhe))


# ------------------------------------------------------------ linha de comando
def _yaml(tmp_path, texto):
    p = tmp_path / "config.yaml"
    p.write_text(texto, encoding="utf-8")
    return str(p)


def test_checar_config_limpa(tmp_path, capsys):
    assert chk.main([_yaml(tmp_path, "n_permutacoes: 200\n")]) == 0
    assert "CRITICO" not in capsys.readouterr().out


def test_checar_config_com_critico(tmp_path, capsys):
    rc = chk.main([_yaml(tmp_path, "n_permutacoes: 10\nholdout_fracao: 0.0\n"), "--ficha"])
    out = capsys.readouterr().out
    assert rc == 1 and "[CRITICO]" in out and "n_permutacoes: 200" in out and "# Ficha anti-vazamento" in out


def test_checar_config_invalida_mostra_mensagem(tmp_path, capsys):
    assert chk.main([_yaml(tmp_path, "chave_que_nao_existe: 1\n")]) == 2
    assert "chave_que_nao_existe" in capsys.readouterr().err


def test_checar_config_inexistente(tmp_path, capsys):
    assert chk.main([str(tmp_path / "nao.yaml")]) == 2
    assert capsys.readouterr().err


def test_checar_com_dados_reais(tmp_path, capsys):
    import numpy as np
    rng = np.random.default_rng(0)
    wn = np.arange(1000, 1100, 2.0)
    linhas = ["classe,amostra," + ",".join(str(w) for w in wn)]
    for i in range(24):    # 12 amostras fisicas, 2 leituras cada
        linhas.append(f"{'ab'[i % 2]},m{i // 2}," + ",".join(f"{v:.4f}" for v in rng.normal(size=len(wn))))
    (tmp_path / "treino.csv").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    y = _yaml(tmp_path, f"modo_entrada: csv\narquivo_csv: {tmp_path / 'treino.csv'}\ncoluna_classe: classe\n"
                        "coluna_grupo: amostra\nfaixa_min_cm: 1000\nfaixa_max_cm: 1098\nmax_lvs: 15\n")
    assert chk.main([y, "--dados"]) == 0
    out = capsys.readouterr().out
    assert "holdout" in out and "max_lvs" not in out   # o pipeline ja limita as LVs ao treino de cada dobra


def test_checar_dados_ilegiveis(tmp_path, capsys):
    y = _yaml(tmp_path, f"modo_entrada: csv\narquivo_csv: {tmp_path / 'nao.csv'}\ncoluna_classe: classe\n")
    rc = chk.main([y, "--dados"])
    err = capsys.readouterr().err
    assert rc == 2 and "CSV nao encontrado" in err and "GUA-E008" in err


def test_checar_resultado(tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "resumo_modelo.txt").write_text(
        "  Permutation p-value : 0.2000\n  Accuracy (CV) : 0.81\n", encoding="utf-8")
    rc = chk.main([_yaml(tmp_path, "n_permutacoes: 200\n"), "--resultado", str(tmp_path), "--idioma", "EN"])
    out = capsys.readouterr().out
    assert rc == 1 and "[CRITICAL]" in out and "0.200" in out


# ------------------------------------------------------------ assistente.py
@pytest.mark.parametrize("pergunta,tipo,id_", [
    ("o que e VIP", "termo", "vip"),                       # antes caia no grafico
    ("o que e figuras de merito", "termo", "figuras de merito"),
    ("meu Q2 esta baixo", "duvida", "q2_baixo"),
    ("GUA-E005", "erro", "GUA-E005"),
    ("o que significa q residual", "duvida", "t2_q_altos"),
])
def test_respostas_diretas(pergunta, tipo, id_):
    r = ass.responder_estruturado(pergunta)
    assert (r["tipo"], r["id"]) == (tipo, id_), r


def test_negacao_nao_vira_resposta_direta():
    # "nao esta baixo" nao pode responder com confianca "Q2 baixo" (antes o gatilho 'nao' casava)
    r = ass.responder_estruturado("meu Q2 nao esta baixo")
    assert not (r["tipo"] == "duvida" and r["id"] == "q2_baixo")


def test_responder_e_wrapper_do_estruturado():
    for q in ("o que e VIP", "qual a capital da Franca?"):
        assert ass.responder(q) == ass.responder_estruturado(q)["texto"]


def test_diagnosticar_inclui_codigo_de_erro():
    hits = ass.diagnosticar("ValueError: Input X contains NaN.")
    assert any(h.get("tipo") == "erro" and h.get("id") == "GUA-E001" for h in hits), hits


def test_sugerir_usa_o_registro():
    s = ass.sugerir(cfg(n_permutations=10, frac_holdout=0.0))
    txt = [x.texto for x in s]
    assert txt[0].startswith("Corrigir antes de rodar:")
    assert any(x.campo == "holdout_fracao" and x.valor == 0.2 for x in s)
    # o mesmo achado nao aparece duas vezes
    assert len(txt) == len(set(txt))


def test_sugerir_com_dados():
    d = ResumoDados(n_leituras=6, n_variaveis=3, n_unidades=6, n_nan=2)
    assert any("NaN" in x.texto for x in ass.sugerir(cfg(), dados=d))


def test_checar_e_ficha_no_assistente():
    assert "[CRITICO]" in ass.checar(cfg(n_permutations=10))
    assert ass.ficha_vazamento(cfg()).startswith("# Ficha")
