# -*- coding: utf-8 -*-
"""Tecla "?" nas listas numeradas (cada valor explicado), perguntas soltas, varias etapas no Personalizado,
remocao de codigos da Codificacao DX e a tela [3.5] no molde das demais."""
from __future__ import annotations

import json

import guaraci.guaraci as g
from guaraci import catalogo_perguntas as cp
from guaraci import catalogo_valores as cv


def _tela(cli, monkeypatch, *teclas, modo="avancado"):
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    g._STATE["modo_usuario"] = modo
    cli.dizer(*teclas, "Q")
    try:
        g.main([])
    except SystemExit:
        pass
    return cli.saida()


def test_todo_valor_de_todo_campo_de_escolha_tem_texto_em_pt_e_en():
    for f in g._CONFIG_SPEC:
        for op in f.get("opcoes") or ():
            for lang in ("PT", "EN"):
                txt = cv.texto_valor(f["key"], op, lang)
                assert txt and len(txt) > 15, (f["key"], op, lang)
    ok = {f["key"]: {str(o) for o in f.get("opcoes") or ()} for f in g._CONFIG_SPEC}
    for chave, vals in cv.VALORES.items():
        assert set(vals) <= ok[chave], (chave, set(vals) - ok[chave])        # nenhum valor explicado que nao existe
        assert all(a and b and a != b for a, b in vals.values()), chave


def test_interrogacao_na_lista_de_pre_processamento_explica_cada_item(cli, monkeypatch):
    saida = _tela(cli, monkeypatch, "2.5", "1", "?", "7", "", "V", "V")
    assert "Usa as etapas marcadas" in saida and "ordem fixa" in saida          # Personalizado explicado
    saida = _tela(cli, monkeypatch, "2.5", "1", "?", "", "", "V", "V")          # Enter = o campo inteiro, todos os valores
    for rotulo in ("Autoscaling", "Mean-centering", "PQN+MC", "AirPLS+SG+MC"):
        assert rotulo + ":" in saida.replace("\n", " ") or rotulo in saida
    assert "Corrige espalhamento multiplicativo" in saida


def test_interrogacao_em_campo_sim_nao_explica_sim_e_nao(cli, monkeypatch):
    saida = _tela(cli, monkeypatch, "2.5", "2", "?", "1", "", "V", "V")
    assert "Liga:" in saida


def test_personalizado_aceita_varias_etapas_e_o_pipeline_as_aplica(cli, monkeypatch):
    from guaraci.preprocessamento import build_preprocessor
    cfg = g.Config()
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    cli.dizer("7", "s")                      # pre_processamento: [7] Personalizado, com a confirmacao dos campos analiticos
    assert g._editar_campo(cfg, "pre_processamento")
    cli.dizer("2,5,6", "s")                  # etapas: [2] snv, [5] sg, [6] mc
    assert g._editar_campo(cfg, "etapas_pre_processamento")
    assert tuple(cfg.etapas_pre_processamento) == ("snv", "sg", "mc")
    assert [n for n, _ in build_preprocessor(cfg).steps] == ["snv", "sg", "mc"]
    cli.dizer("snv,mc", "s")                 # tambem por nome
    assert g._editar_campo(cfg, "etapas_pre_processamento")
    assert [n for n, _ in build_preprocessor(cfg).steps] == ["snv", "mc"]


def test_pergunta_solta_responde_interrogacao_com_texto_proprio(cli):
    assert cp.texto_pergunta("Numero de componentes (Enter=2)", "PT")
    assert cp.texto_pergunta("Number of components (Enter=2)", "EN")
    assert cp.texto_pergunta("pergunta que ninguem cadastrou") is None
    for trechos, (pt, en) in cp.PERGUNTAS:
        assert pt and en and pt != en and all(t == t.lower() for t in trechos)
    cli.dizer("?", "2")
    assert g._ask("  [x]Numero de componentes (Enter=2):[/x] ") == "2"
    assert "Quantos componentes" in cli.saida()
    cli.dizer("?", "4")                                      # sem texto proprio: mostra a pergunta e a ajuda da tela
    assert g._ask("  [x]Uma pergunta nova qualquer sem cadastro:[/x] ") == "4"
    assert "Uma pergunta nova qualquer sem cadastro" in cli.saida()


def test_codificacao_remove_um_codigo_e_todos(cli, monkeypatch):
    g._CODIGOS_PATH.write_text(json.dumps({"AAA": "Classe A", "BBB": "Classe B"}), encoding="utf-8")
    _tela(cli, monkeypatch, "2.9", "5", "1", "s", "V", "V")
    assert json.loads(g._CODIGOS_PATH.read_text(encoding="utf-8")) == {"BBB": "Classe B"}
    _tela(cli, monkeypatch, "2.9", "5", "bbb", "n", "V", "V")                    # recusa: nada muda
    assert json.loads(g._CODIGOS_PATH.read_text(encoding="utf-8")) == {"BBB": "Classe B"}
    _tela(cli, monkeypatch, "2.9", "5", "T", "s", "V", "V")
    assert json.loads(g._CODIGOS_PATH.read_text(encoding="utf-8")) == {}
    saida = _tela(cli, monkeypatch, "2.9", "5", "V", "V")
    assert "Nenhum codigo do usuario" in saida


def test_codificacao_abre_sem_importar_o_pipeline():
    import subprocess
    import sys
    cod = ("import sys, guaraci.guaraci as g\n"
           "g.pq.CODIGO_ESPECIE\n"
           "print('guaraci.pipeline' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", cod], capture_output=True, text=True, timeout=120)
    assert r.stdout.strip() == "False", r.stderr


def test_tela_3_5_usa_o_molde_das_demais_telas_da_secao(cli, monkeypatch):
    saida = _tela(cli, monkeypatch, "3.5", "V", "V")
    assert "│   [1]" in saida and "ASCA" in saida and "Separa a variacao" in saida     # tabela numerada dentro do painel
    assert "[10]" in saida and "3 CSV" in saida and "? ajuda | G Guaraci" in saida
    assert "● Visual" not in saida.split("Tecnicas especiais")[-1].split("Opcao")[0]   # sem a legenda de risco, que aqui nao se aplica
    assert "Precisa de" in saida and "O que faz" in saida            # a coluna verde tem titulo: "Precisa de"
    assert "Testar correcao de sinal" in saida and "Transferir calibracao DS/PDS" in saida
