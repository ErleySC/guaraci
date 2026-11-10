"""Comandos `guaraci ajuda|buscar|assistente` e o assistente nas telas."""
import pytest

import guaraci.guaraci as g
from guaraci import catalogo as cat
from guaraci import cli_comandos as cmd


def test_ajuda_sem_argumento_mostra_o_mapa_de_todos_os_grupos_com_conteudo(capsys):
    assert cmd.subcomando("ajuda", []) == 0
    saida = capsys.readouterr().out
    for gr in cat.grupos():
        if cat.telas(gr["id"]) or cat.comandos(gr["id"]) or [t for t in cat.tecnicas() if t["grupo"] == gr["id"]]:
            assert gr["nome"][0] in saida, gr["id"]
    for t in cat.telas():
        assert f"[{t['tecla']}] {t['nome'][0]}" in saida


@pytest.mark.parametrize("topico", ["pls_da", "dpi", "confusao", "Pre-processamento", "anova", "censura"])
def test_ajuda_topico_explica_no_gabarito(capsys, topico):
    assert cmd.subcomando("ajuda", [topico]) == 0
    assert capsys.readouterr().out.strip()


def test_ajuda_topico_inexistente_sugere_parecidos_ou_erro_de_uso(capsys):
    assert cmd.subcomando("ajuda", ["zzqq"]) == 2
    assert "guaraci ajuda" in capsys.readouterr().err


def test_buscar_exige_termo_e_acha_por_palavra_chave(capsys):
    assert cmd.subcomando("buscar", []) == 2
    capsys.readouterr()
    assert cmd.subcomando("buscar", ["unidade", "independente"]) == 0
    assert "unidade" in capsys.readouterr().out.lower()


def test_assistente_responde_pergunta_livre_sem_rede(capsys):
    assert cmd.subcomando("assistente", ["o", "que", "e", "pseudo-replicacao"]) == 0
    assert "Tratar replicas" in capsys.readouterr().out


def test_main_encaminha_os_tres_comandos(capsys):
    g.main(["buscar", "censura"])
    assert "censura" in capsys.readouterr().out.lower()
    g.main(["ajuda", "dpi"])
    assert "resolucao" in capsys.readouterr().out.lower()


def test_assistente_nas_telas_rota_explicar_sugerir_diagnosticar_buscar(monkeypatch):
    monkeypatch.setattr(g, "_cls", lambda: None)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)

    def roda(*respostas):
        it = iter(respostas)
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(it, ""))
        with g.console.capture() as cap:
            g._abrir_assistente("teste", g.Config())
        return cap.get()

    assert "Rota recomendada" in roda("7", "2", "2")                       # classificar + csv espectral
    assert "Nao ha rota" in roda("7", "4", "6")                            # quantificar + cubo HSI
    assert "Tratar replicas" in roda("8", "pseudo-replicacao")
    assert "ANOVA recusou" in roda("10", "ANOVA recusada: sem unidade")
    assert "icp_politica_censura" in roda("11", "censura")


def test_sugerir_na_tela_so_aplica_quando_o_usuario_escolhe(monkeypatch):
    monkeypatch.setattr(g, "_cls", lambda: None)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    cfg = g.Config()
    cfg.n_permutations = 20
    respostas = iter(["9", ""])                                           # sugerir, nao aplicar nada
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas, ""))
    g._abrir_assistente("teste", cfg)
    assert cfg.n_permutations == 20
    alvo = {"i": None}

    def achar():
        from guaraci import assistente as ass
        for i, x in enumerate(ass.sugerir(cfg), 1):
            if x.campo == "n_permutacoes":
                return str(i)
    alvo["i"] = achar()
    respostas = iter(["9", alvo["i"], ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas, ""))
    g._abrir_assistente("teste", cfg)
    assert cfg.n_permutations == 200
