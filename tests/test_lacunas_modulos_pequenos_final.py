"""Ultimas bordas de modulos pequenos: hardware, log do pipeline, assistente, confianca, previa de espectros, utilidades da CLI e relatorios."""
from __future__ import annotations

import builtins
import io
import logging
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

import guaraci.app_logic as al
import guaraci.assistente as ass
import guaraci.catalogo as cat
import guaraci.cli_assistente as ca
import guaraci.cli_logic as cl
import guaraci.cli_comandos as cc
import guaraci.confianca as conf
import guaraci.erros_entrada as ee
import guaraci.formatos_sinteticos as fs
import guaraci.hardware as hw
import guaraci.log as lg
import guaraci.modos_analise as ma
import guaraci.relatorio_cli as rcli
import guaraci.spectra_preview as sp
import guaraci.tecnicas_avancadas as ta
from guaraci.config import Config


def _quebra(msg="falhou", exc=OSError):
    def f(*a, **k):
        raise exc(msg)
    return f


# ---- hardware -----------------------------------------------------------------------------------------------------------------------------------


def test_limite_de_memoria_do_container_ignora_o_valor_de_sem_limite(monkeypatch):
    real = builtins.open

    def falso(caminho, *a, **k):
        if "memory" in str(caminho):
            return io.StringIO(str(2**63))
        return real(caminho, *a, **k)

    monkeypatch.setattr(builtins, "open", falso)
    assert hw._cgroup_ram_limit_gb() is None


def test_sonda_de_hardware_sem_psutil_usa_ctypes_e_cai_nos_padroes_quando_tudo_falha(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    info = hw.hardware_probe()
    assert info["psutil_ok"] is False and info["ram_total_gb"] > 0
    import ctypes
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=types.SimpleNamespace(GlobalMemoryStatusEx=_quebra("sem kernel32"))),
                        raising=False)
    monkeypatch.setattr(os, "cpu_count", _quebra("sem cpu", RuntimeError))
    # Sem nenhuma fonte de RAM (nem /proc/meminfo, nem sysconf): RAM desconhecida
    # (None), nunca um numero padrao inventado (V86).
    monkeypatch.setattr(hw, "_CAMINHO_MEMINFO", "/nao/existe/meminfo")
    monkeypatch.setattr(os, "sysconf", _quebra("sem sysconf", ValueError), raising=False)
    info = hw.hardware_probe()
    assert info["ram_total_gb"] is None and info["ram_livre_gb"] is None
    assert info["ram_fonte"] is None and info["cpu_fisicos"] == 1


def test_sonda_de_hardware_que_falha_por_inteiro_nao_inventa_ram(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", _quebra("sonda", RuntimeError))
    monkeypatch.setattr(hw, "_ram_sem_psutil", lambda: (None, None, None))
    info = hw.hardware_probe()
    assert info["ram_total_gb"] is None and info["ram_livre_gb"] is None


def test_sonda_de_hardware_mantem_o_disco_padrao_quando_a_leitura_do_disco_falha(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, "disk_usage", _quebra("sem disco"))
    info = hw.hardware_probe()
    assert info["psutil_ok"] is True and info["disco_livre_gb"] == 5.0


def test_verificacao_de_ram_assume_ok_quando_a_sonda_quebra(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", _quebra("sonda", RuntimeError))
    assert hw._verificar_ram(1.0, "teste") is True


# ---- log do pipeline ------------------------------------------------------------------------------------------------------------------------------


def test_log_do_pipeline_nao_derruba_o_processo_se_a_saida_falha_e_aceita_arquivo(tmp_path, monkeypatch):
    registro = logging.LogRecord("guaraci", logging.INFO, __file__, 1, "mensagem", None, None)
    manipulador = lg._StdoutHandler()
    erros = []
    monkeypatch.setattr(manipulador, "handleError", lambda r: erros.append(r))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(write=_quebra("pipe fechado"), flush=lambda: None))
    manipulador.emit(registro)
    assert erros == [registro]


def test_log_configura_arquivo_so_uma_vez(tmp_path):
    raiz = logging.getLogger("guaraci")
    antes = list(raiz.handlers)
    try:
        lg.configurar("INFO", str(tmp_path / "a.log"))
        lg.configurar("INFO", str(tmp_path / "b.log"))
        assert sum(isinstance(h, logging.FileHandler) for h in raiz.handlers) == 1
    finally:
        for h in list(raiz.handlers):
            if h not in antes:
                h.close()
                raiz.removeHandler(h)


# ---- LogThreadSafe e leitura do log ----------------------------------------------------------------------------------------------------------------


def test_captura_de_log_continua_quando_o_espelho_quebra_e_nao_repete_figura_nem_aviso():
    class _Quebrado:
        def write(self, s):
            raise OSError("pipe")

        def flush(self):
            raise ValueError("fechado")

    log = al.LogThreadSafe(tee=_Quebrado())
    log.write("Salvo: fig.png\n")
    log.flush()
    assert "Salvo: fig.png" in log.text()
    texto = "  -> pasta/Graficos/fig1.png\n  -> pasta/Graficos/fig1.png\n  [AVISO] a\n  [AVISO] a\n  [AVISO]   \n"
    assert al.figures_completed(texto) == ["fig1"]
    assert al.log_warnings(texto) == ["a"]


# ---- assistente e confianca --------------------------------------------------------------------------------------------------------------------------


def test_assistente_lista_rotas_e_recusa_termo_vazio():
    assert ass.rotas() and ass.explicar("   ") is None


def test_sugestoes_do_assistente_cobrem_erro_de_config_nada_a_sugerir_e_tabela_de_elementos():
    cfg = Config(mode="multielementar", n_permutations=50, default_preprocessing="msc_sg_mc")
    sugestoes = ass.sugerir(cfg, "PT")
    assert any(s.campo == "pre_processamento" and s.valor == "autoscaling" for s in sugestoes)
    limpa = ass.sugerir(Config(mode="sintetico", n_permutations=200), "EN")
    assert limpa and all(isinstance(s.texto, str) and s.texto for s in limpa)


def test_busca_na_confianca_acha_por_nome_parcial_e_recusa_vazio():
    assert conf.buscar_entrada("   ") is None
    tipo, ident, nome, _estado = conf.entradas(None, opcoes=False)[0]
    assert conf.buscar_entrada(nome.split()[0].lower())[0] == "tecnica"
    assert conf.buscar_entrada(ident) == (tipo, ident)


def test_pagina_de_confianca_de_opcao_com_e_sem_selo():
    com_selo = conf.pagina("opcao", "ddsimca", "PT")
    sem_selo = conf.pagina("opcao", "pasta_dados", "EN")
    assert com_selo.splitlines()[0] and "Sem selo proprio" not in com_selo
    assert "No seal of its own" in sem_selo


# ---- erros de entrada, previa de espectros, utilidades da CLI --------------------------------------------------------------------------------------------


def test_idioma_pela_flag_com_igual_e_motivos_de_arquivo_ruim(tmp_path, monkeypatch):
    assert ee.idioma(["--idioma=en"]) == "en"
    ruim = tmp_path / "x.csv"
    ruim.write_bytes(b"\xff\xfe")
    assert ee._causa(UnicodeDecodeError("utf-8", b"", 0, 1, "x"), str(ruim)) == "binario"
    assert ee._causa(ValueError("outro"), str(ruim)) == "formato"
    assert ee._causa(ValueError("x"), str(tmp_path / "nao_existe.csv")) == "formato"


def test_previa_de_espectros_devolve_nada_quando_o_arquivo_nao_serve(tmp_path):
    assert sp.preview_spectra_dx(str(tmp_path / "nao_existe"), 400, 4000) == (None, None, None)
    assert sp.preview_spectra_csv(str(tmp_path / "nao_existe.csv"), "classe", 400, 4000) == (None, None, None)


def test_utilidades_da_cli_cobrem_descricao_sem_espaco_subpastas_e_pasta_inacessivel(tmp_path, monkeypatch):
    assert len(cl.truncate_desc_by_sentence("a" * 80, 20)) <= 20
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.dx").write_text("x", encoding="utf-8")
    assert cl.count_dx(str(tmp_path)) == 1                       # sem .dx na raiz, conta as subpastas imediatas
    (tmp_path / "b.dx").write_text("x", encoding="utf-8")
    assert cl.count_dx(str(tmp_path)) == 1                       # com .dx na raiz, so' a raiz
    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    cl._CACHE_DX.clear()                        # a contagem guardada (3 s) esconderia a falha de leitura
    monkeypatch.setattr(os, "scandir", _quebra("sem permissao"))
    assert cl.count_dx(str(tmp_path)) == 0


def test_paleta_aplica_estilo_e_mapa_de_cor_do_catalogo_e_tolera_estilo_desconhecido(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt.style, "use", _quebra("estilo desconhecido"))
    monkeypatch.setitem(ca.PALETAS_COR, "teste_cmap", {"cores": ["#111111", "#222222"], "cmap": "viridis"})
    # `apply_palette` muda o GLOBAL `paleta_cores._ATIVA`; o setitem acima so' restaura o dicionario do catalogo
    import guaraci.paleta_cores as pc
    monkeypatch.setattr(pc, "_ATIVA", None)
    with __import__("matplotlib").rc_context():
        ca.apply_palette("teste_cmap")
        assert plt.rcParams["image.cmap"] == "viridis"


def test_escritores_de_formato_sintetico_recusam_formas_incompativeis(tmp_path):
    with pytest.raises(ValueError):
        fs.escrever_andi_ms(str(tmp_path / "a.cdf"), [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        fs.escrever_eem_dat(str(tmp_path / "e.dat"), [300, 310], [400, 410, 420], np.zeros((2, 2)))


def test_modos_de_analise_sem_grupo_da_chave_valem_sempre():
    cfg = Config()
    assert ma.selecionado(cfg, "chave_que_nao_esta_em_nenhum_grupo") is True


def test_tema_escuro_declarado_pela_variavel_de_ambiente(monkeypatch):
    import guaraci.guaraci_theme as th
    monkeypatch.setenv("GUARACI_TERMINAL", "escuro")
    assert th.terminal_claro() is False
    monkeypatch.setenv("GUARACI_TERMINAL", "claro")
    assert th.terminal_claro() is True


# ---- fusao multibloco, relatorio e subcomandos -----------------------------------------------------------------------------------------------------


def test_fusao_multibloco_com_uma_amostra_nao_separa_calibracao_de_validacao():
    X = np.ones((1, 4))
    with pytest.raises(ValueError, match="insuficiente"):
        ta.rodar_fusao_multibloco({"a": X, "b": X}, np.array([1.0]))


def test_fatores_categoricos_ignoram_coluna_de_outro_tamanho_e_de_valores_nao_hashaveis():
    rotulos = np.array(list("aabb"))
    meta = pd.DataFrame({"lista": [[1], [2], [3], [4]], "ok": list("xyxy")})
    disponiveis = ta.fatores_categoricos_disponiveis(rotulos, meta)
    assert "lista" not in disponiveis and "ok" in disponiveis
    assert "sessao" not in ta.fatores_categoricos_disponiveis(rotulos, pd.DataFrame({"sessao": ["s1", "s2"]}))


def test_relatorio_cli_recusa_argumento_invalido_pasta_ausente_e_conta_formato_que_falha(tmp_path, monkeypatch, capsys):
    assert rcli.executar(["--formato"]) == 2
    assert rcli.executar([]) == 2
    pasta = tmp_path / "run"
    (pasta / "Relatorios").mkdir(parents=True)
    (pasta / "Relatorios" / "resumo_modelo.txt").write_text("x", encoding="utf-8")
    from guaraci import reports
    for nome_fn, _usa, _ext in rcli.FORMATOS.values():
        monkeypatch.setattr(reports, nome_fn, _quebra("gerador quebrou", RuntimeError))
    assert rcli.executar([str(pasta), "--formato", "pdf"]) == 1
    assert "FALHOU" in capsys.readouterr().err


def test_subcomandos_da_cli_cobrem_exemplos_executaveis_ajuda_por_parecido_e_bordas(tmp_path, monkeypatch, capsys):
    from guaraci import assistente as a
    monkeypatch.chdir(tmp_path)
    for cenario, executor in (("comparar", "_rodar_anova_exemplo"), ("monitorar", "_rodar_monitorar_exemplo"),
                              ("transferir", "_rodar_transferir_exemplo"), ("vip", "_rodar_vip_exemplo")):
        monkeypatch.setattr(cc, executor, lambda pasta: 0)
        assert cc._cmd_exemplo([cenario, str(tmp_path / cenario), "--rodar"]) == 0
    monkeypatch.setattr(a, "explicar", lambda termo, lang="PT": None)
    monkeypatch.setattr(a, "buscar", lambda termo, lang, n: [{"tipo": "t", "titulo": "T", "id": "i", "trecho": "x"}])
    assert cc._cmd_ajuda(["algo"]) == 0
    monkeypatch.setattr(a, "buscar", lambda termo, lang, n: [])
    assert cc._cmd_buscar(["xyz"]) == 0 and cc._cmd_ajuda(["xyz"]) == 2
    assert cc._cmd_assistente([]) == 0
    with pytest.raises(ValueError):
        cc.subcomando("nao_existe", [])
    monkeypatch.setattr(cat, "grupos", lambda: [{"id": "vazio", "nome": ("V", "E"), "faz": ("f", "e")}])
    assert "Mapa do GUARACI" in cc._mapa_de_ajuda()
    capsys.readouterr()
