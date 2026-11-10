"""Contrato do catalogo declarativo unico (guaraci.catalogo)."""
import subprocess
import sys
from pathlib import Path

import pytest

from guaraci import catalogo as cat
from guaraci import modos_analise as ma
from guaraci.catalogo_validacao import EVIDENCIAS, SELOS

RAIZ = Path(__file__).resolve().parents[1]


def test_catalogo_cumpre_o_contrato_de_texto_e_referencias():
    assert cat.validar_catalogo() == []


def test_textos_ao_usuario_nao_carregam_rotulo_de_trabalho_de_curso():
    """Vistoria 2026-10 (V108): exemplos das opcoes e as telas 'Sobre' (app e
    CLI) falam do produto, nao de um trabalho de curso ou projeto de pesquisa."""
    import re
    rx = re.compile(r"\bTCC\b|research project|pesquisa em quimiometria", re.IGNORECASE)
    for k, o in cat.OPCOES.items():
        for lingua in o.get("exemplos") or ():
            for v in lingua.values():
                assert not rx.search(str(v)), k
    for rel in ("src/guaraci/app_tabs/sobre.py", "src/guaraci/guaraci.py"):
        assert not rx.search((RAIZ / rel).read_text(encoding="utf-8")), rel


def test_toda_opcao_do_config_tem_entrada_completa_no_catalogo(pq):
    spec = {s["key"] for s in pq._CONFIG_SPEC}
    assert spec <= set(cat.OPCOES), sorted(spec - set(cat.OPCOES))
    for k in spec:
        o = cat.opcao(k)
        assert o["faz"][0] and o["faz"][1] and o["use"] and o["nao_use"], k


def test_descricao_do_spec_vem_do_catalogo_e_nao_diverge(pq):
    for s in pq._CONFIG_SPEC:
        assert s["desc"] == cat.faz(s["key"], "PT"), s["key"]


def test_visoes_historicas_sao_derivadas_do_catalogo():
    import guaraci.cli_assistente as ca
    assert ca.MENU_FIELDS == cat.menu_fields()
    assert ca.HELP_DB.keys() == cat.OPCOES.keys()
    assert ca.FIELD_NAMES.keys() == cat.OPCOES.keys()
    for k, h in ca.HELP_DB.items():
        assert h["PT"]["desc"] == cat.faz(k, "PT") and h["EN"]["desc"] == cat.faz(k, "EN")


def test_teclas_de_tela_sao_unicas_e_nao_colidem_com_acoes():
    teclas = [t["tecla"].upper() for t in cat.telas()]
    assert len(teclas) == len(set(teclas))
    from guaraci.catalogo_dados import ACOES_MENU
    assert not set(teclas) & set(ACOES_MENU)


def test_todo_grupo_usado_existe_e_todo_grupo_de_tela_tem_tela():
    ids = {g["id"] for g in cat.grupos()}
    assert {t["grupo"] for t in cat.telas()} <= ids
    assert {t["grupo"] for t in cat.tecnicas()} <= ids


def test_tecnicas_do_registro_e_do_catalogo_coincidem():
    from guaraci.technique_registry import REGISTRY
    assert {e.id for e in REGISTRY} == set(cat.TECNICAS)
    for e in REGISTRY:   # o registro nao tem texto proprio: vem do catalogo
        assert e.quando_usar == cat.TECNICAS[e.id]["use"][0] and e.limitacao == cat.TECNICAS[e.id]["nao_use"][0]


def test_graficos_do_catalogo_cobrem_exatamente_os_de_modos_analise():
    do_pipeline = {k for k, v in cat.GRAFICOS.items() if "comando" not in v}     # os de comando (monitorar) ficam fora do pipeline
    assert do_pipeline == set(ma._FIG_OBJETIVOS) == set(ma._FIG_DESCRICAO)
    assert all(v["comando"] in cat.COMANDOS for v in cat.GRAFICOS.values() if "comando" in v)
    # conjunto basico pequeno por objetivo (3 a 5 essenciais; quantificacao tem 1)
    for obj in ("exploratorio", "classificacao"):
        n = sum(1 for g in cat.graficos() if obj in g["padrao"] and obj in g["objetivos"])
        assert 3 <= n <= 5, (obj, n)


def test_evidencias_apontam_para_testes_que_existem():
    """Toda evidencia aponta teste que existe em `tests/` (vai junto e o usuario reproduz), sem excecao."""
    for eid, e in EVIDENCIAS.items():
        assert e["testes"], eid
        for t in e["testes"]:
            assert (RAIZ / "tests" / t).is_file(), (eid, t)


def test_selo_so_com_evidencia_que_vai_junto_e_nenhum_validado():
    """Politica de dados de terceiros + D8 variante (b), provisoria: o selo se apoia so' em simulacao/testes."""
    for sid, s in SELOS.items():
        assert s["evid"], sid
        assert s["estado"] != "validado", sid
        assert all(EVIDENCIAS[e]["tipo"] in ("simulacao", "testes") for e in s["evid"]), sid
    assert {e["tipo"] for e in EVIDENCIAS.values()} <= {"simulacao", "testes"}


def test_validar_catalogo_reprova_evidencia_que_nao_vai_junto_e_selo_validado(monkeypatch):
    from guaraci import catalogo_validacao as cv
    ev = dict(cv.EVIDENCIAS)
    ev["x"] = dict(ev["mcr_testes"], tipo="dataset_publico")
    monkeypatch.setattr(cat, "EVIDENCIAS", ev)
    monkeypatch.setattr(cat, "SELOS", dict(cv.SELOS, mcr_als=dict(cv.SELOS["mcr_als"], estado="validado")))
    probs = cat.validar_catalogo()
    assert any("evidencia x" in p for p in probs) and any("selo mcr_als" in p and "validado" in p for p in probs)


def test_confianca_e_gerada_do_registro_e_reproduzivel_com_o_pacote():
    c = cat.confianca("tecnica", "pls_da")
    assert c["estado"] == "em_validacao" and c["limites"] and "ponto de partida" in c["calibracao"]
    assert all(r.startswith("pytest tests/") for e in c["evidencias"] for r in e["reproduzir"])
    assert all(not e["nao_executado"] for e in c["evidencias"])
    assert cat.confianca("tecnica", "ddsimca")["estado"] == "sem_validacao_publica"
    assert cat.confianca("opcao", "dpi")["selo"] is None


def test_texto_proibido_pega_relato_de_validacao_e_hedge():
    assert cat.texto_proibido("nos testes obtivemos 95%")
    assert cat.texto_proibido("validado em dataset publico")
    assert cat.texto_proibido("pode ajudar a escolher")
    assert cat.texto_proibido("Medido: 4 processos") is not None
    assert cat.texto_proibido("Compara previsto contra medido") is None
    assert cat.texto_proibido("Define a pasta de saida.") is None


def test_catalogo_nao_puxa_bibliotecas_pesadas():
    cod = ("import sys, guaraci.catalogo as c; c.validar_catalogo();"
           "print(any(m in sys.modules for m in ('numpy','sklearn','scipy','matplotlib','pandas')))")
    r = subprocess.run([sys.executable, "-c", cod], capture_output=True, text=True, cwd=str(RAIZ),
                       env={**__import__("os").environ, "PYTHONPATH": str(RAIZ / "src")}, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


@pytest.mark.parametrize("canal, esperado_tag", [("cli", True), ("web", False)])
def test_alias_de_nome_execucao_aparece_so_no_canal_certo(canal, esperado_tag):
    campos = cat.campos_da_tela("projeto", canal=canal)
    assert ("tag" in campos) is esperado_tag
    assert ("nome_execucao" in campos) is (not esperado_tag)
