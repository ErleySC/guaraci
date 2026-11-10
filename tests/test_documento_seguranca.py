"""SECURITY.md só afirma o que existe: prazo como meta, versões suportadas, canal sem promessa, endurecimento real (V14)."""
from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
TEXTO = (RAIZ / "SECURITY.md").read_text(encoding="utf-8")


def test_tem_as_secoes_e_o_resumo_em_ingles():
    for secao in ("## Como relatar uma vulnerabilidade", "## Modelo de ameaça", "### Endurecimento do modo servidor"):
        assert secao in TEXTO, secao
    assert "English summary" in TEXTO and "business days" in TEXTO


def test_prazo_e_meta_e_igual_ao_do_suporte():
    assert "meta" in TEXTO and "7 dias úteis" in TEXTO
    assert "Resposta esperada em até 7 dias." not in TEXTO
    assert "7 dias úteis" in (RAIZ / "SUPPORT.md").read_text(encoding="utf-8")


def test_versoes_suportadas_sao_a_ultima_publicada_e_o_master():
    assert "a última versão publicada e o ramo `master`" in TEXTO
    assert "apenas a mais recente do ramo" not in TEXTO


def test_aponta_o_canal_privado_do_github_e_nao_traz_endereco_de_email():
    assert "Se o repositório tiver" not in TEXTO
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", TEXTO)
    assert "ainda não há um canal privado" not in TEXTO
    assert "Report a vulnerability" in TEXTO and "/security/advisories/new" in TEXTO


def test_descreve_o_endurecimento_que_o_codigo_implementa():
    for trecho in ('showErrorDetails = "none"', "50 MB", "`.csv`, `.txt` ou `.tsv`", "1000", "2000", "0o600", "mesmos bytes"):
        assert trecho in TEXTO, trecho
    assert "não recebe" in TEXTO and "`config.toml`" in TEXTO       # `streamlit run` direto não traz o config.toml


def test_os_numeros_do_texto_batem_com_a_configuracao_e_o_codigo():
    toml = (RAIZ / ".streamlit/config.toml").read_text(encoding="utf-8")
    assert re.search(r"maxUploadSize\s*=\s*50\b", toml) and 'showErrorDetails = "none"' in toml
    assert "maxUploadSize" in (RAIZ / "Dockerfile").read_text(encoding="utf-8")


def test_comentarios_do_app_nao_dizem_que_o_bat_liga_o_modo_local():
    bat = (RAIZ / "iniciar_guaraci.bat").read_text(encoding="utf-8", errors="replace")
    assert "GUARACI_LOCAL_MODE" not in bat                          # o .bat só abre a CLI
    for rel in ("src/guaraci/app_web.py", "src/guaraci/web_seguranca.py"):
        fonte = (RAIZ / rel).read_text(encoding="utf-8")
        assert "(iniciar_guaraci.bat e" not in fonte and "`iniciar_guaraci.bat` e `guaraci web` ligam" not in fonte, rel
