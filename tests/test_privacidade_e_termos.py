"""Rascunhos do aviso de privacidade e dos termos de uso do demonstrativo hospedado.

O texto descreve o que o codigo faz (limites de envio, limpeza de sessao) e nao afirma conformidade legal. Enquanto o mantenedor
nao aprova os rascunhos, eles nao sao citados no README nem no aplicativo (o teste `test_rascunhos_ainda_nao_sao_publicados` registra
isso: ao aprovar, remova-o e ponha o link).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from guaraci import docs_pacote as dp
from guaraci import web_seguranca as ws

RAIZ = Path(__file__).resolve().parents[1]
DOCS = {"privacidade": RAIZ / "docs" / "usuario" / "privacidade.md", "termos": RAIZ / "docs" / "usuario" / "termos_de_uso.md"}
CONTATO = "GitHub Discussions do repositório"


def _texto(chave: str) -> str:
    return DOCS[chave].read_text(encoding="utf-8")


def _milhar(n: int) -> str:
    return f"{n:,}".replace(",", " ")


@pytest.mark.parametrize("chave", list(DOCS))
def test_documento_existe_e_esta_no_pacote_como_topico(chave):
    assert DOCS[chave].is_file()
    assert chave in dp.TOPICOS
    assert dp.ler(chave).replace("\r\n", "\n") == _texto(chave).replace("\r\n", "\n")


@pytest.mark.parametrize("trecho", [
    "## 1. O que o demonstrativo recebe", "## 2. Para que os dados são usados", "## 3. Onde ficam e por quanto tempo",
    "## 4. Registros de erro", "## 5. Hospedagem por terceiro", "## 6. O que você deve e não deve enviar",
    "fora do Brasil", "6 horas", "Não envie dados pessoais", "registro do servidor", "não é uma declaração de conformidade",
])
def test_privacidade_tem_as_secoes_e_as_limitacoes(trecho):
    assert trecho in _texto("privacidade")


@pytest.mark.parametrize("trecho", [
    "## 2. Não é laudo", "## 3. Sem garantia", "GPL-3.0", "seções 15 e 16", "como estão", "Não é laudo".lower(),
    "Não enviar dados pessoais", "terceiros",
])
def test_termos_tem_as_secoes_e_a_isencao(trecho):
    assert trecho.lower() in _texto("termos").lower()


@pytest.mark.parametrize("chave", list(DOCS))
def test_nao_afirma_conformidade_nem_traz_contato_real(chave):
    t = _texto(chave)
    minusculo = t.lower()
    for proibido in ("em conform" + "idade", "LGPD " + "garante", "conforme a LG" + "PD", "GDPR " + "compliant"):
        assert proibido.lower() not in minusculo, proibido
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", t), "e-mail no texto"
    assert not re.search(r"https?://", t), "endereco externo no texto"
    assert CONTATO in t and "definir no lote" not in t, "o contato e' o das Discussions do repositorio"


def test_os_limites_do_texto_sao_os_do_codigo():
    for chave in DOCS:
        t = _texto(chave)
        assert "50 MB" in t
        assert _milhar(ws.MAX_LINHAS_TABELA_SERVIDOR) in t
        assert _milhar(ws.MAX_COLUNAS_TABELA_SERVIDOR) in t
        assert "20 milhões" in t and ws.MAX_CELULAS_TABELA_SERVIDOR == 20_000_000
    assert ws.IDADE_MAXIMA_HORAS == 6.0
    toml = (RAIZ / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert re.search(r"^maxUploadSize\s*=\s*50\b", toml, flags=re.M)
    assert re.search(r"^gatherUsageStats\s*=\s*false", toml, flags=re.M)


def test_o_texto_nao_cita_pasta_variavel_de_ambiente_nem_nome_de_plataforma():
    for chave in DOCS:
        t = _texto(chave)
        for proibido in ("GUARACI_SESSION_DIR", "GUARACI_LOCAL_MODE", "/tmp", "streamlit.app", "Community Cloud"):
            assert proibido not in t, proibido


def test_o_envio_no_aplicativo_diz_para_nao_enviar_dados_pessoais():
    frase = "Do not upload personal or confidential data."
    app = (RAIZ / "src" / "guaraci" / "app_web.py").read_text(encoding="utf-8")
    assert re.search(re.escape(f'"{frase}": {{"PT": "Não envie dados pessoais', ), app)
    for tela in ("dados.py", "predicao.py"):
        fonte = (RAIZ / "src" / "guaraci" / "app_tabs" / tela).read_text(encoding="utf-8")
        assert f'T("{frase}")' in fonte, tela


def test_rascunhos_ainda_nao_sao_publicados():
    """Pendente de aprovacao do mantenedor: sem link no README nem no app. Ao aprovar, remova este teste."""
    for rel in ("README.md", "README.pt-BR.md", "src/guaraci/app_tabs/sobre.py", "src/guaraci/app_tabs/inicio.py"):
        t = (RAIZ / rel).read_text(encoding="utf-8")
        assert "privacidade.md" not in t and "termos_de_uso.md" not in t, rel
