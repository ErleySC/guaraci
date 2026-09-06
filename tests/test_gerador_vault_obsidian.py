# -*- coding: utf-8 -*-
"""Testes de `scripts/gerar_vault_obsidian.py` e da guarda de privacidade
compartilhada (`scripts/privacidade_amostras.py`, Passo 169).

Contra-prova exigida pela instrução: um cenário SINTÉTICO contendo um
identificador proibido deve fazer a guarda falhar -- sem isso, um guarda
quebrado (ex.: regex que nunca casa nada) passaria como "limpo" e o vault
gerado poderia vazar dado real sem ninguém perceber.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import gerar_vault_obsidian as gvo  # noqa: E402
from privacidade_amostras import (  # noqa: E402
    VazamentoDePrivacidade,
    checar_conteudos_ou_falhar,
)


def _identificador_sintetico(especie: str, ano: str, sep: str = "-") -> str:
    """Monta um identificador em tempo de execução (nunca como literal no
    código-fonte) -- mesma técnica de `tests/test_sem_identificador_real.py`
    (`_id`): um literal com ano real neste arquivo seria pego pela própria
    varredura de `test_sem_identificador_real.py`, já que ela varre TODO
    arquivo versionado, este incluído."""
    return f"{especie}{sep}04-11-{ano}"


# ─────────────────────────────────────────────────────────────────────────
#  Contra-prova da guarda de privacidade (Passo 169)
# ─────────────────────────────────────────────────────────────────────────

def test_guarda_acusa_identificador_real_no_conteudo():
    ident = _identificador_sintetico("CAP", "2020")
    plano = {
        "60-Achados/nota.md": f"Achado sobre a amostra {ident}_T1.",
    }
    with pytest.raises(VazamentoDePrivacidade):
        checar_conteudos_ou_falhar(plano)


def test_guarda_acusa_identificador_real_no_nome_do_arquivo():
    ident = _identificador_sintetico("AND", "2020")
    plano = {
        f"60-Achados/{ident}_T1.md": "conteúdo sem nada de especial",
    }
    with pytest.raises(VazamentoDePrivacidade):
        checar_conteudos_ou_falhar(plano)


def test_guarda_acusa_caminho_absoluto_windows():
    plano = {
        "10-Tecnicas/nir.md": r"gerado a partir de C:\Users\alguem\projeto\docs\x.md",
    }
    with pytest.raises(VazamentoDePrivacidade):
        checar_conteudos_ou_falhar(plano)


def test_guarda_acusa_caminho_absoluto_unix():
    plano = {
        "10-Tecnicas/nir.md": "gerado a partir de /home/alguem/projeto/docs/x.md",
    }
    with pytest.raises(VazamentoDePrivacidade):
        checar_conteudos_ou_falhar(plano)


def test_guarda_aceita_identificador_sentinela_2099():
    plano = {
        "60-Achados/nota.md": "Identificador de exemplo: CAP-04-11-2099_T1.",
    }
    checar_conteudos_ou_falhar(plano)  # não levanta


def test_guarda_nao_acusa_conteudo_legitimo():
    plano = {
        "10-Tecnicas/raman.md": "RMSEP 0,144 %m/m; versão 31.9.0; ISO-8601 2026-09-05.",
        "20-Modulos/conformal.py.md": "conformal.py -- predição conformal one-class.",
    }
    checar_conteudos_ou_falhar(plano)  # não levanta


# ─────────────────────────────────────────────────────────────────────────
#  O gerador real produz um plano limpo (integra guarda + geração)
# ─────────────────────────────────────────────────────────────────────────

def test_plano_real_do_gerador_passa_na_guarda():
    plano, _contagens, _avisos = gvo.montar_plano()
    checar_conteudos_ou_falhar(plano)  # não levanta


def test_plano_real_nao_tem_wikilink_quebrado():
    """Achado real (2026-09-06): um trecho de docs/PROGRESSO.md descrevendo
    a sintaxe `[[nome.py]]` em prosa (não como link de verdade) virou, ao
    ser copiado para uma nota, um wikilink de verdade apontando para uma
    nota "nome.py" inexistente. Este teste garante que isso não volta a
    passar despercebido: todo `[[alvo]]`/`[[alvo|rótulo]]` do plano tem
    que resolver a uma chave real do próprio plano."""
    plano, _contagens, _avisos = gvo.montar_plano()
    stems_existentes = {Path(rel).stem for rel in plano}
    quebrados = []
    for rel, conteudo in plano.items():
        for m in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", conteudo):
            alvo = m.group(1).strip()
            if alvo not in stems_existentes:
                quebrados.append((rel, alvo))
    assert not quebrados, f"wikilink(s) quebrado(s) no plano: {quebrados}"


def test_plano_real_tem_as_categorias_esperadas():
    plano, contagens, _avisos = gvo.montar_plano()
    # cli_assistente.TECNICAS (11 chaves) + modalidades fora do catálogo
    # (Config.mode: "imagem"/"hsi" -- ver MODOS_FORA_DO_CATALOGO)
    assert contagens["10-Tecnicas"] == len(gvo.parse_tecnicas_catalog()) + len(
        gvo.MODOS_FORA_DO_CATALOGO)
    assert contagens["20-Modulos"] > 0
    assert contagens["30-Conceitos"] > 0
    assert contagens["40-Validacoes"] > 0
    assert all(rel.startswith(("00-MOC/", "10-Tecnicas/", "20-Modulos/",
                                "30-Conceitos/", "40-Validacoes/", "50-Decisoes/",
                                "60-Achados/", "90-Canvas/", "README-VAULT.md",
                                "Estado-Atual.md"))
               for rel in plano)


# ─────────────────────────────────────────────────────────────────────────
#  Escrita em disco: idempotência e pasta protegida
# ─────────────────────────────────────────────────────────────────────────

def test_escrever_vault_e_idempotente(tmp_path):
    plano = {"10-Tecnicas/x.md": "conteudo fixo", "20-Modulos/y.py.md": "outro"}
    e1, _i1, _a1 = gvo.escrever_vault(tmp_path, plano)
    e2, i2, _a2 = gvo.escrever_vault(tmp_path, plano)
    assert e1 == 2
    assert e2 == 0
    assert i2 == 2


def test_pasta_protegida_nunca_e_tocada(tmp_path):
    plano1 = {"10-Tecnicas/x.md": "v1"}
    gvo.escrever_vault(tmp_path, plano1)
    canario = tmp_path / gvo._PASTA_PROTEGIDA / "minha-nota.md"
    canario.write_text("não mexa aqui", encoding="utf-8")

    plano2 = {"10-Tecnicas/x.md": "v2"}  # x.md muda, nada sobre a pasta protegida
    gvo.escrever_vault(tmp_path, plano2)

    assert canario.read_text(encoding="utf-8") == "não mexa aqui"
    assert (tmp_path / "10-Tecnicas" / "x.md").read_text(encoding="utf-8") == "v2"


def test_nota_removida_da_fonte_vai_para_arquivo_nao_e_apagada(tmp_path):
    plano1 = {"60-Achados/vai-sumir.md": "conteudo antigo"}
    gvo.escrever_vault(tmp_path, plano1)

    plano2: dict[str, str] = {}  # a nota não existe mais no plano novo
    gvo.escrever_vault(tmp_path, plano2)

    assert not (tmp_path / "60-Achados" / "vai-sumir.md").exists()
    arquivada = tmp_path / "99-Arquivo" / "60-Achados" / "vai-sumir.md"
    assert arquivada.exists()
    assert arquivada.read_text(encoding="utf-8") == "conteudo antigo"


def test_manifesto_e_json_valido(tmp_path):
    gvo.escrever_vault(tmp_path, {"10-Tecnicas/x.md": "v1"})
    manifesto = json.loads((tmp_path / gvo._MANIFESTO).read_text(encoding="utf-8"))
    assert manifesto == ["10-Tecnicas/x.md"]
