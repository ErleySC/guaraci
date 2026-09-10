# -*- coding: utf-8 -*-
"""Testes do motor de consulta por grafo (Passos 173-175).

Consultas representativas rodam contra o vault REAL (gerado por
`gerar_vault_obsidian.py` na mesma máquina/CI) -- se o vault não existir
ainda, os testes o geram primeiro. Desatualização e cobertura incompleta
são simuladas com vaults sintéticos em `tmp_path`, para não depender do
estado real do git nem mutar histórico.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parents[1]
_SCRIPTS = str(_RAIZ / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import consultar_vault as cvv  # noqa: E402
import gerar_vault_obsidian as gvo  # noqa: E402


@pytest.fixture(scope="module")
def vault_real(tmp_path_factory):
    destino = tmp_path_factory.mktemp("vault_consulta")
    plano, _contagens, _avisos = gvo.montar_plano()
    gvo.escrever_vault(destino, plano)
    return cvv.carregar_vault(destino)


# ─────────────────────────────────────────────────────────────────────────
#  Consultas representativas (Passo 173)
# ─────────────────────────────────────────────────────────────────────────

def test_consulta_mcr_als_traz_modulo_conceito_e_achados(vault_real):
    """O exemplo da própria instrução: consultar 'MCR-ALS' não pode
    devolver só uma nota isolada."""
    r = cvv.consultar("MCR-ALS", vault_real, saltos=1)
    assert r.principal is not None
    stems = {n.stem for n in ([r.principal] + r.correspondencias + [n for n, _ in r.vizinhos])}
    assert "mcr_als.py" in stems  # módulo
    assert "mcr-als-aviso-de-escopo" in stems  # conceito
    # pelo menos 2 dos achados/passos que documentam o histórico real do MCR-ALS
    achados_mcr_als = {s for s in stems if "mcr-als" in s}
    assert len(achados_mcr_als) >= 2


def test_consulta_por_tecnica(vault_real):
    r = cvv.consultar("Raman", vault_real, saltos=1)
    assert r.principal is not None
    stems = {n.stem for n in ([r.principal] + r.correspondencias + [n for n, _ in r.vizinhos])}
    assert "raman-espectroscopia-raman" in stems


def test_consulta_por_modulo_traz_dependentes_por_grafo(vault_real):
    r = cvv.consultar("conformal.py", vault_real, saltos=1)
    assert r.principal is not None and r.principal.stem == "conformal.py"
    stems = {n.stem for n in ([r.principal] + r.correspondencias + [n for n, _ in r.vizinhos])}
    # dependentes reais de conformal.py (parse_modulos() calcula "usado_por")
    modulos = gvo.parse_modulos()
    dependentes = modulos["conformal"].usado_por
    assert dependentes, "conformal.py deveria ter pelo menos 1 dependente real"
    assert any(f"{d}.py" in stems for d in dependentes)


def test_consulta_por_decisao_traz_modulo_relacionado(vault_real):
    r = cvv.consultar("bootstrap_vip", vault_real, saltos=1)
    assert r.principal is not None
    stems = {n.stem for n in ([r.principal] + r.correspondencias + [n for n, _ in r.vizinhos])}
    assert "pipeline.py" in stems  # autolinkado a partir do texto da decisão


def test_consulta_sem_correspondencia_nao_inventa_nada(vault_real):
    r = cvv.consultar("termo-que-certamente-nao-existe-no-vault-xyz123", vault_real)
    assert r.principal is None
    saida = cvv.formatar_resultado(r, head=None)
    assert "Nenhuma nota encontrada" in saida


# ─────────────────────────────────────────────────────────────────────────
#  Alerta de desatualização (Passo 174)
# ─────────────────────────────────────────────────────────────────────────

def _nota_sintetica(commit: str) -> cvv.Nota:
    return cvv.Nota(rel="10-Tecnicas/x.md", stem="x", titulo="X", tags=[],
                     fonte=["docs/x.md"], commit=commit, gerado_em="2020-01-01",
                     corpo="corpo")


def test_aviso_desatualizacao_dispara_com_commit_antigo():
    nota_antiga = _nota_sintetica(commit="0000000")
    aviso = cvv.aviso_desatualizacao([nota_antiga], head="abc1234")
    assert aviso is not None
    assert "0000000" in aviso
    assert "abc1234" in aviso


def test_aviso_desatualizacao_nao_dispara_com_commit_atual():
    nota_atual = _nota_sintetica(commit="abc1234")
    aviso = cvv.aviso_desatualizacao([nota_atual], head="abc1234")
    assert aviso is None


def test_aviso_desatualizacao_nao_dispara_sem_head_conhecido():
    nota = _nota_sintetica(commit="0000000")
    assert cvv.aviso_desatualizacao([nota], head=None) is None


def test_formatar_resultado_inclui_aviso_quando_desatualizado(vault_real):
    r = cvv.consultar("conformal.py", vault_real, saltos=0)
    saida = cvv.formatar_resultado(r, head="commit-fabricado-que-nao-existe")
    assert "AVISO" in saida
    assert "desatualizado" in saida.lower()


# ─────────────────────────────────────────────────────────────────────────
#  Verificação de cobertura sob demanda (Passo 175)
# ─────────────────────────────────────────────────────────────────────────

def test_cobertura_do_vault_real_e_completa(vault_real):
    saida = cvv.checar_cobertura_do_vault(vault_real)
    assert "Cobertura COMPLETA." in saida
    assert "Módulos" in saida and "Técnicas" in saida
    assert "Passos de PROGRESSO.md" in saida and "Datasets" in saida


def test_cobertura_acusa_modulo_faltando(vault_real):
    """Remove 1 nota de módulo do vault carregado e confirma que a
    cobertura sob demanda acusa a lacuna -- prova que o comando não
    sempre diz "completo" incondicionalmente."""
    notas_incompletas = dict(vault_real)
    algum_modulo = next(s for s, n in notas_incompletas.items()
                         if n.rel.startswith("20-Modulos/"))
    del notas_incompletas[algum_modulo]
    saida = cvv.checar_cobertura_do_vault(notas_incompletas)
    assert "Cobertura INCOMPLETA." in saida
