"""O ROADMAP lista as lacunas conhecidas (tecnicas, mercado e achados que ficaram para a v1.1), sem data prometida nem codigo interno."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
TEXTO = (RAIZ / "docs" / "projeto" / "ROADMAP.md").read_text(encoding="utf-8")
SECAO = TEXTO.partition("## Itens identificados para a v1.1 e depois")[2]


def test_a_secao_de_itens_identificados_existe():
    assert SECAO.strip(), "falta a secao '## Itens identificados para a v1.1 e depois'"


@pytest.mark.parametrize("termo", [
    "FAAS", "GFAAS", "MIP-OES", "DMA", "a confirmar pelo mantenedor", "analisador direto de mercúrio",
    "curva de calibração univariada", "OPLS para regressão", "RPIQ", "LOD e LOQ multivariados", "RMSEC",
    "`.spa`", "trilha de auditoria", "PyPI", "Raman", "`confianca_%`", "limiar conformal", "T² de treino",
    "`_DIR_LOCAL`", "remissões internas", "cabeçalho SPDX", "subcomando de predição",
])
def test_lacuna_conhecida_esta_no_roadmap(termo):
    assert termo in SECAO, termo


def test_o_iPLS_sem_grupos_foi_resolvido_e_saiu_do_roteiro():
    """A limitacao (CV interna do iPLS sem grupos) foi corrigida: nao pode voltar a ser listada como pendente."""
    assert "sem passar os grupos" not in SECAO
    assert "validação interna agrupada na seleção de variáveis por intervalos" not in SECAO


def test_sem_data_prometida_codigo_interno_nem_alegacao_regulatoria():
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b|\b(?:jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\w*\s+de\s+20\d\d", SECAO, flags=re.I)
    assert not re.search(r"\bV\d{2,3}\b|\bA\d{1,2}-\d{2}\b|\bPasso\s+\d+", SECAO)
    assert "nenhuma alegação de conformidade regulatória" in SECAO
