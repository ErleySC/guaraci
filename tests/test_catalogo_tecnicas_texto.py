# -*- coding: utf-8 -*-
"""Contrato de texto do catalogo de tecnicas (`cli_assistente.TECNICAS`,
tela `[1]` do menu principal) e do aviso de maturidade do HSI -- Passo
164/165 da auditoria de documentacao (2026-09-05).

Achado que motivou este teste: o `preproc_rec` do Raman e o aviso de
maturidade do HSI embutiam detalhe de auditoria interna na tela de uso
corrente (data especifica de aprovacao em portao de aceite; contagem
`n=12`/`n=2` por classe) -- informacao que pertence ao dossie de calibracao
privado (fora do repositorio), nao a uma tela que o usuario le em segundos
para decidir como configurar uma execucao. Este teste fixa o criterio:
texto de tela pode dizer QUE foi validado e apontar pro documento
completo, nunca reproduzir o numero/data/metodologia em si."""
from __future__ import annotations

import re

from guaraci.cli_assistente import TECNICAS
from guaraci.guaraci import _AVISO_MATURIDADE_HSI_EN, _AVISO_MATURIDADE_HSI_PT

# Padroes que NAO devem aparecer em texto de tela de uso corrente --
# pertencem ao dossie de calibracao privado, nao ao catalogo/aviso.
_PADRAO_DATA = re.compile(r"202\d-\d\d-\d\d")
_PADRAO_PASSO = re.compile(r"\bpasso\s*\d+\b", re.IGNORECASE)
_PADRAO_N_IGUAL = re.compile(r"\bn\s*=\s*\d+\b", re.IGNORECASE)
_PADRAO_P_VALOR = re.compile(r"\bp\s*[<=]\s*0[.,]\d+\b", re.IGNORECASE)
_PADRAO_ANTES_DE_USAR = re.compile(
    r"antes de usar para resultado publicavel|before using.*publishable",
    re.IGNORECASE)


def _sem_detalhe_de_auditoria(texto: str) -> list:
    """Retorna a lista de padroes proibidos encontrados em `texto`."""
    achados = []
    for nome, padrao in (
        ("data (AAAA-MM-DD)", _PADRAO_DATA),
        ("referencia a 'Passo N'", _PADRAO_PASSO),
        ("contagem 'n=N'", _PADRAO_N_IGUAL),
        ("p-valor", _PADRAO_P_VALOR),
        ("instrucao 'antes de usar para publicar'", _PADRAO_ANTES_DE_USAR),
    ):
        if padrao.search(texto):
            achados.append(nome)
    return achados


def test_catalogo_de_tecnicas_nao_embute_detalhe_de_auditoria():
    for chave, entrada in TECNICAS.items():
        for lang in ("PT", "EN"):
            for campo in ("desc", "preproc_rec", "faixa", "nome"):
                texto = str(entrada.get(lang, {}).get(campo, ""))
                achados = _sem_detalhe_de_auditoria(texto)
                assert not achados, (
                    f"TECNICAS['{chave}']['{lang}']['{campo}'] tem detalhe de "
                    f"auditoria que deveria estar so' em "
                    f"o dossie de calibracao privado: {achados} -- texto: {texto!r}")


def test_catalogo_de_tecnicas_nao_relata_validacao_na_descricao():
    """Regra vigente (2026-10-03, substitui o Passo 165): a descricao diz o que a tecnica e';
    a evidencia de validacao fica so' no assistente de confianca (`guaraci confianca`)."""
    from guaraci.catalogo import texto_proibido
    for chave, entrada in TECNICAS.items():
        for lang in ("PT", "EN"):
            desc = entrada[lang]["desc"]
            assert texto_proibido(desc) is None, (chave, lang, desc)
            assert "validad" not in desc.lower() and "validacao_dados_publicos" not in desc, (chave, lang)

def test_aviso_maturidade_hsi_nao_embute_detalhe_de_auditoria():
    for texto in (_AVISO_MATURIDADE_HSI_PT, _AVISO_MATURIDADE_HSI_EN):
        achados = _sem_detalhe_de_auditoria(texto)
        assert not achados, (
            f"Aviso de maturidade do HSI tem detalhe de auditoria que "
            f"deveria estar so' no dossie de calibracao privado: {achados} -- "
            f"texto: {texto!r}")
    # Regra vigente (2026-10-03): texto de tela diz o limite de uso e aponta o assistente de
    # confianca; nome de dataset e numero de validacao ficam so' la'.
    from guaraci.catalogo import texto_proibido
    for texto in (_AVISO_MATURIDADE_HSI_PT, _AVISO_MATURIDADE_HSI_EN):
        assert texto_proibido(texto) is None and "validacao_dados_publicos" not in texto, texto
    assert "tecla c" in _AVISO_MATURIDADE_HSI_PT and "key c" in _AVISO_MATURIDADE_HSI_EN


def test_rotulo_da_faixa_cabe_na_tabela_da_tela_de_tecnicas():
    """A tabela da tela [2.4] corta a coluna Faixa em 32 caracteres; com o faixa_txt inteiro, 8 das 15
    tecnicas apareciam com reticencias ("Tempo de retencao (min) ou..."). O rotulo da tabela
    (faixa_curta, ou faixa_txt quando curto) precisa caber inteiro; o texto completo segue no detalhe."""
    for chave, entrada in TECNICAS.items():
        for lang in ("PT", "EN"):
            rot = str(entrada[lang]["faixa_curta"]).split(" (use")[0]
            assert len(rot) <= 32, (chave, lang, rot)
            assert entrada[lang]["faixa"], (chave, lang)

def test_uv_vis_quantitativo_nao_normaliza_por_amostra():
    """Beer-Lambert: A = eps*b*c. SNV/MSC dividem cada espectro pela propria escala e apagam c (analito unico:
    R2cv 0,00 com snv_mc contra 0,999 com mc no checkup mir_raman_uv). O padrao da tela [2.4] tem de preservar a escala."""
    from guaraci.catalogo_tecnicas import INSTRUMENTOS
    assert "snv" not in INSTRUMENTOS["uv-vis"]["preproc"] and "msc" not in INSTRUMENTOS["uv-vis"]["preproc"]


def test_raman_faixa_padrao_fora_da_borda_rayleigh():
    from guaraci.catalogo_tecnicas import INSTRUMENTOS
    assert INSTRUMENTOS["raman"]["faixa_min"] >= 150.0


def test_generico_aceita_tabela_csv_e_nao_aplica_preset_espectral():
    """'Generico' = qualquer dado multivariado: o menu tem de oferecer CSV e o
    preset nao pode ser MSC+SG (derivada de 25 pontos falha com <25 colunas e
    nao tem sentido entre variaveis de unidades diferentes)."""
    from guaraci.catalogo_tecnicas import INSTRUMENTOS
    g = INSTRUMENTOS["generico"]
    assert g["mode"] == "csv" and "csv" in [m for m, _ in g["formatos"]]
    assert g["preproc"] == "autoscaling"
