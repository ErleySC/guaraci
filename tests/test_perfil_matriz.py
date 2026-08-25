# -*- coding: utf-8 -*-
"""Trocar de matriz e' trocar de PERFIL -- nunca editar codigo-fonte.

Este e' o teste de aceitacao do requisito multimatriz. Ele roda o pipeline
completo em duas matrizes com naturezas diferentes (uma em cm-1, outra em
nm; vocabularios distintos) mudando UM campo de configuracao, e verifica
que a saida fala a lingua da matriz certa.

O que motivou: rodando o pipeline sobre milho em grao, o model card
afirmava "quantificacao de adulterante em oleo vegetal amazonico" e o log
dizia "60 adulterados + 0 puros" (auditoria mestre de 2026-08-17, sec.
1.6). Nenhum numero estava errado -- as frases estavam.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from guaraci.perfil_matriz import (UnknownProfileError, apply_profile,
                                   load_profile)


# ── Contrato do carregador ───────────────────────────────────────────────────

@pytest.mark.parametrize("nome", ["generico", "oleo_nir", "milho_nir",
                                   "mel_vis_nir"])
def test_perfis_embutidos_carregam(nome):
    """Todo perfil distribuido no pacote precisa carregar e ter vocabulario."""
    p = load_profile(nome)
    assert p.nome == nome
    assert p.vocabulario.matriz
    assert p.unidade_eixo in ("cm-1", "nm")


def test_matriz_sem_perfil_falha_com_mensagem_acionavel():
    """Matriz sem perfil cadastrado NAO pode cair num padrao em silencio:
    rodar mel com a faixa e o vocabulario de oleo produz numeros que
    parecem validos e afirmacoes quimicas que nao sao."""
    with pytest.raises(UnknownProfileError) as exc:
        load_profile("cafe_raman")
    msg = str(exc.value)
    assert "cafe_raman" in msg
    assert "disponiveis" in msg.lower()
    # a mensagem tem que dizer o que FAZER, nao so' o que falhou
    assert "yaml" in msg.lower()


def test_perfil_de_usuario_por_caminho(tmp_path):
    """Uma matriz nova nao exige entrar no pacote: basta um YAML proprio."""
    yaml_txt = (
        "descricao: 'Cafe torrado por Raman'\n"
        "unidade_eixo: 'cm-1'\n"
        "eixo_min: 200.0\n"
        "eixo_max: 3200.0\n"
        "vocabulario:\n"
        "  classe: 'cultivar'\n"
        "  matriz: 'cafe torrado'\n"
        "  alvo: 'o teor de robusta em arabica'\n"
    )
    caminho = tmp_path / "cafe_raman.yaml"
    caminho.write_text(yaml_txt, encoding="utf-8")
    p = load_profile(str(caminho))
    assert p.vocabulario.matriz == "cafe torrado"
    assert (p.eixo_min, p.eixo_max) == (200.0, 3200.0)


def test_perfil_nao_sobrescreve_escolha_explicita_do_usuario(pq):
    """O perfil e' um PADRAO da matriz, nao uma imposicao: quem definiu a
    faixa na configuracao mandou nela."""
    cfg = pq.Config()
    cfg.wn_min, cfg.wn_max = 1234.0, 5678.0          # escolha explicita
    apply_profile(cfg, load_profile("milho_nir"))
    assert (cfg.wn_min, cfg.wn_max) == (1234.0, 5678.0)

    cfg2 = pq.Config()                                # tudo no default
    apply_profile(cfg2, load_profile("milho_nir"))
    assert (cfg2.wn_min, cfg2.wn_max) == (1100.0, 2498.0)


def test_faixa_de_trabalho_declarada_marca_extrapolacao():
    """Predicao fora da faixa calibrada precisa ser detectavel; sem faixa
    declarada, o perfil nao inventa um limite."""
    milho = load_profile("milho_nir")             # faixa [6, 10] de proteina
    assert not milho.fora_da_faixa_de_trabalho(8.0)
    assert milho.fora_da_faixa_de_trabalho(25.0)
    generico = load_profile("generico")           # sem faixa declarada
    assert not generico.fora_da_faixa_de_trabalho(1e9)
    assert generico.faixa_trabalho is None, (
        "quem consome precisa distinguir 'dentro da faixa' de 'faixa nao "
        "declarada' -- sao coisas diferentes")


# ── Aceitacao: mesmo dado, duas matrizes, zero linha de codigo alterada ──────

def _csv_espectral(caminho: Path, eixo: np.ndarray, n_per_class: int = 12,
                   semente: int = 7) -> None:
    """Escreve um CSV no formato que `dados_io.load_csv` le (uma coluna
    por canal, cabecalho = eixo espectral) + colunas classe/conc."""
    import pandas as pd

    rng = np.random.default_rng(semente)
    linhas, classes, concs = [], [], []
    for k, cls in enumerate(("A", "B")):
        for i in range(n_per_class):
            teor = float(i) / n_per_class * 10.0
            base = 0.4 + 0.1 * k + 0.02 * np.sin(eixo / (eixo.max() / 6.0))
            linhas.append(base + 0.004 * teor + rng.normal(0, 0.002,
                                                           eixo.size))
            classes.append(cls)
            concs.append(teor)
    df = pd.DataFrame(np.array(linhas),
                      columns=[f"{v:.1f}" for v in eixo])
    df.insert(0, "classe", classes)
    df.insert(1, "conc", concs)
    df.to_csv(caminho, index=False)


def _rodar_com_perfil(pq, base: Path, perfil: str, eixo: np.ndarray):
    """Roda executar() mudando SO' cfg.matrix_profile. Devolve a pasta de saida."""
    from conftest import achar_pastas_run

    pasta = base / perfil
    pasta.mkdir(parents=True, exist_ok=True)
    csv = pasta / "espectros.csv"
    _csv_espectral(csv, eixo)

    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="conc",
        output_root_folder=str(pasta / "saida"),
        matrix_profile=perfil,                     # <<< a UNICA diferenca
        group_by_mae_id=False,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=5,
        n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=3, frac_holdout=0.0,
    )
    pq.executar(cfg)
    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, f"executar() nao criou saida para o perfil '{perfil}'"
    return Path(runs[0])


def _model_card(pasta: Path) -> str:
    cards = list(pasta.rglob("model_card.md"))
    assert cards, "model_card.md nao foi gerado"
    return cards[0].read_text(encoding="utf-8")


@pytest.mark.slow
def test_aceitacao_multimatriz_milho_e_oleo_sem_tocar_em_codigo(
        pq, tmp_path):
    """Duas matrizes, dois eixos (nm e cm-1), dois vocabularios -- uma linha
    de configuracao de diferenca.

    Se este teste exigir edicao de codigo-fonte para passar em uma matriz
    nova, e' falha de arquitetura, nao do teste.
    """
    saida_milho = _rodar_com_perfil(
        pq, tmp_path, "milho_nir", np.linspace(1100.0, 2498.0, 220))
    saida_oleo = _rodar_com_perfil(
        pq, tmp_path, "oleo_nir", np.linspace(4000.0, 10000.0, 220))

    card_milho = _model_card(saida_milho)
    card_oleo = _model_card(saida_oleo)

    # Cada model card declara a SUA matriz...
    assert "milho em grao" in card_milho
    assert "oleo vegetal" in card_oleo
    # ...e nao a da outra. Este era exatamente o defeito medido em 2026-08-17.
    assert "oleo vegetal" not in card_milho, (
        "o model card do milho afirma a matriz de oleo -- vocabulario ainda "
        "preso ao dominio de origem")
    assert "milho" not in card_oleo

    # O vocabulario de classe tambem acompanha a matriz.
    assert "variedade" in card_milho
    assert "especie" in card_oleo

    # E o perfil usado fica registrado, para quem le o card depois.
    assert "milho_nir" in card_milho
    assert "oleo_nir" in card_oleo


@pytest.mark.slow
def test_perfil_inexistente_aborta_o_pipeline_antes_de_predizer(pq, tmp_path):
    """Matriz sem perfil -> falha com mensagem acionavel, nunca uma predicao."""
    csv = tmp_path / "e.csv"
    _csv_espectral(csv, np.linspace(4000.0, 10000.0, 120))
    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="conc",
        output_root_folder=str(tmp_path / "saida"),
        matrix_profile="matriz_que_nao_existe",
    )
    with pytest.raises(UnknownProfileError):
        pq.executar(cfg)


def test_perfis_sao_empacotados_com_o_pacote():
    """Os YAMLs precisam viajar dentro da wheel.

    Sem `[tool.setuptools.package-data]`, eles ficam de fora do build e
    `guaraci perfis` lista ZERO perfis num ambiente instalado por pip --
    exatamente o cenario de quem nunca viu o projeto. Achado testando
    instalacao limpa em 2026-08-18; passava despercebido porque em
    desenvolvimento o pacote e' lido da arvore de fontes, onde os arquivos
    obviamente estao.
    """
    import guaraci
    from guaraci.perfil_matriz import DIR_PERFIS

    # DIR_PERFIS resolve a partir do modulo instalado, nao do cwd.
    assert DIR_PERFIS.is_dir(), f"{DIR_PERFIS} nao existe no pacote instalado"
    encontrados = sorted(p.stem for p in DIR_PERFIS.glob("*.yaml"))
    assert {"generico", "oleo_nir", "milho_nir", "mel_vis_nir"} <= set(
        encontrados), f"perfis faltando no pacote: {encontrados}"
    assert DIR_PERFIS.is_relative_to(Path(guaraci.__file__).parent), (
        "os perfis precisam morar DENTRO do pacote para serem empacotados")
