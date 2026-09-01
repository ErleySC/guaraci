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

from guaraci.perfil_matriz import (PERFIS_TECNICA, UnknownProfileError,
                                   apply_profile, combine_profiles,
                                   load_profile, perfis_disponiveis,
                                   save_profile)


# ── Contrato do carregador ───────────────────────────────────────────────────

@pytest.mark.parametrize("nome", ["generico", "oleo_nir", "milho_nir",
                                   "mel_vis_nir"])
def test_perfis_embutidos_carregam(nome):
    """Todo perfil distribuido no pacote precisa carregar e ter vocabulario."""
    p = load_profile(nome)
    assert p.nome == nome
    assert p.vocabulario.matriz
    assert p.unidade_eixo in ("cm-1", "nm")


@pytest.mark.parametrize("nome,nivel_esperado", [
    ("bancada", "high"), ("celular", "medium"), ("scanner", "high"),
])
def test_perfis_tecnica_de_imagem_carregam(nome, nivel_esperado):
    """Bloco 8a (2026-08-25): perfis de TECNICA DE AQUISICAO de imagem
    (bancada/celular/scanner) usam os mesmos load_profile/MatrixProfile dos
    perfis espectrais, com os 3 campos novos preenchidos. O nivel aqui e'
    so' INFORMATIVO -- o nivel real e' sempre decidido pelos dados, nunca
    pelo perfil (ver dados_imagem.py)."""
    p = load_profile(nome)
    assert p.nome == nome
    assert p.unidade_eixo == "indice"  # nao cm-1/nm -- nao e' matriz espectral
    assert p.resolucao_esperada
    assert p.formatos_aceitos
    assert p.nivel_agrupamento_tipico == nivel_esperado


def test_perfis_espectrais_nao_declaram_campos_de_imagem():
    """Os 3 campos novos (Bloco 8a) sao None em todo perfil espectral --
    nunca fazem sentido fora de mode='imagem'."""
    for nome in ("generico", "oleo_nir", "milho_nir", "mel_vis_nir"):
        p = load_profile(nome)
        assert p.resolucao_esperada is None
        assert p.formatos_aceitos is None
        assert p.nivel_agrupamento_tipico is None


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
    assert not milho.outside_working_range(8.0)
    assert milho.outside_working_range(25.0)
    generico = load_profile("generico")           # sem faixa declarada
    assert not generico.outside_working_range(1e9)
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


# ── Central de perfis (Agente 5B, 2026-09-01) — matriz x tecnica ───────────

def test_perfis_disponiveis_separa_matriz_de_tecnica():
    """`generico` serve as duas dimensoes (fallback); os 3 perfis de tecnica
    de imagem NUNCA aparecem na lista de matriz, e vice-versa."""
    matriz = perfis_disponiveis(apenas="matriz")
    tecnica = perfis_disponiveis(apenas="tecnica")

    assert "generico" in matriz and "generico" in tecnica
    assert PERFIS_TECNICA <= set(tecnica)
    assert not (PERFIS_TECNICA & set(matriz)), (
        "perfil de tecnica vazou pra lista de matriz")
    assert {"oleo_nir", "milho_nir", "mel_vis_nir"} <= set(matriz)
    assert not ({"oleo_nir", "milho_nir", "mel_vis_nir"} & set(tecnica)), (
        "perfil de matriz vazou pra lista de tecnica")


def test_perfis_disponiveis_sem_filtro_lista_tudo():
    todos = set(perfis_disponiveis())
    assert todos == set(perfis_disponiveis(apenas="matriz")) | set(
        perfis_disponiveis(apenas="tecnica"))


def test_perfil_de_tecnica_carrega_campos_de_garantia():
    """Os campos do Bloco 8a (resolucao/formatos/nivel de garantia) --
    antes mortos (nunca lidos por ninguem fora deste modulo) -- precisam
    estar de fato populados nos 3 perfis de tecnica, senao a central de
    perfis nao teria nada de novo pra mostrar."""
    for nome in PERFIS_TECNICA:
        p = load_profile(nome)
        assert p.nivel_agrupamento_tipico in ("high", "medium", "none")
        assert p.formatos_aceitos or p.resolucao_esperada


# ── Indicador de cobertura validada (Agente 5B, item pendente fechado) ────

def test_referencia_distingue_perfil_validado_de_declarado():
    """`referencia` nao-vazia = validado com dado publico real (paper/
    dataset citado); vazia = so' declarado. Contra-prova contra o estado
    real dos 8 perfis embutidos -- se um perfil ganhar validacao nova (ou
    perder), o selo da UI (guaraci.py:_rotulo_opcao, app_quimiometria.py:
    _rotulo_perfil) muda sozinho, sem precisar editar codigo de UI."""
    validados = {"milho_nir", "oleos_comestiveis_nir"}
    nao_validados = {"oleo_nir", "mel_vis_nir", "bancada", "celular", "scanner"}
    for nome in validados:
        assert load_profile(nome).referencia, f"{nome} deveria ter referencia"
    for nome in nao_validados:
        assert not load_profile(nome).referencia, f"{nome} nao deveria ter referencia"


# ── Perfil combinado (Agente 5B, "criar/salvar perfil combinado") ─────────

def test_combine_profiles_funde_matriz_e_tecnica_sem_misturar_campos():
    """Vocabulario/faixa vem da matriz; resolucao/formatos/garantia vem da
    tecnica -- contra-prova de que a fusao nao troca as fontes."""
    matriz = load_profile("mel_vis_nir")
    tecnica = load_profile("celular")
    combinado = combine_profiles("mel_celular", matriz, tecnica)

    assert combinado.nome == "mel_celular"
    assert combinado.vocabulario.matriz == matriz.vocabulario.matriz
    assert combinado.eixo_min == matriz.eixo_min
    assert combinado.resolucao_esperada == tecnica.resolucao_esperada
    assert combinado.formatos_aceitos == tecnica.formatos_aceitos
    assert combinado.nivel_agrupamento_tipico == tecnica.nivel_agrupamento_tipico
    # default_preprocessing: tecnica tem valor (autoscaling) -> vence.
    assert combinado.default_preprocessing == tecnica.default_preprocessing


def test_combine_profiles_sem_tecnica_usa_defaults_de_matriz():
    matriz = load_profile("milho_nir")
    combinado = combine_profiles("so_matriz", matriz, None)
    assert combinado.default_preprocessing == matriz.default_preprocessing
    assert combinado.resolucao_esperada is None
    assert combinado.nivel_agrupamento_tipico is None


def test_roundtrip_save_load_preserva_as_duas_dimensoes(tmp_path):
    """Mesma disciplina de roundtrip que ja pegou um bug real de Config
    nesta sessao (test_config_io.py) -- salvar/carregar um perfil
    combinado precisa preservar matriz E tecnica juntas."""
    matriz = load_profile("oleos_comestiveis_nir")
    tecnica = load_profile("bancada")
    combinado = combine_profiles("oleo_bancada", matriz, tecnica)

    caminho = tmp_path / "oleo_bancada.yaml"
    save_profile(combinado, str(caminho))
    lido = load_profile(str(caminho))

    assert lido.vocabulario.matriz == matriz.vocabulario.matriz
    assert lido.faixa_trabalho == matriz.faixa_trabalho
    assert lido.referencia == matriz.referencia
    assert lido.resolucao_esperada == tecnica.resolucao_esperada
    assert lido.formatos_aceitos == tecnica.formatos_aceitos
    assert lido.nivel_agrupamento_tipico == tecnica.nivel_agrupamento_tipico
