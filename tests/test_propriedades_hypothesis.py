# -*- coding: utf-8 -*-
"""Testes de propriedade (Hypothesis) para os invariantes que ja se
mostraram FRAGEIS nesta sessao (Passo 85): agrupamento (vazou em guarda de
identificador, sanitizador e split de quantificacao em auditorias
anteriores) e roundtrip de config.yaml (vazou agora mesmo, Passo 83/84 --
`matrix_profile` resetava em silencio no ciclo salvar/carregar).

Cada propriedade abaixo tem uma CONTRA-PROVA: um teste que mostra a
checagem falhando de proposito quando o invariante e' violado, para
provar que a propriedade nao passa por vacuidade.

NAO converte mecanicamente os testes manuais existentes (test_modo_cego.py,
test_contrato_validacao_agrupada.py) -- este arquivo ADICIONA cobertura via
geracao automatica de casos onde isso agrega valor real (mais formas de
rotulo/valor do que os poucos exemplos fixos escritos a mao cobrem), sem
duplicar o que ja esta' testado la.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from hypothesis import example, given, strategies as st

from guaraci.config_io import _CONFIG_SPEC as _SPEC_MODULO

_SPEC_POR_KEY = {s["key"]: s for s in _SPEC_MODULO}

# Alfabeto de texto p/ campos "str"/"str_opcional"/"list": ASCII imprimivel +
# Latin-1 (cobre PT com acento), sem caracteres de controle. Newline/tab
# ficam FORA de proposito -- o escritor de config.yaml e' de linha unica por
# campo; um newline literal embutido no valor e' uma limitacao separada e
# conhecida do formato (YAML single-quoted dobra quebra de linha em espaco),
# nao o invariante que este arquivo protege (perda de campo/tipo no
# roundtrip). Nenhum campo real do projeto (caminho, nome de coluna, nome de
# especie) contem newline na pratica.
_ALFABETO_TEXTO = st.characters(
    min_codepoint=32, max_codepoint=0x00FF, blacklist_categories=("Cc", "Cs"))


# =========================================================================
#  1. ROUNDTRIP DE CONFIG -- generaliza o bug do Passo 83/84 (matrix_profile
#  perdia valor no ciclo salvar/carregar) para TODO campo de _CONFIG_SPEC,
#  nao so' o que ja foi corrigido.
# =========================================================================

def _estrategia_valor_bruto(spec):
    """Gera um valor 'como o usuario digitaria' para o tipo do campo --
    depois passado por `_coagir_valor` (a mesma coercao que `load_config`
    aplica), entao a comparacao final e' sempre contra a forma CANONICA
    esperada, nao contra o valor bruto."""
    tipo = spec["tipo"]
    if tipo == "bool":
        return st.booleans()
    if tipo == "int":
        lo = spec.get("min"); hi = spec.get("max")
        lo = -1000 if lo is None else int(lo)
        hi = 1000 if hi is None else int(hi)
        return st.integers(min_value=lo, max_value=hi)
    if tipo == "float":
        lo = spec.get("min"); hi = spec.get("max")
        lo = 0.0 if lo is None else float(lo)
        hi = (lo + 1000.0) if hi is None else float(hi)
        return st.floats(min_value=lo, max_value=hi,
                          allow_nan=False, allow_infinity=False)
    if tipo == "choice":
        return st.sampled_from(list(spec["opcoes"]))
    if tipo == "preproc":
        from guaraci.config_io import _PRE_PROC_FRIENDLY
        return st.sampled_from(sorted(_PRE_PROC_FRIENDLY.values()))
    if tipo == "list":
        # itens ja' "limpos" (sem espaco de borda) -- `_coagir_valor` faz
        # strip()+drop-vazio por design (normalizacao de entrada de
        # menu/YAML), entao gerar item com espaco de borda testaria a
        # normalizacao, nao a preservacao de valor.
        item = st.text(alphabet=_ALFABETO_TEXTO, min_size=1, max_size=15) \
            .map(lambda s: s.strip()).filter(lambda s: s != "")
        return st.lists(item, max_size=4)
    if tipo == "str_opcional":
        return st.one_of(
            st.none(),
            st.text(alphabet=_ALFABETO_TEXTO, min_size=1, max_size=1) \
                .map(lambda s: s.strip()).filter(lambda s: s != ""),
            st.text(alphabet=_ALFABETO_TEXTO, min_size=2, max_size=25) \
                .map(lambda s: s.strip()).filter(lambda s: s != ""),
        )
    # "str": sem normalizacao em _coagir_valor (fallback = str(val)) --
    # inclusive espaco de borda precisa sobreviver exato.
    return st.text(alphabet=_ALFABETO_TEXTO, max_size=25)


def _estrategia_spec_e_bruto():
    """Combina spec+valor numa SO estrategia (em vez de `st.data()` com dois
    draws independentes) especificamente para poder fixar `@example`s
    deterministicos abaixo -- com `st.data()` nao da' p/ mirar um (spec,
    bruto) especifico, so' deixar a busca aleatoria por conta do shrinker."""
    return st.sampled_from(_SPEC_MODULO).flatmap(
        lambda spec: st.tuples(st.just(spec), _estrategia_valor_bruto(spec)))


@given(_estrategia_spec_e_bruto())
# Casos adversariais fixos (nao deixados so' pra busca aleatoria achar):
# medido ANTES da correcao deste Passo 85 que so' 80 exemplos aleatorios
# passaram raso por cima desses 3 casos sem os achar nenhuma vez -- a
# propriedade so' pega a regressao de verdade com estes @example fixados.
@example((_SPEC_POR_KEY["pasta_dados"], "010"))            # YAML octal implicito -> vira int 8
@example((_SPEC_POR_KEY["pasta_dados"], "1.50"))            # YAML float -> perde zero a direita
@example((_SPEC_POR_KEY["pasta_dados"], " com espaco "))    # strip implicito do escalar sem aspas
@example((_SPEC_POR_KEY["coluna_classe"], "null"))          # palavra reservada YAML -> None
@example((_SPEC_POR_KEY["coluna_classe"], "0x1A"))          # YAML hex implicito -> vira int 26
@example((_SPEC_POR_KEY["excluir_classes"], ["?0"]))        # "?" em item de lista -> vira {0: None}
@example((_SPEC_POR_KEY["excluir_classes"], ["0?"]))        # "?" em item de lista -> erro de parse
def test_config_roundtrip_preserva_todo_campo_do_spec(pq, spec_e_bruto):
    """Para QUALQUER campo de `_CONFIG_SPEC` e QUALQUER valor valido do seu
    tipo: `Config -> save_config -> load_config` devolve o mesmo valor
    (na forma canonica pos-coercao). Generaliza o achado do Passo 83/84
    (so' `matrix_profile`) para os ~35 campos do spec de uma vez."""
    spec, bruto = spec_e_bruto
    alvo = pq._coagir_valor(spec, bruto)

    cfg = pq.Config()
    setattr(cfg, spec["attr"], alvo)

    fd, caminho = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        pq.save_config(cfg, caminho)
        recarregado = pq.load_config(caminho, base=pq.Config())
    finally:
        os.remove(caminho)

    obtido = getattr(recarregado, spec["attr"])
    if spec["tipo"] == "float":
        assert obtido == pytest.approx(alvo)
    else:
        assert obtido == alvo, (
            f"{spec['key']} ({spec['attr']}): salvou {alvo!r}, "
            f"recarregou {obtido!r} -- perdeu valor no ciclo "
            f"salvar/carregar (classe do bug do Passo 83/84).")


def test_contraprova_campo_fora_do_spec_nao_sobrevive_ao_roundtrip(
        pq, monkeypatch, tmp_path):
    """Contra-prova da propriedade acima: reproduz a CLASSE exata do bug do
    Passo 83 (campo fora de `_CONFIG_SPEC`) removendo `perfil_matriz` de
    uma copia do spec e confirmando que o valor SE PERDE no roundtrip --
    prova que o teste de propriedade acima teria pego a regressao, se
    `_CONFIG_SPEC` estivesse quebrado assim quando o campo foi adicionado."""
    import guaraci.config_io as cio
    spec_quebrado = [s for s in cio._CONFIG_SPEC if s["key"] != "perfil_matriz"]
    monkeypatch.setattr(cio, "_CONFIG_SPEC", spec_quebrado)

    cfg = pq.Config(matrix_profile="oleos_comestiveis_nir")
    caminho = str(tmp_path / "config.yaml")
    pq.save_config(cfg, caminho)
    recarregado = pq.load_config(caminho, base=pq.Config())

    assert recarregado.matrix_profile == "generico", (
        "com o campo fora do spec o valor deveria se perder (voltar ao "
        "default 'generico') -- se sobreviveu mesmo assim, o teste de "
        "propriedade principal nao esta' protegendo nada de verdade.")


# =========================================================================
#  2. QUANTIFICACAO CEGA -- generaliza test_modo_cego.py: para QUALQUER par
#  de rotulos verdadeiro/veneno (nao so' os fixos "A"/"B"/"Z" do teste
#  manual) com os MESMOS rotulos preditos, o resultado em mode cego e'
#  identico.
# =========================================================================

@given(
    verdadeiros=st.lists(st.text(alphabet=_ALFABETO_TEXTO, min_size=1, max_size=8),
                         min_size=1, max_size=12),
    veneno=st.lists(st.text(alphabet=_ALFABETO_TEXTO, min_size=1, max_size=8),
                    min_size=1, max_size=12),
    preditos=st.lists(st.text(alphabet=_ALFABETO_TEXTO, min_size=1, max_size=8),
                      min_size=1, max_size=12),
)
# Reproduz EXATAMENTE o cenario de envenenamento do teste manual original
# (test_modo_cego.py::test_quantificacao_cega_ignora_rotulo_verdadeiro_
# envenenado) como caso fixo -- garante que a busca aleatoria nunca deixa
# de exercitar o cenario que motivou este arquivo, independente do
# orcamento (`max_examples`) do profile ativo (Frente 2, Bloco 13d).
@example(verdadeiros=["A", "A", "B", "B"], veneno=["Z", "Z", "Z", "Z"],
         preditos=["A", "B", "B", "A"])
def test_modo_cego_nunca_depende_do_rotulo_verdadeiro(
        pq, verdadeiros, veneno, preditos):
    n = min(len(verdadeiros), len(veneno), len(preditos))
    verdadeiros = np.array(verdadeiros[:n])
    veneno = np.array(veneno[:n])
    preditos = np.array(preditos[:n])

    cfg = pq.Config(label_mode="cego")
    rot_ok, modo_ok = pq.labels_for_quantification(cfg, verdadeiros, preditos)
    rot_env, modo_env = pq.labels_for_quantification(cfg, veneno, preditos)

    assert modo_ok == modo_env == "cego"
    assert np.array_equal(rot_ok, rot_env), (
        "o rotulo verdadeiro influenciou a saida do mode cego")
    assert np.array_equal(rot_ok, preditos.astype(str))


@given(
    verdadeiros=st.lists(st.text(alphabet=_ALFABETO_TEXTO, min_size=1, max_size=8),
                         min_size=1, max_size=12),
    veneno=st.lists(st.text(alphabet=_ALFABETO_TEXTO, min_size=1, max_size=8),
                    min_size=1, max_size=12),
)
def test_contraprova_modo_controle_de_fato_muda_com_o_veneno(
        pq, verdadeiros, veneno):
    """Contra-prova: no mode `controle` (que LE o rotulo verdadeiro por
    design), envenenar a verdade muda o resultado sempre que verdadeiros !=
    veneno -- prova que a comparacao usada acima distingue 'ignorou o
    rotulo' de 'os dois casos calharam de dar igual', e o teste principal
    nao esta' passando por vacuidade."""
    n = min(len(verdadeiros), len(veneno))
    verdadeiros = np.array(verdadeiros[:n])
    veneno = np.array(veneno[:n])
    if np.array_equal(verdadeiros, veneno):
        return   # nao ha' veneno de fato -- nada a provar neste exemplo

    cfg = pq.Config(label_mode="controle")
    rot_ok, _ = pq.labels_for_quantification(cfg, verdadeiros, verdadeiros.copy())
    rot_env, _ = pq.labels_for_quantification(cfg, veneno, verdadeiros.copy())

    assert not np.array_equal(rot_ok, rot_env), (
        "em mode controle o resultado deveria mudar com o veneno; se nao "
        "mudou, a comparacao do teste principal nao prova nada")


# =========================================================================
#  3. SPLIT GROUP-AWARE -- generaliza o padrao ja usado em
#  test_contrato_validacao_agrupada.py para tamanhos/numero de grupos
#  aleatorios, sobre os TRES splitters de selecao de amostras (Passo 87:
#  Kennard-Stone, Duplex, SPXY) diretamente (funcoes puras, sem precisar
#  rodar o pipeline inteiro).
# =========================================================================

def _rodar_kennard_stone(pq, X, y, mae, frac_cal):
    return pq.kennard_stone_split_group_aware(X, mae, frac_cal)


def _rodar_duplex(pq, X, y, mae, frac_cal):
    return pq.duplex_split_group_aware(X, mae, frac_cal)


def _rodar_spxy(pq, X, y, mae, frac_cal):
    return pq.spxy_split_group_aware(X, y, mae, frac_cal)


_SPLITTERS_GROUP_AWARE = {
    "kennard_stone": _rodar_kennard_stone,
    "duplex": _rodar_duplex,
    "spxy": _rodar_spxy,
}


@given(
    nome_splitter=st.sampled_from(sorted(_SPLITTERS_GROUP_AWARE)),
    n_grupos=st.integers(min_value=4, max_value=15),
    n_replicas=st.integers(min_value=1, max_value=4),
    n_features=st.integers(min_value=2, max_value=8),
    frac_cal=st.floats(min_value=0.2, max_value=0.8),
    seed=st.integers(min_value=0, max_value=10_000),
)
# NAO existe (ate' onde verificado, 2026-08-27) um bug HISTORICO conhecido
# especifico desta propriedade -- ao contrario do roundtrip de config
# acima, que fixa @example para reproduzir bugs JA achados. Os 3 exemplos
# abaixo sao cobertura DEFENSIVA de fronteira: `n_grupos=4` e' o limiar
# EXATO onde `kennard_stone_split_group_aware`/`duplex_split_group_aware`/
# `spxy_split_group_aware` ligam o colapso por grupo
# (`len(np.unique(mae_subset)) >= 4`) -- o tipo de condicao onde um erro
# de off-by-one classicamente se esconde. Garantem que os 3 splitters SAO
# exercitados nesse limiar em toda rodada, independente de o `sampled_from`
# aleatorio calhar de sortear cada um la' (Frente 2, Bloco 13d).
@example(nome_splitter="kennard_stone", n_grupos=4, n_replicas=2,
         n_features=3, frac_cal=0.5, seed=0)
@example(nome_splitter="duplex", n_grupos=4, n_replicas=2,
         n_features=3, frac_cal=0.5, seed=0)
@example(nome_splitter="spxy", n_grupos=4, n_replicas=2,
         n_features=3, frac_cal=0.5, seed=0)
def test_split_group_aware_nunca_separa_grupo(
        pq, nome_splitter, n_grupos, n_replicas, n_features, frac_cal, seed):
    rng = np.random.default_rng(seed)
    X_list, y_list, mae_list = [], [], []
    for g in range(n_grupos):
        centro = rng.normal(size=n_features)
        teor = rng.uniform(0, 10)
        for _ in range(n_replicas):
            X_list.append(centro + rng.normal(scale=0.05, size=n_features))
            y_list.append(teor + rng.normal(scale=0.05))
            mae_list.append(f"G{g}")
    X = np.array(X_list)
    y = np.array(y_list)
    mae = np.array(mae_list)

    idx_cal, idx_val = _SPLITTERS_GROUP_AWARE[nome_splitter](pq, X, y, mae, frac_cal)

    grupos_cal = set(mae[idx_cal])
    grupos_val = set(mae[idx_val])
    assert not (grupos_cal & grupos_val), (
        f"[{nome_splitter}] grupo(s) {grupos_cal & grupos_val} apareceram "
        f"em cal E val")
    assert set(idx_cal.tolist()) | set(idx_val.tolist()) == set(range(len(X)))
    assert not (set(idx_cal.tolist()) & set(idx_val.tolist()))


def test_contraprova_split_ingenuo_de_fato_vaza_grupo():
    """Contra-prova: um split ingenuo (metade/metade por POSICAO, ignorando
    grupo) sobre dados construidos para intercalar grupos DEVE vazar --
    prova que a checagem `grupos_cal & grupos_val` usada acima distingue
    split seguro de split vazado (mesmo padrao de
    test_deteccao_de_vazamento_pega_metodo_ficticio_sem_agrupamento em
    test_contrato_validacao_agrupada.py)."""
    mae = np.array(["G1", "G2", "G1", "G2"])   # intercalado de proposito
    n = len(mae)
    idx_cal, idx_val = np.arange(0, n // 2), np.arange(n // 2, n)

    grupos_cal, grupos_val = set(mae[idx_cal]), set(mae[idx_val])
    assert grupos_cal & grupos_val, (
        "o split ingenuo deveria vazar grupo por construcao; se nao "
        "vazou, os dados de teste nao intercalam grupos como esperado e "
        "o detector estaria sendo testado contra um caso vazio")
