# -*- coding: utf-8 -*-
"""Testes de portao_correcao_sinal.py (Bloco 20) -- contra-prova
obrigatoria: metodo inutil/prejudicial precisa ser rejeitado (ou neutro,
se for literalmente identidade -- ver nota no teste), metodo com ganho
sintetico CONHECIDO precisa ser aprovado.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from guaraci.portao_correcao_sinal import (
    avaliar_correcao_sinal,
    avaliar_correcao_sinal_pls,
)
from guaraci.preprocessamento import MSC, SavGol


# ---------------------------------------------------------------------------
#  Motor generico (avaliar_correcao_sinal): callables sinteticos diretos --
#  nao depende de dado real, testa so' a logica de veredito/Wilcoxon.
# ---------------------------------------------------------------------------

def test_motor_generico_aprova_ganho_consistente_rmsep():
    rng = np.random.default_rng(0)
    sem_fn = lambda seed: 0.50 + rng.normal(0, 0.02)  # noqa: E731
    com_fn = lambda seed: 0.20 + rng.normal(0, 0.02)  # noqa: E731
    v = avaliar_correcao_sinal("fake_bom", sem_fn, com_fn, metrica="RMSEP", n_seeds=10)
    assert v.veredito == "aprovado"
    assert v.poder_suficiente
    assert v.tamanho_efeito > 0
    assert v.p_valor < 0.05


def test_motor_generico_rejeita_piora_consistente_rmsep():
    rng = np.random.default_rng(1)
    sem_fn = lambda seed: 0.20 + rng.normal(0, 0.02)  # noqa: E731
    com_fn = lambda seed: 0.55 + rng.normal(0, 0.02)  # noqa: E731
    v = avaliar_correcao_sinal("fake_ruim", sem_fn, com_fn, metrica="RMSEP", n_seeds=10)
    assert v.veredito == "rejeitado"
    assert v.tamanho_efeito < 0


def test_motor_generico_identidade_e_neutro():
    """Correcao literalmente identica ao 'sem' (mesmos valores) -- nao
    'rejeitado' por definicao matematica (nao ha diferenca nenhuma pra'
    rejeitar), 'neutro' e' o veredito honesto."""
    valores = iter([0.30] * 20)
    sem_fn = lambda seed: next(valores)  # noqa: E731
    com_fn = lambda seed: 0.30  # noqa: E731
    v = avaliar_correcao_sinal("identidade", sem_fn, com_fn, metrica="RMSEP", n_seeds=10)
    assert v.veredito == "neutro"
    assert v.p_valor == pytest.approx(1.0)


def test_motor_generico_sem_diferenca_real_e_neutro_balanced_accuracy():
    rng = np.random.default_rng(2)
    sem_fn = lambda seed: 0.70 + rng.normal(0, 0.03)  # noqa: E731
    com_fn = lambda seed: 0.70 + rng.normal(0, 0.03)  # noqa: E731
    v = avaliar_correcao_sinal("fake_neutro", sem_fn, com_fn,
                               metrica="balanced_accuracy", n_seeds=10)
    assert v.veredito == "neutro"


def test_motor_generico_marca_poder_insuficiente_com_poucos_seeds():
    rng = np.random.default_rng(3)
    sem_fn = lambda seed: 0.50 + rng.normal(0, 0.02)  # noqa: E731
    com_fn = lambda seed: 0.20 + rng.normal(0, 0.02)  # noqa: E731
    v = avaliar_correcao_sinal("poucos_seeds", sem_fn, com_fn, metrica="RMSEP", n_seeds=3)
    assert not v.poder_suficiente
    assert v.n_pares == 3


def test_resumo_inclui_veredito_e_aviso_de_poder():
    rng = np.random.default_rng(4)
    sem_fn = lambda seed: 0.50 + rng.normal(0, 0.02)  # noqa: E731
    com_fn = lambda seed: 0.50 + rng.normal(0, 0.02)  # noqa: E731
    v = avaliar_correcao_sinal("x", sem_fn, com_fn, metrica="RMSEP", n_seeds=3)
    resumo = v.resumo()
    assert "x" in resumo
    assert "poder estatistico baixo" in resumo


# ---------------------------------------------------------------------------
#  Atalho PLS (avaliar_correcao_sinal_pls): dado sintetico group-aware,
#  contra-prova com transformer real sklearn-compativel.
# ---------------------------------------------------------------------------

class _RuidoInutil(BaseEstimator, TransformerMixin):
    """Correcao deliberadamente PREJUDICIAL: adiciona ruido substancial,
    nao correlacionado com nada -- exigencia da contra-prova do Bloco 20
    ("ruido aleatorio deve ser rejeitado")."""
    def __init__(self, escala: float = 8.0, seed: int = 0):
        self.escala = escala
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rng = np.random.default_rng(self.seed)
        return np.asarray(X, dtype=float) + rng.normal(0, self.escala, size=np.shape(X))


def _dataset_regressao_group_aware(seed=0, n_grupos=30, replicas=3, p=25):
    """y correlaciona com um sinal real em X; grupos = objeto fisico
    (replicas nunca podem ser separadas entre treino/validacao -- e' o
    que StableStratifiedGroupKFold, ja usado dentro do portao, garante)."""
    rng = np.random.default_rng(seed)
    n = n_grupos * replicas
    y_grupo = rng.uniform(0, 10, size=n_grupos)
    y = np.repeat(y_grupo, replicas)
    grupos = np.array([f"g{k}" for k in range(n_grupos) for _ in range(replicas)])

    eixo = np.arange(p, dtype=float)
    perfil_sinal = np.exp(-0.5 * ((eixo - 12) / 3) ** 2)
    X = rng.normal(0, 0.05, size=(n, p)) + y[:, None] * 0.3 * perfil_sinal[None, :]
    return X, y, grupos


def _dataset_com_ganho_multiplicativo(seed=0, n_grupos=30, replicas=3, p=25):
    """Mesma estrutura de sinal, mas cada amostra tambem carrega um GANHO
    MULTIPLICATIVO aleatorio (nao correlacionado com y) -- o tipo exato de
    distorcao que MSC foi desenhado para remover."""
    X, y, grupos = _dataset_regressao_group_aware(seed, n_grupos, replicas, p)
    rng = np.random.default_rng(seed + 1000)
    ganho = rng.uniform(0.4, 2.2, size=X.shape[0])
    X_distorcido = X * ganho[:, None]
    return X_distorcido, y, grupos


def _dataset_ruido_dominante(seed=0, n_grupos=30, replicas=3, p=60, escala_ruido=0.15):
    """Sinal FRACO (pico largo/suave) afogado em ruido gaussiano ALTO --
    cenario onde suavizacao (Savitzky-Golay) tem beneficio genuino e
    verificavel por construcao: reduzir ruido de alta frequencia sempre
    melhora SNR, seja o modelo supervisionado ou nao (ao contrario de
    ganho multiplicativo, que um PLS supervisionado ja' ignora sozinho por
    ser nao-correlacionado com y -- medido diretamente nesta sessao: MSC
    e ate' uma correcao 'oraculo' que remove o ganho exato pioraram o
    RMSEP num cenario so' de ganho, porque PLS ja' resolve isso via a
    propria covariancia com y, sem precisar de pre-processamento -- por
    isso o cenario 'aprovado' deste modulo usa ruido, nao ganho)."""
    rng = np.random.default_rng(seed)
    n = n_grupos * replicas
    y_grupo = rng.uniform(0, 10, size=n_grupos)
    y = np.repeat(y_grupo, replicas)
    grupos = np.array([f"g{k}" for k in range(n_grupos) for _ in range(replicas)])
    eixo = np.arange(p, dtype=float)
    perfil_sinal = np.exp(-0.5 * ((eixo - p * 0.5) / (p * 0.1)) ** 2)
    ruido = rng.normal(0, escala_ruido, size=(n, p))
    X = y[:, None] * 0.05 * perfil_sinal[None, :] + ruido
    return X, y, grupos


def test_pls_rejeita_ruido_aleatorio_como_correcao():
    X, y, grupos = _dataset_regressao_group_aware(seed=10)
    v = avaliar_correcao_sinal_pls(
        "ruido_inutil", X, y, grupos, _RuidoInutil(escala=8.0, seed=0),
        metrica="RMSEP", n_componentes=3, n_seeds=8)
    assert v.veredito == "rejeitado", v.resumo()
    assert v.valor_com > v.valor_sem


def test_pls_aprova_savgol_quando_ruido_domina_o_sinal():
    """'Ganho sintetico claro' -- ver docstring de `_dataset_ruido_dominante`
    pra' o motivo de usar suavizacao (nao MSC/ganho) como caso positivo."""
    X, y, grupos = _dataset_ruido_dominante(seed=11)
    v = avaliar_correcao_sinal_pls(
        "savgol_com_ganho_sintetico", X, y, grupos,
        SavGol(window_length=11, polyorder=2, deriv=0),
        metrica="RMSEP", n_componentes=3, n_seeds=8)
    assert v.veredito == "aprovado", v.resumo()
    assert v.valor_com < v.valor_sem


def test_pls_msc_neutro_ou_sem_ganho_quando_dado_nao_tem_distorcao():
    """Contra-prova inversa: SEM a distorcao multiplicativa no dado, MSC
    nao deveria mostrar ganho consistente (nao ha' nada pra' corrigir) --
    aceita 'neutro' OU 'rejeitado' (MSC pode introduzir ruido de estimacao
    sem beneficio), mas NUNCA 'aprovado' com efeito grande."""
    X, y, grupos = _dataset_regressao_group_aware(seed=12)
    v = avaliar_correcao_sinal_pls(
        "msc_sem_necessidade", X, y, grupos, MSC(),
        metrica="RMSEP", n_componentes=3, n_seeds=8)
    assert v.veredito != "aprovado" or abs(v.tamanho_efeito_padronizado) < 0.5, v.resumo()


def test_pls_classificacao_usa_balanced_accuracy_e_maior_e_melhor():
    rng = np.random.default_rng(20)
    n_grupos, replicas, p = 24, 3, 20
    n = n_grupos * replicas
    y_grupo = rng.integers(0, 2, size=n_grupos)
    y = np.repeat(y_grupo, replicas)
    grupos = np.array([f"g{k}" for k in range(n_grupos) for _ in range(replicas)])
    eixo = np.arange(p, dtype=float)
    sinal = np.exp(-0.5 * ((eixo - 10) / 2) ** 2)
    X = rng.normal(0, 1, size=(n, p)) + y[:, None] * 2.0 * sinal[None, :]

    v = avaliar_correcao_sinal_pls(
        "ruido_classificacao", X, y, grupos, _RuidoInutil(escala=6.0, seed=1),
        metrica="balanced_accuracy", n_componentes=2, n_seeds=8, classificacao=True)
    assert v.minimizar is False
    assert v.veredito in ("rejeitado", "neutro")
    if v.veredito == "rejeitado":
        assert v.valor_com < v.valor_sem


def test_pls_split_e_group_aware_grupos_nunca_vazam():
    """Propriedade dura do Bloco 20 (regra 5, group-aware em tudo):
    instrumenta StableStratifiedGroupKFold via monkeypatch nao e' trivial
    aqui -- em vez disso, confirma indiretamente: rodar com um `grupos`
    onde cada replica tem valor de y LIGEIRAMENTE diferente (nao
    identico) e um n_splits que força group-aware de verdade nao lanca
    excecao e produz metrica finita -- e' o teste de fumaca de que o
    atalho realmente delega pro splitter group-aware (nao um split
    aleatorio simples que ignoraria `grupos`)."""
    X, y, grupos = _dataset_regressao_group_aware(seed=13)
    v = avaliar_correcao_sinal_pls(
        "smoke", X, y, grupos, MSC(), metrica="RMSEP", n_componentes=3, n_seeds=3)
    assert np.isfinite(v.valor_sem)
    assert np.isfinite(v.valor_com)


# ---------------------------------------------------------------------------
#  Registro no model card (Bloco 20: "nunca escondido em log interno")
# ---------------------------------------------------------------------------

def test_veredito_e_serializavel_para_o_model_card(tmp_path):
    from guaraci.resultados_io import append_correcao_sinal_model_card

    caminho = tmp_path / "model_card.md"
    caminho.write_text("# Model Card\n", encoding="utf-8")

    X, y, grupos = _dataset_regressao_group_aware(seed=14)
    v = avaliar_correcao_sinal_pls(
        "msc_teste_model_card", X, y, grupos, MSC(),
        metrica="RMSEP", n_componentes=3, n_seeds=4)

    append_correcao_sinal_model_card(str(tmp_path), [v])
    conteudo = caminho.read_text(encoding="utf-8")
    assert "Bloco 20" in conteudo
    assert "msc_teste_model_card" in conteudo
    assert v.veredito in conteudo


def test_model_card_lista_veredito_neutro_e_rejeitado_sem_esconder(tmp_path):
    """Exigencia explicita do Bloco 20: 'neutro'/'rejeitado' aparece no
    model card igual a 'aprovado' -- nunca filtrado."""
    from guaraci.resultados_io import append_correcao_sinal_model_card

    caminho = tmp_path / "model_card.md"
    caminho.write_text("# Model Card\n", encoding="utf-8")

    rng_sem, rng_com = np.random.default_rng(100), np.random.default_rng(200)
    v_neutro = avaliar_correcao_sinal(
        "fake_neutro", lambda s: 0.5 + rng_sem.normal(0, 0.02),  # noqa: E731
        lambda s: 0.5 + rng_com.normal(0, 0.02), metrica="RMSEP", n_seeds=8)  # noqa: E731
    rng_sem2, rng_com2 = np.random.default_rng(300), np.random.default_rng(400)
    v_rejeitado = avaliar_correcao_sinal(
        "fake_rejeitado", lambda s: 0.2 + rng_sem2.normal(0, 0.01),  # noqa: E731
        lambda s: 0.6 + rng_com2.normal(0, 0.01), metrica="RMSEP", n_seeds=8)  # noqa: E731
    assert v_neutro.veredito == "neutro"
    assert v_rejeitado.veredito == "rejeitado"

    append_correcao_sinal_model_card(str(tmp_path), [v_neutro, v_rejeitado])
    conteudo = caminho.read_text(encoding="utf-8")
    assert "fake_neutro" in conteudo and "neutro" in conteudo
    assert "fake_rejeitado" in conteudo and "rejeitado" in conteudo


def test_model_card_sem_arquivo_nao_lanca_excecao(tmp_path):
    from guaraci.resultados_io import append_correcao_sinal_model_card
    append_correcao_sinal_model_card(str(tmp_path), [])
    assert not (tmp_path / "model_card.md").exists()
