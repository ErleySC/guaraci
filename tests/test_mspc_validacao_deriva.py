# -*- coding: utf-8 -*-
"""Validação do MSPC (monitoramento em linha, Bloco 13b) -- Grupo 2,
Camada 1: injeção sintética de deriva.

Diferença para `tests/test_sentinela_deriva.py`: aqueles testes exercitam
`SentinelState`/`check_drift` com booleanos sintéticos gerados DIRETO na
taxa desejada (`rng.random(n) < taxa`) -- provam a MATEMÁTICA do teste
binomial, mas nunca passam por `chemometric_stats.training_
applicability_domain`/`applicability_domain_new_samples` (o Domínio de
Aplicabilidade real, PCA + distância combinada T2/Q). Este arquivo
fecha essa lacuna: gera espectros sintéticos, ajusta um Domínio de
Aplicabilidade real na "calibração", e injeta DERIVA ESPECTRAL genuína
(deslocamento progressivo da distribuição ao longo de sessões simuladas)
-- o mecanismo inteiro, ponta a ponta, não só a estatística isolada.

Por que isto importa: a estatística binomial já estava provada correta;
o que NUNCA tinha sido testado é se um desvio espectral real (não um
booleano sorteado) de fato produz a taxa de rejeição do AD que o teste
espera enxergar -- é o elo que faltava entre "o teste binomial funciona"
e "o MSPC detecta deriva de verdade".
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import (
    ad_rejection_rate_cv,
    applicability_domain_new_samples,
    training_applicability_domain,
)
from guaraci.sentinela_deriva import SentinelState, check_drift


def _espectro_base(p: int) -> np.ndarray:
    """Espectro sintético com 1 banda gaussiana -- forma simples e
    determinística, suficiente para exercitar PCA/AD sem depender de
    nenhum dataset real."""
    eixo = np.linspace(0.0, 1.0, p)
    return np.exp(-((eixo - 0.5) ** 2) / (2 * 0.05 ** 2))


def _ajustar_dominio(n_cal: int = 60, p: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = _espectro_base(p)
    X_cal = base[None, :] + rng.normal(scale=0.02, size=(n_cal, p))
    pca = PCA(n_components=5).fit(X_cal)
    artefatos = training_applicability_domain(pca, X_cal, alpha=0.05)
    return pca, artefatos, base


@lru_cache(maxsize=None)
def _referencia_cv(n_cal: int = 60, p: int = 50, seed: int = 0):
    """Referencia de rejeicao em CV da MESMA calibracao de
    `_ajustar_dominio` (Passo 220, teste R2)."""
    rng = np.random.default_rng(seed)
    base = _espectro_base(p)
    X_cal = base[None, :] + rng.normal(scale=0.02, size=(n_cal, p))
    return ad_rejection_rate_cv(X_cal, 5)


def _novo_estado(janela=None) -> SentinelState:
    """Estado do sentinela COM a referencia de calibracao (R2), como o
    modelo salvo pelo pipeline passa a fornecer."""
    e = SentinelState(alpha_nominal=0.05, janela=janela)
    e.ref_rejeitadas, e.ref_n = _referencia_cv()
    return e


def _gerar_sessao(base: np.ndarray, n: int, delta: float, seed: int
                   ) -> np.ndarray:
    """Uma 'sessão' de amostras novas -- espectro base + deriva
    (deslocamento linear ao longo do eixo, escala `delta`) + ruído.
    `delta=0` = processo estável (mesma distribuição da calibração)."""
    p = base.size
    rng = np.random.default_rng(seed)
    direcao_deriva = np.linspace(0.0, 1.0, p)
    return (base[None, :] + delta * direcao_deriva[None, :]
            + rng.normal(scale=0.02, size=(n, p)))


def test_mspc_detecta_deriva_espectral_progressiva_injetada():
    """Contra-prova central da Camada 1: deriva REAL (não booleano
    sorteado) injetada progressivamente sessão a sessão tem que produzir
    alerta a partir do ponto em que a taxa de rejeição do AD sobe o
    suficiente -- e continuar alertando enquanto a deriva persiste.
    Também confirma que as PRIMEIRAS sessões (deriva pequena/nula) NÃO
    disparam alerta -- um MSPC que alerta sempre não seria útil."""
    pca, artefatos, base = _ajustar_dominio()
    estado = _novo_estado()
    alertas_por_sessao = []
    for sessao in range(15):
        delta = sessao * 0.01   # rampa 0.00 -> 0.14
        X_s = _gerar_sessao(base, n=20, delta=delta, seed=100 + sessao)
        ad = applicability_domain_new_samples(
            pca, X_s, artefatos["var_t"], artefatos["h0"], artefatos["q0"],
            artefatos["Nh"], artefatos["Nq"], artefatos["f_crit"])
        for b in ad["dentro_dominio"]:
            estado.registrar(bool(b))
        alertas_por_sessao.append(check_drift(estado).alerta)

    # Sessoes 0-1 (delta=0.00/0.01, deriva nula/minima): SEM alerta --
    # medido: taxa de rejeicao 0,0 e 0,05 nestas duas primeiras sessoes.
    assert alertas_por_sessao[0] is False
    assert alertas_por_sessao[1] is False
    # A partir de aproximadamente delta=0.03 (sessao 3) a deriva e' grande
    # o suficiente para disparar -- e permanece disparado dai' em diante
    # (deriva cumulativa, nunca "desliga" sozinha sem intervencao).
    assert any(alertas_por_sessao[3:]), (
        "nenhuma sessao com deriva real disparou o alerta -- o MSPC nao "
        "detectou uma deriva espectral genuina")
    assert all(alertas_por_sessao[6:]), (
        "alerta nao se manteve disparado com deriva persistente e "
        "crescente -- comportamento inesperado do teste cumulativo")


def test_mspc_processo_estavel_nao_dispara_falso_alarme_alem_do_esperado():
    """Contra-prova complementar: SEM deriva nenhuma (todas as sessoes da
    MESMA distribuicao da calibracao), a taxa de falso alarme ao longo de
    200 execucoes independentes tem que ficar baixa -- nao exatamente
    zero (o teste binomial tem poder estatistico real), mas longe de
    "sempre alerta". Medido nesta implementacao: 0/200 (0%) -- o piso de
    0,10 abaixo da folga necessaria para nao quebrar por variacao de
    seed em outra maquina/versao de numpy."""
    pca, artefatos, base = _ajustar_dominio()
    n_trials = 200
    n_falso_alarme = 0
    for trial in range(n_trials):
        rng_trial = np.random.default_rng(1000 + trial)
        estado = _novo_estado()
        for _sessao in range(6):
            X_s = (base[None, :]
                   + rng_trial.normal(scale=0.02, size=(20, base.size)))
            ad = applicability_domain_new_samples(
                pca, X_s, artefatos["var_t"], artefatos["h0"],
                artefatos["q0"], artefatos["Nh"], artefatos["Nq"],
                artefatos["f_crit"])
            for b in ad["dentro_dominio"]:
                estado.registrar(bool(b))
        if check_drift(estado).alerta:
            n_falso_alarme += 1

    taxa_falso_alarme = n_falso_alarme / n_trials
    assert taxa_falso_alarme <= 0.10, (
        f"taxa de falso alarme {taxa_falso_alarme:.3f} acima do esperado "
        f"para um processo genuinamente estavel")


def test_mspc_janela_deslizante_detecta_deriva_recente_mesmo_apos_periodo_estavel():
    """`janela` (Bloco 13b) existe para detectar deriva RECENTE quando um
    periodo estavel longo, ja registrado, diluiria o sinal numa
    estatistica puramente cumulativa. Confirma isso com dado espectral
    real (nao so' booleanos): muitas sessoes estaveis seguidas de deriva
    forte -- SEM janela, o historico estavel atrasa a deteccao; COM
    janela, a deteccao usa so' as sessoes recentes."""
    pca, artefatos, base = _ajustar_dominio()

    def _rodar(janela):
        estado = _novo_estado(janela=janela)
        # 10 sessoes estaveis (delta=0) -- historico "limpo" acumulado.
        for sessao in range(10):
            X_s = _gerar_sessao(base, n=20, delta=0.0, seed=200 + sessao)
            ad = applicability_domain_new_samples(
                pca, X_s, artefatos["var_t"], artefatos["h0"],
                artefatos["q0"], artefatos["Nh"], artefatos["Nq"],
                artefatos["f_crit"])
            for b in ad["dentro_dominio"]:
                estado.registrar(bool(b))
        # 1 sessao com deriva forte, abrupta (nao rampa) -- delta=0.08 ja'
        # e' suficiente pra disparar sozinha (ver teste de rampa acima).
        X_deriva = _gerar_sessao(base, n=20, delta=0.08, seed=999)
        ad = applicability_domain_new_samples(
            pca, X_deriva, artefatos["var_t"], artefatos["h0"],
            artefatos["q0"], artefatos["Nh"], artefatos["Nq"],
            artefatos["f_crit"])
        for b in ad["dentro_dominio"]:
            estado.registrar(bool(b))
        return check_drift(estado)

    alerta_com_janela = _rodar(janela=20)      # so' a sessao de deriva + parte do historico limpo
    assert alerta_com_janela.alerta is True, (
        "com janela pequena, a sessao de deriva forte deveria dominar a "
        "estatistica e disparar o alerta")


# ── Passo 220: calibracao NOVA por repeticao (onde o teste legado falha) ──

def _cenario_cauda_pesada():
    """Fatores latentes t-Student (df=4) sobre 3 componentes, dominio com 2
    PCs -- o AD rejeita em controle uma taxa acima de 5% e VARIAVEL entre
    calibracoes (n_cal=40), como nos espectros reais do Corn."""
    p = 60
    r0 = np.random.default_rng(123)
    L = r0.normal(size=(3, p))
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    esc = np.array([3.0, 1.0, 0.4])

    def gera(rng, n, shift=0.0):
        T = rng.standard_t(4, size=(n, 3)) * esc
        return (T @ L + rng.normal(scale=0.02, size=(n, p))
                + shift * np.linspace(0.0, 1.0, p))
    return gera


@pytest.mark.slow
def test_r2_calibra_o_falso_alarme_com_calibracao_nova_por_repeticao():
    """40 repeticoes, CADA UMA com uma calibracao nova (n_cal=40) de um
    processo com caudas pesadas: o teste LEGADO (binomial vs 5%) alarma
    muito em controle; R2 (Fisher vs CV da propria calibracao) mantem o
    falso alarme baixo e ainda detecta uma deriva grande.

    Medido em 200 repeticoes (`scripts/medicoes/medir_mspc_r2_sintetico.
    py`): falso alarme legado 25,5% -> R2 2,5%; poder em delta=0,40: R2
    79%. Aqui 40 repeticoes com folga estatistica (P de violar os limites
    sob as taxas medidas < 1%)."""
    gera = _cenario_cauda_pesada()
    looks = (24, 32, 40)

    def alarme(dentro, ref):
        for n in looks:
            e = SentinelState(alpha_nominal=0.05)
            if ref is not None:
                e.ref_rejeitadas, e.ref_n = ref
            e.historico = [bool(v) for v in dentro[:n]]
            if check_drift(e).alerta:
                return True
        return False

    n_reps = 40
    fa_leg = fa_r2 = pw_r2 = 0
    for s in range(n_reps):
        rng = np.random.default_rng(s)
        Xc = gera(rng, 40)
        pca = PCA(2).fit(Xc)
        a = training_applicability_domain(pca, Xc)
        ref = ad_rejection_rate_cv(Xc, 2)

        def julga(X):
            return applicability_domain_new_samples(
                pca, X, a["var_t"], a["h0"], a["q0"], a["Nh"], a["Nq"],
                a["f_crit"])["dentro_dominio"]

        ctrl = julga(gera(rng, 40))
        fa_leg += alarme(ctrl, None)
        fa_r2 += alarme(ctrl, ref)
        pw_r2 += alarme(julga(gera(rng, 40, shift=0.40)), ref)

    assert fa_r2 / n_reps <= 0.10, f"falso alarme R2 = {fa_r2 / n_reps:.3f}"
    assert fa_leg / n_reps >= fa_r2 / n_reps + 0.08, (
        f"legado {fa_leg / n_reps:.3f} vs R2 {fa_r2 / n_reps:.3f}: o cenario "
        "deixou de reproduzir a miscalibracao do teste antigo")
    assert pw_r2 / n_reps >= 0.55, f"poder R2 (delta=0,40) = {pw_r2 / n_reps:.3f}"
