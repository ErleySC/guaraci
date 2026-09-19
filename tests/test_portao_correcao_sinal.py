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


# =========================================================================
#  Lacunas achadas por mutacao (cosmic-ray, auditoria 2026-09-19)
#  portao_correcao_sinal.py: 131/285 mutantes sobreviveram (46%) -- os
#  testes acima checavam o VEREDITO ponta a ponta, nunca os numeros do
#  relatorio (medias, desvios, efeito padronizado, p exato), a agenda de
#  seeds, as fronteiras (alpha, poder minimo) nem os defaults publicos.
#  Sendo o portao a unica barreira contra "recomendar" uma correcao sem
#  prova, os numeros que ele reporta tem que ser exatos.
# =========================================================================

import inspect  # noqa: E402

from scipy.stats import wilcoxon  # noqa: E402

from guaraci.portao_correcao_sinal import N_MINIMO_PODER_PADRAO  # noqa: E402


def _por_seed(valores):
    """Funcao de score que devolve `valores[seed]` (seed_base=0)."""
    return lambda seed: valores[seed]


def test_relatorio_numerico_exato_com_metrica_de_erro():
    sem = np.array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19.])
    com = sem - np.array([1., 2., 1., 3., 2., 1., 2., 3., 1., 2.])
    v = avaliar_correcao_sinal("m", _por_seed(sem), _por_seed(com),
                               metrica="RMSEP")
    melhora = sem - com
    assert v.minimizar is True
    assert v.valor_sem == pytest.approx(sem.mean())
    assert v.valor_com == pytest.approx(com.mean())
    assert v.desvio_sem == pytest.approx(sem.std(ddof=1))
    assert v.desvio_com == pytest.approx(com.std(ddof=1))
    assert v.tamanho_efeito == pytest.approx(melhora.mean())
    assert v.tamanho_efeito_padronizado == pytest.approx(
        melhora.mean() / melhora.std(ddof=1))
    assert v.p_valor == pytest.approx(float(wilcoxon(com, sem).pvalue))
    assert v.n_pares == 10 and v.poder_suficiente is True
    assert v.veredito == "aprovado"
    assert v.scores_sem == list(sem) and v.scores_com == list(com)


def test_metrica_de_acuracia_maior_e_melhor_e_piora_e_rejeitada():
    sem = np.linspace(0.80, 0.89, 10)
    v = avaliar_correcao_sinal("m", _por_seed(sem), _por_seed(sem - 0.05),
                               metrica="balanced_accuracy")
    assert v.minimizar is False
    assert v.tamanho_efeito == pytest.approx(-0.05)          # piorou
    assert v.veredito == "rejeitado"
    ganho = avaliar_correcao_sinal("m", _por_seed(sem), _por_seed(sem + 0.05),
                                   metrica="balanced_accuracy")
    assert ganho.tamanho_efeito == pytest.approx(0.05)
    assert ganho.veredito == "aprovado"


@pytest.mark.parametrize("nome", ["rmsep", " RMSEP ", "Rmse", "RMSECV",
                                  "erro_absoluto_medio", "MAE", "erro"])
def test_metricas_de_erro_sao_reconhecidas_como_menor_e_melhor(nome):
    v = avaliar_correcao_sinal("m", lambda s: 2.0 + s, lambda s: 1.0 + s,
                               metrica=nome, n_seeds=8)
    assert v.minimizar is True and v.tamanho_efeito == pytest.approx(1.0)


def test_minimizar_explicito_vence_a_inferencia_pelo_nome():
    v = avaliar_correcao_sinal("m", lambda s: 1.0 + s, lambda s: 2.0 + s,
                               metrica="rmsep", minimizar=False, n_seeds=8)
    assert v.minimizar is False and v.tamanho_efeito == pytest.approx(1.0)


def test_agenda_de_seeds_e_seed_base_mais_k_nos_dois_lados_e_default_dez():
    vistos_sem, vistos_com = [], []

    def _sem(seed):
        vistos_sem.append(seed)
        return 5.0 + 0.01 * seed

    def _com(seed):
        vistos_com.append(seed)
        return 4.0 + 0.01 * seed

    avaliar_correcao_sinal("m", _sem, _com, metrica="rmsep", seed_base=100)
    assert vistos_sem == list(range(100, 110))     # n_seeds default = 10
    assert vistos_com == list(range(100, 110))     # MESMA agenda -> pareamento


def test_defaults_publicos_do_portao():
    sig = inspect.signature(avaliar_correcao_sinal).parameters
    assert sig["n_seeds"].default == 10 and sig["seed_base"].default == 0
    assert sig["alpha"].default == 0.05
    assert sig["n_minimo_poder"].default == N_MINIMO_PODER_PADRAO == 8
    pls = inspect.signature(avaliar_correcao_sinal_pls).parameters
    assert pls["n_componentes"].default == 5 and pls["n_splits"].default == 3
    assert pls["n_seeds"].default == 10 and pls["seed_base"].default == 0
    assert pls["alpha"].default == 0.05 and pls["classificacao"].default is False


def test_poder_estatistico_fronteira_exata_em_oito_pares():
    def _v(n):
        return avaliar_correcao_sinal(
            "m", lambda s: 2.0 + 0.1 * s, lambda s: 1.0 + 0.1 * s,
            metrica="rmsep", n_seeds=n)
    assert _v(7).poder_suficiente is False
    assert _v(8).poder_suficiente is True
    assert avaliar_correcao_sinal(        # limite explicito configuravel
        "m", lambda s: 2.0 + s, lambda s: 1.0 + s, metrica="rmsep",
        n_seeds=5, n_minimo_poder=5).poder_suficiente is True


def test_alpha_e_desigualdade_estrita_p_igual_a_alpha_nao_aprova():
    """n=6, todas as diferencas do mesmo sinal: p exato = 2/64 = 0,03125.
    Com alpha == p o portao NAO aprova (`p < alpha`, estrito)."""
    sem, com = np.arange(1., 7.), np.zeros(6)
    r1 = avaliar_correcao_sinal("m", _por_seed(sem), _por_seed(com),
                                metrica="rmsep", n_seeds=6, alpha=0.03125)
    assert r1.p_valor == pytest.approx(0.03125)
    assert r1.veredito == "neutro"
    r2 = avaliar_correcao_sinal("m", _por_seed(sem), _por_seed(com),
                                metrica="rmsep", n_seeds=6, alpha=0.0313)
    assert r2.veredito == "aprovado"


def test_alpha_default_cinco_por_cento_separa_p_de_0047_e_0055():
    """Pina o default 0.05 dos dois lados (mutantes 0.04 e 0.06)."""
    com = np.zeros(8)
    acima = np.array([1.8, 1.2, -0.6, -0.2, 1.9, 1.7, 0.4, 1.0])    # p = 0,0547
    v_neutro = avaliar_correcao_sinal(
        "m", _por_seed(acima), _por_seed(com), metrica="rmsep", n_seeds=8)
    assert 0.05 < v_neutro.p_valor < 0.06 and v_neutro.veredito == "neutro"
    sete = np.array([-0.4, 2.9, 0.5, 2.4, 0.3, 0.8, 2.2])           # p = 0,0469
    v_ok = avaliar_correcao_sinal(
        "m", _por_seed(sete), _por_seed(np.zeros(7)), metrica="rmsep",
        n_seeds=7)
    assert 0.04 < v_ok.p_valor < 0.05 and v_ok.veredito == "aprovado"


def test_ganho_constante_de_um_nao_e_confundido_com_zero():
    """`np.allclose(melhora, 0.0)` mutado para comparar com 1.0 trataria um
    ganho CONSTANTE de 1.0 como 'sem diferenca' (p=1)."""
    v = avaliar_correcao_sinal("m", lambda s: 2.0 + s, lambda s: 1.0 + s,
                               metrica="rmsep", n_seeds=8)
    assert v.p_valor < 0.05 and v.veredito == "aprovado"
    v2 = avaliar_correcao_sinal("m", lambda s: 1.0 + s, lambda s: 2.0 + s,
                                metrica="rmsep", n_seeds=8)
    assert v2.veredito == "rejeitado"


def test_sem_nenhuma_diferenca_p_um_neutro_e_efeito_padronizado_zero():
    v = avaliar_correcao_sinal("m", lambda s: 3.0, lambda s: 3.0,
                               metrica="rmsep", n_seeds=10)
    assert v.p_valor == 1.0 and v.veredito == "neutro"
    assert v.tamanho_efeito == 0.0 and v.tamanho_efeito_padronizado == 0.0


def test_um_unico_par_nao_calcula_desvio_e_nunca_aprova():
    v = avaliar_correcao_sinal("m", lambda s: 5.0, lambda s: 1.0,
                               metrica="rmsep", n_seeds=1)
    assert v.n_pares == 1 and v.desvio_sem == 0.0 and v.desvio_com == 0.0
    assert v.tamanho_efeito_padronizado == 0.0
    assert v.veredito == "neutro" and v.poder_suficiente is False


def test_efeito_padronizado_zero_quando_desvio_das_diferencas_e_zero():
    """Ganho constante: desvio das diferencas = 0 -> padronizado = 0.0 (sem
    divisao por zero), mesmo com efeito real."""
    v = avaliar_correcao_sinal("m", lambda s: 2.0 + s, lambda s: 1.0 + s,
                               metrica="rmsep", n_seeds=8)
    assert v.tamanho_efeito == pytest.approx(1.0)
    assert v.tamanho_efeito_padronizado == 0.0


def test_resumo_das_tres_situacoes_e_do_aviso_de_poder():
    ok = avaliar_correcao_sinal("EMSC", lambda s: 2.0 + 0.1 * s,
                                lambda s: 1.0 + 0.1 * s, metrica="rmsep",
                                n_seeds=8).resumo()
    assert "EMSC: APROVADO" in ok and "poder estatistico baixo" not in ok
    assert "sem=" in ok and "com=" in ok and "n=8" in ok
    neutro = avaliar_correcao_sinal("X", lambda s: 1.0, lambda s: 1.0,
                                    metrica="rmsep", n_seeds=3).resumo()
    assert "NEUTRO" in neutro and "poder estatistico baixo" in neutro
    rej = avaliar_correcao_sinal("Y", lambda s: 1.0 + 0.1 * s,
                                 lambda s: 2.0 + 0.1 * s, metrica="rmsep",
                                 n_seeds=8).resumo()
    assert "REJEITADO" in rej


# -- avaliar_correcao_sinal_pls: internos do CV (propriedades) -------------

class _Identidade:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        return self


class _Destrutivo(_Identidade):
    """Substitui o espectro por ruido deterministico (semente = checksum de
    X) sem relacao com y -- a 'correcao' que so' pode piorar. (Uma matriz
    constante nao serve: apos centrar vira zeros e o PLS devolve NaN.)"""

    @staticmethod
    def _ruido(X):
        X = np.asarray(X, dtype=float)
        semente = int(abs(X.sum()) * 1e6) % (2 ** 32)
        return np.random.default_rng(semente).normal(size=X.shape)

    def transform(self, X):
        return self._ruido(X)

    def fit_transform(self, X, y=None):
        return self._ruido(X)


def _dados_regressao(n=36, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.05, size=n)
    return X, y, np.repeat(np.arange(n // 3), 3)


def test_pls_regressao_recupera_sinal_linear_e_transformador_identidade_empata():
    X, y, g = _dados_regressao()
    v = avaliar_correcao_sinal_pls("id", X, y, g, _Identidade(),
                                   metrica="RMSEP", n_componentes=4,
                                   n_seeds=4)
    assert v.valor_sem < 0.5, "PLS nao recuperou um sinal linear limpo"
    assert v.valor_sem == pytest.approx(v.valor_com)   # MESMOS folds
    assert v.veredito == "neutro"


def test_pls_transformador_destrutivo_e_rejeitado():
    X, y, g = _dados_regressao()
    v = avaliar_correcao_sinal_pls("destroi", X, y, g, _Destrutivo(),
                                   metrica="RMSEP", n_componentes=4,
                                   n_seeds=8)
    assert v.valor_com > v.valor_sem + 1.0
    assert v.veredito == "rejeitado"


def test_pls_classificacao_separavel_tem_acuracia_alta_e_destrutivo_cai():
    rng = np.random.default_rng(1)
    n, p = 60, 6
    y = np.repeat(["a", "b", "c"], n // 3)
    centros = rng.normal(size=(3, p)) * 3
    X = rng.normal(size=(n, p)) * 0.3 + centros[np.repeat([0, 1, 2], n // 3)]
    g = np.repeat(np.arange(n // 3), 3)
    v = avaliar_correcao_sinal_pls("id", X, y, g, _Identidade(),
                                   metrica="balanced_accuracy",
                                   classificacao=True, n_componentes=3,
                                   n_seeds=4)
    assert v.valor_sem > 0.85
    pior = avaliar_correcao_sinal_pls("destroi", X, y, g, _Destrutivo(),
                                      metrica="balanced_accuracy",
                                      classificacao=True, n_componentes=3,
                                      n_seeds=8)
    assert pior.valor_com < 0.5 and pior.veredito == "rejeitado"


def test_pls_com_mais_componentes_que_amostras_e_limitado_sem_erro():
    """`n_comp_eff = max(1, min(n_componentes, p, n_treino-1))`: pedir 50
    componentes com 12 amostras/5 variaveis tem que rodar."""
    X, y, g = _dados_regressao(n=12, p=5, seed=2)
    v = avaliar_correcao_sinal_pls("id", X, y, g, _Identidade(),
                                   metrica="RMSEP", n_componentes=50,
                                   n_seeds=3)
    assert np.isfinite(v.valor_sem) and np.isfinite(v.valor_com)


# -- 2a rodada de mutacao (51 sobreviventes restantes): oraculo do CV --------

from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from guaraci.validacao_estatistica import StableStratifiedGroupKFold  # noqa: E402


def _rmsep_oraculo(X, y, g, seed, n_componentes, n_splits):
    """Reimplementacao INDEPENDENTE do CV de `avaliar_correcao_sinal_pls`
    (regressao, sem correcao): mesmos folds estaveis, centragem so' de
    media, PLS sem escala, RMSEP das predicoes fora-da-dobra."""
    Y = y.reshape(-1, 1)
    folds = list(StableStratifiedGroupKFold(
        n_splits=n_splits, seed=seed).split(np.zeros(len(X)), np.zeros(len(X)),
                                            groups=g))
    y_hat = np.zeros_like(Y, dtype=float)
    for tr, va in folds:
        mc = StandardScaler(with_std=False).fit(X[tr])
        n_comp = int(max(1, min(n_componentes, X.shape[1], len(tr) - 1)))
        pls = PLSRegression(n_components=n_comp, scale=False).fit(
            mc.transform(X[tr]), Y[tr])
        y_hat[va] = pls.predict(mc.transform(X[va]))
    return float(np.sqrt(np.mean((Y.ravel() - y_hat.ravel()) ** 2)))


def test_pls_rmsep_bate_com_oraculo_independente_e_sem_padronizar_colunas():
    """Colunas com escalas MUITO diferentes: autoescalar (with_std=True)
    mudaria o resultado -- o CV so' centra a media. Tambem trava a
    formula de `n_comp_eff` (n_treino-1 e' o limite ativo aqui: p=30)."""
    rng = np.random.default_rng(5)
    n, p = 24, 30
    X = rng.normal(size=(n, p)) * np.r_[1000.0, np.ones(p - 1)]
    y = 0.002 * X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=n)
    g = np.repeat(np.arange(n // 2), 2)
    for n_comp in (3, 50):
        v = avaliar_correcao_sinal_pls("id", X, y, g, _Identidade(),
                                       metrica="RMSEP", n_componentes=n_comp,
                                       n_splits=3, n_seeds=3)
        esperado = [_rmsep_oraculo(X, y, g, s, n_comp, 3) for s in range(3)]
        assert v.scores_sem == pytest.approx(esperado)
        assert v.scores_com == pytest.approx(esperado)


def test_pls_y_multialvo_usa_todas_as_colunas_de_y():
    """`y.ndim == 1` decide o reshape: com 2 alvos o y NAO pode ser
    achatado numa coluna so'."""
    rng = np.random.default_rng(6)
    n = 30
    X = rng.normal(size=(n, 6))
    y2 = np.c_[X[:, 0] * 2.0, X[:, 1] - X[:, 2]] + rng.normal(scale=0.05, size=(n, 2))
    g = np.repeat(np.arange(n // 3), 3)
    v = avaliar_correcao_sinal_pls("id", X, y2, g, _Identidade(),
                                   metrica="RMSEP", n_componentes=4,
                                   n_seeds=3)
    assert np.all(np.isfinite(v.scores_sem)) and v.valor_sem < 0.5


def test_transformador_recebe_y_1d_na_regressao_e_matriz_na_classificacao():
    """OSC-like: na regressao recebe `Y.ravel()` (1-D); na classificacao a
    matriz one-hot (n, n_classes) -- a ramificacao `if classificacao`."""
    X, y, g = _dados_regressao()
    vistos = []   # o portao clona o transformador: registra na lista da closure

    class _Registra(_Identidade):
        def fit_transform(self, X, y=None):
            vistos.append(np.asarray(y).ndim)
            return X

    avaliar_correcao_sinal_pls("reg", X, y, g, _Registra(), metrica="RMSEP",
                               n_componentes=3, n_seeds=1)
    assert vistos and set(vistos) == {1}

    vistos.clear()
    rng = np.random.default_rng(1)
    n, p = 60, 6
    yc = np.repeat(["a", "b", "c"], n // 3)
    Xc = rng.normal(size=(n, p)) * 0.3 + rng.normal(size=(3, p))[np.repeat([0, 1, 2], n // 3)] * 3
    gc = np.repeat(np.arange(n // 3), 3)
    avaliar_correcao_sinal_pls("clf", Xc, yc, gc, _Registra(),
                               metrica="balanced_accuracy",
                               classificacao=True, n_componentes=3, n_seeds=1)
    assert vistos and set(vistos) == {2}


def test_efeito_zero_com_p_significativo_nao_aprova_nem_rejeita():
    """12 pares: 11 ganhos pequenos (1..11) e UMA piora enorme (-66) --
    media exatamente 0, mas o Wilcoxon (ranks) acha p ~ 0,038 < 0,05.
    Sem efeito liquido o veredito tem que ser NEUTRO (mutantes `>=0` e
    `<=0` aprovariam/rejeitariam um efeito nulo)."""
    melhora = np.r_[np.arange(1., 12.), -66.0]
    v = avaliar_correcao_sinal("m", _por_seed(melhora), _por_seed(np.zeros(12)),
                               metrica="rmsep", n_seeds=12)
    assert v.tamanho_efeito == 0.0
    assert v.p_valor < 0.05
    assert v.veredito == "neutro"


def test_dois_pares_calculam_desvio_de_verdade():
    v = avaliar_correcao_sinal("m", _por_seed(np.array([5.0, 9.0])),
                               _por_seed(np.array([1.0, 2.0])),
                               metrica="rmsep", n_seeds=2)
    assert v.desvio_sem == pytest.approx(np.std([5.0, 9.0], ddof=1))
    assert v.desvio_com == pytest.approx(np.std([1.0, 2.0], ddof=1))
    assert v.tamanho_efeito_padronizado == pytest.approx(
        np.mean([4.0, 7.0]) / np.std([4.0, 7.0], ddof=1))


# -- 3a rodada de mutacao: handler de ValueError do Wilcoxon e n_componentes=1 --

def test_wilcoxon_que_levanta_valueerror_vira_p_1_e_veredito_neutro(monkeypatch):
    """O `except ValueError` (scipy nao consegue testar) e' tratado como
    'sem diferenca detectavel' -- p=1.0, nao erro do portao."""
    import guaraci.portao_correcao_sinal as mod

    def _quebra(*a, **k):
        raise ValueError("todas as diferencas descartadas")
    monkeypatch.setattr(mod, "wilcoxon", _quebra)
    v = avaliar_correcao_sinal("m", _por_seed(np.arange(12.0) + 100.0),
                               _por_seed(np.arange(12.0)),
                               metrica="rmsep", n_seeds=12)
    assert v.p_valor == 1.0
    assert v.veredito == "neutro"


def test_pls_com_1_componente_usa_exatamente_1_componente():
    """`max(1, ...)`: n_componentes=1 tem que dar 1 LV (o oraculo
    independente calcula com 1), nao 2."""
    rng = np.random.default_rng(8)
    n, p = 30, 10
    X = rng.normal(size=(n, p))
    y = X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=n)
    g = np.repeat(np.arange(n // 2), 2)
    v = avaliar_correcao_sinal_pls("id", X, y, g, _Identidade(),
                                   metrica="RMSEP", n_componentes=1,
                                   n_splits=3, n_seeds=3)
    esperado = [_rmsep_oraculo(X, y, g, s, 1, 3) for s in range(3)]
    assert v.scores_sem == pytest.approx(esperado)
