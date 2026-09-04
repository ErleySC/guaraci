# -*- coding: utf-8 -*-
"""portao_correcao_sinal.py -- Portao de aceite automatico para tecnicas
de correcao de sinal (Bloco 20).

PROBLEMA QUE RESOLVE. Ate' esta rodada, uma tecnica de correcao de sinal
(EMSC/OSC, transferencia de calibracao PDS/DS, pooled-vs-local, selecao
de variaveis) podia ser adicionada ao leque configuravel e "recomendada"
via `technique_registry`/assistente `G` sem NENHUMA prova de que ela
melhora o resultado no caso real do usuario -- so' com teste ESTRUTURAL
(a funcao roda sem excecao, produz a forma certa de saida). O portao
fecha essa lacuna: roda o MESMO pipeline com e sem a tecnica, sob o
MESMO split bloqueado (group-aware, por sessao/objeto fisico -- nunca
split aleatorio simples) repetido em `n_seeds` particoes independentes,
e decide via teste de Wilcoxon PAREADO (mesmo metodo ja' usado e validado
na comparacao N-PLS vs. PLS-DA por pixel, Passo 132) se a diferenca e'
estatisticamente real.

CONVENCAO DE CHAMADA (por que nao e' `avaliar_correcao_sinal(metodo,
dados, split_bloqueado, metrica)` literal): "o pipeline" e' completamente
diferente entre quantificacao (PLS-R + RMSEP), classificacao (PLS-DA +
balanced_accuracy) e transferencia de calibracao (regressao PDS/DS +
RMSEP no instrumento escravo) -- um unico formato de "dados" nao serve
aos tres. A funcao central (`avaliar_correcao_sinal`) recebe, em vez de
dados brutos, DUAS FUNCOES `avaliar_sem_fn(seed) -> metrica` e
`avaliar_com_fn(seed) -> metrica`: cada uma roda o pipeline INTEIRO
(preprocessamento + split group-aware + fit + predict + metrica) para um
dado `seed`, com/sem a tecnica testada. Isso preserva o requisito real --
MESMO split bloqueado nos dois lados, repetido em `n_seeds` particoes --
sem forcar um formato de dado que nao serve a todo caso de uso. Para o
caso comum (alternar UM transformer sklearn dentro de um Pipeline PLS
fixo, o caso de EMSC/OSC/PDS/DS), `avaliar_correcao_sinal_pls` e' o
atalho que recebe dados de verdade (X, y, grupos)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.stats import wilcoxon

__all__ = [
    "VeredictoCorrecaoSinal",
    "avaliar_correcao_sinal",
    "avaliar_correcao_sinal_pls",
]

#: Metricas onde MENOR e' melhor (erro) -- qualquer coisa fora desta lista
#: e' tratada como "maior e' melhor" (acuracia/Q2/R2/etc), a menos que
#: `minimizar` seja passado explicitamente.
_METRICAS_MENOR_MELHOR = {
    "rmsep", "rmse", "rmsecv", "erro_absoluto_medio", "mae", "erro",
}

#: Minimo de pares (seeds) pro teste de Wilcoxon ser considerado
#: confiavel. n=6 e' o minimo TEORICO pro teste de sinal bicaudal a
#: alpha=0.05 conseguir rejeitar (todas as diferencas com o mesmo sinal,
#: nenhuma zero) -- n<6 nunca detecta nada, mesmo com efeito real. n=8 da
#: uma margem pratica (permite ate' 1-2 pares empatados/invertidos por
#: ruido sem perder poder de detectar um efeito real consistente).
N_MINIMO_PODER_PADRAO = 8


@dataclass
class VeredictoCorrecaoSinal:
    """Resultado do portao para UMA tecnica, em UM dataset/cenario.

    `tamanho_efeito` esta' na unidade da propria metrica (ex.: pontos de
    RMSEP), sempre no sentido "positivo = a correcao ajudou" (ja' ajustado
    pela direcao de `minimizar`). `tamanho_efeito_padronizado` e' esse
    mesmo numero dividido pelo desvio-padrao das diferencas pareadas
    (aproximacao de Cohen's d para amostras pareadas) -- compara-vel entre
    metricas de escalas diferentes.

    `veredito` nunca e' calculado sem reportar `poder_suficiente` junto --
    um "neutro"/"rejeitado" com `poder_suficiente=False` significa "nao da'
    pra' concluir nada com confianca", nao "confirmadamente sem efeito"."""
    metodo: str
    metrica: str
    minimizar: bool
    valor_sem: float
    valor_com: float
    desvio_sem: float
    desvio_com: float
    tamanho_efeito: float
    tamanho_efeito_padronizado: float
    p_valor: float
    n_pares: int
    poder_suficiente: bool
    veredito: str
    scores_sem: List[float] = field(default_factory=list)
    scores_com: List[float] = field(default_factory=list)

    def resumo(self) -> str:
        """Uma linha, pronta pra' exibir na interface/manifesto -- nunca
        esconder o veredito em log interno (exigencia do Bloco 20)."""
        rotulo = {"aprovado": "APROVADO (ganho comprovado)",
                  "rejeitado": "REJEITADO (piora ou sem ganho)",
                  "neutro": "NEUTRO (sem diferenca detectavel)"}[self.veredito]
        aviso_poder = "" if self.poder_suficiente else " [poder estatistico baixo -- conclusao fraca]"
        return (f"{self.metodo}: {rotulo} -- {self.metrica} "
                f"sem={self.valor_sem:.4g} com={self.valor_com:.4g} "
                f"(p={self.p_valor:.3f}, n={self.n_pares}){aviso_poder}")


def avaliar_correcao_sinal(
        nome_metodo: str,
        avaliar_sem_fn: Callable[[int], float],
        avaliar_com_fn: Callable[[int], float],
        *, metrica: str,
        minimizar: Optional[bool] = None,
        n_seeds: int = 10, seed_base: int = 0,
        alpha: float = 0.05,
        n_minimo_poder: int = N_MINIMO_PODER_PADRAO,
        ) -> VeredictoCorrecaoSinal:
    """Portao de aceite generico: chama `avaliar_sem_fn(seed)` e
    `avaliar_com_fn(seed)` para `seed` em `seed_base..seed_base+n_seeds-1`
    (cada chamada deve rodar o MESMO split group-aware bloqueado por
    `seed` -- responsabilidade de quem implementa as funcoes, ver
    docstring do modulo), compara via Wilcoxon pareado, devolve o
    veredito estruturado."""
    if minimizar is None:
        minimizar = metrica.strip().lower() in _METRICAS_MENOR_MELHOR

    scores_sem = [float(avaliar_sem_fn(seed_base + k)) for k in range(n_seeds)]
    scores_com = [float(avaliar_com_fn(seed_base + k)) for k in range(n_seeds)]

    sem_arr = np.array(scores_sem)
    com_arr = np.array(scores_com)
    melhora = (sem_arr - com_arr) if minimizar else (com_arr - sem_arr)

    n_pares = len(melhora)
    poder_suficiente = n_pares >= n_minimo_poder

    if np.allclose(melhora, 0.0):
        p_valor = 1.0
    else:
        try:
            resultado_teste = wilcoxon(com_arr, sem_arr)
            p_valor = float(resultado_teste.pvalue)
        except ValueError:
            # todas as diferencas near-zero apos o zero_method descartar --
            # scipy levanta ValueError em vez de devolver p=1; tratamos
            # como "sem diferenca detectavel", nao como erro do portao.
            p_valor = 1.0

    tamanho_efeito = float(melhora.mean())
    desvio_melhora = float(melhora.std(ddof=1)) if n_pares > 1 else 0.0
    tamanho_efeito_padronizado = (tamanho_efeito / desvio_melhora
                                  if desvio_melhora > 1e-300 else 0.0)

    if p_valor < alpha and tamanho_efeito > 0:
        veredito = "aprovado"
    elif p_valor < alpha and tamanho_efeito < 0:
        veredito = "rejeitado"
    else:
        veredito = "neutro"

    return VeredictoCorrecaoSinal(
        metodo=nome_metodo, metrica=metrica, minimizar=minimizar,
        valor_sem=float(sem_arr.mean()), valor_com=float(com_arr.mean()),
        desvio_sem=float(sem_arr.std(ddof=1)) if n_pares > 1 else 0.0,
        desvio_com=float(com_arr.std(ddof=1)) if n_pares > 1 else 0.0,
        tamanho_efeito=tamanho_efeito,
        tamanho_efeito_padronizado=tamanho_efeito_padronizado,
        p_valor=p_valor, n_pares=n_pares, poder_suficiente=poder_suficiente,
        veredito=veredito, scores_sem=scores_sem, scores_com=scores_com)


def avaliar_correcao_sinal_pls(
        nome_metodo: str, X: np.ndarray, y: np.ndarray, grupos: np.ndarray,
        transformador_correcao, *, metrica: str, n_componentes: int = 5,
        n_splits: int = 3, n_seeds: int = 10, seed_base: int = 0,
        alpha: float = 0.05, classificacao: bool = False,
        ) -> VeredictoCorrecaoSinal:
    """Atalho do portao pro caso comum: alternar UM transformer sklearn
    (`transformador_correcao`, ex. `EMSC()`/`OSC()`) dentro de um Pipeline
    PLS fixo (PLS-R se `classificacao=False`, `metrica='RMSEP'`; PLS-DA se
    `classificacao=True`, `metrica='balanced_accuracy'`), sob CV
    group-aware (`StableStratifiedGroupKFold`) -- MESMOS folds nos dois
    lados (com/sem) pra' cada seed, garantindo o pareamento exigido pelo
    teste. `transformador_correcao` precisa seguir a interface sklearn
    (`fit`/`transform`, ou `fit(X,y)` se for supervisionado como OSC --
    clonado via `sklearn.base.clone` a cada fold, nunca reaproveitado
    "sujo" de um fold anterior)."""
    from sklearn.base import clone
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import LabelBinarizer, StandardScaler

    from guaraci.validacao_estatistica import StableStratifiedGroupKFold

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    grupos = np.asarray(grupos)

    if classificacao:
        lb = LabelBinarizer()
        Y = np.asarray(lb.fit_transform(y))
        if Y.ndim == 1 or Y.shape[1] == 1:
            Y = np.hstack([1 - Y.reshape(-1, 1), Y.reshape(-1, 1)])
        y_estratificacao = y
    else:
        Y = y.reshape(-1, 1) if y.ndim == 1 else y
        y_estratificacao = None

    def _rodar(seed: int, usar_correcao: bool) -> float:
        splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
        alvo_estratificacao = y_estratificacao if y_estratificacao is not None \
            else np.zeros(len(X))
        folds = list(splitter.split(np.zeros(len(X)), alvo_estratificacao, groups=grupos))

        y_hat_geral = np.zeros_like(Y, dtype=float)
        contador = np.zeros(len(Y), dtype=int)
        for idx_tr, idx_va in folds:
            X_tr, X_va = X[idx_tr].copy(), X[idx_va].copy()
            if usar_correcao:
                transf = clone(transformador_correcao)
                if classificacao:
                    X_tr = transf.fit_transform(X_tr, Y[idx_tr])
                else:
                    X_tr = transf.fit_transform(X_tr, Y[idx_tr].ravel())
                X_va = transf.transform(X_va)
            mc = StandardScaler(with_std=False)
            X_tr = mc.fit_transform(X_tr)
            X_va = mc.transform(X_va)
            n_comp_eff = int(max(1, min(n_componentes, X_tr.shape[1], len(idx_tr) - 1)))
            pls = PLSRegression(n_components=n_comp_eff, scale=False)
            pls.fit(X_tr, Y[idx_tr])
            y_hat_geral[idx_va] += np.asarray(pls.predict(X_va), dtype=float)
            contador[idx_va] += 1
        contador[contador == 0] = 1
        y_hat_geral = y_hat_geral / contador[:, None]

        if classificacao:
            classes_preditas = lb.classes_[np.argmax(y_hat_geral, axis=1)]
            return float(balanced_accuracy_score(y, classes_preditas))
        erro = Y.ravel() - y_hat_geral.ravel()
        return float(np.sqrt(np.mean(erro ** 2)))

    return avaliar_correcao_sinal(
        nome_metodo,
        avaliar_sem_fn=lambda seed: _rodar(seed, usar_correcao=False),
        avaliar_com_fn=lambda seed: _rodar(seed, usar_correcao=True),
        metrica=metrica, n_seeds=n_seeds, seed_base=seed_base, alpha=alpha)
