# -*- coding: utf-8 -*-
"""mcr_als.py -- Resolucao de Curvas Multivariada por Minimos Quadrados
Alternados (MCR-ALS), Bloco 14.

REFERENCIA (verificada no Crossref em 2026-09-04): Tauler, R. (1995).
"Multivariate curve resolution applied to second order data." Chemometrics
and Intelligent Laboratory Systems, 30(1), 133-146.
DOI: 10.1016/0169-7439(95)00047-X. Revisao de restricoes: Tauler, R. & de
Juan, A. (2006). "Multivariate Curve Resolution." In Practical Guide to
Chemometrics, 2nd ed., ch. 11. DOI: 10.1201/9781420018301.ch11.

PROBLEMA QUE RESOLVE. Um conjunto de espectros de MISTURA D (n_amostras x
n_variaveis) e' modelado como D ~= C @ S^T, onde C (n_amostras x
n_componentes) sao os perfis de concentracao de cada componente puro em
cada amostra e S^T (n_componentes x n_variaveis) sao os espectros puros
de cada componente. Diferente de PCA/PLS, os fatores C e S sao forcados a
ter significado fisico direto (concentracao e espectro, nao combinacoes
lineares abstratas) via restricoes: nao-negatividade (concentracao e
absorbancia NIR nao sao negativas), normalizacao (fixa a escala, que de
outra forma e' arbitraria entre C e S) e, opcionalmente, unimodalidade
(um so pico por perfil -- relevante quando as amostras tem ordem natural,
ex. tempo de eluicao; NAO se aplica a conjuntos de amostras sem ordem).

AMBIGUIDADE ROTACIONAL (limitacao conhecida do metodo, nao um bug desta
implementacao): para qualquer solucao (C, S) que satisfaca as restricoes,
existe em geral uma familia de solucoes alternativas (C @ T, T^-1 @ S^T)
igualmente compativeis com os dados e as restricoes, para matrizes de
rotacao/escala T nao-triviais. MCR-ALS converge para UMA solucao dessa
familia, escolhida pela inicializacao -- nao ha' garantia de unicidade.
Este modulo NUNCA reporta (C, S) como solucao exata; `mcr_als` inclui um
aviso textual, e `avaliar_incerteza_rotacional` estima a sensibilidade da
solucao a' inicializacao rodando multiplos starts e alinhando componentes
por correlacao maxima (nao e' o calculo formal de banda de ambiguidade de
MCR-BANDS/Jaumot et al. -- e' um proxy mais barato, documentado como tal).

AVISO DE ESCOPO (Bloco 23, medido em 2026-09-04 -- ver
`docs/PROGRESSO.md` Passo 125/136 e
`scripts/medicoes/validar_mcr_als_oleos_reais.py`): MCR-ALS e' ferramenta
de INTERPRETACAO e resolucao de componentes espectrais -- NAO
recomendado para quantificar traco/adulterante MINORITARIO sem
informacao supervisionada. Testado contra o acervo real de oleo (duas
combinacoes especie+adulterante, uma com sinal supervisionado forte
[PLS-R Q2=0,82], outra fraco): em NENHUMA das duas o componente
recuperado correlacionou com o teor declarado (melhor |r|=0,17 e 0,09,
ambos p>0,24) -- mesmo quando PLS-R SUPERVISIONADO consegue prever o
teor bem no mesmo dado. Nao e' bug (lack-of-fit baixo confirma
reconstrucao de espectro boa): MCR-ALS e' NAO-SUPERVISIONADO, otimiza
fidelidade de reconstrucao (variancia total explicada), nao correlacao
com uma variavel externa que nunca ve -- quando o adulterante e'
minoritario (1-15%), a maior fonte de variancia espectral raramente e' o
sinal do adulterante. Use PLS-R (`pipeline.pls_regressao_pooled`) pra
quantificar; MCR-ALS serve pra' extrair/inspecionar espectros puros e
proporcoes RELATIVAS quando o objetivo e' entender a mistura, nao prever
um numero com garantia estatistica.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import nnls, linear_sum_assignment

__all__ = [
    "MCRALSResultado",
    "mcr_als",
    "avaliar_incerteza_rotacional",
    "MCRALSResultadoSupervisionado",
    "mcr_als_com_restricao_correlacao",
]

AVISO_AMBIGUIDADE_ROTACIONAL = (
    "MCR-ALS nao tem solucao unica: (C, S) e' UMA solucao dentre uma familia "
    "de rotacoes/escalas igualmente compativeis com os dados sob as "
    "restricoes usadas. Nao trate C/S como o perfil 'verdadeiro' sem checar "
    "a sensibilidade a' inicializacao (ver avaliar_incerteza_rotacional)."
)


@dataclass
class MCRALSResultado:
    """Resultado de uma resolucao MCR-ALS.

    `C` (n_amostras, n_componentes): perfis de concentracao.
    `S` (n_componentes, n_variaveis): perfis espectrais puros (S^T na
    notacao D ~= C @ S).
    `lof_percent`: lack-of-fit final, 100 * ||D - C@S|| / ||D|| (Frobenius).
    `historico_lof`: lof_percent a cada iteracao (para diagnostico de
    convergencia).
    `aviso_ambiguidade_rotacional`: sempre presente, ver modulo docstring.
    """
    C: np.ndarray
    S: np.ndarray
    lof_percent: float
    n_iter: int
    convergiu: bool
    historico_lof: List[float] = field(default_factory=list)
    aviso_ambiguidade_rotacional: str = AVISO_AMBIGUIDADE_ROTACIONAL


def _init_variaveis_puras(D: np.ndarray, n_componentes: int) -> np.ndarray:
    """Inicializa C escolhendo `n_componentes` amostras "mais puras" (as que
    mais diferem do espectro medio, em norma), na ordem em que sao
    encontradas apos deflacao gulosa -- versao simplificada do principio
    SIMPLISMA (Windig & Guilment, 1991): a amostra mais pura de um
    componente e' a que tem menor contribuicao relativa dos outros.
    Deflacao gulosa: escolhe a amostra mais distante da media, remove sua
    direcao, repete -- garante que os `n_componentes` inicializadores nao
    sejam colineares entre si."""
    restante = D.copy()
    escolhidas: List[int] = []
    for _ in range(n_componentes):
        normas = np.linalg.norm(restante, axis=1)
        idx = int(np.argmax(normas))
        escolhidas.append(idx)
        direcao = restante[idx] / (normas[idx] + 1e-300)
        proj = restante @ direcao
        restante = restante - np.outer(proj, direcao)
    C0 = np.clip(D[escolhidas, :] @ D.T, 0.0, None).T  # (n, k), nao-negativo
    if C0.max() > 0:
        C0 = C0 / C0.max()
    return C0


def _passo_minimos_quadrados(alvo: np.ndarray, base: np.ndarray,
                              nao_negativo: bool) -> np.ndarray:
    """Resolve `alvo ~= base @ X` para X (colunas de alvo, uma nnls/lstsq
    por coluna). `base` e' (m, k), `alvo` e' (m, p), retorna X (k, p)."""
    k = base.shape[1]
    p = alvo.shape[1]
    X = np.empty((k, p))
    if nao_negativo:
        for j in range(p):
            X[:, j], _ = nnls(base, alvo[:, j])
    else:
        X, *_ = np.linalg.lstsq(base, alvo, rcond=None)
    return X


def _normalizar_S(S: np.ndarray, C: np.ndarray, modo: str) -> tuple:
    """Normaliza cada linha (componente) de S e reescala a coluna
    correspondente de C para que C @ S nao mude (so resolve a escala
    arbitraria entre C e S, nao a ambiguidade rotacional)."""
    if modo == "soma_unitaria":
        escala = S.sum(axis=1)
    elif modo == "norma_unitaria":
        escala = np.linalg.norm(S, axis=1)
    else:
        raise ValueError(f"modo de normalizacao desconhecido: {modo!r}")
    escala_segura = np.where(np.abs(escala) > 1e-300, escala, 1.0)
    S_norm = S / escala_segura[:, None]
    C_norm = C * escala_segura[None, :]
    return C_norm, S_norm


def _aplicar_unimodalidade(C: np.ndarray) -> np.ndarray:
    """Forca cada coluna de C a ter um so pico: a partir do indice de
    maximo, torna a sequencia nao-crescente para ambos os lados (assume
    que a ordem das linhas de C e' significativa, ex. tempo/profundidade
    -- nao adequado para amostras sem ordem natural)."""
    C_uni = C.copy()
    for j in range(C.shape[1]):
        col = C_uni[:, j]
        pico = int(np.argmax(col))
        for i in range(pico - 1, -1, -1):
            col[i] = min(col[i], col[i + 1])
        for i in range(pico + 1, len(col)):
            col[i] = min(col[i], col[i - 1])
        C_uni[:, j] = col
    return C_uni


def mcr_als(D: np.ndarray, n_componentes: int, *,
            nao_negativo_c: bool = True, nao_negativo_s: bool = True,
            normalizacao: str = "soma_unitaria",
            unimodal_c: bool = False,
            c_inicial: Optional[np.ndarray] = None,
            s_inicial: Optional[np.ndarray] = None,
            max_iter: int = 200, tol: float = 1e-6) -> MCRALSResultado:
    """Resolve D (n_amostras, n_variaveis) ~= C @ S em `n_componentes`
    perfis de concentracao/espectro puros, via ALS com restricoes.

    Passa `c_inicial` (n_amostras, n_componentes) OU `s_inicial`
    (n_componentes, n_variaveis) para controlar a inicializacao (ex. em
    `avaliar_incerteza_rotacional`, que varia a inicializacao de proposito
    para medir sensibilidade). Sem nenhum dos dois, usa deflacao gulosa
    por variaveis puras (ver `_init_variaveis_puras`).

    `unimodal_c=True` so faz sentido quando a ORDEM das linhas de D e'
    significativa (ex. perfil de eluicao); em conjuntos de amostras sem
    ordem natural, deixe False (default)."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise ValueError("D precisa ser 2D (n_amostras, n_variaveis)")
    if n_componentes < 1:
        raise ValueError("n_componentes precisa ser >= 1")

    norma_D = float(np.linalg.norm(D))
    if norma_D < 1e-300:
        raise ValueError("D e' (quase) todo zero -- nada para resolver")

    if s_inicial is not None:
        S = np.asarray(s_inicial, dtype=float).copy()
        C = _passo_minimos_quadrados(D.T, S.T, nao_negativo_c).T
    else:
        C = (np.asarray(c_inicial, dtype=float).copy() if c_inicial is not None
             else _init_variaveis_puras(D, n_componentes))
        S = _passo_minimos_quadrados(D, C, nao_negativo_s)

    historico_lof: List[float] = []
    lof_prev = np.inf
    convergiu = False
    it = 0
    for it in range(1, max_iter + 1):
        S = _passo_minimos_quadrados(D, C, nao_negativo_s)
        if unimodal_c:
            C = _aplicar_unimodalidade(C)
        C_r, S_r = _normalizar_S(S, C, normalizacao)
        C, S = C_r, S_r
        C = _passo_minimos_quadrados(D.T, S.T, nao_negativo_c).T

        residuo = D - C @ S
        lof = 100.0 * float(np.linalg.norm(residuo)) / norma_D
        historico_lof.append(lof)
        if lof_prev - lof < tol and it > 1:
            convergiu = True
            break
        lof_prev = lof

    return MCRALSResultado(C=C, S=S, lof_percent=historico_lof[-1],
                            n_iter=it, convergiu=convergiu,
                            historico_lof=historico_lof)


@dataclass
class MCRALSResultadoSupervisionado(MCRALSResultado):
    """Resultado de `mcr_als_com_restricao_correlacao` -- mesmos campos de
    `MCRALSResultado`, mais o ajuste linear que ancorou o componente alvo
    à referência conhecida.

    `coef_regressao`: `(a, b)` de `C[calibracao, alvo] = a + b*y_referencia`
    na ÚLTIMA iteração (a mesma reta usada para prever `y` de uma amostra
    nova a partir do `C` que o MCR-ALS lhe atribuir: `y_pred = (c - a)/b`).
    `correlacao_calibracao`: correlação de Pearson entre o perfil do
    componente alvo e `y_referencia` nas amostras de calibração, no `C`
    FINAL -- por construção, próxima de 1 quando a restrição convergiu
    (não é evidência de que o componente seja quimicamente o analito
    certo, só de que a restrição foi imposta com sucesso).
    """
    coef_regressao: "tuple[float, float]" = (0.0, 1.0)
    correlacao_calibracao: float = float("nan")


def mcr_als_com_restricao_correlacao(
        D: np.ndarray, n_componentes: int, *,
        indice_componente_alvo: int,
        y_referencia: np.ndarray,
        indices_calibracao: np.ndarray,
        nao_negativo_c: bool = True, nao_negativo_s: bool = True,
        normalizacao: str = "soma_unitaria",
        unimodal_c: bool = False,
        c_inicial: Optional[np.ndarray] = None,
        s_inicial: Optional[np.ndarray] = None,
        max_iter: int = 200, tol: float = 1e-6
        ) -> MCRALSResultadoSupervisionado:
    """MCR-ALS com restrição de correlação (Bayat, Marín-García, Ghasemi &
    Tauler, *Anal. Chim. Acta* 1113:52-65, 2020, DOI
    10.1016/j.aca.2020.03.057) -- proposta T9 da rodada multiagente de
    2026-09-10.

    A CADA iteração, depois do passo usual de atualização de `C`, o perfil
    do componente `indice_componente_alvo` nas amostras de
    `indices_calibracao` é SUBSTITUÍDO pela predição de uma regressão
    linear simples contra `y_referencia` (valores de referência
    conhecidos, ex. teor declarado) -- ancora esse componente ao valor
    conhecido, sem impor a mesma restrição aos DEMAIS componentes nem às
    amostras SEM referência (ex. um conjunto de validação/predição, cujo
    `C` continua livre, sujeito só às restrições usuais de
    não-negatividade/normalização).

    MOTIVAÇÃO (achado #`mcr-als-aviso-de-escopo`, ver módulo): a versão
    NÃO supervisionada foi testada contra o acervo real de óleo e NÃO
    recuperou o adulterante minoritário (Passos 131/136, `|r|` de 0,17 e
    0,09). Esta variante NUNCA foi testada contra esse mesmo caso — é uma
    proposta nova, não uma correção do achado negativo anterior. Retorno
    esperado é INTERPRETATIVO (espectro do adulterante com âncora no teor
    conhecido), não um substituto do PLS-R para quantificação
    (`pipeline.pls_regressao_pooled`/`pls_regression_by_species`) — usar
    com a mesma ressalva de escopo do `mcr_als` não supervisionado.

    `indices_calibracao` NÃO deve incluir amostras de um grupo de
    validação usado para medir desempenho (a restrição usa `y_referencia`
    diretamente -- avaliar nas mesmas amostras seria circular).
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise ValueError("D precisa ser 2D (n_amostras, n_variaveis)")
    if n_componentes < 1:
        raise ValueError("n_componentes precisa ser >= 1")
    if not (0 <= indice_componente_alvo < n_componentes):
        raise ValueError(
            f"indice_componente_alvo={indice_componente_alvo} fora do "
            f"intervalo [0, {n_componentes})")

    y_referencia = np.asarray(y_referencia, dtype=float)
    indices_calibracao = np.asarray(indices_calibracao, dtype=int)
    if len(y_referencia) != len(indices_calibracao):
        raise ValueError(
            f"y_referencia ({len(y_referencia)}) e indices_calibracao "
            f"({len(indices_calibracao)}) precisam do mesmo comprimento.")
    if len(indices_calibracao) < 2:
        raise ValueError(
            "Precisa de >=2 amostras de calibracao para ajustar a reta "
            "de restricao de correlacao.")

    norma_D = float(np.linalg.norm(D))
    if norma_D < 1e-300:
        raise ValueError("D e' (quase) todo zero -- nada para resolver")

    if s_inicial is not None:
        S = np.asarray(s_inicial, dtype=float).copy()
        C = _passo_minimos_quadrados(D.T, S.T, nao_negativo_c).T
    else:
        C = (np.asarray(c_inicial, dtype=float).copy() if c_inicial is not None
             else _init_variaveis_puras(D, n_componentes))
        S = _passo_minimos_quadrados(D, C, nao_negativo_s)

    def _restringir(C_atual: np.ndarray) -> "tuple[np.ndarray, tuple[float, float]]":
        c_calib = C_atual[indices_calibracao, indice_componente_alvo]
        # np.polyfit(x, y, 1) -> [inclinacao, intercepto], ambos p/ y ~= b*x + a
        b, a = np.polyfit(y_referencia, c_calib, deg=1)
        c_calib_ancorado = a + b * y_referencia
        if nao_negativo_c:
            # a restricao de correlacao nao pode reintroduzir concentracao
            # negativa -- mesma disciplina de nao-negatividade do resto
            # do ALS.
            c_calib_ancorado = np.clip(c_calib_ancorado, 0.0, None)
        C_novo = C_atual.copy()
        C_novo[indices_calibracao, indice_componente_alvo] = c_calib_ancorado
        return C_novo, (float(a), float(b))

    historico_lof: List[float] = []
    lof_prev = np.inf
    convergiu = False
    coef_regressao = (0.0, 1.0)
    it = 0
    for it in range(1, max_iter + 1):
        S = _passo_minimos_quadrados(D, C, nao_negativo_s)
        if unimodal_c:
            C = _aplicar_unimodalidade(C)
        C_r, S_r = _normalizar_S(S, C, normalizacao)
        C, S = C_r, S_r
        C = _passo_minimos_quadrados(D.T, S.T, nao_negativo_c).T
        C, coef_regressao = _restringir(C)

        residuo = D - C @ S
        lof = 100.0 * float(np.linalg.norm(residuo)) / norma_D
        historico_lof.append(lof)
        if lof_prev - lof < tol and it > 1:
            convergiu = True
            break
        lof_prev = lof

    c_calib_final = C[indices_calibracao, indice_componente_alvo]
    if np.std(c_calib_final) > 1e-300 and np.std(y_referencia) > 1e-300:
        correlacao = float(np.corrcoef(c_calib_final, y_referencia)[0, 1])
    else:
        correlacao = float("nan")

    return MCRALSResultadoSupervisionado(
        C=C, S=S, lof_percent=historico_lof[-1], n_iter=it,
        convergiu=convergiu, historico_lof=historico_lof,
        coef_regressao=coef_regressao, correlacao_calibracao=correlacao)


def _alinhar_componentes(referencia: np.ndarray, alvo: np.ndarray) -> np.ndarray:
    """Casa colunas de `alvo` (n, k) com as de `referencia` (n, k) pela
    correlacao maxima (assignment otimo via Hungarian/`linear_sum_assignment`
    -- necessario porque MCR-ALS nao preserva a ordem/rotulo dos componentes
    entre execucoes independentes). Retorna a permutacao de indices de
    `alvo` que melhor casa com `referencia`."""
    k = referencia.shape[1]
    custo = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            r = np.corrcoef(referencia[:, i], alvo[:, j])[0, 1]
            custo[i, j] = -abs(r) if np.isfinite(r) else 0.0
    _, ordem = linear_sum_assignment(custo)
    return ordem


def avaliar_incerteza_rotacional(D: np.ndarray, n_componentes: int, *,
                                  n_inicializacoes: int = 5,
                                  seed: int = 0,
                                  **kwargs_mcr_als) -> dict:
    """Roda `mcr_als` com `n_inicializacoes` inicializacoes aleatorias
    diferentes e mede o quanto C/S variam entre execucoes, apos alinhar os
    componentes por correlacao (label switching entre runs e' esperado).

    NAO e' o calculo formal de banda de ambiguidade rotacional (MCR-BANDS,
    Jaumot & Tauler 2010) -- e' um proxy mais barato: desvio-padrao, entre
    execucoes, da proporcao relativa de cada componente por amostra.
    Um desvio alto sinaliza que a solucao e' sensivel a' inicializacao (uso
    de C/S de UMA execucao como se fosse definitivo seria enganoso); um
    desvio baixo e' evidencia (nao prova) de que a rotacao esta' bem
    restringida pelas constraints usadas.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    resultados: List[MCRALSResultado] = []
    for _ in range(n_inicializacoes):
        c0 = rng.random((n, n_componentes))
        res = mcr_als(D, n_componentes, c_inicial=c0, **kwargs_mcr_als)
        resultados.append(res)

    proporcoes_ref = resultados[0].C / (resultados[0].C.sum(axis=1, keepdims=True) + 1e-300)
    todas_alinhadas = [proporcoes_ref]
    for res in resultados[1:]:
        prop = res.C / (res.C.sum(axis=1, keepdims=True) + 1e-300)
        ordem = _alinhar_componentes(proporcoes_ref, prop)
        todas_alinhadas.append(prop[:, ordem])

    pilha = np.stack(todas_alinhadas, axis=0)  # (n_init, n_amostras, k)
    desvio_padrao_proporcao = pilha.std(axis=0)  # (n_amostras, k)

    return {
        "resultados": resultados,
        "desvio_padrao_proporcao": desvio_padrao_proporcao,
        "desvio_padrao_medio": float(desvio_padrao_proporcao.mean()),
        "lof_percent_por_execucao": [r.lof_percent for r in resultados],
        "aviso": (
            "Proxy de sensibilidade a' inicializacao, nao banda de "
            "ambiguidade formal (MCR-BANDS). " + AVISO_AMBIGUIDADE_ROTACIONAL
        ),
    }
