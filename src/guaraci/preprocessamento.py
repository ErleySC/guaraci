"""
preprocessamento.py — Transformers sklearn-compatíveis de pré-processamento
espectral (SNV, Savitzky-Golay, MSC) e o construtor de pipeline de
pré-processamento.

Extraído de pipeline.py como parte da modularização (Fase H). Depende de
Config só para type hint de `build_preprocessor` (import guardado por
TYPE_CHECKING, para não criar import circular com pipeline.py, que importa
este módulo). pipeline.py reexporta estes nomes, então `pipeline.SNV`,
`pipeline.build_preprocessor(...)` etc. continuam funcionando sem
alteração.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from guaraci.pipeline import Config

__all__ = [
    "SNV",
    "SavGol",
    "MSC",
    "PQN",
    "EMSC",
    "OSC",
    "AirPLS",
    "RemocaoRaiosCosmicos",
    "build_preprocessor",
    "descrever_preprocessamento",
    "NORMALIZACOES",
    "interpretar_faixas",
    "formatar_faixas",
    "interpretar_janela",
    "mascara_fora_das_faixas",
    "avaliar_faixas_excluir",
    "normalizar_espectros",
    "preparar_eixo_espectral",
    "aplicar_no_eixo_do_modelo",
    "MODOS_SEM_EIXO_FISICO",
]


class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: per-sample z-score (scatter correction)."""

    def fit(self, X, y=None):
        # sklearn >= 1.9: um Pipeline so' de etapas SEM estado precisa que a
        # ultima etapa se declare ajustada (`n_features_in_`); sem isto,
        # `Pipeline.transform` levanta NotFittedError (m-04 do parecer de
        # 2026-09-25: preset custom sem mean-centering).
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        return (X - mu) / sd


class SavGol(BaseEstimator, TransformerMixin):
    """Savitzky-Golay filter (smoothing or derivative)."""

    def __init__(self, window_length: int = 25, polyorder: int = 2, deriv: int = 1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        # sklearn >= 1.9: um Pipeline so' de etapas SEM estado precisa que a
        # ultima etapa se declare ajustada (`n_features_in_`); sem isto,
        # `Pipeline.transform` levanta NotFittedError (m-04 do parecer de
        # 2026-09-25: preset custom sem mean-centering).
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return savgol_filter(np.asarray(X, dtype=float),
                             window_length=self.window_length,
                             polyorder=self.polyorder,
                             deriv=self.deriv, axis=1)


def _msc_aplicar(X, ref) -> np.ndarray:
    """MSC com referencia `ref` ja ajustada -- FONTE UNICA da formula, usada
    por `MSC.transform` e por `model_export` (que reaplica o modelo sem
    sklearn). Antes eram duas copias identicas, guardadas so' por um teste
    de igualdade numerica (achado da varredura de duplicacao, 2026-09-19)."""
    X = np.asarray(X, dtype=float)
    ref = np.asarray(ref, dtype=float)
    x_mean = float(ref.mean())
    xc = ref - x_mean
    var_x = float(xc @ xc)

    y_mean = X.mean(axis=1)                  # (n,)
    if var_x < 1e-12:
        # Referencia degenerada (variancia ~0) -- ver docstring de MSC: sem
        # regressao possivel, so' centra pela media de cada amostra.
        return X - y_mean[:, None]

    Yc = X - y_mean[:, None]                 # (n, p)
    b = (Yc @ xc) / var_x                     # (n,) -- Cov(ref, X_i)/Var(ref)
    a = y_mean - b * x_mean                   # (n,)

    b_seguro = np.where(np.abs(b) > 1e-12, b, 1.0)
    out = (X - a[:, None]) / b_seguro[:, None]
    b_quase_zero = np.abs(b) <= 1e-12
    if b_quase_zero.any():
        out[b_quase_zero] = (X - a[:, None])[b_quase_zero]
    return out


class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction. Uses mean training spectrum as
    reference; for each sample estimates (a, b) such that X_i ~ a + b * ref and
    returns (X_i - a) / b. Stateful: must remain inside Pipeline+CV.

    VETORIZADO em 2026-08-07 (achado da auditoria metodologica -- ver
    AUDITORIA_METODOLOGICA_2026-08-07.md, secao "Dívida de
    engenharia observada"): a regressao de 2 parametros (a, b) por amostra
    e' uma regressao linear simples (1 preditor + intercepto), que tem
    forma fechada:
        b = Cov(ref, X_i) / Var(ref)
        a = mean(X_i) - b * mean(ref)
    resolvida para TODAS as amostras de uma vez via operacoes matriciais,
    em vez de um `np.linalg.lstsq` por amostra num loop Python (o mesmo
    resultado, so' mais lento -- desperdicio notavel com matrizes
    espectrais grandes
    espectrais reais). Verificado numericamente contra a versao anterior
    em 20 casos aleatorios + casos estruturados (b=0/1/2): diff < 1e-8.

    Unico caso em que o resultado MUDA de proposito: referencia de treino
    com variancia ~0 (espectro medio CONSTANTE em todo o eixo -- nao
    acontece com dado espectral real, exigiria um instrumento sem
    absolutamente nenhum sinal). Nesse caso a regressao e' mal-posta;
    `lstsq` antigo devolvia a solucao de NORMA MINIMA via SVD (um artefato
    numerico, nao uma resposta cientifica definida), a versao atual cai no
    MESMO fallback ja usado por amostra quando b~=0 (so' subtrai a media),
    mais previsivel que o artefato do SVD.
    """

    def fit(self, X, y=None):
        self.ref_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        return _msc_aplicar(X, self.ref_)


class PQN(BaseEstimator, TransformerMixin):
    """Probabilistic Quotient Normalization (Dieterle, Ross, Schlotterbeck &
    Senn, *Anal. Chem.* 78:4281-4290, 2006, DOI 10.1021/ac051632c).

    Proposta T6 da rodada multiagente de 2026-09-10: `cli_assistente.py`
    já recomendava "SNV ou PQN + MC (após binning)" para RMN, mas PQN não
    existia no pacote (achado R5b). Divide cada espectro por um fator de
    diluição estimado como a MEDIANA das razões ponto a ponto entre ele e
    um espectro de referência -- robusto a poucos picos que variam muito
    (ex.: um metabólito que muda de concentração não domina a estimativa
    de diluição, ao contrário de uma normalização por área/soma total).

    Referência = espectro MEDIANO do conjunto de TREINO (fit), o mesmo
    padrão de referência-por-treino de `MSC`/`EMSC` -- ajustada dentro do
    Pipeline+CV para não vazar entre folds.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.ref_ = np.median(X, axis=0)
        return self

    def transform(self, X):
        return _pqn_aplicar(X, self.ref_)


def _pqn_aplicar(X, ref) -> np.ndarray:
    """Divide cada linha de X pela mediana dos quocientes X/ref (compartilhada por `PQN.transform`
    e pela predicao do modelo portatil, para as duas nunca divergirem)."""
    X = np.asarray(X, dtype=float)
    ref = np.asarray(ref, dtype=float)
    # So' usa canais onde a referencia e' numericamente significativa --
    # divisao por ~0 produziria razao instavel/infinita sem informacao
    # real sobre diluicao.
    escala = float(np.max(np.abs(ref))) if ref.size else 0.0
    limiar = escala * 1e-6 if escala > 0 else 1e-12
    mask_valida = np.abs(ref) > limiar
    if not mask_valida.any():
        # Referencia degenerada (treino ~constante em 0) -- nao ha'
        # quociente estimavel; devolve o espectro sem alteracao em vez
        # de propagar NaN/Inf silenciosamente.
        return X
    razoes = X[:, mask_valida] / ref[mask_valida]
    fator = np.median(razoes, axis=1)
    fator_seguro = np.where(np.abs(fator) > 1e-12, fator, 1.0)
    return X / fator_seguro[:, None]


class EMSC(BaseEstimator, TransformerMixin):
    """Extended Multiplicative Signal Correction (Martens & Stark, 1991,
    J. Pharm. Biomed. Anal. 9(8):625-635, DOI 10.1016/0731-7085(91)80188-F
    -- verificado no Crossref em 2026-09-04).

    Generaliza MSC: alem do termo multiplicativo contra o espectro medio
    de referencia (igual ao MSC), ajusta tambem uma linha de base
    POLINOMIAL (ordem `ordem_polinomial`, no EIXO de indice do canal --
    ver nota abaixo) e, opcionalmente, espectros de INTERFERENTES
    conhecidos, tudo numa unica regressao por amostra:

        X_i(canal) ~= a_i + b_i*ref(canal) + sum_k c_ik*canal^k
                      + sum_j d_ij*interferente_j(canal)

    A correcao remove tudo (a_i, os termos polinomiais, os interferentes)
    exceto o termo multiplicativo b_i, igual ao MSC:

        X_corrigido_i = (X_i - a_i - sum_k c_ik*canal^k
                          - sum_j d_ij*interferente_j) / b_i

    NOTA: sem `eixo` explicito, usa o INDICE do canal (0..p-1) normalizado
    -- suficiente para capturar uma tendencia suave de linha de base;
    nao precisa do numero de onda fisico real para isso (a forma
    polinomial de uma linha de base nao muda por reescala/deslocamento
    linear do eixo). `ordem_polinomial=0` sem `interferentes` reduz ao
    MSC (so' com solver por minimos quadrados em vez da forma fechada
    usada em `MSC`, resultado numericamente equivalente).

    Ganho NAO e' garantido por cenario: confirme com
    `portao_correcao_sinal.avaliar_correcao_sinal_pls` (seeds repetidas,
    Wilcoxon pareado) no seu proprio dataset antes de adota-lo como padrao."""

    def __init__(self, eixo=None, ordem_polinomial: int = 2, interferentes=None):
        self.eixo = eixo
        self.ordem_polinomial = ordem_polinomial
        self.interferentes = interferentes

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        self.ref_ = X.mean(axis=0)

        eixo = np.asarray(self.eixo, dtype=float) if self.eixo is not None \
            else np.arange(p, dtype=float)
        desvio = eixo.std()
        eixo_norm = (eixo - eixo.mean()) / (desvio if desvio > 1e-300 else 1.0)

        colunas = [np.ones(p), self.ref_]
        for k in range(1, int(self.ordem_polinomial) + 1):
            colunas.append(eixo_norm ** k)
        if self.interferentes is not None:
            interf = np.atleast_2d(np.asarray(self.interferentes, dtype=float))
            for linha in interf:
                colunas.append(linha)
        self.base_ = np.column_stack(colunas)   # (p, n_termos)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        base = self.base_
        coefs, *_ = np.linalg.lstsq(base, X.T, rcond=None)   # (n_termos, n)
        b = coefs[1]                                          # termo multiplicativo (ref)
        b_seguro = np.where(np.abs(b) > 1e-12, b, 1.0)
        base_aditiva = base.copy()
        base_aditiva[:, 1] = 0.0                              # zera a contribuicao de b
        reconstrucao_aditiva = base_aditiva @ coefs           # (p, n)
        Xc = (X.T - reconstrucao_aditiva) / b_seguro[None, :]
        return Xc.T


class OSC(BaseEstimator, TransformerMixin):
    """Orthogonal Signal Correction (Wold, Antti, Lindgren & Ohman, 1998,
    Chemom. Intell. Lab. Syst. 44(1-2):175-185, DOI
    10.1016/S0169-7439(98)00109-9 -- verificado no Crossref em 2026-09-04).

    Ao contrario de SNV/MSC/SG (nao-supervisionados, so' olham X), OSC usa
    `y` para remover de X so' a variacao ORTOGONAL ao alvo -- por isso
    `fit` exige `y` (nao e' opcional aqui). Dentro de um `Pipeline`
    (`cross_val_predict`/`cross_val_score`), o sklearn ja' passa `y` de
    treino para o `fit` de cada etapa automaticamente -- sem risco de
    vazamento adicional ao ja' existente no restante do pipeline.

    NIPALS por componente: parte do 1o PC de X (deflacionado pelos
    componentes OSC anteriores), ortogonaliza iterativamente o escore `t`
    em relacao a `y` (projeta fora a parte de `t` correlacionada com y),
    recalcula o loading `w` contra esse `t` ortogonal, ate' convergir;
    deflaciona X por esse componente e repete para `n_componentes`.

    Ganho NAO e' garantido por cenario (OSC pode piorar o RMSEP quando a
    variacao removida carrega informacao sobre y) -- confirme com `avaliar_correcao_sinal_pls` no seu proprio
    dataset antes de usar OSC como pre-processamento padrao; nao use so'
    porque ajudou em outro conjunto de dados."""

    def __init__(self, n_componentes: int = 1, max_iter: int = 100, tol: float = 1e-8):
        self.n_componentes = n_componentes
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.mean_ = X.mean(axis=0)
        Xr = X - self.mean_
        Yc = Y - Y.mean(axis=0)

        n_comp = int(max(1, min(self.n_componentes, X.shape[1], X.shape[0] - 1)))
        YtY_pinv = np.linalg.pinv(Yc.T @ Yc)

        ws, ps = [], []
        for _ in range(n_comp):
            u, s, _vt = np.linalg.svd(Xr, full_matrices=False)
            t = u[:, 0] * s[0]
            t_ant = None
            for _it in range(self.max_iter):
                t_orth = t - Yc @ (YtY_pinv @ (Yc.T @ t))
                w = Xr.T @ t_orth
                norma_w = np.linalg.norm(w)
                w = w / norma_w if norma_w > 1e-300 else w
                t_novo = Xr @ w
                t_novo_orth = t_novo - Yc @ (YtY_pinv @ (Yc.T @ t_novo))
                if t_ant is not None and np.linalg.norm(t_novo_orth - t_ant) < self.tol:
                    t = t_novo_orth
                    break
                t_ant, t = t_novo_orth, t_novo_orth
            tt = float(t @ t)
            p_load = (Xr.T @ t / tt) if tt > 1e-300 else np.zeros(X.shape[1])
            Xr = Xr - np.outer(t, p_load)
            ws.append(w)
            ps.append(p_load)

        self.w_ = np.column_stack(ws)   # (p, n_comp)
        self.p_ = np.column_stack(ps)   # (p, n_comp)
        return self

    def transform(self, X):
        Xc = np.asarray(X, dtype=float) - self.mean_
        for k in range(self.w_.shape[1]):
            t = Xc @ self.w_[:, k]
            Xc = Xc - np.outer(t, self.p_[:, k])
        return Xc


def _whittaker_smooth(x: np.ndarray, w: np.ndarray, lam: float, porder: int = 1) -> np.ndarray:
    """Suavizador de Whittaker ponderado: ajusta `z` que minimiza
    `sum(w*(x-z)^2) + lam*sum(diff(z, porder)^2)` -- baseline penalizado
    por curvatura, resolvido em forma fechada via sistema esparso
    (banda), ponto central do algoritmo airPLS (ver docstring de
    `AirPLS`)."""
    m = x.shape[0]
    E = sparse.eye(m, format="csc")
    for _ in range(porder):
        E = E[1:] - E[:-1]
    W = sparse.diags(w, 0, shape=(m, m))
    A = sparse.csc_matrix(W + lam * (E.T @ E))
    B = W @ x
    return np.asarray(spsolve(A, B))


class AirPLS(BaseEstimator, TransformerMixin):
    """Correcao de linha de base por minimos quadrados penalizados
    adaptativos e iterativamente reponderados (Zhang, Chen & Liang,
    2010, *Analyst* 135(5):1138-1146, DOI 10.1039/b922045c --
    verificado no Crossref em 2026-09-04). Passo 144/145 da auditoria
    das 11 tecnicas (2026-09-04) -- proposta especificamente para
    remover fluorescencia de fundo de espectros Raman, mas nao
    restrita a essa tecnica (opera em qualquer eixo de indice de
    canal).

    Ajusta iterativamente uma linha de base suave `z` (Whittaker
    smoother, penalidade `lam` sobre a curvatura de ordem `porder`) que
    so' segue os pontos ABAIXO do espectro (ruido/baseline de
    verdade), nunca os PICOS (acima da linha de base): a cada
    iteracao, pontos onde o residuo `x - z` e' positivo (picos) recebem
    peso zero na proxima suavizacao; pontos negativos (baseline)
    recebem peso crescente com a magnitude do residuo. Converge quando
    a soma dos residuos negativos fica pequena o suficiente (ou apos
    `itermax` iteracoes) -- nao precisa de deteccao de pico manual nem
    de informacao previa sobre onde estao os picos.

    `lam=100` (default) segue o valor de referencia do artigo original;
    `lam` grande demais deixa a linha de base rigida (nao acompanha a
    curvatura real do fundo, sub-corrige); pequeno demais deixa a linha
    de base seguir os proprios picos (sobre-corrige). Contra-prova
    sintetica em `tests/test_airpls.py`.

    NAO EM `build_preprocessor`/nenhum preset por default: correcao de
    sinal nova so' vira recomendacao padrao depois de passar pelo
    portao de aceite (`portao_correcao_sinal.avaliar_correcao_sinal_pls`);
    `lam` padrao e' ponto de partida, ajustavel.
    Disponivel via `cfg.apply_airpls`/campo `custom` do
    `build_preprocessor`, mesmo padrao de EMSC/OSC."""

    def __init__(self, lam: float = 100.0, porder: int = 1,
                 itermax: int = 15, tol: float = 1e-3):
        self.lam = lam
        self.porder = porder
        self.itermax = itermax
        self.tol = tol

    def fit(self, X, y=None):
        # sklearn >= 1.9: um Pipeline so' de etapas SEM estado precisa que a
        # ultima etapa se declare ajustada (`n_features_in_`); sem isto,
        # `Pipeline.transform` levanta NotFittedError (m-04 do parecer de
        # 2026-09-25: preset custom sem mean-centering).
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def _baseline_uma_amostra(self, x: np.ndarray) -> np.ndarray:
        m = x.shape[0]
        w = np.ones(m)
        z = x.copy()
        soma_abs_x = float(np.abs(x).sum())
        for i in range(1, int(self.itermax) + 1):
            z = _whittaker_smooth(x, w, float(self.lam), int(self.porder))
            d = x - z
            negativos = d[d < 0]
            dssn = float(np.abs(negativos.sum())) if negativos.size else 0.0
            if soma_abs_x > 1e-300 and dssn < self.tol * soma_abs_x:
                break
            if i == self.itermax:
                break
            w[d >= 0] = 0.0
            if negativos.size and dssn > 1e-300:
                w[d < 0] = np.exp(i * np.abs(negativos) / dssn)
            # Peso de borda: usa negativos.max() (residuo NEGATIVO mais
            # proximo de zero, nao a magnitude maxima) -- formula do
            # algoritmo original (Zhang, Chen & Liang 2010): usar
            # abs(negativos).max() aqui (magnitude maxima) faz o peso da
            # borda explodir exponencialmente nas ultimas iteracoes
            # (dssn encolhe a cada volta), dominando o ajuste e
            # degradando a linha de base -- bug medido e corrigido
            # durante a contra-prova sintetica deste modulo (Passo 145).
            pico_extremo = np.exp(i * negativos.max() / dssn) if (
                negativos.size and dssn > 1e-300) else 1.0
            w[0] = pico_extremo
            w[-1] = pico_extremo
        return z

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        baselines = np.vstack([self._baseline_uma_amostra(x) for x in X])
        return X - baselines


def _z_modificado(v: np.ndarray) -> np.ndarray:
    """z-score modificado de Iglewicz & Hoaglin, `0,6745 (v - mediana) / MAD`,
    como em Whitaker & Hayes (2018). MAD nula (sinal sem ruido) -> z = 0 onde
    v e' a mediana e infinito onde nao e'."""
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    dev = v - med
    if mad <= 0.0:
        return np.where(np.abs(dev) > 0.0, np.inf, 0.0)
    return 0.6745 * dev / mad


class RemocaoRaiosCosmicos(BaseEstimator, TransformerMixin):
    """Remocao de raios cosmicos (spikes) de espectros Raman pelo z-score
    modificado da PRIMEIRA DIFERENCA do espectro (Whitaker & Hayes, 2018,
    *Chemometr. Intell. Lab. Syst.* 179:82-84, DOI
    10.1016/j.chemolab.2018.06.009 -- verificado no Crossref em
    2026-10-16).

    Um raio cosmico e' um salto de 1-3 pontos muito mais abrupto que
    qualquer banda Raman: na diferenca `x[i+1] - x[i]` ele aparece como
    um par de valores enormes (subida e descida), enquanto bandas e
    ruido ficam perto da mediana. Com o z-score modificado
    (`0,6745 (d - mediana) / MAD`, robusto ao proprio spike), cada
    diferenca com `|z| > limiar` e' marcada. Os pontos entre uma subida
    e a descida marcada seguinte (ate' `largura_max` pontos) sao o spike;
    cada um e' trocado pela MEDIA dos vizinhos NAO marcados numa janela
    de `+-janela` pontos, como no artigo (sem vizinho limpo na janela:
    interpolacao linear entre os pontos limpos mais proximos).

    Diferenca deliberada do codigo de exemplo do artigo: la' o indice da
    diferenca marcada e' usado direto como indice do ponto, o que troca o
    ponto ANTERIOR ao salto e deixa o meio de um spike de 3 pontos
    intacto; aqui a marcacao vai para os pontos de dentro do salto.

    `limiar=3,5` segue a convencao de Iglewicz & Hoaglin para o z-score
    modificado, usada no artigo; em espectros com bandas muito finas
    (poucos pontos por banda), suba o limiar se bandas reais forem
    marcadas. Opera espectro a espectro e nao aprende nada no `fit`:
    nao ha vazamento entre dobras da validacao cruzada.

    NAO entra em nenhum preset por default (`cfg.apply_despike=False`):
    quando ligado, vira a PRIMEIRA etapa de qualquer pre-processamento
    (o spike tem de sair antes de SNV/derivada o espalharem)."""

    def __init__(self, limiar: float = 6.0, janela: int = 5, largura_max: int = 5,
                 razao_isolamento: float = 3.0):
        self.limiar = limiar
        self.janela = janela
        self.largura_max = largura_max
        self.razao_isolamento = razao_isolamento

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def mascara(self, x: np.ndarray) -> np.ndarray:
        """Pontos de `x` (1-D) marcados como raio cosmico."""
        x = np.asarray(x, dtype=float)
        m = x.shape[0]
        spike = np.zeros(m, dtype=bool)
        if m < 3:
            return spike
        d = np.diff(x)
        z = _z_modificado(d)
        marcadas = np.flatnonzero(np.abs(z) > float(self.limiar))
        larg = max(1, int(self.largura_max))
        razao = float(self.razao_isolamento)

        def _isolado(i: int, lado: int) -> bool:
            # o salto tem de ser `razao` vezes maior que as diferencas logo
            # FORA dele (flanco de banda sobe/desce em varios passos
            # parecidos; raio cosmico, num passo so')
            fora = [abs(d[j]) for j in (i + lado, i + 2 * lado) if 0 <= j < d.size]
            return not fora or abs(d[i]) > razao * max(fora)

        k = 0
        while k < marcadas.size:
            i = int(marcadas[k])
            fim = None
            for k2 in range(k + 1, marcadas.size):
                j = int(marcadas[k2])
                if j - i > larg:
                    break
                if np.sign(z[j]) != np.sign(z[i]):
                    fim = k2
                    break
            if fim is None:
                k += 1
                continue
            j = int(marcadas[fim])
            if razao <= 0 or (_isolado(i, -1) and _isolado(j, +1)):
                spike[i + 1:j + 1] = True
                k = fim + 1
            else:
                k += 1
        return spike

    def _corrigir(self, x: np.ndarray) -> np.ndarray:
        spike = self.mascara(x)
        if not spike.any():
            return x
        out = x.copy()
        limpos = np.flatnonzero(~spike)
        if limpos.size == 0:
            return out
        w = max(1, int(self.janela))
        m = x.shape[0]
        for i in np.flatnonzero(spike):
            viz = np.arange(max(0, i - w), min(m, i + w + 1))
            viz = viz[~spike[viz]]
            out[i] = (float(x[viz].mean()) if viz.size
                      else float(np.interp(i, limpos, x[limpos])))
        return out

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.vstack([self._corrigir(x) for x in X]) if X.size else X.copy()


# ---------------------------------------------------------------------------
# Faixas excluidas do eixo (chave `faixas_excluir`) e janela de referencia
# (`ims_rip_janela`). Funcoes puras, sem estado: o MESMO codigo roda no treino
# (pipeline), na predicao do .joblib e na do JSON portatil.
# ---------------------------------------------------------------------------
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_RE_FAIXA = re.compile(rf"^\s*({_NUM})\s*-\s*({_NUM})\s*$")

#: valores aceitos em `normalizacao` (opt-in; "nenhuma" = comportamento anterior).
NORMALIZACOES = ("nenhuma", "rip", "area")


def _faixa_de_texto(item) -> Tuple[float, float]:
    texto = str(item).strip().replace(",", ".")
    m = _RE_FAIXA.match(texto)
    if m is None:
        raise ValueError(
            f"faixa {str(item)!r} invalida: use \"a-b\" na unidade do eixo, com ponto decimal "
            f"(ex.: 4.5-5.0 ou -0.5-0.5)")
    a, b = float(m.group(1)), float(m.group(2))
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f"faixa {str(item)!r} invalida: valores nao finitos")
    if not a < b:
        raise ValueError(
            f"faixa {str(item)!r} invalida: o inicio ({a:g}) deve ser MENOR que o fim ({b:g}); "
            f"escreva {b:g}-{a:g} se queria o contrario")
    return a, b


def _fundir(pares) -> List[Tuple[float, float]]:
    fundidas: List[Tuple[float, float]] = []
    for a, b in sorted(pares):
        if fundidas and a <= fundidas[-1][1]:
            fundidas[-1] = (fundidas[-1][0], max(fundidas[-1][1], b))
        else:
            fundidas.append((a, b))
    return fundidas


def interpretar_faixas(itens) -> List[Tuple[float, float]]:
    """Lista de textos "a-b" (ou texto unico separado por `;` ou `,`) -> [(a, b)], ordenada e SEM
    sobreposicao (faixas que se tocam sao fundidas). Levanta ValueError com a faixa culpada."""
    if itens is None:
        return []
    if isinstance(itens, str):
        itens = [x for x in (itens.split(";") if ";" in itens else itens.split(",")) if x.strip()]
    return _fundir(_faixa_de_texto(x) for x in itens if str(x).strip())


def formatar_faixas(faixas) -> Tuple[str, ...]:
    """[(a, b)] -> ("a-b", ...) na forma canonica (a que o config.yaml grava)."""
    return tuple(f"{a:g}-{b:g}" for a, b in faixas)


def _como_faixas(faixas) -> List[Tuple[float, float]]:
    """Aceita textos "a-b" ou pares (a, b) ja' numericos."""
    if faixas is None:
        return []
    if not isinstance(faixas, str) and len(faixas) and all(
            isinstance(f, (tuple, list)) and len(f) == 2 for f in faixas):
        return _fundir((float(a), float(b)) for a, b in faixas)
    return interpretar_faixas(faixas)


def interpretar_janela(valor) -> Optional[Tuple[float, float]]:
    """Texto "a-b" (uma unica janela) -> (a, b), ou None se vazio."""
    if valor is None or not str(valor).strip():
        return None
    return _faixa_de_texto(valor)


def mascara_fora_das_faixas(eixo, faixas) -> np.ndarray:
    """True nas posicoes do eixo que FICAM (fora de todas as faixas, extremos incluidos)."""
    eixo = np.asarray(eixo, dtype=float)
    fica = np.ones(eixo.shape, dtype=bool)
    for a, b in _como_faixas(faixas):
        fica &= ~((eixo >= a) & (eixo <= b))
    return fica


def avaliar_faixas_excluir(eixo_dados, eixo_apos_faixa, faixas) -> dict:
    """Confere as faixas excluidas contra o eixo e devolve o que o pipeline precisa dizer.

    `eixo_dados`: eixo lido do arquivo; `eixo_apos_faixa`: o que sobra de faixa_min_cm/faixa_max_cm.
    Levanta ValueError se (1) alguma faixa nao cruza o eixo dos dados (provavel unidade errada) ou
    (2) sobram menos de 2 variaveis. Devolve {"fica": mascara sobre eixo_apos_faixa, "n_removidas",
    "fracao", "aviso_metade": bool, "fora_da_faixa": [faixas que ja' estavam fora do corte]}."""
    eixo_dados = np.asarray(eixo_dados, dtype=float)
    eixo = np.asarray(eixo_apos_faixa, dtype=float)
    fx = _como_faixas(faixas)
    lo, hi = float(np.nanmin(eixo_dados)), float(np.nanmax(eixo_dados))
    fora, fora_do_corte = [], []
    for a, b in fx:
        if b < lo or a > hi:
            fora.append(f"{a:g}-{b:g}")
        elif not ((eixo >= a) & (eixo <= b)).any():
            fora_do_corte.append(f"{a:g}-{b:g}")
    if fora:
        raise ValueError(
            f"faixas_excluir {fora} nao cruza(m) o eixo dos dados ({lo:g} a {hi:g}); confira a unidade do "
            f"eixo (ex.: ppm) ou apague a chave (faixas_excluir: [])")
    fica = mascara_fora_das_faixas(eixo, fx)
    n_rem = int((~fica).sum())
    if eixo.size - n_rem < 2:
        raise ValueError(
            f"faixas_excluir {list(formatar_faixas(fx))} removem todas as variaveis que sobraram da faixa "
            f"{float(eixo.min()):g} a {float(eixo.max()):g}")
    frac = n_rem / max(eixo.size, 1)
    return {"fica": fica, "n_removidas": n_rem, "fracao": frac, "aviso_metade": frac > 0.5,
            "fora_da_faixa": fora_do_corte}


def normalizar_espectros(X, eixo, modo: str, janela=None) -> np.ndarray:
    """Normalizacao por espectro (sem estado, sem vazamento entre dobras).

    modo "nenhuma": devolve X. "rip": divide cada espectro pela intensidade MAXIMA dentro de
    `janela` ("a-b" no eixo; o pico do ion reagente no IMS, ou o padrao interno no RMN). "area": divide
    pela area (trapezio) do espectro sobre o eixo informado. Levanta ValueError se a janela nao
    cruza o eixo, ou se algum espectro tem referencia <= 0 (a divisao inverteria o sinal ou explodiria)."""
    X = np.asarray(X, dtype=float)
    modo = str(modo or "nenhuma").lower()
    if modo == "nenhuma":
        return X
    eixo = np.asarray(eixo, dtype=float)
    if modo == "rip":
        jan = interpretar_janela(janela)
        if jan is None:
            raise ValueError("normalizacao 'rip' exige ims_rip_janela (ex.: 7-9, na unidade do eixo)")
        cols = (eixo >= jan[0]) & (eixo <= jan[1])
        if not cols.any():
            raise ValueError(
                f"ims_rip_janela {jan[0]:g}-{jan[1]:g} nao cruza o eixo do modelo ({eixo.min():g} a "
                f"{eixo.max():g}); a janela do RIP precisa estar DENTRO da faixa usada "
                f"(faixa_min_cm/faixa_max_cm)")
        ref = X[:, cols].max(axis=1)
        rotulo = "intensidade do RIP"
    elif modo == "area":
        # np.trapz foi renomeado em NumPy 2.0; a soma explicita evita depender da versao.
        ordem = np.argsort(eixo)
        e, Xo = eixo[ordem], X[:, ordem]
        ref = np.sum(0.5 * (Xo[:, 1:] + Xo[:, :-1]) * np.diff(e), axis=1)
        rotulo = "area"
    else:
        raise ValueError(f"normalizacao {modo!r} invalida; use {list(NORMALIZACOES)}")
    ruim = ~(np.isfinite(ref) & (ref > 0))
    if ruim.any():
        raise ValueError(
            f"normalizacao '{modo}': {int(ruim.sum())} espectro(s) com {rotulo} <= 0 ou invalida "
            f"(indices {np.flatnonzero(ruim)[:5].tolist()}); a divisao nao teria significado. "
            f"Confira a janela / a linha de base, ou use normalizacao: nenhuma")
    return X / ref[:, None]


#: modos de entrada cujo "eixo" nao e' um eixo fisico (indice de elemento, de feature de cor...):
#: faixas excluidas e normalizacao por janela nao se aplicam.
MODOS_SEM_EIXO_FISICO = ("multielementar", "imagem", "hsi", "eem")


def preparar_eixo_espectral(cfg, X, eixo_dados, eixo):
    """Etapa 1a1 do pipeline (depois do corte wn_min/wn_max, ANTES do pre-processamento): remove as
    `faixas_excluir` e aplica a `normalizacao` por espectro. Devolve (X, eixo, info), com `info` para
    o log/resumo/pacote. A mesma ordem e' reaplicada na predicao (`aplicar_no_eixo_do_modelo`)."""
    X = np.asarray(X, dtype=float)
    eixo = np.asarray(eixo, dtype=float)
    faixas = interpretar_faixas(getattr(cfg, "exclude_ranges", ()))
    norm = str(getattr(cfg, "normalization", "nenhuma") or "nenhuma").lower()
    janela = getattr(cfg, "rip_window", None)
    info = {"faixas": list(formatar_faixas(faixas)), "n_removidas": 0, "fracao": 0.0, "avisos": [],
            "normalizacao": norm, "janela_rip": janela if norm == "rip" else None}
    if str(getattr(cfg, "mode", "")) in MODOS_SEM_EIXO_FISICO:
        if faixas or norm != "nenhuma":
            info["avisos"].append(
                f"faixas_excluir/normalizacao ignoradas no modo {cfg.mode!r}: as colunas nao sao um eixo "
                f"fisico (ppm, ms, cm-1...)")
        info["faixas"], info["normalizacao"], info["janela_rip"] = [], "nenhuma", None
        return X, eixo, info
    if faixas:
        av = avaliar_faixas_excluir(eixo_dados, eixo, faixas)
        X, eixo = X[:, av["fica"]], eixo[av["fica"]]
        info["n_removidas"], info["fracao"] = av["n_removidas"], av["fracao"]
        if av["aviso_metade"]:
            info["avisos"].append(
                f"faixas_excluir remove {av['fracao'] * 100:.0f}% das variaveis da faixa usada "
                f"(> 50%): confira se as faixas estao na unidade certa do eixo")
        if av["fora_da_faixa"]:
            info["avisos"].append(
                f"faixas_excluir {av['fora_da_faixa']} ja' estao fora de faixa_min_cm/faixa_max_cm "
                f"(sem efeito)")
    if norm != "nenhuma":
        X = normalizar_espectros(X, eixo, norm, janela)
    return X, eixo, info


def aplicar_no_eixo_do_modelo(X_interp, eixo_modelo, faixas, norm, janela):
    """Predicao: X ja' interpolado em `eixo_modelo` (o eixo de TREINO, que ja' nao tem as faixas
    excluidas). Reaplica a exclusao (idempotente) e a normalizacao como no treino. Devolve (X, eixo)."""
    eixo_modelo = np.asarray(eixo_modelo, dtype=float)
    fx = _como_faixas(faixas)
    if fx:
        fica = mascara_fora_das_faixas(eixo_modelo, fx)
        X_interp, eixo_modelo = np.asarray(X_interp, dtype=float)[:, fica], eixo_modelo[fica]
    return normalizar_espectros(X_interp, eixo_modelo, norm, janela), eixo_modelo


def build_preprocessor(cfg: "Config") -> Pipeline:
    """Pre-processamento de `cfg` (ver `_build_preprocessor_preset`), com a
    remocao de raios cosmicos (`cfg.apply_despike`, desligada por padrao)
    como PRIMEIRA etapa de qualquer preset: o spike tem de sair antes de
    SNV/MSC/derivada o espalharem pelo espectro. E' por espectro e sem
    estado, entao nao vaza entre dobras."""
    pipe = _build_preprocessor_preset(cfg)
    if str(getattr(cfg, "alinhamento", "nenhum") or "nenhum").lower() == "cow":
        # Alinhamento de retencao: a referencia e' escolhida em `fit`, ou seja, so' com o
        # treino de cada dobra (nao vaza), e o modelo exportado a leva para amostras novas.
        from guaraci.alinhamento_retencao import AlinhadorCOW
        pipe = Pipeline([("cow", AlinhadorCOW(
            n_segmentos=int(getattr(cfg, "alinhamento_segmentos", 30)),
            folga=int(getattr(cfg, "alinhamento_folga", 10))))] + list(pipe.steps))
    if getattr(cfg, "apply_despike", False):
        despike = RemocaoRaiosCosmicos(limiar=float(getattr(cfg, "despike_limiar", 6.0)),
                                       janela=int(getattr(cfg, "despike_janela", 5)))
        return Pipeline([("despike", despike)] + list(pipe.steps))
    return pipe


def _build_preprocessor_preset(cfg: "Config") -> Pipeline:
    """Builds preprocessor according to cfg.default_preprocessing.

    Presets:
        'snv_sg_mc'   : SNV -> SG -> mean-centering (Rinnan et al. 2009,
                        recommended for FTIR/NIR with scatter)
        'autoscaling' : StandardScaler (mean + unit variance)
                        — recommended when SG derivative destroys signal
                        or for NIR without pronounced scatter
        'mc'          : mean-centering only
        'snv_mc'      : SNV -> mean-centering (sem derivada; UV-Vis, fluorescencia, RMN)
        'pqn_mc'      : PQN (Dieterle et al. 2006) -> mean-centering (RMN/IMS com diluicao
                        variavel; exclua agua e padrao interno antes -- `faixas_excluir`)
        'airpls_sg_mc': AirPLS (Zhang, Chen & Liang 2010) -> SG ->
                        mean-centering — recomendado p/ Raman apos o
                        portao de aceite (Bloco 20), `lam` padrao
                        ajustavel; NAO e'
                        SNV/MSC (baseline Raman e' aditiva, nao
                        multiplicativa como espalhamento NIR).
        'airpls_snv_sg_mc': AirPLS -> SNV -> SG -> mean-centering -- Raman de
                        quantificacao: alem da fluorescencia, normaliza a
                        escala do sinal (potencia, foco, densidade).
        'custom'      : honors apply_snv / apply_sg / apply_mc / apply_airpls

    Mean-centering / autoscaling are kept INSIDE the Pipeline so that
    cross_val_predict does not leak statistics between folds.
    """
    preset = (cfg.default_preprocessing or "custom").lower()

    if preset == "autoscaling":
        return Pipeline([("auto", StandardScaler(with_mean=True, with_std=True))])
    if preset == "mc":
        return Pipeline([("mc", StandardScaler(with_std=False))])
    if preset == "snv_mc":
        return Pipeline([("snv", SNV()), ("mc", StandardScaler(with_std=False))])
    if preset == "pqn_mc":
        return Pipeline([("pqn", PQN()), ("mc", StandardScaler(with_std=False))])
    if preset == "snv_sg_mc":
        return Pipeline([
            ("snv", SNV()),
            ("sg",  SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)),
            ("mc",  StandardScaler(with_std=False)),
        ])
    if preset == "airpls_sg_mc":
        return Pipeline([
            ("airpls", AirPLS(lam=cfg.airpls_lam)),
            ("sg",  SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)),
            ("mc",  StandardScaler(with_std=False)),
        ])
    if preset == "airpls_snv_sg_mc":
        # Raman quantitativo: AirPLS tira a fluorescencia de fundo (aditiva), SNV normaliza a escala
        # (potencia do laser, foco, densidade da amostra) e so' entao SG e mean-centering. O SNV vem
        # DEPOIS do AirPLS: normalizar antes dividiria a fluorescencia pelo desvio e deformaria a linha de base.
        return Pipeline([
            ("airpls", AirPLS(lam=cfg.airpls_lam)),
            ("snv", SNV()),
            ("sg",  SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)),
            ("mc",  StandardScaler(with_std=False)),
        ])
    if preset == "msc_sg_mc":
        # MSC->SG+MC: default preset for diffuse FT-NIR with strong scatter.
        # MSC is stateful (reference = training mean) -> kept inside
        # Pipeline to avoid leakage between CV folds.
        return Pipeline([
            ("msc", MSC()),
            ("sg",  SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)),
            ("mc",  StandardScaler(with_std=False)),
        ])
    # custom — uses individual flags
    etapas: List[Tuple[str, BaseEstimator]] = []
    if cfg.apply_airpls:
        etapas.append(("airpls", AirPLS(lam=cfg.airpls_lam)))
    if cfg.apply_snv:
        etapas.append(("snv", SNV()))
    if cfg.apply_pqn:
        etapas.append(("pqn", PQN()))
    if cfg.apply_emsc:
        etapas.append(("emsc", EMSC(ordem_polinomial=cfg.emsc_ordem_polinomial)))
    if cfg.apply_sg:
        etapas.append(("sg", SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)))
    if cfg.apply_mc:
        etapas.append(("mc", StandardScaler(with_std=False)))
    if cfg.apply_osc:
        # OSC precisa de y (Y_bin one-hot) -- sklearn.Pipeline.fit(X, y)
        # ja passa y para toda etapa que aceite, entao entra por ultimo
        # sem nenhuma alteracao de assinatura em quem chama build_preprocessor.
        etapas.append(("osc", OSC(n_componentes=cfg.osc_n_componentes)))
    if not etapas:
        etapas.append(("mc", StandardScaler(with_std=False)))
    return Pipeline(etapas)


def descrever_preprocessamento(cfg: "Config") -> Tuple[List[str], str, bool]:
    """(rotulos curtos, descricao, usa Savitzky-Golay) do pre-processamento ATIVO.

    Fonte unica para o nome da pasta da execucao, o resumo do modelo e o aviso de
    janela do SG: as tres leem o mesmo `build_preprocessor`, entao nao divergem
    (antes, o preset AirPLS e o Personalizado saiam rotulados como SNV-SG-MC)."""
    sg = f"SG(w={cfg.sg_window},p={cfg.sg_polyorder},d={cfg.sg_deriv})"
    curto = {"despike": "Despike", "airpls": "AirPLS", "snv": "SNV", "msc": "MSC", "pqn": "PQN", "emsc": "EMSC",
             "sg": f"SG{cfg.sg_deriv}", "mc": "MC", "auto": "AUTO", "osc": "OSC", "cow": "COW"}
    longo = {"despike": f"Despike(Whitaker-Hayes, z>{getattr(cfg, 'despike_limiar', 6.0):g})",
             "airpls": "AirPLS", "snv": "SNV", "msc": "MSC", "pqn": "PQN",
             "emsc": f"EMSC(ordem={cfg.emsc_ordem_polinomial})", "sg": sg, "mc": "mean-centering",
             "auto": "Autoscaling (mean + unit variance)", "osc": f"OSC(n={cfg.osc_n_componentes})",
             "cow": f"COW(segmentos={getattr(cfg, 'alinhamento_segmentos', 30)}, "
                    f"folga={getattr(cfg, 'alinhamento_folga', 10)})"}
    passos = [n for n, _ in build_preprocessor(cfg).steps]
    descr = "Mean-centering" if passos == ["mc"] else " -> ".join(longo[p] for p in passos)
    return [curto[p] for p in passos] or ["raw"], descr, "sg" in passos
