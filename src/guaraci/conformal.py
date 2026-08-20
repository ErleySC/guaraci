"""
conformal.py — Predicao conformal para autenticacao one-class.

POR QUE ESTE MODULO EXISTE
--------------------------
O DD-SIMCA converte T2/Q numa aproximacao chi-quadrado cujos graus de
liberdade (`Nh`, `Nq`) sao estimados dos dados pelo metodo dos momentos:
`N = 2*(media/desvio)^2`. Isso exige estimar um DESVIO -- e desvio exige
mais de uma observacao independente.

Num dataset de referencia interno havia **1 unica amostra fisica pura
por classe**, medido -- e esse e' o regime que este modulo precisa
tratar sem mentir.
Com uma observacao, `media_e_dof_momentos` nao consegue estimar desvio e
cai num PISO (`N=1.0`) que nao e' um grau de liberdade medido: e' um
default. O `f_crit` resultante vira uma constante
(`chi2.ppf(0.95, 2) = 5.99`) que nao depende de nenhuma propriedade
estimada da distribuicao. O alpha declarado (0,05) e' nominal, nao efetivo.

A predicao conformal ataca isso de frente: a cobertura nao vem de assumir
uma forma de distribuicao, vem de ordenar escores de nao-conformidade num
conjunto de calibracao. E, principalmente, ela torna o limite de n
EXPLICITO em vez de mascara-lo.

O LIMITE DURO, QUE NENHUM METODO REVOGA
---------------------------------------
Com `n` observacoes de calibracao INDEPENDENTES e permutaveis, o menor
nivel de erro que um procedimento conformal pode garantir e' `1/(n+1)`:

    alpha_min = 1/(n+1)   =>   n >= (1-alpha)/alpha

    alpha=0.25 -> n>=3      alpha=0.05 -> n>=19
    alpha=0.10 -> n>=9      alpha=0.01 -> n>=99

Com n=1, `alpha_min = 0.5`. Isto e' aritmetica, nao escolha de metodo:
nao existe procedimento -- conformal, DD-SIMCA, ou qualquer outro -- que
garanta 5% de erro a partir de uma unica amostra independente. A diferenca
e' que aqui a impossibilidade e' RETORNADA (`alcancavel=False`), enquanto o
metodo dos momentos a esconde num piso.

ESCOLHA DO ESCORE DE NAO-CONFORMIDADE
-------------------------------------
Usamos a distancia combinada do DD-SIMCA,
`f = (T2/h0)*Nh + (Q/q0)*Nq` (Eq. 3 de Kucheryavskiy, Rodionova &
Pomerantsev 2024), como escore de nao-conformidade.

Razoes:
  1. E' a MESMA grandeza que o DD-SIMCA ja usa para decidir, entao a
     comparacao entre os dois metodos isola o efeito da REGRA DE DECISAO
     (chi2 assumido vs. quantil empirico), sem confundir com mudanca de
     representacao.
  2. Ela combina distancia no espaco do modelo (T2) e distancia AO modelo
     (Q), que e' o que caracteriza um objeto atipico em SIMCA.
  3. O conformal usa apenas a ORDEM dos escores, entao h0/q0/Nh/Nq
     entram so' como fatores de escala monotonos -- a garantia de
     cobertura NAO depende de eles estarem corretamente estimados. E'
     justamente o ponto: o escore pode ser mal calibrado que a cobertura
     conformal continua valida.

GROUP-AWARE POR CONSTRUCAO
--------------------------
A validade conformal exige PERMUTABILIDADE entre as observacoes de
calibracao. Replicas tecnicas da mesma amostra fisica NAO sao permutaveis
com amostras de outras -- sao quase copias. Calibrar com espectros
individuais infla `n` sem adicionar informacao e quebra a garantia.
Por isso `ConformalOneClass` exige `mae_id` e colapsa cada grupo a UM
escore (a mediana do grupo, robusta a uma replica atipica). O `n` que
entra em `alpha_min` e' o numero de AMOSTRAS FISICAS.

Referencias:
    Vovk V., Gammerman A. & Shafer G. (2005). Algorithmic Learning in a
    Random World. Springer. (conformal prediction; validade por
    permutabilidade)
    Papadopoulos H. et al. (2002). Inductive Confidence Machines for
    Regression. ECML. (split/inductive conformal)
    Kucheryavskiy S., Rodionova O. & Pomerantsev A. (2024). J.
    Chemometrics 38(7):e3556. (distancia combinada usada como escore)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "alpha_alcancavel",
    "n_minimo_para_alpha",
    "limiar_conformal",
    "ConformalOneClass",
]


def alpha_alcancavel(n: int) -> float:
    """Menor nivel de erro garantivel com `n` observacoes de calibracao.

    `alpha_min = 1/(n+1)`. Nenhum procedimento conformal garante um alpha
    menor que este -- e' propriedade da ordenacao de n+1 valores
    permutaveis, nao limitacao de implementacao.
    """
    n = int(n)
    if n < 1:
        return 1.0
    return 1.0 / (n + 1)


def n_minimo_para_alpha(alpha: float) -> int:
    """Menor `n` de calibracao que sustenta o `alpha` pedido: (1-alpha)/alpha."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha deve estar em (0,1), recebido {alpha}")
    return int(math.ceil((1.0 - alpha) / alpha))


def limiar_conformal(scores_calib: np.ndarray, alpha: float = 0.05
                     ) -> Dict[str, Any]:
    """Limiar conformal split/inductive a partir de escores de calibracao.

    Aceita um objeto novo quando `score <= limiar`. O limiar e' o
    k-esimo menor escore de calibracao, com

        k = ceil((n+1) * (1-alpha))

    que e' a correcao de amostra finita que da' a garantia
    `P(score_novo <= limiar) >= 1-alpha` sob permutabilidade.

    Quando `k > n` o limiar seria "+infinito" -- isto e', aceitar tudo --
    porque nao ha' escores suficientes para sustentar o alpha pedido.
    Nesse caso devolve `alcancavel=False` e `limiar=nan` EM VEZ de um
    numero, para que nenhum resultado seja reportado como se tivesse
    garantia que nao tem.

    Returns
    -------
    dict com: limiar (float|nan), alcancavel (bool), n_calibracao (int),
    alpha_nominal (float), alpha_alcancavel (float), k (int), aviso (str|None)
    """
    s = np.asarray(scores_calib, dtype=float)
    s = s[np.isfinite(s)]
    n = int(s.size)
    a_min = alpha_alcancavel(n)
    res: Dict[str, Any] = {
        "limiar": float("nan"),
        "alcancavel": False,
        "n_calibracao": n,
        "alpha_nominal": float(alpha),
        "alpha_alcancavel": a_min,
        "k": 0,
        "aviso": None,
    }
    if n == 0:
        res["aviso"] = "Nenhum escore de calibracao finito."
        return res

    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    res["k"] = k
    if k > n:
        res["aviso"] = (
            f"alpha={alpha:.3g} NAO e' alcancavel com {n} amostra(s) de "
            f"calibracao independentes: exigiria k={k} escores ordenados, "
            f"mas so' ha' {n}. Menor alpha garantivel: {a_min:.3g} "
            f"(precisa de n>={n_minimo_para_alpha(alpha)} para alpha="
            f"{alpha:.3g}). Limiar NAO definido -- reportar como "
            f"nao-estimavel, nunca como 'aceita tudo'.")
        return res

    res["limiar"] = float(np.sort(s)[k - 1])
    res["alcancavel"] = True
    if n < n_minimo_para_alpha(alpha) * 2:
        res["aviso"] = (
            f"Limiar valido, mas com {n} amostras de calibracao a garantia "
            f"e' fragil: qualquer amostra a menos derruba o alpha "
            f"alcancavel para {alpha_alcancavel(n - 1):.3g}.")
    return res


class ConformalOneClass:
    """Detector one-class conformal, group-aware por construcao.

    Recebe escores de nao-conformidade ja calculados (tipicamente a
    distancia combinada do DD-SIMCA -- ver docstring do modulo) e os
    `mae_id` correspondentes, colapsa por amostra fisica e deriva o
    limiar conformal.

    Nao ajusta modelo espectral nenhum: e' uma CAMADA DE DECISAO sobre um
    escore existente. Isso e' deliberado -- mantem a comparacao com o
    DD-SIMCA restrita a regra de decisao.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.info_: Dict[str, Any] = {}
        self.limiar_: float = float("nan")

    @staticmethod
    def _colapsar_por_grupo(scores: np.ndarray,
                            mae_id: np.ndarray) -> np.ndarray:
        """Um escore por amostra fisica (mediana do grupo).

        Mediana, nao media: com 3 replicas, uma leitura atipica desloca a
        media do grupo inteiro e contamina a calibracao; a mediana de 3
        ignora um outlier isolado.
        """
        grupos = np.asarray(mae_id, dtype=str)
        return np.array([float(np.median(scores[grupos == g]))
                         for g in np.unique(grupos)], dtype=float)

    def fit(self, scores_calib: np.ndarray,
            mae_id: Optional[np.ndarray] = None) -> "ConformalOneClass":
        """Calibra o limiar. `mae_id` ausente = cada escore e' tratado como
        amostra independente -- so' correto se NAO houver replicas."""
        s = np.asarray(scores_calib, dtype=float)
        if mae_id is None:
            log.warning(
                "[Conformal] mae_id ausente: cada espectro sera' tratado "
                "como amostra INDEPENDENTE. Se ha' replicas tecnicas, a "
                "garantia de cobertura nao vale (permutabilidade violada) "
                "e o n de calibracao esta' inflado.")
            s_grupo = s
        else:
            s_grupo = self._colapsar_por_grupo(s, mae_id)
        self.info_ = limiar_conformal(s_grupo, self.alpha)
        self.limiar_ = self.info_["limiar"]
        if not self.info_["alcancavel"]:
            log.warning("[Conformal] %s", self.info_["aviso"])
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Aceita (True) quando o escore nao excede o limiar.

        Com limiar nao-estimavel, devolve tudo `False` -- decisao
        conservadora e COERENTE com o aviso: sem garantia, nao ha'
        autenticacao a declarar. Nunca devolver "aceita tudo", que seria
        indistinguivel de um modelo permissivo bem calibrado.
        """
        s = np.asarray(scores, dtype=float)
        if not self.info_.get("alcancavel", False):
            return np.zeros(s.shape, dtype=bool)
        return s <= self.limiar_
