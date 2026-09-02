"""hsi_applicability.py — Dominio de aplicabilidade para o espaco de
pixels do HSI (Passo 108 da `INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md`).

Reaproveita `chemometric_stats.training_applicability_domain`/
`applicability_domain_new_samples` SEM ALTERACAO -- essas funcoes ja'
sao genericas o suficiente (operam em qualquer matriz `(n_amostras,
n_variaveis)`, sem nenhuma suposicao especifica de espectro
vibracional) para aceitar pixels HSI diretamente: 1 pixel = 1 "amostra"
no mesmo sentido que 1 espectro e' 1 amostra no fluxo tabular -- e' a
MESMA granularidade que `hsi_classification.py` (Passo 98) usa para
treinar o PLS-DA por pixel, entao o dominio de aplicabilidade avalia
EXATAMENTE o espaco que o classificador ve.

A UNICA coisa nova aqui e' a checagem de COMPATIBILIDADE DE CAMERA
ANTES de chamar `applicability_domain_new_samples`: cameras diferentes
tem numero de bandas DIFERENTE (Kaki/VIS=224, Kaki/VIS_COR=249,
confirmado por leitura direta -- ver Passo 104), entao
`pca.transform(X_new)` do sklearn simplesmente LEVANTARIA um erro de
shape incompatibilidade em vez de uma decisao "fora do dominio"
interpretavel. Detectado e reportado explicitamente como incompatibi-
lidade de sensor, nunca deixado virar um traceback cru nem uma
comparacao numerica sem sentido entre eixos espectrais diferentes.

Para o caso onde 2 combinacoes REALMENTE compartilham o mesmo sensor
(fisicamente o MESMO modelo de camera -- Specim FX10 para todo VIS
deste dataset, confirmado por leitura das especificacoes de cada
camera no JSON de anotacoes: mesmo `id` "VIS" reaparece com o mesmo
numero de bandas em Kaki/Avocado/Kiwi/Mango/Papaya), a comparacao
numerica E' valida e testa algo real: dados de uma FRUTA diferente da
calibrada devem cair fora do dominio (quimica/reflectancia diferente),
mesmo eixo espectral.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import (applicability_domain_new_samples,
                                        training_applicability_domain)

__all__ = [
    "HSIApplicabilityDomain",
    "train_hsi_applicability_domain",
    "evaluate_hsi_applicability_domain",
]


@dataclass
class HSIApplicabilityDomain:
    pca: object
    n_bandas: int
    artefatos: Dict[str, object]


def train_hsi_applicability_domain(
        X_train: np.ndarray, *, n_components: int = 5, alpha: float = 0.05,
        ) -> HSIApplicabilityDomain:
    """Calibra o dominio de aplicabilidade a partir dos pixels de treino
    (mesma matriz `X` usada por `hsi_classification.
    fit_predict_pixel_plsda`). `n_components` capado por
    `min(n_components, n_amostras-1, n_bandas)`, mesma defesa ja' usada
    em `hsi_identification.train_hsi_identification_ensemble`."""
    X_train = np.asarray(X_train, dtype=float)
    n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    if n_comp < 1:
        raise ValueError(
            f"train_hsi_applicability_domain: dados insuficientes para "
            f"nenhum componente (n_amostras={X_train.shape[0]}, "
            f"n_bandas={X_train.shape[1]}).")
    pca = PCA(n_components=n_comp).fit(X_train)
    artefatos = training_applicability_domain(pca, X_train, alpha=alpha)
    return HSIApplicabilityDomain(
        pca=pca, n_bandas=X_train.shape[1], artefatos=artefatos)


def evaluate_hsi_applicability_domain(
        dominio: HSIApplicabilityDomain, X_new: np.ndarray,
        ) -> Dict[str, object]:
    """Avalia pixels novos contra `dominio`. Se `X_new` tiver um numero
    de bandas DIFERENTE do treino (camera/sensor incompativel), devolve
    `sensor_compativel=False` e `dentro_dominio` todo `False` -- NUNCA
    tenta `pca.transform` com shape incompativel (isso levantaria um
    erro cru do sklearn em vez de uma decisao interpretavel)."""
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim != 2:
        raise ValueError(
            f"evaluate_hsi_applicability_domain espera X_new 2D "
            f"(n_amostras, n_bandas), recebeu shape {X_new.shape}.")

    if X_new.shape[1] != dominio.n_bandas:
        n = X_new.shape[0]
        return {
            "sensor_compativel": False,
            "motivo": (f"sensor incompativel: dominio calibrado com "
                      f"{dominio.n_bandas} bandas, X_new tem "
                      f"{X_new.shape[1]} -- nao comparavel."),
            "dentro_dominio": np.zeros(n, dtype=bool),
            "fracao_dentro": 0.0 if n else float("nan"),
        }

    artefatos = dominio.artefatos
    resultado = applicability_domain_new_samples(
        dominio.pca, X_new,
        var_t=artefatos["var_t"],  # type: ignore[arg-type]
        h0=artefatos["h0"], q0=artefatos["q0"],  # type: ignore[arg-type]
        Nh=artefatos["Nh"], Nq=artefatos["Nq"],  # type: ignore[arg-type]
        f_crit=artefatos["f_crit"])  # type: ignore[arg-type]
    resultado_dict: Dict[str, object] = dict(resultado)
    resultado_dict["sensor_compativel"] = True
    resultado_dict["motivo"] = None
    return resultado_dict
