"""hsi_identification.py — Conjunto aberto para objetos HSI (Passo 106
da `INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md`).

Adapta o padrao ja' validado de `identificacao.py` (PCA + distancia T2-
like ao centroide + `ConformalOneClass`, "desconhecido" quando a
cobertura nao e' validada) para o nivel de OBJETO do HSI, com uma
diferenca estrutural real: em `identificacao.py` todas as combinacoes
especie x adulterante compartilham o MESMO PCA (1 instrumento, 1 eixo
espectral). Aqui cada combinacao fruta x camera tem seu PROPRIO numero
de bandas (Kaki/VIS=224, Kaki/VIS_COR=249, ...) -- nao da pra' projetar
tudo no mesmo espaco PCA. Por isso o ensemble guarda um PCA + var_t
POR combinacao, nao um so' compartilhado.

Granularidade de calibracao -- MEDIDA antes de decidir (Passo 106 exige
isso explicitamente), nao presumida: contagem real de objetos fisicos
distintos por fruta (28 a 88) e por fruta x camera (24 a 87), as duas
>= `conformal.n_minimum_for_alpha(0.05)` (=19) em TODAS as combinacoes
do dataset -- as duas granularidades SAO calibraveis. Escolhida a mais
FINA (fruta x camera) porque tambem passa no minimo E evita misturar a
variancia espectral de cameras diferentes numa mesma calibracao (o
espectro medio de Kaki/VIS e Kaki/VIS_COR nao sao comparaveis mesmo
depois de qualquer normalizacao simples -- sao sensores diferentes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

from guaraci.conformal import ConformalOneClass, achievable_alpha
from guaraci.hsi_pixels import extract_roi_spectra
from guaraci.identificacao import CoverageStatus

__all__ = [
    "HSIIdentificationResult",
    "aggregate_object_spectrum",
    "train_hsi_identification_ensemble",
    "identify_hsi_object",
]


def aggregate_object_spectrum(cubo: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Espectro representativo de UM objeto: media dos pixels da ROI.
    Distinto da classificacao por pixel do Passo 98 (que agrega ROTULOS
    PREDITOS por voto majoritario) -- aqui agregamos o ESPECTRO BRUTO,
    antes de qualquer predicao, para o proposito de identificacao de
    conjunto aberto (mesmo nivel de "1 espectro representa 1 amostra"
    que `identificacao.py` usa)."""
    pixels = extract_roi_spectra(cubo, mascara)
    return pixels.mean(axis=0)


@dataclass
class HSIIdentificationResult:
    objeto_identificado: Optional[str]           # "fruta|camera" ou None
    candidatos_ambiguos: List[str] = field(default_factory=list)
    cobertura_status: Optional[CoverageStatus] = None
    alpha_alcancavel: Optional[float] = None
    escores: Dict[str, float] = field(default_factory=dict)


def train_hsi_identification_ensemble(
        cubos: Sequence[np.ndarray], mascaras: Sequence[np.ndarray],
        group_ids: Sequence[str], frutas: Sequence[str],
        cameras: Sequence[str], *, alpha_nominal: float = 0.05,
        ) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Calibra um `ConformalOneClass` POR combinacao fruta x camera
    presente nos dados de treino -- 1 PCA proprio por combinacao (ver
    docstring do modulo). O "grupo" de calibracao e' o OBJETO FISICO
    (`group_id`) -- e' a unidade independente real deste dataset, nao
    ha' nocao adicional de "sessao" acima disso (diferente do
    `session_from_mae_id` da tabela tabular)."""
    espectros_por_objeto: Dict[str, np.ndarray] = {}
    fruta_por_objeto: Dict[str, str] = {}
    camera_por_objeto: Dict[str, str] = {}
    for cubo, mascara, gid, fruta, camera in zip(
            cubos, mascaras, group_ids, frutas, cameras):
        espectro = aggregate_object_spectrum(cubo, mascara)
        if gid in espectros_por_objeto:
            # Front+back do MESMO objeto -- media das 2 gravacoes (ainda
            # 1 vetor por objeto fisico, nao 2 "amostras independentes").
            espectros_por_objeto[gid] = (espectros_por_objeto[gid] + espectro) / 2.0
        else:
            espectros_por_objeto[gid] = espectro
            fruta_por_objeto[gid] = fruta
            camera_por_objeto[gid] = camera

    combinacoes = sorted({(fruta_por_objeto[g], camera_por_objeto[g])
                          for g in espectros_por_objeto})

    ensemble: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for fruta, camera in combinacoes:
        objetos = [g for g in espectros_por_objeto
                  if fruta_por_objeto[g] == fruta and camera_por_objeto[g] == camera]
        n_grupos = len(objetos)
        X_obj = np.array([espectros_por_objeto[g] for g in objetos])
        n_bandas = X_obj.shape[1]

        # n_grupos<=1: nao da' pra' fitar PCA nenhum (0 graus de liberdade
        # com 1 amostra so') -- registra a entrada MESMO ASSIM (pca=None),
        # NOT_VALIDATED_N1, em vez de omitir a combinacao do ensemble em
        # silencio (que poderia ser confundido com "essa fruta nem existe
        # no treino" por quem consome o resultado).
        if n_grupos <= 1:
            ensemble[(fruta, camera)] = {
                "pca": None, "var_t": None, "centroide": None,
                "conformal": None, "n_grupos": n_grupos, "n_bandas": n_bandas,
                "cobertura_status": CoverageStatus.NOT_VALIDATED_N1,
                "alpha_alcancavel": None,
            }
            continue

        n_comp = min(5, X_obj.shape[0] - 1, n_bandas)
        pca = PCA(n_components=n_comp).fit(X_obj)
        T = pca.transform(X_obj)
        var_t = np.var(T, axis=0)
        var_t = np.where(var_t > 1e-12, var_t, 1e-12)
        centroide = T.mean(axis=0)
        scores = np.sum(((T - centroide.reshape(1, -1)) ** 2) / var_t.reshape(1, -1),
                        axis=1)

        conformal = ConformalOneClass(alpha=alpha_nominal).fit(
            scores, mae_id=np.array(objetos))
        if conformal.info_["alcancavel"]:
            status = CoverageStatus.VALIDATED
            alpha_alcancavel: Optional[float] = float(alpha_nominal)
        else:
            status = CoverageStatus.NOT_VALIDATED_N2_WEAK
            alpha_alcancavel = achievable_alpha(n_grupos)

        ensemble[(fruta, camera)] = {
            "pca": pca, "var_t": var_t, "centroide": centroide,
            "conformal": conformal, "n_grupos": n_grupos, "n_bandas": n_bandas,
            "cobertura_status": status, "alpha_alcancavel": alpha_alcancavel,
        }

    return ensemble


def identify_hsi_object(
        ensemble: Dict[Tuple[str, str], Dict[str, Any]],
        cubo: np.ndarray, mascara: np.ndarray, camera: str,
        n_candidatos: int = 3) -> HSIIdentificationResult:
    """Identifica a combinacao fruta x camera de UM objeto novo contra o
    `ensemble` calibrado -- so' compara com entradas da MESMA `camera`
    (numero de bandas incompativel entre cameras diferentes, nunca
    projeta um espectro de N bandas no PCA de outra camera com M!=N
    bandas). Nunca forca uma classe: `objeto_identificado` so' e'
    preenchido quando EXATAMENTE uma combinacao e' aceita sob cobertura
    VALIDADA -- mesma regra de `identificacao.identify_sample`."""
    entradas_compat = {k: v for k, v in ensemble.items() if k[1] == camera}
    if not entradas_compat:
        return HSIIdentificationResult(
            objeto_identificado=None, candidatos_ambiguos=[],
            cobertura_status=None, alpha_alcancavel=None, escores={})

    espectro = aggregate_object_spectrum(cubo, mascara)

    escores: Dict[str, float] = {}
    aceitos: List[str] = []
    for (fruta, cam), info in entradas_compat.items():
        rotulo = f"{fruta}|{cam}"
        if info["pca"] is None:
            continue  # NOT_VALIDATED_N1 -- sem PCA calibrado, nada a comparar
        if espectro.shape[0] != info["n_bandas"]:
            continue  # defesa extra -- nao deveria acontecer se camera bate
        T = info["pca"].transform(espectro.reshape(1, -1))
        score = float(np.sum(
            ((T - info["centroide"].reshape(1, -1)) ** 2) / info["var_t"].reshape(1, -1)))
        escores[rotulo] = score
        conformal = info["conformal"]
        if (info["cobertura_status"] == CoverageStatus.VALIDATED
                and conformal is not None
                and bool(conformal.predict(np.array([score]))[0])):
            aceitos.append(rotulo)

    ordenados = sorted(escores.items(), key=lambda kv: kv[1])
    candidatos_proximos = [rotulo for rotulo, _ in ordenados[:n_candidatos]]

    if len(aceitos) == 1:
        rotulo_ok = aceitos[0]
        fruta_ok, cam_ok = rotulo_ok.split("|", 1)
        return HSIIdentificationResult(
            objeto_identificado=rotulo_ok, candidatos_ambiguos=[],
            cobertura_status=CoverageStatus.VALIDATED,
            alpha_alcancavel=ensemble[(fruta_ok, cam_ok)]["alpha_alcancavel"],
            escores=escores)

    if len(aceitos) > 1:
        return HSIIdentificationResult(
            objeto_identificado=None, candidatos_ambiguos=sorted(aceitos),
            cobertura_status=CoverageStatus.VALIDATED,
            alpha_alcancavel=min(
                ensemble[(f, c)]["alpha_alcancavel"]
                for f, c in (r.split("|", 1) for r in aceitos)),
            escores=escores)

    return HSIIdentificationResult(
        objeto_identificado=None, candidatos_ambiguos=candidatos_proximos,
        cobertura_status=None, alpha_alcancavel=None, escores=escores)
