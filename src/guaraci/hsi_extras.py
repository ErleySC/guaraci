# -*- coding: utf-8 -*-
"""hsi_extras.py -- orquestracao testavel (sem Rich) dos modulos HSI que
existiam implementados e testados em isolamento mas sem nenhum caminho de
execucao real:
dominio de aplicabilidade (`hsi_applicability`), reamostragem de classes
minoritarias (`hsi_resampling`) e multiway N-PLS (`hsi_multiway`).

NAO cobre a identificacao por categoria x camera (fora do produto): sem
entrada multi-camera que possa ser exercitada de ponta a ponta, nao foi
exposto; registrado como limitacao.

Le a MESMA pasta de cubos ENVI do caminho generico de `run_hsi_pipeline`
(1 subpasta por classe) e reaproveita as mesmas primitivas (leitura,
quality gate + segmentacao, dataset por pixel) -- nao reimplementa
nenhuma etapa."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, cast

import numpy as np

from guaraci.hsi_applicability import (
    evaluate_hsi_applicability_domain,
    train_hsi_applicability_domain,
)
from guaraci.hsi_io import load_hsi_folder_dataset
from guaraci.hsi_multiway import comparar_npls_vs_pixelwise
from guaraci.hsi_pipeline import apply_quality_gate_and_segment
from guaraci.hsi_pixels import build_pixel_dataset
from guaraci.hsi_resampling import (
    class_evaluability_report,
    oversample_minority_groups,
)

__all__ = [
    "HSIDadosPreparados",
    "preparar_dados_hsi",
    "rodar_hsi_dominio_aplicabilidade",
    "rodar_hsi_reamostragem",
    "rodar_hsi_multiway",
]


@dataclass
class HSIDadosPreparados:
    cubos: List[np.ndarray]
    mascaras: List[np.ndarray]
    grupos: List[str]
    rotulos: List[str]
    X: np.ndarray
    y: np.ndarray
    pixel_groups: np.ndarray
    n_rejeitados: int


def preparar_dados_hsi(pasta: str, *, calibracao: str = "nenhuma", referencia_branco: str = "",
                       referencia_escuro: str = "", eixo_bandas: str = "recusar") -> HSIDadosPreparados:
    """Leitura + quality gate + segmentacao + dataset por pixel, sobre uma
    pasta de cubos ENVI (mesma convencao do modo `hsi` generico).

    `calibracao`, `referencia_branco`, `referencia_escuro` e `eixo_bandas` sao os mesmos que o fluxo principal
    le de `hsi_calibration`, `hsi_white_ref`, `hsi_dark_ref` e `hsi_band_axis`: sem eles os extras leriam os cubos
    brutos e aceitariam cubos de eixos diferentes, enquanto o modelo principal nao. Os padroes preservam o
    comportamento antigo. A faixa espectral e o SNV por pixel do fluxo principal NAO sao aplicados aqui."""
    cubos, rotulos, mae_id, _wl, _meta = load_hsi_folder_dataset(
        pasta, calibracao=calibracao, referencia_branco=referencia_branco,
        referencia_escuro=referencia_escuro, eixo_bandas=eixo_bandas)
    if mae_id is None:
        grupos = [f"__sem_grupo_{i}__" for i in range(len(cubos))]
    else:
        grupos = [str(g) for g in mae_id]
    filtrado = apply_quality_gate_and_segment(
        cubos, grupos, list(rotulos), [""] * len(cubos))
    if not filtrado["cubos"]:
        raise ValueError(
            f"Todas as {len(cubos)} gravacoes falharam no quality gate: "
            f"{filtrado['motivos_rejeicao'][:5]}")
    X, y, pixel_groups = build_pixel_dataset(
        filtrado["cubos"], filtrado["mascaras"], filtrado["group_ids"],
        filtrado["rotulos"])
    return HSIDadosPreparados(
        cubos=filtrado["cubos"], mascaras=filtrado["mascaras"],
        grupos=filtrado["group_ids"], rotulos=filtrado["rotulos"],
        X=X, y=y, pixel_groups=pixel_groups,
        n_rejeitados=filtrado["n_rejeitados"])


def rodar_hsi_dominio_aplicabilidade(
        dados: HSIDadosPreparados, *, fracao_teste: float = 0.3,
        seed: int = 42, n_components: int = 5) -> Dict[str, Any]:
    """Calibra o dominio de aplicabilidade nos pixels de objetos de
    TREINO e avalia os pixels de objetos de TESTE -- split por objeto
    fisico (nunca por pixel: pixels do mesmo objeto sao replicas)."""
    grupos_unicos = np.array(sorted(set(dados.pixel_groups.tolist())))
    if len(grupos_unicos) < 2:
        raise ValueError(
            "Dominio de aplicabilidade HSI precisa de >=2 objetos fisicos "
            "(1 p/ calibrar, 1 p/ avaliar).")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(grupos_unicos)
    n_teste = min(max(1, int(round(fracao_teste * len(perm)))), len(perm) - 1)
    grupos_teste = set(perm[:n_teste].tolist())
    is_teste = np.array([g in grupos_teste for g in dados.pixel_groups.tolist()])

    dominio = train_hsi_applicability_domain(
        dados.X[~is_teste], n_components=n_components)
    aval = evaluate_hsi_applicability_domain(dominio, dados.X[is_teste])
    por_objeto: Dict[str, float] = {}
    dentro = np.asarray(aval["dentro_dominio"], dtype=bool)
    pg_teste = dados.pixel_groups[is_teste]
    for g in sorted(grupos_teste):
        m = pg_teste == g                       # todo grupo de teste veio de `pixel_groups`: tem ao menos um pixel
        por_objeto[str(g)] = float(dentro[m].mean())
    return {
        "n_objetos_treino": len(grupos_unicos) - n_teste,
        "n_objetos_teste": n_teste,
        "fracao_dentro": float(cast(float, aval["fracao_dentro"])),
        "sensor_compativel": bool(aval["sensor_compativel"]),
        "fracao_dentro_por_objeto": por_objeto,
    }


def rodar_hsi_reamostragem(dados: HSIDadosPreparados, *, seed: int = 42
                            ) -> Dict[str, Any]:
    """Relatorio de avaliabilidade por classe + contagem de objetos/pixels
    antes e depois de `oversample_minority_groups`."""
    avaliab = class_evaluability_report(dados.y, dados.pixel_groups)
    X2, y2, g2 = oversample_minority_groups(
        dados.X, dados.y, dados.pixel_groups, seed=seed)

    def _contagem(y, g):
        return {c: {"pixels": int((y == c).sum()),
                    "objetos": len(set(g[y == c].tolist()))}
                for c in sorted(set(y.tolist()))}

    return {
        "avaliabilidade": {c: {"n_grupos": e.n_grupos, "avaliavel": e.avaliavel,
                                "nota": e.nota} for c, e in avaliab.items()},
        "antes": _contagem(dados.y, dados.pixel_groups),
        "depois": _contagem(y2, g2),
    }


def rodar_hsi_multiway(dados: HSIDadosPreparados, *, n_componentes: int = 2,
                        n_splits: int = 3, seed: int = 42) -> Dict[str, Any]:
    """Compara N-PLS multiway contra PLS-DA por pixel (mesmo split
    group-aware por objeto) -- exploratorio."""
    try:
        return dict(comparar_npls_vs_pixelwise(
            dados.cubos, dados.mascaras, dados.rotulos, dados.grupos,
            n_componentes=n_componentes, n_splits=n_splits, seed=seed))
    except ValueError as e:
        # Achado real (1a vez exercitado com entrada de usuario): com poucos
        # objetos fisicos por classe, a selecao de LVs (Wold) do PLS-DA por
        # pixel dentro de cada fold ficava com validacao vazia e estourava um
        # erro cru do scikit-learn ("0 sample(s)") -- verificado: 6
        # gravacoes falham, 10 funcionam.
        raise ValueError(
            f"Multiway HSI nao rodou com {len(set(dados.grupos))} objeto(s) "
            f"fisico(s) e n_splits={n_splits}: poucos objetos por classe "
            f"para o split group-aware + selecao de LVs dentro de cada fold "
            f"(erro original: {e}). Use mais objetos por classe (>=5 e' um "
            f"ponto de partida) ou menos folds.") from e
