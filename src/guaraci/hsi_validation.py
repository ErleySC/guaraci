"""hsi_validation.py — Validacao externa por particao nativa de origem
(Passo 101 da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`).

Quando o dado tem particao nativa por DIA de medicao (cada dia e' uma
sessao de medicao/lote separada, nao um agrupamento arbitrario). Usamos
essa particao como teste de GENERALIZACAO REAL (Passo 101): treinar
so' com dias mais cedo, testar em dias nunca vistos -- diferente do
Passo 98 (que treina/testa objetos aleatorios misturados entre TODOS
os dias).

Reaproveita `figuras.specificity_by_class` (ja' existente) + sklearn
p/ sensibilidade (recall)/precisao por classe -- reportadas SEMPRE
separadas (teste interno x externo), nunca uma media unica que
esconderia queda de desempenho no externo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from guaraci.figuras import specificity_by_class
from guaraci.hsi_classification import fit_predict_pixel_plsda
from guaraci.hsi_pixels import build_pixel_dataset

__all__ = ["ExternalValidationReport", "run_external_validation_by_day",
           "run_internal_validation_group_aware", "run_cv_by_object",
           "intervalo_clopper_pearson", "NIVEL_IC"]

#: Nivel dos intervalos de confianca de sensibilidade/especificidade.
NIVEL_IC = 0.95


def intervalo_clopper_pearson(k: int, n: int, nivel: float = NIVEL_IC) -> Tuple[float, float]:
    """IC exato (Clopper & Pearson, 1934, Biometrika 26:404-413) para uma proporcao
    k/n. n = 0 -> (nan, nan). Cada objeto e' uma tentativa: com poucos objetos o
    intervalo e' largo, e e' isso que deve aparecer."""
    if n <= 0:
        return (float("nan"), float("nan"))
    from scipy.stats import beta
    a = (1.0 - nivel) / 2.0
    lo = 0.0 if k <= 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return (lo, hi)


@dataclass
class ExternalValidationReport:
    classes: List[str]
    n_objetos_teste_interno: int
    n_objetos_teste_externo: int
    sensibilidade_interna: Dict[str, float]
    especificidade_interna: Dict[str, float]
    precisao_interna: Dict[str, float]
    sensibilidade_externa: Dict[str, float]
    especificidade_externa: Dict[str, float]
    precisao_externa: Dict[str, float]
    # --- fase 2 (R11-c): como a validacao foi feita e com que base de objetos ---
    #: "cv_objetos" (CV por objeto sobre TODOS os objetos), "holdout" (um sorteio de
    #: objetos de teste) ou "externa_por_dia".
    tipo_validacao: str = "holdout"
    n_dobras: int = 0                          # CV: numero de dobras (= n objetos na leave-one-object-out)
    n_objetos_total: int = 0                   # objetos fisicos disponiveis
    n_objetos_avaliados: int = 0               # objetos com predicao fora da amostra
    n_objetos_por_classe: Dict[str, int] = field(default_factory=dict)   # entre os avaliados
    ic_sensibilidade_interna: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    ic_especificidade_interna: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    acuracia_objetos: float = float("nan")
    ic_acuracia_objetos: Tuple[float, float] = (float("nan"), float("nan"))
    acuracia_balanceada_objetos: float = float("nan")
    nivel_ic: float = NIVEL_IC
    #: predicao por objeto (agregada dos pixels) obtida SEM o objeto no ajuste (CV) ou
    #: dos objetos de teste (holdout); vazio se a validacao nao a produziu.
    predicoes_objeto: Dict[str, Any] = field(default_factory=dict)
    confianca_fora_da_amostra: bool = False
    avisos: List[str] = field(default_factory=list)


class _MetricasPorClasse(TypedDict):
    """Retorno de `_metricas_por_classe`: cada chave com o tipo que `ExternalValidationReport` espera."""
    sensibilidade: Dict[str, float]
    precisao: Dict[str, float]
    especificidade: Dict[str, float]
    n_objetos: Dict[str, int]
    ic_sensibilidade: Dict[str, Tuple[float, float]]
    ic_especificidade: Dict[str, Tuple[float, float]]


def _metricas_por_classe(y_real_por_objeto: Dict[str, str],
                          y_pred_por_objeto: Dict[str, str],
                          classes: Sequence[str]) -> _MetricasPorClasse:
    gids = sorted(y_real_por_objeto)
    y_real = [y_real_por_objeto[g] for g in gids]
    y_pred = [y_pred_por_objeto[g] for g in gids]

    cm = confusion_matrix(y_real, y_pred, labels=list(classes))
    sens = recall_score(y_real, y_pred, labels=list(classes),
                        average=None, zero_division=0)
    prec = precision_score(y_real, y_pred, labels=list(classes),
                           average=None, zero_division=0)
    espec = np.asarray(specificity_by_class(cm), dtype=float)
    # Classe sem objeto no teste: sensibilidade/precisao INDEFINIDAS (NaN), nao
    # 0.00 -- com 1 objeto de teste o relatorio dizia "sensibilidade 0.00" para
    # a classe ausente, como se o modelo tivesse errado.
    presentes = np.array([c in set(y_real) for c in classes])
    preditas = np.array([c in set(y_pred) for c in classes])
    sens = np.where(presentes, sens, np.nan)
    prec = np.where(preditas, prec, np.nan)
    espec = np.where([len(set(y_real) - {c}) > 0 for c in classes], espec, np.nan)

    # contagens de OBJETOS por classe + IC exato (cada objeto e' uma tentativa)
    n_obj: Dict[str, int] = {}
    ic_sens: Dict[str, Tuple[float, float]] = {}
    ic_espec: Dict[str, Tuple[float, float]] = {}
    for c in classes:
        reais_c = [g for g in gids if y_real_por_objeto[g] == c]
        n_obj[c] = len(reais_c)
        tp = sum(1 for g in reais_c if y_pred_por_objeto[g] == c)
        ic_sens[c] = intervalo_clopper_pearson(tp, len(reais_c))
        outros = [g for g in gids if y_real_por_objeto[g] != c]
        tn = sum(1 for g in outros if y_pred_por_objeto[g] != c)
        ic_espec[c] = intervalo_clopper_pearson(tn, len(outros))
    return {
        "sensibilidade": dict(zip(classes, sens.tolist())),
        "precisao": dict(zip(classes, prec.tolist())),
        "especificidade": dict(zip(classes, espec.tolist())),
        "n_objetos": n_obj, "ic_sensibilidade": ic_sens, "ic_especificidade": ic_espec,
    }


def _resumo_global(y_real: Dict[str, str], y_pred: Dict[str, str],
                   sens: Dict[str, float]) -> Dict[str, Any]:
    gids = sorted(y_real)
    n = len(gids)
    k = sum(1 for g in gids if y_real[g] == y_pred[g])
    validas = [v for v in sens.values() if np.isfinite(v)]
    return {"acuracia_objetos": (k / n) if n else float("nan"),
            "ic_acuracia_objetos": intervalo_clopper_pearson(k, n),
            "acuracia_balanceada_objetos": float(np.mean(validas)) if validas else float("nan")}


def run_external_validation_by_day(
        cubos: Sequence[np.ndarray], mascaras: Sequence[np.ndarray],
        group_ids: Sequence[str], rotulos: Sequence[str],
        dias: Sequence[str], dias_externos: Sequence[str], *,
        fracao_teste_interno: float = 0.25, seed: int = 42,
        max_lvs: int = 8, n_splits_wold: int = 2,
        max_pixels_por_gravacao: Optional[int] = None,
        ) -> ExternalValidationReport:
    """Treina com objetos de dias FORA de `dias_externos`, valida:
      - internamente: numa fracao dos MESMOS dias de treino (objetos
        nunca vistos, mas do mesmo lote/sessao) -- split group-aware por
        objeto.
      - externamente: em TODOS os objetos de `dias_externos` (lote/sessao
        nunca vista em nenhuma etapa do treino).

    `dias` deve ter o MESMO comprimento de `cubos`/`mascaras`/
    `group_ids`/`rotulos` (1 entrada por gravacao).

    `max_pixels_por_gravacao`: repassado a `build_pixel_dataset` (Passo
    104) -- necessario para objetos de resolucao alta (ex. uma camera,
    ~97000 pixels/imagem vs. ~4096 de outra) nao estourarem memoria.
    `None` (default) preserva o comportamento antigo."""
    dias_arr = np.asarray(dias)
    n = len(cubos)
    if not (len(mascaras) == len(group_ids) == len(rotulos) == len(dias_arr) == n):
        raise ValueError(
            "run_external_validation_by_day: todas as sequencias de "
            "entrada devem ter o mesmo comprimento (1 por gravacao).")

    e_externo = np.isin(dias_arr, list(dias_externos))
    if not e_externo.any():
        raise ValueError(
            f"Nenhuma gravacao pertence a dias_externos={list(dias_externos)} "
            f"-- confira os valores de `dias` (dias disponiveis: "
            f"{sorted(set(dias_arr.tolist()))}).")
    if e_externo.all():
        raise ValueError(
            "Todas as gravacoes pertencem a dias_externos -- nao sobra "
            "nada para treinar.")

    def _sel(seq, mascara_bool):
        return [v for v, m in zip(seq, mascara_bool) if m]

    cubos_dev = _sel(cubos, ~e_externo)
    mascaras_dev = _sel(mascaras, ~e_externo)
    grupos_dev = _sel(group_ids, ~e_externo)
    rotulos_dev = _sel(rotulos, ~e_externo)

    cubos_ext = _sel(cubos, e_externo)
    mascaras_ext = _sel(mascaras, e_externo)
    grupos_ext = _sel(group_ids, e_externo)
    rotulos_ext = _sel(rotulos, e_externo)

    X_dev, y_dev, pg_dev = build_pixel_dataset(
        cubos_dev, mascaras_dev, grupos_dev, rotulos_dev,
        max_pixels_por_gravacao=max_pixels_por_gravacao, seed=seed)
    X_ext, y_ext, pg_ext = build_pixel_dataset(
        cubos_ext, mascaras_ext, grupos_ext, rotulos_ext,
        max_pixels_por_gravacao=max_pixels_por_gravacao, seed=seed)

    objetos_dev = np.unique(pg_dev)
    rng = np.random.default_rng(seed)
    n_teste_interno = max(1, int(round(len(objetos_dev) * fracao_teste_interno)))
    objetos_teste_interno = set(
        rng.choice(objetos_dev, size=n_teste_interno, replace=False))
    mascara_teste_interno = np.array([g in objetos_teste_interno for g in pg_dev])

    resultado = fit_predict_pixel_plsda(
        X_dev[~mascara_teste_interno], y_dev[~mascara_teste_interno],
        pg_dev[~mascara_teste_interno],
        X_ext, pg_ext,   # predicoes no conjunto EXTERNO
        max_lvs=max_lvs, n_splits_wold=n_splits_wold, seed=seed)
    predicoes_externo = resultado["predicoes_objeto"]
    n_components = resultado["n_components"]

    resultado_interno = fit_predict_pixel_plsda(
        X_dev[~mascara_teste_interno], y_dev[~mascara_teste_interno],
        pg_dev[~mascara_teste_interno],
        X_dev[mascara_teste_interno], pg_dev[mascara_teste_interno],
        n_components=n_components)   # MESMO n_components do modelo externo
    predicoes_interno = resultado_interno["predicoes_objeto"]

    classes = sorted(set(rotulos))
    y_real_interno = {g: y_dev[pg_dev == g][0] for g in objetos_teste_interno}
    y_pred_interno = {g: r.classe_predita for g, r in predicoes_interno.items()}
    y_real_externo = {g: y_ext[pg_ext == g][0] for g in np.unique(pg_ext)}
    y_pred_externo = {g: r.classe_predita for g, r in predicoes_externo.items()}

    m_interno = _metricas_por_classe(y_real_interno, y_pred_interno, classes)
    m_externo = _metricas_por_classe(y_real_externo, y_pred_externo, classes)

    return ExternalValidationReport(
        classes=classes,
        n_objetos_teste_interno=len(y_real_interno),
        n_objetos_teste_externo=len(y_real_externo),
        sensibilidade_interna=m_interno["sensibilidade"],
        especificidade_interna=m_interno["especificidade"],
        precisao_interna=m_interno["precisao"],
        sensibilidade_externa=m_externo["sensibilidade"],
        especificidade_externa=m_externo["especificidade"],
        precisao_externa=m_externo["precisao"],
        tipo_validacao="externa_por_dia",
        n_objetos_total=int(len(np.unique(np.concatenate([pg_dev, pg_ext])))),
        n_objetos_avaliados=len(y_real_interno),
        n_objetos_por_classe=m_interno["n_objetos"],
        ic_sensibilidade_interna=m_interno["ic_sensibilidade"],
        ic_especificidade_interna=m_interno["ic_especificidade"],
        **_resumo_global(y_real_interno, y_pred_interno, m_interno["sensibilidade"]),
        predicoes_objeto=dict(predicoes_interno))


def run_internal_validation_group_aware(
        cubos: Sequence[np.ndarray], mascaras: Sequence[np.ndarray],
        group_ids: Sequence[str], rotulos: Sequence[str], *,
        fracao_teste: float = 0.25, seed: int = 42, max_lvs: int = 8,
        n_splits_wold: int = 2,
        max_pixels_por_gravacao: Optional[int] = None,
        pixels: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        ) -> ExternalValidationReport:
    """Validacao interna group-aware para dataset SEM particao nativa por
    dia/sessao (dado proprio do usuario, Passo 111 -- INSTRUCAO_HSI_DADO_
    PROPRIO.md) -- ao contrario de `run_external_validation_by_day`
    (dado particionado por dia de medicao), aqui
    NAO existe uma fatia "nunca vista em nenhuma etapa" pra' reservar
    como validacao externa de verdade: o holdout e' uma amostra aleatoria
    de OBJETOS FISICOS inteiros (nunca pixel isolado, nunca metade de um
    objeto em treino e metade em teste -- mesma garantia de
    `hsi_pixels.build_pixel_dataset`), sorteada entre TODOS os objetos.

    Por isso os campos `*_externa` do relatorio devolvido ficam VAZIOS
    (`{}`) e `n_objetos_teste_externo=0` -- declara explicitamente que
    so' ha' validacao interna disponivel, nunca finge ter uma fatia
    externa que nao existe (quem chama -- `hsi_pipeline.py` -- deve
    propagar isso pro log/tela, nao esconder atras de um numero zerado
    sem explicacao).

    `group_ids` NAO deve conter `None` -- quando o nivel de agrupamento
    detectado por `hsi_io.load_hsi_folder_dataset` for "none" (Bloco 8),
    quem chama deve resolver cada gravacao para um id UNICO antes de
    passar aqui (nunca compartilhar um placeholder entre gravacoes
    distintas -- isso colapsaria objetos fisicos DIFERENTES no mesmo
    grupo, o oposto do que a garantia de agrupamento existe pra' evitar).
    """
    n = len(cubos)
    if not (len(mascaras) == len(group_ids) == len(rotulos) == n):
        raise ValueError(
            "run_internal_validation_group_aware: todas as sequencias de "
            "entrada devem ter o mesmo comprimento (1 por gravacao).")

    if pixels is not None:     # dataset por pixel ja' montado pelo chamador (evita ler os cubos de novo)
        X, y, pg = pixels
    else:
        X, y, pg = build_pixel_dataset(
            cubos, mascaras, group_ids, rotulos,
            max_pixels_por_gravacao=max_pixels_por_gravacao, seed=seed)

    objetos = np.unique(pg)
    if len(objetos) < 2:
        raise ValueError(
            f"run_internal_validation_group_aware: so' {len(objetos)} "
            f"objeto(s) fisico(s) distinto(s) -- precisa de pelo menos 2 "
            f"pra' formar um holdout treino/teste group-aware.")

    rng = np.random.default_rng(seed)
    n_teste = max(1, int(round(len(objetos) * fracao_teste)))
    n_teste = min(n_teste, len(objetos) - 1)  # nunca esvazia o treino
    objetos_teste = set(rng.choice(objetos, size=n_teste, replace=False))
    mascara_teste = np.array([g in objetos_teste for g in pg])

    resultado = fit_predict_pixel_plsda(
        X[~mascara_teste], y[~mascara_teste], pg[~mascara_teste],
        X[mascara_teste], pg[mascara_teste],
        max_lvs=max_lvs, n_splits_wold=n_splits_wold, seed=seed)
    predicoes = resultado["predicoes_objeto"]

    classes = sorted(set(rotulos))
    y_real = {g: y[pg == g][0] for g in objetos_teste}
    y_pred = {g: r.classe_predita for g, r in predicoes.items()}
    m_interno = _metricas_por_classe(y_real, y_pred, classes)

    return ExternalValidationReport(
        classes=classes,
        n_objetos_teste_interno=len(y_real),
        n_objetos_teste_externo=0,
        sensibilidade_interna=m_interno["sensibilidade"],
        especificidade_interna=m_interno["especificidade"],
        precisao_interna=m_interno["precisao"],
        sensibilidade_externa={}, especificidade_externa={},
        precisao_externa={},
        tipo_validacao="holdout", n_dobras=0, n_objetos_total=int(len(objetos)),
        n_objetos_avaliados=len(y_real), n_objetos_por_classe=m_interno["n_objetos"],
        ic_sensibilidade_interna=m_interno["ic_sensibilidade"],
        ic_especificidade_interna=m_interno["ic_especificidade"],
        **_resumo_global(y_real, y_pred, m_interno["sensibilidade"]),
        predicoes_objeto=dict(predicoes),
        avisos=[f"holdout de {len(y_real)} objeto(s) de teste em {len(objetos)}: metrica indicativa, "
                f"com intervalos largos."])


def _dobras_por_objeto(classe_do_objeto: Dict[str, str], n_dobras: int, seed: int) -> List[List[str]]:
    """Objetos -> dobras, estratificado por classe (cada classe e' repartida por todas as
    dobras em rodizio) e sem nunca separar um objeto: a unidade e' o objeto fisico."""
    objetos = sorted(classe_do_objeto)
    if n_dobras <= 0 or n_dobras >= len(objetos):
        return [[g] for g in objetos]                       # leave-one-object-out
    rng = np.random.default_rng(seed)
    dobras: List[List[str]] = [[] for _ in range(n_dobras)]
    contador = 0
    for classe in sorted(set(classe_do_objeto.values())):
        do_classe = [g for g in objetos if classe_do_objeto[g] == classe]
        for g in rng.permutation(do_classe):
            dobras[contador % n_dobras].append(str(g))
            contador += 1
    return [d for d in dobras if d]


def run_cv_by_object(
        X: np.ndarray, y: np.ndarray, pixel_groups: np.ndarray, *,
        n_dobras: int = 5, max_lvs: int = 8, n_splits_wold: int = 2, seed: int = 42,
        ) -> ExternalValidationReport:
    """Validacao cruzada POR OBJETO fisico sobre TODOS os objetos (R11-c): cada objeto
    e' previsto por um PLS-DA ajustado so' com os OUTROS objetos (nenhum pixel dele no
    ajuste nem na escolha das variaveis latentes, que e' refeita DENTRO de cada treino).
    `n_dobras <= 0` (ou >= n objetos) = leave-one-object-out; senao k-fold estratificado
    por classe. Sensibilidade/especificidade por classe saem com IC exato
    (Clopper-Pearson) e o numero de OBJETOS de cada classe; a predicao por objeto
    (`predicoes_objeto`, de onde sai a confianca/heterogeneidade) e' FORA DA AMOSTRA.

    Cuidado de leitura: o IC trata cada objeto como uma tentativa independente; as
    dobras compartilham objetos de treino, entao e' aproximado (um pouco otimista)."""
    X = np.asarray(X)
    y = np.asarray(y).astype(str)
    pg = np.asarray(pixel_groups).astype(str)
    objetos = sorted(set(pg.tolist()))
    classe_do_objeto: Dict[str, str] = {}
    for g in objetos:
        cls = set(y[pg == g].tolist())
        if len(cls) != 1:
            raise ValueError(f"run_cv_by_object: o objeto '{g}' tem mais de uma classe ({sorted(cls)}).")
        classe_do_objeto[g] = cls.pop()
    classes = sorted(set(classe_do_objeto.values()))
    if len(objetos) < 3 or len(classes) < 2:
        raise ValueError(
            f"run_cv_by_object: precisa de pelo menos 2 classes e 3 objetos fisicos "
            f"(ha' {len(classes)} classe(s) e {len(objetos)} objeto(s)).")
    avisos: List[str] = []
    n_por_classe_total = {c: sum(1 for g in objetos if classe_do_objeto[g] == c) for c in classes}
    poucas = [c for c, n in n_por_classe_total.items() if n < 3]
    if poucas:
        avisos.append("classe(s) com menos de 3 objetos (" + ", ".join(
            f"{c}: {n_por_classe_total[c]}" for c in poucas) + "): sensibilidade com intervalo muito largo.")
    dobras = _dobras_por_objeto(classe_do_objeto, n_dobras, seed)
    y_real: Dict[str, str] = {}
    y_pred: Dict[str, str] = {}
    predicoes: Dict[str, Any] = {}
    n_sem_modelo = 0
    for fold in dobras:
        no_teste = np.isin(pg, fold)
        if len(set(y[~no_teste].tolist())) < 2:
            n_sem_modelo += len(fold)       # treino sem 2 classes: nao ha' modelo
            continue
        try:
            res = fit_predict_pixel_plsda(
                X[~no_teste], y[~no_teste], pg[~no_teste], X[no_teste], pg[no_teste],
                max_lvs=max_lvs, n_splits_wold=n_splits_wold, seed=seed)
        except ValueError:                  # poucos grupos no treino para escolher as VL
            res = fit_predict_pixel_plsda(
                X[~no_teste], y[~no_teste], pg[~no_teste], X[no_teste], pg[no_teste],
                n_components=1, seed=seed)
        for g, r in res["predicoes_objeto"].items():
            y_real[g] = classe_do_objeto[g]
            y_pred[g] = r.classe_predita
            predicoes[g] = r
    if n_sem_modelo:
        avisos.append(f"{n_sem_modelo} objeto(s) sem predicao: o treino sem eles ficaria com uma so' classe.")
    if not y_real:
        raise ValueError("run_cv_by_object: nenhuma dobra pode ser ajustada (treino com uma so' classe).")
    m = _metricas_por_classe(y_real, y_pred, classes)
    return ExternalValidationReport(
        classes=classes,
        n_objetos_teste_interno=len(y_real), n_objetos_teste_externo=0,
        sensibilidade_interna=m["sensibilidade"], especificidade_interna=m["especificidade"],
        precisao_interna=m["precisao"],
        sensibilidade_externa={}, especificidade_externa={}, precisao_externa={},
        tipo_validacao="cv_objetos", n_dobras=len(dobras), n_objetos_total=len(objetos),
        n_objetos_avaliados=len(y_real), n_objetos_por_classe=m["n_objetos"],
        ic_sensibilidade_interna=m["ic_sensibilidade"], ic_especificidade_interna=m["ic_especificidade"],
        **_resumo_global(y_real, y_pred, m["sensibilidade"]),
        predicoes_objeto=predicoes, confianca_fora_da_amostra=True, avisos=avisos)
