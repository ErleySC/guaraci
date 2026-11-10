"""ANOVA de uma via (e com bloco) + pos-teste de Tukey/Games-Howell sobre a
UNIDADE FISICA INDEPENDENTE -- nunca sobre replicas, espectros ou dobras de CV.

Por que existe um modulo proprio (e nao so' `scipy.stats.f_oneway`): o erro
classico em quimiometria e' testar grupos usando cada replica/espectro como se
fosse uma observacao independente (pseudo-replicacao, Hurlbert 1984). Replicas
da mesma amostra sao correlacionadas; contadas como independentes, encolhem o
erro e inflam a taxa de falso positivo. Aqui a regra e' fail-closed:

1. toda resposta vem com o id da sua unidade (amostra fisica ou sessao);
2. a resposta e' AGREGADA (media) por unidade ANTES do teste;
3. sem unidade resolvivel, o modulo RECUSA com mensagem -- nao ha' fallback
   silencioso para "cada linha e' uma unidade";
4. dobras de validacao cruzada nunca sao unidades (as dobras compartilham dado
   de treino e nao sao independentes; Nadeau & Bengio 2003): `origem=
   "dobras_cv"` e' recusado.

Premissas: Brown-Forsythe (Levene com centro = mediana; o campo chama-se
`levene_p` por compatibilidade). No `auto`, se rejeita, Welch ANOVA + Games-Howell
(`tukey_hsd(..., equal_var=False)`); senao ANOVA classica + Tukey (balanceado)
ou Tukey-Kramer (desbalanceado). Normalidade dos residuos (Shapiro) so' como
alerta -- com n pequeno o teste tem pouco poder e "nao rejeitar" nao prova
normalidade. Kruskal-Wallis opcional, rotulado como comparacao de POSTOS.

Varias respostas (ex.: varios elementos): Tukey DENTRO de cada resposta e
Benjamini-Hochberg (`false_discovery_control`) ENTRE respostas -- duas camadas
de correcao, documentadas na saida.

Referencias: Tukey (1949) doi:10.2307/3001913; Kramer (1956)
doi:10.2307/3001469; Games & Howell (1976) doi:10.3102/10769986001002113;
Nadeau & Bengio (2003) doi:10.1023/A:1024068626366.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

log = logging.getLogger(__name__)

__all__ = [
    "ResultadoANOVA",
    "ResultadoKruskal",
    "agregar_por_unidade",
    "anova_uma_via",
    "anova_varias_respostas",
    "diagnosticar_premissas",
    "kruskal_dunn",
    "recomendar_metodo",
    "N_MINIMO_PODER",
    "RAZAO_VARIANCIA_LIMITE",
    "ASSIMETRIA_LIMITE",
    "CURTOSE_LIMITE",
    "TEXTO_KRUSKAL",
    "TEXTO_PREMISSAS",
]

#: Abaixo disto (unidades por grupo) o teste tem pouco poder -- aviso, nao bloqueio.
N_MINIMO_PODER = 6
#: Pior erro por familia medido do Games-Howell (grupo de poucas unidades com a maior variancia; `guaraci manual limitacoes`, secao 2).
FWER_MAXIMO_GAMES_HOWELL_PEQUENO = 0.27

#: Limiares da RECOMENDACAO de metodo (regra pratica, nao teste). Razao entre a
#: maior e a menor variancia por grupo acima de 4 (= razao de desvios-padrao > 2);
#: assimetria dos residuos |g1| > 1; curtose em excesso > 3 (caudas pesadas).
#: Estritamente ACIMA do limite recomenda a troca; no limite, nao.
RAZAO_VARIANCIA_LIMITE = 4.0
ASSIMETRIA_LIMITE = 1.0
CURTOSE_LIMITE = 3.0

#: Frase obrigatoria na saida do Kruskal-Wallis/Dunn.
TEXTO_KRUSKAL = (
    "O Kruskal-Wallis compara DISTRIBUICOES (dominancia estocastica: um grupo "
    "tende a dar valores maiores que outro). So' equivale a 'diferenca de "
    "medianas' se as formas das distribuicoes forem parecidas.")

#: Texto obrigatorio do diagnostico de premissas.
TEXTO_PREMISSAS = (
    "Shapiro-Wilk e' informacao, nao porteiro: com n pequeno tem pouco poder "
    "(nao rejeitar nao prova normalidade) e com n grande rejeita por desvios "
    "minimos -- olhe o grafico quantil-quantil dos residuos. Escolher o teste "
    "em duas etapas (testar normalidade e depois decidir) nao e' recomendado "
    "como regra automatica (Rochon, Gondan & Kieser 2012, comparacao de dois "
    "grupos, doi:10.1186/1471-2288-12-81). "
    "O Tukey e' razoavelmente robusto a pequenos desvios com n igual; com "
    "variancias desiguais use Games-Howell.")

_ORIGENS_RECUSADAS = {
    "dobras_cv": ("dobras de validacao cruzada nao sao unidades independentes "
                  "(compartilham dado de treino; Nadeau & Bengio 2003). Compare "
                  "modelos com o teste t corrigido de Nadeau-Bengio, nao com ANOVA."),
    "replicas": ("replicas da mesma amostra nao sao unidades independentes; "
                 "informe o id da amostra fisica em `unidade`."),
    "espectros": ("espectros da mesma amostra nao sao unidades independentes; "
                  "informe o id da amostra fisica em `unidade`."),
}


@dataclass
class ResultadoANOVA:
    """Saida completa de uma ANOVA sobre unidades."""
    resposta: str
    n_linhas: int
    n_unidades: int
    grupos: List[str]
    n_por_grupo: Dict[str, int]
    media_por_grupo: Dict[str, float]
    tabela: Dict[str, Dict[str, float]]      # 'entre'/'dentro'/'total' -> SQ, gl, QM, F, p
    metodo: str                              # 'ANOVA classica' | 'Welch ANOVA' | 'ANOVA com bloco'
    F: float
    p: float
    eta2: float
    omega2: float
    eta2_ic95: tuple
    omega2_ic95: tuple
    levene_p: float
    shapiro_p: Optional[float]
    pos_teste: str                           # 'Tukey' | 'Tukey-Kramer' | 'Games-Howell'
    comparacoes: List[Dict[str, Any]]        # par a par: diferenca, IC95 simultaneo, p ajustado
    kruskal: Optional[Dict[str, Any]] = None
    p_fdr: Optional[float] = None            # preenchido por anova_varias_respostas
    avisos: List[str] = field(default_factory=list)


def _erro_unidade(n_linhas: int, grupo: np.ndarray, motivo: str) -> ValueError:
    k = len(set(map(str, grupo)))
    return ValueError(
        f"ANOVA recusada: {motivo} ({n_linhas} linha(s), {k} grupo(s)). "
        "Informe em `unidade` o id da amostra fisica (ou da sessao) de cada "
        "linha -- replicas e espectros da mesma amostra precisam do MESMO id. "
        "Sem isso o teste trataria replicas como independentes "
        "(pseudo-replicacao) e o p-valor sairia otimista.")


def agregar_por_unidade(resposta: Sequence[float], grupo: Sequence[Any],
                        unidade: Sequence[Any]) -> tuple:
    """Media da resposta por unidade. Devolve (y_unidade, grupo_unidade,
    bloco_ou_None ignorado, n_nan_descartados). Unidade em dois grupos e'
    erro (a unidade e' a amostra fisica; ela pertence a um grupo so')."""
    y = np.asarray(resposta, dtype=float)
    g = np.asarray(grupo, dtype=object).astype(str)
    u = np.asarray(unidade, dtype=object).astype(str)
    if not (len(y) == len(g) == len(u)):
        raise ValueError("resposta, grupo e unidade precisam ter o mesmo tamanho")
    validos = np.isfinite(y)
    n_nan = int((~validos).sum())
    y, g, u = y[validos], g[validos], u[validos]
    ys, gs, us = [], [], []
    for uid in dict.fromkeys(u):               # ordem estavel
        m = u == uid
        grupos_da_unidade = set(g[m])
        if len(grupos_da_unidade) > 1:
            raise ValueError(
                f"unidade '{uid}' aparece em mais de um grupo "
                f"({sorted(grupos_da_unidade)}): a unidade independente pertence "
                "a um grupo so'. Confira os ids.")
        ys.append(float(y[m].mean()))
        gs.append(str(next(iter(grupos_da_unidade))))
        us.append(uid)
    return np.asarray(ys), np.asarray(gs, dtype=object), np.asarray(us, dtype=object), n_nan


@lru_cache(maxsize=256)
def _q_critico(k: int, gl: float, confianca: float) -> float:
    """Quantil da amplitude studentizada (caro: ~0,15 s). Em varias respostas
    com os mesmos grupos/unidades, k e gl se repetem -- cache evita refazer."""
    return float(stats.studentized_range.ppf(confianca, k, gl))


def _pos_teste_par_a_par(amostras: List[np.ndarray], nomes: List[str],
                         variancias_iguais: bool, alpha: float) -> List[Dict[str, Any]]:
    """Tukey-Kramer (variancias iguais; = Tukey quando balanceado) ou
    Games-Howell (variancias desiguais), com IC simultaneo e p ajustado.
    Mesmas formulas de `scipy.stats.tukey_hsd` (conferido em teste), mas com o
    quantil critico em cache: com 40 elementos o custo caia de ~16 s para
    poucos segundos."""
    k = len(amostras)
    n = np.array([len(a) for a in amostras], dtype=float)
    m = np.array([a.mean() for a in amostras])
    v = np.array([a.var(ddof=1) for a in amostras])
    gl_dentro = float(n.sum() - k)
    msw = float(((n - 1) * v).sum() / gl_dentro)
    pares = [(i, j) for i in range(k) for j in range(i + 1, k)]
    dif = np.array([m[i] - m[j] for i, j in pares])
    if variancias_iguais:
        se = np.array([np.sqrt(msw / 2 * (1 / n[i] + 1 / n[j])) for i, j in pares])
        gl = np.full(len(pares), gl_dentro)
    else:
        vi = v / n
        se = np.array([np.sqrt((vi[i] + vi[j]) / 2) for i, j in pares])
        gl = np.array([(vi[i] + vi[j]) ** 2 / (vi[i] ** 2 / (n[i] - 1) + vi[j] ** 2 / (n[j] - 1))
                       for i, j in pares])
    q = np.abs(dif) / se       # se > 0: variancia dentro > 0 (recusada antes) e grupos sem variancia vao p/ a classica
    p = stats.studentized_range.sf(q, k, gl)
    out = []
    for idx, (i, j) in enumerate(pares):
        crit = _q_critico(k, round(float(gl[idx]), 10), 1 - alpha)
        out.append({"a": nomes[i], "b": nomes[j], "diferenca": float(dif[idx]),
                    "ic95": (float(dif[idx] - crit * se[idx]), float(dif[idx] + crit * se[idx])),
                    "p_ajustado": float(min(1.0, p[idx]))})
    return out


def _tabela_uma_via(amostras: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
    todos = np.concatenate(amostras)
    media = todos.mean()
    k, n = len(amostras), len(todos)
    sq_entre = float(sum(len(a) * (a.mean() - media) ** 2 for a in amostras))
    sq_dentro = float(sum(((a - a.mean()) ** 2).sum() for a in amostras))
    gl_e, gl_d = k - 1, n - k
    qm_e, qm_d = sq_entre / gl_e, sq_dentro / gl_d
    F = qm_e / qm_d            # qm_d > 0: variancia dentro zero e' recusada em anova_uma_via
    p = float(stats.f.sf(F, gl_e, gl_d))
    return {
        "entre": {"SQ": sq_entre, "gl": gl_e, "QM": qm_e, "F": F, "p": p},
        "dentro": {"SQ": sq_dentro, "gl": gl_d, "QM": qm_d},
        "total": {"SQ": sq_entre + sq_dentro, "gl": n - 1},
    }


def _variancia_inutilizavel(a: np.ndarray) -> bool:
    """Grupo sem variancia utilizavel pelo Welch/Games-Howell, que dividem por ela: constante (empates) OU com amplitude tao
    pequena que a variancia sofre underflow (ex.: 1.5e-292 ao quadrado = 0) e o peso n/variancia nao e' finito."""
    if np.ptp(a) == 0:
        return True
    v = float(np.var(a, ddof=1))
    return not (v > 0) or not np.isfinite(len(a) / v)


def _welch(amostras: List[np.ndarray]) -> tuple:
    """Welch ANOVA (variancias desiguais). Devolve (F, gl1, gl2, p)."""
    k = len(amostras)
    n = np.array([len(a) for a in amostras], dtype=float)
    m = np.array([a.mean() for a in amostras])
    v = np.array([a.var(ddof=1) for a in amostras])
    w = n / v
    mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = a / b
    gl2 = (k ** 2 - 1) / (3 * lam)
    return float(F), float(k - 1), float(gl2), float(stats.f.sf(F, k - 1, gl2))


def _eta_omega(amostras: List[np.ndarray]) -> tuple:
    t = _tabela_uma_via(amostras)
    sst = t["total"]["SQ"]
    eta2 = t["entre"]["SQ"] / sst      # sst >= SQ dentro > 0
    omega2 = (t["entre"]["SQ"] - t["entre"]["gl"] * t["dentro"]["QM"]) / (sst + t["dentro"]["QM"])
    return float(eta2), float(omega2)


def _ic_bootstrap(amostras: List[np.ndarray], n_boot: int, seed: int) -> tuple:
    """IC 95% percentil de eta2/omega2 reamostrando UNIDADES dentro de cada grupo."""
    rng = np.random.default_rng(seed)
    e, o = [], []
    for _ in range(n_boot):
        re = [a[rng.integers(0, len(a), len(a))] for a in amostras]
        if sum(((x - x.mean()) ** 2).sum() for x in re) == 0:
            continue
        ee, oo = _eta_omega(re)
        e.append(ee)
        o.append(oo)
    if not e:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    return (tuple(float(x) for x in np.percentile(e, [2.5, 97.5])),
            tuple(float(x) for x in np.percentile(o, [2.5, 97.5])))


def _bloco_aditivo(y: np.ndarray, g: np.ndarray, b: np.ndarray, nomes: List[str],
                   alpha: float) -> tuple:
    """ANOVA de duas vias SEM interacao (fator + bloco), SQ do fator ajustada
    ao bloco (tipo II), por minimos quadrados com numpy. Pos-teste: diferencas
    de medias ajustadas (LS-means) com erro-padrao do contraste e p pela
    distribuicao da amplitude studentizada (Tukey-Kramer generalizado)."""
    blocos = sorted(set(b))
    def matriz(com_fator: bool, com_bloco: bool) -> np.ndarray:
        cols = [np.ones(len(y))]
        if com_fator:
            cols += [(g == n).astype(float) for n in nomes[1:]]
        if com_bloco:
            cols += [(b == bl).astype(float) for bl in blocos[1:]]
        return np.column_stack(cols)
    def sq_res(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r), beta, np.linalg.matrix_rank(X)
    sq_full, beta, rank_full = sq_res(matriz(True, True))
    sq_sem_fator, _, rank_b = sq_res(matriz(False, True))
    sq_sem_bloco, _, rank_f = sq_res(matriz(True, False))
    gl_res = len(y) - rank_full
    gl_f, gl_b = rank_full - rank_b, rank_full - rank_f
    if gl_res <= 0 or gl_f <= 0:
        raise ValueError("delineamento com bloco sem graus de liberdade residuais "
                         "suficientes (poucas unidades por combinacao grupo x bloco)")
    qm_res = sq_full / gl_res
    sq_f, sq_b = sq_sem_fator - sq_full, sq_sem_bloco - sq_full
    F = (sq_f / gl_f) / qm_res if qm_res > 0 else float("inf")
    p = float(stats.f.sf(F, gl_f, gl_res)) if np.isfinite(F) else 0.0
    Fb = (sq_b / gl_b) / qm_res if (qm_res > 0 and gl_b > 0) else float("nan")
    tabela = {
        "entre": {"SQ": sq_f, "gl": gl_f, "QM": sq_f / gl_f, "F": F, "p": p},
        "bloco": {"SQ": sq_b, "gl": gl_b, "QM": sq_b / gl_b if gl_b else float("nan"),
                  "F": Fb, "p": float(stats.f.sf(Fb, gl_b, gl_res)) if np.isfinite(Fb) else float("nan")},
        "dentro": {"SQ": sq_full, "gl": gl_res, "QM": qm_res},
        "total": {"SQ": float(((y - y.mean()) ** 2).sum()), "gl": len(y) - 1},
    }
    X = matriz(True, True)
    cov = qm_res * np.linalg.pinv(X.T @ X)
    k = len(nomes)
    efeitos = np.concatenate([[0.0], beta[1:k]])   # efeito do grupo relativo ao 1o
    var_ef = np.zeros((k, k))
    var_ef[1:, 1:] = cov[1:k, 1:k]
    comps = []
    for i in range(k):
        for j in range(i + 1, k):
            d = efeitos[i] - efeitos[j]
            se = float(np.sqrt(var_ef[i, i] + var_ef[j, j] - 2 * var_ef[i, j]))
            q = abs(d) / se * np.sqrt(2) if se > 0 else float("inf")
            p_ij = float(stats.studentized_range.sf(q, k, gl_res)) if np.isfinite(q) else 0.0
            crit = float(stats.studentized_range.ppf(1 - alpha, k, gl_res)) / np.sqrt(2)
            comps.append({"a": nomes[i], "b": nomes[j], "diferenca": float(d),
                          "ic95": (float(d - crit * se), float(d + crit * se)),
                          "p_ajustado": min(1.0, p_ij)})
    return tabela, F, p, comps


def anova_uma_via(resposta: Sequence[float], grupo: Sequence[Any],
                  unidade: Optional[Sequence[Any]], *,
                  nome_resposta: str = "resposta",
                  bloco: Optional[Sequence[Any]] = None,
                  origem: str = "medida",
                  alpha: float = 0.05,
                  kruskal: bool = False,
                  variancias: str = "auto",
                  calcular_pos_teste: bool = True,
                  n_boot: int = 2000,
                  seed: int = 0) -> ResultadoANOVA:
    """ANOVA sobre a unidade independente. `unidade` e' obrigatoria (id da
    amostra fisica ou da sessao por linha); `bloco` opcional (ex.: sessao,
    lote de digestao) vira ANOVA de duas vias sem interacao, com o bloco
    resolvido por unidade. `calcular_pos_teste=False` pula o Tukey/Games-
    Howell (caro) quando so' o p da ANOVA interessa (triagem, simulacao).
    `variancias`: 'auto' (decide por Levene e AVISA no resultado), 'iguais'
    (ANOVA classica + Tukey/Tukey-Kramer) ou 'desiguais' (Welch + Games-Howell)."""
    if variancias not in ("auto", "iguais", "desiguais"):
        raise ValueError(f"variancias deve ser 'auto', 'iguais' ou 'desiguais' (recebido {variancias!r})")
    if origem in _ORIGENS_RECUSADAS:
        raise ValueError(f"ANOVA recusada: {_ORIGENS_RECUSADAS[origem]}")
    g_bruto = np.asarray(grupo, dtype=object)
    if unidade is None:
        raise _erro_unidade(len(g_bruto), g_bruto, "unidade independente nao informada")
    u_bruto = np.asarray(unidade, dtype=object)
    if any(x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == ""
           for x in u_bruto):
        raise _erro_unidade(len(g_bruto), g_bruto, "ha' linha(s) sem id de unidade")

    avisos: List[str] = []
    y_u, g_u, u_u, n_nan = agregar_por_unidade(resposta, grupo, unidade)
    if n_nan:
        avisos.append(f"{n_nan} linha(s) com resposta ausente/NaN descartada(s)")
    n_linhas = len(g_bruto)
    if len(y_u) == n_linhas - n_nan and n_linhas - n_nan > 0:
        avisos.append("cada linha e' uma unidade distinta: confirme que as linhas "
                      "NAO sao replicas/espectros da mesma amostra")

    nomes = sorted(str(x) for x in set(g_u))
    contagem = {n: int((g_u == n).sum()) for n in nomes}
    unitarios = [n for n in nomes if contagem[n] < 2]
    if unitarios:
        avisos.append(f"grupo(s) com 1 unidade so', excluido(s): {unitarios}")
        mant = np.isin(g_u, [n for n in nomes if n not in unitarios])
        y_u, g_u, u_u = y_u[mant], g_u[mant], u_u[mant]
        nomes = [n for n in nomes if n not in unitarios]
        contagem = {n: contagem[n] for n in nomes}
    if len(nomes) < 2:
        raise ValueError(
            f"ANOVA recusada: menos de 2 grupos com >= 2 unidades independentes "
            f"(grupos validos: {nomes}; unidades: {len(y_u)}). Traga mais amostras "
            "fisicas por grupo.")
    amostras = [y_u[g_u == n] for n in nomes]
    if sum(((a - a.mean()) ** 2).sum() for a in amostras) == 0:
        raise ValueError("ANOVA recusada: variancia dentro dos grupos e' zero "
                         "(respostas constantes por grupo); o teste F nao e' definido.")
    baixos = [n for n in nomes if contagem[n] < N_MINIMO_PODER]
    if baixos:
        avisos.append(f"grupo(s) com menos de {N_MINIMO_PODER} unidades ({baixos}): "
                      "poder baixo -- um resultado nao significativo nao indica "
                      "ausencia de efeito")

    levene_p = float(stats.levene(*amostras, center="median").pvalue)
    variancias_iguais = {"auto": not (levene_p < alpha), "iguais": True, "desiguais": False}[variancias]
    sem_variancia = [n for n, a in zip(nomes, amostras) if _variancia_inutilizavel(a)]
    if not variancias_iguais and sem_variancia:
        # Welch e Games-Howell dividem pela variancia de cada grupo: com um
        # grupo de variancia zero (empates) ficam indefinidos (achado por
        # Hypothesis, 2026-10-02: p = NaN). Usa a ANOVA classica e avisa.
        variancias_iguais = True
        avisos.append(f"grupo(s) com variancia zero ({sem_variancia}): Welch/Games-Howell "
                      "indefinidos; usada a ANOVA classica com Tukey-Kramer -- "
                      "interpretar com cautela (variancias claramente desiguais)")
    residuos = np.concatenate([a - a.mean() for a in amostras])
    shapiro_p = None
    if len(residuos) <= 5000:      # residuos nao constantes: variancia dentro zero e' recusada antes
        shapiro_p = float(stats.shapiro(residuos).pvalue)
        if shapiro_p < alpha:
            avisos.append(f"residuos com desvio de normalidade (Shapiro p={shapiro_p:.3g}); "
                          "considere a transformacao log ou o Kruskal-Wallis")
        elif len(residuos) < 20:
            avisos.append("Shapiro com poucas unidades tem pouco poder: nao rejeitar "
                          "nao demonstra normalidade")

    if bloco is not None:
        b_u = _bloco_por_unidade(bloco, unidade, u_u)
        tabela, F, p, comps = _bloco_aditivo(y_u, g_u, b_u, nomes, alpha)
        metodo, pos = "ANOVA com bloco (duas vias, sem interacao)", "Tukey-Kramer (medias ajustadas ao bloco)"
        # Com bloco nao ha' versao robusta a variancias desiguais aqui: o pos-teste usa o QM residual unico.
        # Mesmo criterio da recomendacao (razao > RAZAO_VARIANCIA_LIMITE ou Brown-Forsythe rejeita).
        var_g = [float(np.var(a, ddof=1)) for a in amostras]
        razao = (1.0 if max(var_g) == 0 else float("inf") if min(var_g) == 0 else max(var_g) / min(var_g))
        if razao > RAZAO_VARIANCIA_LIMITE or levene_p < alpha:
            avisos.append(f"variancias desiguais entre grupos (razao {razao:.1f}; Brown-Forsythe p={levene_p:.3g}): "
                          "o pos-teste com bloco supoe variancias iguais e nao tem versao robusta aqui -- "
                          "trate as comparacoes par a par como exploratorias")
    else:
        tabela = _tabela_uma_via(amostras)
        if variancias_iguais:
            F, p = tabela["entre"]["F"], tabela["entre"]["p"]
            metodo = "ANOVA classica"
            balanceado = len(set(contagem.values())) == 1 and not sem_variancia
            pos = "Tukey" if balanceado else "Tukey-Kramer"
        else:
            F, gl1, gl2, p = _welch(amostras)
            tabela["entre"].update({"F_welch": F, "gl1_welch": gl1, "gl2_welch": gl2, "p_welch": p})
            metodo, pos = "Welch ANOVA", "Games-Howell"
            # `levene_p` e' o Brown-Forsythe (Levene com centro = mediana). Fora do `auto` quem chama escolheu
            # variancias='desiguais' (pedido do usuario ou padrao de um comando): o texto nao afirma qual dos dois.
            avisos.append(f"Brown-Forsythe (centro = mediana) rejeitou variancias iguais (p={levene_p:.3g}): "
                          "Welch ANOVA + Games-Howell" if variancias == "auto"
                          else "Welch ANOVA + Games-Howell (variancias='desiguais')")
            if baixos:
                avisos.append(f"Games-Howell com grupo(s) de menos de {N_MINIMO_PODER} unidades ({baixos}): em simulacao o erro por familia "
                              f"chegou a {FWER_MAXIMO_GAMES_HOWELL_PEQUENO:.2f} (nominal {alpha:g}) quando o grupo pequeno tem a maior "
                              "variancia; trate as comparacoes par a par como exploratorias (guaraci manual limitacoes)")
        comps = []
        if not calcular_pos_teste:
            pos = pos + " (nao calculado: calcular_pos_teste=False)"
        else:
            comps = _pos_teste_par_a_par(amostras, nomes, variancias_iguais, alpha)

    eta2, omega2 = _eta_omega(amostras)
    e_ic, o_ic = _ic_bootstrap(amostras, n_boot, seed)
    kw: Optional[Dict[str, Any]] = None
    if kruskal:
        h = stats.kruskal(*amostras)
        kw = {"H": float(h.statistic), "p": float(h.pvalue),
              "nota": "comparacao de POSTOS (distribuicoes), nao de medias"}
    for a in avisos:
        log.info("[ANOVA %s] %s", nome_resposta, a)
    return ResultadoANOVA(
        resposta=nome_resposta, n_linhas=n_linhas, n_unidades=int(len(y_u)),
        grupos=nomes, n_por_grupo=contagem,
        media_por_grupo={n: float(a.mean()) for n, a in zip(nomes, amostras)},
        tabela=tabela, metodo=metodo, F=float(F), p=float(p),
        eta2=eta2, omega2=omega2, eta2_ic95=e_ic, omega2_ic95=o_ic,
        levene_p=levene_p, shapiro_p=shapiro_p, pos_teste=pos,
        comparacoes=comps, kruskal=kw, avisos=avisos)


def _bloco_por_unidade(bloco: Sequence[Any], unidade: Sequence[Any],
                       unidades_mantidas: np.ndarray) -> np.ndarray:
    b = np.asarray(bloco, dtype=object).astype(str)
    u = np.asarray(unidade, dtype=object).astype(str)
    out = []
    for uid in unidades_mantidas:
        vals = set(b[u == uid])
        if len(vals) > 1:
            raise ValueError(f"unidade '{uid}' aparece em mais de um bloco ({sorted(vals)})")
        out.append(next(iter(vals)))
    return np.asarray(out, dtype=object)


def anova_varias_respostas(respostas: Mapping[str, Any],
                           grupo: Sequence[Any], unidade: Optional[Sequence[Any]],
                           **kwargs: Any) -> Dict[str, Any]:
    """Uma ANOVA por resposta (ex.: por elemento) + Benjamini-Hochberg entre
    respostas. Camadas de correcao: Tukey/Games-Howell DENTRO de cada resposta
    (comparacoes par a par) e FDR ENTRE respostas (p da ANOVA de cada uma).
    Resposta que nao pode ser testada entra em `recusadas` com o motivo."""
    resultados: Dict[str, ResultadoANOVA] = {}
    recusadas: Dict[str, str] = {}
    for nome, y in respostas.items():
        try:
            resultados[nome] = anova_uma_via(y, grupo, unidade, nome_resposta=nome, **kwargs)
        except ValueError as e:
            if "unidade" in str(e) and "nao informada" in str(e):
                raise
            recusadas[nome] = str(e)
    if resultados:
        nomes = list(resultados)
        p_adj = stats.false_discovery_control([resultados[n].p for n in nomes], method="bh")
        for n, pa in zip(nomes, p_adj):
            resultados[n].p_fdr = float(pa)
    return {"resultados": resultados, "recusadas": recusadas,
            "correcao": "Tukey/Games-Howell dentro de cada resposta; "
                        "Benjamini-Hochberg (FDR) entre respostas"}


# ---------------------------------------------------------------------------
# Kruskal-Wallis + Dunn (alternativa nao parametrica ao ANOVA/Tukey)
# ---------------------------------------------------------------------------

@dataclass
class ResultadoKruskal:
    """Kruskal-Wallis + Dunn sobre unidades independentes."""
    resposta: str
    n_linhas: int
    n_unidades: int
    grupos: List[str]
    n_por_grupo: Dict[str, int]
    posto_medio_por_grupo: Dict[str, float]
    H: float
    gl: int
    p: float
    epsilon2: float                           # H / (n - 1), em [0, 1]
    ajuste: str                               # 'holm' | 'bh'
    regra_ajuste: str                         # texto da regra, para o relatorio
    comparacoes: List[Dict[str, Any]]         # Dunn: z, p bruto e p ajustado por par
    p_fdr: Optional[float] = None             # preenchido quando ha varias respostas
    texto: str = TEXTO_KRUSKAL
    avisos: List[str] = field(default_factory=list)


def _ajustar_p(p: np.ndarray, metodo: str) -> np.ndarray:
    """Holm (controla a taxa de erro por familia) ou Benjamini-Hochberg (FDR)."""
    m = len(p)
    if metodo == "bh":
        return np.asarray(stats.false_discovery_control(p, method="bh"), dtype=float)
    ordem = np.argsort(p)
    aj = np.empty(m)
    maximo = 0.0
    for pos, idx in enumerate(ordem):
        maximo = max(maximo, (m - pos) * p[idx])
        aj[idx] = min(1.0, maximo)
    return aj


_REGRA_AJUSTE = {
    "holm": "Holm: p ajustado = maximo cumulativo de (m - posicao) x p, com os p ordenados do menor para o maior",
    "bh": "Benjamini-Hochberg: controla a taxa de falsas descobertas (FDR), menos conservador que Holm",
}


def _dunn(amostras: List[np.ndarray], nomes: List[str], ajuste: str) -> List[Dict[str, Any]]:
    """Teste de Dunn (1964, doi:10.1080/00401706.1964.10490181) com postos
    GLOBAIS e correcao de empates na variancia:
        z_ij = (R_i - R_j) / sqrt( [N(N+1)/12 - sum(t^3 - t)/(12(N-1))] (1/n_i + 1/n_j) )
    p bilateral pela normal; correcao de multiplos testes por `ajuste`."""
    todos = np.concatenate(amostras)
    N = len(todos)
    postos = stats.rankdata(todos)
    _, contagens = np.unique(todos, return_counts=True)
    var_base = N * (N + 1) / 12 - float((contagens ** 3 - contagens).sum()) / (12 * (N - 1))
    cortes = np.cumsum([0] + [len(a) for a in amostras])
    pm = [float(postos[cortes[i]:cortes[i + 1]].mean()) for i in range(len(amostras))]
    n = [len(a) for a in amostras]
    pares = [(i, j) for i in range(len(amostras)) for j in range(i + 1, len(amostras))]
    z = np.array([(pm[i] - pm[j]) / np.sqrt(var_base * (1 / n[i] + 1 / n[j])) for i, j in pares])
    p_bruto = 2 * stats.norm.sf(np.abs(z))
    p_aj = _ajustar_p(p_bruto, ajuste)
    return [{"a": nomes[i], "b": nomes[j], "diferenca_postos": pm[i] - pm[j], "z": float(z[k]),
             "p_bruto": float(p_bruto[k]), "p_ajustado": float(p_aj[k])}
            for k, (i, j) in enumerate(pares)]


def kruskal_dunn(resposta: Sequence[float], grupo: Sequence[Any],
                 unidade: Optional[Sequence[Any]], *,
                 nome_resposta: str = "resposta", origem: str = "medida",
                 ajuste: str = "holm") -> ResultadoKruskal:
    """Kruskal-Wallis (com correcao de empates, `scipy.stats.kruskal`) e Dunn
    par a par sobre a UNIDADE independente (mesmas recusas do ANOVA).
    Tamanho de efeito: epsilon^2_H = H / (n - 1) = H / ((n^2 - 1)/(n + 1))."""
    if ajuste not in _REGRA_AJUSTE:
        raise ValueError(f"ajuste deve ser 'holm' ou 'bh' (recebido {ajuste!r})")
    if origem in _ORIGENS_RECUSADAS:
        raise ValueError(f"Kruskal-Wallis recusado: {_ORIGENS_RECUSADAS[origem]}")
    g_bruto = np.asarray(grupo, dtype=object)
    if unidade is None:
        raise _erro_unidade(len(g_bruto), g_bruto, "unidade independente nao informada")
    u_bruto = np.asarray(unidade, dtype=object)
    if any(x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == ""
           for x in u_bruto):
        raise _erro_unidade(len(g_bruto), g_bruto, "ha' linha(s) sem id de unidade")
    avisos: List[str] = []
    y_u, g_u, _, n_nan = agregar_por_unidade(resposta, grupo, unidade)
    if n_nan:
        avisos.append(f"{n_nan} linha(s) com resposta ausente/NaN descartada(s)")
    nomes = sorted(str(x) for x in set(g_u))
    contagem = {n: int((g_u == n).sum()) for n in nomes}
    unitarios = [n for n in nomes if contagem[n] < 2]
    if unitarios:
        avisos.append(f"grupo(s) com 1 unidade so', excluido(s): {unitarios}")
        mant = np.isin(g_u, [n for n in nomes if n not in unitarios])
        y_u, g_u = y_u[mant], g_u[mant]
        nomes = [n for n in nomes if n not in unitarios]
        contagem = {n: contagem[n] for n in nomes}
    if len(nomes) < 2:
        raise ValueError(
            f"Kruskal-Wallis recusado: menos de 2 grupos com >= 2 unidades independentes "
            f"(grupos validos: {nomes}; unidades: {len(y_u)}). Traga mais amostras "
            "fisicas por grupo.")
    amostras = [y_u[g_u == n] for n in nomes]
    if np.ptp(np.concatenate(amostras)) == 0:
        raise ValueError("Kruskal-Wallis recusado: todos os valores sao iguais "
                         "(todos empatados); a estatistica H nao e' definida.")
    baixos = [n for n in nomes if contagem[n] < N_MINIMO_PODER]
    if baixos:
        avisos.append(f"grupo(s) com menos de {N_MINIMO_PODER} unidades ({baixos}): "
                      "poder baixo -- um resultado nao significativo nao indica "
                      "ausencia de efeito")
    h = stats.kruskal(*amostras)
    n_tot = len(y_u)
    postos = stats.rankdata(np.concatenate(amostras))
    cortes = np.cumsum([0] + [len(a) for a in amostras])
    pm = {nm: float(postos[cortes[i]:cortes[i + 1]].mean()) for i, nm in enumerate(nomes)}
    for a in avisos:
        log.info("[Kruskal %s] %s", nome_resposta, a)
    return ResultadoKruskal(
        resposta=nome_resposta, n_linhas=len(g_bruto), n_unidades=int(n_tot), grupos=nomes,
        n_por_grupo=contagem, posto_medio_por_grupo=pm, H=float(h.statistic),
        gl=len(nomes) - 1, p=float(h.pvalue), epsilon2=float(h.statistic) / (n_tot - 1),
        ajuste=ajuste, regra_ajuste=_REGRA_AJUSTE[ajuste],
        comparacoes=_dunn(amostras, nomes, ajuste), avisos=avisos)


# ---------------------------------------------------------------------------
# Diagnostico de premissas + recomendacao (nunca troca o metodo em silencio)
# ---------------------------------------------------------------------------

def diagnosticar_premissas(resposta: Sequence[float], grupo: Sequence[Any],
                           unidade: Sequence[Any], *, alpha: float = 0.05) -> Dict[str, Any]:
    """Premissas do ANOVA de um fator, sobre as unidades. Devolve residuos
    (valor - media do grupo), pontos do grafico quantil-quantil, Shapiro-Wilk
    (informacao), Levene (centro = media) e Brown-Forsythe (centro = mediana),
    variancia por grupo e a razao maior/menor, assimetria e curtose dos residuos
    padronizados por grupo (a forma nao se confunde com a heterogeneidade)."""
    y_u, g_u, _, _ = agregar_por_unidade(resposta, grupo, unidade)
    todos = sorted(str(x) for x in set(g_u))
    nomes = [n for n in todos if (g_u == n).sum() >= 2]
    amostras = [y_u[g_u == n] for n in nomes]
    if len(amostras) < 2:
        raise ValueError("diagnostico recusado: menos de 2 grupos com >= 2 unidades independentes")
    res = np.concatenate([a - a.mean() for a in amostras])
    ordenado = np.sort(res)
    teoricos = stats.norm.ppf((np.arange(1, len(res) + 1) - 0.5) / len(res))
    var = {n: float(np.var(a, ddof=1)) for n, a in zip(nomes, amostras)}
    menor, maior = min(var.values()), max(var.values())
    if maior == 0:
        razao = 1.0
    elif menor == 0:
        razao = float("inf")
    else:
        razao = maior / menor
    constante = bool(np.ptp(res) == 0)
    shapiro_p = float(stats.shapiro(res).pvalue) if len(res) <= 5000 and not constante else None
    bf = float(stats.levene(*amostras, center="median").pvalue)
    lv = float(stats.levene(*amostras, center="mean").pvalue)
    # Forma medida nos residuos PADRONIZADOS por grupo (valor - media)/dp do grupo:
    # variancias desiguais, sozinhas, inflam a curtose dos residuos juntados (mistura
    # de escalas) e seriam lidas como "caudas pesadas". Grupo sem variancia sai.
    padronizados = np.concatenate([(a - a.mean()) / a.std(ddof=1) for a in amostras if np.ptp(a) > 0]
                                  or [np.zeros(1)])
    forma_constante = bool(np.ptp(padronizados) == 0)
    assim = 0.0 if forma_constante else float(stats.skew(padronizados))
    curt = 0.0 if forma_constante else float(stats.kurtosis(padronizados))
    n_por = [len(a) for a in amostras]
    return {"grupos": nomes, "residuos": res, "qq_teorico": teoricos, "qq_amostral": ordenado,
            "shapiro_p": shapiro_p, "levene_p": lv, "brown_forsythe_p": bf,
            "variancia_por_grupo": var, "razao_variancias": float(razao),
            "assimetria": assim, "curtose_excesso": curt,
            "n_por_grupo": dict(zip(nomes, n_por)),
            "razao_n": float(max(n_por) / min(n_por)), "alpha": alpha,
            "texto": TEXTO_PREMISSAS}


def recomendar_metodo(diag: Dict[str, Any]) -> Dict[str, str]:
    """Sugere o metodo e a razao em UMA linha. E' recomendacao, nao decisao:
    quem chama mostra ao usuario. Ordem das regras:
      1. assimetria |g1| > 1 ou curtose em excesso > 3 -> Kruskal-Wallis/Dunn;
      2. razao de variancias > 4 ou Brown-Forsythe p < alpha -> Games-Howell;
      3. senao -> ANOVA/Tukey."""
    if abs(diag["assimetria"]) > ASSIMETRIA_LIMITE or diag["curtose_excesso"] > CURTOSE_LIMITE:
        motivo = (f"residuos assimetricos ou de caudas pesadas (assimetria {diag['assimetria']:.2f}, "
                  f"curtose {diag['curtose_excesso']:.2f}) -> Kruskal-Wallis/Dunn")
        return {"metodo": "kruskal-dunn", "razao": motivo}
    if diag["razao_variancias"] > RAZAO_VARIANCIA_LIMITE or diag["brown_forsythe_p"] < diag["alpha"]:
        motivo = (f"variancias muito diferentes (razao {diag['razao_variancias']:.1f}; "
                  f"Brown-Forsythe p = {diag['brown_forsythe_p']:.3g}) -> Games-Howell")
        return {"metodo": "games-howell", "razao": motivo}
    return {"metodo": "tukey", "razao": "sem desvio relevante de premissas -> ANOVA/Tukey"}
