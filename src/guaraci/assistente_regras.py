"""assistente_regras.py -- registro UNICO das regras do assistente (offline, deterministico, sem rede).

Cada regra olha um `Caso` -- a configuracao, um resumo dos dados e o resultado de uma execucao,
o que houver -- e, quando se aplica, devolve um `Achado` com severidade, mensagem, causa, o que
fazer, opcionalmente um campo/valor que o usuario pode aceitar, e as referencias de onde vem a regra.

Momentos:
  config     so' a configuracao (antes de carregar dados)
  dados      configuracao + resumo dos dados (pre-voo, antes de rodar)
  resultado  depois da execucao (le o resumo_modelo.txt)

Regras desta lista NUNCA alteram a configuracao: quem aplica e' o usuario. Limiar sem norma
publicada esta marcado como heuristico no proprio texto. Toda referencia tem DOI ou URL conferidos
(ver REFERENCIAS). Teste: tests/test_assistente_regras.py.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["ResumoDados", "Caso", "Regra", "Achado", "REGRAS", "REFERENCIAS", "resumir_dados", "avaliar",
           "resultado_de_resumo", "formatar_achados", "citar", "MOMENTOS", "SEVERIDADES"]

MOMENTOS = ("config", "dados", "resultado")
SEVERIDADES = ("critico", "aviso", "info")
_ORDEM_SEV = {s: i for i, s in enumerate(SEVERIDADES)}

#: Nivel de significancia usado pelo teste de permutacao do pipeline (nao configuravel hoje).
ALPHA_PADRAO = 0.05

# Modos cujo eixo e' numero de onda e cujo pre-processamento e' espectral.
_MODOS_TABULARES = ("multielementar",)
_MODOS_SEM_EIXO_CM = ("imagem", "hsi", "gcms", "hplc", "eem", "rmn", "multielementar")

#: id -> (citacao curta, DOI ou URL). Todas conferidas no Crossref ou na pagina do editor em 2026-10-15.
REFERENCIAS: Dict[str, Tuple[str, str]] = {
    "beleites2013": ("Beleites et al. (2013) Anal. Chim. Acta 760:25-33", "https://doi.org/10.1016/j.aca.2012.11.007"),
    "brereton2014": ("Brereton & Lloyd (2014) J. Chemom. 28:213-225", "https://doi.org/10.1002/cem.2609"),
    "brown2001": ("Brown, Cai & DasGupta (2001) Stat. Sci. 16:101-133", "https://doi.org/10.1214/ss/1009213286"),
    "bro2003": ("Bro & Smilde (2003) J. Chemom. 17:16-33", "https://doi.org/10.1002/cem.773"),
    "rinnan2009": ("Rinnan, van den Berg & Engelsen (2009) TrAC 28:1201-1222", "https://doi.org/10.1016/j.trac.2009.07.007"),
    "guo2017": ("Guo et al. (2017) Anal. Methods 9:4410-4417", "https://doi.org/10.1039/C7AY01363A"),
    "kapoor2023": ("Kapoor & Narayanan (2023) Patterns 4:100804", "https://doi.org/10.1016/j.patter.2023.100804"),
    "kjeldahl2010": ("Kjeldahl & Bro (2010) J. Chemom. 24:558-564", "https://doi.org/10.1002/cem.1346"),
    "varma2006": ("Varma & Simon (2006) BMC Bioinformatics 7:91", "https://doi.org/10.1186/1471-2105-7-91"),
    "westad2015": ("Westad & Marini (2015) Anal. Chim. Acta 893:14-24", "https://doi.org/10.1016/j.aca.2015.06.056"),
}


def citar(ref_id: str) -> str:
    c, u = REFERENCIAS[ref_id]
    return f"{c}, {u}"


# ------------------------------------------------------------------ entradas
@dataclass
class ResumoDados:
    """O que as regras precisam saber dos dados, sem carregar o motor."""
    n_leituras: int
    n_variaveis: int
    n_unidades: int                                # unidades independentes (mae_id); = n_leituras se nao houver
    leituras_por_unidade_max: int = 1
    unidades_por_classe: Dict[str, int] = field(default_factory=dict)
    n_nan: int = 0
    n_constantes: int = 0
    eixo_min: Optional[float] = None
    eixo_max: Optional[float] = None


def resumir_dados(X: Any, eixo: Any = None, rotulos: Any = None, mae_id: Any = None) -> ResumoDados:
    """Resumo dos dados carregados. So' numpy; nao altera nada."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X deve ser 2D (amostras x variaveis), veio com {X.ndim} dimensoes")
    n, p = X.shape
    unid = np.asarray(mae_id) if mae_id is not None else np.arange(n)
    _, cont = np.unique(unid, return_counts=True)
    por_classe: Dict[str, int] = {}
    if rotulos is not None:
        rot = np.asarray(rotulos)
        for c in np.unique(rot):
            por_classe[str(c)] = int(len(np.unique(unid[rot == c])))
    with np.errstate(invalid="ignore"):
        dp = np.nanstd(X, axis=0) if n > 1 else np.zeros(p)
    ex = np.asarray(eixo, dtype=float) if eixo is not None and len(eixo) else None
    return ResumoDados(n_leituras=int(n), n_variaveis=int(p), n_unidades=int(len(cont)),
                       leituras_por_unidade_max=int(cont.max()) if len(cont) else 0,
                       unidades_por_classe=por_classe, n_nan=int(np.isnan(X).sum()),
                       n_constantes=int(np.sum(np.nan_to_num(dp) == 0.0)),
                       eixo_min=float(np.nanmin(ex)) if ex is not None else None,
                       eixo_max=float(np.nanmax(ex)) if ex is not None else None)


@dataclass
class Caso:
    cfg: Any = None
    dados: Optional[ResumoDados] = None
    resultado: Optional[Dict[str, Any]] = None

    def c(self, attr: str, padrao: Any = None) -> Any:
        return getattr(self.cfg, attr, padrao) if self.cfg is not None else padrao

    def r(self, chave: str) -> Any:
        return (self.resultado or {}).get(chave)


@dataclass(frozen=True)
class Regra:
    id: str
    momento: str
    severidade: str
    quando: Callable[[Caso], bool]
    mensagem: Tuple[str, str]           # formatos (PT, EN) com {chaves} de `valores`
    causa: Tuple[str, str]
    fazer: Tuple[str, str]
    valores: Callable[[Caso], Dict[str, Any]] = lambda c: {}
    campo: Optional[str] = None         # chave YAML que o usuario pode mudar
    valor: Callable[[Caso], Any] = lambda c: None
    refs: Tuple[str, ...] = ()


@dataclass
class Achado:
    id: str
    momento: str
    severidade: str
    mensagem: str
    causa: str
    fazer: str
    campo: Optional[str] = None
    valor: Any = None
    refs: Tuple[str, ...] = ()


# ------------------------------------------------------------------ auxiliares
def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _espectral(c: Caso) -> bool:
    return str(c.c("mode", "") or "") not in _MODOS_SEM_EIXO_CM


def _tem_grupo(c: Caso) -> bool:
    return bool(c.c("group_column") or c.c("group_file"))


def _n_perm(c: Caso) -> int:
    try:
        return int(c.c("n_permutations", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _valida_modelo(c: Caso) -> bool:
    """O objetivo exploratorio so' gera PCA/HCA/loadings, sem validacao cruzada nem holdout."""
    if c.cfg is None:
        return True
    from guaraci.modos_analise import EXPLORATORIO, resolve_objective
    return resolve_objective(c.cfg) != EXPLORATORIO


def _eh_classificacao(c: Caso) -> bool:
    """Metricas de classe (acerto, permutacao, R2Y/Q2 do PLS-DA) so' valem para o objetivo de classificacao:
    numa quantificacao elas vem do modelo auxiliar e nao dizem nada sobre o teor (auditoria A-08)."""
    if c.cfg is None:
        return True
    from guaraci.modos_analise import CLASSIFICACAO, resolve_objective
    return resolve_objective(c.cfg) == CLASSIFICACAO


def _n_teste(c: Caso) -> int:
    if not _valida_modelo(c):
        return 0
    d = c.dados
    f = _num(c.c("frac_holdout", 0.0)) or 0.0
    return int(round(f * d.n_unidades)) if d else 0


def _wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Intervalo de Wilson de 95% para uma proporcao p observada em n unidades (Wilson 1927); ao contrario
    do intervalo normal, continua dentro de [0, 1] e nao encolhe a zero com n pequeno."""
    if n <= 0:
        return 0.0, 1.0
    d = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meia = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, centro - meia), min(1.0, centro + meia)


def _razao_classes(c: Caso) -> float:
    v = [x for x in (c.dados.unidades_por_classe.values() if c.dados else []) if x > 0]
    return (max(v) / min(v)) if len(v) >= 2 else 1.0


def _faixa_sem_dados(c: Caso) -> bool:
    d = c.dados
    if d is None or d.eixo_min is None or not _espectral(c):
        return False
    lo, hi = _num(c.c("wn_min")), _num(c.c("wn_max"))
    if lo is None or hi is None or lo >= hi:
        return False
    return hi < d.eixo_min or lo > d.eixo_max

def _com_eixo(c: Caso) -> bool:
    """O modo de entrada tem eixo fisico (ppm, ms, cm-1...), onde faixas excluidas fazem sentido."""
    return str(c.c("mode", "") or "") not in ("imagem", "hsi", "eem", "multielementar")


def _faixas_excluidas(c: Caso):
    from guaraci.preprocessamento import interpretar_faixas
    try:
        return interpretar_faixas(c.c("exclude_ranges", ()))
    except ValueError:
        return []


def _excluidas_fora_do_eixo(c: Caso):
    d = c.dados
    if d is None or d.eixo_min is None or not _com_eixo(c):
        return []
    return [f"{a:g}-{b:g}" for a, b in _faixas_excluidas(c) if b < d.eixo_min or a > d.eixo_max]


def _fracao_excluida(c: Caso) -> float:
    """Parte da faixa usada (corte de faixa x eixo dos dados) coberta pelas faixas excluidas."""
    d = c.dados
    if d is None or d.eixo_min is None or not _com_eixo(c):
        return 0.0
    lo, hi = max(d.eixo_min, _num(c.c("wn_min")) or d.eixo_min), min(d.eixo_max, _num(c.c("wn_max")) or d.eixo_max)
    if hi <= lo:
        return 0.0
    tirado = sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in _faixas_excluidas(c))
    return tirado / (hi - lo)


def _parece_ppm_sem_exclusao(c: Caso) -> bool:
    """RMN 1H (modo rmn, ou eixo que vai de ~0 a 8-20 com centenas de pontos, onde a agua cai em 4.7) sem
    nenhuma faixa excluida. Heuristica: nao ha como saber a unidade de um CSV."""
    d = c.dados
    if d is None or d.eixo_min is None or not _com_eixo(c) or _faixas_excluidas(c):
        return False
    if str(c.c("mode", "") or "") == "rmn":
        return True
    return d.eixo_min <= 1.0 and 8.0 <= d.eixo_max <= 20.0 and d.n_variaveis >= 200 and d.eixo_min > -10.0

def _rip_sem_janela(c: Caso) -> bool:
    return str(c.c("normalization", "nenhuma") or "nenhuma") == "rip" and not str(c.c("rip_window", "") or "").strip()


def _rip_janela_fora_da_faixa(c: Caso) -> bool:
    if str(c.c("normalization", "nenhuma") or "nenhuma") != "rip" or _rip_sem_janela(c):
        return False
    from guaraci.preprocessamento import interpretar_janela
    try:
        jan = interpretar_janela(c.c("rip_window"))
    except ValueError:
        return False
    lo, hi = _num(c.c("wn_min")), _num(c.c("wn_max"))
    return bool(jan and lo is not None and hi is not None and (jan[1] < lo or jan[0] > hi))


def _auditoria_critica(c: Caso) -> List[str]:
    return list((c.resultado or {}).get("_auditoria_critica", []) or [])


def _extras_ausentes(c: Caso, tipo: str) -> List[Any]:
    """Extras opcionais ausentes de que a config depende, do `tipo` ("formato" ou "recurso")."""
    if c.cfg is None:
        return []
    from guaraci import extras_opcionais as ext
    return [e for e, t in ext.extras_exigidos_por(c.cfg) if t == tipo]


def _valores_extra(c: Caso, tipo: str) -> Dict[str, Any]:
    from guaraci import extras_opcionais as ext
    e = _extras_ausentes(c, tipo)[0]
    return dict(modo=str(c.c("mode", "") or ""), pacote=e.pacote, comando=ext.texto_instalar(e))


# ------------------------------------------------------------------ regras
REGRAS: Tuple[Regra, ...] = (
    # ---------------------------------------------------------------- config
    Regra("faixa_invertida", "config", "critico",
          lambda c: _espectral(c) and (_num(c.c("wn_min")) or 0) >= (_num(c.c("wn_max")) or 0),
          ("A faixa espectral esta invertida ou vazia: faixa_min_cm = {lo:g} e faixa_max_cm = {hi:g}.",
           "The spectral range is inverted or empty: faixa_min_cm = {lo:g} and faixa_max_cm = {hi:g}."),
          ("Com o minimo maior ou igual ao maximo, o corte remove todas as variaveis.",
           "With the minimum at or above the maximum, the cut removes every variable."),
          ("Troque os dois valores de lugar ou digite a faixa da sua tecnica.",
           "Swap the two values or type your technique's range."),
          valores=lambda c: dict(lo=_num(c.c("wn_min")) or 0, hi=_num(c.c("wn_max")) or 0)),
    Regra("extra_formato_ausente", "config", "critico",
          lambda c: bool(_extras_ausentes(c, "formato")),
          ("O formato de entrada '{modo}' exige o pacote opcional '{pacote}', que nao esta instalado.",
           "The input format '{modo}' needs the optional package '{pacote}', which is not installed."),
          ("Sem o pacote, cada arquivo vira erro de leitura e a execucao termina com 'nenhuma amostra valida'.",
           "Without the package every file becomes a read error and the run ends with 'no valid sample'."),
          ("Instale com {comando} (tudo de uma vez: pip install \".[all]\") ou exporte os dados para CSV.",
           "Install with {comando} (everything at once: pip install \".[all]\") or export the data to CSV."),
          valores=lambda c: _valores_extra(c, "formato")),
    Regra("extra_recurso_ausente", "config", "aviso",
          lambda c: bool(_extras_ausentes(c, "recurso")),
          ("A opcao ligada depende do pacote opcional '{pacote}', que nao esta instalado: o recurso que o usa ficara de fora.",
           "An enabled option depends on the optional package '{pacote}', which is not installed: the feature that uses it will be left out."),
          ("A execucao segue, mas sem esse recurso (o pacote nao faz parte da instalacao base).",
           "The run goes on without that feature (the package is not part of the base install)."),
          ("Instale com {comando} (tudo de uma vez: pip install \".[all]\") ou desligue a opcao.",
           "Install with {comando} (everything at once: pip install \".[all]\") or turn the option off."),
          valores=lambda c: _valores_extra(c, "recurso")),
    Regra("rip_sem_janela", "config", "critico", _rip_sem_janela,
          ("normalizacao: rip sem ims_rip_janela.", "normalizacao: rip without ims_rip_janela."),
          ("A normalizacao divide cada espectro pelo pico do RIP; sem a janela onde ele esta, nao ha pico a usar.",
           "The normalisation divides each spectrum by the RIP peak; without the window where it sits there is no peak to use."),
          ("Informe ims_rip_janela (ex.: 7-9, na unidade do eixo) ou use normalizacao: nenhuma ou area.",
           "Set ims_rip_janela (e.g. 7-9, in axis units) or use normalizacao: nenhuma or area.")),
    Regra("rip_janela_fora_da_faixa", "config", "critico", _rip_janela_fora_da_faixa,
          ("ims_rip_janela {jan} esta fora da faixa usada ({lo:g} a {hi:g}).", "ims_rip_janela {jan} is outside the range in use ({lo:g} to {hi:g})."),
          ("A normalizacao e' feita sobre o espectro ja cortado por faixa_min_cm/faixa_max_cm; o RIP cortado nao pode servir de referencia.",
           "The normalisation runs on the spectrum already cut by faixa_min_cm/faixa_max_cm; a cut-off RIP cannot be the reference."),
          ("Amplie a faixa para incluir o RIP ou mude a janela.", "Widen the range to include the RIP or change the window."),
          valores=lambda c: dict(jan=str(c.c("rip_window")), lo=_num(c.c("wn_min")) or 0, hi=_num(c.c("wn_max")) or 0)),
    Regra("permutacoes_sem_resolucao", "config", "critico",
          lambda c: 0 < _n_perm(c) and 1.0 / (_n_perm(c) + 1) > ALPHA_PADRAO,
          ("Com {n} permutacoes o menor p-valor possivel e' 1/{n1} = {pmin:.3f}, acima de {alpha}.",
           "With {n} permutations the smallest possible p-value is 1/{n1} = {pmin:.3f}, above {alpha}."),
          ("O p-valor de permutacao conta quantas permutacoes empatam ou superam o modelo real; com poucas, "
           "nenhum resultado chega a ser significativo.",
           "The permutation p-value counts how many permutations tie or beat the real model; with few, no "
           "result can reach significance."),
          ("Use pelo menos {minimo} permutacoes; 200 e' o padrao.", "Use at least {minimo} permutations; 200 is the default."),
          valores=lambda c: dict(n=_n_perm(c), n1=_n_perm(c) + 1, pmin=1.0 / (_n_perm(c) + 1), alpha=ALPHA_PADRAO,
                                 minimo=int(math.ceil(1 / ALPHA_PADRAO)) - 1),
          campo="n_permutacoes", valor=lambda c: 200),
    Regra("permutacoes_poucas", "config", "aviso",
          lambda c: 0 < _n_perm(c) < 100 and 1.0 / (_n_perm(c) + 1) <= ALPHA_PADRAO,
          ("Poucas permutacoes deixam o p-valor do modelo grosseiro; 200 e o padrao.",
           "Few permutations make the model p-value coarse; 200 is the default."),
          ("O p-valor so' anda em passos de 1/{n1}.", "The p-value only moves in steps of 1/{n1}."),
          ("Para publicar, use 200 ou mais.", "To publish, use 200 or more."),
          valores=lambda c: dict(n1=_n_perm(c) + 1), campo="n_permutacoes", valor=lambda c: 200),
    Regra("grupo_nao_declarado", "config", "aviso",
          lambda c: str(c.c("mode", "") or "") in ("csv", "multielementar") and not _tem_grupo(c),
          ("Declare a unidade independente em `coluna_grupo` (amostra fisica, lote ou sessao): "
           "sem ela a validacao pode tratar replicas como amostras independentes.",
           "Declare the independent unit in `coluna_grupo` (physical sample, batch or session): "
           "without it validation may treat replicates as independent samples."),
          ("Leituras da mesma amostra fisica em treino e em teste deixam a metrica otimista.",
           "Readings of the same physical sample in both training and test make the metric optimistic."),
          ("Informe a coluna do CSV que identifica a amostra fisica.", "Give the CSV column that identifies the physical sample."),
          refs=("guo2017", "kapoor2023")),
    Regra("preproc_espectral_em_tabela", "config", "aviso",
          lambda c: str(c.c("mode", "") or "") in _MODOS_TABULARES
          and str(c.c("default_preprocessing", "")) not in ("autoscaling", "mc"),
          ("Em tabela de elementos use autoescala (MSC, SNV e Savitzky-Golay sao espectrais).",
           "On an element table use autoscaling (MSC, SNV and Savitzky-Golay are spectral)."),
          ("Esses metodos supoem variaveis vizinhas num eixo continuo, o que uma tabela de elementos nao tem.",
           "Those methods assume neighbouring variables on a continuous axis, which an element table lacks."),
          ("Troque o pre-processamento para autoescala.", "Switch the pre-processing to autoscaling."),
          campo="pre_processamento", valor=lambda c: "autoscaling", refs=("bro2003",)),
    Regra("autoescala_em_espectro", "config", "aviso",
          lambda c: _espectral(c) and str(c.c("default_preprocessing", "")) == "autoscaling",
          ("Autoescala num espectro da a cada numero de onda a mesma variancia.",
           "Autoscaling a spectrum gives every wavenumber the same variance."),
          ("Regioes de ruido passam a pesar tanto quanto as bandas; em espectro o usual e' centrar na media.",
           "Noise regions weigh as much as bands; on spectra the usual choice is mean-centring."),
          ("Prefira um pre-processamento espectral terminado em centragem (ex.: msc_sg_mc), ou confirme "
           "a escolha comparando os dois com a mesma validacao.",
           "Prefer a spectral pre-processing ending in centring (e.g. msc_sg_mc), or confirm the choice by "
           "comparing both with the same validation."),
          refs=("bro2003",)),
    Regra("sem_holdout", "config", "aviso",
          lambda c: (_num(c.c("frac_holdout", 0.2)) or 0.0) <= 0.0,
          ("Sem holdout externo: o desempenho sai so' da validacao cruzada.",
           "No external holdout: performance comes from cross-validation only."),
          ("A mesma validacao que escolhe o numero de variaveis latentes tambem mede o erro.",
           "The same validation that picks the number of latent variables also measures the error."),
          ("Separe uma fracao de unidades independentes que nao participe de nenhuma escolha (holdout_fracao).",
           "Hold out a fraction of independent units that takes no part in any choice (holdout_fracao)."),
          campo="holdout_fracao", valor=lambda c: 0.2, refs=("westad2015", "varma2006")),
    Regra("holdout_fora_da_faixa", "config", "aviso",
          lambda c: 0.0 < (_num(c.c("frac_holdout", 0.2)) or 0.0) < 0.15 or (_num(c.c("frac_holdout", 0.2)) or 0.0) > 0.35,
          ("Holdout de {p}% das unidades: fora da faixa usual de 15 a 35%.",
           "Holdout of {p}% of the units: outside the usual 15 to 35% range."),
          ("Abaixo de 15% o erro do holdout vem de poucas unidades e oscila muito; acima de 35% sobra pouco "
           "para calibrar e escolher o numero de variaveis latentes.",
           "Below 15% the holdout error comes from few units and swings widely; above 35% too little is left "
           "to calibrate and choose the number of latent variables."),
          ("Use um holdout entre 15 e 35% (0.2 e' o padrao).", "Use a holdout between 15 and 35% (0.2 is the default)."),
          valores=lambda c: dict(p=int(round((_num(c.c("frac_holdout", 0.2)) or 0.0) * 100))),
          campo="holdout_fracao", valor=lambda c: 0.2, refs=("westad2015",)),
    Regra("preproc_bruto_em_espectro", "config", "aviso",
          lambda c: _espectral(c) and str(c.c("default_preprocessing", "")) == "raw",
          ("Espectro sem pre-processamento (raw).", "Spectrum with no pre-processing (raw)."),
          ("Espalhamento e deslocamento de linha de base entram no modelo como se fossem quimica e custam "
           "variaveis latentes.",
           "Scatter and baseline offsets enter the model as if they were chemistry and cost latent variables."),
          ("Para FT-NIR difuso, comece por msc_sg_mc e compare com raw usando a mesma validacao.",
           "For diffuse FT-NIR, start with msc_sg_mc and compare it with raw under the same validation."),
          campo="pre_processamento", valor=lambda c: "msc_sg_mc", refs=("rinnan2009",)),
    Regra("max_lvs_alto", "config", "aviso",
          lambda c: int(_num(c.c("max_lvs", 0)) or 0) > 40,
          ("max_lvs = {m}: teto acima de 40 variaveis latentes.", "max_lvs = {m}: cap above 40 latent variables."),
          ("Com tantas variaveis latentes possiveis a selecao pode escolher um modelo que ajusta ruido "
           "(sobreajuste), e o tempo de execucao cresce.",
           "With so many possible latent variables the selection may pick a model that fits noise "
           "(overfitting), and run time grows."),
          ("Baixe max_lvs para 20 a 30 e confira se o numero escolhido nao bate no teto.",
           "Lower max_lvs to 20 to 30 and check that the chosen number does not hit the cap."),
          valores=lambda c: dict(m=int(_num(c.c("max_lvs", 0)) or 0)),
          campo="max_lvs", valor=lambda c: 25, refs=("westad2015",)),

    # ---------------------------------------------------------------- dados
    Regra("dados_com_nan", "dados", "critico",
          lambda c: bool(c.dados and c.dados.n_nan > 0),
          ("Os dados tem {n} valor(es) faltante(s) (NaN).", "The data has {n} missing value(s) (NaN)."),
          ("PLS e PCA nao aceitam celulas vazias; a execucao pararia no ajuste do modelo.",
           "PLS and PCA do not accept empty cells; the run would stop at model fitting."),
          ("Procure celulas vazias ou com texto no arquivo; se a falta for real, retire a amostra ou a variavel.",
           "Look for empty or text cells in the file; if the gap is real, drop the sample or the variable."),
          valores=lambda c: dict(n=c.dados.n_nan if c.dados else 0)),
    Regra("faixa_fora_do_eixo", "dados", "critico", _faixa_sem_dados,
          ("A faixa {lo:g}-{hi:g} nao cruza o eixo dos dados ({emin:g}-{emax:g}).",
           "The range {lo:g}-{hi:g} does not overlap the data axis ({emin:g}-{emax:g})."),
          ("Depois do corte nao sobra nenhuma variavel.", "No variable is left after the cut."),
          ("Ajuste faixa_min_cm e faixa_max_cm para dentro de {emin:g}-{emax:g}.",
           "Set faixa_min_cm and faixa_max_cm inside {emin:g}-{emax:g}."),
          valores=lambda c: dict(lo=_num(c.c("wn_min")) or 0, hi=_num(c.c("wn_max")) or 0,
                                 emin=c.dados.eixo_min if c.dados else 0, emax=c.dados.eixo_max if c.dados else 0)),
    Regra("faixas_excluir_fora_do_eixo", "dados", "critico",
          lambda c: bool(_excluidas_fora_do_eixo(c)),
          ("faixas_excluir {faixas} nao cruza o eixo dos dados ({emin:g} a {emax:g}).",
           "faixas_excluir {faixas} does not overlap the data axis ({emin:g} to {emax:g})."),
          ("A faixa esta numa unidade diferente da do eixo (ex.: Hz em vez de ppm), entao nada seria excluido e a regiao que voce queria tirar ficaria no modelo.",
           "The range is in a different unit from the axis (e.g. Hz instead of ppm), so nothing would be excluded and the region you wanted out would stay in the model."),
          ("Escreva as faixas na unidade do eixo ({emin:g} a {emax:g}) ou esvazie faixas_excluir.",
           "Write the ranges in the axis unit ({emin:g} to {emax:g}) or empty faixas_excluir."),
          valores=lambda c: dict(faixas=_excluidas_fora_do_eixo(c), emin=c.dados.eixo_min if c.dados else 0,
                                 emax=c.dados.eixo_max if c.dados else 0)),
    Regra("faixas_excluir_mais_da_metade", "dados", "aviso",
          lambda c: _fracao_excluida(c) > 0.5,
          ("faixas_excluir remove {p:.0f}% da faixa usada (mais de 50%).", "faixas_excluir removes {p:.0f}% of the range in use (over 50%)."),
          ("Sobra menos da metade do eixo; e' comum quando a faixa foi escrita numa unidade errada ou com a e b trocados de lugar.",
           "Less than half of the axis is left; this is common when a range was written in the wrong unit or with a and b swapped."),
          ("Confira cada faixa contra o espectro medio; se for proposital, siga.",
           "Check each range against the mean spectrum; if it is intended, go ahead."),
          valores=lambda c: dict(p=_fracao_excluida(c) * 100.0)),
    Regra("rmn_sem_faixas_excluir", "dados", "aviso",
          _parece_ppm_sem_exclusao,
          ("O eixo parece RMN 1H (em ppm) e faixas_excluir esta vazio.", "The axis looks like 1H NMR (in ppm) and faixas_excluir is empty."),
          ("A agua residual (~4.7 ppm) varia de uma amostra para outra e domina o divisor de SNV/PQN; o TSP/TMS (0 ppm) e' constante e nao carrega informacao. Heuristica: o programa nao sabe a unidade de um CSV.",
           "Residual water (~4.7 ppm) varies between samples and dominates the SNV/PQN divisor; TSP/TMS (0 ppm) is constant and carries no information. Heuristic: the program cannot know a CSV's unit."),
          ("Se o dado e' RMN 1H em ppm, use faixas_excluir: [4.5-5.0, -0.5-0.5] (a tela [2.4] preenche isso ao escolher RMN).",
           "If the data are 1H NMR in ppm, use faixas_excluir: [4.5-5.0, -0.5-0.5] (screen [2.4] fills this when you pick NMR)."),
          campo="faixas_excluir", valor=lambda c: "4.5-5.0, -0.5-0.5"),
    Regra("replicas_sem_validacao_agrupada", "dados", "critico",
          lambda c: bool(c.dados and c.dados.leituras_por_unidade_max > 1 and not c.c("group_by_mae_id", True)),
          ("Ha ate {k} leituras por amostra fisica e a validacao por grupo esta desligada.",
           "There are up to {k} readings per physical sample and group validation is off."),
          ("Replicas da mesma amostra caem em treino e teste: a metrica mede reconhecer a amostra, nao prever uma nova.",
           "Replicates of one sample fall in both training and test: the metric measures recognising the sample, "
           "not predicting a new one."),
          ("Ligue a validacao por grupo.", "Turn group validation on."),
          valores=lambda c: dict(k=c.dados.leituras_por_unidade_max if c.dados else 0),
          campo="validacao_group_aware", valor=lambda c: True, refs=("guo2017", "kapoor2023")),
    Regra("classe_com_uma_unidade", "dados", "critico",
          lambda c: bool(c.dados and _valida_modelo(c) and any(v < 2 for v in c.dados.unidades_por_classe.values())),
          ("Classe com uma so' unidade independente: {classes}.", "Class with a single independent unit: {classes}."),
          ("Uma classe com uma unidade nao pode estar ao mesmo tempo no treino e no teste.",
           "A class with one unit cannot be in training and test at the same time."),
          ("Meca mais amostras fisicas dessa classe ou retire-a da analise.",
           "Measure more physical samples of that class or remove it from the analysis."),
          valores=lambda c: dict(classes=", ".join(sorted(k for k, v in (c.dados.unidades_por_classe.items()
                                                                          if c.dados else []) if v < 2)))),
    Regra("dobras_maior_que_grupos", "dados", "info",
          lambda c: bool(c.dados and _valida_modelo(c) and c.c("group_by_mae_id", True) and c.dados.unidades_por_classe
                         and int(c.c("n_splits_cv", 5) or 5) > min(c.dados.unidades_por_classe.values()) >= 2),
          ("cv_dobras = {k}, mas a menor classe tem {m} unidade(s): a validacao vai usar {m} dobras.",
           "cv_dobras = {k}, but the smallest class has {m} unit(s): validation will use {m} folds."),
          ("Cada dobra precisa de ao menos uma unidade de cada classe; o pipeline reduz as dobras sozinho.",
           "Each fold needs at least one unit of each class; the pipeline reduces the folds itself."),
          ("Nada a corrigir; para mais dobras, meca mais unidades da menor classe.",
           "Nothing to fix; for more folds, measure more units of the smallest class."),
          valores=lambda c: dict(k=int(c.c("n_splits_cv", 5) or 5),
                                 m=min(c.dados.unidades_por_classe.values()) if c.dados and c.dados.unidades_por_classe else 0)),
    Regra("holdout_pequeno", "dados", "aviso",
          lambda c: bool(c.dados and 0 < _n_teste(c) < 25),
          ("O holdout tera cerca de {n} unidades: um acerto de 90% nele e' compativel com {lo:.0%} a {hi:.0%} (intervalo de Wilson, 95%).",
           "The holdout will have about {n} units: a 90% hit rate on it is compatible with {lo:.0%} to {hi:.0%} (Wilson interval, 95%)."),
          ("A incerteza de uma proporcao cai com a raiz do numero de unidades testadas.",
           "The uncertainty of a proportion falls with the square root of the number of tested units."),
          ("Reporte o intervalo junto com a metrica e nao compare modelos por diferencas menores que ele; "
           "validar um classificador costuma pedir 75 a 100 unidades de teste.",
           "Report the interval with the metric and do not compare models by differences smaller than it; "
           "validating a classifier usually needs 75 to 100 test units."),
          valores=lambda c: dict(n=_n_teste(c), lo=_wilson(0.9, _n_teste(c))[0], hi=_wilson(0.9, _n_teste(c))[1]),
          refs=("beleites2013", "brown2001")),
    Regra("classes_desbalanceadas", "dados", "aviso",
          lambda c: _razao_classes(c) >= 3.0,
          ("A maior classe tem {r:.1f} vezes mais unidades que a menor.",
           "The largest class has {r:.1f} times more units than the smallest."),
          ("Com classes desiguais a acuracia simples premia quem chuta a classe maior (limiar heuristico: 3x).",
           "With unequal classes plain accuracy rewards guessing the largest class (heuristic threshold: 3x)."),
          ("Leia a acuracia balanceada e o acerto por classe, nao a acuracia global.",
           "Read balanced accuracy and per-class hit rate, not overall accuracy."),
          valores=lambda c: dict(r=_razao_classes(c)), refs=("brereton2014",)),
    Regra("variaveis_constantes", "dados", "info",
          lambda c: bool(c.dados and c.dados.n_constantes > 0),
          ("{n} variavel(is) constante(s) em todas as amostras.", "{n} variable(s) constant across all samples."),
          ("Variavel sem variacao nao carrega informacao e pode quebrar a autoescala (divisao por zero).",
           "A variable with no variation carries no information and can break autoscaling (division by zero)."),
          ("Corte a faixa para fora dessas regioes, se forem bordas do espectro.",
           "Cut the range away from those regions if they are spectrum edges."),
          valores=lambda c: dict(n=c.dados.n_constantes if c.dados else 0)),

    # ---------------------------------------------------------------- resultado
    Regra("auditoria_critica", "resultado", "critico",
          lambda c: bool(_auditoria_critica(c)),
          ("A auditoria de delineamento marcou problema critico: {lista}.",
           "The design audit flagged a critical problem: {lista}."),
          ("Com problema critico de delineamento as metricas nao medem o que parecem medir.",
           "With a critical design problem the metrics do not measure what they seem to."),
          ("Resolva os itens da auditoria (bloco 'Auditoria de delineamento' do resumo_modelo.txt) antes de citar qualquer metrica.",
           "Resolve the audit items ('Auditoria de delineamento' block of resumo_modelo.txt) before citing any metric."),
          valores=lambda c: dict(lista="; ".join(_auditoria_critica(c))), refs=("kapoor2023",)),
    Regra("permutacao_nao_significativa", "resultado", "critico",
          lambda c: _eh_classificacao(c) and (_num(c.r("Permutation p-value")) or 0.0) > ALPHA_PADRAO,
          ("p-valor de permutacao = {p:.3f}: o modelo nao se distingue do acaso.",
           "Permutation p-value = {p:.3f}: the model is not distinguishable from chance."),
          ("Classes embaralhadas produzem desempenho igual ou melhor em mais de {alpha:.0%} das vezes.",
           "Shuffled classes give equal or better performance more than {alpha:.0%} of the time."),
          ("Nao reporte o modelo como discriminante; reveja delineamento, pre-processamento e numero de unidades.",
           "Do not report the model as discriminant; review design, pre-processing and number of units."),
          valores=lambda c: dict(p=_num(c.r("Permutation p-value")) or 0.0, alpha=ALPHA_PADRAO),
          refs=("westad2015",)),
    Regra("acuracia_perfeita", "resultado", "aviso",
          lambda c: (_eh_classificacao(c) and max(_num(c.r("Accuracy (CV)")) or 0.0, _num(c.r("Holdout accuracy")) or 0.0) >= 0.999
                     and (_num(c.r("N unidades independentes")) or 0) < 30),
          ("Acerto de 100% com {n:g} unidades independentes.", "100% hit rate with {n:g} independent units."),
          ("Acerto perfeito com poucas unidades costuma vir de vazamento (replicas ou lote em treino e teste, "
           "selecao feita fora da validacao) ou de classes separadas por sessao, nao pela quimica.",
           "A perfect hit rate with few units usually comes from leakage (replicates or batch in training and "
           "test, selection done outside validation) or classes separated by session, not chemistry."),
          ("Confira a ficha de vazamento e a auditoria de delineamento antes de citar o numero.",
           "Check the leakage sheet and the design audit before citing the number."),
          valores=lambda c: dict(n=_num(c.r("N unidades independentes")) or 0), refs=("kapoor2023", "guo2017")),
    Regra("r2_muito_acima_de_q2", "resultado", "aviso",
          lambda c: (_eh_classificacao(c) and _num(c.r("R2Y")) is not None and _num(c.r("Q2")) is not None
                     and (_num(c.r("R2Y")) or 0) - (_num(c.r("Q2")) or 0) > 0.3),
          ("R2Y = {r2:.2f} e Q2 = {q2:.2f}: diferenca de {d:.2f}.", "R2Y = {r2:.2f} and Q2 = {q2:.2f}: a gap of {d:.2f}."),
          ("O modelo descreve a calibracao muito melhor do que preve amostras fora dela (limiar heuristico: 0,3).",
           "The model describes calibration much better than it predicts outside it (heuristic threshold: 0.3)."),
          ("Reduza max_lvs, confirme a validacao por grupo e compare com o erro do holdout.",
           "Lower max_lvs, confirm group validation and compare with the holdout error."),
          valores=lambda c: dict(r2=_num(c.r("R2Y")) or 0, q2=_num(c.r("Q2")) or 0,
                                 d=(_num(c.r("R2Y")) or 0) - (_num(c.r("Q2")) or 0)),
          refs=("kjeldahl2010",)),
    Regra("cv_muito_acima_do_holdout", "resultado", "aviso",
          lambda c: (_eh_classificacao(c) and _num(c.r("Balanced accuracy")) is not None and _num(c.r("Holdout balanced acc")) is not None
                     and (_num(c.r("Balanced accuracy")) or 0) - (_num(c.r("Holdout balanced acc")) or 0) > 0.15),
          ("Acuracia balanceada {vc:.2f} na validacao cruzada e {ho:.2f} no holdout.",
           "Balanced accuracy {vc:.2f} in cross-validation and {ho:.2f} on the holdout."),
          ("A validacao cruzada sai otimista quando participa das escolhas do modelo (limiar heuristico: 0,15).",
           "Cross-validation comes out optimistic when it takes part in model choices (heuristic threshold: 0.15)."),
          ("Cite o holdout, com o intervalo; confira se o holdout tem poucas unidades.",
           "Cite the holdout, with its interval; check whether the holdout has few units."),
          valores=lambda c: dict(vc=_num(c.r("Balanced accuracy")) or 0, ho=_num(c.r("Holdout balanced acc")) or 0),
          refs=("varma2006", "westad2015")),
    Regra("lvs_no_teto", "resultado", "info",
          lambda c: str(c.r("LVs no teto (max_lvs)") or "").strip().lower() in ("true", "sim", "1"),
          ("O numero de variaveis latentes escolhido bateu no teto (max_lvs).",
           "The chosen number of latent variables hit the cap (max_lvs)."),
          ("A selecao queria mais componentes do que o teto permite.", "The selection wanted more components than the cap allows."),
          ("Antes de subir o teto, veja o grafico de selecao: muitas componentes com poucas unidades costuma ser sobreajuste.",
           "Before raising the cap, look at the selection plot: many components with few units is usually overfitting."),
          refs=("kjeldahl2010",)),
)


# ------------------------------------------------------------------ avaliacao
def avaliar(cfg: Any = None, dados: Optional[ResumoDados] = None, resultado: Optional[Dict[str, Any]] = None,
            lang: str = "PT", momentos: Optional[Iterable[str]] = None) -> List[Achado]:
    """Achados das regras que se aplicam, do mais grave ao menos grave. Sem `momentos`, avalia o que os
    dados permitem: config sempre (se ha cfg), dados se ha resumo, resultado se ha resultado."""
    L = 1 if str(lang).upper() == "EN" else 0
    if momentos is None:
        momentos = [m for m, ok in (("config", cfg is not None), ("dados", dados is not None),
                                    ("resultado", resultado is not None)) if ok]
    momentos = set(momentos)
    caso = Caso(cfg, dados, resultado)
    out: List[Achado] = []
    for r in REGRAS:
        if r.momento not in momentos:
            continue
        try:
            aplica = bool(r.quando(caso))
        except (TypeError, ValueError, AttributeError, ZeroDivisionError):
            aplica = False   # dado ausente ou de outro tipo: a regra nao se pronuncia
        if not aplica:
            continue
        v = r.valores(caso)
        out.append(Achado(r.id, r.momento, r.severidade, r.mensagem[L].format(**v), r.causa[L].format(**v),
                          r.fazer[L].format(**v), r.campo, r.valor(caso) if r.campo else None, r.refs))
    out.sort(key=lambda a: _ORDEM_SEV[a.severidade])
    return out


_ROTULO_SEV = {"critico": ("CRITICO", "CRITICAL"), "aviso": ("AVISO", "WARNING"), "info": ("INFO", "INFO")}


def formatar_achados(achados: Sequence[Achado], lang: str = "PT", refs: bool = True) -> str:
    L = 1 if str(lang).upper() == "EN" else 0
    if not achados:
        return "Nenhum problema encontrado pelas regras do assistente." if L == 0 else "No problem found by the assistant rules."
    rot = (("Por que", "O que fazer", "Aceitar", "Fonte"), ("Why", "What to do", "Accept", "Source"))[L]
    linhas: List[str] = []
    for a in achados:
        linhas.append(f"[{_ROTULO_SEV[a.severidade][L]}] {a.mensagem}")
        linhas.append(f"  {rot[0]}: {a.causa}")
        linhas.append(f"  {rot[1]}: {a.fazer}")
        if a.campo:
            linhas.append(f"  {rot[2]}: {a.campo}: {str(a.valor).lower() if isinstance(a.valor, bool) else a.valor}")
        if refs and a.refs:
            linhas.append(f"  {rot[3]}: " + "; ".join(citar(x) for x in a.refs))
    return "\n".join(linhas)


# ------------------------------------------------------------------ resultado
_LINHA = re.compile(r"^  (\S[^:]*?)\s*:\s(.*)$")
_CRITICO = re.compile(r"^\s+\[CRITICO\]\s+([^:]+):", re.M)


def resultado_de_resumo(texto: str) -> Dict[str, Any]:
    """Le o resumo_modelo.txt gravado pelo pipeline (linhas `  chave : valor`, 2 espacos) num dicionario.
    Numeros viram float; o resto fica texto. `_auditoria_critica` lista os nomes dos achados criticos."""
    out: Dict[str, Any] = {}
    for linha in (texto or "").splitlines():
        m = _LINHA.match(linha)
        if not m:
            continue
        chave, valor = m.group(1).strip(), m.group(2).strip()
        if chave in out:
            continue
        f = _num(valor.split()[0]) if valor else None
        out[chave] = f if (f is not None and len(valor.split()) == 1) else valor
    out["_auditoria_critica"] = [x.strip() for x in _CRITICO.findall(texto or "")]
    return out
