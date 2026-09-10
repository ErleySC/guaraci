# -*- coding: utf-8 -*-
"""auditoria_delineamento.py -- Auditoria de delineamento experimental
(Bloco 11, `guaraci audit`).

Consolida, num comando so', checagens que ate' aqui viviam espalhadas: em
`pipeline.validate_input` (duplicatas), em `cfg.grouping_guarantee`
(agrupamento), em scripts de auditoria PRIVADOS fora do repositorio
(confundimento classe x sessao -- `medir_confundimento_data.py`,
`medir_deriva_vs_quimica.py`), e numa funcao que existia mas nunca tinha
CHAMADOR nenhum (`perfil_matriz.MatrixProfile.outside_working_range`
-- achado desta revisao, registrado em `~/.guaraci_local/PROGRESSO.md`).

Cada checagem devolve um `AuditFinding` (nome curto, severidade,
mensagem) -- roda por padrao, mas e' SILENCIAVEL individualmente, com
JUSTIFICATIVA OBRIGATORIA (a checagem continua aparecendo no relatorio,
so' com severidade "silenciado" e a justificativa anexada -- nunca some
do relatorio so' porque foi silenciada, isso seria esconder, nao
documentar uma decisao).

FALTA (registrado, nao implementado nesta rodada -- ver PROGRESSO.md):
ordem de leitura correlacionada com o alvo (teor/classe). O metodo ja'
existe, validado, num script PRIVADO (`medir_ordem_leitura.py`,
`medir_deriva_vs_quimica.py`) que extrai o timestamp de aquisicao do
audit trail JCAMP-DX -- mas `dados_io.load_data`/`parse_dx` nao expoe
esse timestamp hoje, e portar isso exigiria mudanca no parser DX
(trabalho de engenharia real, nao so' wiring). Fica de fora deste
fechamento do Bloco 11 para nao inflar o escopo sem avisar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import numpy as np

from guaraci.conformal import n_minimum_for_alpha
from guaraci.dados_io import session_from_mae_id

__all__ = [
    "AuditFinding",
    "check_grouping",
    "check_class_session_confounding",
    "check_duplicates",
    "check_insufficient_n",
    "check_validation_use_range",
    "check_external_validation",
    "run_audit",
]

_SEVERIDADES_VALIDAS = {"ok", "aviso", "critico", "silenciado"}


@dataclass
class AuditFinding:
    nome: str
    severidade: str
    mensagem: str

    def __post_init__(self) -> None:
        if self.severidade not in _SEVERIDADES_VALIDAS:
            raise ValueError(
                f"severidade '{self.severidade}' invalida -- use uma de "
                f"{sorted(_SEVERIDADES_VALIDAS)}")


def check_grouping(cfg: "Any") -> AuditFinding:
    """Reaproveita `cfg.grouping_guarantee` (Bloco 8) -- ja' calculado
    pelo carregador de dados, so' reportado aqui."""
    nivel = getattr(cfg, "grouping_guarantee", "unknown")
    if nivel == "high":
        return AuditFinding("agrupamento", "ok",
                             "Garantia de agrupamento: high (mae_id "
                             "confiavel para toda amostra).")
    if nivel == "medium":
        return AuditFinding("agrupamento", "aviso",
                             "Garantia de agrupamento: medium (CSV de "
                             "associacao, nao estrutura de pasta -- "
                             "confira a fonte).")
    return AuditFinding(
        "agrupamento", "critico",
        f"Garantia de agrupamento: {nivel} -- validacao pode ter caido em "
        "StratifiedKFold sem protecao contra vazamento de replica. Trate "
        "metricas como exploratorias.")


def check_class_session_confounding(
        rotulos: np.ndarray, mae_id: Optional[np.ndarray]
        ) -> AuditFinding:
    """Uma classe confinada numa UNICA sessao de coleta (`session_from_
    mae_id`), enquanto o dataset tem multiplas sessoes no total, nao tem
    como separar efeito de classe de deriva instrumental/temporal daquela
    sessao -- mesmo confundimento medido em auditoria anterior (scripts
    privados `medir_confundimento_data.py`/`medir_deriva_vs_quimica.py`),
    aqui generalizado para qualquer classe/matriz, nao so' oleos."""
    if mae_id is None:
        return AuditFinding(
            "confundimento_classe_sessao", "aviso",
            "mae_id ausente -- nao e' possivel checar confundimento "
            "classe x sessao sem identificador de amostra fisica.")

    rotulos = np.asarray(rotulos, dtype=str)
    sessao = np.array([session_from_mae_id(m) for m in mae_id], dtype=str)
    sessoes_totais = len(set(sessao))
    if sessoes_totais <= 1:
        return AuditFinding(
            "confundimento_classe_sessao", "critico",
            "Todo o dataset veio de 1 UNICA sessao de coleta -- nenhuma "
            "classe pode ser distinguida de deriva instrumental/temporal. "
            "Isso nao e' um problema por classe, e' estrutural do dataset "
            "inteiro.")

    classes_confinadas = []
    for classe in sorted(set(rotulos)):
        sessoes_da_classe = {s for s, r in zip(sessao, rotulos) if r == classe}
        if len(sessoes_da_classe) == 1:
            classes_confinadas.append(classe)

    if not classes_confinadas:
        return AuditFinding(
            "confundimento_classe_sessao", "ok",
            f"Nenhuma classe confinada a 1 unica sessao "
            f"({sessoes_totais} sessoes no total).")
    return AuditFinding(
        "confundimento_classe_sessao", "critico",
        f"{len(classes_confinadas)} classe(s) confinada(s) a 1 unica "
        f"sessao de coleta (de {sessoes_totais} sessoes no total): "
        f"{classes_confinadas}. Deriva instrumental daquela sessao fica "
        "indistinguivel de efeito de classe para essas classes.")


def check_duplicates(X: np.ndarray, wavenumbers: np.ndarray,
                      rotulos: np.ndarray,
                      conc: Optional[np.ndarray] = None,
                      mae_id: Optional[np.ndarray] = None
                      ) -> AuditFinding:
    """Reaproveita `pipeline.validate_input` (ja' detecta duplicatas
    exatas e aproximadas) -- roda de novo aqui so' para REPORTAR, nao
    para limpar (o pipeline de treino ja' faz a limpeza; a auditoria e'
    so' visibilidade)."""
    from guaraci.pipeline import validate_input

    _X, _wn, _rot, _conc, _mae, relatorio = validate_input(
        X, wavenumbers, rotulos, conc, mae_id)
    # relatorio: Dict[str, object] (pipeline.validate_input, fora do escopo
    # do mypy) -- cast em vez de int() direto: object nao e' SupportsInt
    # para o overload resolver do mypy, mesmo sendo sempre um int em tempo
    # de execucao (chave sempre populada com contagem inteira).
    n_exatas = int(cast(int, relatorio.get("n_duplicatas_exatas", 0)))
    n_aprox = int(cast(int, relatorio.get("n_duplicatas_aproximadas", 0)))
    if n_exatas == 0 and n_aprox == 0:
        return AuditFinding("duplicatas", "ok",
                             "Nenhuma duplicata exata ou aproximada detectada.")
    severidade = "critico" if n_exatas > 0 else "aviso"
    return AuditFinding(
        "duplicatas", severidade,
        f"{n_exatas} duplicata(s) EXATA(s), {n_aprox} amostra(s) com alta "
        "correlacao (possiveis replicas nao identificadas como tal) -- "
        "risco de vazamento treino/teste se cairem em folds diferentes.")


def check_insufficient_n(rotulos: np.ndarray,
                          mae_id: Optional[np.ndarray],
                          alpha_conformal_referencia: float = 0.05
                          ) -> AuditFinding:
    """Quantas SESSOES independentes (nao amostras/espectros) cada classe
    tem -- e' o `n` que sustenta (ou nao) uma garantia estatistica real,
    ver `conformal.py`/`plano_amostral.py`."""
    if mae_id is None:
        return AuditFinding(
            "n_insuficiente", "aviso",
            "mae_id ausente -- nao e' possivel contar sessoes "
            "independentes por classe.")

    rotulos = np.asarray(rotulos, dtype=str)
    sessao = np.array([session_from_mae_id(m) for m in mae_id], dtype=str)
    n_minimum_conformal = n_minimum_for_alpha(alpha_conformal_referencia)

    classes_fracas = []
    for classe in sorted(set(rotulos)):
        n_sessoes_classe = len({s for s, r in zip(sessao, rotulos)
                                 if r == classe})
        if n_sessoes_classe < n_minimum_conformal:
            classes_fracas.append((classe, n_sessoes_classe))

    if not classes_fracas:
        return AuditFinding(
            "n_insuficiente", "ok",
            f"Toda classe tem >= {n_minimum_conformal} sessoes "
            f"independentes (minimo p/ alpha={alpha_conformal_referencia} "
            "conformal).")
    detalhe = ", ".join(f"{c} (n={n})" for c, n in classes_fracas)
    return AuditFinding(
        "n_insuficiente", "aviso",
        f"{len(classes_fracas)} classe(s) com menos de "
        f"{n_minimum_conformal} sessoes independentes (minimo p/ "
        f"alpha={alpha_conformal_referencia} conformal): {detalhe}. Gates "
        "estatisticos para essas classes ficam nao-validados (ver "
        "identificacao.py/plano_amostral.py) -- nao e' erro, e' o regime "
        "honesto do dataset atual.")


def check_validation_use_range(conc: Optional[np.ndarray], cfg: "Any"
                                ) -> AuditFinding:
    """`perfil_matriz.MatrixProfile.outside_working_range` existia
    sem NENHUM chamador em producao (achado desta revisao) -- finalmente
    usado aqui: compara a faixa de teor OBSERVADA na calibracao contra a
    faixa de trabalho DECLARADA no perfil da matriz."""
    from guaraci.perfil_matriz import cfg_profile

    if conc is None:
        return AuditFinding(
            "faixa_validacao_uso", "aviso",
            "Sem dados de concentracao -- nao e' possivel comparar faixa "
            "de calibracao com faixa de trabalho declarada.")
    conc_v = np.asarray(conc, dtype=float)
    conc_v = conc_v[~np.isnan(conc_v)]
    if conc_v.size == 0:
        return AuditFinding(
            "faixa_validacao_uso", "aviso",
            "Nenhum valor de concentracao valido para comparar.")

    perfil = cfg_profile(cfg)
    if not perfil.faixa_trabalho:
        return AuditFinding(
            "faixa_validacao_uso", "aviso",
            "Perfil de matriz nao declara faixa de trabalho -- nada a "
            "comparar (ausencia de declaracao nao e' garantia de que a "
            "faixa calibrada cobre o uso pretendido).")

    lo_cal, hi_cal = float(conc_v.min()), float(conc_v.max())
    fora_lo = perfil.outside_working_range(lo_cal)
    fora_hi = perfil.outside_working_range(hi_cal)
    if not fora_lo and not fora_hi:
        return AuditFinding(
            "faixa_validacao_uso", "ok",
            f"Faixa calibrada [{lo_cal:.2f}, {hi_cal:.2f}] dentro da "
            f"faixa de trabalho declarada {perfil.faixa_trabalho}.")
    return AuditFinding(
        "faixa_validacao_uso", "aviso",
        f"Faixa calibrada [{lo_cal:.2f}, {hi_cal:.2f}] NAO cobre "
        f"totalmente a faixa de trabalho declarada {perfil.faixa_trabalho} "
        "-- predicoes em amostras dentro da faixa de trabalho mas fora da "
        "faixa calibrada seriam extrapolacao.")


def check_external_validation() -> AuditFinding:
    """Informativo, nao derivado dos dados desta execucao -- ver
    docs/MANUAL.md secao 9: benchmark PLS/pre-processamento contra
    dataset publico (Tecator) existe; classificacao/DD-SIMCA/OPLS-DA
    ainda nao tem benchmark externo."""
    return AuditFinding(
        "validacao_externa", "aviso",
        "Benchmark contra dataset publico: PLS-R/pre-processamento "
        "cobertos (Tecator, ver docs/BENCHMARK_TECATOR.md). "
        "Classificacao/DD-SIMCA/OPLS-DA AINDA sem benchmark externo -- "
        "nao valide conclusoes de metodo sem esse benchmark.")


def run_audit(X: np.ndarray, wavenumbers: np.ndarray,
              rotulos: np.ndarray, cfg: "Any",
              conc: Optional[np.ndarray] = None,
              mae_id: Optional[np.ndarray] = None,
              *, silenciar: Optional[Dict[str, str]] = None
              ) -> List[AuditFinding]:
    """Roda todas as checagens (D5-like: sempre por padrao, nunca opt-in).

    `silenciar`: {nome_da_checagem: justificativa}. A checagem AINDA
    aparece no relatorio, com severidade "silenciado" e a justificativa
    anexada -- silenciar nunca faz uma checagem desaparecer do relatorio
    sem deixar rastro (justificativa vazia/None e' rejeitada).
    """
    silenciar = silenciar or {}
    for nome, justificativa in silenciar.items():
        if not justificativa or not justificativa.strip():
            raise ValueError(
                f"silenciar['{nome}'] precisa de justificativa nao-vazia "
                "-- checagem nunca some do relatorio sem motivo registrado.")

    achados = [
        check_grouping(cfg),
        check_class_session_confounding(rotulos, mae_id),
        check_duplicates(X, wavenumbers, rotulos, conc, mae_id),
        check_insufficient_n(rotulos, mae_id),
        check_validation_use_range(conc, cfg),
        check_external_validation(),
    ]

    resultado: List[AuditFinding] = []
    for achado in achados:
        if achado.nome in silenciar:
            resultado.append(AuditFinding(
                nome=achado.nome, severidade="silenciado",
                mensagem=f"{achado.mensagem} [SILENCIADO: "
                         f"{silenciar[achado.nome]}]"))
        else:
            resultado.append(achado)
    return resultado
