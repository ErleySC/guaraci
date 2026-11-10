# -*- coding: utf-8 -*-
"""auditoria_delineamento.py -- Auditoria de delineamento experimental
(Bloco 11, `guaraci audit`).

Consolida, num comando so', checagens que ate' aqui viviam espalhadas: em
`pipeline.validate_input` (duplicatas), em `cfg.grouping_guarantee`
(agrupamento), em medicoes avulsas de confundimento classe x sessao e de
deriva x quimica (feitas fora do pacote, sem script publicado), e numa
funcao que existia mas nunca tinha CHAMADOR nenhum
(`perfil_matriz.MatrixProfile.outside_working_range`).

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
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from guaraci.conformal import n_minimum_for_alpha
from guaraci.dados_io import session_from_mae_id

__all__ = [
    "AuditFinding",
    "check_grouping",
    "check_class_session_confounding",
    "check_unidade_independente",
    "check_classe_unica",
    "check_grupo_em_varias_classes",
    "check_duplicates",
    "check_insufficient_n",
    "check_validation_use_range",
    "check_external_validation",
    "resolver_unidade_agrupamento",
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
        "metricas como exploratorias. Para proteger as replicas: informe "
        "arquivo_grupos (CSV arquivo,grupo), coluna_grupo (CSV) ou organize "
        "uma subpasta por amostra fisica.")


def check_grupo_em_varias_classes(rotulos: np.ndarray,
                                  mae_id: Optional[np.ndarray]) -> AuditFinding:
    """R2-17 (revisao cega 2026-09-26): um grupo (amostra fisica) com espectros
    de classes diferentes e' incoerente -- ou o rotulo ou o grupo esta errado --
    e rodava sem aviso. Replicas de uma amostra sao da MESMA classe."""
    if mae_id is None:
        return AuditFinding("grupo_em_varias_classes", "ok",
                            "Sem chave de grupo: nada a checar.")
    rot = np.asarray(rotulos, dtype=str)
    mae = np.asarray(mae_id, dtype=str)
    mistos = sorted(g for g in set(mae) if len(set(rot[mae == g])) > 1)
    if not mistos:
        return AuditFinding("grupo_em_varias_classes", "ok",
                            "Cada grupo tem espectros de uma unica classe.")
    return AuditFinding(
        "grupo_em_varias_classes", "aviso",
        f"{len(mistos)} grupo(s) com espectros de MAIS de uma classe "
        f"({', '.join(mistos[:5])}{' ...' if len(mistos) > 5 else ''}). Replicas da "
        "mesma amostra deveriam ter a mesma classe: confira o rotulo ou o grupo.")


def check_class_session_confounding(
        rotulos: np.ndarray, mae_id: Optional[np.ndarray]
        ) -> AuditFinding:
    """Uma classe confinada numa UNICA sessao de coleta (`session_from_
    mae_id`), enquanto o dataset tem multiplas sessoes no total, nao tem
    como separar efeito de classe de deriva instrumental/temporal daquela
    sessao -- mesmo confundimento medido em auditoria anterior (scripts
    privados `medir_confundimento_data.py`/`medir_deriva_vs_quimica.py`),
    aqui generalizado para qualquer classe/matriz."""
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
        f"{', '.join(map(str, classes_confinadas))}. Deriva instrumental daquela sessao fica "
        "indistinguivel de efeito de classe para essas classes.")


def check_unidade_independente(rotulos: np.ndarray,
                               mae_id: Optional[np.ndarray],
                               unidade: str = "mae_id",
                               garantia: str = "high") -> AuditFinding:
    """B4c do parecer de 2026-09-25: o `mae_id` e' a unidade independente da
    validacao? Quando o `mae_id` e' MAIS FINO que a sessao de coleta (cada
    nivel de teor/diluicao tem o seu `mae_id`, mas todos vem do mesmo
    amostra-base), a CV por `mae_id` poe diluicoes da mesma sessao em folds
    diferentes -- e o PLS-DA aprende a SESSAO. Medido com teste nulo (rotulo
    sorteado por sessao, sem sinal de classe): bal.acc 0,90-0,98 agrupando
    por `mae_id`, contra 0,22-0,35 (acaso = 0,25) agrupando por sessao.

    - `unidade="mae_id"` e mae_id mais fino que a sessao: "aviso"; "critico"
      quando cada sessao contem UMA classe so' (a classe e' determinada pela
      sessao: o modelo pode estar distinguindo a sessao, nao a classe).
    - `unidade="sessao"`: "ok" (ou aviso se sobram <3 unidades).
    """
    if mae_id is None:
        return AuditFinding(
            "unidade_independente", "aviso",
            "mae_id ausente -- sem chave de grupo nao ha' unidade "
            "independente declarada (ver garantia de agrupamento).")
    rotulos = np.asarray(rotulos, dtype=str)
    mae = np.asarray(mae_id, dtype=str)
    sessao = np.array([session_from_mae_id(m) for m in mae], dtype=str)
    n_mae, n_sess = len(set(mae)), len(set(sessao))
    if unidade == "sessao":
        if n_sess < 3:
            return AuditFinding(
                "unidade_independente", "aviso",
                f"Agrupando por sessao restam so' {n_sess} unidade(s): a "
                "validacao agrupada fica degenerada (precisa de >=3).")
        return AuditFinding(
            "unidade_independente", "ok",
            f"Validacao agrupada por SESSAO de coleta ({n_sess} unidades "
            f"independentes; {n_mae} mae_id distintos).")
    if garantia == "medium" and all(s == m for s, m in zip(sessao, mae)):
        # R3-01 (revisao cega): grupo DECLARADO pelo usuario cujo id nao segue o
        # padrao de nome da sessao -- o software nao tem como saber se replicas/
        # diluicoes da mesma sessao receberam ids diferentes. Antes: "[OK] coincidem".
        return AuditFinding(
            "unidade_independente", "aviso",
            f"NAO VERIFICAVEL: {n_mae} grupos declarados por voce, mas os ids nao "
            "seguem o padrao de sessao, entao nao da' para checar se replicas ou "
            "diluicoes da MESMA sessao/lote receberam grupos diferentes. Se receberam, "
            "a validacao vaza por sessao e o desempenho e' OTIMISTA (teste nulo: "
            "balanced accuracy 0,72 CV / 0,82 holdout com acaso 0,33). Agrupe pelo "
            "nivel mais grosso que e' de fato independente (sessao/lote/dia).")
    if n_sess >= n_mae:
        return AuditFinding(
            "unidade_independente", "ok",
            f"mae_id e sessao coincidem ({n_mae} unidades): agrupar por "
            "mae_id equivale a agrupar por sessao.")
    aninhada = all(
        len({r for r, s2 in zip(rotulos, sessao) if s2 == sess}) == 1
        for sess in set(sessao))
    base = (f"O mae_id ({n_mae} valores) e' mais fino que a sessao de coleta "
            f"({n_sess} sessoes): niveis de teor/diluicoes da MESMA sessao "
            "podem cair em folds diferentes. ")
    if aninhada:
        return AuditFinding(
            "unidade_independente", "critico",
            base + "Alem disso cada sessao tem UMA classe so': o modelo pode "
            "estar distinguindo a SESSAO, nao a classe (medido em teste nulo: "
            "bal.acc 0,90-0,98 contra acaso 0,25). Use "
            "unidade_agrupamento=sessao e trate os numeros por mae_id como "
            "otimistas.")
    return AuditFinding(
        "unidade_independente", "aviso",
        base + "Se a sessao e' a unidade independente do seu delineamento, "
        "use unidade_agrupamento=sessao.")


def check_classe_unica(rotulos: np.ndarray, cfg: "Any") -> AuditFinding:
    """M-19(d): um dataset de UMA classe "concluia" com balanced accuracy 1,0 e
    o aviso ficava so' no log. Em classificacao/autenticacao (N1/N2) isso e'
    critico -- as metricas de classificacao sao degeneradas; em quantificacao
    (N3, p.ex. um unico material) e' um delineamento legitimo."""
    n = len(set(np.asarray(rotulos, dtype=str).tolist()))
    if n >= 2:
        return AuditFinding("classe_unica", "ok", f"{n} classes presentes.")
    if str(getattr(cfg, "level", "N1")) == "N3":
        return AuditFinding(
            "classe_unica", "ok",
            "Uma unica classe (quantificacao de um so' material): as metricas "
            "de classificacao nao se aplicam; valem as de regressao.")
    return AuditFinding(
        "classe_unica", "critico",
        "Apenas UMA classe presente: accuracy, balanced accuracy, kappa e F1 "
        "sao degeneradas (1,0 por construcao) e NAO medem desempenho. Nao cite "
        "esses numeros; use outro nivel/objetivo ou traga mais classes.")


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
        "estatisticos para essas classes ficam nao-validados (sao necessarias mais "
        "sessoes de coleta independentes) -- nao e' erro, e' o regime "
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
            "Perfil de tecnica nao declara faixa de trabalho -- nada a "
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
    """Informativo, nao derivado dos dados desta execucao.

    Os padroes do motor sao ponto de partida, nao universais; nao validam o estudo do usuario."""
    return AuditFinding(
        "validacao_externa", "aviso",
        "Os padroes do motor sao ponto de partida, nao universais, e nao validam "
        "o seu estudo: confirme as conclusoes com validacao externa propria (amostras "
        "independentes).")


def resolver_unidade_agrupamento(rotulos: np.ndarray,
                                 mae_id: Optional[np.ndarray],
                                 unidade: str = "auto",
                                 garantia: str = "high") -> Tuple[str, str]:
    """Resolve `unidade_agrupamento="auto"` em "mae_id" ou "sessao".

    Devolve `(unidade_efetiva, motivo)`; `motivo` e' "" quando nada precisa ser
    dito. Qualquer valor diferente de "auto" passa direto (a escolha explicita
    do usuario nunca e' trocada).

    "auto" so' vira "sessao" quando e' o unico jeito honesto de validar: o
    `mae_id` e' mais fino que a sessao E cada sessao tem UMA classe so' (o
    caso "critico" de `check_unidade_independente`, em que o modelo pode estar
    aprendendo a sessao) E validar por sessao e' possivel (>= 3 sessoes e toda
    classe em >= 2 sessoes). Se a classe tem uma unica sessao, deixar a
    sessao inteira de fora da validacao tira a classe inteira do treino: nesse
    caso fica "mae_id" e o motivo diz o que falta coletar.
    """
    if unidade != "auto":
        return unidade, ""
    if mae_id is None or garantia != "high":
        return "mae_id", ""
    rotulos = np.asarray(rotulos, dtype=str)
    mae = np.asarray(mae_id, dtype=str)
    sessao = np.array([session_from_mae_id(m) for m in mae], dtype=str)
    if len(set(sessao)) >= len(set(mae)):
        return "mae_id", ""
    aninhada = all(len({r for r, s2 in zip(rotulos, sessao) if s2 == sess}) == 1
                   for sess in set(sessao))
    if not aninhada:
        return "mae_id", ""
    sessoes_da_classe = {c: {s for s, r in zip(sessao, rotulos) if r == c}
                         for c in sorted(set(rotulos))}
    de_uma = [c for c, ss in sessoes_da_classe.items() if len(ss) < 2]
    if len(set(sessao)) < 3 or de_uma:
        quais = f" ({', '.join(de_uma)})" if de_uma else ""
        return "mae_id", (
            "cada sessao de coleta tem uma classe so', o que pede validar por "
            "sessao, mas isso e' impossivel agora: ha' classe(s) com uma unica "
            f"sessao{quais} ou menos de 3 sessoes no total. Os numeros por "
            "mae_id sao OTIMISTAS; colete cada classe em mais sessoes.")
    return "sessao", (
        "cada sessao de coleta tem uma classe so': validando por SESSAO "
        "(unidade_agrupamento=auto escolheu sessao).")


def run_audit(X: np.ndarray, wavenumbers: np.ndarray,
              rotulos: np.ndarray, cfg: "Any",
              conc: Optional[np.ndarray] = None,
              mae_id: Optional[np.ndarray] = None,
              *, silenciar: Optional[Dict[str, str]] = None,
              unidade: Optional[str] = None
              ) -> List[AuditFinding]:
    """Roda todas as checagens (D5-like: sempre por padrao, nunca opt-in).

    `silenciar`: {nome_da_checagem: justificativa}. A checagem AINDA
    aparece no relatorio, com severidade "silenciado" e a justificativa
    anexada -- silenciar nunca faz uma checagem desaparecer do relatorio
    sem deixar rastro (justificativa vazia/None e' rejeitada).
    """
    silenciar = silenciar or {}
    if unidade is None:
        unidade = str(getattr(cfg, "group_unit", "mae_id") or "mae_id")
    if unidade == "auto":      # sem o pipeline para resolver: o que valeria sem decidir nada
        unidade = resolver_unidade_agrupamento(
            rotulos, mae_id, "auto", str(getattr(cfg, "grouping_guarantee", "high")))[0]
    for nome, justificativa in silenciar.items():
        if not justificativa or not justificativa.strip():
            raise ValueError(
                f"silenciar['{nome}'] precisa de justificativa nao-vazia "
                "-- checagem nunca some do relatorio sem motivo registrado.")

    achados = [
        check_grouping(cfg),
        check_class_session_confounding(rotulos, mae_id),
        check_grupo_em_varias_classes(rotulos, mae_id),
        check_classe_unica(rotulos, cfg),
        check_unidade_independente(
            rotulos, mae_id, unidade,
            str(getattr(cfg, "grouping_guarantee", "high"))),
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
