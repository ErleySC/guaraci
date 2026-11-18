# -*- coding: utf-8 -*-
"""idioma_console.py -- o console da execucao no idioma escolhido.

O nucleo (pipeline, classificadores, conformal...) grava as mensagens de console em portugues, e o
painel do CLI e do app web le esse texto por regex (`app_logic`: `[AVISO]`, `[N/7]`, `-> caminho`,
`fold externo K/N concluido`). Reescrever cada mensagem no nucleo mexeria no contrato desse
parsing e no nucleo congelado; aqui a TRADUCAO acontece na saida: o handler de console de
`guaraci.log` e o `print` das etapas passam cada linha por `traduzir_linha`.

Regras:
- so' traduz quando o idioma da sessao e' ingles (`idioma_relatorio.idioma()`);
- as ETIQUETAS que o painel le ficam intactas (`[AVISO]`, `[N/7]`, `->`, a linha do fold externo);
- frase sem par na tabela sai como veio (o teste de vazamento acusa a lacuna em vez de esconder);
- a auditoria de delineamento reaproveita a tabela de `idioma_relatorio.achado`.
ASCII por convencao do projeto (as frases do nucleo vem sem acento, salvo as listadas).
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Pattern, Tuple

from guaraci import idioma_relatorio as _ir

__all__ = ["traduzir_linha", "traduzir_texto", "FRASES_CONSOLE", "saida_traduzida"]

#: (padrao PT da linha SEM a indentacao, modelo EN com os grupos). Primeiro que casa vence.
FRASES_CONSOLE: List[Tuple[str, str]] = [
    (r"\[MODO\] Figuras pertinentes a este objetivo \((\d+)\): (.*)",
     r"[MODE] Figures relevant to this objective (\1): \2"),
    (r"\[MODO\] Objetivo cientifico: (.*?)\s+\(nivel (.*?), objetivo configurado: (.*?)\)",
     r"[MODE] Scientific objective: \1  (level \2, configured objective: \3)"),
    (r"\[INFO\] Perfil de tecnica: (\S+) \(Matriz nao especificada -- nenhum padrao de matriz aplicado\) \| "
     r"eixo \[(.*?)\] \(unidade nao declarada\) \| pre-proc (.*)",
     r"[INFO] Technique profile: \1 (Matrix not specified -- no matrix default applied) | axis [\2] "
     r"(unit not declared) | preproc \3"),
    (r"\[INFO\] Perfil de tecnica: (.*?) \| eixo \[(.*?)\] (.*?) \| pre-proc (.*)",
     r"[INFO] Technique profile: \1 | axis [\2] \3 | preproc \4"),
    (r"\[INFO\] Synthetic MODE — cenario '(.*?)': (\d+) espectros, (\d+) amostras fisicas, (\d+) lotes; "
     r"variancia dentro das amostras \(replicas\) = (.*?) do total\.",
     r"[INFO] Synthetic MODE — scenario '\1': \2 spectra, \3 physical samples, \4 batches; within-sample "
     r"(replicate) variance = \5 of the total."),
    (r"\[INFO\] N2 \(autenticação por espécie\): rótulos de espécie preservados para DD-SIMCA one-class "
     r"\(puros=(\d+) \| adulterados=(\d+)\)\. DD-SIMCA forçado para mode 'puros'\.",
     r"[INFO] N2 (authentication by species): species labels kept for one-class DD-SIMCA (pure=\1 | "
     r"adulterated=\2). DD-SIMCA forced to mode 'puros'."),
    (r"\[INFO\] Truncamento espectral \[(.*?)\]: (\d+)/(\d+) variaveis removidas, restam (\d+) \((.*?)\)\.",
     r"[INFO] Spectral truncation [\1]: \2/\3 variables removed, \4 left (\5)."),
    (r"\[INFO\] Unidade de agrupamento '(.*?)': cada sessao de coleta tem uma classe so', o que pede validar "
     r"por sessao, mas isso e' impossivel agora: ha' classe\(s\) com uma unica sessao \((.*?)\) ou menos de 3 "
     r"sessoes no total\. Os numeros por (.*?) sao OTIMISTAS; colete cada classe em mais sessoes\.",
     r"[INFO] Grouping unit '\1': each collection session holds a single class, which calls for validating by "
     r"session, but that is impossible now: there are classes with a single session (\2) or fewer than 3 "
     r"sessions in total. The numbers by \3 are OPTIMISTIC; collect each class in more sessions."),
    (r"\[INFO\] Unidade de agrupamento '(.*?)': (.*)", r"[INFO] Grouping unit '\1': \2"),
    (r"\[INFO\] Auditoria de delineamento: (\d+) critico\(s\), (\d+) aviso\(s\) -- ver model card para o "
     r"detalhe por checagem\.",
     r"[INFO] Design audit: \1 critical, \2 warning(s) -- see the model card for the detail of each check."),
    (r"\[INFO\] Validacao group-aware ATIVA: (\d+) grupos unicos via (.*?)\.",
     r"[INFO] Group-aware validation ON: \1 unique groups via \2."),
    (r"\[INFO\] Hold-out \(StratifiedGroup \(replicas juntas, por classe\)(.*?)\): (\d+) amostras reservadas "
     r"\(frac=(.*?)\)\. Pipeline rodara em (\d+) amostras\.",
     r"[INFO] Hold-out (StratifiedGroup (replicates together, by class)\1): \2 samples held out (frac=\3). "
     r"The pipeline will run on \4 samples."),
    (r"\[INFO\] Hold-out \((.*?)\): (\d+) amostras reservadas \(frac=(.*?)\)\. Pipeline rodara em (\d+) amostras\.",
     r"[INFO] Hold-out (\1): \2 samples held out (frac=\3). The pipeline will run on \4 samples."),
    (r"\[INFO\] Amostras\s*: (.*)", r"[INFO] Samples  : \1"),
    (r"\[INFO\] Variaveis\s*: (.*)", r"[INFO] Variables: \1"),
    (r"\[INFO\] mae_id\s*: sim \((\d+) grupos\)", r"[INFO] mae_id   : yes (\1 groups)"),
    (r"\[INFO\] mae_id\s*: nao(.*)", r"[INFO] mae_id   : no\1"),
    (r"LVs otimas \(Wold parcimonia\): (\d+) \(min-RMSECV em (\d+) LVs, delta <(.*?)\)",
     r"Optimal LVs (Wold parsimony): \1 (min-RMSECV at \2 LVs, delta <\3)"),
    (r"\[INFO\] CV aninhada ativada \(selecao_lv_cv_aninhada=True, default a partir da v1\.0 -- corrige o vies "
     r"de reusar o mesmo dado p/ escolher n_opt e avaliar a metrica reportada\)\. Pode levar ~4-5x mais tempo "
     r"nesta etapa; desative no config\.yaml \(selecao_lv_cv_aninhada: false\) p/ iteracao rapida, aceitando a "
     r"metrica otimista\.",
     r"[INFO] Nested CV on (selecao_lv_cv_aninhada=True, default since v1.0 -- removes the bias of reusing the "
     r"same data to choose n_opt and to score the reported metric). This step may take ~4-5x longer; turn it off "
     r"in config.yaml (selecao_lv_cv_aninhada: false) for quick iteration, accepting the optimistic metric."),
    (r"Balanced accuracy\s+NAIVE \(n_opt=(\d+) fixo, reportada ate' v1\.0\)\s*= (.*)",
     r"Balanced accuracy  NAIVE (fixed n_opt=\1, reported up to v1.0)        = \2"),
    (r"Balanced accuracy\s+ANINHADA \(n_opt por fold, honesta, reportada a partir da v1\.0\) = (.*)",
     r"Balanced accuracy  NESTED (n_opt per fold, honest, reported since v1.0) = \1"),
    (r"Bal\.Acc observada = (.*?)\s+\|\s+p = (.*?)\s+\|\s+bal\.acc media H0 = (.*)",
     r"Observed Bal.Acc = \1  |  p = \2  |  mean bal.acc under H0 = \3"),
    (r"\[5/7\] Metricas finais \(CV\):", r"[5/7] Final metrics (CV):"),
    (r"\[6/7\] Gerando figuras\.\.\.", r"[6/7] Generating figures..."),
    (r"\[6c/7\] Avaliacao em holdout \((\d+) amostras\)\.\.\.", r"[6c/7] Hold-out evaluation (\1 samples)..."),
    (r"\[DD-SIMCA\] \((.*?), treino='(.*?)'\) — (\d+) amostras de treino",
     r"[DD-SIMCA] (\1, training='\2') — \3 training samples"),
    (r"\[DDSimca\] Classe '(.*?)': (\d+) amostra\(s\) de treino atipica\(s\) \(indices (.*?) de (\d+), deteccao "
     r"robusta mediana/MAD\)\. Nao removidas automaticamente -- considere investigar essas replicas\.",
     r"[DDSimca] Class '\1': \2 atypical training sample(s) (indices \3 of \4, robust median/MAD detection). "
     r"Not removed automatically -- consider inspecting these replicates."),
    (r"\[DDSimca\] Classe '(.*?)': limiar calibrado com apenas (\d+) amostra\(s\) fisica\(s\) independente\(s\) "
     r"\((.*?)\)\. (Nh=.*?, Nq=.*?) -- com tao poucos grupos a taxa de falsa rejeicao de amostras puras novas e' "
     r"desconhecida e pode ser muito alta; nao tome a regiao de aceitacao como margem de seguranca\.",
     r"[DDSimca] Class '\1': threshold calibrated with only \2 independent physical sample(s) (\3). \4 -- with "
     r"so few groups the false-rejection rate of new pure samples is unknown and may be very high; do not take "
     r"the acceptance region as a safety margin."),
    (r"Classe(\s+)sens\(LOGO\)(\s+)grupos(\s+)esp\(adult\)", r"Class\1sens(LOGO)\2groups\3spec(adult)"),
    (r"\[AVISO\] (.*?): Sensibilidade estimada por LOGO com apenas (\d+) grupos de replica\. Incerteza alta; IC "
     r"bootstrap nao e confiavel neste regime\. Interpretar como exploratoria\.",
     r"[AVISO] \1: Sensitivity estimated by LOGO with only \2 replicate groups. High uncertainty; the bootstrap CI "
     r"is not reliable in this regime. Read it as exploratory."),
    (r"\[AVISO\] alpha=(.*?) NAO e' alcancavel com (\d+) (amostra\(s\) de calibracao|grupo\(s\) de calibracao) "
     r"independentes: exigiria k=(\d+) escores ordenados, mas so' ha' (\d+)\. Menor alpha garantivel: (.*?) "
     r"\(precisa de n>=(\d+) para alpha=(.*?)\)\. Limiar NAO definido -- reportar como nao-estimavel, nunca como "
     r"'aceita tudo'\.",
     r"[AVISO] alpha=\1 is NOT reachable with \2 independent calibration samples/groups: it would need k=\4 "
     r"ordered scores, but there are only \5. Smallest guaranteed alpha: \6 (needs n>=\7 for alpha=\8). "
     r"Threshold NOT defined -- report as not estimable, never as 'accepts everything'."),
    (r"\[AVISO\] Dominio de aplicabilidade calibrado com poucos (amostras|grupos \(amostras fisicas\)) por classe "
     r"\(minimo (\d+); (\d+) classe\(s\) abaixo de (\d+)\)\. Nesse regime ele costuma rejeitar muitas amostras "
     r"legitimas: a taxa esperada de alarmes e' desconhecida e pode ser muito maior que os 5% nominais\. Use a "
     r"taxa de rejeicao em validacao cruzada como estimativa e nao trate 'fora do dominio' como prova de problema "
     r"na amostra\.",
     r"[AVISO] Applicability domain calibrated with few samples/groups per class (minimum \2; \3 class(es) below "
     r"\4). In this regime it tends to reject many legitimate samples: the expected alarm rate is unknown and may "
     r"be much higher than the nominal 5%. Use the cross-validated rejection rate as the estimate and do not take "
     r"'outside the domain' as proof of a problem with the sample."),
    (r"\[AD\] Taxa esperada de alarmes em amostras legitimas \(rejeicao do dominio em CV por grupo\): (.*)",
     r"[AD] Expected alarm rate on legitimate samples (domain rejection in group CV): \1"),
    (r"\[Identificacao\] (\d+) combinacoes especie x adulterante calibradas, (\d+) com cobertura VALIDADA\. Ver "
     r"model card para o detalhe por combinacao\.",
     r"[Identification] \1 species x adulterant combinations calibrated, \2 with VALIDATED coverage. See the "
     r"model card for the detail of each combination."),
    (r"\[Identificacao\] NENHUMA combinacao especie x adulterante tem cobertura estatistica validada neste "
     r"dataset -- Identificar/Quantificar em amostra nova ficam bloqueados por padrao \(ver cobertura_status\)\.",
     r"[Identification] NO species x adulterant combination has validated statistical coverage in this dataset "
     r"-- Identify/Quantify on a new sample are blocked by default (see cobertura_status)."),
    (r"\[Pureza\] DD-SIMCA por especie \(Detectar/pureza\): (\d+) especies calibradas, (\d+) com alpha sustentado "
     r"por evidencia \(LOGO medido ou >= (\d+) grupos\); nas demais a taxa de falsa rejeicao de puros novos e' "
     r"desconhecida e pode ser alta\.",
     r"[Purity] DD-SIMCA by species (Detect/purity): \1 species calibrated, \2 with alpha backed by evidence "
     r"(measured LOGO or >= \3 groups); for the others the false-rejection rate of new pure samples is unknown "
     r"and may be high."),
    (r"-> (.*?) \(formato portatil, sem pickle\)", r"-> \1 (portable format, no pickle)"),
    (r"\[7/7\] PLS regressao — PULADA: objective=(.*?) \(level=(.*?)\)\. A regressao da resposta pertence ao Modo "
     r"Quantificacao \(N3\); nos modos Exploratorio/Classificacao ela seria um resultado fora de escopo \(e em N1 "
     r"as 3 classes dominam o sinal da resposta, produzindo R2~0\)\.",
     r"[7/7] PLS regression — SKIPPED: objective=\1 (level=\2). Regression of the response belongs to the "
     r"Quantification mode (N3); in the Exploratory/Classification modes it would be an out-of-scope result (and "
     r"in N1 the 3 classes dominate the response signal, giving R2~0)."),
    (r"\[7/7\] PLS regressao — PULADA: (.*)", r"[7/7] PLS regression — SKIPPED: \1"),
    (r"Resultados em: (.*)", r"Results in: \1"),
    (r"\[WARNING\] Grouping guarantee: (\w+) -- ver 'Grouping guarantee' no resumo/model card para o que isso "
     r"significa para a validacao desta execucao\.",
     r"[WARNING] Grouping guarantee: \1 -- see 'Grouping guarantee' in the summary/model card for what this "
     r"means for the validation of this run."),
    (r"\[1/7\] Pre-processamento \(preset='(.*?)'\)", r"[1/7] Preprocessing (preset='\1')"),
    (r"\[AVISO\] Faixa espectral: so' (.*?) das (\d+) variaveis carregam sinal \((.*?)\)\.",
     r"[AVISO] Spectral range: only \1 of the \2 variables carry signal (\3)."),
    (r"regiao ruidosa: (.*)", r"noisy region: \1"),
    (r"regiao morta: (.*)", r"dead region: \1"),
    (r"faixa com sinal: (.*?) \(atual: (.*?)\)", r"range with signal: \1 (current: \2)"),
    (r"\[ATENCAO\] LVs otimas \((\d+)\) == max_lvs \((\d+)\): RMSECV ainda nao atingiu plateau\. Aumente max_lvs "
     r"\(ex: (\d+)\) e rode novamente\.",
     r"[ATENCAO] Optimal LVs (\1) == max_lvs (\2): RMSECV has not reached a plateau yet. Raise max_lvs "
     r"(e.g. \3) and run again."),
    (r"\[4/7\] Teste de permutacao — PULADO: objective=(.*?)\. Significancia de classificacao nao e' pertinente "
     r"fora do modo Classificacao \(economiza (\d+) refits de CV\)\.",
     r"[4/7] Permutation test — SKIPPED: objective=\1. Classification significance does not apply outside the "
     r"Classification mode (saves \2 CV refits)."),
    (r"\[5b/7\] BCa CI 95% — PULADO: objective=(.*?)\. Intervalo de confianca de metricas de classificacao nao e' "
     r"pertinente fora do modo Classificacao\.",
     r"[5b/7] BCa CI 95% — SKIPPED: objective=\1. A confidence interval of classification metrics does not apply "
     r"outside the Classification mode."),
    (r"\[7/7\] PLS regressao POR ESPECIE \((\d+) especies — calibracao separada para evitar confusao "
     r"inter-especies\)",
     r"[7/7] PLS regression BY SPECIES (\1 species — separate calibration to avoid confusion between species)"),
    (r"Modo de rotulo na quantificacao: (.*)", r"Label mode in quantification: \1"),
    (r"Especies modeladas: (\d+)", r"Species modelled: \1"),
    (r"DModY critico \(SIMCA\): (.*?)\s+\|\s+amostras fora: (\d+)", r"Critical DModY (SIMCA): \1  |  samples outside: \2"),
    (r"\[Modelo\] Atualizado com regressao por especie \((\d+) especies\) -> (.*)",
     r"[Modelo] Updated with regression by species (\1 species) -> \2"),
    (r"Faixa de validacao: (.*?)\s+DP=(.*)", r"Validation range: \1  SD=\2"),
    (r"Linearidade \(falta de ajuste\): (.*)", r"Linearity (lack of fit): \1"),
    (r"\[RESSALVA\] rejeicao NAO conclusiva: erro puro estimado so' de replicas tecnicas do mesmo mae_id "
     r"\(subestima o erro de preparo\) e preparos independentes do mesmo teor contam como niveis distintos: o "
     r"teste tende a rejeitar linearidade -- use nivel_por='conc' se ha' preparos independentes",
     r"[RESSALVA] rejection NOT conclusive: pure error estimated only from technical replicates of the same mae_id "
     r"(underestimates the preparation error) and independent preparations of the same content count as distinct "
     r"levels: the test tends to reject linearity -- use nivel_por='conc' if there are independent preparations"),
    (r"\[!\] Validacao agrupada pelo grupo DECLARADO pelo usuario \(garantia MEDIA\): o software nao verifica que "
     r"os grupos sao de fato independentes\.",
     r"[!] Validation grouped by the group DECLARED by the user (MEDIUM guarantee): the software does not check "
     r"that the groups are really independent."),
    (r"\[INFO\] Loading CSV: (.*)", r"[INFO] Loading CSV: \1"),
]

#: trechos que aparecem DENTRO de linhas ja' traduzidas (aplicados depois da tabela acima): os nomes das
#: figuras do plano (`modos_analise._FIG_DESCRICAO`) e dos objetivos.
_TRECHOS: List[Tuple[str, str]] = [
    (" puros preservados", " pure kept"),
    ("Espectros medios por classe", "Mean spectra by class"),
    ("T2 de Hotelling e Q por amostra", "Hotelling T2 and Q by sample"),
    ("Dendrogramas por pre-processamento", "Dendrograms by preprocessing"),
    ("R2cv por especie x adulterante", "R2cv by species x adulterant"),
    ("Dendrograma HCA", "HCA dendrogram"),
    ("Efeito do pre-processamento", "Preprocessing effect"),
    ("Matriz de confusao + metricas por classe", "Confusion matrix + metrics by class"),
    ("Curvas ROC/AUC multiclasse", "Multiclass ROC/AUC curves"),
    ("Selecao de variaveis latentes", "Latent variable selection"),
    ("Contribuicao de score", "Score contribution"),
    ("DD-SIMCA (autenticacao de pureza)", "DD-SIMCA (purity authentication)"),
    ("Selecao de variaveis (iPLS/sPLS-DA/SPA/AG)", "Variable selection (iPLS/sPLS-DA/SPA/GA)"),
    ("Comparacao de pipelines de pre-proc", "Preprocessing pipeline comparison"),
    ("Teste de permutacao (Wold)", "Permutation test (Wold)"),
    ("Avaliacao em holdout externo", "External hold-out evaluation"),
    ("Teste de incerteza de Martens", "Martens uncertainty test"),
    ("Regressao PLS + figuras de merito analiticas", "PLS regression + analytical figures of merit"),
    ("Auto-Benchmark de regressao", "Regression auto-benchmark"),
    ("Exploratorio", "Exploratory"), ("Classificacao", "Classification"), ("Quantificacao", "Quantification"),
]

_RE: List[Tuple[Pattern[str], str]] = [(re.compile(p), en) for p, en in FRASES_CONSOLE]
_RE_AUDITORIA = re.compile(r"\[AUDITORIA\]\[(CRITICO|AVISO|INFO)\] (\w+): (.*)", re.DOTALL)
_NIVEL_EN = {"CRITICO": "CRITICAL", "AVISO": "WARNING", "INFO": "INFO"}
#: linhas que o painel le por regex com o texto em PT: nunca traduzir
_INTOCAVEIS = re.compile(r"fold externo \d+/\d+ concluido|\[7d/7\].*fold \d+/\d+ concluido")


def traduzir_linha(linha: str, lang: Optional[str] = None) -> str:
    """A linha de console no idioma da sessao (so' traduz para ingles; em PT devolve igual)."""
    if _ir.idioma(lang) != "en" or not linha.strip() or _INTOCAVEIS.search(linha):
        return linha
    ind = linha[: len(linha) - len(linha.lstrip())]
    corpo = linha[len(ind):]
    m = _RE_AUDITORIA.fullmatch(corpo)
    if m:
        nivel, chave, msg = m.groups()
        # a etiqueta do painel e' "[AUDITORIA][CRITICO]": fica; a mensagem vai pela tabela da auditoria
        return f"{ind}[AUDITORIA][{nivel}] {chave}: {_ir.achado(msg, 'en')}"
    for rx, en in _RE:
        mm = rx.fullmatch(corpo)
        if mm:
            saida = mm.expand(en)
            for pt, en_t in _TRECHOS:
                saida = saida.replace(pt, en_t)
            return ind + saida
    return linha


def traduzir_texto(texto: str, lang: Optional[str] = None) -> str:
    """`traduzir_linha` em cada linha de um bloco."""
    if _ir.idioma(lang) != "en" or not texto:
        return texto
    return "\n".join(traduzir_linha(l, lang) for l in texto.split("\n"))


class _SaidaTraduzida:
    """Proxy de `sys.stdout` que traduz cada linha COMPLETA antes de escrever (o resto do texto fica no
    buffer ate' o fim da linha ou o `flush`). Atributos que nao sao de escrita vao para o stream real."""

    def __init__(self, real, lang: Optional[str]) -> None:
        self._real = real
        self._lang = lang
        self._pendente = ""

    def write(self, s: str) -> int:
        self._pendente += s
        if "\n" in self._pendente:
            linhas, self._pendente = self._pendente.rsplit("\n", 1)
            self._real.write(traduzir_texto(linhas, self._lang) + "\n")
        return len(s)

    def flush(self) -> None:
        if self._pendente:
            self._real.write(traduzir_linha(self._pendente, self._lang))
            self._pendente = ""
        self._real.flush()

    def __getattr__(self, nome: str):
        return getattr(self._real, nome)


class saida_traduzida:
    """Contexto que troca `sys.stdout` pelo proxy tradutor durante a execucao (CLI `run`, `demo` e a execucao
    pelo menu). Em portugues nao troca nada."""

    def __init__(self, lang: Optional[str] = None) -> None:
        self._lang = lang
        self._antigo: Optional[Any] = None

    def __enter__(self):
        import sys
        if _ir.idioma(self._lang) == "en" and not isinstance(sys.stdout, _SaidaTraduzida):
            self._antigo = sys.stdout
            sys.stdout = _SaidaTraduzida(sys.stdout, self._lang)
        return self

    def __exit__(self, *exc) -> None:
        import sys
        if self._antigo is not None:
            try:
                sys.stdout.flush()
            finally:
                sys.stdout = self._antigo
