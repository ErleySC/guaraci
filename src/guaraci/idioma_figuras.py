# -*- coding: utf-8 -*-
"""Texto DENTRO das figuras no idioma da sessao (titulos, eixos, legendas, caixas, rotulos de barra).

As funcoes de figura (muitas no nucleo congelado) escrevem os textos num idioma fixo -- a maioria em
ingles, algumas em portugues sem acento. Em vez de reescrever cada uma, `traduzir_figura(fig)` roda
logo antes de gravar (`figuras.save` e os poucos `savefig` diretos) e troca cada texto pelo do idioma
da sessao (`idioma_relatorio.idioma()`), por um glossario de frases.

- Uma passada so', da frase mais longa para a mais curta: um texto ja' traduzido nao e' traduzido de
  novo, e "Validation (n=" vence "Validation".
- Frases com mathtext (`$...$`) entram no glossario inteiras; formula solta nao casa com nada.
- Frase fora do glossario fica como esta'; o teste de idioma das figuras
  (tests/test_idioma_relatorios.py) roda os exemplos e acusa palavra do outro idioma.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

from guaraci.idioma_relatorio import idioma

__all__ = ["traduzir_texto", "traduzir_figura", "textos_da_figura"]

# ---------------------------------------------------------------------------
#  Ingles -> portugues (o grosso das figuras nasce em ingles)
# ---------------------------------------------------------------------------
_EN_PT: List[Tuple[str, str]] = [
    # --- espectros, cromatogramas, EEM, eixos -----------------------------------------------------
    ("Mean chromatogram per class, overlaid (band = ±1 SD) — raw data, before modelling",
     "Cromatograma médio por classe, sobrepostos (faixa = ±1 DP) — dados brutos, antes da modelagem"),
    ("Mean spectra per class (band = ±1 SD) — raw data, before modelling",
     "Espectros médios por classe (faixa = ±1 DP) — dados brutos, antes da modelagem"),
    ("Mean excitation–emission matrix per class (raw data, common colour scale)",
     "Matriz excitação–emissão média por classe (dados brutos, escala de cor comum)"),
    ("Raw spectra (mean ", "Espectros brutos (média "), (" SD per class)", " DP por classe)"),
    ("Intensity (detector signal)", "Intensidade (sinal do detector)"),
    ("Mean intensity (raw)", "Intensidade média (bruta)"),
    ("Intensity", "Intensidade"),
    ("Preprocessed signal", "Sinal pré-processado"),
    ("Normalized value (global max = 1)", "Valor normalizado (máximo global = 1)"),
    ("Wavenumber", "Número de onda"), ("Wavelength", "Comprimento de onda"),
    ("Raman shift", "Deslocamento Raman"), ("Retention time (min)", "Tempo de retenção (min)"),
    ("Chemical shift", "Deslocamento químico"), ("Drift time", "Tempo de deriva"),
    ("Variable index", "Índice da variável"),
    ("Emission wavelength (nm)", "Comprimento de onda de emissão (nm)"),
    ("Excitation wavelength (nm)", "Comprimento de onda de excitação (nm)"),
    ("Spectral variable", "Variável espectral"),
    ("Autoscaling (mean + unit variance)", "autoescalonamento (média + variância unitária)"),
    ("mean-centering", "centralização na média"), ("Mean-centering", "Centralização na média"),
    ("Sample index", "Índice da amostra"), ("Sample (grouped by class)", "Amostra (agrupadas por classe)"),
    # --- PCA / HCA ---------------------------------------------------------------------------------
    ("PCA — exploratory (unsupervised)", "PCA — exploratória (não supervisionada)"),
    ("PCA loading plot — spectral contribution per component",
     "Loadings da PCA — contribuição espectral por componente"),
    ("% of explained variance  | red = positive · blue = negative",
     "% da variância explicada  | vermelho = positivo · azul = negativo"),
    ("PCA scree — ", "Scree da PCA — "), (" components computed, cumulative ", " componentes calculados, acumulado "),
    ("Explained variance (%)", "Variância explicada (%)"), ("Principal component", "Componente principal"),
    ("Per component", "Por componente"), ("Cumulative", "Acumulada"),
    ("Biplot PCA — scores + top-", "Biplot PCA — escores + top-"), (" loadings (labels: ", " loadings (rótulos: "),
    ("HCA by preprocessing — cluster stability", "HCA por pré-processamento — estabilidade dos agrupamentos"),
    ("HCA — dendrogram (centroids, PCA ", "HCA — dendrograma (centroides, PCA "),
    ("Distance (", "Distância ("), (", Euclidean)", ", euclidiana)"),
    ("Comparison of preprocessing pipelines — current: ", "Comparação de pré-processamentos — atual: "),
    (" (not among the presets)", " (fora das predefinições)"),
    ("failed\n", "falhou\n"),
    # --- PLS-DA, VLs, validacao --------------------------------------------------------------------
    ("PLS-DA — LV1 scores per class (model has 1 LV; bar = median)",
     "PLS-DA — escores da VL1 por classe (modelo com 1 VL; barra = mediana)"),
    ("(a) LV selection", "(a) Seleção de VLs"), ("Number of latent variables", "Número de variáveis latentes"),
    ("Optimal: ", "Ótimo: "),
    ("(a) Confusion matrix (CV)", "(a) Matriz de confusão (VC)"),
    ("(a) Confusion matrix — holdout (", "(a) Matriz de confusão — holdout ("),
    ("Proportion per true class", "Proporção por classe verdadeira"),
    ("(b) Cross-validation metrics", "(b) Métricas da validação cruzada"),
    ("(b) Cross-validation vs holdout (overfitting check)", "(b) Validação cruzada vs holdout (checagem de sobreajuste)"),
    ("(b) Per-class metrics (one-vs-rest)", "(b) Métricas por classe (um contra todos)"),
    ("Value (cross-validation)", "Valor (validação cruzada)"),
    ("Metric (CV)", "Métrica (VC)"), ("CV (training)", "VC (treino)"), ("Holdout (test)", "Holdout (teste)"),
    ("External holdout (n=", "Holdout externo (n="), ("External holdout", "Holdout externo"),
    ("Balanced Accuracy (CV fold)", "Acurácia balanceada (dobra da VC)"),
    ("Balanced accuracy (CV group-aware)\nvars: nos metodos com selecao aninhada, MEDIA de variaveis selecionadas "
     "entre as dobras de CV [faixa min-max]",
     "Acurácia balanceada (VC agrupada)\nvars: nos métodos com seleção aninhada, MÉDIA de variáveis selecionadas "
     "entre as dobras da VC [faixa mín-máx]"),
    ("Balanced accuracy (CV group-aware)", "Acurácia balanceada (VC agrupada)"),
    ("Balanced accuracy (CV)", "Acurácia balanceada (VC)"),
    ("Balanced Accuracy", "Acurácia balanceada"), ("Balanced accuracy", "Acurácia balanceada"),
    ("Balanced acc.", "Acurácia bal."), ("Accuracy", "Acurácia"),
    ("Cohen's $\\kappa$", "$\\kappa$ de Cohen"), ("Cohen's kappa", "Kappa de Cohen"), ("Precision", "Precisão"), ("Sensitivity", "Sensibilidade"),
    ("Specificity", "Especificidade"), ("F1-score", "F1"),
    ("ROC curves — PLS-DA (", "Curvas ROC — PLS-DA ("), ("group-aware CV", "VC agrupada"),
    ("CV, not grouped", "VC, não agrupada"), (") | macro AUC = ", ") | AUC macro = "),
    ("False positive rate  (1 – specificity)", "Taxa de falsos positivos  (1 – especificidade)"),
    ("True positive rate  (sensitivity)", "Taxa de verdadeiros positivos  (sensibilidade)"),
    ("Random (AUC = ", "Acaso (AUC = "),
    ("LVs) — 95% limit", "VLs) — limite de 95%"),
    ("(b) T$^2$ vs Q — outlier detection", "(b) T$^2$ vs Q — detecção de outliers"),
    ("Q-residuals / SPE", "Resíduos Q / SPE"), ("Q 95% limit (", "Limite de 95% de Q ("),
    ("Mean spectral contribution per class — LV1", "Contribuição espectral média por classe — VL1"),
    ("Mean contribution ", "Contribuição média "), (" SE per class", " EP por classe"),
    (" variables — discriminant power between classes", " variáveis — poder discriminante entre classes"),
    ("(a) Wold — R$^2$Y vs permutation", "(a) Wold — R$^2$Y vs permutação"),
    ("(b) Wold — Q$^2$Y vs permutation", "(b) Wold — Q$^2$Y vs permutação"),
    ("Similarity (permuted Y, original Y)", "Similaridade (Y permutado, Y original)"),
    ("R$^2$Y (training fit)", "R$^2$Y (ajuste no treino)"), ("Q$^2$Y (CV)", "Q$^2$Y (VC)"),
    ("R2Y intercept: ", "Intercepto do R2Y: "), ("Q2Y intercept: ", "Intercepto do Q2Y: "),
    ("Threshold R2Y", "Limiar R2Y"), ("Threshold Q2Y", "Limiar Q2Y"),
    ("Line (intercept = ", "Reta (intercepto = "), ("Permutations", "Permutações"), ("Observed", "Observado"),
    ("Monte Carlo CV (", "VC Monte Carlo ("),
    ("False Match Rate — FMR (%)", "Taxa de falsa aceitação — FMR (%)"),
    ("False Non-Match Rate — FNMR (%)", "Taxa de falsa rejeição — FNMR (%)"),
    ("]\nOOF predictions, ", "]\npredições fora da dobra, "),
    ("Mean |SHAP value|", "|valor SHAP| médio"), ("SHAP Dependence — ", "Dependência SHAP — "),
    ("\nFeature #", "\nVariável nº "), ("Chance = 1/", "Acaso = 1/"),
    # --- VIP / SR / selecao ------------------------------------------------------------------------
    ("(a) Mean VIP with 95% CI", "(a) VIP médio com IC de 95%"),
    ("(a) VIP and SR across the spectrum", "(a) VIP e SR ao longo do espectro"),
    ("(c) Selection stability", "(c) Estabilidade da seleção"),
    ("Selection frequency (VIP ", "Frequência de seleção (VIP "),
    ("VIP (stratified bootstrap mean)", "VIP (média do bootstrap estratificado)"),
    ("VIP score (with 95% CI)", "VIP (com IC de 95%)"), ("VIP score", "VIP"),
    ("95% CI (bootstrap)", "IC de 95% (bootstrap)"),
    ("Selectivity Ratio", "Razão de seletividade"),
    (" by SR vs VIP", " por SR vs VIP"),
    # --- DD-SIMCA ----------------------------------------------------------------------------------
    ("Cooman's Plot — DD-SIMCA (", "Gráfico de Cooman — DD-SIMCA ("),
    (" scale; dashed = acceptance limit of each model)", "; tracejado = limite de aceitação de cada modelo)"),
    (" — model ", " — modelo "), ("  (≤ 1: accepted by ", "  (≤ 1: aceita por "),
    ("Acceptance boundary", "Fronteira de aceitação"),
    ("Model: ", "Modelo: "), ("| spec.=", "| esp.="),
    ("threshold calibrated on n=", "limiar calibrado com n="), (" physical sample(s)", " amostra(s) física(s)"),
    ("threshold calibrated per SPECTRUM (no mae_id)", "limiar calibrado por ESPECTRO (sem mae_id)"),
    ("(train n=", "(treino n="),
    # --- regressao, Williams, merito ---------------------------------------------------------------
    ("PLS regression coefficients (b) on the pre-processed signal  | fill: positive / negative",
     "Coeficientes da regressão PLS (b) no sinal pré-processado  | preenchimento: positivo / negativo"),
    ("Coefficient b", "Coeficiente b"),
    ("Williams plot — applicability domain of the PLS regression",
     "Gráfico de Williams — domínio de aplicabilidade da regressão PLS"),
    ("Standardized residual (pred − ref) / s", "Resíduo padronizado (pred − ref) / s"),
    ("Leverage h", "Alavancagem h"), (" above h*, ", " acima de h*, "), (" with |r| > 3", " com |r| > 3"),
    ("(b) Predicted vs Reference", "(b) Predito vs referência"),
    ("Predicted vs reference with ±RMSEP band\n", "Predito vs referência com faixa ±RMSEP\n"),
    (" validation points inside ±RMSEP", " pontos de validação dentro de ±RMSEP"),
    ("(c) Residuals — Validation\nBias = mean(pred − ref) = ", "(c) Resíduos — validação\nViés = média(pred − ref) = "),
    ("Residual (predicted − reference)", "Resíduo (predito − referência)"),
    ("Concentration (unit of the reference y)", "Concentração (unidade do y de referência)"),
    ("± RMSEP holdout (", "± RMSEP do holdout ("),
    ("(a) LOD / LOQ per species", "(a) LOD / LOQ por espécie"),
    ("(b) Analytical selectivity (SEL, net analyte signal)", "(b) Seletividade analítica (SEL, sinal líquido do analito)"),
    ("SEL — mean |cos(b, x", "SEL — média de |cos(b, x"),
    ("Predicted value", "Valor predito"), ("Reference value", "Valor de referência"),
    ("Predicted", "Predito"), ("Calibration", "Calibração"), ("Validation", "Validação"),
    ("Species", "Espécie"), ("Adulterant", "Adulterante"), ("Class", "Classe"), ("True", "Verdadeira"),
    ("Value", "Valor"), ("Score", "Escore"),
    # --- transferencia, MCR-ALS, PARAFAC, ASCA, ANOVA ----------------------------------------------
    ("Transfer sample (each corrected by a fit that left it out)",
     "Amostra de transferência (cada uma corrigida por um ajuste que a deixou de fora)"),
    ("Mean signal of the transfer samples", "Sinal médio das amostras de transferência"),
    ("(c) Error per transfer sample, before and after", "(c) Erro por amostra de transferência, antes e depois"),
    ("(a) Before: ", "(a) Antes: "), ("(b) After ", "(b) Após "),
    ("RMS slave − master", "RMS escravo − mestre"), ("RMS after (LOO)", "RMS depois (LOO)"),
    ("RMS before", "RMS antes"), ("Slave after ", "Escravo após "), ("Slave (raw)", "Escravo (bruto)"),
    ("Master", "Mestre"), ("Calibration transfer ", "Transferência de calibração "),
    (" transfer samples", " amostras de transferência"), (", window ", ", janela "),
    ("MCR-ALS resolution", "Resolução MCR-ALS"),
    (" components, relative reconstruction error ", " componentes, erro relativo de reconstrução "),
    ("  (C·S scale is ambiguous: compare shapes, not absolute heights)",
     "  (a escala de C·S é ambígua: compare formas, não alturas absolutas)"),
    ("(a) Concentration profiles C", "(a) Perfis de concentração C"),
    ("(b) Resolved pure spectra S", "(b) Espectros puros resolvidos S"),
    ("Concentration profile C (a.u.)", "Perfil de concentração C (u.a.)"),
    ("Pure spectrum S (a.u.)", "Espectro puro S (u.a.)"),
    ("PARAFAC loadings per mode — ", "Loadings do PARAFAC por modo — "),
    ("PARAFAC with non-negativity constraint (HALS) since v1.0.0; earlier versions or unconstrained software may give "
     "different signs/orders of the factors.",
     "PARAFAC com restrição de não-negatividade (HALS) desde a v1.0.0; versões anteriores ou outros softwares sem "
     "restrição podem dar sinais/ordens diferentes dos fatores."),
    ("PARAFAC without constraint (free ALS), by user choice; the default since v1.0.0 is non-negative (HALS).",
     "PARAFAC sem restrição (ALS livre), por escolha do usuário; o padrão desde a v1.0.0 é não-negativo (HALS)."),
    ("(a) Excitation mode", "(a) Modo de excitação"), ("(b) Emission mode", "(b) Modo de emissão"),
    ("(c) Sample mode (scores)", "(c) Modo das amostras (escores)"),
    ("ASCA — factor ", "ASCA — fator "), (": effect loadings", ": loadings do efeito"),
    (": effect scores (diamond = level, cloud = level + residual) and loadings",
     ": escores do efeito (losango = nível, nuvem = nível + resíduo) e loadings"),
    ("Effect loading", "Loading do efeito"), ("% of total SS", "% da SQ total"),
    ("Component ", "Componente "),
    ("Mean per group ± SD between independent units (points = units)",
     "Média por grupo ± DP entre unidades independentes (pontos = unidades)"),
    ("Mean per group (independent units)", "Média por grupo (unidades independentes)"),
    ("OPLS-DA — predictive × orthogonal (", "OPLS-DA — preditiva × ortogonal ("), (" orth. comp.)", " comp. ortog.)"),
    (" (predictive)", " (preditiva)"), (" (orthogonal ", " (ortogonal "),
    ("Correlation(", "Correlação("), ("Covariance(", "Covariância("),
    ("SHAP value (own class)", "Valor SHAP (própria classe)"),
    ("Raw", "Bruto"), ("Normalization", "Normalização"), ("SG 1st deriv", "SG 1ª derivada"),
    ("SG 2nd deriv", "SG 2ª derivada"),
    # --- marca d'agua -------------------------------------------------------------------------------
    ("PROTOTYPE\nNOT VALIDATED", "PROTÓTIPO\nNÃO VALIDADO"),
    # --- genericos (por ultimo: so' valem se nada mais longo casou) -------------------------------
    (" samples)", " amostras)"), (" variables", " variáveis"),
]

# ---------------------------------------------------------------------------
#  Portugues (sem acento, do nucleo) -> ingles; e a grafia corrigida em portugues
# ---------------------------------------------------------------------------
#: (texto como o codigo grava, portugues correto, ingles)
_PT: List[Tuple[str, str, str]] = [
    ("Carta T2 e Q por amostra (linhas verticais separam lotes)\ndecisao de dominio da tabela: distancia combinada T2+Q",
     "Carta T2 e Q por amostra (linhas verticais separam lotes)\ndecisão de domínio da tabela: distância combinada T2+Q",
     "T2 and Q chart per sample (vertical lines separate batches)\ntable domain decision: combined T2+Q distance"),
    ("Contribuicao ao Q da amostra com maior Q (linha ", "Contribuição ao Q da amostra com maior Q (linha ",
     "Q contribution of the sample with the largest Q (row "),
    ("Tendencia por lote (vermelho = alarme)", "Tendência por lote (vermelho = alarme)", "Trend per batch (red = alarm)"),
    ("lote (amostras na ordem do arquivo)", "lote (amostras na ordem do arquivo)", "batch (samples in file order)"),
    ("amostra (ordem do arquivo)", "amostra (ordem do arquivo)", "sample (file order)"),
    ("contribuicao ao Q", "contribuição ao Q", "Q contribution"),
    ("fracao fora do dominio", "fração fora do domínio", "fraction outside the domain"),
    ("taxa da calibracao (CV)", "taxa da calibração (VC)", "calibration rate (CV)"),
    ("variavel (ordem das colunas)", "variável (ordem das colunas)", "variable (column order)"),
    (" <= limite", " <= limite", " <= limit"), (" > limite", " > limite", " > limit"),
    (", lote ", ", lote ", ", batch "),
    ("T2 de Hotelling", "T2 de Hotelling", "Hotelling T2"), ("Q (residuo)", "Q (resíduo)", "Q (residual)"),
    ("Quantil-quantil dos residuos padronizados (reta y = x)", "Quantil-quantil dos resíduos padronizados (reta y = x)",
     "Quantile-quantile of the standardized residuals (line y = x)"),
    ("quantil teorico (normal)", "quantil teórico (normal)", "theoretical quantile (normal)"),
    ("residuo padronizado (residuo / DP)", "resíduo padronizado (resíduo / DP)", "standardized residual (residual / SD)"),
    ("Valores por unidade, por grupo", "Valores por unidade, por grupo", "Values per unit, per group"),
    ("VIP (media das dobras)", "VIP (média das dobras)", "VIP (mean over folds)"),
    ("VIP com estabilidade (", "VIP com estabilidade (", "VIP with stability ("),
    (" ajustes; azul = VIP > ", " ajustes; azul = VIP > ", " fits; blue = VIP > "),
    (" em >= ", " em >= ", " in >= "),
    ("% das dobras; laranja = instavel)", "% das dobras; laranja = instável)", "% of folds; orange = unstable)"),
    ("Etapa 4 — AG: convergencia da busca genetica", "Etapa 4 — AG: convergência da busca genética",
     "Stage 4 — GA: convergence of the genetic search"),
    ("Etapa 4 — comparacao de metodos de selecao de variaveis",
     "Etapa 4 — comparação de métodos de seleção de variáveis", "Stage 4 — comparison of variable selection methods"),
    ("Etapa 4 — iPLS: desempenho por intervalo espectral", "Etapa 4 — iPLS: desempenho por intervalo espectral",
     "Stage 4 — iPLS: performance per spectral interval"),
    ("Geracao", "Geração", "Generation"), ("Intervalo iPLS", "Intervalo iPLS", "iPLS interval"),
    ("Media da populacao", "Média da população", "Population mean"),
    ("Melhor da geracao", "Melhor da geração", "Best of the generation"),
    ("Modelo global (", "Modelo global (", "Global model ("), ("melhor: ", "melhor: ", "best: "),
    ("Quantificacao especie x adulterante — ", "Quantificação espécie x adulterante — ",
     "Quantification species x adulterant — "),
    (" combinacoes abaixo de R²cv = ", " combinações abaixo de R²cv = ", " combinations below R²cv = "),
    ("R²cv  (limiar de aceite = ", "R²cv  (limiar de aceite = ", "R²cv  (acceptance threshold = "),
    ("S-Plot OPLS-DA — quadrante sup-dir: discriminantes positivos; inf-esq: negativos; centro: ruído",
     "S-Plot OPLS-DA — quadrante sup-dir: discriminantes positivos; inf-esq: negativos; centro: ruído",
     "S-Plot OPLS-DA — upper-right quadrant: positive discriminants; lower-left: negative; centre: noise"),
    ("Auto-Benchmark de regressao -- ", "Comparação automática de regressores -- ", "Regression auto-benchmark -- "),
    ("Auto-Benchmark — ", "Comparação automática — ", "Auto-benchmark — "),
    ("RMSEP por especie (menor = melhor)", "RMSEP por espécie (menor = melhor)", "RMSEP per species (lower = better)"),
    (" iteracoes, test=", " iterações, teste=", " iterations, test="),
    (" especies (", " espécies (", " species ("),
    ("  |  avaliado em: ", "  |  avaliado em: ", "  |  evaluated on: "),
    (")\nIC95% percentil — pre-processamento: ", ")\nIC 95% percentil — pré-processamento: ",
     ")\n95% percentile CI — preprocessing: "),
    ("-fold StratifiedGroupKFold estavel (anti-leakage de replicas)\nPreprocessamento: ",
     "-fold StratifiedGroupKFold estável (sem vazamento de réplicas)\nPré-processamento: ",
     "-fold stable StratifiedGroupKFold (no replicate leakage)\nPreprocessing: "),
    ("Curvas DET (Detection Error Tradeoff) — macro OvR [", "Curvas DET (compromisso entre erros) — macro um contra todos [",
     "DET curves (Detection Error Tradeoff) — macro OvR ["),
    ("IC95%\n[", "IC 95%\n[", "95% CI\n["),
    (" bandas espectrais)\nEixo: ", " bandas espectrais)\nEixo: ", " spectral bands)\nAxis: "),
    ("Mapa de classificação HSI por pixel", "Mapa de classificação HSI por pixel", "HSI per-pixel classification map"),
    ("objeto '", "objeto '", "object '"), ("gravação ", "gravação ", "recording "),
    ("classe verdadeira: ", "classe verdadeira: ", "true class: "),
    ("predição FORA DA AMOSTRA", "predição FORA DA AMOSTRA", "OUT-OF-SAMPLE prediction"),
    ("(modelo ajustado sem este objeto, ", "(modelo ajustado sem este objeto, ", "(model fitted without this object, "),
    ("Variável espectral", "Variável espectral", "Spectral variable"),
    ("PROTOTIPO\nNAO VALIDADO", "PROTÓTIPO\nNÃO VALIDADO", "PROTOTYPE\nNOT VALIDATED"),
]

#: construcoes com numero: (regex, PT, EN), aplicadas depois do glossario.
_REGEX: List[Tuple[str, str, str]] = [
    (r"\bT\$\^2\$ de Hotelling|\bHotelling T\$\^2\$", "T$^2$ de Hotelling", "Hotelling T$^2$"),
    (r"\(a\) T\$\^2\$ \(", "(a) T$^2$ de Hotelling (", "(a) Hotelling T$^2$ ("),
    (r"T\$\^2\$ 95% limit \(|Limite de 95% de T\$\^2\$ \(", "Limite de 95% de T$^2$ (", "T$^2$ 95% limit ("),
    (r"\(b\) Top (\d+) (?:variables|variáveis)", r"(b) As \1 variáveis principais", r"(b) Top \1 variables"),
    (r"\(b\) Top (\d+) por SR vs VIP|\(b\) Top (\d+) by SR vs VIP", r"(b) As \1\2 principais por SR vs VIP",
     r"(b) Top \1\2 by SR vs VIP"),
    (r"^Top (\d+) variáveis — poder discriminante entre classes$",
     r"As \1 variáveis de maior poder discriminante entre classes", r"Top \1 variables — discriminant power between classes"),
    (r": VALID$", ": VÁLIDO", ": VALID"), (r": FAILED$", ": FALHOU", ": FAILED"),
    (r"\bLVs\b", "VLs", "LVs"), (r"\bLV(\d*)\b", r"VL\1", r"LV\1"),
    (r"\b(\d+) VL\)", r"\1 VL)", r"\1 LV)"),
    (r"limite de (T2|Q) \(alpha", r"limite de \1 (alpha", r"\1 limit (alpha"),
    (r"(\d+)/(\d+) pontos de validação", r"\1/\2 pontos de validação", r"\1/\2 validation points"),
]


def _compilar(pares: List[Tuple[str, str]]) -> Tuple[Optional[Pattern[str]], Dict[str, str]]:
    mapa = {de: para for de, para in pares if de != para}
    if not mapa:
        return None, {}
    alternativas = sorted(mapa, key=len, reverse=True)
    letra = "A-Za-zÀ-ÿ"
    partes = []
    for de in alternativas:
        p = re.escape(de)
        if de[:1].isalpha():
            p = f"(?<![{letra}])" + p
        if de[-1:].isalpha():
            p = p + f"(?![{letra}])"
        partes.append(p)
    return re.compile("|".join(partes)), mapa


_GLOSSARIO: Dict[str, Tuple[Optional[Pattern[str]], Dict[str, str]]] = {
    "pt": _compilar(_EN_PT + [(cod, pt) for cod, pt, _ in _PT]),
    "en": _compilar([(cod, en) for cod, _, en in _PT] + [(pt, en) for _, pt, en in _PT]),
}
_REGEX_C = [(re.compile(p), pt, en) for p, pt, en in _REGEX]


def _traduzir_trecho(s: str, lang: str) -> str:
    rx, mapa = _GLOSSARIO[lang]
    if rx is not None:
        s = rx.sub(lambda m: mapa[m.group(0)], s)
    for crx, pt, en in _REGEX_C:
        s = crx.sub(en if lang == "en" else pt, s)
    return s


def traduzir_texto(texto: str, lang: Optional[str] = None) -> str:
    """Um texto de figura no idioma. O glossario so' tem frases inteiras (as com mathtext, como
    "T$^2$ 95% limit", estao escritas com o `$...$`), entao formulas soltas nao casam com nada."""
    if not texto or not any(c.isalpha() for c in texto):
        return texto
    return _traduzir_trecho(texto, idioma(lang))


def _textos(fig: Any) -> List[Any]:
    from matplotlib.text import Text
    return [t for t in fig.findobj(Text) if t.get_text()]


def _rotulos_fixos(eixo: Any) -> Optional[Dict[Any, str]]:
    """Rotulos de tick dados com set_xticklabels: o matplotlib recria os Text dos ticks a cada desenho
    a partir do formatter -- FixedFormatter (`seq`) ou, desde o 3.10, FuncFormatter sobre um dict."""
    fmt = eixo.get_major_formatter()
    seq = getattr(fmt, "seq", None)
    if isinstance(seq, list) and seq:
        return dict(enumerate(seq))
    func = getattr(fmt, "func", None)
    args = getattr(func, "args", ())
    if getattr(getattr(func, "func", None), "__name__", "") == "_format_with_dict" and args \
            and isinstance(args[0], dict):
        return args[0]
    return None


def textos_da_figura(fig: Any) -> List[str]:
    """Todos os textos visiveis da figura (titulos, eixos, legendas, caixas, rotulos de eixo fixos)."""
    saida = [t.get_text() for t in _textos(fig) if t.get_visible()]
    for ax in fig.get_axes():
        for eixo in (ax.xaxis, ax.yaxis):
            rot = _rotulos_fixos(eixo)
            if rot:
                saida.extend(str(x) for x in rot.values())
    return saida


def traduzir_figura(fig: Any, lang: Optional[str] = None) -> None:
    """Troca, no lugar, cada texto da figura pelo do idioma (padrao: o da sessao)."""
    lingua = idioma(lang)
    for t in _textos(fig):
        s = t.get_text()
        novo = traduzir_texto(s, lingua)
        if novo != s:
            t.set_text(novo)
    for ax in fig.get_axes():
        for eixo in (ax.xaxis, ax.yaxis):
            fmt = eixo.get_major_formatter()
            if isinstance(getattr(fmt, "seq", None), list) and fmt.seq:
                fmt.seq = [traduzir_texto(str(x), lingua) for x in fmt.seq]
                continue
            rot = _rotulos_fixos(eixo)
            if rot:
                for k in list(rot):
                    rot[k] = traduzir_texto(str(rot[k]), lingua)
