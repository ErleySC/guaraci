"""traducao_console.py -- o console do pipeline no idioma da sessao.

O nucleo (pipeline.py, auditoria, leitores, comandos de ANOVA/VIP/monitoramento) escreve o log do console em
portugues, e parte dele em ingles. Numa sessao em ingles -- o padrao de quem nunca escolheu idioma -- o
`guaraci run` misturava os dois na mesma tela. Traduzir na fonte mexeria em centenas de mensagens do nucleo
congelado (e no texto que o painel de progresso le por regex); aqui a traducao e' feita na SAIDA, linha a
linha, so' quando o idioma da sessao e' ingles.

Cobertura: as mensagens que `guaraci demo` e os 13 `guaraci exemplo CENARIO --rodar` imprimem
(tests/test_traducao_console.py roda os cenarios em ingles e acusa palavra em portugues que sobrar). Uma
mensagem sem par na tabela sai como esta, em portugues: o teste e' o que impede isso nos caminhos cobertos.

Nao traduz: caminhos de arquivo, nomes de classe e de coluna, chaves de configuracao (`max_lvs`,
`coluna_grupo`...), o arquivo de log gravado em disco (registro tecnico) nem o painel do menu (que tem os
proprios textos nos dois idiomas).
"""
from __future__ import annotations

import io
import re
import sys
from typing import Callable, Dict, List, Optional, Pattern, Tuple, Union

__all__ = ["traduzir_linha", "traduzir_texto", "TradutorSaida", "ativar", "ativo"]

_Repl = Union[str, Callable[["re.Match[str]"], str]]

#: rotulos do objetivo e das figuras que o modos_analise grava em PT.
_OBJ: Dict[str, str] = {"Classificacao": "Classification", "Quantificacao": "Quantification", "Exploratorio": "Exploratory"}
_FIGURAS: Dict[str, str] = {
    "Scores PLS-DA": "PLS-DA scores", "Matriz de confusao + metricas por classe": "Confusion matrix + per-class metrics",
    "Curvas ROC/AUC multiclasse": "Multiclass ROC/AUC curves", "Avaliacao em holdout externo": "External holdout evaluation",
    "DD-SIMCA (autenticacao de pureza)": "DD-SIMCA (purity authentication)",
    "Espectros medios por classe": "Mean spectra per class", "Scores PCA": "PCA scores", "Loadings PCA": "PCA loadings",
    "Biplot PCA (scores + loadings)": "PCA biplot (scores + loadings)",
    "Regressao PLS + figuras de merito analiticas": "PLS regression + analytical figures of merit",
}


#: descricao (PT) dos perfis que vem com o programa (perfis_matriz/*.yaml) -> ingles. Perfil escrito pelo
#: usuario fica como ele escreveu.
DESCRICOES_PERFIL_EN: Dict[str, str] = {
    "Colorimetria digital via camera de bancada (aquisicao controlada)":
        "Digital colorimetry with a bench camera (controlled acquisition)",
    "Colorimetria digital via foto de celular (aquisicao informal)":
        "Digital colorimetry with a phone photo (informal acquisition)",
    "Matriz nao especificada -- nenhum padrao de matriz aplicado":
        "Matrix not specified -- no matrix default applied",
    "NIR dispersivo, 1100-2498 nm": "Dispersive NIR, 1100-2498 nm",
    "FT-NIR de reflectancia difusa (cm-1)": "Diffuse-reflectance FT-NIR (cm-1)",
    "NIR de transmitancia, 8 mm de caminho optico (cm-1)": "Transmittance NIR, 8 mm optical path (cm-1)",
    "Colorimetria digital via scanner (documentos/laminas planas)":
        "Digital colorimetry with a scanner (flat documents/slides)",
    "Vis-NIR, 400-1100 nm (perfil declarado, nao verificado com dado real)":
        "Vis-NIR, 400-1100 nm (declared profile, not checked with real data)",
}


def descricao_perfil(descricao: str, lang: Optional[str] = None) -> str:
    """Descricao de um perfil no idioma da sessao (so' os perfis que vem com o programa tem traducao)."""
    from guaraci.idioma_relatorio import idioma
    if idioma(lang) != "en":
        return descricao
    return " — ".join(DESCRICOES_PERFIL_EN.get(p.strip(), p.strip()) for p in str(descricao).split(" — "))


#: (padrao PT, substituicao EN). Aplicados EM ORDEM, cada um com re.sub sobre a linha inteira: os mais
#: especificos (frases longas) vem antes dos genericos (palavras soltas, rotulos de coluna).
_TABELA: List[Tuple[str, _Repl]] = [
    # ---------------------------------------------------------------- comandos e cabecalho (guaraci.py)
    (r"^Exemplo '([^']+)' gerado em (.*) \(dados sinteticos\)\.$", r"Example '\1' created in \2 (synthetic data)."),
    (r"^Para rodar:$", "To run:"),
    (r"^\[GUARACI\] Executando com (.*) \.\.\.$", r"[GUARACI] Running with \1 ..."),
    (r"^\[GUARACI\] Concluido\. Saida em: ", "[GUARACI] Done. Output in: "),
    (r"^Arquivos gravados:$", "Files written:"),
    (r"^\[MODO\] Objetivo cientifico: (\S+)\s+\(nivel (\S+), objetivo configurado: (\S+)\)$",
     lambda m: f"[MODE] Scientific objective: {_OBJ.get(m.group(1), m.group(1))}  (level {m.group(2)}, "
               f"configured objective: {m.group(3)})"),
    (r"^\[MODO\] Figuras pertinentes a este objetivo \((\d+)\): (.*)$",
     lambda m: f"[MODE] Figures relevant to this objective ({m.group(1)}): "
               + "; ".join(_FIGURAS.get(f.strip(), f.strip()) for f in m.group(2).split(";"))),
    (r"^\[HARDWARE\] RAM total: (.*?)  livre: (.*?)  CPU: ", r"[HARDWARE] RAM total: \1  free: \2  CPU: "),
    (r"^\[INFO\] Perfil de tecnica: (\S+) \((.*?)\) \| eixo (.*?) \| pre-proc ",
     lambda m: f"[INFO] Technique profile: {m.group(1)} ({descricao_perfil(m.group(2), 'en')}) | axis "
               f"{m.group(3).replace('(unidade nao declarada)', '(unit not declared)')} | pre-proc "),
    (r"^\[WARNING\] Grouping guarantee: (\S+) -- ver 'Grouping guarantee' no resumo/model card para o que isso "
     r"significa para a validacao desta execucao\.$",
     r"[WARNING] Grouping guarantee: \1 -- see 'Grouping guarantee' in the summary/model card for what this "
     r"means for the validation of this run."),
    (r"^\[INFO\] Grouping guarantee: (\S+) \((\d+) grupos de amostra fisica\)\.$",
     r"[INFO] Grouping guarantee: \1 (\2 physical-sample groups)."),
    (r"^\[INFO\] Truncamento espectral (.*?): (\d+)/(\d+) variaveis removidas, restam (\d+) \((.*)\)\.$",
     r"[INFO] Spectral truncation \1: \2/\3 variables removed, \4 left (\5)."),
    (r"^\[INFO\] Saida: ", "[INFO] Output: "),
    (r"^\[INFO\] Subpastas: ", "[INFO] Subfolders: "),
    (r"^\[INFO\] Metadados salvos: ", "[INFO] Metadata saved: "),
    (r"^\[INFO\] Auditoria de delineamento: (\d+) critico\(s\), (\d+) aviso\(s\) -- ver model card para o "
     r"detalhe por checagem\.$",
     r"[INFO] Design audit: \1 critical, \2 warning(s) -- see the model card for the detail of each check."),
    (r"^(\s*)\[AUDITORIA\]\[(AVISO|CRITICO|INFO)\] (\w+): (.*)$", lambda m: _auditoria(m)),
    (r"^\[INFO\] Validacao group-aware ATIVA: (\d+) grupos unicos via mae_id\.$",
     r"[INFO] Group-aware validation ON: \1 unique groups via mae_id."),
    (r"^\[INFO\] GroupKFold desabilitado: mae_id indisponivel ou grupos insuficientes — usando StratifiedKFold "
     r"\(estratificada\)\.$",
     "[INFO] GroupKFold disabled: mae_id unavailable or too few groups -- using StratifiedKFold (stratified)."),
    (r"^\[INFO\] Hold-out \((.*)\): (\d+) amostras reservadas \(frac=(.*?)\)\. Pipeline rodara em (\d+) amostras\.$",
     lambda m: f"[INFO] Hold-out ({_holdout(m.group(1))}): {m.group(2)} samples held out (frac={m.group(3)}). "
               f"The pipeline will run on {m.group(4)} samples."),
    (r"^\[INFO\] Objetivo exploratorio: sem holdout \(os graficos usam todas as amostras\)\.$",
     "[INFO] Exploratory objective: no holdout (the plots use every sample)."),
    (r"^\[INFO\] Amostras : ", "[INFO] Samples  : "),
    (r"^\[INFO\] Variaveis: ", "[INFO] Variables: "),
    (r"^\[INFO\] Conc\.    : nao$", "[INFO] Conc.    : no"),
    (r"^\[INFO\] Conc\.    : sim$", "[INFO] Conc.    : yes"),
    (r"^\[INFO\] mae_id   : nao$", "[INFO] mae_id   : no"),
    (r"^\[INFO\] mae_id   : sim \((\d+) grupos\)$", r"[INFO] mae_id   : yes (\1 groups)"),
    (r"^\[INFO\] Unidade de agrupamento 'mae_id': cada sessao de coleta tem uma classe so', o que pede validar "
     r"por sessao, mas isso e' impossivel agora: ha' classe\(s\) com uma unica sessao \((.*?)\) ou menos de (\d+) "
     r"sessoes no total\. Os numeros por mae_id sao OTIMISTAS; colete cada classe em mais sessoes\.$",
     r"[INFO] Grouping unit 'mae_id': each collection session holds a single class, which calls for validating "
     r"by session, but that is impossible now: there are class(es) with a single session (\1) or fewer than \2 "
     r"sessions in total. The numbers by mae_id are OPTIMISTIC; collect each class in more sessions."),
    (r"^\[INFO\] Synthetic MODE — cenario '([^']+)': (\d+) espectros, (\d+) amostras fisicas, (\d+) lotes; "
     r"variancia dentro das amostras \(replicas\) = (.*?) do total\.$",
     r"[INFO] Synthetic MODE -- scenario '\1': \2 spectra, \3 physical samples, \4 batches; within-sample "
     r"variance (replicates) = \5 of the total."),
    (r"^\[INFO\] N2 \(autenticação por espécie\): rótulos de espécie preservados para DD-SIMCA one-class "
     r"\(puros=(\d+) \| adulterados=(\d+)\)\. DD-SIMCA forçado para mode 'puros'\.$",
     r"[INFO] N2 (authentication by species): species labels kept for one-class DD-SIMCA (pure=\1 | "
     r"adulterated=\2). DD-SIMCA forced to mode 'puros'."),
    (r"^\[INFO\] EEM: (\d+) amostras carregadas \(achatadas p/ vetor 1D, (\d+) pontos\)\.$",
     r"[INFO] EEM: \1 samples loaded (flattened to a 1D vector, \2 points)."),
    (r"^\[INFO\] GC-MS ANDI-MS: (\d+) amostras, (\d+) classe\(s\)\.$", r"[INFO] GC-MS ANDI-MS: \1 samples, \2 class(es)."),
    (r"^\[INFO\] (\d+) cubos hiperespectrais carregados, (\d+) bandas\.$", r"[INFO] \1 hyperspectral cubes loaded, \2 bands."),
    (r"^\[INFO\] HSI: Faixa: todas as (\d+) bandas \((.*)\)\.$", r"[INFO] HSI: Range: all \1 bands (\2)."),
    (r"^\[INFO\] HSI: Pre-processamento: SNV por pixel \+ centragem na media \(PLS\)\.$",
     "[INFO] HSI: Preprocessing: per-pixel SNV + mean-centering (PLS)."),
    (r"^\[INFO\] (\d+) imagens carregadas, (\d+) features \(cor\)\.$", r"[INFO] \1 images loaded, \2 features (colour)."),
    (r"^\[PROTOTIPO\] mode='imagem' \(colorimetria digital\) NAO e validado com dataset real\. A validacao so' e' "
     r"agrupada por amostra fisica com subpasta por amostra ou amostras\.csv \(ver 'Grouping guarantee' abaixo\); "
     r"sem isso os relatorios sairao carimbados\.$",
     "[PROTOTYPE] mode='imagem' (digital colorimetry) is NOT validated with a real dataset. Validation is only "
     "grouped by physical sample with one subfolder per sample or amostras.csv (see 'Grouping guarantee' below); "
     "without that the reports will be stamped."),
    (r"^\[AVISO\] imagem: em (\d+)/(\d+) fotos o fundo \(estimado nas bordas\) ocupa mais de (\d+%) da imagem "
     r"\(mediana (\d+%)\) e as estatisticas de cor misturam fundo e objeto; use imagem_segmentacao: otsu \(ou "
     r"distancia_fundo\) ou imagem_recorte\.$",
     r"[WARNING] image: in \1/\2 photos the background (estimated at the borders) covers more than \3 of the "
     r"image (median \4) and the colour statistics mix background and object; use imagem_segmentacao: otsu (or "
     r"distancia_fundo) or imagem_recorte."),
    (r"^\[AVISO\] imagem: a cor do fundo \(borda das fotos\) difere entre classes \((.*?): distancia RGB (\S+) "
     r"contra dispersao interna (\S+)\): o modelo pode estar aprendendo o fundo e nao o objeto\. Use "
     r"imagem_segmentacao ou fotografe todas as classes sobre o mesmo fundo \(heuristica\)\.$",
     r"[WARNING] image: the background colour (photo borders) differs between classes (\1: RGB distance \2 "
     r"against internal spread \3): the model may be learning the background, not the object. Use "
     r"imagem_segmentacao or photograph every class on the same background (heuristic)."),
    (r"^\[AVISO\] n_splits ajustado para (\d+) \(min de grupos/classe = (\d+)\)\.$",
     r"[WARNING] n_splits adjusted to \1 (min groups/class = \2)."),
    (r"^\[AVISO\] n_splits_cv=(\d+) excede minimo por classe \((\d+)\)\. Usando (\d+)\.$",
     r"[WARNING] n_splits_cv=\1 exceeds the per-class minimum (\2). Using \3."),
    # ---------------------------------------------------------------- etapas do pipeline
    (r"^\[1/7\] Pre-processamento \(preset=", "[1/7] Preprocessing (preset="),
    (r"^(\s*)\[AVISO\] Faixa espectral: so' (\S+) das (\d+) variaveis carregam sinal \(SNR>=(\d+)\)\.$",
     r"\1[WARNING] Spectral range: only \2 of the \3 variables carry signal (SNR>=\4)."),
    (r"^(\s*)regiao ruidosa: ", r"\1noisy region: "),
    (r"^(\s*)regiao morta: ", r"\1dead region: "),
    (r"^(\s*)faixa com sinal: (.*) \(atual: (.*)\)$", r"\1range with signal: \2 (current: \3)"),
    (r"^(\s*)Considere reduzir (\S+) e reexecutar; compare Q2 antes de adotar\.$",
     r"\1Consider narrowing \2 and running again; compare Q2 before adopting it."),
    (r"^(\s*)LVs otimas \(Wold parcimonia\): (\d+) \(min-RMSECV em (\d+) LVs, delta <(.*?)\)$",
     r"\1Optimal LVs (Wold parsimony): \2 (min-RMSECV at \3 LVs, delta <\4)"),
    (r"^(\s*)LVs otimas: ", r"\1Optimal LVs: "),
    (r"^(\s*)\[INFO\] CV aninhada ativada \(selecao_lv_cv_aninhada=True, default a partir da (\S+) -- corrige o "
     r"vies de reusar o mesmo dado p/ escolher n_opt e avaliar a metrica reportada\)\. Pode levar (\S+) mais "
     r"tempo nesta etapa; desative no config\.yaml \(selecao_lv_cv_aninhada: false\) p/ iteracao rapida, "
     r"aceitando a metrica otimista\.$",
     r"\1[INFO] Nested CV on (selecao_lv_cv_aninhada=True, default since \2 -- removes the bias of reusing the "
     r"same data to choose n_opt and to measure the reported metric). This step may take \3 longer; switch it "
     r"off in config.yaml (selecao_lv_cv_aninhada: false) for quick iteration, accepting the optimistic metric."),
    (r"^(\s*)\[INFO\] max_lvs=(\d+) limitado a (\d+) pelo tamanho do menor fold de treino \((\d+) amostras\)\.$",
     r"\1[INFO] max_lvs=\2 capped at \3 by the size of the smallest training fold (\4 samples)."),
    (r"^(\s*)\[2b/7\] CV aninhada: fold externo (\d+)/(\d+) concluido \((.*)\)$",
     r"\1[2b/7] Nested CV: outer fold \2/\3 done (\4)"),
    (r"^(\s*)CV aninhada: n_opts por fold externo = ", r"\1Nested CV: n_opts per outer fold = "),
    (r"^(\s*)Balanced accuracy  NAIVE \(n_opt=(\d+) fixo, reportada ate' (\S+)\)",
     r"\1Balanced accuracy  NAIVE (fixed n_opt=\2, reported up to \3)"),
    (r"^(\s*)Balanced accuracy  ANINHADA \(n_opt por fold, honesta, reportada a partir da (\S+)\)",
     r"\1Balanced accuracy  NESTED (n_opt per fold, honest, reported since \2)"),
    (r"^(\s*)\[ATENCAO\] LVs otimas \((\d+)\) == max_lvs \((\d+)\): RMSECV ainda nao atingiu plateau\. Aumente "
     r"max_lvs \(ex: (\d+)\) e rode novamente\.$",
     r"\1[ATTENTION] Optimal LVs (\2) == max_lvs (\3): RMSECV has not reached a plateau yet. Raise max_lvs "
     r"(e.g. \4) and run again."),
    (r"^\[3/7\] PCA exploratoria$", "[3/7] Exploratory PCA"),
    (r"^(\s*)PC1: (.*?)  PC2: (.*?)  acumulado\((\d+)\): ", r"\1PC1: \2  PC2: \3  cumulative(\4): "),
    (r"^\[4/7\] Teste de permutacao \(Y-randomization, n=(\d+)\)$", r"[4/7] Permutation test (Y-randomization, n=\1)"),
    (r"^\[4/7\] Teste de permutacao — PULADO: objective=(\S+)\. Significancia de classificacao nao e' pertinente "
     r"fora do modo Classificacao \(economiza (\d+) refits de CV\)\.$",
     lambda m: f"[4/7] Permutation test -- SKIPPED: objective={_OBJ.get(m.group(1), m.group(1))}. Classification "
               f"significance does not apply outside Classification mode (saves {m.group(2)} CV refits)."),
    (r"^(\s*)Bal\.Acc observada = (.*?)\|  p = (.*?)\|  bal\.acc media H0 = ",
     r"\1Observed bal.acc = \2|  p = \3|  mean bal.acc H0 = "),
    (r"^(\s*)Iteracoes validas: ", r"\1Valid iterations: "),
    (r"^\[5/7\] Metricas finais \(CV\):$", "[5/7] Final metrics (CV):"),
    (r"^\[5/7\] Classificacao de especie que escolhe o modelo de regressao \(CV; a resposta quantificada vem em "
     r"\[7/7\]\):$",
     "[5/7] Species classification that picks the regression model (CV; the quantified response comes in [7/7]):"),
    (r"\[CI nao estimavel: sem variacao entre reamostragens -- todas deram o mesmo valor \(ex\.: acerto perfeito ou "
     r"grupos insuficientes por classe\)\]",
     "[CI not estimable: no variation between resamples -- all gave the same value (e.g. perfect hit rate or "
     "too few groups per class)]"),
    (r"^\[5b/7\] BCa CI 95% — PULADO: objective=(\S+)\. Intervalo de confianca de metricas de classificacao nao "
     r"e' pertinente fora do modo Classificacao\.$",
     lambda m: f"[5b/7] BCa CI 95% -- SKIPPED: objective={_OBJ.get(m.group(1), m.group(1))}. A confidence interval "
               f"of classification metrics does not apply outside Classification mode."),
    (r"^(\s*)\[AVISO\] Apenas UMA classe presente: as metricas globais de classificacao \(accuracy, kappa, F1\) sao "
     r"degeneradas e nao medem desempenho\. Ignore-as neste conjunto\.$",
     r"\1[WARNING] Only ONE class present: the global classification metrics (accuracy, kappa, F1) are "
     r"degenerate and do not measure performance. Ignore them for this set."),
    (r"^\[6/7\] Gerando figuras\.\.\.$", "[6/7] Generating figures..."),
    (r"^\[Razao de seletividade\] Selectivity Ratio \+ contribuicao dos escores\.\.\.$",
     "[Selectivity ratio] Selectivity Ratio + score contribution..."),
    (r"^\[6c/7\] Avaliacao em holdout \((\d+) amostras\)\.\.\.$", r"[6c/7] Holdout evaluation (\1 samples)..."),
    (r"^(\s*)\[AVISO\] alpha=(\S+) NAO e' alcancavel com (\d+) amostra\(s\) de calibracao independentes: exigiria "
     r"k=(\d+) escores ordenados, mas so' ha' (\d+)\. Menor alpha garantivel: (\S+) \(precisa de n>=(\d+) para "
     r"alpha=(\S+)\)\. Limiar NAO definido -- reportar como nao-estimavel, nunca como 'aceita tudo'\.$",
     r"\1[WARNING] alpha=\2 is NOT reachable with \3 independent calibration sample(s): it would need k=\4 "
     r"ordered scores, but there are only \5. Smallest guaranteed alpha: \6 (needs n>=\7 for alpha=\8). "
     r"Threshold NOT defined -- report it as not estimable, never as 'accepts everything'."),
    (r"^\[7d/7\] Dominio de aplicabilidade: calibrando\.\.\.$", "[7d/7] Applicability domain: calibrating..."),
    (r"^(\s*)\[7d/7\] Dominio de aplicabilidade: fold (\d+)/(\d+) concluido \((.*)\)$",
     r"\1[7d/7] Applicability domain: fold \2/\3 done (\4)"),
    (r"^(\s*)\[AVISO\] Dominio de aplicabilidade calibrado com poucos (grupos \(amostras fisicas\)|amostras) por "
     r"classe \(minimo (\d+); (\d+) classe\(s\) abaixo de (\d+)\)\. Nesse regime ele costuma rejeitar muitas "
     r"amostras legitimas: a taxa esperada de alarmes e' desconhecida e pode ser muito maior que os (\S+) "
     r"nominais\. Use a taxa de rejeicao em validacao cruzada como estimativa e nao trate 'fora do dominio' como "
     r"prova de problema na amostra\.$",
     lambda m: f"{m.group(1)}[WARNING] Applicability domain calibrated with few "
               f"{'groups (physical samples)' if m.group(2).startswith('grupos') else 'samples'} per class "
               f"(minimum {m.group(3)}; {m.group(4)} class(es) below {m.group(5)}). In this regime it tends to "
               f"reject many legitimate samples: the expected alarm rate is unknown and may be much higher than "
               f"the nominal {m.group(6)}. Use the cross-validation rejection rate as the estimate and do not take "
               f"'outside the domain' as proof of a problem with the sample."),
    (r"^(\s*)\[AD\] Taxa esperada de alarmes em amostras legitimas \(rejeicao do dominio em CV por grupo\): ",
     r"\1[AD] Expected alarm rate on legitimate samples (domain rejection in group CV): "),
    (r"^(\s*)\[Pureza\] DD-SIMCA por especie \(Detectar/pureza\): (\d+) especies calibradas, (\d+) com alpha "
     r"sustentado por evidencia \(LOGO medido ou >= (\d+) grupos\); nas demais a taxa de falsa rejeicao de puros "
     r"novos e' desconhecida e pode ser alta\.$",
     r"\1[Purity] DD-SIMCA per species (Detect/purity): \2 species calibrated, \3 with alpha backed by evidence "
     r"(measured LOGO or >= \4 groups); for the others the false-rejection rate of new pure samples is unknown "
     r"and may be high."),
    (r"^\[DDSimca\] Classe '(.*?)': (\d+) amostra\(s\) de treino atipica\(s\) \(indices (\[.*?\]) de (\d+), "
     r"deteccao robusta mediana/MAD\)\. Nao removidas automaticamente -- considere investigar essas replicas\.$",
     r"[DDSimca] Class '\1': \2 atypical training sample(s) (indices \3 of \4, robust median/MAD detection). "
     r"Not removed automatically -- consider looking into these replicates."),
    (r"^\[DDSimca\] Classe '(.*?)': limiar calibrado com apenas (\d+) amostra\(s\) fisica\(s\) independente\(s\) "
     r"\(mae_id\)\. Nh=(\S+), Nq=(\S+) -- com tao poucos grupos a taxa de falsa rejeicao de amostras puras novas "
     r"e' desconhecida e pode ser muito alta; nao tome a regiao de aceitacao como margem de seguranca\.$",
     r"[DDSimca] Class '\1': threshold calibrated with only \2 independent physical sample(s) (mae_id). Nh=\3, "
     r"Nq=\4 -- with so few groups the false-rejection rate of new pure samples is unknown and may be very high; "
     r"do not take the acceptance region as a safety margin."),
    (r"^\[DD-SIMCA\] \((.*)\) — (\d+) amostras de treino$", r"[DD-SIMCA] (\1) -- \2 training samples"),
    (r"^(\s*)Classe(\s+)sens\(LOGO\)(\s+)grupos(\s+)esp\(adult\)$", r"\1Class \2sens(LOGO)\3groups\4spec(adult)"),
    (r"\(puros=(\d+), adult=(\d+)\)$", r"(pure=\1, adult=\2)"),
    (r"^(\s*)\[AVISO\] (.+?): Sensibilidade estimada por LOGO com apenas (\d+) grupos de replica\. Incerteza alta; "
     r"IC bootstrap nao e confiavel neste regime\. Interpretar como exploratoria\.$",
     r"\1[WARNING] \2: Sensitivity estimated by LOGO with only \3 replicate groups. High uncertainty; the "
     r"bootstrap CI is not reliable in this regime. Read it as exploratory."),
    (r"^(\s*)Desconhecidos: (\d+)  \|  Ambiguos: (\d+)$", r"\1Unknown: \2  |  Ambiguous: \3"),
    (r"^(\s*)\[Identificacao\] (\d+) combinacoes especie x adulterante calibradas, (\d+) com cobertura VALIDADA\. "
     r"Ver model card para o detalhe por combinacao\.$",
     r"\1[Identification] \2 species x adulterant combinations calibrated, \3 with VALIDATED coverage. See the "
     r"model card for the detail of each combination."),
    (r"^(\s*)\[Identificacao\] NENHUMA combinacao especie x adulterante tem cobertura estatistica validada neste "
     r"dataset -- Identificar/Quantificar em amostra nova ficam bloqueados por padrao \(ver cobertura_status\)\.$",
     r"\1[Identification] NO species x adulterant combination has validated statistical coverage in this "
     r"dataset -- Identify/Quantify on a new sample are blocked by default (see cobertura_status)."),
    (r"^\[Identificar\] mae_id ausente -- ensemble de identificacao NAO pode ser calibrado \(sem ele nao ha' como "
     r"contar sessoes independentes\)\. Retornando ensemble vazio\.$",
     "[Identify] mae_id missing -- the identification ensemble can NOT be calibrated (without it independent "
     "sessions cannot be counted). Returning an empty ensemble."),
    (r"^\[INFO\] Outliers \(", "[INFO] Outliers ("),
    (r"^\[7/7\] PLS regressao — pulado \(sem coluna de concentracao\)$", "[7/7] PLS regression -- skipped (no concentration column)"),
    (r"^\[7/7\] PLS regressao — PULADA: objective=(\S+) \(level=(\S+)\)\. A regressao da resposta pertence ao Modo "
     r"Quantificacao \(N3\); nos modos Exploratorio/Classificacao ela seria um resultado fora de escopo \(e em N1 as "
     r"(\d+) classes dominam o sinal da resposta, produzindo R2~0\)\.$",
     lambda m: f"[7/7] PLS regression -- SKIPPED: objective={_OBJ.get(m.group(1), m.group(1))} (level={m.group(2)}). "
               f"Regression of the response belongs to Quantification mode (N3); in Exploratory/Classification "
               f"modes it would be an out-of-scope result (and in N1 the {m.group(3)} classes dominate the response "
               f"signal, giving R2~0)."),
    (r"^\[7/7\] PLS regressao POR ESPECIE \((\d+) especies — calibracao separada para evitar confusao "
     r"inter-especies\)$",
     r"[7/7] PLS regression PER SPECIES (\1 species -- separate calibration to avoid confusion between species)"),
    (r"^\[7/7\] PLS regressao \(alvo=resposta, (\d+) com resposta > 0 \+ (\d+) com resposta = 0\)$",
     r"[7/7] PLS regression (target=response, \1 with response > 0 + \2 with response = 0)"),
    (r"^(\s*)Modo de rotulo na quantificacao: cego$", r"\1Label mode in quantification: blind"),
    (r"^(\s*)Modo de rotulo na quantificacao: (\S+)$", r"\1Label mode in quantification: \2"),
    (r"^(\s*)Split cal/val: GroupShuffleSplit por mae_id \((\d+) cal / (\d+) val\)$",
     r"\1Cal/val split: GroupShuffleSplit by mae_id (\2 cal / \3 val)"),
    (r"^(\s*)CV interna: GroupKFold n_splits=(\d+) \((\d+) grupos na calibracao\)$",
     r"\1Inner CV: GroupKFold n_splits=\2 (\3 groups in calibration)"),
    (r"->  controle de processo / quantificacao$", "->  process control / quantification"),
    (r"\(unidade do teor; aproximado: so' o ruido de replica; sem leverage nem erro de referencia -> tende a "
     r"SUBestimar\)",
     "(unit of the response; approximate: replicate noise only; no leverage nor reference error -> tends to "
     "UNDERestimate)"),
    (r"^(\s*)DModY critico \(SIMCA\): (\S+)  \|  amostras fora: ", r"\1Critical DModY (SIMCA): \2  |  samples outside: "),
    (r"\[numero publicavel; o RMSEP interno abaixo vem de split 70/30\]",
     "[publishable number; the internal RMSEP below comes from a 70/30 split]"),
    (r"\[numero publicavel; o RMSEP interno vem de split 70/30\]",
     "[publishable number; the internal RMSEP comes from a 70/30 split]"),
    (r"^(\s*)RMSEP \(HOLDOUT externo, ", r"\1RMSEP (external HOLDOUT, "),
    (r"^(\s*)Linearidade \(falta de ajuste\): ", r"\1Linearity (lack of fit): "),
    (r"^(\s*)\[RESSALVA\] rejeicao NAO conclusiva: erro puro estimado so' de replicas tecnicas do mesmo mae_id "
     r"\(subestima o erro de preparo\) e preparos independentes do mesmo teor contam como niveis distintos: o teste "
     r"tende a rejeitar linearidade -- use nivel_por='conc' se ha' preparos independentes",
     r"\1[CAVEAT] rejection NOT conclusive: pure error estimated only from technical replicates of the same mae_id "
     r"(underestimates the preparation error) and independent preparations of the same level count as distinct "
     r"levels: the test tends to reject linearity -- use nivel_por='conc' if there are independent preparations"),
    (r"^(\s*)Especies modeladas: ", r"\1Species modelled: "),
    (r"^(\s*)\[Modelo\] Atualizado com regressao por especie \((\d+) especies\) -> ",
     r"\1[Model] Updated with per-species regression (\2 species) -> "),
    (r"\[(\S+)-(\S+), 95% IC\]", r"[\1-\2, 95% CI]"),
    (r"^(\s*)Faixa de validacao: (.*)  DP=", r"\1Validation range: \2  SD="),
    (r"^(\s*)\[!\] LOD \((\S+)\) esta ABAIXO do RMSEP \((\S+)\): o LOD calculado so' com ruido de replica e' "
     r"irreal\. NAO o cite como limite de deteccao; use o RMSEP como piso do que o modelo distingue\.$",
     r"\1[!] LOD (\2) is BELOW the RMSEP (\3): an LOD computed from replicate noise only is unrealistic. Do NOT "
     r"cite it as a detection limit; use the RMSEP as the floor of what the model distinguishes."),
    (r"^(\s*)Pipeline concluido\.$", r"\1Pipeline finished."),
    (r"^(\s*)Resultados em: ", r"\1Results in: "),
    (r"^(\s*)\[!\] Validacao agrupada pelo grupo DECLARADO pelo usuario \(garantia MEDIA\): o software nao verifica "
     r"que os grupos sao de fato independentes\.$",
     r"\1[!] Validation grouped by the group DECLARED by the user (MEDIUM guarantee): the software does not check "
     r"that the groups are really independent."),
    (r"^(\s*)\[!\] VALIDACAO NAO AGRUPADA: sem chave de grupo \(coluna_grupo/arquivo_grupos\), a CV usou "
     r"StratifiedKFold e replicas da mesma amostra podem estar nos dois lados\. Trate as metricas como "
     r"EXPLORATORIAS\.$",
     r"\1[!] VALIDATION NOT GROUPED: no group key (coluna_grupo/arquivo_grupos), the CV used StratifiedKFold and "
     r"replicates of the same sample may sit on both sides. Treat the metrics as EXPLORATORY."),
    # ---------------------------------------------------------------- comandos sem pipeline
    (r"^\[ANOVA (.+?)\] Welch ANOVA \+ Games-Howell \(variancias='desiguais'\)$",
     r"[ANOVA \1] Welch ANOVA + Games-Howell (variancias='desiguais': unequal variances)"),
    (r"^\[ANOVA (.+?)\] residuos com desvio de normalidade \(Shapiro p=(\S+)\); considere a transformacao log ou o "
     r"Kruskal-Wallis$",
     r"[ANOVA \1] residuals deviate from normality (Shapiro p=\2); consider the log transform or Kruskal-Wallis"),
    (r"^\[ANOVA (.+?)\] (\d+) linha\(s\) com resposta ausente/NaN descartada\(s\)$",
     r"[ANOVA \1] \2 row(s) with missing/NaN response dropped"),
    (r"^\[INFO\] Multielementar: (\d+) amostra\(s\), (\d+) elemento\(s\) \((.*?)\); (\d+) excluido\(s\) por censura "
     r"acima de (\S+): (.*?)\. EM VALIDACAO -- mecanismo verificado em dados sinteticos; padroes sao ponto de "
     r"partida, com poucas unidades independentes por grupo; sem comparacao com resultados publicados\.$",
     lambda m: f"[INFO] Multi-element: {m.group(1)} sample(s), {m.group(2)} element(s) ({m.group(3)}); "
               f"{m.group(4)} excluded by censoring above {m.group(5)}: "
               f"{'none' if m.group(6) == 'nenhum' else m.group(6)}. UNDER VALIDATION -- mechanism checked on "
               f"synthetic data; defaults are a starting point, with few independent units per group; no "
               f"comparison with published results."),
    (r"^\[INFO\] Multielementar: ANOVA por elemento \(Tukey/Games-Howell dentro de cada resposta; Benjamini-Hochberg "
     r"\(FDR\) entre respostas\); (\d+) elemento\(s\) com diferenca entre grupos \(FDR < (\S+)\); sensibilidade da "
     r"censura -- muda: (\S+)$",
     r"[INFO] Multi-element: ANOVA per element (Tukey/Games-Howell within each response; Benjamini-Hochberg (FDR) "
     r"across responses); \1 element(s) differing between groups (FDR < \2); censoring sensitivity -- changes: \3"),
    (r"^(\s*)(.+?) x (.+?): diferenca (\S+) \(CI95 (\S+) a (\S+); p ajustado (\S+)\) ",
     r"\1\2 x \3: difference \4 (CI95 \5 to \6; adjusted p \7) "),
    (r"\(CI95 bootstrap (\S+) a (\S+)\)", r"(CI95 bootstrap \1 to \2)"),
    (r"^(\s*)recommendation: sem desvio relevante de premissas -> ", r"\1recommendation: no relevant departure from the assumptions -> "),
    (r"^(\s*)warning: Welch ANOVA \+ Games-Howell \(variancias='desiguais'\)$",
     r"\1warning: Welch ANOVA + Games-Howell (variancias='desiguais': unequal variances)"),
    (r"^Monitoramento: (\d+) amostra\(s\) de referencia em (\d+) lote\(s\); (\d+) componente\(s\) do PCA; alpha = ",
     r"Monitoring: \1 reference sample(s) in \2 batch(es); \3 PCA component(s); alpha = "),
    (r"^Taxa que a calibracao rejeita em validacao cruzada: ", "Rate the calibration rejects in cross-validation: "),
    (r"^- lote (.+?): (\d+)/(\d+) fora do dominio -> em controle ", r"- batch \1: \2/\3 outside the domain -> in control "),
    (r"^- lote (.+?): (\d+)/(\d+) fora do dominio -> ALARME ", r"- batch \1: \2/\3 outside the domain -> ALARM "),
    (r"^Alarme diz que o processo mudou em relacao a referencia; a causa \(instrumento, materia-prima, operacao\) "
     r"precisa ser investigada no processo\.$",
     "An alarm says the process changed relative to the reference; the cause (instrument, raw material, "
     "operation) has to be investigated in the process."),
    (r"^Transferencia PDS \(janela (\d+)\), (\d+) amostras de transferencia, alpha (\S+) \(escolhido por validacao\)$",
     r"PDS transfer (window \1), \2 transfer samples, alpha \3 (chosen by validation)"),
    (r"^Transferencia DS, (\d+) amostras de transferencia, alpha (\S+) \(escolhido por validacao\)$",
     r"DS transfer, \1 transfer samples, alpha \2 (chosen by validation)"),
    (r"^Diferenca escravo x mestre \(RMS, deixando cada amostra de fora\): (\S+) sem correcao, (\S+) com correcao$",
     r"Slave x master difference (RMS, leaving each sample out): \1 without correction, \2 with correction"),
    (r"^(\d+) espectro\(s\) convertido\(s\): ", r"\1 spectrum/spectra converted: "),
    (r"^Use o CSV convertido com o modelo calibrado no mestre e confira o erro em amostras de referencia\.$",
     "Use the converted CSV with the model calibrated on the master and check the error on reference samples."),
    (r"^VIP com estabilidade \(PLS-DA\); (\d+) variaveis, (\d+) linhas, (\d+) grupos; (\d+) componente\(s\); (\d+) "
     r"ajustes de dobra \((\d+) dobras x (\d+) repeticoes\)$",
     r"VIP with stability (PLS-DA); \1 variables, \2 rows, \3 groups; \4 component(s); \5 fold fits (\6 folds x "
     r"\7 repeats)"),
    (r"^conferencia: soma de VIP\^2 = (\S+) \(esperado: (\d+) variaveis\)$", r"check: sum of VIP^2 = \1 (expected: \2 variables)"),
    (r"^(\d+) variavel\(is\) com VIP > (\S+) em >= (\S+) das dobras \(estavel acima\): ",
     r"\1 variable(s) with VIP > \2 in >= \3 of the folds (stable above): "),
    (r"^(\d+) variavel\(is\) instavel\(is\) \(VIP > (\S+) em (\S+) a (\S+) das dobras\): ",
     r"\1 unstable variable(s) (VIP > \2 in \3 to \4 of the folds): "),
    (r"^(\d+) variavel\(is\) com VIP > (\S+) em menos de (\S+) das dobras \(estavel abaixo\)",
     r"\1 variable(s) with VIP > \2 in fewer than \3 of the folds (stable below)"),
    (r" \(formato portatil, sem pickle\)$", " (portable format, no pickle)"),
    # ---------------------------------------------------------------- por ultimo: prefixos soltos
    (r"^(\s*)\[AVISO\] ", r"\1[WARNING] "),
]

def _holdout(descr: str) -> str:
    return (descr.replace("replicas juntas, por classe", "replicates together, per class")
                 .replace(" puros preservados", " pure samples kept"))


def _auditoria(m: "re.Match[str]") -> str:
    from guaraci.idioma_relatorio import achado
    sev = {"AVISO": "WARNING", "CRITICO": "CRITICAL", "INFO": "INFO"}[m.group(2)]
    return f"{m.group(1)}[AUDIT][{sev}] {m.group(3)}: {achado(m.group(4), 'en')}"


_TABELA_RE: List[Tuple[Pattern[str], _Repl]] = [(re.compile(p), r) for p, r in _TABELA]

_ATIVO = [False]


def ativo() -> bool:
    """True quando a saida do console esta sendo traduzida (sessao em ingles e `ativar()` chamado)."""
    return _ATIVO[0]


def traduzir_linha(linha: str, lang: Optional[str] = None) -> str:
    """A linha no idioma `lang` (padrao: o da sessao). Em portugues volta como esta."""
    from guaraci.idioma_relatorio import idioma
    if idioma(lang) != "en" or not linha.strip():
        return linha
    for padrao, repl in _TABELA_RE:
        linha = padrao.sub(repl, linha)
    return linha


def traduzir_texto(texto: str, lang: Optional[str] = None) -> str:
    """Texto de varias linhas, linha a linha."""
    return "\n".join(traduzir_linha(x, lang) for x in str(texto).split("\n"))


class TradutorSaida(io.TextIOBase):
    """Envolve um stream de texto e traduz cada linha completa antes de escrever. O resto de uma linha
    (sem "\\n" ainda) fica guardado ate' a linha fechar ou ate' `flush()`."""

    def __init__(self, destino):
        super().__init__()
        self._destino = destino
        self._pendente = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        texto = self._pendente + s
        *linhas, self._pendente = texto.split("\n")
        if linhas:
            self._destino.write("\n".join(traduzir_linha(x) for x in linhas) + "\n")
        return len(s)

    def flush(self) -> None:
        if self._pendente:
            self._destino.write(traduzir_linha(self._pendente))
            self._pendente = ""
        self._destino.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._destino, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._destino.fileno()

    @property
    def encoding(self):   # type: ignore[override]
        return getattr(self._destino, "encoding", "utf-8")

    @property
    def errors(self):   # type: ignore[override]
        return getattr(self._destino, "errors", "strict")

    def writable(self) -> bool:
        return True


def ativar() -> Callable[[], None]:
    """Em sessao em ingles, passa sys.stdout e sys.stderr pelo tradutor; devolve a funcao que desfaz. Em
    portugues nao faz nada (devolve uma funcao vazia)."""
    from guaraci.idioma_relatorio import idioma
    if idioma() != "en" or isinstance(sys.stdout, TradutorSaida):
        return lambda: None
    antigos = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = TradutorSaida(antigos[0]), TradutorSaida(antigos[1])
    _ATIVO[0] = True

    def desfazer() -> None:
        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except (OSError, ValueError):
                pass
        sys.stdout, sys.stderr = antigos
        _ATIVO[0] = False
    return desfazer
