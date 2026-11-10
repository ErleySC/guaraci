"""resultados_io.py — Escrita dos artefatos de resultado em disco: resumo do
modelo (resumo_modelo.txt), model card (model_card.md) e identificadores das
amostras (CSV). Também as notas metodológicas e as métricas do modelo PLS.

Extraído de pipeline.py (dívida técnica pós-auditoria): estas funções são
CHAMADAS por executar() para persistir os resultados de uma corrida, mas não
dependem do orquestrador — só de config (__version__, _NIVEL_NOME), numpy,
pandas e sklearn. pipeline.py reexporta todos os nomes, então
`pipeline.save_model_summary(...)`, `pq.generate_model_card(...)`,
`pq.append_regression_summary(...)` etc. seguem funcionando (executar e os
testes consomem via fachada).
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from guaraci.classificadores import ddsimca_alpha_sustentado
from guaraci.config import REPO_URL, Config, __version__, _NIVEL_NOME

if TYPE_CHECKING:
    from guaraci.linearity import LackOfFitResult
    from guaraci.robustness import RobustnessResult
    from guaraci.portao_correcao_sinal import VeredictoCorrecaoSinal
    from guaraci.politica_pooled_local import DecisaoPooledLocal

__all__ = [
    "pls_model_metrics",
    "save_identifiers",
    "save_model_summary",
    "append_regression_summary",
    "anexar_bloco_modelo",
    "append_heatmap_summary",
    "generate_model_card",
    "append_regression_model_card",
    "append_identification_model_card",
    "append_purity_model_card",
    "append_linearity_robustness_model_card",
    "append_correcao_sinal_model_card",
    "append_politica_pooled_local_model_card",
    "save_design_audit",
    "load_design_audit",
    "MARCADOR_DADOS_SINTETICOS",
    "MARCADOR_GARANTIA_AGRUPAMENTO",
    "MARCADOR_LEIA_PRIMEIRO",
    "MARCADOR_MULTIELEMENTAR",
]

# Auditoria de delineamento (Bloco 11) persistida tambem em JSON, ao lado do
# model card. O model card ja' traz os mesmos achados em prosa (para leitura
# humana); o JSON existe porque as INTERFACES precisam dos achados
# estruturados -- o painel de status da aba Projeto do app web mostra
# severidade por achado, e extrair isso de volta do Markdown seria reparsear
# texto que ja' foi serializado uma vez.
NOME_AUDITORIA_DELINEAMENTO = "auditoria_delineamento.json"


def save_design_audit(pasta: str, achados: List[Dict[str, str]]) -> None:
    """Grava os achados da auditoria de delineamento em JSON na pasta de
    resultados. Best-effort: falha de escrita nao derruba a execucao (o
    model card ja' tem os mesmos achados em prosa)."""
    import json
    import logging
    try:
        os.makedirs(pasta, exist_ok=True)
        caminho = os.path.join(pasta, NOME_AUDITORIA_DELINEAMENTO)
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(achados, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logging.getLogger(__name__).warning(
            "auditoria de delineamento nao persistida em JSON: %s", e)


def load_design_audit(pasta: str) -> List[Dict[str, str]]:
    """Le os achados gravados por `save_design_audit`. Lista vazia quando o
    arquivo nao existe (execucao anterior a este formato) ou esta ilegivel --
    quem chama deve tratar vazio como "nao ha informacao", nunca como
    "nenhum problema encontrado"."""
    import json
    caminho = os.path.join(pasta, NOME_AUDITORIA_DELINEAMENTO)
    try:
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return dados if isinstance(dados, list) else []


def pls_model_metrics(modelo: PLSRegression, X: np.ndarray, Y: np.ndarray,
                         Y_cv: np.ndarray) -> Tuple[float, float, float]:
    """R2X via 1 - SS(X - T @ P^T) / SS(X_centered); R2Y via 1 - SS(res)/SS(Y);
    Q2 likewise using CV predictions. Rigorous reconstruction formula."""
    X = np.asarray(X, dtype=float)
    T = np.asarray(modelo.x_scores_,   dtype=float)
    P = np.asarray(modelo.x_loadings_, dtype=float)

    X_c = X - X.mean(axis=0)
    ss_x_total = float(np.sum(X_c ** 2))
    ss_x_res   = float(np.sum((X_c - T @ P.T) ** 2))
    r2x = float(1.0 - ss_x_res / ss_x_total) if ss_x_total > 0 else 0.0

    Y = np.asarray(Y, dtype=float)
    Y_cv = np.asarray(Y_cv, dtype=float)
    Y_hat = np.asarray(modelo.predict(X), dtype=float)
    ss_total = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    if ss_total <= 0:
        return r2x, 0.0, 0.0
    r2y = float(1.0 - float(np.sum((Y - Y_hat) ** 2)) / ss_total)
    q2  = float(1.0 - float(np.sum((Y - Y_cv)  ** 2)) / ss_total)
    return r2x, r2y, q2


def save_identifiers(rotulos: np.ndarray, pred_lab: np.ndarray,
                            scores_pls: np.ndarray, T2: np.ndarray,
                            Q: np.ndarray, t2_lim: float, q_lim: float,
                            pasta: str) -> None:
    """Table with IDs, true/predicted classes and per-sample diagnostics."""
    n = len(rotulos)
    n_lvs_save = min(3, scores_pls.shape[1])
    dados = {
        "ID":             [f"S{i:03d}" for i in range(n)],
        "indice":         np.arange(n),
        "classe_real":    rotulos,
        "classe_predita": pred_lab,
        "correto":        rotulos == pred_lab,
    }
    for k in range(n_lvs_save):
        dados[f"LV{k+1}"] = scores_pls[:, k]
    dados["T2"]          = T2
    dados["T2_outlier"]  = T2 > t2_lim
    dados["Q"]           = Q
    dados["Q_outlier"]   = Q > q_lim
    pd.DataFrame(dados).to_csv(
        os.path.join(pasta, "amostras_identificadores.csv"),
        index=False, sep=";", decimal=",")


#: Notas metodologicas de transparencia (Methods section de artigo) --
#: compartilhadas por save_model_summary (.txt) e generate_model_card (.md),
#: fonte unica para nao divergirem com o tempo.
_NOTAS_METODOLOGICAS: List[Tuple[str, str]] = [
    ("LV selection", "Parsimony rule of thumb (smallest number of LVs with "
     "RMSECV within 2% of the minimum). Prevents overfitting on broad "
     "RMSECV plateaus with high max_lvs. The 2% tolerance is a practical "
     "convention, not a published criterion; the cross-validation basis is "
     "Wold (1978) Technometrics 20:397-405."),
    ("CV-ANOVA F-test", "Eriksson et al. (2008) formula applied to one-hot "
     "Y_bin matrix. With K>2 classes, columns of Y_bin are not independent "
     "(row-sums constrained to 1), so the F-statistic degrees of freedom "
     "are approximate. Report as global omnibus test only; do not "
     "interpret individual class F-values."),
    ("SHAP values", "Computed on the full training set (in-sample) using "
     "TreeExplainer. Represent feature importance for the fitted model, "
     "not out-of-sample generalization. For publication, state: "
     "'SHAP analysis was performed on training data (n=X); "
     "importance rankings may be optimistic.'"),
    ("Benchmark hyperparameters", "SVM (C=10, gamma=scale), RF (300 trees), "
     "GBM (200 trees, lr=0.05), XGBoost (300 trees, lr=0.05) use "
     "literature-based heuristics, not cross-validated tuning. "
     "Comparisons are indicative of order-of-magnitude differences; "
     "nested-CV would be required for rigorous pairwise claims."),
    ("VIP bootstrap", "Group-aware (mae_id): resamples physical measurement "
     "points (T1/T2/T3 kept together), preventing inflated stability "
     "from correlated replicates. Standard stratified bootstrap "
     "(per-sample) would overestimate VIP confidence intervals."),
]


#: Ressalvas do model card em PT, cada uma ligada ao modulo que a torna
#: pertinente (R5-11: o card falava de SHAP/benchmark que nao tinham rodado).
_NOTAS_MODEL_CARD: List[Tuple[str, str, Optional[str]]] = [
    ("Selecao de VLs", "Regra pratica de parcimonia (tolerancia de 2% acima "
     "do RMSECV minimo): evita sobreajuste em platos largos. A tolerancia de "
     "2% e uma convencao pratica, nao um criterio publicado; a base da "
     "validacao cruzada e Wold (1978) Technometrics 20:397-405.", None),
    ("F do CV-ANOVA", "Formula de Eriksson et al. (2008) sobre a matriz Y "
     "one-hot; com mais de 2 classes as colunas nao sao independentes e os "
     "graus de liberdade sao aproximados. Use so' como teste global.",
     "run_cv_anova"),
    ("SHAP", "Calculado no treino inteiro (dentro da amostra): mostra a "
     "importancia para o modelo ajustado, nao generalizacao; pode ser otimista.",
     "run_shap"),
    ("Hiperparametros do benchmark", "SVM, RF, GBM e XGBoost usam heuristicas "
     "da literatura, sem ajuste por validacao cruzada: a comparacao indica ordem "
     "de grandeza, nao diferencas finas.", "run_benchmark"),
    ("Bootstrap do VIP", "Reamostra amostras fisicas (replicas juntas) quando ha' "
     "mae_id; bootstrap por espectro superestimaria a estabilidade.", None),
]


#: Marcadores de PROSA (sujeitos a i18n/reescrita) dos avisos do model card e
#: do resumo_modelo.txt. Instrução de 2026-10-02 (Parte 3): um teste que
#: digita este texto de novo quebra toda vez que a prosa muda de idioma/forma
#: (aconteceu com "GROUPING GUARANTEE" -> "GARANTIA DE AGRUPAMENTO" nesta
#: mesma branch). Testes devem importar ESTAS constantes em vez de
#: redigitar o texto -- a fonte de verdade fica só aqui.
MARCADOR_DADOS_SINTETICOS = "DADOS SINTETICOS DE DEMONSTRACAO"
MARCADOR_GARANTIA_AGRUPAMENTO = "GARANTIA DE AGRUPAMENTO"
MARCADOR_LEIA_PRIMEIRO = "## Leia primeiro"
MARCADOR_MULTIELEMENTAR = "PERFIL MULTIELEMENTAR EM VALIDACAO"
# Os mesmos marcadores no model card em ingles (GUARACI_IDIOMA=en).
MARCADOR_DADOS_SINTETICOS_EN = "SYNTHETIC DEMONSTRATION DATA"
MARCADOR_GARANTIA_AGRUPAMENTO_EN = "GROUPING GUARANTEE"
MARCADOR_LEIA_PRIMEIRO_EN = "## Read first"
MARCADOR_MULTIELEMENTAR_EN = "MULTI-ELEMENT PROFILE UNDER VALIDATION"

#: nome do nivel em ingles (config._NIVEL_NOME e' PT).
_NIVEL_NOME_EN = {
    "N1": "Classification by species",
    "N2": "Pure/adulterated discrimination",
    "N3": "Content quantification",
}

#: _NOTAS_MODEL_CARD em ingles (mesma ordem e mesmas chaves de ativacao).
_NOTAS_MODEL_CARD_EN: List[Tuple[str, str, Optional[str]]] = [
    ("LV selection", "Parsimony rule of thumb (2% tolerance above the minimum "
     "RMSECV): avoids overfitting on broad plateaus. The 2% tolerance is a "
     "practical convention, not a published criterion; the cross-validation "
     "basis is Wold (1978) Technometrics 20:397-405.", None),
    ("CV-ANOVA F", "Eriksson et al. (2008) formula on the one-hot Y matrix; "
     "with more than 2 classes the columns are not independent and the "
     "degrees of freedom are approximate. Use it only as a global test.",
     "run_cv_anova"),
    ("SHAP", "Computed on the whole training set (in-sample): shows importance "
     "for the fitted model, not generalisation; may be optimistic.",
     "run_shap"),
    ("Benchmark hyperparameters", "SVM, RF, GBM and XGBoost use literature "
     "heuristics, without cross-validated tuning: the comparison shows order of "
     "magnitude, not fine differences.", "run_benchmark"),
    ("VIP bootstrap", "Resamples physical samples (replicates together) when "
     "there is a mae_id; per-spectrum bootstrap would overestimate stability.", None),
]


#: arquivo com o resumo canonico (chaves fixas do pipeline, para maquina): quem le metricas do
#: resumo le daqui (resumo_parse.ler_resumo_canonico), nao do .txt, que sai no idioma da sessao.
NOME_RESUMO_CANONICO = "resumo_modelo.json"
_FORMATO_RESUMO = "guaraci-resumo/1"

#: as notas metodologicas do resumo em portugues (mesma ordem de _NOTAS_METODOLOGICAS).
_NOTAS_METODOLOGICAS_PT: List[Tuple[str, str]] = [
    ("Selecao de VLs", "Regra pratica de parcimonia (menor numero de VLs com RMSECV ate' 2% acima do "
     "minimo). Evita sobreajuste em platos largos de RMSECV com max_lvs alto. A tolerancia de 2% e' "
     "uma convencao pratica, nao um criterio publicado; a base da validacao cruzada e' Wold (1978) "
     "Technometrics 20:397-405."),
    ("Teste F do CV-ANOVA", "Formula de Eriksson et al. (2008) aplicada a matriz Y_bin (uma coluna por "
     "classe). Com K>2 classes as colunas de Y_bin nao sao independentes (cada linha soma 1), entao os "
     "graus de liberdade do F sao aproximados. Relate so' como teste global; nao interprete o F de "
     "cada classe."),
    ("Valores SHAP", "Calculados no treino inteiro (dentro da amostra) com TreeExplainer. Mostram a "
     "importancia das variaveis para o modelo ajustado, nao a generalizacao. Para publicar, declare: "
     "'a analise SHAP foi feita nos dados de treino (n=X); a ordem de importancia pode ser otimista.'"),
    ("Hiperparametros da comparacao", "SVM (C=10, gamma=scale), RF (300 arvores), GBM (200 arvores, "
     "lr=0.05) e XGBoost (300 arvores, lr=0.05) usam heuristicas da literatura, sem ajuste por "
     "validacao cruzada. A comparacao indica diferencas de ordem de grandeza; afirmacoes rigorosas par "
     "a par exigiriam validacao cruzada aninhada."),
    ("Bootstrap do VIP", "Agrupado por mae_id: reamostra os pontos de medicao fisicos (T1/T2/T3 "
     "juntos), o que evita estabilidade inflada por replicas correlacionadas. Um bootstrap estratificado "
     "por espectro superestimaria a confianca dos intervalos do VIP."),
]

#: severidade da auditoria no resumo legivel.
_SEVERIDADE_EN = {"CRITICO": "CRITICAL", "AVISO": "WARNING", "OK": "OK", "INFO": "INFO"}


def _texto_resumo(info: Dict[str, object], lang: Optional[str] = None) -> str:
    """Texto do resumo. `lang=None` = canonico (chaves fixas do pipeline, o formato que os leitores
    por regex conhecem); 'pt'/'en' = o resumo_modelo.txt legivel, num idioma so'."""
    from guaraci import idioma_relatorio as _i
    canonico = lang is None
    en = (not canonico) and _i.idioma(lang) == "en"

    def T(pt: str, ing: str, canon: str) -> str:
        return canon if canonico else (ing if en else pt)

    def rot(k: str) -> str:
        return k if canonico else _i.rotulo_linha_resumo(k, lang)

    def val(v: object) -> str:
        if canonico:
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        return _i.valor_resumo(v, lang)

    linhas: List[str] = ["=" * 60,
                         T("  Resumo do modelo PLS-DA", "  PLS-DA Model Summary", "  PLS-DA Model Summary"),
                         "=" * 60, ""]
    # R5-S3/M-19(b) (revisao cega 2026-09-26): o que importa para NAO citar
    # um numero errado vinha depois de ~100 linhas de metricas.
    _aud = info.get("auditoria_delineamento")
    _crit = [a for a in (_aud if isinstance(_aud, list) else [])
             if isinstance(a, dict) and a.get("severidade") == "critico"]

    def msg(a: Dict[str, Any]) -> str:
        m = str(a.get("mensagem", ""))
        return m if canonico else _i.achado(m, lang)

    linhas.append(T("  LEIA PRIMEIRO", "  READ FIRST", "  LEIA PRIMEIRO"))
    if _crit:
        linhas.append(T(f"  {len(_crit)} problema(s) CRITICO(s) de delineamento -- nao cite as",
                        f"  {len(_crit)} CRITICAL design problem(s) -- do not cite the",
                        f"  {len(_crit)} problema(s) CRITICO(s) de delineamento -- nao cite as"))
        linhas.append(T("  metricas abaixo antes de resolver:", "  metrics below before fixing them:",
                        "  metricas abaixo antes de resolver:"))
        for a in _crit:
            linhas.append(f"    - {a.get('nome', '?')}: {msg(a)}")
    else:
        linhas.append(T("  Nenhum problema critico de delineamento detectado.",
                        "  No critical design problem detected.",
                        "  Nenhum problema critico de delineamento detectado."))
    linhas += [T("  Como ler: o desempenho fora da amostra e' o do HOLDOUT externo",
                 "  How to read: out-of-sample performance is the external HOLDOUT",
                 "  Como ler: o desempenho fora da amostra e' o do HOLDOUT externo"),
               T("  (linhas 'Holdout ...'); a validacao cruzada (VC) e' otimista se a",
                 "  ('Holdout ...' lines); cross-validation (CV) is optimistic if the",
                 "  (linhas 'Holdout ...'); a validacao cruzada (CV) e' otimista se a"),
               T("  garantia de agrupamento for 'media' ou 'nenhuma'. Ressalvas: model_card.md.",
                 "  grouping guarantee is 'medium' or 'none'. Caveats: model_card.md.",
                 "  garantia de agrupamento for 'medium' ou 'none'. Ressalvas: model_card.md."),
               ""]
    escalares = [(k, v) for k, v in info.items() if not isinstance(v, (dict, list))]
    estruturados = {k: v for k, v in info.items() if isinstance(v, (dict, list))}
    if canonico:
        # largura pela MAIOR chave de todo o resumo (estruturadas inclusive), como sempre foi
        largura = max(len(k) for k in info.keys()) + 2
    else:
        largura = max([len(rot(k)) for k, _ in escalares] or [0]) + 2
    for k, v in escalares:
        linhas.append(f"  {rot(k):<{largura}s}: {val(v)}")

    # M-19(b): a auditoria de delineamento saia como repr() Python de uma
    # lista de dicts, numa linha, longe das metricas. Agora e' um bloco
    # legivel logo apos elas, uma linha por achado, com a severidade.
    achados = estruturados.pop("auditoria_delineamento", None)
    if isinstance(achados, list) and achados:
        linhas.append("")
        linhas.append(T("  Auditoria de delineamento (ler ANTES de citar as metricas):",
                        "  Design audit (read BEFORE citing the metrics):",
                        "  Auditoria de delineamento (ler ANTES de citar as metricas):"))
        for a in achados:
            sev = str(a.get("severidade", "?")).upper()
            if en:
                sev = _SEVERIDADE_EN.get(sev, sev)
            nome = a.get('nome', '?') if canonico else f"`{a.get('nome', '?')}`"   # id da checagem
            linhas.append(f"    [{sev}] {nome}: {msg(a)}")
    for k, v in estruturados.items():
        linhas.append("")
        linhas.append(f"  {rot(k)}:")
        if isinstance(v, dict):
            for kk, vv in v.items():
                # valor de enum gravado em PT ("normalizacao: nenhuma") sai no idioma do relatorio; o canonico fica
                linhas.append(f"    {kk}: {_i.valor(vv, lang) if (not canonico and isinstance(vv, str)) else vv}")
        else:
            for item in v:
                linhas.append(f"    - {item}")

    # Methodological transparency notes (for article Methods section)
    linhas.append("")
    linhas.append("-" * 60)
    linhas.append(T("  Notas metodologicas (para a revisao do artigo)",
                    "  Methodological Notes (for article peer review)",
                    "  Methodological Notes (for article peer review)"))
    linhas.append("-" * 60)
    notas = _NOTAS_METODOLOGICAS_PT if (not canonico and not en) else _NOTAS_METODOLOGICAS
    for title, note in notas:
        linhas.append("")
        linhas.append(f"  [{title}]")
        # Word-wrap at 70 chars
        line = "    "
        for w in note.split():
            if len(line) + len(w) + 1 > 70:
                linhas.append(line)
                line = "    " + w + " "
            else:
                line += w + " "
        if line.strip():
            linhas.append(line)
    return "\n".join(linhas) + "\n"


def _gravar_canonico(pasta: str, texto: str, anexar: bool = False) -> None:
    """Grava (ou anexa a) o resumo canonico em resumo_modelo.json."""
    import json
    caminho = os.path.join(pasta, NOME_RESUMO_CANONICO)
    if anexar:
        try:
            with open(caminho, encoding="utf-8") as f:
                dados = json.load(f)
        except (OSError, ValueError):
            return    # execucao antiga (so' o .txt): nada a anexar
        texto = str(dados.get("texto", "")) + texto
    from guaraci.idioma_relatorio import idioma
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump({"formato": _FORMATO_RESUMO, "idioma_do_txt": idioma(), "texto": texto},
                  f, ensure_ascii=False, indent=1)


def save_model_summary(pasta: str, info: Dict[str, object]) -> None:
    """resumo_modelo.txt no idioma da sessao (para ler) + resumo_modelo.json com o texto canonico
    (chaves fixas, para os leitores por regex: relatorios, assistente, app)."""
    from guaraci.idioma_relatorio import idioma
    with open(os.path.join(pasta, "resumo_modelo.txt"), "w", encoding="utf-8") as f:
        f.write(_texto_resumo(info, idioma()))
    _gravar_canonico(pasta, _texto_resumo(info, None))


def _anexar_ao_resumo(pasta: str, linhas: List[str], o_que: str) -> None:
    """Anexa um bloco ao resumo: canonico no .json, no idioma da sessao no .txt."""
    from guaraci.idioma_relatorio import linha_resumo
    try:
        with open(os.path.join(pasta, "resumo_modelo.txt"), "a", encoding="utf-8") as f:
            f.write("\n".join(linha_resumo(x) for x in linhas) + "\n")
        _gravar_canonico(pasta, "\n".join(linhas) + "\n", anexar=True)
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel anexar {o_que} ao resumo: {e}")



#: marca das linhas de AVISO do modelo no resumo canonico (uma linha por aviso, com o texto em PT e
#: EN: o relatorio, que pode estar em outro idioma que o do resumo, escolhe o seu). Ler com
#: `resumo_parse.avisos_do_modelo`.
MARCADOR_AVISO_MODELO = "[AVISO-MODELO]"


def anexar_bloco_modelo(pasta: str, titulo: Tuple[str, str],
                        itens: List[Tuple[str, str, str]], *, aviso: bool) -> None:
    """Anexa um bloco (avisos ou registros) ao resumo_modelo.txt/.json e ao model_card.md.

    `itens` = [(codigo, texto_pt, texto_en)]. Com `aviso=True` cada item vira uma linha
    `[AVISO-MODELO] codigo PT=... EN=...` no resumo canonico (lida por `avisos_do_modelo`, que os
    relatorios PDF/DOCX/XLSX usam) e um paragrafo `[AVISO]` no resumo legivel, no idioma da sessao.
    Sem itens, nao faz nada. Append-only, como os outros blocos: o resumo ja foi gravado."""
    if not itens:
        return
    import textwrap
    from guaraci.idioma_relatorio import idioma, t as T
    en = idioma() == "en"
    tit = titulo[1] if en else titulo[0]
    barra = "=" * 60
    txt: List[str] = ["", barra, f"  {tit.upper()}", barra]
    canon: List[str] = ["", barra, f"  {titulo[0].upper()}", barra]
    for codigo, pt, ing in itens:
        corpo = ing if en else pt
        marca = "[WARNING]" if (aviso and en) else ("[AVISO]" if aviso else "-")
        linhas = textwrap.wrap(corpo, width=74)
        txt.append(f"  {marca} {linhas[0]}")
        txt += [f"      {x}" for x in linhas[1:]]
        if aviso:
            canon.append(f"  {MARCADOR_AVISO_MODELO} {codigo} PT={' '.join(pt.split())} EN={' '.join(ing.split())}")
        else:
            canon.append(f"  - {' '.join(pt.split())}")
    try:
        with open(os.path.join(pasta, "resumo_modelo.txt"), "a", encoding="utf-8") as f:
            f.write("\n".join(txt) + "\n")
        _gravar_canonico(pasta, "\n".join(canon) + "\n", anexar=True)
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel anexar '{titulo[0]}' ao resumo: {e}")
    cartao = os.path.join(pasta, "model_card.md")
    if os.path.isfile(cartao):
        md = ["", f"## {tit}", ""] + [f"- {'**' + T('Aviso', 'Warning') + ':** ' if aviso else ''}"
                                      f"{' '.join((ing if en else pt).split())}" for _c, pt, ing in itens]
        try:
            with open(cartao, "a", encoding="utf-8") as f:
                f.write("\n".join(md) + "\n")
        except OSError as e:
            print(f"  [AVISO] Nao foi possivel anexar '{titulo[0]}' ao model card: {e}")


def append_regression_summary(
        pasta: str,
        pooled: Optional[Dict[str, object]] = None,
        tabela_especie: Optional[List[Dict[str, object]]] = None,
        fom_pooled: Optional[Dict[str, float]] = None,
        holdout: Optional[Dict[str, Any]] = None) -> None:
    """Anexa o bloco de regressao PLS (com figuras de merito analiticas de
    Valderrama, Braga & Poppi 2009) ao resumo_modelo.txt.

    Motivacao: LOD/LOQ/SEN/SEL/gamma so eram impressos no console/log; agora
    tambem entram no resumo persistido (que a aba Relatorios do app captura).
    E append-only DE PROPOSITO: a regressao roda depois de o resumo ser gravado
    a 1a vez em executar(), e reordenar esse orquestrador seria arriscado.
    """

    def _fmt(v: object, nd: int = 3, suf: str = "") -> str:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "n/a"
        return f"{fv:.{nd}f}{suf}" if np.isfinite(fv) else "n/a"

    linhas: List[str] = [
        "", "=" * 60,
        "  PLS Regression — Analytical Figures of Merit",
        "  (Valderrama, Braga & Poppi, 2009, Quim. Nova 32(5):1278-1287)",
        "=" * 60,
    ]
    if pooled is not None:
        linhas.append("")
        linhas.append("  Pooled metrics:")
        linhas.append(f"    R2cal .....: {_fmt(pooled.get('r2c'), 4)}")
        linhas.append(f"    R2val .....: {_fmt(pooled.get('r2v'), 4)}")
        linhas.append(f"    RMSEC .....: {_fmt(pooled.get('rmsec'))}")
        linhas.append(f"    RMSECV ....: {_fmt(pooled.get('rmsecv'))}")
        if pooled.get('r2cv') is not None:
            linhas.append(f"    R2cv ......: {_fmt(pooled.get('r2cv'), 4)}")
        linhas.append(f"    RMSEP .....: {_fmt(pooled.get('rmsep'))}")
        if pooled.get('bias') is not None:
            linhas.append(f"    Bias ......: {_fmt(pooled.get('bias'), 4)}")
        linhas.append("    (RMSEP acima = validacao INTERNA: split 70/30 dos dados de")
        linhas.append("     treino, n pequeno; nao e' o numero publicavel.)")
        if holdout is not None:
            linhas.append(
                f"    RMSEP (holdout externo) : {_fmt(holdout.get('rmsep'))}"
                f"   n={holdout.get('n')}  R2={_fmt(holdout.get('r2'), 3)}  "
                f"bias={_fmt(holdout.get('bias'), 3)}   <- numero publicavel")
            if holdout.get('n_sem_modelo'):
                linhas.append(
                    f"      ({holdout['n_sem_modelo']} amostra(s) do holdout "
                    f"sem modelo de regressao para a classe atribuida)")
        # RPD/RER com a faixa de uso ao lado: um RPD nu vira alegacao
        # exagerada em texto (Williams 2014; AACC 39-00.01).
        if pooled.get('rpd') is not None:
            linhas.append(f"    SEP .......: {_fmt(pooled.get('sep'), 3)}")
            linhas.append(f"    RPD .......: {_fmt(pooled.get('rpd'), 2)}"
                          f"  ({pooled.get('rpd_faixa', '-')})")
            linhas.append(f"    RER .......: {_fmt(pooled.get('rer'), 1)}")
        if pooled.get('dmody_crit') is not None:
            linhas.append(f"    DModY critico (SIMCA): "
                          f"{_fmt(pooled.get('dmody_crit'), 3)}")
            linhas.append(f"    Amostras fora do DModY: "
                          f"{pooled.get('n_fora_do_dmody', 'n/a')}")
    if tabela_especie:
        linhas.append("")
        linhas.append("  Per-species figures of merit:")
        hdr = (f"    {'Species':<18s} {'LVs':>3s} {'RMSEP':>7s} {'R2val':>6s} "
               f"{'LOD':>7s} {'LOQ':>7s} {'SEN':>7s} {'SEL':>6s}")
        linhas.append(hdr)
        linhas.append("    " + "-" * (len(hdr) - 4))
        for t in tabela_especie:
            linhas.append(
                f"    {str(t.get('especie', ''))[:18]:<18s} "
                f"{int(t.get('n_lv', 0) or 0):>3d} "  # type: ignore[call-overload]
                # ^ t: Dict[str, object]; .get() devolve object, sem
                # overload de int() compativel -- typing solto de proposito
                # (dado heterogeneo por especie), nao vale apertar aqui.
                f"{_fmt(t.get('rmsep'), 2):>7s} {_fmt(t.get('r2val'), 3):>6s} "
                f"{_fmt(t.get('lod'), 2):>7s} {_fmt(t.get('loq'), 2):>7s} "
                f"{_fmt(t.get('sensibilidade'), 3):>7s} "
                f"{_fmt(t.get('seletividade_media'), 3):>6s}")
            # Bloco 12: LOD/LOQ nunca isolado -- IC (Allegrini & Olivieri
            # 2014, ver chemometric_stats.regression_figures_of_merit) +
            # faixa/desvio-padrao do conjunto de VALIDACAO logo abaixo.
            _lod = t.get("lod")
            if t.get("lod_ic_baixo") is not None and isinstance(_lod, (int, float, np.number)) \
                    and np.isfinite(_lod):
                _conf = t.get("lod_ic_confianca")
                _conf_txt = f"{_conf * 100:.0f}%" if _conf is not None else "?"  # type: ignore[operator]
                linhas.append(
                    f"      IC({_conf_txt}): LOD "
                    f"[{_fmt(t.get('lod_ic_baixo'), 2)}-"
                    f"{_fmt(t.get('lod_ic_alto'), 2)}]  "
                    f"LOQ [{_fmt(t.get('loq_ic_baixo'), 2)}-"
                    f"{_fmt(t.get('loq_ic_alto'), 2)}]  |  "
                    f"validacao: [{_fmt(t.get('validacao_teor_min'), 2)}, "
                    f"{_fmt(t.get('validacao_teor_max'), 2)}] "
                    f"DP={_fmt(t.get('validacao_teor_dp'), 2)}")
            # T1 (rodada multiagente 2026-09-10, Passo 202): margem de
            # predicao conforme por especie (conformal.conformal_margin_
            # regression) -- so' reporta numero quando alcancavel; senao
            # diz explicitamente por que nao (NAO_VALIDADO, mesma
            # disciplina do IC de LOD/LOQ acima).
            if t.get("conformal_alcancavel") is True:
                linhas.append(
                    f"      Intervalo de predicao (conforme, 95%): "
                    f"+/-{_fmt(t.get('conformal_margem'), 2)} "
                    f"({t.get('conformal_n_grupos')} grupos de validacao)")
            elif t.get("conformal_alcancavel") is False:
                linhas.append(
                    f"      Intervalo de predicao: NAO_VALIDADO "
                    f"({t.get('conformal_n_grupos')} grupo(s) de validacao, "
                    f"minimo 19 p/ alpha=0.05)")
    if fom_pooled is not None:
        linhas.append("")
        linhas.append("  Figures of merit (single pooled model):")
        linhas.append(f"    LOD .......: {_fmt(fom_pooled.get('lod'), 2)}")
        linhas.append(f"    LOQ .......: {_fmt(fom_pooled.get('loq'), 2)}")
        linhas.append(f"    SEN .......: {_fmt(fom_pooled.get('sensibilidade'), 3)}")
        linhas.append(f"    gamma .....: {_fmt(fom_pooled.get('sensibilidade_analitica'), 2)}")
        linhas.append(f"    SEL .......: {_fmt(fom_pooled.get('seletividade_media'), 3)}")
        linhas.append(f"    delta_x ...: {_fmt(fom_pooled.get('delta_x_ruido'), 5)}")
    linhas.append("")
    linhas.append("  LOD, LOQ, RMSEP and the validation range are in the unit of the reference concentration.")
    linhas.append("  Note: LOD/LOQ/SEN require physical replicates (mae_id) to")
    linhas.append("  estimate instrumental noise; 'n/a' means none available.")

    _anexar_ao_resumo(pasta, linhas, "regressao")


def append_heatmap_summary(pasta: str, resultado: Dict[str, object]) -> None:
    """Anexa ao resumo_modelo.txt o balanco do heatmap especie x adulterante
    (R2cv por combinacao). Deixa EXPLICITO quantas combinacoes NAO atingem o
    limiar de aceite -- uma quantificacao que so funciona em parte das
    combinacoes nao deve ser lida como sucesso geral. Append-only pelo mesmo
    motivo de append_regression_summary (roda depois do 1o flush do resumo).
    """
    limiar   = float(resultado.get("limiar_r2", 0.70))   # type: ignore[arg-type]
    n_falhas = int(resultado.get("n_falhas", 0))          # type: ignore[call-overload]
    n_ok     = int(resultado.get("n_ok", 0))              # type: ignore[call-overload]
    n_na     = int(resultado.get("n_na", 0))              # type: ignore[call-overload]
    n_total  = int(resultado.get("n_total", 0))           # type: ignore[call-overload]
    matriz   = resultado.get("matriz", {}) or {}
    linhas: List[str] = [
        "", "=" * 60,
        "  Quantificacao por especie x adulterante (R2cv, group-aware)",
        "=" * 60,
        f"  Limiar de aceite: R2cv >= {limiar:.2f}",
        f"  Combinacoes avaliadas: {n_total} | aprovadas: {n_ok} | "
        f"abaixo do limiar: {n_falhas} | sem dados (n/a): {n_na}",
        f"  >> {n_falhas}/{n_total} combinacoes abaixo de R2cv = {limiar:.2f}",
    ]
    if isinstance(matriz, dict) and matriz:
        abaixo = []
        for (esp, ad), r2 in matriz.items():
            try:
                fr = float(r2)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fr) and fr < limiar:
                abaixo.append(f"    {esp} x {ad}: R2cv={fr:.3f}")
        linhas.append("")
        linhas.append("  Combinacoes abaixo do limiar (nao quantificaveis):")
        linhas.extend(abaixo or
                      ["    (nenhuma — todas as combinacoes com dados passaram)"])
    _anexar_ao_resumo(pasta, linhas, "heatmap")


# =========================================================================
#  Model Card (Mitchell et al. 2019, "Model Cards for Model Reporting",
#  FAT* '19) -- resumo de 1 documento por execucao para uso pretendido,
#  dados, metricas e limitacoes. Montagem a partir de dados JA CALCULADOS
#  (resumo/hardware/config) -- nao introduz ciencia nova, so' organiza o
#  que ja existe no padrao esperado por quem avalia "ferramenta profissional"
#  (ML-ops/auditoria), analogo ao model card do Hugging Face Hub.
# =========================================================================

def _faixa_do_resumo(resumo: Dict[str, Any]) -> Tuple[str, str]:
    """(rotulo, valor) da faixa espectral do resumo, com a unidade que a execucao gravou.

    A chave vem do perfil (`Faixa espectral (nm)`, `... (cm⁻¹)` ou sem unidade); resumos
    antigos usam `Faixa espectral (cm-1)`. Nenhuma unidade e' presumida aqui."""
    for chave, valor in resumo.items():
        if str(chave).startswith("Faixa espectral"):
            return str(chave), str(valor)
    return "Faixa espectral", "-"


def _md_tabela(linhas: List[Tuple[str, str]],
               cabecalho: Optional[Tuple[str, str]] = None) -> str:
    """Monta uma tabela Markdown de 2 colunas (Campo | Valor)."""
    # M-19(a) do parecer de 2026-09-25: varios valores trazem " | " como
    # separador interno ("recomendacao=... | n_amostras=... | veredito=...").
    # Num GFM cada "|" abre uma CELULA: a linha virava 4 colunas sob um
    # cabecalho de 2, e o renderizador DESCARTAVA as extras -- inclusive o
    # status "nao validado". O pipe interno e' escapado: sempre 2 celulas.
    def _esc(t: object) -> str:
        return str(t).replace("|", "\\|").replace("\n", " ")

    if cabecalho is None:
        from guaraci.idioma_relatorio import t as _t
        cabecalho = (_t("Campo", "Field"), _t("Valor", "Value"))
    out = [f"| {cabecalho[0]} | {cabecalho[1]} |", "|---|---|"]
    for k, v in linhas:
        out.append(f"| {_esc(k)} | {_esc(v)} |")
    return "\n".join(out)


def generate_model_card(pasta: str, cfg: "Config", resumo: Dict[str, object],
                      hw: Dict[str, Any], classes_unicas: np.ndarray) -> None:
    """Gera `model_card.md` (secoes de Mitchell et al. 2019, adaptado a um
    pipeline quimiometrico de autenticacao/quantificacao). Escrito no MESMO
    ponto de `save_model_summary` -- reaproveita o dict `resumo` inteiro,
    ja com todas as metricas/diagnosticos/integridade de dados calculados,
    em vez de recalcular ou receber duzias de parametros separados.

    Um idioma so' (`idioma_relatorio.idioma()`, o da sessao): rotulos do
    resumo, valores curtos e mensagens da auditoria passam por
    `idioma_relatorio` -- antes o card em PT trazia "Total samples",
    "Accuracy (CV)" etc. no meio.

    Regressao (N2/N3) e' um addendum ANEXADO depois (mesmo padrao de
    `append_regression_summary`), pois so' fica disponivel mais tarde em
    executar() -- ver `append_regression_model_card`.
    """
    from guaraci import idioma_relatorio as _i18n
    lang = _i18n.idioma()

    def T(pt: str, en: str) -> str:
        return _i18n.t(pt, en, lang)

    def R(chave: str) -> str:
        return _i18n.rotulo_resumo(chave, lang)

    nivel = cfg.level
    nivel_nome = T(_NIVEL_NOME.get(nivel, nivel), _NIVEL_NOME_EN.get(nivel, nivel))
    agora = datetime.now().strftime("%Y-%m-%d %H:%M")

    # A matriz e o vocabulario vem do PERFIL (guaraci/perfis_matriz/*.yaml),
    # nunca do codigo: o texto anterior afirmava o dominio de origem em
    # qualquer execucao -- inclusive rodando sobre outra matriz (medido na
    # auditoria mestre de 2026-08-17). Um model card que declara a matriz
    # errada e' pior que um sem matriz nenhuma.
    from guaraci.perfil_matriz import cfg_profile
    perfil = cfg_profile(cfg)
    voc = _i18n.vocabulario(perfil.vocabulario, lang)
    if lang == "en":
        uso_pretendido = {
            "N1": (f"Identification of the {voc['classe']} in {voc['matriz']} from the "
                   "spectrum (multiclass classification by PLS-DA)."),
            "N2": (f"Authentication PER {voc['classe'].upper()} ({voc['conforme']} vs. "
                   f"{voc['nao_conforme']}) by one-class DD-SIMCA, in {voc['matriz']}."),
            "N3": (f"Quantification of {voc['alvo']} in {voc['matriz']}, calibrated "
                   f"separately per {voc['classe']} (PLS-R)."),
        }.get(nivel, "Chemometric analysis of spectral data.")
    else:
        # contrair(): o vocabulario traz o artigo ("o teor...", "a matriz...") e
        # "de o teor ... em a matriz" saia literal no card.
        uso_pretendido = _i18n.contrair({
            "N1": (f"Identificacao de {voc['classe']} em {voc['matriz']} a partir do "
                   "espectro (classificacao multiclasse via PLS-DA)."),
            "N2": (f"Autenticacao POR {voc['classe'].upper()} ({voc['conforme']} vs. "
                   f"{voc['nao_conforme']}) via DD-SIMCA one-class, em {voc['matriz']}."),
            "N3": (f"Quantificacao de {voc['alvo']} em {voc['matriz']}, calibrada "
                   f"separadamente por {voc['classe']} (PLS-R)."),
        }.get(nivel, "Analise quimiometrica de dados espectrais."))

    linhas: List[str] = [
        f"# Model Card -- GUARACI v{__version__}",   # nome do formato (Mitchell et al. 2019)
        "",
        T(f"*Gerado automaticamente em {agora}. Este documento descreve os "
          "artefatos de UMA execucao do pipeline; regenere a cada novo "
          "conjunto de dados ou configuracao.*",
          f"*Generated automatically on {agora}. This document describes the "
          "artefacts of ONE pipeline run; regenerate it for every new dataset "
          "or configuration.*"),
        "",
    ]
    # Passo 57 (achado na revisao com o usuario): dados sinteticos
    # (`dados_io.generate_synthetic_data`) tem sinal construido com uma
    # banda marcadora limpa por classe -- metricas quase perfeitas
    # (accuracy/balanced accuracy/kappa = 1,0000) sao ESPERADAS nesse
    # regime, nao evidencia de desempenho. Sem esta marca, um exemplo
    # sintetico e' indistinguivel de um resultado real no documento --
    # exatamente o tipo de "metrica perfeita sem contexto" que ja foi
    # achado de auditoria grave quando apareceu em dataset real. Primeiro
    # aviso do documento (antes ate' do grouping_guarantee).
    if cfg.mode == "sintetico":
        linhas += [
            f"> ## ⚠️ {T('EXECUTADO COM', 'RUN ON')} "
            f"{T(MARCADOR_DADOS_SINTETICOS, MARCADOR_DADOS_SINTETICOS_EN)}",
            ">",
            T("> Esta execucao usou o gerador sintetico "
              "(`dados_io.generate_synthetic_data`), nao espectros reais "
              "medidos. O sinal de cada classe/adulterante e' construido com "
              "uma banda marcadora limpa -- metricas perfeitas ou quase "
              "perfeitas (acuracia, acuracia balanceada, kappa = 1,0000) sao "
              "ESPERADAS neste regime e NAO indicam desempenho real do "
              "metodo. Use apenas para verificar que o pipeline roda de "
              "ponta a ponta; nunca cite estes numeros como evidencia de "
              "desempenho em dado real.",
              "> This run used the synthetic generator "
              "(`dados_io.generate_synthetic_data`), not real measured "
              "spectra. The signal of each class/adulterant is built with a "
              "clean marker band -- perfect or near-perfect metrics "
              "(accuracy, balanced accuracy, kappa = 1.0000) are EXPECTED in "
              "this regime and do NOT show real performance of the method. "
              "Use it only to check that the pipeline runs end to end; never "
              "cite these numbers as evidence of performance on real data."),
            "",
        ]
    _mr = getattr(cfg, "multielementar_resumo", None) or {}
    if cfg.mode == "multielementar":
        _anova = _mr.get("anova", {})
        _sens = _mr.get("sensibilidade_censura", {})
        _ses = _mr.get("efeito_de_sessao", {})
        _muda = (T("MUDA", "CHANGES") if _sens.get("muda") else
                 (T("nao muda", "does not change") if _sens.get("muda") is False
                  else T("nao pode ser comparada", "cannot be compared")))
        linhas += [
            f"> ## ⚠️ {T(MARCADOR_MULTIELEMENTAR, MARCADOR_MULTIELEMENTAR_EN)}",
            ">",
            f"> {_mr.get('selo', T('em validacao', 'under validation'))}.",
            T(f"> Unidade: {_mr.get('unidade', '?')}. Censura: politica "
              f"'{_mr.get('politica_censura', '?')}', limiar {_mr.get('limiar_censura', '?')}, "
              f"LOD/{_mr.get('divisor_lod', '?')}; transformacao: {_mr.get('transformacao', '?')}.",
              f"> Unit: {_mr.get('unidade', '?')}. Censoring: policy "
              f"'{_mr.get('politica_censura', '?')}', threshold {_mr.get('limiar_censura', '?')}, "
              f"LOD/{_mr.get('divisor_lod', '?')}; transformation: {_mr.get('transformacao', '?')}."),
            T(f"> Elementos modelados: {len(_mr.get('elementos_modelados', []))}; excluidos por "
              f"censura: {_mr.get('elementos_excluidos') or 'nenhum'}.",
              f"> Modelled elements: {len(_mr.get('elementos_modelados', []))}; excluded by "
              f"censoring: {_mr.get('elementos_excluidos') or 'none'}."),
            T(f"> ANOVA por elemento sobre a unidade independente ({_anova.get('correcao', '?')}): "
              f"{len(_anova.get('elementos_com_diferenca_entre_grupos', []))} elemento(s) com "
              f"diferenca entre grupos (FDR < {_anova.get('alpha', '?')}). Tabelas: ",
              f"> ANOVA per element on the independent unit ({_anova.get('correcao', '?')}): "
              f"{len(_anova.get('elementos_com_diferenca_entre_grupos', []))} element(s) with a "
              f"difference between groups (FDR < {_anova.get('alpha', '?')}). Tables: ")
            + "`Tabelas/multielementar_anova_por_elemento.csv`, `Tabelas/multielementar_comparacoes.csv`; "
            + T("detalhe: ", "detail: ") + "`Relatorios/multielementar.json`.",
            T(f"> Sensibilidade da censura (as duas politicas): a conclusao {_muda}",
              f"> Censoring sensitivity (both policies): the conclusion {_muda}")
            + (f": {sorted(_sens.get('diferencas', {}))}" if _sens.get("muda") else "") + ".",
            T("> Efeito de sessao (QC): ", "> Session effect (QC): ")
            + (str(_ses.get("elementos_com_efeito_de_sessao")) if _ses.get("testado") else
               T("nao testado -- ", "not tested -- ") + str(_ses.get("motivo", ""))) + ".",
            "",
        ]
    _marca_agr = T(MARCADOR_GARANTIA_AGRUPAMENTO, MARCADOR_GARANTIA_AGRUPAMENTO_EN)
    if cfg.grouping_guarantee == "medium":
        linhas += [
            f"> ## ⚠️ {_marca_agr}: {T('MEDIA', 'MEDIUM')}",
            ">",
            T("> A validacao foi agrupada pelo grupo que VOCE declarou (coluna ou "
              "arquivo de grupos). O software nao consegue verificar se esses "
              "grupos sao de fato independentes: se replicas ou diluicoes da "
              "mesma sessao/lote receberam grupos diferentes, as metricas sao "
              "otimistas.",
              "> Validation was grouped by the group YOU declared (column or "
              "group file). The software cannot check whether those groups are "
              "really independent: if replicates or dilutions of the same "
              "session/batch got different groups, the metrics are optimistic."),
            "",
        ]
    elif cfg.grouping_guarantee != "high":
        linhas += [
            f"> ## ⚠️ {_marca_agr}: "
            f"{T({'none': 'NENHUMA', 'unknown': 'DESCONHECIDA'}.get(cfg.grouping_guarantee, cfg.grouping_guarantee.upper()), cfg.grouping_guarantee.upper())}",
            ">",
            T("> Sem identificador de grupo (amostra fisica) confiavel: a validacao "
              "caiu em `StratifiedKFold`, sem protecao contra vazamento entre "
              "replicas da mesma amostra. Trate as metricas como exploratorias. "
              "Declare `coluna_grupo` ou `arquivo_grupos` para proteger.",
              "> No reliable group (physical sample) identifier: validation fell "
              "back to `StratifiedKFold`, without protection against leakage "
              "between replicates of the same sample. Treat the metrics as "
              "exploratory. Declare `coluna_grupo` or `arquivo_grupos` to protect it."),
            "",
        ]
    # R5-11: "Leia primeiro" -- os CRITICOS da auditoria no topo, em linguagem
    # simples, em vez de enterrados no meio do documento.
    _aud_lst = resumo.get("auditoria_delineamento")
    _crit = [a for a in (_aud_lst if isinstance(_aud_lst, list) else [])
             if isinstance(a, dict) and a.get("severidade") == "critico"]
    linhas += [T(MARCADOR_LEIA_PRIMEIRO, MARCADOR_LEIA_PRIMEIRO_EN), ""]
    if _crit:
        linhas += [T(f"**{len(_crit)} problema(s) CRITICO(s) no delineamento — nao "
                     "cite as metricas abaixo antes de resolver:**",
                     f"**{len(_crit)} CRITICAL design problem(s) — do not cite the "
                     "metrics below before solving them:**"), ""]
        linhas += [f"- **`{a['nome']}`**: {_i18n.achado(a['mensagem'], lang)}" for a in _crit]
    else:
        linhas += [T("Nenhum problema critico de delineamento detectado. Mesmo assim, "
                     "o numero que vale como desempenho fora da amostra e' o do "
                     "**holdout externo** (abaixo), nao o da validacao cruzada.",
                     "No critical design problem detected. Even so, the number that "
                     "counts as out-of-sample performance is the **external holdout** "
                     "(below), not the cross-validation one.")]
    linhas += [""]
    _faixa_rot, _faixa_val = _faixa_do_resumo(resumo)
    if lang == "en":
        _faixa_rot = _faixa_rot.replace("Faixa espectral", "Spectral range")
    _matriz_txt = re.sub(r"^(a|o|as|os|the)\s+", "", voc["matriz"].strip(), flags=re.IGNORECASE)
    linhas += [
        T("## 1. Detalhes do Modelo", "## 1. Model Details"),
        "",
        _md_tabela([
            (T("Plataforma", "Platform"), T("GUARACI -- Plataforma de Quimiometria",
                                            "GUARACI -- Chemometrics Platform")),
            (T("Versao", "Version"), __version__),
            (T("Tipo de analise", "Analysis type"), f"{nivel} ({nivel_nome})"),
            (T("Algoritmo principal", "Main algorithm"),
             T("PLS-DA (analise discriminante por minimos quadrados parciais)",
               "PLS-DA (Partial Least Squares Discriminant Analysis)") if nivel != "N3" else
             T("PLS-DA + PLS-R por especie", "PLS-DA + PLS-R per species")),
            (T("Pre-processamento", "Pre-processing"),
             _i18n.preprocessamento(resumo.get("Pre-processamento", "-"), lang)),
            (T("Perfil de tecnica", "Technique profile"),
             _perfil_txt(perfil, lang)),
            (T("Matriz", "Matrix"), _matriz_txt),
            (_faixa_rot, _faixa_val),
            (T("Faixa de trabalho declarada", "Declared working range"),
             (f"{perfil.faixa_trabalho[0]} {T('a', 'to')} {perfil.faixa_trabalho[1]}"
              if perfil.faixa_trabalho else T("nao declarada no perfil", "not declared in the profile"))),
            (T("Tag de execucao", "Run tag"), str(resumo.get("Tag", "-"))),
            (T("Licenca", "License"), "GPL-3.0-or-later"),
            (T("Repositorio", "Repository"), REPO_URL.split("://", 1)[1]),
        ]),
        "",
        T("## 2. Uso Pretendido", "## 2. Intended Use"),
        "",
        f"**{T('Uso primario', 'Primary use')}:** {uso_pretendido}",
        "",
        T("**Usuarios primarios:** pesquisadores em quimiometria, laboratorios "
          "de controle de qualidade, projetos academicos "
          "(graduacao/pos-graduacao).",
          "**Primary users:** chemometrics researchers, quality-control "
          "laboratories, academic projects (undergraduate/graduate)."),
        "",
        T("**Fora do escopo:** nao substitui metodos analiticos de referencia "
          "regulamentados (ex.: cromatografia certificada) sem validacao "
          "cruzada formal; nao validado para matrizes/instrumentos fora do "
          "conjunto de calibracao; nao e' um dispositivo medico/forense.",
          "**Out of scope:** does not replace regulated reference analytical "
          "methods (e.g. certified chromatography) without formal cross-"
          "validation; not validated for matrices/instruments outside the "
          "calibration set; not a medical/forensic device."),
        "",
        T("## 3. Fatores Relevantes", "## 3. Relevant Factors"),
        "",
        _md_tabela([
            (T("Classes/especies", "Classes/species"), ", ".join(str(c) for c in classes_unicas)),
            (R("Group-aware (mae_id)"),
             _i18n.valor(resumo.get("Group-aware (mae_id)", "-"), lang)),
            (R("N grupos mae_id"), str(resumo.get("N grupos mae_id", "-"))),
            (R("Imbalance ratio"), f"{resumo.get('Imbalance ratio', 'n/a')}"),
        ]),
        "",
        T("## 4. Metricas de Desempenho (validacao cruzada)",
          "## 4. Performance Metrics (cross-validation)"),
        "",
        _md_tabela([
            (R(k), f"{v:.4f}" if isinstance(v, float) else _i18n.valor(v, lang))
            for k, v in resumo.items()
            if k in ("Accuracy (CV)", "Balanced accuracy", "F1 (macro)",
                     "Cohen's kappa", "R2X", "R2Y", "Q2",
                     "Permutation p-value", "Hotelling T2 (95%)",
                     "Q-residual (95%)", "ROC AUC macro (OvR)",
                     "BCa Accuracy", "BCa Balanced acc.", "BCa F1 (macro)",
                     "BCa Cohen's kappa", "Martens n_significativas",
                     "Martens n_folds_validos", "DModX critical (SIMCA)",
                     "N samples outside DModX")
        ]),
        "",
        T("## 5. Dados de Avaliacao/Treino", "## 5. Evaluation/Training Data"),
        "",
        _md_tabela([
            (R(k), str(resumo.get(k, "-")))
            for k in ("Total samples", "Total variables", "Total classes", "NaN integrity",
                      "Integridade Inf", "Constant variables", "Exact duplicates")
        ]),
        "",
        T("## 6. Acuracia por classe (validacao cruzada)",
          "## 6. Accuracy per class (cross-validation)"),
        "",
        _md_tabela([
            (k.replace("  Acc ", ""), f"{v:.4f}" if isinstance(v, float) else str(v))
            for k, v in resumo.items() if k.startswith("  Acc ")
        ], cabecalho=(T("Classe", "Class"), T("Acuracia", "Accuracy"))),
    ]

    # DD-SIMCA por classe (N2), se disponivel
    ddsimca_linhas = [(k.replace("DD-SIMCA ", "").replace(" sens/esp", ""), str(v))
                      for k, v in resumo.items()
                      if k.startswith("DD-SIMCA ") and k.endswith("sens/esp")]
    if ddsimca_linhas:
        linhas += ["", T("**DD-SIMCA -- sensibilidade/especificidade por classe:**",
                         "**DD-SIMCA -- sensitivity/specificity per class:**"), "",
                   _md_tabela(ddsimca_linhas, cabecalho=(T("Classe", "Class"),
                                                         T("Sens./espec.", "Sens./spec.")))]

    # M-19(c): o holdout externo (dado nunca visto por nenhuma escolha) nao
    # aparecia no model card; so' no resumo. Os numeros que valem como
    # desempenho fora da amostra ficam aqui, ao lado da ressalva de classes.
    _ho_linhas = [(R(k), f"{v:.4f}" if isinstance(v, float) else _i18n.valor(v, lang))
                  for k, v in resumo.items()
                  if k.startswith(("Holdout n", "Holdout accuracy",
                                   "Holdout balanced", "Holdout F1",
                                   "Holdout Cohen", "BCa Holdout",
                                   "Holdout classes"))]
    if _ho_linhas:
        linhas += ["", T("## Holdout externo", "## External holdout"), "",
                   T("Amostras reservadas ANTES de qualquer escolha (VLs, pre-"
                     "processamento, limiares): e' o desempenho fora da amostra.",
                     "Samples set aside BEFORE any choice (LVs, pre-processing, "
                     "thresholds): this is the out-of-sample performance."),
                   "", _md_tabela(_ho_linhas)]

    # Auditoria de delineamento (Bloco 11) -- roda por padrao em toda
    # execucao (pipeline.run_audit), nao opt-in. Icone por
    # severidade para leitura rapida; "silenciado" ainda aparece, com a
    # justificativa anexada na propria mensagem -- nunca some do
    # relatorio so' por ter sido silenciado.
    _achados_audit = resumo.get("auditoria_delineamento")
    if isinstance(_achados_audit, list) and _achados_audit:
        _icone_sev = ({"ok": "OK", "aviso": "WARNING", "critico": "CRITICAL",
                       "silenciado": "SILENCED"} if lang == "en" else
                      {"ok": "OK", "aviso": "AVISO", "critico": "CRITICO",
                       "silenciado": "SILENCIADO"})
        linhas += [
            "", T("## Auditoria de Delineamento", "## Design Audit"), "",
            T("Roda por padrao em toda execucao -- checagens silenciaveis "
              "individualmente, sempre com justificativa registrada (nunca "
              "desaparecem do relatorio so' por terem sido silenciadas).",
              "Runs by default on every run -- checks can be silenced one by "
              "one, always with a recorded reason (they never disappear from "
              "the report just because they were silenced)."),
            "",
            _md_tabela([
                (f"`{a['nome']}`", f"[{_icone_sev.get(a['severidade'], a['severidade'].upper())}] "
                 f"{_i18n.achado(a['mensagem'], lang)}")
                for a in _achados_audit
            ], cabecalho=(T("Checagem", "Check"), T("Resultado", "Result"))),
        ]

    linhas += [
        "",
        T("## 7. Consideracoes Eticas", "## 7. Ethical Considerations"),
        "",
        T("Amostras provenientes de uma unica regiao/instrumento; tamanhos "
          "de referencia por classe podem ser pequenos (ver secao 5). "
          "Resultados nao devem ser generalizados para lotes, safras ou "
          "instrumentos fora do conjunto de calibracao sem revalidacao.",
          "Samples come from a single region/instrument; reference sizes per "
          "class may be small (see section 5). Results must not be "
          "generalised to batches, harvests or instruments outside the "
          "calibration set without revalidation."),
        "",
        T("## 8. Ressalvas e Recomendacoes", "## 8. Caveats and Recommendations"),
        "",
    ]
    for titulo, nota, flag in (_NOTAS_MODEL_CARD_EN if lang == "en" else _NOTAS_MODEL_CARD):
        if flag is None or bool(getattr(cfg, flag, False)):
            linhas.append(f"- **{titulo}:** {nota}")
    # M-03: parametros EFETIVOS (o que de fato rodou) e ajustes de extras por
    # RAM -- o rodape antigo afirmava "nao afeta os resultados" enquanto a RAM
    # livre trocava n_splits_cv e mudava a metrica. Agora o hardware so' pode desligar/reduzir EXTRAS, e tudo fica
    # registrado aqui.
    _ef = resumo.get("parametros_efetivos")
    if isinstance(_ef, dict) and _ef:
        _linhas_ef = []
        for _k, _v in _ef.items():
            if _k == "ajustes_hardware":
                continue
            _linhas_ef.append((f"`{_k}`", _valor_parametro(_v, lang)))
        linhas += ["", T("## Parametros efetivos desta execucao",
                         "## Effective parameters of this run"), "",
                   T("Tudo que define o numero reportado, como foi USADO (e nao "
                     "so' como foi pedido).",
                     "Everything that defines the reported number, as it was USED "
                     "(not only as it was requested)."), "",
                   _md_tabela(_linhas_ef, cabecalho=(T("Parametro", "Parameter"),
                                                     T("Valor", "Value")))]
        _aj = _ef.get("ajustes_hardware") or []
        if _aj:
            linhas += ["", T("**Extras ajustados por RAM livre (nenhum parametro "
                             "cientifico foi alterado):**",
                             "**Extras adjusted to free RAM (no scientific "
                             "parameter was changed):**"), ""]
            linhas += [f"- `{a['parametro']}`: {a['pedido']} -> {a['efetivo']} "
                       f"({a['motivo']})" for a in _aj]
        else:
            linhas += ["", T("Nenhum ajuste por hardware foi aplicado.",
                             "No hardware adjustment was applied.")]
    _rt = hw.get("ram_total_gb")
    _ram_txt = T("nao medida", "not measured") if _rt is None else f"total {_rt} GB"
    linhas += [
        "",
        T(f"*Hardware da execucao: RAM {_ram_txt}, "
          f"CPU {hw.get('cpu_fisicos', '?')} fisicos/{hw.get('cpu_logicos', '?')} "
          f"logicos. O hardware nao altera parametros cientificos; so' extras "
          f"(SHAP, benchmark, Monte Carlo), e cada ajuste aparece acima.*",
          f"*Run hardware: RAM {_ram_txt}, "
          f"CPU {hw.get('cpu_fisicos', '?')} physical/{hw.get('cpu_logicos', '?')} "
          f"logical. Hardware does not change scientific parameters; only extras "
          f"(SHAP, benchmark, Monte Carlo), and every adjustment is listed above.*"),
    ]

    caminho = os.path.join(pasta, "model_card.md")
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel gerar model_card.md: {e}")


def _perfil_txt(perfil: Any, lang: str) -> str:
    """Nome do perfil + descricao. A descricao dos perfis embutidos e' PT; em EN so' o nome."""
    if not getattr(perfil, "descricao", ""):
        return f"`{perfil.nome}`"
    if lang == "en":
        return (f"`{perfil.nome}` (no matrix profile declared -- no matrix default applied)"
                if perfil.nome == "generico" else f"`{perfil.nome}`")
    return f"`{perfil.nome}` ({perfil.descricao})"


def _valor_parametro(v: Any, lang: str) -> str:
    """Valor de `parametros_efetivos` legivel: dict vira `chave=valor; ...`, booleano vira sim/nao."""
    from guaraci import idioma_relatorio as _i18n
    if isinstance(v, bool):
        return _i18n.valor(v, lang)
    if isinstance(v, dict):
        if not v:
            return "-"
        partes = []
        for k, x in v.items():
            if isinstance(x, (list, tuple)):
                x = len(x) if x else 0
            partes.append(f"`{k}`={_valor_parametro(x, lang)}")
        return "; ".join(partes)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(x) for x in v) + "]"
    if isinstance(v, str):
        return _i18n.valor(v, lang)       # enum gravado em PT ("nenhuma") no idioma do relatorio
    return str(v)


def append_regression_model_card(
        pasta: str,
        pooled: Optional[Dict[str, object]] = None,
        tabela_especie: Optional[List[Dict[str, object]]] = None,
        fom_pooled: Optional[Dict[str, float]] = None,
        holdout: Optional[Dict[str, Any]] = None) -> None:
    """Anexa o addendum de regressao (N2/N3) ao model_card.md -- mesmos
    parametros de `append_regression_summary` (chamar as duas juntas nos
    mesmos pontos de executar()), append-only pelo mesmo motivo: a
    regressao roda DEPOIS de `generate_model_card` no fluxo de executar().
    """
    caminho = os.path.join(pasta, "model_card.md")
    if not os.path.isfile(caminho):
        return   # model_card.md nao foi gerado (ex.: generate_model_card falhou)
    from guaraci.idioma_relatorio import t as T

    def _fmt(v: object, nd: int = 3) -> str:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "n/a"
        return f"{fv:.{nd}f}" if np.isfinite(fv) else "n/a"

    linhas: List[str] = ["", T("## 9. Addendum -- Quantificacao (N2/N3, PLS-R)",
                               "## 9. Addendum -- Quantification (N2/N3, PLS-R)"), ""]
    if pooled is not None:
        _linhas_pooled = [
            (T("RMSEP (modelo unico, validacao interna 70/30)",
               "RMSEP (single pooled model, internal 70/30 validation)"), _fmt(pooled.get("rmsep"), 3)),
            (T("R2val (modelo unico, validacao interna)",
               "R2val (single pooled model, internal validation)"), _fmt(pooled.get("r2v"), 4)),
        ]
        if holdout is not None:
            _linhas_pooled.append(
                (T(f"RMSEP (HOLDOUT externo, n={holdout.get('n')}) -- numero publicavel",
                   f"RMSEP (external HOLDOUT, n={holdout.get('n')}) -- publishable number"),
                 _fmt(holdout.get("rmsep"), 3)))
            _linhas_pooled.append(
                (T("R2 (holdout externo)", "R2 (external holdout)"), _fmt(holdout.get("r2"), 3)))
        if pooled.get("dmody_crit") is not None:
            _linhas_pooled.append(
                (T("DModY critico (SIMCA)", "Critical DModY (SIMCA)"), _fmt(pooled.get("dmody_crit"), 3)))
            _linhas_pooled.append(
                (T("Amostras fora do DModY", "Samples outside DModY"),
                 str(pooled.get("n_fora_do_dmody", "n/a"))))
        linhas.append(_md_tabela(_linhas_pooled))
    if tabela_especie:
        def _linha_especie(t: Dict[str, Any]) -> str:
            base = (f"RMSEP={_fmt(t.get('rmsep'), 2)} | "
                    f"LOD={_fmt(t.get('lod'), 2)} | "
                    f"LOQ={_fmt(t.get('loq'), 2)}")
            if t.get("lod_ic_baixo") is None:
                return base
            _conf = t.get("lod_ic_confianca")
            _conf_txt = f"{_conf * 100:.0f}%" if _conf is not None else "?"
            # Bloco 12: LOD/LOQ nunca isolado -- IC (Allegrini & Olivieri
            # 2014) + faixa/desvio-padrao do conjunto de VALIDACAO juntos.
            return (f"{base} | {T('IC', 'CI')}({_conf_txt})=[{_fmt(t.get('lod_ic_baixo'), 2)}-"
                    f"{_fmt(t.get('lod_ic_alto'), 2)}]/"
                    f"[{_fmt(t.get('loq_ic_baixo'), 2)}-"
                    f"{_fmt(t.get('loq_ic_alto'), 2)}] | "
                    f"{T('validacao', 'validation')}=[{_fmt(t.get('validacao_teor_min'), 2)}, "
                    f"{_fmt(t.get('validacao_teor_max'), 2)}] "
                    f"({T('DP', 'SD')}={_fmt(t.get('validacao_teor_dp'), 2)})")

        linhas += ["", T("**Figuras de merito por especie "
                         "(Valderrama, Braga & Poppi, 2009; IC do LOD/LOQ: "
                         "Allegrini & Olivieri, 2014):**",
                         "**Figures of merit per species "
                         "(Valderrama, Braga & Poppi, 2009; LOD/LOQ CI: "
                         "Allegrini & Olivieri, 2014):**"), "",
                   _md_tabela([
                       (str(t.get("especie", "")), _linha_especie(t))
                       for t in tabela_especie
                   ], cabecalho=(T("Especie", "Species"), T("Valor", "Value")))]
    if fom_pooled is not None:
        linhas.append(_md_tabela([
            ("LOD", _fmt(fom_pooled.get('lod'), 2)),
            ("LOQ", _fmt(fom_pooled.get('loq'), 2)),
            (T("Sensibilidade (SEN)", "Sensitivity (SEN)"), _fmt(fom_pooled.get("sensibilidade"), 3)),
        ]))

    try:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel anexar regressao ao model card: {e}")


def append_identification_model_card(pasta: str,
                                      ensemble: Dict[Any, Dict[str, Any]]
                                      ) -> None:
    """Anexa o addendum de Identificacao especie x adulterante (Bloco 9b)
    ao model_card.md -- mesmo padrao append-only de
    `append_regression_model_card` (chamada depois de `generate_model_card`
    no fluxo de `executar()`, sem re-escrever o arquivo).

    Esta e' UM dos 3 lugares onde a ressalva de nao-validacao tem que
    aparecer (D5, Bloco 9b) -- os outros dois sao o log da execucao
    (`pipeline.executar()`) e o manifesto (`predicao.generate_manifest`,
    chave `identification_coverage`). Lista TODA combinacao, nunca so' as
    validadas -- omitir as nao-validadas seria esconder exatamente a
    ressalva que esta secao existe para tornar visivel.

    SEM NUMERO de secao fixo (ao contrario de `append_regression_model_card`,
    que sempre escreve "## 9."): esta funcao e' chamada logo apos o dominio
    de aplicabilidade, ANTES da regressao (secao 9, condicional, so' roda
    quando ha' adulterante+especies suficientes) -- numerar como "## 10."
    produziria "10." aparecendo ANTES de "9." no arquivo sempre que a
    regressao rodasse depois (achado real, verificado lendo o model_card.md
    gerado por uma execucao de teste). Titulo sem numero e' correto em
    qualquer ordem de chamada.
    """
    caminho = os.path.join(pasta, "model_card.md")
    if not os.path.isfile(caminho):
        return
    from guaraci.idioma_relatorio import t as T

    linhas: List[str] = [
        "", T("## Addendum -- Identificacao especie x adulterante (fluxo cego)",
              "## Addendum -- Species x adulterant identification (blind flow)"), "",
        T("Ensemble conformal calibrado por combinacao especie x adulterante "
          "-- `alpha_alcancavel` e' o "
          "MENOR erro que a calibracao atual pode garantir para aquela "
          "combinacao (`1/(n_grupos+1)`); `n/a` quando `n_grupos<=1` (nem um "
          "limiar minimamente informativo pode ser calculado).",
          "Conformal ensemble calibrated per species x adulterant combination "
          "-- `alpha_alcancavel` is the SMALLEST error the current calibration "
          "can guarantee for that combination (`1/(n_grupos+1)`); `n/a` when "
          "`n_grupos<=1` (not even a minimally informative threshold can be "
          "computed)."),
        "",
    ]
    if not ensemble:
        linhas.append(T("*Nenhuma combinacao especie x adulterante calibrada "
                        "nesta execucao (sem `mae_id`, ou dataset sem "
                        "adulterante nomeavel).*",
                        "*No species x adulterant combination calibrated in "
                        "this run (no `mae_id`, or dataset without a named "
                        "adulterant).*"))
    else:
        n_validado = sum(1 for v in ensemble.values()
                          if str(v["cobertura_status"].value) == "validado")
        linhas.append(T(
            f"**{len(ensemble)} combinacoes calibradas, {n_validado} com "
            f"cobertura VALIDADA.** Combinacoes NAO validadas nao "
            f"identificam amostra nenhuma com garantia estatistica -- "
            f"aparecem so' como candidato informacional (ver "
            f"`identificacao.identify_sample`).",
            f"**{len(ensemble)} calibrated combinations, {n_validado} with "
            f"VALIDATED coverage.** NON-validated combinations identify no "
            f"sample with a statistical guarantee -- they appear only as an "
            f"informational candidate (see `identificacao.identify_sample`)."))
        def _fmt_alpha(a: Optional[float]) -> str:
            return "n/a" if a is None else f"{a:.3f}"

        linhas.append("")
        linhas.append(_md_tabela([
            (f"{esp} x {adult}",
             f"n_grupos={v['n_grupos']} | status={v['cobertura_status'].value} | "
             f"alpha_alcancavel={_fmt_alpha(v['alpha_alcancavel'])}")
            for (esp, adult), v in sorted(ensemble.items())
        ], cabecalho=(T("Especie x adulterante", "Species x adulterant"), T("Calibracao", "Calibration"))))

    try:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel anexar identificacao ao model card: {e}")


def append_purity_model_card(pasta: str,
                              modelos_por_especie: Dict[str, Dict[str, Any]]
                              ) -> None:
    """Anexa o addendum de DD-SIMCA de pureza por especie (Detectar,
    Bloco 9b) ao model_card.md -- mesmo padrao append-only das duas
    funcoes acima, mesmo motivo do titulo sem numero fixo.

    Este e' o segundo sinal de Detectar (`predicao.detect_purity`),
    complementar ao dominio de aplicabilidade -- ver docstring de
    `predicao.PurityResult` para a diferenca entre os dois. `confiavel`
    (`n_grupos_calibracao>=3`) e' o mesmo limiar ja usado no aviso de
    `DDSimca.fit` para "regiao larga/conservadora por construcao" --
    abaixo dele, o metodo AINDA decide aceitar/rejeitar, mas sem alpha
    declarado com lastro (fica de fora da soma de Bonferroni).
    """
    caminho = os.path.join(pasta, "model_card.md")
    if not os.path.isfile(caminho):
        return
    from guaraci.idioma_relatorio import t as T

    linhas: List[str] = [
        "", T("## Addendum -- DD-SIMCA de pureza por especie (Detectar, fluxo cego)",
              "## Addendum -- Purity DD-SIMCA per species (Detect, blind flow)"), "",
        T("Segundo sinal de Detectar, complementar ao dominio de "
          "aplicabilidade: o dominio de aplicabilidade e' ajustado em toda a "
          "amostragem (pura + adulterada) e responde 'isto e' parecido com "
          "algo que vimos'; este DD-SIMCA e' ajustado SO' nos puros de cada "
          "especie e responde 'isto e' puro para a especie predita'. Uma "
          "amostra adulterada pode passar no dominio de aplicabilidade (ela "
          "FAZ parte do treino dele) e ainda assim ser rejeitada aqui.",
          "Second Detect signal, complementary to the applicability domain: "
          "the applicability domain is fitted on the whole sampling (pure + "
          "adulterated) and answers 'is this like something we have seen'; "
          "this DD-SIMCA is fitted ONLY on the pure samples of each species "
          "and answers 'is this pure for the predicted species'. An "
          "adulterated sample can pass the applicability domain (it IS part "
          "of its training) and still be rejected here."),
        "",
    ]
    if not modelos_por_especie:
        linhas.append(T("*Nenhum modelo DD-SIMCA de pureza calibrado nesta execucao.*",
                        "*No purity DD-SIMCA model calibrated in this run.*"))
    else:
        def _confiavel(m: Dict[str, Any]) -> bool:
            return ddsimca_alpha_sustentado(
                int(m.get("n_grupos_calibracao", 0)), m.get("sens_logo"))

        n_confiavel = sum(1 for m in modelos_por_especie.values()
                           if _confiavel(m))
        linhas.append(T(
            f"**{len(modelos_por_especie)} especies calibradas, "
            f"{n_confiavel} com alpha sustentado por evidencia** "
            f"(sensibilidade LOGO >= 0,90 com 5 ou mais grupos, ou 19 ou "
            f"mais grupos). Nas demais, a taxa de falsa rejeicao de "
            f"amostras puras novas e' DESCONHECIDA e pode ser alta: o alpha "
            f"declarado (0,05) nao entra na soma de Bonferroni de "
            f"`predict_blind` -- decide aceitar/rejeitar mesmo assim, mas "
            f"sem garantia numerica com lastro.",
            f"**{len(modelos_por_especie)} calibrated species, "
            f"{n_confiavel} with alpha backed by evidence** "
            f"(LOGO sensitivity >= 0.90 with 5 or more groups, or 19 or "
            f"more groups). For the others the false-rejection rate of new "
            f"pure samples is UNKNOWN and may be high: the declared alpha "
            f"(0.05) does not enter the Bonferroni sum of `predict_blind` -- "
            f"it still decides accept/reject, but without a numeric "
            f"guarantee behind it."))
        linhas.append("")
        def _celula(m: Dict[str, Any]) -> str:
            _sens = m.get("sens_logo")
            _txt_sens = f"sens_logo={_sens:.2f} | " if _sens is not None else ""
            return (f"n_grupos_calibracao={m.get('n_grupos_calibracao', 'n/a')}"
                    f" | {_txt_sens}"
                    + T(f"confiavel={'sim' if _confiavel(m) else 'nao'}",
                        f"reliable={'yes' if _confiavel(m) else 'no'}"))

        linhas.append(_md_tabela([
            (esp, _celula(m))
            for esp, m in sorted(modelos_por_especie.items())
        ], cabecalho=(T("Especie", "Species"), T("Calibracao", "Calibration"))))

    try:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel anexar pureza ao model card: {e}")


def append_linearity_robustness_model_card(
        pasta: str,
        linearidade: Optional["LackOfFitResult"] = None,
        robustez: Optional[Dict[str, "RobustnessResult"]] = None,
        ) -> None:
    """Anexa o addendum de linearidade formal + robustez (Bloco 13d) ao
    model_card.md -- mesmo padrao append-only das funcoes acima, mesmo
    motivo do titulo sem numero fixo (chamada em ponto variavel do fluxo,
    dependendo de quais diagnosticos o objetivo/nivel ligou).

    `linearidade`: resultado de `linearity.lack_of_fit_test` (ou `None` --
    diagnostico nao rodado nesta execucao). `robustez`: dict nome->
    `robustness.RobustnessResult` de `robustness.run_robustness_protocol`
    (ou `None`). Os dois sao independentes -- qualquer combinacao
    presente/ausente e' valida (cada um pode nao ser computavel/aplicavel
    sem o outro)."""
    caminho = os.path.join(pasta, "model_card.md")
    if not os.path.isfile(caminho):
        return
    if linearidade is None and not robustez:
        return
    from guaraci.idioma_relatorio import t as T

    def _fmt(v: Optional[float], nd: int = 4) -> str:
        if v is None:
            return "n/a"
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return "n/a"
        return f"{fv:.{nd}f}" if np.isfinite(fv) else "n/a"

    linhas: List[str] = [
        "", T("## Addendum -- Linearidade formal e robustez",
              "## Addendum -- Formal linearity and robustness"), "",
    ]

    if linearidade is not None:
        linhas += [T("**Teste de falta de ajuste** -- "
                     "Draper & Smith, *Applied Regression Analysis*, cap. "
                     "2.6. H0: a reta de calibracao e' adequada (sem "
                     "curvatura sistematica detectavel); rejeitar H0 "
                     "(p < alpha) indica falta de ajuste significativa.",
                     "**Lack-of-fit test** -- Draper & Smith, *Applied "
                     "Regression Analysis*, ch. 2.6. H0: the calibration line "
                     "is adequate (no detectable systematic curvature); "
                     "rejecting H0 (p < alpha) means significant lack of fit."), ""]
        if not linearidade.computavel:
            linhas.append(T(f"*Nao computavel: {linearidade.motivo}*",
                            f"*Not computable: {linearidade.motivo}*"))
        else:
            # Com niveis = grupos (erro puro so' de replicas tecnicas) o teste tende a rejeitar
            # linearidade mesmo numa curva reta (limitacao documentada em `linearity.py`): nesse
            # caso o veredito de rejeicao e' "nao conclusivo", nunca "curvatura detectada"
            # (auditoria de 2026-11-10: o exemplo `quantificar`, linear por construcao, saia
            # com "curvatura detectada").
            so_replicas_tecnicas = getattr(linearidade, "nivel_por", "conc") == "grupo"
            if linearidade.linear:
                veredito = T("sem falta de ajuste detectavel", "no detectable lack of fit")
            elif so_replicas_tecnicas:
                veredito = T("NAO CONCLUSIVO (o erro puro vem so' de replicas tecnicas e subestima o erro real)",
                             "INCONCLUSIVE (pure error comes from technical replicates only and underestimates the real error)")
            else:
                veredito = T("FALTA DE AJUSTE SIGNIFICATIVA (curvatura detectada)",
                             "SIGNIFICANT LACK OF FIT (curvature detected)")
            linhas.append(_md_tabela([
                ("F", _fmt(linearidade.F, 3)),
                (T("p-valor", "p-value"), _fmt(linearidade.p_value, 4)),
                (T("graus de liberdade (falta de ajuste, erro puro)",
                   "degrees of freedom (lack of fit, pure error)"),
                 f"{linearidade.df_lof}, {linearidade.df_pe}"),
                (T("niveis da curva (com replica)", "curve levels (with replicate)"),
                 f"{linearidade.n_niveis} ({linearidade.n_niveis_com_replica})"),
                ("alpha", _fmt(linearidade.alpha, 2)),
                (T("veredito", "verdict"), veredito),
            ]))
            aviso = getattr(linearidade, "aviso_erro_puro", None)
            if aviso:
                # texto fixo nos dois idiomas (o campo do resultado fica em PT, para o log)
                linhas += ["", T(f"*Ressalva:* {aviso}",
                                 "*Caveat:* pure error is estimated from technical replicates of the same "
                                 "mae_id only (it underestimates preparation error) and independent preparations "
                                 "of the same concentration count as distinct levels: the test tends to reject "
                                 "linearity -- use nivel_por='conc' if there are independent preparations")]
        linhas.append("")

    if robustez:
        linhas += [T("**Protocolo de robustez** -- variacao do resultado "
                     "final sob perturbacao controlada (pre-processamento, "
                     "ruido/deriva sinteticos). Reportado como INTERVALO "
                     "(minimo-maximo), nao como aprovado/reprovado -- mesma "
                     "filosofia do LOD/LOQ como intervalo: "
                     "declarar o que os dados sustentam.",
                     "**Robustness protocol** -- variation of the final result "
                     "under controlled perturbation (pre-processing, synthetic "
                     "noise/drift). Reported as an INTERVAL (minimum-maximum), "
                     "not as pass/fail -- same philosophy as LOD/LOQ as an "
                     "interval: state what the data support."), ""]
        linhas.append(_md_tabela([
            (nome, T(f"referencia={_fmt(r.baseline, 4)} | "
                     f"intervalo=[{_fmt(r.minimo, 4)}, {_fmt(r.maximo, 4)}] | "
                     f"mediana={_fmt(r.mediana, 4)} | "
                     f"variacao_absoluta={_fmt(r.variacao_absoluta, 4)} | "
                     f"n_replicas={r.n_replicas}",
                     f"baseline={_fmt(r.baseline, 4)} | "
                     f"interval=[{_fmt(r.minimo, 4)}, {_fmt(r.maximo, 4)}] | "
                     f"median={_fmt(r.mediana, 4)} | "
                     f"absolute_variation={_fmt(r.variacao_absoluta, 4)} | "
                     f"n_replicates={r.n_replicas}"))
            for nome, r in sorted(robustez.items())
        ], cabecalho=(T("Perturbacao", "Perturbation"), T("Resultado", "Result"))))
        linhas.append("")

    try:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel anexar linearidade/robustez ao "
              f"model card: {e}")


def append_correcao_sinal_model_card(
        pasta: str, veredictos: List["VeredictoCorrecaoSinal"]) -> None:
    """Anexa o veredito do portao de aceite de correcao de sinal (Bloco 20,
    `portao_correcao_sinal.avaliar_correcao_sinal`/`_pls`) ao model_card.md
    -- mesmo padrao append-only das funcoes acima, mesmo motivo do titulo
    sem numero fixo (o portao pode rodar em pontos variados do fluxo,
    dependendo de quais tecnicas de correcao o usuario tem configuradas).

    NUNCA esconder um veredito 'neutro'/'rejeitado' -- e' exatamente o que
    o Bloco 20 existe para tornar visivel (uma tecnica sem prova de ganho
    continua DISPONIVEL, mas o usuario ve o veredito ao lado da opcao).
    Lista TODO veredito recebido, na ordem dada -- nunca filtra so' os
    aprovados."""
    caminho = os.path.join(pasta, "model_card.md")
    if not os.path.isfile(caminho):
        return
    if not veredictos:
        return
    from guaraci.idioma_relatorio import t as T

    linhas: List[str] = [
        "", T("## Addendum -- Portao de aceite de correcao de sinal",
              "## Addendum -- Acceptance gate for signal correction"), "",
        T("Cada tecnica de correcao de sinal so' e' recomendada se passar por "
          "este portao: mesmo pipeline com/sem a tecnica, sob o MESMO split "
          "agrupado bloqueado, comparado por teste de Wilcoxon pareado "
          "(mesmo metodo validado na comparacao N-PLS vs. PLS-DA por pixel"
          "). 'Neutro'/'rejeitado' nao remove a tecnica do leque "
          "configuravel -- so' declara que o ganho nao esta' comprovado "
          "neste dataset/cenario.",
          "Each signal-correction technique is only recommended if it passes "
          "this gate: the same pipeline with/without the technique, under the "
          "SAME locked grouped split, compared by a paired Wilcoxon test (same "
          "method validated in the per-pixel N-PLS vs. PLS-DA comparison). "
          "'Neutral'/'rejected' does not remove the technique from the "
          "configurable set -- it only states that the gain is not proven on "
          "this dataset/scenario."), "",
    ]
    linhas.append(_md_tabela([
        (v.metodo, T(f"veredito={v.veredito} | metrica={v.metrica} "
                     f"sem={v.valor_sem:.4g} com={v.valor_com:.4g} | "
                     f"efeito_padronizado={v.tamanho_efeito_padronizado:.3f} | "
                     f"p={v.p_valor:.4f} | n_pares={v.n_pares} | "
                     f"poder_suficiente={'sim' if v.poder_suficiente else 'nao'}",
                     f"verdict={v.veredito} | metric={v.metrica} "
                     f"without={v.valor_sem:.4g} with={v.valor_com:.4g} | "
                     f"standardised_effect={v.tamanho_efeito_padronizado:.3f} | "
                     f"p={v.p_valor:.4f} | n_pairs={v.n_pares} | "
                     f"enough_power={'yes' if v.poder_suficiente else 'no'}"))
        for v in veredictos
    ], cabecalho=(T("Tecnica", "Technique"), T("Resultado", "Result"))))
    linhas.append("")

    try:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel anexar portao de correcao de "
              f"sinal ao model card: {e}")


def append_politica_pooled_local_model_card(
        pasta: str, decisoes: List["DecisaoPooledLocal"]) -> None:
    """Anexa a decisao pooled vs. local por especie (Bloco 26,
    `politica_pooled_local.decidir_pooled_vs_local`) ao model_card.md --
    mesmo padrao append-only das funcoes acima. Lista TODA especie
    avaliada, inclusive as com dados insuficientes (recomendacao pooled
    por definicao, nunca escondida) -- nunca so' as que foram pra local."""
    caminho = os.path.join(pasta, "model_card.md")
    if not os.path.isfile(caminho):
        return
    if not decisoes:
        return
    from guaraci.idioma_relatorio import t as T

    linhas: List[str] = [
        "", T("## Addendum -- Modelo unico (pooled) vs. local por especie",
              "## Addendum -- Single pooled vs. local model per species"), "",
        T("Local (modelo separado por especie) so' e' recomendado quando a "
          "especie tem amostras suficientes (mesmo limiar de "
          "`pls_regression_by_species`) E o portao de aceite "
          "aprova o ganho sobre o modelo unico com poder estatistico suficiente "
          "-- nunca so' porque o RMSEP local saiu menor num unico split.",
          "Local (a separate model per species) is only recommended when the "
          "species has enough samples (same threshold as "
          "`pls_regression_by_species`) AND the acceptance gate approves the "
          "gain over the pooled model with enough statistical power -- never "
          "just because the local RMSEP came out lower on a single split."), "",
    ]
    linhas.append(_md_tabela([
        (d.especie,
         T(f"recomendacao={d.recomendacao} | n_amostras={d.n_amostras} "
           f"(minimo={d.n_minimo}) | ",
           f"recommendation={d.recomendacao} | n_samples={d.n_amostras} "
           f"(minimum={d.n_minimo}) | ")
         + (T("veredito", "verdict") + f"={d.veredito.veredito} "
            f"(p={d.veredito.p_valor:.4f}, n={d.veredito.n_pares})"
            if d.veredito is not None else
            T("portao nao rodado (dados insuficientes)", "gate not run (insufficient data)")))
        for d in decisoes
    ], cabecalho=(T("Especie", "Species"), T("Decisao", "Decision"))))
    linhas.append("")

    try:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel anexar politica pooled vs. "
              f"local ao model card: {e}")
