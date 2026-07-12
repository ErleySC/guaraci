"""resultados_io.py — Escrita dos artefatos de resultado em disco: resumo do
modelo (resumo_modelo.txt), model card (model_card.md) e identificadores das
amostras (CSV). Também as notas metodológicas e as métricas do modelo PLS.

Extraído de pipeline.py (dívida técnica pós-auditoria): estas funções são
CHAMADAS por executar() para persistir os resultados de uma corrida, mas não
dependem do orquestrador — só de config (__version__, _NIVEL_NOME), numpy,
pandas e sklearn. pipeline.py reexporta todos os nomes, então
`pipeline.salvar_resumo_modelo(...)`, `pq.gerar_model_card(...)`,
`pq.anexar_regressao_resumo(...)` etc. seguem funcionando (executar e os
testes consomem via fachada).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from guaraci.config import Config, __version__, _NIVEL_NOME


def metricas_modelo_pls(modelo: PLSRegression, X: np.ndarray, Y: np.ndarray,
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


def salvar_identificadores(rotulos: np.ndarray, pred_lab: np.ndarray,
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
#: compartilhadas por salvar_resumo_modelo (.txt) e gerar_model_card (.md),
#: fonte unica para nao divergirem com o tempo.
_NOTAS_METODOLOGICAS: List[Tuple[str, str]] = [
    ("LV selection", "Wold parsimony criterion (2% RMSECV tolerance above "
     "minimum). Prevents overfitting on broad RMSECV plateaus with "
     "high max_lvs. Reference: Wold (1978) Technometrics 20:397-405."),
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


def salvar_resumo_modelo(pasta: str, info: Dict[str, object]) -> None:
    caminho = os.path.join(pasta, "resumo_modelo.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  PLS-DA Model Summary\n")
        f.write("=" * 60 + "\n\n")
        largura = max(len(k) for k in info.keys()) + 2
        for k, v in info.items():
            if isinstance(v, float):
                f.write(f"  {k:<{largura}s}: {v:.4f}\n")
            else:
                f.write(f"  {k:<{largura}s}: {v}\n")

        # Methodological transparency notes (for article Methods section)
        f.write("\n" + "-" * 60 + "\n")
        f.write("  Methodological Notes (for article peer review)\n")
        f.write("-" * 60 + "\n")
        for title, note in _NOTAS_METODOLOGICAS:
            f.write(f"\n  [{title}]\n")
            # Word-wrap at 70 chars
            words = note.split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 70:
                    f.write(line + "\n")
                    line = "    " + w + " "
                else:
                    line += w + " "
            if line.strip():
                f.write(line + "\n")


def anexar_regressao_resumo(
        pasta: str,
        pooled: Optional[Dict[str, object]] = None,
        tabela_especie: Optional[List[Dict[str, object]]] = None,
        fom_pooled: Optional[Dict[str, float]] = None) -> None:
    """Anexa o bloco de regressao PLS (com figuras de merito analiticas de
    Valderrama, Braga & Poppi 2009) ao resumo_modelo.txt.

    Motivacao: LOD/LOQ/SEN/SEL/gamma so eram impressos no console/log; agora
    tambem entram no resumo persistido (que a aba Relatorios do app captura).
    E append-only DE PROPOSITO: a regressao roda depois de o resumo ser gravado
    a 1a vez em executar(), e reordenar esse orquestrador seria arriscado.
    """
    caminho = os.path.join(pasta, "resumo_modelo.txt")

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
        linhas.append(f"    RMSEP .....: {_fmt(pooled.get('rmsep'))}")
        if pooled.get('bias') is not None:
            linhas.append(f"    Bias ......: {_fmt(pooled.get('bias'), 4)}")
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
                f"{int(t.get('n_lv', 0) or 0):>3d} "
                f"{_fmt(t.get('rmsep'), 2):>7s} {_fmt(t.get('r2val'), 3):>6s} "
                f"{_fmt(t.get('lod'), 2):>7s} {_fmt(t.get('loq'), 2):>7s} "
                f"{_fmt(t.get('sensibilidade'), 3):>7s} "
                f"{_fmt(t.get('seletividade_media'), 3):>6s}")
    if fom_pooled is not None:
        linhas.append("")
        linhas.append("  Figures of merit (single pooled model):")
        linhas.append(f"    LOD .......: {_fmt(fom_pooled.get('lod'), 2, '%')}")
        linhas.append(f"    LOQ .......: {_fmt(fom_pooled.get('loq'), 2, '%')}")
        linhas.append(f"    SEN .......: {_fmt(fom_pooled.get('sensibilidade'), 3)}")
        linhas.append(f"    gamma .....: {_fmt(fom_pooled.get('sensibilidade_analitica'), 2)}")
        linhas.append(f"    SEL .......: {_fmt(fom_pooled.get('seletividade_media'), 3)}")
        linhas.append(f"    delta_x ...: {_fmt(fom_pooled.get('delta_x_ruido'), 5)}")
    linhas.append("")
    linhas.append("  Note: LOD/LOQ/SEN require physical replicates (mae_id) to")
    linhas.append("  estimate instrumental noise; 'n/a' means none available.")

    try:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except Exception as e:
        print(f"  [AVISO] Nao foi possivel anexar regressao ao resumo: {e}")


def anexar_heatmap_resumo(pasta: str, resultado: Dict[str, object]) -> None:
    """Anexa ao resumo_modelo.txt o balanco do heatmap especie x adulterante
    (R2cv por combinacao). Deixa EXPLICITO quantas combinacoes NAO atingem o
    limiar de aceite -- uma quantificacao que so funciona em parte das
    combinacoes nao deve ser lida como sucesso geral. Append-only pelo mesmo
    motivo de anexar_regressao_resumo (roda depois do 1o flush do resumo).
    """
    caminho = os.path.join(pasta, "resumo_modelo.txt")
    limiar   = float(resultado.get("limiar_r2", 0.70))   # type: ignore[arg-type]
    n_falhas = int(resultado.get("n_falhas", 0))          # type: ignore[arg-type]
    n_ok     = int(resultado.get("n_ok", 0))              # type: ignore[arg-type]
    n_na     = int(resultado.get("n_na", 0))              # type: ignore[arg-type]
    n_total  = int(resultado.get("n_total", 0))           # type: ignore[arg-type]
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
                fr = float(r2)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if np.isfinite(fr) and fr < limiar:
                abaixo.append(f"    {esp} x {ad}: R2cv={fr:.3f}")
        linhas.append("")
        linhas.append("  Combinacoes abaixo do limiar (nao quantificaveis):")
        linhas.extend(abaixo or
                      ["    (nenhuma — todas as combinacoes com dados passaram)"])
    try:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except Exception as e:
        print(f"  [AVISO] Nao foi possivel anexar heatmap ao resumo: {e}")


# =========================================================================
#  Model Card (Mitchell et al. 2019, "Model Cards for Model Reporting",
#  FAT* '19) -- resumo de 1 documento por execucao para uso pretendido,
#  dados, metricas e limitacoes. Montagem a partir de dados JA CALCULADOS
#  (resumo/hardware/config) -- nao introduz ciencia nova, so' organiza o
#  que ja existe no padrao esperado por quem avalia "ferramenta profissional"
#  (ML-ops/auditoria), analogo ao model card do Hugging Face Hub.
# =========================================================================

def _md_tabela(linhas: List[Tuple[str, str]]) -> str:
    """Monta uma tabela Markdown de 2 colunas (Campo | Valor)."""
    out = ["| Campo | Valor |", "|---|---|"]
    for k, v in linhas:
        out.append(f"| {k} | {v} |")
    return "\n".join(out)


def gerar_model_card(pasta: str, cfg: "Config", resumo: Dict[str, object],
                      hw: Dict[str, Any], classes_unicas: np.ndarray) -> None:
    """Gera `model_card.md` (secoes de Mitchell et al. 2019, adaptado a um
    pipeline quimiometrico de autenticacao/quantificacao). Escrito no MESMO
    ponto de `salvar_resumo_modelo` -- reaproveita o dict `resumo` inteiro,
    ja com todas as metricas/diagnosticos/integridade de dados calculados,
    em vez de recalcular ou receber duzias de parametros separados.

    Regressao (N2/N3) e' um addendum ANEXADO depois (mesmo padrao de
    `anexar_regressao_resumo`), pois so' fica disponivel mais tarde em
    executar() -- ver `anexar_regressao_model_card`.
    """
    nivel = cfg.nivel
    nivel_nome = _NIVEL_NOME.get(nivel, nivel)
    agora = datetime.now().strftime("%Y-%m-%d %H:%M")

    uso_pretendido = {
        "N1": ("Identificacao de especie de oleo vegetal amazonico a partir "
               "de espectro FT-NIR (classificacao multiclasse via PLS-DA)."),
        "N2": ("Autenticacao de pureza POR ESPECIE (puro vs. adulterado) via "
               "DD-SIMCA one-class, a partir de espectro FT-NIR."),
        "N3": ("Quantificacao do teor (%) de adulterante em oleo vegetal "
               "amazonico, calibrada separadamente por especie (PLS-R)."),
    }.get(nivel, "Analise quimiometrica de espectros FT-NIR.")

    linhas: List[str] = [
        f"# Model Card -- GUARACI v{__version__}",
        "",
        f"*Gerado automaticamente em {agora}. Este documento descreve os "
        "artefatos de UMA execucao do pipeline; regenere a cada novo "
        "conjunto de dados ou configuracao.*",
        "",
        "## 1. Detalhes do Modelo",
        "",
        _md_tabela([
            ("Plataforma", "GUARACI -- Chemometrics Platform"),
            ("Versao", __version__),
            ("Tipo de analise", f"{nivel} ({nivel_nome})"),
            ("Algoritmo principal", "PLS-DA (Partial Least Squares "
             "Discriminant Analysis)" if nivel != "N3" else
             "PLS-DA + PLS-R por especie"),
            ("Pre-processamento", str(resumo.get("Pre-processamento", "-"))),
            ("Faixa espectral", str(resumo.get("Faixa espectral (cm-1)", "-"))),
            ("Tag de execucao", str(resumo.get("Tag", "-"))),
            ("Licenca", "GPL-3.0-or-later (dual-licensed -- ver docs/COMMERCIAL.md)"),
            ("Instituicao", "GEAAp / UFPA"),
            ("Repositorio", "github.com/ErleySC/guaraci"),
        ]),
        "",
        "## 2. Uso Pretendido",
        "",
        f"**Uso primario:** {uso_pretendido}",
        "",
        "**Usuarios primarios:** pesquisadores em quimiometria, laboratorios "
        "de controle de qualidade de oleos vegetais, projetos academicos "
        "(TCC/PIBIC/pos-graduacao).",
        "",
        "**Fora do escopo:** nao substitui metodos analiticos de referencia "
        "regulamentados (ex.: cromatografia certificada) sem validacao "
        "cruzada formal; nao validado para matrizes/instrumentos fora do "
        "dataset de calibracao; nao e' um dispositivo medico/forense.",
        "",
        "## 3. Fatores Relevantes",
        "",
        _md_tabela([
            ("Classes/especies", ", ".join(str(c) for c in classes_unicas)),
            ("Validacao group-aware (mae_id)",
             str(resumo.get("Group-aware (mae_id)", "-"))),
            ("N grupos (mae_id)", str(resumo.get("N grupos mae_id", "-"))),
            ("Razao de desbalanceamento",
             f"{resumo.get('Imbalance ratio', 'n/a')}"),
        ]),
        "",
        "## 4. Metricas de Desempenho (validacao cruzada)",
        "",
        _md_tabela([
            (k, f"{v:.4f}" if isinstance(v, float) else str(v))
            for k, v in resumo.items()
            if k in ("Accuracy (CV)", "Balanced accuracy", "F1 (macro)",
                     "Cohen's kappa", "R2X", "R2Y", "Q2",
                     "Permutation p-value", "Hotelling T2 (95%)",
                     "Q-residual (95%)", "ROC AUC macro (OvR)",
                     "BCa Accuracy", "BCa Balanced acc.", "BCa F1 (macro)",
                     "BCa Cohen's kappa", "Martens n_significativas",
                     "Martens n_folds_validos", "DModX critico (SIMCA)",
                     "N amostras fora do DModX")
        ]),
        "",
        "## 5. Dados de Avaliacao/Treino",
        "",
        _md_tabela([
            ("Total de amostras", str(resumo.get("Total de amostras", "-"))),
            ("Total de variaveis", str(resumo.get("Total de variaveis", "-"))),
            ("Total de classes", str(resumo.get("Total de classes", "-"))),
            ("Amostras com NaN removidas",
             str(resumo.get("Integridade NaN", "-"))),
            ("Amostras com Inf removidas",
             str(resumo.get("Integridade Inf", "-"))),
            ("Variaveis constantes removidas",
             str(resumo.get("Variaveis constantes", "-"))),
            ("Duplicatas exatas", str(resumo.get("Duplicatas exatas", "-"))),
        ]),
        "",
        "## 6. Analises Quantitativas (por classe)",
        "",
        _md_tabela([
            (k.replace("  Acc ", ""), f"{v:.4f}" if isinstance(v, float) else str(v))
            for k, v in resumo.items() if k.startswith("  Acc ")
        ]),
    ]

    # DD-SIMCA por classe (N2), se disponivel
    ddsimca_linhas = [(k.replace("DD-SIMCA ", "").replace(" sens/esp", ""), str(v))
                      for k, v in resumo.items()
                      if k.startswith("DD-SIMCA ") and k.endswith("sens/esp")]
    if ddsimca_linhas:
        linhas += ["", "**DD-SIMCA -- sensibilidade/especificidade por classe:**", "",
                   _md_tabela(ddsimca_linhas)]

    linhas += [
        "",
        "## 7. Consideracoes Eticas",
        "",
        "Amostras provenientes de uma unica regiao/instrumento; tamanhos "
        "de referencia por classe podem ser pequenos (ver secao 5). "
        "Resultados nao devem ser generalizados para lotes, safras ou "
        "instrumentos fora do conjunto de calibracao sem revalidacao.",
        "",
        "## 8. Ressalvas e Recomendacoes",
        "",
    ]
    for titulo, nota in _NOTAS_METODOLOGICAS:
        linhas.append(f"- **{titulo}:** {nota}")
    linhas += [
        "",
        f"*Hardware da execucao: RAM total {hw.get('ram_total_gb', '?')} GB, "
        f"CPU {hw.get('cpu_fisicos', '?')} fisicos/{hw.get('cpu_logicos', '?')} "
        f"logicos (nao afeta os resultados, so' o tempo de execucao).*",
    ]

    caminho = os.path.join(pasta, "model_card.md")
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except Exception as e:
        print(f"  [AVISO] Nao foi possivel gerar model_card.md: {e}")


def anexar_regressao_model_card(
        pasta: str,
        pooled: Optional[Dict[str, object]] = None,
        tabela_especie: Optional[List[Dict[str, object]]] = None,
        fom_pooled: Optional[Dict[str, float]] = None) -> None:
    """Anexa o addendum de regressao (N2/N3) ao model_card.md -- mesmos
    parametros de `anexar_regressao_resumo` (chamar as duas juntas nos
    mesmos pontos de executar()), append-only pelo mesmo motivo: a
    regressao roda DEPOIS de `gerar_model_card` no fluxo de executar().
    """
    caminho = os.path.join(pasta, "model_card.md")
    if not os.path.isfile(caminho):
        return   # model_card.md nao foi gerado (ex.: gerar_model_card falhou)

    def _fmt(v: object, nd: int = 3) -> str:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "n/a"
        return f"{fv:.{nd}f}" if np.isfinite(fv) else "n/a"

    linhas: List[str] = ["", "## 9. Addendum -- Quantificacao (N2/N3, PLS-R)", ""]
    if pooled is not None:
        _linhas_pooled = [
            ("RMSEP (pooled)", _fmt(pooled.get("rmsep"), 3)),
            ("R2val (pooled)", _fmt(pooled.get("r2v"), 4)),
        ]
        if pooled.get("dmody_crit") is not None:
            _linhas_pooled.append(
                ("DModY critico (SIMCA)", _fmt(pooled.get("dmody_crit"), 3)))
            _linhas_pooled.append(
                ("Amostras fora do DModY", str(pooled.get("n_fora_do_dmody", "n/a"))))
        linhas.append(_md_tabela(_linhas_pooled))
    if tabela_especie:
        linhas += ["", "**Figuras de merito por especie "
                   "(Valderrama, Braga & Poppi, 2009):**", "",
                   _md_tabela([
                       (str(t.get("especie", "")),
                        f"RMSEP={_fmt(t.get('rmsep'), 2)} | "
                        f"LOD={_fmt(t.get('lod'), 2)}% | "
                        f"LOQ={_fmt(t.get('loq'), 2)}%")
                       for t in tabela_especie
                   ])]
    if fom_pooled is not None:
        linhas.append(_md_tabela([
            ("LOD", f"{_fmt(fom_pooled.get('lod'), 2)}%"),
            ("LOQ", f"{_fmt(fom_pooled.get('loq'), 2)}%"),
            ("Sensibilidade (SEN)", _fmt(fom_pooled.get("sensibilidade"), 3)),
        ]))

    try:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
    except Exception as e:
        print(f"  [AVISO] Nao foi possivel anexar regressao ao model card: {e}")
