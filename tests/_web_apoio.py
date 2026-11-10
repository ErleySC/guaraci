"""Apoio dos testes das abas web: pasta de resultados sintetica e tokens de cor, importados de dentro das funcoes do AppTest."""
from __future__ import annotations

import os

#: tokens de cor que as abas leem (mesmas chaves do tema real)
TOKENS = {"success_bg": "#e6ffe6", "success": "#060", "warn_bg": "#fff7e0", "warn": "#860", "error_bg": "#ffe6e6", "error": "#900",
          "bg": "#fff", "text": "#000", "muted": "#666", "primary": "#06c", "border": "#ccc", "card": "#f5f5f5"}

RESUMO = """Optimal LVs: 5
Balanced accuracy: 0.91
Acc Classe A: 0.95
Acc Classe B: 0.80
Acc Classe C: 0.50

--- Methodological Notes ---
Notas para a revisao por pares.
"""


def _png(caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1])
    fig.savefig(caminho)
    plt.close(fig)


def criar_pasta_de_resultados(base, *, figuras=("fig1_pca_scores.png", "fig_roc_auc.png"), resumo=RESUMO, model_card="# Model card\n",
                              benchmark=True, mc=True, execucoes_irmas=0):
    """Cria <base>/<execucao>/{Graficos,Tabelas,Relatorios} e devolve o caminho da execucao."""
    raiz = os.path.join(str(base), "resultados")
    pasta = os.path.join(raiz, "execucao_atual")
    for sub in ("Graficos", "Tabelas", "Relatorios"):
        os.makedirs(os.path.join(pasta, sub), exist_ok=True)
    for nome in figuras:
        _png(os.path.join(pasta, "Graficos", nome))
    if resumo is not None:
        for sub in ("Relatorios", "Logs"):
            os.makedirs(os.path.join(pasta, sub), exist_ok=True)
            with open(os.path.join(pasta, sub, "resumo_modelo.txt"), "w", encoding="utf-8") as f:
                f.write(resumo)
    if model_card is not None:
        with open(os.path.join(pasta, "model_card.md"), "w", encoding="utf-8") as f:
            f.write(model_card)
    if benchmark:
        with open(os.path.join(pasta, "Tabelas", "benchmark_classifiers.csv"), "w", encoding="utf-8") as f:
            f.write("Modelo;Acuracia\nPLS-DA;0,9\n")
    if mc:
        with open(os.path.join(pasta, "Tabelas", "monte_carlo_cv.csv"), "w", encoding="utf-8") as f:
            f.write("Metrica;IC\nacc;0,8\n")
    for i in range(execucoes_irmas):
        os.makedirs(os.path.join(raiz, f"execucao_antiga_{i}"), exist_ok=True)
    return pasta
