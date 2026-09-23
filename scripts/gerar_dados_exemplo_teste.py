"""Gera `dados_exemplo.csv` -- um conjunto pequeno de espectros SINTETICOS
(3 classes, 20 amostras por classe, 200 variaveis) para quem quer ver o
GUARACI rodar sem ter dado proprio (roteiro de teste de usabilidade).

Nao e' dado real: as classes so' diferem por picos artificiais. Uso:
    python scripts/gerar_dados_exemplo_teste.py [saida.csv]
Formato: 1 coluna por variavel espectral (cabecalho = numero de onda),
mais a coluna `classe` -- o que o modo `csv` do GUARACI espera."""
import sys

import numpy as np
import pandas as pd

saida = sys.argv[1] if len(sys.argv) > 1 else "dados_exemplo.csv"
rng = np.random.default_rng(7)
wn = np.linspace(4000, 10000, 200)
picos = {"Andiroba": (5200, 7000), "Copaiba": (5600, 7400), "Buriti": (4800, 8200)}
linhas, rotulos = [], []
for classe, (p1, p2) in picos.items():
    base = (np.exp(-((wn - p1) / 250) ** 2) + 0.7 * np.exp(-((wn - p2) / 300) ** 2))
    for _ in range(20):
        ganho = rng.normal(1.0, 0.05)
        deriva = rng.normal(0, 0.02) * (wn - wn.min()) / (wn.max() - wn.min())
        linhas.append(ganho * base + deriva + rng.normal(0, 0.01, wn.size))
        rotulos.append(classe)
df = pd.DataFrame(linhas, columns=[f"{w:.2f}" for w in wn])
df["classe"] = rotulos
df.to_csv(saida, index=False)
print(f"{len(df)} amostras, {len(wn)} variaveis -> {saida}")
