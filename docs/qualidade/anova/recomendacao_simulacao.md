# Simulacao da recomendacao de metodo (2000 repeticoes por cenario e por condicao)

**Público:** quem confere o que foi verificado · **Estado:** gerado por script (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Verdade conhecida: o metodo correto e' o que o mecanismo gerador pede. Medias iguais para o erro tipo I; diferenca de 1.5 desvio no grupo 3 para o poder. Nivel nominal 0.05.

## 1. A recomendacao acerta?

| cenario | correto | recomendou Tukey | Games-Howell | Kruskal/Dunn | acerto |
|---|---|---|---|---|---|
| normal, variancias iguais, n=10 | tukey | 86.9% | 12.8% | 0.2% | **86.9%** |
| normal, variancias iguais, n=4 | tukey | 48.8% | 51.1% | 0.1% | **48.8%** |
| normal, variancias iguais, n desigual 5/10/25 | tukey | 82.1% | 17.8% | 0.1% | **82.1%** |
| normal, variancia 2:1, n=10 | tukey | 70.3% | 29.4% | 0.2% | **70.3%** |
| normal, variancias 1:10, n=10 | games-howell | 7.8% | 92.0% | 0.2% | **92.0%** |
| normal, variancias 1:10, n desigual (grupo de maior variancia tem n=5) | games-howell | 12.2% | 87.7% | 0.1% | **87.7%** |
| t com 3 gl (caudas pesadas), n=15 | kruskal-dunn | 60.8% | 25.4% | 13.9% | **13.9%** |
| lognormal (assimetrica), n=15 | kruskal-dunn | 4.0% | 3.7% | 92.3% | **92.3%** |

## 2. Erro tipo I por familia (FWER) e poder do par com diferenca real

| cenario | FWER Tukey | FWER G-H | FWER Dunn | FWER recomendado | poder Tukey | poder G-H | poder Dunn | poder recomendado |
|---|---|---|---|---|---|---|---|---|
| normal, variancias iguais, n=10 | 0.050 | 0.048 | 0.038 | **0.051** | 0.80 | 0.77 | 0.72 | **0.79** |
| normal, variancias iguais, n=4 | 0.053 | 0.045 | 0.031 | **0.055** | 0.33 | 0.23 | 0.17 | **0.27** |
| normal, variancias iguais, n desigual 5/10/25 | 0.053 | 0.061 | 0.042 | **0.072** | 0.73 | 0.58 | 0.66 | **0.73** |
| normal, variancia 2:1, n=10 | 0.052 | 0.048 | 0.040 | **0.053** | 0.64 | 0.56 | 0.52 | **0.60** |
| normal, variancias 1:10, n=10 | 0.065 | 0.049 | 0.046 | **0.049** | 0.21 | 0.13 | 0.13 | **0.13** |
| normal, variancias 1:10, n desigual (grupo de maior variancia tem n=5) | 0.257 | 0.060 | 0.114 | **0.089** | 0.42 | 0.07 | 0.25 | **0.11** |
| t com 3 gl (caudas pesadas), n=15 | 0.040 | 0.035 | 0.035 | **0.046** | 0.94 | 0.93 | 0.99 | **0.96** |
| lognormal (assimetrica), n=15 | 0.034 | 0.026 | 0.046 | **0.048** | 0.92 | 0.91 | 1.00 | **1.00** |

Com 2000 repeticoes o erro padrao de uma taxa perto de 0,05 e' cerca de 0.005.
