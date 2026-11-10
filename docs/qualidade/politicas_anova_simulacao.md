# Politicas de comparacao de grupos (ANOVA): simulacao com verdade conhecida

**Público:** quem confere o que foi verificado · **Estado:** gerado por script (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

600 repeticoes por cenario e por condicao; nivel 0.05; limite de erro por familia 0.06; erro padrao de Monte Carlo ~0.009 perto de 0,05. Efeito de 2.0 desvios no ultimo grupo para o poder.

## 1. Calibracao (grade A: 104 cenarios)

| politica | FWER maximo | FWER medio | cenarios acima de 0,06 (com margem de MC) | poder medio | pior cenario |
|---|---|---|---|---|---|
| tukey | 0.197 | 0.071 | 43 (30) | 0.786 | normal|var1:10|n=6|k=5|var_alta_n_pequeno |
| games-howell | 0.207 | 0.060 | 34 (18) | 0.683 | lognormal|var1:10|n=6|k=3|var_alta_n_pequeno |
| auto | 0.188 | 0.066 | 40 (26) | 0.765 | normal|var1:10|n=6|k=5|var_alta_n_pequeno |
| recomendar | 0.625 | 0.091 | 45 (31) | 0.709 | lognormal|var1:10|n=30|k=3|var_alta_n_pequeno |
| kruskal-dunn | 0.627 | 0.072 | 29 (21) | 0.611 | lognormal|var1:10|n=30|k=3|var_alta_n_pequeno |

**Politica escolhida na calibracao: `games-howell`** (**nenhuma politica cumpre o criterio**; esta e' a de menos violacoes).

## 2. Confirmacao independente (grade B: 136 cenarios, outros n, razoes, k, distribuicoes e sementes)

| politica | FWER maximo | FWER medio | cenarios acima de 0,06 (com margem de MC) | poder medio | pior cenario |
|---|---|---|---|---|---|
| tukey | 0.215 | 0.066 | 48 (35) | 0.783 | normal|var1:16|n=20|k=4|var_alta_n_pequeno |
| games-howell | 0.270 | 0.058 | 38 (25) | 0.688 | lognormal|var1:16|n=4|k=4|var_alta_n_pequeno |
| auto | 0.187 | 0.057 | 41 (25) | 0.753 | normal|var1:16|n=4|k=4|var_alta_n_pequeno |
| recomendar | 0.530 | 0.078 | 50 (33) | 0.700 | lognormal|var1:16|n=20|k=4|var_alta_n_pequeno |
| kruskal-dunn | 0.535 | 0.060 | 29 (20) | 0.573 | lognormal|var1:16|n=20|k=4|var_alta_n_pequeno |

Na confirmacao, `games-howell` tem FWER maximo 0.270, 38 cenario(s) acima de 0.06 (25 com a margem) e poder medio 0.688.

## 3. Onde cada politica falha (calibracao + confirmacao, FWER > 0,06)

- **tukey**: 65 de 240 cenarios acima do limite com margem; piores: normal|var1:16|n=20|k=4|var_alta_n_pequeno (0.215); normal|var1:16|n=8|k=4|var_alta_n_pequeno (0.202); normal|var1:16|n=4|k=4|var_alta_n_pequeno (0.200); normal|var1:10|n=6|k=5|var_alta_n_pequeno (0.197)
- **games-howell**: 43 de 240 cenarios acima do limite com margem; piores: lognormal|var1:16|n=4|k=4|var_alta_n_pequeno (0.270); lognormal|var1:10|n=6|k=3|var_alta_n_pequeno (0.207); lognormal|var1:16|n=8|k=4|var_alta_n_pequeno (0.202); lognormal|var1:6|n=4|k=4|var_alta_n_pequeno (0.182)
- **auto**: 51 de 240 cenarios acima do limite com margem; piores: normal|var1:10|n=6|k=5|var_alta_n_pequeno (0.188); normal|var1:10|n=6|k=10|var_alta_n_pequeno (0.187); normal|var1:16|n=4|k=4|var_alta_n_pequeno (0.187); lognormal|var1:16|n=4|k=4|var_alta_n_pequeno (0.187)
- **recomendar**: 64 de 240 cenarios acima do limite com margem; piores: lognormal|var1:10|n=30|k=3|var_alta_n_pequeno (0.625); lognormal|var1:10|n=30|k=3|igual (0.597); lognormal|var1:16|n=20|k=4|var_alta_n_pequeno (0.530); lognormal|var1:16|n=20|k=4|igual (0.523)
- **kruskal-dunn**: 41 de 240 cenarios acima do limite com margem; piores: lognormal|var1:10|n=30|k=3|var_alta_n_pequeno (0.627); lognormal|var1:10|n=30|k=3|igual (0.597); lognormal|var1:16|n=20|k=4|var_alta_n_pequeno (0.535); lognormal|var1:16|n=20|k=4|igual (0.525)
