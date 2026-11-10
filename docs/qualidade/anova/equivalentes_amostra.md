# Verificacao por amostra dos mutantes equivalentes

**Público:** quem confere o que foi verificado · **Estado:** gerado por script (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Sorteados 10 de 19 classificados como equivalentes (semente 0). Cada um foi aplicado ao modulo e a saida foi comparada com a do original em entradas aleatorias de borda (`scripts/verificar_equivalentes.py`).

| mutante | trecho | tentativas | saidas diferentes | veredito |
|---|---|---|---|---|
| `anova_grupos:651` const 1 | `or [np.zeros(1)])` | 300 | 0 | equivalente (nenhuma diferenca) |
| `anova_cli:264` Gt->GtE | `if c and not any(c < o or (c == o and k > idx) for idx, o in enumerate` | 300 | 0 | equivalente (nenhuma diferenca) |
| `vip_estabilidade:76` const 1 | `return yv.reshape(-1, 1), np.searchsorted(cortes, yv, side="right")` | 300 | 0 | equivalente (nenhuma diferenca) |
| `vip_estabilidade:85` Sub->Add | `return (Xtr - media) / dp` | 300 | 0 | equivalente (nenhuma diferenca) |
| `anova_grupos:201` const 1.0 | `"p_ajustado": float(min(1.0, p[idx]))})` | 300 | 0 | equivalente (nenhuma diferenca) |
| `vip_estabilidade:83` const 1 | `dp = Xtr.std(axis=0, ddof=1)` | 300 | 0 | equivalente (nenhuma diferenca) |
| `anova_cli:258` And<->Or | `if i in col and j in col:` | 300 | 0 | equivalente (nenhuma diferenca) |
| `anova_grupos:291` Gt->GtE | `F = (sq_f / gl_f) / qm_res if qm_res > 0 else float("inf")` | 300 | 0 | equivalente (nenhuma diferenca) |
| `anova_grupos:293` Gt->GtE | `Fb = (sq_b / gl_b) / qm_res if (qm_res > 0 and gl_b > 0) else float("n` | 300 | 0 | equivalente (nenhuma diferenca) |
| `anova_grupos:317` const 1.0 | `"p_ajustado": min(1.0, p_ij)})` | 300 | 0 | equivalente (nenhuma diferenca) |

Controle de poder (mutantes MORTOS nos mesmos geradores; ao menos um por modulo precisa mostrar diferenca; zero em um mutante morto significa que o gerador nao chega naquela linha, por exemplo uma recusa de entrada):

| mutante morto | tentativas | saidas diferentes |
|---|---|---|
| `anova_grupos:610` Sub->Add | 60 | 47 |
| `anova_grupos:542` Sub->Add | 60 | 28 |
| `vip_estabilidade:130` Lt->Gt | 60 | 0 |
| `vip_estabilidade:161` Gt->Lt | 60 | 60 |

Nenhuma diferenca e' evidencia, nao prova; a prova e' o argumento escrito de cada classificacao, abaixo.

Argumentos escritos de cada classificacao:

- `anova_grupos:651` const 1: equivalente matematico: np.zeros(1) x np.zeros(2): so' o caso de todos os grupos constantes, em que a forma e' constante nos dois
- `anova_cli:264` Gt->GtE: equivalente matematico: equivalente verificado por enumeracao: mesmas letras em 4660 casos (todas as matrizes de significancia, k = 2 a 5) e em 40 sorteios com k ate 7
- `vip_estabilidade:76` const 1: equivalente matematico: reshape(-2, 1): o numpy aceita qualquer dimensao negativa como 'inferir'
- `vip_estabilidade:85` Sub->Add: equivalente matematico: o PLSRegression centra X internamente
- `anova_grupos:201` const 1.0: equivalente matematico: min(1,p) com p <= 1 por construcao (a funcao de sobrevivencia nunca passa de 1)
- `vip_estabilidade:83` const 1: equivalente matematico: ddof 1 -> 2 multiplica todas as colunas pelo mesmo fator: o VIP nao muda
- `anova_cli:258` And<->Or: equivalente matematico: equivalente verificado por enumeracao: mesmas letras em 4660 casos (todas as matrizes de significancia, k = 2 a 5) e em 40 sorteios com k ate 7
- `anova_grupos:291` Gt->GtE: equivalente matematico: com QM residual exatamente 0 (dados perfeitamente aditivos, alcancavel: tests/test_anova_bloco_aditivo_exato.py) o quociente (SQ/gl)/0.0 em numpy e' inf, o mesmo valor do ramo else
- `anova_grupos:293` Gt->GtE: equivalente matematico: com gl do bloco == 0 o quociente 0/0 e' NaN, o mesmo valor do ramo else (e com QM residual 0 o ramo QM e' testado em tests/test_anova_bloco_aditivo_exato.py)
- `anova_grupos:317` const 1.0: equivalente matematico: min(1,p) com p <= 1 por construcao
