# Mutação nas linhas alteradas desde a última verificação completa

**Público:** quem avalia os números · **Estado:** gerado por `scripts/gerar_mutacao_linhas_alteradas.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Teste de mutação (`scripts/mutacao_modulo.py`): o programa troca um operador do código por outro e roda os testes; um teste que falha *mata* o mutante, e o que sobrevive mostra um comportamento que nenhum teste confere. Aqui a mutação foi aplicada só às **linhas alteradas** desde a última verificação completa, módulo por módulo, com os testes de cada módulo. O ANOVA, a CLI do ANOVA e o VIP têm a rodada completa em `docs/qualidade/anova_blindagem.md`.

## 1. Resultado por faixa de linhas

| módulo | linhas | mutantes | mortos | sobreviventes | sobreviventes classificados |
|---|---|---|---|---|---|
| `anova_grupos.py` | 63-64 | 1 | 1 | 0 | 0 |
| `anova_grupos.py` | 426-429 | 1 | 1 | 0 | 0 |
| `avaliacao_modelos.py` | 273-276 | 1 | 1 | 0 | 0 |
| `avaliacao_modelos.py` | 783-785 | 1 | 1 | 0 | 0 |
| `chemometric_stats.py` | 1090-1091 | 4 | 4 | 0 | 0 |
| `classificadores.py` | 665-667 | 3 | 3 | 0 | 0 |
| `config_io.py` | 598-600 | 1 | 1 | 0 | 0 |
| `dados_io.py` | 315-323 | 7 | 7 | 0 | 0 |
| `dados_io.py` | 1181-1192 | 5 | 5 | 0 | 0 |
| `dados_io.py` | 1214-1216 | 2 | 2 | 0 | 0 |
| `dados_io.py` | 1225 | 2 | 2 | 0 | 0 |
| `docs_pacote.py` | 1-58 | 22 | 22 | 0 | 0 |
| `pipeline.py` | 1740-1741 | 1 | 1 | 0 | 0 |
| `pipeline.py` | 2294-2378 | 37 | 36 | 1 | 1 |
| `plano_amostral.py` | 126-140 | 9 | 9 | 0 | 0 |
| `selecao_variaveis.py` | 508 | 2 | 2 | 0 | 0 |
| `selecao_variaveis.py` | 1225-1228 | 1 | 1 | 0 | 0 |

**99 de 100 mutantes mortos; 1 sobrevivente(s); 0 sem classificação.**

## 2. Sobreviventes e a classificação de cada um

| módulo | linha | operador | trecho | classificação |
|---|---|---|---|---|
| `pipeline.py` | 2321 | And<->Or | `assert bca_holdout is not None and conformal_classificacao is not None` | equivalente: e' um `assert` de invariante que so' estreita o tipo Optional; as duas variaveis nascem no mesmo bloco que `metricas_holdout` (qualquer falha ali o zera e o `if` acima nao entra), entao nunca divergem |

## 3. Faixas alteradas sem operador mutável

Comentários, importações ou texto: não há o que mutar.

| módulo | linhas |
|---|---|
| `predicao.py` | 379 |
| `linearity.py` | 178 |
| `vip_cli.py` | 214 |
| `config_io.py` | 263-265 |
| `selecao_variaveis.py` | 599 |
| `dados_io.py` | 1613 |
| `pipeline.py` | 1113 |
| `pipeline.py` | 2174 |

## 4. Não executado nesta rodada

Fica registrado como **não executado** (não como verde); entra na v1.1.

| módulo | linhas | situação |
|---|---|---|
| `hsi_extras.py` | 106-107 | mutação não executada (escopo desta rodada: núcleo estatístico e leitores) |
| `auditoria_delineamento.py` | 376-378 | mutação não executada (escopo desta rodada) |
