# Auditoria metodológica — 2026-08-07

Escopo desta rodada: núcleo científico (`chemometric_stats.py`,
`classificadores.py`, `validacao_estatistica.py`, `preprocessamento.py`) e os
pontos de uso em produção (`pipeline.py`, `selecao_variaveis.py`,
`predicao.py`, `figuras.py`). **Não** cobre `avaliacao_modelos.py`,
`dados_io.py`, `dados_imagem.py`, `reports.py` — ver "Não auditado" no fim.

Método: ler a equação no código, comparar com a equação na publicação
original, e **medir** a divergência. Todo número abaixo saiu de um script em
`docs/auditoria/`, não de estimativa.

Estado do repositório na auditoria: v31.9.0, 663 testes passando (2 skip),
cobertura 67%, `ruff` limpo — os 9 itens da tabela ESTADO ALEGADO do
CLAUDE.md foram reverificados e batem.

---

## Resumo executivo

Cinco achados. Dois são da mesma classe do bug do DD-SIMCA corrigido em
2026-08-08 (implementação diverge do método publicado, sem que nenhum teste
percebesse, porque os testes verificam que a função *roda*, não que ela
*calcula o que diz calcular*).

| # | Achado | Gravidade | Onde | Impacto medido |
|---|---|---|---|---|
| A1 | Teste de permutação/Wold não é group-aware | **CRÍTICA** | `validacao_estatistica.py:365,497` | falso positivo 15,0% (nominal 5%) |
| A2 | Selectivity Ratio usa `w1`, não `b/‖b‖` | **CRÍTICA** | `chemometric_stats.py:50` | Jaccard@20 = 0,39 na seleção de variáveis |
| A3 | Domínio de aplicabilidade usa regra retangular | **ALTA** | `chemometric_stats.py:504-506` | rejeição 11,6% (nominal 5%) |
| A4 | OPLS-DA multiclasse usa alvo derivado de X (LDA) | MÉDIA | `classificadores.py:448-466` | não medido — método não publicado |
| A5 | Duas docstrings contradizem a referência citada | BAIXA | `chemometric_stats.py:173,197` | ≤1% no *n* deste projeto |

O achado A1 é o mais grave porque atinge exatamente o argumento central do
projeto: o Guaraci existe para fazer **validação group-aware**, e o teste que
produz o p-valor citável não é group-aware.

---

## A1 — Teste de permutação não respeita os grupos `mae_id` (CRÍTICA)

**O que o código faz.** `teste_permutacao` e `teste_wold` geram o nulo com
`rng.permutation(len(Y_bin))` (linhas 497 e 365) — permutação por **amostra**.
O `groups` é repassado intacto ao splitter. Resultado: depois de embaralhar,
um mesmo `mae_id` passa a conter réplicas com rótulos **diferentes**.

**Por que está errado.** Em dado agrupado a unidade de troca (exchangeable
unit) é o **grupo**, não a amostra. No dado real, todas as réplicas de um
`mae_id` compartilham um rótulo por construção física — é uma restrição do
delineamento. A permutação por amostra gera conjuntos nulos que **não podem
existir** sob H0 e que são estritamente mais difíceis de classificar. O nulo
fica artificialmente estreito e o valor observado cai na cauda com frequência
demais.

**Medido** (`docs/auditoria/medir_permutacao_grupos.py`, H0 verdadeiro, 12
grupos × 3 réplicas, 3 classes, 120 repetições × 100 permutações):

| Esquema | desvio-padrão do nulo | falso positivo (p<0,05) |
|---|---|---|
| por amostra (implementado) | 0,0864 | **0,150** |
| por grupo (correto) | 0,1532 | 0,042 |

O nulo implementado é ~1,8× estreito demais. **Um p-valor reportado como
<0,05 tem taxa de erro real de ~15% neste regime** — 3× o declarado.

**Correção.** Permutar rótulos entre grupos, preservando a coerência interna:

```python
gid_unicos, inv = np.unique(groups, return_inverse=True)
rot_por_grupo = np.array([y_int[groups == g][0] for g in gid_unicos])
y_perm = rot_por_grupo[rng.permutation(len(gid_unicos))][inv]
```

Vale para os dois testes. Requer que todo grupo tenha rótulo único — verificar
e falhar explicitamente se não tiver.

**Teste que precisa existir:** um que rode H0 verdadeiro com estrutura de
grupo e **falhe** se a taxa de falso positivo sair de [0,02; 0,10].

Referência para permutação restrita em dado agrupado: Winkler A.M. et al.
(2015), *Multi-level block permutation*, NeuroImage 123:253-268.

---

## A2 — Selectivity Ratio implementa a projeção-alvo errada (CRÍTICA)

**O que a literatura define.** Em Rajalahti et al. (2009) e Kvalheim (2020,
*J. Chemometrics* 34:e3211), a projeção-alvo usa o **vetor de regressão
normalizado** `b/‖b‖` como alvo. A propriedade que define o método: o escore
projetado `t_TP` é **proporcional ao vetor de valores preditos ŷ**.

**O que o código faz.** `chemometric_stats.py:50` usa `w1 = W[:, 0]` — o
primeiro peso PLS. Só coincide com `b/‖b‖` quando o modelo tem 1 LV.

**Medido** (`medir_sr_ranking.py`, cenário multi-interferente, 300 variáveis):

| LVs | Jaccard@20 | ρ Spearman | max SR (ref) | max SR (impl) |
|---|---|---|---|---|
| 1 | 1,000 | 1,0000 | 45,8 | 45,8 |
| 2 | **0,394** | 0,708 | 76,1 | 45,8 |
| 3 | **0,386** | 0,774 | 164,2 | 45,8 |
| 8 | **0,386** | 0,756 | 118,8 | 45,8 |

Dois fatos decisivos:

1. `corr(t_TP, ŷ)` = **1,000000 exato** na referência; **0,92** na
   implementação (LV≥2). A propriedade que *define* o método não vale.
2. O SR implementado é **idêntico para 2, 3, 4, 6 e 8 LVs** (45,8 sempre) —
   está congelado na resposta de 1 LV, insensível ao modelo que diz descrever.

**Impacto em produção:** `selecao_variaveis.py:97` (`_mask_sr_top_frac`) ordena
por SR e corta o top-N. Com Jaccard@20 = 0,39, **~60% das variáveis
selecionadas diferem** das que o método publicado selecionaria. Também entra
no relatório oficial via `pipeline.py:1642`.

**Correção:** trocar o alvo por `b/‖b‖` (ver `sr_referencia()` em
`medir_sr_ranking.py`, já escrito e validado).

**Teste que precisa existir:** `assert corr(t_TP, ŷ) == 1` (a menos de
tolerância numérica) para qualquer nº de LVs. Falha com o código atual.

---

## A3 — Domínio de aplicabilidade repete a regra retangular do DD-SIMCA (ALTA)

`dominio_aplicabilidade_amostras_novas` (linhas 504-506) decide
`dentro = (T² ≤ lim) & (Q ≤ lim)`, com α=0,05 independente em cada eixo —
**exatamente a regra que foi removida do `DDSimca.predict()` em 2026-08-08**,
ainda viva aqui.

**Medido** (`medir_achados.py`, 40 simulações, amostras novas da **mesma**
distribuição do treino): rejeição **11,6%** (sd 0,021) contra 5% nominal.
Acima até do 9,75% ingênuo de `1-(1-α)²`, porque os dois limites são eles
mesmos estimados.

**Onde importa:** `predicao.py:264` — o caminho de produção que decide se uma
amostra nova está dentro do domínio do modelo. Uma em cada nove amostras
legítimas é marcada como fora do domínio.

**Correção:** aplicar a mesma distância combinada já usada no DD-SIMCA
(`DDSimca._f_distance` + `chi2.ppf(1-α, Nh+Nq)`), reaproveitando a função que
já existe em vez de uma terceira implementação da regra de decisão.

---

## A4 — OPLS-DA multiclasse constrói o alvo a partir de X (MÉDIA)

`classificadores.py:448-466`: para Y multiclasse, o código ajusta uma
`LinearDiscriminantAnalysis` em `(X, y)` e usa o **primeiro escore discriminante
como o `y` contínuo** do OPLS-DA.

Trygg & Wold (2002) definem OPLS para `y` binário/contínuo; a extensão
multiclasse publicada é O2PLS/OPLS com Y multi-coluna. Usar um alvo derivado
de X **não é método publicado** — o comentário no código explica a motivação
(evitar viés de "primeira classe vs resto"), o que é legítimo como
raciocínio, mas o resultado é uma variante própria, não OPLS-DA.

Dois riscos concretos, nenhum medido nesta rodada:
- o componente "preditivo" fica parcialmente auto-referencial (alvo é função de X);
- com p ≫ n a LDA é mal-condicionada; o `except` cai para PLS2 e **muda o eixo
  do S-Plot silenciosamente** (só um `log.warning`).

**Ação:** ou documentar explicitamente como variante do Guaraci (com essa
palavra) em `VALIDATION.md` e no MANUAL, ou trocar por PLS2 multi-coluna, que é
o caminho publicado. Não deixar como está sem rótulo.

---

## A5 — Duas docstrings contradizem a referência que citam (BAIXA)

**`hotelling_t2_limite` (linha 173).** Cita Tracy-Young-Mason (1992) e afirma
ser "valid for both observations within the calibration set and new
observations". TYM 1992 é precisamente o artigo que estabelece que os dois
casos são **diferentes**: Fase I (amostras do próprio treino) usa
distribuição **Beta**; Fase II (amostras novas) usa **F**. O código implementa
só a de Fase II e a aplica também em contexto de Fase I
(`dominio_aplicabilidade_treino:471`, `figuras.py:533`).

Medido — razão limite-F / limite-Beta:

| n | k=2 | k=3 |
|---|---|---|
| 10 | 2,37× | 3,23× |
| 20 | 1,47× | 1,65× |
| 30 | 1,28× | 1,38× |
| 300 | 1,02× | 1,03× |

**Impacto real neste projeto: baixo.** Onde a função é usada em Fase I, n é da
ordem de centenas (razão ~1,01×). E o caminho `ucl_method="theoretical"`, onde
n=3-4 tornaria o erro 2-3×, **não é o default** (`config.py:204` =
`"empirical"`) e desde 2026-08-08 o `T2_UCL` só alimenta a linha de
diagnóstico, não a decisão. Corrigir a docstring é obrigatório; implementar o
limite Beta é opcional.

**`q_residuos_limite` (linha 197).** Implementa `g·χ²(h)` por casamento de
momentos — que é a aproximação de **Box (1954)** / Nomikos & MacGregor (1995),
não Jackson & Mudholkar (1979), cuja fórmula é outra (baseada em θ₁,θ₂,θ₃,h₀ e
na normal). As duas são legítimas; a atribuição está trocada, em 3 lugares
(`chemometric_stats.py:458,474`, `classificadores.py:31,99`). Corrigir a
citação — o CLAUDE.md exige que toda referência seja verificável.

---

## Verificado e correto

Auditado contra a publicação original e **sem divergência encontrada**:

- **VIP** (`vip_scores`) — Chong & Jun (2005). Fórmula correta, incluindo a
  generalização multi-Y por `‖q_a‖²`. Propriedade `Σ VIP² = p` verificada
  algebricamente.
- **DD-SIMCA distância combinada** — Kucheryavskiy et al. (2024) Eq. 3-4,
  reproduzida corretamente; `_media_e_dof` implementa `N = 2(média/desvio)²`
  corretamente.
- **DD-SIMCA `Q_train` por LOO** — a correção de 2026-07-19 é sólida; resolve o
  viés in-sample na raiz.
- **OPLS-DA (caminho binário)** — Trygg & Wold (2002) Alg. 1, incluindo o
  Gram-Schmidt explícito em `t_orth`, que muitas implementações omitem.
- **SNV / MSC** — Barnes et al. (1989) e Geladi et al. (1985). MSC
  corretamente *stateful* (referência = média do treino, dentro do Pipeline).
- **BCa** — Efron (1987): z₀, aceleração por jackknife e o mapeamento dos
  percentis estão corretos.
- **`StratifiedGroupKFoldEstavel`** — resolve um problema real (partição do
  sklearn instável entre versões) de forma defensável e determinística.
- **DModX / DModY** — normalização e graus de liberdade batem com Eriksson et
  al. (2006).

---

## Dívida de engenharia observada (não medida)

1. **Inconsistência LOO vs in-sample no DD-SIMCA.** `fit()` guarda `Q_train`
   por LOO, mas `score_matrix()` recalcula Q in-sample via `_t2_q`. Se a figura
   de aceitação plota pontos de treino via `score_matrix` contra um `f_crit`
   derivado do `q0` LOO, treino e limite estão em escalas diferentes.
   **Achado por leitura de código, não medido** — verificar em `figuras.py`.
2. **230 `print()` fora de `pipeline.py`**, incluindo 2 em
   `chemometric_stats.py` (módulo de cálculo puro) e 8 em
   `validacao_estatistica.py`. O CLAUDE.md (P6) afirma "os demais módulos já
   usavam `logging` desde antes" — **isso é falso**; corrigir a afirmação.
3. **`MSC.transform` faz um `lstsq` por amostra em laço Python.** Com 934×8192
   é desperdício; a regressão de 2 parâmetros tem forma fechada vetorizável.
   Correto, só lento.
4. **`spectra_preview.py` em 0% de cobertura.**

---

## Plano de correção sugerido

| Ordem | Item | Por quê nesta posição |
|---|---|---|
| 1 | A1 (permutação por grupo) | Único que invalida um número já citável; atinge o argumento central |
| 2 | A2 (SR com `b/‖b‖`) | Muda variáveis selecionadas → muda o modelo final |
| 3 | A3 (AD com distância combinada) | Mesmo bug já corrigido em outro lugar; correção é reúso |
| 4 | A5 (docstrings) | Barato, e o CLAUDE.md exige referência verificável |
| 5 | A4 (rotular OPLS-DA multiclasse) | Decisão de projeto, não bug |

**Regra que fica, análoga à do P10 (figuras):** teste de método científico tem
que verificar a **propriedade que define o método** (`t_TP ∝ ŷ`, taxa de falso
positivo calibrada, α efetivo = α nominal), nunca só que a função devolve um
array do shape certo. Os três achados críticos passaram por 663 testes.

---

## Não auditado nesta rodada

`avaliacao_modelos.py` (Monte Carlo CV, SHAP, DET), `dados_io.py`
(agrupamento `mae_id`, JCAMP-DX), `selecao_variaveis.py` (iPLS, SPA, GA-PLS,
sPLS-DA — só o consumo de SR foi visto), `dados_imagem.py`, `reports.py`.
Nenhuma afirmação deste documento cobre esses módulos.
