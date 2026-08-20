# FASE B — Módulos nunca auditados (2026-08-16)

Somente leitura. Nenhum código de produção alterado.

**Sobre a premissa da fase.** O documento de auditoria pedia para tratar
como suspeito um resultado vazio, já que as duas rodadas anteriores acharam
defeito em todo módulo que examinaram. Não foi o caso: **os quatro módulos
têm achado.** A previsão se manteve.

Também vale registrar o inverso, para não inflar o placar: **três das
suspeitas levantadas pelo próprio documento não se confirmaram.** Estão
listadas no fim como "verificado e correto".

| # | Achado | Gravidade | Arquivo:linha | Impacto | Custo | Status |
|---|---|---|---|---|---|---|
| B3-1 | Template LaTeX afirma metodologia que pode não ter sido usada | **Alta** | reports.py:714,722,728 | manuscrito gerado com método falso | 2 h | ✅ resolvido 2026-08-16 |
| B1-1 | bal.acc do iPLS é máximo-de-N na partição que reporta | **Alta** | selecao_variaveis.py:260,567 | **+0,070** bal.acc medido | 3 h | ✅ resolvido 2026-08-16 (opção **a**: nested-CV) |
| B4-1 | Modo imagem sem gating + validação group-aware silenciosamente off | **Alta** | dados_imagem.py:227 | saída de protótipo indistinguível de validada | 3 h | ✅ resolvido 2026-08-16 (relatórios + figuras carimbados) |
| ~~B2-1~~ | ~~Monte Carlo CV descarta iterações difíceis~~ | ❌ **retratado** | — | **descarte medido: 0,0%** | — | achado inválido, ver seção |
| B1-3 | CV interna de AG/SPA não é group-aware | Média | selecao_variaveis.py:178 | variáveis escolhidas por critério com vazamento | 2 h | ✅ resolvido 2026-08-16 |
| B1-2 | sPLS-DA faz truncamento duro, não soft-thresholding | Baixa | selecao_variaveis.py:266 | Jaccard 0,87 com n_comp=5 | 1 h | ✅ resolvido 2026-08-16 (docstring **e** implementação) |
| B3-2 | `%` dentro de `\caption{}` comenta a chave de fechamento | Baixa | reports.py:646 | LaTeX gerado não compila | 5 min | ✅ resolvido 2026-08-16 |

> **B3-1 resolvido:** `gerar_latex_template` agora lê `Group-aware
> (mae_id)`/`Validacao` do resumo real e condiciona o texto — afirma
> GroupKFold só quando de fato usado. Faixa espectral e nº de permutações
> também deixaram de ser constantes cravadas. Auditados os 6
> `resumo_modelo.txt` já usados no material do TCC: todos "sim" — nenhuma
> figura/tabela já citada veio de execução em fallback, o defeito era só
> no texto gerado. Teste de regressão em `test_reports.py` (compara o
> LaTeX gerado com resumo "sim" vs "nao", confirma que o TEXTO muda).
> **B4-1 herda parcialmente**: o texto do relatório agora reflete o
> fallback corretamente quando ele ocorre, mas o gating de execução
> (bloquear/carimbar modo imagem) proposto no achado original não foi
> implementado nesta sessão — ver docs/CHANGELOG.md.
>
> **B1-2 (completo):** além da docstring, a **implementação** passou a usar
> o soft-thresholding da referência,
> `w_j ← sign(w_j)·max(|w_j|−λ, 0)` com λ = o (keep+1)-ésimo maior |w| —
> o que preserva a parametrização por contagem (mesma ideia do `keepX` do
> mixOmics) e ao mesmo tempo encolhe as sobreviventes. É o encolhimento que
> muda a direção normalizada de w, logo o escore t, logo a deflação, logo o
> conjunto escolhido pelos componentes seguintes.
>
> **B4-1 (completo):** os relatórios já saíam carimbados; faltava a figura
> solta. `salvar()` — ponto único por onde toda figura passa — agora aplica
> marca d'água "PROTOTIPO / NAO VALIDADO" no modo imagem, cobrindo as ~30
> figuras de uma vez. Era o arquivo que acaba colado num slide sem
> contexto nenhum.

---

## B3-1 — O template LaTeX afirma metodologia que pode não ter sido usada

**É o achado mais grave da fase**, porque produz um documento destinado a
virar manuscrito.

[reports.py:714–731](../../src/guaraci/reports.py#L714) mistura valores
interpolados do run real com **constantes cravadas no template**:

```latex
FT-NIR spectra were acquired in the range \SIrange{4000}{10000}{...}.   % cravado
The PLS-DA model was calibrated with {met['lvs']} latent variables,     % real
selected by group-aware cross-validation (GroupKFold, grouping           % CRAVADO
technical replicates to prevent data leakage)
\item \textbf{Y-randomization} (200 permutations)                        % cravado
\item \textbf{Wold's test} (intercepts $R^2Y < 0.40$ and $Q^2Y < 0.05$)   % cravado
```

`met['preproc']` e `met['lvs']` vêm do run. **A afirmação de GroupKFold,
a faixa espectral, o número de permutações e os limiares de Wold, não.**

O pipeline tem um caminho explícito de fallback
([pipeline.py:1184](../../src/guaraci/pipeline.py#L1184)):

```
[INFO] GroupKFold desabilitado: mae_id indisponivel ou grupos insuficientes
       — usando StratifiedKFold (estratificada).
```

Quando esse caminho é tomado — modo imagem (B4-1), CSV sem grupo, ou
`mae_id` ausente — **o LaTeX gerado continua afirmando que houve validação
group-aware.** É a única afirmação do projeto que não pode sair errada:
validação group-aware é o argumento central, o statement of need do JOSS, e
a linha que separa este software dos concorrentes.

Atenuante real, registrada: o `resumo_modelo` grava o rótulo verdadeiro da
CV ([pipeline.py:1957](../../src/guaraci/pipeline.py#L1957), campo
`"Validacao": cv_label`). Ou seja, **a informação correta existe e está a
uma variável de distância** do template que a contradiz. Correção: trocar
as constantes por interpolação do que o run efetivamente fez, e emitir a
frase de group-aware condicionada ao `cv_label`.

## B3-2 — `%` dentro de `\caption{}` quebra o LaTeX gerado

[reports.py:646](../../src/guaraci/reports.py#L646) emite:

```latex
\caption{Nome da figura. % TODO: add chemical interpretation.}
```

Em LaTeX `%` comenta até o fim da **linha** — incluindo a chave `}` que
fecha o `\caption`. O `\label` da linha seguinte é absorvido e o documento
não compila (ou compila com a legenda engolindo o resto do bloco).

Os outros 5 TODO do módulo (linhas 695, 702, 712, 758, 765) estão em
início de linha no corpo do template: são comentários LaTeX legítimos,
deixados de propósito para o autor preencher. **Só o da linha 646 é
defeito.** Correção: mover o TODO para a linha anterior.

Isso responde a pergunta do B3 ("algum TODO emite string literal em
documento gerado?"): sim, um.

## B3-3 — Arredondamento (verificado, sem achado)

Os números do CSV de Monte Carlo são gravados com `round(..., 4)`
([avaliacao_modelos.py:437–445](../../src/guaraci/avaliacao_modelos.py#L437)),
consistente com a precisão que centenas de iterações sustentam. Não
encontrei relatório reportando mais dígitos do que a medição sustenta.

---

## B1-1 — O bal.acc do iPLS é o máximo de N avaliações feitas na partição que o reporta

[selecao_ipls:260](../../src/guaraci/selecao_variaveis.py#L260) escolhe o
melhor intervalo por `balanced_accuracy` calculado sobre `cv_indices`;
[etapa4:567](../../src/guaraci/selecao_variaveis.py#L567) reavalia o
vencedor **na mesma `cv_indices`** e põe o resultado na tabela final.

O comentário do módulo justifica deixar o iPLS fora da correção de
nested-CV porque "a partição em intervalos NÃO usa rótulo". Isso é
verdade e é irrelevante: **a escolha do melhor intervalo usa rótulo**, via
a mesma CV que depois reporta o número. É o mesmo double dipping que o
próprio bloco corrigiu para VIP/SR/sPLS-DA, com uma justificativa que não
endereça o mecanismo real.

### Medição

Script: [`medir_selecao_variaveis.py`](medir_selecao_variaveis.py).
Dados sintéticos com estrutura de réplica (10 grupos × 3 réplicas × 4
classes), `GroupKFold(5)`, 10 intervalos, 12 seeds. Compara o iPLS como
está contra o iPLS com a escolha do intervalo refeita dentro de cada fold.

```
 seed  iPLS como esta  iPLS aninhado     viés
----------------------------------------------
    0          0.6750         0.5750   0.1000
    3          0.5667         0.4583   0.1083
    8          0.6000         0.4500   0.1500
   ...
  viés medio  : +0.0701 pontos de bal.acc
  viés mediano: +0.0667
  positivo em : 12/12 seeds
```

**+7,0 pontos de balanced accuracy, positivo em 12 de 12 seeds.**

**A consequência pior não é o número do iPLS em si, é a tabela.** A Etapa 4
compara Full / iPLS / VIP / SR / sPLS-DA / SPA / AG lado a lado, e
[etapa4:668–671](../../src/guaraci/selecao_variaveis.py#L668) escolhe
automaticamente "o mais parcimonioso dentro de 1% do máximo". Todos os
outros métodos passaram por nested-CV; o iPLS não. **A comparação é entre
números computados sob regras diferentes, com uma diferença medida sete
vezes maior que o limiar de 1% usado para decidir o vencedor.** O iPLS é
sistematicamente favorecido pela regra de escolha automática.

> **✅ RESOLVIDO em 2026-08-16 — opção (a), nested-CV completo.** Entre as
> duas opções levantadas (encaixar o iPLS no mesmo nested-CV dos outros
> métodos, ou marcá-lo na tabela como avaliação não-aninhada), foi feita a
> primeira: a infraestrutura já existia (`_avaliar_subset_nested_cv`) e o
> seletor de intervalo por fold já havia sido escrito e validado na própria
> medição, então o custo marginal do conserto correto era baixo — e ele
> elimina o problema em vez de anotá-lo. `_mask_melhor_intervalo` reescolhe
> o melhor intervalo dentro de cada fold, usando só dados de treino. A
> busca no dataset inteiro continua rodando 1× para a figura/CSV de
> diagnóstico por intervalo (mesmo padrão de SPA/AG). Teste em
> `test_pipeline_core.py`.

## B1-2 — sPLS-DA: truncamento duro onde a referência define soft-thresholding

[sparse_plsda_mask:266](../../src/guaraci/selecao_variaveis.py#L266) tem
docstring "estilo Le Cao et al. 2008 — NIPALS com **soft-selection**", e
implementa `idx = argsort(|w|)[::-1][:keep]` — truncamento **duro**: mantém
as top-k e zera o resto, **sem encolher as sobreviventes**.

Lê Cao et al. (2008) definem sPLS por penalização com **soft-thresholding**
nos vetores de loading: `w_j ← sign(w_j)·(|w_j| − λ)₊` — as sobreviventes
são encolhidas por λ. Verificado por busca, não de memória (ver Fontes).

Duas coisas erradas, portanto: o método diverge da referência citada, e a
docstring chama de "soft-selection" justamente a variante dura. Esta
segunda parte é a mesma classe do achado A5 da auditoria anterior
(docstring contradizendo a referência que ela cita).

### Medição

Com 1 componente os dois dão o **mesmo conjunto** (top-k por |w| é
exatamente o resultado do soft-threshold com λ = (k+1)-ésimo maior |w|).
A partir de 2 componentes as direções normalizadas diferem, a deflação
diverge, e a seleção muda:

```
 n_comp   Jaccard mediano   |dif| mediana de vars
      1             1.000                       0
      2             0.926                       2
      3             0.944                       2
      4             0.880                       6
      5             0.867                       8
```

**Divergência real mas modesta** — bem menor que a do Selectivity Ratio
(Jaccard 0,39 na auditoria anterior). Gravidade baixa; o que justifica
corrigir é a docstring, não o impacto numérico.

## B1-3 — A CV interna dos métodos de busca não é group-aware

[`_cv_local`:178](../../src/guaraci/selecao_variaveis.py#L178) usa
`StratifiedKFold` — réplicas do mesmo `mae_id` caem em treino e validação
da CV **interna** que guia a fitness do AG e a pontuação do SPA. O código
declara isso ("não é group-aware — mae_id não chega até aqui") e argumenta
que "só orienta a otimização; o número científico reportado usa sempre o
fold externo".

**O argumento está metade certo, e a metade que falta importa.** O número
reportado é de fato honesto — o fold externo é group-aware e a busca nunca
o vê. Mas o **produto científico da Etapa 4 não é o bal.acc: é o conjunto
de variáveis selecionadas**, e esse é escolhido por um critério com
vazamento de réplica. Uma busca guiada por uma partição onde réplicas
vazam prefere variáveis que exploram a similaridade entre réplicas — que
é exatamente o artefato que o projeto existe para combater. O software
aplica a si mesmo, no passo de seleção, o erro que denuncia.

Correção: propagar `mae_id` até `etapa4_selecao_variaveis` e usar
`StratifiedGroupKFoldEstavel` (já existe no projeto) em `_cv_local`. Não
medido nesta rodada — a medição exigiria comparar conjuntos selecionados
com e sem grupos na CV interna, o que é o mesmo formato do B1-2 e cabe na
correção.

> **✅ RESOLVIDO em 2026-08-16, exatamente como proposto acima.**
> `_cv_local` aceita `grupos_local` e usa `StratifiedGroupKFoldEstavel`
> quando há ≥2 grupos; `mae_id` propagado de `executar()` →
> `etapa4_selecao_variaveis` → `_avaliar_busca_nested_cv` → `_cv_local`.
> Sem `mae_id`, o comportamento anterior é preservado (necessário para
> modos sem identificador de réplica). Dois testes em
> `test_pipeline_core.py`: um garante que nenhum grupo aparece nos dois
> lados de um fold interno; o outro garante que o caminho sem grupos não
> mudou. **Continua não medido** o impacto no conjunto de variáveis
> selecionadas — a correção foi aplicada por argumento metodológico, não
> por medição de magnitude.

## B1-4 — Determinismo do AG (verificado, sem achado)

`selecao_ag` usa `np.random.default_rng(seed)`
([linha 410](../../src/guaraci/selecao_variaveis.py#L410)) e `_cv_local`
recebe o mesmo `cfg.seed` em todo fold externo. **Dado o seed, o resultado
é determinístico.** Responde a terceira pergunta do B1.

---

## B2-1 — ❌ RETRATADO: o viés que eu afirmei não existe

**Este achado não se sustenta. Medido em 2026-08-16 e retirado.**

O que eu havia afirmado: [avaliacao_modelos.py:406–409](../../src/guaraci/avaliacao_modelos.py#L406)
descarta iterações em que o treino não contém todas as classes; essas
seriam as partições mais difíceis, então o `IC95%` sairia otimista.

```python
if len(np.unique(y_tr)) < len(lb.classes_):
    continue
if len(np.unique(y_te)) < 2:
    continue
```

**O raciocínio ignorou o splitter.** `_stratified_group_shuffle_splits`
estratifica **no nível de grupo** (`StratifiedShuffleSplit` sobre os
grupos, com a classe majoritária de cada um) — ou seja, ele já **garante**
representação proporcional de toda classe em treino e teste. A condição que
o guard testa praticamente não pode ocorrer quando o splitter roda.

Medição ([`medir_monte_carlo_descarte.py`](medir_monte_carlo_descarte.py)),
200 iterações por célula, dados com classes deliberadamente pouco separadas
para que a BA não saturasse:

```
 classes  grupos/cl  descarte  BA sobrev.  BA descart.    IC95 sobrev.      IC95 todas
      14         44      0.0%      0.9998            -   [0.998,1.000]   [0.998,1.000]
      14         20      0.0%      0.9999            -   [1.000,1.000]   [1.000,1.000]
      14         10      0.0%      0.9975            -   [0.976,1.000]   [0.976,1.000]
      14          6      0.0%      0.9346            -   [0.845,1.000]   [0.845,1.000]
      14          4      0.0%      0.8589            -   [0.714,1.000]   [0.714,1.000]
```

**Descarte de 0,0% em todos os regimes viáveis**, inclusive o regime do
dataset de desenvolvimento — a primeira linha da tabela, a de maior
número de grupos por classe. A BA varia de 0,86 a 1,00 entre as células, o que confirma
que o 0% não é artefato de um problema fácil demais. O guard é
**defensivo**, não uma fonte de viés: nada é descartado na prática.

**Fica a lição de método, que vale registrar:** eu classifiquei como achado
"Média" algo que a própria auditoria manda medir antes de afirmar (regra 3
do ciclo). O erro foi ler o `continue` isoladamente, sem confrontá-lo com a
garantia dada pelo splitter 90 linhas acima.

### B2-1b — o que a medição achou no lugar (menor, mas real)

`_stratified_group_shuffle_splits` **levanta `ValueError`** quando o número
de grupos de teste fica abaixo do número de classes (exigência do
`StratifiedShuffleSplit`):

```
14 classes, 3 grupos/classe -> "The test_size = 11 should be greater or
                                equal to the number of classes = 14"
```

A chamada em [monte_carlo_cv:382](../../src/guaraci/avaliacao_modelos.py#L382)
**não** está protegida, então em um dataset com poucos grupos por classe o
Monte Carlo CV inteiro morre e só sobra a mensagem genérica do `except`
amplo do pipeline. **Não afeta este dataset** (~44 grupos/classe, folga de
mais de 10×), e é um modo de falha *ruidoso* — o oposto do descarte
silencioso que eu havia alegado. Correção sugerida (não aplicada): detectar
a condição antes e emitir aviso explícito de "poucos grupos para Monte
Carlo CV com N classes" em vez de deixar estourar.

## B2-2, B2-3, B2-4 — Verificado e correto

As três outras perguntas do B2 não viraram achado:

- **Monte Carlo CV respeita `mae_id`?** Sim.
  [`_stratified_group_shuffle_splits`](../../src/guaraci/avaliacao_modelos.py#L319)
  estratifica no nível de grupo (classe majoritária por grupo) e monta os
  índices por `np.isin(grupos_cv, ...)` — réplicas nunca se separam. O
  benchmark usa `StratifiedGroupKFold`. **Não é o mesmo bug do A1.**
- **SHAP é válido para o modelo usado?** Sim, e a preocupação do documento
  ("SHAP sobre PLS") não se aplica: o `TreeExplainer` é usado apenas em
  RF/GBM/XGBoost ([linha 654](../../src/guaraci/avaliacao_modelos.py#L654)),
  que são exatamente os modelos para os quais ele é exato. O PLS-DA não é
  explicado por SHAP. Nota menor, não achado: o classificador é ajustado em
  100% dos dados e explicado num subconjunto dos mesmos dados — atribuição
  in-sample, que é a prática padrão para SHAP (explica o modelo, não
  generalização), mas a figura não diz isso.
- **A correção do `interpolar_det` (P10) cobre este módulo?** Sim — a
  função vive **neste** módulo
  ([linha 467](../../src/guaraci/avaliacao_modelos.py#L467)) e é usada em
  [537](../../src/guaraci/avaliacao_modelos.py#L537). Não há caminho DET
  paralelo sem a correção.

---

## B4-1 — O modo imagem não tem gating, e desliga a validação group-aware em silêncio

O documento pedia: "confirmar que o gating impede saída rotulada como
publicável; se não existir gating, é achado". **Não existe gating.**

A palavra "protótipo" aparece em docstrings
([dados_imagem.py:19,24,28,227](../../src/guaraci/dados_imagem.py#L19)),
em comentários de config, e no texto de ajuda do CLI
([cli_assistente.py:632](../../src/guaraci/cli_assistente.py#L632)).
**Em nenhum lugar do caminho de execução, de figura ou de relatório.** Um
PDF, Word ou LaTeX gerado em modo imagem é tipograficamente idêntico a um
gerado a partir de FT-NIR validado.

E há um agravante que compõe com o B3-1:
[dados_imagem.py:227](../../src/guaraci/dados_imagem.py#L227) — `conc` e
`mae_id` são **sempre `None`** neste modo. Com `mae_id is None`, o
pipeline cai no fallback da
[linha 1184](../../src/guaraci/pipeline.py#L1184) e usa `StratifiedKFold`.
Ou seja, em modo imagem:

1. não há validação group-aware — o diferencial central do software está
   desligado;
2. o único aviso é uma linha `[INFO]` no console, que ninguém relê;
3. **e o template LaTeX afirma, cravado, que houve GroupKFold** (B3-1).

Os três juntos produzem um documento que declara ter usado a metodologia
que o projeto vende, num modo em que essa metodologia está desligada por
construção. Correção mínima: bloquear a geração de relatório em modo
imagem, ou carimbar "PROTÓTIPO — NÃO VALIDADO" em toda figura e cabeçalho
de relatório, e condicionar a frase de group-aware ao `cv_label` real.

> **✅ RESOLVIDO em 2026-08-16** (as três camadas):
> 1. `executar()` grava `Modo de entrada` no `resumo_modelo.txt` — mesma
>    fonte única que o B3-1 usa para o `cv_label`.
> 2. Aviso de nível **WARNING** (não `[INFO]`) no início da execução em
>    modo imagem, dizendo explicitamente que a validação group-aware está
>    desligada e que os relatórios sairão carimbados.
> 3. PDF, Word e LaTeX carimbam **"PROTOTYPE OUTPUT — NOT VALIDATED"** na
>    **capa**, não em nota de rodapé, com texto de fonte única
>    (`_AVISO_PROTOTIPO_TITULO`/`_CORPO` em `reports.py`) para os três
>    geradores não divergirem. O ponto 3 do agravante (LaTeX afirmando
>    GroupKFold) já havia caído com a correção do B3-1.
>
> Teste de regressão em `test_reports.py`: gera os três formatos em modo
> `imagem` e em modo `dx` e confirma que o carimbo aparece **só** no
> primeiro. **Não implementado:** carimbo nas figuras individuais (só nos
> relatórios) — uma figura exportada solta do diretório `Graficos/`
> continua sem marca de protótipo.

---

## Fontes

- Lê Cao K.-A. et al. (2008). *A Sparse PLS for Variable Selection when
  Integrating Omics Data.* Statistical Applications in Genetics and
  Molecular Biology 7(1):35. Soft-thresholding nos vetores de loading.
  [researchgate](https://www.researchgate.net/publication/23562617_A_Sparse_PLS_for_Variable_Selection_when_Integrating_Omics_Data)
- Lê Cao K.-A. et al. (2011). *Sparse PLS discriminant analysis:
  biologically relevant feature selection and graphical displays for
  multiclass problems.* BMC Bioinformatics 12:253.
  [link.springer.com](https://link.springer.com/article/10.1186/1471-2105-12-253)
- Araújo M.C.U. et al. (2001). *The successive projections algorithm...*
  Chemom. Intell. Lab. Syst. 57:65–73. Conferido contra `_spa_cadeia`:
  a deflação cumulativa está correta; a limitação a `n_starts` pontos de
  partida (em vez de todas as variáveis) está declarada na docstring.
- Ambroise C. & McLachlan G.J. (2002). PNAS 99:6562–6566. Já citado
  corretamente pelo módulo como base da correção de nested-CV.
