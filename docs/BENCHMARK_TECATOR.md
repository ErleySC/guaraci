# Validação externa — dataset Tecator (NIR, teor de gordura em carne)

> Resolve a lacuna citada em `docs/VALIDATION.md` ("sem *benchmark* contra
> um dataset público externo"). Todas as validações anteriores do GUARACI
> usavam dados sintéticos, propriedades matemáticas, ou o dataset próprio
> do autor (óleos amazônicos). Esta página roda o motor real de
> pré-processamento + regressão PLS do GUARACI — não uma reimplementação —
> num dataset público, de terceiros, bem estabelecido na literatura de
> quimiometria, fora de qualquer dado que o autor tenha coletado.

## O dataset

**Tecator**: 215 amostras de carne finamente picada, espectro de
absorbância NIR (transmitância) em 100 canais, 850–1050 nm, medido num
Tecator Infratec Food and Feed Analyzer. Cada amostra tem o teor de
umidade, gordura e proteína determinado por química analítica de
referência. 25 amostras adicionais de extrapolação (alto teor de gordura
ou proteína) não são usadas aqui — seguimos o protocolo padrão da
literatura.

**Fonte**: [StatLib, Carnegie Mellon University](http://lib.stat.cmu.edu/datasets/tecator)
— domínio público, nota de permissão da Tecator AB preservada no arquivo
original. Referência primária (origem do dataset e do split
treino/teste convencional):

> THODBERG, H. H. A review of Bayesian neural networks with an application
> to near infrared spectroscopy. **IEEE Transactions on Neural Networks**,
> v. 7, n. 1, p. 56–72, 1996. doi:10.1109/72.478392

**Split usado** (convenção padrão da literatura para este dataset):
amostras 1–172 = treino, 173–215 = teste. O GUARACI nunca viu o conjunto
de teste durante a seleção de LVs (CV só no treino, `KFold` de 5 folds).

## Metodologia

Script reprodutível: [`scripts/benchmark_tecator.py`](../scripts/benchmark_tecator.py)
(baixa o dataset da fonte original a cada execução — não redistribuído no
repositório). Roda literalmente `guaraci.preprocessamento.construir_preprocessador()`
e `sklearn.cross_decomposition.PLSRegression` — os mesmos componentes que
`pipeline.pls_regressao_por_especie()` usa internamente — com seleção do
número de variáveis latentes por validação cruzada no conjunto de treino
(mesma metodologia usada em produção), para 4 dos presets de
pré-processamento do GUARACI.

```bash
python scripts/benchmark_tecator.py
```

**Nota sobre `mae_id`**: o Tecator não tem réplicas físicas documentadas
(cada amostra é uma medição independente), então a proteção *group-aware*
(o diferencial central do GUARACI) não é exercitada por este benchmark —
ela não se aplica a este dataset. O que este benchmark valida é o **motor
de pré-processamento espectral + regressão PLS**, não a lógica anti-vazamento
de réplicas (essa já é validada contra fórmula fechada, ver `VALIDATION.md`).

## Resultados (obtidos rodando o script nesta revisão)

| Pré-processamento | LVs | RMSECV (treino) | RMSEP (teste) | R²pred (teste) |
|---|---:|---:|---:|---:|
| MSC → SG(1ª deriv.) → MC | 10 | 2,155 | 2,343 | 0,9674 |
| SNV → SG(1ª deriv.) → MC | 11 | 2,003 | 2,089 | 0,9741 |
| Mean-centering apenas | 14 | 2,697 | 2,011 | 0,9760 |
| Autoscaling | 14 | 2,639 | 2,001 | 0,9762 |

RMSEP em pontos percentuais de gordura (a variável tem faixa 0,9–58,5% no
dataset completo). Seed fixo (42) — resultado reprodutível bit a bit
rodando o script novamente.

## Interpretação

- **RMSEP na faixa 2,0–2,3% de gordura, R²pred 0,97–0,98** — consistente
  com a faixa amplamente reportada na literatura de quimiometria para PLS
  neste dataset (a caracterização do Tecator como *benchmark* padrão de
  regressão em NIR é corroborada por múltiplas fontes independentes —
  OpenML, o pacote R `caret`, dezenas de artigos que o usam para comparar
  métodos). **Não fazemos a alegação mais forte de "reproduz o RMSEP exato
  de um artigo específico X"**: cada publicação usa combinações diferentes
  de pré-processamento, número de LVs e critério de seleção, o que torna
  uma comparação ponto-a-ponto não-trivial sem replicar o protocolo exato
  de cada uma. A alegação verificável e honesta é: **o motor do GUARACI,
  rodando num dataset que o autor nunca viu, produz um resultado dentro do
  intervalo esperado por um método PLS bem calibrado** — não um número
  patológico (o que indicaria um bug de pré-processamento ou vazamento).
- **Achado interessante, não escondido**: o preset `msc_sg_mc` — o melhor
  preset no dataset próprio do autor (óleos amazônicos FT-NIR, Bal.Acc =
  0,923) — é o **pior** dos 4 aqui (RMSEP = 2,343). `autoscaling` (o pior
  em muitos cenários de FT-NIR com espalhamento) é o **melhor** aqui
  (RMSEP = 2,001). Isso é esperado e cientificamente correto: a Tecator
  usa NIT (transmitância), não NIR difuso, então o espalhamento que
  MSC/SNV corrigem em FT-NIR de óleos simplesmente não é o problema
  dominante neste instrumento — **não existe um preset universalmente
  melhor**, o que já era o argumento do `comparar_pre_processamentos` do
  próprio GUARACI.

## Limitações deste benchmark

- Testa o motor de **pré-processamento + regressão PLS**, não a validação
  *group-aware* via `mae_id` (não aplicável — sem réplicas neste dataset).
- Não testa DD-SIMCA, OPLS-DA, nem classificação (Tecator é um problema de
  regressão pura, uma "espécie").
- Comparação com a literatura é qualitativa (faixa esperada), não uma
  replicação exata de um número publicado específico — ver seção acima.
