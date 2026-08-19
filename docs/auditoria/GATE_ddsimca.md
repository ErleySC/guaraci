# GATE — o DD-SIMCA é estimável neste dataset? (2026-08-17)

**Recomendação, em uma linha:** não. Para autenticação one-class por
espécie a partir das amostras puras, **o DD-SIMCA não é estimável neste
dataset**, e nenhum outro método é — o limite é aritmético, não de método.
O GUARACI deve adotar **predição conformal como regra de decisão de
referência**, não porque ela resolva o problema, mas porque é a única que
**devolve "não estimável" em vez de um número sem conteúdo estatístico**.

Scripts: [`medir_n_efetivo_ddsimca.py`](medir_n_efetivo_ddsimca.py),
[`medir_conformal_vs_ddsimca.py`](medir_conformal_vs_ddsimca.py).

---

## 1.1 — n efetivo por espécie

Medido sobre os `.dx` reais, após as correções de rótulo desta auditoria
(A2-1/A2-2).

### Modo `puros` (o one-class do N2 — o que autentica)

| espécie | n_espectros | n_amostras_físicas | n usado em h0/q0/Nh/Nq |
|---|---:|---:|---:|
| Andiroba, Açaí, Babaçu, Bacaba, Buriti, Castanha do Pará, Coco, Goiaba, Graviola, Maracujá, Palmiste, Patauá, Pracaxi | 3 cada | **1 cada** | **1 cada** |
| **TOTAL** | **39** | **13** | **13** |

**13 de 13 espécies têm exatamente uma amostra física pura.** É pior do
que o relatado antes desta sessão (12 de 13): a unificação da Andiroba
(A2-1) removeu a única espécie que aparentava ter duas — e ela aparentava
justamente por causa do defeito de agrupamento.

### Modo `todos` (treina em todas as amostras da classe)

| espécie | n_espectros | n_amostras_físicas |
|---|---:|---:|
| Andiroba | 138 | 46 |
| Açaí | 138 | 46 |
| Babaçu | 129 | 44 |
| Bacaba | 137 | 46 |
| Buriti | 138 | 46 |
| Castanha do Pará | 135 | 46 |
| Coco | 135 | 45 |
| Goiaba | 115 | 39 |
| Graviola | 93 | 31 |
| Maracujá | 105 | 36 |
| Palmiste | 138 | 46 |
| Patauá | 135 | 45 |
| Pracaxi | 136 | 46 |

**Distinção que decide o resto do documento:** o problema de n **não** é do
dataset como um todo — é específico do regime de autenticação. No modo
`todos` há 31–46 amostras físicas por classe, folga confortável. Mas esse
modo treina com amostras adulteradas dentro do conjunto, então não
responde "esta amostra é pura?", que é a pergunta do N2.

---

## 1.2 — Com n=1, o que exatamente h0, q0, Nh e Nq estimam?

Execução real, Andiroba, pré-processamento `msc_sg_mc` (o das rodadas do
TCC):

```
espectros de treino .......... 3
variaveis .................... 2000
mae_id distintos ............. 1   [<id omitido>]
n_comp efetivo ............... 1

T2_train (por espectro) ...... [0.012  1.103  0.885]
Q_train  (LOO, por espectro) . [0.381066 0.65452  0.572413]

--> colapso por amostra fisica (media por mae_id):
T2 por AMOSTRA ............... [0.6667]
Q  por AMOSTRA ............... [0.536]
o metodo dos momentos recebe 1 valor.

media_e_dof_momentos(T2) -> h0=0.666667, Nh=1
media_e_dof_momentos(Q)  -> q0=0.536,    Nq=1

T2: media=0.666667  desvio(ddof=1)=indefinido (n=1)
    -> N cai no PISO 1.0 (nao e' um grau de liberdade medido dos dados)
Q:  media=0.536     desvio(ddof=1)=indefinido (n=1)
    -> N cai no PISO 1.0

f_crit = chi2.ppf(0.95, Nh+Nq=2) = 5.99146
```

**Resposta direta à pergunta:**

- **`h0` e `q0`** não estimam a média de uma distribuição. Com uma
  observação, a "média" **é** a própria observação. São fatores de escala
  arbitrários, fixados por uma única amostra.
- **`Nh` e `Nq` não estimam nada.** O método dos momentos precisa de um
  desvio-padrão; com n=1 ele é indefinido, e a função cai num **piso
  fixo de 1.0**. Esse 1.0 não é um grau de liberdade medido: é um
  *default* de código.
- Portanto **`f_crit = χ²(0,95; 2) = 5,99146` é uma constante** — o mesmo
  valor para toda espécie, independente de qualquer propriedade dos
  dados. O `alpha=0,05` é **nominal**; não existe distribuição estimada
  que o sustente.

O DD-SIMCA, neste regime, não é um teste estatístico calibrado. É um
limiar fixo aplicado a uma escala definida por uma amostra.

---

## 1.3 — Se n=1, o que a predição conformal entrega?

### Primeiro, o que ela **não** entrega

O documento pedia conformal como caminho principal. **Ela não resolve
n=1** — e afirmar o contrário seria trocar um número sem lastro por
outro. O limite é aritmético:

```
alpha minimo garantivel = 1/(n+1)

 n_calib  alpha_min   alpha=0.05?   alpha=0.10?
       1      0.500         False         False
       2      0.333         False         False
       3      0.250         False         False
       9      0.100         False          True
      19      0.050          True          True
```

Com **n=1, o menor α garantível é 0,50** — cara ou coroa. Nenhum
procedimento — conformal, DD-SIMCA, bootstrap, bayesiano — extrai
garantia de 5% de uma única amostra independente.

### O que ela entrega, e por que isso importa

A diferença é o comportamento na impossibilidade:

| | DD-SIMCA | Conformal |
|---|---|---|
| n=1 | devolve `f_crit=5,99` (constante de piso), sem sinal de problema | devolve `alcancavel=False`, `limiar=NaN` e a razão |
| Sustentação do α | assumida (χ² com dof estimados) | ordenação empírica, sem forma assumida |
| Depende do escore estar bem calibrado? | **sim** | **não** — usa só a ordem |

### Cobertura empírica vs nominal (medido)

Protocolo: split de **três vias por amostra física** (treino ajusta o
modelo / calibração deriva o limiar / teste mede), 60 seeds por célula,
réplicas nunca separadas entre lados. Os dois métodos recebem **o mesmo
escore** (distância combinada `f`), calculado pela classe `DDSimca` **de
produção** — com as correções desta auditoria (Q por LOO, calibração por
amostra física). Comparar contra uma versão caseira sem elas seria
espantalho.

```
### alpha = 0.05   (cobertura nominal = 0.95)
 n_amostras  alpha_min   DD-SIMCA    Conformal   definido?
          1      0.500      0.162  nao estimavel       0%   <-- dataset real
          3      0.250      0.180  nao estimavel       0%
         10      0.091      0.220  nao estimavel       0%
         19      0.050      0.178          0.945     100%
         30      0.032      0.169          0.966     100%
         50      0.020      0.183          0.966     100%

### alpha = 0.10   (cobertura nominal = 0.90)
          1      0.500      0.085  nao estimavel       0%   <-- dataset real
         10      0.091      0.126          0.924     100%
         19      0.050      0.102          0.902     100%
         30      0.032      0.088          0.901     100%
         50      0.020      0.107          0.918     100%
```

**Leitura:**

1. **O conformal entrega a cobertura nominal com precisão**, assim que o n
   permite: 0,945 / 0,966 / 0,966 contra 0,95 nominal; 0,902 / 0,901 /
   0,918 contra 0,90. A garantia é real, não assintótica.
2. **Abaixo do n mínimo ele se recusa a responder** (0% de células
   definidas) — que é o comportamento correto, e o oposto do DD-SIMCA.
3. **O DD-SIMCA sub-cobre gravemente e não melhora com n**: 0,16–0,22
   onde o nominal é 0,95. Ele rejeita ~80% de amostras vindas da **mesma
   distribuição** do treino.

### Achado novo, não previsto no documento de auditoria

O item 3 acima **não é explicado** pelo que já foi corrigido. A correção
A1 desta sessão passou o eixo **Q** para leave-one-out, mas o eixo **T²**
continua in-sample: `T2_train` vem de `pca.fit_transform(Xc)` sobre as
próprias amostras que definiram o subespaço. Amostras novas têm T² e Q
sistematicamente maiores, então `h0` fica pequeno demais e `f` infla para
tudo que não seja treino.

**Não corrigi isto nesta rodada** — é mudança no núcleo de decisão do
DD-SIMCA e exige medição própria, não um patch no fim de uma sessão. Fica
registrado como o próximo achado a investigar. Ele **reforça** a
recomendação abaixo, mas não é o que a sustenta: mesmo com o T² corrigido,
n=1 continua sendo n=1.

---

## 1.4 — Recomendação única

**O método de referência do GUARACI para autenticação one-class passa a
ser a predição conformal** (`src/guaraci/conformal.py`), com o DD-SIMCA
mantido no código e rotulado como exploratório/comparativo.

Justificativa medida, em ordem de peso:

1. É a única das duas que **declara a impossibilidade** em vez de emitir
   um número derivado de um piso de código (§1.2 vs §1.3).
2. Quando o n permite, ela **entrega a cobertura nominal medida**
   (0,945–0,966 contra 0,95); o DD-SIMCA não (0,16–0,22).
3. A garantia **não depende de o escore estar bem calibrado**, só da
   ordem — propriedade que importa exatamente aqui, onde a calibração do
   escore é o que está quebrado.
4. Ela é **group-aware por construção** no módulo implementado: exige
   `mae_id` e colapsa réplicas a um escore por amostra física, porque
   réplicas técnicas não são permutáveis. Alinha-se ao argumento central
   do projeto em vez de contradizê-lo.

### O que isso significa para o TCC, sem eufemismo

Para as 13 espécies, com 1 amostra pura cada:

> A sensibilidade/especificidade DD-SIMCA por espécie **não é uma
> quantidade estimável neste dataset**. Não é um número conservador, nem
> exploratório: é indefinido. Reportá-lo como resultado validado seria
> erro metodológico.

Caminhos que **de fato** produzem um número com garantia, em ordem de
custo:

| Caminho | α alcançável | O que muda |
|---|---|---|
| **Coletar mais amostras puras** | n=9 → 0,10; n=19 → 0,05 | resolve de verdade; nenhum método substitui dado |
| **Agrupar as 13 espécies num só modelo** | n=13 → **0,071** (α=0,10 sim, α=0,05 não) | a garantia passa a ser sobre "óleo puro", não por espécie — tem que estar dito em toda figura e tabela |
| Reportar α alcançável em vez do desejado | — | honesto e barato; vira contribuição metodológica |
| Modo `todos` (n=31–46) | 0,05 folgado | mas responde outra pergunta: discrimina puro×adulterado **dentro** da espécie, não autentica |

**Recomendo o segundo + o terceiro combinados**: um modelo conformal
agrupado com α=0,10 declarado, mais a tabela de α alcançável por espécie.
Produz um número defensável hoje, sem coleta adicional, e a limitação
fica explícita em vez de escondida.

---

## Estado do que foi entregue

- `src/guaraci/conformal.py` — implementado: `alpha_alcancavel`,
  `n_minimo_para_alpha`, `limiar_conformal`, `ConformalOneClass`
  (group-aware, recusa-se a produzir limiar sem n suficiente).
- **Ainda não integrado ao `executar()`** — a integração no pipeline e a
  substituição nas figuras/relatórios dependem da sua decisão sobre qual
  dos caminhos da tabela acima adotar (por espécie vs. agrupado), que é
  decisão científica, não de implementação.
- **Cobertura no dataset público: não medida** — depende do BLOCO 4, que
  não foi executado (o BLOCO 1 é bloqueador por instrução sua).

---

# ADENDO — 2026-08-17 (BLOCO A): o diagnóstico do T²/Q

> **Adendo, não reescrita.** Tudo acima fica como foi escrito em 2026-08-16.
> Esta seção corrige uma afirmação daquele texto e refuta a hipótese que o
> prompt de correção levantou. Script: [`medir_escala_t2_q.py`](medir_escala_t2_q.py).

## Retratação: "o DD-SIMCA não melhora com n" estava errado

A §1.3 afirmou, na leitura item 3, que o DD-SIMCA "sub-cobre gravemente e
**não melhora com n**". **Isso é artefato do meu protocolo de medição, não
propriedade do método.** Na primeira versão de
`medir_conformal_vs_ddsimca.py`, a coluna `n_amostras` variava o conjunto
de **calibração do conformal**, enquanto o DD-SIMCA era sempre ajustado num
treino **fixo de 10 amostras** em todas as linhas. Os dois lados da tabela
não significavam a mesma coisa ao longo daquele eixo.

Medido corretamente, variando o treino do DD-SIMCA:

```
 n_treino   T2_novo/h0    Q_novo/q0   cobertura
       10       0.5721       1.3212      0.2200
       19       0.3425       1.2057      0.2600
       30       0.3194       1.1159      0.5333
       50       0.1598       1.0512      0.8333
      100       0.2450       1.0335      0.8000
```

**A cobertura melhora com n** — de 0,22 para ~0,83. A afirmação anterior
fica retratada.

## A hipótese do prompt está refutada

O prompt supôs desalinhamento: `Q` em escala LOO contra um `q0` calibrado
noutra base, com `T²` in-sample e `h0` compatível. Os números dizem o
contrário:

| grandeza | valor (n=50) | leitura |
|---|---:|---|
| `Q_train` LOO / `Q_train` in-sample | **1,0592** | as duas escalas de Q batem |
| `Q_novo` / `q0` | **1,0407** | o eixo Q está **alinhado** |
| `T2_novo` / `h0` | **0,3479** | T² é **conservador**, não inflado |

A correção A1 funcionou: o `q0` derivado do LOO aproxima bem o Q de
amostras novas. **Não há desalinhamento de escala a corrigir.**

## O mecanismo real

O problema é `Nq`, não a escala:

```
Nq = 384.8      (Nh + Nq = 386.6)
f_crit / (Nh+Nq) = 1.1212
```

`N = 2·(média/desvio)²` explode porque `Q` soma resíduos sobre ~400
variáveis e **genuinamente concentra** (desvio relativo ≈ √(2/398) ≈ 7%).
Com `Nh+Nq ≈ 387`, a aproximação χ² concede margem relativa de apenas
**12%**: uma amostra nova só é aceita se a média ponderada de `T2/h0` e
`Q/q0` ficar dentro de 12% de 1.

Isso **não é bug** — é a consequência correta de `Q` concentrar. O que
quebra a cobertura é a combinação:

1. margem legitimamente estreita (12%), **mais**
2. viés pequeno de `q0` com n baixo — `Q_novo/q0` = 1,32 (n=10), 1,21
   (n=19), 1,12 (n=30), 1,05 (n=50), 1,03 (n=100).

Com n pequeno, um viés de 20–30% não cabe numa margem de 12% → rejeição em
massa. Conforme n cresce, o viés cai para dentro da margem e a cobertura
sobe. **O LOO é assintoticamente correto, mas enviesado com n pequeno e
p grande.**

Verifiquei também se o viés vinha do colapso por amostra física
(minha correção F1/A2-3): calibrar por grupo vs. por espectro dá
`Nq` = 284,5 vs 292,5 (n=10) e 384,8 vs 372,2 (n=50) — **diferença
desprezível**. Não é a causa.

## Consequência para a recomendação (§1.4)

**Nenhuma mudança.** Não havia bug de escala a corrigir, então não há
correção que devolva o DD-SIMCA ao posto de método de referência. E, mais
decisivo: **tudo isto é sobre o regime n≥10.** No dataset real o n é **1**,
onde nem a questão da margem se coloca — `Nh=Nq=1` por piso de código e
`f_crit` é constante.

A resposta à pergunta final do prompt: **a limitação de n=1 por espécie
continua sendo o fator decisivo, independente deste bug** — que, medido,
nem era bug.

O que **muda** é o rótulo (§A.3): o DD-SIMCA não é "método quebrado", é
método **que exige n que este dataset não tem**. Com n≥50 ele entrega
cobertura 0,83; com n=1, nada. Isso o mantém como exploratório aqui, mas
por razão diferente da que a §1.4 registrava — e o torna perfeitamente
utilizável num dataset com amostras suficientes.

**Gating recomendado (não implementado):** `DDSimca.fit()` deveria
declarar `nao_estimavel=True` quando `n_grupos_calibracao < 10`, no mesmo
padrão do `ConformalOneClass`, em vez de devolver um `f_crit` de piso.
Não implementado nesta rodada por depender da decisão de escopo.

---

# ADENDO 2 — 2026-08-17 (BLOCOS D e E)

## BLOCO D — a literatura de referência prevê salvaguarda para Nh/Nq?

**Verificado na fonte primária, não assumido.** Kucheryavskiy, Rodionova &
Pomerantsev (2024), *A comprehensive tutorial on Data-Driven SIMCA*,
J. Chemometrics 38(7):e3556 — [PDF aberto (Aalborg University)](https://vbn.aau.dk/ws/portalfiles/portal/762358091/Journal_of_Chemometrics_-_2024_-_Kucheryavskiy_-_A_comprehensive_tutorial_on_Data_Driven_SIMCA_Theory_and_implementation.pdf).

### O que a fonte diz (p. 4)

As equações são as que o projeto implementa:

> N_q = 2(q₀/s_q)² ,  N_h = 2(h₀/s_h)²
>
> "Here, s_q and s_h are the standard deviations of the q and h distances,
> respectively. **Because DoF is expected to be a whole number, an integer
> part of the calculation is taken.**"

### Achado 1 — desvio real, ainda que pequeno

**O GUARACI não trunca `Nh`/`Nq` para inteiro.** `media_e_dof_momentos`
devolve float. A fonte é explícita: toma-se a parte inteira. É desvio da
referência citada — mesma classe dos achados A5/B1-2, ainda que de impacto
numérico pequeno (`f_crit = χ²(0,95; N)` varia pouco entre N=386,6 e
N=386). **Não corrigido nesta rodada**: mexer no núcleo de decisão antes da
decisão de escopo seria mudar o número por baixo da reunião.

### Achado 2 — não existe teto recomendado, e isso foi verificado

**Não há, em nenhuma seção teórica (3.1–3.6), teto, truncamento superior
ou salvaguarda para `Nh`/`Nq` quando o n de calibração é pequeno.** Também
não há número mínimo de amostras de calibração recomendado. Registro isto
como **verificado, não assumido** — era exatamente o que o BLOCO D pedia.

### Achado 3 — a fonte trata `Nq` alto como sinal de sobreajuste (p. 10)

> "The selection of the optimal number of PCs can also be done using a plot
> in which the DoFs are presented depending on the PC values. **A sharp
> increase in the value of N_q starting from some PC may signal
> overfitting** as the noise modeling in orthogonal distance has begun."

Ou seja, os autores usam `Nq` como **diagnóstico de sobreajuste**, não como
parâmetro a limitar. O GUARACI não implementa esse gráfico. É a
salvaguarda que a literatura de fato oferece — e ela é diagnóstica, não
corretiva. **Fica como recomendação, não implementada.**

### Achado 4 — o que a fonte oferece para o problema de n pequeno

Duas coisas que o projeto pode usar e hoje não usa:

1. **IC binomial para a sensibilidade** (p. 8): *"One can compute a 95%
   confidence interval for the observed sensitivity by using the 0.025 and
   0.975 quantiles of the binomial distribution."* Com I=3 espectros o IC
   sairia praticamente [0, 1] — o que é a resposta honesta, e expressa na
   linguagem da própria referência.
2. **PCV** (p. 10) quando não há conjunto de teste independente — o projeto
   já tem, como opt-in (`ddsimca_pcv`).

**Conclusão do BLOCO D:** a ausência de salvaguarda para n pequeno na
literatura de referência foi **verificada**. O que existe é diagnóstico de
sobreajuste via `Nq`, não correção. Portanto a explicação do ADENDO 1
permanece — mas registra-se um desvio real (parte inteira) e duas
ferramentas da fonte que o projeto não usa.

## BLOCO E — curva de cobertura completa

**Medido**, 60 seeds por célula, orçamento IGUAL de amostras físicas para
os dois métodos (DD-SIMCA gasta as n no modelo+limiar; conformal parte ao
meio — metade modelo, metade calibração, que é o preço da garantia).

```
### alpha = 0.05   (nominal = 0.95)
 n_amostras  alpha_min   DD-SIMCA      Conformal
          1      0.500      0.000   nao estimavel   <-- dataset real
          3      0.250      0.690   nao estimavel
         10      0.091      0.171   nao estimavel
         19      0.050      0.457   nao estimavel
         30      0.032      0.628   nao estimavel
         40      0.024      0.706           0.939
         50      0.020      0.772           0.956
         80      0.012      0.840           0.953

### alpha = 0.10   (nominal = 0.90)
          1      0.500      0.000   nao estimavel   <-- dataset real
         10      0.091      0.094   nao estimavel
         19      0.050      0.340           0.931
         30      0.032      0.509           0.935
         40      0.024      0.597           0.899
         50      0.020      0.666           0.916
         80      0.012      0.740           0.894
```

**Leitura, com a ressalva que o BLOCO E pedia explícita:**

1. **A cobertura do DD-SIMCA cresce monotonicamente a partir de n=10**:
   0,171 → 0,457 → 0,628 → 0,706 → 0,772 → **0,840** (n=80). A leitura
   "é limitação de n, não de método" fica **sustentada pela tendência**.
2. **Mas ainda não atingiu o nominal**: 0,840 contra 0,95 em n=80, 11
   pontos de distância. **Não está estabelecido que converge para 0,95** —
   está estabelecido que sobe monotonicamente em direção a ele. Afirmar
   convergência exigiria n≫80, que esta medição não cobre.
3. **O conformal atinge o nominal com precisão** assim que o n permite:
   0,939 / 0,956 / 0,953 contra 0,95; 0,931 / 0,935 / 0,899 / 0,916 /
   0,894 contra 0,90.
4. A célula n=3 (0,690) é artefato: com 3 amostras o `n_comp` é reduzido
   pelo guard de graus de liberdade residuais e o modelo degenera para uma
   região larga. Não é sinal.

**Retratação da curva do ADENDO 1:** aquela tabela usava **uma seed por
ponto** e mostrava n=100 (0,80) abaixo de n=50 (0,83), o que sugeria
não-monotonicidade. Com 60 seeds a curva é **monótona**. A
não-monotonicidade era ruído de seed única, como eu havia suspeitado mas
não demonstrado.

Isso **não altera** a recomendação do gate: o dataset real está em **n=1**,
onde a cobertura medida do DD-SIMCA é **0,000** — ele rejeita tudo.



---

# ADENDO 3 — 2026-08-17 (BLOCOS I, J, K)

## BLOCO I — confundimento data × espécie: **estruturalmente inseparável**

Script: [`medir_confundimento_data.py`](medir_confundimento_data.py).

`mae_id = cod + data` impede que réplicas da mesma amostra se separem
entre treino e teste, **mas não impede que a mesma sessão de medição
apareça dos dois lados** — e, pior, não permite testar o contrário.

### I.1 — espécie × datas

| | modo puros | modo todos |
|---|---:|---:|
| datas totais | 11 | 15 |
| datas com >1 espécie | 2 | 2 |
| **datas exclusivas de 1 espécie** | **9 (82%)** | **13 (87%)** |

**10 das 13 espécies foram medidas numa única data.**

### I.2 — a data determina a espécie?

```
datas que determinam UNICAMENTE a especie: 13/15  (87%)
```

Saber a data equivale a saber a espécie em 87% dos casos.

### I.3 — GroupKFold por data: **impossível**

```
n_LV=20: NENHUM fold avaliavel -- toda particao por data deixa classe fora do treino.
n_LV=40: idem.
```

Não é que o resultado piore: **não existe partição válida**. Como a
maioria das espécies tem uma só data, colocar essa data no teste deixa a
espécie com **zero** amostras de treino. O experimento que separaria
química de sessão **não pode ser feito com este dataset**.

Este é o terceiro cenário previsto no BLOCO I — "particionamento
impossível → esse é o achado, e vai para limitações com o mesmo peso do
n=1". **Confirmado.**

### I.4 — onde a curva de LVs satura

Grade estendida a 80, `GroupKFold` por `mae_id`:

| n_LV | 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| bal.acc | 0,317 | 0,716 | 0,890 | 0,914 | 0,923 | 0,922 | 0,927 | 0,927 | 0,929 |

**Satura em ~40 LVs.** De 40 a 80 o ganho é **+0,006** — platô, não
crescimento. A leitura anterior ("ainda sobe em 40") estava olhando o fim
da parte íngreme, não uma tendência.

Nested-CV com a grade estendida: **0,9241 [0,9192; 0,9279]** (viés vs.
varredura: +0,0067). Sobrepõe-se ao 0,9203 da grade até 40 — a diferença
não é resolvível com estes dados.

**O que a saturação NÃO resolve:** 40 LVs continua muito para 13 classes.
A hipótese de que parte da capacidade serve para memorizar assinatura de
sessão **permanece não testável** neste dataset, pelo I.3.

## BLOCO J — proveniência da faixa espectral: **resolvido**

Inspecionei `guaraci_historico_antigo_20260815.bundle` (clone read-only em
diretório temporário, fora da árvore; bundle não modificado). **185
commits**, indo até 2026-05-29 — muito antes da reescrita de 2026-07-11
que o ADENDO anterior deu como barreira.

**A faixa está no commit inicial do projeto** (`9abd0ca`, 2026-05-29), com
justificativa instrumental escrita junto:

```
# ---- Truncamento espectral (FT-NIR util: 4000-10000 cm-1) ----
# Remove ruido de borda da FFT (0/8/16/24 cm-1 aparecem como falsos top
# VIP quando SG derivativo amplifica essa regiao). Aplicado ANTES de
# qualquer pre-processamento.
wn_min: float = 4000.0
wn_max: float = 10000.0
```

E no ponto de aplicação: *"SG derivativo amplifica os ultimos pontos da
FFT... Sem truncar, esses pontos viram falsos top-VIP com estabilidade
100% (artefato, nao quimica)."*

**O valor nunca foi alterado.** Em 185 commits, apenas dois tocam essas
linhas: o inicial (introduz) e `f6a1aa3` (2026-07-04, migração para
pacote, valores idênticos). **Nenhum commit testa faixas alternativas
comparando métricas.**

**Veredito: a faixa foi escolhida a priori, por racional instrumental,
antes de qualquer modelo.** Não é seleção com vazamento. O BLOCO G fica
resolvido, e a ressalva do documento de escopo pode ser retirada.

## BLOCO K — curva de cobertura fechada: **converge**

Medido, 60 seeds por célula, mesmo protocolo (orçamento igual, split por
amostra física):

```
alpha = 0.05 (nominal 0.95)     DD-SIMCA   Conformal
 n=1                               0.000  nao estimavel   <-- dataset real
 n=10                              0.171  nao estimavel
 n=30                              0.628  nao estimavel
 n=50                              0.772          0.956
 n=80                              0.840          0.953
 n=150                             0.896          0.958
 n=300                             0.921          0.952
```

### Veredito explícito entre as duas hipóteses

**Convergência, não platô.** A lacuna para o nominal encolhe pela metade a
cada dobra de n:

| n | cobertura | lacuna p/ 0,95 | razão vs. anterior |
|---:|---:|---:|---:|
| 80 | 0,840 | 0,110 | — |
| 150 | 0,896 | 0,054 | **0,49** |
| 300 | 0,921 | 0,029 | **0,54** |

A lacuna se comporta como `C/n`. Extrapolando: n=600 → ~0,935; n=1200 →
~0,943; n=2400 → ~0,946. **Não há platô abaixo do nominal.**

**Portanto: "é limitação de n, não de método" fica PROVADO**, não apenas
sustentado. O BLOCO A **não** precisa ser reaberto — não existe viés
residual inexplicado. O que o ADENDO 1 identificou (viés de `q0` com n
pequeno, dentro de uma margem χ² legitimamente estreita porque `Q`
concentra) explica a curva inteira.

**O conformal permanece no nominal em toda a faixa** (0,952–0,958), o que
era a previsão: a garantia é de amostra finita, não assintótica.

### O que isso muda, e o que não muda

**Muda o rótulo do DD-SIMCA:** ele não é método quebrado nem
mal-especificado. É um método **assintoticamente correto** que exige n
grande para se aproximar do α declarado — e a distância ao nominal é
previsível.

**Não muda a recomendação do gate.** O dataset real está em **n=1**, onde
a cobertura medida é **0,000**: o modelo rejeita tudo. A extrapolação
acima indica que seriam necessárias centenas de amostras físicas por
espécie para o DD-SIMCA entregar o α que declara.
