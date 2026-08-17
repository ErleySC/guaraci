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
