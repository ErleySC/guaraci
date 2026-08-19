# Decisão de escopo — material para a reunião (2026-08-17)

Uma página. Todos os números vêm de medição nos dados reais, com script
reproduzível em `docs/auditoria/`. Onde não medi, está escrito "não medido".

---

## O que o gate mediu

O desenho atual do TCC é **one-class por espécie**: treinar um modelo
DD-SIMCA nas amostras puras de cada espécie e perguntar "esta amostra é
pura?". A auditoria mediu quantas amostras **físicas** puras existem por
espécie — não quantos espectros:

| | espectros | amostras físicas (`mae_id`) |
|---|---:|---:|
| Cada uma das 13 espécies | 3 | **1** |
| Total | 39 | **13** |

**As 3 medidas por espécie são réplicas técnicas do mesmo ponto de
amostragem, não amostras independentes.**

## Por que o desenho (C) não é estimável

O DD-SIMCA converte a distância ao modelo numa aproximação χ² cujos graus
de liberdade são estimados dos dados por `N = 2·(média/desvio)²`. Isso
exige estimar um **desvio-padrão**, e desvio exige mais de uma observação
independente.

Executado na Andiroba, com os números reais:

```
T2 por AMOSTRA ....... [0.6667]      <- uma observacao
desvio(ddof=1) ....... indefinido (n=1)
   -> N cai no PISO 1.0 (default de codigo, nao grau de liberdade medido)
f_crit = chi2.ppf(0.95, 2) = 5.99146   <- constante, igual p/ toda especie
```

Consequência: `h0`/`q0` são a própria observação; `Nh`/`Nq` não estimam
nada; e o limiar é uma **constante**, independente dos dados. O α=0,05 é
nominal — não existe distribuição estimada que o sustente.

**Isso não se resolve trocando de método.** O limite conformal
`α_min = 1/(n+1)` dá, com n=1, **α mínimo = 0,50** — cara ou coroa. Não é
limitação de software: é aritmética.

---

## Os três desenhos, com números reais

| | O que valida | n disponível | Validação | Estimável hoje? |
|---|---|---|---|---|
| **(A)** Discriminante de espécie (PLS-DA) + quantificação de teor (PLS-R) | classificação de espécie; teor de adulterante | **562 grupos `mae_id`**; 31–46 amostras físicas por classe | GroupKFold real por `mae_id`, **LV por CV aninhada** | **Sim** — bal.acc **0,9203** [0,9192; 0,9217] |
| **(B)** One-class agrupado (13 espécies como classe única "óleo puro") | pureza vs. adulteração, **sem** resolver espécie | **n = 13** amostras físicas | conformal | **Sim, com α=0,10** (α=0,05 **não**: α_min = 0,0714) |
| **(C)** One-class por espécie *(desenho atual)* | pureza por espécie | **n = 1** por espécie | — | **Não** |

### Detalhe do (A) — número corrigido por CV aninhada de LVs

O valor 0,9167 do rascunho anterior foi obtido **varrendo** n_LV e
reportando o melhor na mesma partição — o viés de máximo-de-N que o achado
B1-1 corrigiu no iPLS. Refeito com a escolha do n_LV numa partição
**interna** ao fold de treino (`StratifiedGroupKFoldEstavel`, 3 folds
internos), avaliada no fold externo que a busca nunca viu, 5 folds
externos, 5 seeds:

| | bal.acc | IC95% |
|---|---:|---|
| Varredura (enviesado) | 0,9243 | [0,9230; 0,9267] |
| **CV aninhada (honesto)** | **0,9203** | **[0,9192; 0,9217]** |
| viés | **+0,0041** | |

**O número para a reunião é 0,9203 [0,9192; 0,9217].**

O viés é pequeno (0,4 ponto) — ao contrário do iPLS (+7,0 pontos) — porque
a curva de LV é **monótona e saturante**: escolher o máximo de uma curva
monótona quase não tem winner's curse. A CV interna escolhe 30–40 LVs,
consistente com a externa.

**Ressalvas que precisam ir junto do número:**

- **A curva ainda sobe em 40 LVs** (o teto da grade testada):

  | n_LV | 5 | 10 | 20 | 30 | 40 | 60 | 80 |
  |---|---|---|---|---|---|---|---|
  | bal.acc | 0,317 | 0,716 | 0,890 | 0,914 | 0,923 | 0,927 | 0,929 |

  **Medido: satura em ~40 LVs** — de 40 a 80 o ganho é +0,006. Com a grade
  estendida a 80, a nested-CV dá 0,9241 [0,9192; 0,9279], que se sobrepõe
  ao 0,9203. A diferença não é resolvível com estes dados.
- 40 LVs para 13 classes continua sendo muito. Ver a seção de
  confundimento abaixo: a hipótese de que parte dessa capacidade memoriza
  assinatura de sessão **não é testável neste dataset**.
- Ainda **sem seleção de variáveis** e sem otimização de pré-processamento
  — é validação preliminar, não o pipeline de publicação.
- **Sem truncamento espectral o mesmo modelo dá 0,4809.** O truncamento é
  metade do resultado (ver seção sobre a origem da faixa, abaixo).

### A faixa [4000, 10000] cm⁻¹ foi escolhida cega ao resultado? **Sim.**

**Resolvido.** Inspecionei o bundle do histórico purgado
(`guaraci_historico_antigo_20260815.bundle`, 185 commits até 2026-05-29 —
anterior à reescrita que antes bloqueava a verificação).

A faixa está no **commit inicial do projeto**, com a justificativa
instrumental escrita junto: *"Remove ruído de borda da FFT (0/8/16/24 cm⁻¹
aparecem como falsos top VIP quando SG derivativo amplifica essa região).
Aplicado ANTES de qualquer pré-processamento."*

**O valor nunca foi alterado** — em 185 commits, só dois tocam essas
linhas: o inicial e a migração para pacote (valores idênticos). Nenhum
commit testa faixas alternativas comparando métricas.

**Não é seleção com vazamento.** A ressalva anterior está retirada.

### O problema que substitui esse: confundimento data × espécie

**Este é o achado que muda a interpretação do 0,92.**

| | modo puros | modo todos |
|---|---:|---:|
| datas de medição | 11 | 15 |
| **datas exclusivas de 1 espécie** | **82%** | **87%** |

**10 das 13 espécies foram medidas numa única data.** Saber a data
equivale a saber a espécie em 87% dos casos.

Tentei rodar o PLS-DA com `GroupKFold` **por data** (nenhuma sessão em
treino e teste ao mesmo tempo). **Não existe partição válida**: como a
maioria das espécies tem uma só data, pôr essa data no teste deixa a
espécie com zero amostras de treino.

**Consequência:** não é possível, com este dataset, distinguir quanto do
0,92 é discriminação **química** entre espécies e quanto é assinatura de
**sessão de medição** (deriva de instrumento, condições do dia). O
experimento que separaria as duas coisas não pode ser feito.

Isso **não invalida** o número — o modelo classifica espécie a 0,92 sob
validação group-aware honesta. Mas restringe o que se pode **afirmar**
sobre ele, e essa limitação tem o mesmo peso do n=1: entra nas limitações
do TCC, não em nota de rodapé.

**O que resolveria:** remedir um subconjunto de espécies em sessões
cruzadas (cada espécie em ≥2 datas, cada data com ≥2 espécies). Poucas
amostras bastariam — o objetivo é quebrar o confundimento, não aumentar n.

### Detalhe do (B) — confirmado nos dados reais

```
amostras fisicas puras (13 especies juntas) : 13
alpha minimo garantivel = 1/(n+1) ......... : 0.0714
alpha=0.05 alcancavel? .................... : False
alpha=0.10 alcancavel? .................... : True
```

O que muda: a garantia passa a ser sobre **"óleo amazônico puro"**, não
por espécie. Uma amostra rejeitada é "não compatível com óleo puro"; o
método **não diz qual espécie**. Isso tem de estar escrito em toda figura,
tabela e legenda — não em nota de rodapé.

---

## Custo até novembro — estimativa honesta, não otimista

| | Trabalho restante | Estimativa |
|---|---|---|
| **(A)** | Ligar LV aninhada no pipeline; reexecutar N1+N3; refazer figuras e tabelas; escrever. O motor já existe e está validado. | **3–5 semanas** |
| **(B)** | Integrar `conformal.py` ao `executar()`; decidir e documentar o escore; figuras novas de aceitação; escrever a limitação em todo lugar. Módulo pronto, integração não. | **4–6 semanas** |
| **(A)+(B)** | Os dois acima, com sobreposição parcial de escrita. | **7–10 semanas** — **não cabe** com folga até novembro |
| **(C)** | Não há trabalho que o torne estimável sem coletar mais amostras. | — |

Estimativas contam só implementação e execução; **não** incluem revisão da
orientadora, banca, nem imprevisto. Considerando que a experiência desta
auditoria foi que cada módulo olhado de perto revelou um achado, eu
trataria a faixa alta como a mais provável.

### O caminho que resolveria (C) de verdade

Coletar mais amostras físicas puras por espécie:

| amostras puras por espécie | α alcançável |
|---:|---|
| 1 (hoje) | 0,50 |
| 3 | 0,25 |
| 9 | 0,10 |
| 19 | 0,05 |

Nenhum método estatístico substitui isso. Se houver acesso a mais óleos
puros das mesmas espécies, é a intervenção de maior retorno — e a única
que preserva o desenho original.

---

## O que eu não decido

**A escolha entre (A) e (B) é sua e da orientadora.** Elas respondem
perguntas científicas diferentes, não são versões melhor/pior do mesmo
trabalho:

- **(A)** entrega um resultado forte e já medido, mas responde
  "que espécie é esta?" — não é autenticação de pureza.
- **(B)** preserva a pergunta original (pureza), com garantia estatística
  real, ao preço de perder a resolução por espécie e de aceitar α=0,10 em
  vez de 0,05.

O que a auditoria fecha é apenas isto: **(C), como está, não produz número
reportável**, e nenhuma escolha de método muda esse fato.

---

## Pendente para depois da reunião

- **BLOCO 2** (IC da Etapa 4, ≥20 seeds) e **BLOCO 4** (datasets públicos):
  válidos, mas quais figuras/tabelas fazem sentido depende do desenho
  escolhido.
- **BLOCO 3** (inventário do que regenerar): metade do inventário deixa de
  existir se o desenho mudar.
- **Push ao repositório remoto**: aguarda esta decisão **e** a conversa
  sobre autoria/distribuição de dados.
