# MANUAL DO GUARACI — PLATAFORMA DE QUIMIOMETRIA

> Manual de instruções das funcionalidades do projeto. Mantido atualizado a
> cada mudança relevante de funcionalidade, interface ou fluxo. Para
> instalação, citação e licença, ver `README.md`.

O **GUARACI** é uma plataforma de quimiometria multitécnica para autenticação
e caracterização de matrizes complexas — óleos amazônicos por FT-NIR e
qualquer dado espectral ou tabular (Raman, UV-Vis, FTIR, cromatografia).
Cobre todo o fluxo científico — análise exploratória, classificação,
autenticação, quantificação e relatórios de publicação — com validação
estatística rigorosa e proteção contra vazamento de réplicas em cada etapa.

## SUMÁRIO

1. [As três formas de usar](#1-as-três-formas-de-usar)
2. [Modos de análise e objetivo científico](#2-modos-de-análise-e-objetivo-científico)
3. [Estrutura de saída dos resultados](#3-estrutura-de-saída-dos-resultados)
4. [Fontes de dados de entrada](#4-fontes-de-dados-de-entrada)
5. [Funcionalidades científicas](#5-funcionalidades-científicas)
6. [Fluxo típico na interface web](#6-fluxo-típico-na-interface-web)
7. [Mapa dos módulos (para desenvolvedores)](#7-mapa-dos-módulos-para-desenvolvedores)
8. [Desenvolvimento](#8-desenvolvimento)
9. [Referências](#referências)

---

## 1 As três formas de usar

O código fica no pacote `guaraci` (em `src/`). Instale uma vez com
`pip install -e .` (disponibiliza o comando `guaraci`); sem instalar, use
`PYTHONPATH=src`.

| Forma | Comando | Para quem |
|---|---|---|
| **Web (Streamlit)** | `streamlit run app_quimiometria.py` | Uso visual, 8 abas guiadas. Demo público: <https://guaraci.streamlit.app> |
| **Assistente de terminal** | `guaraci` (ou `PYTHONPATH=src python -m guaraci.guaraci`) | Menu interativo colorido, sem editar código |
| **Pipeline direto** | `python -m guaraci.pipeline --rodar` | Execução automatizada a partir de `config.yaml` |

As três formas compartilham o mesmo motor (`pipeline.py`) e a mesma
configuração (`config.yaml` / classe `Config`) — não há divergência de
resultado entre elas.

### 1.1 Painel de acompanhamento em tempo real (terminal)

Ao rodar pelo assistente de terminal, a execução é acompanhada por um painel
ao vivo (biblioteca *Rich*) que mostra, a cada instante: o objetivo científico
resolvido (seção 2), o percentual real de progresso e a etapa em andamento
(lidos diretamente do log do motor, não estimados por tempo decorrido), o
tempo estimado restante, a lista de figuras já concluídas frente ao total
planejado para o objetivo em curso, e os avisos não fatais emitidos durante
a execução. O mesmo mecanismo de captura de log alimenta a barra de
progresso do aplicativo web.

---

## 2 Modos de análise e objetivo científico

O **modo de análise** define o que o pipeline faz. Na interface aparecem com
nomes amigáveis; internamente são identificados como N1/N2/N3.

### 2.1 Os três níveis (N1 / N2 / N3)

- **N1 — Classificação (por espécie).**
  Identifica a qual espécie/classe cada amostra pertence (por exemplo, 13 a
  14 óleos amazônicos). Método: **PLS-DA** com `GroupKFold` anti-vazamento
  de réplicas (as réplicas T1/T2/T3 do mesmo ponto amostral nunca são
  separadas entre treino e validação).

- **N2 — Discriminação (puro vs. adulterado).**
  Autentica a pureza **por espécie** via **DD-SIMCA** *one-class* (T² e
  Q-resíduos com limites de aceitação específicos por classe).

- **N3 — Quantificação (% de adulterante).**
  Estima o teor de adulterante por **regressão PLS calibrada por espécie**
  (`pls_regressao_por_especie`). Reporta também as **figuras de mérito
  analíticas** — LOD, LOQ, sensibilidade, sensibilidade analítica (γ) e
  seletividade, segundo Valderrama, Braga e Poppi (2009) — calculadas
  automaticamente a partir das réplicas físicas (T1/T2/T3) de cada espécie.
  Além do bloco de texto no console e no `resumo_modelo.txt` ("*Analytical
  Figures of Merit*"), essas figuras de mérito têm agora uma **representação
  gráfica dedicada** (`figS3_merito_regressao.png`, ver seção 5) — LOD/LOQ e
  seletividade por espécie lado a lado, com indicação explícita ("n/a") para
  espécies sem réplicas físicas suficientes para estimar o ruído
  instrumental. O *split* calibração/validação dessa regressão aceita dois
  métodos via `divisao_cal_val` no `config.yaml` (hiperparâmetro avançado,
  **não** exposto no aplicativo/CLI — mesmo padrão de
  `ipls_n_intervalos`/`vip_threshold_sel`): `"aleatoria"` (padrão,
  `GroupShuffleSplit` *group-aware*) ou `"kennard_stone"` (Kennard e Stone,
  1969 — cobertura maximamente representativa do espaço espectral em vez de
  aleatória; com réplicas físicas, colapsa cada grupo T1/T2/T3 num espectro
  médio antes de selecionar, preservando o mesmo invariante de nunca separar
  réplicas entre calibração e validação).

### 2.2 Objetivo científico: Exploratório, Classificação, Quantificação

Além do nível N1/N2/N3, cada execução resolve um **objetivo científico** —
**Exploratório**, **Classificação** ou **Quantificação** — que determina
**exclusivamente** quais figuras, cálculos e campos de relatório são
gerados. Essa camada (`modos_analise.py`) foi introduzida numa auditoria de
qualidade que constatou que, antes, os níveis N2 e N3 geravam
**exatamente o mesmo conjunto de figuras** — incluindo a figura de regressão
dentro de uma corrida de classificação, e vice-versa — misturando resultados
de propósitos distintos na mesma pasta de saída.

O campo `objetivo` (em `Config`/`config.yaml`, valor padrão `"auto"`) deriva
do nível quando não é definido explicitamente, preservando o comportamento
histórico:

| Nível | Objetivo derivado (`auto`) |
|---|---|
| N1 | Classificação |
| N2 | Classificação |
| N3 | Quantificação |

Pode ser sobreposto explicitamente para `"exploratorio"`, `"classificacao"`
ou `"quantificacao"`. O **Modo Exploratório** é a funcionalidade nova: gera
apenas as análises não supervisionadas (PCA, HCA, *loadings*,
pré-processamento), sem PLS-DA nem regressão — útil para uma primeira
inspeção do conjunto de dados antes de comprometer-se com um modelo
supervisionado.

**O que cada objetivo gera:**

| Objetivo | Figuras/relatórios pertinentes |
|---|---|
| **Exploratório** | PCA (*scores*), HCA (dendrograma), *loadings* PCA, efeito do pré-processamento |
| **Classificação** | PLS-DA (*scores*), matriz de confusão, ROC/AUC, VIP, seleção de LVs, Selectivity Ratio, DD-SIMCA, OPLS-DA, Etapa 4, teste de Wold, holdout, teste de Martens, Auto-Benchmark, Monte Carlo CV, SHAP |
| **Quantificação** | Regressão PLS + figuras de mérito analíticas (LOD/LOQ/SEN/SEL) |

Figuras de **contexto geral** — PCA de *scores* (visão geral) e o painel de
*outliers* T²/Q — aparecem em **qualquer** objetivo, pois derivam do próprio
ajuste do modelo usado para a visão geral da amostragem, independentemente
do propósito específico da corrida.

**Otimização de desempenho:** a filtragem por objetivo não suprime apenas a
*figura* — também evita a **computação** cara que só interessaria à
classificação. O teste de permutação (*Y-randomization*, tipicamente 200
reamostragens) e o intervalo de confiança BCa das métricas de classificação
(acurácia, acurácia balanceada, F1, kappa de Cohen) só são calculados quando
o objetivo resolvido é Classificação; fora dela, essas etapas são puladas
por completo (medição: cerca de 75% de redução no tempo total de uma corrida
de Quantificação frente a uma corrida de Classificação equivalente), e os
respectivos campos somem do `resumo_modelo.txt`/`model_card.md` em vez de
aparecerem como valores não computados.

**Curadoria de figuras por tipo de análise (regras adicionais):**
- **DD-SIMCA** (autenticação de pureza) é um conceito de **N2**. Em **N1**
  (classificação por espécie), o toggle `ddsimca`/`executar_ddsimca` é
  **ignorado com aviso** — não agrega a um estudo de identificação de
  espécie. Em N2 ele é sempre ligado automaticamente (não precisa
  configurar).
- **OPLS-DA** não é específico de nenhum nível — no Guaraci ele discrimina
  **espécie** (mesmo alvo do PLS-DA, via LDA quando há mais de duas
  classes), então continua disponível como extra no objetivo Classificação.

**Conjunto padrão de fábrica (~7 a 9 figuras "*core*"), qualquer nível:**
PCA (*scores*), PLS-DA (*scores*), *outliers* T²/Q, matriz de confusão,
ROC/AUC, curva de seleção de LVs, VIP + Selectivity Ratio — mais *bootstrap*
VIP e avaliação em *holdout* quando os respectivos parâmetros estão ativos
(padrão). N2 soma a figura de aceitação DD-SIMCA; N3 soma a figura de
regressão PLS e a figura de mérito analítica dedicada. Tudo o mais —
OPLS-DA, Etapa 4 (seleção de variáveis), comparação de *pipelines* de
pré-processamento, HCA comparativo, teste de Wold, CV-ANOVA, Auto-Benchmark,
Monte Carlo CV, SHAP, figuras detalhadas (`figuras_detalhadas`) — é *opt-in*:
o usuário liga explicitamente quando quiser ir além do conjunto padrão.

---

## 3 Estrutura de saída dos resultados

Cada execução grava seus resultados em uma hierarquia de pastas que separa
**amostra/conjunto de dados**, **objetivo científico** (seção 2.2) e o
conteúdo por categoria:

```text
<pasta_saida_raiz>/
  <Amostra>/
    <Modo>/
      <Execução>/
        Graficos/
        Tabelas/
        Relatorios/
        Modelos/
```

- **`<Amostra>`** — identificador do conjunto de dados. Vem do rótulo livre
  `tag` (em `Config`/`config.yaml`) quando preenchido; senão é derivado
  automaticamente do modo de entrada (nome do arquivo CSV, nome da pasta de
  espectros, ou `"sintetico"` para dados de teste).
- **`<Modo>`** — rótulo amigável do objetivo científico resolvido:
  `Exploratorio`, `Classificacao` ou `Quantificacao`.
- **`<Execução>`** — identificador único da corrida (nível, pré-processamento
  e data/hora), no formato `PLSDA_OE_<nivel>_<preproc>_<AAAAMMDD_HHMMSS>`.
- **`Graficos/`** — todas as figuras (`.png`/`.pdf`/`.svg`, conforme
  `formato_saida`), incluindo subpastas de figuras detalhadas (por exemplo,
  `ddsimca/`).
- **`Tabelas/`** — dados tabulares em CSV (identificadores de amostra,
  metadados, teste de Martens, comparação de *pipelines*, benchmarks).
- **`Relatorios/`** — `resumo_modelo.txt` e `model_card.md`.
- **`Modelos/`** — modelo final exportado (`modelo_plsda.joblib`).

Exemplo real (execução de Classificação com `tag="oleos_essenciais"`):

```text
resultados_tcc/oleos_essenciais/Classificacao/PLSDA_OE_N2_MSC-SG1-MC_20260705_222028/
  Graficos/fig1_pca_scores.png
  Graficos/fig2_plsda_scores.png
  ...
  Tabelas/amostras_identificadores.csv
  Relatorios/resumo_modelo.txt
  Relatorios/model_card.md
  Modelos/modelo_plsda.joblib
```

> **Compatibilidade com resultados anteriores:** o aplicativo web continua
> lendo `resumo_modelo.txt`/`model_card.md` de execuções geradas antes desta
> reestruturação (que usavam a pasta `logs/` em vez de `Relatorios/`) sem
> necessidade de migração manual.

---

## 4 Fontes de dados de entrada

Configuráveis via `modo_entrada` (aplicativo, CLI ou `config.yaml`):

| Modo | Origem | Observação |
|---|---|---|
| `dx` | Espectros JCAMP-DX (FT-NIR/Raman/MIR) | Padrão; uma subpasta por classe |
| `csv` | Tabela genérica (colunas espectrais + uma coluna de classe) | Qualquer dado tabular |
| `imagem` | **Colorimetria digital (protótipo)** | Ver adiante |
| `sintetico` | Dados simulados | Para testes/demonstração |

**Modo `imagem` (colorimetria digital, protótipo):** extrai estatísticas de
cor (média/desvio-padrão por canal em RGB, HSV e Lab — 18 variáveis) de cada
fotografia e, opcionalmente, textura (GLCM, requer `pip install
scikit-image`). Mesma convenção de pastas do modo `dx` (uma subpasta por
classe). A partir da extração, toda a maquinaria quimiométrica (PCA, PLS-DA,
DD-SIMCA, seleção de variáveis, figuras de mérito) funciona sem alteração —
cada estatística de cor vira uma "variável", exatamente como um comprimento
de onda.

**Duas configurações obrigatórias ao usar `modo="imagem"`:**
1. `pre_processamento` deve ser `autoscaling` ou `mc` — **nunca** um preset
   com Savitzky-Golay (`msc_sg_mc`/`snv_sg_mc`), que pressupõe um sinal
   espectral contínuo, sem sentido para um vetor curto de estatísticas de
   cor discretas.
2. `faixa_min_cm`/`faixa_max_cm` devem cobrir o intervalo `0` a
   `n_features-1` (por exemplo, `-1` a `100`) — o eixo de variáveis aqui é
   um índice simbólico, não um número de onda real, e os padrões de fábrica
   (4000–10000) descartariam todas as variáveis.

Sem caso de uso específico ainda amarrado (protótipo genérico) — cabe ao
usuário definir a região de interesse via `imagem_recorte` (recorte
retangular relativo, `config.yaml`) antes da extração.

---

## 5 Funcionalidades científicas

**Pré-processamento espectral** (dentro do `Pipeline` do *scikit-learn*, sem
vazamento entre *folds* de validação cruzada): SNV, MSC, Savitzky-Golay
(suavização ou derivada), *mean-centering*, *autoscaling*. Presets prontos:
`msc_sg_mc` (melhor desempenho no conjunto de referência), `snv_sg_mc`,
`mc`, `autoscaling`, `custom`.

**Análise exploratória:** PCA (*scores* e *loadings*), HCA (dendrograma de
Ward sobre componentes principais).

**Classificação e discriminação:** PLS-DA, OPLS-DA (com S-*Plot*), DD-SIMCA
(com Cooman's *Plot*).

**Seleção de variáveis (Etapa 4)** — sempre executados: iPLS (por
intervalos), VIP ≥ 1, Selectivity Ratio (20% superior), sPLS-DA esparso.
Opcionais (mais lentos, ligar quando quiser comparar mais a fundo):
- **SPA/APS** — Algoritmo das Projeções Sucessivas (Araújo et al., 2001):
  monta cadeias de variáveis com baixa colinearidade entre si.
- **AG** — Algoritmo Genético (GA-PLS, Leardi, 2000): evolui uma população
  de subconjuntos de variáveis por seleção, cruzamento e mutação, usando
  acurácia balanceada via validação cruzada como aptidão.

Todos os métodos — sempre ligados e opcionais — são avaliados sob o **mesmo
esquema de validação cruzada *group-aware***, permitindo comparação direta
numa única tabela e figura.

**Validação estatística:** teste de permutação, teste de Wold (R²Y/Q²Y),
CV-ANOVA (Eriksson, Trygg e Wold, 2008), *bootstrap* BCa (intervalo de
confiança da acurácia), *holdout* externo *group-aware*, Monte Carlo CV
(IC 95%). O teste de permutação e o intervalo BCa são calculados apenas no
objetivo Classificação (seção 2.2).

**Teste de incerteza de Martens** (Martens e Martens, 2000) — *opt-in* via
`teste_martens` (aplicativo e CLI, aba/menu Validação): *jackknifing*
*group-aware* dos coeficientes de regressão PLS (reaproveita a mesma
validação cruzada de seleção de LVs). Produz um **teste de hipótese
formal** (estatística *t* + valor-p) de significância por variável — mais
rigoroso que VIP/Selectivity Ratio, que são medidas de *magnitude* sem
valor-p associado. Em modelos multiclasse, o resultado por variável é o
máximo |*t*| entre as classes (significativa se discrimina pelo menos uma).
Gera `Tabelas/teste_martens.csv` (comprimento de onda, *t*, *p*,
significativo) e um resumo (número de variáveis significativas) no
`resumo_modelo.txt`/`model_card.md`.

**DModX / DModY** (nomenclatura padrão SIMCA-P/Unscrambler, Eriksson et al.,
2006) — sempre calculados, sem *toggle*: são o **mesmo** T²/Q-resíduo e
resíduo de predição já usados nas figuras (`fig3_outliers`/
`fig7_pls_regressao`), apenas **normalizados e nomeados** na escala/convenção
que usuários vindos dessas ferramentas comerciais já reconhecem (DModX ≈ 1 =
resíduo típico; acima do limite crítico = fora do modelo). Não geram figura
nova (seria redundante com as já existentes) — aparecem como resumo (limite
crítico + número de amostras fora) no console, `resumo_modelo.txt` e
`model_card.md`. DModX é sempre reportado (classificação); DModY aparece
quando há regressão (Quantificação).

**Comparação de modelos (Auto-Benchmark):** PLS-DA vs. SVM RBF vs. *Random
Forest* vs. *Gradient Boosting* vs. XGBoost, sob a mesma validação cruzada
*group-aware*. Curvas DET e interpretabilidade via **SHAP**
(*TreeExplainer*).

**Auto-Benchmark de regressão (Quantificação):** PLS-R (o modelo já
calibrado por `pls_regressao_por_especie`, reaproveitado sem reajuste) vs.
Ridge, Lasso, *Elastic Net*, SVR (RBF) e *Random Forest Regressor* — um
modelo **por espécie** (mesma arquitetura da quantificação, calibração
separada evita que a variação entre espécies confunda o sinal de
adulteração), com o **mesmo *split* calibração/validação** (determinístico,
mesma semente/`divisao_cal_val` do PLS-R) e o mesmo pré-processamento, para
uma comparação honesta ponto a ponto. *Opt-in* via `benchmark_regressao`
(aplicativo e CLI, categoria Avançado — mesmo padrão do Auto-Benchmark de
classificação): gera `benchmark_regressao.csv` (RMSEP/R² agregado e por
espécie) e `fig_benchmark_regressores.png` (*boxplot* de RMSEP por espécie,
menor é melhor).

**Predição em amostras novas (aplicativo e CLI):** aplica um modelo já
treinado (`modelo_plsda.joblib`, salvo automaticamente ao final de cada
execução) a espectros novos, sem rodar o pipeline inteiro de novo. Entrada:
CSV com colunas = número de onda (sem coluna de classe). Saída: classe
predita, confiança (%) e **dois diagnósticos complementares** de
confiabilidade:
- **Ajuste ao modelo PLS-DA** (colunas `T2`/`Q`/`aceito`) — o quanto a
  amostra se afasta do que o modelo de classificação capturou.
- **Domínio de aplicabilidade** (colunas `AD_*`, Jaworska, Nikolova-Jeliazkova
  e Aldenberg, 2005) — o quanto a amostra é um espectro atípico frente ao
  conjunto de calibração em geral, via T²/Q num PCA exploratório
  independente da classe. Reaproveita
  `chemometric_stats.dominio_aplicabilidade_amostras_novas`; só aparece se
  o modelo foi salvo por uma versão do pipeline que exporta esses artefatos
  (retrocompatível — modelos antigos continuam predizendo normalmente, só
  sem essas colunas extras).

Disponível em dois lugares, com a **mesma lógica científica**
(`predicao.py`, compartilhado, sem duplicação):
- **Aplicativo web** — aba 🔮 *Prediction*.
- **CLI** — menu principal, tecla `[B]` *Predição em Lote*: pede o caminho
  do modelo, do CSV de espectros novos e do CSV de saída (Enter = mesmo
  nome do CSV de entrada mais `_predicao.csv`), e imprime um resumo por
  classe mais os dois diagnósticos acima. Útil para automação/*scripts* e
  integração com LIMS sem precisar do navegador.

> **Segurança — upload de modelo em implantação pública.** Um arquivo
> `.joblib` é um *pickle*: carregá-lo **executa código** contido no
> arquivo. Em uso local (própria máquina) isso é seguro, pois o modelo é
> do próprio usuário. Mas num demonstrativo hospedado publicamente, aceitar
> *upload* de `.joblib` de qualquer visitante é um vetor de execução remota
> de código (RCE). Por isso, na implantação pública, defina a variável de
> ambiente **`GUARACI_DISABLE_MODEL_UPLOAD=1`**: o aplicativo esconde o
> carregador de modelo e passa a aceitar apenas **caminho local**
> (controlado pelo operador). O *upload* de CSV de espectros permanece
> liberado (dado inerte). Sem a variável (padrão), o *upload* fica
> habilitado com um aviso — apropriado para uso local com um único usuário.

**Figuras:** conjunto essencial por padrão (cerca de 8 a 10 figuras, a
depender do objetivo — seção 2.2) com opção de figuras detalhadas adicionais
(`figuras_detalhadas=True`). Formatos PNG/PDF/SVG, DPI configurável.

**Figura de mérito analítica dedicada (Quantificação):**
`figS3_merito_regressao.png` — dois painéis lado a lado: LOD/LOQ por espécie
e seletividade média por espécie, seguindo Valderrama, Braga e Poppi (2009).
Antes desta funcionalidade, esses valores só apareciam como tabela de texto
no `resumo_modelo.txt`/`model_card.md`.

**Relatórios:** PDF, Word (`.docx`), Excel (cinco abas), LaTeX e PowerPoint,
com capa de projeto (nome, autor, instituição, objetivo — o "tipo de
estudo" é derivado automaticamente do objetivo científico da execução).

**Model Card (`model_card.md`):** documento de uma página gerado
automaticamente ao final de toda execução, no padrão *Model Cards for Model
Reporting* (Mitchell et al., 2019) — o mesmo formato usado por plataformas
de *ML-ops* (por exemplo, Hugging Face Hub) para trilha de auditoria e
transparência. Seções: detalhes do modelo (versão, algoritmo,
pré-processamento), uso pretendido (e fora de escopo, específico por
objetivo), fatores relevantes (classes, validação cruzada *group-aware*),
métricas de desempenho, dados de avaliação/treino (integridade, tamanho),
análises quantitativas por classe, considerações éticas e ressalvas
metodológicas (as mesmas do `resumo_modelo.txt`, fonte única). Em
Quantificação, ganha um adendo com as figuras de mérito de regressão.
Aparece na aba **Relatórios** do aplicativo (prévia e download `.md`
próprio) e em `Relatorios/` de toda execução (CLI e aplicativo).

---

## 6 Fluxo típico na interface web

1. **Projeto** — preencha nome, autor, instituição e objetivo (campos
   descritivos, entram na capa dos relatórios).
2. **Dados** — faça *upload* de um CSV ou aponte a pasta de espectros `.dx`
   (uma subpasta por classe).
3. **Pré-processamento** — escolha o preset e confira a visualização de
   antes/depois.
4. **Modelo** — escolha o nível de análise e, se necessário, o objetivo
   científico explícito (seção 2.2), ajuste os parâmetros (variáveis
   latentes, *holdout*, validação, módulos extras, figuras) e clique em
   **▶️ Run pipeline**.
5. **Validação / Predição / Relatórios** — inspecione as métricas por
   classe e as figuras geradas, e baixe os relatórios e o ZIP de
   resultados.
6. **Sobre** — identidade do projeto, comparativo com softwares
   comerciais, licença (GPL-3.0-or-later) e como citar (APA/ABNT/BibTeX).

Tema claro/escuro: menu ⋮ → *Settings* → *Theme* (segue a preferência do
sistema operacional por padrão).

Cabeçalho: logo, versão e badges (licença/instituição) ficam sempre
visíveis no topo, antes das abas. Quando o app roda **sem** `config.yaml`
local (caso do deploy público em `guaraci.streamlit.app`, que não tem
acesso aos dados reais de pesquisa), aparece um aviso de **modo
demonstração** explicando que os espectros são sintéticos.

---

## 7 Mapa dos módulos (para desenvolvedores)

O motor do pipeline é modularizado por responsabilidade. `pipeline.py`
funciona como **fachada**: reexporta todos os símbolos públicos dos módulos
abaixo, então `import pipeline as pq; pq.X` continua funcionando sem
alteração, não importa em qual arquivo `X` esteja implementado de fato.

| Módulo | Responsabilidade |
|---|---|
| `pipeline.py` | Orquestrador `executar()`, menu de terminal legado e **fachada de reexport** de todos os módulos |
| `modos_analise.py` | **Objetivo científico** (Exploratório/Classificação/Quantificação, seção 2.2): fonte única que decide quais figuras/relatórios cada execução gera (`resolver_objetivo`, `deve_gerar`, `descrever_plano`). `descrever_plano` filtra tanto pelo objetivo quanto pelos módulos opt-in ligados (DD-SIMCA, OPLS-DA, Benchmark...) — alimenta o painel ao vivo do terminal **e** a prévia "O que será gerado" da aba Model do app web, que atualiza em tempo real conforme os toggles mudam |
| `config_io.py` | **Fonte única da configuração**: `_CONFIG_SPEC` (campo amigável ↔ atributo), ler/gravar/validar/coagir (`carregar_config`, `salvar_config`, `_coagir_valor`, `_validar_semantico`) |
| `resultados_io.py` | Escrita dos artefatos de uma corrida: `resumo_modelo.txt`, `model_card.md`, identificadores CSV, notas metodológicas, métricas PLS |
| `config.py` | *dataclass* `Config`, fonte única de `__version__`/`_NIVEL_NOME` e das constantes de nome de pasta (`NOME_GRAFICOS`/`NOME_TABELAS`/`NOME_RELATORIOS`/`NOME_MODELOS`, seção 3) |
| `chemometric_stats.py` | VIP, Selectivity Ratio, teste de incerteza de Martens, Hotelling T², Q-resíduos, variância explicada, figuras de mérito (LOD/LOQ/SEN/SEL), domínio de aplicabilidade |
| `paleta_cores.py` | Paleta e marcadores de máxima distintividade por classe |
| `dados_io.py` | *Parsing* JCAMP-DX/ASDF, CSV e modo sintético; metadados do `TITLE`; seleção de amostras Kennard-Stone; despacha a leitura via `io_registry.py` |
| `io_registry.py` | *Registry* de leitores de dados: mapeia `cfg.modo` (`dx`/`csv`/`imagem`/`sintetico`) ao leitor correspondente |
| `dados_imagem.py` | Colorimetria digital (`modo="imagem"`, protótipo): extração de *features* RGB/HSV/Lab e textura opcional |
| `preprocessamento.py` | *Transformers* SNV/SavGol/MSC e `construir_preprocessador` |
| `classificadores.py` | DD-SIMCA, OPLS-DA |
| `figuras.py` | Camada de plotagem (todas as figuras do pipeline, incluindo `fig_merito_regressao`) |
| `validacao_estatistica.py` | BCa, CV-ANOVA, permutação, teste de Wold, validação cruzada manual |
| `hardware.py` | Detecção de RAM/CPU/disco, auto-ajuste de `Config`, guarda de RAM |
| `selecao_variaveis.py` | Etapa 4 completa: iPLS, sPLS-DA, SPA/APS, AG e figuras da etapa |
| `avaliacao_modelos.py` | PLS-DA, Auto-Benchmark, Monte Carlo CV, curvas DET, SHAP — modelos de comparação vêm de `model_registry.py` |
| `model_registry.py` | *Registry* de modelos de *benchmark*: fonte única da lista PLS-DA/SVM/RF/GBM/XGBoost |
| `predicao.py` | Predição em amostras novas a partir de um `.joblib` salvo — compartilhado entre aplicativo e CLI |
| `reports.py` | Geração de relatórios do aplicativo web (PDF/Word/Excel/LaTeX/PowerPoint) |
| `app_logic.py` | Lógica pura da interface web (progresso, formatação, coleta de configuração, leitura de artefatos, captura de log — `LogThreadSafe`), testável sem *Streamlit* |
| `cli_logic.py` | Lógica pura da CLI de terminal (truncamento, validação de faixas, contagem de arquivos), testável sem *Rich* |
| `resumo_parse.py` | *Parsing* puro do `resumo_modelo.txt`: `parse_metricas_modelo` e `parse_acuracia_por_classe` |
| `spectra_preview.py` | Carregamento/plotagem de amostra de espectros para prévia (abas Data e Preprocessing) |
| `app_tabs/` | Um módulo por aba do aplicativo web (`projeto`, `dados`, `preprocessamento`, `modelo`, `validacao`, `predicao`, `relatorios`, `sobre`) |

Os módulos acima vivem no pacote `src/guaraci/`. Interfaces de usuário:
`app_quimiometria.py` (web — fica na **raiz**, é o ponto de entrada do
*Streamlit*) e `guaraci/guaraci.py` (assistente de terminal — **único**
ponto de entrada interativo). Tema visual compartilhado: `guaraci_theme.py`,
`design_tokens.py`.

**CLI unificada:** `guaraci/cli_assistente.py` foi um assistente hierárquico
completo e independente — hoje é só um módulo de **dados/i18n
compartilhado** (rótulos, textos de ajuda, perfis, paletas, técnicas
analíticas) que `guaraci.py` consome; não tem mais `main()` nem menus
próprios e não deve ser executado diretamente.

---

## 8 Desenvolvimento

```bash
pytest tests/                 # suíte completa (inclui o teste fim a fim "slow")
pytest tests/ -m "not slow"   # só os testes rápidos
ruff check .                  # lint estático
```

- **Testes:** `test_pipeline_smoke.py` e `test_pipeline_core.py` (unidade),
  `test_modos_analise.py` (objetivo científico), `test_figuras_regressao.py`
  (regressão de figuras e contrato de não vazamento entre modos),
  `test_fachada_reexport.py` (protege o contrato de reexport da fachada
  contra regressões futuras).
- **Integração contínua (GitHub Actions):** *lint* (`ruff`) e testes com
  cobertura a cada envio (*push*) ou *pull request*. O *Dependabot* abre
  *pull requests* semanais de atualização de dependências.

---

## REFERÊNCIAS

ARAÚJO, M. C. U. et al. The successive projections algorithm for variable
selection in spectroscopic multicomponent analysis. **Chemometrics and
Intelligent Laboratory Systems**, v. 57, n. 2, p. 65-73, 2001.

ERIKSSON, L.; TRYGG, J.; WOLD, S. CV-ANOVA for significance testing of PLS
and OPLS models. **Journal of Chemometrics**, v. 22, n. 11-12, p. 594-600,
2008.

ERIKSSON, L. et al. **Multi- and megavariate data analysis**: principles
and applications. Umeå: Umetrics AB, 2006.

JAWORSKA, J.; NIKOLOVA-JELIAZKOVA, N.; ALDENBERG, T. QSAR applicability
domain estimation by projection of the training set in descriptor space: a
review. **SAR and QSAR in Environmental Research**, v. 16, n. 5, p. 445-466,
2005.

KENNARD, R. W.; STONE, L. A. Computer aided design of experiments.
**Technometrics**, v. 11, n. 1, p. 137-148, 1969.

LEARDI, R. Application of genetic algorithm-PLS for feature selection in
spectral data sets. **Journal of Chemometrics**, v. 14, n. 5-6, p. 643-655,
2000.

MARTENS, H.; MARTENS, M. Modified jack-knife estimation of parameter
uncertainty in bilinear modelling by partial least squares regression
(PLSR). **Food Quality and Preference**, v. 11, n. 1-2, p. 5-16, 2000.

MITCHELL, M. et al. Model cards for model reporting. In: CONFERENCE ON
FAIRNESS, ACCOUNTABILITY, AND TRANSPARENCY (FAT*), 2019, Atlanta.
**Proceedings [...]**. New York: ACM, 2019. p. 220-229.

VALDERRAMA, P.; BRAGA, J. W. B.; POPPI, R. J. Estado da arte de figuras de
mérito em calibração multivariada. **Química Nova**, v. 32, n. 5,
p. 1278-1287, 2009.

WOLD, S. Cross-validatory estimation of the number of components in factor
and principal components models. **Technometrics**, v. 20, n. 4,
p. 397-405, 1978.

---

*Última revisão do manual: prévia "O que será gerado" em tempo real na aba
Model (web), 8ª aba **Sobre** (identidade, licença, como citar), cabeçalho
com logo/versão/badges e aviso de modo demonstração no deploy público sem
`config.yaml` local.*
