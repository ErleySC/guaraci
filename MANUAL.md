# Manual do GUARACI — Plataforma de Quimiometria

> Manual de instruções das funcionalidades do projeto. Mantido atualizado a
> cada mudança relevante. Para instalação, citação e licença, ver `README.md`.

O **GUARACI** é uma plataforma de quimiometria multi-técnica para autenticação
e caracterização de matrizes complexas — óleos amazônicos por FT-NIR, e
qualquer dado espectral/tabular (Raman, UV-Vis, FTIR, cromatografia). Cobre
todo o fluxo científico: análise exploratória, classificação, autenticação,
quantificação e relatórios de publicação, com validação estatística rigorosa
e anti-vazamento de réplicas em cada etapa.

**Sumário**
1. [As três formas de usar](#1-as-três-formas-de-usar)
2. [Modos de análise (N1 / N2 / N3)](#2-modos-de-análise-n1--n2--n3)
3. [Fontes de dados de entrada](#3-fontes-de-dados-de-entrada)
4. [Funcionalidades científicas](#4-funcionalidades-científicas)
5. [Fluxo típico na interface web](#5-fluxo-típico-na-interface-web)
6. [Mapa dos módulos (para desenvolvedores)](#6-mapa-dos-módulos-para-desenvolvedores)
7. [Desenvolvimento](#7-desenvolvimento)

---

## 1. As três formas de usar

O código fica no pacote `guaraci` (em `src/`). Instale uma vez com
`pip install -e .` (disponibiliza o comando `guaraci`); sem instalar, use
`PYTHONPATH=src`.

| Forma | Comando | Para quem |
|---|---|---|
| **Web (Streamlit)** | `streamlit run app_quimiometria.py` | Uso visual, 7 abas guiadas. Demo público: <https://guaraci.streamlit.app> |
| **Assistente de terminal** | `guaraci` (ou `PYTHONPATH=src python -m guaraci.guaraci`) | Menu interativo colorido, sem editar código |
| **Pipeline direto** | `python -m guaraci.pipeline --rodar` | Execução automatizada a partir de `config.yaml` |

As três formas compartilham o mesmo motor (`pipeline.py`) e a mesma
configuração (`config.yaml` / classe `Config`) — não há divergência de
resultado entre elas.

---

## 2. Modos de análise (N1 / N2 / N3)

O **modo de análise** define o que o pipeline faz. Na interface aparecem com
nomes amigáveis; internamente são identificados como N1/N2/N3.

- **N1 — Classificação (por espécie).**
  Identifica a qual espécie/classe cada amostra pertence (ex.: 13–14 óleos
  amazônicos). Método: **PLS-DA** com GroupKFold anti-vazamento de réplicas
  (as réplicas T1/T2/T3 do mesmo ponto amostral nunca são separadas entre
  treino e validação).

- **N2 — Discriminação (puro vs. adulterado).**
  Autentica a pureza **por espécie** via **DD-SIMCA** one-class (T² e
  Q-resíduos com limites de aceitação específicos por classe).

- **N3 — Quantificação (% de adulterante).**
  Estima o teor de adulterante por **regressão PLS calibrada por espécie**
  (`pls_regressao_por_espécie`). Reporta também as **figuras de mérito
  analíticas** — LOD, LOQ, sensibilidade, sensibilidade analítica (γ) e
  seletividade, segundo Valderrama, Braga & Poppi (2009) — calculadas
  automaticamente a partir das réplicas físicas (T1/T2/T3) de cada espécie.
  Aparecem no console/log de execução logo após RMSEP e R² **e também são
  gravadas no `resumo_modelo.txt`** (bloco "Analytical Figures of Merit"),
  que a aba **Relatórios** do app captura — antes só saíam no console.
  O split calibração/validação dessa regressão aceita dois métodos via
  `divisao_cal_val` no `config.yaml` (hiperparâmetro avançado, **não**
  exposto no app/CLI — mesmo padrão de `ipls_n_intervalos`/`vip_threshold_sel`):
  `"aleatoria"` (padrão, `GroupShuffleSplit` group-aware) ou
  `"kennard_stone"` (Kennard & Stone, 1969 — cobertura maximamente
  representativa do espaço espectral em vez de aleatória; com réplicas
  físicas, colapsa cada grupo T1/T2/T3 num espectro médio antes de
  selecionar, preservando o mesmo invariante de nunca separar réplicas
  entre calibração e validação).

**Curadoria de figuras por tipo de análise:** o pipeline só gera gráficos
relevantes para o modo de análise escolhido, mesmo que um módulo tenha sido
ligado manualmente por engano:
- **DD-SIMCA** (autenticação de pureza) é um conceito de **N2**. Em **N1**
  (classificação por espécie), o toggle `ddsimca`/`executar_ddsimca` é
  **ignorado com aviso** — não agrega a um estudo de identificação de
  espécie. Em N2 ele é sempre ligado automaticamente (não precisa configurar).
- **Regressão PLS + figuras de mérito** (LOD/LOQ/SEN/SEL) só rodam em
  **N2/N3**; em N1 são puladas (a variação entre espécies domina o sinal de
  adulteração, tornando a regressão sem sentido).
- **OPLS-DA** não é específico de nenhum nível — no Guaraci ele discrimina
  **espécie** (mesmo alvo do PLS-DA, via LDA quando há >2 classes), então
  continua disponível como extra em qualquer nível.

**Conjunto padrão de fábrica (~7–9 figuras "core"), qualquer nível:**
PCA (scores), PLS-DA (scores), outliers T²/Q, matriz de confusão, ROC/AUC,
curva de seleção de LVs, VIP+Selectivity Ratio — mais bootstrap VIP e
avaliação em holdout quando os respectivos parâmetros estão ativos (padrão).
N2 soma a figura de aceitação DD-SIMCA; N3 soma a figura de regressão PLS.
Tudo o mais — OPLS-DA, Etapa 4 (seleção de variáveis), comparação de
pipelines de pré-processamento, HCA comparativo, teste de Wold, CV-ANOVA,
Auto-Benchmark, Monte Carlo CV, SHAP, figuras detalhadas (`figuras_detalhadas`)
— é **opt-in**: o usuário liga explicitamente quando quiser ir além do
conjunto padrão.

---

## 3. Fontes de dados de entrada

Configuráveis via `modo_entrada` (app, CLI ou `config.yaml`):

| Modo | Origem | Observação |
|---|---|---|
| `dx` | Espectros JCAMP-DX (FT-NIR/Raman/MIR) | Padrão; 1 subpasta por classe |
| `csv` | Tabela genérica (colunas espectrais + 1 coluna de classe) | Qualquer dado tabular |
| `imagem` | **Colorimetria digital (protótipo)** | Ver abaixo |
| `sintetico` | Dados simulados | Para testes/demonstração |

**Modo `imagem` (colorimetria digital, protótipo):** extrai estatísticas de
cor (média/desvio-padrão por canal em RGB, HSV e Lab — 18 variáveis) de cada
foto, e opcionalmente textura (GLCM, requer `pip install scikit-image`).
Mesma convenção de pastas do modo `dx` (1 subpasta por classe). A partir da
extração, toda a maquinaria quimiométrica (PCA, PLS-DA, DD-SIMCA, seleção de
variáveis, figuras de mérito) funciona sem alteração — cada estatística de
cor vira uma "variável", exatamente como um comprimento de onda.

⚠️ **Duas configurações obrigatórias ao usar `modo="imagem"`:**
1. `pre_processamento` deve ser `autoscaling` ou `mc` — **nunca** um preset
   com Savitzky-Golay (`msc_sg_mc`/`snv_sg_mc`), que pressupõe um sinal
   espectral contínuo, sem sentido para um vetor curto de estatísticas de
   cor discretas.
2. `faixa_min_cm`/`faixa_max_cm` devem cobrir o intervalo `0`–`n_features-1`
   (ex.: `-1` a `100`) — o eixo de variáveis aqui é um índice simbólico, não
   um número de onda real, e os padrões de fábrica (4000–10000) descartariam
   todas as variáveis.

Sem caso de uso específico ainda amarrado (protótipo genérico) — cabe ao
usuário definir a região de interesse via `imagem_recorte` (recorte
retangular relativo, `config.yaml`) antes da extração.

---

## 4. Funcionalidades científicas

**Pré-processamento espectral** (dentro do `Pipeline` do scikit-learn, sem
vazamento entre folds de validação cruzada): SNV, MSC, Savitzky-Golay
(suavização ou derivada), mean-centering, autoscaling. Presets prontos:
`msc_sg_mc` (melhor desempenho no dataset de referência), `snv_sg_mc`, `mc`,
`autoscaling`, `custom`.

**Análise exploratória:** PCA (scores e loadings), HCA (dendrograma de Ward
sobre componentes principais).

**Classificação e discriminação:** PLS-DA, OPLS-DA (com S-Plot), DD-SIMCA
(com Cooman's Plot).

**Seleção de variáveis (Etapa 4)** — sempre executados: iPLS (por
intervalos), VIP ≥ 1, Selectivity Ratio (top 20%), sPLS-DA esparso.
Opcionais (mais lentos, ligar quando quiser comparar mais a fundo):
- **SPA/APS** — Algoritmo das Projeções Sucessivas (Araújo et al., 2001):
  monta cadeias de variáveis com baixa colinearidade entre si.
- **AG** — Algoritmo Genético (GA-PLS): evolui uma população de
  subconjuntos de variáveis por seleção, cruzamento e mutação, usando
  acurácia balanceada via validação cruzada como aptidão.

Todos os métodos — sempre-ligados e opcionais — são avaliados sob o **mesmo
esquema de validação cruzada group-aware**, permitindo comparação direta
numa única tabela e figura.

**Validação estatística:** teste de permutação, teste de Wold (R²Y/Q²Y),
CV-ANOVA (Eriksson), bootstrap BCa (intervalo de confiança da acurácia),
holdout externo group-aware, Monte Carlo CV (IC95%).

**Teste de incerteza de Martens** (Martens & Martens, 2000) — opt-in via
`teste_martens` (app e CLI, aba/menu Validação): jackknifing *group-aware*
dos coeficientes de regressão PLS (reaproveita a mesma CV de seleção de
LVs). Produz um **teste de hipótese formal** (estatística t + p-valor) de
significância por variável — mais rigoroso que VIP/Selectivity Ratio, que
são medidas de *magnitude* sem p-valor associado. Em modelos multiclasse, o
resultado por variável é o máximo |t| entre as classes (significativa se
discrimina pelo menos uma). Gera `dados/teste_martens.csv`
(comprimento de onda, t, p, significativo) e um resumo (nº de variáveis
significativas) no `resumo_modelo.txt`/`model_card.md`.

**DModX / DModY** (nomenclatura padrão SIMCA-P/Unscrambler, Eriksson et al.
2006) — sempre calculados, sem toggle: são o **mesmo** T²/Q-resíduo e
resíduo de predição já usados nas figuras (`fig3_outliers`/`fig7_pls_regressao`),
apenas **normalizados e nomeados** na escala/convenção que usuários vindos
dessas ferramentas comerciais já reconhecem (DModX ≈ 1 = resíduo típico;
acima do limite crítico = fora do modelo). Não geram figura nova (seria
redundante com as já existentes) — aparecem como resumo (limite crítico +
número de amostras fora) no console, `resumo_modelo.txt` e `model_card.md`.
DModX é sempre reportado (classificação); DModY aparece quando há
regressão (N2/N3).

**Comparação de modelos (Auto-Benchmark):** PLS-DA vs. SVM RBF vs. Random
Forest vs. Gradient Boosting vs. XGBoost, sob a mesma CV group-aware. Curvas
DET e interpretabilidade via **SHAP** (TreeExplainer).

**Auto-Benchmark de regressão (N2/N3):** PLS-R (o modelo já calibrado por
`pls_regressao_por_espécie`, reaproveitado sem refit) vs. Ridge, Lasso,
Elastic Net, SVR (RBF) e Random Forest Regressor — um modelo **por espécie**
(mesma arquitetura da quantificação, calibração separada evita que a
variação inter-espécies confunda o sinal de adulteração), com o **mesmo
split calibração/validação** (determinístico, mesmo `seed`/`divisao_cal_val`
do PLS-R) e o mesmo pré-processamento, para uma comparação honesta
apples-to-apples. Opt-in via `benchmark_regressao` (app e CLI, categoria
Avançado — mesmo padrão do Auto-Benchmark de classificação): gera
`benchmark_regressao.csv` (RMSEP/R² pooled + por espécie) e
`fig_benchmark_regressores.png` (boxplot de RMSEP por espécie, menor =
melhor).

**Predição em amostras novas (app e CLI):** aplica um modelo já treinado
(`modelo_plsda.joblib`, salvo automaticamente ao final de cada execução) a
espectros novos, sem rodar o pipeline inteiro de novo. Entrada: CSV com
colunas = número de onda (sem coluna de classe). Saída: classe predita,
confiança (%) e **dois diagnósticos complementares** de confiabilidade:
- **Ajuste ao modelo PLS-DA** (colunas `T2`/`Q`/`aceito`) — o quanto a
  amostra se afasta do que o modelo de classificação capturou.
- **Domínio de Aplicabilidade** (colunas `AD_*`, Jaworska et al. 2005) —
  o quanto a amostra é um espectro atípico frente ao dataset de calibração
  em geral, via T²/Q num PCA exploratório independente da classe. Reaproveita
  `chemometric_stats.dominio_aplicabilidade_amostras_novas`; só aparece se o
  modelo foi salvo por uma versão do pipeline que exporta esses artefatos
  (retrocompatível — modelos antigos continuam predizendo normalmente, só
  sem essas colunas extras).

Disponível em dois lugares, com a **mesma lógica científica** (`predicao.py`,
compartilhado, zero duplicação):
- **App web** — aba 🔮 *Prediction*.
- **CLI** — menu principal, tecla `[B]` *Predição em Lote*: pede o caminho
  do modelo, do CSV de espectros novos e do CSV de saída (Enter = mesmo
  nome do CSV de entrada + `_predicao.csv`), e imprime um resumo por classe
  + os dois diagnósticos acima. Útil para automação/scripts e integração
  com LIMS sem precisar do navegador.

> **⚠️ Segurança — upload de modelo em deploy público.** Um arquivo `.joblib`
> é um *pickle*: carregá-lo **executa código** contido no arquivo. Em uso local
> (sua máquina) isso é seguro, pois o modelo é seu. Mas num **demo hospedado
> público**, aceitar upload de `.joblib` de qualquer visitante é um vetor de
> execução remota de código (RCE). Por isso, no deploy público, defina a
> variável de ambiente **`GUARACI_DISABLE_MODEL_UPLOAD=1`**: o app esconde o
> uploader de modelo e passa a aceitar apenas **caminho local** (controlado pelo
> operador). O upload de CSV de espectros permanece liberado (dado inerte). Sem
> a variável (padrão), o upload fica habilitado com um aviso — apropriado para
> uso local single-user.

**Figuras:** conjunto essencial por padrão (~8 figuras) com opção de
figuras detalhadas adicionais (`figuras_detalhadas=True`). Formatos
PNG/PDF/SVG, DPI configurável.

**Relatórios:** PDF, Word (`.docx`), Excel (5 abas), LaTeX e PowerPoint, com
capa de projeto (nome, autor, instituição, objetivo — o "tipo de estudo" é
derivado automaticamente do modo de análise escolhido).

**Model Card (`model_card.md`):** documento de 1 página gerado
automaticamente ao final de toda execução, no padrão *Model Cards for Model
Reporting* (Mitchell et al., 2019, FAT\*'19) — o mesmo formato usado por
plataformas de ML-ops (ex.: Hugging Face Hub) para trilha de auditoria e
transparência. Seções: detalhes do modelo (versão, algoritmo,
pré-processamento), uso pretendido (e fora-de-escopo, específico para
N1/N2/N3), fatores relevantes (classes, group-aware CV), métricas de
desempenho, dados de avaliação/treino (integridade, tamanho), análises
quantitativas por classe, considerações éticas e ressalvas metodológicas
(as mesmas do `resumo_modelo.txt`, fonte única). Em N2/N3, ganha um
addendum com as figuras de mérito de regressão. Aparece na aba
**Relatórios** do app (preview + download `.md` próprio) e no `logs/` de
toda execução (CLI e app).

---

## 5. Fluxo típico na interface web

1. **Projeto** — preencha nome, autor, instituição e objetivo (campos
   descritivos, entram na capa dos relatórios).
2. **Dados** — faça upload de um CSV ou aponte a pasta de espectros `.dx`
   (uma subpasta por classe).
3. **Pré-processamento** — escolha o preset e confira a visualização de
   antes/depois.
4. **Modelo** — escolha o modo de análise, ajuste os parâmetros (variáveis
   latentes, holdout, validação, módulos extras, figuras) e clique em
   **▶️ Run pipeline**.
5. **Validação / Predição / Relatórios** — inspecione as métricas por
   classe e as figuras geradas, e baixe os relatórios e o ZIP de resultados.

Tema claro/escuro: menu ⋮ → *Settings* → *Theme* (segue a preferência do
sistema operacional por padrão).

---

## 6. Mapa dos módulos (para desenvolvedores)

Desde a Fase H, o motor do pipeline está modularizado por responsabilidade.
`pipeline.py` funciona como **fachada**: reexporta todos os símbolos
públicos dos módulos abaixo, então `import pipeline as pq; pq.X` continua
funcionando sem alteração, não importa em qual arquivo `X` esteja
implementado de fato.

| Módulo | Responsabilidade |
|---|---|
| `pipeline.py` | `Config`, `_CONFIG_SPEC`, orquestrador `executar()`, IO de configuração, CLI embutido e fachada de reexport |
| `chemometric_stats.py` | VIP, Selectivity Ratio, **teste de incerteza de Martens** (`teste_incerteza_martens`, jackknifing), Hotelling T², Q-resíduos, variância explicada, figuras de mérito (LOD/LOQ/SEN/SEL), **domínio de aplicabilidade** (`dominio_aplicabilidade` + variantes `_treino`/`_amostras_novas`, T²+Q — usado por `predicao.py` na predição em lote) |
| `paleta_cores.py` | Paleta e marcadores de máxima distintividade por classe |
| `dados_io.py` | Parsing JCAMP-DX/ASDF, CSV e modo sintético; metadados do `TITLE`; **seleção de amostras Kennard-Stone** (`kennard_stone`, `kennard_stone_split`); despacha a leitura via `io_registry.py` |
| `io_registry.py` | **Registry de leitores de dados** (item 20 da auditoria): mapeia `cfg.modo` ("dx"/"csv"/"imagem"/"sintetico") ao leitor correspondente. Novo formato de entrada = `registrar_leitor("modo", funcao)`, sem editar `dados_io.carregar_dados()` |
| `dados_imagem.py` | Colorimetria digital (`modo="imagem"`, protótipo): extração de features RGB/HSV/Lab + textura opcional |
| `preprocessamento.py` | Transformers SNV/SavGol/MSC + `construir_preprocessador` |
| `classificadores.py` | DD-SIMCA, OPLS-DA |
| `figuras.py` | Camada de plotagem (todas as figuras do pipeline) |
| `validacao_estatistica.py` | BCa, CV-ANOVA, permutação, teste de Wold, CV manual |
| `hardware.py` | Detecção de RAM/CPU/disco, auto-ajuste de `Config`, guarda de RAM |
| `selecao_variaveis.py` | Etapa 4 completa: iPLS, sPLS-DA, SPA/APS, AG + figuras da etapa |
| `avaliacao_modelos.py` | PLS-DA, Auto-Benchmark, Monte Carlo CV, curvas DET, SHAP — modelos de comparação vêm de `model_registry.py` |
| `model_registry.py` | **Registry de modelos de benchmark** (item 20 da auditoria): fonte única da lista PLS-DA/SVM/RF/GBM/XGBoost, usada por `benchmark_classificadores()` **e** `monte_carlo_cv()` (antes duplicada nas duas funções e divergente) |
| `predicao.py` | Predição em amostras novas a partir de um `.joblib` salvo — compartilhado entre app (aba Prediction) e CLI (menu `[B]`) |
| `reports.py` | Geração de relatórios do app web (PDF/Word/Excel/LaTeX/PowerPoint), extraída de `app_quimiometria.py` |
| `app_logic.py` | Lógica pura da UI web (progresso, formatação, coleta de config, leitura de artefatos), testável sem Streamlit |
| `cli_logic.py` | Lógica pura da CLI de terminal (truncamento, validação de faixas, contagem de arquivos), testável sem Rich |
| `resumo_parse.py` | **Parsing puro do resumo_modelo.txt** (item 19): `parse_metricas_modelo` (12 métricas) e `parse_acuracia_por_classe`, antes duplicados nos 5 geradores de relatório e na aba Validation |
| `spectra_preview.py` | Carregamento/plotagem de amostra de espectros para prévia (abas Data e Preprocessing) |
| `app_tabs/` | Um módulo por aba do app web (`projeto`, `dados`, `preprocessamento`, `modelo`, `validacao`, `predicao`, `relatorios`) — item 18: cada aba é uma função `render(...)`, `app_quimiometria.py` só orquestra |
| `io_registry.py` | **Registry de leitores de dados** (item 20): mapeia `cfg.modo` ao leitor; novo formato = `registrar_leitor(...)` sem editar `dados_io.carregar_dados()` |

Os módulos acima vivem no pacote `src/guaraci/`. Interfaces de usuário:
`app_quimiometria.py` (web — fica na **raiz**, é o entry point do Streamlit)
e `guaraci/guaraci.py` (assistente de terminal — **único** ponto de entrada
interativo; ver nota abaixo). Tema visual compartilhado: `guaraci_theme.py`,
`design_tokens.py`.

**CLI unificada (item 16 da auditoria):** `guaraci/cli_assistente.py` foi um
assistente hierárquico completo e independente — hoje é só um módulo de
**dados/i18n compartilhado** (rótulos, textos de ajuda, perfis, paletas,
técnicas analíticas) que `guaraci.py` consome; não tem mais `main()` nem
menus próprios e não deve ser executado diretamente.

---

## 7. Desenvolvimento

```bash
pytest tests/                 # suíte completa (inclui o teste end-to-end 'slow')
pytest tests/ -m "not slow"   # só os testes rápidos
ruff check .                  # lint estático
```

- **Testes:** `test_pipeline_smoke.py` e `test_pipeline_core.py` (unidade),
  `test_figuras_regressao.py` (regressão de figuras), `test_fachada_reexport.py`
  (protege o contrato de reexport da fachada contra regressões futuras).
- **CI (GitHub Actions):** lint (`ruff`) e testes com cobertura a cada push
  ou pull request. O Dependabot abre PRs semanais de atualização de
  dependências.

---

*Última revisão do manual: modo de entrada `imagem` (colorimetria digital,
protótipo), seleção de variáveis SPA/APS e AG (Algoritmo Genético), figuras
de mérito analíticas (N3).*
