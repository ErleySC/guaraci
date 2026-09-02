# MANUAL DO GUARACI — PLATAFORMA DE QUIMIOMETRIA

> Manual de instruções das funcionalidades do projeto. Mantido atualizado a
> cada mudança relevante de funcionalidade, interface ou fluxo. Para
> instalação, citação e licença, ver `README.md`.

O **GUARACI** é uma plataforma de quimiometria multitécnica para autenticação
e caracterização de matrizes complexas, a partir de qualquer dado espectral
ou tabular (FT-NIR, NIR, MIR, Raman, UV-Vis, cromatografia). O que é
específico de cada matriz vive num **perfil** (`guaraci/perfis_matriz/*.yaml`),
nunca no código-fonte — ver a seção "Perfis de matriz".
Cobre todo o fluxo científico — análise exploratória, classificação,
autenticação, quantificação e relatórios de publicação — com validação
estatística rigorosa e proteção contra vazamento de réplicas em cada etapa.

## SUMÁRIO

1. [As três formas de usar](#1-as-três-formas-de-usar)
2. [Modos de análise e objetivo científico](#2-modos-de-análise-e-objetivo-científico)
3. [Estrutura de saída dos resultados](#3-estrutura-de-saída-dos-resultados)
4. [Fontes de dados de entrada](#4-fontes-de-dados-de-entrada)
4b. [Perfis de matriz e mode cego](#4b-perfis-de-matriz-e-mode-cego)
5. [Funcionalidades científicas](#5-funcionalidades-científicas)
6. [Fluxo típico na interface web](#6-fluxo-típico-na-interface-web)
7. [Mapa dos módulos (para desenvolvedores)](#7-mapa-dos-módulos-para-desenvolvedores)
8. [Desenvolvimento](#8-desenvolvimento)
9. [Limitações conhecidas](#9-limitações-conhecidas)
10. [Referências](#referências)

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

### 1.1 Checklist pré-execução (terminal)

Antes de confirmar a execução (`[R]`), o assistente de terminal mostra um
checklist com uma varredura barata dos arquivos `.dx` (~0,3 s para milhares
de arquivos — só lê o cabeçalho JCAMP-DX, não os 8192 pontos espectrais),
que antecipa dois efeitos que **mudam o N da análise** e antes só apareciam
no meio do log, depois de a execução já ter começado:

- **Descarte por faixa espectral incompatível** — datasets reais podem
  misturar janelas de aquisição (ex.: NIR completo `[0, 15797]` vs. faixa
  estreita `[300, 4000]`); o motor mantém só a faixa dominante e descarta o
  resto. O checklist mostra quantos espectros serão descartados e de qual
  espécie, antes de rodar.
- **Amostras sem `mae_id`** — entram na análise **sem** proteção contra
  vazamento de réplica (o diferencial central do projeto). O checklist
  avisa quantas.

Também mostra uma **estimativa de tempo** (faixa, não valor exato — ex.:
`~6-15 min`), calibrada em medição real, não em regra de bolso. Quando
`n_jobs_permutation=1` e há muitas permutações, o checklist sugere
explicitamente subir esse valor: o resultado é idêntico (mesmo seed, mesma
partição), só o tempo de execução muda.

### 1.2 Painel de acompanhamento em tempo real (terminal)

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

O **mode de análise** define o que o pipeline faz. Na interface aparecem com
nomes amigáveis; internamente são identificados como N1/N2/N3.

### 2.1 Os três níveis (N1 / N2 / N3)

> `N1`/`N2`/`N3` é o código interno de `cfg.level` (usado em `config.yaml`,
> nunca exibido como termo primário na CLI/web). No nome da pasta de saída
> (seção 3), cada nível vira um slug amigável: `N1`→`PorEspecie`,
> `N2`→`Autenticacao`, `N3`→`Quantificacao` (corrigido em 2026-07-13 —
> versões anteriores gravavam `N1`/`N2`/`N3` cru no nome do diretório).

- **N1 — Classificação (por espécie).**
  Identifica a qual classe cada amostra pertence (espécie, variedade, tipo —
  o nome vem do perfil de matriz). Método: **PLS-DA** com `GroupKFold` anti-vazamento
  de réplicas (as réplicas T1/T2/T3 do mesmo ponto amostral nunca são
  separadas entre treino e validação).

- **N2 — Discriminação (puro vs. adulterado).**
  Autentica a pureza **por espécie** via **DD-SIMCA** *one-class* (T² e
  Q-resíduos com limites de aceitação específicos por classe).
  A **sensibilidade** (fração de puros aceitos) é estimada por
  **leave-one-group-out (LOGO) por réplica `mae_id`**, não por
  re-substituição: para cada grupo de réplica, o modelo é retreinado nos
  demais e testado no grupo retido. Toda figura, tabela e relatório mostra o
  **número de grupos** ao lado da sensibilidade; com **menos de 10 grupos** o
  valor vem acompanhado de aviso de incerteza (exploratório) e, com **apenas
  1 grupo** de puros, é reportado como `n/a (não validado)` — sem replicação
  independente não há validação possível. A especificidade (rejeição de
  adulterados) permanece uma medida externa legítima.

- **N3 — Quantificação (% de adulterante).**
  Estima o teor de adulterante por **regressão PLS calibrada por espécie**
  (`pls_regression_by_species`). Reporta também as **figuras de mérito
  analíticas** — LOD, LOQ, sensibilidade, sensibilidade analítica (γ) e
  seletividade, segundo Valderrama, Braga e Poppi (2009) — calculadas
  automaticamente a partir das réplicas físicas (T1/T2/T3) de cada espécie.
  **LOD/LOQ como intervalo, não ponto (Bloco 12):** o ruído instrumental
  (`delta_x`) usado no LOD/LOQ é ele próprio uma *estimativa* com graus de
  liberdade limitados (poucas réplicas físicas por espécie) — reportar só o
  valor pontual esconde essa incerteza. Cada LOD/LOQ vem acompanhado de um
  **intervalo de confiança** (95% por padrão, via qui-quadrado sobre a
  variância pooled — conceito de reportar o limite de detecção como
  intervalo, não número único, segue Allegrini e Olivieri, 2014, *Anal.
  Chem.* 86(15):7858-7866) **e** da **faixa e desvio-padrão do próprio
  conjunto de validação** ao lado — um LOD pequeno não diz nada sozinho se
  o teor de validação variou pouco. Aparece no console, no
  `resumo_modelo.txt` e no `model_card.md` sempre junto do LOD/LOQ pontual,
  nunca isolado. Além do bloco de texto, essas figuras de mérito têm uma
  **representação gráfica dedicada** (`figS3_merito_regressao.png`, ver
  seção 5) — LOD/LOQ e seletividade por espécie lado a lado, com indicação
  explícita ("n/a") para espécies sem réplicas físicas suficientes para
  estimar o ruído instrumental. **RPD/RER** (`chemometric_stats.rpd_rer`)
  já estavam implementados antes deste bloco — reverificado por leitura
  direta do código nesta revisão, não presumido da memória: razão SD/SEP e
  amplitude/SEP, com as faixas de interpretação de Williams (2014), já
  integrados ao pooled do `pls_regression_by_species`. O *split*
  calibração/validação dessa regressão aceita dois
  métodos via `cal_val_split` no `config.yaml` (hiperparâmetro avançado,
  **não** exposto no aplicativo/CLI — mesmo padrão de
  `ipls_n_intervalos`/`vip_threshold_sel`): `"aleatoria"` (padrão,
  `GroupShuffleSplit` *group-aware*) ou `"kennard_stone"` (Kennard e Stone,
  1969 — cobertura maximamente representativa do espaço espectral em vez de
  aleatória; com réplicas físicas, colapsa cada grupo T1/T2/T3 num espectro
  médio antes de selecionar, preservando o mesmo invariante de nunca separar
  réplicas entre calibração e validação).
  Gera ainda o **mapa de calor espécie × adulterante**
  (`fig_heatmap_especie_adulterante.png`): um **R²cv** (R² em validação
  cruzada *group-aware* — mede o acerto do teor em amostras não vistas no
  treino) por combinação de espécie e adulterante. Existe porque a regressão
  que *junta* os adulterantes de uma espécie mascara que alguns adulterantes
  simplesmente **não são quantificáveis** (o sinal afoga no ruído). Cada
  célula **abaixo do limiar de aceite (R²cv = 0,70)** aparece **hachurada e em
  negrito** (nunca some), célula sem dados suficientes vira cinza com "n/a", e
  o título + o `resumo_modelo.txt` trazem o **contador de falhas** (ex.: "3/9
  combinações abaixo de R²cv = 0,70") — para que uma quantificação que só
  funciona em parte das combinações não seja lida como sucesso geral. O
  adulterante de cada amostra é derivado do `mae_id`. *(No mode sintético, ative
  com `synthetic_adulterants` no `config.yaml`, ex.: `["S","M","A"]`.)*

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
| **Exploratório** | PCA (*scores*), HCA (dendrograma), *loadings* PCA, *biplot* PCA (scores + loadings sobrepostos), efeito do pré-processamento |
| **Classificação** | PLS-DA (*scores*), matriz de confusão, ROC/AUC, VIP, seleção de LVs, Selectivity Ratio, DD-SIMCA, OPLS-DA, Etapa 4, teste de Wold, holdout, teste de Martens, Auto-Benchmark, Monte Carlo CV, SHAP |
| **Quantificação** | Regressão PLS + figuras de mérito analíticas (LOD/LOQ/SEN/SEL) |

Figuras de **contexto geral** — **espectros médios por classe** (banda =
±1 desvio-padrão, dado bruto antes de qualquer modelagem), PCA de *scores*
(visão geral) e o painel de *outliers* T²/Q — aparecem em **qualquer**
objetivo, pois oferecem contexto químico/diagnóstico válido
independentemente do propósito específico da corrida (a primeira nem
depende de um modelo ajustado — é só o dado bruto agrupado por classe).

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
  (classificação por espécie), o toggle `ddsimca`/`run_ddsimca` é
  **ignorado com aviso** — não agrega a um estudo de identificação de
  espécie. Em N2 ele é sempre ligado automaticamente (não precisa
  configurar).
- **OPLS-DA** não é específico de nenhum nível — no Guaraci ele discrimina
  **espécie** (mesmo alvo do PLS-DA, via PLS2 quando há mais de duas
  classes), então continua disponível como extra no objetivo Classificação.

**Conjunto padrão de fábrica (~8 a 10 figuras "*core*"), qualquer nível:**
espectros médios por classe, PCA (*scores*), PLS-DA (*scores*), *outliers*
T²/Q, matriz de confusão, ROC/AUC, curva de seleção de LVs, VIP +
Selectivity Ratio — mais *bootstrap* VIP e avaliação em *holdout* quando os
respectivos parâmetros estão ativos (padrão). N2 soma a figura de aceitação
DD-SIMCA; N3 soma a figura de regressão PLS e a figura de mérito analítica
dedicada. Tudo o mais — OPLS-DA, Etapa 4 (seleção de variáveis), *biplot*
PCA, comparação de *pipelines* de pré-processamento, HCA comparativo, teste
de Wold, CV-ANOVA, Auto-Benchmark, Monte Carlo CV, SHAP, figuras detalhadas
(`figuras_detalhadas`) — é *opt-in*: o usuário liga explicitamente quando
quiser ir além do conjunto padrão.

---

## 3 Estrutura de saída dos resultados

Cada execução grava seus resultados em uma hierarquia de pastas que separa
**amostra/conjunto de dados**, **objetivo científico** (seção 2.2) e o
conteúdo por categoria:

```text
<output_root_folder>/
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
  automaticamente do mode de entrada (nome do arquivo CSV, nome da pasta de
  espectros, ou `"sintetico"` para dados de teste).
- **`<Modo>`** — rótulo amigável do objetivo científico resolvido:
  `Exploratorio`, `Classificacao` ou `Quantificacao`.
- **`<Execução>`** — identificador único da corrida (nível, pré-processamento
  e data/hora), no formato `PLSDA_OE_<slug-do-nivel>_<preproc>_<AAAAMMDD_HHMMSS>`.
  `<slug-do-nivel>` é o nome amigável do nível (`PorEspecie`/`Autenticacao`/
  `Quantificacao` — corrigido em 2026-07-13; versões anteriores usavam
  `N1`/`N2`/`N3` cru aqui).
- **`Graficos/`** — todas as figuras (`.png`/`.pdf`/`.svg`, conforme
  `output_format`), incluindo subpastas de figuras detalhadas (por exemplo,
  `ddsimca/`).
- **`Tabelas/`** — dados tabulares em CSV (identificadores de amostra,
  metadados, teste de Martens, comparação de *pipelines*, benchmarks).
- **`Relatorios/`** — `resumo_modelo.txt` e `model_card.md`.
- **`Modelos/`** — modelo final exportado (`modelo_plsda.joblib`).

Exemplo real (execução de Classificação com `tag="oleos_essenciais"`):

```text
resultados_tcc/oleos_essenciais/Classificacao/PLSDA_OE_Autenticacao_MSC-SG1-MC_20260705_222028/
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
| `imagem` | Colorimetria digital (protótipo só sem garantia de agrupamento — ver adiante) | Ver adiante |
| `hsi` | Imageamento hiperespectral (protótipo "mínimo viável" — ver adiante) | Distinto de `imagem`: por pixel, não por foto |
| `sintetico` | Dados simulados | Para testes/demonstração |

**Modo `imagem` (colorimetria digital):** extrai estatísticas de
cor (média/desvio-padrão por canal em RGB, HSV e Lab — 18 variáveis) de cada
fotografia e, opcionalmente, textura (GLCM, requer `pip install
scikit-image`). Mesma convenção de pastas do mode `dx` (uma subpasta por
classe). A partir da extração, toda a maquinaria quimiométrica (PCA, PLS-DA,
DD-SIMCA, seleção de variáveis, figuras de mérito) funciona sem alteração —
cada estatística de cor vira uma "variável", exatamente como um comprimento
de onda.

**Três níveis de garantia de agrupamento (Bloco 8, 2026-08-25).** Ao
contrário do mode `dx` (que sempre tem `mae_id` real, extraído do
metadado `##TITLE=`), o mode `imagem` detecta automaticamente qual fonte
de agrupamento por amostra física está disponível, nesta ordem de
prioridade:

1. **`high`** — subpasta por amostra física: `Classe/Amostra/*.jpg`, um
   nível extra opcional dentro da subpasta de classe. Cada subpasta de
   amostra vira um grupo; as fotos dentro dela são réplicas do mesmo
   grupo. Sem parsing de nome, sem ambiguidade. Detectado automaticamente
   quando **toda** subpasta de classe contém só subpastas (nunca arquivo
   solto).
2. **`medium`** — CSV de associação manual (`amostras.csv` na raiz da
   pasta de dados, colunas `arquivo,id_amostra`, caminho relativo à raiz).
   Usado quando o nível `high` não está presente. **Toda** imagem
   carregada precisa aparecer no CSV — cobertura parcial falha com
   mensagem explícita listando os arquivos faltantes, nunca processa
   parcialmente em silêncio.
3. **`none`** — nem subpasta por amostra nem CSV presentes: o sistema
   aceita processar mesmo assim (uso direto, "só jogar as fotos e
   rodar"), mas cai em `StratifiedKFold` (sem proteção contra vazamento
   entre fotos da mesma amostra) e declara isso explicitamente em três
   lugares — log da execução, `model_card.md` e o manifesto do modelo
   (`*.manifest.json`) — nunca só em comentário/docstring interno. Os
   geradores de relatório (PDF/Word/LaTeX) carimbam **"PROTOTYPE OUTPUT —
   NO GROUPING GUARANTEE"** só neste nível.

EXIF **não** é usado como fonte de agrupamento — avaliado e descartado:
recompressão/edição de apps de galeria/mensageria apaga o metadado na
prática, e mesmo quando presente, fotos tiradas numa janela de tempo curta
não garantem "mesma amostra física".

Perfis de técnica de aquisição (`bancada`/`celular`/`scanner`, em
`guaraci/perfis_matriz/`) declaram resolução esperada, formatos aceitos e
o nível de garantia **tipicamente** alcançável por aquele fluxo de
trabalho — informativo, nunca restritivo: o nível real de cada execução é
sempre decidido pelos dados fornecidos.

Exemplo mínimo dos três modos de uso:

```
# Nível "high" — pasta organizada por amostra física
dados/Puro/amostra01/foto1.jpg
dados/Puro/amostra01/foto2.jpg
dados/Puro/amostra02/foto1.jpg
dados/Adulterado/amostra01/foto1.jpg

# Nível "medium" — pasta flat + CSV de associação na raiz
dados/Puro/foto1.jpg
dados/Puro/foto2.jpg
dados/amostras.csv        # arquivo,id_amostra
                           # Puro/foto1.jpg,S1
                           # Puro/foto2.jpg,S1

# Nível "none" — pasta flat, sem CSV (uso direto, sem garantia)
dados/Puro/foto1.jpg
dados/Puro/foto2.jpg
```

**Duas configurações obrigatórias ao usar `mode="imagem"`:**
1. `pre_processamento` deve ser `autoscaling` ou `mc` — **nunca** um preset
   com Savitzky-Golay (`msc_sg_mc`/`snv_sg_mc`), que pressupõe um sinal
   espectral contínuo, sem sentido para um vetor curto de estatísticas de
   cor discretas.
2. `faixa_min_cm`/`faixa_max_cm` devem cobrir o intervalo `0` a
   `n_features-1` (por exemplo, `-1` a `100`) — o eixo de variáveis aqui é
   um índice simbólico, não um número de onda real, e os padrões de fábrica
   (4000–10000) descartariam todas as variáveis.

Sem caso de uso específico ainda amarrado (protótipo genérico) — cabe ao
usuário definir a região de interesse via `image_crop` (recorte
retangular relativo, `config.yaml`) antes da extração.

**Modo `hsi` (imageamento hiperespectral, protótipo "mínimo viável"):**
DISTINTO do mode `imagem` acima — opera **por pixel** de um cubo
hiperespectral (formato ENVI, par `.hdr`+`.bin`), não por foto inteira.
Fluxo: quality gate (saturação, SNR, fração de pixels válidos) →
segmentação objeto/fundo (PCA+Otsu) → extração dos pixels da ROI, cada
um marcado com o `group_id` do objeto físico de origem (mesmo conceito
de `mae_id`, nunca dois pixels do mesmo objeto em lados diferentes de
um split) → PLS-DA por pixel (reaproveita `PLSDAClassifier`) com
agregação por objeto (classe majoritária + heterogeneidade) → mapa de
classificação espacial → explicabilidade cruzada (VIP × tabela de
atribuição química) → validação externa por partição nativa de dia de
medição.

Acessível pela tecla **`[X]`** do menu principal da CLI
(`hsi_dataset_folder` aponta para a pasta com `manifest.json` + os
arquivos ENVI — ver `scripts/download_datasets/baixar_deephs_kaki.py`
para obter o dataset público usado na validação). Orquestrado por
`hsi_pipeline.run_hsi_pipeline`, não por `pipeline.executar()` — a forma
de dado (por pixel, agregação por objeto) é fundamentalmente diferente
da matriz amostras×variáveis que os demais modes compartilham.

Validado com o dataset público DeepHS Fruit (Kaki/câmera VIS, Varga,
Makowski & Zell, IJCNN 2021) — desempenho ainda modesto (desbalanceamento
severo de classes no dataset), números honestos em
`docs/VALIDACAO_PUBLICA.md` §7 e `docs/PROGRESSO.md`. Não usar para
resultado publicável sem validação adicional em dado próprio.

---

## 4b Perfis de matriz e mode cego

### 4b.1 Perfil de matriz — trocar de matriz sem tocar em código

Tudo que é propriedade da **matriz**, e não do **método**, vive num arquivo
de perfil (`src/guaraci/perfis_matriz/*.yaml`):

| Campo | O que define |
|---|---|
| `unidade_eixo` | `cm-1` ou `nm` — aparece nos rótulos e no model card |
| `eixo_min` / `eixo_max` | faixa espectral usada; só se aplica se você não definiu a sua |
| `default_preprocessing` | preset inicial adequado à matriz |
| `vocabulario` | como a saída chama as coisas: `classe`, `matriz`, `alvo`, `conforme`, `nao_conforme` |
| `faixa_trabalho` | faixa do analito coberta pela calibração; usada para marcar extrapolação |
| `referencia` | de onde vieram esses valores (nunca inventar) |

```bash
guaraci perfis                 # lista os perfis embutidos
guaraci --perfil=milho_nir     # ou o caminho de um YAML seu
```

Também editável **dentro** do assistente interativo, em `[2] Dados` →
*Perfil de matriz* — não é só um flag de lançamento. Até 2026-08-27 esse
campo não estava em `_CONFIG_SPEC`: não tinha entrada de menu e, mais
grave, **`[S] Salvar Perfil` seguido de `[L] Carregar` resetava
silenciosamente para `generico`**, mesmo se o usuário tivesse escolhido
outro perfil na sessão que gerou o `--perfil=` original (achado da
auditoria de acessibilidade do Passo 83). Corrigido: o campo agora
persiste em `config.yaml` como qualquer outro.

**Achado relacionado (Passo 85, 2026-08-27).** Verificando a extensão do
bug acima com um teste de propriedade (Hypothesis), apareceu uma segunda
via de perda silenciosa de valor no ciclo salvar/carregar: um campo de
texto (`str`) cujo conteúdo *parece* outra coisa em YAML — `"010"`
(octal implícito → inteiro `8`), `"1.50"` (float → perde o zero,
vira `1.5`), `"0x1A"` (hex implícito → `26`) — era escrito **sem aspas**
em `config.yaml` e, ao recarregar, vinha de volta como outro valor. Não
era específico de `perfil_matriz`: qualquer campo `str`/`str_opcional`
(ex.: nome de coluna, caminho) sofria o mesmo se o usuário digitasse algo
que o YAML interpreta implicitamente. Corrigido em `_fmt_yaml`
(`config_io.py`): a decisão de citar ou não uma string agora usa o
próprio `yaml.safe_load` como oráculo (`yaml.safe_load(v) != v` → cita),
em vez de uma lista fixa de palavras reservadas — cobre qualquer forma
que o YAML reinterprete, não só as 6 que alguém lembrou de listar.

Uma SEGUNDA via apareceu no mesmo teste, num campo `list` (`excluir_classes`):
um item contendo `?` — `"0?"` nem chegava a *parsear* de volta (erro), `"?0"`
virava um mapa `{0: None}` em silêncio — porque `?` só é perigoso dentro do
CONTEXTO de uma lista `[a, b]`, não como escalar solto (o oráculo acima não
via problema nele sozinho). Corrigido adicionando `?` ao conjunto de
caracteres que força aspas em item de lista.

**Testes de propriedade (Hypothesis).** A partir do Passo 85,
`tests/test_propriedades_hypothesis.py` gera automaticamente valores para
todo campo de `_CONFIG_SPEC` (não só os que alguém lembrou de escrever à
mão) e confirma que sobrevivem ao ciclo salvar/carregar — foi este teste
que achou o bug acima. Roda como dependência de desenvolvimento
(`pip install -e .[dev]`), nunca em produção.

**Por que o vocabulário importa.** Antes dos perfis, rodar o pipeline sobre
milho em grão gerava um model card afirmando *"quantificação de adulterante
em óleo vegetal amazônico"*. Nenhum número estava errado — a frase estava. O
motor nunca lê esses termos para decidir nada; eles só aparecem em texto, e é
essa separação que impede o vocabulário de uma matriz de contaminar os
resultados de outra.

**Matriz sem perfil não roda.** `UnknownProfileError` é levantado
**antes de carregar qualquer dado**, com a lista de perfis existentes e a
instrução de como escrever um novo. Rodar mel com a faixa e o vocabulário de
óleo produziria números que parecem válidos e afirmações químicas que não são.

**Escrevendo um perfil novo:** copie `generico.yaml`, preencha, e passe o
caminho. Não precisa entrar no pacote nem fazer fork.

### 4b.2 Modo cego — o padrão, e por quê

Quem envia uma amostra desconhecida para um modelo de quantificação **não
sabe a classe dela**. Se a calibração souber, o número medido descreve um
cenário que o usuário final nunca terá em mãos.

| Modo | O que a calibração por classe usa | Quando usar |
|---|---|---|
| `cego` (**padrão**) | a classe **predita** pelo classificador | sempre que o número for para fora |
| `controle` | a classe **verdadeira** | só para isolar erro de quantificação de erro de classificação, no desenvolvimento |

```bash
guaraci                     # cego
guaraci --mode=controle     # marcado como tal em toda a saída
```

No mode cego, um erro do classificador se propaga para a quantificação — e é
**correto** que se propague: é o que aconteceria em produção. Quando não há
classificador ajustado, o mode reportado é `controle-forcado`, nunca `cego`:
um resultado de controle disfarçado de cego seria pior que um resultado de
controle assumido.

### 4b.3 O que o software nunca grava

O parser JCAMP lê apenas os 9 campos de que precisa. **`##AUDIT TRAIL`
(operador, local) e `##$Detector model` não são lidos em momento nenhum** —
não é sorte, é contrato, e há um teste que falha se alguém os adicionar à
lista "porque pode ser útil".

O `##TITLE` **é** lido (dele saem classe, teor e o agrupamento de réplicas),
mas não é gravado: `metadados.csv` passa por `sanitizar_metadados()`, que
remove `title_original`, `arquivo`, `cod`, `data`, `mae_id` e `subpasta`, e
os substitui por `grupo_replica` (`G000`, `G001`…) — anônimo, e preservando
exatamente o agrupamento que sustenta a validação *group-aware*.

Para sanitizar os **arquivos de origem** (útil antes de depositar espectros
num repositório público), use
`python scripts/sanitizar_dx.py <entrada> <saida>` — ele escreve cópias e
**nunca** sobrescreve os originais.

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
Desde a v31.3.0, VIP/SR/sPLS-DA usam **seleção aninhada (nested-CV)**: a
máscara de variáveis é recalculada a cada *fold* usando só as amostras de
treino daquele *fold*, não um VIP/SR pré-calculado no dataset inteiro —
evita o viés de seleção (*double dipping*, Ambroise & McLachlan, 2002)
que inflava o `balanced_accuracy` reportado em versões anteriores. iPLS
não precisou dessa correção (a partição em intervalos não usa rótulo).
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
`fig7_pls_regression`), apenas **normalizados e nomeados** na escala/convenção
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
calibrado por `pls_regression_by_species`, reaproveitado sem reajuste) vs.
Ridge, Lasso, *Elastic Net*, SVR (RBF) e *Random Forest Regressor* — um
modelo **por espécie** (mesma arquitetura da quantificação, calibração
separada evita que a variação entre espécies confunda o sinal de
adulteração), com o **mesmo *split* calibração/validação** (determinístico,
mesma semente/`cal_val_split` do PLS-R) e o mesmo pré-processamento, para
uma comparação honesta ponto a ponto. *Opt-in* via `benchmark_regressao`
(aplicativo e CLI, categoria Avançado — mesmo padrão do Auto-Benchmark de
classificação): gera `benchmark_regressao.csv` (RMSEP/R² agregado e por
espécie) e `fig_benchmark_regressors.png` (*boxplot* de RMSEP por espécie,
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

> **Segurança — carregar um `.joblib` executa código.** Um arquivo `.joblib`
> é um *pickle*: carregá-lo **executa qualquer código contido nele**, antes
> de qualquer validação de conteúdo ser possível. Por isso, todo
> carregamento de modelo (CLI e aplicativo) passa por
> `guaraci.predicao.load_model(caminho, confiar=True)` — um portão
> único que **recusa carregar sem confirmação explícita**: na CLI, uma
> pergunta (s/n); no aplicativo, uma caixa de seleção obrigatória. Além
> disso, todo `modelo_plsda.joblib` exportado pelo pipeline vem com um
> **manifesto** (`<modelo>.joblib.manifest.json`, hash SHA-256 + versões de
> biblioteca) ao lado — se o arquivo for trocado ou corrompido depois que o
> manifesto foi gerado, o carregamento é **bloqueado antes** de o pickle
> executar (não apenas avisado depois). Detalhes completos, incluindo o que
> essas proteções **não** resolvem (não há verificação automática de "isto
> é seguro" para pickle — a decisão de confiar é sempre humana), em
> `SECURITY.md`.
>
> Em implantação pública (demonstrativo hospedado), o operador ainda pode
> definir **`GUARACI_DISABLE_MODEL_UPLOAD=1`** para desabilitar o *upload*
> de `.joblib` pela interface web por completo, aceitando só caminho local
> controlado pelo próprio operador. O *upload* de CSV de espectros
> permanece liberado (dado inerte, não executa nada ao ser lido).

**Figuras:** conjunto essencial por padrão (cerca de 8 a 10 figuras, a
depender do objetivo — seção 2.2) com opção de figuras detalhadas adicionais
(`detailed_figures=True`). Formatos PNG/PDF/SVG, DPI configurável.

### 2.2b Transferência de calibração entre instrumentos (Passo 86 — `transferencia_calibracao.py`)

Um modelo PLS calibrado com espectros de um instrumento tipicamente degrada
quando aplicado a espectros de OUTRO instrumento (mesma amostra,
espectrômetro diferente): deriva de comprimento de onda, resposta de
detector, ótica — produz um deslocamento sistemático que o modelo nunca viu
no treino. `transferencia_calibracao.py` implementa dois métodos clássicos
(Wang, Veltkamp e Kowalski, 1991 — *Multivariate instrument
standardization*, *Analytical Chemistry* 63(23):2750-2756, DOI
`10.1021/ac00023a016`) para corrigir isso a partir de um PEQUENO conjunto de
amostras medidas nos DOIS instrumentos (*amostras de transferência*), sem
recalibrar o modelo do zero:

- **`piecewise_direct_standardization`** (PDS) — uma regressão ridge POR
  CANAL do instrumento mestre, contra uma janela local de canais vizinhos
  do escravo. É o método primário: o deslocamento entre instrumentos é
  predominantemente LOCAL (um pico desloca poucos canais, não o espectro
  inteiro).
- **`direct_standardization`** (DS) — uma única regressão ridge global
  (todos os canais de uma vez). Mais simples, mas o `F` denso (p×p) tende a
  superajustar com poucas amostras de transferência — na validação contra o
  Corn (abaixo), DS não reduziu o erro de forma relevante; PDS reduziu.

`apply_standardization(X_novo, transform)` aplica a transformação aprendida
a espectros novos do instrumento escravo antes de entrar no modelo
calibrado no mestre.

**Validado contra o Corn** (`test_validacao_publica.py`, dataset público —
as mesmas 80 amostras medidas em 3 espectrômetros: m5, mp5, mp6): um PLS de
proteína calibrado só no m5, aplicado direto no mp5, tem RMSEP ≈ 0,51 —
quase 3,5× o RMSEP do m5 sozinho (≈ 0,148). Com PDS (15 amostras de
transferência, janela=5, `alpha`=0,001), o RMSEP no mp5 cai para ≈ 0,16 —
praticamente o mesmo nível do m5 sozinho.

**Limitações conhecidas** (ver seção 9 para a lista completa do projeto):
- **Quantas amostras de transferência**: a validação acima usou 15 — abaixo
  disso a regressão ridge por janela fica instável (poucas amostras
  relativas à largura da janela). Não há um mínimo teórico fechado; 15-20
  amostras cobrindo a faixa de variação do analito é o que a literatura
  original recomenda e o que este projeto validou.
- **`alpha` (regularização) e `janela` são sensíveis ao par de
  instrumentos** — os valores usados na validação (janela=5, alpha=0,001)
  foram medidos empiricamente contra o Corn, não são universais. Um par de
  instrumentos com deslocamento maior/menor pode pedir janela mais larga ou
  `alpha` diferente; não há afinação automática hoje.
- **Pressupõe deslocamento predominantemente linear e local** (o que PDS
  corrige bem). Não corrige efeitos não-lineares fortes de detector, nem
  substitui uma calibração própria no instrumento de destino quando há
  amostras suficientes para isso.

### 2.2c Seleção de amostras de calibração — Kennard-Stone, Duplex, SPXY (Passo 87)

Além do Kennard-Stone (1969) já usado internamente pelo pipeline de
quantificação, `dados_io.py` agora expõe também:

- **`duplex_split`** (Snee, 1977 — *Validation of Regression Models:
  Methods and Examples*, *Technometrics* 19(4):415-428, DOI
  `10.1080/00401706.1977.10489581`) — em vez de encher primeiro o treino
  para só depois sobrar a validação (como KS faz), cresce os DOIS
  conjuntos em paralelo, alternando — os dois ficam representativos do
  espaço, não só o treino. `frac_treino=0.5` é o Duplex clássico
  (alternância estrita); outros valores enviesam a alternância.
- **`spxy_split`** (Galvão et al., 2005 — *A method for calibration and
  validation subset partitioning*, *Talanta* 67(4):736-740, DOI
  `10.1016/j.talanta.2005.03.025`) — o mesmo algoritmo guloso do KS, mas a
  distância combina X (espectro) **e** y (referência/teor), normalizadas:
  `d = d_x/max(d_x) + d_y/max(d_y)`. Cobre o espaço espectral E a faixa do
  analito ao mesmo tempo — KS puro pode deixar de fora o extremo do TEOR
  se ele não for também um extremo espectral (`test_selecao_amostras.py`
  tem um caso sintético que reproduz exatamente isso).

Todos os três (`kennard_stone_split`, `duplex_split`, `spxy_split`) têm
variante `*_group_aware` (mesmo padrão de
`kennard_stone_split_group_aware`): com `mae_id` disponível (≥4 grupos),
colapsa cada grupo de réplicas físicas num espectro médio antes de rodar o
método, depois expande de volta — nenhum dos três separa réplica física
entre calibração e validação (garantido por teste de propriedade
Hypothesis, `tests/test_propriedades_hypothesis.py`).

**CLI** — menu principal, tecla `[K]` *Seleção de Amostras* (Bloco 10, ao
lado do planejamento de coleta): pede um CSV com os espectros (1 amostra
por linha), opcionalmente uma coluna de referência/teor (habilita SPXY),
o método e a fração de calibração, e grava uma cópia do CSV com uma coluna
extra marcando `calibracao`/`validacao` por amostra. Só separa/marca — o
CSV original nunca é alterado.

### 2.3 Planejamento de coleta (Bloco 10 — `plano_amostral.py`/`plano_coleta.py`)

Antes de coletar dados, dois módulos ajudam a planejar **quanto** coletar
e **como** distribuir a coleta sem introduzir confundimentos evitáveis:

- **`plano_amostral.py` — quanto coletar.** Duas fontes de garantia,
  nunca misturadas: (1) o gate **conformal** (Identificar/agrupado) —
  `n_minimum_conformal(alpha)` reaproveita `conformal.n_minimum_for_alpha`
  diretamente, garantia *distribution-free* real; (2) **DD-SIMCA por
  espécie** — `ddsimca_sample_size_guidance(cobertura_alvo)`
  **nunca promete** uma cobertura-alvo acima do platô medido (~0,94-0,945,
  ver seção 9) — acima disso, devolve `alcancavel=False` e recomenda o
  gate conformal em vez de sugerir um `n` que não resolveria o problema.
- **`plano_coleta.py` — como distribuir.** `plan_collection(classes,
  n_por_classe, n_sessoes)` distribui as amostras entre sessões de forma
  **balanceada** (nenhuma sessão fica dominada por uma única classe — o
  confundimento classe×sessão faria qualquer deriva instrumental
  daquela sessão ficar indistinguível de efeito de classe) e
  **aleatoriza a ordem de leitura dentro de cada sessão** (evita ler as
  amostras em ordem correlacionada com teor/classe, que confundiria
  deriva instrumental com sinal químico). Gera alertas automáticos:
  réplica técnica (T1/T2/T3) não conta como amostra independente extra;
  recomendação de brancos/controles intercalados; aviso forte se só 1
  sessão foi pedida (impossível separar classe de deriva temporal nesse
  caso). `plan_from_statistical_target` combina os dois módulos
  numa chamada só.
- **Saída:** Markdown (formato primário) + planilha Excel (`openpyxl`,
  já dependência do projeto — sem biblioteca nova), com a ordem de
  leitura completa por sessão e a lista de alertas. PDF **opcional**
  (`export_pdf`, `fpdf2` — já dependência do projeto via `reports.py`,
  usa `FPDF.table()` nativo), mesmo conteúdo, layout imprimível para
  levar ao laboratório.
- **CLI** — menu principal, tecla `[J]` *Planejamento de Coleta*: pede
  classes, número de sessões, e o alvo estatístico (conformal ou
  DD-SIMCA); gera Markdown+Excel sempre, e pergunta se também quer PDF
  (opcional, `(s/N)`); mostra o resumo + alertas na tela.

### 2.4 Auditoria de delineamento (Bloco 11 — `auditoria_delineamento.py`)

Roda **por padrão em toda execução** (não é opt-in), logo após a
validação de integridade dos dados. Consolida checagens que antes viviam
espalhadas — algumas já existiam em produção, outras só em scripts
privados de auditoria, uma (`outside_working_range`) existia sem
nenhum chamador:

| Checagem | O que verifica |
|---|---|
| `agrupamento` | `cfg.grouping_guarantee` (Bloco 8) — se a validação tem proteção real contra vazamento de réplica. |
| `confundimento_classe_sessao` | Se alguma classe está confinada a **1 única sessão de coleta** (`session_from_mae_id`) enquanto o dataset tem múltiplas sessões — deriva instrumental daquela sessão fica indistinguível de efeito de classe. |
| `duplicatas` | Reaproveita `pipeline.validate_input` — duplicatas exatas/aproximadas (risco de vazamento treino/teste). |
| `n_insuficiente` | Quantas sessões independentes cada classe tem, frente ao mínimo do gate conformal (`conformal.n_minimum_for_alpha`). |
| `faixa_validacao_uso` | Faixa de teor **observada** na calibração vs. faixa de trabalho **declarada** no perfil da matriz (`perfil_matriz.outside_working_range`) — extrapolação silenciosa. |
| `validacao_externa` | Informativo: PLS-R/pré-processamento têm benchmark público (Tecator); classificação/DD-SIMCA/OPLS-DA ainda não. |

Cada checagem é **silenciável individualmente**, mas exige justificativa
não-vazia — a checagem silenciada continua aparecendo no relatório
(severidade `silenciado` + a justificativa anexada), nunca desaparece
sem deixar rastro. Resultado integrado ao `model_card.md` (seção
"Auditoria de Delineamento").

**CLI** — menu principal, tecla `[U]` *Auditoria de Delineamento*: roda
`run_audit` isoladamente sobre o dataset já configurado em `[2] Dados`,
sem exigir rodar o pipeline de classificação/quantificação inteiro (ex.:
auditar antes de decidir se vale a pena treinar). Reaproveita
`load_data`/`validate_input` — mesmo caminho de dados que `executar()`
usa antes de chamar `run_audit`, nenhuma lógica duplicada. Sem fonte de
dados configurada, reporta o erro amigável e aponta para `[2] Dados` em
vez de quebrar.

**Fora do escopo desta versão (registrado, não implementado):** ordem de
leitura correlacionada com o alvo (teor/classe). O método já existe,
validado, num script privado que extrai o timestamp de aquisição do
*audit trail* JCAMP-DX — mas o parser DX em produção não expõe esse
timestamp hoje; portar isso exige mudança no parser, não só *wiring*.

### 2.5 Sentinela de deriva do domínio de aplicabilidade (Bloco 13b — `sentinela_deriva.py`)

Uma amostra isolada fora do domínio de aplicabilidade pode ser só ruído
de amostragem — o próprio *alpha* nominal já prevê uma fração de
rejeições legítimas. O que importa em produção contínua é outra
pergunta: **a taxa de rejeição está subindo ao longo do tempo** — sinal
de que o instrumento/matriz/processo derivou desde a calibração?

`SentinelState` acumula 1 booleano (`AD_dentro_dominio`) por amostra
julgada, com `janela` opcional (`None` = cumulativo sem limite, nunca
descarta dado silenciosamente; um inteiro ativa janela deslizante para
detectar deriva **recente** especificamente — o *trade-off* fica
explícito ao chamador, não escondido atrás de um default mágico).
`check_drift` testa **H0: taxa de rejeição = alpha nominal** contra
**H1: taxa > alpha nominal** via **teste binomial exato** unilateral
(`scipy.stats.binomtest`) — não um limiar cru tipo "2× o nominal", que
teria taxa de falso alarme dependente de `n` sem justificativa formal.

**Defaults justificados, não escolhidos a dedo:**
- `n_minimo` (abaixo disso, não testa — sem poder estatístico): por
  padrão `conformal.n_minimum_for_alpha(alpha_nominal)` — o MESMO mínimo
  prático já usado em todo o projeto para o gate conformal atingir aquele
  *alpha* (ex.: 0,05 → 19). Reaproveitado por consistência, não recalculado.
- `significancia` do teste de deriva: 0,05 por padrão — o mesmo *alpha*
  nominal usado em todo o projeto para os próprios gates (DD-SIMCA, AD,
  conformal).

Verificado por simulação (não só "não quebra"): gerando exatamente na
taxa nominal (H0 verdadeiro), a taxa de falso alarme observada em 300
repetições Monte Carlo fica próxima do `significancia` declarado; uma
deriva real (taxa de rejeição 6× o nominal) dispara o alerta com `n`
suficiente.

**Persistência e integração:** `save_state`/`load_state` (JSON)
permitem que a sentinela sobreviva entre execuções — uso real (LIMS
chamando o pipeline ao longo de dias/semanas) não mantém um processo
Python vivo o tempo todo. CLI — menu `[B]` *Predição em Lote* atualiza
automaticamente a sentinela persistida ao lado do modelo
(`<modelo>.joblib.sentinela.json`) a cada rodada, quando o pacote tem
artefatos de domínio de aplicabilidade, e mostra o status no resumo.

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
acesso aos dados reais de pesquisa), aparece um aviso de **mode
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
| `modos_analise.py` | **Objetivo científico** (Exploratório/Classificação/Quantificação, seção 2.2): fonte única que decide quais figuras/relatórios cada execução gera (`resolve_objective`, `should_generate`, `describe_plan`). `describe_plan` filtra tanto pelo objetivo quanto pelos módulos opt-in ligados (DD-SIMCA, OPLS-DA, Benchmark...) — alimenta o painel ao vivo do terminal **e** a prévia "O que será gerado" da aba Model do app web, que atualiza em tempo real conforme os toggles mudam |
| `config_io.py` | **Fonte única da configuração**: `_CONFIG_SPEC` (campo amigável ↔ atributo), ler/gravar/validar/coagir (`load_config`, `save_config`, `_coagir_valor`, `_validar_semantico`) |
| `resultados_io.py` | Escrita dos artefatos de uma corrida: `resumo_modelo.txt`, `model_card.md`, identificadores CSV, notas metodológicas, métricas PLS |
| `config.py` | *dataclass* `Config`, fonte única de `__version__`/`_NIVEL_NOME` e das constantes de nome de pasta (`NOME_GRAFICOS`/`NOME_TABELAS`/`NOME_RELATORIOS`/`NOME_MODELOS`, seção 3) |
| `chemometric_stats.py` | VIP, Selectivity Ratio, teste de incerteza de Martens, Hotelling T², Q-resíduos, variância explicada, figuras de mérito (LOD/LOQ/SEN/SEL), domínio de aplicabilidade |
| `paleta_cores.py` | Paleta e marcadores de máxima distintividade por classe |
| `dados_io.py` | *Parsing* JCAMP-DX/ASDF, CSV e mode sintético; metadados do `TITLE`; seleção de amostras Kennard-Stone; despacha a leitura via `io_registry.py` |
| `io_registry.py` | *Registry* de leitores de dados: mapeia `cfg.mode` (`dx`/`csv`/`imagem`/`sintetico`) ao leitor correspondente |
| `dados_imagem.py` | Colorimetria digital (`mode="imagem"`, protótipo): extração de *features* RGB/HSV/Lab e textura opcional |
| `hsi_io.py` | Leitor ENVI (`.hdr`+`.bin`) genérico + leitor específico do dataset DeepHS Fruit/Kaki (`mode="hsi"`) |
| `hsi_quality.py` | Quality gate de cubo HSI: saturação, SNR (Immerkaer 1996), fração de pixels válidos |
| `hsi_segmentation.py` | Segmentação objeto/fundo por PCA(PC1)+Otsu, inferência de fundo pela borda da cena |
| `hsi_pixels.py` | Extração de espectros de pixel da ROI + `group_id` de objeto físico (base do split group-aware) |
| `hsi_classification.py` | PLS-DA por pixel (reaproveita `PLSDAClassifier`), seleção de LVs por Wold, agregação por objeto |
| `hsi_chemistry.py` | Explicabilidade cruzada: VIP × tabela de atribuição química (banda↔composto conhecido) |
| `hsi_validation.py` | Validação externa por partição nativa de dia/lote — sensibilidade/especificidade/precisão sempre separadas |
| `hsi_figures.py` | Mapa de classificação espacial por pixel (reaproveita `figuras.save` e a paleta da mascote) |
| `hsi_pipeline.py` | Orquestração ponta-a-ponta do `mode="hsi"` (não usa `pipeline.executar()` — forma de dado diferente) |
| `preprocessamento.py` | *Transformers* SNV/SavGol/MSC e `build_preprocessor` |
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
| `resumo_parse.py` | *Parsing* puro do `resumo_modelo.txt`: `parse_model_metrics` e `parse_accuracy_by_class` |
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

## 9 Limitações conhecidas

Esta seção existe porque declarar limites **aumenta** a confiança no
software — o oposto de esconder os pontos fracos. Cada item abaixo foi
verificado no código desta revisão (ou está marcado explicitamente quando
o número vem de uma rodada anterior contra o dataset real, não
re-executada nesta sessão).

- **Sensibilidade DD-SIMCA (N2) depende do número de grupos de réplica
  pura.** É estimada por *leave-one-group-out* (LOGO) por `mae_id` — ver
  seção 2.1. Com um único grupo de puros por espécie (réplicas físicas da
  **mesma** amostra, não amostras independentes), a sensibilidade **não é
  validável** e o campo mostra `n/a (não validado)`, nunca um número
  inflado. Para ter sensibilidade defensável é preciso ≥2 amostras puras
  **fisicamente independentes** por espécie. A especificidade (rejeição de
  adulterados) não tem essa limitação — é medida em amostras que nunca
  entraram no treino.

- **DD-SIMCA não converge para a cobertura nominal só com mais amostras
  de calibração — há um platô assintótico (medido em 2026-08-26, ver
  `scripts/medicoes/medir_ddsimca_cobertura_vs_n.py`).** Simulação com DGP
  gaussiano controlado (modelo bem especificado, o cenário mais favorável
  possível ao método): a cobertura empírica sobe rápido até n≈150 e depois
  **estanca** num platô de ~0,94-0,945 — inclusive em n=1200, sem sinal de
  aproximar o nominal 0,95 (α=0,05).

  | n | cobertura | desvio | déficit (1−cobertura) |
  |---|---|---|---|
  | 5 | 0,8450 | 0,1510 | 0,1550 |
  | 10 | 0,8957 | 0,0697 | 0,1043 |
  | 20 | 0,9038 | 0,0666 | 0,0962 |
  | 40 | 0,9230 | 0,0346 | 0,0770 |
  | 80 | 0,9306 | 0,0218 | 0,0694 |
  | 150 | 0,9428 | 0,0201 | 0,0572 |
  | 300 | 0,9425 | 0,0122 | 0,0575 |
  | 600 | 0,9448 | 0,0103 | 0,0552 |
  | 1200 | 0,9411 | 0,0070 | 0,0589 |

  Nenhuma forma funcional simples (C/n, C/√n, exponencial) ajusta essa
  curva inteira — C/n teve R²=−1,49 (pior que uma reta horizontal),
  porque a forma real é "convergência rápida + platô persistente", não
  uma curva suave até zero. **Consequência prática:** não existe `n`
  finito que garanta cobertura-alvo abaixo do platô (~0,94-0,945 nesta
  configuração) via DD-SIMCA (método paramétrico χ²-momentos) — para
  cobertura-alvo mais exigente, só o gate conformal (`identificacao.py`/
  `conformal.py`, `ConformalOneClass`) tem garantia formal
  *distribution-free*, sem esse piso assintótico. Este achado motivou a
  reformulação do P1 do Bloco 10 (`guaraci plan`): a orientação de
  tamanho amostral para DD-SIMCA não promete atingir qualquer cobertura
  aumentando `n`.

  **Retratação:** uma instrução anterior desta sessão citou 3 pontos
  específicos (0,840@n=80 / 0,921@n=300 / 0,943@n=1200) como "já medidos
  e validados" — busca no repositório inteiro (código, docs, scripts de
  medição, arquivo de acompanhamento local) não encontrou nenhum artefato
  que os sustentasse. A medição real acima não bate com nenhum dos três
  (diferenças de +0,09, +0,02 e −0,002, sem padrão de erro consistente) —
  os números citados não vieram de medição real.

- **~~`Q2` muda com a versão do scikit-learn~~ — RESOLVIDO em 2026-08-05.**
  Registrado aqui porque afeta a comparação com resultados anteriores.
  *O problema:* o `StratifiedGroupKFold` do scikit-learn muda a partição
  entre versões **mesmo com `random_state` fixo** — medido com dados
  idênticos, **42% das amostras caíam em fold diferente** entre 1.7.2 e
  1.9.0 (10 de 24 grupos de réplica trocando de lado). Isso fazia `Q2`,
  `RMSECV`, acurácia, F1, kappa e até o nº de LVs ótimas dependerem da
  versão instalada.
  *A correção:* o Guaraci passou a usar partição própria
  (`StableStratifiedGroupKFold`), com ordenação fixada por hash
  determinístico — mesma partição em qualquer versão de scikit-learn,
  numpy ou Python, em qualquer sistema operacional (verificado: hash da
  partição idêntico em 1.7.2 e 1.9.0).
  **Atenção ao comparar com rodadas antigas:** números de validação cruzada
  gerados antes desta versão não são diretamente comparáveis com os de
  agora — a partição mudou (uma vez, de propósito). Reexecute antes de
  citar. Continua sendo boa prática informar o ambiente
  (`requirements-lock.txt`) junto de qualquer número publicado.

- **Regressão de teor agrupando espécies não funciona (R²≈0).** A variação
  espectral entre espécies (~90% da variância total) domina completamente
  o sinal de adulteração — o modelo "aprende a prever a média" em vez de
  quantificar. A granularidade correta é **por espécie** (padrão do
  Guaraci, `pls_regression_by_species`) ou, mais fino ainda, **por espécie
  × adulterante** (o mapa de calor da seção 2.1) — nem todo par
  espécie/adulterante é quantificável, e o heatmap marca explicitamente
  quais falham em vez de escondê-los numa média.

- **A identificação de adulterante não pode ser agregada entre espécies —
  a matriz-hospedeira domina o sinal de adulteração mais que o próprio
  adulterante (medido e RE-VERIFICADO com script reprodutível em
  2026-08-26, design do Bloco 9).** Antes de calibrar um identificador de
  adulterante (conjunto aberto, um por soja/algodão/milho, agregando as 13
  espécies), testou-se se a assinatura espectral de cada adulterante é
  consistente entre matrizes — pré-requisito para que uma classe agregada
  seja estatisticamente válida (exchangeability). Calculou-se, para cada
  amostra física adulterada (réplicas T1/T2/T3 colapsadas por `mae_id`,
  espectro pré-processado), o desvio em relação à média da própria
  espécie pura (delta), e mediu-se quanto dessa direção é explicada por
  ESPÉCIE vs. por ADULTERANTE (R² tipo ANOVA/PERMANOVA one-way, Anderson
  2001 — `scripts/medicoes/medir_especie_vs_adulterante_permanova.py`):

  | Corte de teor | R² por adulterante (3 classes) | R² por espécie (13 classes) |
  |---|---|---|
  | ≥0% (n=549 amostras físicas) | 0,0032 | 0,5566 (bruto) / 0,5105 (direção normalizada) |
  | ≥10% (n=216 amostras físicas) | 0,0153 | 0,5873 (bruto) / 0,5160 (direção normalizada) |

  **Espécie explica de 21× a 175× mais variância do delta que o tipo de
  adulterante** — bem mais forte que a estimativa anterior ("6 a 13×"),
  não mais fraca. A relação **se fortalece** em concentrações mais altas
  no lado do adulterante (R² sobe de 0,0032 para 0,0153), consistente com
  sinal químico ficando mais evidente com mais teor. Isso reforça que o
  efeito de matriz-hospedeira é real e sistemático, não um artefato de
  baixa razão sinal-ruído: "soja em Andiroba" e "soja em Castanha do
  Pará" não compartilham uma direção espectral comum o suficiente para
  serem tratadas como a mesma população estatística. **A decisão de
  design permanece a mesma e fica mais bem sustentada, não reaberta:** o
  identificador de adulterante (Bloco 9) é calibrado por combinação
  espécie×adulterante, com cobertura reportada como **não validável** no
  dataset atual (mesmo padrão e mesma linguagem do gate DD-SIMCA, Seção
  4.6 do relatório PIBIC): 36 das 38 combinações espécie×adulterante têm
  exatamente 1 sessão de coleta independente (`session_from_mae_id`,
  Bloco 9b — `grupos_mae_id` inflado pela variação de teor dentro da
  mesma sessão não conta como independência real). A ressalva "não
  validado" se propaga para o resultado final de qualquer predição em
  amostra nova, não fica só no relatório de auditoria interno.

  **Retratação (2026-08-26):** a tabela e a frase "6 a 13×" publicadas
  antes desta correção NUNCA tiveram um script reprodutível — varredura
  de lastro (Passo 62, revisão desta sessão) não encontrou nenhum
  artefato em `scripts/medicoes/` nem nos scripts privados de auditoria
  que os produzisse. Pior: a aritmética do próprio texto publicado não
  fechava — das 4 razões possíveis na tabela antiga (0,1866/0,0034=54,9×
  · 0,1034/0,0034=30,4× · 0,2182/0,0140=15,6× · 0,1147/0,0140=8,2×),
  nenhuma estava genuinamente entre 6 e 13×. Remedido do zero com a
  metodologia acima: o R² por adulterante bateu quase exatamente com o
  citado (0,0032 vs. 0,0034 e 0,0153 vs. 0,0140 — forte indício de que a
  metodologia reconstruída está alinhada com a original), mas o R² por
  espécie saiu ~3× maior que o citado (0,55-0,59 vs. 0,19-0,22) — a causa
  exata dessa diferença específica não foi determinada (o método exato
  usado para gerar os números antigos não está documentado em código em
  lugar nenhum). A conclusão qualitativa (espécie domina, decisão de
  design válida) não muda em nenhum dos dois casos — e fica mais forte
  com os números corrigidos, não mais fraca.

- **Bloco 9b (implementado e verificado em 2026-08-25): fluxo completo
  Detectar → Identificar → Quantificar em amostra nova, no mode cego.**
  `identificacao.py` calibra um ensemble conformal (`ConformalOneClass`,
  ver `conformal.py`) por combinação espécie×adulterante, reaproveitando o
  mesmo espaço PCA do domínio de aplicabilidade — sem ajustar um espaço
  novo por combinação, a maioria tem poucos espectros de 1-2 sessões. A
  contagem de sessão independente usa `dados_io.session_from_mae_id`
  (**não** o `mae_id` bruto: uma amostra adulterada tem um `mae_id` por
  NÍVEL DE TEOR, não por sessão de coleta — contar bruto infla o `n` de 1
  sessão real para até 15). Reexecutado contra o dataset real após essa
  correção: **confirma exatamente** os números já citados acima (36
  combinações com 1 sessão, 2 com 2 — Andiroba×soja e Maracujá×algodão,
  as únicas com `alpha_alcançável` real de 0,333 em vez de `n/a`).
  `Identificar` nunca força uma classe: só preenche resultado quando
  exatamente UMA combinação é aceita sob cobertura estatisticamente
  validada (nenhuma, no dataset atual). `Quantificar` é bloqueado por
  `Identificar` (nunca lança exceção — devolve motivo estruturado:
  `identificacao_desconhecida` ou `identificacao_ambigua`). A ressalva
  aparece nos 3 lugares combinados (log da execução, addendum "Identificação
  espécie×adulterante" no `model_card.md`, e `<modelo>.joblib.manifest.json`
  → `identification_coverage`)
  — mesmo padrão já usado para `grouping_guarantee` (Bloco 8). A predição
  em lote (CLI, menu "Predição em Lote") ganha as colunas
  `classe_identificada`, `identificacao_cobertura`,
  `identificacao_alpha_alcancavel`, `identificacao_candidatos`,
  `teor_estimado`, `quantificacao_motivo_bloqueio` e `alpha_total`
  (limite de união/Bonferroni sobre os alpha de Detectar+Identificar —
  Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2022) sempre que o
  modelo carregado tiver o ensemble; modelos exportados antes do
  Bloco 9b continuam funcionando sem essas colunas (retrocompatível).

  **Detectar tem DOIS sinais complementares, não um só.** Revisão com o
  usuário (mesmo dia): o domínio de aplicabilidade (AD) é ajustado em toda
  a amostragem — puros **e** adulterados juntos — e responde "isto é
  parecido com algo que vimos no treino"; uma amostra adulterada passa
  tranquilamente pelo AD, porque ela faz parte do próprio treino do AD.
  Isso não é o que "Detectar" deveria checar num fluxo de autenticação. O
  DD-SIMCA por espécie (`predicao.detect_purity`, novo) é ajustado só nos
  puros e responde "isto é puro para a espécie predita pela classificação
  N1" — até esta correção, esse modelo só existia dentro de uma rodada N2
  isolada e nunca era persistido no `.joblib` para aplicar depois a uma
  amostra nova. Contra-prova dedicada (`test_ad_e_ddsimca_nao_sao_
  redundantes`): uma amostra sintética adulterada passa no AD e é
  corretamente rejeitada pelo DD-SIMCA de pureza — os dois sinais não
  colapsam no mesmo resultado. Mesmo tratamento honesto de cobertura do
  restante do Bloco 9b: `n_grupos_calibracao<3` (o caso comum é 1 amostra
  pura por sessão) não impede o DD-SIMCA de decidir aceitar/rejeitar
  (método paramétrico χ², não recusa como o conformal), mas o alpha
  declarado fica de fora da soma de Bonferroni — sem lastro numérico não
  entra na conta. Ressalva também nos 3 lugares (`purity_coverage` no
  manifesto, addendum próprio no `model_card.md`, log da execução).

- **Modo imagem (colorimetria digital): protótipo só quando não há fonte
  de agrupamento por amostra física.** Desde o Bloco 8 (2026-08-25), o
  carregador (`dados_imagem.py`) detecta automaticamente 3 níveis de
  garantia de agrupamento — `high` (subpasta por amostra) e `medium` (CSV
  de associação) têm a MESMA proteção anti-vazamento que dx/sintético;
  só `none` (nem estrutura nem CSV) cai em `StratifiedKFold` sem proteção
  e carimba "PROTOTYPE OUTPUT" nos relatórios. `conc` continua sempre
  `None` (sem quantificação neste protótipo, em qualquer nível), e o eixo
  de "variáveis" retornado não corresponde a comprimento de onda físico.
  Não usar nível `none` para resultado publicável sem validação
  adicional; `high`/`medium` têm a mesma garantia group-aware dos outros
  mode de entrada.

- **Validado majoritariamente em FT-NIR.** O motor de pré-processamento e
  modelagem é agnóstico ao tipo de espectro (o parser JCAMP-DX aceita
  FT-NIR, NIR, MIR e Raman, e a interface oferece presets de
  pré-processamento para MIR), mas **nenhuma rodada com dado real de MIR
  ou Raman foi validada neste projeto** — os resultados reportados vêm de
  datasets públicos de NIR (Eigenvector Corn, Tecator; ver
  `docs/VALIDACAO_PUBLICA.md`). Relatos de uso com outras técnicas
  são bem-vindos, mas trate como não testado até prova em contrário.

- **Carregar um modelo `.joblib` executa código arbitrário.** É uma
  limitação do formato pickle, não do Guaraci — ver `SECURITY.md` na raiz
  para a política completa e as proteções implementadas (confirmação
  explícita obrigatória + manifesto de integridade).

- **`mae_id` mal formado vira "órfão" (grupo de 1 amostra), perdendo a
  proteção anti-vazamento só para essa amostra.** Quando o nome do arquivo
  `.dx` não casa com o padrão esperado no `##TITLE=` (JCAMP-DX), o parser
  não trava o carregamento — atribui um `mae_id` único (`orfao_<arquivo>`)
  a essa amostra isolada, para não desabilitar o `GroupKFold` do dataset
  inteiro por causa de um arquivo mal nomeado. O `[INFO]` impresso no
  console/log mostra quantos arquivos viraram órfãos; **conferir sempre
  esse número** — muitos órfãos indicam um problema sistemático de
  nomenclatura, não ruído isolado.

- **Auto-Benchmark e Monte Carlo CV usam hiperparâmetros por heurística de
  literatura, sem *tuning* por validação cruzada interna** (mesmo padrão
  para todos os classificadores/regressores comparados — SVM, Random
  Forest, XGBoost, Ridge, Lasso, Elastic Net, SVR). É uma comparação justa
  entre modelos *fora da caixa*, não a melhor versão possível de cada um;
  não usar essas tabelas para afirmar que um algoritmo é *inerentemente*
  melhor que outro sem otimizar os hiperparâmetros de ambos.

- **Benchmark contra dataset público externo: feito (Tecator), parcial.**
  Ver `docs/BENCHMARK_TECATOR.md` — motor de pré-processamento + regressão
  PLS testado num dataset de terceiros (NIR, teor de gordura em carne,
  Thodberg 1996), fora do dataset próprio do autor. RMSEP dentro da faixa
  esperada da literatura. Cobre pré-processamento+PLS; **ainda falta**
  benchmark de classificação/DD-SIMCA/OPLS-DA contra dataset externo, e
  um segundo dataset (ex.: corn, Eigenvector) para reforçar a
  generalização além de um único caso.

- **Classes com poucas amostras ou espectralmente próximas têm recall mais
  baixo** — reporte sempre a **matriz de confusão completa**, nunca só a
  acurácia agregada, que pode esconder uma classe minoritária mal
  classificada. Classes quimicamente muito próximas — por exemplo, duas
  espécies da mesma família botânica, ou duas variedades do mesmo grão —
  podem se sobrepor no espectro: é limitação **química** da técnica, não do
  código, e nenhum pré-processamento a remove. *(Recall por classe e número
  de espectros descartados por faixa incompatível variam por dataset;
  confira o `resumo_modelo.txt` e a matriz de confusão da sua própria
  rodada.)*

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

ANGELOPOULOS, A. N.; BATES, S. A gentle introduction to conformal
prediction and distribution-free uncertainty quantification. **arXiv**,
2107.07511, 2022.

VOVK, V.; GAMMERMAN, A.; SHAFER, G. **Algorithmic learning in a random
world**. New York: Springer, 2005.

---

*Última revisão do manual: Bloco 9b FECHADO (2026-08-25/26) — Passos 56-57
da revisão final. (56) O manifesto tinha um ÚNICO booleano
`quantificacao_disponivel` calculado só de "existe algum pipeline de
regressão", `true` mesmo com 0/N combinações validadas (quando
`predict_blind` nunca quantificaria de fato) — substituído por
`quantificacao_disponivel_com_garantia` (exige combinação validada COM
modelo de regressão da mesma espécie) + `quantificacao_possivel_sem_
garantia` (maquinaria existe, sem garantia estatística). (57) Execução com
dado sintético não se autodeclarava em lugar nenhum — métricas quase
perfeitas (accuracy/kappa=1,0000, regime ESPERADO do gerador sintético)
ficavam indistinguíveis de resultado real; `dados_sinteticos` agora no
manifesto + aviso no topo do `model_card.md` sempre que
`cfg.mode=="sintetico"`. Antes, mesmo Bloco 9b: Detectar fechado com um
segundo sinal complementar (`predicao.
detect_purity`, DD-SIMCA por espécie ajustado só nos puros, persistido no
`.joblib` pela primeira vez — antes só existia dentro de uma rodada N2
isolada); o domínio de aplicabilidade sozinho responde "parecido com o
treino", não "puro", e uma amostra adulterada passa por ele sem problema
(contra-prova dedicada confirma que os dois sinais não colapsam). Também
corrigido: addendum de Identificação no `model_card.md` tinha número de
seção fixo ("## 10.") que aparecia ANTES do addendum de Quantificação
("## 9.") sempre que a regressão também rodava (append-only, ordem de
escrita = ordem no arquivo) — título sem número agora. Antes, no mesmo
Bloco 9b: fluxo completo Detectar
→ Identificar → Quantificar em amostra nova (mode cego), implementado e
verificado contra o dataset real: `identificacao.py` novo (ensemble
conformal por combinação espécie×adulterante), `pipeline.
pls_regression_by_species` agora persiste os modelos de regressão POR
ESPÉCIE ajustados (antes só calculava métricas de CV, não guardava o
modelo pronto para uso), `predicao.predict_blind`/`quantify_sample` novos,
CLI (menu Predição em Lote) ganha as colunas do fluxo cego. Achado
corrigido durante a implementação: a contagem de "sessão de coleta
independente" não pode usar `mae_id` bruto (infla até 15× por diluição de
teor dentro da mesma sessão) — `dados_io.session_from_mae_id` novo,
reexecução contra o dataset real confirma os números já citados acima (36
combinações com 1 sessão, 2 com 2). Antes: novo `docs/VALIDATION.md` (cartão de visita
técnico) — tabela com 11 linhas de validação contra sklearn/fórmulas
fechadas (PLS-DA, SNV normalização+invariância de espalhamento, VIP, MSC,
DD-SIMCA T²/Q-resíduos, CV-ANOVA, BCa, teste de permutação, OPLS-DA),
valores obtidos rodando a suíte nesta sessão, com seção honesta do que
ainda NÃO está validado (dataset público externo, cobertura empírica do
BCa). Linkado em README.md/README.pt-br.md. Antes: terminologia da
interface (CLI e README) revisada
para liderar com o nome amigável do mode de análise ("Classificação por
espécie (N1)" em vez de "N1 — Classificação..."); o código interno N1/N2/N3
passa a aparecer como referência técnica secundária, nunca como o rótulo
principal — a tabela de equivalência nível↔objetivo (seção 2.2) já estava
correta e não mudou. Antes disso: `SECURITY.md` novo (raiz) — carregamento
de modelo `.joblib` agora passa por `load_model(confiar=True)`
obrigatório (CLI: confirmação s/n; app: caixa de seleção) + manifesto
SHA-256 gerado junto de todo modelo exportado, que bloqueia o carregamento
se o arquivo for trocado depois. Antes disso: nova seção 9 "Limitações
2 figuras que faltavam (CLAUDE.md seção 5): **espectros médios por classe**
(`fig0_espectros_medios_classe.png`, banda ±1 DP, dado bruto, gerada em
qualquer objetivo — contexto químico antes da modelagem) e **biplot PCA**
(`fig_biplot_pca.png`, scores + top-12 loadings sobrepostos, objetivo
Exploratório) — as outras 2 figuras "que faltavam" (RMSECV×LVs e o heatmap
espécie×adulterante) já existiam, achado ao verificar antes de implementar.
Bug real corrigido no biplot antes do commit: escala única calibrada pelo
maior score conjunto (PC1+PC2) desenhava vetores com componente forte no
eixo de menor alcance fora da área visível — corrigido calibrando por eixo
e usando o mais restritivo (`_escala_vetores_biplot`, com teste de
regressão dedicado). Antes disso: nova seção 9 "Limitações
conhecidas" (item do roadmap CLAUDE.md) — 9 itens verificados no código
desta revisão (DD-SIMCA/LOGO, regressão por espécie, mode imagem
protótipo, FT-NIR vs. MIR/Raman não validado, `.joblib`/RCE, `mae_id`
órfão, hiperparâmetros do benchmark sem tuning, sem dataset público
externo, recall por classe) — números específicos de dataset real
marcados como não re-executados nesta sessão. Antes: mapa de calor espécie × adulterante (N3,
`fig_heatmap_especie_adulterante.png` — nome sem "N3" desde 2026-07-13) — R²cv por combinação, com células
reprovadas hachuradas e contador de falhas no título e no relatório;
sensibilidade DD-SIMCA (N2) agora estimada por
leave-one-group-out honesto por réplica `mae_id` — sempre exibida com o número
de grupos e aviso de incerteza; `n/a (não validado)` quando há um só grupo de
puros (substitui a re-substituição, que inflava até 100%). Antes: prévia "O que
será gerado" em tempo real na aba Model (web), 8ª aba **Sobre** (identidade,
licença, como citar), cabeçalho com logo/versão/badges e aviso de mode
demonstração no deploy público sem `config.yaml` local.*
