# MANUAL DO GUARACI — PLATAFORMA DE QUIMIOMETRIA

**Público:** todos · **Estado:** manual · **Vale para:** a versão do programa que traz este texto (`guaraci --version`); o histórico de mudanças está no `CHANGELOG.md`, na página do projeto

> Manual de instruções das funcionalidades do projeto. Mantido atualizado a
> cada mudança relevante de funcionalidade, interface ou fluxo. Para
> instalação, citação e licença, ver o `README.md` (na página do projeto, <https://github.com/ErleySC/guaraci>).

O **GUARACI** é uma plataforma de quimiometria multitécnica para autenticação
e caracterização de matrizes complexas, a partir de qualquer dado espectral
ou tabular (FT-NIR, NIR, MIR, Raman, UV-Vis, cromatografia). O que é
específico de cada matriz vive num **perfil** (`guaraci/perfis_matriz/*.yaml`),
nunca no código-fonte — ver a seção "Perfis de técnica".
Cobre todo o fluxo científico — análise exploratória, classificação,
autenticação, quantificação e relatórios de publicação — com validação
estatística rigorosa e proteção contra vazamento de réplicas em cada etapa.

## SUMÁRIO

0. [Primeiros passos (comece aqui)](#0-primeiros-passos-comece-aqui)
1. [As três formas de usar](#1-as-três-formas-de-usar)
2. [Modos de análise e objetivo científico](#2-modos-de-análise-e-objetivo-científico)
3. [Estrutura de saída dos resultados](#3-estrutura-de-saída-dos-resultados)
4. [Fontes de dados de entrada](#4-fontes-de-dados-de-entrada)
4b. [Perfis de técnica e modo cego](#4b-perfis-de-técnica-e-modo-cego)
5. [Funcionalidades científicas](#5-funcionalidades-científicas)
6. [Fluxo típico na interface web](#6-fluxo-típico-na-interface-web)
7. [Mapa dos módulos (para desenvolvedores)](#7-mapa-dos-módulos-para-desenvolvedores)
8. [Desenvolvimento](#8-desenvolvimento)
9. [Limitações conhecidas](#9-limitações-conhecidas)
10. [Perguntas frequentes e problemas comuns](#10-perguntas-frequentes-e-problemas-comuns)
11. [Glossário](#11-glossário)
12. [Requisitos mínimos](#12-requisitos-mínimos)
13. [Como reportar erro](#13-como-reportar-erro)
10. [Referências](#referências)

---

## 0 Primeiros passos (comece aqui)

Este caminho leva você do zero ao primeiro resultado, na ordem, sem
conhecer o projeto nem quimiometria. Se travar, anote **onde** e **por
quê** — é a informação mais útil que você pode dar.

**1. Instale** (uma vez; precisa de Python 3.11 ou mais novo). Na pasta do
projeto:

```bash
pip install -e .
```

Para a versão visual e os relatórios em PDF/Word/Excel, instale também os extras:
`pip install -e .[web,reports]`.

Alguns formatos e recursos dependem de pacotes opcionais que a instalação base não traz: `opus`
(Bruker OPUS), `spc` (Thermo/Galactic SPC), `rmn` (pasta Bruker de RMN), `hplc` (pasta
Agilent/Waters), `multiway` (PARAFAC), `imagem` (textura de imagem) e `robusto` (PCV do DD-SIMCA).
A tela `[2.4]` escreve `(extra: pip install ...)` ao lado do formato quando o pacote falta e
avisa na hora em que você o escolhe; `guaraci doctor` e `guaraci checar` listam o que falta e
quais técnicas isso afeta. Para instalar tudo de uma vez:
`pip install ".[all]"` (de dentro da pasta clonada; para desenvolver: `pip install -e ".[all]"`).
Se preferir que o GUARACI instale só o que falta (navegador, relatórios e os leitores disponíveis para o seu Python,
sem o `xgboost` e o `shap`), rode `guaraci doctor --instalar`: ele mostra a lista, pede confirmação (`--sim` não pergunta)
e usa o `pip` do mesmo Python do GUARACI.

**2. Confira o ambiente.** Este comando diz se falta alguma dependência (e grava um arquivo
`guaraci_doctor.txt` na pasta atual):

```bash
guaraci doctor
```

**3. Veja funcionar com dados de teste** (não precisa de nenhum arquivo
seu; leva alguns minutos e grava tudo numa pasta `GUARACI_Demo/` **dentro
da pasta em que você rodou o comando**; no Windows, ao terminar ele abre
o explorador de arquivos nessa pasta):

```bash
guaraci demo
```

No fim, ele mostra o caminho dos resultados. Abra a pasta `Graficos/`
(figuras) e o arquivo `Relatorios/resumo_modelo.txt` (números).

**4. Use os seus dados.** Organize uma pasta com **uma subpasta por
classe** (por exemplo, `dados/ClasseA/`, `dados/ClasseB/`), cada uma com
os espectros dessa classe (arquivos `.dx`) — ou uma tabela CSV com uma
coluna de classe. Depois abra o assistente:

```bash
guaraci
```

No menu, digite **`2`** (Dados). A tela lista os campos, cada um com um
número (os números podem mudar — vá pelo **nome**, não pelo número):
- **Modo de entrada**: `dx` (espectros `.dx` em pastas), `csv` (uma
  tabela), ou um dos formatos de instrumento (`opus`, `spc`, `sp`, `rmn`,
  `hplc`, `gcms`, `eem`);
- para `csv`: **Arquivo CSV** (caminho do arquivo) e **Coluna de classe**
  (nome da coluna que diz a classe; padrão `classe`);
- para os demais, a **pasta de entrada** está na tela **`[1]` Projeto**.

Cada mudança pede confirmação (`s`) e um Enter para continuar. Se não tem
dados próprios, gere um arquivo de exemplo (números **inventados**):
`python scripts/gerar_dados_exemplo_teste.py` cria `dados_exemplo.csv`.
Volte com **`0`** e digite **`R`** (rodar). O assistente mostra um checklist antes de começar e um painel de
progresso durante. Para rodar sem menu, a partir de um arquivo `config.yaml`:
`guaraci run config.yaml`. Prefere clicar? `guaraci web` (modo local; `streamlit run app_quimiometria.py` sobe o modo servidor, só com upload — ver `guaraci manual seguranca`)
abre a versão visual, com as mesmas etapas na barra lateral (Projeto →
Dados → Pré-processamento → Modelo → Validação).

**5. Onde estão os resultados.** Numa pasta nova dentro da pasta de saída
(padrão `resultados_guaraci/`), com `Graficos/`, `Tabelas/`, `Relatorios/` e
`Modelos/`. O `Relatorios/model_card.md` é a página que resume o que o
modelo faz, com o que não faz.

**6. Como ler o resultado sem ser especialista.**
- *Acurácia balanceada* — de 0 a 1, quão bem o modelo acerta em média
  **cada classe** (1 = acerta tudo). É a nota principal da classificação.
- Se o resumo disser **"não validado"** ou **"n/a"**, não é defeito: o
  programa está recusando dar um número que os seus dados não sustentam
  (poucas amostras independentes). Leve isso a sério — é a característica
  mais importante do GUARACI, não uma falha.
- Várias medições da **mesma amostra física** são tratadas como um grupo
  (não se separam entre treino e teste) — por isso os números são mais
  baixos, e mais honestos, do que em programas que não fazem isso.

**7. Avisos de maturidade, em linguagem comum.**
- *Monitoramento de deriva do instrumento (MSPC)*: os limiares padrão são
  ponto de partida, não universais, e ajustáveis por opção. Trate
  como um alarme a confirmar, não como veredito; ele pode demorar até um
  lote para avisar.
- *Fusão de duas técnicas (ex.: NIR + MIR)*: **nem sempre melhora** o
  resultado em relação à melhor técnica sozinha. Use só para
  comparar.
- *Aumento de dados (VRM)*: efeito pequeno (~0,2–1%); é experimental e
  não está em nenhum menu — só via código.
- *Dual-sPLS*: aparece no benchmark de regressão; pode melhorar em alguns
  ajustes e ser neutro em outros — não é substituto automático do PLS.
- *Técnicas avançadas (`[T]`)*: são ferramentas de exploração; resultados
  delas ajudam a **entender** os dados, não substituem a validação do
  modelo principal.
- *Imageamento hiperespectral (`[X]`)* é só no terminal e ainda é
  protótipo.

Detalhes de cada item estão nas seções seguintes e na seção 9.

---

## 1 As três formas de usar

O código fica no pacote `guaraci` (em `src/`). Instale uma vez com
`pip install -e .` (disponibiliza o comando `guaraci`); sem instalar, use
`PYTHONPATH=src`.

| Forma | Comando | Para quem |
|---|---|---|
| **Web (Streamlit)** | `guaraci web` (local) · `streamlit run app_quimiometria.py` (modo servidor) | Uso visual, navegação lateral em 4 grupos + telas Início/Visualização. Demo público: <https://guaraci.streamlit.app> |
| **Assistente de terminal** | `guaraci` (ou `PYTHONPATH=src python -m guaraci.guaraci`) | Menu interativo colorido, sem editar código |
| **Pipeline direto** | `python -m guaraci.pipeline --rodar` | Execução automatizada a partir de `config.yaml` |

As três formas compartilham o mesmo motor (`pipeline.py`) e o mesmo formato de
configuração (`config.yaml` / classe `Config`) — não há divergência de
resultado entre elas **quando lêem o mesmo arquivo**. Mas cada uma lê um
arquivo diferente:

| Forma | Qual `config.yaml` |
|---|---|
| Assistente (`guaraci`) | `~/.guaraci/config.yaml` (o caminho é impresso ao abrir) |
| `guaraci run [arquivo]` | o `arquivo` dado; sem argumento, `./config.yaml` (o caminho é impresso) |
| Web local (`guaraci web`) | edita e usa a configuração da sessão; grava em `~/.guaraci/` só o que você salva |
| Web servidor | nenhum (só upload; configuração isolada por sessão) |

Edições feitas nos menus do assistente só valem para a sessão até você
rodar ou salvar; sair sem rodar **não** grava.

> **Idioma das figuras:** todos os textos das figuras (eixos, títulos, legendas) ficam em **inglês**, o idioma de publicação; a interface e os relatórios seguem o idioma escolhido. Nas figuras de linha (ROC, espectros médios) cada classe tem, além da cor, um **estilo de linha** próprio, legível em daltonismo e em impressão P&B.

> **Leitor de tela?** Rode a CLI com `GUARACI_MODO_LINEAR=1`: a tela não é limpa a cada ação e as caixas perdem os caracteres de desenho, então o leitor lê só o conteúdo novo. `NO_COLOR=1` tira também cores e negrito.

> **Terminal de fundo claro?** A CLI usa cores pensadas para fundo escuro. Em fundo branco, rode com `GUARACI_TERMINAL=claro` (ou deixe o terminal definir `COLORFGBG`); todas as cores passam a ter contraste ≥ 5:1.

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
`n_jobs_permutacao=1` e há muitas permutações, o checklist sugere
explicitamente subir esse valor: o resultado é idêntico (mesmo seed, mesma
partição), só o tempo de execução muda.

**Classe com uma só unidade independente.** Validar exige ao menos duas unidades
(amostras físicas ou grupos `mae_id`) em cada classe: uma classe com uma só não pode estar
no treino e no teste ao mesmo tempo. Em Classificação e Quantificação, antes de rodar, o
programa lista essas classes e pergunta se deve retirá-las desta análise (`(s/N)`; em inglês
`(y/N)`). Respondendo sim, elas vão para o campo `excluir_classes` e a análise roda sem mais
perguntas; para trazê-las de volta, apague o campo em **Dados**. Se a falta de unidades só
aparecer depois de separar o holdout, a análise para com "Validação cruzada impossível",
mostra as classes e oferece retirá-las e rodar de novo. No objetivo **Exploratório** nada é
retirado (seção 2.2).

### 1.2 Painel de acompanhamento em tempo real (terminal)

Ao rodar pelo assistente de terminal, a execução é acompanhada por um painel
ao vivo (biblioteca *Rich*) que mostra, a cada instante: o objetivo científico
resolvido (seção 2), o percentual real de progresso e a etapa em andamento
(lidos diretamente do log do motor, não estimados por tempo decorrido), o
tempo estimado restante, a lista de figuras já concluídas frente ao total
planejado para o objetivo em curso, e os avisos não fatais emitidos durante
a execução. O mesmo mecanismo de captura de log alimenta a barra de
progresso do aplicativo web.

Quando a análise termina, com sucesso ou com erro, o painel fecha, a tela volta ao terminal
e aparece o resumo (ou a mensagem de erro) seguido de "Enter para continuar". No Windows a
edição rápida do console fica desligada durante a análise, porque um clique na janela
pausava a saída e parecia travamento. Se o log ficar 120 s sem linha nova, o programa grava
a pilha de todas as threads em `~/.guaraci/travamento_*.txt`, para anexar a um relato de
erro (seção 13); a análise continua.

---

## 2 Modos de análise e objetivo científico

O **modo de análise** define o que o pipeline faz. Na interface aparecem com
nomes amigáveis; internamente são identificados como N1/N2/N3.

### 2.1 Os três níveis (N1 / N2 / N3)

> `N1`/`N2`/`N3` é o código interno de `cfg.level` (usado em `config.yaml`,
> nunca exibido como termo primário na CLI/web). No nome da pasta de saída
> (seção 3), cada nível vira um slug amigável: `N1`→`PorEspecie`,
> `N2`→`Autenticacao`, `N3`→`Quantificacao` (versões antigas gravavam `N1`/`N2`/`N3` cru no nome do diretório).

- **N1 — Classificação (por espécie).**
  Identifica a qual classe cada amostra pertence (espécie, variedade, tipo —
  o nome vem do perfil de técnica). Método: **PLS-DA** com `GroupKFold` anti-vazamento
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
  **LOD/LOQ como intervalo, não ponto:** o ruído instrumental
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
  estimar o ruído instrumental. **Unidade:** LOD, LOQ, RMSEP e a faixa de validação saem na unidade
  do teor de referência dos seus dados (não há "%" fixo): leia o número na unidade em que você
  informou a concentração.
  **Faixa de decisão em 3 estados:** todo teor previsto é
  categorizado contra esse mesmo LOD/LOQ — `nao_detectavel` (< LOD),
  `zona_cinzenta` (entre LOD e LOQ: detecção possível, quantificação não
  confiável) e `quantificado_com_confianca` (≥ LOQ). As colunas
  `faixa_decisao`, `lod` e `loq` saem no CSV de predição da **CLI e da aba
  Predição do app web** (paridade garantida por
  `tests/test_paridade_predicao_cli_web.py`); na web a coluna vem colorida
  e há um resumo com a contagem de amostras em cada faixa. Quando o modelo
  da espécie não tem LOD/LOQ persistido (ou não computável, sem réplicas
  suficientes), a faixa é `None` e a interface diz isso — nunca
  "não detectável" por omissão.
  **RPD/RER** (`chemometric_stats.rpd_rer`)
  já estavam implementados antes deste bloco — reverificado por leitura
  direta do código nesta revisão, não presumido da memória: razão SD/SEP e
  amplitude/SEP, com as faixas de interpretação de Williams (2014), já
  integrados ao pooled do `pls_regression_by_species`. O *split*
  calibração/validação dessa regressão aceita dois
  métodos via `cal_val_split` (campo de `Config`, só via código — não é chave do `config.yaml`; hiperparâmetro avançado,
  **não** exposto no aplicativo/CLI — mesmo padrão de
  `ipls_n_intervalos`/`vip_threshold_sel`): `"aleatoria"` (padrão,
  `GroupShuffleSplit` *group-aware*) ou `"kennard_stone"` (Kennard e Stone,
  1969 — cobertura maximamente representativa do espaço espectral em vez de
  aleatória; com réplicas físicas, colapsa cada grupo T1/T2/T3 num espectro
  médio antes de selecionar, preservando o mesmo invariante de nunca separar
  réplicas entre calibração e validação).
  Gera ainda o **mapa de calor espécie × adulterante**
  (`fig_heatmap_species_by_adulterant.png`): um **R²cv** (R² em validação
  cruzada *group-aware* — mede o acerto do teor em amostras não vistas no
  treino) por combinação de espécie e adulterante. Existe porque a regressão
  que *junta* os adulterantes de uma espécie mascara que alguns adulterantes
  simplesmente **não são quantificáveis** (o sinal afoga no ruído). Cada
  célula **abaixo do limiar de aceite (R²cv = 0,70)** aparece **hachurada e em
  negrito** (nunca some), célula sem dados suficientes vira cinza com "n/a", e
  o título + o `resumo_modelo.txt` trazem o **contador de falhas** (ex.: "3/9
  combinações abaixo de R²cv = 0,70") — para que uma quantificação que só
  funciona em parte das combinações não seja lida como sucesso geral. O
  adulterante de cada amostra é derivado do `mae_id`. *(No modo sintético, ative
  com `synthetic_adulterants` (campo de `Config`, só via código — não é chave do `config.yaml`), ex.: `["S","M","A"]`.)*

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

No Exploratório **não há holdout** e uma classe com uma só unidade **não interrompe** a
análise: os gráficos usam todas as amostras e todas as classes. O motor ainda ajusta um
PLS-DA interno (usado no painel T²/Q e no resumo), com divisão simples por grupo
(`GroupKFold`); a linha "Validation" do resumo diz "exploratorio: nao valida o modelo". Os
números de acurácia e Q² desse resumo não valem como validação nesse objetivo.

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
  `tag` (campo de `Config`; na CLI, tecla `[N]`; não é chave do `config.yaml`) quando preenchido; senão é derivado
  automaticamente do mode de entrada (nome do arquivo CSV, nome da pasta de
  espectros, ou `"sintetico"` para dados de teste).
- **`<Modo>`** — rótulo amigável do objetivo científico resolvido:
  `Exploratorio`, `Classificacao` ou `Quantificacao`.
- **`<Execução>`** — identificador único da corrida (nível, pré-processamento
  e data/hora), no formato `PLSDA_OE_<slug-do-nivel>_<preproc>_<AAAAMMDD_HHMMSS>`.
  `<slug-do-nivel>` é o nome amigável do nível (`PorEspecie`/`Autenticacao`/
  `Quantificacao`; versões antigas usavam
  `N1`/`N2`/`N3` cru aqui).
- **`Graficos/`** — todas as figuras (`.png`/`.pdf`/`.svg`, conforme
  `output_format`). Rodando pelo comando `guaraci`, elas saem em subpastas
  por grupo de análise (`01_dados_preprocessamento/`, `02_exploracao_PCA_HCA/`,
  `03_PLS-DA/`, ..., `07_DD-SIMCA/`, `10_quantificacao/`): figuras da mesma
  pasta se corroboram, e o `README.txt` da pasta diz o que cada grupo mostra.
  Pela API (`import guaraci`) a pasta continua plana, salvo
  `guaraci.figuras.ORGANIZAR_POR_GRUPO = True`.
- **`Tabelas/`** — dados tabulares em CSV (identificadores de amostra,
  metadados, teste de Martens, comparação de *pipelines*, benchmarks).
- **`Relatorios/`** — `resumo_modelo.txt` e `model_card.md`.
- **`Modelos/`** — modelo final exportado (`modelo_plsda.joblib`).

Exemplo real (execução de Classificação com `tag="lote_a"`):

```text
resultados_guaraci/lote_a/Classificacao/PLSDA_OE_Autenticacao_MSC-SG1-MC_20260705_222028/
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
| `hsi` | Imageamento hiperespectral — aceita cubo próprio, offline (ver adiante) | Distinto de `imagem`: por pixel, não por foto |
| `opus` | OPUS binário (Bruker FT-IR/FT-NIR) | Requer `pip install -e ".[opus]"`; sem extensão fixa (tenta cada arquivo) |
| `spc` | SPC (Galactic/Thermo GRAMS), `.spc` | Requer `[spc]` (instalado de um commit git fixado: precisa do `git`); multi-subarquivo usa só o 1º |
| `sp` | PerkinElmer `.sp` | Sem dependência extra |
| `rmn` | RMN Bruker, espectro já processado (`pdata/<N>`) | Requer `[rmn]`; 1 pasta de experimento = 1 amostra |
| `hplc` | Cromatograma Agilent (`.D`)/Waters (`.raw`) | Requer `[hplc]` (Python ≤ 3.13: a dependência não tem wheel para 3.14); ver `hplc_detector` (`UV`/`FID`) e `cromatografia_canal` (abaixo) |
| `gcms` | GC-MS ANDI-MS/netCDF (`.CDF`) | Sem dependência extra (via `scipy`); TIC por padrão, espectro de massas por scan com `cromatografia_canal` (abaixo) |
| `eem` | Fluorescência excitação-emissão (`.dat`), achatada em vetor 1D | Perde a estrutura 2D — ver ressalva abaixo |
| `multielementar` | Tabela amostra × elemento do ICP-OES/ICP-MS (CSV) | **Em validação**; exige `icp_unidade` e `coluna_grupo` — ver seção 5.8 |
| `sintetico` | Dados simulados | Para testes/demonstração |

**Canais do cromatograma (`cromatografia_canal`) e teto de pontos.** Padrão (vazio): `hplc` lê o 1º canal do detector e `gcms`
lê o TIC, como antes. `todos` ou uma lista (`254,280` em `hplc`; `57,71,85` em `gcms`) lê vários canais: cada amostra vira UM vetor,
a matriz tempo × canal achatada **por canal** (canal 1 em todos os tempos, depois o canal 2...), e o eixo passa a ser a posição
1..n. Um só comprimento de onda dá um cromatograma comum (eixo = tempo em min). `hplc` lê os canais DAD/`.ch` do Agilent e do
Waters pelo `rainbow-api` (exige o mesmo eixo de tempo nos canais escolhidos); `gcms` soma as intensidades por unidade de massa
(m/z arredondado ao inteiro) a partir do layout esparso do ASTM E1948 e exige a mesma faixa de m/z em todos os arquivos (senão,
liste os m/z). Mais de 500 000 variáveis (tempos × canais) pedem `cromatograma_max_pontos` ou menos canais.
O eixo comum de espectros (`dx`, `opus`, `spc`, `sp`, `rmn`) tem teto `espectro_max_pontos` (padrão 2000, 0 = sem limite); o de
cromatogramas (`hplc`, `gcms`) tem `cromatograma_max_pontos` (padrão 0 = sem limite). Quando o eixo é reduzido, o log e o resumo
trazem o AVISO "n original → n final", o método e o passo da grade; a redução com razão ≥ 2:1 é média por intervalo (nunca amostra
pontual). **Não validado com arquivo real de fabricante:** o `.uv` Agilent e o ANDI-MS com espectros dos testes são sintéticos,
escritos no layout que o `rainbow-api` e o ASTM E1948 descrevem.

**Modo `imagem` (colorimetria digital):** extrai estatísticas de
cor (média/desvio-padrão por canal em RGB, HSV e Lab — 18 variáveis) de cada
fotografia e, opcionalmente, textura (GLCM, requer `pip install
scikit-image`). Mesma convenção de pastas do mode `dx` (uma subpasta por
classe). A partir da extração, toda a maquinaria quimiométrica (PCA, PLS-DA,
DD-SIMCA, seleção de variáveis, figuras de mérito) funciona sem alteração —
cada estatística de cor vira uma "variável", exatamente como um comprimento
de onda.

**Três níveis de garantia de agrupamento.** Ao
contrário do mode `dx` (que só tem `mae_id` quando o `##TITLE=` segue a
convenção opcional da seção 4.1, ou quando você informa `arquivo_grupos`),
o mode `imagem` detecta automaticamente qual fonte
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
usuário definir a região de interesse via `image_crop` (campo de `Config`, só via código; recorte
retangular relativo, `config.yaml`) antes da extração.

**Modo `hsi` (imageamento hiperespectral):** DISTINTO do mode `imagem`
acima — opera **por pixel** de um cubo hiperespectral (formato ENVI,
par `.hdr`+`.bin`), não por foto inteira. Fluxo: leitura genérica
(`hsi_io.load_hsi_folder_dataset`) → quality gate (saturação, SNR,
fração de pixels válidos) → segmentação objeto/fundo (PCA+Otsu) →
extração dos pixels da ROI, cada um marcado com o `group_id` do objeto
físico de origem (mesmo conceito de `mae_id`, nunca dois pixels do
mesmo objeto em lados diferentes de um split) → PLS-DA por pixel
(reaproveita `PLSDAClassifier`) com agregação por objeto (classe
majoritária + heterogeneidade) → mapa de classificação espacial →
validação.

**Aceita o cubo do PRÓPRIO usuário** — `hsi_pasta_dataset` (`Config.hsi_dataset_folder`)
aponta pra' QUALQUER pasta com cubos ENVI, convenção de subpasta-por-
classe (mesma do mode `dx`/`imagem`), sem exigir manifesto nem layout
específico. Agrupamento por amostra física reaproveita a mesma
hierarquia de 3 níveis do mode `imagem` (subpasta por amostra física /
CSV `amostras.csv` de associação na raiz / nenhuma fonte, sempre
declarado explicitamente, nunca presumido). Funcionamento **offline**
verificado por teste com o socket de rede desabilitado
(`tests/test_hsi_offline_prova.py`) — não é alegação.

`run_hsi_pipeline` usa o caminho genérico para qualquer pasta de cubos; a explicabilidade cruzada por tabela de atribuição química e a partição por dia de medição exigem metadados que uma pasta genérica não tem.

Extras, no fim da tela `[X]` (só CLI): `[D]` domínio de
aplicabilidade (calibra em objetos de treino, avalia objetos de teste),
`[R]` reamostragem de classes minoritárias (duplica objetos — equaliza
pixels, não cria objetos novos) e `[M]` comparação N-PLS multiway ×
PLS-DA por pixel (exploratória; precisa de ≥~5 objetos por classe). O HSI não tem página no aplicativo web (ver seção 9).

Acessível pela tecla **`[X]`** do menu principal da CLI. Orquestrado
por `hsi_pipeline.run_hsi_pipeline`, não por `pipeline.executar()` — a
forma de dado (por pixel, agregação por objeto) é fundamentalmente
diferente da matriz amostras×variáveis que os demais modes
compartilham. Por isso o modo `hsi` vale só enquanto a tela `[X]` está
aberta: ao sair, o modo de entrada anterior volta, e `[R]` continua
rodando os dados configurados em `[2]`. Versões antigas deixavam o modo
gravado, e o `[R]` falhava depois de uma simples visita à tela.

Com classes muito desbalanceadas o desempenho em HSI tende a ser modesto. Rodar em cubo próprio funciona mecanicamente; como qualquer perfil novo, o desempenho na SUA técnica é não testado até você testar.

**Modos de instrumento (`opus`/`spc`/`sp`/`rmn`/`hplc`/`gcms`/`eem`):**
convertem o formato binário/nativo de fabricante para o mesmo fluxo de
`dx`/`csv` — pré-processamento, PLS-DA/PLS-R, DD-SIMCA, tudo funciona sem
alteração. Convenção de pasta: uma subpasta por classe (mesma do mode `dx`),
cada arquivo (`opus`/`spc`/`sp`/`gcms`) ou subpasta-amostra (`rmn`: pasta com
`pdata/<N>/` dentro; `hplc`: pasta terminando em `.D`/`.raw`) é uma amostra;
sem subpasta, tudo vira 1 classe só (nome da pasta raiz). Nenhum tem
`mae_id` real (agrupamento por réplica física) — mesma situação honesta já
documentada para `csv` genérico, `grouping_guarantee` não é elevado
artificialmente. `eem` é a exceção estrutural: fluorescência
excitação-emissão é uma MATRIZ 2D por amostra, achatada (flatten) em vetor
1D para caber no mesmo contrato de pipeline — funciona, mas perde a
estrutura espectral 2D explícita; a decomposição PARAFAC de verdade fica na
tela **[3.5] Técnicas especiais** (item 5), sobre a mesma pasta (ver a seção
5.7.1, que também trata do espalhamento Rayleigh/Raman). Testado ponta a ponta contra arquivo/pasta REAL de fabricante (SPC, `.sp`, RMN Bruker, HPLC Agilent com o 1º canal UV, EEM Horiba)
na máquina do mantenedor; os arquivos não vão no repositório e o desfecho está em `docs/qualidade/rodada_local_formatos_reais.json` (só na cópia do repositório).
`opus`, `gcms`, `hplc` (Agilent e Waters) foram conferidos também com arquivos reais **públicos** de outros projetos (cópias fora do repositório): 4 arquivos OPUS de FT-IR do repositório de dados do spectrochempy, mais um de background e um de série (uma absorbância bate com o valor que a suíte do spectrochempy espera, 2,72740 em 2303,87 cm⁻¹); 8 arquivos ANDI-MS de GC-MS do PyMassSpec (o TIC do arquivo e a soma das intensidades de cada scan batem, e a matriz tempo × m/z bate com um laço independente); e 12 pastas do repositório de testes do `rainbow-api` (Agilent `.D` com `.uv` DAD, `.ch`, FID, CAD, ELSD e MS; Waters `.raw`), onde o GUARACI devolve exatamente o que o `rainbow` lê. A decodificação do binário de Agilent e Waters é do `rainbow`: ela **não** foi comparada com o software do fabricante. Não são lidos: arquivo OPUS de série ("assembled", vários espectros num arquivo) e arquivo só de background, recusados com erro explícito (antes o background entrava como amostra); pasta Agilent MassHunter de alta resolução ou ICP-MS; e o arquivo `.dx` zipado do OpenLab (extraia a pasta `.D`). O leitor JCAMP-DX foi conferido com 124 arquivos públicos (a suíte oficial do JCAMP-DX e espectros de gás do NIST), e SPC, `.sp` e RMN Bruker com arquivos públicos de outros projetos.
Antes de confiar num formato com o seu instrumento, confira a leitura de uma amostra conhecida.

### 4.1 Como organizar os seus dados (o que o GUARACI espera, modo a modo)

O GUARACI **não exige nenhuma convenção de nome nos seus arquivos**. A tabela diz de onde vêm a classe e o grupo (a "amostra
física" a que as réplicas pertencem) em cada modo. O grupo importa: réplicas da mesma amostra precisam ficar do mesmo lado,
treino ou teste, senão a validação sai otimista. Sem grupo o GUARACI roda, mas avisa (auditoria `agrupamento`, crítica) e trata
os números como exploratórios.

| Modo | Estrutura da pasta ou do arquivo | A classe vem de | O grupo (réplicas) vem de |
|---|---|---|---|
| `dx` | `pasta/ClasseA/*.dx`, `pasta/ClasseB/*.dx` | nome da subpasta | `arquivo_grupos` (CSV `arquivo,grupo`) |
| `dx`, pasta sem subpastas | `pasta/*.dx` | 1ª parte do nome do arquivo, cortando em `-` ou `_` (`controle_01.dx` → `controle`; `dx_parte_classe` escolhe outra parte) | `arquivo_grupos` |
| `opus`, `spc`, `sp`, `gcms` | uma subpasta por classe; cada arquivo é uma amostra | nome da subpasta (sem subpasta, tudo vira 1 classe, o nome da pasta) | `arquivo_grupos` |
| `rmn` | `pasta/ClasseA/<amostra>/<expno>/pdata/<N>/` | nome da subpasta | `arquivo_grupos` |
| `hplc` | `pasta/ClasseA/<amostra>.D` (ou `.raw`) | nome da subpasta | `arquivo_grupos` |
| `eem` | `pasta/ClasseA/*.dat` | nome da subpasta | `arquivo_grupos` |
| `csv`, `multielementar` | uma linha por amostra; cabeçalhos numéricos = eixo (espectro), cabeçalhos com nome = variáveis nomeadas | `coluna_classe` | `coluna_grupo` |
| `imagem`, `hsi` | `pasta/Classe/Amostra/*.jpg` (ou cubos ENVI) | subpasta de classe | subpasta da amostra, detectada sozinha (garantia `high`); ou CSV de associação (`medium`) |

**O `##TITLE=` do JCAMP-DX é opcional.** O GUARACI reconhece uma convenção de título (`COD-DD-MM-AAAA[_AD-x-N%]_Tn`, com `COD` de 2 a 4
letras) que preenche espécie, teor e réplica (`mae_id`) sozinha. Se **nenhum** arquivo segue essa convenção, o que é o caso normal de
quem tem os seus próprios dados, o log diz só isso (`[INFO]`) e a classe vem da subpasta ou do nome. Se **só alguns** seguem,
os que não seguem viram grupos de 1 amostra (`orfao_<arquivo>`) e o log avisa quantos são. Dois cuidados com o nome do arquivo:
um trecho `N,NN%` (por exemplo `trat_5%_01.dx`) é lido como teor (`dx_teor_no_nome`); desligue se `%` for outra coisa no seu nome.

**Como informar o grupo sem renomear nada.** Crie um CSV com duas colunas, `arquivo` (nome do arquivo sem a extensão) e `grupo`
(o mesmo valor para as réplicas da mesma amostra), e aponte `arquivo_grupos` para ele (campo "Arquivo de grupos" na tela de Dados):

```csv
arquivo,grupo
amostra_01,A
amostra_02,A
amostra_03,B
```

Tudo isso vale para `guaraci run config.yaml`, para o assistente e para a versão web. Para ver o efeito antes de usar os seus dados,
`guaraci exemplo classificar` cria uma pasta de exemplo com esse formato.

---

## 4b Perfis de técnica e modo cego

### 4b.1 Perfil de técnica — trocar de técnica ou de amostra sem tocar em código

Tudo que é propriedade da **técnica de medida** (eixo, faixa, vocabulário), e não do **método**, vive num arquivo
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
guaraci --perfil=nir_dispersivo_1100_2498nm     # ou o caminho de um YAML seu
```

Também editável **dentro** do assistente interativo, em `[2] Dados` →
*Perfil de técnica* — não é só um flag de lançamento. Versões antigas não tinham esse
campo em `_CONFIG_SPEC`: não tinha entrada de menu e, mais
grave, **`[S] Salvar Perfil` seguido de `[L] Carregar` resetava
silenciosamente para `generico`**, mesmo se o usuário tivesse escolhido
outro perfil na sessão que gerou o `--perfil=` original (achado da
auditoria de acessibilidade). Corrigido: o campo agora
persiste em `config.yaml` como qualquer outro.

**Achado relacionado.** Verificando a extensão do
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

**Testes de propriedade (Hypothesis).** Hoje,
`tests/test_propriedades_hypothesis.py` gera automaticamente valores para
todo campo de `_CONFIG_SPEC` (não só os que alguém lembrou de escrever à
mão) e confirma que sobrevivem ao ciclo salvar/carregar — foi este teste
que achou o bug acima. Roda como dependência de desenvolvimento
(`pip install -e .[dev]`), nunca em produção.

**Por que o vocabulário importa.** Antes dos perfis, o model card de uma execução podia afirmar um vocabulário de outra técnica (por exemplo, uma frase de quantificação de adulterante de um tipo de amostra sobre outro). Nenhum número estava errado — a frase estava. O motor nunca lê esses termos para decidir nada; eles só aparecem em texto, e é essa separação que impede o vocabulário de um perfil de contaminar os resultados de outro.

**Técnica sem perfil não roda.** `UnknownProfileError` é levantado **antes de carregar qualquer dado**, com a lista de perfis existentes e a instrução de como escrever um novo. Rodar com a faixa e o vocabulário de outra técnica produziria números que parecem válidos e afirmações químicas que não são.

**Escrevendo um perfil novo:** copie `generico.yaml`, preencha, e passe o
caminho. Não precisa entrar no pacote nem fazer fork.

**`codigos_adulterante`:** a Identificação (§5)
lê o "segundo fator" de cada amostra do último token do `mae_id`
(`{código}-{data}-{letra}{teor}`) — a letra vinha de um mapa GLOBAL e
fixo (`dados_io.ADULTERANTE_NOME`, só as letras A/M/S, com nomes neutros),
então um perfil com sua própria convenção de letras nunca formava
combinação nenhuma. Declare `codigos_adulterante` no seu perfil (ex.:
`{"X": "quitosana", "Y": "amido"}`) para usar seu próprio mapa, sem
tocar em código-fonte — vazio (default) preserva o mapa global de
sempre. A ESTRUTURA do token (1 letra + dígitos no último segmento)
continua fixa; só o dicionário letra→nome é parametrizável.

### 4b.2 Perfil de aquisição de imagem + perfil combinado

Separado do perfil de técnica espectral (eixo, faixa e vocabulário) existe o perfil de
**aquisição de imagem** (como a foto foi capturada) — usado hoje pelo
mode `imagem` (`bancada.yaml`/`celular.yaml`/`scanner.yaml`, cada um
com `unidade_eixo="indice"`, resolução esperada, formatos aceitos e o
nível de garantia de agrupamento TÍPICO daquele fluxo de trabalho — só
informativo, o nível real é sempre decidido pelos dados, nunca pelo
perfil, ver §4). Carregado com o mesmo `load_profile` do perfil de
técnica.

Esses 3 são exemplos PRÉ-CADASTRADOS de conveniência, não um limite
rígido — uma técnica nova (câmera de drone, microscópio digital, etc.)
é reconhecida automaticamente assim que declara os mesmos campos
(resolução esperada/formatos aceitos/nível de agrupamento típico), sem
precisar entrar no pacote nem editar `perfil_matriz.py` (a
listagem `guaraci perfis` filtrada por "técnica" classifica pelo
CONTEÚDO do perfil, não por uma lista fixa de nomes).

**Combinar e salvar** (dentro do assistente interativo, `[2] Dados`,
depois de escolher tanto `Perfil de técnica` quanto `Perfil de aquisição
de imagem`): funde os dois com `perfil_matriz.combine_profiles`
(vocabulário/faixa/eixo sempre do perfil de técnica; resolução/formatos/garantia
típica sempre da técnica; `default_preprocessing` da técnica quando ela
declarar um) e grava o resultado como um YAML novo em
`~/.guaraci/perfis_matriz/<nome>.yaml`, pronto pra reusar digitando o
caminho no campo `Perfil de técnica` numa próxima sessão — nunca precisa
recombinar toda vez.

**Ao combinar um perfil de técnica ESPECTRAL (nm/cm⁻¹) com uma técnica
de imagem, o eixo herdado é o do perfil de técnica** (`apply_profile` só sobrescreve
`wn_min`/`wn_max` quando o perfil declara um `eixo_min`/`eixo_max`) —
usar diretamente um perfil como `visnir_400_1100nm` (400–1100nm) combinado
com `celular` produziria uma faixa que descarta TODAS as 18 variáveis
simbólicas de cor do mode `imagem` (achado da auditoria de
adaptabilidade). Combine com o perfil `generico` (sem eixo
declarado) quando a matriz não tiver perfil espectral próprio, ou ajuste
`wn_min`/`wn_max` manualmente depois de aplicar o perfil combinado.

### 4b.3 Modo cego — o padrão, e por quê

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

No modo cego, um erro do classificador se propaga para a quantificação — e é
**correto** que se propague: é o que aconteceria em produção. Quando não há
classificador ajustado, o mode reportado é `controle-forcado`, nunca `cego`:
um resultado de controle disfarçado de cego seria pior que um resultado de
controle assumido.

### 4b.4 O que o software nunca grava

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

**Remoção de raios cósmicos** (Raman; desligada por padrão):
`remover_raios_cosmicos: true` marca saltos de 1 a 3 pontos pelo *z-score*
modificado da primeira diferença (Whitaker e Hayes, 2018) e troca cada ponto
marcado pela média dos vizinhos limpos. Entra antes de qualquer preset e opera
espectro a espectro, sem vazamento entre *folds*. Ajustes:
`raios_cosmicos_limiar` (padrão 6, o valor do artigo) e `raios_cosmicos_janela`
(padrão 5 pontos de cada lado).

**Análise exploratória:** PCA (*scores* e *loadings*), HCA (dendrograma de
Ward sobre componentes principais).

**Classificação e discriminação:** PLS-DA, OPLS-DA (com S-*Plot*), DD-SIMCA
(com Cooman's *Plot*).

**Seleção de variáveis (Etapa 4)** — sempre executados: iPLS (por
intervalos), VIP ≥ 1, Selectivity Ratio (20% superior), sPLS-DA esparso.
VIP/SR/sPLS-DA usam **seleção aninhada (nested-CV)**: a
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

> **XGBoost é opcional e pode ficar indisponível.** É o único modelo deste
> comparativo que depende de um pacote externo com biblioteca nativa
> compilada; se o pacote estiver ausente, **ou presente mas bloqueado pelo
> sistema** (achado real: `xgboost.dll` barrada pelo Controle de Aplicativo
> do Windows/Smart App Control), ele é **pulado com um aviso no log**
> (`[Benchmark] XGBoost indisponivel neste ambiente: ...`, com a causa
> provável e a ação sugerida) — as demais técnicas (PLS-DA, SVM RBF, Random
> Forest, Gradient Boosting) continuam rodando normalmente, nunca travam por
> causa dele. Use `guaraci.model_registry.tecnicas_indisponiveis(n_opt, cfg)`
> para consultar o motivo programaticamente antes de rodar.

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
execução) a espectros novos, sem rodar o pipeline inteiro de novo. **Um modelo salvo por uma versão
anterior à correção do limite de Q-resíduos é recusado, com a mensagem "Retreine o modelo"**: o limite de aceitação dele
vinha do ajuste sobre os próprios dados de treino e não é comparável com o dos modelos atuais; nada é predito com ele.
Entrada:
CSV com colunas = número de onda (sem coluna de classe). Saída: classe
predita, confiança (%) e **dois diagnósticos complementares** de
confiabilidade:
- **Ajuste ao modelo PLS-DA** (colunas `T2`/`Q`/`aceito`) — o quanto a
  amostra se afasta do que o modelo de classificação capturou.
- **Domínio de aplicabilidade** (colunas `AD_*`, Jaworska, Nikolova-Jeliazkova
  e Aldenberg, 2005) — o quanto a amostra é um espectro atípico frente ao
  conjunto de calibração em geral, via T²/Q num PCA exploratório
  independente da classe. Reaproveita
  `chemometric_stats.applicability_domain_new_samples`; só aparece se
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
> `guaraci manual seguranca`.
>
> Em implantação pública (demonstrativo hospedado), o operador ainda pode
> definir **`GUARACI_DISABLE_MODEL_UPLOAD=1`** para desabilitar o *upload*
> de `.joblib` pela interface web por completo, aceitando só caminho local
> controlado pelo próprio operador. O *upload* de CSV de espectros
> permanece liberado (dado inerte, não executa nada ao ser lido).

**Figuras:** conjunto essencial por padrão (cerca de 8 a 10 figuras, a
depender do objetivo — seção 2.2) com opção de figuras detalhadas adicionais
(`figuras_detalhadas`). Formatos PNG/PDF/SVG, DPI configurável.

### 5.1 Transferência de calibração entre instrumentos

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
  superajustar com poucas amostras de transferência; por isso PDS é o método primário.

`apply_standardization(X_novo, transform)` aplica a transformação aprendida
a espectros novos do instrumento escravo antes de entrar no modelo
calibrado no mestre.

**Valores padrão:** `janela=5` e `alpha=0,001` são ponto de partida, não
universais, e ajustáveis por opção. Confira sempre o RMSEP no instrumento
escravo com e sem a transformação, nas suas próprias amostras de transferência.

**Limitações conhecidas** (ver seção 9 para a lista completa do projeto):
- **Quantas amostras de transferência**: com poucas amostras a regressão
  ridge por janela fica instável (poucas amostras relativas à largura da
  janela). Não há um mínimo teórico fechado; 15-20 amostras cobrindo a
  faixa de variação do analito é o que a literatura original recomenda.
- **`alpha` (regularização) e `janela` são sensíveis ao par de
  instrumentos** — os valores padrão (janela=5, alpha=0,001) são
  ponto de partida, não são universais. Um par de
  instrumentos com deslocamento maior/menor pode pedir janela mais larga ou
  `alpha` diferente; não há afinação automática hoje.
- **Pressupõe deslocamento predominantemente linear e local** (o que PDS
  corrige bem). Não corrige efeitos não-lineares fortes de detector, nem
  substitui uma calibração própria no instrumento de destino quando há
  amostras suficientes para isso.

### 5.2 Seleção de amostras de calibração — Kennard-Stone, Duplex, SPXY

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

**CLI** — menu principal, tecla `[K]` *Seleção de Amostras* (ao
lado do planejamento de coleta): pede um CSV com os espectros (1 amostra
por linha; separador `,` ou `;` detectado sozinho — `;` implica vírgula
decimal, como no Excel em português), opcionalmente uma coluna de
referência/teor (habilita SPXY), a **coluna que identifica a amostra
física** (Enter aceita a detectada: `mae_id`, `amostra`, `grupo`,
`sample_id`...), o método e a fração de calibração, e grava uma cópia do
CSV com uma coluna extra marcando `calibracao`/`validacao` por amostra. Só
separa/marca — o CSV original nunca é alterado.

A tela usa sempre as variantes `*_group_aware`: com a coluna de amostra
física e pelo menos 4 amostras distintas, réplicas nunca são separadas
entre calibração e validação. Sem essa coluna (ou com menos de 4 grupos) a
tela **avisa** que réplicas podem ser separadas. Versões antigas da tela
usavam as variantes sem grupo.

### 5.3 Planejamento de coleta

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

### 5.4 Auditoria de delineamento

Roda **por padrão em toda execução** (não é opt-in), logo após a
validação de integridade dos dados. Consolida checagens que antes viviam
espalhadas — antes espalhadas em várias partes do código:

| Checagem | O que verifica |
|---|---|
| `agrupamento` | `cfg.grouping_guarantee` — se a validação tem proteção real contra vazamento de réplica. |
| `confundimento_classe_sessao` | Se alguma classe está confinada a **1 única sessão de coleta** (`session_from_mae_id`) enquanto o dataset tem múltiplas sessões — deriva instrumental daquela sessão fica indistinguível de efeito de classe. |
| `duplicatas` | Reaproveita `pipeline.validate_input` — duplicatas exatas/aproximadas (risco de vazamento treino/teste). |
| `n_insuficiente` | Quantas sessões independentes cada classe tem, frente ao mínimo do gate conformal (`conformal.n_minimum_for_alpha`). |
| `faixa_validacao_uso` | Faixa de teor **observada** na calibração vs. faixa de trabalho **declarada** no perfil da matriz (`perfil_matriz.outside_working_range`) — extrapolação silenciosa. |
| `validacao_externa` | Informativo: os padrões do motor são ponto de partida, não universais, e não validam o seu estudo. |

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

**Fora do escopo desta versão (registrado, não implementado):** ordem de leitura correlacionada com o alvo (teor/classe). O método depende do timestamp de aquisição gravado no *audit trail* JCAMP-DX, que o parser deste pacote não lê de propósito (a proveniência não viaja com os resultados); portar isso exige mudança no parser, não só *wiring*.

### 5.5 Sentinela de deriva do domínio de aplicabilidade

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
`check_drift` testa **H0: a taxa de rejeição em produção é igual à taxa que a
PRÓPRIA calibração rejeita em validação cruzada** contra **H1: maior**, via
**teste exato de Fisher unilateral de duas amostras** (`scipy.stats.
fisher_exact`) — teste R2. A referência
(`ad_cv_rejeitadas`/`ad_cv_n`) é calculada no treino por
`chemometric_stats.ad_rejection_rate_cv` (group-aware por `mae_id`), gravada
no pacote de modelo e chega ao sentinela pelas colunas `AD_ref_cv_rejeitadas`/
`AD_ref_cv_n` da predição. **Por que não comparar com o 5% nominal**: um domínio
de aplicabilidade calibrado com n finito, em espectros reais, rejeita em
controle uma taxa diferente de 5% e diferente a cada calibração (efeito da
estimação de parâmetros da Fase I, Jensen et al. 2006, DOI
10.1080/00224065.2006.11918623); o teste contra 5% dá falso alarme acima do
nominal; R2 o reduz. **Custo de R2**:
a detecção da troca de instrumento pode levar até 1 lote a mais
que o teste contra o nominal, que acusa sempre no 1º. **Modelos salvos antes dessa correção** não têm a
referência: o sentinela cai no teste binomial contra o nominal, com aviso
`[teste LEGADO …]` na mensagem e `DriftAlert.teste == "binomial_nominal"` —
retreinar grava a referência. Não é um limiar cru tipo "2× o nominal", que
teria taxa de falso alarme dependente de `n` sem justificativa formal.

**Defaults justificados, não escolhidos a dedo:**
- `n_minimo` (abaixo disso, não testa — sem poder estatístico): por
  padrão `conformal.n_minimum_for_alpha(alpha_nominal)` — o MESMO mínimo
  prático já usado em todo o projeto para o gate conformal atingir aquele
  *alpha* (ex.: 0,05 → 19). Reaproveitado por consistência, não recalculado.
- `significancia` do teste de deriva: 0,05 por padrão — o mesmo *alpha*
  nominal usado em todo o projeto para os próprios gates (DD-SIMCA, AD,
  conformal).

Verificado por simulação (não só "não quebra"): 200
repetições sintéticas com calibração nova a cada repetição (falso alarme
2,5% com R2 contra 25,5% do teste antigo, em processo de caudas pesadas). Os
limiares padrão são ponto de partida, não universais, e ajustáveis por opção.

**Persistência e integração:** `save_state`/`load_state` (JSON)
permitem que a sentinela sobreviva entre execuções — uso real (LIMS
chamando o pipeline ao longo de dias/semanas) não mantém um processo
Python vivo o tempo todo. Orquestração unificada em
`sentinela_deriva.hook_apos_predicao` (carregar → registrar → salvar →
checar deriva), chamada dos DOIS pontos de predição — CLI (menu `[B]`
*Predição em Lote*) **e a aba web Predição** (antes,
só o CLI atualizava a sentinela) — atualizam automaticamente a sentinela
persistida ao lado do modelo (`<modelo>.joblib.sentinela.json`) a cada
rodada, quando o pacote tem artefatos de domínio de aplicabilidade, e
mostram o status/alerta no resumo. Na aba web, um modelo carregado por
caminho local usa esse caminho; um modelo enviado por upload usa um
caminho estável POR SESSÃO do navegador (a sentinela acumula entre
predições repetidas na mesma sessão).

**Figura de mérito analítica dedicada (Quantificação):**
`figS3_merito_regressao.png` — dois painéis lado a lado: LOD/LOQ por espécie
e seletividade média por espécie, seguindo Valderrama, Braga e Poppi (2009).
Antes desta funcionalidade, esses valores só apareciam como tabela de texto
no `resumo_modelo.txt`/`model_card.md`.

**Relatórios:** PDF, Word (`.docx`), Excel (4 abas + Benchmark quando houver), LaTeX e PowerPoint,
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

### 5.6 Conjunto conforme de classificação, exportação portátil e explicabilidade fora do treino

**Conjunto de predição conforme para classificação** (`conformal.
conformal_margin_classification`/`conformal_prediction_set`, LAC —
Sadinle, Lei & Wasserman 2019): até aqui, classificação era argmax
pontual, sem nenhuma noção de incerteza por classe — inconsistente com
o rigor já aplicado à Quantificação (T1 acima). Agora, quando o holdout
externo tem grupos independentes suficientes (>=19 para *alpha*=0,05),
a predição em lote (CLI e aba web) traz também o CONJUNTO de classes
plausíveis para cada amostra (coluna `classes_plausiveis` no CSV),
calibrado no mesmo holdout nunca visto pelo ajuste do modelo — pode
conter mais de uma classe (ambiguidade real) ou nenhuma (quando nem a
classe mais provável atinge a cobertura pedida). Sem grupos suficientes,
a coluna traz `NAO_VALIDADO`, nunca um conjunto fabricado.

**Explicabilidade fora do treino**: quando o holdout existe, o SHAP
(Auto-Benchmark) passa a explicar o HOLDOUT em vez de uma subamostra do
próprio treino — a figura declara `avaliado em: holdout` ou `avaliado
em: treino (subamostra, otimista)`, nunca omitindo qual dos dois foi
usado.

**Exportação portátil de modelo** (`model_export.py`): ao lado do
`<modelo>.joblib`, a execução grava automaticamente
`<modelo>.portatil.json` — formato JSON puro, SEM pickle, para os
presets de pré-processamento suportados (`snv_sg_mc`/`msc_sg_mc`/`mc`/
`autoscaling`). Ao contrário do `.joblib` (que executa código
arbitrário ao carregar — risco já tratado em outros pontos do produto,
ver seção de Segurança), o JSON não executa nada ao ser lido, e um
consumidor externo mínimo (outra linguagem, sem `scikit-learn`/
`joblib`) consegue reaplicar o modelo a partir só dos números do
arquivo. Escopo deliberadamente menor que o `.joblib` completo: cobre
só pré-processamento + PLS-DA/PLS-R + classes, não DD-SIMCA por
espécie/ensemble de identificação/domínio de aplicabilidade/conjunto
conforme (esses continuam só no `.joblib`).

### 5.7 Técnicas Avançadas (ASCA, EPO/GLSW, MCR-ALS, fusão multibloco)

Quatro técnicas exploratórias/diagnósticas, **não integradas ao pipeline de
treino automático** (rodam à parte, sob demanda) — acessíveis pela tecla
**`[T]`** da CLI ou pela página **Advanced Techniques** (grupo Analisar) do
aplicativo web. ASCA/EPO-GLSW/MCR-ALS usam o dataset já configurado na
aba/menu Dados; fusão multibloco pede 2 pastas/arquivos na hora (é a única
que estruturalmente precisa de 2 datasets).

- **ASCA** (ANOVA-Simultaneous Component Analysis, Smilde et al. 2005) —
  decompõe os espectros por fatores do delineamento (espécie sempre
  disponível; qualquer coluna de metadado categórica de baixa cardinalidade
  quando o formato de entrada a fornece — ex.: sessão, adulterante) e testa
  a significância de cada um por permutação, *group-aware* por `mae_id`
  quando disponível. Escopo honesto: decomposição por efeito MARGINAL,
  exata só para fatores ortogonais — `ss_desbalanco` grande sinaliza fatores
  correlacionados demais (a extensão ASCA+ que corrigiria isso não está
  implementada, ver seção 9).
- **EPO / GLSW** (External Parameter Orthogonalisation / Generalized Least
  Squares Weighting) — remove (EPO) ou atenua (GLSW) a variação de um fator
  de perturbação CONHECIDO (sessão, espécie-hospedeira, instrumento),
  estimada de espectros diferença entre amostras que compartilham a
  condição a preservar mas diferem no fator a remover. Diagnóstico
  exploratório: **não se aplica** quando o fator de perturbação é colinear
  com o alvo científico (ver seção 9).
- **MCR-ALS** (Multivariate Curve Resolution — Alternating Least Squares) —
  resolve a matriz espectral em perfis puros de concentração/espectro.
  Puro (sem referência) ou com **restrição de correlação** (Bayat et al.
  2020) quando há teor/concentração conhecido em parte das amostras: ancora
  um componente-alvo escolhido a esses valores de referência a cada
  iteração. Interpretativo — **não substitui PLS-R** para quantificação com
  garantia (ver seção 9).
- **Fusão multibloco** — funde ≥2 blocos medidos na MESMA amostra física
  (ex.: FT-NIR + MIR) numa única regressão PLS-R, cada bloco
  pré-processado SEPARADAMENTE antes de concatenar (nunca a matriz já
  concatenada — corromperia a fronteira espectral entre técnicas). A fusão
  **nem sempre supera** o melhor bloco sozinho — compare sempre contra cada bloco treinado isoladamente
  antes de adotar (ver seção 9).

Lógica real (testável sem interface) em `tecnicas_avancadas.py` — fonte
única entre CLI e web, mesmo padrão da auditoria de delineamento.

---

#### 5.7.1 EEM: espalhamento Rayleigh/Raman e PARAFAC (não-negativo por padrão)

**Espalhamento.** Uma EEM bruta traz diagonais de espalhamento Rayleigh (1ª ordem, em = ex; 2ª ordem,
em = 2·ex) e Raman da água (deslocamento de 3400 cm⁻¹, 1ª e 2ª ordem) que não são fluorescência.
Sobre a EEM bruta o PARAFAC não serve (nos dados sintéticos de verificação, erro de reconstrução de 0,82
e correlação escore × concentração de até 0,51; no PCA, o PC1 é a diagonal de Rayleigh). A chave
`eem_espalhamento` trata isso na leitura:

| valor | o que faz |
|---|---|
| `nenhum` (padrão) | não toca no dado (resultados existentes não mudam); **avisa** se a EEM bruta tem espalhamento forte |
| `nan` | as bandas e a região sem sinal (em abaixo de ex) viram dado **faltante**: o PARAFAC as ignora (ALS com dados faltantes, por EM); no classificador (vetor achatado) essas colunas valem 0 em todas as amostras e são descartadas como constantes |
| `zero` | viram 0, como dado (o PARAFAC as ajusta como zeros: não recomendado) |
| `interp` | as bandas são preenchidas por interpolação linear 2D dos vizinhos válidos (Zepp et al., 2004, *Marine Chemistry* 89:15); o PARAFAC ainda trata a região como faltante |

`eem_espalhamento_largura_nm` dá a largura **total** (nm) da banda removida por ordem, na ordem Rayleigh 1ª,
Rayleigh 2ª, Raman 1ª, Raman 2ª (vazio = 30 nm; um número vale para todas; a última largura vale para as que
faltarem). `eem_raman: true` inclui o Raman da água; é opcional porque a banda cruza a emissão de
fluoróforos reais (a tirosina, por exemplo, cai sobre o Raman de 1ª ordem). Larguras menores que as caudas
do espalhamento deixam resto de espalhamento e pioram o PARAFAC: confira o erro de reconstrução e o CORCONDIA.
NaN já presentes nos arquivos (convenção de exportadores) são tratados como dado faltante em qualquer modo.
Lembre-se de aplicar a **mesma** configuração ao ajustar e ao prever amostras novas.

**PARAFAC não-negativo.** Desde a v1.0.0 o PARAFAC da tela [3.5] usa restrição de não-negatividade
(HALS) e reporta o CORCONDIA (consistência do núcleo, Bro & Kiers, 2003). Versões anteriores e outros
programas sem restrição (ALS livre) podem dar **sinais e ordens diferentes** dos fatores (um perfil inteiro
negativo, escore negativo para um fluoróforo presente); os valores dos perfis também podem diferir. Quem
quiser o ALS livre define `parafac_nao_negativo: false` (os sinais dos perfis são então normalizados para soma
positiva, só por convenção). A tela, `parafac_resumo.json`, `parafac_resumo.txt` e a figura registram
`restricao` (`nao-negativo` ou `nenhuma`), o algoritmo e a nota correspondente, em português e inglês.
Escolha o número de componentes pelo maior valor que mantém o CORCONDIA alto (> ~60–80 %); com dados
faltantes o CORCONDIA usa o tensor com as células faltantes preenchidas pelo modelo.

---

### 5.8 Perfil multielementar (ICP-OES / ICP-MS) e ANOVA sobre a unidade independente

> **Selo: EM VALIDAÇÃO.** Mecanismo verificado em testes com dados sintéticos
> (leitura, censura, QC e ANOVA sobre a unidade). Com poucas unidades
> independentes por grupo o poder é baixo.

**O que é.** Um modo de entrada (`modo_entrada: multielementar`) para a
tabela **amostra × elemento** exportada do ICP-OES ou ICP-MS. Não é
espectro: cada coluna é um elemento (`Cd 111`, `Cd 228.802`, `Ca`). OES e MS
usam o mesmo perfil.

**O que o arquivo precisa ter** (CSV; separador `;` ou `,` e decimal
detectados sozinhos):

| Campo da configuração | O que é | Obrigatório |
|---|---|---|
| `arquivo_csv` | o CSV exportado do instrumento | sim |
| `icp_unidade` | unidade das concentrações (`mg/kg`, `ug/L`, ...; `mista` se as colunas têm unidades diferentes ou a fonte não declara) | **sim** — sem ela o arquivo é recusado |
| `coluna_classe` | o grupo a comparar (origem, produtor, sistema) | sim |
| `coluna_grupo` | a **unidade independente** (amostra física, lote, fazenda) | **sim** — sem ela a classificação e a ANOVA são recusadas |
| `icp_coluna_amostra` | código de cada linha (nunca é gravado em disco) | sim (padrão `Amostra`) |
| `icp_coluna_tipo` | `amostra`, `branco`, `qc`, `crm`, `ld` ou `crm_cert` (ver abaixo) | não (vazio = todas são amostras) |
| `icp_coluna_sessao` | sessão/corrida analítica | não (com QC, mede o efeito de sessão) |
| `icp_padrao_interno` | elemento usado como padrão interno | não (relatório de deriva) |

**Pela linha de comando** (`guaraci run config.yaml`; também no assistente, no menu de dados, e na interface web com upload do arquivo). Exemplo mínimo:

```yaml
modo_entrada: multielementar
arquivo_csv: dados/perfil_icp.csv
coluna_classe: Origem          # o grupo a comparar
coluna_grupo: Lote             # a unidade independente (obrigatória)
icp_unidade: mg/kg             # ou 'mista'
icp_coluna_amostra: Amostra
icp_politica_censura: substituir   # ou excluir
pre_processamento: autoscaling     # obrigatório: o padrão (MSC/SG) é espectral e é recusado
```

**Linhas especiais na coluna de tipo** (todas fora do modelo; só as `amostra` entram):

| `Tipo` | O que carrega | Para quê |
|---|---|---|
| `branco` | brancos analíticos | o relatório compara a média do branco com o LD (alerta se acima) |
| `qc` | controle de qualidade/padrão repetido | deriva do padrão interno e efeito de sessão |
| `crm` | material de referência medido (uma ou mais linhas) | recuperação |
| `ld` | uma linha com o limite de detecção de cada elemento | resolve `<LD` sem número e a comparação branco × LD |
| `crm_cert` | **uma** linha com o valor **certificado** de cada elemento do material de referência (mesma unidade das concentrações) | denominador da recuperação do `crm`: recuperação = média medida ÷ valor certificado |

Sem a linha `crm_cert` o programa **não inventa** recuperação: o relatório
mostra "recuperação do MRC: não verificada (sem valor certificado)". Mais de uma
linha `crm_cert` é erro (qual valia?). O valor certificado vem do certificado
do seu material; o programa não traz valores de nenhum material.

Nome de elemento ambíguo (duas colunas que viram o mesmo elemento) é **erro**,
nunca adivinhação. Pré-processamento: só `Autoscaling` (recomendado) ou
`Mean-centering`; MSC/SNV/Savitzky-Golay são recusados (são espectrais).
Ordem fixa: censura → `log10(1+x)` opcional (`icp_log`) → autoescala ajustada
só no treino de cada dobra.

**Célula vazia não é censura.** Vazio = ausência (não há limite a substituir): a linha sai nas duas políticas e a contagem aparece à parte (`linhas_descartadas_por_ausencia`) no relatório.

**Valores censurados** (`<LOD`, `<LOQ`, `ND`, `BDL`, `<0,05`): a fração
censurada por elemento é registrada. Elemento acima de `icp_limiar_censura`
(padrão 50%) sai do modelo e é listado. Duas políticas: `substituir` (por
LOD/√2; o LOD vem do próprio token, `<0,05`) e `excluir` (a linha com valor
censurado sai). **A análise é sempre refeita com as duas** e o relatório diz
se a conclusão muda — nenhuma é escolhida pelo resultado.

**O que sai** (além da classificação PLS-DA de sempre, com validação agrupada
pela `coluna_grupo`):
- `Tabelas/multielementar_anova_por_elemento.csv` — ANOVA por elemento sobre a
  unidade independente: F, p, p corrigido por FDR entre elementos, η² e ω² com
  IC bootstrap sobre unidades, Brown-Forsythe (coluna `levene_p`: Levene com
  centro = mediana), avisos;
- `Tabelas/multielementar_comparacoes.csv` — todas as comparações par a par
  por **Welch ANOVA + Games-Howell** (política fixa: a escolha automática
  Tukey × Games-Howell inflava o erro por família para ~0,25 em simulação),
  com IC simultâneo de nível 1 − α e p ajustado;
- `Relatorios/multielementar.json` — política de censura, sensibilidade,
  QC (recuperação de material de referência, brancos, deriva do padrão
  interno), efeito de sessão (ANOVA com as linhas de QC);
- `Graficos/fig_multielementar_medias.png` — média por grupo dos elementos
  com diferença (sem letras de agrupamento);
- `Relatorios/` PDF/DOCX/XLSX/TEX (`guaraci relatorio PASTA`): para a execução multielementar, a seção "Perfil multielementar" traz a tabela de ANOVA por elemento (F, p, p corrigido por FDR, η² com IC), o método e a correção usados, as comparações par a par significativas, a sensibilidade à censura e os alertas de QC (deriva do padrão interno, branco acima do LD, recuperação do MRC fora de 80–120% ou não verificada). O relatório não mostra faixa espectral nem SNR (não se aplicam);
- selo "EM VALIDAÇÃO" no model card; o modelo exportado (`.portatil.json`) leva
  unidade, elementos, política de censura e transformação.

**ANOVA: quando usar e quando não.** Use para comparar a média de **uma
resposta escalar por unidade independente** (concentração de um elemento,
teor, escore de LV, métrica por sessão) entre grupos. O módulo agrega réplicas
por unidade **antes** do teste e **recusa**: ANOVA sem unidade declarada,
replicas, espectros e **dobras de validação cruzada** (as dobras compartilham
treino e não são independentes; para comparar modelos use o teste t corrigido
de Nadeau-Bengio, já usado no Auto-Benchmark). Na política `auto`, variâncias
desiguais (Brown-Forsythe, isto é, Levene com centro = mediana) → Welch ANOVA +
Games-Howell; senão Tukey (balanceado) ou Tukey-Kramer. Com bloco o pós-teste
supõe variâncias iguais, e a saída avisa quando a razão de variâncias passa de 4
ou o Brown-Forsythe rejeita.

**Por que a unidade importa — medido.** Simulação sob H0 (2000 execuções,
réplicas correlacionadas): tratando réplicas como independentes, 57,0% de
falsos positivos; agregando por unidade, 4,8% (IC95 3,9–5,8%).

**Limites.** Só CSV (XLSX e formatos proprietários de instrumento não);
razões isotópicas fora do escopo; poder baixo com menos de 6 unidades por
grupo (medido: 0,35 com 6 por grupo para efeito de 1 desvio-padrão). Com o Games-Howell (o padrão do `guaraci anova`), o erro por
família chegou a 0,27 no pior cenário simulado (grupo de poucas unidades com a maior variância; nominal 0,05): o programa avisa quando
algum grupo tem menos de 6 unidades e as comparações par a par desses grupos devem ser lidas como exploratórias.

### 5.9 Escolha do teste: diagnóstico de premissas, Kruskal-Wallis + Dunn

> **Selo: EM VALIDAÇÃO.** Conferido contra SciPy e contra implementações
> ingênuas escritas no teste; sem exemplo publicado conferido na fonte.

`guaraci anova` aceita `--metodo` (o padrão foi escolhido por simulação: `docs/qualidade/politicas_anova_simulacao.md` (só na cópia do repositório); **nenhuma política cumpre o erro por família ≤ 0,06 em todos os cenários**, e a falha se concentra em variância alta no grupo de poucas unidades, ver `guaraci manual limitacoes`, §2):

| `--metodo` | O que roda | Quando |
|---|---|---|
| `games-howell` (**padrão**; com `--bloco` o padrão é `auto`) | Welch ANOVA + Games-Howell | uso corrente: é a política que menos passa do limite de erro por família na simulação |
| `auto` | ANOVA; o Brown-Forsythe (Levene com centro = mediana) decide Tukey × Games-Howell **e o resultado diz qual foi** | quando quiser a regra em duas etapas (infla o erro por família quando o grupo pequeno tem a maior variância); com `--bloco` roda a ANOVA com bloco e **avisa** se as variâncias parecem desiguais |
| `tukey` | ANOVA clássica + Tukey/Tukey-Kramer | só a pedido: variâncias parecidas e n igual |
| `kruskal-dunn` | Kruskal-Wallis + Dunn (`--ajuste holm` padrão, ou `bh`) | forma ruim, valores extremos, poucas unidades; compara distribuições, não médias |
| `recomendar` | mostra o diagnóstico, a razão em uma linha e a sugestão, mas **roda o padrão** e diz o que foi sugerido | dúvida sobre as premissas |

**O programa nunca troca o método em silêncio.** O que foi rodado por
resposta fica no resumo e em `config_efetiva.yaml`; se você pediu um método
e o diagnóstico sugeria outro, a saída diz isso e **mantém o seu pedido**. Com `recomendar` vale o mesmo: a sugestão aparece, o padrão roda, e para usar a sugestão você pede o método.

**Diagnóstico de premissas** (por resposta): resíduos, gráfico
quantil-quantil (na figura `figuras/anova_<resposta>.png`), Shapiro-Wilk,
Levene (média) e Brown-Forsythe (mediana), razão entre a maior e a menor
variância por grupo, assimetria e curtose dos resíduos padronizados por grupo.
Regra de recomendação, nesta ordem (estritamente acima do limite troca; no
limite, não): |assimetria| > 1 ou curtose em excesso > 3 → Kruskal-Wallis/Dunn;
razão de variâncias > 4 ou Brown-Forsythe p < alfa → Games-Howell; senão
ANOVA/Tukey. É regra prática, não teste. A curtose amostral de caudas pesadas é
instável (muda muito com a amostra): por isso o texto manda **olhar o
gráfico quantil-quantil**.

**Quanto confiar na recomendação (simulação com verdade conhecida, 2000
repetições por cenário, `scripts/simular_recomendacao.py`, só na cópia do repositório;
tabelas completas em `docs/qualidade/anova/recomendacao_simulacao.md`, só na cópia do repositório).**
Três grupos, nível 0,05; "acerto" é recomendar o método que o mecanismo
gerador pede.

| cenário | acerto | erro tipo I por família do método recomendado (Tukey fixo) |
|---|---|---|
| normal, variâncias iguais, n = 10 | 87% | 0,051 (0,050) |
| normal, variâncias iguais, n = 4 | 49% | 0,055 (0,053) |
| normal, variâncias iguais, n desigual (5/10/25) | 82% | 0,072 (0,053) |
| normal, variância 2:1, n = 10 | 70% | 0,053 (0,052) |
| normal, variâncias 1:10, n = 10 | 92% | 0,049 (0,065) |
| normal, variâncias 1:10, n desigual (5/10/25, maior variância no menor grupo) | 88% | 0,089 (0,257) |
| lognormal (assimétrica), n = 15 | 92% | 0,048 (0,034) |
| t com 3 gl (caudas pesadas), n = 15 | 14% | 0,046 (0,040) |

O que isso quer dizer, sem enfeite: (1) onde o Tukey infla (variâncias muito
diferentes com n desigual) a recomendação reduz o erro de 0,257 para 0,089,
mas **não o leva a 0,05**; sempre rodar Games-Howell ficaria em 0,060 nesse
cenário, ao custo de menos poder quando as premissas valem (poder de 0,58
contra 0,73 no desenho 5/10/25 com variâncias iguais). (2) Com n muito
pequeno a razão de variâncias dispara por acaso: com n = 4 por grupo
a regra recomenda Games-Howell em 51% das vezes mesmo com variâncias iguais
(conservador, mas perde poder). (3) Com caudas pesadas moderadas a regra quase
nunca sugere o Kruskal-Wallis (14%); nesse cenário o Tukey manteve o erro (0,040), e
o Kruskal-Dunn teria mais poder (0,99 contra 0,94). (4) O erro por família do
método recomendado fica entre 0,046 e 0,055 em seis dos oito cenários da tabela e chega a
0,072 e 0,089 nos outros dois (n desigual): escolher o método pelos próprios dados
tem custo. Os limiares (razão 4, |assimetria| 1, curtose 3) são escolha do
projeto e **não foram ajustados a essa simulação**; o resultado diz o quanto
a regra vale, não que ela seja ótima.

**Shapiro é informação, não porteiro**: com n pequeno tem pouco poder e com n
grande rejeita por desvios mínimos. Testar normalidade para depois escolher o
teste não é recomendado como regra automática (Rochon, Gondan & Kieser 2012,
dois grupos, doi:10.1186/1471-2288-12-81). O Tukey é razoavelmente robusto a
pequenos desvios com n igual; com variâncias desiguais use Games-Howell.

**Kruskal-Wallis + Dunn.** H com correção de empates (`scipy.stats.kruskal`);
tamanho de efeito ε²<sub>H</sub> = H/(n−1); Dunn (1964,
doi:10.1080/00401706.1964.10490181) com postos globais e correção de empates
na variância, z por par, p bilateral e ajuste de Holm (padrão) ou
Benjamini-Hochberg — a regra aparece no relatório. **O Kruskal-Wallis
compara distribuições (dominância estocástica) e só equivale a "diferença de
medianas" se as formas forem parecidas.** Os mesmos recusos do ANOVA valem
(unidade obrigatória; réplicas, espectros e dobras de CV não são unidades).

**Saídas** (`--saida`): `tabelas/anova_por_resposta.csv`,
`tabelas/anova_comparacoes.csv`, `relatorio/anova_relatorio.md|html|tex`
(tabela, letras de grupos homogêneos, frase obrigatória, figuras por caminho
relativo), `figuras/*.png` (`--sem-figuras` desliga) e `config_efetiva.yaml`.
`--idioma en` traduz as mensagens e o relatório.

### 5.10 VIP com estabilidade

> **Selo: EM VALIDAÇÃO.** Conferido contra um PLS1 por NIPALS escrito na mão
> (formula de Wold, Sjöström & Eriksson 2001,
> doi:10.1016/S0169-7439(01)00155-1) e pela propriedade Σ VIP² = nº de
> variáveis; dados sintéticos.

    guaraci vip TABELA.csv --classe COL --grupo COL [--componentes 2]
    guaraci vip TABELA.csv --resposta COL --grupo COL

O VIP é recalculado em cada dobra de treino de uma validação agrupada
(`StableStratifiedGroupKFold`: réplicas do mesmo `--grupo` ficam juntas),
repetida com sementes diferentes. Para cada variável sai o VIP com todos os
dados, o VIP médio das dobras com intervalo 2,5–97,5%, e a **frequência com
que VIP > limiar** (padrão 1): estável acima ≥ 80%, estável abaixo ≤ 20%,
instável no meio. Sem `--grupo` o comando roda, mas **avisa** que as réplicas
vazam entre treino e teste e o resultado fica otimista.

**VIP > 1 é regra prática, não teste de significância.** Escolher variáveis
olhando o VIP e depois validar o modelo com os mesmos dados da escolha dá
resultado otimista; por isso `guaraci vip` **não seleciona nem reduz
variáveis** — a seleção aninhada (seção 5) é o caminho para isso.

### 5.11 Telas, teclas, ajuda e assistentes (CLI)

Menus, `--help`, ajuda, assistentes e esquema do config saem de **um catálogo**
(`guaraci.catalogo`): cada opção diz o que faz, quando usar e quando não usar.
Evidência de validação não aparece na descrição; fica no assistente de confiança.

**Toda tela** mostra a trilha (`Menu › Grupo › Tela`), o que faz, o que você vai
precisar e o mesmo rodapé: `? ajuda | G Guaraci | I idioma | C confianca | R rodar | V voltar`.
`v`, `0`, `q` e Enter voltam (sair do programa é só `q` no menu principal); `c` abre o assistente de
confiança; `g` (ou `a`) abre o assistente. `?` **pergunta primeiro qual opção** você quer
entender: digite o número (ou parte do nome) e aparece só a ajuda daquela opção (num campo:
Campo, Faz, Use quando, Nao use, Impacto, Padrão, Faixa e Exemplos); uma letra (`G`, `I`, `C`,
`R`, `V`, `+`, `M`, `Q`) explica a tecla; Enter mostra o painel da tela inteira (Tela, Onde,
Faz, Use quando, Depois, Precisa de e a lista das opções). `?N` abre direto o detalhe da
opção N. No menu principal, `?` seguido do número do grupo lista as telas dele, Enter explica
os grupos e as teclas, e `A` abre o índice de todos os campos.
Em telas de fluxo (planejamento, seleção, auditoria, predição, HSI), `?` vale em qualquer
pergunta e mostra o painel da tela sem perder a resposta em andamento.

O menu é organizado **pelo objetivo** (a ordem abaixo é a do catálogo; "avançado" = some no
Modo Básico, mas a tecla continua funcionando):

| Grupo do menu | Telas |
|---|---|
| Começar | `R` Rodar análise · `E` Exemplo guiado · `P` Perfis prontos · `G` Assistente |
| Planejar e preparar dados | `F` Planejar, selecionar e auditar · `J` Planejamento de coleta (avançado) · `1` Projeto · `2` Dados · `8` Técnica analítica · `9` Codificação DX (avançado) · `3` Pré-processamento · `K` Seleção de amostras (avançado) · `U` Auditoria de delineamento (avançado) |
| Explorar | `T` Técnicas especiais (avançado): ASCA, EPO/GLSW, MCR-ALS, fusão multibloco, PARAFAC, COW |
| Classificar ou quantificar | `4` Modelagem · `5` Validação · `6` Benchmark e Monte Carlo (avançado) · `B` Predição em lote |
| Comparar grupos | `W` Comparar grupos (ANOVA) |
| Monitorar processos | `D` Monitorar lotes |
| Imagens | `X` Imageamento hiperespectral (avançado) |
| Gráficos e relatórios | `7` Visualização · `Z` Relatórios de uma execução |
| Qualidade e confiança | `C` Assistente de confiança |
| Ajuda | `?` Ajuda · `O` Sobre · `H` Hardware (avançado) |

**Perfis prontos (`P`)**: duas escolhas, objetivo (explorar, classificar, quantificar...) e rigor
(rápido, padrão, publicação...); o programa mostra o que vai mudar e o tempo medido no exemplo
sintético, e pergunta antes de aplicar. A tela `F` encadeia planejar a coleta → selecionar
amostras → auditar o delineamento (`p` pula uma etapa).

Mudanças de tecla: Sobre passou de `A` para `O`; em Dados, fundir perfis passou de `C`
para `F`; em Perfis, salvar passou de `S` para `W`; em Visualização, transparência passou
de `A` para `T`; mostrar opções avançadas passou de `V` para `+`; em Codificação DX e
Sobre os painéis agora são `1` a `4`. O Modo Básico esconde as telas avançadas do menu
(as teclas continuam funcionando) e o modo de entrada esconde as opções que não se aplicam
a ele (`+` mostra todas).

**Comandos novos** (`guaraci --help` lista todos, agrupados como o menu):
- `guaraci anova TABELA.csv --resposta COL --grupo COL --unidade COL`: ANOVA sobre a unidade
  independente (padrão Welch + Games-Howell; Tukey ou Tukey-Kramer a pedido; FDR entre
  respostas; tamanho de efeito). **Sem `--unidade` o comando recusa** (código 2).
- `guaraci exemplo [CENARIO] [PASTA] [--rodar]`: gera dado sintético pequeno e o `config.yaml`
  do cenário (`explorar`, `classificar`, `quantificar`, `multielementar`, `comparar`, `imagem`,
  `hsi`, `transferir`, `monitorar`); `guaraci run PASTA/config.yaml` roda de ponta a ponta
  (`--rodar` faz os dois).
- `guaraci monitorar --referencia REF.csv --novos NOVOS.csv --lote COL`: monitoramento
  multivariado (T² e Q com limites por PCA calibrados na referência; julgamento por lote). Alarme
  diz que o processo mudou, não a causa. Tela `D`.
- `guaraci transferir --mestre M.csv --escravo E.csv --novos N.csv [--metodo pds|ds]`: transfere
  espectros do instrumento escravo para o espaço do mestre (amostras de transferência medidas nos
  dois); mostra o erro antes e depois. O `--alpha` (regularização do ridge) é escolhido por
  validação, deixando uma amostra de transferência de fora, quando você não o passa; um valor fixo
  não serve a todas as escalas (em absorbância, `alpha = 1` zerava a transformação). Com poucas
  amostras de transferência (menos de 3) a validação é impossível e o programa usa um valor quase sem
  regularização.
- `guaraci relatorio PASTA_DA_EXECUCAO [--formato pdf,docx,xlsx,tex,pptx | todos]`: relatório de uma
  execução já feita (extras `[reports]`). Tela `Z`.
- `guaraci confianca [METODO]`: selo (`validado`, `em validação`, `limitado`, `sem validação
  pública`, `informativo`), evidência, limites e como reproduzir; texto gerado do registro.
- `guaraci manual [TOPICO|SECAO]`: mostra, sem precisar do repositório, os documentos que acompanham o
  programa instalado (`manual`, `seguranca`, `limitacoes`, `suporte`) ou uma seção do manual
  pelo número (ex.: `guaraci manual 5.8`). Toda remissão a "seção N" nas mensagens do programa se lê assim.
- `guaraci ajuda [TOPICO]`, `guaraci buscar TERMO`, `guaraci assistente PERGUNTA`: respostas do
  catálogo, sem rede e sem LLM. O assistente também está na tecla `a` (rota recomendada,
  explicar, sugerir, diagnosticar, buscar); ele nunca aplica nada sozinho.
  **Dúvidas de quimiometria:** o assistente responde por regra, no gabarito *isto significa / por que
  acontece / o que fazer agora / onde ver na CLI*, a 12 perguntas comuns: Q2 baixo; R² alto com Q2
  baixo; sobreajuste; quantas variáveis latentes; qual pré-processamento; T² ou Q altos; classe nova;
  alarme de deriva; ANOVA ou Tukey; valores abaixo do limite no ICP; poucas amostras; réplicas em
  treino e teste. Fora dessas regras ele diz **"não sei responder isso com segurança"**, lista o que
  cobre e aponta a ajuda `?` e este manual (seções 5.11 e 9); ele não inventa resposta. As regras
  vivem em `guaraci/catalogo_duvidas.py`. Não há modelo de linguagem: resposta livre fica para
  a v1.1+.

**Gráficos**: `graficos: padrao` (padrão) gera só as figuras essenciais do objetivo (explorar 4,
classificar 4, quantificar 3); `graficos: todos` ou `--mais-graficos` gera todas; uma lista de ids
gera só essas (`guaraci buscar grafico` mostra cada id entre parênteses). Os gráficos de
etapas que você liga (benchmark, OPLS-DA, SHAP, Monte Carlo...) saem em qualquer seleção.

**Ajuste automático por contexto** (`ajuste_automatico`, ligado por padrão;
`--sem-ajuste-automatico` desliga): só compatibilidade, com linha `Ajustado: X → Y. Motivo: Z.
Desfazer: ...` no log, em `config_efetiva.yaml` e em `Relatorios/ajustes_automaticos.json`. Hoje:
pré-processamento espectral em tabela de elementos ou em imagem vira autoescala; modos imagem,
gcms, hplc, eem e rmn com a faixa NIR padrão deixam de cortar o eixo; o nível N1 desliga o DD-SIMCA
e o N2 liga; etapas de classificação ficam desligadas em objetivo que não é classificação. **Nunca
é automático:** alpha, validação, unidade de análise, política de censura, seleção de variáveis,
método final (viram sugestão). Combinação sem remédio seguro é recusada (ex.: nível N3 no modo csv
sem `coluna_concentracao`). O `cfg` do usuário volta aos valores escolhidos depois da execução.

**Saturação do detector (`saturacao: auto | avisar | nenhuma`).** Nos modos `dx`, `opus`, `spc`,
`sp` e `csv`, o programa procura absorbância travada num teto (um plato de valores idênticos,
por exemplo A = 3,0 com cubeta de caminho óptico longo). Variável com 10 % ou mais das amostras
no teto (alargada em 3 pontos de cada lado) e espectro com 20 % ou mais das variáveis no teto não
carregam informação da amostra. `auto` (padrão) tira os dois do modelo e registra em
`Relatorios/saturacao.json`, no aviso do terminal e na linha "Saturação do detector" do resumo;
`avisar` só avisa e não remove nada; `nenhuma` nem procura. O teto é a classe de valores repetidos
mais alta, não o máximo global, e dado sem plato nunca é tocado. A opção é independente de
`ajuste_automatico`. Mudar de `auto` para `avisar` ou `nenhuma` muda o conjunto de variáveis e,
portanto, os números do modelo.

### 5.12 RMN e IMS: faixas excluídas, normalização por espectro e o preset PQN+MC

**Faixas excluídas (`faixas_excluir`).** Lista de faixas `"a-b"` na unidade do eixo (ppm, ms,
cm⁻¹...), por exemplo `[4.5-5.0, -0.5-0.5]`. As variáveis dentro delas (extremos incluídos) saem
do cálculo logo depois do corte de `faixa_min_cm`/`faixa_max_cm` e **antes** de qualquer
pré-processamento. Por quê: SNV e PQN dividem cada espectro por uma medida do próprio espectro, e
a água residual do RMN (cerca de 4,7 ppm) varia dezenas de vezes entre amostras, enquanto o
TSP/TMS (0 ppm) é um padrão constante; os dois dominam o divisor e escondem a química. Em um
conjunto sintético de RMN de ¹H (3 classes, água residual de 30 a 400 vezes o sinal dos
metabólitos), a acurácia da validação cruzada com SNV+MC foi de 0,60 sem a exclusão para
0,80 com ela (dado simulado: mede o mecanismo, não o desempenho com amostras reais).

- **Padrão.** Vazio no motor geral: nenhum resultado existente muda. Escolher **RMN** na tela
  [2.4] grava `faixas_excluir: [4.5-5.0, -0.5-0.5]` e avisa na confirmação, no checklist e no
  resumo; escolher qualquer outra técnica zera a chave. Para manter a água no modelo, deixe a
  lista vazia em [2.3] Dados.
- **Confira as caudas.** A água de uma presaturação mal ajustada tem asas que passam de 4,5–5,0
  ppm. Olhe o espectro médio e alargue a faixa se for o caso; o programa não adivinha.
- **Validação.** O formato precisa ser `a-b` com `a < b` e ponto decimal (`5.0-4.5` é recusada);
  faixas que se tocam são fundidas. `guaraci checar CONFIG --dados` recusa uma faixa que não cruza o
  eixo dos dados (provável unidade errada, como Hz no lugar de ppm), avisa quando as faixas tiram
  mais de 50% da faixa usada e avisa quando o eixo parece RMN de ¹H em ppm e a lista está vazia
  (heurística: o programa não sabe a unidade de um CSV). Na execução, o mesmo cruzamento com o eixo
  vira erro claro.
- **No modelo exportado.** O `.joblib` e o `.portatil.json` guardam `faixas_excluir` e o eixo já
  sem elas; a predição (CLI, web e JSON portátil) reaplica a mesma exclusão, então uma água muito
  diferente numa amostra nova não muda a resposta.
- **Onde não vale.** Modos sem eixo físico (`multielementar`, `imagem`, `hsi`, `eem`) ignoram a
  chave, com aviso. Não há binning: agrupe em janelas de 0,01 a 0,04 ppm antes de importar.

**Preset `pqn_mc` (PQN + mean-centering).** Seletor *PQN+MC* (no fim da lista, para não mudar a
numeração antiga). Equivale a `Personalizado` com `etapas_pre_processamento: [pqn, mc]`: divide
cada espectro pela mediana dos quocientes contra o espectro mediano do treino (Dieterle et al.,
*Anal. Chem.* 78:4281, 2006), o que corrige diluição sem ser dominado por um pico que varia. O
fator de diluição conhecido é recuperado em teste. Limite: o quociente usa todas as variáveis; em
espectros esparsos, canais sem sinal e as asas da água pesam na mediana e o fator estimado deixa de
seguir a diluição. Nos dados sintéticos de RMN deste manual, com a água excluída, SNV+MC foi igual ou
melhor que PQN+MC em todos os cenários testados, por isso **o padrão da técnica RMN continua
SNV+MC**; PQN+MC fica como opção para quem tem diluição muito variável e espectros com binning.

**Normalização por espectro (`normalizacao`, `ims_rip_janela`).** Opcional, desligada por padrão
(`nenhuma`). Roda sobre o espectro já cortado e sem as faixas excluídas, antes do
pré-processamento, **sem estado** (não vaza entre dobras) e é reaplicada na predição:

| `normalizacao` | O que faz |
|---|---|
| `nenhuma` | Padrão: nada muda. |
| `rip` | Divide cada espectro pela intensidade **máxima** dentro de `ims_rip_janela` (`"a-b"`, ex.: `7-9` ms), o pico do íon reagente (RIP) no IMS. A janela precisa estar dentro da faixa usada e cobrir o pico apesar da deriva de T e P. |
| `area` | Divide pela área do espectro (regra do trapézio sobre o eixo usado). |

Um espectro com referência menor ou igual a zero é recusado com os índices. **O que os dados
sintéticos de IMS mostraram** (RIP + produtos, ganho de injeção variável de 20% a 50%, 8 conjuntos):
- **SNV já resolve o ganho.** `snv_sg_mc` e `snv_mc` deram o mesmo R² de validação cruzada para o
  teor (0,98 com ganho de 20% e de 50%) e acurácia 1,00 nas classes; só `mc`, sem normalizar, caiu para
  0,72 com ganho de 50%.
- **Por RIP, só quando o RIP não cai com o analito.** Pela conservação de carga o RIP *diminui* quando
  o analito sobe; dividir por ele comprime justamente a resposta que se quer medir. Com esse efeito,
  `rip`+MC deu R² de 0,68 (contra 0,98 do SNV); sem ele (ganho de 50%, RIP constante), 0,99, empatado
  com o SNV. `area`+MC ficou no meio (0,91 a 0,97).
- **Na execução completa** com o mesmo dado (30 lotes, `max_lvs: 10`): Q² de 0,97 com SNV+SG+MC (10
  variáveis latentes, o teto) e com SNV+MC (3), 0,95 com `area`+MC e 0,30 com `rip`+MC.
- **Recomendação.** O padrão da tela [2.4] para IMS não mudou (SNV+SG+MC). Para IMS, o SNV já
  normaliza o ganho; use `normalizacao: area` se não quiser SNV e `rip` só com RIP que não varia com
  o analito. A janela do SG padrão (25 pontos, 1,25 ms a 20 kHz) é bem mais larga que os picos (0,16 a
  0,26 ms); se o modelo bater no teto de variáveis latentes, `SNV+MC` é a alternativa sem derivada.

## 6 Fluxo típico na interface web

A navegação é uma **barra lateral** (versões antigas tinham 8 abas
horizontais). Ela tem duas telas fixas no topo e quatro grupos numerados
que refletem a ordem do trabalho:

- **◆ Início** — painel de status (seção 6.1).
- **🎨 Visualização** — cores das figuras (seção 6.2).
- **① Preparar** — Projeto · Dados · Pré-processamento
- **② Executar** — Modelo
- **③ Analisar** — Validação · Predição · Técnicas Avançadas · Relatórios
- **④ Referência** — Sobre

O grupo que contém a tela aberta vem expandido; nas telas fixas abre-se o
grupo ①. Ao lado de um item aparece um **selo** quando há algo real por
trás dele: o número de espectros em *Dados* (só depois de carregar a
prévia), `cego` em *Predição* (quando existe uma predição na sessão) e `✓`
em *Relatórios* (quando existe uma execução). Nenhum selo é exibido sem
dado — a mesma regra dos números do painel.

Acima da tela há uma barra com o título e o subtítulo da tela atual, o
**tema ativo** (somente leitura: o Streamlit não expõe API para trocar o
tema por código, então ali fica o estado real e o caminho ⋮ → *Settings* →
*Theme*) e dois atalhos: **Exportar relatório** e **Ir para Modelo →**.

Ordem de trabalho:

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

### 6.1 Tela Início — painel de status

A tela **◆ Início** resume o estado real da sessão. Abre com uma faixa de
**próxima ação sugerida** (com o botão que leva à tela certa) e quatro
cartões: *Dados carregados*, *Espectros médios por classe*, *Resultado da
predição* e *Faixa de decisão · teor estimado*.

Regra que vale para todos: **evidência ou silêncio**. Onde não há número
real, o cartão mostra `—` e explica o que falta — nunca zero fingindo ser
resultado.

- **Sem nada carregado** — cada cartão diz por onde começar; a faixa
  sugere carregar os espectros.
- **Com prévia de dados carregada** (botão *Load spectra preview* na tela
  Dados) — contagem de espectros e classes da prévia; "amostras físicas" e
  "variáveis espectrais" ficam `—`, porque esses números só existem depois
  de rodar.
- **Com execução concluída** — contagens lidas do `resumo_modelo.txt`
  daquela execução (amostras, variáveis, classes e nº de grupos `mae_id`,
  que é a contagem de *amostras físicas*), matriz/técnica do perfil ativo,
  e o resultado da
  **auditoria de delineamento** (seção 5.4) daquela execução: quantos
  críticos/avisos e o detalhe por checagem. Um achado crítico vira a
  próxima ação sugerida, com o texto real do achado.

Quando a execução é antiga demais para ter o registro da auditoria em
disco, o painel diz *"auditoria indisponível"* — nunca "nenhum problema
encontrado", que seria uma afirmação sem lastro.

O cartão **Faixa de decisão** desenha a barra de três zonas (não
detectado · zona cinza · confiável) com o marcador na posição do teor
estimado, e só a desenha quando LOD e LOQ existem de fato naquele
resultado (seção 5, figuras de mérito). Sem LOD/LOQ finitos ele diz isso,
em vez de desenhar uma faixa sem escala.

### 6.2 Tela Visualização — cores das figuras

A tela **🎨 Visualização** concentra a personalização de cor (antes ficava
dentro da aba Modelo → 🖼️ Figures).

À esquerda, três esquemas: **padrão Guaraci**, **seguro para daltonismo**
(Wong 2011/Okabe-Ito) e **alto contraste**, cada um com a amostra das
cores reais e uma descrição. Abaixo, **cor por classe**, que permite
trocar a cor de cada classe individualmente — as classes vêm da prévia
carregada, então esse controle só aparece quando existe dado.

À direita, a **pré-visualização ao vivo**: é a figura de espectros médios
de verdade, desenhada pela mesma função que o pipeline usa
(`spectra_preview.plot_mean_spectra` → `paleta_cores.map_class_colors`),
não uma amostra decorativa. Sem espectros carregados a tela diz isso e
mostra apenas as amostras de cor.

A escolha é gravada no MESMO arquivo de preferências que a CLI usa
(`~/.guaraci/visual_config.json`), então vale para as duas interfaces.

Duas regras importantes:

- A paleta vale para as figuras da **próxima execução** — figuras já
  gravadas em disco não são recoloridas (isso exigiria rodar o pipeline
  de novo).
- Uma paleta com **menos cores que o número de classes** é recusada na
  hora de desenhar (daria a mesma cor a duas espécies): o pipeline volta à
  paleta de máxima distintividade e registra o motivo no log.

> Nota: versões antigas só alteravam, ao escolher a paleta,
> `rcParams["axes.prop_cycle"]` do matplotlib, e **nenhuma** figura do
> pipeline usa esse ciclo (todas passam a cor explicitamente). O menu
> confirmava a troca e as figuras saíam idênticas.

Tema claro/escuro: menu ⋮ → *Settings* → *Theme* (segue a preferência do
sistema operacional por padrão). O degradê da barra lateral é montado a
partir dos tokens do tema **ativo** (`design_tokens`, lidos de
`st.context.theme`), então ele se redesenha ao trocar de tema em vez de
ficar preso à paleta de um tema só.

Marca: a logo e a versão ficam no topo da barra lateral. A logo (ver `docs/projeto/identidade_visual.md` §1.4, só na cópia do repositório) é um PNG transparente exibido sobre
uma moldura de fundo claro fixo — medido: a tinta mais escura da marca tem
contraste 1,40:1 contra o fundo do tema escuro, o que a tornaria invisível
sem essa moldura. Quando o app roda **sem** `config.yaml` local (caso do
deploy público em `guaraci.streamlit.app`, que não tem acesso aos dados
reais de pesquisa), aparece um aviso de **modo demonstração** explicando
que os espectros são sintéticos.

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
| `dados_io.py` | *Parsing* JCAMP-DX/ASDF, CSV e modo sintético; metadados do `TITLE`; seleção de amostras Kennard-Stone, Duplex e SPXY (com variantes *group-aware*); despacha a leitura via `io_registry.py` |
| `io_registry.py` | *Registry* de leitores de dados: mapeia `cfg.mode` (`dx`, `csv`, `imagem`, `sintetico` e, via `leitores_avancados.py`, `opus`/`spc`/`sp`/`rmn`/`hplc`/`gcms`/`eem`) ao leitor correspondente |
| `dados_imagem.py` | Colorimetria digital (`mode="imagem"`, protótipo): extração de *features* RGB/HSV/Lab e textura opcional |
| `agrupamento_pastas.py` | Hierarquia de 3 níveis de garantia de agrupamento (subpasta/CSV/nenhuma), extraída de `dados_imagem.py` pra' ser reaproveitada por `hsi_io.py` sem duplicar |
| `hsi_io.py` | Leitor ENVI (`.hdr`+`.bin`) genérico; `load_hsi_folder_dataset` (pasta do próprio usuário) + (`mode="hsi"`) |
| `hsi_quality.py` | Quality gate de cubo HSI: saturação, SNR (Immerkaer 1996), fração de pixels válidos |
| `hsi_segmentation.py` | Segmentação objeto/fundo por PCA(PC1)+Otsu, inferência de fundo pela borda da cena |
| `hsi_pixels.py` | Extração de espectros de pixel da ROI + `group_id` de objeto físico (base do split group-aware) |
| `hsi_classification.py` | PLS-DA por pixel (reaproveita `PLSDAClassifier`), seleção de LVs por Wold, agregação por objeto |
| `hsi_chemistry.py` | Explicabilidade cruzada: VIP × tabela de atribuição química (banda↔composto conhecido) — exige a tabela de atribuição química |
| `hsi_validation.py` | Validação externa por partição nativa de dia/lote (exige esse metadado); `run_internal_validation_group_aware` — só interna, sem particao por dia (dataset genérico) |
| `hsi_figures.py` | Mapa de classificação espacial por pixel (reaproveita `figuras.save` e a paleta da mascote) |
| `hsi_pipeline.py` | Orquestração ponta-a-ponta do `mode="hsi"` (não usa `pipeline.executar()` — forma de dado diferente); despacha pelo modo de dado (`modo_dataset`) |
| `preprocessamento.py` | *Transformers* SNV/SavGol/MSC e `build_preprocessor` |
| `classificadores.py` | DD-SIMCA, OPLS-DA |
| `figuras.py` | Camada de plotagem (todas as figuras do pipeline, incluindo `fig_regression_merit`) |
| `validacao_estatistica.py` | BCa, CV-ANOVA, permutação, teste de Wold, validação cruzada manual |
| `anova_grupos.py` | ANOVA de uma via/com bloco + Tukey, Tukey-Kramer ou Games-Howell sobre a unidade independente; FDR entre respostas (seção 5.8) |
| `icp_multielementar.py` | Perfil multielementar (ICP-OES/ICP-MS): leitura, censura com sensibilidade, QC, leitor `multielementar` e relatório (seção 5.8) |
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
| `app_tabs/` | Um módulo por tela do aplicativo web (`inicio`, `visualizacao`, `projeto`, `dados`, `preprocessamento`, `modelo`, `validacao`, `predicao`, `tecnicas`, `relatorios`, `sobre`) |
| `tecnicas_avancadas.py` | Orquestração testável (sem Rich/Streamlit) de ASCA/EPO-GLSW/MCR-ALS/fusão multibloco — fonte única entre CLI (`[T]`) e web (`app_tabs/tecnicas.py`) |
| `leitores_avancados.py` | Registra no `io_registry` os leitores de formato de instrumento (`opus`/`spc`/`sp`/`rmn`/`hplc`/`gcms`/`eem`) |

Os módulos acima vivem no pacote `src/guaraci/`. Interfaces de usuário:
`app_quimiometria.py` (web — fica na **raiz**, é o ponto de entrada do
*Streamlit*) e `guaraci/guaraci.py` (assistente de terminal — **único**
ponto de entrada interativo). Tema visual compartilhado: `guaraci_theme.py`,
`design_tokens.py`.

**CLI unificada:** `guaraci/cli_assistente.py` foi um assistente hierárquico
completo e independente — hoje é só um módulo de **dados
compartilhado** (paletas, fontes, referências) e de visões derivadas do
catálogo (`HELP_DB`, `MENU_FIELDS`, `TECNICAS`...) que `guaraci.py` consome; não tem mais `main()` nem menus
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

- **Perfil multielementar (ICP) em validação.** Mecanismo verificado em testes com
  dados sintéticos; poucas unidades independentes por grupo reduzem o poder; declare a
  unidade das concentrações (sem ela o rótulo é `mista`). Só CSV. Ver seção 5.8.

- **Cenários e telas (CLI).** A matriz `scripts/matriz_cenarios.py` (só na cópia do repositório) executa
  os cenários pelo CLI. Das 27 telas, 18 existem só no CLI (o web não tem,
  por exemplo, imageamento hiperespectral, monitoramento e comparação de grupos). A
  **priorização por amostragem ativa** (tela `J`, "refinar") só tem o que priorizar com
  modelo treinado sobre `mae_id` no formato que codifica sessão e adulterante (modo
  `dx`: código da espécie, data da coleta e token de teor, ex. `ABC-03-01-2099-A1.05`); com
  outros dados o programa responde "nada a priorizar". Selo `limitado`: provado pelo CLI com
  `mae_id` sintético nesse formato (`tests/test_amostragem_ativa_cli.py`). Os assistentes são
  determinísticos e só respondem o que está no catálogo; não há LLM.

- **Pré-processamento com estado ajustado no treino inteiro em parte das
  validações internas (m-06).** MSC, autoscaling e centragem guardam
  estatísticas do treino. Na seleção de LVs, na CV aninhada, na permutação e,
  desde 2026-09-26, no **LOGO do DD-SIMCA**, o pré-processamento é reajustado
  dentro de cada dobra (medido antes da correção: sensibilidade ~0,02 otimista
  com 10 grupos em MSC/autoscaling; zero com SNV ou 30 grupos). Na Etapa 4 e no
  domínio de aplicabilidade ele ainda vem do treino completo. Além disso, a Etapa 4 escolhe o
  método pela mesma validação cruzada que o reporta (*winner's curse*): o
  número do método escolhido é otimista. Só o holdout externo é livre disso.
  Correção (pré-processar dentro de cada fold) está no backlog.
- **A unidade independente precisa ser declarada.** Quando o delineamento tem sessões ou lotes além das réplicas (`mae_id`), agrupar só por `mae_id` pode deixar vazamento por sessão e produzir números otimistas. O padrão é `unidade_agrupamento: auto`: o GUARACI valida por sessão quando cada sessão tem uma classe só e isso é possível (pelo menos 3 sessões e toda classe em 2 ou mais); se alguma classe tem uma única sessão, fica em `mae_id` e o log diz o que falta coletar. O relatório diz qual unidade foi usada (`unidade_agrupamento`); compare com `unidade_agrupamento: sessao` antes de citar um número. A tela de Auditoria também mostra, por classe, quantas sessões independentes faltam para um resultado honesto (5) e para a garantia conformal (19).
- **Coberturas medidas por revisão cega independente (2026-09-26, dados
  sintéticos hierárquicos).** DD-SIMCA calibrado por `mae_id` aceitava 0,82–0,83
  das amostras puras novas (nominal 0,95); **corrigido** (a calibração pela média
  do grupo inflava os graus de liberdade): agora 0,93 com 30 grupos e 0,95 com
  100. Com 10 grupos ainda ~0,77. IC BCa por grupos subcobre com poucos grupos (G=6: 0,73;
  G=10: 0,86; G=40: 0,95). Domínio de aplicabilidade aceita ~0,91–0,93 de
  amostras novas normais. Identificação espécie×adulterante exige na prática
  ~20 grupos (não 19). Grupo declarado por você mais fino que a sessão, com ids
  que o software não consegue interpretar, deixa a validação vazar em silêncio
  (balanced accuracy 0,72 CV / 0,82 holdout sob H0, acaso 0,33): agrupe pelo nível
  independente de verdade. LOD/LOQ abaixo do RMSEP é irreal (o log avisa).
- **CV-ANOVA é conservador e o teste de Wilcoxon entre modelos é fraco (S-1/S-2).**
  Sob H0 simulada o CV-ANOVA rejeitou só 0,25% das vezes (nominal 5%): quase
  nunca dá falso positivo, mas perde poder. Wilcoxon sobre 5 folds tem p
  mínimo 0,0625 e **nunca** é significativo a 5%. Trate ambos como
  indicativos; o teste de permutação (Y-randomization) é o principal. O t
  corrigido de Nadeau-Bengio foi acrescentado ao benchmark (coluna própria,
  2026-09-26); o Wilcoxon fica só por compatibilidade.
- **LOD/LOQ é aproximado e tende a subestimar (m-02).** Usa só o ruído de
  réplica; omite os termos de leverage e de erro de referência da fórmula
  de Allegrini & Olivieri (2014).
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
  de calibração — há um platô assintótico (medido por simulação em
  2026-08-26).** Simulação com DGP
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
  reformulação do planejamento de coleta (tecla `[J]`): a orientação de
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

- **A identificação de adulterante não é agregada entre espécies.** A assinatura espectral de um adulterante depende da matriz hospedeira; agregar combinações diferentes numa só classe violaria a permutabilidade que a calibração conformal exige. Por isso o identificador é calibrado por combinação espécie×adulterante, e a cobertura sai como **não validável** quando uma combinação tem menos de duas sessões de coleta independentes (`session_from_mae_id`: o `mae_id` bruto codifica o nível de teor, não a sessão, e contá-lo infla o `n`). A ressalva "não validado" se propaga para o resultado final de qualquer predição em amostra nova, não fica só no relatório de auditoria interno.
  A medição que sustenta a decisão pode ser refeita com dado próprio: ela calcula quanto do desvio em relação à média da espécie pura é explicado por espécie e por adulterante (R² tipo PERMANOVA, Anderson 2001).

  **Retratação (2026-08-26):** a tabela e a frase "6 a 13×" publicadas
  antes desta correção NUNCA tiveram um script reprodutível — varredura
  de lastro (revisão desta sessão) não encontrou nenhum
  artefato de medição nem nos scripts privados de auditoria
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

- **Fluxo cego (implementado e verificado em 2026-08-25): fluxo completo
  Detectar → Identificar → Quantificar em amostra nova, no modo cego.**
  `identificacao.py` calibra um ensemble conformal (`ConformalOneClass`,
  ver `conformal.py`) por combinação espécie×adulterante, reaproveitando o
  mesmo espaço PCA do domínio de aplicabilidade — sem ajustar um espaço
  novo por combinação, a maioria tem poucos espectros de 1-2 sessões. A
  contagem de sessão independente usa `dados_io.session_from_mae_id`
  (**não** o `mae_id` bruto: uma amostra adulterada tem um `mae_id` por
  NÍVEL DE TEOR, não por sessão de coleta — contar bruto infla o `n` de 1
  sessão real para até 15). Combinações com uma só sessão ficam com `alpha_alcançável` `n/a`.
  `Identificar` nunca força uma classe: só preenche resultado quando
  exatamente UMA combinação é aceita sob cobertura estatisticamente
  validada. `Quantificar` é bloqueado por
  `Identificar` (nunca lança exceção — devolve motivo estruturado:
  `identificacao_desconhecida` ou `identificacao_ambigua`). A ressalva
  aparece nos 3 lugares combinados (log da execução, addendum "Identificação
  espécie×adulterante" no `model_card.md`, e `<modelo>.joblib.manifest.json`
  → `identification_coverage`)
  — mesmo padrão já usado para `grouping_guarantee`. A predição
  em lote (CLI, menu "Predição em Lote") ganha as colunas
  `classe_identificada`, `identificacao_cobertura`,
  `identificacao_alpha_alcancavel`, `identificacao_candidatos`,
  `teor_estimado`, `quantificacao_motivo_bloqueio` e `alpha_total`
  (limite de união/Bonferroni sobre os alpha de Detectar+Identificar —
  Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2022) sempre que o
  modelo carregado tiver o ensemble; modelos exportados antes do
  fluxo cego continuam funcionando sem essas colunas (retrocompatível).

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
  restante do fluxo cego: sem evidência de que o alpha se sustenta (sensibilidade
  LOGO ≥ 0,90 com 5 ou mais grupos, ou 19 ou mais grupos; o caso comum é 1
  amostra pura por sessão), a taxa de falsa rejeição de puros novos é
  desconhecida e pode ser alta. Isso não impede o DD-SIMCA de decidir
  aceitar/rejeitar (método paramétrico χ², não recusa como o conformal),
  mas o alpha declarado fica de fora da soma de Bonferroni — sem lastro
  numérico não entra na conta. Ressalva também nos 3 lugares (`purity_coverage` no
  manifesto, addendum próprio no `model_card.md`, log da execução).

- **Modo imagem (colorimetria digital): protótipo só quando não há fonte
  de agrupamento por amostra física.** Desde 2026-08-25, o
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
  ou Raman foi validada neste projeto**; os valores padrão foram pensados
  para NIR. Relatos de uso com outras técnicas
  são bem-vindos, mas trate como não testado até prova em contrário.

- **Carregar um modelo `.joblib` executa código arbitrário.** É uma
  limitação do formato pickle, não do Guaraci — ver `guaraci manual seguranca`
  para a política completa e as proteções implementadas (confirmação
  explícita obrigatória + manifesto de integridade).

- **`mae_id` mal formado vira "órfão" (grupo de 1 amostra), perdendo a
  proteção anti-vazamento só para essa amostra.** Quando **alguns** arquivos
  `.dx` casam com a convenção opcional de `##TITLE=` (seção 4.1) e outros
  não, o parser não trava o carregamento — atribui um `mae_id` único
  (`orfao_<arquivo>`) a cada um dos que não casam, para não desabilitar o
  `GroupKFold` do dataset inteiro por causa de um arquivo mal nomeado.
  (Se **nenhum** casa, não há `mae_id` nenhum: use `arquivo_grupos`.) O `[INFO]` impresso no
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

- **Valores padrão são ponto de partida.** Os valores padrão do motor (pré-processamento,
  regressão PLS, limiares) seguem a literatura citada e não são universais; são ajustáveis por opção. Isso não
  valida o seu estudo: confirme as conclusões com validação externa própria.

- **Classes com poucas amostras ou espectralmente próximas têm recall mais
  baixo** — reporte sempre a **matriz de confusão completa**, nunca só a
  acurácia agregada, que pode esconder uma classe minoritária mal
  classificada. Classes quimicamente muito próximas — por exemplo, duas
  espécies da mesma família botânica, ou duas variedades do mesmo produto —
  podem se sobrepor no espectro: é limitação **química** da técnica, não do
  código, e nenhum pré-processamento a remove. *(Recall por classe e número
  de espectros descartados por faixa incompatível variam por dataset;
  confira o `resumo_modelo.txt` e a matriz de confusão da sua própria
  rodada.)*

- **ASCA** decompõe por efeito MARGINAL (Smilde et al. 2005), exato só
  para fatores ortogonais (delineamento balanceado). Com fatores
  fortemente correlacionados ou muito desbalanceados, variância de um
  fator pode "vazar" para outro — `ss_desbalanco` (exibido no resultado)
  sinaliza quando isso é relevante. A extensão ASCA+ (Thiel, Féraud &
  Govaerts 2017), que corrige isso via modelo linear geral, **não está
  implementada** (esforço maior, avaliado e registrado como backlog).

- **EPO/GLSW não se aplicam quando o fator de perturbação é colinear com o alvo científico:** remover essas direções removeria o próprio sinal que se quer quantificar (a ordem de leitura colinear com o teor é um exemplo). Ferramenta exploratória, não integrada ao pipeline de treino automático.

- **MCR-ALS é interpretativo:** a solução não é única (ambiguidade rotacional) e nada garante que um componente recuperado corresponda ao analito; não substitui PLS-R para quantificação com garantia.

- **Fusão multibloco (nível 1, concatenação) pode não superar o melhor bloco
  sozinho** — o bloco com mais variáveis domina a decomposição PLS por
  contagem de colunas, não por conteúdo informativo. Compare sempre
  contra cada bloco treinado isoladamente antes de adotar a fusão.

- **HSI é só CLI** (tecla `[X]`): o aplicativo web não tem página HSI
  (só o campo da pasta na aba Dados). Os extras HSI (domínio de
  aplicabilidade, reamostragem, multiway) herdam isso; a identificação por objeto e câmera não está exposta.

- **Leitores de instrumento novos (`opus`/`spc`/`sp`/`rmn`/`hplc`/`gcms`/
  `eem`) não têm `mae_id` real** (nenhuma noção de réplica física
  conhecida a partir do nome/estrutura de arquivo) — mesma situação
  honesta já documentada para `csv` genérico; `grouping_guarantee` não é
  elevado artificialmente. `opus` e `gcms` não têm arquivo real de teste
  disponível neste ambiente (testados com double controlado, mesma
  limitação já documentada para o parser de baixo nível).

---

## 10 Perguntas frequentes e problemas comuns

**O programa abriu em inglês.** Quem abre pela primeira vez vê a interface em inglês. A tecla `I`
no menu troca para português, e a escolha fica gravada em `~/.guaraci`.

**A análise parou com "Validação cruzada impossível".** Alguma classe tem uma só unidade
independente. O programa oferece retirá-la (`excluir_classes`) e rodar de novo; ver seção 1.1.
No objetivo Exploratório isso não acontece.

**O painel parou de mudar.** Espere a mensagem final: a análise pode passar alguns minutos numa
etapa longa. Se nada mudar por 120 s, o arquivo `~/.guaraci/travamento_*.txt` registra onde ela
estava; anexe-o ao relato de erro (seção 13).

**O PowerShell diz que a execução de scripts está desabilitada ao ativar o ambiente.**
Libere só para aquela janela com `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` e repita
`.venv\Scripts\Activate.ps1`. Outra saída é não ativar nada e chamar `.venv\Scripts\guaraci.exe`
direto. O passo a passo completo está em `docs/usuario/instalacao.md` (só na cópia do repositório;
na página do projeto: <https://github.com/ErleySC/guaraci>).

**No Linux, `python3 -m venv` falha com "ensurepip is not available".** No Debian e no Ubuntu o
módulo `venv` vem em pacote separado (`python3.12-venv`, ou o da versão que você usa). Instale-o e
crie o ambiente de novo. O Tk **não** é necessário: as figuras vão para arquivo, sem abrir janela.

**`guaraci doctor` mostra `[ausente]` em algo.** Em um recurso opcional (por exemplo o `streamlit`)
isso não é erro: só quer dizer que aquele extra não foi instalado. Para o navegador, instale
`".[web]"`; para relatórios em PDF/Word/Excel, `".[reports]"`; para tudo, `".[all]"`.
Mantenha as aspas: no `zsh`, sem elas o shell interpreta os colchetes.

**O terminal fica ilegível em fundo branco.** A paleta supõe fundo escuro. Defina
`GUARACI_TERMINAL=claro` antes de rodar (PowerShell: `$env:GUARACI_TERMINAL = "claro"`;
Linux: `export GUARACI_TERMINAL=claro`).

**O Windows bloqueou um arquivo (Controle de Aplicativo).** Rode o mesmo comando de novo: o Windows
costuma liberar a biblioteca depois da primeira checagem. Se persistir, o Python usado
provavelmente não é o do instalador oficial; recrie o ambiente com o oficial. Evite instalar em
pasta sincronizada com a nuvem (serviços de sincronização): a sincronização pode travar arquivos e corromper o ambiente.

**Abri o CSV e dá erro de leitura.** Confira o separador de colunas e o separador decimal. Se o
decimal for a vírgula, o separador de colunas precisa ser outro (por exemplo `;`); a mensagem do
programa explica o caso. Tabelas de centenas de MB consomem muita memória só para abrir (ver
seção 12).

**Apareceu um aviso dizendo que réplicas podem estar vazando entre treino e teste.** Significa que
o programa não sabe quais linhas são réplicas da mesma amostra física, então a validação cruzada
pode separar réplicas de uma mesma amostra e dar um número otimista. Informe o identificador da
amostra física (a coluna `mae_id`, ver glossário) ou use a organização de pastas que a codifica
(seção 4).

**O modelo parece bom demais, ou instável.** Com poucas amostras por classe
ou poucos grupos independentes, as métricas ficam instáveis. Veja `guaraci manual limitacoes` e
`guaraci buscar "poucas amostras"`. Um número bom em validação sem grupos, ou com a seleção de
variáveis feita fora da validação, não vale como evidência.

**Uma execução parou no meio.** Se a pasta da execução tem o arquivo `EXECUCAO_INCOMPLETA.txt`, ela
não terminou: os arquivos que existem ali podem estar parciais. Rode de novo.

**`guaraci web` abre mas o navegador não alcança o endereço (WSL).** Use o endereço
`http://localhost:...` que o Streamlit imprime no terminal e trabalhe numa pasta dentro do Linux
(`~/guaraci`), não em `/mnt/c/...`.

**Quero uma resposta rápida sobre um termo ou opção.** `guaraci buscar TERMO` procura no catálogo,
`guaraci ajuda` mostra o mapa, e o assistente (`guaraci assistente PERGUNTA`, ou a tecla `a`) responde
por regra, sem rede e sem modelo de linguagem. Se nada resolver, ver a seção 13.

---

## 11 Glossário

- **Amostra física:** o objeto real que foi medido (um frasco, uma folha). Várias leituras dele são
  **réplicas**. É a unidade independente que a validação deve respeitar.
- **`mae_id`:** coluna (ou código derivado das pastas) que diz a qual amostra física cada linha
  pertence. Serve para que a validação nunca separe réplicas da mesma amostra entre treino e teste.
- **Unidade independente:** o que de fato conta como "uma observação" estatística (a amostra física,
  não a réplica nem o espectro). Testes e intervalos que usam réplicas como se fossem unidades
  produzem números otimistas.
- **LV (variável latente):** cada componente do PLS. Poucas LVs subajustam, muitas sobreajustam;
  o programa escolhe por validação cruzada.
- **PLS-DA / PLS-R:** PLS para classificar (discriminação) / para prever uma quantidade (regressão).
- **RMSEC, RMSECV, RMSEP:** erro quadrático médio de calibração (nos dados que treinaram), de
  validação cruzada e de predição (em amostras que o modelo nunca viu). O RMSEP é o que mais se
  aproxima do desempenho real.
- **R², Q²:** fração da variância explicada no ajuste / na validação cruzada. Um R² alto com Q² baixo
  indica sobreajuste.
- **Validação cruzada (CV):** repetir treino e teste em partes diferentes dos dados. **LOGO** deixa um
  grupo inteiro (uma amostra física) de fora a cada rodada.
- **Seleção aninhada:** escolher variáveis ou parâmetros **dentro** de cada dobra de treino, para a
  escolha não "espiar" os dados de teste.
- **VIP:** importância de cada variável no PLS; costuma-se destacar as de VIP maior que 1.
- **DD-SIMCA:** modelo de uma classe só (autenticação): aceita ou rejeita a amostra como "da classe".
  Não diz qual é a outra classe.
- **T² e Q (e DModX):** distâncias da amostra ao modelo (dentro do espaço do modelo e fora dele).
  Valores altos sugerem amostra diferente das de calibração.
- **Domínio de aplicabilidade:** região em que o modelo foi calibrado e onde suas previsões valem.
- **Conjunto conforme:** em vez de uma só classe, devolve o conjunto de classes compatíveis com um nível
  de confiança escolhido.
- **LOD / LOQ:** menor quantidade que o método detecta / quantifica com confiança. Aqui são
  aproximações (ver seção 9).
- **SNV, MSC, Savitzky-Golay, centragem, autoescalamento:** pré-processamentos espectrais (seção 5).
- **Perfil de técnica:** conjunto de ajustes de uma técnica analítica (NIR, Raman, UV-Vis...);
  vive em arquivo, não no código (seção 4b).
- **Modo cego:** o programa não olha o rótulo verdadeiro das amostras novas ao prever (seção 4b.3).
- **Objetivo científico:** Exploratório, Classificação ou Quantificação (seção 2.2); decide quais
  figuras e relatórios saem.

---

## 12 Requisitos mínimos

- **Python:** 3.11 a 3.13. Sistemas: Windows, Linux e WSL; macOS **não foi testado**.
- **Memória e processador:** 4 GB de RAM e 2 núcleos bastam para o cenário típico (dezenas de amostras
  e centenas de variáveis), conforme medido pelo autor em dados sintéticos. Tabelas CSV de
  centenas de MB exigem bem mais memória só para serem abertas; com pouca RAM, prefira tabelas
  menores ou uma máquina maior.
- **Instalação:** alguns minutos com boa conexão (o `pip` baixa as bibliotecas científicas); o
  extra `".[all]"` baixa e ocupa bem mais que o núcleo. Reserve folga em disco para os resultados:
  cada execução grava de poucos MB, com gráficos e tabelas.
- **Conexão:** só para instalar. Depois, o programa funciona offline.
- Conferir no seu computador: `guaraci doctor` (Python, memória, processador e bibliotecas).

---

## 13 Como reportar erro

O GUARACI é mantido por uma pessoa, no tempo livre: a meta é responder em até 7 dias úteis, sem
garantia de prazo nem de solução. O texto completo está em `guaraci manual suporte`.

| Assunto | Canal |
|---|---|
| Dúvida de uso | GitHub Discussions do repositório; antes, este manual e `guaraci buscar` |
| Erro (bug) | GitHub Issues, com o formulário de erro |
| Sugestão | GitHub Issues, com o formulário de sugestão |
| Vulnerabilidade de segurança | **Nunca em issue pública**: siga `guaraci manual seguranca` |

Ao relatar um erro, inclua: a versão (`guaraci --version`), o sistema operacional, o comando ou a tela
usada e a **mensagem completa**, copiada como apareceu. Se a análise rodou, anexe só o que puder ser
público: **não anexe dado que você não pode tornar público**, nem amostras de terceiros.

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

*O que mudou de uma versão para outra está no `CHANGELOG.md`, na página do projeto.*
