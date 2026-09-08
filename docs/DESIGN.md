# GUARACI — Identidade visual e navegação (proposta, Agente 5)

> **Status: PROPOSTA, não implementada.** Nenhum arquivo de UI (`design_tokens.py`,
> `guaraci_theme.py`, `app_quimiometria.py`, `app_tabs/*.py`) foi tocado para
> produzir este documento. Seções 1 e 2 exigem aprovação explícita antes de
> qualquer implementação (regra da auditoria de 2026-09-01).

## 0. Achado que muda o escopo desta seção

O projeto **já tem** um sistema de tokens de cor em produção
([`src/guaraci/design_tokens.py`](../src/guaraci/design_tokens.py), reexportado
por [`guaraci_theme.py`](../src/guaraci/guaraci_theme.py) para o tema Rich do
CLI): paleta "forest/amber" — verde-floresta como cor primária, âmbar como
destaque. É usado hoje em **todo** o CLI (tags `[a]`/`[f]`/`[s]`/`[r]`/`[g]`
espalhadas por `guaraci.py`, ~200KB) e no app Streamlit.

A paleta extraída da mascote (seção 1) tem **laranja vívido como cor
dominante** — não existe hoje nenhum tom de laranja como cor primária no
sistema; o tom mais próximo (`PR`/rust, `#B85030`) é usado para **alerta**, não
para ação. Ou seja: a mascote e o sistema de cores em produção não contam a
mesma história. Isso não estava previsto na instrução original (que presumia
"extrair a paleta da mascote" como se fosse a primeira paleta do projeto) —
é um achado que descobri ao ler o código antes de escrever este documento.

Três caminhos possíveis, com custo bem diferente:

- **(A) Migração completa** para a paleta da mascote (laranja primário) —
  reescreve `design_tokens.py`, `guaraci_theme.py` e todo tag Rich em
  `guaraci.py`/`cli_assistente.py`. Risco alto (arquivo de 200KB, testado por
  saída de texto em alguns lugares), mas resolve a inconsistência de vez.
- **(B) Manter o forest/amber atual como identidade oficial**, tratar a
  mascote como ilustração (usada no ícone/splash), sem forçar paleta de UI a
  bater com ela. Risco zero, mas a Seção 5.1 original fica sem efeito prático.
- **(C) Híbrido** — mantém verde como `primary` (já é), recolore o `accent`
  atual (`#B8963E`, âmbar discreto) para o dourado mais vívido da mascote
  (`#FFC100`, ou uma versão dessaturada dele) e usa o laranja da mascote só
  no **elemento de identidade** (o sol, logo, splash/cabeçalho), nunca em
  botão/ação. Menor blast radius: não mexe no significado semântico já
  aprendido pelo usuário (verde=sucesso, âmbar/dourado=destaque), só afina os
  tons e adiciona o sol como símbolo.

Este documento define a paleta extraída (seção 1) independente de qual
caminho for escolhido — ela é a matéria-prima; a decisão A/B/C fica para você.

## 1. Paleta extraída da mascote (medida, não estimada)

Extraída por amostragem de pixel real do PNG (`assets/guaraci_icon.png`),
agrupando por matiz (HSV) e tirando a moda + média dos 200 tons mais
frequentes de cada grupo — não são valores "de olho".

> **Reamostrada em 2026-09-08**, com a troca da marca (ver §1.4). Os números
> abaixo são os da marca ATUAL. Os tokens implementados hoje
> (`design_tokens.py`/`guaraci_theme.py`) foram derivados da marca ANTERIOR
> e **não** foram alterados nesta rodada — a comparação está em §1.4, como
> decisão aberta, não como mudança silenciosa de cor do produto.

| Papel | Hex (moda) | Hex (média, mais neutro) | Onde aparece na mascote |
|---|---|---|---|
| Laranja | `#D5672C` | `#EE862D` | lado esquerdo do cocar de nós/folhas (23,2% dos pixels) |
| Dourado | `#F9A233` | `#F3A334` | sol central dentro do frasco (11,0%) |
| Verde | `#19824E` | `#397845` | lado direito do cocar, moléculas (29,5%) |
| Verde escuro | `#1E4935` | `#1C4734` | contorno do frasco e do cão, base da marca (14,7%) |
| Branco | `#FFFFFF` | `#FDFDFC` | vazados internos do desenho (21,5%) |

### 1.1 Variantes por estado (derivadas, HSL ±L)

| Cor | Normal | Hover (+8% claro) | Pressed (−8% escuro) | Desabilitado (−40% sat, cinza) |
|---|---|---|---|---|
| Laranja | `#FF6400` | `#FF7F2B` | `#D95700` | `#C9A88F` |
| Dourado | `#FFC100` | `#FFD23F` | `#D9A400` | `#D9C99B` |
| Verde | `#46A41C` | `#5CC42E` | `#398416` | `#A8C79A` |
| Erro (rust, já existente) | `#B85030` | `#CC6440` | `#9C4326` | — |

### 1.2 Regras de uso (a cor carrega significado, não decoração)

- **Laranja** → ação primária (botão principal, CTA), **nunca** para status
  (não usar para "erro" nem "sucesso" — já existem cores para isso).
- **Verde** → confirmação, sucesso, estado válido. Mantém o significado que
  o sistema atual já usa (`success`/`PG`) — não inverter.
- **Dourado/âmbar** → bordas, molduras, elementos de identidade/destaque
  secundário. Não usar como cor de texto de corpo (contraste insuficiente
  em fundo claro).
- **Grafite escuro** → texto e fundo de área técnica (painéis de código,
  tabelas densas). É a cor que já existe como `text`/`PW`+`PD` no sistema
  atual — sem conflito, só reafirma o que já é.
- **Nunca** usar a paleta de identidade (esta tabela) para colorir
  classes/espécies em gráficos — isso é papel de
  [`paleta_cores.py`](../src/guaraci/paleta_cores.py) (paleta de máxima
  distintividade perceptual, Glasbey-based, otimizada para daltonismo). São
  dois sistemas de cor com propósitos diferentes; misturá-los pioraria a
  legibilidade dos gráficos científicos, que é o que mais importa lá.

### 1.3 Elemento simbólico: o sol

O sol continua sendo o elemento mais reaproveitável como marca gráfica
isolada (funciona em tamanho pequeno, silhueta simples, carrega a cor
dourada). Na marca de 2026-09 ele passou do laptop da mascote para **dentro
do frasco**, no centro da composição — ganhou peso, não perdeu. Usado como
favicon/ícone de app (via `guaraci_icon.png`) e como marcador no cabeçalho
padrão de cada tela (seção 3).

### 1.4 Troca de marca em 2026-09-08 — o que mudou, e o que NÃO mudou

A marca anterior (cachorro cientista de jaleco, com cocar, notebook e
erlenmeyer ao lado) foi substituída por uma composição mais gráfica: a
silhueta de um **frasco de Erlenmeyer** contendo o **sol** e a **cabeça do
cão**, com estruturas **moleculares** na base e um **cocar de nós e folhas**
em gradiente laranja→verde no topo. Arquivo anterior preservado em
`assets/legado/guaraci_icon_2026-07.png` (e `.ico`) — arquivado, não
apagado.

Diferenças medidas entre as duas amostragens (mesmo método):

| Papel | Marca 2026-07 (moda) | Marca 2026-09 (moda) | Leitura |
|---|---|---|---|
| Laranja | `#FF6400` | `#D5672C` | menos saturado, mais terroso |
| Dourado | `#FFC100` | `#F9A233` | puxou para laranja |
| Verde | `#46A41C` | `#19824E` | saiu do verde-amarelado para verde-azulado |
| Escuro | `#181E21` (grafite neutro) | `#1E4935` (verde escuro) | **a marca nova não tem neutro escuro** |
| Claro | `#FFFBD4` (creme) | `#FFFFFF` (branco) | perdeu o tom quente |

**O que NÃO mudou (de propósito):** os tokens em `design_tokens.py`
(`primary`/`accent` etc.) e a paleta Rich do CLI (`guaraci_theme.py`).
Trocá-los muda a cor de CLI, web e cabeçalhos de uma vez — é decisão de
produto, não consequência automática de trocar um arquivo de imagem. Fica
registrado aqui como pendência explícita: **realinhar os tokens à marca
nova, ou assumir que a paleta do produto é independente da mascote.**

Nota de contraste (medida, não estimada): a tinta mais escura da marca nova
(`#0E3724`, base do frasco) tem contraste **1,40:1** contra o fundo do tema
escuro (`#0F1613`) — invisível na prática (o mínimo para elemento gráfico é
3:1, WCAG 1.4.11). Por isso a moldura da logo no cabeçalho do app tem fundo
claro fixo (`#FDFDFD`), onde o contraste é **12,4:1** nos dois temas — mesma
convenção já usada para as figuras científicas ("papel" branco, intencional
em qualquer tema). O PNG em si continua com fundo transparente.

## 2. Tipografia

**Restrição real que já existe no projeto**: o Streamlit não tem injeção de
CSS custom hoje — foi removida de propósito (`app_quimiometria.py`, comentário
sobre `_active_theme()`: "Não há mais estado paralelo `dark_mode` nem CSS
`!important` pintando widgets à mão, origem do bug de cor ao trocar tema").
Reintroduzir CSS para forçar uma fonte custom arrisca reproduzir exatamente
essa classe de bug. Recomendação: usar a chave `font` nativa do
`[theme]` do Streamlit (`.streamlit/config.toml`, suportada desde Streamlit
1.36) — não CSS injetado.

- **Família única**: `"Source Sans Pro", sans-serif` (é a família nativa do
  tema padrão do Streamlit — trocar por outra via a chave `font` do tema é
  possível, mas cada opção fora das nativas do Streamlit — `"sans serif"`,
  `"serif"`, `"monospace"` — teria que ser testada nas duas telas antes de
  virar padrão; não testei nenhuma alternativa nesta rodada, então não
  proponho uma). O CLI usa fonte de terminal do usuário (monoespaçada, fora
  do nosso controle — só a cor/negrito/dim é nosso).
- **Hierarquia (máx. 4 níveis, já é aproximadamente o que o Streamlit usa)**:
  1. Título de aba (`st.subheader` / painel Rich com `[hdr]`)
  2. Rótulo de campo/seção (`st.markdown("**...**")` / texto normal)
  3. Corpo/ajuda (`st.caption` / `[dim]`)
  4. Dado tabular/código (fonte monoespaçada, `st.dataframe`/`st.code`)

## 3. Cabeçalho/rodapé padrão (conceito — implementação fica para 5.3)

Todo tela (CLI e web) segue: **cabeçalho identificado** (nome da aba + ícone
do sol pequeno, ex. "🔆 Predição em Lote") → **contexto** (1 linha: o que essa
tela faz) → conteúdo → ações → **rodapé de navegação** (voltar/ajuda/atalho).
Isso já existe parcialmente (algumas abas têm `st.caption` de contexto logo
no topo, ex. `app_tabs/dados.py`: *"📂 Step 2: Upload or select spectra
folder..."*) — a proposta é tornar isso **obrigatório e no mesmo formato**
em todas, não decisão caso a caso. Não implementado aqui.

## 4. Proposta de agrupamento de navegação

**Correção à lista original**: a instrução original listava 6 grupos
assumindo abas que incluem "predição" e "sentinela de deriva" como entradas
de menu próprias do grupo "Prever". O Agente 1 já confirmou por leitura de
código que a lista real de menu do CLI é **18 entradas** (não 19 — não existe
"0"), e que a sentinela de deriva **não é uma aba própria**: ela roda
embutida dentro do fluxo de `B` (Predição em Lote), sem entrada de menu
separada. A predição de amostra única (não-lote) também não existe como aba
do CLI — só existe no app Streamlit. Ajustei o agrupamento para bater com a
lista real:

| Grupo | Abas reais (tecla) |
|---|---|
| **Preparar** | `2` Dados · `3` Pré-processamento · `9` Codificação DX · `P` Perfis prontos |
| **Planejar** | `J` Planejamento de coleta · `K` Seleção de amostras · `U` Auditoria de delineamento |
| **Modelar** | `4` Modelagem · `6` Métodos avançados · `8` Técnicas avançadas |
| **Validar** | `5` Validação · `7` Visualização |
| **Prever** | `B` Predição em lote *(inclui a sentinela de deriva, embutida no fluxo — não é aba separada)* |
| **Sistema** | `1` Projeto · `H` Hardware · `G` Guaraci (assistente) · `?` Ajuda · `A` Sobre |

O agrupamento em si (a lógica de fluxo de trabalho: preparar → planejar →
modelar → validar → prever, mais um grupo de sistema) faz sentido e bate com
a ordem em que um usuário novo realmente usaria o pipeline — mantenho a
proposta. O grupo "Prever" fica mais magro do que a instrução original
supunha (1 aba só, não 3), o que é esperado — é a ala do produto com menos
frentes hoje.

**Proposta de implementação (não feita ainda)**: menu principal mostra os 6
grupos; escolher um grupo abre um submenu com as abas dele. Atalho direto
preservado — digitar a tecla original (`B`, `U`, etc.) direto no menu
principal continua indo direto para a aba, sem passar pelo grupo, para quem
já decorou o atalho.

## 5. Pendências desta seção

Aguardando decisão sua: caminho A/B/C da seção 0 (migração de paleta),
aprovação do agrupamento de navegação da seção 4, e se a família tipográfica
recomendada (nativa do Streamlit) está OK ou se você quer testar alguma
alternativa antes de fixar. Nada em `src/guaraci/app_tabs/*.py`,
`design_tokens.py` ou `guaraci_theme.py` foi alterado.

**Atualização 2026-09-01**: seções 0 (caminho A, migração completa) e 4
(agrupamento em 6 grupos) foram aprovadas e já implementadas — ver commits
`209537f` e `52bb9c1`. Seção 3 (cabeçalho/rodapé padrão) e a parte de
ajuda contextual/progresso/confirmação da instrução original (5.3/5.4)
também já foram implementadas nas 18 abas do CLI e nas 8 do app web — ver
commits `fcf3244` e `e07321d`. Só a tipografia (seção 2) segue sem decisão
explícita — mantido o padrão nativo do Streamlit, nenhuma alternativa foi
testada.

**Atualização 2026-09-08**: o app **web** deixou de ter 8 abas horizontais —
a navegação virou barra lateral com 2 telas fixas (Início, Visualização) e os
mesmos 4 grupos de fluxo já usados como legenda (① Preparar · ② Executar ·
③ Analisar · ④ Referência), a pedido do mockup v4. A estrutura vive em
`src/guaraci/app_nav.py` (dado puro, sem Streamlit), e é dela que o gerador do
vault lê a ordem das telas. Isso **reverte** duas decisões registradas antes:
"abas horizontais, sem sidebar" e "não pintar widget nativo do Streamlit" — a
segunda com duas salvaguardas que a tentativa antiga não tinha (cores vindas
dos tokens do tema **ativo**, e nenhum `!important`). O menu da **CLI** (18
telas em 6 grupos, seção 4 acima) não mudou. Ver `docs/MANUAL.md` §6 e
`docs/PROGRESSO.md` Passos 195-199.

---

# GUARACI — Central de perfis (Agente 5B)

> **Status 2026-09-01: implementado por completo.** Ver commits `d50155c`
> e `c1939ce`. 2º campo em `Config` (`acquisition_profile`, independente
> de `matrix_profile`); seletor navegável (era texto livre) na CLI e no
> Streamlit, agrupado por dimensão via `perfis_disponiveis()`; exposição
> do dado antes morto (resolução/formatos/nível de garantia típico) em
> cada opção do seletor; indicador de cobertura validada (✅/⚠, derivado
> do campo `referencia` que já existia — `milho_nir`/`oleos_comestiveis_nir`
> validados com dado público, os demais 5 perfis só declarados); e
> `combine_profiles`/`save_profile` para fundir matriz+técnica num perfil
> de usuário reusável (`[C]` na aba Dados do CLI).

## 5B.1 Diagnóstico (corrige a premissa da instrução original)

A instrução original presumia que a aba `P` ("Perfis Prontos") já era um
seletor de perfil de matriz química, e que o mecanismo de perfil de
**técnica de aquisição** (Bloco 8, modo imagem) vivia "disperso" em outro
lugar isolado. Lendo o código (não supondo), as duas metades dessa premissa
estão erradas — de um jeito que muda o que faz sentido propor:

1. **A aba `P` não tem nada a ver com perfil de matriz.** `_menu_profiles`
   (`src/guaraci/guaraci.py:3221`) é uma biblioteca de **presets de
   objetivo/rigor de análise** — "Explorar Dados", "Autenticar Pureza",
   "Quantificar Teor", "Pesquisa Acadêmica", "Alta Rigorosidade" etc. — que
   aplicam um conjunto de campos de `Config` (nível de rigor estatístico,
   paleta de cor) de uma vez. É a mesma lista que alimenta os 3 botões de
   atalho "Recommended analysis" da aba Data do app Streamlit
   (`PROFILES`, `app_tabs/dados.py`). Repurposar `P` para virar "central de
   perfis de matriz+técnica" quebraria essa feature existente, que
   funciona e é referenciada em `CLAUDE.md` seção 6.
2. **Perfil de matriz química e perfil de técnica de aquisição já são o
   MESMO mecanismo, não dois.** `src/guaraci/perfil_matriz.py` define uma
   única dataclass `MatrixProfile` com campos de matriz (unidade do eixo,
   faixa espectral, vocabulário, código de classe) **e** campos de técnica
   de aquisição (`resolucao_esperada`, `formatos_aceitos`,
   `nivel_agrupamento_tipico` — comentário no código: "Bloco 8a,
   2026-08-25"), lidos do mesmo diretório `perfis_matriz/*.yaml` (8
   arquivos: `oleo_nir`, `milho_nir`, `oleos_comestiveis_nir`,
   `mel_vis_nir` — sabor matriz; `bancada`, `celular`, `scanner` — sabor
   técnica; `generico` — fallback de qualquer um dos dois). Não há
   dispersão para unificar — a unificação de dados já existe.

O problema real, achado por leitura de código e confirmado por grep (zero
referência aos 3 campos de técnica em qualquer lugar fora do próprio
`perfil_matriz.py`), é outro, em três partes:

- **Os campos de técnica de aquisição nunca são lidos por ninguém.**
  `resolucao_esperada`/`formatos_aceitos`/`nivel_agrupamento_tipico` são
  carregados do YAML, viram atributos da dataclass, e não aparecem em
  nenhuma tela, nem são consumidos por `dados_imagem.py` (o módulo que de
  fato processa imagens). São dados mortos do ponto de vista do usuário —
  existem, mas ninguém nunca vê.
- **`Config` só guarda UM perfil por vez** (`matrix_profile: str`). Hoje
  não há como escolher "isto é mel" (matriz, pro vocabulário do relatório)
  **e** "foi fotografado com celular" (técnica, pra saber a garantia de
  agrupamento típica) ao mesmo tempo — escolher um descarta as
  informações do outro, mesmo os dois sendo relevantes juntos
  especificamente no modo imagem.
- **O único ponto de acesso interativo é um campo de texto livre**
  (`_CONFIG_SPEC` tipo `"str"`, sem `opcoes`) dentro da aba `2` Dados,
  junto com 8 outros campos não relacionados. O usuário digita o nome de
  cor (ex. `oleo_nir`) sem ver a lista — pra descobrir os nomes precisa
  sair da sessão interativa e rodar `guaraci perfis` (subcomando
  não-interativo, lista e encerra) num terminal separado.

## 5B.2 Escopo proposto (aguardando aprovação)

- **Não mexer na aba `P`** — ela resolve um problema diferente e válido
  (presets de análise), manter como está.
- **Novo ponto de acesso interativo** — dentro da aba `2` Dados, ao
  selecionar/editar o campo `perfil_matriz`, trocar o prompt de texto
  livre por uma lista navegável (mesmo padrão já usado em `_menu_profiles`
  — número + Enter, `[?]` pra detalhe) que mostra os 8 perfis, agrupados
  visualmente em duas seções ("Perfis de matriz" / "Perfis de técnica de
  aquisição de imagem") — a separação é só de apresentação, o carregamento
  continua sendo o mesmo `load_profile()` para os dois grupos.
- Para cada perfil de técnica, a lista exibe o que hoje é dado morto:
  resolução esperada, formatos aceitos, nível de garantia de agrupamento
  típico — a mesma transparência de cobertura que já existe no resto do
  projeto (LOGO honesto, ressalva de cobertura não-validada), agora visível
  no lugar certo.
- **Decisão de arquitetura que precisa de aprovação explícita**: dar a
  `Config` um segundo campo (`acquisition_profile: Optional[str] = None`)
  para permitir combinar matriz + técnica simultaneamente no modo imagem —
  hoje é estruturalmente impossível (um campo só). Sem esse segundo campo,
  a proposta de "central de perfis com duas dimensões independentes" da
  instrução original não é implementável de verdade, só apresentável.
- Indicar na lista qual combinação já tem histórico de validação com dado
  real (Corn/Mendeley/acervo privado) e qual é inédita — mesmo padrão de
  honestidade de cobertura do resto do projeto. Hoje nenhum perfil de
  imagem (`bancada`/`celular`/`scanner`) tem qualquer validação publicada;
  isso precisa aparecer explicitamente, não só no código-fonte.
- Criar/salvar perfil combinado: permitir ao usuário salvar um YAML próprio
  que referencie os dois perfis escolhidos (ou os funda num terceiro
  arquivo) para reuso — formato exato a definir na implementação, não
  neste documento.

## 5B.3 Implementação (só depois de aprovação)

Reaproveitar `perfil_matriz.load_profile`/`MatrixProfile` sem duplicar
lógica; teste de contrato garantindo que salvar/carregar um perfil
combinado preserva as duas dimensões (mesmo padrão de roundtrip que já
pegou um bug real de `Config` nesta sessão).

---

# GUARACI — Assistente G (Agente 6, proposta)

> **Status: PROPOSTA, não implementada.** Pausa obrigatória — aguardando
> aprovação de escopo (6.2) antes de qualquer implementação, mesma regra
> das seções anteriores.

## 6.1 Diagnóstico do que `G` faz hoje

Lido `_abrir_assistente`/`_guaraci_revisar_config`/`_guaraci_navegar_secoes`
(`src/guaraci/guaraci.py:1073-1062`). `G` hoje é **decorativo**, nos 3
sentidos que a instrução original cogitava:

- **Passivo**: as duas opções ("Revisar configuração atual" / "Informações
  sobre uma seção") só ecoam o que já está na tela — não lê dado nenhum
  do dataset carregado, não roda nenhuma verificação.
- **Estático**: `_guaraci_revisar_config` aplica um punhado de regras
  hardcoded sobre 9 campos fixos de `Config` (ex.: "pre-proc == 'raw' →
  avisa"); não usa `run_audit`, `applicability_domain`,
  `achievable_alpha` nem nenhuma outra função de diagnóstico real que já
  existe no projeto.
- **Desatualizado por construção**: `_guaraci_navegar_secoes` é um dict
  hardcoded de 8 seções (`"1"` a `"8"`) com descrição escrita à mão — não
  inclui `H`/`B`/`J`/`U`/`K`/`P`/`?`/`A` (6 das 18 abas reais faltam), e
  não deriva de `_CONFIG_SPEC`/`_HELP_DB` (que já existem e já alimentam
  a ajuda `[?]` de cada campo — `G` reinventa uma segunda fonte, pior e
  menor, em vez de reaproveitar a que já existe).

## 6.2 Escopo proposto (aguardando aprovação)

O pedido original tem 4 frentes ((a) diagnosticar, (b) sugerir+executar,
(c) responder, (d) listar técnicas). Fazer as 4 por completo é um projeto
maior que qualquer um dos Agentes 1-5B feitos até aqui nesta sessão — a
proposta abaixo separa o que é **cirurgicamente viável agora** (reaproveita
lógica 100% existente) do que fica para uma rodada futura, para não
prometer mais do que dá pra entregar com o mesmo rigor de teste do resto
do projeto.

**Achado que reduz o risco da frente (a)**: as verificações de diagnóstico
que a instrução pede como exemplo — "dataset sem identificador de
agrupamento", "n insuficiente", "faixa de validação diferente da faixa de
uso", "classe desbalanceada, duplicatas, variáveis constantes" — **já são,
literalmente, as 6 funções `check_*` de `auditoria_delineamento.py`**
(`check_grouping`, `check_class_session_confounding`, `check_duplicates`,
`check_insufficient_n`, `check_validation_use_range`,
`check_external_validation`), já testadas, já usadas pela aba `U`. Não é
lógica nova para escrever — é *wiring* de algo que já existe.

### Fase 1 (proposta para implementar agora, se aprovada)

- **(a) Diagnosticar**: novo item de menu no assistente, "Diagnosticar
  dados carregados" — só roda quando o usuário pede (não em todo `[G]`,
  que seria lento e a maioria das aberturas é só pra ajuda pontual).
  Carrega os dados da `Config` atual e chama `run_audit` — mesmo motor,
  mesma saída da aba `U`, apresentada dentro do assistente. Regra dura já
  válida hoje (aba `U` já segue): nunca inventa número, nunca esconde
  ressalva.
- **(d) Listar técnicas**: novo módulo `technique_registry.py` (mesmo
  padrão de `model_registry.py`/`io_registry.py` já usados no projeto) —
  uma entrada por método (DD-SIMCA, conformal, PLS-DA, PLS-R
  pooled/por-espécie, Kennard-Stone/Duplex/SPXY, PDS/DS, LOD/LOQ/RPD/RER,
  linearidade/robustez, perfis de matriz/técnica) com nome, categoria,
  quando usar, limitação conhecida. "Nunca desatualizado" não vem de AST
  introspection (caro e frágil para código científico) e sim do mesmo
  mecanismo que já existe no projeto para `_CONFIG_SPEC`↔`MENU_FIELDS`
  (`tests/test_interfaces_configuraveis.py`): um teste de contrato que
  falha se uma função pública nova de método científico não tiver entrada
  correspondente no registry — reprovar o build é o que impede
  desatualização de verdade, não a promessa de que "ninguém vai esquecer".
- **(b) Sugerir+executar — só 1 caso, como prova do padrão**: depois do
  diagnóstico rodar, se `check_insufficient_n` (ou equivalente) apontar
  que o `n` atual não atinge o alpha desejado, oferecer "Seu n permite α
  mínimo de X. Quer ver quantas amostras faltam para Y?" chamando
  `achievable_alpha`/`n_minimum_for_alpha` (já existem, só compor o
  texto). Os demais exemplos do pedido original ("quer que eu tente
  extrair agrupamento de mae_id?", "quer marcar como extrapolação?") ficam
  para uma rodada futura — cada um precisa de uma ação de escrita
  diferente (mexer em `Config`, re-rodar validação) que merece o mesmo
  cuidado de teste que o resto do projeto tem, não convém empacotar todos
  juntos sem verificação individual.

### Fase 2 (fora do escopo desta rodada — registrar, não implementar)

- **(c) Responder perguntas livres**: o projeto não tem nenhuma dependência
  de LLM/NLP hoje (é uma ferramenta determinística) — "responder" não pode
  virar um chat de linguagem natural sem mudar a natureza do projeto. Uma
  versão viável é um FAQ curado (poucas perguntas canônicas, casadas por
  palavra-chave, resposta montada a partir do estado real da sessão) — mas
  isso é decisão de escopo própria, não encaixa como "correção cirúrgica"
  desta rodada. Registrado como pendência, não implementado agora.
- Ações de escrita do item (b) além do único caso da Fase 1 (extrair
  agrupamento de `mae_id`, marcar predição como extrapolação, abrir
  `guaraci plan` pré-preenchido).

## 6.3 Implementação (só depois de aprovação da Fase 1 acima)

- `_abrir_assistente` ganha um 3º item de menu ("Diagnosticar dados
  carregados"), reaproveitando `pq.load_data`/`pq.validate_input`/
  `run_audit` exatamente como `_menu_audit` já faz (mesmo padrão de
  `console.status` da padronização 5.3).
- Novo `technique_registry.py`: lista de entradas + teste de contrato
  (mesmo padrão de `test_interfaces_configuraveis.py`) garantindo que
  toda função pública das categorias listadas tem entrada correspondente.
- `_guaraci_navegar_secoes` passa a derivar de `_CONFIG_SPEC` (todas as 18
  abas, não 8 hardcoded) em vez do dict estático atual — fecha o gap de
  desatualização descrito em 6.1 como efeito colateral direto da correção,
  não como escopo novo.
- Testes: cada achado de diagnóstico precisa de contra-prova (dataset
  construído para violar a regra dispara o alerta; dataset limpo não
  dispara) — já existem para as 6 funções `check_*` reaproveitadas
  (`tests/test_auditoria_delineamento.py`); só a sugestão de α mínimo
  (item novo) precisa de teste próprio.

> **Status 2026-09-01: Fase 1 implementada, 3 dos 4 itens acima.** Ver
> commit seguinte a este. Implementado: item 1 (`_guaraci_diagnosticar`,
> opção `[3]` do assistente, reaproveita `run_audit` sem lógica nova),
> item 2 (`technique_registry.py`, 18 entradas, 2 testes de contrato:
> toda `referencia` resolve pra símbolo real, e os módulos de propósito
> único têm cobertura completa de `__all__`), e o caso único de
> sugerir+executar (α mínimo alcançável pra classe mais fraca, função
> pura `_sugestao_alpha_classe_fraca`, testável sem console). **Não
> implementado na hora, fechado depois** (ver commit posterior a este):
> `_guaraci_navegar_secoes` agora deriva de `_t()`/`d_*` (a mesma fonte
> que já alimenta o rodapé/ajuda de cada aba), cobrindo as 18 abas reais
> em vez do dict estático de 8 — não veio de `_CONFIG_SPEC` como a
> proposta original cogitava (esse não tinha o metadado certo), mas da
> fonte de tradução, que é o registro correto no nível de aba.

> **Status 2026-09-01: Fase 2 parcial.** FAQ curado implementado — `[5]`
> no assistente, 4 perguntas canônicas escolhidas por número (não é chat
> de linguagem livre; o projeto não tem dependência de LLM), respostas
> ancoradas no `cfg` real da sessão quando aplicável (regra dura: nunca
> texto genérico solto — a resposta de "qual método usar" muda de verdade
> com `cfg.level`, testado). A sugestão de α mínimo (Fase 1) ganhou a
> parte "executar": responder 's' agora chama
> `plano_coleta.plan_from_statistical_target` de verdade e mostra o
> `n_por_classe`/alertas do plano, não só o texto da sugestão.
> **Não implementado, avaliado e descartado por não ter ação segura
> mapeável**: "extrair agrupamento de `mae_id`" — `grouping_guarantee` já
> É derivado de `mae_id` automaticamente no carregamento; quando vem
> baixo, é porque a estrutura de pasta/CSV genuinamente não permite
> extrair — não há re-tentativa que o assistente possa oferecer sem
> inventar uma garantia que os dados não sustentam. **Não implementado,
> registrado como trabalho futuro real** (precisa mexer em
> `predicao.py`, código já testado, com o mesmo cuidado do resto do
> projeto): marcar predição fora da faixa de trabalho do perfil como
> extrapolação — `MatrixProfile.outside_working_range` existe e é usado
> em `check_validation_use_range` (tempo de calibração), mas nunca é
> chamado em `predict_samples`/`predict_blind` (tempo de predição).
