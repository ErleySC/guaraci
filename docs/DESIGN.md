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

| Papel | Hex (moda) | Hex (média, mais neutro) | Onde aparece na mascote |
|---|---|---|---|
| Laranja | `#FF6400` | `#E26900` | fundo gradiente (topo do balão/frasco) |
| Dourado | `#FFC100` | `#FDA801` | moldura do balão, sol no laptop |
| Verde | `#46A41C` | `#47890B` | fundo (base), folha no cocar |
| Grafite escuro | `#181E21` | `#3E423F` | contorno, laptop, texto |
| Creme claro | `#FFFBD4` | `#FDF2C7` | reflexo/luz, favo de mel de fundo |

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

O sol no laptop da mascote é o elemento mais reaproveitável como marca
gráfica isolada (funciona em tamanho pequeno, silhueta simples, já
carrega a cor dourada). Proposta: usar como favicon/ícone de app (já é,
via `guaraci_icon.png`) e como marcador no cabeçalho padrão de cada tela
(seção 3).

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
