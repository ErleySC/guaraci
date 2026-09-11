# Escopo — Fusão multibloco e Monitoramento em linha (MSPC)

> Documento de ESCOPO (Item 3 da instrução de 2026-09-10, "Dependabot,
> medição replicada do #12, escopo de produto grande, e dívida de tipo
> do hsi_pipeline"). Não é código, é planejamento — mesmo tratamento já
> dado a outros itens de grande porte (ex.: linearidade/robustez do
> Bloco 13d): medir e propor escopo antes de qualquer implementação.
> Registrado também em `docs/BACKLOG_MULTIAGENTE.md` (P4/P5) para virar
> instrução própria quando priorizado.

## 1. Fusão multibloco

### 1.1 O que significa, concretamente, para o GUARACI

Hoje cada técnica espectroscópica/cromatográfica roda como um bloco
isolado: FT-NIR, HSI, EEM, GC-MS etc. cada uma produz sua própria matriz
`X` (amostras × variáveis) e seu próprio modelo. "Fusão multibloco"
significa combinar **dois ou mais blocos medidos na MESMA amostra
física** num único modelo, em vez de reportar dois modelos separados
que nunca se falam. Três níveis de fusão são candidatos, em ordem
crescente de esforço de engenharia:

1. **Concatenação de baixo nível (low-level fusion)**: concatenar as
   colunas dos blocos (depois de pré-processar cada um separadamente)
   numa matriz única `X_fundida = [X_bloco1 | X_bloco2]` e rodar o PLS/
   PLS-DA já existente sobre ela. Exige só alinhamento por amostra —
   nenhum algoritmo novo.
2. **Fusão em nível médio (mid-level / feature fusion)**: reduzir cada
   bloco a um resumo (scores de PCA/PLS, VIP, ou saída do MCR-ALS já
   existente) antes de concatenar — reduz dimensionalidade e mitiga um
   bloco de alta dimensão "dominar" o outro por contagem de variáveis.
3. **Fusão por decisão (late fusion)**: treinar um modelo por bloco e
   combinar as PREDIÇÕES (votação, média ponderada por confiança) — é o
   que `hsi_multiway.comparar_npls_vs_pixelwise` já faz informalmente ao
   comparar duas abordagens no mesmo dado, mas nunca fundindo blocos de
   TÉCNICAS diferentes.

Nível 1 é o único que corresponde à literatura de "fusão multibloco"
propriamente dita (PLS multibloco / MB-PLS, SO-PLS, ComDim); os outros
dois são combinações mais simples já parcialmente ao alcance do que
existe.

### 1.2 Par de técnicas/dataset para prova de conceito real

Candidato concreto, já parcialmente integrado — não hipotético:

**Mendeley `10.17632/ctgg7k4m5g.2` (óleos comestíveis, ver
`docs/VALIDACAO_PUBLICA.md` §2/2b)**. O artigo original (Ottaway et al.
2021) mediu as **MESMAS amostras físicas de óleo** com 4 técnicas
(NIR 8mm, NIR 24mm, MIR 50µm, Raman) para o mesmo alvo (índice de
peróxido). O GUARACI hoje só baixa e usa `NIR8mm1A.csv` — os outros 3
arquivos-irmãos (`MIR50um1A.csv`, `Raman1A.csv`, `NIR24mm1A.csv`) já
existem no mesmo repositório Mendeley, com correspondência de amostra
por índice de linha (a verificar antes de qualquer código: confirmar
que a ordem de linhas é a mesma entre `NIR8mm1A.csv` e `MIR50um1A.csv`,
ou se há uma coluna de ID de amostra explícita — não visto na
integração atual porque só 1 arquivo foi lido).

Por que este par é o melhor candidato disponível hoje:
- Mesmo alvo de regressão já validado no GUARACI (índice de peróxido,
  `test_regressao_peroxido_roda_sem_excecao_e_calibra_razoavel`) — a
  fusão pode ser medida contra a baseline NIR-só já existente
  (RMSEP/R² já reportados em §2 de `VALIDACAO_PUBLICA.md`).
- NIR 8mm (RMSEP publicado 4,9) e MIR 50µm são fisicamente
  complementares (bandas de overtone vs. fundamental) — é exatamente o
  tipo de par onde a literatura de fusão multibloco relata ganho real,
  não um par arbitrário.
- Nenhum dataset novo para baixar/licenciar — mesmo script
  `baixar_mendeley_oleos.py` já teria acesso ao arquivo MIR (só precisa
  passar a baixá-lo e parseá-lo).
- Risco de "alegação científica falsa" é baixo: o próprio artigo já
  testou e publicou RMSEP por técnica individual, então a fusão pode ser
  comparada contra um número PUBLICADO, não só contra a baseline interna.

Alternativa descartada: HSI (imagem + espectro já fundidos dentro do
mesmo cubo — não são dois BLOCOS de técnicas diferentes, é uma única
medição multidimensional) e EEM (mesma situação — excitação×emissão é
1 técnica, não 2 blocos).

### 1.3 Esforço estimado

Comparável a `eem_multiway.py` (137 linhas + integração ao PARAFAC
existente, Passo 149) somado ao trabalho de alinhamento por amostra já
resolvido para grupos físicos em `hsi_pixels.build_pixel_dataset`
(mesmo padrão de garantia, aplicado a "bloco" em vez de "pixel"). Não é
comparável ao HSI completo (Passos 94-116, ~15 módulos) — é mais
próximo de **1 técnica nova de integração** (como GC-MS/Passo 150, ~1
semana de trabalho incremental): 1 função de alinhamento + parser do
2º arquivo + 1 preset de pré-processamento por bloco + comparação
contra baseline single-block.

Itens concretos, em ordem:
1. Estender `baixar_mendeley_oleos.py` para baixar `MIR50um1A.csv`
   também (mesmo padrão de retry/checksum de `_http_retry.py`).
2. Verificar alinhamento de amostra entre os 2 arquivos (linha a linha
   ou por coluna de ID) — bloqueante, decide se o PoC é viável sem
   trabalho extra de correspondência.
3. Nova função `preprocessamento.build_multiblock_dataset` (ou módulo
   `fusao_multibloco.py` próprio, seguindo a convenção de 1 arquivo por
   técnica/capacidade): concatena blocos já pré-processados
   separadamente (nível 1 da seção 1.1), com guarda explícita de que
   `n_amostras` bate entre os blocos (nunca concatenar silenciosamente
   blocos desalinhados).
4. Rodar PLS de regressão sobre a matriz fundida, comparar RMSEP/R²
   contra a baseline NIR-só já medida — reportar os 2 números lado a
   lado, nunca só o da fusão (mesma disciplina do resto do projeto:
   comparação explícita, não afirmação solta de "melhorou").
5. Teste de contrato (`tests/test_validacao_publica_mendeley.py` ganha
   1 teste novo) + atualização de `docs/VALIDACAO_PUBLICA.md`.

## 2. Monitoramento em linha / MSPC

### 2.1 É extensão natural da sentinela, ou arquitetura nova?

**Extensão natural.** `sentinela_deriva.py` (202 linhas, Bloco 13b) já
implementa o núcleo estatístico de MSPC (*Multivariate Statistical
Process Control*) que importa para este produto: um teste binomial
exato sobre taxa de rejeição do domínio de aplicabilidade ao longo do
tempo, com estado persistente (`save_state`/`load_state`) e alerta
(`DriftAlert`) quando a taxa sobe além do esperado. O que falta não é
um algoritmo novo — é a ORQUESTRAÇÃO em volta dele: hoje
`update_with_predictions`/`check_drift` são chamados manualmente, lote a
lote, por quem roda o script. "Em linha" pede que isso rode sozinho.

### 2.2 O que "em linha" significa de forma realista aqui

O GUARACI é **local-first, sem servidor dedicado** (roda como app
Streamlit local ou CLI, ver `docs/DESIGN.md`) — "em linha" no sentido de
um dashboard MSPC industrial (OPC-UA, streaming contínuo, servidor
sempre ativo) não é uma extensão, é outro produto, fora do que este
projeto é hoje. A interpretação realista, e a única que não exige
arquitetura nova:

> **Execução agendada e frequente, não um servidor contínuo.** A cada
> novo lote de amostras processado (predição via `predict_samples`/
> `predict_blind`, que já produz a coluna `AD_dentro_dominio`), rodar
> automaticamente `update_with_predictions` + `check_drift` sobre o
> estado persistido, e emitir alerta (arquivo, log, ou — se o usuário
> tiver e-mail/webhook configurado externamente — notificação) quando
> `DriftAlert.alerta` for `True`.

Isso é mais próximo do que já existe (execução por lote, local, sem
servidor) do que a categoria "monitoramento em linha" sugere à primeira
vista — não é uma promessa exagerada, é honestidade sobre o que este
projeto pode sustentar sem virar um produto de infraestrutura
diferente.

### 2.3 Escopo mínimo viável

1. **Hook automático no ponto de predição**: `predicao.predict_samples`/
   `predict_blind` (ou o comando `guaraci run`, Passo 208, que já
   executa não-interativamente) passam a chamar
   `update_with_predictions`+`check_drift` automaticamente quando
   `cfg` apontar um caminho de estado da sentinela (`cfg.sentinela_state_path`,
   campo novo, opcional — comportamento atual preservado quando ausente,
   mesma disciplina de retrocompatibilidade de `bootstrap_bca_ci`
   com `groups`, Passo 202/#13).
2. **Saída do alerta**: reaproveita o padrão de log/console já usado no
   projeto (`[WARNING]`) — nenhum canal novo (e-mail/webhook) nesta
   primeira fase; é o candidato mais provável para virar escopo
   PRÓPRIO depois, dado que exigiria credenciais/configuração externa
   ao projeto (SMTP, webhook secret) — superfície de risco diferente
   (ver seção "Ações que exigem permissão explícita" das regras gerais
   desta sessão), não algo para decidir dentro deste documento de
   escopo.
3. **Agendamento**: fica a cargo do usuário/SO (Task Scheduler no
   Windows, cron em Linux/Mac, chamando `guaraci run config.yaml`
   periodicamente) — o GUARACI não precisa de um daemon próprio; é
   consistente com o projeto ser uma ferramenta local-first, não um
   serviço.
4. **Teste**: `tests/test_sentinela_deriva.py` (195 linhas hoje) ganha
   testes do novo hook (predição real dispara `update_with_predictions`
   quando configurado; não dispara quando `cfg.sentinela_state_path`
   ausente).

### 2.4 Esforço estimado

Menor que a fusão multibloco — é 1 campo novo de `Config` +
`_CONFIG_SPEC` (ver `config_io.py`) + 1 chamada condicional nos 2 pontos
de predição já existentes + testes. Sem parser novo, sem dataset novo,
sem algoritmo novo (`sentinela_deriva.py` já faz o trabalho
estatístico). Comparável a T4 (correção de deriva por QC/brancos,
Passo 4 da rodada anterior desta sessão) em tamanho de diff, não ao HSI
completo.

## 3. O que este documento NÃO decide

Nenhum código foi escrito. Prioridade relativa entre os dois itens,
entre eles e o resto do backlog (`docs/BACKLOG_MULTIAGENTE.md`), e o
momento de iniciar cada um ficam para quando o autor emitir instrução
própria — mesma disciplina já aplicada a P4/P5 na rodada de
2026-09-10 anterior a esta.
