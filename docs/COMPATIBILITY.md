# Política de compatibilidade

> Este documento não existia antes desta auditoria (Bloco B, 2026-08-26).
> A instrução que motivou esta rodada presumia haver "um mecanismo de
> alias do Config" de uma rodada anterior a estender — verificado por
> `git grep -in "alias\|DeprecationWarning"` em `src/` e `docs/`: **não
> existe nenhum mecanismo de depreciação/alias em funcionamento hoje**.
> O `_CONFIG_SPEC` mapeia chave-YAML↔atributo-Python (ex.: `modo_entrada`
> ↔ `mode`) por motivo de idioma/legibilidade, não por compatibilidade
> entre versões, e chave desconhecida no YAML é **erro**, não aviso (ver
> seção "config.yaml" abaixo). Este documento estabelece a política pela
> primeira vez, não descreve algo pré-existente.

## O que o SemVer cobre a partir da v1.0.0

A superfície coberta por garantia de compatibilidade é exatamente o que
está listado em `__all__` de cada módulo de `src/guaraci/` — nada além
disso. Concretamente:

- **Assinatura de função pública**: nome, ordem/nome dos parâmetros,
  tipos, valores default. Mudar qualquer um desses de forma incompatível
  exige bump de major (ou minor com depreciação, ver abaixo).
- **Formato de dataclass/dict de resultado** das funções públicas —
  campos existentes não somem nem mudam de tipo/significado (`AuditFinding`,
  `CollectionPlan`, `SentinelState`, `IdentificationResult`,
  `QuantificationResult`, `BlindPredictionResult`, `MatrixProfile` etc.).
- **Esquema de `config.yaml`**: as chaves listadas em `_CONFIG_SPEC`
  (`config_io.py`) — nome da chave, tipo, opções válidas, min/max.
- **Nomes de coluna de saída** (CSV/Excel/relatório) gerados pelas
  funções de `resultados_io.py`/`predicao.py`/`reports.py` -- cobertos
  pela POLÍTICA (mudar um nome de coluna sem depreciação é incompatível,
  mesma regra de tudo acima), mas **sem proteção automatizada hoje** --
  ver "Dívida conhecida" abaixo. É a superfície mais provável de ser
  consumida por scripts de usuário automatizando algo em cima do
  GUARACI (ex.: um script que lê `resumo_modelo.txt`/CSV de saída por
  nome de coluna), então a política vale igual mesmo sem o teste --
  só não há alarme automático se alguém violar sem querer.
- **Códigos de saída da CLI** (`guaraci` como comando): `0` sucesso,
  `1` erro de execução, `2` uso incorreto — contrato já documentado no
  próprio `--help` do programa (`guaraci.py:_TEXTO_AJUDA`).

Módulos de módulo-para-módulo dentro do pacote (ex.: `pipeline.py`
reexportando símbolos de `chemometric_stats.py`) são parte do contrato
**pelo objeto reexportado**, não pelo caminho de import — `pipeline.X`
e `chemometric_stats.X` continuam sendo o mesmo objeto (`is`), protegido
por `tests/test_fachada_reexport.py`.

## O que pode mudar sem aviso

- Qualquer identificador com prefixo `_` (função, classe, constante,
  campo de dataclass) — é implementação interna, mesmo quando outro
  módulo do pacote o consome via getattr dinâmico ou quando um teste o
  exercita diretamente (ex.: `_cv_predict_manual`, `_menu_*` de
  `guaraci.py`, `_CFG`/`_check_balance`/`_menu_interativo` de
  `pipeline.py`). O prefixo é o sinal — não "ninguém usa", e sim "não é
  contrato estável".
- Texto de mensagens de log/console, texto de ajuda do menu interativo,
  wording de avisos — comportamento visível mas não estruturado; útil
  para o usuário, não para código que faz parsing.
- Nomes de arquivo/módulo dentro de `src/guaraci/` (ex.: `dados_io.py`,
  `figuras.py` continuam em português) — nunca foram parte do contrato
  de nenhuma versão; só os identificadores *dentro* deles (função/
  classe/campo) são cobertos, conforme decidido e verificado na
  auditoria de nomenclatura desta mesma sessão (ver `PROGRESSO.md`,
  linha "65/66").
- Figuras/relatórios cuja geração depende de `modos_analise.should_generate`
  (quais figuras aparecem para qual objetivo científico) — o *conjunto*
  de figuras geradas pode mudar; a assinatura de cada função `fig_*`
  individual, uma vez pública, segue a regra normal acima.

## Casos especiais documentados (não são dívida, são decisão)

- **`comparar_pipelines`** (campo de `Config`) e **`executar_ag`**/
  `selecao_ag` (campo de `Config`, chave YAML `selecao_ag`) — mantidos
  em português deliberadamente, decisão registrada antes desta sessão.
  Não serão migrados sem uma decisão nova e explícita.
- **`config.yaml` com chave desconhecida é ERRO, não aviso silencioso**
  — decisão deliberada (`config_io.load_config`, corrigida em
  2026-08-20): uma chave digitada errada ou de uma versão diferente do
  software rodar com o DEFAULT em silêncio é pior, para um pipeline
  científico, do que falhar alto. Isso significa que **remover uma
  chave de `_CONFIG_SPEC` é sempre incompatível** (quebra qualquer
  `config.yaml` antigo que a use) — não existe caminho de depreciação
  suave para chaves de config hoje; renomear uma chave exige manter a
  chave antiga funcionando (alias real) ou tratar como major bump.
- **`bootstrap_vip`** (`pipeline.py`) é uma função pública já marcada
  `DEPRECATED` no próprio corpo (redireciona para
  `bootstrap_vip_stratified`) sem nenhum chamador restante no pacote —
  mantida pública precisamente por ser o primeiro caso real de
  depreciação do projeto; serve de modelo para o mecanismo abaixo.

## Prazo e mecanismo de depreciação (política nova, a partir de agora)

1. Uma função/campo/parâmetro público a remover ou renomear é marcado
   com `DeprecationWarning` (`warnings.warn(..., DeprecationWarning,
   stacklevel=2)`) apontando o substituto, no mesmo padrão que
   `bootstrap_vip` já demonstra em prosa (formalizar o `warnings.warn`
   nela é o primeiro item de dívida técnica desta política, não feito
   nesta sessão).
2. Fica assim por **no mínimo 1 versão minor inteira** antes de poder
   ser removida num major seguinte.
3. Exceção: chaves de `config.yaml` (ver acima) — sem alias funcionando,
   uma renomeação de chave É incompatível já na minor em que acontece,
   a menos que a chave antiga seja mantida em paralelo (aceita e
   convertida para o campo novo) até o major seguinte.
4. Toda depreciação é registrada no `CHANGELOG.md` na versão em que
   começa E na versão em que a remoção de fato acontece.

## Dívida conhecida (registrada, não implementada)

**Nomes de coluna de CSV/Excel/relatório não têm proteção de contrato
automatizada.** `tests/test_contrato_api_publica.py` introspecciona
assinatura de função/campo de dataclass/schema de `_CONFIG_SPEC` -- mas
não existe objeto Python introspectável que declare as colunas de um
`DataFrame` retornado por `benchmark_classifiers`, `predict_samples`,
`predict_blind`, ou as colunas dos CSVs escritos por
`resultados_io.save_identifiers`, sem rodar a função e inspecionar o
resultado de fato. Cobrir isso exigiria algo no espírito de
`test_golden_valores.py` (rodar o pipeline sintético, capturar
`df.columns`/cabeçalho do CSV, comparar contra um golden) -- trabalho de
escopo comparável a um Passo próprio, registrado aqui em 2026-08-26
(Passo 77) para não ser esquecido, não implementado por decisão
explícita nesta sessão (candidato a instrução futura, depois de
datasets públicos e Hypothesis estarem em andamento).

Consequência prática enquanto isso não existe: um nome de coluna pode
mudar silenciosamente sem nenhum teste falhar, mesmo violando a
política acima. Quem depende de nomes de coluna específicos (scripts de
usuário automatizando algo em cima do GUARACI) não tem hoje o mesmo
nível de proteção que a assinatura de função/schema de config já têm.

## Aplicação mecânica

Este documento não é a garantia — é a explicação da garantia. O que
falha automaticamente se alguém violar o que está descrito aqui é o
teste de contrato (`tests/test_contrato_api_publica.py`, Passo 73) --
**exceto** para nomes de coluna de saída, ver "Dívida conhecida" acima.
