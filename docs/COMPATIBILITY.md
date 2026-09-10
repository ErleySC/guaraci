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
- **Nomes de coluna de saída** (CSV/Excel) gerados pelas funções de
  `resultados_io.py`/`predicao.py`/`avaliacao_modelos.py`/
  `selecao_variaveis.py`/`plano_coleta.py`/`guaraci.py` (menu de seleção
  de amostras) -- cobertos pela POLÍTICA (mudar um nome de coluna sem
  depreciação é incompatível, mesma regra de tudo acima) **e, desde
  2026-08-27 (Passo 89), com proteção automatizada**:
  `tests/test_contrato_saida_tabular.py` roda cada função contra dado
  sintético e compara as colunas reais contra um snapshot golden
  (`tests/golden/contrato_saida_tabular.json`) -- mesmo mecanismo de
  `test_contrato_api_publica.py`, arquivo separado porque a captura
  exige EXECUÇÃO real (não introspecção estática: não existe objeto
  Python que declare as colunas de um `DataFrame` sem rodar a função).
  `reports.py::generate_excel_report` não tem entrada própria: verificado
  por leitura que ele copia `list(df.columns)` verbatim dos CSVs já
  cobertos, então protegê-los já protege a reempacotação em Excel.
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

## Dívida fechada (2026-08-27, Passo 89)

**Nomes de coluna de CSV/Excel agora têm proteção de contrato
automatizada.** Registrada como dívida em 2026-08-26 (Passo 77):
`tests/test_contrato_api_publica.py` introspecciona assinatura de
função/campo de dataclass/schema de `_CONFIG_SPEC`, mas não existe
objeto Python introspectável que declare as colunas de um `DataFrame`
sem rodar a função e inspecionar o resultado de fato. Fechada com
`tests/test_contrato_saida_tabular.py` (Passo 89): roda `save_identifiers`,
`sanitizar_metadados`, `benchmark_classifiers`, `monte_carlo_cv`,
`benchmark_regression_by_species`, `etapa4_selecao_variaveis` (4 CSVs),
`plano_coleta.export_excel`, o menu `[K]` de seleção de amostras,
`predict_samples`, e uma execução completa de `executar()` (para
`teste_martens.csv`/`comparacao_pipelines.csv`, construídos inline no
orquestrador) contra dado sintético, comparando as colunas reais contra
um snapshot golden. Contra-prova: monkeypatch de `save_identifiers`
renomeando `classe_predita` -> `classe_pred` de verdade (não só no dict
do teste) confirma que o mesmo detector usado pelo teste principal
acusa a mudança.

## Aplicação mecânica

Este documento não é a garantia — é a explicação da garantia. O que
falha automaticamente se alguém violar o que está descrito aqui é o
teste de contrato (`tests/test_contrato_api_publica.py`, Passo 73, para
assinatura/schema; `tests/test_contrato_saida_tabular.py`, Passo 89,
para nomes de coluna de saída).
