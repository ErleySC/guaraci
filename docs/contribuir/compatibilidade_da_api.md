# Política de compatibilidade

**Público:** quem mantém ou contribui com o código · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

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

> A política vale a partir da **v1.0.0**, o primeiro release público. Toda mudança de API
> é registrada pelo golden (`tests/test_contrato_api_publica.py`) e listada no CHANGELOG.

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
- **`apply_quality_gate_and_segment`/`fit_predict_pixel_plsda`** (Passo
  212, 2026-09-10): o tipo de retorno mudou de `Dict[str, object]` para
  `TypedDict` (`QualityGateResult`/`PixelPLSDAResult`) — correção da
  raiz de uma dívida de tipo (`object` solto se propagava por ~10 call
  sites de `hsi_pipeline.py`/`hsi_validation.py`, exigindo `cast()`
  local em cada um). `tests/test_contrato_api_publica.py` acusou a
  mudança (assinatura textual mudou); **decisão explícita do autor**:
  regravar o golden, aceitando como mudança ADITIVA/não-quebradora em
  runtime (`TypedDict` é só anotação de tipo — não existe em runtime,
  `dict.__getitem__`/`.keys()`/etc. continuam idênticos para qualquer
  chamador) **sem decidir bump de versão correspondente** — a v1.0.0
  (Passo 213/214) é o primeiro marco onde o SemVer desta política passa
  a valer de fato; este caso pré-1.0 fica registrado aqui como o
  precedente do que conta como "aditivo" quando a política acima
  (`Assinatura de função pública`) entrar em vigor de verdade.

- **`parse_spc`/`parse_sp`** (`importadores_proprietarios.py`, Grupo 1 do
  mapa de completude, 2026-09-12): duas funções públicas novas (leitores
  de formato SPC/PerkinElmer .sp), mesmo padrão aditivo de `parse_opus`
  já existente. `tests/test_contrato_api_publica.py` acusou a mudança
  (dois símbolos novos no módulo); golden regravado — aditivo puro, sem
  bump de versão correspondente decidido (mesmo precedente do caso
  `apply_quality_gate_and_segment` acima: pré-v1.0.0, a política de SemVer
  ainda não está em vigor de fato).
- **`parse_rmn_bruker`/`parse_cromatograma_hplc`** (`importadores_
  proprietarios.py`, fechamento final do Grupo 1, 2026-09-12): duas
  funções públicas novas (leitor de espectro RMN Bruker processado e de
  cromatograma HPLC/GC Agilent/Waters), mesmo padrão aditivo acima.
  Golden regravado — mesmo precedente, aditivo puro, sem bump de versão.
- **Teste R2 do sentinela de deriva** (Passo 220, 2026-09-19): (a)
  `chemometric_stats.ad_rejection_rate_cv` -- funcao publica nova; (b)
  `SentinelState` ganha `ref_rejeitadas`/`ref_n` e `DriftAlert` ganha
  `teste`/`taxa_referencia`, todos campos OPCIONAIS com default (estado
  `.sentinela.json` antigo continua carregando); (c) `predict_samples`
  ganha 2 colunas de saida (`AD_ref_cv_rejeitadas`/`AD_ref_cv_n`) so'
  quando o pacote de modelo as tem. **Mudanca de COMPORTAMENTO
  intencional e autorizada**: com a referencia, `check_drift` deixa de
  testar contra o `alpha` nominal e passa a testar contra a taxa de
  rejeicao da propria calibracao (Fisher, duas amostras) -- o teste antigo
  dava falso alarme acima do nominal. Modelos sem a referencia (salvos antes
  do Passo 220) continuam no teste antigo, com aviso na mensagem e em
  `DriftAlert.teste == "binomial_nominal"`. Golden regravado; aditivo na
  assinatura, nao no comportamento do alerta.
- **`predicao.anexar_colunas_fluxo_cego`** (auditoria de confiabilidade de
  2026-09-19, varredura de duplicacao): funcao publica nova que passa a ser
  a FONTE UNICA das 12 colunas do fluxo cego (Detectar->Identificar->
  Quantificar), antes montadas por duas copias identicas no menu do CLI e na
  aba web. Nenhum nome de coluna mudou (contrato de saida tabular intacto);
  aditivo, mesma nota de honestidade dos itens abaixo (pos-v1.0.0).
- **`model_export.py`** (modulo inteiro, novo -- fechamento do Grupo 2,
  exportacao portatil de modelo em JSON puro, sem pickle/joblib):
  `export_portable_json`, `save_portable_json`, `load_portable_json`,
  `predict_portable`, `UnsupportedModelError`. Mesmo padrao aditivo e
  mesma nota de honestidade dos itens abaixo (pos-v1.0.0).
- **`benchmark_classifiers`/`fig_shap_benchmark`** (`avaliacao_modelos.py`,
  fechamento do Grupo 2 -- explicabilidade fora do treino): dois
  parametros OPCIONAIS novos em cada uma (`X_holdout_raw`/
  `y_holdout_int`, default `None`) -- assinatura antiga continua valendo
  sem mudanca (chamador que nao passa os dois novos cai no comportamento
  ANTERIOR, explicar uma subamostra do treino). Mesmo padrao aditivo e
  mesma nota de honestidade dos itens abaixo (pos-v1.0.0).
- **`fusao_multibloco.py`** (modulo inteiro, novo -- fechamento do Grupo 2,
  fusao multibloco de nivel 1): `build_multiblock_dataset`,
  `fit_evaluate_pls_regression`, `RegressionFitResult`,
  `BlockMismatchError`. Mesmo padrao aditivo (modulo novo, nenhum
  simbolo existente mudou) e mesma nota de honestidade do item logo
  abaixo (pos-v1.0.0).
- **`conformal_margin_classification`/`conformal_prediction_set`**
  (`conformal.py`) e **`hook_apos_predicao`** (`sentinela_deriva.py`),
  fechamento do Grupo 2 (conjunto de predição conforme para classificação
  multiclasse + orquestração compartilhada CLI/web da sentinela de
  deriva): três funções públicas novas, mesmo padrão aditivo (nenhuma
  assinatura existente mudou). Golden regravado — **nota de honestidade,
  diferente dos dois casos acima**: estes três símbolos foram adicionados
  DEPOIS da tag `v1.0.0` (`cc0e14c`, 2026-09-11), quando a política de
  SemVer descrita no topo deste documento já estava, pela própria data,
  em vigor — o precedente "pré-v1.0.0, política ainda não vale" citado
  pelos dois itens de Grupo 1 acima (`parse_spc`/`parse_sp`,
  `parse_rmn_bruker`/`parse_cromatograma_hplc`, ambos também datados de
  2026-09-12) não se sustenta cronologicamente para nenhum dos três: a
  prática real do projeto, desde a tag v1.0.0, tem sido regravar o golden
  como aditivo sem decidir um bump de minor correspondente, em vez de
  seguir a própria regra escrita acima ("mudar assinatura pública... exige
  bump de major/minor"). Registrado aqui em vez de silenciado; a
  reconciliação (bump de minor retroativo cobrindo todas as adições
  aditivas pós-v1.0.0, ou uma emenda explícita à política permitindo
  adição pura de símbolo sem bump) fica para decisão do autor, não
  presumida por esta rodada.
- **`aumento_dados.py`** (módulo inteiro, novo — VRM opcional, nunca
  default) e **`dual_spls.py`** (módulo inteiro, novo — Dual-sPLS norma
  lasso, opção de regularização ao lado de Ridge/Lasso/Elastic Net em
  `benchmark_regression_by_species`): candidatos do Grupo 2 do mapa de
  completude, implementados e conferidos pelo portão de aceite (Bloco 20)
  nesta rodada (varredura final, 2026-09-20). Mesmo padrão aditivo (módulos novos, nenhum símbolo
  existente mudou de assinatura) e mesma nota de honestidade dos itens
  acima (pós-v1.0.0): golden regravado sem bump de minor decidido,
  reconciliação continua pendente de decisão do autor.
- **`tecnicas_avancadas.py`** (módulo inteiro, novo — orquestração testável
  de ASCA/EPO-GLSW/MCR-ALS/fusão multibloco) e **`leitores_avancados.py`**
  (módulo inteiro, novo — registra 7 leitores de formato no `io_registry`:
  `opus`/`spc`/`sp`/`rmn`/`hplc`/`gcms`/`eem`): fecham a lacuna encontrada
  na auditoria de acessibilidade CLI/web de 2026-09-23 (funções já
  implementadas e testadas em isolamento, mas sem caminho de execução real
  em nenhuma interface). `config.py` ganhou 1 campo novo (`hplc_detector`)
  e o `mode` passou a aceitar 7 valores novos (mesmo campo, mudança
  aditiva). Mesmo padrão aditivo e mesma nota de honestidade dos dois itens
  acima: golden regravado sem bump de minor decidido, reconciliação
  continua pendente de decisão do autor.
- **`hsi_extras.py`** (módulo inteiro, novo — orquestração de domínio de
  aplicabilidade/reamostragem/multiway HSI, 2026-09-23): mesmo padrão
  aditivo e mesma nota de honestidade dos itens acima; golden regravado.
- **`catalogo`: grupos e pastas do menu da CLI** (guia v2, 2026-10-15):
  constantes novas `GRUPOS_CLI`, `SECOES_TELA`, `TECLAS_ANTIGAS`; funções
  novas `grupos_cli`, `telas_do_grupo_cli`, `secoes_da_tela`, `guia_tela`;
  `campos_da_tela` ganha o argumento nomeado opcional `basicos_primeiro`
  (default `False`, comportamento antigo). Golden regravado — aditivo puro,
  sem bump de versão.

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

## Registro de mudanças aditivas

- 2026-10-16: novo módulo `guaraci.grupos_figuras` (tabela prefixo → subpasta
  das figuras, README da pasta e chave de ordem) e chave
  `figuras.ORGANIZAR_POR_GRUPO` (padrão `False`). Aditivo: nenhuma assinatura
  existente mudou; golden regravado. Pela CLI a chave é ligada e as figuras
  saem em `Graficos/<grupo>/`; `app_logic.list_figures` ordena pelo nome plano,
  então os relatórios escolhem as mesmas figuras com ou sem subpastas.
- 2026-10-16: novo módulo `guaraci.figuras_extras` (scree, coeficientes,
  Williams, faixa ±RMSEP, MCR-ALS, PARAFAC, ASCA, mapa EEM, cromatograma,
  transferência). Aditivo; golden regravado.
- 2026-10-16: `figuras.fig7_pls_regression` ganhou os keyword-only opcionais
  `y_holdout`, `yhat_holdout` e `rmsep_holdout` (pontos do holdout externo no
  predito x referência) e `pipeline.pls_regressao_pooled` ganhou
  `wavenumbers=None` (eixo da figura de coeficientes b do modelo final).
  Aditivo, padrões mantêm o comportamento anterior; só figuras, nenhum número
  de saída muda. Golden regravado.

- 2026-10-16: nova classe `guaraci.preprocessamento.RemocaoRaiosCosmicos`
  (Whitaker & Hayes 2018), campos `Config.apply_despike` (padrão `False`),
  `despike_limiar` e `despike_janela`, e chaves de config
  `remover_raios_cosmicos`, `raios_cosmicos_limiar`, `raios_cosmicos_janela`.
  Aditivo: desligada, `build_preprocessor` devolve o mesmo pipeline de antes;
  golden regravado.

- 2026-10-16: `guaraci.confianca.entradas` e `confianca.lista` ganharam o
  keyword-only `lang="PT"` (nomes de técnicas e opções no idioma da interface) e
  `catalogo.tecnicas("EN")` devolve o nome em inglês. Aditivo; o padrão mantém
  o comportamento anterior. Golden regravado.

- 2026-11-06: `figuras.save` passou a devolver o caminho gravado (antes
  `None`) e `texto_leiame` ganhou o parâmetro opcional `lang`. Também entraram
  os módulos novos `guaraci.idioma_relatorio` e `guaraci.idioma_figuras`.
  Aditivo; golden regravado.

- 2026-11-08 (checkup das técnicas): `eem_multiway.consistencia_nucleo` é uma
  função nova, e `ParafacEEMResultado` ganhou o campo `consistencia_nucleo`.
  `icp_multielementar.aplicar_censura` ganhou o keyword-only `por_elemento=False`
  e `TabelaMultielementar` ganhou o campo `lod`.
  `leitores_avancados.load_generic_folder_dataset` ganhou `integrar=False`.
  Aditivo: os padrões mantêm as chamadas antigas válidas. Golden regravado.

- 2026-11-08 (fase 2 do checkup das técnicas, dentro da v1.0.0): entraram
  módulos novos (`hsi_blocos`, `extras_opcionais`) e funções e constantes novas
  em `alinhamento_retencao`, `classificadores`, `dados_imagem`, `eem_io`,
  `eem_multiway`, `eixo_espectral`, `hsi_validation`, `preprocessamento`,
  `resumo_parse` e outros. Vários leitores e funções ganharam parâmetros
  opcionais com padrão no fim da assinatura (`ignorar`, `mascara`, `max_pontos`,
  `lazy`, `nao_negativo`, `rotulos_treino`, entre outros). Os campos novos dos
  dataclasses têm padrão. Em `Config`, alguns campos novos entraram no meio da
  lista; quem cria `Config` por nome, que é o uso documentado, não é afetado.
  Algumas anotações de retorno ficaram mais precisas e nenhum parâmetro antigo
  mudou ou saiu. **Mudança de comportamento:** `parafac_eem` passou a usar
  `nao_negativo=True` por padrão. Esse e os outros padrões que mudam resultado
  estão no CHANGELOG da v1.0.0. Golden regravado.

- 2026-11-10 (pendências da fase 2, dentro da v1.0.0): a função nova
  `spectra_preview.remover_faixas_excluidas`, e `preparar_dados_hsi` ganhou 4
  parâmetros keyword-only cujos padrões mantêm o comportamento anterior.
  Aditivo; golden regravado.

- 2026-11-10 (validação por sessão automática, dentro da v1.0.0):
  `unidade_agrupamento` aceita o valor `auto`, que passa a ser o padrão.
  Entraram duas funções novas em `plano_coleta`, e `run_audit` ganhou o
  keyword-only `unidade`. As assinaturas são aditivas. **Mudança de
  comportamento:** com o padrão `auto`, um delineamento com sessões pode passar
  a ser validado por sessão (ver CHANGELOG). Golden regravado.

- 2026-11-10 (instalação guiada, dentro da v1.0.0): entraram em
  `extras_opcionais` a constante `EXTRAS_DE_CONFORTO` e as funções
  `plano_instalacao` e `requisitos_do_extra`, usadas por
  `guaraci doctor --instalar`. Aditivo; golden regravado.

- 2026-11-10 (auditoria, dentro da v1.0.0): em `transferencia_calibracao`,
  `direct_standardization` e `piecewise_direct_standardization` passaram de
  `alpha: float = 1.0` para `alpha: Optional[float] = None`. Quem passa `alpha`
  explícito não é afetado. **Mudança de comportamento:** sem `alpha`, ele é
  escolhido por validação deixando uma amostra de fora; o padrão fixo de 1,0
  zerava a transformação em escala de absorbância. O comando `guaraci
  transferir --alpha` segue o mesmo padrão. Golden regravado.

## Aplicação mecânica

Este documento não é a garantia — é a explicação da garantia. O que
falha automaticamente se alguém violar o que está descrito aqui é o
teste de contrato (`tests/test_contrato_api_publica.py`, Passo 73, para
assinatura/schema; `tests/test_contrato_saida_tabular.py`, Passo 89,
para nomes de coluna de saída).

- 2026-11-10 (saturação do detector, dentro da v1.0.0): o módulo novo
  `saturacao` e o campo `Config.saturacao` (`auto | avisar | nenhuma`, padrão
  `auto`) entraram. **Mudança de comportamento:** nos modos `dx`, `opus`, `spc`,
  `sp` e `csv`, dado com plato de absorbância no teto do detector passa a ter as
  variáveis e os espectros saturados fora do modelo (ver CHANGELOG); `avisar`
  e `nenhuma` mantêm o comportamento anterior. Golden regravado.

- 2026-11-10 (instalação só a partir do clone, dentro da v1.0.0): `extras_opcionais.COMANDO_TUDO` e
  `comando_pip` passam a devolver `pip install ".[all]"` e `pip install ".[extra]"` (a instalação é sempre
  a partir do clone: não há pacote no PyPI). Só muda texto; nenhuma assinatura mudou.
