# PROGRESSO — Passos 92-95 (2026-09-01)

## Passo 92 — Verificação da literatura citada em INSTRUCAO_HSI_MINIMO_VIAVEL.md

3 referências citadas na instrução, verificadas ANTES de qualquer uso em
código/documentação (regra "não citar se não confirmar"):

- `S0031320325004960` ("Revisão de classificação HSI entre domínios") —
  **confirmado**: "Cross-domain hyperspectral image classification",
  *Pattern Recognition* 168, dez/2025. Tema bate exatamente.
- `S0169743926002212` ("Revisão sobre transferência de calibração e
  incerteza") — **confirmado** via WebSearch (snippet retornou o PII
  exato): "Chemometric and machine-learning strategies for calibration
  transfer", *Chemometrics and Intelligent Laboratory Systems*, 2026.
- `S2772375526007070` ("Framework de padronização e reprodutibilidade em
  HSI") — **NÃO confirmado**. O ISSN implícito (2772-3755) corresponde a
  um periódico real (*Smart Agricultural Technology*, Elsevier,
  tematicamente compatível), mas o artigo especifico nunca apareceu em
  nenhuma busca (WebSearch por PII exato, por título aproximado, por
  termos-chave da descrição). Acesso direto ao ScienceDirect bloqueado
  (WebFetch: 403; Browser pane: CAPTCHA Cloudflare) — sem via alternativa
  de confirmação disponível nesta sessão. **Não citado** em nenhum lugar.

## Passo 93 — Busca de dataset público de HSI (prioridade sobre implementação)

Candidato escolhido: **DeepHS Fruit** (Varga, Makowski & Zell, IJCNN
2021, arXiv:2104.09808, github.com/cogsys-tuebingen/deephs_fruit) —
subconjunto Kaki (caqui) / câmera VIS (Specim FX10, 224 bandas,
397,66-1003,81 nm), 56 gravações / 38 frutas físicas, rótulo real
`ripeness_state` (unripe/perfect/overripe).

Candidato alternativo considerado (Mendeley `gjwx64sgkp`, bagas de uva,
CC BY 4.0) — descartado: não foi possível confirmar se distribui cubo
BRUTO ou só espectro já extraído (o segundo não serve para
segmentação/mapa espacial, Passos 96-99).

Formato confirmado por leitura DIRETA (HTTP Range requests no
`Kaki.zip` de 2,2G, sem baixar o arquivo inteiro — leitura só do
directorio central + membros necessários): par ENVI `.hdr` (texto) +
`.bin` (float32, BIP, sem header embutido). Comprimentos de onda vêm à
parte, no JSON de anotações oficial do dataset (`cameras[].wavelengths`
por câmera).

**Agrupamento por objeto físico** (crítico para o Passo 97): confirmado
por leitura direta do JSON de anotações que "frente"/"costas" da MESMA
fruta compartilham `storage_days` e `ripeness_state` dentro do mesmo
dia — group_id = `f"{day}_{numero_da_fruta}"`.

## Passo 94 — `src/guaraci/hsi_io.py`

Leitor ENVI genérico (`load_envi_cube`, aceita bip/bil/bsq, qualquer
`data type` ENVI suportado, `wavelengths` externo quando o `.hdr` não
traz) + leitor específico do subconjunto DeepHS/Kaki
(`load_deephs_kaki_dataset`). 12 testes (11 sintéticos + 1 contra o
dataset real, `GUARACI_DATASETS_DIR`-gated). Commit `5f3ec85`.

## Passo 95 — `src/guaraci/hsi_quality.py`

Quality gate fail-fast (saturação/faixa, SNR via Immerkaer 1996, fração
de pixels válidos) — rejeita com motivo único e específico, nunca
processa em silêncio. Contra-prova obrigatória da instrução (cubo
saturado e cubo de SNR baixo, ambos rejeitados) — 8 testes. Calibração
radiométrica por referência branco/preto **não implementada** nesta
rodada: o dataset escolhido já vem calibrado e não há cubo de referência
bruto disponível para testar essa etapa de verdade — documentado, não
escondido. Commit `f3de9ca`.

## Dataset baixado e infraestrutura de reprodução

`scripts/download_datasets/baixar_deephs_kaki.py` — usa HTTP Range para
extrair só os 112 arquivos (56 gravações × .hdr+.bin) do Kaki.zip de
2,2G sem baixar o arquivo inteiro, cada um com SHA256+tamanho pinado
(verificado ANTES de gravar, mesma regra de
`baixar_mendeley_oleos.py`). Testado de verdade: cache-hit (pins batem
com os 112 arquivos já extraídos) e extração fresca (pasta vazia, exercita
o caminho de rede real) — ambos confirmados por execução direta, não
suposto.

Licença do DeepHS Fruit: **não declarada formalmente** (sem SPDX no
repo/README, API do GitHub devolve `license: None`) — ver retratação em
`docs/VALIDACAO_PUBLICA.md` §4 (uma busca inicial via WebSearch sugeriu
CC BY-SA 4.0; não confirmado por verificação direta, corrigido antes de
entrar em qualquer citação).

## Passo 96 — `src/guaraci/hsi_segmentation.py`

PCA (PC1) + Otsu (implementado do zero -- scikit-image e' dependencia
OPCIONAL do projeto). Distincao documentada do PCA de dominio de
aplicabilidade (`chemometric_stats.applicability_domain`) -- uso
espacial por pixel de UMA cena, nao distancia a um modelo pre-treinado.
Sem mascara de referencia no dataset -- validado por INSPECAO VISUAL
DOCUMENTADA (`resultados_hsi_segmentacao/kaki_segmentacao_amostra.png`,
gitignorado): confirma visualmente que a mascara isola o contorno real
da fruta. Cena sintetica com objeto conhecido: IoU>0.8. Commit `7de3727`.

## Passo 97 — `src/guaraci/hsi_pixels.py`

Extracao de espectros de pixel da ROI + `group_id` de objeto fisico
replicado por pixel (frente/costas da MESMA fruta compartilham
group_id, confirmado por leitura direta do JSON de anotacoes). Contra-
prova OBRIGATORIA (Hypothesis, numero de objetos e pixels/objeto
aleatorios): `StableStratifiedGroupKFold` (o splitter group-aware JA
padronizado no projeto) nunca separa pixels do mesmo objeto entre
treino/validacao, em nenhum fold. Commit `1c179f2`.

## Passo 98 — `src/guaraci/hsi_classification.py`

PLS-DA por pixel (reaproveita `avaliacao_modelos.PLSDAClassifier`, nao
reimplementado), split group-aware, numero de LVs por parsimonia de
Wold (mesmo criterio de `pipeline.py`, generalizado p/ classificacao
via 1-balanced_accuracy). Agregacao por objeto: classe majoritaria +
heterogeneidade (fracao de pixels em desacordo).

**Medido contra o dataset real** (8 objetos de teste de 38 totais,
split group-aware, seed=0, n_components=8 selecionado por Wold):
**3/8 objetos corretos** -- o modelo colapsa quase todo para "perfect"
(classe majoritaria, 42/56 gravacoes; overripe=12, unripe=2).
Desbalanceamento severo, nao corrigido nesta rodada (fora do escopo do
"minimo viavel" -- rebalanceamento/reponderacao seria proximo passo
natural, nao feito aqui p/ nao inflar o resultado por ajuste ad-hoc).
Reportado honestamente, mesmo padrao ja' registrado p/ o Mendeley
(`docs/VALIDACAO_PUBLICA.md` §2: bal.acc 0,35 CV). Confirma que o
pipeline mecanico (segmentacao -> extracao -> classificacao ->
agregacao) funciona ponta-a-ponta sobre dado real -- nao que o
desempenho e' bom.

---

# PROGRESSO — Passos 84-87 (2026-08-27)

> Log de progresso do checkout ativo (OneDrive). Convenção: um bloco por
> Passo, evidência ou silêncio (nenhuma prosa de "corrigido"/"confirmado"
> sem comando/teste que sustente a afirmação).

## Passo 84 — Extensão do bug de `matrix_profile` (Passo 83)

**Pergunta:** o bug corrigido no Passo 83 (`matrix_profile` resetava para
`"generico"` no ciclo salvar/carregar de `config.yaml`, porque o campo
nunca esteve em `_CONFIG_SPEC`) afetou alguma validação pública já
reportada como concluída (Corn, Mendeley)?

**Resposta: NÃO.** Evidência:
- `tests/test_validacao_publica.py` e `tests/test_validacao_publica_mendeley.py`
  constroem `pq.Config(matrix_profile=...)` diretamente em memória e chamam
  `pq.executar(cfg)` na sequência — `grep -n "save_config\|load_config"`
  nos dois arquivos retorna vazio.
- `save_config`/`load_config` só são acionados pelo menu interativo de
  terminal (`_menu_interativo`, `pipeline.py:2971-3027`) e pelo fluxo
  `[S]`/`[L]` da CLI — nenhum dos dois entra no caminho das validações.
- `.github/workflows/test.yml` (jobs `validacao-publica` e
  `validacao-publica-mendeley`) roda `pytest tests/test_validacao_publica*.py`
  direto, sem etapa de `config.yaml` no meio.

Nenhuma revalidação necessária; nenhum número publicado mudou.

## Passo 85 — Hypothesis (testes de propriedade)

- `hypothesis>=6.100,<7.0` adicionado como dependência de desenvolvimento
  (`pyproject.toml` extra `[dev]`; NUNCA em `requirements.txt`, que é o
  manifesto de deploy).
- `tests/test_propriedades_hypothesis.py`: 3 propriedades + 3 contra-provas
  documentadas — roundtrip de `config.yaml` (generaliza o Passo 83 para
  TODOS os campos de `_CONFIG_SPEC`), quantificação cega nunca depende do
  rótulo verdadeiro, split group-aware nunca separa réplica física (cobre
  os 3 splitters do Passo 87 desde que existiram).
- **Achado real, ANTES de qualquer commit**: o próprio teste de roundtrip
  achou 2 bugs de silêncio em `_fmt_yaml` (`config_io.py`) —
  (1) string `str`/`str_opcional` com forma YAML-ambígua ('010'→int 8
  octal, '1.50'→perde zero, '0x1A'→26) saía sem aspas; (2) item de lista
  contendo `?` quebrava ou virava mapa em silêncio dentro de `[a, b]`.
  Corrigido usando `yaml.safe_load` como oráculo + `?` no conjunto de
  caracteres que força aspas em item de lista. Confirmado por
  reversão manual: sem a correção, os `@example` fixados no teste falham
  de forma determinística (não dependiam de sorte da busca aleatória —
  medido: 80 exemplos aleatórios sozinhos NÃO pegavam o bug de forma
  confiável, por isso os `@example` foram fixados).
- Commit: `test: Hypothesis (testes de propriedade) + achado real de
  config.yaml (Passo 85)`.

## Passo 86 — Transferência de calibração entre instrumentos

- `src/guaraci/transferencia_calibracao.py` (novo módulo, `__all__` desde
  o início): Direct Standardization (DS) e Piecewise Direct
  Standardization (PDS) — Wang, Veltkamp & Kowalski (1991),
  *Multivariate instrument standardization*, DOI `10.1021/ac00023a016`
  (verificado no Crossref).
- `tests/test_transferencia_calibracao.py`: contrato de forma/erro +
  redução de erro em dados sintéticos + contra-prova (mestre/escravo SEM
  relação real → PDS não melhora).
- **Validado contra o Corn real** (3 espectrômetros, mesmas 80 amostras):
  RMSEP proteína m5→mp5 sem transferência ≈ 0,51; com PDS (15 amostras de
  transferência, janela=5, alpha=0,001) ≈ 0,16 — quase o nível do m5
  sozinho (≈ 0,148). Hiperparâmetros medidos empiricamente contra o
  dataset, não adivinhados (ver `check_corn_transfer.py` no scratchpad da
  sessão para a varredura). DS não reduziu o erro de forma relevante neste
  par de instrumentos — achado honesto, não escondido.
- Reexportado em `pipeline.py`; contrato de fachada
  (`tests/test_fachada_reexport.py`) e contrato de API pública
  (`tests/golden/contrato_api_publica.json`) atualizados.
- Limitações documentadas em `docs/MANUAL.md` §2.2b (nº mínimo de amostras
  de transferência, sensibilidade de `alpha`/`janela`, pressupõe
  deslocamento linear/local).

## Passo 87 — Seleção de amostras (Kennard-Stone, Duplex, SPXY)

- Kennard-Stone (`kennard_stone`/`kennard_stone_split`/
  `kennard_stone_split_group_aware`) já existia — reaproveitado, não
  reimplementado.
- Completado com `duplex_split`/`duplex_split_group_aware` (Snee, 1977,
  DOI `10.1080/00401706.1977.10489581`) e `spxy_split`/
  `spxy_split_group_aware` (Galvão et al., 2005, DOI
  `10.1016/j.talanta.2005.03.025`) em `src/guaraci/dados_io.py`, mesma
  disciplina group-aware do Kennard-Stone (nunca separa réplica física
  entre calibração/validação — garantido por teste de propriedade
  Hypothesis parametrizado nos 3 splitters).
- `tests/test_selecao_amostras.py`: contrato de partição, proporção,
  casos degenerados (n=0/1/2), group-aware, e uma contra-prova específica
  do motivo de existir do SPXY (KS puro pode deixar de fora o extremo do
  TEOR se ele não for também extremo espectral; SPXY não deixa — caso
  sintético reproduz isso).
- Integrado à CLI: menu principal, tecla `[K]` *Seleção de Amostras*
  (Bloco 10, ao lado do planejamento de coleta `[J]`) — lê um CSV de
  espectros, roda o método escolhido, grava cópia com coluna
  `calibracao`/`validacao`. 3 testes CLI ponta-a-ponta (Kennard-Stone,
  SPXY com coluna alvo, contra-prova de arquivo ausente).
- Documentado em `docs/MANUAL.md` §2.2c.

## Estado da suíte (Passos 84-87)

Commit do Passo 85: 987 testes (incl. Corn real) + ruff limpos.
Passos 86+87: 1008 testes + ruff limpos.

---

# Bloco 13d + varredura geral (2026-08-27, mesma sessão)

## Frente 1 — Bloco 13d: linearidade e robustez formais

- `src/guaraci/linearity.py` (novo, `__all__` desde o commit inicial):
  `lack_of_fit_test` — teste F de falta de ajuste clássico (Draper &
  Smith, cap. 2.6), nível da curva = grupo de réplica física (`mae_id`,
  L2). Contra-prova: curvatura sintética deliberada produz F
  significativo, e F cresce com a magnitude da curvatura.
- `src/guaraci/robustness.py` (novo, `__all__` desde o commit inicial):
  perturbação controlada (pré-processamento, ruído gaussiano, deriva de
  linha de base) + protocolo que reporta variação como INTERVALO, nunca
  binário (R2). Cobre PLS-R e PLS-DA (R3). Contra-prova: perturbação
  maior produz variação maior.
- Integrado ao dossiê via `append_linearity_robustness_model_card`
  (mesmo mecanismo append-only de regressão/identificação/pureza).
- **Validado contra Corn E Mendeley reais**: nos dois, sem `mae_id`
  (réplica física), o teste de linearidade reporta corretamente "não
  computável" — achado honesto (L2), não um bug. Protocolo de robustez
  roda e reporta intervalo em ambos (RMSEP no Corn, bal.acc no
  Mendeley).
- **Decisão de escopo NÃO tomada sozinha (reportada)**: os dois módulos
  NÃO estão fiados automaticamente em `executar()` via novo campo de
  `Config` — isso mudaria o comportamento/custo padrão de toda execução
  do pipeline (robustez roda múltiplos refits) e é uma decisão de
  produto, não um ajuste mecânico "dentro do que já é interno". As
  funções existem, são públicas, testadas e validadas contra dado real;
  faltaria só a decisão de fiação automática + nome/default do flag de
  `Config`, se for para acontecer.
- `mypy`: os 3 módulos novos desta sessão (`linearity.py`,
  `robustness.py`, `transferencia_calibracao.py` do Passo 86) passam
  limpos e cabem no critério já documentado (sem I/O/UI/estado global)
  — adicionados ao gate da CI (ver Frente 3a).
- Commit: `feat: linearidade formal (lack-of-fit) + protocolo de
  robustez (Bloco 13d, Frente 1)`.

## Frente 2 — Infraestrutura de Hypothesis fortalecida

- Auditoria dos 3 grupos de propriedade existentes: só o roundtrip de
  config tinha `@example`. Adicionado `@example` para quantificação
  cega (reproduz o cenário de envenenamento do teste manual original) e
  3 `@example` defensivos de fronteira para o split group-aware (limiar
  `n_grupos=4` onde o colapso por grupo liga) — documentado
  explicitamente que não há bug histórico conhecido para essa
  propriedade (ao contrário do roundtrip), para não sugerir cobertura
  que não existe.
- Profile diferenciado (`conftest.py`): `dev` (50 exemplos, local) vs
  `ci` (300 exemplos, auto-selecionado via `CI=true`, já setado pelo
  GitHub Actions — nenhuma mudança em `test.yml` necessária).
  `max_examples=` por teste removido em favor do profile ativo.
- `.hypothesis/` e `.pytest_cache/` no `.gitignore` (cache local, não
  fonte de verdade).
- `CONTRIBUTING.md`: nova seção documentando a lição medida no Passo 85
  e a convenção resultante.
- Commit: `test: fortalece infraestrutura de Hypothesis -- profile
  CI/local + @example auditados (Bloco 13d, Frente 2)`.

## Frente 3 — Varredura geral

**3a — type-checking.** Medido por comando direto (`mypy` local): os 3
módulos novos desta sessão (linearidade, robustez, transferência de
calibração) passam limpos e cabem no critério de escopo já documentado
(pyproject.toml) — adicionados ao gate da CI, custo zero (nenhum erro
para corrigir). `dados_io.py`/`guaraci.py`/`config_io.py` continuam
FORA do gate por critério — têm I/O/UI, fora do escopo por desenho, não
por descuido.

**3b — segurança.** Nenhum `eval`/`exec`/desserialização insegura nova
encontrado. `subprocess`/`os.system` existentes são todos strings
literais ou listas de argumento (sem `shell=True` com entrada do
usuário), já auditados em 2026-08-07. O único script de download
(`baixar_mendeley_oleos.py`) já segue a disciplina correta (HTTPS,
tamanho+SHA-256 pinados, verificados ANTES de gravar em disco) —
documentado como convenção obrigatória em `docs/VALIDACAO_PUBLICA.md`
§6 para qualquer script futuro. `pip-audit` contra o ambiente inteiro
(incl. `hypothesis`): **nenhuma vulnerabilidade conhecida**.

**3c — documentação de alto nível.** `README.md`/`README.pt-br.md`:
lista de funcionalidades atualizada (mode cego, planejamento
experimental, auditoria de delineamento, linearidade/robustez,
transferência de calibração, seleção de amostras). `paper/paper.md`:
contagem de testes stale (779) corrigida para "1000+"; parágrafo novo
cobrindo as funcionalidades pós-reposicionamento, com 4 referências
novas em `paper.bib` (Wang-Veltkamp-Kowalski 1991, Snee 1977, Galvão et
al. 2005, Draper & Smith 1998), DOIs verificados no Crossref.
`CITATION.cff`: verificado — versão/data consistentes com
`pyproject.toml`, nenhuma mudança necessária (bumping de data sem bump
de versão seria enganoso).

**3d — comparativo com concorrentes.** Verificado por busca (não
presumido): Kennard-Stone **já é** funcionalidade padrão do Unscrambler
(confirmado); PDS/transferência de calibração é método clássico,
razoável supor presente em suites comerciais maduras mesmo sem
confirmação direta — por isso **NÃO adicionados** à tabela comparativa
como diferenciais (seria uma alegação de exclusividade sem lastro). Não
encontrada evidência de que concorrentes ofereçam planejamento
experimental automatizado, auditoria de confundimento ou identificação
de conjunto aberto calibrada por predição conforme — mas ausência de
evidência não é prova; por isso essas funcionalidades foram adicionadas
como itens de lista (fato, sem comparação) na seção *Features*, não
como linha nova na tabela comparativa (que faz uma alegação
competitiva). Tabela comparativa do README mantida como estava.

## Estado da suíte (Bloco 13d + varredura)

1042 testes (incl. Corn e Mendeley reais) + ruff limpos após Frente 1 e
Frente 2. Frente 3 é só documentação (README/paper/CONTRIBUTING/
VALIDACAO_PUBLICA) — suíte completa reconfirmada mesmo assim, mesma
disciplina.

---

# Três pendências técnicas remanescentes (2026-08-27, mesma sessão)

## Passo 89 — Contrato de colunas de saída: FECHADO (implementação)

Dívida registrada em 2026-08-26 (Passo 77) e documentada em
`docs/COMPATIBILITY.md` desde então. Levantamento por comando direto
(`grep -rn "to_csv|to_excel|Workbook(" src/guaraci/`) de todos os pontos
de geração tabular: `avaliacao_modelos.py` (3), `guaraci.py` (2, incl.
o menu `[K]` do Passo 87), `pipeline.py` (3, inline dentro de
`executar()`), `resultados_io.py` (1), `selecao_variaveis.py` (4),
`plano_coleta.py` (Excel via openpyxl). `reports.py::generate_excel_report`
e `auditoria_delineamento.py`/`sentinela_deriva.py`/`linearity.py`/
`robustness.py` verificados como FORA de escopo (por leitura direta, não
suposição) — o primeiro copia colunas verbatim de CSVs já cobertos; os
demais não produzem saída tabular própria.

`tests/test_contrato_saida_tabular.py` (novo): snapshot golden
(`tests/golden/contrato_saida_tabular.json`) gerado por EXECUÇÃO REAL
contra dado sintético (nunca lista digitada à mão) — mesmo mecanismo de
`test_contrato_api_publica.py`. Cobre: `save_identifiers`,
`sanitizar_metadados`, `benchmark_classifiers`, `monte_carlo_cv`,
`benchmark_regression_by_species`, `etapa4_selecao_variaveis` (4 CSVs:
ipls/spa/ag/tabela-final), `plano_coleta.export_excel` (2 abas), o menu
`[K]` de seleção de amostras, `predict_samples`, e uma execução completa
de `executar()` (para `teste_martens.csv`/`comparacao_pipelines.csv`,
construídos INLINE no orquestrador, não atrás de função pública própria
— só rodar de verdade protege esses dois).

Contra-prova: monkeypatch de `save_identifiers` renomeando
`classe_predita` → `classe_pred` NUM CSV REAL (não só num dict de
teste) confirma que `_diferencas` (o mesmo detector do teste principal)
acusa a mudança.

`docs/COMPATIBILITY.md` atualizado: seção "Dívida conhecida" virou
"Dívida fechada (2026-08-27, Passo 89)".

## Passo 90 — Escopo do mypy: DECISÃO EXPLÍCITA = expandir (medido, implementado)

Medido por comando direto (`mypy` por arquivo, os 38 módulos de
`src/guaraci/`): **10 já no gate** (todos limpos), **17 fora do gate com
0 erros**, **11 fora do gate com erros** (1 a 13 cada).

Critério aplicado (o mesmo já documentado em `pyproject.toml`: sem
I/O pesado, sem UI, sem estado global) — **não** "0 erros = incluir
automaticamente": `figuras.py`, `app_logic.py`, `cli_assistente.py`,
`cli_logic.py`, `guaraci_theme.py`, `log.py`, `spectra_preview.py`
(importa `streamlit`, verificado por leitura) ficam FORA por serem
UI/renderização/orquestração — mesmo com 0 erros hoje, incluí-los
arriscaria ruído futuro conforme a integração de Streamlit/Rich
aprofunda, sem pegar bug de cálculo (mesma razão já documentada para
excluir `guaraci.py`/`pipeline.py`, que TÊM 11 e 10 erros
respectivamente e continuam explicitamente fora, decisão pré-existente
não reaberta aqui). `dados_imagem.py` fica fora por razão técnica real
(não "nunca foi feito"): importa `tifffile` via scikit-image, que usa
sintaxe Python 3.12 incompatível com `python_version=3.10` do mypy —
erro de SINTAXE de terceiro que interrompe a checagem inteira, não
corrigível no nosso código. `avaliacao_modelos.py`/`dados_io.py`/
`resultados_io.py` ficam fora: I/O real (CSV) é parte central do que
fazem, não incidental, e tinham 4-13 erros cada.

**Decisão: EXPANDIR.** Adicionados 14 módulos (10 já limpos +
4 corrigidos): `conformal.py`, `config.py`, `plano_amostral.py`,
`selecao_variaveis.py`, `sentinela_deriva.py`, `predicao.py`,
`io_registry.py`, `config_io.py`, `perfil_matriz.py`, `plano_coleta.py`,
`paleta_cores.py` (2 `# type: ignore` não utilizados removidos),
`auditoria_delineamento.py` (2 `int(object)` corrigidos com `cast`),
`identificacao.py` (1 tipo de chave de dict corrigido — `tuple(list)`
não prova comprimento 2 pro checador, trocado por desempacotamento
explícito), `model_registry.py` (1 comentário `# type: ignore`
malformado removido, era redundante com `ignore_missing_imports=true`
já setado globalmente). Gate: 10 → 24 módulos. `.github/workflows/test.yml`
atualizado com a lista completa.

## Passo 91 — Comparativo do README: RECONFIRMADO + 1 linha nova

Reconfirmado por nova busca (2026-08-27): Kennard-Stone **é** recurso
padrão do Unscrambler (fonte: busca anterior desta sessão, sem mudança).
Decisão de não reivindicar seleção de amostras/transferência de
calibração como diferencial permanece válida.

Avaliado o CONJUNTO completo (não só seleção de amostras isolada):
planejamento experimental (`plano_amostral.py`+`plano_coleta.py`),
auditoria de delineamento automática (`auditoria_delineamento.py`),
modo cego com conjunto aberto calibrado por predição conforme
(`identificacao.py`), sentinela de deriva (`sentinela_deriva.py`),
dossiê de linearidade/robustez opcional. Busca dedicada (2026-08-27)
por "sample size guidance + confounding audit + conformal open-set
identification" em ferramentas comerciais: nenhuma evidência de suite
comercial bundlando essa combinação — os resultados encontrados são
literatura acadêmica/de fronteira (predição conforme auditada 2026,
D-optimal design), não recurso de produto integrado.

**Decisão: adicionar 1 linha ao comparativo** (README.md e
README.pt-br.md), fraseada com o mesmo cuidado epistêmico de antes
("não encontrado em documentação pública até 2026-08", não "nenhum
concorrente tem") — mais uma nota explícita ao lado da tabela dizendo
que Kennard-Stone/transferência de calibração NÃO são reivindicados
como diferenciais, para que a mudança não pareça contradizer a decisão
anterior sobre esses dois itens especificamente.
