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
