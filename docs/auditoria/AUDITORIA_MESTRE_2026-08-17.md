# Auditoria mestre — GUARACI (2026-08-17)

Execução do `PROMPT_AUDITORIA_MESTRE_GUARACI` (passes 0 a 6) em rodada
única. Toda métrica desta página foi produzida por comando executado nesta
sessão; o que não foi medido está marcado `NÃO VERIFICADO`.

Scripts novos desta rodada, todos reexecutáveis:
[`medir_ordem_leitura.py`](medir_ordem_leitura.py),
[`medir_deriva_vs_quimica.py`](medir_deriva_vs_quimica.py),
[`medir_ad_vies_insample.py`](medir_ad_vies_insample.py),
[`medir_truncagem_nh_nq.py`](medir_truncagem_nh_nq.py),
[`datasets_publicos.py`](datasets_publicos.py).

---

## 0. Inventário

| Item | Medido | Comando |
|---|---|---|
| Módulos / linhas em `src/guaraci` | 31 arquivos, 20.333 linhas | `wc -l src/guaraci/*.py` |
| Arquivos rastreados no Git | 141 | `git ls-files \| wc -l` |
| Espectros / `.dx` no Git | **0** | `git ls-files \| grep -iE '\.(dx\|jdx\|spc\|npy)$'` |
| Testes | **732 passam, 2 skip** (no início); **737 passam, 2 skip** ao fim, com os 5 testes novos desta rodada | `pytest -q` |
| Cobertura `src/guaraci` | **70 %** | `pytest --cov=src/guaraci` |
| `ruff` / `mypy` (7 módulos do gate) | limpos | `ruff check .`, `mypy …` |
| CVEs nas dependências | **nenhuma** | `pip-audit --vulnerability-service osv` |
| `except` amplos | 53, 100 % com `noqa` justificado | `grep -rn "except Exception"` |
| TODO/FIXME reais | 6, todos em templates LaTeX de `reports.py` | `grep -rn TODO src/` |
| Commits | 96 (91 do autor, 5 do Dependabot) | `git log --all --format='%an'` |
| Commits locais não enviados | 11 | `git status -sb` |
| Dados reais | 3.882 `.dx`, **fora do repositório** | `find "…/dados oleos" -name '*.dx'` |

**Divergência com o contexto herdado.** O prompt afirma que a faixa
`[4000, 10000] cm⁻¹` está "provada no bundle" como inalterada em **185
commits**. O histórico atual tem **96 commits no total** (2026-07-04 a
2026-08-17) — ele foi reescrito em 2026-08-16 na purga dos espectros
(S4). A alegação pode ser verdadeira sobre o histórico antigo, mas **não é
verificável no repositório de hoje**. O mesmo vale como limite do Passe 4:
o histórico atual não serve como prova temporal de autoria.

---

## 1. Metodologia

### 1.1 Separação identificação × quantificação — REQUISITO NÃO ATENDIDO

| Verificação | Resultado | Evidência |
|---|---|---|
| O rótulo entra no caminho de quantificação? | **Sim, por construção** | `pipeline.py:785` `r2cv_especie_adulterante(…, rotulos, mae_id, …)`; `pipeline.py:868` `pls_regressao_por_especie(…, rotulos, …)` — a calibração é feita **dentro de cada espécie×adulterante conhecidos** |
| Vazamento indireto (pré-proc ajustado no conjunto todo) | **Não** — MSC/SG/centragem vivem dentro do `Pipeline` do sklearn | `preprocessamento.py:116-141`, comentário em `:128` |
| Modo cego é o padrão? | **Não existe modo cego** | `grep -rni "cego\|modo_controle\|--modo" src/` → 0 ocorrências |
| Teste que falha se o rótulo vazar | **Não existe** | — |
| O modelo de teor é exportado para uso em amostra nova? | **Não** | `pipeline.py:2189-2201`: o `.joblib` leva `pls_final` (PLS-DA), `label_binarizer`, `wavenumbers`, limites — **nenhum modelo de regressão** |
| Resposta a amostra não adulterada / adulterante fora do treino | força uma classe | `predicao.py:245` `idx_pred = Y_norm.argmax(axis=1)` — sempre devolve a classe de maior score |
| Detecção de fora-de-domínio no caminho padrão | **Sim** (colunas `AD_*`) | `predicao.py:279-289` |

**Consequência.** A quantificação do GUARACI hoje é uma **rotina de
validação interna**, não um recurso entregável: ela exige saber a espécie e
o adulterante, e o modelo resultante não é salvo. A arquitetura alvo
(`detectar → identificar → quantificar`, cada etapa cega) não existe.
Isso não é vazamento acidental — é o desenho atual. Mas contraria o
requisito de projeto e precisa ser dito explicitamente em qualquer texto
que apresente a quantificação como capacidade do software.

### 1.2 Ordem de leitura — **P0, medido**

`medir_ordem_leitura.py`, sobre os 3.785 `.dx` cujo `##TITLE` é
parseável. Auditorias anteriores usaram a data do `##TITLE` (data da
**amostra**); esta usa o `##AUDIT TRAIL` (data/hora da **leitura**), que é
outro campo e nunca tinha sido lido.

**O fato declarado pelo autor está CONFIRMADO:** puro e adulterado de cada
espécie compartilham a sessão de leitura em **13 de 14** espécies
catalogadas (a exceção é um código não catalogado, sem amostra pura).

**E a exposição que isso cria também está confirmada, e é grave:**

| | |
|---|---|
| Blocos espécie×adulterante×sessão avaliados | 47 |
| Blocos com correlação significativa entre ordem de leitura e teor | **47 (100 %)** — esperado sob H₀: 5 % |
| ρ de Spearman médio / mediano | **+0,997 / +1,000** |
| ρ > 0 | 47/47 (binomial p = 1,4 × 10⁻¹⁴) |
| Sessões de leitura | 27 dias, 2020-11-26 a 2021-02-02 |

Dentro de cada sessão as amostras foram lidas **da menor para a maior
concentração**, sem exceção. Portanto "tempo decorrido na sessão" e "teor"
são a mesma variável, e qualquer deriva temporal (aquecimento da fonte,
purga de CO₂/H₂O, degradação da janela, evaporação/oxidação da alíquota)
entra no modelo como se fosse sinal de adulteração. **Nenhuma alteração de
software separa os dois** — a separação teria que vir do delineamento
(ordem aleatorizada, brancos intercalados, releituras).

Não há releituras que permitam estimar a deriva: dos 652 `mae_id`, apenas
4 foram lidos em mais de um dia, e o intervalo dentro de um grupo de
réplicas tem mediana de 219 s (as três réplicas são consecutivas).

### 1.3 O modelo de teor não sobrevive à troca de sessão

`medir_deriva_vs_quimica.py`, com a **mesma receita do pipeline em
produção** (`msc_sg_mc`, SG 25/2/1, regra de LV de `pipeline.py:836`),
mudando apenas a partição. Seis espécies têm soja lida em duas sessões.

| Partição | R² mediana | RMSE mediano |
|---|---|---|
| Dentro da sessão, CV group-aware por `mae_id` (o que o pipeline reporta) | +0,023 (nas 7 sessões grandes: **0,66 a 0,95**) | 1,93 |
| Treina numa sessão, prediz a outra | **−1,388** (12 de 14 abaixo de zero) | 6,47 (**3,3×**) |

**Confundidores declarados** — este teste *não* isola deriva: as duas
sessões diferem também em **lote** (Lote 1 × LOTE 02) e em **faixa de
teor** (0–5 % × 0–15 %).

Restringindo à faixa comum 0–5 %, o resultado muda de figura e fica pior
para a alegação de quantificação: **o R² intra-sessão também colapsa**
(mediana −1,04; negativo em 14/14, inclusive nos blocos com n = 45–54). Ou
seja, o desempenho alto só aparece quando a faixa vai até ~15 %, numa
sessão única em que a ordem é monotônica.

**Leitura honesta:** o desempenho de quantificação hoje reportado é uma
propriedade do delineamento, não uma capacidade demonstrada de quantificar
adulteração. Isso vale para os números do TCC e para qualquer texto que os
cite.

### 1.4 Validação, métricas e limites

| Item | Estado | Evidência |
|---|---|---|
| CV aninhada nos métodos comparados | sim | `selecao_variaveis.py`, `_avaliar_busca_nested_cv` |
| RMSEC / RMSECV / RMSEP / R²pred / bias | presentes | `pipeline.py:1036`, `:2473` |
| **RPD e RER** | **ausentes** | `grep -rn "\brpd\b\|\brer\b" src/` → 0 |
| LOD / LOQ | implementados (Valderrama, Braga & Poppi 2009, via NAS/SEN) | `chemometric_stats.py:415-486` |
| LOD/LOQ na prática | **saem `N/A` sem réplicas físicas suficientes** — na execução do dataset público, `N/A` | saída do pipeline |
| Incerteza como intervalo | sim (BCa) | `validacao_estatistica.py:204` |
| Aderência a AOCS/AOAC/ASTM/ISO | não declarada | — |

**Truncagem de `Nh`/`Nq` — RETRATAÇÃO.** O contexto herdado lista como
pendência o fato de `media_e_dof_momentos` devolver graus de liberdade
contínuos em vez de inteiros (Kucheryavskiy et al. 2024, p. 4). Medido em
6 cenários × 200 repetições (`medir_truncagem_nh_nq.py`): **a diferença de
cobertura é ≤ 0,0033 em todos**. É cosmético, não numérico. Não vale
tratar como achado.

**Mas a medição expôs um achado maior — corrigido nesta sessão.**

### 1.5 Domínio de aplicabilidade rejeitava a própria calibração — **corrigido**

`dominio_aplicabilidade_treino` calculava o Q de treino **in-sample**
(`chemometric_stats.py:587`, antes da correção). Com n < p — o regime deste
projeto — a PCA reconstrói cada amostra de treino quase exatamente, `q0` e
`Nq` saem otimistas e o limite passa a rejeitar amostras legítimas.

É a **mesma classe de defeito já corrigida no DD-SIMCA em 2026-07-19**
(`DDSimca._q_residuals_loo`, CLAUDE.md P1) e que **não tinha sido
propagada** para o AD — que é justamente o caminho que roda em produção em
`predicao.py` (colunas `AD_*`).

Medido com as funções de produção (`medir_ad_vies_insample.py`), fração de
amostras da própria distribuição de treino aceitas, α = 0,05 (esperado
0,95):

| n | p | k | antes | depois (Q por LOO) |
|---:|---:|---:|---:|---:|---:|
| 20 | 1200 | 3 | **0,144** | 0,941 |
| 30 | 1200 | 3 | **0,205** | 0,949 |
| 50 | 1200 | 5 | **0,187** | 0,962 |
| 100 | 1200 | 5 | **0,425** | 0,960 |
| 300 | 1200 | 10 | **0,574** | 0,962 |
| 50 | 100 | 5 | **0,562** | 0,976 |

**Por que passou despercebido:** os testes de AD existentes usam n = 80–200
com p = 15–30 (n ≫ p), o único regime em que o viés é pequeno.

Correção aplicada:
- `q_residuos_loo` promovida a função pura em `chemometric_stats.py`;
- `dominio_aplicabilidade_treino` passa a usá-la;
- `DDSimca._q_residuals_loo` delega para ela (a duplicação foi o que
  permitiu que uma das cópias ficasse para trás — mesmo padrão do achado A3);
- teste novo no regime real (`test_dominio_aplicabilidade_nao_rejeita_treino_no_regime_n_menor_que_p`),
  que **falha** com o código anterior (media 0,390 contra piso 0,85).

### 1.6 Generalização entre matrizes

**O teste foi executado, e o resultado é majoritariamente positivo.**
`datasets_publicos.py` roda o pipeline no Eigenvector *Corn* (80 amostras,
700 canais, 1100–2498 nm — matriz não-óleo) **sem alterar uma linha de
código-fonte**, só por configuração:

| Métrica (proteína, espectrômetro m5) | GUARACI | Literatura |
|---|---|---|
| RMSEP | **0,171 %m/m** | faixa típica de PLS: 0,1–0,2 |
| R²val / LVs | 0,865 / 8 | — |

O motor de quantificação está correto. Isso **separa bug de limitação**: o
que falha no dataset próprio é o delineamento (§1.2/§1.3), não o algoritmo.

Mas a mesma execução expôs onde a plataforma ainda é "software de óleo":

| Hardcode | Local | Efeito na matriz nova |
|---|---|---|
| `CODIGO_ESPECIE` (13 óleos amazônicos) | `dados_io.py:35` | matriz nova não tem código; cai no fallback |
| `ADULTERANTE_NOME` (A/M/S) | `dados_io.py:43` | conceito de adulterante fixo em 3 |
| `BANDAS_NIR` (atribuições de óleo vegetal) | `figuras.py:681-690` | anotações químicas de óleo desenhadas sobre espectro de milho |
| Texto do model card | `reports.py` | model card do milho diz *"Quantificacao do teor (%) de adulterante em **oleo vegetal amazonico**"* |
| Vocabulário da saída | `pipeline.py:2389` | milho é reportado como *"60 adulterados + 0 puros"* |
| Sem gate de nº de classes | `pipeline.py` | com 1 classe, reporta **`Accuracy (CV) = 1.0000`** e `R2X=R2Y=Q2=0,0000` sem aviso |

Não há conceito de "matriz" na configuração (`grep -ni "matriz" config.py`
→ 0). Adicionar matriz nova não exige tocar no código *para rodar*, mas
exige para que a saída não afirme química errada.

---

## 2. Código e dados

| Item | Estado | Evidência |
|---|---|---|
| Roda ponta a ponta em ambiente limpo | sim | execução do corn, acima |
| Seeds fixadas | majoritariamente — 2 pontos usam `default_rng(42)` fixo em vez de `cfg.seed` | `avaliacao_modelos.py:107`, `:785` |
| Lockfile / versão de Python | `requirements-lock.txt`, `requires-python = ">=3.10"` | `pyproject.toml:12` |
| Dados brutos separados dos processados | sim — nada versionado | `.gitignore` cobre `dados/`, `*.dx`, `resultados*/` |
| Espectro em HEAD ou histórico | **nenhum** | `git log --all --diff-filter=A --name-only` |
| Identificador de amostra em arquivo versionado | **3 reais** — ver retratação abaixo | medição contra os 652 `mae_id` do dataset |
| Duplicação consolidada nesta rodada | `q_residuos_loo` (2 cópias → 1) | `chemometric_stats.py`, `classificadores.py` |

**RETRATAÇÃO — identificadores de amostra versionados.** Eu havia marcado
este item como limpo com base numa varredura por *espectros* e nomes
institucionais. `docs/auditoria/exposicao_dado_publico.md` (escrito antes
desta rodada) apontava o contrário, e ao reverificar ele está certo: há
**26 ocorrências** de identificador no padrão `COD-DD-MM-AAAA[-Xteor]` em
arquivos versionados. Medindo contra os 652 `mae_id` reais do dataset,
**3 correspondem a amostras físicas reais**:

| Identificador | Onde | Contém |
|---|---|---|
| *(código de espécie + data)* | `dados_io.py:52,202`, `tests/…` | espécie + data de coleta |
| *(o mesmo, com sufixo de adulteração)* | `dados_io.py:50,203`, `tests/test_heatmap…:17` | + adulterante e teor |
| *(outra espécie + data)* | `cli_assistente.py:926,934` | espécie + data de coleta |

Os outros 5 **não existem** no dataset — são exemplos inventados, e não
expõem nada. (Os identificadores reais não são reproduzidos aqui: o
registro do achado precisa do fato e do local, não do identificador.)

São metadados, não espectros, e já estão no remoto público.

**Resolvido em 2026-08-18** (rodada de desacoplamento): os 3 foram
substituídos por identificadores sintéticos de ano 2099 — mesma estrutura,
mesma lógica de parsing exercitada, nenhuma colisão possível com o acervo
real. 13 ocorrências em 6 arquivos. O **histórico publicado continua com
os originais**; ver a proposta de estratégia no relatório de
desacoplamento.

**Dois defeitos abertos em `predicao.py` — o caminho de produção:**

1. **Regra retangular residual** (`predicao.py:255`): a coluna `aceito` é
   `(T2 ≤ t2_ucl) & (Q ≤ q_ucl)` — exatamente a regra corrigida no DD-SIMCA
   (2026-08-08) e no AD (achado A3), com α conjunto efetivo ≈ 0,0975 em vez
   de 0,05. As colunas `AD_*` usam a distância combinada correta, mas a
   coluna com o nome mais autoritativo (`aceito`) não.
2. **Fallback de `q_ucl` circular** (`predicao.py:236`): se o pacote não
   traz `q_ucl`, o limite é calculado a partir das **próprias amostras
   novas** (`np.percentile(Q_new, 99) * 1.5`). Um lote inteiro fora do
   domínio elevaria o limite e seria aceito. Pacotes gerados pelo pipeline
   atual sempre trazem `q_ucl` (`pipeline.py:2201`), então hoje é
   **latente** — mas dispara com qualquer `.joblib` anterior à v25.

---

## 3. Segurança

| Item | Resultado |
|---|---|
| Segredos no HEAD e em todo o histórico | **nenhum** (varredura por padrões de token/chave em 96 commits) |
| `pickle`/`joblib` | protegido: `confiar=True` obrigatório + verificação de SHA-256 do manifesto **antes** do `joblib.load` (`predicao.py:74-120`) |
| `eval` / `exec` | nenhum |
| `subprocess` / `os.system` | 3 usos, todos com argumentos constantes; `subprocess.run(["chcp","65001"], shell=True)` em `guaraci.py:39` é o único com `shell=True` — não explorável (sem entrada do usuário), mas o `shell=True` é dispensável |
| YAML | `yaml.safe_load` (`config_io.py:320`) |
| Dependências com CVE | **nenhuma** (`pip-audit`, serviço OSV) |
| Permissões de CI | `docs.yml` declara escopo mínimo; **`test.yml` e `draft-pdf.yml` não declaram `permissions:`** e herdam o padrão do repositório |

---

## 4. Titularidade e licenciamento

> **Não é aconselhamento jurídico.** É um mapa de fatos. A conclusão sobre
> titularidade é do NIT/Agência de Inovação e de um advogado, não desta
> auditoria.

### 4.1 Código — a declaração de autoria única é coerente com o repositório

| Verificação | Resultado |
|---|---|
| Autores nos commits | `Erley <erleysdacosta@gmail.com>` em 91/96; Dependabot em 5; **nenhum coautor humano** |
| E-mail institucional em autor ou committer | **nenhum** |
| Código de terceiro/orientador incorporado | nenhum indício no histórico |
| Limite desta verificação | o histórico foi reescrito em 2026-08-16; a janela 2026-07-04→2026-08-17 **não** representa o desenvolvimento real e não serve como prova temporal |

### 4.2 Dados — aqui está o risco real, e ele é medido

Os metadados internos dos `.dx` (`##AUDIT TRAIL`) registram, em **100 % dos
3.785 arquivos**:

- **local o nome de uma instituição** — o instrumento é institucional;
- **operador** — 3 rótulos distintos, correspondentes a pelo menos 2
  pessoas, **nenhuma delas o autor** (contagens 1537 / 1517 / 828).

Isso responde por medição, não por declaração, à pergunta "houve técnico
operando?": **sim**. Consequências, aplicando a regra de decisão do prompt:

- o **código continua sendo do autor** — nada no histórico contradiz isso;
- os **espectros não devem ser publicados** no repositório, e o uso no TCC
  pede agradecimento/autorização;
- a proveniência é **documentada, nunca apagada** — remover os arquivos é
  correto; apagar o registro de que vieram de um instrumento institucional
  operado por terceiros seria misatribuição;
- quem operou o instrumento entra em `ACKNOWLEDGMENTS`, **não** em
  `AUTHORS` nem como coautor do software.

**Risco adicional, específico e acionável:** um dos rótulos de operador
gravados nos arquivos é um **apelido informal e depreciativo** aplicado a
uma pessoa nomeada, presente em **828 arquivos**. Se esses espectros forem
publicados (Zenodo, material suplementar, dataset anexo ao artigo) sem
sanitizar o `AUDIT TRAIL`, isso vai junto. É um dano reputacional a um
terceiro identificável. **Sanitizar o `AUDIT TRAIL` é pré-requisito de
qualquer publicação dos dados brutos.**

### 4.3 Mapeamento de ocorrências (3 baldes)

| Balde | Ocorrências | Ação |
|---|---|---|
| **DADO** | nenhum em arquivo versionado; o nome de uma instituição + nome de operador nos `.dx` **fora** do repo | manter fora; sanitizar antes de qualquer publicação |
| **AFILIAÇÃO DO AUTOR** | link Lattes (`lattes.cnpq.br/5755582193284309`) em `CITATION.cff:18,82`, `README.md:167`, `README.pt-br.md:236`, `app_tabs/sobre.py:15`, `guaraci.py:2768` | **decisão do usuário** — não removido |
| **CRÉDITO A TERCEIRO** | nenhum registrado hoje; deveria haver `ACKNOWLEDGMENTS` para quem operou o instrumento | criar, com autorização das pessoas |

Nenhuma menção a instituição, laboratório, grupo de pesquisa, agência de
fomento ou edital em arquivo versionado — a decisão de 2026-08-07 foi
cumprida.

### 4.4 Licenciamento

| Item | Estado |
|---|---|
| `LICENSE` | **GPL-3.0-or-later**, texto padrão íntegro (674 linhas) |
| Cláusula "somente acadêmico/não comercial" | **ausente** — o item marcado como P0 no prompt **não se aplica** |
| Duplo licenciamento | já documentado (`docs/COMMERCIAL.md`) e coerente com `LICENSE`, `CITATION.cff`, `pyproject.toml` |
| Licenças das dependências | todas permissivas (BSD/MIT/Apache/PSF) — nenhuma GPL/AGPL/não comercial que contamine |
| DOI do Zenodo | ~~`10.5281/zenodo.21311867` **resolve**~~ — **RETRATADO em 2026-08-19**: devolve **HTTP 410**. Os dois depósitos (v31.1.0 e v31.1.1) foram retirados pelo autor em 2026-08-04; o concept não tem versão viva (`/versions` → `total: 0`). A auditoria de 2026-08-17 marcou "resolve" sem medir o código de status. |

**Ressalva a levar ao NIT:** `COMMERCIAL.md` afirma que *"o autor detém o
copyright integral e o direito de licenciar sob outros termos"*. Essa
afirmação só se sustenta se a Lei 9.609/98, art. 4º §3º não atingir o caso
— o que depende do mapa de fatos de 4.1/4.2, não do código. A oferta de
licença comercial é a parte mais exposta se a titularidade for questionada.

**Ponto técnico de licença:** GPL-3.0 não cobre uso em rede. Um terceiro
pode rodar um fork modificado do app Streamlit como serviço fechado sem
publicar as modificações. Se a intenção é fechar essa porta, a licença
teria que ser **AGPL-3.0**. É uma decisão, não um defeito.

---

## 5. CLI, documentação e referências

| Item | Estado |
|---|---|
| `--help` | existe, mas mínimo: `demo\|doctor\|--version` — sem `--modo`, sem flags de configuração |
| Modo cego como padrão na CLI | **não existe** (ver §1.1) |
| Saída declara valor + incerteza + dentro/fora do domínio + versão | parcialmente: versão e domínio sim; matriz usada, não |
| Model card | **existe e é gerado automaticamente**, mas com texto de matriz hardcoded (ver §1.6) |
| Referências bibliográficas | 16 DOIs citados; **14 resolviam, 2 estavam errados** |

**Dois DOIs incorretos, corrigidos nesta rodada** (as referências são
reais; os identificadores é que estavam errados):

| Referência | DOI antes | DOI correto (verificado no Crossref) |
|---|---|---|
| Geladi & Kowalski (1986), *Anal. Chim. Acta* 185:1-17 | `10.1016/0003-2670(85)85121-2` | `10.1016/0003-2670(86)80028-9` |
| Geladi, MacDougall & Martens (1985), *Appl. Spectrosc.* 39(3):491-500 | `10.1366/0003702854248684` | `10.1366/0003702854248656` |

Nenhuma referência inventada foi encontrada.

---

## 6. Repositório e CI

| Item | Estado |
|---|---|
| Estrutura, `.gitignore`, CHANGELOG, SemVer | coerentes |
| CI | lint (`ruff`) + typecheck (`mypy`, 7 módulos) + testes em Ubuntu/Windows/macOS, Python 3.10–3.13 |
| Pendência | `permissions:` ausente em `test.yml` e `draft-pdf.yml`; 5 PRs do Dependabot sem revisão; 11 commits locais não enviados |

---

## 7. Correções priorizadas

| # | Prio | Achado | Impacto se não corrigir | Local | Esforço | Bloqueia entrega? |
|---|---|---|---|---|---|---|
| 1 | **P0** | Ordem de leitura monotônica com o teor (ρ=+0,997, 47/47 blocos) | Todo número de quantificação é inseparável de deriva de sessão | delineamento (não é software) | re-medição | **Sim** |
| 2 | **P0** | Quantificação não transfere entre sessões; na faixa comum 0–5 % não funciona nem intra-sessão | Alegar "quantifica teor" não se sustenta | — | declarar | **Sim** |
| 3 | **P0** | `AUDIT TRAIL` dos `.dx` carrega o nome de uma instituição + nome de terceiro (um deles com apelido depreciativo, 828 arquivos) | Exposição de terceiro identificável se os dados forem publicados | dados brutos | sanitizar | **Sim, para publicar dados** |
| 4 | ~~P0~~ | ~~AD rejeitava 43–86 % da própria calibração~~ | — | `chemometric_stats.py` | — | **✅ corrigido nesta rodada** |
| 5 | **P1** | Sem modo cego / sem teste de vazamento de rótulo na quantificação | Requisito de projeto não atendido | `pipeline.py`, CLI | médio | Sim, para "produto" |
| 6 | **P1** | Regra retangular residual em `predicao.py:255` (`aceito`) | α conjunto ≈ 0,0975 em vez de 0,05 no caminho de produção | `predicao.py` | pequeno | Não |
| 7 | **P1** | Fallback de `q_ucl` derivado das próprias amostras novas | Limite circular com pacotes antigos | `predicao.py:236` | pequeno | Não |
| 8 | **P1** | Model card e saída afirmam "óleo vegetal amazônico"/"adulterados" em qualquer matriz | Afirmação química falsa fora de óleos | `reports.py`, `pipeline.py:2389` | pequeno | Não |
| 9 | **P1** | Sem gate de nº de classes: reporta `Accuracy = 1.0000` com 1 classe | Métrica sem sentido apresentada como resultado | `pipeline.py` | pequeno | Não |
| 10 | ~~P1~~ | ~~2 DOIs incorretos~~ | — | `cli_assistente.py` | — | **✅ corrigido nesta rodada** |
| 11 | **P2** | RPD e RER ausentes | Figuras de mérito incompletas para quantificação | `chemometric_stats.py` | pequeno | Não |
| 12 | **P2** | `permissions:` ausente em 2 workflows; `shell=True` dispensável | Superfície de ataque desnecessária | `.github/workflows/`, `guaraci.py:39` | trivial | Não |
| 13 | **P2** | 2 seeds fixas em `default_rng(42)` em vez de `cfg.seed` | Reprodutibilidade parcialmente fora do controle do usuário | `avaliacao_modelos.py:107,785` | trivial | Não |

---

## 8. Veredito

> **Fechável nesta semana:** os itens P1/P2 de código (5 a 13) — são
> pequenos, localizados e cobertos por testes. Os dois P0 de software já
> foram fechados nesta rodada.
>
> **Não fechável nesta semana:** os P0 1 e 2. Eles não são defeitos de
> código: são consequências do delineamento experimental de 2020-2021, e a
> única correção real é remedir com ordem de leitura aleatorizada, brancos
> intercalados e pelo menos duas sessões independentes por combinação.
>
> **O que impede:** o software está mais correto do que os dados que ele
> analisa. A validação externa prova isso — no *Corn* público, o mesmo
> motor entrega RMSEP dentro da faixa da literatura. O caminho mais curto
> para um resultado defensável **não** passa por mexer no código: passa por
> reposicionar a alegação (autenticação/exploração, com a quantificação
> declarada como exploratória e limitada à sessão de calibração) ou por
> uma nova campanha de medidas.

