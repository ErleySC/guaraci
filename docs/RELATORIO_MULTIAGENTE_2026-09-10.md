# Relatório multiagente — estado, segurança, técnicas e mercado (2026-09-10)

Rodada de quatro frentes: estado/segurança (Agente 1, correções aplicadas),
técnicas novas (Agente 2), problemas documentados da área (Agente 3) e
concorrência (Agente 4). Os Agentes 2–4 **só propõem**: nada deles foi
implementado. Commit de referência: `8b784de`, com as correções `02d65fe` e
`676e9c4` por cima. O vault foi regenerado antes da pesquisa, e
`consultar_vault.py --cobertura` deu COMPLETA (68/68 módulos, 13/13 técnicas,
110/110 Passos, 14/14 datasets).

---

## 1. Resumo executivo

- O CI estava vermelho **em todo push desde 2026-09-05**, por dois defeitos do próprio workflow: um `\n` literal que abortava o mypy e um job de validação sem o extra que instala o `tensorly`. Os dois foram corrigidos. Três jobs de validação pública ainda podem falhar de forma intermitente por indisponibilidade do Zenodo.
- Segurança: nenhuma vulnerabilidade de código ou de dependência encontrada. O achado real foi de **privacidade**: a guarda de caminho absoluto só varria o vault, e por isso um caminho de máquina com o nome do usuário foi publicado no `PROGRESSO.md`. A guarda agora varre o repositório e o texto foi corrigido. O caminho **continua no histórico git**.
- Achado mais crítico da rodada (Agente 3, conferido no código): a validação cruzada e o holdout agrupam por `mae_id`, mas o próprio código documenta que, em amostra adulterada, o `mae_id` é **um por nível de teor** (15 diluições de uma sessão). A alegação central de "validação anti-vazamento" é, portanto, mais fraca do que o README sugere para dados adulterados.
- Concorrência (Agente 4): o comparativo público do README faz afirmações que a documentação dos concorrentes contradiz ou não sustenta. PLS_Toolbox/Solo e Unscrambler **têm** CV por grupo; o "❌ Reproducible" não tem evidência. Os concorrentes estão à frente em intervalo de predição por amostra, fusão multibloco, leitores de formato, exportação portátil do modelo e monitoramento em linha.
- **Decisões urgentes do autor:** (1) como tratar esse agrupamento, porque mexe em lógica científica já validada e em números já citados; (2) corrigir o comparativo do README, uma alegação pública sem sustentação num repositório público.

---

## 2. Agente 1 — estado, bugs e segurança

### 2.1 Estado medido (não citado de memória)

| Item | Resultado |
|---|---|
| Suíte completa (fast + slow), linha de base `8b784de` | **1460 passed, 42 skipped, 0 failed** (615 s) |
| Cobertura total (`--cov=.`, mesmo comando do CI) | 85,2% (piso do CI: 60%) |
| Núcleo científico | chemometric_stats 97% · classificadores 97% · preprocessamento 97% · validacao_estatistica **95%** (no limite do gate de 95%) · agregado 96,5% |
| `ruff check .` | limpo |
| `mypy` (gate do CI) | limpo, 50 arquivos (49 anteriores + `eem_multiway.py`) |
| `pip-audit` (serviço OSV; o serviço PyPI caiu com *connection reset*) | **0 vulnerabilidades conhecidas** em 139 pacotes do ambiente, incluindo tensorly 0.9.0, brukeropus 1.4.3, fpdf2 2.8.7, hypothesis 6.165.10, prcv 1.2.1 e streamlit 1.61.0. `gc-ims-tools` não foi adotado. |
| Suíte completa após as correções | ver §2.5 |

### 2.2 Bugs corrigidos

| # | Sintoma | Causa raiz | Correção | Contra-prova |
|---|---|---|---|---|
| B1 | Job `typecheck` falhando em todo push desde `8b784de` | `\n` **literal** na lista do mypy em `test.yml`. O bash reduz `\n` sem aspas a `n`, e o mypy aborta com `Cannot read file 'n'` antes de checar qualquer módulo. | `02d65fe` | Bloco `run:` simulado: o HEAD passava 1 argumento fora de `src/guaraci/*.py` (`n`); a versão corrigida passa 0. |
| B2 | Job `validacao-publica-eem-zenodo` falhando em todo push desde `7396666` (2026-09-05) | Instalava `pip install -e .` sem `[multiway]`, e os 2 testes de PARAFAC davam `ModuleNotFoundError: tensorly`. O job `test` não pegava porque instala `requirements.txt`, que lista tensorly. | `02d65fe` | No `pyproject.toml`, tensorly só entra pelo extra `multiway` (checado com `tomllib`). A confirmação final depende do CI após o push. |
| B3 | `eem_multiway.py` (rodada multiway) fora do gate de tipos | Módulo puro criado sem entrar na lista | `02d65fe` | mypy limpo no módulo e no gate completo |
| B4 | Caminho absoluto de máquina (usuário + pasta do acervo) publicado em `docs/PROGRESSO.md` (Passo 131, `22b5511`, repositório público desde 2026-08-16) | `PADRAO_CAMINHO_ABSOLUTO` só era aplicado pelo gerador do vault; a varredura de arquivos versionados só checava identificador de amostra | `676e9c4`: varredura de caminho absoluto sobre todo arquivo versionado, com usuário sentinela `alguem` como única exceção (mesma lógica do ano 2099) | O teste novo falhou antes da correção listando 7 ocorrências (1 real + 6 placeholders); depois, 22 passed nos módulos de privacidade |

**Severidade de B4:** baixa. Expõe nome de usuário e estrutura de pasta, nenhum dado espectral ou identificador de amostra. O caminho **continua no histórico git** (a decisão registrada é manter o histórico); reescrever ou não é decisão do autor.

### 2.3 Reportados, não corrigidos

| # | Achado | Por que não foi corrigido |
|---|---|---|
| R1 | Jobs `hplc-azeite`, `gcims-urina` e `eem-zenodo` falharam por **HTTP 504 / read timeout do Zenodo** (2026-09-08 00:10 e 18:35 UTC). Em HEAD agora respondem 200. Os scripts `baixar_zenodo_*` usam `urllib` sem retry; o job do Corn usa `curl --retry 3`. | Retry com backoff é robustez, mas altera a semântica declarada do gate ("fonte fora do ar = falha"). Decisão do autor. |
| R2 | Módulos fora do gate mypy com erros: `hsi_pipeline.py` (31), `resultados_io.py` (13), `avaliacao_modelos.py` (6). O erro de `dados_imagem.py` vem do `tifffile` (sintaxe 3.12 com `python_version = 3.10`), não do projeto. Passam sem erro e estão fora do gate: `cli_logic`, `app_logic`, `spectra_preview`, `hsi_figures`. | São orquestração ou figura, fora da política "só módulos puros". É dívida, não bug. |
| R3 | `consultar_vault.py` casa substring: `"EPO"` devolve notas com "t**empo**" como nota principal. | Pode dar falsa impressão de registro prévio, justamente o que a consulta existe para evitar. Correção simples (fronteira de palavra para siglas), mas é ferramenta, não código científico; fica para priorização. |
| R4 | 5 PRs do Dependabot abertas (#8–#11 de 2026-08-08, actions do CI; #20 de 2026-09-04, 22 atualizações Python). | Revisão de dependência não entra nesta rodada. |
| R5 | Inconsistências de documentação achadas pelos Agentes 2 e 3 (conferidas): (a) a docstring de `hsi_uncertainty.py` l.17-19 diz que "Quantificar" tem "intervalo de predição" com alpha próprio, e `predicao.py` l.684-685 e o código dizem que não há; (b) `cli_assistente.py` l.1669/1677 recomenda PQN para RMN, e PQN não existe no pacote; (c) `auditoria_delineamento.check_external_validation` devolve mensagem fixa dizendo que classificação não tem benchmark externo, desatualizada frente a `VALIDACAO_PUBLICA.md`. | Correção de texto trivial, mas (a) e (b) dependem da decisão sobre as propostas T1/T6 (§3): implementar ou remover a menção. |

### 2.4 Varredura de segurança

| Vetor | Resultado |
|---|---|
| `eval`/`exec`/`pickle.loads`/`marshal`/`shelve`/`yaml.load`/`allow_pickle=True` em `src/`, `scripts/` e `app_quimiometria.py` | **nenhuma ocorrência** |
| `subprocess` | só listas de argumentos com valores internos (`git rev-parse`, abrir pasta de resultado). O único `shell=True` é `chcp 65001` constante (builtin do cmd.exe). `os.system` só com `cls`/`clear` constantes. |
| `joblib.load` (pickle) | mitigação P5 intacta: SHA-256 do manifesto conferido **antes** do load, aviso explícito na CLI e na web |
| `zipfile.extractall` (`baixar_zenodo_hplc_azeite.py`, `baixar_zenodo_eem_azeite.py`) | só depois de SHA-256 e tamanho pinados; além disso, `zipfile` remove componentes `..` e absolutos. Sem vetor explorável. |
| Leitura de zip por HTTP Range (`baixar_zenodo_gcims_urina.py`, `baixar_deephs_*`) | o nome gravado vem de lista **pinada no script ou sidecar**, nunca do zip remoto, e o SHA-256 é conferido antes de gravar. Sem path traversal. |
| Upload na web | só o basename do nome enviado é usado (`app_logic.py` l.298-317) |
| OPUS / NetCDF | OPUS via `brukeropus` opcional (leitura de arquivo local); ANDI/NetCDF por leitor próprio (`gcms_io.py`, sem `netCDF4`/`pyms`) |
| Guarda de privacidade | identificador de amostra: todo arquivo versionado + valores gerados (já existia); caminho absoluto: **agora** também em todo arquivo versionado (B4) |

### 2.5 Suíte após as correções

**1462 passed, 42 skipped, 0 failed** (580 s). São +2 em relação à linha de base: os dois testes novos de caminho absoluto. Cobertura total 85,2%; núcleo científico idêntico (97 / 97 / 97 / 95%). `ruff` e `mypy` (gate de 50 arquivos) limpos.

---

## 3. Agente 2 — técnicas novas propostas

Os 10 itens têm DOI conferido no Crossref. A lista vai do **menor esforço e maior retorno ao maior esforço**. As lacunas citadas estão registradas no vault, no PROGRESSO, no MANUAL ou no CLAUDE.md local; nenhuma foi inventada aqui.

| # | Técnica | Lacuna que resolve | Esforço | Referência (Crossref ok) | Registro prévio |
|---|---|---|---|---|---|
| T1 | **Predição conforme para regressão**, com escore por grupo: intervalo do teor com cobertura ≥ 1−α | Quantificação sem intervalo e fora do `alpha_total` (`predicao.py` l.684-685); a mesma lacuna aparece como problema #10 do Agente 3 | Baixo: reaproveita `conformal.conformal_threshold`; com <19 grupos, `NAO_VALIDADO` | Lei et al. 2018, JASA, 10.1080/01621459.2017.1307116 | nenhum (só a docstring contraditória, R5a) |
| T2 | **EPO / GLSW**: remove as direções de um fator de perturbação rotulado (hospedeira, rodada, instrumento) | Matriz-hospedeira domina o adulterante (21–175×); quantificação não sobrevive à troca de sessão | Baixo (SVD, ~40 linhas cada). Precisa do rótulo do fator no portão: usar `avaliar_correcao_sinal` genérico | Roger et al. 2003, CILS, 10.1016/S0169-7439(03)00051-0 · Martens et al. 2003, J. Chemom., 10.1002/cem.780 | nenhum. **Não se aplica ao dataset próprio**: ordem × teor é colinear |
| T3 | **ASCA+**: partição simultânea da variância por fator do delineamento, com permutação por unidade experimental | O "21–175×" é R² one-way de script de medição, fora do produto | Baixo a médio; poderia entrar em `auditoria_delineamento` | Smilde et al. 2005, Bioinformatics, 10.1093/bioinformatics/bti476 · Thiel et al. 2017, J. Chemom., 10.1002/cem.2895 | parcial (PERMANOVA por script) |
| T4 | **Correção de deriva por QC/brancos intercalados** (curva suave do QC contra a ordem de aquisição) | `plano_coleta` manda intercalar brancos e nada os consome; é a única via de software compatível com o achado ordem × teor, **só para coletas futuras**. Também cobre o problema #3/#6 do Agente 3. | Baixo no algoritmo, médio no dado. **PRÉ-REQUISITO:** QC com ordem de aquisição (o GC-IMS Zenodo tem 14 QCs; a ordem de injeção não foi confirmada) | Dunn et al. 2011, Nat. Protoc., 10.1038/nprot.2011.335 | parcial (só o planejamento) |
| T5 | **di-PLS**: PLS invariante ao domínio, usando espectros não rotulados do domínio novo | Espécie/sessão/instrumento novos exigem hoje amostras rotuladas (≥6) ou padrões de transferência | Médio (sem implementação pronta verificada) | Nikzad-Langerodi et al. 2018, Anal. Chem., 10.1021/acs.analchem.8b00498 | nenhum |
| T6 | **PQN** | Recomendação da interface sem implementação (R5b); ganho de desempenho esperado baixo (RMN público já em bal.acc 1,000) | Muito baixo (~15 linhas) | Dieterle et al. 2006, Anal. Chem., 10.1021/ac051632c | parcial (só a recomendação) |
| T7 | **Covariância Ledoit-Wolf / LDA com encolhimento** | Mahalanobis inflada por mal-condicionamento (Passo 112); separabilidade em sinal fraco. Ganho **diagnóstico**, não resgata o limite físico do Passo 121. | Baixo (já no scikit-learn) | Ledoit & Wolf 2004, J. Multivar. Anal., 10.1016/S0047-259X(03)00096-4 | nenhum |
| T8 | **Espectro + variável de delineamento** (espécie no modelo) | Alternativa intermediária ao binário pooled/local (Passo 139); sobrepõe-se a T2, então implementar só um dos dois | Médio-baixo; o erro do N1 se propaga ao teor e precisa ser medido | Jørgensen et al. 2004, J. Chemom., 10.1002/cem.890 | nenhum |
| T9 | **MCR-ALS com restrição de correlação** (supervisionado) | O aviso de escopo do MCR-ALS diz "não recomendado sem informação supervisionada"; essa variante nunca foi testada. Retorno **interpretativo**. | Médio | Bayat et al. 2020, Anal. Chim. Acta, 10.1016/j.aca.2020.03.057 | parcial (versão não supervisionada: NEGATIVO, Passos 131/136) |
| T10 | **LWR** (PLS local por vizinhança) | Espécie da amostra nova incerta; baixa prioridade, sobrepõe-se a T8 e ao Passo 139 | Médio | Næs et al. 1990, Anal. Chem., 10.1021/ac00206a003 | nenhum |

**Rejeitadas por inadequação:**
- ComBat (Johnson et al. 2007, 10.1093/biostatistics/kxj037): no confundimento registrado, removeria o próprio sinal.
- GSAM: pré-requisito de adição de padrão por amostra, que o projeto não tem.
- Wavelet: nenhuma lacuna que ela resolva melhor que SG/AirPLS.

**Descartadas por já existirem ou já terem sido avaliadas (19):**
- **Já existentes:** EMSC (inclusive com `interferentes=`, que existe mas não está exposto em `Config`), OSC (rejeitado no óleo), icoshift (registrado), COW, AirPLS, PARAFAC/N-PLS, DS/PDS, pooled vs. local, plano de coleta, sentinela de deriva, auditoria de confundimento, PERMANOVA, NAS/figuras de mérito, seleção de variáveis aninhada, OPLS-DA.
- **Já avaliadas com resultado NEGATIVO:** MCR-ALS não supervisionado para traço minoritário, subtração de fundo instrumental, seleção de banda/reformulação contínua (Passo 112).

---

## 4. Agente 3 — problemas documentados da área × cobertura do GUARACI

Legenda: **R** resolve · **RP** resolve parcialmente · **DNC** detecta mas não corrige · **NR** não resolve. Os 23 DOIs foram conferidos no Crossref. As afirmações de código marcadas com ✔ foram **reconferidas por mim** nesta rodada.

| # | Problema documentado | Fonte | GUARACI | Como, ou o que falta |
|---|---|---|---|---|
| 1 | CV mal desenhada em dado agrupado | Esbensen & Geladi 2010, 10.1002/cem.1310 · Kapoor & Narayanan 2023, 10.1016/j.patter.2023.100804 | **RP** | `StableStratifiedGroupKFold` exige grupos e é estável entre versões. **Falta:** CV e holdout agrupam por `mae_id` ✔ (`pipeline.py` l.1766), e o docstring de `dados_io.session_from_mae_id` ✔ registra que o `mae_id` adulterado é um por nível de teor. Diluições da mesma sessão caem em folds diferentes. Efeito nas métricas **não medido**. |
| 2 | Sem validação externa | Westad & Marini 2015, 10.1016/j.aca.2015.06.056 · Consonni et al. 2010, 10.1002/cem.1290 | **RP** | Holdout por grupo ligado por padrão; KS/Duplex/SPXY group-aware; validação por dia (HSI); 14 datasets públicos. **Falta:** o holdout tabular sai das mesmas sessões do treino e não mede transferência para sessão nova. |
| 3 | Efeito de lote/sessão não corrigido | Leek et al. 2010, 10.1038/nrg2825 · Wehrens et al. 2016, 10.1007/s11306-016-1015-8 | **DNC** | `check_class_session_confounding` detecta e `plano_coleta` previne; não há correção de efeito de sessão (ver T2/T4) |
| 4 | Ordem de leitura confundida com o alvo | Broadhurst & Kell 2006, 10.1007/s11306-006-0037-z | **NR** | O pacote não detecta: o método só existe em script privado, porque o parser DX não expõe o timestamp. `plano_coleta` aleatoriza a ordem das coletas futuras. No dado já coletado nada separa deriva de teor. |
| 5 | Transferência entre instrumentos | Feudale et al. 2002, 10.1016/S0169-7439(02)00085-0 · Workman 2018, 10.1177/0003702817736064 | **RP** | DS/PDS com portão de aceite (PDS confirmado, DS retratado). **Falta:** métodos sem padrões de transferência (ver T5). |
| 6 | Deriva de calibração não monitorada | Kourti & MacGregor 1995, 10.1016/0169-7439(95)80036-9 | **DNC** | `sentinela_deriva.check_drift` (binomial sobre a taxa de rejeição do domínio). **Falta:** deriva de viés que continua dentro do domínio fica invisível; não há carta de controle com amostras de referência. |
| 7 | Seleção de variáveis com vazamento | Ambroise & McLachlan 2002, 10.1073/pnas.102102699 · Brereton 2006, 10.1016/j.trac.2006.10.005 | **R** | Máscara refeita por fold (`_avaliar_subset_nested_cv`, `_avaliar_busca_nested_cv`); viés medido antes da correção: +0,070 |
| 8 | Pré-processamento ajustado antes do split | Kapoor & Narayanan 2023 · Westad & Marini 2015 | **RP** | Pipeline sklearn por fold na CV principal, permutação e portão. **Falta:** a Etapa 4 recebe `X_processed` ✔ ajustado no treino inteiro (`pipeline.py` l.1732). Com MSC, a referência sai de todas as amostras; magnitude **não medida**. |
| 9 | Caixa-preta sem atribuição química | Rudin 2019, 10.1038/s42256-019-0048-x · Mehmood et al. 2012, 10.1016/j.chemolab.2012.07.010 | **RP** | VIP/SR/Martens, bootstrap de VIP por grupo, cruzamento VIP × química (só VIS/fruta). **Falta:** tabela de bandas para NIR/MIR/Raman de óleos; SHAP calculado no treino. |
| 10 | Quantificação sem intervalo por amostra | Olivieri et al. 2006, 10.1351/pac200678030633 | **NR** | Teor pontual + faixa de decisão; LOD/LOQ só global (ver T1) |
| 11 | Classificação sem incerteza | Rodionova et al. 2016, 10.1016/j.trac.2016.01.010 · Pomerantsev 2008, 10.1002/cem.1147 | **RP** | Conformal one-class por grupo com recusa, Bonferroni no `alpha_total`, DD-SIMCA. **Falta:** a espécie do N1 é argmax pontual, sem conjunto de predição. |
| 12 | Nº de VLs escolhido de forma otimista | Filzmoser et al. 2009, 10.1002/cem.1225 · Varma & Simon 2006, 10.1186/1471-2105-7-91 | **RP** | Parcimônia 1,02 × RMSECV; holdout avaliado à parte. **Falta:** as métricas de CV da classificação vêm de `preds_por_lv[n_opt]` ✔, os mesmos folds que escolheram `n_opt`; não há rdCV. |
| 13 | Métricas infladas / IC ingênuo | Chicco & Jurman 2020, 10.1186/s12864-019-6413-7 · Brereton & Lloyd 2014, 10.1002/cem.2609 | **RP** | bal.acc, kappa, F1 macro. **Falta:** `bootstrap_bca_ci` ✔ não aceita grupos e reamostra espectros, então com réplicas o IC sai estreito demais. |
| 14 | Y-randomização incompleta | Westerhuis et al. 2008, 10.1007/s11306-007-0099-6 | **RP** | Permutação por grupo (falso positivo medido: 15,0% por amostra → 4,2% por grupo). **Falta:** `n_opt` fixo dentro do laço, o que deixa o p levemente otimista. |
| 15 | "Melhor modelo" escolhido no mesmo CV | Varma & Simon 2006 | **NR** (declarado) | Benchmark e `compare_pipelines` sem CV aninhada; o portão é a exceção |
| 16 | Pseudo-replicação | Brereton 2006 | **DNC** | `check_insufficient_n` conta sessões; `session_from_mae_id` corrige o n no conformal, **mas não na CV** (#1) |
| 17 | Duplicatas entre treino e teste | Kapoor & Narayanan 2023 | **R** | `validate_input` e `check_duplicates` detectam (a remoção não foi conferida linha a linha) |
| 18 | Predição fora do domínio | Kourti & MacGregor 1995 · Pomerantsev 2008 | **R** | Domínio de aplicabilidade T²/Q por amostra; `check_validation_use_range` |

**Placar:** 3 R · 9 RP · 3 DNC · 3 NR.

**Lacunas genuínas, em ordem de impacto:**
1. Unidade de independência `mae_id` × sessão (#1, #2, #16).
2. Intervalo de predição na quantificação (#10 = T1).
3. Ordem de leitura não detectada no pacote (#4).
4. VLs e permutação otimistas (#12, #14).
5. IC BCa sem grupos (#13).
6. Deriva dentro do domínio (#6).
7. Correção de sessão e atribuição química tabular (#3, #9).
8. Benchmark sem aninhamento (#15).

**Não medido nesta rodada (declarado):** tamanho do vazamento de MSC na Etapa 4; efeito de agrupar por sessão nas métricas; remoção de duplicatas.

---

## 5. Agente 4 — concorrência e comunidade

**Fontes usadas:**
- documentação oficial aberta: Eigenvector docs; manual do Unscrambler 9.6 via archive.org, versão antiga e não conferida contra a atual; guias Sartorius SIMCA-Q e SIMCA 17;
- avaliações do Capterra;
- artigos com DOI conferido no Crossref: Kucheryavskiy 2020 (mdatools), 10.1016/j.chemolab.2020.103937; Andersson 2009, 10.1002/cem.1248; Ezenarro & Schorn-García 2025, 10.1002/cem.70036.

**Fontes descartadas:** o ResearchGate respondeu 403 em todas as tentativas, então **nenhuma discussão de lá entra como evidência**. O que só apareceu em resumo de busca ficou fora.

### 5.1 Limitações por categoria

| Categoria | Achado | Fonte | Recorrência | GUARACI |
|---|---|---|---|---|
| Custo | PLS_Toolbox exige MATLAB e contrato de manutenção pago para receber atualizações. Os preços publicados são de 2008, escritos pela Eigenvector (concorrente). Unscrambler X tem nota 3,3/5 em custo-benefício, a menor entre os critérios. | eigenvector.com (preços e produto); Capterra | recorrente, com fontes fracas | **resolve**: GPL-3.0; a licença comercial só vale para uso embarcado fechado (`docs/COMMERCIAL.md`) |
| Curva de aprendizado | Uma queixa no Capterra, contradita pela nota média (4,3/5) | Capterra | **anedótico** | não usar como argumento; ninguém mediu isso no GUARACI |
| CV anti-vazamento | **Não é ausência.** PLS_Toolbox/Solo têm CV "Custom" com vetor de grupos e documentam a "replicate sample trap". O manual do Unscrambler manda manter réplicas no mesmo segmento. A limitação real é que **não é o padrão**: o padrão particiona por linha. | eigenvectordocs.com (Using_Cross-Validation, Crossval); manual Unscrambler 9.6 | recorrente (2 fornecedores) | **resolve parcialmente**: group-aware por padrão, também no holdout, **mas por `mae_id`, não por sessão** (= §4 #1) |
| Validação só por CV na prática | Revisão sistemática: muitos modelos NIRS em alimentos só com CV, com estimativa otimista | Ezenarro & Schorn-García 2025 | recorrente | **resolve** o ponto "só CV" (holdout group-aware por padrão) |
| Incerteza | Unscrambler dá "Predicted with Deviation" por amostra e teste de incerteza de Martens | manual Unscrambler 9.6 | confirmação de recurso | **não resolve** o intervalo por amostra (= §4 #10, T1); Martens existe (`chemometric_stats.py`, jackknife group-aware), mas não é diferencial |
| Reprodutibilidade | PLS_Toolbox tem linha de comando completa e SIMCA-Q tem API Python/C#: o pipeline é scriptável. Nenhum relato aberto de resultado que muda entre versões. | eigenvector.com; guia SIMCA-Q | sem evidência de limitação | splitter estável + `generate_manifest` resolvem o risco **próprio**; não sustenta "concorrente não é reprodutível" |
| Multitécnica | PLS_Toolbox `multiblock`; SIMCA O2PLS (integração de dados) | eigenvectordocs.com (Multiblock); guia SIMCA-Q | confirmação de recurso | **não resolve**: zero implementação de fusão multibloco; várias técnicas, uma de cada vez |
| Formatos | PLS_Toolbox tem cerca de 35 leitores (SPC, PerkinElmer, Thermo, ASD, netCDF, PI, SQL…). Model_Exporter gera o modelo em .py/.m/XML. | eigenvectordocs.com (Importing_Data, Model_Exporter) | recorrente (2 fornecedores) | **parcial**: JCAMP, CSV, OPUS, ANDI-MS e ENVI; o modelo é `.joblib`, preso às versões de Python e sklearn |
| Automação / processo | SIMCA-online e Aspen Process Pulse: monitoramento em tempo real com alarmes. Automação costuma ser produto à parte. | sartorius.com; aspentech.com | recorrente (2 fornecedores) | **parcial e bem menor**: `sentinela_deriva.py` offline, lote a lote. A CLI sem argumentos abre o assistente, e não há `guaraci run config.yaml`. |

### 5.2 Onde os concorrentes estão à frente (lacunas reais)
1. Intervalo de predição por amostra na quantificação (Unscrambler). É a mesma lacuna de §4 #10 e da proposta T1.
2. Fusão multibloco (PLS_Toolbox, SIMCA).
3. Variedade de leitores de instrumento (PLS_Toolbox).
4. Exportação portátil do modelo (Eigenvector Model_Exporter).
5. Monitoramento em linha / MSPC (SIMCA-online, Aspen).
6. Execução não interativa pela CLI. PLS_Toolbox e SIMCA-Q são scriptáveis; o GUARACI só oferece a API Python para isso.
7. Maturidade de interface, suporte e treinamento. É fato de estrutura: o GUARACI é projeto de um autor.

### 5.3 Correções necessárias no comparativo do README (propostas, não aplicadas)
- **L23-24** ("the project's methodological differentiator") e **L56** ("Group-aware validation … ⚠️ limited"): os concorrentes têm CV por grupo configurável. O defensável é "group-aware **por padrão**, também no holdout", com a ressalva de que o agrupamento é por amostra física, não por sessão.
- **L57** ("Reproducible … ❌"): sem evidência. Trocar por "possível via script, não é o fluxo padrão" ou remover a linha.
- **L27** ("no closed-format lock-in"): vale para o código e para os formatos de entrada. O arquivo de modelo é `.joblib`, dependente de versão.
- **L62-66** ("calibration transfer … reasonably assumed present"): é suposição; verificar na documentação ou reescrever sem "assumed".
- **L59** e **L55**: nada contradiz. Podem ficar.

---

## 6. O que o vault já sabia

- **Agente 2:** de **31 candidatas** cogitadas, **21** já estavam registradas no vault, no PROGRESSO ou no código, total ou parcialmente. Foram 19 descartadas por já existirem ou já terem sido avaliadas, e 4 propostas têm registro parcial (T3, T4, T6, T9; a contagem de candidatas distintas é 21 porque dois descartes são subcasos). **10** não tinham registro nenhum. Sem a consulta prévia, EMSC, OSC, MCR-ALS não supervisionado e subtração de fundo teriam sido repropostos; os três últimos já tinham resultado negativo medido.
- **Agente 3:** dos **18** problemas, **9** já tinham nota correspondente no vault (alguns só na versão HSI), **7** tinham registro parcial e **2** nenhum (#12, #15). O vault **não** registrava nenhuma das falhas específicas achadas agora: agrupamento por `mae_id` na CV, BCa sem grupos, `X_processed` da Etapa 4, `n_opt` fixo na permutação. O achado ordem × teor (ρ +0,997) está só no CLAUDE.md local, não no vault.
- **Limite da ferramenta:** a busca por sigla casa substring (R3), então "registro prévio" por sigla curta precisa de conferência manual. O Agente 2 fez essa conferência para EPO.
- **Agente 4:** o vault **não tinha pesquisa própria sobre concorrentes**. Havia só a decisão `50-Decisoes/decisao-adicionar-1-linha-ao-comparativo.md` (acrescentar a L59 com cautela e não reivindicar Kennard-Stone nem transferência de calibração) e as notas-espelho do próprio README. Nenhuma das 7 lacunas de §5.2 nem das correções de §5.3 estava registrada. A cautela registrada na L59 se confirmou.
- **Convergência entre frentes, mesclada e não duplicada:** o intervalo de predição da quantificação apareceu nos Agentes 2 (T1), 3 (#10) e 4 (Unscrambler); o agrupamento por sessão, nos Agentes 3 (#1) e 4 (ressalva da CV); a correção de efeito de sessão por QC, nos Agentes 2 (T4) e 3 (#3/#6).

---

## 7. Recomendação de próximo passo (impacto × esforço; decisão do autor)

**Acionável agora (baixo risco, fora da lógica científica):**
1. Conferir o CI verde após o push (B1/B2). Se o Zenodo voltar a falhar, decidir sobre retry (R1).
2. Decidir sobre o caminho que ainda está no histórico git (B4).
3. Corrigir as três inconsistências de documentação (R5), junto com a decisão sobre T1/T6.
4. **Reescrever o comparativo do README (§5.3):** alegação pública contradita por documentação dos concorrentes. Esforço baixo e sem risco técnico; o texto precisa da aprovação do autor.

**Decisão científica (alto impacto; toca lógica validada, então só com aprovação):**
5. **Medir primeiro, depois decidir:** métricas com agrupamento por sessão (`session_from_mae_id`) contra `mae_id`, em dataset público com estrutura equivalente e no dataset próprio (Agente 3, #1). É esforço baixo de medição e o resultado define se números citados precisam de errata.
6. `bootstrap_bca_ci` por grupo (#13) e métricas de CV sem viés de seleção de VLs (#12): esforço baixo a médio, afeta os ICs reportados.

**Técnicas e produto (médio prazo):**
7. T1 (intervalo conforme para regressão): fecha a lacuna #10, a contradição R5a e a lacuna frente ao Unscrambler, a única proposta que aparece nas três frentes de pesquisa.
8. T2 (EPO/GLSW) + T3 (ASCA+): diagnóstico e remoção de fator, testáveis no EEM Zenodo.
9. T4 (QC/brancos): só com dado de QC ordenado.
10. Lacunas de produto do Agente 4, sem referência científica a conferir: execução não interativa pela CLI (`guaraci run config.yaml`), fusão multibloco, exportação portátil do modelo e mais leitores de formato. Precisam de avaliação de esforço antes de entrar na fila.

Nada disso foi implementado nesta rodada.
