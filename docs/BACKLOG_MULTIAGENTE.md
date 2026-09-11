# Backlog — rodada multiagente de 2026-09-10

Rastreamento de **todo item** de `docs/RELATORIO_MULTIAGENTE_2026-09-10.md`
(Passo 200), para que "nada fica de fora" seja verificável. Cada linha tem
um status:

- `implementado nesta rodada` — código mudou, com contra-prova.
- `medido, decisão pendente` — número já existe, falta decisão do autor.
- `backlog — não priorizado ainda` — registrado, nada mudou.

Atualizado pela última vez no fechamento da Fase 6 (consolidação final)
da instrução "IMPLEMENTAR TODOS OS ACHADOS" de 2026-09-10, Passos
201–208. Nenhum item do relatório original ficou sem status.

---

## A. Bugs reportados (Agente 1, §2.3 do relatório)

| # | Item | Status | Nota |
|---|---|---|---|
| R1 | Retry com backoff em `baixar_zenodo_*` | **corrigido (Passo 202, commit `2d85345`)** — `scripts/download_datasets/_http_retry.py`, backoff exponencial, HTTP 4xx não retentado, falha após esgotar tentativas continua sendo falha real | |
| R2 | `avaliacao_modelos.py` (6 erros) e `resultados_io.py` (13 erros) | **corrigido (Passo 202)** — ambos `type: ignore` obsoletos (código de erro do mypy mudou de versão para versão) mais narrowing real com `isinstance` em 1 ponto; suíte de ambos os módulos passando | |
| R2 | `hsi_pipeline.py` (31 erros) | **corrigido na origem e commitado (Passo 212, 2026-09-10)** — `apply_quality_gate_and_segment` (`hsi_pipeline.py`) e `fit_predict_pixel_plsda` (`hsi_classification.py`) ganharam `TypedDict` (`QualityGateResult`, `PixelPLSDAResult`) no lugar de `Dict[str, object]` — a raiz de fato, não um remendo por call site: os 3 `cast()` locais em `hsi_validation.py` e o `# type: ignore` em `hsi_multiway.py` ficaram desnecessários e foram removidos. `mypy` limpo nos 54 módulos do gate + `hsi_pipeline.py` (antes: 31 erros, agora incluído no gate da CI); `ruff` limpo; suíte completa **1534 passed/42 skipped** (zero mudança de comportamento — só anotação de tipo, `cast()`/`TypedDict` são no-ops em runtime). O retorno das 2 funções, já públicas (`__all__`), mudou de assinatura textual (`Dict[str, object]` → `TypedDict`), o que reprovava `tests/test_contrato_api_publica.py` — **decisão do autor (2026-09-10): regravar o golden agora**, aceitando a mudança como aditiva/não-quebradora em runtime, sem decidir bump de versão (fica para quando a diretriz de versionamento for destravada). `PixelPLSDAResult` exportado em `__all__` de `hsi_classification.py` pela mesma razão de `QualityGateResult` já estar em `hsi_pipeline.py` (consistência). | |
| R3 | `consultar_vault.py` casa substring em sigla (`"EPO"` → "tempo") | **corrigido (Passo 202, commit `2d85345`)** — sigla curta (maiúscula/dígito/hífen, até 6 chars) exige fronteira de palavra; termo comum continua com substring simples | |
| R4 | 5 PRs do Dependabot abertas, nunca revisadas | **5/5 mergeadas (2026-09-10)** | Reconfirmado contra o `master` avançado (16 commits desde a última revisão): `#8`/`#9`/`#10`/`#20` com CI 100% verde e `mergeStateStatus=CLEAN` de imediato — mergeadas via `gh pr merge --squash --delete-branch`. `#11` (`upload-pages-artifact` v3→v5) tinha 1 check `validacao-publica` falho por infraestrutura (`curl: Failed to connect to eigenvector.com`, timeout de rede baixando `corn.mat` no runner — não relacionado ao bump em si, que só toca `docs.yml`); `gh run rerun --failed` confirmou ser flake (100% verde na 2ª tentativa) — mergeada em seguida. Todas as 5 mergeadas, nenhuma exigiu revisão além da rotina (bumps mecânicos, sem código de aplicação tocado). |
| R5a | Docstring de `hsi_uncertainty.py` alega intervalo de predição inexistente | **fechado (Passo 203, T1)** — deixou de ser inconsistência: T1 implementou o intervalo, a frase agora descreve algo real; nota datada explica a mudança de status aspiracional → real | |
| R5b | `cli_assistente.py` recomenda PQN para RMN sem implementação | **fechado (Passo 202, T6)** — PQN implementado, a recomendação passou a ser verdadeira | vinculado a T6 |
| R5c | `auditoria_delineamento.check_external_validation` com mensagem fixa desatualizada | **corrigido (Passo 201, commit `9f5fe01`)** — mensagem distingue PLS-DA (14 datasets públicos) de DD-SIMCA/OPLS-DA (ainda sem benchmark externo) | |

## B. Medição crítica (Agente 3 #1, Fase 1 — bloqueante)

| # | Item | Status | Nota |
|---|---|---|---|
| M1 | Medir `mae_id` vs `session_from_mae_id` na CV/holdout principal (classificação N1, dataset privado) | **medido (Passo 201)** | `scripts/medicoes/medir_mae_id_vs_sessao.py`. Resultado: não é uma diferença numérica — 8/13 espécies têm 1 única sessão; agrupar por sessão produz folds vazios (medido: 1432/144/96/0/0 em 5 folds), estruturalmente inexecutável com o splitter real do projeto. |
| M2 | Confirmar escopo: só dataset privado, ou também alguma das 14 validações públicas | **confirmado (Passo 201), ANTES da medição numérica** | `grep group_by_mae_id=False` em todos os `tests/test_validacao_publica*.py` — nenhum dataset público agrupa por `mae_id`; o achado é exclusivo do dataset privado adulterado. `docs/VALIDACAO_PUBLICA.md` não precisa de correção. |
| M3 | Propagar a correção ao pipeline principal | **inaplicável, registrado (Passo 201)** — não há substituição executável de `mae_id` por sessão hoje (ver M1); corrigir de fato exige coleta futura com mais sessões independentes por espécie, fora de escopo de código. Nenhuma mudança em `pipeline.py`. | |
| M4 | Verificar se documento já entregue (relatório PIBIC) cita métrica do dataset privado como proteção contra vazamento **entre sessões** | **decisão/verificação pendente do autor** — sem visibilidade do que já foi submetido externamente | |

## C. Problemas documentados da área — Agente 3 (18 itens, §4 do relatório)

| # | Problema | Status no relatório | Ação nesta rodada |
|---|---|---|---|
| 1 | CV mal desenhada em dado agrupado | RP | medido (M1): impossibilidade estrutural, não correção de código — ver M1/M3 |
| 2 | Sem validação externa | RP | backlog — não priorizado (falta transferência de sessão nova) |
| 3 | Efeito de lote/sessão não corrigido | DNC | backlog — ver T2/T4 |
| 4 | Ordem de leitura confundida com o alvo | NR | backlog — pré-requisito de timestamp no parser DX, fora de escopo desta rodada |
| 5 | Transferência entre instrumentos | RP | backlog — ver T5 |
| 6 | Deriva de calibração não monitorada | DNC | backlog — carta de controle não implementada |
| 7 | Seleção de variáveis com vazamento | R | já resolvido antes desta rodada; nenhuma ação |
| 8 | Pré-processamento ajustado antes do split (Etapa 4) | RP | backlog — magnitude do vazamento de MSC não medida |
| 9 | Caixa-preta sem atribuição química | RP | backlog — tabela de bandas NIR/MIR/Raman de óleos não existe |
| 10 | Quantificação sem intervalo por amostra | NR | **resolvido (Passo 203, T1)** — ver acima |
| 11 | Classificação sem incerteza | RP | já parcialmente resolvido (conformal one-class); backlog o resto |
| 12 | Nº de VLs escolhido de forma otimista | RP | **RETRATAÇÃO do Passo 202 — replicado (10 seeds, Passo 211), resultado agora CONSISTENTE com a hipótese original, decisão de propagar pendente de aprovação do autor.** A medição do Passo 202 foi **1 única execução** (1 seed de split) — insuficiente para decidir, como a própria nota já registrava. `scripts/medicoes/medir_vieses_selecao_lv_replicado.py` repetiu a mesma comparação em 10 seeds independentes + Wilcoxon pareado (mesmo padrão de `comparar_npls_pixelwise_mango.py`): **a versão NAIVE (atual) teve balanced_accuracy maior que a ANINHADA em 10/10 seeds** (médias 0,9183 vs. 0,8756, delta médio +0,0427, Wilcoxon p=0,0020) — **direção OPOSTA à do Passo 202, e agora SIM na direção da hipótese original do relatório** (a seleção atual infla a métrica reportada por reusar o mesmo dado para escolher `n_opt` e avaliar). O número isolado do Passo 202 (nested maior) foi artefato daquele 1 split específico, não um efeito real — retratado explicitamente aqui, não silenciado. Custo real da réplica: 2445s (~40,7 min) para as 10 seeds, reportado ANTES de rodar (estimativa ~2250s com base no tempo do Passo 202, 48s+177s/seed). **Decisão pré-aprovada da instrução de 2026-09-10 satisfeita** (resultado consistente e significativo nas réplicas) — mas **NÃO propagado ao pipeline nesta rodada**, por instrução explícita de aguardar aprovação do autor mesmo com resultado consistente. Propor ao autor: trocar a seleção de `n_opt` por CV aninhada custaria ~4,4× o tempo de CV (ver #14) mas removeria um viés otimista real e mensurável na métrica principal reportada (`balanced_accuracy`). |
| 13 | Métricas infladas / IC ingênuo | RP | **corrigido e medido (Passo 202)** — `bootstrap_bca_ci(groups=...)`, retrocompatível; largura do IC de balanced_accuracy sobe de 0,0316 para 0,0411 (+30%) no dataset privado; propagado a `pipeline.py` (CV e holdout) |
| 14 | Y-randomização incompleta | RP | **avaliado (Passo 202), documentado como proibitivo** — mover a seleção de LVs para dentro de cada permutação custaria ~150-200× o tempo atual (200 permutações × custo ~4,4× medido em #12, atualizado com a média das 10 seeds replicadas do Passo 211 — antes 3,7× de 1 única execução); mantido como limitação declarada |
| 15 | "Melhor modelo" escolhido no mesmo CV | NR (declarado) | **avaliado (Passo 202)** — mesma restrição computacional de #14; mantido como limitação já documentada em `resultados_io._NOTAS_METODOLOGICAS` |
| 16 | Pseudo-replicação | DNC | = item M1/M3 |
| 17 | Duplicatas entre treino e teste | R | já resolvido; remoção linha a linha não reconferida (backlog, esforço baixo) |
| 18 | Predição fora do domínio | R | já resolvido; nenhuma ação |

## D. Técnicas novas — Agente 2 (T1–T10, §3 do relatório)

| # | Técnica | Status |
|---|---|---|
| T1 | Predição conforme para regressão | **implementado (Passo 203)** — `conformal.conformal_margin_regression`, wired em `pls_regression_by_species`/`quantify_sample`/`predict_blind` (alpha_total). Fecha #10 e o achado R5a. |
| T2 | EPO / GLSW | **implementado (Passo 205)** — `epo_glsw.py` (EPO + GLSW + `build_difference_matrix`). Não é um transformer de Pipeline padrão (recebe matriz de diferenças pronta, ver docstring). Não se aplica ao dataset próprio (ordem×teor colinear). |
| T3 | ASCA (+) | **implementado (Passo 204)** — `asca.py`, decomposição marginal (Smilde et al. 2005) + permutação por unidade experimental. Escopo honesto: NÃO é a extensão "+" completa (Thiel et al. 2017) para fatores correlacionados; `ss_desbalanco` sinaliza quando ela seria necessária. |
| T4 | Correção de deriva por QC/brancos | **implementado (Passo 207)** — `deriva_qc.corrigir_deriva_por_qc` (QC-RLSC). Pré-requisito de dado (QC com ordem de aquisição) não satisfeito hoje — função utilizável só com dado fornecido pelo chamador, não ligada a fluxo automático. |
| T5 | di-PLS | **backlog — não priorizado nesta rodada** (instrução determina registrar, não implementar) |
| T6 | PQN | **implementado (Passo 202)** — `preprocessamento.PQN` + `cfg.apply_pqn` (preset custom). Fecha R5b. |
| T7 | Ledoit-Wolf / LDA com encolhimento | **implementado (Passo 202)** — `chemometric_stats.mahalanobis_distance_shrinkage` (`estimador='shrinkage'`/`'raw'` lado a lado). Ganho diagnóstico, não promete resgatar separação real. |
| T8 | Espectro + variável de delineamento | **backlog — não priorizado** (sobrepõe-se a T2, instrução manda registrar) |
| T9 | MCR-ALS com restrição de correlação | **implementado (Passo 206)** — `mcr_als_com_restricao_correlacao`. Nunca testada contra o acervo real (proposta nova, não correção do achado negativo da versão não supervisionada). |
| T10 | LWR (PLS local por vizinhança) | **backlog — não priorizado** (sobrepõe-se a T8/Passo 139) |

## E. Lacunas de produto — Agente 4 (§5.2 do relatório, Fase 5)

| # | Lacuna | Status |
|---|---|---|
| P1 | Execução não interativa pela CLI (`guaraci run config.yaml`) | **implementado (Passo 208)** — comando `run`, 10 testes |
| P2 | Exportação portátil do modelo (alternativa ao `.joblib` puro) | **avaliado (Passo 208), não implementado** — ONNX/PMML exportam 1 modelo por vez (não cabe a estrutura multi-espécie/classes customizadas); `skops` exigiria registrar cada classe própria. Projeto de migração próprio, fora desta rodada. |
| P3 | Mais leitores de formato de instrumento | **avaliado (Passo 208), não implementado** — candidatos priorizados: SPC (Galactic/Thermo) e PerkinElmer `.sp`, maior demanda genérica frente aos ~35 do PLS_Toolbox. Não iniciado. |
| P4 | Fusão multibloco | **escopo registrado (2026-09-10)** — `docs/ESCOPO_FUSAO_MULTIBLOCO_E_MSPC.md` §1: candidato de prova de conceito identificado (Mendeley `ctgg7k4m5g.2`, NIR8mm+MIR50µm nas MESMAS amostras, já parcialmente integrado), esforço estimado comparável a `eem_multiway.py` (Passo 149) + 1 técnica de integração nova. Não iniciar sem instrução própria. |
| P5 | Monitoramento em linha / MSPC | **escopo registrado (2026-09-10)** — `docs/ESCOPO_FUSAO_MULTIBLOCO_E_MSPC.md` §2: extensão de `sentinela_deriva.py` (núcleo estatístico já existe), "em linha" reinterpretado como execução agendada via `guaraci run` (local-first, sem servidor/daemon novo). Esforço estimado menor que P4. Não iniciar sem instrução própria. |
| P6 | Intervalo de predição por amostra (Unscrambler) | = T1 (mesma lacuna, ver D) |
| P7 | Maturidade de interface/suporte/treinamento | **backlog — fato de estrutura, não é item de engenharia** |

## F. Correções do comparativo do README — Agente 4 (§5.3 do relatório)

| # | Linha do README | Status |
|---|---|---|
| C1 | L23-24 / L56 — "methodological differentiator" / "Group-aware … ⚠️ limited" | **corrigido (Passo 202, commit `0338132`)** — título/prosa trocados para "enabled by default"; tabela: concorrente "✅ disponível, configurado pelo usuário" vs. "✅ por padrão, incl. holdout". Aplicado em README.md e README.pt-br.md. |
| C2 | L57 — "Reproducible … ❌" sem evidência | **corrigido (Passo 202)** — trocado por "possível via script (PLS_Toolbox/MATLAB, SIMCA-Q), não é o fluxo padrão da GUI" |
| C3 | L27 — "no closed-format lock-in" sem qualificar o `.joblib` | **corrigido (Passo 202)** — mission statement reescrito; linha nova na tabela: modelo é `.joblib`/pickle preso à versão, concorrente exporta `.py`/`.m`/XML (mesmo achado do P2) |
| C4 | L62-66 — transferência de calibração "reasonably assumed present" | **corrigido (Passo 202)** — reescrito sem apresentar suposição como fato verificado |
| C5 | L59 / L55 — nada contradiz, mantidos sem mudança | não se aplica (confirmado correto) |

---

## Itens explicitamente fora desta rodada (com razão)

- **M3** (propagar `session_from_mae_id` ao pipeline principal): a instrução e a regra de ouro exigem aprovação explícita do autor sobre o número medido antes de tocar em lógica científica já validada. Fica pendente até essa aprovação chegar.
- **T5, T8, T10**: a própria instrução (Fase 4) manda registrar no backlog, não implementar.
- **P4, P5, P7**: escopo grande de produto; a instrução manda registrar para instrução futura dedicada, não iniciar aqui.
- **Problemas #2, #3, #4, #5, #6, #8, #9, #11 (parte), #17 (parte)** do Agente 3: nenhum é bug de baixo risco nem está na lista priorizada de Fases 3/4 desta instrução; ficam no backlog científico geral, sem prazo.
