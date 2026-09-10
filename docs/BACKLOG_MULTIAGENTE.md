# Backlog — rodada multiagente de 2026-09-10

Rastreamento de **todo item** de `docs/RELATORIO_MULTIAGENTE_2026-09-10.md`
(Passo 200), para que "nada fica de fora" seja verificável. Cada linha tem
um status:

- `implementado nesta rodada` — código mudou, com contra-prova.
- `medido, decisão pendente` — número já existe, falta decisão do autor.
- `backlog — não priorizado ainda` — registrado, nada mudou.

Atualizado pela última vez no fechamento da instrução de 2026-09-10
("IMPLEMENTAR TODOS OS ACHADOS"), Passos 201+.

---

## A. Bugs reportados (Agente 1, §2.3 do relatório)

| # | Item | Status | Nota |
|---|---|---|---|
| R1 | Retry com backoff em `baixar_zenodo_*` | _preenchido na Fase 2_ | |
| R2 | `avaliacao_modelos.py` (6 erros) e `resultados_io.py` (13 erros) | **corrigido (Passo 202)** — ambos `type: ignore` obsoletos (código de erro do mypy mudou de versão para versão) mais narrowing real com `isinstance` em 1 ponto; suíte de ambos os módulos passando | |
| R2 | `hsi_pipeline.py` (31 erros) | **backlog — esforço maior**, não corrigido nesta rodada. Raiz sistêmica: um valor `object` (de config/manifesto lido de forma solta) se propaga por ~10 chamadas de função tipadas ao longo do módulo (`build_pixel_dataset`, `run_internal_validation_group_aware`, `run_external_validation_by_day`, `PLSDAClassifier`, `enrich_object_results`). Corrigir de verdade exige anotar o tipo na ORIGEM da leitura, não remendar cada call site — risco de introduzir tipo sutilmente errado num módulo científico sem tempo para validar cada ponto. | |
| R3 | `consultar_vault.py` casa substring em sigla (`"EPO"` → "tempo") | _preenchido na Fase 2_ | |
| R4 | 5 PRs do Dependabot abertas, nunca revisadas | _preenchido na Fase 2_ | |
| R5a | Docstring de `hsi_uncertainty.py` alega intervalo de predição inexistente | _preenchido na Fase 2_ | |
| R5b | `cli_assistente.py` recomenda PQN para RMN sem implementação | _preenchido na Fase 2_ | vinculado a T6 |
| R5c | `auditoria_delineamento.check_external_validation` com mensagem fixa desatualizada | _preenchido na Fase 2_ | |

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
| 10 | Quantificação sem intervalo por amostra | NR | _preenchido na Fase 4 (T1)_ |
| 11 | Classificação sem incerteza | RP | já parcialmente resolvido (conformal one-class); backlog o resto |
| 12 | Nº de VLs escolhido de forma otimista | RP | _preenchido na Fase 3_ |
| 13 | Métricas infladas / IC ingênuo | RP | _preenchido na Fase 3_ |
| 14 | Y-randomização incompleta | RP | _preenchido na Fase 3_ |
| 15 | "Melhor modelo" escolhido no mesmo CV | NR (declarado) | _preenchido na Fase 3_ |
| 16 | Pseudo-replicação | DNC | = item M1/M3 |
| 17 | Duplicatas entre treino e teste | R | já resolvido; remoção linha a linha não reconferida (backlog, esforço baixo) |
| 18 | Predição fora do domínio | R | já resolvido; nenhuma ação |

## D. Técnicas novas — Agente 2 (T1–T10, §3 do relatório)

| # | Técnica | Status |
|---|---|---|
| T1 | Predição conforme para regressão | _preenchido na Fase 4_ |
| T2 | EPO / GLSW | _preenchido na Fase 4_ |
| T3 | ASCA+ | _preenchido na Fase 4_ |
| T4 | Correção de deriva por QC/brancos | _preenchido na Fase 4_ |
| T5 | di-PLS | **backlog — não priorizado nesta rodada** (instrução determina registrar, não implementar) |
| T6 | PQN | _preenchido na Fase 4_ |
| T7 | Ledoit-Wolf / LDA com encolhimento | _preenchido na Fase 4_ |
| T8 | Espectro + variável de delineamento | **backlog — não priorizado** (sobrepõe-se a T2, instrução manda registrar) |
| T9 | MCR-ALS com restrição de correlação | _preenchido na Fase 4_ |
| T10 | LWR (PLS local por vizinhança) | **backlog — não priorizado** (sobrepõe-se a T8/Passo 139) |

## E. Lacunas de produto — Agente 4 (§5.2 do relatório, Fase 5)

| # | Lacuna | Status |
|---|---|---|
| P1 | Execução não interativa pela CLI (`guaraci run config.yaml`) | _avaliado na Fase 5_ |
| P2 | Exportação portátil do modelo (alternativa ao `.joblib` puro) | _avaliado na Fase 5_ |
| P3 | Mais leitores de formato de instrumento | _avaliado na Fase 5_ |
| P4 | Fusão multibloco | **backlog — escopo grande, não iniciar sem instrução própria** |
| P5 | Monitoramento em linha / MSPC | **backlog — escopo grande, não iniciar sem instrução própria** |
| P6 | Intervalo de predição por amostra (Unscrambler) | = T1 (mesma lacuna, ver D) |
| P7 | Maturidade de interface/suporte/treinamento | **backlog — fato de estrutura, não é item de engenharia** |

## F. Correções do comparativo do README — Agente 4 (§5.3 do relatório)

| # | Linha do README | Status |
|---|---|---|
| C1 | L23-24 / L56 — "methodological differentiator" / "Group-aware … ⚠️ limited" | _preenchido na Fase 2_ |
| C2 | L57 — "Reproducible … ❌" sem evidência | _preenchido na Fase 2_ |
| C3 | L27 — "no closed-format lock-in" sem qualificar o `.joblib` | _preenchido na Fase 2_ |
| C4 | L62-66 — transferência de calibração "reasonably assumed present" | _preenchido na Fase 2_ |
| C5 | L59 / L55 — nada contradiz, mantidos sem mudança | não se aplica (confirmado correto) |

---

## Itens explicitamente fora desta rodada (com razão)

- **M3** (propagar `session_from_mae_id` ao pipeline principal): a instrução e a regra de ouro exigem aprovação explícita do autor sobre o número medido antes de tocar em lógica científica já validada. Fica pendente até essa aprovação chegar.
- **T5, T8, T10**: a própria instrução (Fase 4) manda registrar no backlog, não implementar.
- **P4, P5, P7**: escopo grande de produto; a instrução manda registrar para instrução futura dedicada, não iniciar aqui.
- **Problemas #2, #3, #4, #5, #6, #8, #9, #11 (parte), #17 (parte)** do Agente 3: nenhum é bug de baixo risco nem está na lista priorizada de Fases 3/4 desta instrução; ficam no backlog científico geral, sem prazo.
