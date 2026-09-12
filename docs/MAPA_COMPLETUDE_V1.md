# GUARACI — MAPA DE COMPLETUDE PARA v1.0.0

**Propósito deste documento:** registro permanente de tudo que falta, está parcial, ou não tem garantia, levantado antes da publicação — para nunca mais repetir o erro de declarar "pronto para publicar" enquanto lacunas reais (como o HSI da rodada anterior) ficam de fora sem registro. Este documento é a fonte única de verdade sobre "o que falta" até a publicação. Deve ser atualizado a cada grupo fechado, nunca reescrito do zero.

**Regra de uso:** nenhum item sai deste documento por estar implementado sem contra-prova real (mesma disciplina de toda a sessão). Nenhum item é descartado sem razão registrada. Ao final de todos os grupos, se não houver pendência aberta aqui, o projeto está de fato pronto para publicação — não antes.

---

## GRUPO 1 — Leitores de formato / compatibilidade de equipamento

Objetivo: um usuário com qualquer um destes equipamentos consegue usar o dado bruto real, sem pré-processar fora do GUARACI primeiro.

| Item | Estado atual | O que falta |
|---|---|---|
| JCAMP-DX (`.dx`) | Completo, núcleo histórico | — |
| CSV genérico | Completo | — |
| OPUS (Bruker FT-IR/FT-NIR) | Implementado (`brukeropus`) | Testado só estruturalmente em alguns pontos — confirmar cobertura real |
| NetCDF/ANDI-MS (GC-MS) | Implementado, parser próprio | — |
| ENVI (HSI) | Implementado, genérico (aceita dado do usuário) | — |
| Imagem colorimétrica (JPG/PNG) | Implementado, 3 níveis de agrupamento | — |
| SPC (Galactic/Thermo) | Não lido | Prioridade alta — formato muito comum em espectroscopia comercial |
| PerkinElmer `.sp` | Não lido | Prioridade alta |
| RMN bruto (Bruker/Varian, FID/espectro processado nativo) | Só aceita dado já binado de terceiro | Sem isso, RMN só serve para reanalisar dataset público, não para uso de laboratório real |
| GC-IMS bruto (`.mea`) | Só aceita tabela de pico já extraída | Formato do instrumento nunca lido diretamente |
| Fluorescência EEM bruta | Só aceita formato específico de 1 dataset público | Sem parser genérico de matriz excitação-emissão |
| HPLC cromatograma bruto | Só aceita tabela de pico já extraída | Sem leitor de cromatograma bruto (nenhum formato de instrumento) |
| UV-Vis bruto de equipamento comercial | Só CSV genérico testado | Formato proprietário (ex.: Agilent, Shimadzu) não avaliado |

**Prioridade sugerida dentro do grupo:** SPC e `.sp` primeiro (maior demanda genérica, formato bem documentado, bibliotecas Python maduras já identificadas). RMN bruto e GC-IMS bruto depois (formato mais fechado, exige mais engenharia reversa ou biblioteca de terceiro com licença a confirmar).

---

## GRUPO 2 — Análises e capacidades faltantes

| Item | Estado atual | O que falta |
|---|---|---|
| Conjunto de predição para regressão (T1) | Implementado | — |
| Conjunto de predição para classificação multiclasse | Não existe | Hoje é argmax pontual sem intervalo por classe — inconsistente com o rigor já aplicado à regressão |
| Fusão multibloco | Só escopo (`docs/ESCOPO_FUSAO_MULTIBLOCO_E_MSPC.md`) | Zero código — candidato de prova de conceito já identificado (Mendeley NIR8mm+MIR) |
| Monitoramento em linha / MSPC | Só escopo | Zero código — extensão de `sentinela_deriva.py` já desenhada |
| ASCA+ completo (fatores correlacionados) | Só a versão marginal (Smilde et al. 2005) | Extensão Thiel et al. 2017 não implementada |
| Explicabilidade fora do treino | SHAP só no próprio treino | Reconhecidamente otimista, sem correção |
| di-PLS (domínio novo sem rótulo) | Registrado, não implementado (T5) | Backlog explícito |
| Espectro + variável de delineamento (T8) | Registrado, não implementado | Sobrepõe-se a EPO/GLSW, decidir um dos dois |
| PLS local por vizinhança (T10) | Registrado, não implementado | Baixa prioridade |
| Exportação portátil de modelo | Preso a `.joblib` | ONNX/PMML avaliados e rejeitados (não cabem na estrutura multi-espécie); alternativa própria não desenhada |
| Execução scriptável rica | `guaraci run config.yaml` existe | Recente, pouco testado em uso real; sem exemplos de integração externa documentados |

---

## GRUPO 3 — Performance, leveza e limpeza (MEDIDO em 2026-09-11)

Medido por instrução dedicada (ver scripts versionados em `scripts/medicoes/`). Metodologia e limiares: Nielsen/Miller — 0,1s instantâneo, 1,0s fluxo de pensamento preservado, 10s limite de atenção (acima disso, indicação de progresso obrigatória).

| Item | Resultado medido |
|---|---|
| Tempo de inicialização (CLI) | Média 3,89s ± 0,11s (10 execuções cold start) — tolerável p/ Nielsen, mas **1,0s é um `time.sleep(1.0)` fixo** na mensagem de boas-vindas ([guaraci.py:976](../src/guaraci/guaraci.py)), não computação real. Import (`-X importtime`) responde por ~2,0-2,4s (sklearn ~1,0s, scipy ~0,5s, pandas ~0,4s, matplotlib ~0,3s); nenhuma dependência pesada opcional (streamlit/tensorly/brukeropus/shap/xgboost) é importada no boot — já corretamente isolada. |
| Tempo de inicialização (Streamlit) | Boot até aceitar TCP: 1,24s ± 0,22s (5 execuções). Até 1ª resposta HTTP 200: 1,54s ± 0,61s. **Limitação declarada**: mede o shell estático, não o render completo pós-websocket (exigiria Playwright — deliberadamente não adicionado só para esta medição). |
| Uso de memória com dataset grande | HSI sintético (64×64×224, 60 gravações, ~210MB de dado bruto — 64×64 é a resolução REAL do dataset público DeepHS Kaki/VIS): **pico de 1,27GB de RSS**, ~6× o tamanho do dado bruto. Tabular sintético na escala do maior dataset público integrado (Mendeley NIR 8mm, n=100×11500 canais, também a ordem de grandeza do dataset privado do autor por achado #12): **pico de 1,04GB de RSS** — ~37-110× o tamanho do dado bruto (a ordem de grandeza de p²×8 bytes = 1,06GB para p=11500 é suspeita e aponta para uma estrutura O(p²) em algum ponto do pipeline; não isolada nesta rodada, ver candidato a investigação abaixo). GC-IMS bruto: **N/A, sem leitor implementado** (Grupo 1). |
| Peso total de dependências | Ambiente limpo `pip install -e .[dev]`: **28 pacotes, 415MB**. `pip install -e .[all]`: **79 pacotes, 877MB** — top 5 em disco: llvmlite 119MB (via `shap`, extra `benchmark`), scipy 118MB (core), pyarrow 90MB (via `streamlit`, extra `web`), pandas 69MB (core), xgboost 58MB (extra `benchmark`). Todas as dependências pesadas e de uso raro (shap/xgboost/tensorly/brukeropus/prcv/scikit-image/streamlit) **já estão corretamente isoladas como extras opcionais** em `pyproject.toml` e confirmadas ausentes do import padrão do CLI — nenhum candidato a lazy-import pendente. |
| Custo da CV aninhada (`selecao_lv_cv_aninhada`, default `True` desde v1.0) | Com parâmetros REAIS (`n_splits_cv=5, n_repeats_cv=3, max_lvs=40`, escala Mendeley NIR 8mm): **sem** CV aninhada 271,7s, **com** 979,6s → **razão 3,61×**, compatível com a estimativa de "~4-5×" já documentada em `config.py`. Ambos os tempos absolutos (4,5min e 16,3min) estão MUITO acima do limite de atenção de Nielsen (10s). **Achado de UX confirmado**: não há NENHUMA indicação de progresso dentro do laço de CV aninhada ([pipeline.py:1911-1948](../src/guaraci/pipeline.py)) — só um log antes ("CV aninhada ativada...") e um depois; até 16 minutos de silêncio total. *(Nota de retratação: a 1ª rodada, com parâmetros reduzidos p/ velocidade, mediu razão ~1× e contradizia a estimativa do código — descartada e substituída por esta, documentada no próprio script.)* |
| Código morto / import não usado / função duplicada | `vulture` sobre `src/guaraci/` + `app_quimiometria.py`: 89 candidatos brutos → **86 falsos positivos confirmados por busca de uso real** (campos de dataclass/NamedTuple que são API pública de resultado, `render()` chamado só de `app_quimiometria.py` fora do escopo do scan, atributos write-only de terceiro — python-docx/openpyxl/fpdf2 —, callbacks `header`/`footer` do FPDF, contrato de interface do sklearn `get_n_splits`, campo `dwLength` de struct ctypes do Windows, atributos `_train_`/`T_` estilo sklearn) → **3 confirmados mortos e removidos** com suíte completa antes/depois (1533 passados, 0 regressão): `_ler_citation()` órfã em `guaraci.py`, `_tem_imagem_direta_ou_em_subpasta()` e `_subpasta_e_grupo_de_amostras()` órfãs em `dados_imagem.py`. Duplicação: varredura leve (blocos ≥8 linhas) em `guaraci.py`/`figuras.py`/`pipeline.py` achou **1 duplicação real de ~30 linhas** (carregar dados → validar → `run_audit` → renderizar achados → painel de resumo) entre `_guaraci_diagnosticar` ([guaraci.py:1128](../src/guaraci/guaraci.py)) e o menu de Auditoria de Delineamento ([guaraci.py:4113](../src/guaraci/guaraci.py)) — não consolidada nesta rodada (risco de comportamento, fora do escopo de uma medição). |
| Cobertura de novos módulos no gate mypy | `ruff check .` e os 7 módulos do gate mypy: **limpos** (ruff pegou 3 erros nos scripts novos desta própria rodada, corrigidos). |
| Tamanho do pacote publicável | `python -m build`: wheel 615KB comprimido / 1,83MB descomprimido (96 arquivos, só `guaraci/` + dist-info — **limpo**, nada de teste/scratch/dataset). sdist 944KB (226 arquivos: `src/`+`tests/`, decisão legítima de incluir testes no sdist — **também limpo**, sem vazamento). |
| Teste de mutação no núcleo científico crítico | `cosmic-ray` (mutmut não roda nativo no Windows). **conformal.py**: 230 mutantes, 57 sobreviventes (24,8%) — achado mais preocupante: em `ConformalOneClass`, o valor-padrão do fallback `.get("alcancavel", False)` pode virar `True` sem nenhum teste notar (inverte a semântica fail-safe→fail-open quando a chave está ausente, caminho real de código, não hipotético). **classificadores.py**: 1064 mutantes, 392 sobreviventes (36,8%) — achados mais preocupantes: as guardas de convergência/degenerescência do NIPALS PLS1 e do ajuste OPLS-DA (`if nt < 1e-12`, `if nto < 1e-12`, critério de tolerância) sobrevivem a inversão por `not` sem nenhum teste falhar — a lógica de parada por degenerescência numérica do algoritmo central está descoberta. **chemometric_stats.py**: 2527 mutantes — **medição parcial, interrompida em 528/2527 (20,9%) por custo de tempo** (suíte relevante ~18s/mutante × 2527 ≈ várias horas); 149 sobreviventes (28,2%) nos 528 rodados. Retomável (`cosmic-ray exec` reaproveita o progresso salvo em `cr_chemometric.sqlite`), não incluída no total abaixo. Taxa zero não era esperada em nenhum dos três — confirmado. |

**Achado colateral (não investigado nesta rodada, registrado para não se perder):** `scripts/download_datasets/baixar_deephs_fruit_todas.py` depende de um sidecar `_deephs_fruit_todas_pins.json` que **não existe no repositório** — o script quebra com `RuntimeError` antes de baixar qualquer coisa. Por isso a medição de memória do HSI usou cubo sintético em vez do dataset público real.

**Candidato a investigação futura (não confirmado):** o pico de RSS do caso tabular (1,04GB para ~9-28MB de dado bruto em p=11500 variáveis) tem ordem de grandeza compatível com uma matriz p×p (11500² × 8 bytes ≈ 1,06GB) formada em algum ponto do pipeline — não isolado nesta rodada; profiling por `cProfile` aponta `chemometric_stats.training_applicability_domain`/`q_residuals_loo` como a etapa dominante em tempo (~60% do total), candidata natural a investigar primeiro.

**Scripts versionados desta medição:** `scripts/medicoes/medir_tempo_inicializacao.py`, `scripts/medicoes/medir_uso_memoria.py`, `scripts/medicoes/medir_custo_cv_aninhada.py`.

---

## GRUPO 4 — Testes que faltam

| Item | Estado |
|---|---|
| Teste de mutação (mutation testing) | Nunca rodado — cobertura de linha (85%) não mede se o teste pegaria um bug real |
| `@example` determinístico nas técnicas mais recentes (MCR-ALS, EPO/GLSW, ASCA, T1-T9) | Auditoria de Hypothesis só cobriu os testes existentes antes daquela rodada específica |
| Teste de carga/memória com dataset genuinamente grande | Não existe |
| Teste de usabilidade real (pessoa de fora usando às cegas) | Nunca feito — nenhuma auditoria de agente substitui isso |
| Cobertura de novos módulos no gate mypy | Verificar se todo módulo novo desde a última auditoria está no gate |

---

## GRUPO 5 — Diferenciação competitiva real

**Já à frente de concorrente estabelecido, com prova:**
- Predição conforme (regressão e one-class) com cobertura declarada.
- Validação group-aware por padrão, incluindo holdout.
- Portão de aceite que rejeita automaticamente correção de sinal sem prova de ganho.
- Honestidade de cobertura (declara "não validado" em vez de forçar número).

**Ainda atrás:**
- Variedade de leitor de formato (~5 vs. ~35 do PLS_Toolbox) — ver Grupo 1.
- Exportação portátil de modelo — ver Grupo 2.
- Fusão multibloco e monitoramento em linha — ver Grupo 2.
- Maturidade de suporte/treinamento — fato estrutural, não item de engenharia.

---

## O QUE É EXCLUSÃO DELIBERADA, NÃO LACUNA

Deep learning geral, fusão multimodal dependente de sensor caro fora de alcance, LIMS empresarial completo, conformidade regulatória formal (nunca alegada, por decisão consciente), transferência de calibração sem padrão físico de referência. Todos descartados com razão registrada em rodadas anteriores.

---

## ORDEM DE IMPLEMENTAÇÃO PROPOSTA

1. ~~Grupo 3 primeiro (medição de performance)~~ — **MEDIDO em 2026-09-11** (ver tabela acima). Pendências residuais dentro do próprio grupo: teste de mutação de `chemometric_stats.py` incompleto (528/2527, retomável); decisão sobre os 2 achados de UX (sleep de 1s no boot do CLI, ausência de progresso na CV aninhada) e sobre o achado de possível O(p²) no caso tabular — nenhum corrigido ainda, aguardando decisão do autor (risco de mudar comportamento testado).
2. Grupo 1 (leitores de formato) — SPC e `.sp` primeiro; RMN/GC-IMS/EEM brutos depois.
3. Grupo 4 (testes) — em paralelo aos Grupos 1 e 2; completar a rodada de mutação de `chemometric_stats.py` iniciada no Grupo 3.
4. Grupo 2 (análises) — conjunto de predição multiclasse primeiro; fusão multibloco e MSPC depois.
5. Grupo 5 — consequência natural de fechar 1, 2 e 3; revisar o comparativo do README ao final.

---

## CRITÉRIO DE PUBLICAÇÃO

O projeto está pronto para publicar quando todos os grupos acima estiverem numa de duas condições: implementado com contra-prova real, ou excluído com razão registrada e explícita. Nenhum item pode permanecer ambíguo no momento da publicação. Este documento é o checklist final — revisar linha por linha antes de qualquer criação de repositório novo ou tag pública.
