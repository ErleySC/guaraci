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
| Tempo de inicialização (CLI) | Média 3,89s ± 0,11s (10 execuções cold start) — tolerável p/ Nielsen. **Corrigido**: o `time.sleep(1.0)` fixo na mensagem de boas-vindas ([guaraci.py:966](../src/guaraci/guaraci.py)) não tinha motivo funcional (`git log -S`: presente desde o commit original v31.0.0, nunca documentado) — reduzido para 0,4s. Import (`-X importtime`) responde por ~2,0-2,4s (sklearn ~1,0s, scipy ~0,5s, pandas ~0,4s, matplotlib ~0,3s); nenhuma dependência pesada opcional (streamlit/tensorly/brukeropus/shap/xgboost) é importada no boot — já corretamente isolada. |
| Tempo de inicialização (Streamlit) | Boot até aceitar TCP: 1,24s ± 0,22s (5 execuções). Até 1ª resposta HTTP 200: 1,54s ± 0,61s. **Limitação declarada**: mede o shell estático, não o render completo pós-websocket (exigiria Playwright — deliberadamente não adicionado só para esta medição). |
| Uso de memória com dataset grande | HSI sintético (64×64×224, 60 gravações, ~210MB de dado bruto — 64×64 é a resolução REAL do dataset público DeepHS Kaki/VIS): **pico de 1,27GB de RSS**, ~6× o tamanho do dado bruto (não investigado a fundo — ordem de grandeza menor, prioridade mais baixa). Tabular sintético na escala do maior dataset público integrado (Mendeley NIR 8mm, n=100×11500 canais): **pico de 1,04GB de RSS** via `psutil` (RSS do processo). **Investigado com `tracemalloc` de granularidade de linha (Passo 216) — hipótese de matriz p×p REFUTADA**: o pico rastreado pelo tracemalloc foi de só **410,7MB**, menos da metade do RSS total, e nenhuma alocação isolada relevante aparece nesse valor — as maiores são cópias `X_raw[idx]` inteiramente ordinárias do split de holdout (~26MB e ~22MB, [pipeline.py:1688](../src/guaraci/pipeline.py)/[:1717](../src/guaraci/pipeline.py), lineares em n×p, não quadráticas), renderização de figuras matplotlib (~63MB+42MB+21MB) e maquinário de import (~57MB+12MB). Os ~630MB que faltam para bater com o RSS não são visíveis ao tracemalloc — mais consistente com memória de trabalho de BLAS/LAPACK (SVD para PCA/PLS num p=11500 "largo") e retenção normal de páginas pelo alocador do SO do que com um vazamento identificável. **Classificado como arquitetura aceitável para datasets largos (p≫n), não um bug — nenhuma correção aplicada.** GC-IMS bruto: **N/A, sem leitor implementado** (Grupo 1). |
| Peso total de dependências | Ambiente limpo `pip install -e .[dev]`: **28 pacotes, 415MB**. `pip install -e .[all]`: **79 pacotes, 877MB** — top 5 em disco: llvmlite 119MB (via `shap`, extra `benchmark`), scipy 118MB (core), pyarrow 90MB (via `streamlit`, extra `web`), pandas 69MB (core), xgboost 58MB (extra `benchmark`). Todas as dependências pesadas e de uso raro (shap/xgboost/tensorly/brukeropus/prcv/scikit-image/streamlit) **já estão corretamente isoladas como extras opcionais** em `pyproject.toml` e confirmadas ausentes do import padrão do CLI — nenhum candidato a lazy-import pendente. |
| Custo da CV aninhada (`selecao_lv_cv_aninhada`, default `True` desde v1.0) | Com parâmetros REAIS (`n_splits_cv=5, n_repeats_cv=3, max_lvs=40`, escala Mendeley NIR 8mm): **sem** CV aninhada 271,7s, **com** 979,6s → **razão 3,61×**, compatível com a estimativa de "~4-5×" já documentada em `config.py`. Ambos os tempos absolutos (4,5min e 16,3min) estão MUITO acima do limite de atenção de Nielsen (10s). **Corrigido**: laço de CV aninhada ([pipeline.py:1911-1957](../src/guaraci/pipeline.py)) não emitia NENHUM log entre início e fim (até 16min de silêncio) — agora emite `[2b/7] CV aninhada: fold externo K/N concluído` a cada fold externo, e `app_logic.log_progress()` (usado tanto pelo painel do CLI quanto pela barra do app web — mesma função, paridade automática) reconhece o novo marcador e avança a fração suavemente, mesmo padrão já usado para a etapa `[6/7]` (achado de 2026-08-07). 4 testes novos travam o comportamento (nunca regride, 1 atualização por fold, não afeta outras etapas). *(Nota de retratação: a 1ª rodada da medição de tempo, com parâmetros reduzidos p/ velocidade, tinha medido razão ~1× e contradizia a estimativa do código — descartada e substituída pela acima, documentada no próprio script.)* |
| Código morto / import não usado / função duplicada | `vulture` sobre `src/guaraci/` + `app_quimiometria.py`: 89 candidatos brutos → **86 falsos positivos confirmados por busca de uso real** (campos de dataclass/NamedTuple que são API pública de resultado, `render()` chamado só de `app_quimiometria.py` fora do escopo do scan, atributos write-only de terceiro — python-docx/openpyxl/fpdf2 —, callbacks `header`/`footer` do FPDF, contrato de interface do sklearn `get_n_splits`, campo `dwLength` de struct ctypes do Windows, atributos `_train_`/`T_` estilo sklearn) → **3 confirmados mortos e removidos** com suíte completa antes/depois (1533 passados, 0 regressão): `_ler_citation()` órfã em `guaraci.py`, `_tem_imagem_direta_ou_em_subpasta()` e `_subpasta_e_grupo_de_amostras()` órfãs em `dados_imagem.py`. **Duplicação registrada, não consolidada** (risco de comportamento, fora do escopo desta rodada): ~30 linhas (carregar dados → validar → `run_audit` → renderizar achados → painel de resumo) entre `_guaraci_diagnosticar` ([guaraci.py:1128](../src/guaraci/guaraci.py)) e o menu de Auditoria de Delineamento ([guaraci.py:4113](../src/guaraci/guaraci.py)). |
| Cobertura de novos módulos no gate mypy | `ruff check .` e os 7 módulos do gate mypy: **limpos** a cada lote de correção desta rodada. |
| Tamanho do pacote publicável | `python -m build`: wheel 615KB comprimido / 1,83MB descomprimido (96 arquivos, só `guaraci/` + dist-info — **limpo**, nada de teste/scratch/dataset). sdist 944KB (226 arquivos: `src/`+`tests/`, decisão legítima de incluir testes no sdist — **também limpo**, sem vazamento). |
| Teste de mutação no núcleo científico crítico | `cosmic-ray` (mutmut não roda nativo no Windows). **conformal.py — FECHADO**: 230 mutantes, 57→**54 sobreviventes (23,5%)** após correção. O furo crítico (`ConformalOneClass`: default do fallback `.get("alcancavel", False)` podia virar `True` sem nenhum teste notar — fail-safe→fail-open com `info_` incompleto, cenário real de modelo `.joblib` legado) e um segundo furo irmão (aviso de não-alcançável em `fit()` podia ter o `not` removido sem quebrar nada) foram fechados com 2 testes novos, confirmados mortos 100% via cosmic-ray local. **classificadores.py — os 2 furos priorizados FECHADOS**: 1064 mutantes, 392→404 sobreviventes agregado (aumento pequeno entre rodadas sem nenhuma mudança de código-fonte entre elas — mais consistente com ruído de timeout/ambiente do que regressão real, não perseguido a fundo dado o custo). O ternário de componentes ortogonais (`fit()`, X de posto 1 — comum quando `n_ortho` pede mais do que o dado sustenta) e o `isinstance()` do retorno de `PLSRegression.transform()` foram fechados com 1 teste novo, ambos confirmados mortos 100%. 3 outros sobreviventes (guardas `nt<1e-12`/tolerância de convergência em `_nipals_pls1`, `nto<1e-12` em `fit()`) foram investigados a fundo — **muito provavelmente mutantes equivalentes** para PLS1 de 1 y (prova: `u_new` é sempre um múltiplo escalar POSITIVO de `y`, então `w` converge em direção exata na 2ª iteração sempre — confirmado empiricamente em 10 casos aleatórios de formas variadas, sempre exatamente 2 iterações) — documentado no teste em vez de forçado com dado artificial; o `isinstance()` de uso não coberto é dead code confirmado por inspeção do source do sklearn (`transform(X, y=None)` sempre devolve ndarray puro em toda a faixa suportada `>=1.3`). **chemometric_stats.py**: relançado como job **isolado** (2527 mutantes) após o achado de corrupção de árvore de trabalho por concorrência (ver nota) — em andamento, não bloqueia o restante; reportar taxa de sobrevivência e mutantes mais preocupantes quando concluir (pode ser em sessão futura). |

**Achado metodológico desta rodada (Passo 216):** rodar `cosmic-ray` em paralelo com a suíte de verificação de OUTRA correção, ou matar (`TaskStop`) um `cosmic-ray exec` no meio de um mutante, CORROMPE a árvore de trabalho (o mutante fica escrito em disco) e pode fazer testes completamente não-relacionados falharem por contaminação, não por regressão real — aconteceu 2× nesta sessão, ambos detectados (`git diff`/re-execução isolada) antes de qualquer conclusão errada ser reportada. Regra adotada: nunca rodar suíte de verificação nem outro `cosmic-ray` sobre o MESMO arquivo enquanto um `cosmic-ray exec` estiver ativo nele; ao interromper, sempre `git status`/`git diff` no arquivo-alvo antes de continuar.

**Achado colateral (não investigado nesta rodada, registrado para não se perder):** `scripts/download_datasets/baixar_deephs_fruit_todas.py` depende de um sidecar `_deephs_fruit_todas_pins.json` que **não existe no repositório** — o script quebra com `RuntimeError` antes de baixar qualquer coisa. Por isso a medição de memória do HSI usou cubo sintético em vez do dataset público real.

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

1. ~~Grupo 3 primeiro (medição de performance)~~ — **MEDIDO e as pendências de maior prioridade FECHADAS em 2026-09-11/12** (ver tabela acima): furos críticos de mutation testing em `conformal.py` e `classificadores.py` corrigidos com contra-prova; hipótese de memória O(p²) investigada e refutada; progresso na CV aninhada implementado (CLI + app web); sleep artificial reduzido. Única pendência residual: teste de mutação de `chemometric_stats.py` rodando isolado em background (2527 mutantes, horas) — não bloqueia os próximos grupos.
2. Grupo 1 (leitores de formato) — SPC e `.sp` primeiro; RMN/GC-IMS/EEM brutos depois.
3. Grupo 4 (testes) — em paralelo aos Grupos 1 e 2; reportar `chemometric_stats.py` quando a rodada isolada do Grupo 3 concluir.
4. Grupo 2 (análises) — conjunto de predição multiclasse primeiro; fusão multibloco e MSPC depois.
5. Grupo 5 — consequência natural de fechar 1, 2 e 3; revisar o comparativo do README ao final.

---

## CRITÉRIO DE PUBLICAÇÃO

O projeto está pronto para publicar quando todos os grupos acima estiverem numa de duas condições: implementado com contra-prova real, ou excluído com razão registrada e explícita. Nenhum item pode permanecer ambíguo no momento da publicação. Este documento é o checklist final — revisar linha por linha antes de qualquer criação de repositório novo ou tag pública.
