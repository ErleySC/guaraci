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

## GRUPO 3 — Performance, leveza e limpeza (nunca medido até agora)

Esta é a lacuna que nenhuma instrução anterior tocou.

| Item | Estado |
|---|---|
| Tempo de inicialização (CLI) | Não medido |
| Tempo de inicialização (Streamlit) | Não medido |
| Uso de memória com dataset grande (cubo HSI, GC-IMS bruto) | Não medido |
| Peso total de dependências (contagem, tamanho de instalação) | Não medido — lista cresceu bastante (sklearn, scipy, streamlit, hypothesis, fpdf2, tensorly, brukeropus, netCDF4, matplotlib, pandas e mais) |
| Custo da CV aninhada (2,4-3,7× medido) vs. expectativa de UX aceitável | Medido em isolamento, nunca comparado contra tolerância real de espera do usuário |
| Código morto / import não usado / função duplicada | Nunca varrido — só documentação recebeu esse tipo de auditoria |
| Tamanho do pacote publicável (`pyproject.toml`, o que entra no build) | Confirmado uma vez, não revalidado desde então com todo o código novo |

**Ação sugerida:** este grupo precisa de medição antes de qualquer otimização — não presumir que está pesado ou leve sem medir.

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

1. Grupo 3 primeiro (medição de performance) — rápido, não muda comportamento, informa decisão nos outros grupos.
2. Grupo 1 (leitores de formato) — SPC e `.sp` primeiro; RMN/GC-IMS/EEM brutos depois.
3. Grupo 4 (testes) — em paralelo aos Grupos 1 e 2, mais uma rodada dedicada de teste de mutação no núcleo científico existente.
4. Grupo 2 (análises) — conjunto de predição multiclasse primeiro; fusão multibloco e MSPC depois.
5. Grupo 5 — consequência natural de fechar 1, 2 e 3; revisar o comparativo do README ao final.

---

## CRITÉRIO DE PUBLICAÇÃO

O projeto está pronto para publicar quando todos os grupos acima estiverem numa de duas condições: implementado com contra-prova real, ou excluído com razão registrada e explícita. Nenhum item pode permanecer ambíguo no momento da publicação. Este documento é o checklist final — revisar linha por linha antes de qualquer criação de repositório novo ou tag pública.
