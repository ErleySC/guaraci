# Auditoria de UX e consistência — 2026-09-24

Base: `INSTRUCAO_REFORMULACAO_UX.md` (Parte H — auditoria antes de qualquer
mudança) + anotações do autor (`auterações guaraci.docx`, 10 capturas).
Commit auditado: `6a5901a`. **Nada foi alterado no código**; este documento
só registra achados com a evidência de cada um.

Legenda de evidência: **[exec]** reproduzido executando código;
**[leitura]** confirmado lendo o código-fonte (linha citada).

---

## 1. Bugs confirmados (corrigir antes da reformulação)

> **Status (2026-09-24, mesmo dia):** B1–B11 **corrigidos**, cada um com
> teste de regressão: `tests/test_auditoria_ux_2026_09_24.py` (B2, B4–B10),
> `tests/test_menu_hsi.py` (B1), `tests/test_selecao_amostras.py` (B3, B11).
> Os testes novos foram rodados contra o código antigo e falham lá.
> Detalhe por item no `docs/CHANGELOG.md` → "Não lançado".
> As seções 2 a 6 abaixo continuam válidas: são a reformulação da
> instrução, que depende das pausas (a), (b) e (c).

| # | Bug | Evidência | Efeito para o usuário |
|---|---|---|---|
| B1 | Abrir **[X] HSI** grava `cfg.mode = "hsi"` *antes* de pedir a pasta. Sair com `[0]` não desfaz. | [leitura] `guaraci.py:3577` · [exec] `get_reader('hsi')` → `ValueError: Modo de entrada desconhecido: 'hsi'` | Depois de só *olhar* a tela HSI: o cabeçalho de todas as telas passa a dizer "Técnica: HSI" (capturas 6–10), a caixa de execução diz "✔ Pronto para executar" e o **[R] Rodar Pipeline falha**, porque não existe leitor de dados para o modo `hsi`. É a causa real da captura 9 (R7). |
| B2 | Rodapé anuncia `[?] Ajuda do campo`, mas a tecla **não faz nada** em 4 de 5 telas de configuração. | [exec] teste com entrada simulada: `?` abre ajuda só em `_menu_modeling`; em `_menu_preprocessing`, `_menu_validation`, `_menu_advanced` e `_menu_visualization` cai em "inválido". Causa: só `_loop_menu` (`guaraci.py:2362`) trata `?`; as outras 4 telas têm loop próprio. | Promessa visível que não funciona: [3], [5], [6], [7]. |
| B3 | **[K] Seleção de Amostras** usa as versões de Kennard-Stone/Duplex/SPXY **que não respeitam grupo**. | [leitura] `guaraci.py:4079-4084` chama `kennard_stone_split`/`duplex_split`/`spxy_split`; as variantes `*_group_aware` existem em `dados_io.py:421/557/630` e a tela nem pergunta pela coluna de grupo. | Réplicas da mesma amostra física (ex.: T1/T2/T3) podem cair uma na calibração e outra na validação: o vazamento que o GUARACI promete evitar. O `MANUAL.md` §2.2c afirma que "nenhum dos três separa réplica física", o que é verdade para a biblioteca, mas **não para a tela [K]**. |
| B4 | **[7] › [D] Grade › [4] Transparência** aceita qualquer número, sem checar a faixa. | [leitura] `guaraci.py:2675` · [exec] `grid.alpha=5.0` → `ValueError: alpha (5.0) is outside 0-1 range` ao criar qualquer figura | O valor fica gravado em `~/.guaraci`, então **todas as execuções seguintes** perdem as figuras até alguém editar o arquivo à mão. |
| B5 | **[7] › [A] Transparência dos Pontos** não tem efeito nenhum, e a própria nota no código admite isso. | [leitura] `guaraci.py:5350-5356` | Opção visível que não faz nada. Precisa ser ligada de verdade (21 chamadas `ax.scatter` em `figuras.py`) ou escondida. |
| B6 | Perfil **"Acessibilidade"** sobrescreve 11 campos de **análise** (max_lvs, DD-SIMCA, OPLS-DA, seleção de variáveis…), não só as cores. | [leitura] `cli_assistente.py:1311-1319` | Quem estava em Quantificar (N3) e escolhe Acessibilidade só pelas cores fica com DD-SIMCA ligado e LVs trocados, sem aviso. Confirma o R4 da instrução. |
| B7 | Perfis de **rigor** não definem o nível e não chamam `_ajustar_toggles_por_nivel`. | [leitura] `guaraci.py:4525-4546` (compare com `guaraci.py:2263`, onde a edição manual chama o ajuste) | "Alta Rigorosidade ~3-6 h" aplicado em N3 liga Benchmark/Monte Carlo/SHAP, que ficam **inertes** em Quantificação: o tempo anunciado não se aplica e o menu mostra opções "Sim" que não rodam. |
| B8 | Depois de aplicar um perfil, "Rodar agora? **[S/n]**": o Enter (resposta padrão) **inicia a execução**. | [leitura] `guaraci.py:4625-4629` | Um Enter distraído dispara uma execução de até 6 h. Padrão seguro: `[s/N]`. |
| B9 | **[U] Auditoria**: `[G]` abre o assistente e depois **sai da tela** em vez de voltar a ela. | [leitura] `guaraci.py:4449-4451` | Inconsistente com [J]/[K]/[B], que redesenham e continuam. |
| B10 | [3] Pré-processamento e [6] Métodos Avançados: Enter vazio = "inválido" + pausa; nas outras telas Enter = voltar. | [leitura] `guaraci.py:2474`, `2568` | Pequeno, mas é exatamente a inconsistência que o autor sente. |
| B11 | [K] lê o CSV com `pd.read_csv` padrão (separador `,`). | [leitura] `guaraci.py:4028` | CSV exportado do Excel em PT-BR (`;` e vírgula decimal, o mesmo formato que o próprio GUARACI grava em `[B]`) vira 1 coluna, e a mensagem diz "CSV sem colunas numéricas", sem dizer o porquê. `[9]` já detecta `;` vs `,` (`guaraci.py:2934`); reaproveitar. |

---

## 2. Tabela de auditoria — CLI, tela por tela

Colunas: **Rodapé** = mostra `? G I 0` · **[?] ok** = a tecla funciona ·
**Ex/padrão** = entradas com exemplo e valor padrão · **Valid.** = checa a
entrada antes do próximo passo · **Web** = existe equivalente no app web.

| Tecla · Tela | Tipo | Rodapé | [?] ok | Ex/padrão | Valid. | Web | Observações |
|---|---|---|---|---|---|---|---|
| Menu principal | menu | ✔ | ✔ | — | — | ✔ | Pares confusos: [6] "Métodos Avançados" × [T] "Técnicas Avançadas"; [8] em Modelar; HSI em Prever. `[M] Modo` só esconde campos dentro das telas, não itens do menu. |
| [1] Projeto | config (`_loop_menu`) | ✔ | ✔ | parcial | ✔ | ✔ | Modelo a copiar. |
| [2] Dados | config (`_loop_menu`) | ✔ | ✔ | parcial | ✔ | ✔ | `perfil_tecnica` sem entrada no `_HELP_DB` (cai no texto curto). |
| [3] Pré-processamento | config (loop próprio) | ✔ | **✘ (B2)** | parcial | ✔ | ✔ | Enter = inválido (B10). |
| [4] Modelagem | config (`_loop_menu`) | ✔ | ✔ | parcial | ✔ | ✔ | Itens 9–12 com enchimento "Além dos métodos acima…" (4 campos: `selecao_spa/ag/cars/uve`). |
| [5] Validação | config (loop próprio) | ✔ | **✘ (B2)** | parcial | ✔ | parcial | `n_jobs_permutacao` sem entrada no `_HELP_DB`. |
| [6] Métodos Avançados | config (loop próprio) | ✔ | **✘ (B2)** | parcial | ✔ | ✔ | Enter = inválido (B10). |
| [7] Visualização | config + 4 subtelas | ✔ | **✘ (B2)** | ✘ nas subtelas | **✘ (B4)** | ✔ | [A] sem efeito (B5); subtelas P/F/D/A sem ajuda nem idioma. |
| [8] Técnica Analítica | lista | ✔ | ✔ (detalhe) | — | ✔ | parcial | Pertence a *Preparar*. Sem aviso de compatibilidade técnica × dado (R7). |
| [9] Codificação DX | ações | parcial (sem `?`) | ✘ | ✔ | ✔ | ✘ | Lista fixa de 13 espécies de óleo embutida no código: domínio do TCC, não da plataforma. |
| [P] Perfis Prontos | lista | ✔ | ✔ (detalhe) | — | — | parcial (`dados.py`) | B6, B7, B8. Tempos fixos, não medidos. |
| [J] Planejamento de Coleta | assistente | **só G/0** | ✘ | **✘** (classes sem exemplo; alpha/cobertura sem explicar) | parcial | **✘** | Jargão solto: "gate conformal", "DD-SIMCA", "réplica/branco". |
| [K] Seleção de Amostras | assistente | **só G/0** | ✘ | parcial | parcial | **✘** | B3 (grupo), B11 (separador). Não diz *quando* usar KS × Duplex × SPXY. |
| [U] Auditoria de Delineamento | assistente | **só G/0** | ✘ | — | ✔ | ✔ (`projeto.py`) | B9. Texto cita "[2] Dados", uma tecla embutida no texto. |
| [T] Técnicas Avançadas | menu + 4 assistentes | **só 0** | ✘ | parcial | parcial | ✔ | Mostra nomes de variável ao usuário (`grupo_interesse`, `fator_incomodo`), "0-indexado", "Mendeley", "PoC (Grupo 2)", "ver docstring". Modo de entrada da fusão é texto livre, sem validar. |
| [B] Predição em Lote | assistente | **só G/0** | ✘ | parcial | ✔ | ✔ | Bloco de resultado com ~25 linhas de texto expositivo e referências internas ("Bloco 9b", "Bloco 12"). |
| [X] HSI | assistente | **só 0** | ✘ | ✘ | parcial | **✘** | B1. Texto de auditoria na tela (DeepHS, Kaki, VIS, `docs/VALIDACAO_PUBLICA.md seção 7`), o que é R9. Extras [D]/[R]/[M] sem explicação. |
| [H] Hardware | painel | só G/0 | ✘ | — | — | parcial | Recomendações citam "[6]" no texto (quebram se as teclas mudarem). |
| [G] Assistente | menu | só Q | — | — | — | parcial | Não lê o registro de técnicas nem o vault; texto próprio. |
| [?] Ajuda | lista | ✔ | ✔ | — | — | ✘ | Cobre só os **49 campos** do `_CONFIG_SPEC`, em uma tabela única sem paginação. **Não cobre** telas (J/K/U/T/B/X/P/9/H), técnicas nem perfis. |
| [A] Sobre | painel | — | — | — | — | ✔ | — |
| [R] Checklist/execução | ação | — | — | — | ✔ | ✔ | Status "Pronto" calculado pela contagem de `.dx`, independente do modo (ver B1). |

**Números medidos** [exec]:
- 46 de 49 descrições de campo passam de 44 caracteres e **aparecem cortadas com "…"** na lista (R8).
- 2 campos sem entrada de ajuda própria: `perfil_tecnica`, `n_jobs_permutacao`.
- Só **3 telas** ([1], [2], [4]) usam o loop genérico `_loop_menu`; todas as outras têm loop e rodapé próprios, escritos à mão. Não existe molde único (R2).

---

## 3. Técnicas sem caminho de uso (pedido do autor: "nenhuma técnica sem que ela seja utilizada")

| Técnica | Estado | Evidência |
|---|---|---|
| **Transferência de calibração (DS/PDS)** | Implementada, validada no Corn, registrada no `technique_registry` (#11, #12). **Nenhuma tela CLI ou web a executa.** | [leitura] `pipeline.py:828` só *importa*; nenhuma chamada. A Parte C da instrução propõe o objetivo "Transferir entre instrumentos", que hoje não tem motor ligado. |
| Amostragem ativa | Só CLI (dentro de [J]); sem web | `grep` nas `app_tabs`: 0 ocorrências |
| Planejamento de coleta, Seleção de amostras, HSI, Codificação | Só CLI | idem |

Para cumprir a regra "toda técnica do registro tem caminho de execução", DS/PDS precisa de uma tela antes do teste de cobertura da Parte B. Sem ela, o teste já nasceria falhando.

---

## 4. Documentação desatualizada ou inconsistente

| Documento | Problema |
|---|---|
| `docs/MANUAL.md` §2.2c | Diz que a seleção de amostras não separa réplicas; a tela [K] separa (B3). O rodapé do manual afirma "verificado contra o código, seção por seção". |
| `docs/MANUAL.md` (estrutura) | Numeração quebrada: dentro do capítulo **5** as subseções se chamam **2.2b, 2.2c, 2.3 … 2.7**. Títulos com jargão interno ("Passo 86", "Bloco 10", "Grupo 2"), o mesmo R9 da tela, agora no manual. O §9 "Limitações conhecidas" tem ~300 linhas. |
| Vault Obsidian | Gerado em `59b083c`; o repositório está em `6a5901a`. `consultar_vault.py --cobertura` → **"Cobertura INCOMPLETA"** (1 commit atrasado). O vault cobre módulos e técnicas, **não telas nem campos**, e a Parte B vai exigir isso. |
| Tela HSI / `_AVISO_MATURIDADE_HSI_*` | Texto de auditoria destinado ao usuário final (R9). Mover o detalhe para `[?]` e deixar só o selo de maturidade (Parte F). |
| Texto de telas com teclas embutidas | "[2] Dados" (U, T), "Desativar em [6]" (H). Tudo quebra na Parte E se as teclas mudarem. Gerar o texto a partir do mapa de teclas, não escrevê-lo à mão. |
| `mypy` local | Quebrado: `tifffile` usa sintaxe do Py 3.12 e o `pyproject.toml:135` fixa `python_version = "3.10"`. A regra 3 da instrução ("mypy limpo por lote") não é verificável localmente até resolver isso. `ruff`: limpo. |

---

## 5. Correções à própria instrução (o que ela pressupõe e não bate com o código)

1. **Parte G, NO_COLOR já funciona.** O Rich respeita `NO_COLOR` sozinho ([exec] `NO_COLOR=1` → `console.no_color=True`). Falta só um teste que trave esse comportamento. Problema real: `force_terminal=True` (`guaraci_theme.py:94`) emite códigos de cor mesmo quando a saída é redirecionada para arquivo ou leitor de tela.
2. **Parte C, "tempo estimado real, medido".** Não existe infraestrutura de medição; os tempos dos perfis são texto fixo. Precisa de um benchmark por combinação objetivo × rigor com dado sintético, que a própria Parte C já pede como teste e-2-e: aproveitar o mesmo teste para medir.
3. **Parte C, objetivo "Transferir entre instrumentos"** depende de criar a tela DS/PDS (seção 3).
4. **Parte E, "usar o [M] Modo já existente"**: hoje o modo Iniciante esconde **campos**, não **itens do menu principal**. O Modo Básico da instrução é uma funcionalidade nova, não um ajuste.
5. **Parte A, teste de contrato "por introspecção de todas as telas registradas"**: não existe registro de telas; o despacho é uma cadeia de `if/elif` em `main()` (`guaraci.py` ~5980). O registro precisa ser criado primeiro, porque o teste depende dele.

---

## 6. Ordem recomendada

1. **Bugs B1–B11** (baixo risco, pequenos, sem pausa obrigatória). B1, B3 e B4 primeiro: são os que produzem resultado errado ou quebram execução.
2. **Registro de telas + molde único** (Parte A): resolve B2/B9/B10 de vez, em vez de remendar 4 loops.
3. **Registro de ajuda** (Parte B), incluindo telas e campos.
4. **Textos e selos** (Parte F).
5. **Tela DS/PDS**, depois Perfis objetivo × rigor (Parte C), com **pausa (b)** antes de mexer no formato salvo.
6. **Mapa de teclas novo** (Parte E), com **pausa (a)**.
7. Manual/vault regenerados a partir do registro (Partes I/J).
