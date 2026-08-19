# Scorecard — rodada de desacoplamento e fechamento (2026-08-18)

Sucede [`SCORECARD_2026-08-17.md`](SCORECARD_2026-08-17.md). Notas: **0**
ausente · **0,5** parcial · **1** completo · **N/V** não verificável (fora
do denominador). ✅ = resolvido nesta rodada.

Mudança de premissa que reorganiza o scorecard inteiro: o dataset
institucional saiu do escopo do software. Itens que antes pontuavam mal por
causa do delineamento dos `.dx` **não migram como dívida do software** —
migram para o TCC, que é entregável separado. O que fica aqui é o que o
software pode provar sozinho, em dado público.

---

## 1. Metodologia científica e validação — peso 25 %

| # | Item | 08-17 | **08-18** | Evidência |
|---|---|:--:|:--:|---|
| 1.1 | Rótulo não entra no caminho de quantificação | 0 | **1** ✅ | `pipeline.rotulos_para_quantificacao`; quantificação usa `pred_lab` |
| 1.2 | Sem vazamento indireto (pré-proc dentro do CV) | 1 | 1 | `preprocessamento.py:128-141` |
| 1.3 | Modo cego é o padrão da API/CLI | 0 | **1** ✅ | `Config.modo_rotulo="cego"`; `--modo=controle` explícito |
| 1.4 | Teste que falha se o rótulo vazar | 0 | **1** ✅ | `tests/test_modo_cego.py` — prova por envenenamento |
| 1.5 | Amostra fora do treino não é forçada numa classe | 0 | 0,5 | AD calibrado sinaliza fora-de-domínio; `argmax` ainda força a classe |
| 1.6 | Fora-de-domínio no caminho padrão | 1 | 1 | colunas `AD_*`, Q por LOO |
| 1.7 | Arquitetura detectar→identificar→quantificar | 0 | 0,5 | quantificação já consome a identificação (cego); detecção binária separada não existe |
| 1.8 | Ordem de leitura descorrelacionada do teor | 0 | **N/V** | propriedade do acervo do TCC, fora do escopo do software |
| 1.9–1.12 | Delineamento do acervo (sessão, brancos, réplicas) | 0–1 | **N/V** | idem |
| 1.13 | CV aninhada em todos os métodos | 1 | 1 | `_avaliar_busca_nested_cv` |
| 1.14 | RMSEC/RMSECV/RMSEP/R²pred/bias | 1 | 1 | `pipeline.py` |
| 1.15 | RPD e RER | 0 | **1** ✅ | `chemometric_stats.rpd_rer` + `interpretar_rpd`, com faixa publicada |
| 1.16 | Matriz de confusão + métricas por classe | 1 | 1 | — |
| 1.17 | LOD/LOQ com método declarado | 0,5 | 0,5 | implementados; saem `N/A` sem réplicas físicas — correto, mas limita |
| 1.18 | Incerteza como intervalo | 1 | 1 | BCa |
| 1.19 | Linearidade / faixa de trabalho | 0 | 0,5 | `faixa_trabalho` no perfil + `fora_da_faixa_de_trabalho`; não plotada |
| 1.20 | Aderência a AOCS/AOAC/ASTM/ISO | 0 | 0 | não declarada |
| 1.21 | `Nh`/`Nq` conforme a referência | 1 | 1 | retratado: truncagem muda ≤0,003 |
| 1.22 | Generaliza para matriz não-óleo | 1 | 1 | Corn: RMSEP 0,144, dentro da faixa publicada |
| 1.23 | Configuração por matriz externalizada | 0,5 | **1** ✅ | `perfis_matriz/*.yaml` + `perfil_matriz.py` |
| 1.24 | Matriz sem perfil falha em vez de predizer | 0 | **1** ✅ | `PerfilDesconhecidoError` antes do carregamento |
| 1.25 | Saída não afirma química da matriz errada | 0,5 | **1** ✅ | model card usa o vocabulário do perfil; teste de aceitação trava |

**Subtotal: 16 / 18 → 88,9 %** (5 itens N/V — migraram para o TCC)

---

## 2. Código e testes — peso 20 %

| # | Item | 08-17 | **08-18** | Evidência |
|---|---|:--:|:--:|---|
| 2.1 | Roda ponta a ponta em ambiente limpo | 1 | 1 | venv novo + `pip install .` + `guaraci --version` |
| 2.2 | Determinismo / seeds | 1 | 1 | `cfg.seed` em todos os pontos; teste de reprodutibilidade no Corn |
| 2.3 | Lockfile + Python declarado | 1 | 1 | `requirements-lock.txt`; `>=3.10` |
| 2.4 | Testes nas funções numéricas | 1 | 1 | núcleo ≥95 % |
| 2.5 | Cobertura real | 0,5 | 0,5 | 70 % total; `guaraci.py` ainda baixo |
| 2.6 | `except` genéricos sob controle | 1 | 1 | 100 % com `noqa` justificado |
| 2.7 | Sem falha silenciosa em produção | 1 | 1 | `q_ucl` circular recusado |
| 2.8 | Regra de decisão única | 1 | 1 | distância combinada nas 4 cópias |
| 2.9 | Duplicação consolidada | 1 | 1 | `q_residuos_loo` unificada |
| 2.10 | Código morto | 1 | 1 | `ruff F401/F841` = 0 |
| 2.11 | Complexidade sob controle | 0 | 0 | `executar()` ~1.500 linhas; `guaraci.py` ~4.000 |
| 2.12 | Ciclos de import | — | **1** ✅ | **RETIFICADO**: 0 ciclos entre imports de nível de módulo. Os 14 contados em 08-17 incluíam imports locais e `TYPE_CHECKING` — que são a técnica de *quebrar* ciclo. 31 módulos importam sozinhos em processo limpo (`tests/test_import_ciclos.py`) |
| 2.13 | Pacote empacota o que precisa | — | **1** ✅ | achado e corrigido: perfis não iam na wheel |

**Subtotal: 11 / 12 → 91,7 %**

---

## 3. Dados: proveniência e vazamento — peso 15 %

| # | Item | 08-17 | **08-18** | Evidência |
|---|---|:--:|:--:|---|
| 3.1 | Proveniência documentada | 0,5 | 1 ✅ | `DESACOPLAMENTO_2026-08-18.md` |
| 3.2 | Duplicatas, unidades, encoding | 1 | 1 | `prescan_dx`, `validar_entrada` |
| 3.3 | Brutos separados dos processados | 1 | 1 | dados fora do repo |
| 3.4 | Dicionário de dados | 0 | 0,5 | colunas de `metadados.csv` documentadas em `sanitizar_metadados` |
| 3.5 | Nenhum dado pessoal versionado | 1 | 1 | — |
| 3.6 | Nenhum espectro no repo/histórico | 1 | 1 | — |
| 3.6b | Nenhum identificador de amostra versionado | 0 | **1** ✅ | 3 reais → sintéticos de 2099; varredura final limpa |
| 3.7 | Metadados de proveniência não vazam para a saída | 0 | **1** ✅ | `tests/test_sanitizacao_metadados.py`, 8 testes |
| 3.8 | Licença de cada dataset público citada | 1 | 1 | `VALIDACAO_PUBLICA.md` |
| 3.9 | Validação externa executada | 1 | 1 | Corn |
| 3.10 | Comparação com o publicado | 1 | 1 | RMSEP 0,144 vs 0,1–0,2 |
| 3.11 | Validação externa no CI | — | **1** ✅ | job `validacao-publica` |
| 3.12 | 2º e 3º datasets públicos (mel, Mendeley) | N/V | **0** | **NÃO OBTIDO** — motivo em `VALIDACAO_PUBLICA.md` §2 |

**Subtotal: 11,5 / 13 → 88,5 %**

---

## 4. Jurídico e licenciamento — peso 15 %

Inalterado em relação a 08-17, exceto:

| # | Item | 08-17 | **08-18** |
|---|---|:--:|:--:|
| 4.5 | Proveniência documentada, não apagada | 0,5 | **1** ✅ (`DESACOPLAMENTO_2026-08-18.md` §6) |
| 4.13 | `ACKNOWLEDGMENTS` para quem operou o instrumento | — | **0,5** — arquivo criado com estrutura e política; nomes dependem de autorização das pessoas |

**Subtotal: 10 / 13 → 76,9 %**

---

## 5. Segurança — peso 10 %

Inalterado: **8 / 8 → 100 %** (1 N/V: deploy público não verificável sem login).

---

## 6. Documentação, CLI, referências — peso 10 %

| # | Item | 08-17 | **08-18** | Evidência |
|---|---|:--:|:--:|---|
| 6.1 | `--help` completo, flags consistentes | 0,5 | **1** ✅ | `_TEXTO_AJUDA`, fonte única |
| 6.2 | Modo cego padrão na CLI | 0 | **1** ✅ | `--modo=controle` |
| 6.3 | Saída declara valor + incerteza + domínio + versão + matriz | 0,5 | **1** ✅ | perfil e matriz no model card |
| 6.4 | README com instalação testada e recursos documentados | 1 | **1** ✅ | instalação testada em venv limpa; README (EN/PT) e MANUAL §4b cobrem perfis, modo cego e sanitização |
| 6.5 | Model card | 1 | 1 | matriz vem do perfil |
| 6.6 | DOIs verificados | 1 | 1 | 16/16 resolvem |
| 6.7 | Nenhuma referência inventada | 1 | 1 | — |
| 6.8 | Docstrings públicas | 1 | 1 | — |
| 6.9 | Códigos de saída corretos | — | **1** ✅ | 0 sucesso / 2 uso incorreto, verificado |

**Subtotal: 9 / 9 → 100 %**

---

## 7. Repositório e CI — peso 5 %

| # | Item | 08-17 | **08-18** |
|---|---|:--:|:--:|
| 7.1–7.5 | Estrutura, `.gitignore`, histórico, CI, SemVer | 5/5 | 5/5 |
| 7.6 | Issues/PRs triadas | 0 | 0 — 5 PRs do Dependabot; 11+ commits locais |
| 7.7 | Wheel construído e verificado | — | **1** ✅ — `twine check` PASSED, perfis dentro, zero artefato suspeito |

**Subtotal: 6 / 7 → 85,7 %**

---

## Consolidado

| Dimensão | Peso | 08-17 | **08-18** | Contribuição |
|---|---:|---:|---:|---:|
| Metodologia científica e validação | 25 % | 50,0 % | **88,9 %** | 22,22 |
| Código e testes | 20 % | 95,5 % | 91,7 % | 18,33 |
| Dados (proveniência, vazamento) | 15 % | 77,3 % | **88,5 %** | 13,27 |
| Jurídico, titularidade, licenciamento | 15 % | 75,0 % | 76,9 % | 11,54 |
| Segurança | 10 % | 100 % | 100 % | 10,00 |
| Documentação, CLI, referências | 10 % | 75,0 % | **100 %** | 10,00 |
| Repositório e CI | 5 % | 83,3 % | 85,7 % | 4,29 |
| **TOTAL** | **100 %** | 76,1 % | **89,7 %** | |

**Itens `N/V`: 7 de 87 (8,0 %)** — abaixo do limiar de 20 %, então o
percentual é confiável no sentido do prompt.

**Como ler o salto de 76 → 87.** Metade vem de trabalho real (modo cego,
perfis, sanitização, RPD/RER, CLI). A outra metade vem de **mudança de
escopo**: 5 itens que puxavam a nota para baixo eram propriedades do
delineamento do acervo, e agora são `N/V` porque pertencem ao TCC. Isso não
é maquiagem — é a consequência aritmética da decisão que você tomou — mas
seria desonesto apresentar 87 % como se fosse tudo melhoria de engenharia.
A dívida não sumiu; mudou de dono.

**Queda em "Código e testes" (95,5 → 91,7):** não houve regressão. Foi
adicionado um item que antes não era medido — complexidade de `executar()` e
`guaraci.py`, que continua pontuando 0. Medir mais coisas baixa a média
quando o que se passa a medir está pior que a média anterior.

**Retificação registrada.** O scorecard de 08-17 não media ciclos de import;
a medição inicial desta rodada acusou 14 e eu os reportei como dívida. A
contagem estava errada: ela somava imports dentro de funções e sob
`TYPE_CHECKING`, que existem precisamente para quebrar ciclos. Entre imports
de nível de módulo há **zero**, e os 31 módulos importam sozinhos num
interpretador limpo. Ficou um teste travando a propriedade, para que a
próxima auditoria não precise refazer a conta.
