# Fase de design e funcionalidade: o que pode mudar e o que não pode sem repetir a verificação

**Público.** Estado: manual. Atualizado junto do hash verificado em `docs/qualidade/`.

Depois do hash final verificado, a próxima fase mexe em design e funcionalidade. Este documento separa o que isso pode tocar **livremente** do que **exige repetir a verificação**, e diz como o repositório impede o segundo caso de passar despercebido.

## 1. Pode mudar sem repetir a verificação científica

Mudança de apresentação, que não altera número, método nem validação:

| área | onde |
|---|---|
| textos do catálogo (nomes, descrições, ajuda, dúvidas do assistente, rótulos) | `src/guaraci/catalogo_*.py`, `confianca.py`, `assistente.py` |
| layout, cores, tipografia, tokens de design | `design_tokens.py`, `guaraci_theme.py`, `preferencias_visuais.py`, `paleta_cores.py`, `app_tabs/`, `.streamlit/` |
| menus, ordem das telas, atalhos | `app_nav.py`, `guaraci.py` (menus), `cli_comandos.py` (texto de ajuda), `cli_assistente.py` |
| documentação e exemplos | `docs/`, `README*.md`, `MANUAL.md`, `exemplos.py` (dados sintéticos) |

Mesmo assim, **depois de qualquer mudança** os testes de contrato do catálogo, de links e de texto proibido precisam passar.

## 2. Não pode mudar sem repetir `scripts/verificar_hash_final.py`

Tudo que produz ou valida um número:

- cálculo da comparação de grupos (`anova_grupos.py`) e a política padrão da CLI (`anova_cli.py`);
- pré-processamento (`preprocessamento.py`), regressão e classificação (`chemometric_stats.py`, `classificadores.py`, `dual_spls.py`, `pipeline.py`);
- **validação cruzada** e partição estável (`validacao_estatistica.py`), agrupamento por unidade independente (`agrupamento_pastas.py`, `dados_io.py`);
- **seleção de variáveis** (`selecao_variaveis.py`, `vip_estabilidade.py`);
- dados de instrumento (`importadores_proprietarios.py`, `leitores_avancados.py`, `eem_io.py`, `gcms_io.py`, `icp_multielementar.py`), HSI (`hsi_*.py`), transferência de calibração, MSPC, conformal e domínio de aplicabilidade.

A lista exata, com o SHA-256 de cada arquivo, está em `docs/qualidade/nucleo_congelado.json`. O teste `tests/test_nucleo_congelado.py` **falha** se um desses arquivos mudar sem o manifesto ser regenerado (`python scripts/congelar_nucleo.py`) — e regenerar o manifesto é exatamente o gesto de dizer "repeti a verificação".

## 3. Como repetir

1. Árvore limpa, tudo commitado.
2. `python scripts/verificar_hash_final.py` (suíte completa com os testes lentos, `ruff`, `mypy`, `bandit`, `semgrep`, `pip-audit`, matriz de cenários sintéticos, contrato do catálogo, links, auditores de exposição e de termos de matriz, instalação limpa, matriz Python 3.10 a 3.13).
3. O resultado fica em `resultado_<hash>.json`; a verificação falha se o `HEAD` mudar durante a execução. Um documento "verificado em `<hash>`" só vale para aquele hash; um teste confere que `<hash>..HEAD` toca só arquivos gerados em `docs/qualidade/`.

## 4. Regra prática para quem for mexer

> Se a mudança altera **o que o programa calcula ou como valida**, é do §2. Se altera **como aparece**, é do §1. Na dúvida, é do §2.

## 5. Resumo do que pode e do que não pode

| pode mudar (design) | não pode mudar sem nova verificação do núcleo |
|---|---|
| layout, cores, tipografia, temas claro e escuro | os arquivos listados em `docs/qualidade/nucleo_congelado.json` |
| textos de interface e de ajuda da CLI, mensagens | os números de saída (o teste de valores de referência falha se algum mudar) |
| ordem de telas, menus, atalhos | limiares, políticas padrão e seeds |
| figuras: estilo, cores, rótulos, tamanho | os dados plotados e os formatos de arquivo de modelo (`.json`, `.joblib`) |

## 6. Mapa: onde fica cada tela, componente, figura e texto

| o quê | onde (arquivo → função/estrutura) |
|---|---|
| menu principal da CLI e telas por assunto | `guaraci.py` → `_print_main_menu`, `_loop_menu`, `_menu_project`, `_menu_data`, `_menu_preprocessing`, `_menu_modeling`, `_menu_validation`, `_menu_advanced`, `_menu_visualization`, `_menu_prediction`, `_menu_hsi`, `_menu_plan`, `_menu_selecao_amostras`, `_menu_tec_*` |
| cabeçalho, rodapé e ajuda de cada tela da CLI | `guaraci.py` → `_cabecalho_tela`, `_rodape_tela`, `_ajuda_da_tela`, `_print_submenu_compact` |
| boas-vindas, despedida, mensagens de status | `guaraci.py` → `_exibir_boas_vindas`; `guaraci_theme.py` → `ok`, `warn`, `err`, `info` |
| tema e cores do terminal, largura | `guaraci_theme.py` → `terminal_claro`, `ansi`, `_W` |
| subcomandos (`anova`, `vip`, `monitorar`, `transferir`, `exemplo`, `confianca`, `assistente`, `ajuda`, `manual`…) | `cli_comandos.py` → `SUBCOMANDOS`, `texto_ajuda`, `_cmd_*`; ajuda de cada um em `anova_cli.py`, `vip_cli.py`, `monitorar_cli.py`, `transferir_cli.py` |
| assistente por perguntas e catálogo de textos | `assistente.py`, `cli_assistente.py`, `catalogo*.py` (nomes, rotas, dúvidas, opções, técnicas) |
| páginas do aplicativo web | `app_web.py` (estrutura, traduções), `app_nav.py` (grupos e ordem), `app_tabs/*.py` (uma por página: início, projeto, dados, pré-processamento, modelo, validação, técnicas, visualização, predição, relatórios, sobre) |
| tema do aplicativo web | `design_tokens.py` (tokens claro/escuro), `.streamlit/config.toml`, `preferencias_visuais.py` |
| paleta, marcadores e transparência das figuras | `paleta_cores.py`; escolha pelo usuário em `cli_assistente.py` → `apply_palette` e no seletor de paleta do web |
| figuras do fluxo principal e do HSI | `figuras.py`, `hsi_figures.py`, `spectra_preview.py` (e as funções `fig_*` de cada módulo de método) |
| relatórios (LaTeX, Word, PDF, slides) | `reports.py`, `relatorio_cli.py`, `relatorio_blocos.py` |
| documentos do manual dentro do pacote | `dados_docs/` (gerados por `scripts/gerar_docs_pacote.py` a partir de `docs/` e `MANUAL.md`) |
| mensagens de erro de entrada | `erros_entrada.py`, `web_seguranca.py` (modo servidor: sem detalhe técnico ao visitante) |

## 7. Pontos de design já conhecidos (nada implementado)

| ponto | onde | classe |
|---|---|---|
| paleta pensada para fundo escuro; em terminal claro algumas cores perdem contraste | `guaraci_theme.py` (`terminal_claro`, `ansi`) | design |
| terminal estreito (menos de 80 colunas) trunca linhas do menu; a largura útil é fixada entre 60 e 100 | `guaraci_theme.py` → `_W`; `guaraci.py` → `_print_main_menu`, `_print_submenu_compact` | design |
| primeira abertura das telas: falta orientação sobre o que fazer primeiro (CLI e web) | `guaraci.py` → `_exibir_boas_vindas`; `app_tabs/inicio.py` | design |
| bilinguismo parcial: parte dos textos da CLI e do web só existe em um idioma | `app_web.py` (tabela de traduções), `catalogo*.py`, mensagens soltas em `guaraci.py` | design |
| pastas de saída diferentes por comando (`resultados_anova`, `resultados_vip`, `resultados_monitoramento` …) e sem regra única | `anova_cli.py`, `vip_cli.py`, `monitorar_cli.py`, `transferir_cli.py` (argumento `--saida`) | design |
| rastros técnicos em mensagens de erro (nome de exceção, caminho) em alguns caminhos da CLI | `guaraci.py` (tratamento de exceção em `main`), `erros_entrada.py` | design |
| não existe subcomando de predição: o fluxo só existe na tela de predição do menu e na página web | `cli_comandos.py` (`SUBCOMANDOS`), `guaraci.py` → `_menu_prediction`, `app_tabs/predicao.py` | v1.1 (funcionalidade nova) |

## 8. Regras da fase de design

1. Cada rodada de design termina com `python scripts/checar_design.py` **verde** e um commit de **assunto único**.
2. Nenhuma verificação completa durante o design. A verificação completa (`scripts/verificar_hash_final.py`) roda **uma vez**, depois, no hash final, junto com o teste de usabilidade externo.
3. Se `checar_design.py` disser que o núcleo mudou, a mudança não é de design: desfaça, ou trate como mudança científica (com teste novo e manifesto regenerado de propósito).
4. Funcionalidade nova não entra: vai para o roteiro da v1.1.

## 9. Como usar `scripts/checar_design.py`

```bash
python scripts/checar_design.py            # confere tudo (meta: até 10 minutos no Windows; medido: 2,2 minutos)
python scripts/checar_design.py --rapido   # sem a camada de testes de interface (só núcleo, estilo, varreduras, instantâneo)
python scripts/checar_design.py --base     # regrava o instantâneo de comparação (só depois de aprovar a mudança visual)
```

Etapas, em ordem; as marcadas **bloqueia** fazem o comando sair com código diferente de 0:

| etapa | o que confere | bloqueia |
|---|---|---|
| `ruff` | estilo e erros estáticos | sim |
| `nucleo` | o manifesto do núcleo continua batendo com os arquivos | **sim** (núcleo mudou) |
| `valores` | números de referência (valores de referência e contrato da API) | **sim** |
| `varreduras` | e-mail, caminho de usuário, termos sensíveis, termos de matriz | sim |
| `testes_interface` | testes de menus, telas, paleta, vocabulário, CLI e ajuda | sim |
| `instantaneo` | `--help`, `--version` e telas comparados com o instantâneo salvo; as diferenças são **listadas**, não bloqueiam | não |

O instantâneo fica em `~/.guaraci_local/design_base/` (fora do repositório): textos de `--help` e `--version`, a saída de texto do demo,
lista e tamanho das figuras do demo, tempo de abertura e (com `--base --com-wheel`) o tamanho do wheel. Para regenerar depois de uma
mudança visual aprovada: `python scripts/checar_design.py --base`.
