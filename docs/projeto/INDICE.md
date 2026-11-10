# Índice dos documentos

**Público:** todos · **Estado:** gerado por `scripts/gerar_indice_docs.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Todo documento versionado aparece uma vez, no destino a que pertence (menos os `docs/qualidade/resultado_<hash>.json`, um por verificação final). A política (o que vive onde, o que nunca entra) está em [`../POLITICA_DE_DOCUMENTACAO.md`](../POLITICA_DE_DOCUMENTACAO.md).

## Comece aqui

| documento | o que é |
|---|---|
| [`CHANGELOG.md`](../../CHANGELOG.md) | Histórico de mudanças por versão. |
| [`MANUAL.md`](../../MANUAL.md) | Manual de uso: telas, comandos, técnicas, opções e limitações de cada uma. |
| [`README.md`](../../README.md) | Visão geral em inglês: o que é, instalação, primeiros passos e limitações. |
| [`README.pt-BR.md`](../../README.pt-BR.md) | Versão completa em português do README. |
| [`docs/index.md`](../index.md) | Página inicial do site de documentação. |

## Usuário

| documento | o que é |
|---|---|
| [`docs/usuario/expectativas_teste_usabilidade.md`](../usuario/expectativas_teste_usabilidade.md) | O que se espera aprender com um teste de usabilidade e como interpretar os resultados. |
| [`docs/usuario/guia_do_testador_externo.md`](../usuario/guia_do_testador_externo.md) | Guia para quem testa o programa sem conhecer o projeto: o que fazer e o que relatar. |
| [`docs/usuario/instalacao.md`](../usuario/instalacao.md) | Instalação passo a passo no Windows, Linux e WSL, com requisitos mínimos e problemas comuns. |
| [`docs/usuario/privacidade.md`](../usuario/privacidade.md) | Rascunho do aviso de privacidade do demonstrativo hospedado: o que recebe, onde fica, por quanto tempo e o que não enviar. |
| [`docs/usuario/roteiro_teste_usabilidade.md`](../usuario/roteiro_teste_usabilidade.md) | Roteiro de tarefas para um teste de usabilidade. |
| [`docs/usuario/template_feedback_teste.md`](../usuario/template_feedback_teste.md) | Modelo de formulário para registrar o resultado de um teste. |
| [`docs/usuario/termos_de_uso.md`](../usuario/termos_de_uso.md) | Rascunho dos termos de uso do demonstrativo hospedado: não é laudo, sem garantia, o que não enviar. |

## Científico

| documento | o que é |
|---|---|
| [`docs/cientifico/glossario_referencias_no_codigo.md`](../cientifico/glossario_referencias_no_codigo.md) | Glossário único dos rótulos que aparecem em comentários e docstrings (Passo, Bloco, achados). |
| [`docs/cientifico/limitacoes.md`](../cientifico/limitacoes.md) | O que o projeto não garante, com a evidência de cada limitação. |
| [`docs/cientifico/numeros_publicos.yaml`](../cientifico/numeros_publicos.yaml) | Contagens próprias do repositório citadas nos READMEs (sem número de conjunto de dados). |
| [`docs/cientifico/referencias.md`](../cientifico/referencias.md) | Tabela única de referências (fonte, DOI, data de conferência, onde é citada), gerada do registro. |
| [`docs/cientifico/referencias_verificadas.yaml`](../cientifico/referencias_verificadas.yaml) | Tabela única de referências: fonte, DOI, data de conferência e onde é citada. |
| [`docs/cientifico/validacao_numerica.md`](../cientifico/validacao_numerica.md) | Conferência dos métodos contra as fórmulas que os definem, com tolerâncias. |

## Contribuir

| documento | o que é |
|---|---|
| [`ACKNOWLEDGMENTS.md`](../../ACKNOWLEDGMENTS.md) | Atribuição do software e da literatura em que o projeto se apoia. |
| [`CODE_OF_CONDUCT.md`](../../CODE_OF_CONDUCT.md) | Código de conduta da comunidade. |
| [`CONTRIBUTING.md`](../../CONTRIBUTING.md) | Como relatar erro, propor mudança e rodar os testes. |
| [`GOVERNANCE.md`](../../GOVERNANCE.md) | Projeto mantido por uma pessoa: quem decide e como alguém vira colaborador. |
| [`SECURITY.md`](../../SECURITY.md) | Política de segurança: carregamento de modelo, modo servidor e canal para relato de vulnerabilidade. |
| [`SUPPORT.md`](../../SUPPORT.md) | Onde tirar dúvida, relatar erro ou falar com o mantenedor, e o prazo de resposta esperado. |
| [`docs/api/chemometric_stats.md`](../api/chemometric_stats.md) | Referência da API: estatística quimiométrica. |
| [`docs/api/classificadores.md`](../api/classificadores.md) | Referência da API: classificadores. |
| [`docs/api/predicao.md`](../api/predicao.md) | Referência da API: predição. |
| [`docs/api/preprocessamento.md`](../api/preprocessamento.md) | Referência da API: pré-processamento. |
| [`docs/api/selecao_variaveis.md`](../api/selecao_variaveis.md) | Referência da API: seleção de variáveis. |
| [`docs/api/validacao_estatistica.md`](../api/validacao_estatistica.md) | Referência da API: validação estatística. |
| [`docs/contribuir/arquitetura.md`](../contribuir/arquitetura.md) | Mapa dos módulos do pacote e de como se encaixam. |
| [`docs/contribuir/compatibilidade_da_api.md`](../contribuir/compatibilidade_da_api.md) | Contrato da API pública e política de compatibilidade entre versões. |

## Projeto

| documento | o que é |
|---|---|
| [`docs/POLITICA_DE_DOCUMENTACAO.md`](../POLITICA_DE_DOCUMENTACAO.md) | O que vive onde, o que nunca entra em arquivo versionado e como isso é verificado. |
| [`docs/projeto/FASE_DESIGN.md`](../projeto/FASE_DESIGN.md) | Fase de design depois do núcleo congelado: o que entra, o que não entra e como verificar. |
| [`docs/projeto/INDICE.md`](../projeto/INDICE.md) | Este índice (gerado). |
| [`docs/projeto/MAPA_DE_MIGRACAO.md`](../projeto/MAPA_DE_MIGRACAO.md) | De onde cada documento veio: caminho antigo, caminho novo. |
| [`docs/projeto/ROADMAP.md`](../projeto/ROADMAP.md) | Roteiro por fase, da v1.0.0 à v1.1+: o que entra, o que não entra, critério de pronto e dependência. |
| [`docs/projeto/escopo_fusao_multibloco_e_mspc.md`](../projeto/escopo_fusao_multibloco_e_mspc.md) | Escopo da fusão multibloco e do controle estatístico multivariado de processo. |
| [`docs/projeto/estudo_novas_tecnicas.md`](../projeto/estudo_novas_tecnicas.md) | Estudo de viabilidade de seis técnicas candidatas (UV-Vis, ICP-OES, MIP-OES, FAAS, GFAAS, DMA), sem código. |
| [`docs/projeto/identidade_visual.md`](../projeto/identidade_visual.md) | Identidade visual e padrões de interface (terminal e web). |
| [`docs/projeto/roteiro_v1_1.md`](../projeto/roteiro_v1_1.md) | Roteiro de técnicas e capacidades previstas para a v1.1 e depois. |

## Qualidade (gerado por script)

| documento | o que é |
|---|---|
| [`docs/qualidade/anova/cobertura.json`](../qualidade/anova/cobertura.json) | Cobertura de ramos dos módulos de ANOVA e VIP. |
| [`docs/qualidade/anova/equivalentes_amostra.md`](../qualidade/anova/equivalentes_amostra.md) | Amostra dos mutantes classificados como equivalentes, com a verificação. |
| [`docs/qualidade/anova/mutacao_anova_cli.json`](../qualidade/anova/mutacao_anova_cli.json) | Resultado do teste de mutação de `anova_cli.py` (para `pulados`: `tests/_pulados.py`, a guarda de testes pulados). |
| [`docs/qualidade/anova/mutacao_anova_grupos.json`](../qualidade/anova/mutacao_anova_grupos.json) | Resultado do teste de mutação de `anova_grupos.py` (para `pulados`: `tests/_pulados.py`, a guarda de testes pulados). |
| [`docs/qualidade/anova/mutacao_pulados.json`](../qualidade/anova/mutacao_pulados.json) | Resultado do teste de mutação de `pulados.py` (para `pulados`: `tests/_pulados.py`, a guarda de testes pulados). |
| [`docs/qualidade/anova/mutacao_relatorio_blocos.json`](../qualidade/anova/mutacao_relatorio_blocos.json) | Resultado do teste de mutação de `relatorio_blocos.py` (para `pulados`: `tests/_pulados.py`, a guarda de testes pulados). |
| [`docs/qualidade/anova/mutacao_vip_cli.json`](../qualidade/anova/mutacao_vip_cli.json) | Resultado do teste de mutação de `vip_cli.py` (para `pulados`: `tests/_pulados.py`, a guarda de testes pulados). |
| [`docs/qualidade/anova/mutacao_vip_cli_py310_linha214.json`](../qualidade/anova/mutacao_vip_cli_py310_linha214.json) | Resultado do teste de mutação de `vip_cli.py` numa linha, em outra versão de Python. |
| [`docs/qualidade/anova/mutacao_vip_cli_py311_linha214.json`](../qualidade/anova/mutacao_vip_cli_py311_linha214.json) | Resultado do teste de mutação de `vip_cli.py` numa linha, em outra versão de Python. |
| [`docs/qualidade/anova/mutacao_vip_estabilidade.json`](../qualidade/anova/mutacao_vip_estabilidade.json) | Resultado do teste de mutação de `vip_estabilidade.py` (para `pulados`: `tests/_pulados.py`, a guarda de testes pulados). |
| [`docs/qualidade/anova/mutantes_classificados.json`](../qualidade/anova/mutantes_classificados.json) | Classificação dos mutantes sobreviventes. |
| [`docs/qualidade/anova/recomendacao_simulacao.json`](../qualidade/anova/recomendacao_simulacao.json) | Dados da simulação da recomendação do ANOVA. |
| [`docs/qualidade/anova/recomendacao_simulacao.md`](../qualidade/anova/recomendacao_simulacao.md) | Simulação da recomendação automática de método do ANOVA. |
| [`docs/qualidade/anova_blindagem.md`](../qualidade/anova_blindagem.md) | Relatório de blindagem do ANOVA: oráculos independentes, mutação e cobertura. |
| [`docs/qualidade/cobertura/mutacao_cobertura_excecoes.json`](../qualidade/cobertura/mutacao_cobertura_excecoes.json) | Teste de mutação da guarda de exclusões de cobertura (`scripts/cobertura_excecoes.py`), primeira rodada. |
| [`docs/qualidade/cobertura/mutacao_cobertura_excecoes_2.json`](../qualidade/cobertura/mutacao_cobertura_excecoes_2.json) | Teste de mutação da guarda de exclusões de cobertura, nova rodada só dos sobreviventes. |
| [`docs/qualidade/cobertura_lacunas.md`](../qualidade/cobertura_lacunas.md) | Linhas e ramos que a suíte não cobre, por camada, com a categoria de cada lacuna e as exclusões registradas (gerado da cobertura medida). |
| [`docs/qualidade/dependencias_de_maquina.md`](../qualidade/dependencias_de_maquina.md) | Testes que usam RAM, CPU, relógio, caminho, ambiente ou rede, com a decisão de cada um (gerado por varredura). |
| [`docs/qualidade/estado_do_projeto.md`](../qualidade/estado_do_projeto.md) | Estado do projeto medido por script: documentos por destino, auditores, links e referências. |
| [`docs/qualidade/mapa_tecnica_tarefa.md`](../qualidade/mapa_tecnica_tarefa.md) | Mapa técnica × tarefa × comando × exemplo sintético × teste (gerado do catálogo). |
| [`docs/qualidade/matriz_cenarios.md`](../qualidade/matriz_cenarios.md) | Matriz de cenários sintéticos rodados pela CLI, com resultado e tempo. |
| [`docs/qualidade/mutacao/anova_grupos__426-429.json`](../qualidade/mutacao/anova_grupos__426-429.json) | Resultado do teste de mutação de `anova_grupos.py`, linhas 426-429 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/anova_grupos__63-64.json`](../qualidade/mutacao/anova_grupos__63-64.json) | Resultado do teste de mutação de `anova_grupos.py`, linhas 63-64 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/avaliacao_modelos__273-276.json`](../qualidade/mutacao/avaliacao_modelos__273-276.json) | Resultado do teste de mutação de `avaliacao_modelos.py`, linhas 273-276 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/avaliacao_modelos__783-785.json`](../qualidade/mutacao/avaliacao_modelos__783-785.json) | Resultado do teste de mutação de `avaliacao_modelos.py`, linhas 783-785 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/chemometric_stats__1090-1091.json`](../qualidade/mutacao/chemometric_stats__1090-1091.json) | Resultado do teste de mutação de `chemometric_stats.py`, linhas 1090-1091 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/classificadores__665-667.json`](../qualidade/mutacao/classificadores__665-667.json) | Resultado do teste de mutação de `classificadores.py`, linhas 665-667 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/config_io__598-600.json`](../qualidade/mutacao/config_io__598-600.json) | Resultado do teste de mutação de `config_io.py`, linhas 598-600 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/dados_io__1181-1192.json`](../qualidade/mutacao/dados_io__1181-1192.json) | Resultado do teste de mutação de `dados_io.py`, linhas 1181-1192 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/dados_io__1214-1216.json`](../qualidade/mutacao/dados_io__1214-1216.json) | Resultado do teste de mutação de `dados_io.py`, linhas 1214-1216 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/dados_io__1225.json`](../qualidade/mutacao/dados_io__1225.json) | Resultado do teste de mutação de `dados_io.py`, linhas 1225 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/dados_io__315-323.json`](../qualidade/mutacao/dados_io__315-323.json) | Resultado do teste de mutação de `dados_io.py`, linhas 315-323 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/docs_pacote__1-58.json`](../qualidade/mutacao/docs_pacote__1-58.json) | Resultado do teste de mutação de `docs_pacote.py`, linhas 1-58 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/pipeline__1740-1741.json`](../qualidade/mutacao/pipeline__1740-1741.json) | Resultado do teste de mutação de `pipeline.py`, linhas 1740-1741 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/pipeline__2294-2378_1a_rodada.json`](../qualidade/mutacao/pipeline__2294-2378_1a_rodada.json) | Resultado do teste de mutação de `pipeline.py`, linhas 2294-2378 1a rodada (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/pipeline__2294-2378_reexecucao.json`](../qualidade/mutacao/pipeline__2294-2378_reexecucao.json) | Resultado do teste de mutação de `pipeline.py`, linhas 2294-2378 reexecucao (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/plano_amostral__126-140.json`](../qualidade/mutacao/plano_amostral__126-140.json) | Resultado do teste de mutação de `plano_amostral.py`, linhas 126-140 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/selecao_variaveis__1225-1228.json`](../qualidade/mutacao/selecao_variaveis__1225-1228.json) | Resultado do teste de mutação de `selecao_variaveis.py`, linhas 1225-1228 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao/selecao_variaveis__508.json`](../qualidade/mutacao/selecao_variaveis__508.json) | Resultado do teste de mutação de `selecao_variaveis.py`, linhas 508 (para `mutacao_linhas_alteradas.md`). |
| [`docs/qualidade/mutacao_linhas_alteradas.md`](../qualidade/mutacao_linhas_alteradas.md) | Teste de mutação nas linhas alteradas desde a última verificação completa: mortos, sobreviventes classificados e módulos não executados (gerado). |
| [`docs/qualidade/nucleo_congelado.json`](../qualidade/nucleo_congelado.json) | Manifesto do núcleo que produz ou valida um número, com o SHA-256 de cada arquivo. |
| [`docs/qualidade/politicas_anova_simulacao.json`](../qualidade/politicas_anova_simulacao.json) | Dados da simulação das políticas do ANOVA. |
| [`docs/qualidade/politicas_anova_simulacao.md`](../qualidade/politicas_anova_simulacao.md) | Simulação das políticas de comparação múltipla do ANOVA: erro tipo I e poder. |
| [`docs/qualidade/rodada_local_formatos_reais.json`](../qualidade/rodada_local_formatos_reais.json) | Registro (sem arquivos) da rodada local dos leitores contra arquivos reais de fabricante: contagens e hash. |
| [`docs/qualidade/selecao_aninhada_rotulos_aleatorios.json`](../qualidade/selecao_aninhada_rotulos_aleatorios.json) | Evidência de que a seleção aninhada fica no acaso com rótulos sorteados. |
