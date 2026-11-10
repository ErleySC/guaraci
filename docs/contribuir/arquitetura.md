# Arquitetura do GUARACI — guia para quem assume a manutenção

**Público:** quem mantém ou contribui com o código · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Objetivo: um mantenedor novo consegue achar onde cada coisa vive, entender as decisões que não estão óbvias no código e rodar tudo (m-24 do parecer de 2026-09-25: *bus factor* = 1).

## Fluxo de uma execução

```
Config (config.py, _CONFIG_SPEC em config_io.py — fonte única dos campos)
  → dados_io.load_data (registry de leitores: io_registry.py / leitores_avancados.py)
      devolve (wn, X, rótulos, conc, mae_id, metadados)  + grupo → garantia de agrupamento
  → pipeline.executar  (_executar_impl)
      truncamento espectral → holdout agrupado estratificado → pré-processamento
      → CV (StableStratifiedGroupKFold) → permutação → métricas/BCa → DD-SIMCA/AD
      → regressão por espécie / pooled (holdout externo) → figuras → resultados_io
  → saída: <pasta>/<Amostra>/<Modo>/<execução>/{Graficos,Tabelas,Modelos,Relatorios}
```

Duas interfaces finas sobre o mesmo núcleo: CLI (`guaraci.py`, Rich) e web (`app_quimiometria.py` + `app_tabs/`). Nenhuma lógica científica vive nelas.

## Decisões que não são óbvias

| Decisão | Onde | Por quê |
|---|---|---|
| Validação agrupada por padrão, com declaração do **nível de garantia** (`high/medium/none`) | `dados_io.load_data`, `garantia_agrupamento_prevista` | Sem chave de grupo o software não finge proteção: declara e avisa |
| Unidade independente configurável (`mae_id` × `sessao`) | `chave_unidade_validacao`, `auditoria_delineamento` | Diluições da mesma sessão em folds diferentes deixam o modelo aprender a sessão |
| Splitter estável próprio | `validacao_estatistica.StableStratifiedGroupKFold` | O do sklearn muda a partição entre versões |
| O hardware nunca altera parâmetro científico | `hardware.py` | Métrica dependente da RAM foi um achado bloqueante |
| Web em **modo servidor** por padrão | `web_seguranca.py`, `SECURITY.md` | Pickle executa código; caminhos livres leem o servidor |
| Modelo `.joblib` só com manifesto **assinado** (HMAC local) | `predicao.load_model` | Hash prova integridade, não origem |
| Predição recusa extrapolação | `eixo_espectral.py` | `np.interp` repete a borda e produzia 99% de confiança |
| Correções locais de metadado só por opção explícita | `dados_io.definir_correcoes_locais` | Reclassificação silenciosa quebrava a reprodutibilidade |
| Números públicos e citações verificados por script | `docs/cientifico/numeros_publicos.yaml`, `scripts/verificar_citacoes.py` | Alegação sem fonte única envelhece |

## Como rodar tudo

```bash
pip install -e ".[all]"            # ambiente completo (Python 3.11–3.13)
pytest tests                       # a suíte inteira; a contagem atual está em docs/qualidade/estado_do_projeto.md
python scripts/verificar_citacoes.py          # DOIs contra Crossref/DataCite (rede)
python scripts/pisos_de_versao.py > /tmp/p.txt && pip-audit -r /tmp/p.txt --no-deps --disable-pip
ruff check . ; mypy src/guaraci app_quimiometria.py ; bandit -r src -ll
guaraci web                        # interface web local
```



## Mapa de módulos (gerado do primeiro parágrafo de cada docstring)

| Módulo | Papel |
|---|---|
| `src/guaraci/app_tabs/__init__.py` | app_tabs — um módulo por tela do app web (item 18 da auditoria: quebrar o |
| `src/guaraci/app_tabs/dados.py` | app_tabs/dados.py — Tela (Data): entrada de dados (.dx, CSV local/upload) |
| `src/guaraci/app_tabs/inicio.py` | app_tabs/inicio.py — Tela (Home): painel de status do projeto. |
| `src/guaraci/app_tabs/modelo.py` | app_tabs/modelo.py — Tela (Model): parâmetros avançados + execução do |
| `src/guaraci/app_tabs/predicao.py` | app_tabs/predicao.py — Tela (Prediction): aplica um modelo `.joblib` |
| `src/guaraci/app_tabs/preprocessamento.py` | app_tabs/preprocessamento.py — Tela (Preprocessing): preset espectral + |
| `src/guaraci/app_tabs/projeto.py` | app_tabs/projeto.py — Tela (Project): identificação do projeto e status |
| `src/guaraci/app_tabs/relatorios.py` | app_tabs/relatorios.py — Tela (Reports): downloads (ZIP/PDF/Word/Excel/ |
| `src/guaraci/app_tabs/sobre.py` | app_tabs/sobre.py — Tela (About): identidade do projeto, diferenciais, |
| `src/guaraci/app_tabs/tecnicas.py` | app_tabs/tecnicas.py — Tela (Advanced Techniques): ASCA, EPO/GLSW, |
| `src/guaraci/app_tabs/validacao.py` | app_tabs/validacao.py — Tela (Validation): figuras e métricas da última |
| `src/guaraci/app_tabs/visualizacao.py` | app_tabs/visualizacao.py — Tela (Visualisation): cor dos gráficos. |
| `src/guaraci/__init__.py` | GUARACI — plataforma quimiométrica multitécnica (pacote). |
| `src/guaraci/agrupamento_pastas.py` | agrupamento_pastas.py — Hierarquia de 3 niveis de garantia de |
| `src/guaraci/alinhamento_retencao.py` | alinhamento_retencao.py -- COW (Correlation Optimized Warping) para |
| `src/guaraci/amostragem_ativa.py` | amostragem_ativa.py -- Amostragem ativa orientada por incerteza |
| `src/guaraci/app_logic.py` | app_logic.py — Lógica PURA da interface web (Streamlit), sem dependência do |
| `src/guaraci/app_nav.py` | app_nav.py — Definição PURA da navegação do app web (sem Streamlit). |
| `src/guaraci/asca.py` | asca.py — ASCA (ANOVA-Simultaneous Component Analysis), Smilde, Jansen, |
| `src/guaraci/auditoria_delineamento.py` | auditoria_delineamento.py -- Auditoria de delineamento experimental |
| `src/guaraci/assistente.py` | assistente.py — Assistente Guaraci: determinístico, offline, sem LLM e sem rede. |
| `src/guaraci/aumento_dados.py` | aumento_dados.py -- Aumento de dados espectral (data augmentation) |
| `src/guaraci/avaliacao_modelos.py` | avaliacao_modelos.py — Comparacao de classificadores e interpretabilidade: |
| `src/guaraci/chemometric_stats.py` | chemometric_stats.py — Diagnósticos quimiométricos puros (VIP, SR, Hotelling T², |
| `src/guaraci/classificadores.py` | classificadores.py — Classificadores quimiométricos avançados: DD-SIMCA |
| `src/guaraci/cli_assistente.py` | cli_assistente.py — Modulo de dados/i18n compartilhado do GUARACI. |
| `src/guaraci/cli_logic.py` | cli_logic.py — Lógica PURA da CLI de terminal (guaraci.py), sem dependência |
| `src/guaraci/config.py` | config.py — Objeto de configuração central do pipeline (dataclass Config). |
| `src/guaraci/config_io.py` | config_io.py — Serialização, validação e schema (_CONFIG_SPEC) da Config. |
| `src/guaraci/conformal.py` | conformal.py — Predicao conformal para autenticacao one-class. |
| `src/guaraci/dados_imagem.py` | dados_imagem.py — Extracao de dados quimiometricos a partir de IMAGENS |
| `src/guaraci/dados_io.py` | dados_io.py — Parsing de JCAMP-DX/ASDF e carregamento de dados (.dx, CSV, |
| `src/guaraci/deriva_qc.py` | deriva_qc.py — Correção de deriva de sinal ao longo da ordem de |
| `src/guaraci/design_tokens.py` | design_tokens.py — Fonte UNICA de cor da plataforma GUARACI (web + CLI + figuras). |
| `src/guaraci/dual_spls.py` | dual_spls.py -- Dual-sPLS (PLS esparso via norma dual), variante lasso. |
| `src/guaraci/eem_io.py` | eem_io.py -- leitor de matrizes de excitacao-emissao (EEM) exportadas por |
| `src/guaraci/eem_multiway.py` | eem_multiway.py -- Decomposicao PARAFAC de matrizes de fluorescencia |
| `src/guaraci/eixo_espectral.py` | eixo_espectral.py -- cobertura do eixo espectral na PREDICAO (M-06). |
| `src/guaraci/epo_glsw.py` | epo_glsw.py — EPO e GLSW: removem/atenuam a variação de um fator de |
| `src/guaraci/figuras.py` | figuras.py — Toda a camada de plotagem do pipeline: setup do matplotlib, |
| `src/guaraci/fusao_multibloco.py` | fusao_multibloco.py -- Fusao multibloco de baixo nivel (Grupo 2, prova |
| `src/guaraci/formatos_sinteticos.py` | formatos_sinteticos.py — Escritores de formatos de instrumento com dados sintéticos de verdade conhecida. |
| `src/guaraci/gcms_io.py` | gcms_io.py -- Leitor de GC-MS bruto no formato ANDI-MS/NetCDF (ASTM |
| `src/guaraci/guaraci.py` | guaraci.py v1.0.0 — Interface profissional GUARACI para o pipeline quimiometrico |
| `src/guaraci/guaraci_theme.py` | guaraci_theme.py — Tema visual compartilhado da plataforma GUARACI. |
| `src/guaraci/hardware.py` | hardware.py — Deteccao e adaptacao a hardware (RAM/CPU/disco): probe, auto- |
| `src/guaraci/hsi_applicability.py` | hsi_applicability.py — Dominio de aplicabilidade para o espaco de |
| `src/guaraci/hsi_classification.py` | hsi_classification.py — PLS-DA por pixel + agregacao por voto |
| `src/guaraci/hsi_extras.py` | hsi_extras.py -- orquestracao testavel (sem Rich) dos modulos HSI que |
| `src/guaraci/hsi_figures.py` | hsi_figures.py — Mapa de classificacao espacial de cubo HSI (Passo 99 |
| `src/guaraci/hsi_io.py` | hsi_io.py — Estrutura interna de cubo hiperespectral (HSI) e leitor do |
| `src/guaraci/hsi_multiway.py` | hsi_multiway.py -- Decomposicao multiway do cubo hiperespectral: PARAFAC |
| `src/guaraci/hsi_pipeline.py` | hsi_pipeline.py — Orquestracao ponta-a-ponta do modo de entrada `hsi` |
| `src/guaraci/hsi_pixels.py` | hsi_pixels.py — Extracao de espectros de pixel da ROI (mascara do Passo |
| `src/guaraci/hsi_quality.py` | hsi_quality.py — Quality gate de cubo hiperespectral (Passo 95 da |
| `src/guaraci/hsi_resampling.py` | hsi_resampling.py — Reamostragem estratificada group-aware para |
| `src/guaraci/hsi_segmentation.py` | hsi_segmentation.py — Segmentacao espacial de cubo hiperespectral por |
| `src/guaraci/hsi_uncertainty.py` | hsi_uncertainty.py — Propagacao de incerteza formal para resultados |
| `src/guaraci/hsi_validation.py` | hsi_validation.py — Validacao externa por particao nativa de origem |
| `src/guaraci/identificacao.py` | identificacao.py -- Identificacao de combinacao especie x adulterante (Bloco 9b). |
| `src/guaraci/importadores_proprietarios.py` | importadores_proprietarios.py -- Leitores de formato binario proprietario |
| `src/guaraci/io_registry.py` | io_registry.py — Registry de leitores de dados (item 20 da auditoria). |
| `src/guaraci/leitores_avancados.py` | leitores_avancados.py -- registra no `io_registry` os leitores de formato |
| `src/guaraci/linearity.py` | linearity.py -- Teste formal de linearidade da curva de calibracao |
| `src/guaraci/log.py` | log.py — ponto UNICO de configuracao de logging do GUARACI (CLAUDE.md P6). |
| `src/guaraci/mcr_als.py` | mcr_als.py -- Resolucao de Curvas Multivariada por Minimos Quadrados |
| `src/guaraci/model_export.py` | model_export.py -- Exportacao portatil de modelo (Grupo 2, JSON puro). |
| `src/guaraci/model_registry.py` | model_registry.py — Registry de modelos do Auto-Benchmark / Monte Carlo CV |
| `src/guaraci/modos_analise.py` | modos_analise.py — Camada de OBJETIVO CIENTIFICO da analise. |
| `src/guaraci/paleta_cores.py` | paleta_cores.py — Paleta de cores e marcadores de máxima distintividade |
| `src/guaraci/perfil_matriz.py` | Perfis de matriz — o que muda quando a amostra muda. |
| `src/guaraci/pipeline.py` | Chemometric pipeline for multivariate analytical data (FT-NIR and other techniques). |
| `src/guaraci/plano_amostral.py` | plano_amostral.py -- Orientacao de tamanho amostral (P1, Bloco 10 -- |
| `src/guaraci/plano_coleta.py` | plano_coleta.py -- Distribuicao de amostras entre sessoes de coleta e |
| `src/guaraci/politica_pooled_local.py` | politica_pooled_local.py -- Politica automatica pooled vs. local |
| `src/guaraci/portao_correcao_sinal.py` | portao_correcao_sinal.py -- Portao de aceite automatico para tecnicas |
| `src/guaraci/predicao.py` | Predicao em amostras desconhecidas a partir de um modelo salvo (.joblib). |
| `src/guaraci/preferencias_visuais.py` | preferencias_visuais.py — Preferencias visuais persistentes do usuario |
| `src/guaraci/preprocessamento.py` | preprocessamento.py — Transformers sklearn-compatíveis de pré-processamento |
| `src/guaraci/reports.py` | reports.py — Geração de relatórios do app web (PDF, Word, Excel, LaTeX, |
| `src/guaraci/resultados_io.py` | resultados_io.py — Escrita dos artefatos de resultado em disco: resumo do |
| `src/guaraci/resumo_parse.py` | resumo_parse.py — Parsing PURO do resumo_modelo.txt (item 19 da auditoria: |
| `src/guaraci/robustness.py` | robustness.py -- Protocolo de robustez (Bloco 13d, Frente 1, decisao |
| `src/guaraci/selecao_variaveis.py` | selecao_variaveis.py — Etapa 4: selecao de variaveis espectrais (iPLS por |
| `src/guaraci/sentinela_deriva.py` | sentinela_deriva.py -- Monitoramento continuo de deriva do dominio de |
| `src/guaraci/spectra_preview.py` | spectra_preview.py — Carregamento/plotagem de amostra de espectros para |
| `src/guaraci/technique_registry.py` | technique_registry.py — Catalogo dos metodos cientificos do GUARACI |
| `src/guaraci/tecnicas_avancadas.py` | tecnicas_avancadas.py -- orquestracao testavel (sem Rich/Streamlit) para |
| `src/guaraci/transferencia_calibracao.py` | transferencia_calibracao.py -- Transferencia de calibracao entre |
| `src/guaraci/validacao_estatistica.py` | validacao_estatistica.py — Testes de significancia e intervalos de confianca |
| `src/guaraci/web_seguranca.py` | web_seguranca.py -- Politica de seguranca da interface web (Streamlit). |
| `src/guaraci/__main__.py` | `python -m guaraci` = o comando `guaraci` (R5-21 da revisao cega de 2026-09-26). |
| `src/guaraci/anova_cli.py` | anova_cli.py -- `guaraci anova`: comparação de grupos sobre a unidade independente (Tukey, Games-Howell, Kruskal-Wallis/Dunn, recomendação de método), com figuras, relatório md/html/tex e mensagens PT/EN. |
| `src/guaraci/anova_grupos.py` | ANOVA de uma via (e com bloco) + Tukey/Games-Howell, Kruskal-Wallis + Dunn, diagnóstico de premissas e recomendação de método, sobre a unidade independente. |
| `src/guaraci/vip_estabilidade.py` | VIP do PLS recalculado nas dobras da validação agrupada (frequência de VIP > limiar e intervalo por variável). |
| `src/guaraci/vip_cli.py` | vip_cli.py -- `guaraci vip`: VIP com estabilidade pela CLI (tabela, figura, relatório md/html/tex, PT/EN). |
| `src/guaraci/relatorio_blocos.py` | relatório .md/.html/.tex a partir de blocos neutros (usado por `anova` e `vip`). |
| `src/guaraci/catalogo.py` | catalogo.py -- catalogo declarativo unico do GUARACI. |
| `src/guaraci/catalogo_ajuda.py` | catalogo_ajuda.py -- dados do Assistente Guaraci (rotas, glossario, diagnosticos). |
| `src/guaraci/catalogo_dados.py` | catalogo_dados.py -- dados declarativos do catalogo (grupos, telas, tecnicas, |
| `src/guaraci/catalogo_duvidas.py` | catalogo_duvidas.py -- as duvidas mais comuns de quimiometria, respondidas por regra (dado declarativo). |
| `src/guaraci/catalogo_opcoes.py` | catalogo_opcoes.py -- opcoes de configuracao do catalogo (fonte unica de texto). |
| `src/guaraci/catalogo_tecnicas.py` | catalogo_tecnicas.py -- tecnicas e graficos do catalogo (dado declarativo). |
| `src/guaraci/catalogo_validacao.py` | catalogo_validacao.py -- registro de evidencias e selos (fonte do assistente |
| `src/guaraci/cli_comandos.py` | cli_comandos.py -- `guaraci --help` e subcomandos novos, derivados do catalogo. |
| `src/guaraci/confianca.py` | confianca.py -- Assistente de Confianca (texto puro, offline). |
| `src/guaraci/contexto.py` | contexto.py -- ajuste automatico por contexto (regras declarativas, offline). |
| `src/guaraci/csv_robusto.py` | Leitura de CSV tolerante ao que o Excel e a locale pt-BR realmente geram. |
| `src/guaraci/exemplos.py` | exemplos.py -- `guaraci exemplo <cenario>`: dado sintetico pequeno + config pronta. |
| `src/guaraci/icp_multielementar.py` | Perfil multielementar (ICP-OES / ICP-MS): tabela amostra x elemento. |
| `src/guaraci/monitorar_cli.py` | monitorar_cli.py -- `guaraci monitorar`: monitorar lotes de um processo (MSPC) pela CLI. |
| `src/guaraci/relatorio_cli.py` | relatorio_cli.py -- `guaraci relatorio`: gera os relatorios de uma execucao pela CLI. |
| `src/guaraci/transferir_cli.py` | transferir_cli.py -- `guaraci transferir`: transferencia de calibracao entre instrumentos (DS/PDS). |

*As 18 últimas linhas do mapa (módulos do CLI como centro de trabalho, do `__main__` ao `transferir_cli`) foram acrescentadas em 2026-10-05 a partir da primeira linha da docstring de cada módulo; não há script de regeneração do mapa inteiro (pendência registrada no livro-razão).*

Referências nos comentários (`Passo N`, `M-xx`, `CLAUDE.md` etc.): ver [`glossario_referencias_no_codigo.md`](../cientifico/glossario_referencias_no_codigo.md).
