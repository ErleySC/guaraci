# Changelog — GUARACI

Histórico de versões do pipeline quimiométrico. Extraído do cabeçalho de
`pipeline.py` (a versão atual vive em `pipeline.__version__`).

> Ordem histórica original preservada como estava no código-fonte.

```
NAO LANCADO (pos-v31.9.0) — 2026-08-07 — Seguranca: fecha bypass da
             mitigacao de RCE via pickle no app web (CRITICO) + 2 achados
             menores.
             [AUDITORIA DE SEGURANCA] GUARACI_DISABLE_MODEL_UPLOAD=1
             (mitigacao documentada em SECURITY.md p/ deploy publico)
             desabilitava APENAS o uploader de .joblib, deixando um
             segundo caminho de entrada pelo qual um visitante remoto NAO
             autenticado conseguia fazer o servidor carregar um pickle
             escolhido por ele -- RCE, apesar da mitigacao estar
             corretamente configurada. Passo a passo omitido de proposito
             enquanto a correcao nao estiver implantada no deploy publico
             (ver nota de divulgacao adiada em docs/auditoria/
             AUDITORIA_SEGURANCA_2026-08-07.md); permanece no historico
             do Git p/ quem precisar auditar. Corrigido em 2
             camadas: (1) campo de caminho local tambem oculto quando
             upload_bloqueado=True -- nesse modo a aba Predicao nao
             carrega nada pela web; (2) nova app_logic.caminho_upload_temp()
             isola uploads por sessao (uuid aleatorio via st.session_state)
             em vez de caminho fixo compartilhado -- fecha a
             previsibilidade e corrige de brinde uma condicao de corrida
             real entre sessoes concorrentes. Achado menor (BAIXA):
             os.system(f'open "{pasta_run}"') ao abrir a pasta de
             resultados do `guaraci demo` -- pasta_run e' sempre gerado
             internamente (nao explora'vel hoje), mas e' o padrao que vira
             injecao de comando real se um dia alimentado por input do
             usuario; trocado por subprocess.run() com lista de
             argumentos, que nunca passa por shell.
             Relatorio completo: docs/auditoria/AUDITORIA_SEGURANCA_2026-08-07.md.
             701 testes passam (697 + 4 novos), ruff limpo.

NAO LANCADO (pos-v31.9.0) — 2026-08-07 — CLI: estado do usuario sai do
             diretorio de instalacao do pacote.
             [_CFG_PATH / _USER_DIR] config.yaml, perfis/, flags de
             idioma/modo e codigos_usuario.json eram gravados DENTRO do
             diretorio onde guaraci.py esta instalado -- quebra em
             qualquer instalacao read-only (pip de sistema, Docker,
             `pip install --user` em alguns casos). salvar_config() logo
             antes de rodar o pipeline nao tinha NENHUMA guarda contra
             isso, derrubando o CLI com PermissionError no pior momento
             possivel. Movido para Path.home()/".guaraci". Migracao
             automatica e' best-effort (nunca sobrescreve, nunca apaga a
             origem), chamada uma vez no inicio de main() -- nao na
             importacao do modulo, pra nao escrever no HOME de quem so'
             esta importando (ex.: testes). Verificado com o ambiente real
             do autor: config.yaml/.cli_modo_usuario/perfis/ migrados com
             conteudo identico, arquivos antigos intactos. De brinde,
             achado um gap de isolamento pre-existente num teste (escrevia
             de verdade dentro do checkout do pacote a cada rodada).
             697 testes passam (693 + 4 novos), ruff limpo.

NAO LANCADO (pos-v31.9.0) — 2026-08-07 — Testes: spectra_preview.py
             cobertura 0% -> 94%.
             Modulo de previa de espectros da UI web (abas Data/
             Preprocessing) nunca tinha teste. 12 testes cobrindo
             preview_espectros_dx (estrutura multi-pasta, pasta vazia,
             arquivo .dx corrompido excluido sem derrubar os demais,
             reamostragem p/ grade de referencia diferente),
             preview_espectros_csv (colunas nao-numericas, coluna de
             classe ausente) e plot_espectros_media (inversao de eixo com
             wavenumber decrescente). 693 testes passam (681 + 12 novos).

NAO LANCADO (pos-v31.9.0) — 2026-08-07 — Performance: MSC.transform
             vetorizado (forma fechada, sem loop de lstsq).
             A regressao de 2 parametros (a, b tal que X_i~a+b*ref) por
             AMOSTRA usava np.linalg.lstsq num loop Python -- e' regressao
             linear simples, que tem forma fechada (b=Cov(ref,X_i)/
             Var(ref)), resolvida p/ todas as amostras de uma vez.
             Verificado numericamente identico ao lstsq por amostra (20
             casos aleatorios + estruturados, diff<1e-8); medido 1.5x mais
             rapido em escala real do projeto (934x8192). Unica mudanca de
             comportamento, documentada e testada: referencia de treino
             com variancia ~0 (nao ocorre com dado real) -- antes dava a
             solucao de norma minima do SVG (artefato sem significado
             cientifico), agora cai no mesmo fallback ja usado p/ b~=0.
             681 testes passam (678 + 3 novos).

NAO LANCADO (pos-v31.9.0) — 2026-08-07 — print() -> logging nos 2 modulos
             do nucleo cientifico que ainda faltavam.
             P6 (2026-07-13) migrou pipeline.py; a tabela ESTADO ALEGADO
             do CLAUDE.md afirmava (nunca reverificado com grep correto,
             sem excluir falsos positivos de console.print()) que "os
             demais modulos ja usavam logging". Nao era verdade:
             chemometric_stats.py e validacao_estatistica.py -- 2 dos 4
             modulos do nucleo -- tinham 10 print() ao todo (chamadas de
             progresso do teste de Wold/permutacao + avisos de taxa de
             falha). Como esses 2 caminhos so' rodam de dentro de
             executar() (que ja chama log.py:configurar() antes de
             qualquer coisa), a saida em producao fica identica -- so'
             passa a ser roteavel/silenciavel. 678 testes passam (sem
             novos, so' migracao).

NAO LANCADO (pos-v31.9.0) — 2026-08-07 — CI: matriz de teste reduzida em
             PRs (cota de minutos do Actions esgotada).
             Multiplicador de minutos do GitHub Actions: 1x Linux / 2x
             Windows / 10x macOS. A matriz cheia (10 combinacoes, incl. 2
             macOS) rodava por INTEIRO a cada push de PR. Em
             `pull_request`: 3 combinacoes (Ubuntu 3.10/3.13 + Windows
             3.11, sem macOS). Em `push` p/ master/main (uma vez por
             merge): matriz cheia mantida. Selecao via
             `github.event_name == 'pull_request' && fromJSON(...) ||
             fromJSON(...)` no matrix.include.

NAO LANCADO (pos-v31.9.0) — 2026-08-07 — CLI: 2 bugs de robustez achados
             num "checkup geral" de interface pedido explicitamente.
             [BUG DO PROGRESSO] A etapa "[6/7]" (figuras+DD-SIMCA+OPLS-DA+
             holdout) concentra a maior parte do tempo real de execucao,
             mas so' tinha 2 marcadores de texto OPCIONAIS entre inicio e
             fim -- progresso ficava CRAVADO em 6/7=0.857 durante toda a
             fase (medido: 96.1% das amostras de progresso presas nesse
             numero antes da correcao, 31.1% depois). progresso_do_log()
             ganhou parametro opcional total_figuras_planejadas: quando a
             etapa atual e' a 6, soma bonus fracionario proporcional a
             figuras ja salvas -- retrocompativel (None preserva
             comportamento antigo exato).
             [EOF INFINITO] main() girava para sempre (chamando
             os.system("cls") a cada iteracao) quando stdin chegava a EOF
             permanente (pipe fechado, sessao SSH caindo, automacao
             alimentando sequencia fixa de comandos) -- _input() engolia
             EOFError internamente e devolvia "", que nunca bate com
             nenhuma opcao de menu, entao o try/except que JA existia ao
             redor da leitura nunca disparava. Reproduzido: >350 redesenhos
             em 8s sem terminar. Corrigido trocando por input() direto
             nesse UNICO ponto, deixando o EOFError propagar ate' o
             handler que ja existia.
             677 testes passam (672 + 5, bug do progresso) / 678 (+1, EOF).

NAO LANCADO (pos-v31.9.0) — 2026-08-07 — Auditoria metodologica do nucleo
             cientifico: 5 achados (A1-A5), mesma classe do bug do P1.
             [A1, CRITICO] Teste de permutacao/Wold permutava rotulos por
             AMOSTRA, ignorando mae_id -- apos embaralhar, um mesmo grupo
             de replica fisica ficava com rotulos diferentes, impossivel
             sob H0. Medido: falso positivo de 15.0% contra 5% nominal (12
             grupos x 3 replicas, 120 repeticoes). O mais grave: atinge o
             argumento central do projeto (validacao group-aware) -- o
             teste que produz o p-valor citavel nao era group-aware.
             Corrigido: _gerar_permutacoes_rotulo() permuta a atribuicao
             de rotulo ENTRE grupos (Winkler et al. 2015), preservando
             coerencia de mae_id.
             [A2, CRITICO] Selectivity Ratio usava o peso PLS w1 em vez do
             vetor de regressao normalizado b/||b|| (Rajalahti et al.
             2009) -- so' coincidem com 1 LV. Medido: corr(t_tp,yhat) --
             a propriedade que define o metodo -- caia de 1.000000 p/
             ~0.92 com >=2 LVs; SR congelado na resposta de 1 LV p/
             qualquer numero de LVs; Jaccard@20 do ranking = 0.39. Usado
             por selecao_variaveis.py p/ SELECIONAR variaveis -- metodo
             anterior escolhia um conjunto diferente do que a literatura
             escolheria.
             [A3, ALTA] Dominio de aplicabilidade usava a MESMA regra
             retangular (T2<=UCL E Q<=UCL, alpha independente por eixo) ja
             corrigida no DD-SIMCA (P1, 2026-08-08 -- ver acima). Medido:
             rejeicao de 11.6% contra 5% nominal em amostras da propria
             distribuicao do treino. Usado em producao por predicao.py.
             Corrigido por REUSO: media_e_dof_momentos()/distancia_
             combinada() extraidas do DD-SIMCA p/ chemometric_stats.py,
             compartilhadas em vez de reimplementadas pela 3a vez.
             [A4, MEDIA -- decisao do autor] OPLS-DA multiclasse construia
             o alvo continuo via LDA(X,y) -- nao e' o metodo publicado
             (Trygg & Wold 2002 definem OPLS p/ y binario/continuo; a
             extensao multiclasse publicada e' OPLS/O2PLS com Y
             multi-coluna via PLS2). Trocado pelo caminho publicado:
             OPLSDAWrapper._alvo_continuo() usa o 1o escore Y de um PLS2
             ajustado em (X,Y).
             [A5, BAIXA] Docstring de hotelling_t2_limite contradizia a
             propria referencia citada (afirmava validade em Fase I
             quando TYM 1992 define Fase I via Beta, nao F). Corrigida.
             Retratacao registrada no proprio relatorio: alegacao inicial
             sobre q_residuos_limite (atribuicao a Jackson & Mudholkar por
             engano) verificada como FALSA apos busca adicional -- a
             atribuicao ja existente estava correta, nenhuma mudanca de
             codigo para esse item.
             Relatorio completo: docs/auditoria/AUDITORIA_METODOLOGICA_2026-08-07.md.
             672 testes passam (663 + 9 liquidos), ruff e mypy limpos.

NAO LANCADO (pos-v31.9.0) — 2026-08-08 — DD-SIMCA: diagnostico robusto
             (mediana/MAD) de replicas de treino atipicas.
             [OUTLIERS ROBUSTOS] Terceiro item da pesquisa de "novas
             tecnologias" pedida: Kucheryavskiy/Rodionova/Pomerantsev
             (2024) recomendam explicitamente estimadores ROBUSTOS
             (mediana/IQR) para DETECTAR outliers no treino, revertendo
             para estimadores classicos so' depois de removidos. Dado que
             este projeto opera com nc=3-4 amostras puras (excluir uma so'
             por suspeita pode derrubar o modelo inteiro abaixo do minimo
             de graus de liberdade), a decisao de escopo foi deliberada:
             `_outliers_robustos_mad()` (z-score modificado, Iglewicz &
             Hoaglin 1993) SO' SINALIZA -- nunca remove automaticamente.
             Verificado empiricamente que funciona no cenario real (2
             replicas proximas + 1 divergente -> divergente sinalizada) e
             documentado honestamente que a uniao dos 2 eixos (T2 e Q) tem
             ~10% de falso positivo mesmo em n=20 (medido: 3/30 seeds) --
             o proprio T2/Q_train ja e' instavel com so' 2 graus de
             liberdade residuais, entao o aviso deve ser lido como "vale
             conferir", nunca como "esta errado".
             `n_train` do modelo continua o numero ORIGINAL de amostras
             sempre -- testado explicitamente que nenhuma e' removida.
             Exposto em score_matrix() (`outliers_treino`) e no resumo
             (`DD-SIMCA {classe} AVISO treino`).
             663 testes passam (eram 657), ruff limpo; mypy limpo nos 7
             modulos puros do gate de CI (pipeline.py tem debito de
             tipagem pre-existente, fora do escopo, confirmado identico
             antes/depois via git stash).

NAO LANCADO (pos-v31.9.0) — 2026-08-08 — DD-SIMCA: diagnostico complementar
             por Procrustes Cross-Validation (PCV), opt-in via
             cfg.ddsimca_pcv.
             [PCV] Pesquisa de literatura atualizada (pedido explicito:
             "busque por novas tecnologias") achou Kucheryavskiy/Zhilin/
             Rodionova/Pomerantsev (2020) Anal. Chem. 92(17):11842-11850 e
             Pomerantsev/Rodionova (2021) Talanta 226:122104 ("Procrustes
             Cross-Validation of SHORT datasets in PCA context" -- mesmos
             autores do DD-SIMCA, atacando exatamente o problema de poucas
             amostras puras deste projeto). Integrado via pacote opcional
             `prcv` (extra [robusto] novo em pyproject.toml). Nova funcao
             `sensibilidade_ddsimca_pcv()` gera um "PV-set" por reamostragem
             e reporta sensibilidade sobre ele, ao LADO do LOGO (nunca em
             vez dele).
             Caveat cientifico verificado empiricamente, nao suposto: com
             todas as replicas puras de uma classe pertencendo ao MESMO
             grupo mae_id (n_grupos=1, o caso mais comum neste dataset), o
             PV-set so' reproduz ruido de MEDICAO (T1/T2/T3 da mesma
             amostra), nunca variacao entre amostras fisicas diferentes --
             PCV nao fabrica a informacao que falta, nenhuma tecnica de
             validacao fabrica. Testado tambem que passar o split de CV
             agrupado por mae_id quando so' existe 1 grupo faz `pcvpca`
             falhar (ValueError de shape) -- corrigido com fallback para
             leave-one-out por amostra individual nesse caso, unica
             estrutura possivel quando nao ha' mais de 1 grupo a proteger.
             O aviso reportado deixa esse limite explicito sempre que
             n_grupos<2, para o numero nao ser lido como equivalente ao
             LOGO.
             Wiring completo: campo em Config/_CONFIG_SPEC, menu CLI
             (menu_modelagem) E aba do app web (modelo.py) -- os testes de
             alcancabilidade de campo (test_interfaces_configuraveis.py,
             AST-based reachability em test_guaraci_cli.py) pegaram os 2
             pontos que faltavam antes do commit.
             657 testes passam (eram 651), ruff e mypy limpos.

NAO LANCADO (pos-v31.9.0) — 2026-08-08 — DD-SIMCA: regra de decisao corrigida
             para a distancia combinada do metodo publicado (fecha P1
             residual do CLAUDE.md).
             [REGRA RETANGULAR -> DISTANCIA COMBINADA] predict() aceitava um
             objeto se T2<=UCL(T2) E Q<=UCL(Q) independentemente -- uma
             regiao retangular. O docstring da classe ja documentava isso
             como divergencia do metodo citado (Rodionova/Pomerantsev), mas
             sem a formula exata para corrigir. Pesquisa de literatura
             atualizada achou Kucheryavskiy, Rodionova & Pomerantsev (2024)
             J. Chemometrics 38(7):e3556 -- tutorial dos proprios autores do
             DD-SIMCA com as Eq. 3-4 exatas: distancia combinada
             f=(T2/h0)*Nh+(Q/q0)*Nq comparada a UM UNICO f_crit=chi2(1-alpha,
             Nh+Nq), com Nh/Nq estimados DOS DADOS por metodo dos momentos
             (a mesma matematica que chemometric_stats.q_residuos_limite ja
             usava so' para Q -- estendida agora para T2 tambem, unificando
             os dois eixos sob o "data-driven" que da nome ao metodo).
             Com alpha independente por eixo a rejeicao conjunta efetiva era
             ~1-(1-alpha)^2~=0.0975 (quase o dobro do alpha=0.05 declarado)
             -- medido num caso sintetico controlado: regra antiga aceitava
             93.85% dos pontos de uma distribuicao conhecida (deveria ser
             ~95%), regra nova aceita 96.80%; 2.95% dos pontos MUDAM de
             classificacao entre as duas regras (nao e' so' um campo novo
             sem uso). A regra estava duplicada em 3 lugares (predict(),
             sensibilidade_ddsimca_logo(), especificidade no pipeline) --
             unificada numa so' fonte de verdade (score_matrix() agora
             expoe "f"/"f_crit", os 3 usos comparam contra eles).
             Figuras (fig_sprint3_ddsimca_acceptance, fig_ddsimca_
             individuais) atualizadas: a "caixa" de duas linhas retas
             perpendiculares (T2_norm=1, Q_norm=1) nunca foi a regiao de
             aceitacao real do modelo -- agora desenham a reta diagonal
             unica que a distancia combinada de fato usa
             (_fronteira_ddsimca()), senao a figura continuaria mostrando
             uma fronteira diferente da que o codigo usa para decidir.
             Golden test regravado: especificidade/n_desconhecidos do
             cenario sintetico N2 mudaram (ex.: Esp_A 61.9->38.1%) --
             direcao esperada: a regra antiga super-rejeitava em geral
             (inclusive amostras da propria classe), inflando especificidade
             como efeito colateral; a regra correta aceita mais amostras no
             total (proprias e estranhas), entao a especificidade cai para
             um valor mais honesto.
             651 testes passam (eram 644), ruff e mypy limpos.

NAO LANCADO (pos-v31.9.0) — 2026-08-07 — Figuras: curva DET era uma reta sem
             significado, rotulos do biplot ilegiveis, painel de execucao
             apagava a tela, e diagnostico novo de faixa espectral.
             [CURVA DET ERRADA] `sklearn.metrics.det_curve` devolve os pontos
             em ordem de limiar CRESCENTE, o que deixa `fmr` DECRESCENTE.
             `np.interp` exige `xp` crescente e NAO ordena sozinho -- a
             interpolacao degenerava e devolvia `fnmr[-1]` constante para todo
             FMR > 0. Resultado: TODA figura DET gerada ate hoje era uma RETA
             HORIZONTAL, nao uma curva. Nenhum erro era lancado. O teste
             existente so' verificava que o arquivo .png existia, por isso o
             defeito sobreviveu. Extraida `interpolar_det()` como funcao pura
             + 3 testes de propriedade (monotonicidade, extremos, degenerado);
             verificado que o teste FALHA com o codigo antigo. A diagonal, que
             era rotulada "Ref. diagonal" (induzindo a leitura errada de que a
             curva deveria segui-la), agora e' identificada como a linha de
             EER, e o EER de cada classificador aparece na legenda.
             [BIPLOT ILEGIVEL] Dois defeitos somados: (a) o top-N por
             magnitude selecionava canais VIZINHOS da mesma banda (no espectro
             real: 5875/5883/5891/5899... = 2 bandas contadas 12 vezes) e (b)
             nao havia anti-colisao de rotulos, entao os numeros de onda saiam
             impressos uns por cima dos outros. Corrigido com
             `selecionar_loadings_distintos()` (separacao espectral minima +
             piso relativo de magnitude, para nao completar a cota com ruido:
             o titulo passa a mostrar "top-5" quando so' ha' 5 bandas reais) e
             `afastar_rotulos()` (agrupa em colunas por x e empilha em y,
             convergencia garantida em uma passada + linha-guia ate a seta).
             Uma primeira versao por repulsao par-a-par iterativa OSCILAVA e
             deixava 7 pares sobrepostos mesmo apos 120 iteracoes -- medido,
             descartado e substituido.
             [TELA PRETA] `figuras_concluidas`/`avisos_do_log` cresciam sem
             teto; numa corrida completa (26 figuras + varios avisos) o painel
             passava de 35 linhas num terminal de 24. O `Live` do Rich perde o
             controle do cursor quando o bloco nao cabe na janela: a tela fica
             preta com so' o cursor piscando, embora o calculo siga rodando
             normalmente por baixo. Painel limitado (4 avisos mais recentes +
             contador do que ficou de fora, lista de figuras truncada) e
             `vertical_overflow="crop"` como rede de seguranca. Medido: pior
             caso caiu de 35 para 22 linhas.
             [FAIXA ESPECTRAL] Novo `diagnosticar_faixa_espectral()`: separa
             regiao MORTA (sem sinal) de RUIDOSA (dominada por alta
             frequencia) via SNR entre componente suave e residuo, e sugere a
             faixa com sinal. Emite AVISO e entra no resumo_modelo.txt; NUNCA
             corta sozinho -- mudar a faixa muda o resultado, e a decisao e'
             do usuario. Verificado que nao da' falso positivo em espectro
             que usa a faixa inteira.
             [np.interp SEM ORDENAR — latente] Auditoria do mesmo tipo de bug
             achou 3 outros sitios sem ordenacao do eixo: dados_io (preenche
             NaN), predicao (aplica modelo a amostra nova) e spectra_preview.
             O ABB MB3600 grava numero de onda CRESCENTE, entao NAO afeta os
             resultados deste dataset -- mas um .dx de terceiro em ordem
             decrescente (convencao comum em FTIR) daria predicao errada em
             silencio. Corrigidos os tres.
             634 testes passam (eram 617), ruff e mypy limpos.

NAO LANCADO (pos-v31.9.0) — 2026-08-06 — UI: markup cru, vazamento de PT em
             EN, padronizacao de booleanos, reset por nivel, 7 campos
             inalcancaveis por qualquer menu, e limpeza de identificacao.
             [Sem numero de versao de proposito -- ver nota de 2026-08-05
             abaixo: proxima versao publicada sera v1.0.0.]
             Markup do Rich cru na tela: Text("[g]Yes[/g]") tratava a string
             como literal em vez de interpretar -- precisa ser
             Text.from_markup(). Afetava PT e EN, so' nao tinha sido notado
             porque o default de Config() tem os campos booleanos em False
             (ramo que nao tinha o bug). Junto: varredura completa por
             vazamento de portugues em modo EN ("Modo:", "(automatico)",
             rotulo de nivel sem traducao).
             Wold/CV-ANOVA rodavam SEM checar o objetivo cientifico
             (diferente do teste de permutacao, que ja tinha esse guard) --
             em Quantificacao, refaziam refits de CV com rotulos de CLASSE
             e escreviam metrica sem sentido no resumo. Corrigido com o
             mesmo guard do teste de permutacao.
             Campos booleanos (20 no total) padronizados para escolha
             numerada [1]=Sim [2]=Nao -- antes caiam em texto livre, e
             _coagir_valor tratava QUALQUER entrada fora de um punhado de
             palavras magicas como False SEM AVISO (digitar "y" virava
             False silenciosamente).
             Trocar "nivel" agora desliga automaticamente toggles inertes
             nesse nivel/objetivo (DD-SIMCA, OPLS-DA, Etapa4, Wold,
             CV-ANOVA, Martens, Benchmark, Monte Carlo, SHAP,
             Benchmark-regressao), espelhando a logica real de
             pipeline.executar()/modos_analise.py, com aviso explicito do
             que mudou.
             Achado sistemico: n_jobs_permutacao, teste_martens,
             benchmark_regressao, figuras_detalhadas,
             imagem_incluir_textura, objetivo, selecao_ag e selecao_spa
             existiam no Config/_CONFIG_SPEC/HELP_DB, mas nunca tinham sido
             colocados em NENHUM menu -- so' editaveis a mao no YAML.
             Adicionados aos menus corretos. Teste sistemico novo varre
             todo o _CONFIG_SPEC via AST contra toda funcao menu_* para a
             classe inteira do bug nao voltar.
             Identificacao: exemplos de citacao do README (APA/ABNT/BibTeX)
             atualizados para bater com o que o software gera desde a
             remocao do branding institucional (ver nota de 2026-08-05
             abaixo); contradicao de copyright corrigida em README.md/
             README.pt-br.md/COMMERCIAL.md (autor + instituicao
             vs "o autor retem integralmente o copyright" no mesmo
             documento -- ficava so' com Erley, conforme decisao ja
             registrada no CLAUDE.md); CITATION.cff perde o bloco
             preferred-citation.institution (mantido affiliation:, que e'
             padrao e legitimo).
             617 testes (era 573), ruff limpo, mypy limpo.

NAO LANCADO (pos-v31.9.0) — 2026-08-05 — CORRECAO CIENTIFICA: particao de
             validacao cruzada estavel entre versoes do scikit-learn.
             [Sem numero de versao de proposito: a proxima versao publicada
             sera a v1.0.0, reiniciando o SemVer -- decisao do autor em
             2026-08-04. Nao criar tag v31.10.0.]
             O StratifiedGroupKFold do sklearn muda a particao entre versoes
             MESMO com random_state fixo. Medido com dados identicos (72
             amostras, 3 classes, 24 grupos de replica, random_state=42):
             42% das amostras caem em fold diferente entre sklearn 1.7.2 e
             1.9.0 -- 10 dos 24 grupos trocam de lado.
             Efeito: Q2, RMSECV, acuracia, F1, kappa e ate' o numero de LVs
             otimas dependiam da versao de biblioteca instalada. Para um
             projeto cujo argumento central e' validacao group-aware
             REPRODUTIVEL, era contradicao direta: os numeros de uma
             monografia mudariam ao reinstalar o ambiente.
             Correcao: StratifiedGroupKFoldEstavel (validacao_estatistica.py)
             -- mesma heuristica gulosa de estratificacao com grupos, com a
             ordenacao fixada por hash blake2b determinístico do id do grupo
             + seed. Estavel entre versoes de Python/numpy/sklearn e entre
             plataformas. Verificado: hash da particao IDENTICO em sklearn
             1.7.2 e 1.9.0 (antes, dois hashes distintos).
             ATENCAO: numeros de validacao cruzada de versoes anteriores NAO
             sao comparaveis com os desta -- a particao mudou uma vez, de
             proposito. Reexecutar antes de citar.
             Achado pelo golden test novo (tests/test_golden_valores.py), que
             trava 26 valores numericos do pipeline; no run sintetico so' o
             Q2 denunciava, pois as demais metricas estao saturadas em 1,0.
             Tambem nesta versao: correcao do `guaraci doctor`, que reportava
             fpdf2 como ausente com o pacote instalado (nome de modulo `fpdf`
             != nome do pacote pip) e exibia "rich ?" por rich nao expor
             __version__; launcher da area de trabalho, que chamava o Python
             do PATH (sem dependencias) em vez do venv; e requirements-lock
             .txt, cuja primeira versao era ININSTALAVEL (llvmlite 0.36.0,
             que exige Python <3.10, junto de numpy 2.5.1).
             573 testes (era 562), ruff limpo, mypy limpo.

v31.9.0 — 2026-08-04 — CORRECAO CIENTIFICA no DD-SIMCA + itens de comunidade
             JOSS. Achados de auditoria adversarial (2026-07-19) sobre a
             correcao LOGO da v31.1.x -- os dois bugs abaixo so' se
             manifestam quando o modelo e' testado em dado que nao viu, ou
             seja, a re-substituicao anterior jamais os revelaria:
             (1) o "small-n guard" elevava T2_ucl/Q_ucl ate max(estatistica
                 de treino) sempre que nc < 20 -- caminho SEMPRE ativo em
                 producao (ddsimca_treinar_em="puros" forca nc=3-4). Efeito:
                 alpha efetivo = 0, aceitacao de 100% do proprio treino por
                 construcao. Removido;
             (2) Q_ucl estruturalmente enviesado para baixo com n<<p: Q_train
                 era in-sample, e com poucos puros e milhares de variaveis a
                 PCA reconstroi o proprio treino quase exatamente, colapsando
                 o UCL a ponto de rejeitar QUALQUER amostra retida. Medido:
                 4 grupos estatisticamente identicos davam sensibilidade LOGO
                 0,0. Corrigido com residuo leave-one-out
                 (DDSimca._q_residuals_loo); o mesmo cenario passa a dar ~1,0
                 e um outlier real continua sendo rejeitado.
             ATENCAO: numeros de DD-SIMCA produzidos por versoes anteriores
             a esta nao sao comparaveis com os desta. Reexecutar antes de
             citar em texto.
             Achado nao corrigido (registrado, menor prioridade): a regiao de
             aceitacao e' T2<=UCL E Q<=UCL com alpha independente em cada, o
             que da alpha conjunto ~0,10, nao 0,05 -- nao e' o metodo de
             distancia combinada de Rodionova/Pomerantsev citado na docstring.
             Requer reescrever predict(), nao so' recalibrar um limite.
             Comunidade/empacotamento: CODE_OF_CONDUCT.md (Contributor
             Covenant 2.1), templates de issue (.yml) e de PR, classifiers
             trove + keywords em ingles + urls no pyproject.toml.
             Correcao de metadados: CITATION.cff estava travado em 31.1.1
             (a auditoria de 2026-07-13 reportou, erradamente, que estava
             sincronizado); READMEs e CITATION passam a usar o CONCEPT DOI
             do Zenodo (10.5281/zenodo.21311867), que sempre resolve para a
             ultima versao, em vez do DOI versionado da v31.1.1.
v31.8.0 — 2026-07-13 — MkDocs + GitHub Pages (item #12) e secao State of the
             field no paper JOSS (item #14):
             (1) mkdocs.yml novo: tema Material, plugin mkdocstrings (API
                 do nucleo cientifico: chemometric_stats, classificadores,
                 preprocessamento, validacao_estatistica, selecao_variaveis,
                 predicao), reaproveita docs/*.md existentes (nenhuma
                 duplicacao de conteudo -- so' SECURITY.md, que vive na raiz
                 do repo, e' copiado p/ dentro de docs/ NO CI, nunca
                 commitado, p/ nao divergir do original);
             (2) docs/index.md novo: landing page propria do site (nao e'
                 copia do README -- proposito diferente, texto proprio);
             (3) .github/workflows/docs.yml novo: builda com `mkdocs build
                 --strict` (link quebrado novo falha o job) e publica em
                 GitHub Pages via actions/deploy-pages;
             (4) paper/paper.md: nova secao "State of the field" comparando
                 com mdatools (Kucheryavskiy 2020), hyperSpec (Beleites &
                 Sergo) e pyChemometrics (Correia) -- as 3 citacoes foram
                 verificadas via busca antes de escrever (regra do
                 CLAUDE.md 0.2: nunca inventar referencia), nao geradas de
                 memoria. paper.bib ganha as 3 entradas correspondentes.
v31.7.0 — 2026-07-13 — Modo Iniciante/Avancado nos submenus da CLI (CLAUDE.md secao 6):
             (1) Toggle global [M] no menu principal (Modo: Iniciante/
                 Avancado), persistido em .cli_modo_usuario (mesmo padrao
                 de persistencia do idioma). Default: Iniciante;
             (2) `_print_submenu_compact` ganha `campos_avancados`/
                 `mostrar_avancado`: quando o modo e' Iniciante e o reveal
                 local nao foi ativado, esconde os campos marcados como
                 avancados e retorna a lista de campos REALMENTE exibidos
                 (a fonte da verdade p/ indexacao numerica -- callers
                 precisam usar essa lista, nao o `fields` original, senao
                 o numero digitado pelo usuario aponta pro campo errado
                 quando ha' campos escondidos). Compatibilidade: chamadores
                 que nao passam `campos_avancados` (menu_preproc,
                 menu_avancado, menu_visualizacao) continuam vendo todos os
                 campos, sem mudanca de comportamento;
             (3) revelacao LOCAL [V] por submenu (menu_modelagem via
                 `_loop_menu`, menu_validacao com loop proprio): expande
                 so' aquela visita ao menu, sem mudar o modo da sessao
                 inteira -- design pedido explicitamente pelo autor;
             (4) Campos escondidos por padrao: menu_modelagem ->
                 opls_da/ddsimca/modo_ddsimca/selecao_variaveis_etapa4
                 (nivel N2 ja forca DD-SIMCA automaticamente); menu_validacao
                 -> n_permutacoes/teste_wold/teste_cv_anova (testes extras,
                 tuning fino). menu_preproc (so' 2 campos), menu_avancado
                 (ja e' uma secao separada de modulos pesados) e
                 menu_tecnica/menu_codificacao (nao sao listas de
                 hiperparametro) foram deixados de fora, de proposito;
             (5) Verificado interativamente via CLI real (stdin scriptado):
                 modo Iniciante esconde 4/6 campos em Modelagem com o aviso
                 "[V] Mostrar opcoes avancadas (4 ocultas)"; [V] revela
                 tudo so' naquela visita; [M] alterna o modo global e a
                 proxima entrada no menu ja reflete os 6 campos completos.
```

```
v31.6.0 — 2026-07-13 — Cobertura do nucleo cientifico -> 95% (CLAUDE.md P4):
             (1) classificadores.py 93% -> 97%: testes de propriedade para
                 casos degenerados do NIPALS PLS1 (X todo-zero interrompe
                 sem divergir), da deflacao OPLS (componente ortogonal com
                 norma ~0), do fallback LDA->PLS2 quando a LDA multiclasse
                 falha (matriz de dispersao intra-classe singular --
                 forcado via monkeypatch, dificil de reproduzir
                 naturalmente), e de sensibilidade_ddsimca_logo com folds
                 sem dado suficiente (nao lanca excecao, fica inconclusivo);
             (2) validacao_estatistica.py 90% -> 95%: teste_permutacao e
                 teste_wold agora tem cobertura para o caso em que TODA
                 iteracao do loop falha (fold impossivel apos embaralhar
                 rotulos) -- p-valor vira 1.0 nao-informativo, slopes viram
                 NaN, nunca um calculo sobre lista vazia. Injetado via cv
                 fake deterministico (falha a partir da 2a chamada), nao
                 uma coincidencia estatistica fragil;
             (3) chemometric_stats.py (98%) e preprocessamento.py (100%)
                 ja estavam acima da meta antes desta sessao -- confirmado,
                 nao alterado. Cobertura TOTAL do projeto continua 64%
                 (fora do escopo desta rodada: dados_io/figuras/guaraci.py).
```

```
v31.5.0 — 2026-07-13 — print() -> logging em pipeline.py (CLAUDE.md P6, parcial):
             (1) 164 chamadas `print()` em `pipeline.py` migradas para
                 `log.info()` (`log = logging.getLogger(__name__)`).
                 `src/guaraci/log.py` novo: ponto único de configuração,
                 com handler que escreve em `sys.stdout` NO MOMENTO do
                 emit (não uma referência capturada na importação) --
                 necessário para continuar funcionando dentro do
                 `contextlib.redirect_stdout` que o CLI e o worker do app
                 web usam para capturar o log e alimentar o painel de
                 progresso ao vivo. Verificado com teste de integração
                 dedicado que roda o pipeline sintético de verdade e
                 confirma que os regex do painel (`app_logic.py`) ainda
                 casam com o texto capturado antes E depois da migração;
             (2) PARCIAL DE PROPÓSITO: o painel do CLI/app web continua
                 fazendo parsing de texto por regex, não consumindo
                 registros de logging estruturados -- essa reescrita
                 (a solução completa que o CLAUDE.md P6 propõe) é um
                 projeto à parte, não feito aqui. `log.info()` preserva o
                 mesmo texto que `print()` produzia, então resolve a
                 inconsistência entre módulos mas não a fragilidade de
                 fundo (mudar uma string ainda quebraria o painel).
```

```
v31.4.0 — 2026-07-13 — Preparação para submissão JOSS:
             (1) Benchmark contra dataset público externo (Tecator, NIR,
                 teor de gordura em carne — Thodberg 1996): roda o motor
                 real de pré-processamento + regressão PLS do GUARACI
                 (não uma reimplementação) no split oficial 172/43,
                 RMSEP 2,0-2,3% / R²pred 0,97-0,98, dentro da faixa
                 esperada da literatura. Script reprodutível
                 (`scripts/benchmark_tecator.py`, baixa o dado da fonte
                 original a cada execução) + write-up completo
                 (`docs/BENCHMARK_TECATOR.md`) com metodologia, resultados
                 e limitações honestas (não cobre DD-SIMCA/classificação/
                 group-aware — Tecator não tem réplicas físicas). Fecha a
                 lacuna citada em VALIDATION.md/MANUAL.md;
             (2) fix(web): mesmo bug de sincronização de widget do preset
                 (v31.3.0) também corrigido no botão pré-existente
                 "↺ Reload config.yaml" — extraído para
                 `_sincronizar_widgets_com_cfg()` compartilhada;
             (3) docs: `paper.md`/`CONTRIBUTING.md` sincronizados com a
                 contagem real de testes (550+, não 525+/498+);
                 `paper.bib` ganha a referência Thodberg1996.
```

```
v31.3.0 — 2026-07-13 — Correções da auditoria multidisciplinar de 15 etapas (2026-07-12):
             (1) BREAKING: Etapa 4 (seleção de variáveis) corrige viés de
                 seleção não-aninhada. VIP>=threshold, SR top-fração e
                 sPLS-DA calculavam a máscara de variáveis a partir de um
                 modelo ajustado no DATASET INTEIRO (double dipping —
                 Ambroise & McLachlan, 2002, PNAS) antes de avaliar por CV.
                 Agora usam nested-CV (`_avaliar_subset_nested_cv`,
                 `selecao_variaveis.py`): a máscara é recalculada a cada
                 fold usando só os dados de treino daquele fold. Resultados
                 numéricos de balanced_accuracy da Etapa 4 para esses 3
                 métodos NÃO são comparáveis com versões anteriores (tende
                 a cair, o que é o objetivo — número anterior era otimista).
                 iPLS não precisou de correção: a partição em intervalos não
                 usa rótulo, só a escolha do "melhor intervalo" usa CV (viés
                 padrão de qualquer seleção de hiperparâmetro, não double
                 dipping). `etapa4_selecao_variaveis()` não recebe mais
                 `vip`/`sr` pré-calculados como parâmetro;
             (2) menu "Visualização" da CLI (`guaraci.py`) tinha 4 opções
                 (H/M/B/V — heatmap espectral, matriz de confusão, biplot,
                 variância×wavelength) que sempre falhavam: apontavam para
                 funções que nunca existiram em nenhum módulo do projeto,
                 mascaradas por um `except Exception` genérico. Opções
                 removidas do menu (gerar essas figuras fora de uma
                 execução completa é feature nova, não bugfix — fica para
                 quando for implementada de verdade);
             (3) tooltip do assistente "G" (`GUARACI_TIPS["nivel"]`) não
                 seguia a convenção "nome amigável primeiro, código entre
                 parênteses" já usada nos rótulos de menu (P8) — corrigido;
             (4) CLAUDE.md sincronizado com o estado real do código (a
                 tabela "estado alegado" e a lista de problemas P1-P9
                 estavam desatualizadas desde a v31.2.0);
             (5) BREAKING: mesma correção de (1), agora para AG/SPA (Etapa
                 4, opt-in) — achado colateral mais grave que o de VIP/SR:
                 a fitness do AG e a pontuação do SPA usavam a MESMA CV do
                 número final reportado (double dipping por construção,
                 não só double dipping indireto via modelo pré-ajustado).
                 Corrigido com `_avaliar_busca_nested_cv`: busca refeita a
                 cada fold externo, usando só o treino daquele fold; custo
                 de execução sobe ~N vezes (N = nº de folds), aceitável
                 pois ambos já são opt-in/documentados como lentos;
             (6) nomes de pasta de saída (`PLSDA_OE_N2_...`) e de figura
                 (`figN3_heatmap_...`) não expõem mais N1/N2/N3 cru — slugs
                 amigáveis (`_NIVEL_SLUG_PASTA`, `config.py`):
                 `PLSDA_OE_Autenticacao_...`, `fig_heatmap_especie_adulterante.png`.
                 `cfg.nivel` continua "N1"/"N2"/"N3" internamente; só o nome
                 em disco mudou (P8 residual, decisão aprovada explicitamente
                 por ser mudança de formato de saída);
             (7) 3 presets por objetivo científico — "Explorar Dados" /
                 "Autenticar Pureza" / "Quantificar Teor" (CLI: `menu_perfis`;
                 app web: aba Dados) — reaproveitam `PROFILES`
                 (`cli_assistente.py`), mesma fonte usada pelos perfis de
                 rigor já existentes. CLI: aplicar um perfil agora pergunta
                 "Rodar agora?" e chama `_rodar_pipeline` direto. App web:
                 corrigido também um bug de sincronização de estado dos
                 widgets do Streamlit (`key=` estático só honra o valor novo
                 se escrito direto em `st.session_state[key]`, não bastando
                 apagar a chave) — sem essa correção os presets mudavam
                 `cfg_base` mas os widgets da aba Model continuavam
                 mostrando o valor antigo.
```

```
v31.2.0 — 2026-07-12 — Mudanças de COMPORTAMENTO CIENTÍFICO (CLAUDE.md P1/P2/P5):
             (1) BREAKING: sensibilidade DD-SIMCA deixa de ser re-substituição
                 (treino==teste, inflava para ~100%) e passa a ser estimada por
                 leave-one-group-out (LOGO) sobre mae_id. O dict de resultado
                 passa a expor `n_grupos` e um `aviso` quando n_grupos<10.
                 Resultados numéricos de sensibilidade gerados por versões
                 anteriores NÃO são comparáveis com esta versão;
             (2) heatmap espécie×adulterante (R²cv) passa a ser figura nativa
                 de `executar()` no objetivo Quantificação, com contagem de
                 combinações abaixo de R²cv=0.70 no título;
             (3) `predicao.carregar_modelo` passa a exigir `confiar=True`
                 explícito (joblib/pickle executa código arbitrário) e cada
                 modelo salvo passa a vir com manifesto SHA-256
                 (docs/SECURITY.md);
             (4) auditoria dos blocos `except Exception`/`except:` (P3):
                 maioria estreitada para o tipo real de erro; 1 bug real
                 corrigido (fallback silencioso LDA→PLS2 no OPLS-DA sem log);
             (5) 2 figuras novas: espectros médios por classe (±1 desvio) e
                 biplot PCA (scores+loadings);
             (6) vocabulário N1/N2/N3 aposentado da UI (P8; mantido como
                 apelido interno) — ver tabela de equivalência no MANUAL;
             (7) docs/VALIDATION.md e seção "Limitações conhecidas" no MANUAL.
```

```
v08  base: Sprints 1-3, GroupKFold mae_id, spectral truncation
v10  2026-05-28  max_lvs=30; ddsimca_n_components=7;
                   C2: comparar_pipelines uses max_lv=cfg.max_lvs (was min(8,..))
v11 — 2026-05-28 — C3: HCA dendrogram (Ward); C4: DD-SIMCA one-class
                   (trains only on pure samples, sens/spec); C5: N3 PLS reg GroupKFold
                   by mae_id; C6: T2 outliers per class in model summary
v12 — 2026-05-28 — M1: pure(*)/adulterated(o) markers in score plots;
                   M2: sens/spec in DD-SIMCA acceptance plot titles
v13 — 2026-05-28 — M3: chemical annotation of VIP bands; M4: accuracy per
                   class in resumo_modelo.txt
v14 — 2026-05-28 — FINDING: MSC->SG+MC = 0.923 bal.acc on full dataset
                   (1807) vs autoscaling 0.472 (AUTO advantage was
                   artifact of 80% subset). Changes:
                   (1) preset "msc_sg_mc" in construir_preprocessador;
                   (2) preprocessamento_padrao default = "msc_sg_mc";
                   (3) frac_holdout default = 0.20;
                   (4) gerar_nome_saida case "msc_sg_mc" -> "MSC-SGd-MC";
                   (5) M1: stars -> circle with black edge (avoids cluttering 1807pts);
                   (6) DD-SIMCA reverts to training on ALL samples (3 pure/class
                       makes one-class infeasible; requires >=15 pure/class)
v15 — 2026-05-28 — (1) holdout_preserva_puros=True: pure samples always in training
                       (fixes "pure=0" in 4 classes after holdout);
                   (2) automatic warning "LVs at ceiling" (console + summary);
                   (3) DD-SIMCA acceptance plot in LOG-LOG scale
                       (fixes data squeezed in corner; Pomerantsev standard)
v16 — 2026-05-28 — Organization/visualization:
                   (1) salvar() accepts subfolder; (2) fig3 Hotelling T2 in
                   log scale (Y) and T2vsQ in log-log (centers the cloud);
                   (3) score_contribution split into 2 figs (spectrum +
                   top-discriminant tall/readable with side legend);
                   (4) DD-SIMCA: 14 individual plots in ddsimca/ subfolder
v17 — 2026-05-28 — MAXIMUM PERCEPTUAL DISTINCTIVENESS color system:
                   (1) PALETTE Trubetskoy/Glasbey 20 colors (deltaE_min 27.4
                       vs ~15 before; eliminates 3 near-identical blues/2 greens);
                   (2) optional detection of glasbey/colorcet libs;
                   (3) SEQUENTIAL deterministic assignment (adjacent contrast)
                       replaces hash; (4) secondary SHAPE channel
                       (mapear_marcadores_classes, 14 shapes) for
                       colorblindness/B&W; (5) edge_para_cor by luminance
v18 — 2026-05-28 — Axis readability: _ticks_x_inteiros() applies
                   MaxNLocator(integer, nbins=10) when >15 ticks
                   (LV selection and PLS regression with 30-50 LVs no longer
                   overlap numbers); <=15 shows all values.
v19 — 2026-05-28 — V3 HCA/VIP:
                   (1) HCA on centroids in PCA(hca_n_pcs=65) — reduces
                       noise; (2) dendrogram axes inverted
                       (orientation=top: species on lower X axis colored
                       and rotated, distance on left Y axis);
                   (3) fig_hca_comparacao_pipelines: dendrogram panel
                       (raw/SNV/MSC/SG1/SG2/SNV+SG1/MSC+SG1/norm);
                   (4) automatic cluster interpretation (k=2);
                   (5) VIP: y-lim on real range + statistics box
                       (min/max/mean/std/n>=1) — checks real dispersion
                   + Config flags: mostrar_marcadores_classe/elipses_grupo
v20 — 2026-05-28 — Organization Q1: folder PLSDA_OE_{level}_{preproc}_
                   {YYYYMMDD_HHMMSS} with subfolders dados/ figuras/
                   modelos/ logs/. Figures->figuras/; metadata,
                   identifiers, comparison->dados/; summary->logs/;
                   final model (joblib: preproc+PLS+LB+wavenumbers)
                   ->modelos/. Sprint1 audit (A1,A2,A3,A5,A6,A11):
                   confirmed ALREADY implemented in previous versions.
v22 — 2026-05-29 — Phase 0 (rigor fixes):
                   B1: validar_entrada synchronizes mae_id with the SAME mask
                       for NaN/Inf removal (before, 1 NaN silently disabled
                       GroupKFold = replica leakage);
                   B4: DD-SIMCA 'todos' mode no longer reports misleading
                       in-sample "spec" (spec=n/a; mode label in
                       summary makes clear that sens/spec != authentication);
                   B7: Q-residual in summary with adaptive notation (:.4g
                       when <1e-3) instead of displaying 0.0000.
v21 — 2026-05-28 — STAGE 4 (variable selection) + class exclusion:
                   (1) Config.excluir_classes (e.g. Copaiba anomalous batch);
                   (2) iPLS (intervals), selection by VIP>=threshold, by SR
                       (top fraction), sPLS-DA (NIPALS soft-selection);
                   (3) single evaluator _avaliar_subset_cv (group-aware CV,
                       MC re-fitted per fold = no leakage);
                   (4) figures fig_etapa4_ipls_intervalos +
                       fig_etapa4_comparacao_metodos; CSVs in dados/;
                   (5) most PARSIMONIOUS method selected (bal.acc within
                       1% of max, fewer variables) in summary.
v24 — 2026-05-29 — Sprint v24: Publication Figures:
                   (1) fig_loadings_pca: PCA Loading Plot PC1/PC2 (bars
                       colored by sign, NIR inverted X axis);
                   (2) fig_roc_auc: Multiclass ROC curves OvR (scores
                       group-aware CV; macro AUC in title and summary);
                   (3) fig_splot_opls: OPLS-DA S-Plot (covariance x
                       correlation with t_pred; top-N annotated; colormap
                       RdBu_r; ref. Bylesjo 2006);
                   (4) fig_cooman_ddsimca: DD-SIMCA Cooman's Plot (pairs
                       A x B; sqrt(dQ) scale; subplot grid;
                       ref. Pomerantsev 2020).
                   Integration: aucs_roc added to resumo_modelo.txt.
v23 — 2026-05-29 — ACCESSIBLE LAYER (no code editing):
                   (1) _CONFIG_SPEC: single source mapping friendly names
                       <-> Config attributes, with type,
                       description and options for validation;
                   (2) salvar_config/carregar_config: commented YAML in
                       plain language; defaults preserved for missing keys;
                       unknown keys ignored;
                   (3) menu_interativo: terminal assistant (CMD-style)
                       to edit fields, save/load and run without opening
                       the code editor;
                   (4) new __main__: --rodar (uses config.yaml), --codigo
                       (legacy CFG), or interactive menu when in terminal;
                   (5) config.yaml template generated (excludes Copaiba
                       anomalous batch, max_lvs=40). Pipeline logic INTACT.
v27  benchmark_classificadores integrated into executar()
v28  Monte Carlo CV (IC95%); SHAP TreeExplainer; DET curves (linear+log)
v29  hardware_probe; auto RAM tiers (4 levels); RAM guards; cleanup util
v30  PowerPoint export; .streamlit/config.toml; CLAUDE.md; English i18n
v31  Bugfix: (1) iPLS/comparar_pipelines Q2 overflow guard (ss_res non-finite
             → q2=nan; eliminates -3.9e31 artifact in narrow intervals);
          (2) Wold permutation loop: filter non-finite r2/q2 before polyfit
             (fixes NaN intercept when degenerate model produces blow-up);
          (3) Wold fig: correct N/A status when intercept is NaN;
          (4) resumo_modelo.txt: Q2 NaN shown as "n/a" instead of crash;
          (5) config.yaml: N1 14-class, pasta dados/, ddsimca todos
```
