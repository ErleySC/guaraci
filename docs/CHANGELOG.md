# Changelog — GUARACI

Histórico de versões do pipeline quimiométrico. Extraído do cabeçalho de
`pipeline.py` (a versão atual vive em `pipeline.__version__`).

> Ordem histórica original preservada como estava no código-fonte.

```
NAO LANCADO (pos-v31.9.0) — 2026-08-17 — CLI: persistencia de estado volta a
             funcionar -- 3 wrappers eram no-op SILENCIOSO (varredura de
             bugs). `_carregar_visual_cfg`, `_salvar_visual_cfg` e
             `_carregar_codigos_usuario` procuravam implementacoes em
             `cli_assistente` que NUNCA existiram la'; `getattr(..., None)`
             devolvia None e os wrappers caiam no fallback vazio. O proprio
             codigo ja registrava "esse esta quebrado ... fora do escopo
             desta feature consertar isso" desde 2026-07-13.
             Impacto 1 (menu Visualizacao): as 4 opcoes (Paleta/Fonte/Grid/
             Alpha) gravavam no dict, chamavam _salvar_visual_cfg(),
             imprimiam "OK Paleta: X" e NAO persistiam nada -- confirmacao
             falsa, valor de volta ao default na proxima abertura. Pelo
             mesmo caminho, `_sincronizar_dpi` e TODA a aplicacao de estilo
             em `_rodar_pipeline` (paleta/fonte/grid/alpha nos rcParams do
             matplotlib) liam {} e caiam nos defaults. O usuario ja tinha
             `~/.guaraci/visual_config.json` com paleta="publicacao"
             gravado desde 2026-06-01, ignorado desde entao.
             Impacto 2 (codigos de especie): o menu gravava certo
             (`_salvar_cod`) e listava certo (`_cod_usr`) -- as duas tem
             implementacao propria e sempre funcionaram -- mas a UNICA
             linha que injeta os codigos no pipeline
             (`pq.CODIGO_ESPECIE.update(...)`) usava o wrapper quebrado e
             recebia {}. Codigo cadastrado aparecia no menu e era ignorado
             na analise.
             CORRIGIDO: as tres leem/gravam direto em _USER_DIR, no mesmo
             padrao de `_cod_usr`/`_salvar_cod`. Novo `_VISUAL_PATH`
             (~/.guaraci/visual_config.json), incluido na migracao de
             estado legado -- inclusive a partir do cwd, porque a versao
             antiga gravava por caminho RELATIVO (era por isso que havia um
             visual_config.json na raiz do repo). Falha de escrita agora
             AVISA em vez de sumir. 3 testes novos.
             Varredura sistematica: enumerados TODOS os `_try(...)` e
             `getattr(_cli, ...)` de guaraci.py contra o modulo real --
             estes 3 eram os unicos ausentes; as constantes estao todas OK.

NAO LANCADO (pos-v31.9.0) — 2026-08-17 — Robustez: `zip(strict=True)` no
             pareamento mae_id<->metadados e cadeia de excecao preservada.
             (1) dados_io: `zip(mae_ids, meta_rows)` monta o array de
             grupos; as duas listas sao preenchidas em lockstep, entao
             comprimentos diferentes sao bug de programacao -- e um zip()
             normal esconderia isso TRUNCANDO em silencio, gerando um
             mae_arr mais curto que X e deslocando o grupo de todas as
             amostras seguintes. Como mae_id e' justamente o que impede
             vazamento de replica, o erro passaria despercebido. Agora
             estoura. (Auditados os outros 23 `zip()` sem strict do pacote:
             todos com comprimento garantido por construcao --
             np.unique(return_counts=True), labels=lb.classes_, listas
             preenchidas no mesmo laco -- nao alterados, para nao virar
             churn com risco de quebra.)
             (2) config_io: `raise RuntimeError(...) from _e_yaml` no
             ImportError do PyYAML -- sem o `from`, o traceback dizia
             "During handling of the above exception, another exception
             occurred", sugerindo falha no tratamento de erro em vez de
             dependencia ausente.

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — DD-SIMCA: Q de treino sai da
             escala in-sample na figura de aceitacao (achado A1, o achado
             original da Fase A que faltava aplicar). `fit()` calibra
             q0/Nq/f_crit a partir de `Q_train` LEAVE-ONE-OUT, mas
             `score_matrix()` -> `_t2_q()` recalculava Q IN-SAMPLE para
             TODAS as linhas, inclusive as de treino -- e uma amostra de
             treino reconstroi a si mesma de forma otimista, porque ajudou
             a definir a PCA que depois a reconstroi. A figura de aceitacao
             plotava entao os pontos numa escala e a fronteira noutra.
             Medido (scripts/medicoes/medir_ddsimca_loo_vs_insample.py,
             p=8192, 40 seeds/celula): Q in-sample e' **10 a 15x menor** que
             o LOO no regime real (nc=3-4 puros/classe) -- em eixo log,
             mais de uma decada de folga visual inventada.
             IMPACTO NA DECISAO, medido e declarado: **0,0%** dos pontos de
             treino mudam de lado da fronteira com nc=3-4 (sobe a 7-12% com
             nc>=6). E' defeito de fidelidade da FIGURA, nao de numero --
             consistente com o golden test nao ter mudado com esta
             correcao (a especificidade vem de amostras adulteradas, que
             nao estao no treino, e `predict()` nao passa por
             `score_matrix`).
             CORRIGIDO: `score_matrix(X, mask_treino=..., y=...)` usa o
             Q_train LOO armazenado para as linhas de treino, mantendo
             in-sample para amostras novas (que e' o valor CORRETO para
             elas -- nao participaram do ajuste). Guarda de desalinhamento:
             se X nao confere com o usado em fit(), mantem in-sample e
             AVISA, em vez de trocar Q pelas linhas erradas. 2 testes novos.

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — dados_io: unificacao de mae_id da
             Andiroba + exclusao de espectro com pureza indeterminada
             (achados A2-1 e A2-2, segunda parte).
             A2-1: a regra `mae_id = cod + data` separava em dois grupos as
             replicas puras de Andiroba lidas em datas diferentes --
             um grupo com (T2,T3) e outro com (T1), triplicatas
             COMPLEMENTARES somando exatamente {1,2,3}. Duas amostras
             fisicas distintas teriam cada uma sua propria triplicata (ou
             ao menos ambas comecando em T1); a complementaridade exata e'
             assinatura de UMA amostra com leituras separadas. Unificado
             via `_ALIAS_MAE_ID`.
             RESSALVA REGISTRADA NO CODIGO: as duas datas estao muito
             apartadas, o que e' incomum para replicas tecnicas e enfraquece
             a hipotese. A unificacao foi aplicada mesmo assim porque e' a
             escolha CONSERVADORA nas duas hipoteses: se e' a mesma amostra,
             corrige o vazamento; se sao distintas, agrupa a mais, reduz o
             n efetivo e retira a Andiroba do rol de especies com LOGO
             estimavel -- deixa de reivindicar um numero, nunca inventa um.
             NAO unificar e' que era arriscado: mantinha a Andiroba como a
             UNICA especie com LOGO calculavel, calculado sobre um par que
             pode ser a mesma amostra. Confirmar no caderno de coleta
             continua valendo.
             A2-2 (2a parte): o espectro sem metadado recuperavel -- unico dos 7
             orfaos sem informacao recuperavel do TITLE. Entrava com
             conc=0.0, isto e', como PURO, contaminando o treino one-class
             de uma classe cujo conjunto puro tem pouquissimos espectros -- uma
             fracao de contaminacao alta o bastante para inviabilizar o
             modelo one-class. Excluido via
             `_TITLES_PUREZA_INDETERMINADA`, com aviso NOMINAL no log --
             nunca em silencio. Nao e' regra generica de "TITLE ilegivel ->
             descartar": um TITLE ilegivel cuja pureza seja recuperavel do
             nome do arquivo continua sendo carregado.
             Verificado contra o dataset de desenvolvimento: orfaos 1 ->
             **0**, 100%% dos arquivos parseados, e a especie que aparentava
             ter duas amostras puras passa a ter um unico mae_id com
             triplicatas. Contagens em documentacao local, nao publicada.
             1 teste novo.

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — Etapa 4: sPLS-DA passa a usar o
             soft-thresholding da referencia (achado B1-2, correcao
             completa). A docstring ja havia sido corrigida antes; agora a
             IMPLEMENTACAO tambem: `sparse_plsda_mask` fazia truncamento
             DURO (`argsort(|w|)[:keep]`, zerando o resto sem encolher as
             sobreviventes) onde Le Cao et al. (2008) definem a esparsidade
             por penalizacao com soft-threshold
             `w_j <- sign(w_j)*max(|w_j|-lambda, 0)`. Implementado com
             lambda = o (keep+1)-esimo maior |w|, o que preserva a
             parametrizacao por CONTAGEM (mesma ideia do `keepX` do
             mixOmics) e ao mesmo tempo encolhe as sobreviventes -- e' o
             encolhimento que muda a direcao normalizada de w, logo o
             escore t, logo a deflacao, logo o conjunto escolhido pelos
             componentes seguintes. Divergencia previamente medida entre as
             duas variantes: Jaccard 1,000 com 1 componente (identicas por
             construcao) caindo a ~0,87 com 5. 1 teste novo (cardinalidade
             exata + coincidencia com top-k no 1o componente).

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — Monte Carlo CV: falha por poucos
             grupos vira aviso explicito (achado B2-1b, encontrado ao
             RETRATAR o B2-1). `_stratified_group_shuffle_splits` levanta
             ValueError quando n_grupos_teste < n_classes (exigencia do
             StratifiedShuffleSplit) e a chamada em `monte_carlo_cv` nao
             estava protegida -- num dataset com poucos grupos de replica
             por classe o MC CV inteiro morria e o usuario so' via a
             mensagem generica do except amplo do pipeline, sem saber a
             causa. Agora a condicao e' detectada ANTES, com aviso que diz
             o numero de grupos, o de classes e o que fazer (reduzir
             classes, aumentar test_size, ou coletar mais amostras
             fisicas). Nao afeta o dataset de desenvolvimento, que tem folga de mais
             de uma ordem de grandeza.

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — Figuras: marca d'agua de
             PROTOTIPO no modo imagem (residual do B4-1). Os relatorios
             PDF/Word/LaTeX ja saiam carimbados, mas uma figura .png
             exportada solta da pasta Graficos/ circulava sem contexto
             nenhum -- e' justamente o arquivo que acaba colado num slide
             ou num texto. `salvar()` e' o ponto unico por onde TODA figura
             passa, entao a marca entra la' e cobre as ~30 figuras de uma
             vez. 1 teste novo (verifica a propriedade -- texto de aviso
             presente na figura -- nao pixels).

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — Etapa 4: iPLS entra no nested-CV
             e a CV interna das buscas vira group-aware (achados B1-1 e
             B1-3 da auditoria de modulos nao auditados).
             B1-1: o bal.acc do iPLS reportado na tabela comparativa era o
             MAXIMO de `ipls_n_intervalos` avaliacoes feitas na MESMA
             particao de CV que depois reportava o numero (selecao_ipls
             escolhia o melhor intervalo sobre `cv_indices`; etapa4
             reavaliava o vencedor na mesma `cv_indices`). Vies de
             maximo-de-N. O comentario do modulo justificava excluir o
             iPLS do nested-CV porque "a particao em intervalos NAO usa
             rotulo" -- verdade, e irrelevante: a ESCOLHA do melhor
             intervalo usa. Medido: **+0,070 pontos de balanced accuracy,
             positivo em 12/12 seeds** (
             scripts/medicoes/medir_selecao_variaveis.py). O agravante era a tabela: os
             outros 6 metodos ja passavam por nested-CV, e
             `etapa4_selecao_variaveis` elege automaticamente o metodo
             "mais parcimonioso dentro de 1% do maximo" -- o vies era 7x
             o criterio de desempate, favorecendo sistematicamente o iPLS.
             Corrigido com `_mask_melhor_intervalo` (escolha do intervalo
             refeita a cada fold, so' com dados de treino) via
             `_avaliar_subset_nested_cv`, o mesmo caminho de VIP/SR/
             sPLS-DA. A busca no dataset inteiro continua rodando 1x para
             a figura/CSV de diagnostico por intervalo (mesmo padrao ja
             usado por SPA/AG).
             B1-3: `_cv_local` -- a CV INTERNA que guia a fitness do AG e
             a pontuacao do SPA -- usava sempre `StratifiedKFold`, entao
             replicas do mesmo `mae_id` caiam em treino e validacao da
             particao que escolhe as VARIAVEIS. A justificativa no codigo
             ("so' orienta a otimizacao; o numero reportado usa o fold
             externo group-aware") estava metade certa: o numero e' de
             fato honesto, mas o produto cientifico da Etapa 4 nao e' o
             bal.acc -- sao as variaveis selecionadas, e uma busca guiada
             por particao com vazamento prefere justamente as variaveis
             que exploram similaridade entre replicas. Corrigido:
             `_cv_local` aceita `grupos_local` e usa
             `StratifiedGroupKFoldEstavel` quando ha' >=2 grupos;
             `mae_id` propagado de `executar()` ate
             `etapa4_selecao_variaveis` -> `_avaliar_busca_nested_cv`.
             Sem `mae_id`, comportamento anterior preservado.
             4 testes novos em test_pipeline_core.py.

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — Relatorios: saida do modo imagem
             sai carimbada como PROTOTIPO (achado B4-1). O modo
             `modo="imagem"` (colorimetria digital) nunca foi validado com
             dataset real e `dados_imagem` devolve `mae_id=None` SEMPRE,
             entao o pipeline cai no fallback `StratifiedKFold` e a
             validacao group-aware -- o diferencial central do projeto --
             fica desligada. A unica mencao a "prototipo" vivia em
             docstrings e no texto de ajuda do CLI: nunca no caminho de
             execucao, de figura ou de relatorio. Um PDF gerado em modo
             imagem era tipograficamente identico ao de uma analise FT-NIR
             validada -- e, combinado com o B3-1 (corrigido na mesma
             sessao), afirmava ter usado GroupKFold.
             Corrigido em 3 camadas: (1) `executar()` grava "Modo de
             entrada" no resumo_modelo.txt (mesma fonte unica que o B3-1
             usa para o cv_label); (2) aviso de nivel WARNING (nao INFO) no
             inicio da execucao em modo imagem; (3) PDF, Word e LaTeX
             carimbam "PROTOTYPE OUTPUT - NOT VALIDATED" na CAPA (nao em
             nota de rodape), com texto de fonte unica
             (`_AVISO_PROTOTIPO_TITULO`/`_CORPO`) para os geradores nao
             divergirem. 1 teste novo em test_reports.py (confirma que o
             carimbo aparece SO' no modo imagem).

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — RETRATACAO de achado de auditoria
             (B2-1): a alegacao de que `monte_carlo_cv` produzia IC95%
             otimista ao descartar iteracoes sem todas as classes no treino
             **nao se sustentou na medicao e foi retirada**. O raciocinio
             original leu o `continue` isoladamente, sem confrontar com a
             garantia dada por `_stratified_group_shuffle_splits` 90 linhas
             acima: ele estratifica NO NIVEL DE GRUPO, entao ja garante
             toda classe representada em treino e teste, e a condicao do
             guard praticamente nao pode ocorrer. Medido (200 iteracoes por
             celula, classes deliberadamente pouco separadas p/ a BA nao
             saturar): **descarte de 0,0%** em todos os regimes viaveis,
             incluindo o regime do dataset de desenvolvimento, com BA
             variando de 0,86 a 1,00 entre celulas -- confirmando que o 0%
             nao e' artefato de problema facil demais. Script:
             scripts/medicoes/medir_monte_carlo_descarte.py. O guard e'
             defensivo, nao fonte de vies. Registrado como exemplo de que
             reverificar a propria auditoria e' obrigatorio (mesma licao
             da retratacao do q_residuos_limite no P11).
             Achado MENOR encontrado no lugar (nao corrigido, nao afeta
             este dataset): `_stratified_group_shuffle_splits` levanta
             ValueError quando n_grupos_teste < n_classes (exigencia do
             StratifiedShuffleSplit), e a chamada em monte_carlo_cv nao
             esta protegida -- num dataset com poucos grupos por classe o
             MC CV inteiro morre com mensagem generica. E' modo de falha
             RUIDOSO (o oposto do descarte silencioso alegado).

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — dados_io: amostras adulteradas
             deixam de entrar no dataset como PURAS (achado A2-2 da
             auditoria de gate 0). Inspecao manual dos arquivos que
             `carregar_dx` reportava como "isolated orphans" (TITLE nao
             casou com `_RE_TITLE`) achou algo mais serio que agrupamento
             perdido: parte deles tinha erro de digitacao no ##TITLE=
             (virgula extra antes do "%", digito de triplicata cortado ou
             duplicado) que quebrava o parse -- e o fallback por nome de
             arquivo TAMBEM falhava em extrair o teor, entao o arquivo
             entrava no dataset com conc=0.0/puro=True. Amostras
             adulteradas estavam contaminando o conjunto "puros" usado
             para treinar o DD-SIMCA one-class das classes afetadas.
             CORRIGIDO com `_CORRECOES_TITLE_CONHECIDAS`: tabela explicita
             (nao regex generico, de proposito) reconhecendo os TITLEs
             malformados por correspondencia EXATA, cada um verificado
             lendo o proprio ##TITLE= (nao adivinhado). Os erros eram de
             digitacao no campo -- virgula extra no teor, digito de
             replicata ausente ou duplicado, marcador de replica faltando.
             A tabela vive fora da arvore versionada: especie, teor e
             defeito de cada arquivo sao metadado de amostra. Um deles
             NAO foi corrigido de proposito -- convencao de nome
             diferente, zero informacao recuperavel do TITLE; permanece
             marcado como pendencia de verificacao manual contra o
             caderno de coleta.
             Verificado contra o dataset de desenvolvimento (nao so'
             sintetico): titles nao-conformes caem, concentracoes
             extraidas sobem, mae_id parseados sobem e os orfaos isolados
             caem para um unico caso. Contagens em documentacao local,
             nao publicada. 1 teste novo em test_pipeline_core.py.

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — DD-SIMCA: limiar calibrado por
             AMOSTRA FISICA (mae_id), nao por espectro (achado F1/A2-3 da
             auditoria de gate 0). `DDSimca.fit()` calculava h0/q0/Nh/Nq
             (os graus de liberdade que definem a regiao de aceitacao,
             Eq. 3-4 de Kucheryavskiy/Rodionova/Pomerantsev 2024) a partir
             de `len(Xc)` -- os ESPECTROS de treino tratados como
             observacoes independentes. Com 3 replicas tecnicas (T1/T2/T3)
             da MESMA amostra fisica, isso e' o mesmo vazamento de replica
             que o projeto existe para impedir, cometido no proprio
             calculo do limiar que decide aceitacao/rejeicao: ruido de
             replica era lido como se fosse informacao sobre variabilidade
             ENTRE amostras.
             Medido no dataset de desenvolvimento (auditoria A2, 2026-08-16): quase
             toda classe tem exatamente 1 mae_id puro independente -- ou
             seja, para praticamente todo o dataset, Nh/Nq nao-degenerados
             (>1) vinham inteiramente de ruido de replica, nunca de
             variabilidade real entre amostras.
             CORRIGIDO: `fit()` aceita `mae_id` opcional; quando presente,
             h0/q0/Nh/Nq sao estimados da MEDIA de T2/Q por `mae_id`, nao
             por espectro. Com 1 grupo (regime real da maioria das
             especies), Nh=Nq=1.0 (o minimo honesto, mesmo raciocinio do
             P1: numero mais largo/conservador substituindo confianca
             espuria, nao um defeito). Propagado ao pipeline (`ddsimca.fit`
             em pipeline.py) e aos dois consumidores internos que fitam um
             DDSimca temporario (`sensibilidade_ddsimca_logo`,
             `sensibilidade_ddsimca_pcv`) -- sem isso, a "estimativa
             honesta" do LOGO mediria aceitacao contra um limiar com o
             MESMO vies que o LOGO existe para corrigir.
             Sem `mae_id` (None): comportamento anterior preservado, com
             aviso explicito no log e `calibrado_por_amostra=False`
             exposto no modelo/score_matrix -- necessario quando nao ha'
             identificador de replica (ex.: modo_entrada="imagem", achado
             B4-1). Figuras de aceitacao (fig_sprint3_ddsimca_acceptance,
             fig_ddsimca_individuais) passam a mostrar `n_grupos_calibracao`
             ao lado do limiar (mesmo criterio de aceite do P1: nunca
             mostrar limiar sem dizer com quantas amostras fisicas
             independentes ele foi calibrado).
             IMPACTO NUMERICO MEDIDO (golden test, dataset sintetico):
             especificidade Esp_A 38,1%->23,8%, Esp_C 25,0%->12,5%,
             n_desconhecidos 14->8 -- a regiao de aceitacao ficou mais
             larga/permissiva, na direcao esperada (menos confianca
             espuria = menos rejeicao "confiante"). Qualquer numero de
             especificidade/sensibilidade DD-SIMCA de execucoes anteriores
             a este commit foi calibrado pelo metodo antigo (por espectro)
             e precisa ser reexecutado antes de ser citado.
             44 testes em test_classificadores.py (5 novos, propriedade
             matematica: duplicar replicas da MESMA amostra nao deve
             inflar o limiar quando mae_id esta disponivel; sem mae_id,
             1 amostra fisica com varias replicas produz Nh/Nq
             espuriamente nao-degenerados).

NAO LANCADO (pos-v31.9.0) — 2026-08-16 — Relatorios: template LaTeX nao
             afirma mais "group-aware cross-validation (GroupKFold)"
             CRAVADO no texto (achado B3-1 da auditoria de modulos nao
             auditados). O pipeline pode cair para StratifiedKFold quando
             mae_id esta indisponivel (mesmo caso do achado acima, ex.:
             modo_entrada="imagem") -- nesse caso o manuscrito gerado
             continuava alegando validacao group-aware que nao rodou
             naquela execucao. `gerar_latex_template` agora le
             `Group-aware (mae_id)` e `Validacao` do resumo_modelo.txt real
             (mesmos campos gravados por pipeline.py) e condiciona o texto:
             afirma GroupKFold so' quando de fato usado, com o `cv_label`
             real; caso contrario, avisa explicitamente que a protecao
             NAO foi aplicada naquela execucao. Faixa espectral e numero
             de permutacoes tambem deixaram de ser constantes cravadas
             (\SIrange{4000}{10000} e "200 permutations" fixos no template)
             -- agora interpolados do resumo real, ja que ambos sao
             configuraveis (cfg.wn_min/wn_max, cfg.n_permutacoes).
             Auditoria dos runs ja usados no material do TCC (run_N1/N2/N3
             .log + 6 resumo_modelo.txt em resultados_tcc/): todos com
             "Group-aware (mae_id): sim" -- nenhuma figura/tabela ja citada
             veio de execucao em fallback. O defeito era so' no TEXTO do
             LaTeX gerado, nao nos numeros ja obtidos.
             Teste de regressao: gera o LaTeX duas vezes (resumo com
             group-aware sim/nao) e confirma que o texto muda de fato, nao
             so' que compila (tests/test_reports.py).

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
             (ver nota de divulgacao adiada em
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
             Relatorio completo: AUDITORIA_SEGURANCA_2026-08-07.md.
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
             Relatorio completo: AUDITORIA_METODOLOGICA_2026-08-07.md.
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
             [NOTA de 2026-08-19 -- a entrada acima fica como registro
             historico, mas nao descreve mais o estado atual: os dois
             depositos no Zenodo foram retirados PELA CONTA DONA em
             2026-08-04, motivo 'duplicate' (21311868 = v31.1.0;
             21313436 = v31.1.1). O concept DOI devolve HTTP 410 e
             /versions devolve total: 0 -- nao resolve mais para versao
             nenhuma. Badge e link foram removidos dos READMEs e do
             CITATION.cff; ver PR #16 para o estado atual da citacao.]
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
v14 — 2026-05-28 — FINDING: MSC->SG+MC foi o melhor preset no dataset
             entao em uso (metrica retirada em 2026-08-18: derivada de
             dataset institucional, fora do escopo publico do software)
                   vs autoscaling (metrica retirada em 2026-08-18 --
                   dataset institucional fora do escopo publico) (AUTO advantage was
                   artifact of 80% subset). Changes:
                   (1) preset "msc_sg_mc" in construir_preprocessador;
                   (2) preprocessamento_padrao default = "msc_sg_mc";
                   (3) frac_holdout default = 0.20;
                   (4) gerar_nome_saida case "msc_sg_mc" -> "MSC-SGd-MC";
                   (5) M1: stars -> circle with black edge (avoids cluttering dense plots);
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
