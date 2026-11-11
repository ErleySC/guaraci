# Changelog — GUARACI

**Público:** todos · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Histórico de versões do GUARACI, do mais novo para o mais antigo, no estilo
[Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/). A versão atual vive em
`guaraci.__version__` (hoje `1.0.0`). A política de compatibilidade está em
`docs/contribuir/compatibilidade_da_api.md`.

## Não lançado

Análise crítica do repositório público (2026-11-11):

- **Etapa principal que falha não termina mais como "concluído".** Se a regressão de uma Quantificação
  falha inteira, se o RMSEP do holdout externo não é calculado ou se o modelo de regressão não entra no
  pacote exportado, a execução segue para salvar o resto, mas o resumo final diz "Pipeline INCOMPLETO",
  a pasta ganha `ETAPAS_COM_FALHA.txt`, o `model_card.md` lista as etapas que faltaram e `guaraci run`
  sai com código 1. Antes isso era uma linha `[AVISO]` no meio do log e o código de saída era 0.
- **Documentação alinhada com a CI:** README, manual, guia de instalação e `limitacoes.md` diziam que
  macOS e Docker não eram verificados, mas a CI roda a suíte completa no macOS (3.11 e 3.13) e constrói
  a imagem Docker; o texto agora diz exatamente o que é conferido e o que não é. Saiu a menção ao
  Python 3.10, que não é mais suportado.
- **README:** a tabela comparativa passa a citar `mdatools`, scikit-learn e `chemotools` pelo que eles
  de fato oferecem, e há uma seção dizendo quando outra ferramenta é melhor escolha.

CI da publicação (2026-11-11): no primeiro CI do repositório público os testes passaram em todos os jobs
(nenhuma falha), mas o job fechava vermelho por configuração:

- **Segurança:** o piso do `pypdf` (extra `[dev]`, só usado nos testes) subiu de 6.1 para 6.21.4, a última
  correção de segurança apontada pelo `pip-audit`.
- **Testes na CI:** o job instala também o extra `[imagem]` (scikit-image + tifffile), o único que faltava,
  e o perfil `ci` da guarda de pulados aceita o pulo "bash utilizavel ausente" (até 8 testes), porque o
  runner Windows do Actions não tem a distro WSL.

Correções da auditoria de 2026-11-10, sobre o commit `11a3202`:

- **`guaraci transferir` (DS/PDS):** o `alpha` padrão deixou de ser 1,0 fixo e passou a ser escolhido por
  validação deixando uma amostra de transferência de fora. Em escala de absorbância o 1,0 zerava a
  transformação: o erro de predição depois de transferir igualava o desvio do teor, enquanto a tela
  mostrava melhora. Quem passa `--alpha` não é afetado.
- **LOD, LOQ e faixa de validação:** saem na unidade do teor de referência, sem o "%" fixo que o terminal, o
  resumo e o model card imprimiam mesmo quando o teor não era porcentagem.
- **Instalação:** nenhum texto manda mais instalar por nome pelo PyPI (o pacote não está lá e o nome pertence a
  outro projeto); a instalação é sempre `pip install ".[extra]"` de dentro do clone.
- **Avisos de classificação em quantificação:** acerto perfeito, p-valor de permutação, R2Y × Q2 e CV × holdout
  deixam de aparecer numa execução de quantificação (vinham do modelo auxiliar). Os nomes de arquivo `PLSDA` ficam.
- **RPD:** o código não atribui mais os cortes de interpretação a Williams nem à AACC (não conferidos).
- **Testes:** DS/PDS, VIP e Selectivity Ratio ganharam oráculos independentes (Ridge do scikit-learn, fórmulas
  fechadas) que matam os mutantes sobreviventes da auditoria.
- **Selectivity Ratio:** X é centrado dentro do cálculo. O resultado só mudava com pré-processamento sem
  centragem na média (preset `custom` sem mean-centering).
- **Linearidade:** quando o erro puro vem só de réplicas técnicas, a rejeição aparece como "não conclusiva"
  e a ressalva sai no model card e no console; antes dizia "curvatura detectada".
- **Documentação:** o README e o guia de instalação avisam que o nome `guaraci` no PyPI é de outro projeto;
  `docs/cientifico/validacao_numerica.md` foi reescrito.

**Saturação do detector (absorbância no teto)**
- Novo `guaraci.saturacao`: detecta o teto do detector (plato de valores idênticos, ex.: A = 3,0
  com cubeta de 10 mm em óleos) nos modos `dx`, `opus`, `spc`, `sp` e `csv`. Variável com 10 % ou
  mais das amostras no teto (alargada em 3 pontos de cada lado) e espectro com 20 % ou mais das
  variáveis no teto saem do modelo (nova opção `saturacao`: `auto` remove e registra, padrão;
  `avisar` só avisa; `nenhuma` não procura; independe de `ajuste_automatico`).
- O que saiu fica no aviso do terminal, em `Relatorios/saturacao.json` e na linha "Saturação do
  detector" do resumo. Dado sem plato no teto nunca é tocado.
- Medido em 1.874 espectros NIR de óleos (cubeta de 10 mm): 279 variáveis e 68 espectros fora.
  RMSEP do holdout 4,22 → 3,82 e R² 0,04 → 0,16; é ganho real, mas modesto (o limite é a faixa
  de teor de 1 a 15 %, não o software).

**Regressão (teor)**
- A CV da regressão não herda mais o número de dobras da classificação (limitado pela classe com
  menos grupos): usa `n_splits_cv` e só é limitada pelo número de grupos do próprio ajuste.

**Q-resíduo**
- `q_residuals_limit`: com cauda pesada (h < 1) e limite abaixo da mediana de Q, usa o percentil
  empírico em vez do ajuste chi² por momentos, que fazia quase toda amostra virar outlier Q.

## v1.0.0 — primeiro release público (2026-11-10)

Primeiro release coberto pela política de SemVer: mudar de forma incompatível a assinatura de
função pública, o formato de dataclass de resultado, o esquema de `config.yaml` ou o nome de
coluna de saída exige versão maior.

**Validação sem vazamento por padrão**
- Validação cruzada group-aware: réplicas da mesma amostra física nunca ficam em dobras de
  treino e de teste diferentes (CV, holdout estratificado por classe e grupo, bootstrap BCa,
  seleção de variáveis SPA/AG/CARS/iPLS).
- Seleção do número de variáveis latentes por CV aninhada: a métrica reportada não reusa os
  folds que escolheram o modelo.
- Unidade de agrupamento declarável (`auto`, `mae_id` ou `sessao`) e testes de permutação.
- **Padrão alterado:** `unidade_agrupamento: auto`. O GUARACI valida por sessão quando cada sessão tem uma
  classe só e isso é possível (pelo menos 3 sessões e toda classe em 2 ou mais sessões). Quando alguma classe
  tem uma única sessão, ele fica em `mae_id` e o log diz o que falta coletar. Se você escolher a unidade, ela
  nunca é trocada. A tela de Auditoria mostra o plano de coleta: por classe, quantas sessões faltam para um
  resultado honesto (5) e para a garantia conformal (19).

**Métodos**
- Classificação: PLS-DA, OPLS-DA (Trygg & Wold, 2002, sem ortogonalização extra; `predict`
  equivalente ao PLS1 com k+1 componentes), DD-SIMCA com limites estimados por grupo e
  sensibilidade leave-one-group-out, SVM/RF de referência.
- Quantificação: PLS-R com intervalos e RMSEP de holdout; transferência de calibração (DS/PDS).
- Exploração e interpretação: PCA, VIP, SR, SHAP, MCR-ALS, PARAFAC/N-PLS, ASCA, EPO/GLSW.
- Monitoramento: MSPC (T² e Q) e sentinela de deriva.
- Imagem hiperespectral: classificação por pixel com agregação por objeto físico.

**Entrada de dados**
- CSV/XLSX, JCAMP-DX e leitores de instrumento (OPUS, SPC, PerkinElmer `.sp`, RMN Bruker
  processado, cromatogramas Agilent/Waters, ANDI-MS, EEM), cubos ENVI e pastas de imagens.
- Metadados que identificam a amostra de origem nunca são gravados em disco; um teste varre os
  artefatos gerados.

**Interfaces**
- Terminal bilíngue (PT/EN) com menu inicial em seis grupos numerados, modo básico e avançado,
  ajuda por campo e rodapé de teclas único.
- App web Streamlit (modo servidor por padrão: só upload, pasta de sessão isolada, modelo só
  em JSON portátil; `guaraci web` liga o modo local).
- API Python com contrato público versionado (`tests/golden/contrato_api_publica.json`).

**Validação numérica**
- Conferência contra as fórmulas de definição e oráculo externo em R para Dual-sPLS (ver
  `docs/cientifico/validacao_numerica.md`). Limites conhecidos em `docs/cientifico/limitacoes.md`.

**Integração final (CLI v2, desempenho, assistente, métodos isolados)**

**CLI antes do lançamento**
- Quem abre o programa pela primeira vez vê a interface em inglês; a escolha feita com `I` continua gravada em
  `~/.guaraci`. Os testes de tela fixam o português por conta própria (`conftest.py`).
- Voltar sobe um nível só: a lista de técnicas especiais, o assistente e as telas de passos (seleção de
  amostras, predição, comparar, monitorar, HSI, relatórios, hardware) reabrem depois de rodar e só saem com
  voltar. Entrada encerrada (EOF) ou Ctrl+C conta como voltar, para a tela não reabrir sem fim.
- O número de uma tela avançada oculta abre a tela direto (antes pedia `[+]` e um segundo toque).
- Textos em português que apareciam no modo inglês: opções dos campos, faixas e erros de validação, paletas,
  grade, tipos de arquivo, perfis de matriz, ajuda, Sobre, `doctor`, níveis de hardware. Perguntas em inglês
  usam `(y/n)`. Nomes reais de pasta e coluna (`dados`, `resultados`, `classe`, `Amostra`) não são traduzidos.
- `docs_pacote.lista(idioma="PT")`: parâmetro novo, opcional; a chamada sem argumento não muda.
- `guaraci relatorio --formato all` vale o mesmo que `todos`.
- Painel de execução (travamento relatado): o `Live` do Rich trocava `sys.stdout` e, ao sair, devolvia o buffer do
  log da análise. O log escapava para a tela (painel parado em "0.0% Starting...") e, no fim, o painel de erro ou de
  conclusão e o "Enter para continuar" iam para o buffer: a tela parecia travada para sempre. Agora o `Live` não
  mexe em `sys.stdout` e a saída real é restaurada depois da análise.
- Classe com uma só unidade independente: antes de rodar, o menu oferece retirar essas classes (`excluir_classes`)
  em vez de "rodar mesmo assim" uma análise que para com erro de validação cruzada. A regra de dobras não diz mais
  "nada a corrigir" quando a menor classe tem 1 unidade.
  Se a falta de unidades só aparece depois do holdout, a análise levanta `ValidacaoCruzadaImpossivel` com as
  classes, e o menu oferece retirá-las e rodar de novo sem repetir as perguntas.
- Objetivo exploratório (só PCA/HCA/loadings): não reserva mais holdout e não para por classe com uma só unidade;
  os gráficos usam todas as amostras e classes. O PLS-DA interno usa divisão simples, marcada como não validada.
- Painel de execução: no Windows desliga a edição rápida do console enquanto roda (um clique na janela
  pausava a saída e parecia travamento); sem linha nova no log por 120 s, grava a pilha de todas as threads
  em `~/.guaraci/travamento_*.txt` e avisa no painel, sem interromper a análise.
- Tecla `?` com um painel só em toda tela (Tela/Campo, Onde, Faz, Use quando, Nao use, Impacto, Padrão, Faixa,
  Exemplos e a lista de opções com o que cada uma faz). Antes: algumas telas imprimiam uma linha cinza, outras
  uma lista solta, outras nada; o menu de grupo mostrava a ajuda da última tela aberta (a ajuda registrada não
  era limpa ao voltar). Menu principal, menus de grupo, telas de configuração (com as opções ocultas também),
  exemplo guiado, perfis, técnica, técnicas especiais, codificação, Sobre, hardware, assistente `G`, confiança
  e as escolhas internas (paleta, fonte, grade, método de seleção, extras HSI) explicam cada opção; `N` ou
  `?N` abre o detalhe. Faixas escritas em inglês no catálogo (“Valid system path”) e `True`/`False` aparecem
  em português na tela em português. O `?` pergunta primeiro o número ou a letra e mostra só aquela opção
  (Enter = painel da tela inteira), para não despejar a tela toda antes da pergunta.
  `tests/test_ajuda_unificada.py` percorre todas as telas, grupos e campos.

**Documentação**
- Valores padrão descritos como o que são: ponto de partida, não universais, ajustáveis por opção. Sai a frase
  "calibrados em dados de referência", que apontava para dados que não acompanham o programa (catálogo de
  confiança, auditoria de delineamento, MANUAL, limitações, artigo).
- README em inglês e em português reescritos; MANUAL com o Exploratório sem holdout, a retirada de classes com
  uma só unidade e o painel de execução (seções 1.1, 1.2, 2.2 e 10).

**Dependências**
- Pisos novos para fechar vulnerabilidades conhecidas apontadas pelo `pip-audit` (como na CI): Pillow 13.0 (cinco
  corrigidas na 13.0) e fpdf2 2.8.8 (duas); no lock, certifi 2026.10.23.
- **Python mínimo passa a ser 3.11.** O 3.10 saiu de suporte em outubro de 2026, e o Pillow 13.0 (com as correções
  acima), o numpy 2.3+ e o scipy 1.16+ já exigem 3.11. CI em 3.11, 3.12 e 3.13.

**Métodos isolados (sem ligação ao pipeline)**
- `robpca`: ROBPCA (Hubert, Rousseeuw & Vanden Branden 2005) em numpy/scipy. Conferido número a número
  contra `rrcov::PcaHubert` 1.7-5 em 9 conjuntos (`tests/oraculos_metodos/`); em 60 conjuntos aleatórios, 56
  coincidem e nos 4 restantes o próprio rrcov varia com a semente. Sem o modo `skew` do rrcov.
- `pcv_regressao`: Procrustes cross-validation para PLS (Kucheryavskiy, Rodionova & Pomerantsev 2023).
  Conferido contra o pacote R `pcv` 1.2.0 em 7 conjuntos; a parte ortogonal do pseudo-conjunto é
  aleatória nos dois e é conferida pela definição. Teste de verdade conhecida mostra o comportamento em um cenário.
  Sem PCR.
- `test_alcance_modulos`: tetos subiram de 3 para 5 (bibliotecas não expostas) e de 20 para 22 (total) para
  declarar esses dois módulos; reverter é apagar os módulos ou ligá-los ao CLI.

- Assistente: "Revisar configuracao" volta a avisar sobre espectro sem pre-processamento (raw), max_lvs acima de 40 e holdout fora de 15-35%, agora como regras do registro (valem tambem no pre-voo e em `guaraci checar`).

**Assistente**
- Registro único de regras (`assistente_regras`): cada regra lê configuração, resumo dos dados
  ou `resumo_modelo.txt`, tem severidade, causa, ação e referência com DOI. `sugerir` passou a
  usar o registro (parâmetro novo opcional `dados`).
- Pré-voo pela linha de comando: `python -m guaraci.assistente_checar config.yaml [--dados]
  [--resultado PASTA] [--ficha]`; sai com 1 quando há achado crítico.
- Códigos de erro estáveis `GUA-E001` a `GUA-E013` (`assistente_erros`), com causa e ação, casados
  com as mensagens reais do scikit-learn, do Python e dos leitores; a mensagem original nunca é
  escondida. `guaraci assistente GUA-E005` explica um código.
- Ficha anti-vazamento (`assistente_vazamento`) no formato de Kapoor & Narayanan (2023): o que o
  programa não sabe vira pergunta ao autor, nunca "ok".
- Conjuntos de avaliação de perguntas em `tests/perguntas_assistente/` rodados nos testes;
  nenhuma resposta direta errada é aceita.
- Busca BM25 com sinônimos (`assistente_busca`): ranqueia as dúvidas por relevância, recusa pergunta
  sem relação com o catálogo e só responde direto quando uma dúvida se destaca com folga; em dúvida,
  mostra lista. Quatro respostas diretas erradas medidas nos conjuntos cegos viraram acerto ou lista.
- Corrigido: o gatilho "erra" do sobreajuste casava "errada" (adjetivo); palavra solta só responde em
  pergunta curta ou de definição.
- Corrigido: negação ("Q2 não está baixo") deixava de casar com a dúvida errada; "o que é VIP"
  respondia com o gráfico em vez do termo.

Mudanças aditivas (versão menor pela política de compatibilidade).

**Desempenho e máquinas fracas**
- A barra de progresso não fica mais parada no fim da execução. A CV do domínio de
  aplicabilidade, que roda ao salvar o modelo, passou a registrar cada fold
  (`[7d/7] ... fold k/N`), e o painel mostra o contador. As permutações em paralelo também
  passam a mostrar o progresso.
- O Q leave-one-out (`q_residuals_loo`) usa a matriz de Gram quando n ≤ p, com os mesmos
  valores (diferença de ~1e-13). A CV do domínio ficou 31 vezes mais rápida em 243 × 1500.
- A seleção do número de VLs faz um ajuste de PLS por fold, e não um por VL e fold. As
  predições são as mesmas.
- Durante `executar()` o BLAS roda com 1 thread. Para outro valor, use
  `GUARACI_BLAS_THREADS` ou `OMP_NUM_THREADS`.
- `auto_adjust_hardware_config` limita as figuras a 300 dpi quando há menos de 6 GB de RAM
  livre. Também limita `n_jobs_permutation` aos núcleos físicos, com ~0,4 GB por processo.
  Nenhum número muda.
- Execução completa em dados sintéticos de 243 × 1500 com a configuração padrão: de 260 s
  para 55 s, com os mesmos valores do golden de pipeline.

**Pré-processamento**
- Remoção de raios cósmicos para Raman (`RemocaoRaiosCosmicos`; Whitaker & Hayes 2018,
  DOI 10.1016/j.chemolab.2018.06.009): chaves `remover_raios_cosmicos` (padrão `false`),
  `raios_cosmicos_limiar` (6) e `raios_cosmicos_janela` (5). Ligada, entra antes de
  qualquer preset. No cenário sintético `raman` o R²cv do teor volta de 0,80 para 0,91.

**Modo sintético realista**
- Novo `simulador_espectral` e chave `sintetico_cenario` (`nir`, `raman`, `rmn` ou
  `classico`). Os cenários realistas incluem:
  - mistura de Beer-Lambert com composição que varia entre amostras;
  - espalhamento e linha de base do tipo MSC/EMSC;
  - réplicas correlacionadas e efeito de lote;
  - ruído que cresce com o sinal;
  - espectros anômalos;
  - fluorescência e raios cósmicos no Raman, e deslocamento por pH no RMN.
- **Muda o padrão:** o modo sintético passa a gerar o cenário `nir`. Para o gerador
  antigo, use `sintetico_cenario: classico`.

**Tela [2.4] grava a unidade do eixo e a técnica (R6)**
- Novas chaves de configuração `eixo_unidade` (`cm-1`, `nm`, `raman_shift`, `min`, `ppm`, `ms`,
  `indice`; vazio = não declarada) e `tecnica_analitica`. A tela [2.4] as grava a cada técnica
  (e limpa a da técnica anterior; o Genérico não declara unidade). Só mudam rótulo, orientação
  e texto: nenhum número do modelo muda.
- Figuras: o eixo ganha o rótulo da unidade ("Wavelength (nm)", "Raman shift (cm-1)", "Retention
  time (min)", "Chemical shift (ppm)", "Drift time (ms)", "Variable index") em vez de "Spectral
  variable", e **só vai do maior para o menor em cm-1 e ppm**. **Mudança de padrão:** antes, todo
  dado em modo `csv` saía com o eixo invertido (NIR dispersivo, UV-Vis, fluorescência,
  cromatografia, IMS e Raman em CSV); agora esses saem crescentes. Sem unidade declarada
  (configs antigas) o comportamento antigo continua (rótulo neutro; orientação pelo modo de entrada).
  Um perfil de matriz em cm-1 cujo `rotulo_eixo` diz "Raman" também deixa de inverter o eixo.
- Resumo: `Faixa espectral (nm)` etc. com a unidade certa e `Tecnica (perfil)` com a técnica da [2.4]
  (antes `-`). Em `parametros_efetivos` entram `faixa_espectral` e `unidade_eixo`; o campo antigo
  `faixa_espectral_cm1` fica quando a unidade é cm-1 ou Raman (e quando nenhuma foi declarada) e
  some quando a unidade declarada é outra, porque ali o nome mentia.

**Raman (R7)**
- Novo preset `airpls_snv_sg_mc` (AirPLS, SNV, Savitzky-Golay, centragem na média). **Mudança de
  padrão:** a [2.4] do Raman passa a gravar esse preset (antes `airpls_sg_mc`) e
  `remover_raios_cosmicos: true`. Motivo: a intensidade Raman absoluta depende de potência, foco e
  densidade da amostra, e só o SNV a remove; no Raman sintético de quantificação (CV agrupada),
  R² do holdout 0,55 (padrão anterior) e R²cv 0,86 foram para 0,96 e 0,99. Sair do Raman na [2.4]
  desliga de novo a remoção de raios cosmicos; quem a ligou à mão sem passar pelo Raman não é afetado.
- A exportação portátil do modelo (`modelo_plsda.portatil.json`) passa a aceitar `airpls_sg_mc`,
  `airpls_snv_sg_mc` e a etapa de raios cosmicos (com os parâmetros usados no treino); a predição
  portátil bate com a do `.joblib`. O preset `custom` segue recusado.

**Formatos que dependem de pacotes opcionais (P6)**
- A tela [2.4] escreve `(extra: pip install ...)` ao lado do formato (OPUS, SPC, RMN Bruker, HPLC
  Agilent/Waters) quando o pacote não está instalado, avisa na hora da escolha e avisa dos recursos
  opcionais da técnica (PARAFAC na fluorescência 3D, textura na imagem). `guaraci doctor` e
  `guaraci checar` listam os extras ausentes e as técnicas que afetam; `guaraci checar` reprova
  (crítico) config cujo `modo_entrada` precisa de um pacote ausente. Os extras seguem opcionais;
  `pip install "guaraci-chemometrics[all]"` instala todos.

**Leitores (fase 2 do checkup)**
- **CSV genérico com cabeçalhos nomeados ou mistos** (`dados_io`): colunas com nome (`pH`, `composto_a`) viram variáveis
  nomeadas (eixo = posição 1..k), mesmo misturadas com cabeçalhos numéricos; metadado (`classe`, `grupo`, `conc`, `lote`)
  continua fora. Se os cabeçalhos numéricos formam um bloco espectral (monotônico e ≥ 4× as colunas nomeadas), o CSV segue
  sendo espectro e as colunas nomeadas são ignoradas com aviso, como antes. Os nomes aparecem nos eixos das figuras
  (até 40 variáveis; o eixo fica crescente, na ordem do arquivo) e no resumo do modelo.
- **GUA-E007 não diagnostica mais cabeçalho de CSV.** O texto do erro de "menos de 2 colunas de variáveis" ganhou o código novo
  `GUA-E014` (`GUA-E001` a `GUA-E014`); `GUA-E007` ficou só para faixa que não cruza o eixo.
- **Reamostragem do eixo avisada** (`dx`, `opus`, `spc`, `sp`, `rmn`, `hplc`, `gcms`): quando o eixo comum tem menos pontos que
  alguma amostra, o log e o resumo do modelo dizem "n original → n final", o método e o passo. Chaves novas:
  `espectro_max_pontos` (padrão 2000, o teto de sempre) e `cromatograma_max_pontos`.
- **Mudança de padrão (cromatogramas `hplc`/`gcms`): sem teto de pontos** (antes 2000). A decimação pontual por `np.interp`
  fazia a altura de um pico variar até 8,5 % entre injeções idênticas, conforme a fase da amostragem na grade; sem decimar, a
  variação fica abaixo de 1 % (teste). Cromatogramas de dezenas de milhares de pontos geram matrizes maiores: `cromatograma_max_pontos`
  restaura um teto.
- **Mudança de método (todos os leitores que reduzem o eixo):** com razão nativo/grade ≥ 2:1 a redução é média por intervalo, não
  interpolação pontual (sem aliasing: conserva a área de linhas mais finas que o passo). Abaixo de 2:1 e sem redução, o resultado é o mesmo de antes.
- **`cromatografia_canal`** (padrão vazio = comportamento atual): `hplc` lê todos os canais DAD do Agilent/Waters ou uma lista de
  comprimentos de onda; `gcms` lê o espectro de massas por scan agregado em unidades de massa (ANDI-MS, ASTM E1948) ou uma lista de m/z.
  Testado com `.uv` Agilent e ANDI-MS sintéticos; sem arquivo real de fabricante.
- `formatos_sinteticos.escrever_andi_ms` aceita `espectros=` (layout esparso do E1948); `dados_io.load_dx` aceita `max_pontos`.

**EEM (fluorescência 3D) e PARAFAC**
- Espalhamento Rayleigh/Raman de 1ª e 2ª ordem: novas chaves `eem_espalhamento` (`nenhum`|`nan`|`zero`|`interp`),
  `eem_espalhamento_largura_nm` (largura por ordem; padrão 30 nm) e `eem_raman` (Raman da água, 3400 cm⁻¹).
  O padrão **continua `nenhum`** (não muda resultados existentes); com EEM bruta de espalhamento forte e nada
  aplicado, a leitura avisa. Motivo: sobre a EEM bruta o PARAFAC não serve (erro de reconstrução 0,82,
  correlação escore × concentração ≤ 0,51; com `nan` nos dados sintéticos de verificação, correlação ≥ 0,99 e
  congruência de Tucker dos espectros ≥ 0,99).
- PARAFAC com dados faltantes (ALS por EM): a região removida não derruba mais colunas nem amostras; NaN já
  presentes nos arquivos EEM passam a ser lidos como dado faltante (antes todas as amostras sumiam). No
  classificador, a região mascarada vale 0 em todas as amostras e é descartada como constante, igual no
  ajuste e na predição.
- **PARAFAC não-negativo (HALS) desde a v1.0.0** (já era assim; agora está documentado e registrado):
  versões anteriores ou outros programas sem restrição podem dar sinais/ordens diferentes dos fatores; o
  CORCONDIA é reportado. A tela [3.5], `parafac_resumo.json`/`.txt` e a figura trazem uma nota em PT e EN, e o
  resultado grava `restricao` e o algoritmo. Nova chave `parafac_nao_negativo` (padrão `true`); `false` usa o
  ALS livre, com a nota correspondente.
- API: `parafac_eem` ganha `nao_negativo`, `mascara_faltantes`, `max_iter_em` e `tol_em`;
  `ParafacEEMResultado` ganha `restricao`, `algoritmo`, `n_faltantes` e `n_iter_em` (acréscimos, sem quebra);
  `rodar_parafac_eem` ganha `nao_negativo` e `mascara_faltantes`; novas funções em `eem_io`.

**RMN e IMS**
- **Muda o padrão da tela [2.4] para RMN:** escolher a técnica RMN passa a gravar
  `faixas_excluir: [4.5-5.0, -0.5-0.5]` (água residual e TSP/TMS), com aviso na confirmação, no
  checklist e no resumo; escolher outra técnica zera a chave. Motivo: sem a exclusão, a água (30 a
  400 vezes o sinal dos metabólitos) dominava o divisor do SNV e derrubava a acurácia de um conjunto
  sintético de 0,85 para 0,60. No motor geral (`config.yaml`, API) a chave é **vazia por padrão**:
  nenhuma execução existente muda.
- Nova chave `faixas_excluir` (lista de `"a-b"` na unidade do eixo), aplicada depois do corte de faixa e
  antes do pré-processamento. Vai no pacote do modelo (`.joblib` e `.portatil.json`) e a predição reaplica
  a mesma exclusão. Validação de formato (`a < b`), de unidade (faixa fora do eixo é recusada) e aviso
  quando tira mais de 50% da faixa; `guaraci checar` ganhou três regras (`faixas_excluir_fora_do_eixo`,
  `faixas_excluir_mais_da_metade`, `rmn_sem_faixas_excluir`).
- Novo preset nomeado `pqn_mc` (seletor *PQN+MC*, no fim da lista): PQN + mean-centering, exportável no JSON
  portátil. O padrão do RMN **continua** `snv_mc`: nos dados sintéticos, com a água excluída, SNV+MC foi
  igual ou melhor que PQN+MC em todos os cenários testados.
- Novas chaves opcionais `normalizacao` (`nenhuma`, `rip`, `area`; padrão `nenhuma`) e `ims_rip_janela`:
  normalização por espectro antes do pré-processamento, sem estado e reaplicada na predição. O padrão do IMS
  na tela [2.4] **não muda** (ver o manual, seção 5.12, para a evidência).
- Contrato da API pública (`tests/golden/contrato_api_publica.json`) regravado: só acrescenta as quatro chaves
  novas, os três campos novos da `Config` e a opção `PQN+MC`.

**HSI (imageamento hiperespectral)**
- Calibração radiométrica branco/escuro opt-in (`hsi_calibracao: branco_escuro`, `hsi_referencia_branco`,
  `hsi_referencia_escuro`): R = (I − D)/(W − D) com WHITEREF/DARKREF no padrão Specim (ao lado do cubo, na
  pasta da sessão ou por caminho), com recusa se W ≤ D, branco saturado, referência de outra câmera ou cubo
  sem referência. A mensagem de rejeição de cubos em DN aponta para a chave.
- Eixo de bandas comparado entre cubos (tolerância 1 nm): duas câmeras com o mesmo número de bandas deixam de
  ser misturadas sem aviso (recusa com mensagem); `hsi_eixo_bandas: reamostrar` interpola até meia banda.
- **Padrão alterado:** a validação interna do HSI passa a ser a validação cruzada por OBJETO sobre todos os objetos
  (`hsi_validacao: cv_objetos`, 5 dobras; `hsi_cv_dobras: 0` = leave-one-object-out) no lugar de um único holdout de
  20% (1 a 3 objetos de teste), com IC de Clopper-Pearson e número de objetos por classe, e a confiança por objeto
  passa a ser fora da amostra (antes era do modelo aplicado aos próprios objetos). Motivo: com 3 objetos de teste a
  sensibilidade por classe se apoiava em 1 ou 2 objetos. `hsi_validacao: holdout` volta ao comportamento anterior.
  O `hsi_relatorio.txt` ganhou linhas e passou a ser acompanhado de `hsi_relatorio.json`.
- Cubos lidos por `np.memmap` em blocos de linhas; controle de qualidade, segmentação (PC1 por covariância em blocos)
  e extração de pixels sem copiar o cubo para float64. Sem mudança de resultado nos dados de teste; menos memória.
- Arquivos `WHITEREF_*`/`DARKREF_*` deixam de ser lidos como amostras (antes entravam como cubos e eram rejeitados
  ou, pior, viravam objetos de uma classe).

**Imagem digital**
- `imagem_referencia` (cartão cinza/branco, coordenadas relativas): divide cada canal pela média do cartão, por
  foto, para tirar o efeito multiplicativo da iluminação da sessão. Desligado por padrão. Fotografe com o mesmo
  cartão em todas as sessões.
- `imagem_segmentacao: otsu | distancia_fundo`: estatísticas de cor só do objeto. Desligado por padrão.
- Aviso quando, sem segmentação nem recorte, o fundo ocupa mais da metade da foto ou a cor do fundo separa as
  classes. Parâmetros de recorte/cartão/segmentação gravados no modelo exportado (`imagem`).

**Relatórios (PDF, DOCX, XLSX, PPTX, LaTeX) e ficha anti-vazamento** (fase 2 do checkup)
- Quantificação (PLS-R): o relatório mostra as métricas de regressão (RMSEP do holdout externo e
  do interno, RMSECV, R2cv, RMSEC, R2cal, viés, SEP/RER/RPD com a faixa de uso, LOD/LOQ, tabela por
  espécie) com a base de cada número, e deixa de mostrar o molde de classificação (R2Y/Q2Y/R2X
  zerados, "Classes 1", "Balanced accuracy -"). Motivo: o PDF/DOCX tinha só o molde PLS-DA e as
  métricas de regressão só apareciam no XLSX. Classificação não mostra métricas de regressão.
  Execuções antigas continuam legíveis (o relatório lê o `resumo_modelo` como sempre).
- O pré-processamento aparece inteiro (por exemplo SNV -> centragem na média) em todos os
  formatos; antes a linha era cortada no primeiro passo. Muda o texto do relatório, não o modelo.
- `resumo_modelo.txt` ganha a linha `R2cv` (1 - PRESS/SS_tot sobre as mesmas predições de
  validação cruzada do RMSECV). Valor novo, nenhum existente muda.
- HSI: a ficha anti-vazamento tem ramo próprio e diz só o que o código HSI faz, lendo
  `hsi_relatorio.json` (o `.txt` nas execuções antigas): tipo de validação (CV por objeto, o padrão, ou
  holdout único), calibração branco/escuro quando ligada, eixo de bandas, IC de Clopper-Pearson, n de
  objetos avaliados, confiança por objeto fora ou dentro da amostra, SNV por pixel só se o preset tiver
  SNV; antes herdava MSC/EMSC, validação aninhada e holdout de 20% que o HSI não usa.
  `guaraci relatorio PASTA_HSI` gera PDF, DOCX, XLSX e LaTeX (PPTX não; a mensagem diz isso), com o
  tipo de validação, sensibilidade/especificidade com IC e n por classe. `hsi_relatorio.txt` ganha a
  linha "Aviso: ..." quando há menos de 10 objetos AVALIADOS fora da amostra (na CV são todos os
  objetos; no holdout, só os de teste): sensibilidade/especificidade são métrica indicativa.
- Multielementar (ICP): nova linha `Tipo=CRM_CERT` (valor certificado do material de referência, uma
  só) dá o denominador da recuperação do MRC; sem ela o relatório diz que a recuperação NÃO foi
  calculada, e o mesmo vale para brancos sem `Tipo=LD` e para o padrão interno ausente. Mais de uma
  linha `CRM_CERT` é erro. O arquivo do exemplo (`guaraci exemplo multielementar`) passou a trazer
  as linhas `ld`, `branco`, `crm` e `crm_cert` (as amostras e os resultados do modelo não mudam).
- Multielementar: PDF/DOCX/XLSX/LaTeX trazem a tabela de ANOVA por elemento (com a correção de
  comparações múltiplas), a sensibilidade à censura e os alertas de QC (deriva do padrão interno,
  branco acima do LD, recuperação do MRC); a faixa espectral e o SNR não aparecem.
- API pública (contrato regravado, só acréscimos): `resumo_parse.modo_do_resumo`, `parse_regressao`,
  `parse_hsi_relatorio`, `N_TESTE_HSI_INDICATIVO`; `assistente_vazamento.itens_hsi`;
  `relatorio_blocos.renderizar_tex_corpo`; campo `crm_certificado` em `TabelaMultielementar`.

Nos três blocos seguintes (P2, R14, R9) nenhum padrão mudou: tudo é aviso novo ou chave opt-in (sem a chave, o resultado é o de antes).

**Quantificação com mais de uma classe (P2)**
- Aviso (resumo, model card, console, PDF/Word/Excel) quando a acurácia balanceada da classificação
  que escolhe o modelo de teor fica abaixo de 0,80: a validação pode ficar instável (R2 agregado
  negativo numa varredura sintética) e o aviso indica a saída (`coluna_classe` constante).
- Nova chave `quantificacao_rotulo_treino: verdadeiro` (padrão `previsto`): treina o teor com a classe
  verdadeira e valida roteando pela prevista. Sintético: R2 de validação -2,17 -> 0,995.

**Classes ordenadas (R14)**
- Detecção objetiva de classes em rampa (colinearidade das médias de classe no espaço dos scores,
  espaçamento regular e queda de sensibilidade da classe do meio) com aviso no resumo e nos relatórios.
- Nova chave `classificador: plslda` (padrão `plsda`): LDA com encolhimento de Ledoit-Wolf sobre os
  scores do PLS, que resolve o mascaramento da classe do meio. Exportação portátil em JSON passa a
  `versao_formato` 3 quando o modelo é PLS-LDA (os arquivos antigos continuam lidos).
- Texto do exemplo `imagem`: o 0,88 vem do PLS-DA.

**Alinhamento de cromatogramas (R9, parte do pipeline)**
- Nova chave `alinhamento: cow` (padrão desligado; `alinhamento_segmentos`, `alinhamento_folga`):
  COW dentro de cada dobra da CV, com a referência escolhida só nos dados de treino da dobra; a mesma
  referência e os mesmos segmentos valem para o teste e para a previsão, e o alinhador vai no pacote
  exportado. Registrado no log, no resumo e no model card.
- O COW ficou vetorizado (nós idênticos ao da versão em laço, 25-40x mais rápido) e com cache.

**Pendências pequenas da fase 2 (nenhum padrão mudou)**
- `guaraci checar CONFIG --resultado PASTA_HSI` lê a pasta de uma execução HSI (que não grava `resumo_modelo.txt`) e a ficha
  descreve o que o HSI fez; uma pasta inexistente dá mensagem e saída 2, não traceback.
- Código novo `GUA-E015` (`GUA-E001` a `GUA-E015`): `faixas_excluir` ou `ims_rip_janela` fora do eixo, ou `faixas_excluir`
  que apaga todas as variáveis. Antes saíam como "Erro sem código conhecido".
- `guaraci exemplo hsi` grava `pre_processamento: snv_mc` e a faixa aberta (0 a 999999), como a tela [2.4]; a execução do
  exemplo não avisa mais que o preset não se aplica ao HSI.
- A prévia da web (aba Dados) não mostra as variáveis dentro de `faixas_excluir`, e diz quantas tirou.
- Os extras do HSI (tela [X]) leem os cubos com a mesma calibração branco/escuro e a mesma checagem do eixo de bandas do
  modelo principal. A faixa espectral e o SNV por pixel continuam sem ser aplicados nos extras.

**JCAMP-DX conferido com arquivos reais públicos (124 arquivos)**
- Comparado com o pacote `jcamp` nos 124 arquivos da suíte oficial de teste do JCAMP-DX e nos espectros de gás do NIST:
  58 batiam, agora 90. Três erros reais corrigidos no leitor `.dx`: (1) arquivos AFFN com os rótulos de abscissa das linhas
  deslocados em um ponto (o formato do NIST) perdiam um valor por linha e distorciam o espectro sem erro; quando os valores
  lidos somam exatamente `NPOINTS`, a ordem do arquivo manda e os rótulos só conferem; (2) comentário `$$` depois dos dados
  (`$$No of bytes of data = ...`, do INTTODX) fazia o arquivo ser lido como ASDF e falhar, e comentário `$$` num valor de
  cabeçalho quebrava a conversão; (3) rótulos com espaço antes do `=` (`##NPOINTS = 8192`) não eram reconhecidos e o
  espectro vinha vazio (a norma ignora espaços, hífens, barras e caixa no rótulo).
- Arquivo composto (vários blocos ou `##BLOCKS=`), tabela de picos (`##PEAK TABLE=`) e `##NTUPLES=` agora dão erro dizendo o
  que são, em vez de espectro misturado ou vazio. Um `.dx` com várias `##XYDATA=` antes lido como o primeiro bloco agora é recusado.
- Também conferidos com arquivos reais, sem diferença: SPC (11 arquivos, Galactic e Thermo), `.sp` do PerkinElmer e o RMN Bruker
  de teste do `nmrglue` (idêntico ao `nmrglue`). Sem arquivo real público para OPUS, Agilent e Waters: continuam só com arquivos
  sintéticos.

**Dados de quem usa o GUARACI pela primeira vez**
- O aviso "N arquivos com ##TITLE= fora do padrão" tratava a convenção de título opcional como se fosse a esperada. Agora, se
  **nenhum** arquivo a segue (o caso normal de quem tem os próprios dados), o log é um `[INFO]` que diz de onde vêm classe e
  grupo e como informar o grupo (`arquivo_grupos`); o aviso só aparece quando **alguns** seguem e outros não.
- A auditoria crítica `agrupamento` (sem proteção contra vazamento de réplica) passa a dizer como resolver: `arquivo_grupos`,
  `coluna_grupo` ou uma subpasta por amostra física. A tradução para o relatório em inglês acompanha.
- Manual: seção 4.1 "Como organizar os seus dados" (de onde vêm classe e grupo em cada modo, o `##TITLE=` opcional, o CSV de
  grupos). Corrigidas três afirmações: o modo `dx` não "sempre tem `mae_id`"; os órfãos só existem quando alguns títulos casam;
  e a referência a `tests/fixtures/{spc,sp,...}`, que não existem no repositório, virou a descrição da rodada local com arquivos
  reais (registro em `docs/qualidade/rodada_local_formatos_reais.json`).

**Instalação completa: `guaraci doctor --instalar`**
- O `doctor` já listava o que faltava; agora também instala. `guaraci doctor --instalar` (com `--sim` para não perguntar) monta o
  comando `pip` com os requisitos exatos do projeto (lidos dos metadados do pacote instalado, já filtrados pelo marcador de
  Python, por exemplo `rainbow-api` só para Python < 3.14) e instala o navegador (`web`), os relatórios (`reports`) e os extras
  de formato/recurso (OPUS, SPC, RMN, HPLC, PARAFAC, textura, PCV) que ainda não estão no ambiente. O `benchmark` (xgboost, shap;
  ~1 GB) fica de fora de propósito: só serve à comparação com outros algoritmos.
- Se o GUARACI roda direto da pasta de fontes (sem metadados), o comando imprime `pip install -e ".[extras]"` em vez de adivinhar.
- README e `docs/usuario/instalacao.md` passam a instalar `".[web,reports]"` e a rodar `doctor --instalar`.

**Decisões de padrão e leitores conferidos com arquivos públicos**
- Decididos pelo autor e mantidos: `eem_espalhamento = nenhum` (`nan` só para quem vai rodar PARAFAC), `quantificacao_rotulo_treino = previsto` e
  os seis padrões da fase 2 (preset Raman + raios cósmicos, eixo crescente para nm/min/ms, cromatograma sem limite de pontos, `cv_objetos` na HSI, `faixas_excluir` na RMN,
  `unidade_agrupamento = auto`). Nada muda no comportamento.
- Conferidos com arquivos reais públicos de outros projetos (cópias fora do repositório): 4 arquivos OPUS de FT-IR (spectrochempy; o valor em 2303,87 cm⁻¹ bate com o da suíte deles),
  8 arquivos ANDI-MS de GC-MS (PyMassSpec; TIC exato e matriz tempo × m/z igual a um laço independente) e 12 pastas Agilent/Waters do `rainbow-api` (UV/DAD, FID, CAD, ELSD, MS),
  com o GUARACI devolvendo exatamente o que o `rainbow` lê. A decodificação Agilent/Waters é do `rainbow` e não foi comparada com o software do fabricante.
- **Correção (OPUS):** um arquivo só de background (`rf`/`igrf`) ou de série (vários espectros num arquivo, que o `brukeropus` não abre) era lido como se fosse uma amostra, e um
  interferograma entrava como espectro. Agora o fallback recusa referência, interferograma, fase e traço com `ValueError` explicando o motivo; e os espectros derivados `atr`, `km`, `logr`, `ra`,
  `e`, `p`, `pas` passam a ser preferidos ao background quando não há `a`/`t`/`r`.
- Pasta Agilent sem nenhum detector cromatográfico (MassHunter Q-TOF, ICP-MS) agora diz isso, em vez de `disponiveis: []`.
