# Mapa técnica × tarefa × comando × exemplo × teste

**Público:** quem avalia o que o programa cobre · **Estado:** gerado por `scripts/gerar_mapa_tecnica_tarefa.py` a partir do catálogo (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

## 1. Cenários sintéticos por tarefa

Cada cenário gera dados sintéticos por código próprio e a configuração pronta; nenhum dado de terceiros.

| tarefa | cenário | o que faz | como rodar | teste |
|---|---|---|---|---|
| explorar | `explorar` | Olha a estrutura de espectros sinteticos de 3 classes, sem treinar modelo. | `guaraci exemplo explorar` e `guaraci run config.yaml` | `tests/test_exemplos_e_comandos.py` |
| classificar | `classificar` | Classifica espectros sinteticos de 3 especies com validacao por grupo (lote). | `guaraci exemplo classificar` e `guaraci run config.yaml` | `tests/test_exemplos_e_comandos.py` |
| quantificar | `quantificar` | Estima um teor sintetico de 0 a 20 por regressao PLS com validacao por grupo. | `guaraci exemplo quantificar` e `guaraci run config.yaml` | `tests/test_exemplos_e_comandos.py` |
| comparar | `multielementar` | Classifica origens e compara elementos sobre o lote, com censura e QC. | `guaraci exemplo multielementar` e `guaraci run config.yaml` | `tests/test_exemplos_e_comandos.py` |
| comparar | `comparar` | Compara a media de uma resposta entre 3 grupos, agregando por unidade independente. | `guaraci exemplo comparar` e `guaraci anova tabela.csv --resposta teor --grupo grupo --unidade amostra` | `tests/test_exemplos_e_comandos.py` |
| imagens | `imagem` | Gera 36 fotos sinteticas de 3 classes (6 amostras com 2 fotos cada por classe) e classifica pela cor. As classes (clara, media, escura) sao niveis ordenados de cor: a acuracia balanceada abaixo de 1,00 (cerca de 0,9) vem do PLS-DA, nao dos dados; com classificador: plslda (LDA sobre os escores) as 3 classes saem com 1,00. | `guaraci exemplo imagem` e `guaraci run config.yaml` | `tests/test_exemplos_e_comandos.py` |
| imagens | `hsi` | Gera 16 cubos ENVI sinteticos de 2 classes (4 objetos por classe, 2 vistas cada) e roda o fluxo HSI. | `guaraci exemplo hsi` e `guaraci run config.yaml` | `tests/test_exemplos_e_comandos.py` |
| quantificar | `transferir` | Gera 12 amostras lidas nos dois instrumentos e 20 espectros novos do segundo, e os converte para o primeiro. | `guaraci exemplo transferir` e `guaraci transferir --mestre mestre.csv --escravo escravo.csv --novos novos_escravo.csv` | `tests/test_exemplos_e_comandos.py` |
| monitorar | `monitorar` | Calibra em 5 lotes em controle e acompanha 6 lotes novos, os 2 ultimos com deriva simulada. | `guaraci exemplo monitorar` e `guaraci monitorar --referencia referencia.csv --novos novos.csv --lote lote --id amostra` | `tests/test_exemplos_e_comandos.py` |
| classificar | `fluorescencia_eem` | Gera 18 matrizes excitacao-emissao sinteticas de 3 classes (2 fluoroforos, a classe muda a proporcao) e classifica o vetor achatado. | `guaraci exemplo fluorescencia_eem` e `guaraci run config.yaml` | `tests/test_exemplos_e_comandos.py` |
| classificar | `cromatograma_gcms` | Gera 18 cromatogramas de TIC sinteticos de 3 classes (razao entre dois picos e pequeno deslocamento de retencao) e classifica. | `guaraci exemplo cromatograma_gcms` e `guaraci run config.yaml` | `tests/test_exemplos_e_comandos.py` |
| modelar | `vip` | Gera 3 classes com 12 variaveis (so as 3 primeiras separam as classes) e mostra o VIP e a estabilidade nas dobras agrupadas. | `guaraci exemplo vip` e `guaraci vip tabela.csv --classe classe --grupo amostra` | `tests/test_exemplos_e_comandos.py` |

## 2. Técnicas de análise do catálogo

| técnica | tarefa | selo | evidência (tipos) | exemplo de uso |
|---|---|---|---|---|
| `ddsimca` | classificar | sem_validacao_publica | 1 testes | nivel: N2 (o DD-SIMCA liga no nivel N2) |
| `conformal_one_class` | classificar | em_validacao | 1 testes | automatico: o pacote do modelo de [R] Rodar treina, [3.3] Predicao (modo cego) aplica; API guaraci.conformal.ConformalOneClass |
| `pls_da` | classificar | em_validacao | 1 testes | nivel: N1 |
| `pls_r_pooled` | quantificar | em_validacao | 2 testes | nivel: N3 |
| `pls_r_por_especie` | quantificar | sem_validacao_publica | 1 testes | nivel: N3 |
| `conformal_regressao` | quantificar | em_validacao | 1 testes | automatico em [R] Rodar com objetivo quantificacao (intervalo por amostra); API guaraci.conformal.conformal_margin_regression |
| `conformal_classificacao` | classificar | em_validacao | 1 testes | automatico em [R] Rodar com objetivo classificacao; API guaraci.conformal.conformal_margin_classification |
| `identificacao_conjunto_aberto` | classificar | sem_validacao_publica | 1 testes | [3.3] Predicao com rotulo cego; API guaraci.predicao.predict_blind |
| `kennard_stone` | dados | em_validacao | 1 testes | tela [2.8] Selecao de amostras |
| `duplex` | dados | em_validacao | 1 testes | tela [2.8] Selecao de amostras |
| `spxy` | dados | em_validacao | 1 testes | tela [2.8] Selecao de amostras |
| `direct_standardization` | quantificar | em_validacao | 1 testes | guaraci transferir --metodo ds (ou guaraci exemplo transferir --rodar) |
| `piecewise_direct_standardization` | quantificar | em_validacao | 1 testes | guaraci transferir --metodo pds (ou guaraci exemplo transferir --rodar) |
| `figuras_de_merito_regressao` | quantificar | em_validacao | 1 testes | saem sozinhas na quantificacao (nivel N3) |
| `rpd_rer` | quantificar | em_validacao | 1 testes | saem sozinhos na quantificacao (nivel N3) |
| `linearidade` | quantificar | em_validacao | 1 testes | automatico em [R] Rodar com objetivo quantificacao; API guaraci.linearity.lack_of_fit_test |
| `robustez` | confianca | em_validacao | 1 testes | [3.4] Avancado > protocolo de robustez (quantificacao); API guaraci.robustness.run_robustness_protocol |
| `perfil_matriz` | dados | nao_aplicavel | 1 testes | perfil_matriz: nir_ft_difuso |
| `perfil_tecnica_aquisicao` | imagens | nao_aplicavel | 1 testes | perfil_tecnica: (modo imagem) |
| `mcr_als` | explorar | limitado | 1 testes | tela [3.5] Tecnicas especiais |
| `anova_tukey_unidade` | comparar | em_validacao | 1 simulacao, 1 testes | guaraci anova  (ou modo multielementar) |
| `kruskal_dunn_unidade` | comparar | em_validacao | 2 testes | guaraci anova  --metodo kruskal-dunn |
| `diagnostico_premissas` | comparar | em_validacao | 1 simulacao, 1 testes | guaraci anova  --metodo recomendar |
| `vip_estabilidade` | modelar | em_validacao | 1 testes | guaraci exemplo vip --rodar |
| `perfil_multielementar` | comparar | em_validacao | 1 testes | modo_entrada: multielementar |
| `parafac_eem` | explorar | em_validacao | 1 testes | tela [3.5] Tecnicas especiais, item 5 (modo eem) |
| `cow_alinhamento` | dados | em_validacao | 1 testes | tela [3.5] Tecnicas especiais, item 6 (modo hplc ou gcms) |
| `portao_correcao_sinal` | dados | em_validacao | 1 testes | tela [3.5] Tecnicas especiais, item 7 |
| `amostragem_ativa` | dados | limitado | 1 testes | tela [2.7] Planejamento de coleta, refinar com um modelo treinado |
| `pooled_vs_local` | quantificar | sem_validacao_publica | 1 testes | tela [3.5] Tecnicas especiais, item 8 (nivel N3) |
| `monitoramento_mspc` | monitorar | em_validacao | 1 simulacao | guaraci exemplo monitorar --rodar |

## 3. Técnicas de medida

| técnica | faixa típica | pré-processamento recomendado | modo de entrada |
|---|---|---|---|
| `ft-nir` | 4000-10000 cm-1 (tipico) ou 4000-12000 cm-1 | MSC+SG+MC | dx |
| `nir` | 700-2500 nm | SNV+SG+MC ou MSC+SG+MC | dx |
| `mir` | 400-4000 cm-1 | SNV+SG+MC (1a derivada; e o aplicado); SNV+MC ou SG+MC como alternativas (MSC menos comum em MIR) | dx |
| `raman` | 200-4000 cm-1 (Raman shift; abaixo de ~200 cm-1 entra a borda do filtro Rayleigh) | AirPLS+SNV+SG+MC com remocao de raios cosmicos (a [2.4] liga os dois): AirPLS tira a fluorescencia de fundo (aditiva, por isso sem MSC) e o SNV tira a escala do sinal (potencia, foco, densidade), que senao fica no modelo; AirPLS+SG+MC e a alternativa sem normalizacao | dx |
| `uv-vis` | 190-900 nm | Mean-centering (Beer-Lambert: a absorbancia e proporcional a concentracao e o SNV apaga essa escala); SNV+MC so para classificar pela forma do espectro | csv |
| `fluorescencia` | 200-800 nm (use CSV) | MC (a intensidade e proporcional a concentracao; SNV divide cada espectro pelo proprio desvio e apaga essa informacao -- use SNV+MC so para classificar por forma do espectro) | csv |
| `hplc` | Tempo de retencao ou tabela de compostos (sem corte) | MC (cromatograma inteiro: autoscaling amplifica o ruido da linha de base); alinhe a retencao (COW) antes | csv |
| `gc-ms` | Retencao, compostos ou m/z (sem corte) | MC (cromatograma inteiro: autoscaling amplifica o ruido da linha de base), apos alinhar os picos (COW) | csv |
| `nmr` | 0-12 ppm (deslocamento quimico) | SNV ou PQN + MC (apos binning) | csv |
| `ims` | 5-50 ms (tempo de deriva) | SNV+SG+MC (apos alinhamento por tempo de deriva) | csv |
| `eem` | Indice da grade (sem unidade fisica) | MC (PARAFAC usa a matriz sem pre-processamento) | eem |
| `icp` | Nao se aplica (colunas sao elementos) | Autoscaling (elementos em escalas diferentes), com log opcional | multielementar |
| `imagem` | Indice da variavel de cor (sem unidade fisica) | Autoscaling (MSC e SG nao fazem sentido para estatisticas de cor) | imagem |
| `hsi` | A do sensor (nm), lida do cabecalho ENVI | SNV+MC por pixel (confira a faixa do sensor) | hsi |
| `generico` | Configurar manualmente no menu Dados | Autoscaling (variaveis em unidades diferentes); MSC/SG apenas para espectros | csv |

## 4. Comandos

| comando | para que serve |
|---|---|
| `guaraci run [CONFIG]` | Executa o pipeline uma vez a partir de CONFIG (padrao: ./config.yaml) e sai, sem menu: para automacao e CI. |
| `guaraci demo` | Roda o pipeline com dados sinteticos; nao precisa de dado seu. |
| `guaraci exemplo [CENARIO] [PASTA]` | Gera dados sinteticos e a configuracao prontos para um cenario (sem argumento, lista os cenarios). |
| `guaraci web` | Abre a interface web local no navegador (127.0.0.1; so esta maquina). |
| `guaraci perfis` | Lista os perfis de tecnica disponiveis. |
| `guaraci anova TABELA.csv --resposta COL --grupo COL --unidade COL [--metodo M]` | Compara grupos sobre a unidade independente (obrigatoria): ANOVA com Tukey ou Games-Howell, ou Kruskal-Wallis com Dunn; recusa sem --unidade. |
| `guaraci vip TABELA.csv --classe COL --grupo COL [--componentes N]` | Calcula o VIP do PLS e mostra em quantas dobras agrupadas cada variavel passa do limiar; VIP > 1 e regra pratica, nao teste. |
| `guaraci monitorar --referencia REF.csv --novos NOVOS.csv --lote COL` | Calibra o dominio do processo em lotes de referencia e alarma os lotes novos que sairam dele. |
| `guaraci transferir --mestre M.csv --escravo E.csv --novos N.csv [--metodo ds|pds]` | Converte espectros de um instrumento novo para o espaco do instrumento do modelo (DS ou PDS). |
| `guaraci relatorio PASTA [--formato pdf,docx,xlsx,tex,pptx]` | Gera os relatorios (PDF, Word, Excel, LaTeX, PowerPoint) de uma execucao ja feita. |
| `guaraci assistente [PERGUNTA]` | Responde uma pergunta a partir do catalogo, sem rede (sem argumento, explica o uso; nas telas, tecla G). |
| `guaraci ajuda [TOPICO]` | Explica uma opcao, tecnica, grafico, tela ou termo no mesmo gabarito (sem argumento, mostra o mapa de grupos). |
| `guaraci buscar TERMO` | Procura uma palavra em todo o catalogo (opcoes, tecnicas, graficos, termos, comandos). |
| `guaraci manual [TOPICO|SECAO]` | Mostra o manual e os documentos que acompanham o programa (sem argumento, lista os topicos; com numero, a secao). |
| `guaraci checar CONFIG [--dados] [--resultado PASTA] [--ficha]` | Pre-voo do assistente sem as telas: regras de configuracao e, com --dados, de dados; --resultado le uma execucao. Sai 1 com achado critico. |
| `guaraci confianca [METODO]` | Mostra selo, evidencia, limites e como reproduzir um metodo ou opcao. |
| `guaraci doctor` | Diagnostica o ambiente: dependencias, RAM e CPU. |
