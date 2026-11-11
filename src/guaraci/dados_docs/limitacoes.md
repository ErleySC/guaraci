# Limitações conhecidas

**Público:** quem avalia os métodos e os números · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

O que este projeto **não garante**, escrito junto da evidência de cada ponto. Fonte única: se uma limitação some do código, some daqui no mesmo commit.

## 1. Seleção de variáveis e validação aninhada

**O que foi medido (2026-10-06, `scripts/prova_selecao_aninhada.py`, 30 sementes).** Rótulos sorteados e 200 variáveis de ruído puro, 60 amostras, 3 classes (acaso = 0,333). Acerto balanceado médio:

| método | aninhado (como o pipeline faz) | ingênuo (máscara escolhida com todos os dados) |
|---|---|---|
| VIP ≥ 1 | 0,344 | 0,715 |
| SR (10% melhores) | 0,341 | 0,632 |
| sPLS-DA | 0,346 | 0,689 |
| CARS | 0,339 | 0,537 |
| iPLS (melhor intervalo) | 0,318 | 0,406 |
| SPA/APS | 0,316 | 0,339 |

O aninhado fica no acaso nos seis métodos; o ingênuo passa muito do acaso em VIP, SR, sPLS-DA e CARS (o teste tem poder para enxergar vazamento). Evidência: `docs/qualidade/selecao_aninhada_rotulos_aleatorios.json`; teste de regressão: `tests/test_selecao_aninhada_rotulos_aleatorios.py`. O Auto-Benchmark de classificação e o de regressão usam o pré-processamento dentro do `Pipeline` por dobra e hiperparâmetros fixos (sem ajuste por validação interna), então não há seleção que vaze.

**O que continua sendo limitação:**

- **Escolha ENTRE métodos.** Cada método reescolhe as variáveis dentro da dobra, mas escolher o melhor método da tabela da Etapa 4 usa a mesma validação que mede o resultado: o vencedor é otimista. O programa imprime esse aviso (`AVISO_ESCOLHA_ENTRE_METODOS`). Para um número honesto do método escolhido, use uma validação externa que não participou da escolha.
- **Número de variáveis latentes.** A Etapa 4 recebe o `n_opt` que o pipeline escolheu por validação cruzada nos mesmos dados; o número de componentes não é reescolhido dentro de cada dobra da Etapa 4.
- **Pré-processamento ajustado uma vez.** `X_processed` usa um pré-processador ajustado em todas as amostras (para as figuras e a PCA) e é a entrada da Etapa 4. SNV, Savitzky-Golay e AirPLS agem espectro por espectro (não há o que vazar); MSC, PQN e EMSC usam um espectro de referência calculado com todas as amostras, o que vaza só informação não supervisionada (sem rótulo), de efeito pequeno. O OSC usa o rótulo e é opcional; com ele ligado essa entrada deixa de ser independente do rótulo.

## 2. Método padrão do `guaraci anova` (simulação com verdade conhecida)

**O que foi medido (2026-10-07, `scripts/simular_politicas_anova.py`, 600 repetições por cenário).** Cinco políticas (Tukey fixo, Games-Howell fixo, `auto` por Levene, `recomendar`, Kruskal-Wallis/Dunn) rodaram sob a hipótese nula de médias iguais, em variância 1:1 a 1:16, n = 4 a 30, k = 3 a 10, distribuições normal, lognormal, t de 3 graus e contaminada. A **calibração** (104 cenários) escolheu a política; uma **confirmação independente** (136 cenários, outros n, razões, k e sementes) a conferiu. O critério era erro por família (FWER) ≤ 0,06 em todos os cenários e, entre as que cumprissem, o maior poder. Evidência: `docs/qualidade/politicas_anova_simulacao.md` e `.json`.

**Resultado: nenhuma política cumpre o critério.** O padrão passou a ser o **Games-Howell**, a política com menos cenários acima do limite (18 de 104 na calibração, 25 de 136 na confirmação, contando a margem de Monte Carlo). Tukey (30 e 35) e a recomendação (31 e 33) ficaram atrás nas duas grades; o `auto` empatou na confirmação (25) com poder maior (0,75 contra 0,69). O Tukey só roda a pedido (`--metodo tukey`); com `--bloco` o padrão é `auto`, porque o Games-Howell não tem versão com bloco aqui.

**Onde o padrão falha (FWER do Games-Howell, calibração / confirmação):**

| classe de cenário | FWER médio | FWER máximo |
|---|---|---|
| n e variância iguais entre grupos | 0,053 / 0,049 | 0,143 / 0,182 (n = 4 com assimetria ou caudas pesadas) |
| n desigual | 0,051 / 0,051 | 0,093 / 0,108 |
| variância maior nos grupos grandes | 0,045 / 0,043 | 0,075 / 0,085 |
| **variância maior no grupo de poucas unidades** | **0,090 / 0,087** | **0,207 / 0,270** |

- Com premissas razoáveis (normal, variâncias iguais) o Games-Howell fica em 0,063 no pior caso; o Tukey, em 0,067 e 0,062.
- A falha concentra-se em **variância alta no grupo de poucas unidades**, mais forte com dado assimétrico (lognormal) e n ≤ 8. Nesse caso nenhuma política testada controla o erro; com menos de 6 unidades por grupo trate o resultado como exploratório.
- O poder médio do Games-Howell é 0,68 (Tukey 0,79): é o preço do controle melhor do erro.
- **A recomendação perdeu para o padrão** nas duas grades (FWER máximo 0,63 e 0,53, contra 0,21 e 0,27) e foi rebaixada: `--metodo recomendar` mostra o diagnóstico de premissas e a sugestão, mas **roda o padrão** e diz o que foi sugerido. Para usar a sugestão, peça o método explicitamente.
- O **Kruskal-Wallis/Dunn** compara distribuições, não médias: sob médias iguais e variâncias diferentes ele rejeita muito (FWER até 0,63). Use quando a pergunta for de dominância estocástica ou quando o dado não permitir comparar médias.

Uma simulação anterior (2000 repetições por cenário, `docs/qualidade/anova/recomendacao_simulacao.md`) mediu a regra da recomendação isoladamente: acerta pouco em n = 4 e em caudas pesadas. Está mantida como evidência; a decisão acima é a que vale para o padrão.

O teste `tests/test_anova_fwer_politica.py` refaz na CI um cenário que o padrão cumpre e um que ele não cumpre: se o erro mudar, esta seção muda junto.

## 3. Kruskal-Wallis e Dunn

Comparam distribuições (dominância estocástica). Só equivalem a "diferença de medianas" se as formas dos grupos forem parecidas. O Dunn também não é robusto a variâncias muito diferentes com n desigual (erro por família 0,114 no cenário 1:10 com n desigual, mesma simulação).

## 4. VIP com estabilidade

O VIP resume a importância das variáveis **no modelo PLS ajustado**; não é seleção. `guaraci vip` mostra em quantas dobras agrupadas cada variável passa do limiar, mas não reduz o espectro. Reduzir variáveis pelo VIP só vale dentro da validação aninhada da Etapa 4 (item 1).

## 5. Canais e ambientes

- **Comparação de grupos (`comparar`/`anova`) e `guaraci vip`: só linha de comando.** Não há página no canal web (decisão D-S, `app_nav.TELAS_SO_NO_CLI`).
- **Versões do Python.** O projeto declara `>=3.11` desde a 1.0.0: o Python 3.10 saiu de suporte em outubro de 2026 e o Pillow com as correções de segurança (13.0) exige 3.11. O CI (`.github/workflows/test.yml`) cobre 3.11 a 3.13 em Linux e Windows e 3.11 e 3.13 em macOS (este só no push para o master).

## 6. Plataforma, escopo e unidade independente

- **O resultado depende da unidade independente.** Réplicas, espectros e dobras de validação cruzada não são unidades independentes: tratados como tal, o p-valor e a acurácia saem otimistas (medido: 57% de falsos positivos com réplicas contra 4,8% agregando por unidade). Os comandos pedem a unidade e o relatório diz qual foi usada; quando o delineamento tem sessões ou lotes além das réplicas, compare com `unidade_agrupamento: sessao` antes de citar um número.
- **macOS só na CI (não suportado nesta versão).** A suíte completa passa no macOS da CI (Python 3.11 e 3.13). A verificação manual cobre Windows (Python 3.12) e Linux (Python 3.11 e 3.13, via WSL); no macOS não houve uso real.
- **Docker só na CI.** A CI constrói a imagem e confere que o app web sobe em modo servidor, sem root e sem carregar pickle. Análises dentro do contêiner não são testadas.
- **HSI com dado real só foi validado localmente.** O fluxo por pixel roda com cubos próprios, offline e sob dado sintético na suíte; o desempenho em dado real de imageamento hiperespectral não tem resultado publicável (o conjunto usado não tem licença formal).
- **Leitores de formato: arquivo sintético × arquivo real.** Os importadores têm testes de contrato contra a especificação de cada formato, com arquivos gerados pelo próprio projeto. Esses arquivos seguem a especificação, mas não reproduzem as variações de arquivos reais de instrumento (campos opcionais, versões de firmware, preenchimentos fora do padrão). A rodada contra arquivos reais de fabricante só existe no checkout do mantenedor; fica registrada sem os arquivos (contagens e hash do resultado, `docs/qualidade/rodada_local_formatos_reais.json`). OPUS, ANDI-MS (GC-MS), Agilent e Waters foram conferidos também com arquivos públicos de outros projetos (4 arquivos OPUS de amostra, 8 ANDI-MS e 12 pastas Agilent/Waters; ver o manual, seção 4), sem comparação com o software do fabricante. Quem usar um formato que nunca viu um arquivo real do seu instrumento deve conferir a leitura de uma amostra conhecida antes de confiar.
- **Pergunta livre ao assistente fica para a v1.1+.** Hoje o assistente responde só a partir do catálogo, por regra, offline; fora das regras ele diz que não sabe.
- **Valores padrão são ponto de partida.** Os valores padrão seguem a literatura citada e não são universais; nenhum resultado com dado que não acompanha o programa é publicado. Ajuste por opção e confirme com validação externa própria.
- **Sem uso externo ainda.** Não há usuário externo, release nem citação; o guia para quem quiser testar é `docs/usuario/guia_do_testador_externo.md`.

## 7. O que a verificação por mutação cobre (e o que não)

- **Escopo da mutação.** O ANOVA, a CLI do ANOVA e o VIP têm a rodada completa (`docs/qualidade/anova_blindagem.md`). Nos demais módulos do núcleo, a mutação foi aplicada só às **linhas alteradas** desde a última verificação completa (`docs/qualidade/mutacao_linhas_alteradas.md`): 99 de 100 mutantes mortos e o sobrevivente classificado como equivalente. Linhas antigas desses módulos não foram mutadas nesta rodada.
- **Mutação não executada** em `hsi_extras` e `auditoria_delineamento` (linhas alteradas): decisão de escopo por tempo e carga da máquina; entra na v1.1.
- **Mutantes equivalentes são classificados à mão.** Os 20 sobreviventes de `anova_cli` e o de `pipeline` estão em `docs/qualidade/anova/mutantes_classificados.json`, cada um com a razão; uma amostra foi reverificada por execução (`docs/qualidade/anova/equivalentes_amostra.md`). Mutantes estéticos (tamanho de fonte, espessura de linha) não são afirmados por teste.
- **Cobertura de 100% não prova correção.** A cobertura mede o que foi executado, não o que foi conferido; a conferência é a mutação e os oráculos independentes. O escopo medido é src/guaraci, com 50 linhas excluídas e registradas (≈99,90 % sem exclusões); app_quimiometria.py e scripts fora do escopo.

## 8. Achados da auditoria de 2026-11-10 que seguem abertos

A auditoria independente do commit final corrigiu DS/PDS, Selectivity Ratio, o veredito de linearidade, o aviso sobre o nome no PyPI, o "%" fixo do LOD/LOQ, os avisos de classificação em execuções de quantificação e a atribuição dos cortes do RPD (ver `CHANGELOG.md`). Ficaram abertos, e estão aqui sem maquiagem:

- **Faixas de RPD.** Os cortes (2,0 / 2,5 / 3,0 / 5,0) são uma convenção prática de NIR agrícola, sem fonte única verificada: o código não os atribui a Williams (2014, citado só pela definição do RPD) nem à AACC 39-00.01, porque o texto original não foi conferido e há escalas publicadas com cortes diferentes. Trate a faixa como orientação (Esbensen et al. 2014 criticam cortes fixos) e compare o RMSEP com o critério do seu método oficial.
- **LOD e LOQ omitem leverage e erro da referência.** Só o ruído de réplica entra; o valor tende a subestimar e o programa avisa quando o LOD cai abaixo do RMSEP.
- **Nomes de classificação em execução de quantificação.** A pasta de saída (`PLSDA_OE_...`) e o arquivo `modelo_plsda.joblib` (e `modelo_plsda.portatil.json`) mantêm o prefixo `PLSDA` mesmo numa quantificação: renomeá-los quebraria o contrato de formato lido pela predição, pelo app web e pelos modelos já salvos. Os números e os avisos são os da regressão.
- **Idioma misturado.** Sem idioma escolhido o relatório sai em inglês, e parte do console sai em português. Escolha o idioma explicitamente.
- **Testes de DS/PDS e VIP/SR.** Os mutantes sobreviventes da auditoria foram fechados com oráculos independentes (Ridge do scikit-learn, fórmulas fechadas): 15 de 16 mutantes mortos na checagem manual; o sobrevivente (centrar o mestre no DS) é equivalente, e a divisão `b/‖b‖` do SR também (o SR é invariante à escala de b). Os demais módulos do núcleo seguem com a cobertura de mutação descrita na seção 7.
- **DD-SIMCA em dado limpo.** Marca de 2 a 12 amostras como atípicas em 300 pontos gaussianos limpos (no demo, 6 de 27). Com alpha = 0,05 isso é esperado por construção (cerca de 5 %): leia as "atípicas" como candidatas à inspeção, não como defeito.
- **Kennard-Stone com semente aproximada.** O conjunto difere do algoritmo exato em 45 de 200 sorteios (Jaccard médio 0,95, mínimo 0,59); isso está documentado no código, não na saída do programa.
- **Oráculos externos em R (mdatools, ropls).** Os testes contra esses pacotes não estão nesta versão (não havia R no ambiente de construção); os métodos correspondentes foram conferidos contra Python (pingouin, statsmodels, scikit-posthocs), contra fórmulas fechadas e por propriedades, não contra o R.
- **Escala.** Não há medida de tempo e memória para dezenas de milhares de espectros; o demo (27 s, cerca de 1 GB de pico) é o que foi medido.
