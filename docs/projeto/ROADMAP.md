# Roteiro do projeto: da v1.0.0 à v1.1+

**Público.** Estado: manual. Um quadro por fase: o que entra, o que não entra, quando está pronta, quem executa e do que depende. Números de teste, cobertura e verificação **não** são escritos aqui: vêm de `docs/qualidade/estado_do_projeto.md` e do `resultado_<hash>.json` do hash verificado.

| fase | objetivo em uma linha |
|---|---|
| F1 | fechar a v1.0.0 num hash verificado |
| F2 | decisões que só o mantenedor toma |
| F3 | design e funcionalidade sobre o núcleo congelado |
| F4 | teste de usabilidade com pessoa externa |
| F5 | publicação |
| F6 | v1.1 e depois |

## F1 — Fechar a v1.0.0

| campo | conteúdo |
|---|---|
| objetivo | um hash da linha de desenvolvimento com `resultado_<hash>.json` e 0 falhas em Windows (3.12) e Linux (3.10, 3.11, 3.13) |
| entra | correções de defeito e de teste frágil achadas pela verificação; documentação limpa (um documento vivo por assunto, números gerados por script); este roteiro |
| não entra | funcionalidade nova, dependência nova, técnica nova em código, publicação |
| critério de pronto | `scripts/verificar_hash_final.py` com 0 falhas no hash final; branch publicável reconstruída uma vez a partir dele, com árvore idêntica (a ferramenta de reescrita do histórico fica fora do repositório, no ferramental privado do mantenedor) |
| quem executa | mantenedor com apoio de agente |
| dependência | nenhuma externa; as etapas `pendente_do_autor` (termos sensíveis locais, vault) aparecem separadas e não contam como falha |
| primeiro passo | corrigir as falhas da última verificação e repetir a verificação no hash congelado |

## F2 — Decisões do mantenedor

Só a lista; o detalhe (quem, onde, contato) fica fora do repositório.

| campo | conteúdo |
|---|---|
| objetivo | destravar o que nenhum código resolve |
| entra | licença e acordo de contribuição; governança do repositório; declaração de uso de IA para o artigo; contatos de manutenção e de segurança nos metadados; variável que desliga o envio de modelo no app público; licença formal de conjuntos de dados de terceiros usados só na validação local; nomes da API (português ou inglês) |
| não entra | qualquer execução pelo agente |
| critério de pronto | cada item com decisão registrada fora do repositório e, quando muda arquivo versionado, um commit próprio |
| quem executa | mantenedor |
| dependência | conversas institucionais quando o item pede |
| primeiro passo | desligar o envio de modelo no app público (minutos, sem dependência) |

## F3 — Design e funcionalidade

| campo | conteúdo |
|---|---|
| objetivo | melhorar apresentação e uso sem mexer no que calcula ou valida |
| entra | o que `docs/projeto/FASE_DESIGN.md` §1 permite: textos do catálogo, layout, cores, tipografia, menus, documentação e exemplos |
| não entra | o núcleo listado em `docs/qualidade/nucleo_congelado.json` sem repetir a verificação |
| critério de pronto | cada rodada termina com `scripts/verificar_hash_final.py` em 0 falhas no hash da rodada |
| quem executa | mantenedor com apoio de agente |
| dependência | F1 fechada |
| primeiro passo | escolher a primeira tela a revisar a partir da auditoria de experiência de uso mais recente |

## F4 — Teste externo

| campo | conteúdo |
|---|---|
| objetivo | medir a curva de aprendizado com quem nunca usou o programa |
| entra | `docs/usuario/roteiro_teste_usabilidade.md` (tarefas e tempos-alvo), `docs/usuario/guia_do_testador_externo.md`, `docs/usuario/template_feedback_teste.md` e a folha de registro sem dado pessoal |
| não entra | mudança de código durante a sessão de teste |
| critério de pronto | ao menos uma sessão completa registrada, com tempo por tarefa e a repetição da tarefa 2; achados viram itens de F3 |
| quem executa | mantenedor aplica; pessoa externa executa |
| dependência | F1 fechada (o pacote testado é o do hash verificado) |
| primeiro passo | combinar a primeira sessão e entregar o pacote do hash verificado |

## F5 — Publicação

| campo | conteúdo |
|---|---|
| objetivo | tornar o projeto público e citável |
| entra | registro de DOI do software; relógio de seis meses de histórico público exigido pela revista de software; app público com envio de modelo desativado |
| não entra | números derivados de conjunto de dados sem licença formal |
| critério de pronto | repositório público com a tag da versão, registro de DOI ativo e app público configurado |
| DOI do software | a v1.0 sai **sem DOI** (`CITATION.cff` e README não prometem nenhum); o DOI novo entra numa v1.0.x, depois do depósito, e só então aparece no `CITATION.cff` |
| quem executa | mantenedor |
| dependência | F2 (licença, histórico, governança) e F1 |
| primeiro passo | aprovar o histórico da branch publicável |

## F6 — v1.1 e depois

| campo | conteúdo |
|---|---|
| objetivo | ampliar técnicas e alcance sem perder a validação |
| entra | o quadro abaixo e o radar de `docs/projeto/roteiro_v1_1.md` |
| não entra | qualquer item sem referência conferida em registro oficial de DOI; alegação de conformidade regulatória |
| critério de pronto | cada técnica com dado sintético de verdade conhecida no teste, unidade independente explícita e verificação completa em 0 falhas |
| quem executa | mantenedor com apoio de agente |
| dependência | F1; decisão do mantenedor sobre a ordem |
| primeiro passo | curva de calibração univariada (base de três técnicas) |

### Técnicas candidatas (detalhe em `docs/projeto/estudo_novas_tecnicas.md`)

| prioridade | técnica | esforço | formato de entrada | o que já existe | risco |
|---|---|---|---|---|---|
| 1 | curva de calibração univariada (base) | M | tabela longa: leitura, tipo, concentração nominal, sinal | teste de falta de ajuste; figuras de mérito por regressão inversa | médio: modelo, pesos e faixa são decisões de método a expor |
| 2 | corrida analítica e correção de deriva por padrão interno | M | sequência de brancos, padrões, materiais de referência e verificações, com ordem de leitura | QC, brancos, material de referência e recuperação | médio |
| 3 | UV-Vis | P | matriz amostra × comprimento de onda + metadados de corrida | leitura por CSV, pré-processamentos, PLS, MCR-ALS, EMSC | baixo |
| 3 | DMA (sigla a confirmar: analisador direto de mercúrio ou análise dinâmico-mecânica) | P (se for mercúrio, sobre a curva univariada) | tabela longa: amostra, alíquota, massa, sinal, hora, tipo | censura, QC, ANOVA por grupo | baixo se for mercúrio; outro tipo de dado se for dinâmico-mecânica |
| 4 | ICP-OES | M | amostra × (elemento, linha) com unidade e ordem de leitura | perfil multielementar completo | médio: seleção de linha e deriva |
| 4 | MIP-OES | P (junto do ICP-OES) | o do ICP-OES, com gás do plasma e introdução da amostra | o mesmo do ICP-OES | baixo |
| 5 | FAAS | M | tabela longa por leitura | censura, QC, ANOVA | médio |
| 6 | GFAAS | G | o do FAAS, mais o sinal resolvido no tempo | o mesmo do FAAS | médio |

### Outros itens da v1.1+

| item | esforço | dependência |
|---|---|---|
| assistente que responde pergunta livre (modelo de linguagem local) | G | avaliação de privacidade e de custo |
| páginas web de ANOVA, VIP e HSI (hoje só na linha de comando) | M | F3 |
| validação em imagem hiperespectral real | M | licença formal do conjunto de dados |
| segundo conjunto público de ICP | P | conjunto com licença aberta e unidades independentes declaradas |
| teste de mutação dos módulos ainda não mutados | M | tempo de máquina; só linhas alteradas por rodada |
| suporte a macOS | M | máquina com macOS para rodar a suíte |
| testes de regressão com oráculos externos (PLS-DA, VIP, DD-SIMCA e OPLS-DA contra mdatools e ropls; ANOVA e post-hoc contra pingouin, statsmodels e scikit-posthocs) | M | R e as bibliotecas de referência fora das dependências do projeto, com saídas gravadas em arquivo |
| nome do pacote e do comando, que colidem com o projeto `guaraci` já publicado no PyPI | M | decisão do autor antes de publicar no PyPI |
| nomes da API em inglês | G | decisão da F2 (muda a API pública) |

## Itens identificados para a v1.1 e depois

Lista de trabalho, por área, sem datas. Os itens saíram da vistoria interna do projeto e da comparação com outras ferramentas de quimiometria. Cada linha diz o que falta e por quê; nada aqui está feito, e nenhum item muda resultado numérico sem teste que o prove.

### Prioridade alta

| item | por que importa | primeiro passo |
|---|---|---|
| estatística T² de treino do espaço PLS continua calculada nos próprios dados de treino | com seis ou mais variáveis latentes e calibração pequena, a taxa de rejeição de amostras legítimas fica acima do nominal | estimar o limite por validação cruzada (ou por reamostragem) e medir a taxa de rejeição em simulação |
| limiar conformal no domínio de aplicabilidade | o conjunto conformal fora do domínio sai sem ressalva, e o limiar de rejeição do domínio é calibrado em poucas amostras | calibrar o limiar do domínio com o mesmo esquema do conformal e acrescentar a ressalva à saída |

### Técnicas e métricas

| item | esforço | observação |
|---|---|---|
| curva de calibração univariada (padrão externo e adição de padrão) com LOD e LOQ univariados | M | base de três técnicas desta lista; primeiro item da F6 |
| FAAS, GFAAS e MIP-OES (técnicas atômicas) | M a G | não há leitor nem perfil; entram sobre a curva univariada e a corrida analítica (ver quadro de técnicas) |
| DMA | P | sigla a confirmar pelo mantenedor: analisador direto de mercúrio (então vira variante da curva univariada) ou análise dinâmico-mecânica (outro tipo de dado) |
| OPLS para regressão | M | hoje só há OPLS-DA |
| RPIQ junto de RPD e RMSEP | P | métrica ausente |
| LOD e LOQ multivariados completos | M | a versão atual é aproximada: faltam os termos de alavancagem e de erro em y, como o próprio texto do código admite |
| RMSEC com correção de graus de liberdade | P | hoje sem a correção |
| número de variáveis latentes da regressão por parcimônia | M | hoje é o mínimo global, sem regra de parcimônia |
| método sequencial para a sentinela de deriva | M | falso alarme acumulado e atraso de detecção no método atual (há ressalva no texto) |
| conjunto conformal vazio registrado como texto vazio | P | gravar um marcador explícito |
| eta ao quadrado e intervalo de confiança da ANOVA com bloco ignoram o bloco | P | incluir o bloco no cálculo |
| comparações de Dunn com dispersões muito diferentes | P | ressalva ou método alternativo |
| partição estável pode gerar dobra vazia em casos irreais | P | detectar e avisar |
| unidade da resposta configurável (hoje o texto não presume unidade) | M | padrão neutro para a faixa espectral por técnica |
| concordância numérica do DD-SIMCA com uma implementação de referência independente | M | o método continua marcado como sem validação pública |
| equivalência dos pré-processamentos com uma biblioteca de referência | M | comparar saída a saída em dado sintético |
| preset personalizado aceita correções redundantes | P | recusar ou avisar |
| coluna `confianca_%` não é probabilidade calibrada | P | renomear a coluna e documentar o que ela mede |

### Dados, formatos e exemplos

| item | esforço | observação |
|---|---|---|
| exemplos sintéticos para Raman, infravermelho médio, UV-Vis, RMN e HPLC | M | hoje os exemplos sintéticos não cobrem essas técnicas |
| dado de demonstração do demonstrativo sem confundimento entre sessão e classe | P | o demonstrativo mostra avisos críticos e acerto perfeito sem explicar a origem; parte da explicação já entrou no texto |
| leitor de arquivos `.spa` | M | formato de espectrômetro comercial ausente |
| leitor de CSV com motor mais rápido | M | o motor em Python é cerca de 8 vezes mais lento e usa cerca de 6 vezes mais memória que o motor alternativo |
| NaN ou Inf descarta o espectro sem dizer qual | P | listar os índices descartados |
| intervalo BCa sai `[nan, nan]` no caso degenerado | P | mensagem e valor explícitos |
| validação imediata do arquivo no envio do demonstrativo | P | hoje o erro só aparece ao rodar |
| pasta local de correções fixa no código (`_DIR_LOCAL` em `src/guaraci/dados_io.py`) | P | o caminho de uma pasta de desenvolvimento está escrito no código; deve vir de configuração ou sair do código |

### Interface e experiência

| item | esforço | observação |
|---|---|---|
| subcomando de predição na linha de comando | M | hoje o fluxo de predição só existe na tela |
| paleta de cores para fundo claro no terminal | P | a paleta supõe fundo escuro (a variável `GUARACI_TERMINAL=claro` ameniza) |
| menu em terminal estreito | P | o menu trunca em larguras pequenas |
| rastros técnicos pontuais em mensagens de erro | P | revisar mensagens |
| primeira abertura das telas | P | orientar o primeiro uso |
| pastas de saída inconsistentes entre comandos | M | uniformizar nomes e raiz de saída |
| alinhamento da saída de `guaraci perfis` | P | colunas desalinhadas |
| interface totalmente bilíngue e nomes da API em um idioma só | G | hoje há mistura de português e inglês (depende da decisão sobre os nomes da API) |
| estimativa de tempo antes de rodar | M | a interrupção já não deixa pasta parcial com aparência de completa; falta estimar a duração |
| interface gráfica de desktop e análise interativa | G | lacuna frente a ferramentas com interface própria |
| unidade do eixo e rótulos vindos sempre do perfil | P | restam telas com texto fixo |

### Desempenho e recursos

| item | esforço | observação |
|---|---|---|
| consumo fixo de memória de cerca de 0,7 a 0,8 GB | M | importar sob demanda |
| importação de `pyplot` no topo dos módulos | P | adiar a importação |
| detecção de processadores ignora afinidade e limites de contêiner | P | respeitar cgroup e afinidade |

### Qualidade do código

| item | esforço | observação |
|---|---|---|
| funções de complexidade alta e módulos grandes | G | dividir por responsabilidade |
| comentários `noqa` sem efeito | P | remover os inertes |
| `type: ignore` sem motivo | P | justificar ou corrigir |
| avisos registrados em nível INFO | P | subir para WARNING |
| função `_fmt` duplicada | P | unificar |
| asserções fracas em testes pontuais | P | tornar exatas |
| regra de análise estática de desserialização incompleta | P | cobrir os casos restantes |
| remissões internas que ainda restam em comentários | P | cada comentário deve explicar o motivo sem apontar documento interno |
| citações autor-ano fora do verificador de referências | P | estender o verificador |
| arquivos versionados sem uso aparente | P | provar o desuso nas quatro versões de Python antes de remover |

### Distribuição, segurança e conformidade

| item | esforço | observação |
|---|---|---|
| publicação em índice de pacotes (PyPI e conda) | M | depende da decisão sobre o nome do pacote |
| imagem de contêiner fixada por digest e construída por camadas | P | o Dockerfile não fixa digest |
| manifesto mínimo de implantação do demonstrativo | P | descrever o que a hospedagem precisa |
| avisos de terceiros da imagem e do ambiente, com verificação de cópia | M | ampliar o arquivo de avisos |
| cabeçalho SPDX em cada arquivo | M | hoje só a licença do pacote está declarada |
| trilha de auditoria e assinatura eletrônica | G | exigida em laboratório regulado; **nenhuma alegação de conformidade regulatória** enquanto não existir |
| JSON brutos de qualidade versionados | P | decidir se ficam no repositório |
| idempotência do cofre de notas verificável sem variável local | P | tornar o teste independente do ambiente |
| aviso de privacidade e termos do demonstrativo publicados | P | os rascunhos estão em `docs/usuario/`; falta a aprovação do autor |
