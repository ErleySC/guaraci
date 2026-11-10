# Radar de técnicas e tecnologias — v1.1+ (2026-10-06)

**Público:** quem decide o rumo do projeto · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

**Isto não implementa nada.** Nenhuma linha daqui entra em código ou dependência na v1.0.0. Cada referência foi conferida contra o registro oficial do DOI (Crossref ou DataCite) em 2026-10-06 e consta em `docs/cientifico/referencias_verificadas.yaml`. Referência que não passou na conferência foi descartada (ver o fim).

## Precedente

Há um precedente pontual de busca por técnicas novas (DD-SIMCA com PCV, 2026-08-08: achou Kucheryavskiy et al. 2020 e Pomerantsev e Rodionova 2021, integrado como extra opcional `[robusto]`, função `sensibilidade_ddsimca_pcv()`) e nenhum radar permanente; este documento é o primeiro. Para revisitar: repetir a busca por área a cada release menor e atualizar a tabela, sem mexer em código antes de o autor aprovar.

## Tabela (cada linha: o que faz, onde ajudaria, custo, veredito)

Tipo de fonte: todas as referências abaixo são artigos revisados por pares, exceto as duas marcadas *preprint*. Nenhuma vem de blog.

| técnica / tecnologia | o que faz (1 linha) | cenário do GUARACI que melhora | evidência (DOI conferido) | custo / risco | veredito |
|---|---|---|---|---|---|
| PLS invariante a domínio (di-PLS) | Isto ajusta o PLS para que o modelo valha em outro instrumento ou lote, usando espectros do novo domínio sem rótulo. | transferência de calibração (hoje só DS/PDS, que exigem amostras padrão medidas nos dois instrumentos) | Nikzad-Langerodi et al., *Anal. Chem.* 2018, [10.1021/acs.analchem.8b00498](https://doi.org/10.1021/acs.analchem.8b00498) | implementação própria sobre o PLS existente; sem dependência nova; precisa de dado real de dois instrumentos | **adotar v1.1** |
| Revisões de transferência de calibração | Isto reúne os métodos de transferência e quando cada um falha. | base para comparar DS/PDS com métodos novos e para o texto do assistente de confiança | Feudale et al., *Chemom. Intell. Lab. Syst.* 2002, [10.1016/S0169-7439(02)00085-0](https://doi.org/10.1016/S0169-7439(02)00085-0); Workman, *Appl. Spectrosc.* 2017, [10.1177/0003702817736064](https://doi.org/10.1177/0003702817736064) | só leitura | observar |
| Análise de imagem hiperespectral, tutorial | Isto descreve o fluxo pixel a pixel (qualidade, segmentação, modelo) e seus erros comuns. | HSI (`guaraci run` modo hsi) e a futura validação em HSI real | Amigo et al., *Anal. Chim. Acta* 2015, [10.1016/j.aca.2015.09.030](https://doi.org/10.1016/j.aca.2015.09.030) | só leitura; validação real depende de dado com licença formal | observar |
| Modelos interpretáveis por desenho | Isto defende modelos que o próprio usuário consegue ler, em vez de explicar caixa-preta depois. | política do projeto (PLS, VIP, regras); sustenta manter shap como opcional | Rudin, *Nat. Mach. Intell.* 2019, [10.1038/s42256-019-0048-x](https://doi.org/10.1038/s42256-019-0048-x) | nenhum | observar (já é a política) |
| Revisão de seleção de variáveis em PLS | Isto compara os métodos de seleção e mostra que o ranking muda com o dado. | aviso do viés da escolha entre métodos (Etapa 4, m-06) | Mehmood et al., *Chemom. Intell. Lab. Syst.* 2012, [10.1016/j.chemolab.2012.07.010](https://doi.org/10.1016/j.chemolab.2012.07.010) | só leitura | observar |
| Validação de modelos quimiométricos, tutorial | Isto explica por que validação cruzada sem grupo e sem conjunto externo superestima o desempenho. | texto do MANUAL §validação e do assistente ("R² alto e Q2 baixo") | Westad e Marini, *Anal. Chim. Acta* 2015, [10.1016/j.aca.2015.06.056](https://doi.org/10.1016/j.aca.2015.06.056) | só leitura | **adotar v1.1** (citar no manual) |
| Autenticação: análise discriminante é inadequada | Isto mostra por que PLS-DA não serve sozinho para dizer "é autêntico" e propõe modelar a classe. | DD-SIMCA e conformal (já adotados); texto do assistente de confiança | Rodionova, Titova e Pomerantsev, *TrAC* 2016, [10.1016/j.trac.2016.01.010](https://doi.org/10.1016/j.trac.2016.01.010) | nenhum | adotado (citação) |
| MSPC clássico (T², Q) | Isto monitora um processo por projeções e aponta qual variável causou o alarme. | `guaraci monitorar` (já implementado) | Kourti e MacGregor, *Chemom. Intell. Lab. Syst.* 1995, [10.1016/0169-7439(95)80036-9](https://doi.org/10.1016/0169-7439(95)80036-9) | nenhum | adotado |
| Taxonomia de deriva de conceito | Isto classifica a deriva (súbita, gradual, recorrente) e os detectores adequados a cada uma. | sentinela de deriva (hoje um teste exato de Fisher sobre a taxa de rejeição) | Gama et al., *ACM Comput. Surv.* 2014, [10.1145/2523813](https://doi.org/10.1145/2523813) | detectores novos exigem dado de produção ao longo do tempo (não temos) | observar |
| PCA para dados composicionais | Isto trata o efeito de a soma ser constante (fechamento) e os outliers antes do PCA. | perfil multielementar ICP, quando as variáveis são frações | Filzmoser, Hron e Reimann, *Environmetrics* 2009, [10.1002/env.966](https://doi.org/10.1002/env.966) | transformação log-razão em numpy; **relevância limitada**: ICP em mg/kg não fecha em 100 % | observar |
| Valores abaixo do limite de detecção em dados composicionais | Isto compara formas de substituir o "< LD" sem distorcer a composição. | política de censura do ICP (hoje, política de censura própria) | Palarea-Albaladejo e Martín-Fernández, *Anal. Chim. Acta* 2013, [10.1016/j.aca.2012.12.029](https://doi.org/10.1016/j.aca.2012.12.029) | só comparação com a regra atual | **adotar v1.1** (como comparação, não como troca) |
| Pré-tratamento: não há receita universal | Isto mostra que o melhor pré-tratamento depende do dado e que a ordem importa. | regra do assistente "escolha de pré-tratamento"; opção `pre_processamento` | Engel et al., *TrAC* 2013, [10.1016/j.trac.2013.04.015](https://doi.org/10.1016/j.trac.2013.04.015) | nenhum | adotado na regra do assistente |
| EMSC | Isto corrige espalhamento e efeitos físicos conhecidos usando um modelo do espectro. | portão de correção de sinal (já implementado) | Afseth e Kohler, *Chemom. Intell. Lab. Syst.* 2012, [10.1016/j.chemolab.2012.03.004](https://doi.org/10.1016/j.chemolab.2012.03.004) | nenhum | adotado |
| Redes convolucionais para regressão NIR | Isto treina uma CNN que aceita espectro quase bruto. | quantificação em conjuntos grandes | Cui e Fearn, *Chemom. Intell. Lab. Syst.* 2018, [10.1016/j.chemolab.2018.07.008](https://doi.org/10.1016/j.chemolab.2018.07.008) | dependência pesada (torch ou similar), perde a interpretabilidade, instalação difícil no Windows do autor | **descartar para a v1.x** (contraria leveza e clareza) |
| Formato JCAMP-DX | Isto é o formato de troca de espectros que o GUARACI já lê. | leitura de `.dx` | McDonald e Wilks, *Appl. Spectrosc.* 1988, [10.1366/0003702884428734](https://doi.org/10.1366/0003702884428734) | nenhum | adotado |
| Princípios FAIR | Isto lista o que os dados e metadados precisam ter para serem achados e reutilizados. | metadados do modelo exportado e da `config_efetiva.yaml` | Wilkinson et al., *Sci. Data* 2016, [10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18) | só um checklist | observar |
| Conformal: introdução e conjuntos adaptativos | Isto dá o guia de predição conformal e um método de conjuntos com cobertura adaptativa para classificação. | conformal de classificação (hoje LAC) | Angelopoulos e Bates, *arXiv* 2021 (**preprint**), [10.48550/arXiv.2107.07511](https://doi.org/10.48550/arXiv.2107.07511); Romano, Sesia e Candès, *arXiv* 2020 (**preprint**), [10.48550/arXiv.2006.02544](https://doi.org/10.48550/arXiv.2006.02544) | implementação própria; preprint é estado da arte, não da prática | observar |

## Cobertura das áreas pedidas

| área | linhas acima |
|---|---|
| seleção e validação para dados espectrais | validação (Westad), seleção (Mehmood), pré-tratamento (Engel), autenticação (Rodionova) |
| conformal e incerteza | conformal (preprints) |
| domain adaptation e transferência de calibração | di-PLS, revisões |
| HSI | tutorial |
| modelos interpretáveis | Rudin; CNN descartada |
| detecção de deriva | MSPC, taxonomia de deriva |
| ICP e dados composicionais | Filzmoser, Palarea-Albaladejo |
| formatos abertos de espectro | JCAMP-DX, FAIR |

## Adiados na última adição de recursos (2026-10-06)

Decisão do autor: a última adição antes da v1.0.0 foi Kruskal-Wallis + Dunn, diagnóstico de premissas do ANOVA e VIP com estabilidade. As técnicas abaixo ficam para `v1.1+`, **sem código agora**. DOIs conferidos no Crossref em 2026-10-06 (registro em `docs/cientifico/referencias_verificadas.yaml`).

**Achado ao escrever esta tabela:** quatro das sete já existem em alguma forma no código; a coluna "o que já existe" diz qual. O que fica para `v1.1+` é só o que **falta** em cada uma.

| técnica | o que faz (1 linha) | o que já existe | o que fica para v1.1+ e por que não agora | referência (DOI conferido) | veredito |
|---|---|---|---|---|---|
| iPLS (PLS por intervalos) | Isto ajusta um PLS em cada faixa do espectro e mostra quais faixas predizem melhor. | Etapa 4 (`selecao_variaveis.selecao_ipls`), com seleção aninhada | comando próprio, sem rodar o pipeline completo; escolher faixa e validar nos mesmos dados dá resultado otimista | Nørgaard et al., *Appl. Spectrosc.* 2000, [10.1366/0003702001949500](https://doi.org/10.1366/0003702001949500) | v1.1+ |
| ALS / arPLS (linha de base) | Isto estima e subtrai a linha de base do espectro por mínimos quadrados assimétricos. | `AirPLS` em `preprocessamento.py` (Whittaker reponderado) | a variante arPLS (reponderação de Baek) comparada à atual pelo portão de correção de sinal | Baek et al., *Analyst* 2015, [10.1039/C4AN01061B](https://doi.org/10.1039/C4AN01061B) | v1.1+ |
| SPA-MLR | Isto escolhe poucas variáveis pouco correlacionadas e ajusta uma regressão linear múltipla. | SPA/APS na Etapa 4 (`selecao_variaveis.selecao_spa`), avaliado por PLS | a regressão linear múltipla como modelo final; mais um seletor a validar com seleção aninhada | Araújo et al., *Chemom. Intell. Lab. Syst.* 2001, [10.1016/S0169-7439(01)00119-8](https://doi.org/10.1016/S0169-7439(01)00119-8) | v1.1+ |
| SVM / SVR | Isto ajusta fronteiras ou regressões não lineares por vetores de suporte. | comparadores no Auto-Benchmark (`avaliacao_modelos.py`), não modelos de produção | SVM/SVR como modelo exportável; perde a interpretabilidade do PLS e exige hiperparâmetros com validação aninhada | Smola e Schölkopf, *Stat. Comput.* 2004, [10.1023/B:STCO.0000035301.49549.88](https://doi.org/10.1023/B:STCO.0000035301.49549.88) | v1.1+ |
| k-NN | Isto classifica pela classe dos vizinhos mais próximos. | nada | sem ganho esperado sobre PLS-DA + DD-SIMCA para o caso de uso; sensível a réplicas vazando entre dobras | Cover e Hart, *IEEE Trans. Inf. Theory* 1967, [10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964) | v1.1+ |
| Teste de Bartlett | Isto testa se as variâncias dos grupos são iguais, supondo normalidade. | Levene e Brown-Forsythe (robustos) em `anova_grupos.py` | o Bartlett é sensível a desvio de normalidade; só valeria como comparação | Bartlett, *Proc. R. Soc. A* 1937, [10.1098/rspa.1937.0109](https://doi.org/10.1098/rspa.1937.0109) | v1.1+ |
| Anderson-Darling | Isto testa se os dados vêm de uma distribuição dada, com peso nas caudas. | Shapiro-Wilk como informação e gráfico quantil-quantil | um segundo teste de normalidade não muda a recomendação (o texto já manda olhar o gráfico) | Anderson e Darling, *J. Am. Stat. Assoc.* 1954, [10.1080/01621459.1954.10501232](https://doi.org/10.1080/01621459.1954.10501232) | v1.1+ |

## O que foi descartado e por quê

- Uma referência de validação de modelos que eu tinha em memória trazia o DOI de outro artigo: o registro oficial devolveu um sensor eletroquímico. Foi descartada e substituída pela entrada correta, achada por busca de título e conferida (Westad e Marini 2015). Isto reproduz, em escala pequena, o erro que o parecer de 2026-09-25 apontou (B6).
- Ideias sem fonte conferida nesta rodada (por exemplo detectores de deriva específicos para espectroscopia) **não entram** neste documento.

## Regras deste radar

Nada daqui vira dependência ou código sem o autor aprovar; "adotar v1.1" é uma recomendação, não uma tarefa. Fonte que não puder ser conferida no registro oficial é descartada e dita como tal.
