# Estudo de técnicas candidatas (v1.1+, só leitura, sem código)

**Público.** Estado: manual. Escopo: UV-Vis, ICP-OES, MIP-OES, FAAS, GFAAS e DMA. Nada aqui foi implementado; o congelamento de funcionalidades continua valendo.

**Sobre "DMA".** Este estudo assume **DMA = analisador direto de mercúrio** (decomposição térmica, amalgamação e absorção atômica do mercúrio). Se a intenção era **análise dinâmico-mecânica** (módulo de armazenamento e perda contra temperatura ou frequência), é outro tipo de dado (curvas contra temperatura, transições), entra como técnica separada e não é tratada aqui.

## 0. O que o produto já oferece e serve a todas as técnicas abaixo

| capacidade | onde | observação |
|---|---|---|
| tabela amostra × variável com unidade declarada (recusa sem unidade) | `icp_multielementar` | o perfil não é de espectro: cada coluna é um analito |
| valores censurados (`<LOD`, `<LOQ`, `ND`, `BDL`, `<0,05`), duas políticas e análise de sensibilidade obrigatória | `aplicar_censura`, `sensibilidade_censura` | a conclusão é refeita com as duas políticas e o relatório diz se muda |
| QC, brancos, material de referência (CRM), padrão interno, recuperação | `relatorio_qc` | decide se a corrida é aceitável antes de comparar grupos |
| ANOVA por analito sobre a unidade independente, FDR, Games-Howell / Tukey / Kruskal-Dunn | `anova_grupos`, `guaraci anova` | pseudo-replicação é recusada |
| regressão PLS com validação agrupada e figuras de mérito (LOD/LOQ por regressão inversa, faixa de decisão) | `chemometric_stats`, `pipeline` | para sinal espectral ou tabela |
| teste de falta de ajuste de uma calibração | `linearity.lack_of_fit_test` | só a falta de ajuste, não a curva univariada inteira |
| leitura de CSV por coluna de comprimento de onda; JCAMP-DX; SPC, SP, OPUS (opcionais) | `dados_io`, `importadores_proprietarios` | UV-Vis comercial entra como CSV ou espectro |
| validação agrupada estável e unidade independente configurável | `StableStratifiedGroupKFold` | vale para qualquer técnica com réplicas |

**O que nenhuma das seis técnicas tem hoje:** curva de calibração **univariada** como objeto (ajuste, pesos, resíduos, intervalo inverso, linearidade formal, escolha da faixa), adição de padrão, correção de deriva por padrão interno ao longo da corrida, e um fluxo "corrida analítica" (sequência de brancos, padrões, amostras, CRMs e verificações) como entrada de primeira classe.

## 1. UV-Vis

- **Já coberto:** espectros por CSV (uma coluna por comprimento de onda), pré-processamentos (SNV, derivadas, linha de base penalizada), PLS-R e PLS-DA, MCR-ALS para misturas, EMSC, figuras de mérito por regressão inversa.
- **Falta:** curva de calibração univariada de Lambert-Beer com verificação de linearidade e saturação; escolha do comprimento de onda analítico; subtração de branco por corrida; correção de dispersão (turbidez) como etapa explícita.
- **Estrutura de dado:** matriz amostra × comprimento de onda (200–800 nm) + metadados (tipo: branco, padrão, amostra; concentração nominal; lote; corrida).
- **Validação sintética de verdade conhecida:** espectros de 2 a 3 absorvedores com perfis gaussianos e concentrações conhecidas, ruído aditivo e uma componente de espalhamento conhecida; critérios: recuperação da concentração (viés < 2%), LOD recuperado dentro de 20% do teórico, linearidade detectada quando se injeta curvatura de 5%.
- **Esforço relativo:** baixo (a infraestrutura espectral existe). **Risco:** baixo; o risco é prometer "método validado" para uma matriz que o produto não conhece.

## 2. ICP-OES

- **Já coberto:** o perfil multielementar inteiro (leitura, unidade, censura, QC, CRM, ANOVA por elemento, FDR, classificação por PLS-DA com validação agrupada).
- **Falta:** seleção de linha de emissão por elemento (escolher entre várias linhas do mesmo elemento pela interferência e pela sensibilidade), correção de interferência espectral, correção de deriva por padrão interno ao longo da corrida, diluição e fator de diluição como metadados, limites por linha.
- **Estrutura de dado:** tabela amostra × (elemento, linha) com unidade, tipo de linha (branco, padrão, amostra, CRM, verificação) e a ordem de leitura.
- **Validação sintética:** corrida simulada com deriva multiplicativa lenta conhecida, linhas com interferência conhecida e valores abaixo do LOD; critérios: a correção por padrão interno devolve a concentração com viés < 3%; a escolha de linha recupera a linha sem interferência; a conclusão sob as duas políticas de censura é relatada.
- **Esforço relativo:** médio. **Risco:** médio (seleção de linha e deriva exigem decisões de método do analista, não apenas código).

## 3. MIP-OES

- **Já coberto:** igual ao ICP-OES (mesmo formato tabular; o produto não distingue a fonte de plasma).
- **Falta:** o que o ICP-OES pede, mais perfis de limites típicos (LOD mais altos, sensibilidade menor para alguns elementos), tratamento de efeitos de matriz de nitrogênio (plasma de microondas) como metadado de método e verificação de ruído de fundo.
- **Estrutura de dado:** a mesma do ICP-OES, com o campo "gás do plasma" e "modo de introdução da amostra".
- **Validação sintética:** a mesma corrida simulada do ICP-OES com LOD maiores; critério: a análise de sensibilidade de censura muda a conclusão exatamente nos elementos cuja fração censurada é alta.
- **Esforço relativo:** baixo se feito junto com o ICP-OES; médio sozinho. **Risco:** baixo.

## 4. FAAS (absorção atômica por chama)

- **Já coberto:** tabela amostra × analito, censura, QC, ANOVA, PLS não se aplica (um elemento por leitura, sem espectro).
- **Falta:** a curva de calibração univariada como objeto (é o centro desta técnica), faixa linear dinâmica curta e curvatura em altas concentrações, adição de padrão para matrizes com efeito, correção de deriva por verificação de padrão a cada N amostras, ressintonia (recalibração) quando a verificação falha.
- **Estrutura de dado:** longa (uma linha por leitura: elemento, tipo, concentração nominal, absorbância, hora, replicata); a unidade independente é a amostra, não a leitura.
- **Validação sintética:** curvas com curvatura conhecida (modelo quadrático suave), heterocedasticidade crescente com a concentração, deriva linear no tempo; critérios: o ajuste ponderado recupera a concentração das amostras com viés < 3%; o intervalo inverso cobre o valor verdadeiro em 95% ± margem de Monte Carlo; a falta de ajuste dispara quando a curvatura passa de 3%.
- **Esforço relativo:** médio (a curva univariada é nova e é a base das próximas duas técnicas). **Risco:** médio (escolha de modelo, pesos e faixa são decisões metodológicas que o produto deve expor, não esconder).

## 5. GFAAS (absorção atômica em forno de grafite)

- **Já coberto:** o mesmo que o FAAS.
- **Falta:** o que o FAAS pede, mais o perfil temporal do sinal (altura de pico contra área integrada), modificador de matriz como metadado, correção de fundo, injeções de volume variável, curva por adição de padrão como padrão (matrizes complexas), e limites muito baixos (a censura passa a ser mais frequente, o que reforça a análise de sensibilidade).
- **Estrutura de dado:** a do FAAS, mais o sinal resolvido no tempo por leitura (opcional) e o programa de temperatura como metadado.
- **Validação sintética:** sinais gaussianos no tempo com área conhecida, fundo conhecido e amostras próximas do LOD; critérios: área recuperada com erro < 2%; LOD/LOQ conforme a definição declarada; comportamento fail-closed quando o perfil temporal está truncado.
- **Esforço relativo:** médio a alto (perfil temporal). **Risco:** médio.

## 6. DMA como analisador direto de mercúrio

- **Já coberto:** tabela de um analito (mercúrio) por amostra, censura, QC, CRM, ANOVA por grupo.
- **Falta:** o mesmo da curva univariada; verificação de memória (efeito de memória entre amostras), controle de brancos de sistema por sequência, massa da alíquota e fator de normalização por massa como metadados obrigatórios, o aviso de que **réplicas de uma mesma alíquota não são unidades independentes** (a regra já existe para o ANOVA e se estende).
- **Estrutura de dado:** longa (amostra, alíquota, massa, sinal, hora, tipo).
- **Validação sintética:** sequência com efeito de memória conhecido (decaimento exponencial entre amostras), massas diferentes e CRM no meio; critérios: o efeito de memória é detectado quando passa de 1% da amostra anterior; a conclusão por grupo usa a amostra como unidade, não a alíquota.
- **Esforço relativo:** baixo se a curva univariada já existir. **Risco:** baixo.

## Ordem sugerida (para o autor decidir)

1. **Curva de calibração univariada** (FAAS e DMA a usam como base, UV-Vis e os demais a reaproveitam): esforço médio, risco médio, destrava três técnicas.
2. **Corrida analítica como entrada** (sequência com brancos, padrões, CRMs e verificações) e correção de deriva por padrão interno: destrava ICP-OES, MIP-OES e GFAAS.
3. **UV-Vis e DMA** como casos de uso da curva; **ICP-OES e MIP-OES** com seleção de linha; **GFAAS** por último (perfil temporal).

Cada técnica entra com a política já usada neste projeto: dado sintético de verdade conhecida para o teste, unidade independente explícita, censura com análise de sensibilidade, nenhuma alegação de conformidade regulatória, e dado de terceiros só para calibrar valores padrão, nunca no repositório nem no pacote.
