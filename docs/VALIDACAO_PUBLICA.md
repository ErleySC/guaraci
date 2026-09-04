# Validação pública — a única base de evidência do software

Desde 2026-08-18 o GUARACI é validado exclusivamente em datasets públicos.
Nenhuma métrica desta página vem de dado privado: todas são reproduzíveis
por qualquer pessoa a partir dos datasets citados.

Reproduzir: `pytest tests/test_validacao_publica.py tests/test_validacao_publica_mendeley.py`
com `GUARACI_DATASETS_DIR` apontando para a pasta que contém os arquivos
(o CI faz o download nos jobs `validacao-publica`/`validacao-publica-mendeley`).
Nenhum dos dois datasets é versionado neste repositório — ver
`datasets/README.md` para a política completa e
`scripts/download_datasets/` para os scripts de aquisição.

---

## 1. Tabela consolidada

| Dataset | Matriz | n | Canais / faixa | Alvo | Métrica obtida | Referência da literatura | Estado |
|---|---|---:|---|---|---|---|---|
| Eigenvector **Corn** (m5) — técnica: **NIR Dispersivo** (Passo 146) | milho em grão | 80 | 700 · 1100–2498 nm | proteína | **RMSEP 0,144 %m/m**; R²val 0,912; 8 LVs | RMSEP típico de PLS: **0,1–0,2** | ✅ dentro da faixa |
| **Tecator** | carne moída | 240 | 100 · 850–1050 nm | gordura | RMSEP 2,001 (`autoscaling`) | ver `docs/BENCHMARK_TECATOR.md` | ✅ dentro do esperado |
| **Mel adulterado** (478 × 700, 4 classes) | mel | — | — | puro vs. 3 xaropes | — | Downey, Fouratier & Kelly (2003), *J. Near Infrared Spectrosc.* 11:447-456 | ❌ **NÃO OBTIDO** (origem identificada, sem repositório público, reconfirmado 2026-08-27) |
| Mendeley `10.17632/ctgg7k4m5g.2` (NIR 8mm) | 19 óleos comestíveis diversos | 100 | 11512 · 3899–14999 cm⁻¹ | classificação (8 espécies, n≥5) + índice de peróxido | **Balanced accuracy 0,35 (CV) / 0,475 (holdout)**; R²cal 0,833 (log10 PV) | balanced accuracy: sem alvo publicado nesta forma (ver §2); RMSEP publicado 4,9 **não reproduzido** (ver §2) | 🟢 **INTEGRADO** (2026-08-27) — classificação valida requisito multimatriz; regressão é sanity check, não gate de literatura |
| Mendeley `10.17632/ctgg7k4m5g.2` (**MIR**, arquivo-irmão do NIR 8mm) | mesmas 19 óleos, mesmas 100 amostras | 100 | 3423 · 699–3999 cm⁻¹ | classificação (8 espécies, n≥5) + índice de peróxido | **Balanced accuracy 0,696**; R²cal 0,79/R²val 0,57 (log10 PV) | idem NIR — sem alvo publicado nesta forma; sanity check, não gate | 🟢 **INTEGRADO** (2026-09-04, Passo 142/143) — R²val positivo, ao contrário do NIR 8mm |
| Mendeley `10.17632/ctgg7k4m5g.2` (**Raman**, arquivo-irmão do NIR 8mm) | idem | 99 (1 amostra sem medição Raman, NaN removida) | 1340 · −18 a 1974 cm⁻¹ (Raman shift) | idem | **Balanced accuracy 0,389**; R²cal 0,67/R²val 0,43 (log10 PV) | idem — artigo original sinaliza Raman como possível correlação por acaso neste dataset | 🟢 **INTEGRADO** (2026-09-04, Passo 142/143) — sinal mais fraco que MIR/NIR, coerente com a ressalva do artigo |
| DeepHS Fruit / Kaki / VIS (Varga, Makowski & Zell, IJCNN 2021) | caqui (imageamento hiperespectral, 64×64×224, Specim FX10) | 56 gravações (38 frutas físicas) | 224 · 397,66–1003,81 nm | ripeness_state (unripe/perfect/overripe) por pixel, agregado por objeto | **5/8 objetos corretos** (teste group-aware) — tende à classe majoritária | — (pipeline HSI, sem alvo de literatura comparável ainda) | 🟡 **EM INTEGRAÇÃO** (2026-09-01) — pipeline funciona ponta-a-ponta, desempenho limitado por desbalanceamento severo (ver §7) |
| Mendeley `10.17632/thkcz3h6n6.6` (**Fluorescência**, LED 1) | 24 óleos de oliva (grau EXTRA/VIRGEN/LAMPANTE) | 24 (média de 20 repetições técnicas) | 1024 canais de emissão (índice, sem nm calibrado) | grau de qualidade (3 classes, n≥5) | **Balanced accuracy 0,383 (CV)** — fraco, logo acima do acaso (~0,333) | sem alvo publicado nesta forma | 🟡 **INTEGRADO** (2026-09-04, Passo 142/143) — sinal fraco, n pequeno (ver §2d) |
| Figshare `10.6084/m9.figshare.4307804` (**RMN**) | 97 azeites de oliva (Abruzzo/Itália) | 97 | 125 variáveis ppm (já binadas pelos autores) | província de origem (Pescara/Teramo) | **Balanced accuracy 0,500 — EXATAMENTE o acaso** (binário) | artigo original reporta 99% com LDA + seleção geoestatística de variável (não reproduzido) | 🔴 **NEGATIVO, documentado** (2026-09-04, Passo 142/143) — motor genérico não separa; ver §2e para a hipótese |
| ERIC/Eawag `10.25678/000D3C19` (**UV-Vis**, sensor scan/Spectrolyser) | esgoto bruto (campanha de 25 semanas, flume) | 82 dias (agregado; 533 amostras de laboratório antes da agregação) | 215 · 200–735 nm | DOC (carbono orgânico dissolvido, mg/L) | **R²cal 0,616 / R²val 0,650**; RMSEP 34,2 mg/L; 7 LVs (EMSC+MC) | sem RMSEP publicado para este recorte (sanity check, não gate) | 🟢 **INTEGRADO** (2026-09-04, Passo 147) — EMSC (já aprovado no Corn, §9) capturou sinal real; ver §2g |

O RMSEP do Corn está no meio da faixa publicada — nem baixo demais (o que
sugeriria vazamento) nem alto demais (bug de pré-processamento). É esse
resultado que sustenta a afirmação "o motor está correto", e é o único tipo
de afirmação que este repositório pode fazer sobre desempenho.

---

## 2. Mendeley `10.17632/ctgg7k4m5g.2` — INTEGRADO (2026-08-27)

### O que o dataset realmente é (correção de premissa)

A descrição original do projeto ("azeite adulterado com milho/canola/
amêndoa") **não corresponde à estrutura real dos dados** — verificado
por leitura direta dos arquivos e do artigo original (Ottaway et al.
2021, *Applied Spectroscopy* 75(6):700-709,
doi:10.1177/0003702821994500; ver também Gilbraith et al. 2025,
*Journal of Spectral Imaging*). O dataset tem **19 tipos de óleo puro**
como classes separadas (azeite extra-virgem/leve/puro, abacate,
amendoim, milho, semente de uva, canola, girassol, vários *blends*,
entre outros — ver `OilClassKey.csv`) — corn/canola/almond aparecem só
como **seus próprios óleos puros**, nunca como "azeite adulterado com
X%". Não existe coluna de percentual de adulteração em nenhum arquivo.

O alvo de quantificação real é o **índice de peróxido** (mEq O2/kg —
grau de oxidação/rancidez de óleos envelhecidos naturalmente 5-7 anos),
não um percentual de mistura. É um alvo cientificamente legítimo e
diretamente compatível com a maquinaria de regressão PLS do GUARACI,
só que **não é a mesma alegação** de "detectar adulteração por mistura"
que o dataset privado do autor testa.

### Arquivos usados (dos 10 do dataset)

Só `NIR8mm1A.csv` (caminho óptico 8mm) e `OilClassKey.csv` (legenda
número→nome de classe) são baixados e usados. O artigo original testou
4 técnicas (NIR 8mm, NIR 24mm, MIR 50µm, Raman) para o mesmo alvo
(índice de peróxido); **NIR 8mm teve o RMSEP publicado mais baixo E
mais confiável** (RMSEP=4,9 — o próprio artigo sinaliza o RMSEP do
Raman, 6,9, como resultante de correlação por acaso). Os outros 8
arquivos (MIR, Raman, NIR 24mm/2mm) não são baixados nem usados nesta
integração — sem necessidade para o que está sendo validado.

**NaN**: uma linha de `Raman1A.csv` (não usado aqui) tem NaN em todas
as colunas espectrais — amostra sem aquela medição específica tomada.
`NIR8mm1A.csv` (usado) tem **zero NaN**, confirmado por leitura direta;
o teste (`tests/test_validacao_publica_mendeley.py::_carregar_bruto`)
falha alto se isso mudar no futuro, em vez de descartar em silêncio.

### Classificação (prova do requisito multimatriz)

`test_multimatriz_declara_perfil_correto_e_classifica_acima_do_acaso`:
classes com <5 amostras são excluídas (classificar sobre n=1 não é
checagem honesta) — restam 8 espécies, 78 amostras. Perfil de matriz
novo, `oleos_comestiveis_nir` (NIR, 8mm, cm⁻¹, vocabulário próprio),
aplicado **sem nenhuma alteração de código-fonte** — só
`cfg.matrix_profile`. Confirmado: model card declara "óleo comestível
(NIR, 8mm)" e **não** declara vocabulário de nenhuma outra matriz
(nem `milho em grão`, nem `óleo vegetal` do dataset privado).

Balanced accuracy: **0,35 em CV** (permutação Y-randomization: p=0,167,
**não significativo a p<0,05** — reportado honestamente, não escondido)
e **0,475 no holdout externo** (16 amostras). Regime genuinamente
difícil: n=62 amostras de treino / 8 classes desbalanceadas (5 a 27
amostras/classe) / ~11500 variáveis colineares. O teste garante "acima
do acaso" (piso 0,25, acaso teórico ~0,125 para 8 classes), não
"desempenho de produção" — não é a mesma alegação que o dataset privado
do autor sustenta com muito mais amostras por classe.

### Quantificação (índice de peróxido) — RMSEP publicado NÃO reproduzido

`test_regressao_peroxido_roda_sem_excecao_e_calibra_razoavel`: modelo
PLS **global/pooled** (mesma escolha metodológica do artigo original,
"a global peroxide value model", não estratificado por espécie), alvo
`log10(índice de peróxido)` (assimetria 1,5–165, >100× de amplitude).

**Medido em 2026-08-27, com os presets padrão do GUARACI e holdout
independente de 25 amostras:**

| Transformação | RMSEP | R²cal | R²val |
|---|---|---|---|
| Bruto | 25,9 | 0,71 | **−0,87** |
| log10 | 0,49 (unid. log) | 0,83 | **−0,53** |

Nenhuma das duas reproduz o RMSEP publicado (4,9) — R²val fica
**negativo em ambos os casos** (pior que prever a média) no holdout de
25 amostras. Com n=100 e ~11500 canais colineares, esse é um holdout de
alta variância; o artigo não detalha protocolo suficiente (CV exata,
remoção de outliers, se "RMSEP" ali é holdout independente ou RMSECV)
para reproduzir o número com confiança. **Decisão registrada
(2026-08-26): não perseguir o número publicado por tentativa-e-erro**
— isso ajustaria o teste ao resultado em vez de reproduzi-lo de
verdade. O teste ficou como *sanity check* (roda sem exceção, R²cal >
0,3 — confirma que a calibração captura sinal real), não como gate
contra a literatura. Isso é uma limitação registrada, não escondida.

### Como reproduzir

```
python scripts/download_datasets/baixar_mendeley_oleos.py
GUARACI_DATASETS_DIR=datasets_publicos pytest tests/test_validacao_publica_mendeley.py -v
```

### 2b. MIR e Raman — os arquivos-irmãos do NIR 8mm (Passo 142/143, 2026-09-04)

Auditoria das 11 técnicas do menu `cli_assistente.TECNICAS` (Passo 141)
achou que só FT-NIR tinha validação real; as outras 10 eram metadado de
menu (faixa espectral + preset de pré-processamento) sobre um motor
genérico nunca exercitado contra dado real daquela técnica. Antes de
sair buscando dataset novo por técnica (Passo 142), uma checagem no que
já estava integrado mostrou que **o próprio dataset Mendeley usado para
NIR já contém MIR e Raman das MESMAS 100 amostras** (`MIR1A.csv`,
`Raman1A.csv` — arquivos-irmãos de `NIR8mm1A.csv` no mesmo repositório,
mesma licença CC BY 4.0, mesmo artigo). Confirmado por leitura direta:
`Class`/`PeroxideValue` são idênticos linha-a-linha nos 3 arquivos.
`scripts/download_datasets/baixar_mendeley_oleos.py` foi estendido para
baixar os 2 arquivos novos (SHA256/tamanho pinados a partir da API
pública do Mendeley, que já devolve o hash calculado por eles — conferido
que bate com o hash de `NIR8mm1A.csv` já pinado desde 2026-08-26).

Nenhum perfil de matriz dedicado existe para MIR/Raman de óleos
comestíveis (só NIR tem `oleos_comestiveis_nir`) — os testes usam
`matrix_profile="generico"` com `wn_min`/`wn_max` explícitos por
técnica. Criar perfis dedicados fica como pendência (fora do escopo
aprovado deste passo).

**Medido em 2026-09-04** (mesmo protocolo do §2: classificação com 8
espécies com ≥5 amostras, n=78; regressão pooled em `log10(índice de
peróxido)`, holdout de 25 amostras, seed=0):

| Técnica | n | Canais | Bal. acc. (classif.) | R²cal | R²val | RMSEP (log10) |
|---|---:|---:|---:|---:|---:|---:|
| NIR 8mm (§2, referência) | 100 | 11512 | 0,475 (holdout) | 0,83 | **−0,53** | 0,49 |
| MIR | 100 | 3423 | **0,696** | 0,79 | **0,57** | 0,26 |
| Raman | 99 (1 NaN removida) | 1340 | 0,389 | 0,67 | **0,43** | 0,26 |

Achado que vale registrar sem sicofantear: nesta medição pontual, tanto
MIR quanto Raman tiveram **R²val positivo** — ao contrário do NIR 8mm,
que ficou negativo (§2). Isso não vira "MIR é melhor que NIR para este
problema" — é UMA medição, com holdout de alta variância (mesma ressalva
do §2, n=100), e o próprio artigo original usa o NIR 8mm como a técnica
de referência (RMSEP publicado mais confiável entre as 4). Os testes
(`test_validacao_publica_mendeley_mir_raman.py`) usam limiares de
sanidade com folga sob o valor medido (R²cal > 0,5 MIR / 0,4 Raman;
balanced accuracy > 0,4 MIR / 0,2 Raman) — não um gate de literatura,
mesma política do §2.

Reproduzir:
```
python scripts/download_datasets/baixar_mendeley_oleos.py
GUARACI_DATASETS_DIR=datasets_publicos pytest tests/test_validacao_publica_mendeley_mir_raman.py -v
```

### 2c. Busca de dataset para UV-Vis, Fluorescência e RMN (Passo 142, 2026-09-04)

Antes de integrar Fluorescência/RMN (§2d/§2e), a busca completa por
técnica (candidato, formato, licença, viabilidade — exigido pelo Passo
142) ficou registrada assim:

| Técnica | Candidato | Formato (inspecionado por download direto) | Licença | Viabilidade |
|---|---|---|---|---|
| RMN | Figshare `4307804` (óleo, origem geográfica) | CSV pronto: 125 variáveis ppm já binadas | CC0 | ✅ dado ótimo; ⚠️ **Figshare bloqueia download automatizado** (desafio de bot AWS WAF, não CAPTCHA) — usado via download manual, ver §2e |
| Fluorescência (simples) | Mendeley `thkcz3h6n6` (óleo, grau de qualidade) | CSV com coluna `Data` codificando espectro 1D (1024 pts) como string de lista | CC BY 4.0 | ✅ usado, ver §2d |
| Fluorescência (EEM real) | Mendeley `g6y69g8gwm` (óleo envelhecendo) | Zip de 52 MB, CSVs brutos de instrumento por etapa de envelhecimento — EEM genuína (múltiplas amostras × ~35 excitações × varredura de emissão) | CC BY 4.0 | ⚠️ baixa sem bloqueio, mas exige parser novo (formato bruto, não tabular); **não integrado** — registrado como pendência para quando houver EEM real de verdade no roadmap |
| UV-Vis | Mesmo `g6y69g8gwm` (traz painel UV também) | CSV de relatório de instrumento, só 4 comprimentos de onda | CC BY 4.0 | ⚠️ real mas fraco (4 pontos, não uma varredura) — **não integrado** |
| UV-Vis (melhor candidato) | Artigo "Bangladeshi honey UV-vis-NIR" (1960 amostras, %adulteração 0–40%) — alvo quase idêntico ao caso de uso central do GUARACI | — | — | ❌ página atrás de CAPTCHA (ScienceDirect) — não contornado (regra permanente); nenhum repositório de dados público localizado nesta busca |

HPLC/GC-MS/IMS: busca não iniciada ainda (fora do escopo desta rodada).

### 2d. Fluorescência Molecular — Mendeley `thkcz3h6n6` (Passo 142/143, 2026-09-04)

24 azeites de oliva com grau de qualidade oficial (EXTRA/VIRGEN/
LAMPANTE — 10/8/6 amostras), espectros de emissão de fluorescência
(1024 pontos) em 2 LEDs de excitação × 20 repetições técnicas por
amostra. Licença CC BY 4.0.

Decisões registradas (não escondidas): só o LED 1 foi usado (escolha
arbitrária de um canal, mesmo espírito da escolha do NIR 8mm em §2);
as 20 repetições técnicas por (amostra, LED) foram **médias antes de
treinar** — o motor do GUARACI só suporta agrupamento via convenção
`mae_id` própria do projeto, que não se aplica a um dataset externo;
forçar um `mae_id` artificial contaminaria o vocabulário (o problema
que `perfil_matriz.py` existe para evitar — Passo 141). Colapsar as
repetições elimina o risco de vazamento na raiz, ao custo de n=24 (não
480). O dataset não publica eixo de emissão calibrado em nm — usado
índice de canal.

**Medido em 2026-09-04**: balanced_accuracy = **0,383** (CV) — acima
do acaso (~0,333 para 3 classes), mas um sinal fraco, coerente com
n=24 pequeno. Testei a hipótese de que subtrair o espectro de fundo do
instrumento (fornecido no próprio dataset) melhoraria o sinal —
**resultado idêntico** (0,383 com e sem subtração): o preset padrão
MSC+SG+MC já remove qualquer offset constante por amostra antes da
subtração ter chance de importar.

Reproduzir:
```
python scripts/download_datasets/baixar_mendeley_fluorescencia_oleo.py
GUARACI_DATASETS_DIR=datasets_publicos pytest tests/test_validacao_publica_mendeley_fluorescencia.py -v
```

### 2e. RMN — Figshare `4307804` (Passo 142/143, 2026-09-04) — achado NEGATIVO documentado

97 azeites de oliva da região Abruzzo/Itália, perfil ¹H-NMR já binado
pelos autores originais em 125 variáveis de deslocamento químico (ppm)
— zero NaN, pronto para o motor CSV genérico sem nenhum binning novo
do GUARACI. Alvo: província de origem, codificada no prefixo do
`Sample_ID` (`pe`=Pescara, 50; `te`=Teramo, 47 — confirmado por leitura
das coordenadas lat/long de cada grupo). Licença **CC0**.

**Limitação de acesso** (não do dado): o Figshare bloqueia download
automatizado deste arquivo com um desafio de bot da AWS WAF — não é
CAPTCHA, mas `scripts/download_datasets/baixar_figshare_azeite_nmr.py`
tenta o caminho automático e, se falhar, orienta o download manual com
instruções claras. O arquivo usado nesta auditoria foi obtido assim
(hash SHA256 conferido).

**Achado real, medido em 2026-09-04 — NÃO escondido**: ao contrário de
NIR/MIR/Raman/Fluorescência, a classificação por província com o motor
GENÉRICO do GUARACI (PLS-DA, 125 variáveis) ficou em
**balanced_accuracy = 0,500 — EXATAMENTE o nível do acaso** para um
problema binário. Testado com 4 presets de pré-processamento
diferentes (`msc_sg_mc`/`snv_mc`/`autoscaling`/`sg_mc`) — todos deram o
MESMO 0,500. Hipótese razoável (mesma disciplina do achado `unripe` do
HSI, §7): o próprio GUARACI reporta que só ~32% das 125 variáveis
carregam sinal acima do ruído (aviso `[AVISO] Faixa espectral`) — o
artigo original (que reporta 99% de acurácia) **não** usou PLS-DA
ingênuo sobre todas as variáveis; usou um teste geoestatístico (I de
Moran) para selecionar quais variáveis têm autocorrelação espacial
antes de rodar LDA só' nelas. A separação por província provavelmente
existe em poucas variáveis específicas, não no espectro inteiro —
reproduzir os 99% do artigo exigiria implementar seleção de variável
por geoestatística, fora do escopo deste passo. O teste
(`test_validacao_publica_figshare_azeite_nmr.py`) não tem gate de
"aprendeu algo" — só confirma que o pipeline roda sem exceção e produz
um número válido, porque um gate de sucesso aqui inventaria um
resultado que a medição não mostra.

Reproduzir:
```
python scripts/download_datasets/baixar_figshare_azeite_nmr.py
GUARACI_DATASETS_DIR=datasets_publicos pytest tests/test_validacao_publica_figshare_azeite_nmr.py -v
```

### 2f. NIR Dispersivo — reclassificação do Corn, sem dataset novo (Passo 146, 2026-09-04)

Antes de sair buscando um dataset novo para a técnica #2 do menu (NIR
Dispersivo, `cli_assistente.TECNICAS["nir"]`), a instrução do Passo 146
pediu para reverificar se algum dataset já baixado nesta sessão tinha
sido classificado erroneamente. O Corn (linha do topo desta tabela, §1)
está integrado desde os Passos 78/79 mas **nunca tinha sido atribuído a
nenhuma das 11 técnicas do menu** no levantamento do Passo 141 — contava
só como prova de que "o motor reproduz a literatura", solto da tabela.

**Achado confirmado por busca direta (não presumido):** os 3
instrumentos do Corn (`m5`/`mp5`/`mp6`) são da família **FOSS
NIRSystems 5000/6500**. Digman, Cherney & Cherney (2022), *Sensors*
22(2):658, doi:10.3390/s22020658 — artigo que compara diretamente um
espectrômetro NIR de bancada contra um FT-NIR portátil — descreve
textualmente: "The (FOSS) NIRSystem 6500 (FOSS, Hillerød, Denmark) is a
**scanning monochromator spectrometer** with a wavelength range from
1100 to 2498 nm" — faixa **idêntica** à do Corn (1100-2498nm,
confirmado em `tests/test_validacao_publica.py::
test_guaraci_reproduz_a_literatura_no_corn`). O mesmo artigo usa esse
instrumento como contraponto direto a um FT-NIR (interferômetro de
Michelson) medido no mesmo estudo — monocromador de rede com varredura
mecânica e Fourier-transform são categorias tecnológicas distintas na
própria literatura de instrumentação NIR, não uma distinção inventada
para este projeto.

**Conclusão: o Corn nunca foi FT-NIR.** É NIR Dispersivo de livro-texto,
só nunca conectado à tabela de 11 técnicas até agora. Nenhuma mudança
de código foi necessária — o teste já existente já reproduz RMSEP=0,144
%m/m (proteína, m5), R²val=0,912, 8 LVs (reconfirmado por execução
direta nesta rodada: `corn.mat` baixado de novo, SHA-256 e tamanho
conferidos contra os valores pinados no CI — `e28fd4be...c46b5`,
1445616 bytes, batem). Split: `frac_holdout=0.25`,
`group_by_mae_id=False` — correto para este dataset: cada uma das 80
amostras tem 1 medição por instrumento, sem repetição técnica a
agrupar, logo nenhum grupo pode vazar entre treino e validação.

**Estado da técnica #2 (NIR Dispersivo) na tabela de 11 técnicas:
Parcial → Funcional.** `docs/PROGRESSO.md` Passo 146 tem o registro
completo.

### 2g. UV-Vis — ERIC/Eawag, esgoto bruto (Passo 147, Fase A, 2026-09-04)

Busca dedicada (Zenodo API, ScienceDirect bloqueado por CAPTCHA em pelo
menos 2 candidatos — regra permanente, não contornado) encontrou
Lechevallier et al. (2025), "Dataset on wastewater quality monitoring
with adsorption and reflectance spectroscopy in the UV/Vis range",
*Scientific Data* 12:1296, doi:10.1038/s41597-025-05459-x — campanha de
25 semanas medindo esgoto bruto (Suíça) com 2 espectrofotômetros UV-Vis
(Spectrolyser/`scan`, 200-735nm; ISA, 200-706nm) a cada 2 minutos, e 533
amostras coletadas manualmente e analisadas em laboratório para 9
indicadores (turbidez, DOC, TSS, TOC, N dissolvido, N total, NH4, PO4,
SO4). Dados publicados em ERIC open (Eawag), **licença CC BY** (campo
`license_id` da API pública do portal CKAN,
`opendata.eawag.ch/api/3/action/package_show`).

Só o arquivo `2_data.zip` (~357 MB, CSV) é baixado — o pacote completo
tem mais ~180 GB de cubos hiperespectrais (formato ENVI) que este
projeto não usa; a validação de UV-Vis usa só as tabelas de sensor e
laboratório. Script:
`scripts/download_datasets/baixar_eawag_esgoto_uvvis.py` (mesma
disciplina de segurança do §6: HTTPS, SHA-256+tamanho pinados,
verificados antes de gravar em disco — arquivo grande, então
transmitido em streaming para um temporário e só promovido ao destino
final depois de bater os dois, em vez de carregado inteiro em memória
como os scripts de arquivo pequeno).

**Decisões metodológicas (documentadas, não escondidas):**
- **Sensor**: `scan` (Spectrolyser) — faixa completa publicada (200-735nm)
  sem descarte de canais, ao contrário do ISA (que descarta UV baixo por
  absorção da fibra óptica, conforme o próprio artigo).
- **Alvo**: DOC (carbono orgânico dissolvido) — um dos 5 indicadores
  medidos para as 533 amostras inteiras (turbidez/DOC/N dissolvido/
  NH4/PO4/SO4; TSS/TOC/N total só têm 45 amostras) e o correlato
  clássico de absorbância UV-Vis na literatura de monitoramento de água
  (UV254 como substituto de DOC).
- **Casamento amostra-de-laboratório ↔ espectro do sensor**: pelo
  timestamp mais próximo, tolerância de 3 minutos (o sensor mede a cada
  2 min); 513/529 amostras de laboratório casaram dentro da tolerância.
- **Agrupamento (regra 5, "group-aware em qualquer validação nova")**: o
  motor `mode="csv"` do GUARACI (`load_csv`) trata toda coluna que não
  seja classe/conc como CANAL espectral — não há como passar uma coluna
  de agrupamento arbitrária sem quebrar o parser. Em vez de ignorar o
  risco (várias coletas manuais no MESMO DIA são temporalmente
  autocorrelacionadas), as amostras foram **agregadas por dia** (média
  do alvo e do espectro casado) antes de montar o CSV — mesma solução já
  usada para a Fluorescência (§2d, colapsar repetições técnicas).
  Resultado: **82 dias** (não 533 amostras), cada um uma unidade física
  independente — `group_by_mae_id=False` é correto aqui pela mesma razão
  do Corn (nenhum grupo repetido a proteger depois da agregação).
- **Pré-processamento**: EMSC (já aprovado no portão de aceite contra o
  Corn, §9) + mean-centering, isolado (sem SNV/SG, que vêm ligados por
  padrão em `default_preprocessing != "custom"` — ver achado de bug
  abaixo) — por pedido explícito da instrução ("validar com EMSC já
  disponível como opção de pré-processamento").

**Medido em 2026-09-04** (holdout 25%, seed=0, 82 dias → ~61 treino/CV,
~21 holdout): RMSEC=20,83, RMSECV=31,71, **RMSEP=34,25 mg/L**, **R²cal
0,616 / R²val 0,650**, 7 LVs, RPD=1,79/RER=7,3 (ambos "não utilizável"
pela faixa publicada — DOC em esgoto bruto tem alta variabilidade
diária, 23-269 mg/L). Sanity check, não gate de literatura (não há
RMSEP publicado para este recorte específico: sensor `scan`, alvo DOC,
agregação diária) — o que o teste garante é R²cal/R²val positivos e
substanciais, ou seja, a calibração capturou sinal real, não ruído.

**Achado colateral (bug de rotulagem, não de cálculo) durante este
passo**: `pipeline.generate_output_name` (nome da pasta de saída) só
checava `apply_snv`/`apply_sg`/`apply_mc` no ramo `default_preprocessing
== "custom"` — qualquer execução com EMSC/AirPLS/OSC (ex.: todo o portão
de aceite do §9, ou Raman com AirPLS) tinha esses passos REALMENTE
aplicados mas OMITIDOS do nome da pasta, que então sub-representava o
que rodou (nunca afetou o cálculo em si, só a rastreabilidade do nome).
Corrigido em `src/guaraci/pipeline.py` para declarar os 3, na mesma
ordem em que `preprocessamento.build_preprocessor` os aplica. Teste de
regressão novo:
`tests/test_pipeline_core.py::test_gerar_nome_saida_custom_declara_emsc_airpls_osc`.

Reproduzir:
```
python scripts/download_datasets/baixar_eawag_esgoto_uvvis.py
GUARACI_DATASETS_DIR=<pasta> pytest tests/test_validacao_publica_eawag_esgoto_uvvis.py -v
```

### Histórico — RETRATAÇÃO de 2026-08-18

> **A entrada anterior desta seção afirmava "❌ NÃO OBTIDO" com a rota
> `?version=2` retornando HTTP 403 e exigindo sessão de navegador.
> Reconfirmado por comando direto em 2026-08-26 (Passo 78): o endpoint
> SEM o parâmetro `?version=2` funciona e nunca exigiu sessão.** A
> tentativa de 2026-08-18 usou a rota errada (`.../files?version=2`,
> que de fato devolve erro) em vez do endpoint correto de metadados do
> dataset. Não se sabe se o endpoint mudou de comportamento ou se a
> investigação original simplesmente não tentou essa rota — o registro
> anterior não detalha todas as variações testadas o suficiente para
> distinguir os dois casos.

| Tentativa (2026-08-18) | Resultado |
|---|---|
| `data.mendeley.com/public-api/datasets/ctgg7k4m5g/files?version=2` | **HTTP 403** (reconfirmado 2026-08-26: `{"error":400}`, ainda bloqueado) |
| `api.data.mendeley.com/datasets/ctgg7k4m5g/2` | **HTTP 404** (reconfirmado 2026-08-26: ainda 404) |
| `api.data.mendeley.com/datasets/ctgg7k4m5g/2/files` | **HTTP 404** (reconfirmado 2026-08-26: ainda 404) |
| `doi.org/api/handles/10.17632/ctgg7k4m5g.2` | **200** — o DOI resolve, o dataset existe |
| Página de destino (`data.mendeley.com/datasets/ctgg7k4m5g/2`) | **200**, 125 KB — SPA, links montados por JS |

| Tentativa NOVA (2026-08-26) | Resultado |
|---|---|
| `data.mendeley.com/public-api/datasets/ctgg7k4m5g` (SEM `?version=2`) | **HTTP 200**, JSON completo: metadados + lista de 10 arquivos + `download_url` por arquivo (`ATRAdulteration3.csv`, `ATRPure3.csv`, `MIR1A.csv`, `NIR24mm1A.csv`, `NIR24mm1B.csv`, `NIR2mm1B.csv`, `NIR8mm1A.csv`, `OilClassKey.csv`, `Raman1A.csv`, `Raman2.csv`; ~129,5 MB total) |
| `HEAD` no `download_url` do menor arquivo (`OilClassKey.csv`, 401 bytes) | **302** → redireciona para URL assinada S3 (`prod-dcd-datasets-public-files-eu-west-1.s3...`) — download real, sem autenticação, confirmado funcional |

**Licença confirmada no JSON da API (não só no HTML): CC BY 4.0**
(`data_licence.short_name`) — compatível com uso e redistribuição
mediante atribuição.

Este era o estado em 2026-08-26 (Passo 78, só levantamento de
acessibilidade). A integração completa (download automático,
classificação, quantificação, perfil de matriz, CI multiplataforma)
foi feita em 2026-08-27 (Passo 79) — ver o topo desta seção §2.
Nenhum leitor novo em `io_registry.py` foi necessário: os CSVs do
Mendeley já vêm no formato largo que o `mode="csv"` do GUARACI
consome diretamente (`Class,PeroxideValue,<wavenumbers...>`).

### Mel adulterado (478 amostras, 700 comprimentos de onda, 4 classes) — NÃO OBTIDO, reconfirmado 2026-08-27

**Origem identificada (achado novo, Passo 80): Downey, G.; Fouratier, V.;
Kelly, J.D. (2003). "Detection of honey adulteration by addition of
fructose and glucose using near infrared transflectance spectroscopy."
*Journal of Near Infrared Spectroscopy* 11:447-456.** — mel irlandês
artesanal adulterado com frutose/glicose, transflectância NIR, reflector
dourado, amostra de 0,1mm. As 4 classes (puro, xarope de beterraba
invertido, mistura frutose:glicose, xarope de milho rico em frutose) e
n=478/700 canais batem com a descrição usada neste projeto — citado em
"Comparison of Machine Learning Models in Food Authentication Studies"
(Singh & Domijan, arXiv:1905.07302) e em ao menos outro artigo de
seleção robusta de variáveis (arXiv:2010.10415), nenhum dos dois com
link de dado/código público.

**Buscado ativamente em 2026-08-27** (arXiv, ResearchGate, GitHub,
CRAN, página institucional dos autores) — **nenhum repositório público,
pacote R/Python, ou arquivo suplementar foi localizado** com o dado em
si, só citações do artigo de 2003 em trabalhos posteriores. Diferente
do Mendeley (§2), aqui não há endpoint de API nem link de download a
verificar — a busca chegou ao limite do que é razoável sem contatar os
autores diretamente (fora do escopo desta auditoria automatizada).

**Não foi substituído por um dataset de mel qualquer**: a alegação a
validar é o requisito multimatriz com *n* adequado para classificação
puro vs. adulterado, e um dataset diferente não a sustenta.

Consequência assumida: o perfil `mel_vis_nir` existe e é carregável, mas
está marcado no próprio YAML como **declarado, não validado com dado real**.
O requisito multimatriz foi provado com outros dois pares de matrizes
(milho/óleo privado — §3; óleos comestíveis Mendeley — §2).

---

## 3. Prova do requisito multimatriz

`tests/test_perfil_matriz.py::test_aceitacao_multimatriz_milho_e_oleo_sem_tocar_em_codigo`

Roda o pipeline completo em duas matrizes de naturezas diferentes — uma em
**nm**, outra em **cm⁻¹**, com vocabulários distintos — alterando **um único
campo de configuração** (`cfg.matrix_profile`). Verifica que:

- cada model card declara a sua matriz (`milho em grao` / `oleo vegetal`);
- **nenhum declara a da outra** — este era exatamente o defeito medido em
  2026-08-17, quando o card do milho afirmava "óleo vegetal amazônico";
- o vocabulário de classe acompanha (`variedade` / `especie`);
- o perfil usado fica registrado no card, para quem o lê depois.

Prova adicional com dado público de terceiro (§2, Passo 79):
`tests/test_validacao_publica_mendeley.py::test_multimatriz_declara_perfil_correto_e_classifica_acima_do_acaso`
— perfil `oleos_comestiveis_nir` aplicado ao dataset Mendeley sem
alteração de código, model card declara "óleo comestível (NIR, 8mm)" e
não declara vocabulário de nenhuma outra matriz (nem `milho em grao`
nem `oleo vegetal`).

E `test_perfil_inexistente_aborta_o_pipeline_antes_de_predizer`: matriz sem
perfil cadastrado levanta `UnknownProfileError` **antes** de qualquer
predição, com a lista de perfis disponíveis e a instrução de como escrever
um novo. Nunca cai num padrão de outra matriz em silêncio.

---

## 4. Licenças dos datasets

| Dataset | Licença / termos | Verificado em |
|---|---|---|
| Eigenvector Corn | distribuído publicamente pela Eigenvector Research para benchmarking; ver a página da fonte para os termos | página da fonte, 2026-08-17 |
| Tecator | domínio público (StatLib) | `docs/BENCHMARK_TECATOR.md` |
| Mendeley `ctgg7k4m5g` | **CC BY 4.0** | JSON da API oficial (`data_licence.short_name`), reconfirmado 2026-08-27 |
| DeepHS Fruit (Kaki) | **não declarada formalmente** (sem SPDX no repo/README; `api.github.com/repos/cogsys-tuebingen/deephs_fruit` devolve `license: None`) — autores afirmam publicamente "we make public" o dataset (README, paper IJCNN 2021) e distribuem por HTTP sem autenticação; mesmo tratamento já dado ao Corn nesta tabela | leitura direta do repo + API do GitHub, 2026-09-01 |
| ERIC/Eawag `000D3C19` (esgoto UV-Vis) | **CC BY** | campo `license_id` da API pública do portal CKAN, `opendata.eawag.ch/api/3/action/package_show`, 2026-09-04 |

Nenhum destes arquivos é versionado neste repositório.

**Retratação (2026-09-01):** uma busca inicial (WebSearch, via Papers with
Code) sugeriu licença CC BY-SA 4.0 para o DeepHS Fruit. Verificação
direta (README, ausência de arquivo `LICENSE`, `api.github.com`) **não
confirma** essa licença — corrigido aqui antes de qualquer uso da
alegação em código/documentação, conforme a regra "evidência ou
silêncio" desta auditoria.

**Decisão explícita do usuário (2026-09-01):** apresentadas 3 opções
(manter o tratamento atual / contatar os autores antes de prosseguir /
substituir por dataset com licença explícita), o usuário escolheu
**manter o tratamento atual** — uso exclusivo para validação, nunca
redistribuído, baixado sob demanda a cada execução, mesmo padrão já
aplicado ao Corn nesta mesma tabela. Nenhuma cláusula do dataset proíbe
esse uso; a integração do Passo 93-102 permanece como está.

---

## 5. Métricas de quantificação agora reportadas

A tabela do §1 só é interpretável porque o RMSEP vem acompanhado. Desde
esta rodada o pipeline reporta, para toda regressão:

| Métrica | Onde | Observação |
|---|---|---|
| RMSEC / RMSECV / RMSEP | log, `resumo_modelo.txt` | já existiam |
| R²cal / R²val / bias | idem | já existiam |
| **SEP** | idem | erro-padrão de predição corrigido pelo bias |
| **RPD** | idem | `SD(y_ref) / SEP`, **com a faixa de uso ao lado** |
| **RER** | idem | `amplitude(y_ref) / SEP` |
| LOD / LOQ / SEN / seletividade | `figS3_merito_regressao` | Valderrama, Braga & Poppi (2009); exige réplicas físicas |

RPD e RER nunca saem nus: `interpret_rpd()` anexa a faixa publicada
(Williams 2014, em Williams, Dardenne & Flinn, *J. Near Infrared
Spectrosc.* 22(2):85-93; AACC 39-00.01). Um número cru convida a
comparações indevidas entre estudos; a faixa carrega a referência que a
define.

Nota honesta sobre **LOD/LOQ**: eles são calculados, mas dependem de
réplicas físicas para estimar o ruído instrumental. Em datasets sem
réplicas — o Corn é um deles — saem `N/A`, e é correto que saiam: um LOD
estimado sem base de repetibilidade seria um número inventado.

## 6. Convenção obrigatória para scripts de download (segurança)

Reafirmado por varredura de segurança em 2026-08-27 (Bloco 13d, Frente 3b)
sobre `scripts/download_datasets/baixar_mendeley_oleos.py` — o único
script deste tipo hoje, já em conformidade; documentado aqui para que
**qualquer script de download futuro siga o mesmo padrão desde o
primeiro commit**, não como correção depois do fato:

1. **HTTPS sempre**; nunca HTTP puro.
2. **Tamanho esperado (bytes) E SHA-256 esperado, pinados no código** —
   nunca lidos de um arquivo externo que o próprio download poderia
   trocar.
3. **Verificar os dois ANTES de gravar em disco** (o conteúdo baixado
   fica em memória até passar na checagem — `_baixar_um` em
   `baixar_mendeley_oleos.py` é o padrão de referência). Se o hash não
   bater, `raise RuntimeError` — nunca gravar um arquivo não verificado,
   nunca seguir em frente com um aviso apenas.
4. **URL hardcoded no código**, nunca construída a partir de entrada do
   usuário ou de variável de ambiente (evita um vetor de SSRF/redirect
   trivial).
5. Cache local fora do controle de versão (`$GUARACI_DATASETS_DIR`),
   nunca comitar o dado de terceiro.

O mesmo princípio (checksum pinado, verificado antes de confiar no
conteúdo) já se aplica em `.github/workflows/test.yml` para o download
direto do Corn (`CORN_SHA256`/`CORN_BYTES`) — os dois caminhos (script
Python para Mendeley, `curl`+`sha256sum` inline para Corn) seguem a
mesma regra por caminhos diferentes.

---

## 7. HSI (imageamento hiperespectral) — em integração (2026-09-01)

Ver `INSTRUCAO_HSI_MINIMO_VIAVEL.md` e `docs/PROGRESSO.md` para o
detalhamento passo a passo. Resumo do estado nesta rodada:

- **Literatura citada (Passo 92):** 2 das 3 referências confirmadas por
  busca direta (Crossref/WebSearch) — "Cross-domain hyperspectral image
  classification" (*Pattern Recognition* 168, dez/2025) e "Chemometric
  and machine-learning strategies for calibration transfer"
  (*Chemometrics and Intelligent Laboratory Systems*, 2026). A terceira
  ("Framework de padronização... HSI", PII `S2772375526007070`) **não
  foi confirmada** — o ISSN implícito corresponde a um periódico real
  (*Smart Agricultural Technology*, Elsevier), mas o artigo específico
  não foi localizado por nenhuma via de busca disponível nesta sessão.
  **Não citada** em nenhum lugar do código/documentação, conforme a
  regra do Passo 92.
- **Dataset público (Passo 93):** DeepHS Fruit (Varga, Makowski & Zell,
  IJCNN 2021) — subconjunto Kaki/câmera VIS (56 gravações, 38 frutas
  físicas, rótulo `ripeness_state`). Formato ENVI (`.hdr`+`.bin`)
  confirmado por leitura direta. Ver §4 para a licença (não declarada
  formalmente, mesmo tratamento do Corn).
- **Infraestrutura de leitura (Passo 94) e quality gate (Passo 95):**
  `src/guaraci/hsi_io.py`, `src/guaraci/hsi_quality.py` — testados
  contra o dataset real (`tests/test_hsi_io.py::
  test_load_deephs_kaki_dataset_real`, requer `GUARACI_DATASETS_DIR`
  apontando para `deephs_kaki_vis/`, obtido via
  `scripts/download_datasets/baixar_deephs_kaki.py`).
- **Segmentação (Passo 96):** `src/guaraci/hsi_segmentation.py` — PCA
  (PC1) + Otsu (implementado do zero, não depende de scikit-image).
  Dataset não tem máscara de referência; validado por **inspeção visual
  documentada** (`resultados_hsi_segmentacao/kaki_segmentacao_amostra.png`,
  gitignorado) — o contorno da fruta aparece claramente isolado do fundo.
  **Retratação (2026-09-01):** a primeira versão desta função assumia
  "objeto = minoria de pixels da cena" — correto para a cena sintética
  do teste, mas **invertido** no dataset real (a fruta ocupa ~59% do
  quadro, maioria) — a máscara marcava os CANTOS/fundo como "objeto". O
  erro só apareceu ao reexaminar a própria imagem salva pela inspeção
  visual exigida pelo Passo 96 (o texto do relatório inicial descreveu a
  máscara como "correta" sem essa reconferência — corrigido aqui antes
  de qualquer uso downstream). Corrigido para inferir o fundo pela BORDA
  da imagem (pixels mais externos), não pela fração de área — cobre os
  dois casos (objeto minoria ou maioria). Novo teste de propriedade
  (objeto majoritário) adicionado para não regredir.
- **Extração de ROI + agrupamento por objeto físico (Passo 97):**
  `src/guaraci/hsi_pixels.py`. Contra-prova obrigatória (Hypothesis)
  confirma que `StableStratifiedGroupKFold` nunca separa pixels do mesmo
  objeto físico entre treino/validação, em nenhum fold.
- **PLS-DA por pixel + agregação por voto majoritário (Passo 98):**
  `src/guaraci/hsi_classification.py` — reaproveita `PLSDAClassifier`
  já existente, seleção de LVs por parsimônia de Wold (mesmo critério de
  `pipeline.py`, generalizado para classificação). **Medido contra o
  dataset real** (8 objetos de teste, split group-aware,
  `n_components=5` após a correção da segmentação abaixo): **5/8
  objetos corretos** — o modelo ainda tende a "perfect" (a classe
  majoritária, 42/56 gravações). Regime
  genuinamente difícil por desbalanceamento severo (2 `unripe`, 12
  `overripe`, 42 `perfect`), reportado honestamente — mesmo padrão já
  registrado para o Mendeley (§2: bal.acc 0,35 CV). Não é um resultado
  de produção; confirma que o pipeline mecânico funciona ponta-a-ponta
  sobre dado real, não que o desempenho é bom.
- **Explicabilidade cruzada (Passo 100):** `src/guaraci/hsi_chemistry.py`
  — cruza VIP com tabela de atribuição química (VIS 397-1004nm, fruta;
  3 entradas citando Merzlyak, Solovchenko & Gitelson 2003 e Osborne,
  Fearn & Hindle 1993). Medido contra o dataset real: 4 das 5 bandas de
  maior VIP caem em 540-550nm, dentro da faixa de carotenoides/
  antocianinas — consistente com a fisiologia do amadurecimento.
- **Validação externa (Passo 101):** `src/guaraci/hsi_validation.py` —
  partição nativa por dia de medição (`day_8_m3`+`day_9_m3` como teste
  externo nunca visto no treino). Sensibilidade/especificidade/precisão
  reportadas separadas por classe e por interno/externo — ver tabela
  completa em `docs/PROGRESSO.md` Passo 101. Números ruidosos e por
  vezes contra-intuitivos (n pequeno por classe/partição), reportados
  sem suavizar — inclusive `unripe` com sensibilidade zero nas duas
  partições (só 2 gravações dessa classe no dataset inteiro).
- **Integração ao menu/CLI (Passo 102):** `hsi_pipeline.run_hsi_pipeline`
  orquestra o fluxo completo (leitura → quality gate → segmentação →
  classificação por pixel → mapa espacial → explicabilidade → validação
  externa) numa única chamada, distinto do `mode="imagem"` (nunca
  confundidos — ver docstring de `hsi_pipeline.py`). Acessível pela tecla
  `[X]` do menu principal da CLI (`_menu_hsi`), testado de ponta a ponta
  contra o dataset real, inclusive pelo caminho do usuário (digitar o
  caminho da pasta na tela, não chamar a função Python diretamente).
  Novo campo `hsi_dataset_folder`/`hsi_pasta_dataset` no `_CONFIG_SPEC`,
  alcançável nas duas interfaces (CLI e app web) e coberto pelas redes de
  segurança sistêmicas do projeto (`test_todo_campo_do_spec_e_alcancavel_
  por_algum_menu`, `test_todo_campo_do_config_spec_aparece_no_app/no_
  menu_cli`).

**Fatia "mínimo viável" (Passos 92-102) concluída** — pipeline HSI
completo, testado contra dado público real em cada etapa, integrado ao
menu principal da CLI.

### Passos 103-110 — robustez, validação ampliada, honestidade estatística

Ver `INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md` e `docs/PROGRESSO.md` para
o detalhamento passo a passo. Resumo:

- **Texto/UI (Passo 103):** frase solta "Protótipo 'mínimo viável'" e
  cabeçalho fixo "Técnica: FT-NIR" (herdado do template genérico,
  incorreto para HSI) corrigidos — ver `docs/PROGRESSO.md` Passo 103
  para a decisão registrada de não usar o carimbo formal "PROTOTYPE
  OUTPUT" (critério objetivo diferente, não se aplica ao HSI).
- **Validação expandida (Passo 104):** as 4 frutas restantes do DeepHS
  Fruit (Avocado, Kiwi, Mango, Papaya) e todas as câmeras disponíveis
  (nenhuma fruta tem as 3 câmeras — medido, não presumido) baixadas e
  testadas com o **mesmo pipeline, sem alteração de código** — só os
  parâmetros de `load_deephs_fruit_dataset`. Achado real durante a
  validação: resolução de imagem varia ~24× entre frutas (Kaki
  64×64=4096 pixels/imagem vs. Avocado/VIS ~286×294=~97000) — sem um
  teto de pixels por gravação, o fit de PLS-DA estourava memória
  (2,8GB numa única alocação). Corrigido com subamostragem por
  gravação (`hsi_pixels.build_pixel_dataset(max_pixels_por_gravacao=
  2000)`, pixels reais, nunca inventados, RNG semeado — reprodutível),
  mesmo teto aplicado a TODAS as combinações (comparação justa: câmera
  de alta resolução não ganha mais peso na agregação por objeto só por
  ter mais pixels).

  **Tabela comparativa (validação externa por dia, Passo 101 estendido
  — sensibilidade interno/externo por classe; tabela completa com
  especificidade/precisão em `tests/test_validacao_publica_deephs_
  fruit.py`, reproduzível com `GUARACI_DATASETS_DIR` apontando para
  `deephs_fruit_all/`):**

  | Fruta | Câmera | n objetos | interno/externo n | overripe sens (int/ext) | perfect sens (int/ext) | unripe sens (int/ext) |
  |---|---|---:|---|---|---|---|
  | Avocado | NIR | 24 | 5/5 | 0,00/0,60 | 1,00/0,00 | 0,50/0,00 |
  | Avocado | VIS | 54 | 11/9 | 0,00/0,33 | 1,00/0,33 | 0,00/0,00 |
  | Kiwi | NIR | 35 | 8/2 | 0,00/0,00 | 0,83/1,00 | 0,00/0,00 |
  | Kiwi | VIS | 87 | 19/12 | 0,00/0,40 | 1,00/1,00 | 0,00/0,00 |
  | Mango | VIS | 36 | 6/11 | 0,00/0,00 | 0,00/0,00 | **1,00/1,00** |
  | Mango | VIS_COR | 38 | 7/10 | 0,00/0,00 | 1,00/1,00 | 0,00/0,00 |
  | Papaya | VIS | 26 | 5/5 | 0,00/0,00 | 1,00/1,00 | 0,00/0,00 |
  | Papaya | VIS_COR | 25 | 5/6 | 0,00/0,00 | 1,00/1,00 | 0,00/0,00 |

  Padrão consistente com o já visto no Kaki (§7 acima): colapso quase
  total na classe majoritária (`perfect` na maioria das combinações),
  exceto Mango/VIS onde `unripe` é a classe favorecida — reforça que o
  colapso segue a distribuição real de classes do subconjunto, não um
  viés fixo do modelo. **Achado não-óbvio**: `Kiwi/VIS` é a ÚNICA
  combinação onde as 3 classes têm `n>=19` (limiar de
  `n_minimum_for_alpha(0.05)`, ver Passo 105) — MESMO ASSIM `unripe`
  sai com sensibilidade 0,00 interno E externo. Ou seja, o problema não
  é só tamanho de amostra: há uma dificuldade de separabilidade real
  entre `unripe` e as outras classes nessa combinação, não corrigível
  só com mais dados — achado honesto, não escondido atrás da hipótese
  mais confortável de "só falta n".

- **Desbalanceamento (Passo 105):** `src/guaraci/hsi_resampling.py` —
  reamostragem group-aware (nunca duplica pixel fora do grupo) +
  `class_evaluability_report`, aplicado às 8 novas combinações acima
  (asteriscos no log de teste marcam classes com `n<19`, "não
  avaliável estatisticamente" — mesma linguagem já usada no resto do
  projeto). Contra-prova obrigatória (Hypothesis) confirma que a
  reamostragem nunca separa pixels do mesmo objeto entre
  treino/validação.
- **Conjunto aberto (Passo 106):** `src/guaraci/hsi_identification.py`
  — granularidade de calibração MEDIDA (não presumida): objetos por
  fruta (28-88) e por fruta×câmera (24-87), as duas ≥ 19 em TODAS as
  10 combinações reais — escolhida a mais fina. Contra-prova: tipo
  espectral fora do treino retorna "desconhecido".
- **Incerteza (Passo 107):** `src/guaraci/hsi_uncertainty.py` —
  heterogeneidade de pixel vira relatório formal. Decisão registrada:
  **sem** combinação Bonferroni entre etapas do fluxo HSI (só 1 etapa
  tem alpha formal hoje — Identificação; razão completa no docstring
  do módulo e em `docs/PROGRESSO.md`).
- **Domínio de aplicabilidade (Passo 108):**
  `src/guaraci/hsi_applicability.py` — reaproveita
  `chemometric_stats.training_applicability_domain`/
  `applicability_domain_new_samples` sem alteração (pixel = amostra,
  mesma granularidade do PLS-DA). Sensor incompatível (bandas
  diferentes) detectado explicitamente, nunca um erro cru de shape.
  Contra-prova: cena sintética fora do domínio rejeitada em >90% dos
  pixels.
- **Datasets adicionais (Passo 109):** melhor candidato encontrado —
  **Olive Dataset** (Mendeley `10.17632/8xvhcsdvst.1`), ENVI,
  **CC BY 4.0 confirmado via API oficial** (licença que o DeepHS Fruit
  não tem), matriz nova (azeitona). **Não integrado nesta rodada** —
  decisão de integrar fica para o usuário. Lista completa de
  candidatos avaliados em `docs/PROGRESSO.md` Passo 109.
- **Licença do DeepHS Fruit (Passo 110):** rascunho de e-mail aos
  autores preparado em `docs/RASCUNHOS_CONTATO.md` — **não enviado**,
  aguarda revisão/envio manual do usuário.

### Passo 111 — HSI aceita dado do próprio usuário, offline

Falha de arquitetura corrigida: até aqui, `run_hsi_pipeline` exigia
`manifest.json` de um dataset público específico — o resto do GUARACI
sempre aceitou pasta do próprio usuário; o HSI era a exceção.

`hsi_io.load_hsi_folder_dataset(pasta)`: lê qualquer pasta com cubos
ENVI (`.hdr`+`.bin`), convenção de subpasta-por-classe. Agrupamento por
amostra física reaproveita a hierarquia de 3 níveis já validada no modo
`imagem` (Bloco 8, 2026-08-25) — extraída para `agrupamento_pastas.py`
em vez de duplicada. `run_hsi_pipeline` despacha automaticamente
(presença de `manifest.json`) entre o caminho ORIGINAL (dataset
público, validação externa por dia + explicabilidade química,
inalterado) e o caminho NOVO (só validação interna group-aware, SEM
explicabilidade química — a tabela `ATRIBUICAO_QUIMICA_VIS_FRUTA` é
conhecimento específico do dataset público, aplicá-la a comprimento de
onda arbitrário do usuário seria alegação científica falsa).

**Prova de independência (contra-prova obrigatória, não alegação de
prosa):** `tests/test_hsi_offline_prova.py` — cubo hiperespectral
sintético gerado localmente (zero download), `socket.socket`
monkeypatchado para levantar exceção em qualquer tentativa de conexão
de rede, pipeline completo (leitura → quality gate → segmentação →
classificação → mapa → confiança por objeto → validação) rodando sem
tocar rede — medido, não afirmado.

### Passo 112 — Investigação do problema `unripe` (Kiwi/VIS): 3 hipóteses testadas, nenhuma resolveu

O achado do Passo 104 (tabela acima): `Kiwi/VIS` é a ÚNICA combinação
com as 3 classes `n≥19`, mas `unripe` ainda sai com sensibilidade 0,00
interna E externa. 3 hipóteses testadas contra o dataset real (código
reprodutível em `tests/test_investigacao_unripe_kiwi_vis.py`, requer
`GUARACI_DATASETS_DIR` apontando para `deephs_fruit_all/`):

- **Hipótese A — seleção de banda química** (clorofila 660-680nm +
  carotenoide/antocianina 500-550nm, mesma tabela de
  `hsi_chemistry.ATRIBUICAO_QUIMICA_VIS_FRUTA`, 26 das 224 bandas):
  **não melhora** — `unripe` permanece sens(int/ext)=0,00/0,00,
  idêntico ao espectro completo. Restringir a banda quimicamente
  "óbvia" não resgata o sinal.
- **Hipótese B — fronteira contínua** (`storage_days`, metadado real de
  dias de armazenamento, como proxy de maturação; PLS-R com CV
  group-aware por dia): **não confirma** — Q²=-0,17 (sem generalização
  entre dias, pior que prever a média), e a média predita de
  `unripe` (6,36 dias) fica quase idêntica à de `perfect` (6,17 dias) —
  recast contínuo não separa as classes melhor que a discreta. Ressalva
  própria da hipótese: `storage_days` é um proxy RUIDOSO do rótulo
  visual (`unripe` real tem `storage_days` médio de 6,2, MAIOR que o de
  `perfect`, 5,2 — avaliação humana visual não é função determinística
  do tempo de armazenamento neste dataset).
- **Hipótese C — sobreposição espectral real**: medida com Mahalanobis
  entre centroides (espectro médio por objeto, `unripe` n=28 vs.
  `perfect` n=39) em PCA de baixa dimensão (2 componentes, 86,7% da
  variância, matriz bem condicionada) = **0,384** — distância PEQUENA.
  Efeito por-banda (|diferença de média|/desvio-padrão combinado,
  espectro completo, sem PCA): mediana **0,376**, máximo 0,803 — efeito
  fraco-a-moderado, não uma separação clara. **Achado metodológico
  colateral**: a MESMA distância de Mahalanobis calculada em PCA de
  alta dimensão (10 componentes, 99,9% da variância) sobe para 1,048 —
  não porque há mais separação real, mas por MAL-CONDICIONAMENTO da
  covariância estimada com poucas amostras (n≈30-40/classe) em
  dimensão alta (achado desta investigação, não hipotético — medido e
  reportado em `tests/test_investigacao_unripe_kiwi_vis.py`, que
  confirma explicitamente que o número cresce com a dimensão). A
  leitura confiável (baixa dimensão, bem condicionada) e o efeito
  por-banda univariado concordam: **sobreposição espectral substancial
  entre `unripe` e `perfect`** nesta câmera/resolução (397-1004nm,
  Specim FX10).

**Fechamento (Passo 112)**: nenhuma das 3 hipóteses resgatou a
classificação de `unripe`. A evidência mais robusta (Hipótese C, medida
com cuidado metodológico contra o próprio artefato que ela quase
produziu) aponta para **limite real de separabilidade espectral** entre
os estágios `unripe`/`perfect` de Kiwi nesta banda de câmera — não um
bug de implementação, não resolvível só com seleção de banda (A) nem
com reformulação contínua (B). Nenhuma das 3 foi implementada como
opção configurável no pipeline (nenhuma mostrou melhora real que
justificasse isso). 4 testes novos, suíte completa, ruff/mypy limpos.

### Passo 114 — Hipótese D: a diferença é real (firmeza objetiva confirma), a técnica é que tem sensibilidade fraca

A Hipótese B (acima) levantou uma suspeita: `unripe` tinha
`storage_days` médio MAIOR que `perfect` — fisiologicamente
contraintuitivo o suficiente para questionar se o rótulo `ripeness_state`
em si é confiável (se não fosse, a conclusão certa seria "ruído de
rótulo", não "sobreposição espectral", e a Hipótese C precisaria ser
**retratada**).

O manifest do DeepHS Fruit publica `firmness` — medição OBJETIVA de
firmeza por fruto (penetrômetro ou equivalente), independente do
rótulo visual. Testado diretamente (`tests/
test_investigacao_unripe_kiwi_vis.py::
test_hipotese_d_firmeza_objetiva_confirma_rotulo_nao_e_ruido`, 81/87
objetos com firmeza medida — `firmness=0`/`None` só em `overripe`,
nunca em `unripe`/`perfect`, confirmado por leitura direta do
manifest):

| Classe | n | firmeza média | dp |
|---|---:|---:|---:|
| unripe | 28 | 2083,9 | 459,4 |
| perfect | 39 | 1398,1 | 374,0 |
| overripe | 14 (73 com dado ausente/piso) | 548,2 | 568,1 |

Ordem `unripe > perfect > overripe` — fisiologicamente correta (fruta
amolece ao amadurecer). Mann-Whitney U (unripe vs. perfect):
**p=8,87×10⁻⁸**; Cohen's d=**1,64** (efeito GRANDE, separação limpa).

**Não é retratação — é refinamento.** O rótulo `ripeness_state` é
respaldado por medição independente: NÃO é ruído de rótulo. A
conclusão da Hipótese C (nenhuma hipótese resgata a classificação,
"limite de separabilidade") permanece — mas agora com uma leitura mais
precisa: a diferença FÍSICA entre `unripe` e `perfect` é real e
substancial (d=1,64 na firmeza, contra d≈0,38-0,80 no sinal espectral
por-banda) — a câmera VIS (397-1004nm, reflectância) é que tem
sensibilidade fraca a essa diferença especificamente, não evidência de
que as duas classes sejam fisicamente indistinguíveis em geral. Um
sensor de textura/firmeza (ou NIR com penetração diferente) poderia, em
princípio, separar melhor — questão em aberto, fora do escopo desta
investigação (que cobriu as 3 hipóteses pedidas + a 4ª motivada pelo
próprio achado da B).

**Conclusão definitiva (Passo 121, fecha a investigação — não é mais
"em aberto"):** a diferença física entre `unripe` e `perfect` é real e
estatisticamente robusta (medição objetiva de firmeza, Mann-Whitney
p=8,87×10⁻⁸, Cohen's d=1,64). A câmera VIS do DeepHS Fruit,
especificamente, não captura essa diferença no espectro de
refletância — não é ruído de rótulo, é limite de sensibilidade
espectral desta técnica de aquisição para este estágio de maturação.

**Checagem adicional — câmera NIR do mesmo Kiwi (Passo 121, não
bloqueante, exploratória):** o dataset também tem Kiwi/NIR (58
gravações, 35 objetos: unripe n=7, perfect n=19, overripe n=9 —
amostra BEM menor que a do VIS, resultado abaixo é observacional, não
um teste de hipótese formal). Efeito por-banda (|diferença de
média|/desvio-padrão combinado, espectro NIR completo, mesmo espírito
da Hipótese C): mediana **0,968**, máximo 1,119 — cerca de 2,5× MAIOR
que o efeito medido em VIS (mediana 0,376), fisicamente plausível (NIR
é sensível a mudanças de umidade/estrutura celular durante o
amadurecimento, que a reflectância VIS não capta tão bem). **Mas a
classificação por pixel em Kiwi/NIR também falhou para `unripe`**
(sens 0,00/0,00, mesma tabela do Passo 104) — com n_unripe=7 objetos,
a causa mais provável dessa falha específica é tamanho de amostra
insuficiente para o split group-aware, não falta de sinal espectral.
Ou seja: **evidência adicional de que o limite é da câmera VIS
especificamente**, não da matriz Kiwi em geral — mas não uma
confirmação completa (precisaria de mais objetos `unripe` de NIR pra'
testar isso com uma classificação de verdade, fora do escopo desta
rodada).

Suíte completa, ruff/mypy limpos.

### Passo 123 — Hipótese NIR (registro para investigação futura, NÃO conclusão)

**Hipótese não confirmada, amostra pequena**: o sinal espectral de
maturação do Kiwi parece mais concentrado na faixa NIR (950-1700 nm)
que na VIS, possivelmente ligado a mudanças de composição química
(açúcar, ácido, degradação de clorofila) que absorvem mais fortemente
fora do visível. Se confirmado com amostra maior, isso orientaria a
recomendação de câmera para esse tipo de aplicação.

Esta é uma hipótese especulativa, distinta e mais fraca que a conclusão
DEFINITIVA do Passo 121 acima (que é sobre a câmera VIS especificamente
não captar uma diferença fisiológica REAL e confirmada por firmeza
objetiva) — aqui a pergunta é se a câmera NIR captaria MAIS dessa
diferença, e a amostra é pequena demais pra' afirmar isso com confiança.

**Checagem em outra fruta com câmera NIR disponível (Avocado — única
outra opção no DeepHS Fruit além do Kiwi)**: o padrão NÃO se repete.

| Fruta | Câmera | n (unripe/perfect) | efeito por-banda mediano |
|---|---|---|---:|
| Kiwi | VIS | 28/39 | 0,376 |
| Kiwi | NIR | 7/19 | 0,968 (**2,5× maior** que VIS) |
| Avocado | VIS | 9/29 | 0,548 |
| Avocado | NIR | 6/11 | 0,301 (**0,55×** — MENOR que VIS) |

**Conclusão honesta**: a hipótese "NIR concentra mais sinal de
maturação que VIS" é **específica do Kiwi neste dataset, não um padrão
geral** — o Avocado mostra o efeito na direção OPOSTA (VIS maior que
NIR). Ambas as amostras de NIR são pequenas (n_unripe=6-7), então nem a
direção Kiwi nem a direção Avocado devem ser tratadas como
estabelecidas — apenas registradas como hipóteses concorrentes para
investigação futura com amostra maior. Nenhuma alegação forte é feita
aqui sobre qual câmera é geralmente melhor para maturação de fruta.

## 8. Isolamento físico de dataset de terceiro (Passo 118, P0)

Auditoria por comando direto — checagem P0 explícita, não alegação.

- **`git ls-files` (árvore atual)**: nenhum arquivo com extensão de
  dado bruto de terceiro (`.mat`/`.hdr`/`.bin`/`.raw`/`.zip`) está
  versionado; nenhum arquivo versionado passa de 512 KB.
- **`git rev-list --objects --all` + `cat-file --batch-check`
  (HISTÓRICO COMPLETO, não só a árvore atual)**: o maior blob de todo o
  histórico é `guaraci_icon.png` (~2,7 MB, ícone legítimo do projeto);
  o resto são revisões históricas de `guaraci.py` (código-fonte, não
  dado). **Nenhum dataset público jamais foi commitado, em nenhuma
  revisão** — não é só "não está lá hoje", nunca esteve.
- Tamanho total de `.git`: 24 MB — consistente com "zero blob de
  dataset", nunca teria esse tamanho se algo como o DeepHS Fruit
  (23 GB) tivesse passado por um commit e sido removido depois (objetos
  removidos de commits recentes continuam no `.git` até um `gc`
  agressivo, então esse tamanho pequeno já é evidência forte).
- **Mecanismo único confirmado**: os 3 scripts de
  `scripts/download_datasets/` usam exatamente o mesmo padrão —
  `os.environ.get("GUARACI_DATASETS_DIR", "datasets_publicos")` — sem
  segundo mecanismo de cache paralelo. `datasets_publicos/` (o
  fallback) está coberto pelo `.gitignore`.
- Prova automatizada e repetível em `tests/test_isolamento_datasets.py`
  (roda em toda suíte, não gated por `GUARACI_DATASETS_DIR` — é
  checagem sobre o repositório em si): árvore atual, histórico
  completo, cobertura do `.gitignore`, e consistência do mecanismo
  entre os 3 scripts.

**Resultado**: nenhum achado grave. `datasets/README.md` atualizado com
a política completa e a tabela de todos os datasets já integrados
(incluindo DeepHS Fruit — todas as frutas, que faltava na tabela).

## 9. Portão de aceite (Bloco 20): EMSC/OSC e PDS/DS no Corn (Passos 134-135)

Reproduzível: `GUARACI_DATASETS_DIR=... pytest tests/test_validacao_publica.py -k portao -v`.

Nenhuma técnica de correção de sinal entra recomendada sem prova de
ganho em validação bloqueada (Wilcoxon pareado, 10-20 seeds
independentes, split group-aware repetido — `portao_correcao_sinal.py`,
Bloco 20). Aplicado ao Corn real (único dataset público deste projeto
com replicação entre instrumentos/proteína como alvo contínuo):

| Técnica | Cenário | RMSEP sem | RMSEP com | p (Wilcoxon) | Veredito |
|---|---|---:|---:|---:|---|
| EMSC | Corn/m5, proteína | 0,164 | 0,132 | 0,002 | ✅ aprovado |
| OSC | Corn/m5, proteína | 0,164 | 0,145 | 0,002 | ✅ aprovado |
| PDS | Corn, m5→mp5 | 0,91* | 0,18* | <0,01 | ✅ aprovado (confirma o já conhecido) |
| DS | Corn, m5→mp5 | 0,88* | 0,50* | <0,001 | ✅ aprovado (**retrata** achado anterior) |

\* média sobre múltiplos seeds — mais alta que o RMSEP "sem" de um único
split (0,51, o já publicado) porque splits aleatórios diferentes variam
bastante em dificuldade; o veredito usa a distribuição inteira, não um
ponto só.

**PDS**: contra-prova do próprio mecanismo do portão — reproduz
formalmente (10 seeds) o resultado já conhecido de
`test_transferencia_de_calibracao_reduz_erro_entre_instrumentos_do_corn`
(RMSEP ~0,51→~0,16 num único seed=0). Confirmado: o portão não inventa
nem contradiz um resultado já validado.

**DS — retratação**: uma nota anterior (rodada de transferência de
calibração) registrava "DS não ajudou". Medido agora contra 20 seeds
(não 1): DS **ajuda de verdade** (p<0,001, vence em 16/20 seeds), só que
muito mais fraco e menos consistente que PDS — PDS sempre chega a
~0,15-0,22, DS fica em ~0,44-0,58. A causa provável da alegação anterior:
checar um único split (seed=0), onde por coincidência DS sai
ligeiramente pior (0,510→0,528) — ilusão clássica de N=1. Regra que
fica: nunca generalizar "PDS sempre funciona"/"DS nunca funciona" — o
veredito é por par de instrumentos/dataset, e aqui os dois ajudam, em
graus muito diferentes.

**Acervo privado de óleo** (não registrado nesta página por política —
ver linha 1-5 — mas resumido para contexto): EMSC aprovado (RMSEP
4,70→4,39, p=0,002), OSC **rejeitado** (4,70→4,99, piorou, p=0,002) na
quantificação pooled de teor de adulterante. Resultado real em
`docs/PROGRESSO.md`, Passo 134.
