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
| Eigenvector **Corn** (m5) | milho em grão | 80 | 700 · 1100–2498 nm | proteína | **RMSEP 0,144 %m/m**; R²val 0,912; 8 LVs | RMSEP típico de PLS: **0,1–0,2** | ✅ dentro da faixa |
| **Tecator** | carne moída | 240 | 100 · 850–1050 nm | gordura | RMSEP 2,001 (`autoscaling`) | ver `docs/BENCHMARK_TECATOR.md` | ✅ dentro do esperado |
| **Mel adulterado** (478 × 700, 4 classes) | mel | — | — | puro vs. 3 xaropes | — | Downey, Fouratier & Kelly (2003), *J. Near Infrared Spectrosc.* 11:447-456 | ❌ **NÃO OBTIDO** (origem identificada, sem repositório público, reconfirmado 2026-08-27) |
| Mendeley `10.17632/ctgg7k4m5g.2` (NIR 8mm) | 19 óleos comestíveis diversos | 100 | 11512 · 3899–14999 cm⁻¹ | classificação (8 espécies, n≥5) + índice de peróxido | **Balanced accuracy 0,35 (CV) / 0,475 (holdout)**; R²cal 0,833 (log10 PV) | balanced accuracy: sem alvo publicado nesta forma (ver §2); RMSEP publicado 4,9 **não reproduzido** (ver §2) | 🟢 **INTEGRADO** (2026-08-27) — classificação valida requisito multimatriz; regressão é sanity check, não gate de literatura |
| DeepHS Fruit / Kaki / VIS (Varga, Makowski & Zell, IJCNN 2021) | caqui (imageamento hiperespectral, 64×64×224, Specim FX10) | 56 gravações (38 frutas físicas) | 224 · 397,66–1003,81 nm | ripeness_state (unripe/perfect/overripe) por pixel, agregado por objeto | **5/8 objetos corretos** (teste group-aware) — tende à classe majoritária | — (pipeline HSI, sem alvo de literatura comparável ainda) | 🟡 **EM INTEGRAÇÃO** (2026-09-01) — pipeline funciona ponta-a-ponta, desempenho limitado por desbalanceamento severo (ver §7) |

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

Nenhum destes arquivos é versionado neste repositório.

**Retratação (2026-09-01):** uma busca inicial (WebSearch, via Papers with
Code) sugeriu licença CC BY-SA 4.0 para o DeepHS Fruit. Verificação
direta (README, ausência de arquivo `LICENSE`, `api.github.com`) **não
confirma** essa licença — corrigido aqui antes de qualquer uso da
alegação em código/documentação, conforme a regra "evidência ou
silêncio" desta auditoria.

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
- **Integração ao menu/CLI (Passo 102):** ainda não implementada nesta
  rodada — em andamento.
