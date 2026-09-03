# PROGRESSO — Passo 117 (2026-09-03)

## Passo 117 — Auditoria de adaptabilidade multimatriz/multitecnica

Repetido o teste de aceite multimatriz (`test_perfil_matriz.py`) para
os 3 modos que faltavam, cada um com perfil/dominio FICTICIO nunca
visto pelo pacote (`tests/test_aceitacao_adaptabilidade.py`):

- **Tabular**: YAML de perfil (matriz "resina de cupuacu", inventada)
  escrito num tmp_path como um usuario faria, carregado pelo CAMINHO
  (nao precisa estar em `src/guaraci/perfis_matriz/`) -- roda
  `executar()` ponta-a-ponta, model card declara so' o vocabulario
  certo, zero linha de codigo alterada.
- **Imagem colorimetrica**: tecnica ficticia nova ("microscopio_digital")
  combinada com o perfil "generico" via `combine_profiles`/
  `save_profile` (fluxo do Agente 5B) -- roda mode="imagem"
  ponta-a-ponta.
- **HSI**: dominio TOTALMENTE alheio a fruta -- autenticidade de
  comprimido farmaceutico ("autentico"/"falsificado") via
  `load_hsi_folder_dataset` + `run_hsi_pipeline`, confirmando que o
  carregador do Passo 111 nao tem NENHUMA amarracao ao vocabulario do
  DeepHS Fruit.

**2 achados de arquitetura medidos e reportados (NAO corrigidos
sozinhos -- decisao de escopo fica pro usuario)**:

1. **Identificacao (Bloco 9b) e' estruturalmente amarrada ao conceito
   de "adulterante"**, nao so' de vocabulario: `identificacao.
   train_identification_ensemble` chama `dados_io.
   adulterant_from_mae_id` (regex especifico das letras A/M/S do
   dataset original de oleo) pra' particionar as combinacoes. Pra'
   QUALQUER mae_id que nao siga essa convencao -- inclusive mae_id REAL
   e valido de mode="imagem" nivel "high" -- a Identificacao roda sem
   erro mas devolve SEMPRE 0 combinacoes, e o proprio `model_card.md`
   ja' documenta a causa ("sem adulterante nomeavel"). Achado colateral
   de ORDEM: `resumo["Identificacao (Bloco 9b) ..."]` em `pipeline.py`
   e' escrito DEPOIS que `resumo_modelo.txt` ja' foi salvo em disco --
   nunca aparece nesse arquivo, so' no `model_card.md`.
2. **`perfil_matriz.PERFIS_TECNICA`** e' um frozenset fixo de 3 nomes
   ("bancada"/"celular"/"scanner") usado so' pra' filtrar
   `perfis_disponiveis(apenas="tecnica")` -- uma tecnica nova funciona
   normalmente via `load_profile`/`combine_profiles` (confirmado pelo
   teste de aceite acima), mas nao aparece na listagem filtrada por
   "tecnica" (cai errado em "matriz"). So' afeta descoberta em
   menu/listagem, nunca a execucao do pipeline.

5 testes novos, suite completa (1179 passed), ruff limpo. Commit, push.

---

# PROGRESSO — Passo 114 (2026-09-03)

## Passo 113 — Push

`git push origin master` -- confirmado (`d53f39b..70c686c`), sincronizado
com `origin/master`.

## Passo 114 — Hipótese D: firmeza objetiva confirma o rótulo, não é ruído

Motivada por um achado da Hipótese B (Passo 112): `unripe` tinha
`storage_days` médio MAIOR que `perfect`, contraintuitivo o bastante
pra' suspeitar de ruído de rótulo (o que exigiria retratar a Hipótese
C). Verificado: o manifest do DeepHS Fruit publica `firmness` (medição
objetiva por fruto). Testado em `tests/
test_investigacao_unripe_kiwi_vis.py::
test_hipotese_d_firmeza_objetiva_confirma_rotulo_nao_e_ruido`:
`unripe` (n=28, média=2083,9) > `perfect` (n=39, média=1398,1) >
`overripe` — ordem fisiologicamente correta. Mann-Whitney unripe vs.
perfect: p=8,87×10⁻⁸, Cohen's d=1,64 (efeito grande).

**Não é retratação, é refinamento** (registrado explicitamente em
`docs/VALIDACAO_PUBLICA.md` §7, Passo 114): o rótulo é confiável
(respaldado por medição independente), a conclusão da Hipótese C
("nenhuma hipótese resgata a classificação") permanece — mas agora com
leitura mais precisa: a diferença física é real e substancial (d=1,64
na firmeza vs. d≈0,38-0,80 no sinal espectral), a câmera VIS é que tem
sensibilidade fraca a ela especificamente. Suíte completa, ruff/mypy
limpos, commit, push.

---

# PROGRESSO — Passo 112 (2026-09-02)

## Passo 112 — Investigação do `unripe` (Kiwi/VIS): 3 hipóteses, nenhuma resolveu

Achado do Passo 104: `Kiwi/VIS` é a ÚNICA combinação do DeepHS Fruit com
as 3 classes `n≥19`, mas `unripe` sai com sensibilidade 0,00 interna E
externa mesmo assim — não é só falta de amostra. Investigação rigorosa
de 3 hipóteses, código em `tests/test_investigacao_unripe_kiwi_vis.py`
(gated por `GUARACI_DATASETS_DIR`, reproduzível):

- **A (banda química)**: restringir a 26 bandas de clorofila
  (660-680nm) + carotenoide/antocianina (500-550nm) — mesma tabela de
  `hsi_chemistry.py` — **não melhora** `unripe` (permanece 0,00/0,00).
- **B (fronteira contínua)**: PLS-R de `storage_days` (proxy real de
  maturação) com CV group-aware por dia — **Q²=-0,17** (sem
  generalização), e a média predita de `unripe` (6,36 dias) fica quase
  igual à de `perfect` (6,17 dias). Achado extra: `storage_days` em si
  é um proxy ruidoso do rótulo visual (`unripe` tem média de dias MAIOR
  que `perfect` neste dataset).
- **C (sobreposição espectral)**: Mahalanobis(unripe, perfect) em PCA
  de 2 componentes (bem condicionado) = 0,384 — pequena. Efeito
  por-banda mediano = 0,376 — fraco-moderado. **Achado metodológico
  colateral medido**: a mesma distância em PCA de 10 componentes sobe
  pra' 1,048 só por mal-condicionamento da covariância (n≈30-40/classe
  em dimensão alta) — artefato explicitamente identificado e reportado,
  não confundido com separação real.

**Conclusão honesta**: nenhuma hipótese resgatou a classificação. A
evidência mais robusta (C, medida com controle do próprio artefato de
mal-condicionamento) indica limite real de separabilidade espectral
entre `unripe`/`perfect` de Kiwi nesta câmera (Specim FX10,
397-1004nm) — não bug de implementação. Nenhuma hipótese virou opção
configurável (nenhuma mostrou melhora que justificasse). `docs/
VALIDACAO_PUBLICA.md` §7 atualizado com os números completos. 4 testes
novos, suíte completa (1170 passed), ruff/mypy limpos.

---

# PROGRESSO — Passo 111 (2026-09-02)

## Passo 111 — HSI aceita dado do próprio usuário, offline (INSTRUCAO_HSI_DADO_PROPRIO.md)

Falha de arquitetura corrigida: `run_hsi_pipeline` exigia `manifest.json`
de um dataset público específico (DeepHS Fruit) pra' qualquer uso do
modo `hsi` — o resto do GUARACI sempre aceitou pasta do próprio usuário.

**111a/b** — `hsi_io.load_envi_cube` já era genérico (não precisou
separar de nada). Nova `hsi_io.load_hsi_folder_dataset(pasta)`: lê
qualquer pasta com cubos ENVI (`.hdr`+`.bin`), convenção de subpasta-
por-classe (mesma de `dados_io.py`/`dados_imagem.py`), sem exigir
`manifest.json`. Agrupamento por amostra física reaproveita a hierarquia
de 3 níveis do Bloco 8 — extraída de `dados_imagem.py` pra' um módulo
novo, `agrupamento_pastas.py` (extensão como parâmetro), em vez de
duplicar a lógica; `dados_imagem.py` foi refatorado pra' delegar a ela,
24/24 testes existentes continuam passando sem alteração de contrato.

**111c** — `_menu_hsi` não exige mais `manifest.json`: aceita qualquer
pasta válida, o erro real vem de dentro do pipeline se não houver
cubos. Texto da tela deixa claro que dataset público é só fixture de
validação, não pré-requisito. Painel de resultado declara
explicitamente quando só há validação interna (nunca esconde atrás de
um "0" sem explicação).

**111d (contra-prova)** — `tests/test_hsi_offline_prova.py`: cubo
sintético gerado localmente (zero download), `socket.socket`
monkeypatchado pra' levantar exceção em QUALQUER tentativa de conexão
de rede, pipeline completo (leitura → quality gate → segmentação →
classificação → mapa → confiança por objeto → validação) rodando sem
tocar rede. Inclui teste da própria fixture de bloqueio.

**Decisão de arquitetura**: `run_hsi_pipeline` despacha automaticamente
(presença de `manifest.json`) entre o caminho ORIGINAL (dataset público,
validação externa por dia + explicabilidade química cruzada, inalterado)
e o caminho NOVO (`hsi_validation.run_internal_validation_group_aware`
— só validação interna, sem particao por dia que não existe num dataset
genérico; SEM explicabilidade química cruzada, porque a tabela
`ATRIBUICAO_QUIMICA_VIS_FRUTA` é conhecimento específico do dataset
público — aplicá-la a comprimento de onda arbitrário do usuário seria
alegação científica falsa, não limitação honesta).

4 commits (1 por módulo), suíte completa a cada lote (1170 passed, 9
skipped), ruff/mypy limpos, golden de contrato de API regravado (só
adições).

---

# PROGRESSO — Passo 104 execução real + fechamento (2026-09-02)

## Passo 104 — Validação comparativa executada (achado real: estouro de memória)

Primeira tentativa (`tests/test_validacao_publica_deephs_fruit.py`
contra as 1048 gravações baixadas): Avocado/NIR passou honestamente
(numeros fracos, a maioria das classes com `n<19`), mas Avocado/VIS
**crashou** com `MemoryError` -- tentativa de alocar 2,8GB para UM
unico fit de PLS-DA dentro do loop de selecao de LVs.

**Causa raiz medida** (nao presumida): resolucao de imagem varia MUITO
entre frutas. Kaki: 64x64=4096 pixels/imagem. Avocado/VIS medido
diretamente: ~286x294=~97000 pixels/imagem -- **~24x mais**. Sem teto,
o dataset por-pixel de uma fruta de alta resolucao cresce sem
controle.

**Corrigido**: `hsi_pixels.build_pixel_dataset` ganhou
`max_pixels_por_gravacao` (subamostragem SEM REPOSICAO por gravacao,
RNG semeado -- reprodutivel, pixels retidos sao REAIS, nunca
inventados). `hsi_validation.run_external_validation_by_day` repassa o
parametro. Teto usado na validacao comparativa: 2000 pixels/gravacao
(perto da escala natural do Kaki) -- MESMO teto para todas as 8
combinacoes (comparacao justa, camera de alta resolucao nao ganha mais
peso na agregacao por objeto). 4 novos testes em `tests/
test_hsi_pixels.py` (subamostra ate' o limite, nao subamostra abaixo
do limite, comportamento antigo preservado sem o parametro,
reprodutibilidade por seed).

**Resultado real, apos a correcao**: as 8 combinacoes rodaram sem erro
(869s, ~14,5min). Tabela comparativa completa em
`docs/VALIDACAO_PUBLICA.md` §7. Achado nao-obvio: `Kiwi/VIS` e' a UNICA
combinacao com as 3 classes `n>=19` (limiar do Passo 105) -- MESMO
ASSIM `unripe` sai com sensibilidade 0,00 interno E externo, indicando
que o desbalanceamento de classe NAO e' a unica causa do colapso
nessa combinacao especifica -- ha' dificuldade de separabilidade real,
reportada honestamente em vez de assumida como "so' falta n".

Suite completa + ruff + mypy limpos apos a correcao de memoria.

---

# PROGRESSO — Passo 109 (2026-09-02)

## Passo 109 — Datasets públicos adicionais de HSI (candidatos, NÃO integrados)

Busca ativa além do DeepHS Fruit, mesma disciplina do Passo 93 (formato,
licença, tamanho, cubo bruto vs. processado, ANTES de qualquer
integração). Reportado aqui, integração fica para decisão explícita
(a instrução pede isso: "reportar a lista antes de integrar qualquer
um").

| Candidato | Matriz | Formato | Licença | Tamanho | Veredito |
|---|---|---|---|---|---|
| **Olive Dataset** (Mendeley `10.17632/8xvhcsdvst.1`) | Azeitona em campo (Manzanilla e Gordal), monitoramento sazonal | ENVI (`.hdr`+`.raw`, mesmo formato já suportado por `hsi_io.load_envi_cube` sem alteração), 400-1000nm, 204 bandas, imagens 512×512 | **CC BY 4.0 — confirmado via API oficial da Mendeley** (`data_licence`), SPDX explícito | `demo_Olive_Dataset.zip` = 385MB (viável mesmo sem HTTP Range); `Olive_Dataset.zip` completo = 10GB | ✅ **Melhor candidato** — matriz nova (azeitona/oleaginosa, diferente das 5 frutas do DeepHS), licença explícita (o DeepHS Fruit não tem), formato já suportado sem novo leitor |
| Hyperspectral Pork Belly Dataset (Zenodo `17242553`) | Carne (detecção de corpo estranho) | Cubo 640×1000×184, 942-1723nm, mas o subconjunto de treino já vem como "patches" pré-processados (80 mil), não claramente cubo bruto por amostra | **CC BY 4.0 confirmado** | 18,9GB (subconjunto), 1,6TB total | ⚠️ Matriz nova (carne) e licença boa, mas formato interno do zip não confirmado como cubo bruto — precisaria inspeção direta (mesmo protocolo do Passo 93/104) antes de qualquer integração |
| Barley Hyperspectral Dataset (Univ. Copenhague, ERDA) | Cevada (grão) | NIR-HSI, espectro medido já processado (absorbância média dentro da máscara) | CC BY-**NC** 4.0 (restrição de uso comercial) | não verificado | ⚠️ Matriz de grão (interessante), mas já vem como espectro processado (não cubo bruto) + licença não-comercial — não prioritário |
| HSIFoodIngr-64 | Ingredientes alimentares diversos | não verificado em detalhe | CC BY-**NC-ND** 4.0 (sem derivados) | não verificado | ❌ Licença restritiva demais (proíbe obras derivadas) |

**Nenhum destes foi baixado ou integrado nesta rodada** -- decisão de
integrar (ou não) o Olive Dataset fica para o usuário confirmar, dado o
volume de trabalho que uma integração completa (novo leitor
específico, testes, validação) representaria em cima do que já foi
feito nos Passos 92-108.

---

# PROGRESSO — Passo 108 (2026-09-02)

## Passo 108 — `hsi_applicability.py`

Reaproveita `chemometric_stats.training_applicability_domain`/
`applicability_domain_new_samples` SEM ALTERACAO -- ja' sao genericas
o suficiente p/ aceitar pixels HSI direto (1 pixel = 1 "amostra", MESMA
granularidade que `hsi_classification.py` usa p/ treinar o PLS-DA por
pixel -- o dominio de aplicabilidade avalia exatamente o espaco que o
classificador ve). Unica coisa nova: checagem de compatibilidade de
CAMERA antes de chamar as funcoes existentes -- cameras diferentes tem
numero de bandas diferente (Kaki/VIS=224, Kaki/VIS_COR=249), entao
`pca.transform` cru levantaria um erro de shape em vez de uma decisao
interpretavel. Sensor incompativel agora devolve
`sensor_compativel=False` + motivo explicito, nunca um traceback cru.

Testado tambem o caso onde a comparacao numerica FAZ sentido: 2
combinacoes com o MESMO sensor (Specim FX10, mesmo `id`="VIS" e mesmo
numero de bandas em varias frutas, confirmado no JSON de anotacoes) --
o dominio calibrado numa fruta rejeita corretamente a maioria dos
pixels de outra fruta (quimica/reflectancia diferente, mesmo eixo
espectral).

Contra-prova OBRIGATORIA: cena sintetica deliberadamente fora do
dominio (deslocamento grande) e' rejeitada em >90% dos pixels.

---

# PROGRESSO — Passos 104-107 (2026-09-02)

## Passo 104 — `hsi_io.load_deephs_fruit_dataset` (generalizacao multi-fruta/camera)

Regex de `group_id` generalizado (o antigo exigia sufixo `_m\d+_`
especifico do Kaki -- nao bate com `avocado_day_01_20_front.hdr`).
Premissa de agrupamento por objeto fisico (frente/costas = mesma fruta)
reverificada por leitura direta do JSON de anotacoes p/ as 4 frutas
novas -- zero inconsistencias em 328 gravacoes adicionais. Cada camera
tem numero de bandas proprio (medido, nao presumido): nenhuma fruta tem
as 3 cameras (Kaki: VIS/VIS_COR; Avocado/Kiwi: VIS/NIR; Mango/Papaya:
VIS/VIS_COR) -- `load_deephs_fruit_dataset` levanta erro explicito se
mais de 1 camera sobrar num filtro (wavelengths incompativeis).

Download das 4 frutas novas (Avocado/Kiwi/Mango/Papaya, todas as
cameras disponiveis, 636 gravacoes x 2 arquivos) via HTTP Range
paralelizado (8 workers/fruta) -- script de producao
`baixar_deephs_fruit_todas.py` com pins em sidecar JSON versionado
(1272 arquivos, inline no .py seria ilegivel). Resultados da validacao
comparativa (Passo 104 propriamente dito, sensibilidade/especificidade/
precisao por fruta x camera) reportados numa secao separada assim que o
download terminar -- este bloco documenta so' a infraestrutura.

## Passo 105 — `hsi_resampling.py`

`oversample_minority_groups` duplica OBJETOS FISICOS inteiros (nunca
pixels soltos fora do grupo) das classes minoritarias -- duplicatas
mantem o MESMO `group_id` (nao um id sintetico), o que garante que
NENHUM split group-aware (nem o externo nem a selecao interna de LVs
por Wold) separa uma copia do original. Iguala o PESO em pixels, nunca
fabrica um objeto fisico novo (estrutural).

`class_evaluability_report` reusa `conformal.n_minimum_for_alpha`
(=19 p/ alpha=0.05) -- MESMO limiar ja' padronizado no assistente e no
gate DD-SIMCA/conjunto aberto, nao um limiar novo so' p/ HSI. Contra-
prova Hypothesis obrigatoria: reamostragem nunca separa pixels do
mesmo objeto entre treino/validacao, generalizando a propriedade do
Passo 97.

## Passo 106 — `hsi_identification.py`

Conjunto aberto adaptado de `identificacao.py` p/ o nivel de objeto do
HSI. Diferenca estrutural real (nao no fluxo tabular): cada combinacao
fruta x camera tem seu proprio numero de bandas -- 1 PCA por
combinacao, nao 1 global compartilhado.

Granularidade de calibracao MEDIDA antes de decidir (Passo 106 exige
isso explicitamente): objetos fisicos distintos por fruta (28-88) e por
fruta x camera (24-87), as duas >= n_minimum_for_alpha(0.05)=19 em
TODAS as combinacoes reais do dataset -- escolhida a mais fina (fruta x
camera) por tambem ser calibravel e evitar misturar variancia espectral
de sensores diferentes. `n_grupos<=1` registrado explicitamente como
NOT_VALIDATED_N1 (nao omitido em silencio -- achado durante os testes,
corrigido antes do commit). Contra-prova obrigatoria: tipo espectral
nao presente no treino retorna "desconhecido" (nao aceito por nenhuma
entrada calibrada).

## Passo 107 — `hsi_uncertainty.py`

Heterogeneidade de pixel (ja' calculada no Passo 98) vira relatorio
FORMAL (`enrich_object_results`, nota de confianca textual + numeros
crus) -- wireado em `hsi_pipeline.run_hsi_pipeline` (chave
`confianca_por_objeto`) e na tela `[X]` da CLI (objetos de baixa
concordancia listados explicitamente, nunca escondidos).

**DECISAO REGISTRADA (exigida pela instrucao antes de implementar):**
NAO combinar alpha por Bonferroni entre etapas do fluxo HSI, ao
contrario do fluxo tabular Detectar->Identificar->Quantificar. O fluxo
HSI hoje so' tem UMA etapa com alpha formalmente calibrado
(Identificacao, Passo 106) -- quality gate (Passo 95) e' limiar
deterministico, classificacao+agregacao (Passo 98) e' decisao pontual
+ heterogeneidade descritiva, nenhuma das duas tem alpha proprio.
Bonferroni de 1 alpha so' e' o proprio alpha -- nada a combinar. Se o
HSI ganhar uma etapa de quantificacao formal com intervalo de predicao
proprio no futuro, a combinacao passaria a fazer sentido, espelhando o
fluxo tabular -- nao antes disso. Razao completa documentada no
docstring de `hsi_uncertainty.py`.

---

# PROGRESSO — Passo 103 (2026-09-02)

## Passo 103 — Texto/UI da tela HSI corrigidos

Dois bugs reais reportados pelo usuario ao revisar a tela HSI:

1. **Frase solta "Prototipo 'minimo viavel'"** -- substituida por
   `_AVISO_MATURIDADE_HSI_PT`/`_EN` (fonte unica, mesmo padrao de
   `_AVISO_PROTOTIPO_TITULO`/`_CORPO` em `reports.py`), descrevendo a
   limitacao REAL e especifica ("validado em 1 fruta (Kaki) e 1 camera
   (VIS)... overripe n=12, unripe n=2"), nao um rotulo generico.
   **Decisao registrada**: NAO usar o carimbo formal "PROTOTYPE OUTPUT"
   (`reports.py`) porque o criterio objetivo daquele carimbo (ausencia
   de garantia de agrupamento anti-vazamento) NAO se aplica ao HSI -- o
   HSI TEM garantia real (`group_id` por objeto fisico, Passo 97,
   validada por Hypothesis). Sao limitacoes de natureza diferente;
   reusar o carimbo verbatim seria factualmente impreciso.
2. **Cabecalho fixo "Tecnica: FT-NIR"** herdado do template generico --
   `_print_header`/`_print_status` agora usam `_rotulo_tecnica_efetivo
   (cfg)`, que mostra "HSI" quando `cfg.mode=="hsi"`, "Colorimetria
   digital" quando `cfg.mode=="imagem"` (MESMO bug, corrigido pela MESMA
   fonte -- achado ao varrer as demais telas, nao so' a de HSI), e
   preserva o comportamento antigo (tecnica escolhida em [8]) para
   dx/csv/sintetico. `cfg.mode="hsi"` agora e' setado ANTES do primeiro
   `_print_header`, nao so' apos validar a pasta -- senao a tela ainda
   mostraria o rotulo errado no primeiro render.

**Contra-prova de teste**: `tests/test_menu_hsi.py` renderiza a tela de
verdade (cabecalho + intro, PT e EN) e confere que "FT-NIR"/"prototipo"/
"minimo viavel" NAO aparecem e que "Tecnica: HSI"/"Technique: HSI"
aparecem -- mais 2 testes unitarios de `_rotulo_tecnica_efetivo` (modo
imagem corrigido, modo dx preservado -- contra-prova de nao-regressao).

**Achado no processo, corrigido antes do commit**: o helper de teste
`_renderizar_tela_hsi` setava `_STATE["lang"]="EN"` sem restaurar no
`finally` -- vazava para os testes seguintes na mesma sessao pytest e
quebrou 2 testes de `test_selecao_amostras.py` (coluna `"conjunto"`
virava `"set"` em ingles). Mesma classe de bug ja documentada no
helper `_render` de `test_guaraci_cli.py` (que EU deveria ter copiado
completo, nao so' a parte de `console.file`) -- corrigido, suite
completa voltou a 1113 passando.

---

# PROGRESSO — Passos 92-95 (2026-09-01)

## Passo 92 — Verificação da literatura citada em INSTRUCAO_HSI_MINIMO_VIAVEL.md

3 referências citadas na instrução, verificadas ANTES de qualquer uso em
código/documentação (regra "não citar se não confirmar"):

- `S0031320325004960` ("Revisão de classificação HSI entre domínios") —
  **confirmado**: "Cross-domain hyperspectral image classification",
  *Pattern Recognition* 168, dez/2025. Tema bate exatamente.
- `S0169743926002212` ("Revisão sobre transferência de calibração e
  incerteza") — **confirmado** via WebSearch (snippet retornou o PII
  exato): "Chemometric and machine-learning strategies for calibration
  transfer", *Chemometrics and Intelligent Laboratory Systems*, 2026.
- `S2772375526007070` ("Framework de padronização e reprodutibilidade em
  HSI") — **NÃO confirmado**. O ISSN implícito (2772-3755) corresponde a
  um periódico real (*Smart Agricultural Technology*, Elsevier,
  tematicamente compatível), mas o artigo especifico nunca apareceu em
  nenhuma busca (WebSearch por PII exato, por título aproximado, por
  termos-chave da descrição). Acesso direto ao ScienceDirect bloqueado
  (WebFetch: 403; Browser pane: CAPTCHA Cloudflare) — sem via alternativa
  de confirmação disponível nesta sessão. **Não citado** em nenhum lugar.

## Passo 93 — Busca de dataset público de HSI (prioridade sobre implementação)

Candidato escolhido: **DeepHS Fruit** (Varga, Makowski & Zell, IJCNN
2021, arXiv:2104.09808, github.com/cogsys-tuebingen/deephs_fruit) —
subconjunto Kaki (caqui) / câmera VIS (Specim FX10, 224 bandas,
397,66-1003,81 nm), 56 gravações / 38 frutas físicas, rótulo real
`ripeness_state` (unripe/perfect/overripe).

Candidato alternativo considerado (Mendeley `gjwx64sgkp`, bagas de uva,
CC BY 4.0) — descartado: não foi possível confirmar se distribui cubo
BRUTO ou só espectro já extraído (o segundo não serve para
segmentação/mapa espacial, Passos 96-99).

Formato confirmado por leitura DIRETA (HTTP Range requests no
`Kaki.zip` de 2,2G, sem baixar o arquivo inteiro — leitura só do
directorio central + membros necessários): par ENVI `.hdr` (texto) +
`.bin` (float32, BIP, sem header embutido). Comprimentos de onda vêm à
parte, no JSON de anotações oficial do dataset (`cameras[].wavelengths`
por câmera).

**Agrupamento por objeto físico** (crítico para o Passo 97): confirmado
por leitura direta do JSON de anotações que "frente"/"costas" da MESMA
fruta compartilham `storage_days` e `ripeness_state` dentro do mesmo
dia — group_id = `f"{day}_{numero_da_fruta}"`.

## Passo 94 — `src/guaraci/hsi_io.py`

Leitor ENVI genérico (`load_envi_cube`, aceita bip/bil/bsq, qualquer
`data type` ENVI suportado, `wavelengths` externo quando o `.hdr` não
traz) + leitor específico do subconjunto DeepHS/Kaki
(`load_deephs_kaki_dataset`). 12 testes (11 sintéticos + 1 contra o
dataset real, `GUARACI_DATASETS_DIR`-gated). Commit `5f3ec85`.

## Passo 95 — `src/guaraci/hsi_quality.py`

Quality gate fail-fast (saturação/faixa, SNR via Immerkaer 1996, fração
de pixels válidos) — rejeita com motivo único e específico, nunca
processa em silêncio. Contra-prova obrigatória da instrução (cubo
saturado e cubo de SNR baixo, ambos rejeitados) — 8 testes. Calibração
radiométrica por referência branco/preto **não implementada** nesta
rodada: o dataset escolhido já vem calibrado e não há cubo de referência
bruto disponível para testar essa etapa de verdade — documentado, não
escondido. Commit `f3de9ca`.

## Dataset baixado e infraestrutura de reprodução

`scripts/download_datasets/baixar_deephs_kaki.py` — usa HTTP Range para
extrair só os 112 arquivos (56 gravações × .hdr+.bin) do Kaki.zip de
2,2G sem baixar o arquivo inteiro, cada um com SHA256+tamanho pinado
(verificado ANTES de gravar, mesma regra de
`baixar_mendeley_oleos.py`). Testado de verdade: cache-hit (pins batem
com os 112 arquivos já extraídos) e extração fresca (pasta vazia, exercita
o caminho de rede real) — ambos confirmados por execução direta, não
suposto.

Licença do DeepHS Fruit: **não declarada formalmente** (sem SPDX no
repo/README, API do GitHub devolve `license: None`) — ver retratação em
`docs/VALIDACAO_PUBLICA.md` §4 (uma busca inicial via WebSearch sugeriu
CC BY-SA 4.0; não confirmado por verificação direta, corrigido antes de
entrar em qualquer citação).

## Passo 96 — `src/guaraci/hsi_segmentation.py`

PCA (PC1) + Otsu (implementado do zero -- scikit-image e' dependencia
OPCIONAL do projeto). Distincao documentada do PCA de dominio de
aplicabilidade (`chemometric_stats.applicability_domain`) -- uso
espacial por pixel de UMA cena, nao distancia a um modelo pre-treinado.
Sem mascara de referencia no dataset -- validado por INSPECAO VISUAL
DOCUMENTADA (`resultados_hsi_segmentacao/kaki_segmentacao_amostra.png`,
gitignorado). Cena sintetica com objeto conhecido: IoU>0.8. Commit
`7de3727`.

**RETRATACAO (2026-09-01, mesma rodada):** a versao commitada em
`7de3727` assumia "objeto = MINORIA de pixels da cena" -- correto para
a cena sintetica do teste, mas ERRADO no dataset real: a fruta ocupa
~59% do quadro (maioria), entao a mascara marcava os CANTOS (fundo)
como "objeto" -- inversao silenciosa. O relatorio desta auditoria
descreveu a mascara commitada como "confirma visualmente" sem
reconferir a propria imagem salva com atencao -- so' pego ao usa-la no
Passo 99 (mapa de classificacao) e notar que a fruta aparecia em cinza
(fora da ROI) em vez de colorida. Corrigido: fundo agora e' inferido
pela BORDA da imagem (pixels mais externos), nao pela fracao de area --
cobre objeto minoria OU maioria. Novo teste de propriedade (objeto
majoritario, cena tipo "moldura fina") adicionado. Numeros do Passo 98
abaixo foram RECALCULADOS com a mascara corrigida (a versao original
tinha treinado/testado sobre pixels de FUNDO, nao da fruta).

## Passo 97 — `src/guaraci/hsi_pixels.py`

Extracao de espectros de pixel da ROI + `group_id` de objeto fisico
replicado por pixel (frente/costas da MESMA fruta compartilham
group_id, confirmado por leitura direta do JSON de anotacoes). Contra-
prova OBRIGATORIA (Hypothesis, numero de objetos e pixels/objeto
aleatorios): `StableStratifiedGroupKFold` (o splitter group-aware JA
padronizado no projeto) nunca separa pixels do mesmo objeto entre
treino/validacao, em nenhum fold. Commit `1c179f2`.

## Passo 98 — `src/guaraci/hsi_classification.py`

PLS-DA por pixel (reaproveita `avaliacao_modelos.PLSDAClassifier`, nao
reimplementado), split group-aware, numero de LVs por parsimonia de
Wold (mesmo criterio de `pipeline.py`, generalizado p/ classificacao
via 1-balanced_accuracy). Agregacao por objeto: classe majoritaria +
heterogeneidade (fracao de pixels em desacordo).

**Medido contra o dataset real** (8 objetos de teste de 38 totais,
split group-aware, seed=0, n_components=5 selecionado por Wold, JA' com
a mascara de segmentacao corrigida -- ver retratacao no Passo 96 acima):
**5/8 objetos corretos** -- o modelo ainda tende a "perfect" (classe
majoritaria, 42/56 gravacoes; overripe=12, unripe=2) nos erros restantes.
Desbalanceamento severo, nao corrigido nesta rodada (fora do escopo do
"minimo viavel" -- rebalanceamento/reponderacao seria proximo passo
natural, nao feito aqui p/ nao inflar o resultado por ajuste ad-hoc).
Reportado honestamente, mesmo padrao ja' registrado p/ o Mendeley
(`docs/VALIDACAO_PUBLICA.md` §2: bal.acc 0,35 CV). Confirma que o
pipeline mecanico (segmentacao -> extracao -> classificacao ->
agregacao) funciona ponta-a-ponta sobre dado real -- nao que o
desempenho e' bom.

## Passo 99 — `src/guaraci/hsi_figures.py`

Mapa de classificacao espacial por pixel, reaproveitando `figuras.save`
(pasta/formato/carimbo de prototipo ja' padronizados) e `paleta_cores.
color` (paleta da mascote) -- sem paleta nova. Testado contra o dataset
real apos a correcao da segmentacao (Passo 96). Commit `4150a27`.

## Passo 100 — `src/guaraci/hsi_chemistry.py`

Cruzamento VIP (reaproveita `chemometric_stats.vip_scores` ja'
existente, nao reimplementado) x tabela de atribuicao quimica --
tabela ESPECIFICA da matriz/faixa deste dataset (VIS 397-1004nm,
fruta), 3 entradas citando literatura real (Merzlyak, Solovchenko &
Gitelson 2003 p/ clorofila-a e carotenoides/antocianinas; Osborne,
Fearn & Hindle 1993 p/ agua). Nunca afirma causalidade -- so' "consistente
com" ou a frase padrao de "sem atribuicao obvia".

**Medido contra o dataset real** (top-5 bandas de maior VIP, PLS-DA
5 componentes sobre TODOS os pixels de ROI): 4 das 5 bandas caem entre
540-550nm, dentro da faixa tabelada de carotenoides/antocianinas --
consistente com a fisiologia real do amadurecimento (degradacao de
clorofila + acumulo de carotenoides), achado nao forcado (a tabela foi
escrita ANTES de rodar o VIP real, ver commit).

## Passo 101 — `src/guaraci/hsi_validation.py`

Particao nativa de origem = DIA de medicao (achado por leitura direta
do JSON de anotacoes: cada dia e' uma sessao/lote separado,
`storage_days` cresce por dia). Teste externo = dias `day_8_m3` +
`day_9_m3` (nunca vistos no treino); teste interno = objetos held-out
dos demais 6 dias. Sensibilidade/especificidade/precisao reportadas
SEPARADAS por classe e por interno/externo (reaproveita
`figuras.specificity_by_class`, ja' existente) -- nunca uma media
unica.

**Medido contra o dataset real** (interno n=6, externo n=12):

| classe | sens(int) | sens(ext) | espec(int) | espec(ext) | prec(int) | prec(ext) |
|---|---|---|---|---|---|---|
| overripe | 0,00 | 0,50 | 1,00 | 1,00 | 0,00 | 1,00 |
| perfect | 1,00 | 1,00 | 0,00 | 0,50 | 0,83 | 0,80 |
| unripe | 0,00 | 0,00 | 1,00 | 1,00 | 0,00 | 0,00 |

Numeros ruidosos e as vezes CONTRA-INTUITIVOS (sensibilidade de
overripe MAIOR no externo que no interno) -- efeito esperado de n muito
pequeno por classe/particao (6-12 objetos), reportado sem suavizar.
`unripe` tem sensibilidade/precisao zero nas duas particoes -- o
dataset so' tem 2 gravacoes dessa classe no total (ver Passo 93),
insuficiente para qualquer split aprender o padrao. Nao e' escondido:
e' exatamente o tipo de "queda/limitacao real" que a instrucao pede
para documentar, nao maquiar.

## Passo 102 — Integracao ao menu/CLI

`src/guaraci/hsi_pipeline.py` (novo): orquestra leitura -> quality gate
-> segmentacao -> classificacao por pixel -> mapa espacial ->
explicabilidade -> validacao externa numa unica chamada. Modo `hsi`
adicionado a `Config.mode`/`_CONFIG_SPEC` -- DISTINTO do modo `imagem`
(docstring do modulo explica a diferenca: HSI e' por pixel, `imagem` e'
por foto inteira; nunca confundidos no menu/docs, requisito explicito
da instrucao).

Acessivel pela tecla **[X]** do menu principal da CLI (`_menu_hsi` em
guaraci.py) -- testado de ponta a ponta pelo caminho REAL do usuario
(`tests/test_menu_hsi.py::test_menu_hsi_roda_pipeline_completo_via_cli`,
digita o caminho da pasta na tela, nao chama `run_hsi_pipeline`
diretamente). Novo campo `hsi_dataset_folder` (`hsi_pasta_dataset` no
YAML) alcancavel nas 2 interfaces -- as redes de seguranca sistemicas
do projeto (`test_todo_campo_do_spec_e_alcancavel_por_algum_menu`,
`test_todo_campo_do_config_spec_aparece_no_app`/`no_menu_cli`) pegaram
a lacuna automaticamente antes do commit, exatamente a classe de bug
que esses testes existem para prevenir.

`hsi_pasta_dataset` adicionado a `ALIASES_COM_CAMINHO_PROPRIO`
(cli_assistente.py) -- tem caminho de edicao proprio e melhor
(`_menu_hsi` valida `manifest.json` antes de aceitar) que o editor
generico de campo.

7 modulos HSI puros (`hsi_io/quality/segmentation/pixels/
classification/chemistry/validation.py`) adicionados ao gate de mypy
do CI (`.github/workflows/test.yml`) -- 2 erros reais de tipo achados e
corrigidos em `hsi_validation.py` (parametro reatribuido com tipo
incompativel; retorno `Dict[str, object]` de `fit_predict_pixel_plsda`
usado sem `cast` explicito).

README.md/README.pt-br.md/docs/MANUAL.md atualizados: modo `hsi`
listado ao lado de NIR/MIR tabular e do modo `imagem`, com a mesma
ressalva de maturidade ja' usada para `imagem` (protótipo, nao usar
para resultado publicavel sem validacao adicional).

Suite completa (1108 testes, incluindo o teste de propriedade
obrigatorio do Passo 97) + ruff + mypy (7 modulos novos) limpos antes
do commit.

**Fatia "minimo viavel" da INSTRUCAO_HSI_MINIMO_VIAVEL.md concluida
(Passos 92-102).** Fora de escopo por decisao consciente (registrado
tambem no proprio arquivo da instrucao): deep learning, spectral
unmixing, domain adaptation/few-shot learning, fusao multimodal, sensor
multiespectral embarcado, arquitetura "detector de matriz + especialista
por matriz".

---

# PROGRESSO — Passos 84-87 (2026-08-27)

> Log de progresso do checkout ativo (OneDrive). Convenção: um bloco por
> Passo, evidência ou silêncio (nenhuma prosa de "corrigido"/"confirmado"
> sem comando/teste que sustente a afirmação).

## Passo 84 — Extensão do bug de `matrix_profile` (Passo 83)

**Pergunta:** o bug corrigido no Passo 83 (`matrix_profile` resetava para
`"generico"` no ciclo salvar/carregar de `config.yaml`, porque o campo
nunca esteve em `_CONFIG_SPEC`) afetou alguma validação pública já
reportada como concluída (Corn, Mendeley)?

**Resposta: NÃO.** Evidência:
- `tests/test_validacao_publica.py` e `tests/test_validacao_publica_mendeley.py`
  constroem `pq.Config(matrix_profile=...)` diretamente em memória e chamam
  `pq.executar(cfg)` na sequência — `grep -n "save_config\|load_config"`
  nos dois arquivos retorna vazio.
- `save_config`/`load_config` só são acionados pelo menu interativo de
  terminal (`_menu_interativo`, `pipeline.py:2971-3027`) e pelo fluxo
  `[S]`/`[L]` da CLI — nenhum dos dois entra no caminho das validações.
- `.github/workflows/test.yml` (jobs `validacao-publica` e
  `validacao-publica-mendeley`) roda `pytest tests/test_validacao_publica*.py`
  direto, sem etapa de `config.yaml` no meio.

Nenhuma revalidação necessária; nenhum número publicado mudou.

## Passo 85 — Hypothesis (testes de propriedade)

- `hypothesis>=6.100,<7.0` adicionado como dependência de desenvolvimento
  (`pyproject.toml` extra `[dev]`; NUNCA em `requirements.txt`, que é o
  manifesto de deploy).
- `tests/test_propriedades_hypothesis.py`: 3 propriedades + 3 contra-provas
  documentadas — roundtrip de `config.yaml` (generaliza o Passo 83 para
  TODOS os campos de `_CONFIG_SPEC`), quantificação cega nunca depende do
  rótulo verdadeiro, split group-aware nunca separa réplica física (cobre
  os 3 splitters do Passo 87 desde que existiram).
- **Achado real, ANTES de qualquer commit**: o próprio teste de roundtrip
  achou 2 bugs de silêncio em `_fmt_yaml` (`config_io.py`) —
  (1) string `str`/`str_opcional` com forma YAML-ambígua ('010'→int 8
  octal, '1.50'→perde zero, '0x1A'→26) saía sem aspas; (2) item de lista
  contendo `?` quebrava ou virava mapa em silêncio dentro de `[a, b]`.
  Corrigido usando `yaml.safe_load` como oráculo + `?` no conjunto de
  caracteres que força aspas em item de lista. Confirmado por
  reversão manual: sem a correção, os `@example` fixados no teste falham
  de forma determinística (não dependiam de sorte da busca aleatória —
  medido: 80 exemplos aleatórios sozinhos NÃO pegavam o bug de forma
  confiável, por isso os `@example` foram fixados).
- Commit: `test: Hypothesis (testes de propriedade) + achado real de
  config.yaml (Passo 85)`.

## Passo 86 — Transferência de calibração entre instrumentos

- `src/guaraci/transferencia_calibracao.py` (novo módulo, `__all__` desde
  o início): Direct Standardization (DS) e Piecewise Direct
  Standardization (PDS) — Wang, Veltkamp & Kowalski (1991),
  *Multivariate instrument standardization*, DOI `10.1021/ac00023a016`
  (verificado no Crossref).
- `tests/test_transferencia_calibracao.py`: contrato de forma/erro +
  redução de erro em dados sintéticos + contra-prova (mestre/escravo SEM
  relação real → PDS não melhora).
- **Validado contra o Corn real** (3 espectrômetros, mesmas 80 amostras):
  RMSEP proteína m5→mp5 sem transferência ≈ 0,51; com PDS (15 amostras de
  transferência, janela=5, alpha=0,001) ≈ 0,16 — quase o nível do m5
  sozinho (≈ 0,148). Hiperparâmetros medidos empiricamente contra o
  dataset, não adivinhados (ver `check_corn_transfer.py` no scratchpad da
  sessão para a varredura). DS não reduziu o erro de forma relevante neste
  par de instrumentos — achado honesto, não escondido.
- Reexportado em `pipeline.py`; contrato de fachada
  (`tests/test_fachada_reexport.py`) e contrato de API pública
  (`tests/golden/contrato_api_publica.json`) atualizados.
- Limitações documentadas em `docs/MANUAL.md` §2.2b (nº mínimo de amostras
  de transferência, sensibilidade de `alpha`/`janela`, pressupõe
  deslocamento linear/local).

## Passo 87 — Seleção de amostras (Kennard-Stone, Duplex, SPXY)

- Kennard-Stone (`kennard_stone`/`kennard_stone_split`/
  `kennard_stone_split_group_aware`) já existia — reaproveitado, não
  reimplementado.
- Completado com `duplex_split`/`duplex_split_group_aware` (Snee, 1977,
  DOI `10.1080/00401706.1977.10489581`) e `spxy_split`/
  `spxy_split_group_aware` (Galvão et al., 2005, DOI
  `10.1016/j.talanta.2005.03.025`) em `src/guaraci/dados_io.py`, mesma
  disciplina group-aware do Kennard-Stone (nunca separa réplica física
  entre calibração/validação — garantido por teste de propriedade
  Hypothesis parametrizado nos 3 splitters).
- `tests/test_selecao_amostras.py`: contrato de partição, proporção,
  casos degenerados (n=0/1/2), group-aware, e uma contra-prova específica
  do motivo de existir do SPXY (KS puro pode deixar de fora o extremo do
  TEOR se ele não for também extremo espectral; SPXY não deixa — caso
  sintético reproduz isso).
- Integrado à CLI: menu principal, tecla `[K]` *Seleção de Amostras*
  (Bloco 10, ao lado do planejamento de coleta `[J]`) — lê um CSV de
  espectros, roda o método escolhido, grava cópia com coluna
  `calibracao`/`validacao`. 3 testes CLI ponta-a-ponta (Kennard-Stone,
  SPXY com coluna alvo, contra-prova de arquivo ausente).
- Documentado em `docs/MANUAL.md` §2.2c.

## Estado da suíte (Passos 84-87)

Commit do Passo 85: 987 testes (incl. Corn real) + ruff limpos.
Passos 86+87: 1008 testes + ruff limpos.

---

# Bloco 13d + varredura geral (2026-08-27, mesma sessão)

## Frente 1 — Bloco 13d: linearidade e robustez formais

- `src/guaraci/linearity.py` (novo, `__all__` desde o commit inicial):
  `lack_of_fit_test` — teste F de falta de ajuste clássico (Draper &
  Smith, cap. 2.6), nível da curva = grupo de réplica física (`mae_id`,
  L2). Contra-prova: curvatura sintética deliberada produz F
  significativo, e F cresce com a magnitude da curvatura.
- `src/guaraci/robustness.py` (novo, `__all__` desde o commit inicial):
  perturbação controlada (pré-processamento, ruído gaussiano, deriva de
  linha de base) + protocolo que reporta variação como INTERVALO, nunca
  binário (R2). Cobre PLS-R e PLS-DA (R3). Contra-prova: perturbação
  maior produz variação maior.
- Integrado ao dossiê via `append_linearity_robustness_model_card`
  (mesmo mecanismo append-only de regressão/identificação/pureza).
- **Validado contra Corn E Mendeley reais**: nos dois, sem `mae_id`
  (réplica física), o teste de linearidade reporta corretamente "não
  computável" — achado honesto (L2), não um bug. Protocolo de robustez
  roda e reporta intervalo em ambos (RMSEP no Corn, bal.acc no
  Mendeley).
- **Decisão de escopo NÃO tomada sozinha (reportada)**: os dois módulos
  NÃO estão fiados automaticamente em `executar()` via novo campo de
  `Config` — isso mudaria o comportamento/custo padrão de toda execução
  do pipeline (robustez roda múltiplos refits) e é uma decisão de
  produto, não um ajuste mecânico "dentro do que já é interno". As
  funções existem, são públicas, testadas e validadas contra dado real;
  faltaria só a decisão de fiação automática + nome/default do flag de
  `Config`, se for para acontecer.
- `mypy`: os 3 módulos novos desta sessão (`linearity.py`,
  `robustness.py`, `transferencia_calibracao.py` do Passo 86) passam
  limpos e cabem no critério já documentado (sem I/O/UI/estado global)
  — adicionados ao gate da CI (ver Frente 3a).
- Commit: `feat: linearidade formal (lack-of-fit) + protocolo de
  robustez (Bloco 13d, Frente 1)`.

## Frente 2 — Infraestrutura de Hypothesis fortalecida

- Auditoria dos 3 grupos de propriedade existentes: só o roundtrip de
  config tinha `@example`. Adicionado `@example` para quantificação
  cega (reproduz o cenário de envenenamento do teste manual original) e
  3 `@example` defensivos de fronteira para o split group-aware (limiar
  `n_grupos=4` onde o colapso por grupo liga) — documentado
  explicitamente que não há bug histórico conhecido para essa
  propriedade (ao contrário do roundtrip), para não sugerir cobertura
  que não existe.
- Profile diferenciado (`conftest.py`): `dev` (50 exemplos, local) vs
  `ci` (300 exemplos, auto-selecionado via `CI=true`, já setado pelo
  GitHub Actions — nenhuma mudança em `test.yml` necessária).
  `max_examples=` por teste removido em favor do profile ativo.
- `.hypothesis/` e `.pytest_cache/` no `.gitignore` (cache local, não
  fonte de verdade).
- `CONTRIBUTING.md`: nova seção documentando a lição medida no Passo 85
  e a convenção resultante.
- Commit: `test: fortalece infraestrutura de Hypothesis -- profile
  CI/local + @example auditados (Bloco 13d, Frente 2)`.

## Frente 3 — Varredura geral

**3a — type-checking.** Medido por comando direto (`mypy` local): os 3
módulos novos desta sessão (linearidade, robustez, transferência de
calibração) passam limpos e cabem no critério de escopo já documentado
(pyproject.toml) — adicionados ao gate da CI, custo zero (nenhum erro
para corrigir). `dados_io.py`/`guaraci.py`/`config_io.py` continuam
FORA do gate por critério — têm I/O/UI, fora do escopo por desenho, não
por descuido.

**3b — segurança.** Nenhum `eval`/`exec`/desserialização insegura nova
encontrado. `subprocess`/`os.system` existentes são todos strings
literais ou listas de argumento (sem `shell=True` com entrada do
usuário), já auditados em 2026-08-07. O único script de download
(`baixar_mendeley_oleos.py`) já segue a disciplina correta (HTTPS,
tamanho+SHA-256 pinados, verificados ANTES de gravar em disco) —
documentado como convenção obrigatória em `docs/VALIDACAO_PUBLICA.md`
§6 para qualquer script futuro. `pip-audit` contra o ambiente inteiro
(incl. `hypothesis`): **nenhuma vulnerabilidade conhecida**.

**3c — documentação de alto nível.** `README.md`/`README.pt-br.md`:
lista de funcionalidades atualizada (mode cego, planejamento
experimental, auditoria de delineamento, linearidade/robustez,
transferência de calibração, seleção de amostras). `paper/paper.md`:
contagem de testes stale (779) corrigida para "1000+"; parágrafo novo
cobrindo as funcionalidades pós-reposicionamento, com 4 referências
novas em `paper.bib` (Wang-Veltkamp-Kowalski 1991, Snee 1977, Galvão et
al. 2005, Draper & Smith 1998), DOIs verificados no Crossref.
`CITATION.cff`: verificado — versão/data consistentes com
`pyproject.toml`, nenhuma mudança necessária (bumping de data sem bump
de versão seria enganoso).

**3d — comparativo com concorrentes.** Verificado por busca (não
presumido): Kennard-Stone **já é** funcionalidade padrão do Unscrambler
(confirmado); PDS/transferência de calibração é método clássico,
razoável supor presente em suites comerciais maduras mesmo sem
confirmação direta — por isso **NÃO adicionados** à tabela comparativa
como diferenciais (seria uma alegação de exclusividade sem lastro). Não
encontrada evidência de que concorrentes ofereçam planejamento
experimental automatizado, auditoria de confundimento ou identificação
de conjunto aberto calibrada por predição conforme — mas ausência de
evidência não é prova; por isso essas funcionalidades foram adicionadas
como itens de lista (fato, sem comparação) na seção *Features*, não
como linha nova na tabela comparativa (que faz uma alegação
competitiva). Tabela comparativa do README mantida como estava.

## Estado da suíte (Bloco 13d + varredura)

1042 testes (incl. Corn e Mendeley reais) + ruff limpos após Frente 1 e
Frente 2. Frente 3 é só documentação (README/paper/CONTRIBUTING/
VALIDACAO_PUBLICA) — suíte completa reconfirmada mesmo assim, mesma
disciplina.

---

# Três pendências técnicas remanescentes (2026-08-27, mesma sessão)

## Passo 89 — Contrato de colunas de saída: FECHADO (implementação)

Dívida registrada em 2026-08-26 (Passo 77) e documentada em
`docs/COMPATIBILITY.md` desde então. Levantamento por comando direto
(`grep -rn "to_csv|to_excel|Workbook(" src/guaraci/`) de todos os pontos
de geração tabular: `avaliacao_modelos.py` (3), `guaraci.py` (2, incl.
o menu `[K]` do Passo 87), `pipeline.py` (3, inline dentro de
`executar()`), `resultados_io.py` (1), `selecao_variaveis.py` (4),
`plano_coleta.py` (Excel via openpyxl). `reports.py::generate_excel_report`
e `auditoria_delineamento.py`/`sentinela_deriva.py`/`linearity.py`/
`robustness.py` verificados como FORA de escopo (por leitura direta, não
suposição) — o primeiro copia colunas verbatim de CSVs já cobertos; os
demais não produzem saída tabular própria.

`tests/test_contrato_saida_tabular.py` (novo): snapshot golden
(`tests/golden/contrato_saida_tabular.json`) gerado por EXECUÇÃO REAL
contra dado sintético (nunca lista digitada à mão) — mesmo mecanismo de
`test_contrato_api_publica.py`. Cobre: `save_identifiers`,
`sanitizar_metadados`, `benchmark_classifiers`, `monte_carlo_cv`,
`benchmark_regression_by_species`, `etapa4_selecao_variaveis` (4 CSVs:
ipls/spa/ag/tabela-final), `plano_coleta.export_excel` (2 abas), o menu
`[K]` de seleção de amostras, `predict_samples`, e uma execução completa
de `executar()` (para `teste_martens.csv`/`comparacao_pipelines.csv`,
construídos INLINE no orquestrador, não atrás de função pública própria
— só rodar de verdade protege esses dois).

Contra-prova: monkeypatch de `save_identifiers` renomeando
`classe_predita` → `classe_pred` NUM CSV REAL (não só num dict de
teste) confirma que `_diferencas` (o mesmo detector do teste principal)
acusa a mudança.

`docs/COMPATIBILITY.md` atualizado: seção "Dívida conhecida" virou
"Dívida fechada (2026-08-27, Passo 89)".

## Passo 90 — Escopo do mypy: DECISÃO EXPLÍCITA = expandir (medido, implementado)

Medido por comando direto (`mypy` por arquivo, os 38 módulos de
`src/guaraci/`): **10 já no gate** (todos limpos), **17 fora do gate com
0 erros**, **11 fora do gate com erros** (1 a 13 cada).

Critério aplicado (o mesmo já documentado em `pyproject.toml`: sem
I/O pesado, sem UI, sem estado global) — **não** "0 erros = incluir
automaticamente": `figuras.py`, `app_logic.py`, `cli_assistente.py`,
`cli_logic.py`, `guaraci_theme.py`, `log.py`, `spectra_preview.py`
(importa `streamlit`, verificado por leitura) ficam FORA por serem
UI/renderização/orquestração — mesmo com 0 erros hoje, incluí-los
arriscaria ruído futuro conforme a integração de Streamlit/Rich
aprofunda, sem pegar bug de cálculo (mesma razão já documentada para
excluir `guaraci.py`/`pipeline.py`, que TÊM 11 e 10 erros
respectivamente e continuam explicitamente fora, decisão pré-existente
não reaberta aqui). `dados_imagem.py` fica fora por razão técnica real
(não "nunca foi feito"): importa `tifffile` via scikit-image, que usa
sintaxe Python 3.12 incompatível com `python_version=3.10` do mypy —
erro de SINTAXE de terceiro que interrompe a checagem inteira, não
corrigível no nosso código. `avaliacao_modelos.py`/`dados_io.py`/
`resultados_io.py` ficam fora: I/O real (CSV) é parte central do que
fazem, não incidental, e tinham 4-13 erros cada.

**Decisão: EXPANDIR.** Adicionados 14 módulos (10 já limpos +
4 corrigidos): `conformal.py`, `config.py`, `plano_amostral.py`,
`selecao_variaveis.py`, `sentinela_deriva.py`, `predicao.py`,
`io_registry.py`, `config_io.py`, `perfil_matriz.py`, `plano_coleta.py`,
`paleta_cores.py` (2 `# type: ignore` não utilizados removidos),
`auditoria_delineamento.py` (2 `int(object)` corrigidos com `cast`),
`identificacao.py` (1 tipo de chave de dict corrigido — `tuple(list)`
não prova comprimento 2 pro checador, trocado por desempacotamento
explícito), `model_registry.py` (1 comentário `# type: ignore`
malformado removido, era redundante com `ignore_missing_imports=true`
já setado globalmente). Gate: 10 → 24 módulos. `.github/workflows/test.yml`
atualizado com a lista completa.

## Passo 91 — Comparativo do README: RECONFIRMADO + 1 linha nova

Reconfirmado por nova busca (2026-08-27): Kennard-Stone **é** recurso
padrão do Unscrambler (fonte: busca anterior desta sessão, sem mudança).
Decisão de não reivindicar seleção de amostras/transferência de
calibração como diferencial permanece válida.

Avaliado o CONJUNTO completo (não só seleção de amostras isolada):
planejamento experimental (`plano_amostral.py`+`plano_coleta.py`),
auditoria de delineamento automática (`auditoria_delineamento.py`),
modo cego com conjunto aberto calibrado por predição conforme
(`identificacao.py`), sentinela de deriva (`sentinela_deriva.py`),
dossiê de linearidade/robustez opcional. Busca dedicada (2026-08-27)
por "sample size guidance + confounding audit + conformal open-set
identification" em ferramentas comerciais: nenhuma evidência de suite
comercial bundlando essa combinação — os resultados encontrados são
literatura acadêmica/de fronteira (predição conforme auditada 2026,
D-optimal design), não recurso de produto integrado.

**Decisão: adicionar 1 linha ao comparativo** (README.md e
README.pt-br.md), fraseada com o mesmo cuidado epistêmico de antes
("não encontrado em documentação pública até 2026-08", não "nenhum
concorrente tem") — mais uma nota explícita ao lado da tabela dizendo
que Kennard-Stone/transferência de calibração NÃO são reivindicados
como diferenciais, para que a mudança não pareça contradizer a decisão
anterior sobre esses dois itens especificamente.
