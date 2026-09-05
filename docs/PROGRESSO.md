# PROGRESSO — Passo 152: reavaliação final de HPLC — "suportável, aguardando dataset" (2026-09-04)

## Passo 152 — HPLC: busca ampliada, nenhum dataset compatível encontrado

Instrução: nova busca, mais ampla, por dataset de HPLC com tabela de
picos pronta (formato compatível com o motor atual — amostras como
linhas, picos/áreas como colunas, com alvo supervisionado). Se
encontrado, validar como as demais técnicas tabulares; se não,
documentar HPLC como "tecnicamente suportável, aguardando dataset" —
diferente de placeholder.

**Busca realizada** (mais ampla que a do Passo 141/142, que não tinha
sido iniciada para HPLC): API do Zenodo (múltiplas queries — "HPLC peak
area table", "HPLC honey/olive/wine adulteration", "HPLC-DAD
chemometrics"), Mendeley Data (busca direta), UCI Machine Learning
Repository, e o site especializado `mcrals.info` (via seu espelho no
Zenodo, DOI 10.5281/zenodo.8206165 — coleção de datasets de referência
do grupo Tauler, autoridade em resolução de curva multivariada).

**Candidato mais próximo encontrado**: `adataset.zip`/`bdataset.zip`
(Zenodo 8206165) — dado HPLC-DAD REAL (pesticidas organofosforados em
águas naturais, Tauler, Lacorte & Barceló 1996, *J. Chromatogr. A*
730:177-183), formato MATLAB. **Não serve para o padrão de validação
deste projeto**: é um conjunto de RESOLUÇÃO DE CURVA (1 matriz de
mistura com 3 compostos + 2 matrizes de padrão) — não uma tabela
amostra×alvo com N amostras independentes para PLS/PLS-DA, que é o que
`mode="csv"` do GUARACI espera. Usar esse dado exigiria reformular a
alegação de "quantificação supervisionada" para "resolução de curva
MCR-ALS" — um escopo diferente do que as outras 10 técnicas desta
auditoria validam, não uma substituição equivalente.

**Decisão**: nenhum dataset HPLC compatível com o padrão de validação
encontrado. Registrado como **"suportável, aguardando dataset"** — o
motor genérico (`mode="csv"`, PLS/PLS-DA) já processaria uma tabela de
picos de HPLC sem nenhuma mudança de código, exatamente como já faz
para RMN (variáveis ppm pré-binadas) e as demais técnicas tabulares;
a lacuna é puramente de DADO disponível, não de capacidade do motor —
distinção que evita tanto subestimar (não é "não suportado") quanto
superestimar (não está "validado") a maturidade real desta técnica.

Nenhuma linha de código nova para HPLC nesta rodada (não havia o que
implementar sem um dataset). `docs/VALIDACAO_PUBLICA.md` não ganhou
seção nova para HPLC (nada foi integrado) — só esta nota em
`docs/PROGRESSO.md` e a atualização da tabela de 11 técnicas.

Próximo (Passo 153): consolidação final — atualizar
`docs/VALIDACAO_PUBLICA.md`/README/MANUAL/PROGRESSO.md com o estado
final e completo das 11 técnicas.

---

# PROGRESSO — Passos 150/151: GC-MS validado com dado real; IMS adiado formalmente (Fase D, 2026-09-04)

## Passo 150 — GC-MS: parser ANDI-MS + COW, validado contra 55 amostras reais

Checklist da instrução: (1) avaliar bibliotecas antes de escrever parser
do zero; (2) extrair TIC e picos do formato bruto; (3) implementar
alinhamento de retenção (COW) entre as 55 amostras do dataset de
lavanda; (4) converter para estrutura tabular interna; (5) validar com
split group-aware.

**(1) Bibliotecas avaliadas**: `netCDF4` e `pyms` (PyMassSpec) são as
opções maduras citadas na literatura para ANDI-MS. Antes de adotar
qualquer uma, os arquivos reais do dataset (Mendeley `10.17632/
pgkrc7wyj4.1`, "Lavandula angustifolia essential oil adulteration
dataset", Pokajewicz 2024, 55 amostras comerciais, CC BY 4.0,
confirmado por `file` no binário) se confirmaram netCDF **CLÁSSICO**
(não HDF5) — `scipy.io.netcdf_file` (scipy JÁ é dependência do projeto)
lê direto, sem nenhuma dependência nova. Achado que evitou avaliação de
licença de terceiros por completo.

**(2) TIC extraído**: `src/guaraci/gcms_io.py` novo —
`ler_tic_andi_ms`/`carregar_dataset_gcms` leem `scan_acquisition_time`/
`total_intensity` (nomes de variável padrão ANDI-MS, confirmados por
leitura direta). Baixados e verificados os 55 arquivos reais (~347 MB
total, SHA-256 de cada um vindo da própria API do Mendeley — mesma
prática de scripts anteriores). Extração de picos individuais (não só
TIC) fica fora de escopo — não necessária para o alinhamento por COW.

**(3) COW implementado do zero**: `src/guaraci/alinhamento_retencao.py`
— Nielsen, Carstensen & Smedsgaard (1998), J. Chromatogr. A 805:17-35,
DOI 10.1016/S0021-9673(98)00021-1 (confirmado no Crossref). Programação
dinâmica sobre limites de segmento com folga (`slack`), maximizando a
correlação segmento-a-segmento entre amostra deformada e referência.
Contra-prova obrigatória (regra 9, `tests/test_alinhamento_retencao.py`,
4 testes): cromatograma sintético com warp não-linear CONHECIDO — COW
recupera o alinhamento (correlação sobe 0,486→0,790); sem warp, COW não
piora nada.

**(4)+(5) Validado contra as 55 amostras reais**: TICs resampleados
numa grade de tempo comum (interseção seguRA entre lotes com duração de
corrida ligeiramente diferente, ~75 vs ~82 min) e alinhados por COW
contra uma referência (L01). **Correlação média par-a-par entre as 55
amostras: 0,753 (antes) → 0,884 (depois do COW)** — melhora real,
mensurável, contra dado real. "Group-aware" não se aplica aqui: cada
arquivo é 1 injeção por amostra, sem repetição a proteger.

**Escopo explicitamente NÃO coberto (decisão registrada, não
escondida)**: o dump bruto do dataset (1020 arquivos, só `.CDF`/`.dat`)
não inclui uma tabela de referência com rótulo de autenticidade/
adulteração por amostra — esse rótulo, se existir, está no artigo
companheiro, não no repositório de dados; reconstruí-lo exigiria
identificação de compostos por índice de retenção, escopo maior que
"escrever um parser". Por isso esta validação não tenta classificar/
quantificar adulteração — valida exatamente o que os itens (1)-(5) da
instrução pedem (parser + alinhamento), não além disso.

Novos módulos (`gcms_io.py`, `alinhamento_retencao.py`) adicionados ao
gate de mypy (0 erros). Suíte completa, ruff/mypy limpos.

Reproduzir:
```
python scripts/download_datasets/baixar_mendeley_gcms_lavanda.py
GUARACI_DATASETS_DIR=<pasta> pytest tests/test_validacao_publica_mendeley_gcms_lavanda.py tests/test_alinhamento_retencao.py -v
```

## Passo 151 — IMS: adiamento formal (mesmo tratamento do Bloco 13d)

Checklist: (1) confirmar licença de `gc-ims-tools`; (2) usar a
biblioteca para ler `.mea`; (3) decidir redução da matriz 2D
(retenção×deriva) — tabular ou PARAFAC/multiway; (4) validar contra mel
ou azeite com split group-aware; (5) correção de K₀ se metadados
existirem.

**(1) Licença confirmada**: `gc-ims-tools` (Charisma-Mannheim/
gc-ims-tools) — **BSD-3-Clause**, compatível com GPL-3.0-or-later.

**Por que os itens (2)-(5) foram adiados**: os 2 candidatos reais
encontrados (Mendeley `fr9t5fkkvz`, GC-IMS de 53 azeites por origem
geográfica — Espanha/Itália/Grécia — 157 arquivos, **~6,9 GB**; e
Mendeley `jxj2r45t2x`, GC-IMS de 50 méis por origem botânica — acácia/
canola/melato — 110 arquivos, **~4,8 GB**) são, os dois, **1-2 ordens
de grandeza maiores** que qualquer outro dataset usado nesta auditoria
inteira (o maior até aqui, o esgoto UV-Vis do Passo 147, tinha ~357 MB
— aqui estamos falando de 5-7 GB). Verificado adicionalmente: **nenhum
dos dois dumps brutos inclui uma tabela de rótulo/classe por arquivo**
— os nomes de arquivo são só timestamps de injeção (ex.:
`161201_142928.mea`); o repositório da própria biblioteca
`gc-ims-tools` (checado diretamente via API do GitHub — `docs/source/`
só tem `.rst` de referência de API, nenhum notebook/CSV de exemplo
bundlado) também não traz essa tabela. O rótulo verdadeiro estaria ou
dentro do cabeçalho binário do `.mea` (campo de comentário/descrição do
instrumento G.A.S. — não confirmado sem abrir o formato) ou no artigo
companheiro (Food Research International,
doi:10.1016/j.foodres.2022.111779) — qualquer um dos dois caminhos é
trabalho de escopo próprio, maior que "ler o `.mea` com a biblioteca
já pronta".

**Decisão**: adiar Passo 151 formalmente, mesmo tratamento já dado a
outros itens fora do escopo viável desta rodada (Bloco 13d). Registrado
como pendência real — download de ~5-7 GB e reconstrução de rótulo a
partir de binário/artigo, não "sem tentativa". Nenhuma linha de código
nova para IMS nesta rodada; a licença confirmada e os 2 datasets/
tamanhos ficam documentados para quando uma sessão futura tiver
orçamento de tempo/banda para essa escala.

Próximo (Passo 152): reavaliação final de HPLC (busca mais ampla por
dataset com tabela de picos pronta).

---

# PROGRESSO — Passo 149: parser real para EEM de fluorescência, PARAFAC contra dado real pela 1ª vez (Fase C, 2026-09-04)

## Passo 149 — Fluorescência EEM: dataset melhor encontrado, parser escrito, PARAFAC validado

Instrução (Fase C): escrever parser robusto para o dataset EEM real já
identificado (Mendeley `g6y69g8gwm`, registrado em `eem_multiway.py`
desde o Passo 144/145 como "formato irregular, parser fora de escopo")
e conectar ao PARAFAC generalizado, até então só provado por
contra-prova sintética.

**Achado antes de escrever qualquer parser**: busca por um dataset EEM
alternativo (Zenodo API) encontrou "EEM fluorescence spectral dataset
of olive-oil adulteration samples across five adulterant systems" (DOI
10.5281/zenodo.19755088, CC BY 4.0) — 330 espectros EEM REAIS (35
excitações × 270 emissões) de azeite EVOO (3 marcas) adulterado com 5
óleos (milho/canola/amendoim/soja/noz) em 10 frações conhecidas
(9,09%-90,91%), 3 medições independentes por combinação. **Formato bem
mais regular** que o Mendeley `g6y69g8gwm` (texto tab-separado, 3
linhas de cabeçalho + 270 linhas de dado, confirmado por leitura direta
em 3 arquivos de marcas/rodadas/frações diferentes — mesma grade exata
de comprimentos de onda nos 3). Decisão: usar este dataset em vez de
insistir no parser do formato antigo — a alegação a validar
("PARAFAC funciona em EEM real") não exige ser especificamente aquele
dataset.

**Parser** (`src/guaraci/eem_io.py`, novo): `parse_eem_dat` (1 arquivo,
nunca inventa valor — linha com número de colunas errado ou campo
não-numérico é descartada e contada, nunca completada) +
`carregar_dataset_eem_azeite` (varre a árvore de pastas, mapeia nomes
em chinês → marca/adulterante via substring, calcula fração de azeite
do nome da pasta de razão). **Achado de formato real** (não hipotético):
16/330 pastas (4,8%) nomeiam o arquivo `0.dat` em vez do padrão
`0_RM.dat` — mesmo conteúdo, variação de nomenclatura do dataset
original. Corrigido com busca por glob (`*.dat`) em vez de nome fixo —
as 16 amostras recuperadas sem inventar nada (`_localizar_arquivo_dat`
retorna `None`, nunca adivinha, se achar 0 ou >1 candidato). Com a
correção: **330/330 amostras carregadas, 0% de descarte de linhas**
dentro dos arquivos (formato interno perfeitamente regular).

**Conectado ao PARAFAC** (`eem_multiway.construir_tensor_eem` +
`parafac_eem`, R=3): rodou contra o tensor real (330, 35, 270) pela
primeira vez — erro de reconstrução relativo 0,161, e um dos 3 fatores
recuperados (método não-supervisionado, nunca vê o rótulo) correlaciona
**|r|=0,888** com a fração de azeite REAL. Robusto: idêntico em 5
seeds, faixa 0,85-0,95 para R∈{2..5}.

**Validação group-aware adicional** (regra 5, harness direto com
sklearn — `mode="csv"` não suporta coluna de agrupamento arbitrária,
mesma limitação do Passo 147): PLS sobre o espectro EEM achatado
(9450 canais) prevendo fração de azeite, grupo=(marca,adulterante,razão)
invariante entre as 3 rodadas, `GroupKFold` 5-fold sobre 110 grupos
únicos. **R²=0,976, RMSEP=4,09 p.p.** (n_lv=8; faixa 0,95-0,98 testando
n_lv∈{3,5,8,10} — não frágil a parâmetro).

**Estado da técnica #6 (Fluorescência) na tabela de 11: Parcial→fraco-
mas-real vira Funcional, forte** — o EEM real (a parte que faltava)
agora tem validação completa. `docs/VALIDACAO_PUBLICA.md` §1 e §2h
atualizados; §2c (busca original) e a entrada do Mendeley `g6y69g8gwm`
mantidas com nota de substituição (não apagadas).

Novo módulo `eem_io.py` adicionado ao gate de mypy (0 erros, mesmo
padrão de `hsi_io.py`). Suíte completa, ruff/mypy limpos.

Reproduzir:
```
python scripts/download_datasets/baixar_zenodo_eem_azeite.py
GUARACI_DATASETS_DIR=<pasta> pytest tests/test_validacao_publica_zenodo_eem_azeite.py tests/test_eem_multiway.py -v
```

Próximo (Passo 150/151, Fase D, podem rodar em paralelo): parsers de
formato binário proprietário para GC-MS (NetCDF/ANDI-MS) e IMS (`.mea`).

---

# PROGRESSO — Passo 148: bug real de classificação binária encontrado e corrigido, achado negativo do RMN RETRATADO (Fase B, 2026-09-04)

## Passo 148 — RMN + I de Moran: a investigação achou um bug, não uma limitação

Instrução (Fase B): tentar melhorar o resultado do RMN (documentado como
"0,500 — exatamente o acaso" no Passo 142/143) usando a mesma técnica de
seleção de variável do artigo original (I de Moran), aceitando o
resultado como está — sem insistir numa 3ª abordagem se não melhorasse.

**Passo 1 — referência confirmada via Crossref**: Lamanna, Imparato,
Tano, Braca, D'Ercole & Ghianni (2017), "Territorial origin of olive
oil: representing georeferenced maps of olive oils by NMR profiling",
*Magnetic Resonance in Chemistry* 55(7):639-647, DOI 10.1002/mrc.4566 —
confirmado que usa "the first principal component of NMR variables
selected according to the Moran test" para o mesmo dataset
(Figshare `4307804`).

**Passo 2 — implementação**: `moran_i_mask`/`_pesos_knn`/
`_avaliar_subset_nested_cv_moran` novos em `src/guaraci/
selecao_variaveis.py` (mesmo padrão de VIP/iPLS/CARS/UVE do módulo,
nested-CV respeitada — a máscara é recalculada a cada fold usando só o
treino daquele fold). Escopo deliberadamente restrito: NÃO integrado ao
menu/CLI/`etapa4_selecao_variaveis`, porque nenhum outro dataset deste
projeto publica coordenadas geográficas por amostra — um mecanismo
genérico de "coords" no config/CSV/menu para um método que só um
dataset consegue usar seria especulação, não necessidade. Contra-prova
obrigatória (regra 9) em `tests/test_selecao_moran.py`: dataset
sintético com variáveis geograficamente correlacionadas misturadas com
ruído puro — o método recupera as primeiras e rejeita o segundo;
embaralhar as coordenadas destrói o sinal detectado (5 testes, todos
passando).

**Passo 3 — reavaliação, e a descoberta real**: ao rodar a comparação
Full vs. Moran com um harness de CV limpo (`_avaliar_subset_cv`, direto
em `selecao_variaveis.py`, nunca passa por `pipeline.executar()`), o
resultado **Full (125 variáveis) já saiu em balanced_accuracy≈0,94** —
completamente inconsistente com o "0,500 (acaso)" documentado
anteriormente via `pq.executar()`. Isso não é "o harness ficou mais
generoso" — é sinal de bug. Investigação (reprodução passo a passo do
código de `pipeline.executar()`, instrumentação com monkeypatch em
`classification_metrics` para capturar y_true/y_pred reais em cada
chamada) isolou a causa exata:

`pipeline.py` construía o alvo one-hot com `Y_bin =
LabelBinarizer().fit_transform(rotulos)` e só expandia para 2 colunas
quando `Y_bin.ndim == 1` — mas para EXATAMENTE 2 classes o sklearn já
devolve shape `(n, 1)`, que tem `ndim == 2` (não 1). A expansão nunca
disparava. Com Y_bin de 1 coluna só, `np.argmax(Y_bin, axis=1)` é
SEMPRE 0, e toda predição downstream colapsava na PRIMEIRA classe —
balanced_accuracy trava em exatamente 0,5 para QUALQUER dataset
binário, independente de pré-processamento (por isso os 4 presets
testados no Passo 142/143 davam todos o mesmo 0,500 — o colapso
acontecia DEPOIS de qualquer pré-processamento, na decodificação da
predição final). A checagem CORRETA (`Y_bin.ndim == 1 or Y_bin.shape[1]
== 1`) já existia — e sempre existiu — em
`avaliacao_modelos.PLSDAClassifier.fit`, `hsi_multiway.
NPLSClassifier.fit` e `portao_correcao_sinal.py`; só o caminho de
classificação PRINCIPAL (`pipeline.executar()`, usado por toda execução
N1/N2 deste projeto inteiro) tinha ficado para trás dessa correção.

**Por que isso nunca apareceu antes**: o Figshare `4307804` (RMN) é o
ÚNICO dataset público validado neste projeto com exatamente 2 classes
(Mendeley tem 8 espécies; Fluorescência, 3 graus; HSI, 3 estágios de
maturação) — o bug só se manifesta no caso binário exato.

**Correção**: `src/guaraci/pipeline.py`, 1 condição (`or Y_bin.shape[1]
== 1` adicionado). Contra-prova obrigatória (regra 9):
`tests/test_pipeline_core.py::
test_executar_classificacao_binaria_nao_colapsa_em_uma_classe_so` —
dataset sintético de 2 classes bem separadas via `pq.executar()`
completo (não o harness interno), que colapsaria para balanced_accuracy
≈0,5 com o bug antigo e agora classifica >0,9; verifica também que as
predições cobrem as 2 classes (não uma só).

**Resultado, medido com o bug corrigido**: RMN classifica província
(Pescara/Teramo) com **balanced_accuracy = 1,000 (CV) / 1,000
(holdout)**, robusto a 5 presets de pré-processamento e 10 seeds de CV
(faixa 0,98-1,00). **Consistente com o artigo original** (99% com LDA +
seleção geoestatística) — o motor genérico do GUARACI, SEM nenhuma
seleção de variável, já chega ao mesmo patamar. A comparação Full vs.
Moran (mesmos folds, harness limpo, nunca afetado pelo bug): Full=0,937,
Moran(~31 var)=0,957 — diferença pequena, os dois já perto do teto.
**A seleção geoestatística acabou não sendo o que faltava — o bug de
classificação binária era.**

**Estado da técnica #9 (RMN) na tabela de 11: RETRATADO de
"negativo-documentado" para Funcional, um dos resultados mais fortes da
tabela.** `docs/VALIDACAO_PUBLICA.md` §1 e §2e reescritos com retratação
formal (nota antiga preservada e marcada como retratada, não apagada).

Suíte completa, ruff limpo, golden do contrato de API pública
regravado (`GUARACI_REGRAVAR_GOLDEN=1`, `moran_i_mask` novo em
`selecao_variaveis.__all__`).

Reproduzir:
```
python scripts/download_datasets/baixar_figshare_azeite_nmr.py
GUARACI_DATASETS_DIR=<pasta> pytest tests/test_validacao_publica_figshare_azeite_nmr.py tests/test_selecao_moran.py tests/test_pipeline_core.py::test_executar_classificacao_binaria_nao_colapsa_em_uma_classe_so -v
```

Próximo (Passo 149, Fase C): parser real para o dataset EEM de
fluorescência — candidato melhor que o originalmente cogitado já
identificado (Zenodo `10.5281/zenodo.19755088`, 330 espectros EEM reais,
adulteração quantitativa de azeite com 5 óleos diferentes, CC BY 4.0 —
ver anotação de pesquisa desta sessão).

---

# PROGRESSO — Passo 147: UV-Vis validado com dataset real (Fase A, 2026-09-04)

## Passo 147 — UV-Vis: ERIC/Eawag (esgoto bruto), EMSC como pedido pela instrução

Busca (Zenodo API + WebSearch; ScienceDirect bloqueado por CAPTCHA em
pelo menos 2 candidatos — regra permanente, não contornado) encontrou
Lechevallier et al. (2025), "Dataset on wastewater quality monitoring
with adsorption and reflectance spectroscopy in the UV/Vis range",
*Scientific Data* 12:1296, doi:10.1038/s41597-025-05459-x — campanha de
25 semanas de esgoto bruto (Suíça), 2 espectrofotômetros UV-Vis a cada 2
minutos, 533 amostras de laboratório para 9 indicadores de poluição.
Dados em ERIC open (Eawag), **licença CC BY** confirmada via API do
portal CKAN.

Baixado só `2_data.zip` (~357 MB, CSV) — o pacote completo tem mais
~180 GB de cubos hiperespectrais não usados aqui. Script novo:
`scripts/download_datasets/baixar_eawag_esgoto_uvvis.py` (mesma
disciplina de segurança dos demais — HTTPS, SHA-256+tamanho pinados,
streaming para temporário e só promovido ao destino final depois de
verificado, por ser um arquivo grande demais para carregar inteiro em
memória como os scripts anteriores).

**Metodologia (documentada em detalhe no módulo de teste e em
`docs/VALIDACAO_PUBLICA.md` §2g)**: sensor `scan`/Spectrolyser
(200-735nm, faixa completa sem descarte); alvo DOC (um dos 5
indicadores com cobertura nas 533 amostras); casamento
laboratório↔sensor por timestamp mais próximo (tolerância 3 min,
513/529 casaram); **agregação por dia** (82 dias) para satisfazer a
regra 5 (group-aware) sem precisar de uma coluna `mae_id` que o motor
CSV não suporta — mesma solução já usada para a Fluorescência (§2d).
Pré-processamento: EMSC (já aprovado no Corn, Passo 134) + MC, ISOLADO
(sem SNV/SG) para atribuir o efeito só ao EMSC, por pedido explícito da
instrução.

**Medido em 2026-09-04**: RMSEC=20,83, RMSECV=31,71, RMSEP=34,25 mg/L,
**R²cal=0,616 / R²val=0,650**, 7 LVs. Sanity check (não há RMSEP
publicado para este recorte) — mas R²val positivo e substancial
confirma que a calibração capturou sinal real sobre um espectro UV-Vis
verdadeiro, não é ruído.

**Achado colateral (bug de rotulagem, corrigido)**: ao montar a config
com `default_preprocessing="custom"` + `apply_emsc=True`, o nome da
pasta de saída (`pipeline.generate_output_name`) saiu como
`..._SNV-SG1-MC_...` — sem nenhuma menção a EMSC, porque a função só
checava `apply_snv`/`apply_sg`/`apply_mc`, nunca `apply_emsc`/
`apply_airpls`/`apply_osc`. O cálculo em si estava correto (EMSC de
fato rodou — confirmado lendo `preprocessamento.build_preprocessor`),
só o RÓTULO da pasta mentia sobre o que rodou. Isso afeta retroativamente
a rastreabilidade de QUALQUER execução anterior com EMSC/AirPLS/OSC via
`default_preprocessing="custom"` (todo o portão de aceite do Passo
134/135, e o Raman com AirPLS do Passo 144/145) — o cálculo dessas
execuções nunca esteve errado, só o nome da pasta que as guarda.
Corrigido em `src/guaraci/pipeline.py::generate_output_name` (3 linhas
novas, mesma ordem de `build_preprocessor`). Teste de regressão:
`tests/test_pipeline_core.py::test_gerar_nome_saida_custom_declara_emsc_airpls_osc`.

**Estado da técnica #5 (UV-Vis) na tabela de 11: Parcial → Funcional.**
`docs/VALIDACAO_PUBLICA.md` §1, §2g e §4 atualizados.

Reproduzir:
```
python scripts/download_datasets/baixar_eawag_esgoto_uvvis.py
GUARACI_DATASETS_DIR=<pasta> pytest tests/test_validacao_publica_eawag_esgoto_uvvis.py -v
```

Próximo (Passo 148, Fase B): tentar melhorar o RMN via seleção
geoestatística de variável (I de Moran).

---

# PROGRESSO — Passo 146: NIR Dispersivo reclassificado sem dataset novo (Fase A, 2026-09-04)

## Passo 146 — NIR Dispersivo: Eigenvector Corn reclassificado (achado por reconferência)

Instrução (Fase A do plano de fechamento das 11 técnicas): antes de
buscar dataset novo para NIR Dispersivo, reverificar se algum dataset
já baixado nesta sessão foi classificado erroneamente como FT-NIR
quando na verdade é dispersivo.

O Corn (Eigenvector, `m5`/`mp5`/`mp6`, integrado desde Passo 78/79 — ver
`docs/VALIDACAO_PUBLICA.md` §1) nunca tinha sido atribuído a nenhuma das
11 técnicas do menu no levantamento do Passo 141 — contava só como
prova de que "o motor reproduz a literatura", solto da tabela de 11
técnicas (não era um caso de má classificação como FT-NIR; era ausência
de classificação).

**Achado, confirmado por busca direta (Crossref/leitura de artigo, não
suposição):** os 3 instrumentos do Corn são da família **FOSS
NIRSystems 5000/6500**. Digman, Cherney & Cherney (2022), *Sensors*
22(2):658, doi:10.3390/s22020658 — estudo que compara diretamente um
espectrômetro NIR de bancada contra um FT-NIR portátil — descreve o
"(FOSS) NIRSystem 6500 (FOSS, Hillerød, Denmark)" explicitamente como
"a scanning monochromator spectrometer with a wavelength range from
1100 to 2498 nm" — faixa **idêntica** à do Corn (1100-2498nm,
confirmado em `tests/test_validacao_publica.py`). O mesmo artigo
contrasta esse instrumento diretamente com um FT-NIR (interferômetro de
Michelson) medido no mesmo estudo — monocromador de rede com varredura
mecânica e Fourier-transform são categorias tecnológicas distintas na
própria literatura de instrumentação NIR. **O Corn nunca foi FT-NIR** —
é NIR Dispersivo de livro-texto, só nunca conectado à tabela de 11
técnicas até agora.

Nenhuma mudança de código foi necessária: o teste já existente
(`test_guaraci_reproduz_a_literatura_no_corn`) já reproduz RMSEP=0,144
%m/m (proteína, m5), R²val=0,912, 8 LVs — reconfirmado por execução
direta nesta rodada (dataset baixado de novo, SHA-256/tamanho
conferidos contra os valores pinados no CI: `e28fd4be...c46b5`,
1445616 bytes — batem). Split: `frac_holdout=0.25`,
`group_by_mae_id=False` — correto para este dataset: cada uma das 80
amostras tem 1 medição por instrumento, sem repetição técnica a
agrupar, logo nenhum grupo pode vazar entre treino e validação.
Suíte completa (`pytest tests/test_validacao_publica.py -k corn -v`):
5 testes relacionados ao Corn, todos passando.

**Estado da técnica #2 (NIR Dispersivo) na tabela de 11: Parcial →
Funcional.** `docs/VALIDACAO_PUBLICA.md` §1 e §2f atualizados com a
atribuição de técnica explícita e a citação completa do achado.

Reproduzir:
```
curl -fsSL -o corn.mat https://eigenvector.com/data/Corn/corn.mat
GUARACI_DATASETS_DIR=<pasta que contem corn.mat> pytest tests/test_validacao_publica.py -k corn -v
```

Próximo (Passo 147, ainda Fase A): buscar dataset público de UV-Vis
para quantificação.

---

# PROGRESSO — Fechamento da auditoria das 11 técnicas analíticas (Passos 141-145, 2026-09-04)

## Estado final de cada uma das 11 técnicas do menu (`cli_assistente.TECNICAS`)

| # | Técnica | Estado (Passo 141) | Dataset público | Validação real | Correção complementar |
|---|---|---|---|---|---|
| 1 | **FT-NIR** | Funcional | Mendeley `ctgg7k4m5g` (NIR8mm), CC BY 4.0 | Bal.acc 0,475 (holdout); R²cal 0,83/R²val −0,53 | — (já era o caso de uso original) |
| 2 | **NIR Dispersivo** | **Funcional (novo, reclassificação — Passo 146)** | Eigenvector Corn (`m5`/`mp5`/`mp6`) — já integrado desde Passo 78/79, nunca antes atribuído a uma técnica do menu | RMSEP 0,144 %m/m (proteína, m5); R²val 0,912; 8 LVs | Reclassificação: FOSS NIRSystems 5000/6500 = "scanning monochromator spectrometer" (Digman, Cherney & Cherney 2022, *Sensors* 22(2):658) — tecnologia dispersiva, NÃO Fourier-transform. Nenhum dataset novo necessário |
| 3 | **MIR/FTIR** | **Funcional (novo)** | Arquivo-irmão do NIR8mm no mesmo dataset Mendeley | Bal.acc 0,696; R²cal 0,79/R²val 0,57 | — |
| 4 | **Raman** | **Funcional (novo)** | Arquivo-irmão do NIR8mm | Bal.acc 0,389; R²cal 0,67/R²val 0,43 | **AirPLS implementado e APROVADO no portão de aceite** (RMSEP 0,442→0,424, p=0,002) |
| 5 | **UV-Vis** | **Funcional (novo — Passo 147)** | ERIC/Eawag `000D3C19` (Lechevallier et al. 2025, CC BY) — esgoto bruto, sensor Spectrolyser 200-735nm | R²cal 0,616 / R²val 0,650 (DOC, 82 dias agregados, EMSC+MC, sanity check) | EMSC **já existia** (aprovado desde Passo 134) — usado como pedido, achado colateral: bug de rótulo em `generate_output_name` corrigido (ver abaixo) |
| 6 | **Fluorescência Molecular** | **Funcional, forte (Passo 149)** | Mendeley `thkcz3h6n6` (simples, bal.acc 0,383) + **Zenodo `19755088` (EEM real, INTEGRADO)** | EEM: PARAFAC \|r\|=0,888 com fração real; PLS group-aware R²=0,976, RMSEP=4,09 p.p. | **PARAFAC generalizado rodou contra EEM real pela 1ª vez** (`eem_io.py` novo, parser regular — Mendeley `g6y69g8gwm` permanece fora de escopo, substituído por dataset melhor) |
| 7 | **HPLC** | **Reavaliado (Passo 152) — "suportável, aguardando dataset"** | Busca ampliada (Zenodo API, Mendeley, UCI, site MCR-ALS) não achou tabela de picos pronta com alvo supervisionado — candidato mais próximo (Zenodo `8206165`, Tauler et al. 1996) é dado de resolução de curva (3 compostos), não um conjunto amostra×alvo | Não validado | Nenhuma técnica de correção nova implementada — sem dataset compatível, não há o que validar; motor genérico já suportaria `mode="csv"` se um dataset aparecer |
| 8 | **GC-MS** | **Funcional (Passo 150)** | Mendeley `pgkrc7wyj4` (55 amostras de lavanda, CC BY 4.0) | COW: corr. média par-a-par 0,753→0,884 (55 amostras reais) | `scipy.io.netcdf_file` lê ANDI-MS direto (zero dependência nova); COW implementado do zero e contra-provado; AMDIS/MCR-ALS não necessários para esta validação |
| 9 | **RMN/NMR** | **RETRATADO no Passo 148 → Funcional, forte** | Figshare `4307804` (CC0, já binado) | ~~Bal.acc 0,500 (acaso)~~ **1,000 (CV/holdout)** — o "0,500" era bug de classificação binária em `pipeline.py`, corrigido no Passo 148, não limitação do dado (ver `docs/VALIDACAO_PUBLICA.md` §2e) | icoshift (referência confirmada) — não implementado, dataset já vem pré-binado |
| 10 | **IMS** | **Parcial — adiado formalmente (Passo 151)** | 2 datasets GC-IMS reais achados (Mendeley `fr9t5fkkvz` azeite 6,9GB/157 arq.; `jxj2r45t2x` mel 4,8GB/110 arq., ambos CC BY 4.0) | Não validado | `gc-ims-tools` (BSD-3, compatível) confirmado; **adiado**: ambos datasets ~1-2 ordens de grandeza maiores que qualquer outro deste projeto, E nenhum tem tabela de rótulo/classe no dump bruto (nem no repo da biblioteca) — reconstruir exigiria abrir o binário `.mea` ou o artigo companheiro, escopo maior que "escrever um parser" |
| 11 | **Genérico** | Funcional | — (é o próprio fallback) | — | — |

## Resumo em uma frase (exigido pela instrução original)

> **Nota histórica preservada** (estado em 2026-09-04, ao fechar os
> Passos 141-145, ANTES das Fases A-E do plano de fechamento): "Das 11
> técnicas, 6 têm validação real com dado público agora (FT-NIR, MIR,
> Raman, Genérico com sinal genuíno acima do acaso; Fluorescência com
> sinal fraco mas real; RMN com resultado NEGATIVO honestamente
> documentado); NIR Dispersivo e UV-Vis permanecem sem validação
> própria; HPLC, GC-MS e IMS permanecem como placeholder de fato."
> **Retratado/superado pelas Fases A-E abaixo — ver o resumo atual.**

**Estado final (2026-09-04, ao fechar as Fases A-E do plano de
fechamento das 11 técnicas)**: **9 das 11 técnicas têm validação real,
forte, com dado público** (FT-NIR, NIR Dispersivo, MIR, Raman, UV-Vis,
Fluorescência — incluindo EEM real —, RMN, GC-MS, Genérico). Duas
permanecem sem validação, por limitação GENUÍNA e documentada, não por
falta de tentativa: **HPLC** ("suportável, aguardando dataset" — motor
genérico já processaria uma tabela de picos se um dataset compatível
aparecer) e **IMS** (adiado formalmente — 2 datasets reais identificados,
mas 5-7 GB cada e sem rótulo de referência, escopo maior que "escrever
um parser"). Achado mais significativo da rodada: o "resultado negativo"
do RMN nunca foi real — era um bug de classificação binária em
`pipeline.py` (afetava qualquer dataset com exatamente 2 classes,
encontrado só agora porque o RMN é o único dataset deste projeto com
essa estrutura), corrigido e retratado por completo (§2e). Duas
correções de física foram implementadas e aprovadas contra dado real
(AirPLS para Raman, portão de aceite; COW para GC-MS, contra-prova
sintética + 55 amostras reais); PARAFAC (EEM) e I de Moran (RMN) saíram
da fase "só sintético" para rodar contra dado real pela primeira vez;
EMSC (UV-Vis) e MCR-ALS (cogitado para HPLC) confirmaram já existir,
evitando trabalho duplicado.

---

# PROGRESSO — Passo 144/145: AirPLS (Raman) e PARAFAC/EEM (Bloco 30, 2026-09-04)

## Passo 144 — Levantamento de análises complementares por modalidade

Para cada técnica com sinal real (MIR/Raman/Fluorescência/RMN validados
nos Passos 142/143) e as bloqueadas (HPLC/GC-MS/IMS), busquei correção
específica de física com referência confirmada no Crossref antes de
propor: airPLS (Zhang, Chen & Liang 2010, DOI 10.1039/b922045c) p/
Raman; icoshift (Savorani et al. 2010, DOI 10.1016/j.jmr.2009.11.012)
p/ RMN; COW (Nielsen et al. 1998, DOI 10.1016/s0021-9673(98)00021-1)
p/ HPLC/GC-MS; AMDIS (Stein 1999, DOI 10.1016/s1044-0305(99)00047-1)
p/ deconvolução GC-MS; Mason-Schamp (1958, DOI
10.1016/0003-4916(58)90049-6) p/ correção de mobilidade reduzida em
IMS. **Achado que evitou trabalho duplicado**: a correção de
espalhamento/turbidez proposta p/ UV-Vis **já existe** —
`EMSC` (Martens & Stark 1991) já implementado e já aprovado pelo
portão de aceite desde o Passo 134, disponível pra qualquer técnica via
`cfg.apply_emsc`. Das 7 propostas, só 2 tinham escopo implementável
nesta rodada (zero dependência nova, dataset real disponível): airPLS
e PARAFAC generalizado p/ EEM — as outras ficam bloqueadas até HPLC/
GC-MS/IMS terem parser (Passo 143 não concluído p/ essas 3).

## Passo 145 — AirPLS: implementado, testado, **APROVADO no portão de aceite**

`AirPLS` (`preprocessamento.py`) -- Whittaker smoother com
reponderação iterativa adaptativa (Zhang, Chen & Liang 2010).

**Bug real achado E corrigido durante a própria contra-prova
sintética** (disciplina do Passo 145 funcionando como deveria): a
fórmula do peso de borda usava `abs(negativos).max()` (magnitude
MÁXIMA do resíduo negativo) em vez de `negativos.max()` (resíduo mais
PRÓXIMO de zero, a fórmula correta do artigo original) -- a versão
errada fazia o peso da borda explodir exponencialmente nas últimas
iterações (o denominador `dssn` encolhe a cada volta), dominando o
ajuste e degradando a linha de base progressivamente em vez de
convergir. Medido diretamente: com o bug, razão erro-corrigido/
erro-bruto ficava em ~0.85-0.90 (mal corrigia); corrigido, ~0.04 (quase
recupera os picos puros). A contra-prova sintética (espectro Raman
simulado com fluorescência de fundo larga) só passou depois da
correção -- exatamente o que a regra "contra-prova específica de cada
correção" existe para pegar.

**Portão de aceite (Bloco 20) contra Raman1A.csv real** (Mendeley
ctgg7k4m5g, já integrado nos Passos 142/143), 10 seeds, Wilcoxon
pareado: **APROVADO** -- RMSEP (log10 índice de peróxido) 0,442→0,424,
p=0,002. É o único método deste módulo aprovado direto na primeira
tentativa (EMSC/OSC precisaram de 2 datasets pra' decidir, ver §9 do
VALIDACAO_PUBLICA.md). Novo preset nomeado `airpls_sg_mc` em
`build_preprocessor` (AirPLS→SG→MC, sem MSC/SNV -- baseline Raman é
aditiva, não multiplicativa); `cli_assistente.TECNICAS["raman"]`
atualizado pra recomendar esse preset (era `"sg_mc"`, que na prática
caía em SNV+SG+MC por não bater com nenhum preset nomeado -- bug de
roteamento pré-existente, corrigido de passagem).

## Passo 145 — PARAFAC generalizado para EEM (Fluorescência)

`eem_multiway.py` (módulo novo, não reaproveita `hsi_multiway.py`
diretamente): `construir_tensor_eem` empilha `{amostra: matriz EEM}`
num tensor 3-way (SEM a lógica de redução espacial por ROI da versão
HSI -- EEM de uma campanha compartilha a MESMA grade excitação/emissão
por construção, não precisa reduzir nada); `parafac_eem` reusa a
decomposição de `hsi_multiway.parafac_hsi` (matematicamente genérica)
mas RENOMEIA os fatores pro vocabulário correto (`fator_excitacao`/
`fator_emissao`, não `fator_espacial`/`fator_espectral` -- mesmo
cuidado de vocabulário que motivou `perfil_matriz.py`, Passo 141).

Contra-prova sintética (exigida antes de qualquer dado real): EEM
simulada como mistura linear de 2 componentes puros (perfis excitação/
emissão gaussianos bem separados) com proporções por amostra
conhecidas -- PARAFAC recuperou as 2 proporções com correlação >0,9
cada (melhor correspondência sob a ambiguidade de permutação/sinal do
método), erro de reconstrução <10%.

**Dataset público real (Mendeley `g6y69g8gwm`, 24 azeites, EEM real em
10 etapas de envelhecimento) baixado e inspecionado, mas NÃO
integrado**: o CSV bruto de exportação do instrumento é IRREGULAR por
bloco excitação/amostra (confirmado por tokenização linha-a-linha --
contagem de campos varia de 281 a 1 dependendo da linha, sem
preenchimento consistente) -- escrever um parser robusto pra esse
formato específico é trabalho de escopo próprio, fora do que foi
aprovado ("PARAFAC generalizado", não "parser do formato do
instrumento X"). Registrado como pendência honesta, não escondida --
mesma disciplina do MCR-ALS antes de tocar dado real (Bloco 14).

6 testes novos (3 AirPLS + 4 EEM, um deles reaproveitando o Raman
público já baixado). Contrato de API pública regravado (AirPLS +
2 campos de Config + módulo eem_multiway novos). Ruff/mypy limpos.

---

# PROGRESSO — Passo 142/143 continuação: Fluorescência e RMN (2026-09-04)

## Passo 142/143 — Fluorescência (Mendeley thkcz3h6n6) e RMN (Figshare 4307804), Bloco 29

Busca real de dataset público para UV-Vis/Fluorescência/RMN (Passo 142)
registrada em `docs/VALIDACAO_PUBLICA.md` §2c: RMN achou candidato
ótimo (CC0, já binado) mas com Figshare bloqueando download
automatizado (desafio de bot AWS WAF, contornado via download manual
pelo usuário + verificação de hash); Fluorescência achou 2 candidatos
(versão simples 1D usada, versão EEM real de instrumento bruto
registrada como pendência — parser novo necessário, fora do escopo);
UV-Vis não achou candidato bom o suficiente (melhor achado, honey
Bangladesh 1960 amostras, ficou atrás de CAPTCHA — não contornado, regra
permanente).

**Fluorescência** (`tests/test_validacao_publica_mendeley_fluorescencia.py`):
24 óleos (EXTRA/VIRGEN/LAMPANTE), 20 repetições técnicas colapsadas em
média por amostra (motor do GUARACI só agrupa por `mae_id` próprio,
não aplicável a dataset externo — forçar um `mae_id` artificial
contaminaria vocabulário, o problema que `perfil_matriz.py` existe
para evitar). balanced_accuracy=0,383 (CV) — sinal fraco mas acima do
acaso (~0,333), n=24 pequeno. Testei hipótese de subtração de fundo do
instrumento: resultado IDÊNTICO (MSC+SG+MC já remove offset antes da
subtração importar) — achado negativo registrado, não escondido.

**RMN** (`tests/test_validacao_publica_figshare_azeite_nmr.py`): 97
azeites, 125 variáveis ppm já binadas pelos autores, alvo=província
(Pescara/Teramo). **Achado NEGATIVO real**: balanced_accuracy=0,500 —
EXATAMENTE o acaso, testado com 4 presets de pré-processamento
diferentes (todos idênticos). Hipótese registrada: só ~32% das
variáveis carregam sinal (aviso do próprio GUARACI); o artigo original
usou seleção geoestatística de variável (I de Moran) antes de LDA, não
PLS-DA ingênuo sobre tudo — reproduzir os 99% publicados exigiria essa
seleção, fora de escopo. Teste sem gate de sucesso (só confirma que
roda sem exceção) -- inventar um piso de acerto aqui mentiria sobre o
que foi medido.

CI (`validacao-publica-mendeley`) atualizado para Fluorescência (RMN
deliberadamente FORA do CI -- bloqueio de bot dependeria do IP do
runner, seria flakiness real, não bug). 2 testes novos, suíte
completa, ruff/mypy limpos.

---

# PROGRESSO — Passo 141-143 (2026-09-04)

## Passo 141 — Auditoria de realidade das 11 técnicas do menu

Instrução nova: auditar o catálogo de 11 técnicas analíticas
(`cli_assistente.TECNICAS`) por evidência de código direto, não pela
existência no menu. Achado estrutural: cada entrada do dicionário só
tem 4 campos funcionais (`faixa_min`/`faixa_max`/`preproc`/`mode`) —
selecionar uma técnica no menu (`guaraci._menu_technique`) só escreve
esses 4 valores na config, sem *dispatch* algum por técnica em
nenhum outro lugar do pipeline. Só existem 3 parsers no sistema
(`parse_dx` genérico JCAMP-DX, `load_csv` genérico tabular, `parse_opus`
binário Bruker — este último não ligado a nenhum fluxo alcançável pelo
menu, só chamável direto). Zero correção específica de física de
técnica em qualquer lugar do código (busca por `baseline`/`airPLS`/
`warp`/`binning`/`drift time`: zero ocorrências).

Classificação: **funcional** só FT-NIR (única com validação real
ponta-a-ponta) e Genérico (fallback idêntico em mecânica); as outras 9
são **parcial** — herdam o motor genérico (que roda de fato sobre o
dado se vier em CSV/DX plano) mas sem nenhuma correção física
específica nem dataset próprio validado. Nenhuma é *placeholder* puro.

Achado que decidiu o escopo do Passo 142: o dataset Mendeley
`10.17632/ctgg7k4m5g.2` (Ottaway et al. 2021), já integrado para NIR em
`docs/VALIDACAO_PUBLICA.md` §2, contém `MIR1A.csv` e `Raman1A.csv` —
arquivos-irmãos das MESMAS 100 amostras, mesmo alvo, mesma licença.
Também confirmado: `hsi_multiway.py` tem PARAFAC real (`tensorly`), mas
o wrapper `construir_tensor_amostras` é acoplado a cubo espacial de
imagem — não diretamente reaproveitável para EEM de fluorescência sem
adaptação (só a chamada PARAFAC de baixo nível seria reusável).

Usuário escolheu priorizar "MIR+Raman primeiro" (esforço baixo, reusa
infra pronta) antes de buscar dataset novo para UV-Vis/Fluorescência/
RMN e avaliar a complexidade maior de HPLC/GC-MS/IMS.

## Passo 142/143 — MIR e Raman validados via arquivos-irmãos do NIR (Bloco 28)

`scripts/download_datasets/baixar_mendeley_oleos.py` estendido para
baixar `MIR1A.csv`/`Raman1A.csv` (SHA256/tamanho pinados a partir da API
pública do Mendeley — confirmado que bate com o hash de `NIR8mm1A.csv`
já pinado desde 2026-08-26, então a mesma fonte é confiável para os 2
arquivos novos). Verificado por leitura direta: `Class`/`PeroxideValue`
idênticos linha-a-linha nos 3 arquivos (mesmas 100 amostras, mesma
ordem) — MIR1A.csv sem NaN, Raman1A.csv com 1 linha 100% NaN nas
colunas espectrais (removida antes de treinar).

`tests/test_validacao_publica_mendeley_mir_raman.py` (4 testes, mesmo
protocolo do teste de NIR: classificação 8 espécies n≥5/n=78,
regressão pooled em log10(índice de peróxido), holdout=25, seed=0):

| Técnica | Bal. acc. | R²cal | R²val | RMSEP (log10) |
|---|---:|---:|---:|---:|
| NIR 8mm (referência, §2) | 0,475 | 0,83 | −0,53 | 0,49 |
| MIR | **0,696** | 0,79 | **0,57** | 0,26 |
| Raman | 0,389 | 0,67 | 0,43 | 0,26 |

Achado real (não sicofanteado): MIR e Raman tiveram R²val POSITIVO
nesta medição pontual, ao contrário do NIR 8mm — mas é uma única
medição com holdout de alta variância (n=100), não motivo para afirmar
"MIR é melhor que NIR" como conclusão geral. Nenhum perfil de matriz
dedicado existe para MIR/Raman de óleos comestíveis ainda (usado
`matrix_profile="generico"` com `wn_min`/`wn_max` explícitos) — registrado
como pendência para um Passo futuro, fora do escopo aprovado aqui.

CI (`validacao-publica-mendeley`) atualizado para rodar o arquivo novo
nos 3 SOs. `docs/VALIDACAO_PUBLICA.md` §2b documenta os números. 4
testes novos, suíte completa (1343 passed, 22 skipped). Ruff/mypy
limpos.

---

# PROGRESSO — Passo 140 (2026-09-04)

## Passo 140 — interval-VIP com validação aninhada (Bloco 27, fecha a instrução de portão de aceite)

**Estado do iPLS confirmado por comando direto ANTES de escrever
código** (exigência explícita do bloco): `grep -in "interval.vip\|ivip"
src/guaraci/*.py tests/*.py` → zero ocorrências. `selecao_ipls`
(iPLS, refit completo de PLS-DA por intervalo) e `vip_scores`/
`_mask_vip_threshold` (VIP por variável individual) já existiam,
`_mask_melhor_intervalo` já tinha nested-CV desde o Passo 126 (Bloco
17). A lacuna real: nenhum método combinava "intervalo espectral" com
"VIP" — nem iPLS (usa CV, não VIP) nem VIP (por variável, não por
região).

Implementado só essa lacuna em `selecao_variaveis.py`:
`_vip_por_intervalo` (1 único fit de PLS-DA no espectro inteiro, VIP
agregado — média — dentro de cada intervalo, mais barato que o iPLS que
reajusta 1 modelo POR intervalo — confirmado por teste que conta
instâncias de `PLSRegression`), `_mask_melhor_intervalo_vip` (nested-CV,
mesma disciplina do resto do módulo), `selecao_interval_vip` (função
pública de diagnóstico, mesmo padrão de `selecao_ipls`). Item novo,
sempre ativo (mesmo custo de VIP/SR/iPLS, não opt-in como CARS/UVE/SPA/
AG), na tabela da Etapa 4.

**Testado no cenário de adulterante minoritário, sob o portão do Bloco
20** (exigência explícita): classe minoritária ~17%, sinal concentrado
em 10 de 300 canais. **Achado real durante a validação**: a primeira
tentativa (p=60, amplitude=3,0) deu efeito-teto — balanced_accuracy=1,0
dos dois lados, mesma armadilha já encontrada e corrigida no Passo 132
(Bloco 15) — corrigido aumentando p e reduzindo amplitude/aumentando
ruído até o cenário ficar genuinamente difícil. Resultado real medido
(não escolhido a dedo): **APROVADO**, balanced_accuracy 0,73→0,93,
p=0,002, 10 seeds.

6 testes novos. Suíte completa (1341 passed, 19 skipped — +6 vs. Passo
139). Contrato de API pública regravado (1 nome novo em
`selecao_variaveis.__all__`). Ruff/mypy limpos.

**Fecha a instrução do portão de aceite (Blocos 20-27)**: o portão
central está implementado e provado por contra-prova (identidade→
neutro, ruído→rejeitado, ganho sintético real→aprovado); EMSC/OSC e
PDS/DS foram avaliados formalmente (resultado misto para EMSC/OSC —
aprovado no Corn, rejeitado/aprovado dependendo do cenário no óleo — e
retratação honesta para DS, que ajuda mais do que o registro anterior
dizia); MCR-ALS tem aviso de escopo permanente em 3 superfícies; faixa
de decisão, amostragem ativa, política pooled/local e interval-VIP
estão implementados e testados.

---

# PROGRESSO — Passo 139 (2026-09-04)

## Passo 139 — Política automática pooled vs. local (Bloco 26)

Novo módulo `politica_pooled_local.py`: `decidir_pooled_vs_local(X, y,
rotulos, grupos, especie)` formaliza a decisão entre um modelo pooled
(todas as espécies juntas) e um modelo LOCAL (só aquela espécie) — os
dois já são calibrados hoje (`pipeline.pls_regressao_pooled`/
`pls_regression_by_species`), mas até este passo nunca havia decisão
formal sobre qual usar. **Reaproveita o portão do Bloco 20** (mesmo
motor `avaliar_correcao_sinal`, Wilcoxon pareado), não um mecanismo
paralelo: "sem" = RMSEP da espécie quando o modelo é treinado no pooled
inteiro; "com" = RMSEP quando treinado só nos dados da espécie — os
dois avaliados nas MESMAS amostras de teste da espécie, por seed,
comparação pareada honesta.

**Dois portões, não um**: `local` só é recomendado quando (a) a espécie
tem amostras ≥ `MIN_AMOSTRAS_LOCAL_PADRAO` — **mesmo limiar** já
estabelecido em `pipeline.pls_regression_by_species`
(`min_amostras_adult=6`), confirmado por teste que lê o default via
`inspect.signature` em vez de duplicar o número — E (b) o portão aprova
o ganho com poder estatístico suficiente. Abaixo do limiar de amostras,
a recomendação já sai `"pooled"` sem nem rodar o portão (economiza
custo, decisão já está definida pelo limiar).

**Validado com cenário sintético que espelha o motivo REAL** já
documentado em `pls_regression_by_species` ("variação inter-espécies
domina o sinal de adulteração"): com um offset espectral por espécie
dominando a variância, pooled falha feio (RMSEP=2,28) e local resolve
(RMSEP=0,24, aprovado p=0,008) → recomendação `"local"`. Sem esse
confundimento, local não ajuda (RMSEP levemente pior, mais dados de
treino vencem) → recomendação `"pooled"`, conservadora. Duas espécies
no MESMO dataset podem receber recomendações diferentes — nunca
"local para tudo" de uma vez.

Registrado no model card (`resultados_io.append_politica_pooled_local_
model_card`, addendum "Bloco 26") — lista toda espécie avaliada,
inclusive as com dados insuficientes, nunca só as que foram para local.

7 testes novos. Suíte completa (1335 passed, 19 skipped — +7 vs. Passo
138). Contrato de API pública regravado. Ruff/mypy limpos, módulo
adicionado à allowlist do CI.

---

# PROGRESSO — Passo 138 (2026-09-04)

## Passo 138 — Amostragem ativa orientada por incerteza (Bloco 25)

Novo módulo `amostragem_ativa.py`: `priorizar_amostragem(ensemble,
alpha, erro_por_especie)` prioriza combinações espécie×adulterante por
"impacto esperado por sessão investida" — reaproveita
`identificacao.train_identification_ensemble` (nunca recalibra, só
consome o `ensemble` já calculado) e `conformal.n_minimum_for_alpha`.
Combinações já `VALIDATED` ficam com prioridade 0; entre as não
validadas, prioridade = 1/(1+sessões_faltantes) — quem está mais perto
de cruzar o limiar de validação vem primeiro. `erro_por_especie`
(opcional, RMSEP/1-bal.acc já medido em outro lugar do pipeline) dá
peso extra a espécies com erro pior.

**Escopo decidido conscientemente**: `applicability_domain` NÃO entra
diretamente — é uma propriedade por AMOSTRA NOVA, não um agregado
natural "onde investir a próxima coleta" por combinação; o sinal que
responde essa pergunta é cobertura estatística (sessões faltando), que
é o que o módulo usa. Documentado no docstring, não escondido.

**Achado durante a verificação** (`ConformalOneClass._colapsar_por_
grupo` reduz o escore a UM POR SESSÃO antes de checar `achievable_
alpha`): `cobertura_status` depende SÓ do número de sessões, nunca do
valor dos escores PCA/DD-SIMCA em si — permitiu escrever o teste de
sanidade sem precisar treinar PCA de verdade.

**Teste de sanidade contra o acervo real**
(`scripts/medicoes/amostragem_ativa_oleos_reais.py`): confirma
exatamente o achado já documentado — **38/38 combinações não-validadas**
a alpha=0,05 (nenhuma chega a `n_minimum_for_alpha(0,05)=19` sessões);
36 têm exatamente 1 sessão, 2 têm exatamente 2 (Andiroba/soja,
Maracujá/algodão — batendo exatamente com `docs/MANUAL.md`). A lista
priorizada corretamente ranqueia essas 2 acima das 36 de 1 sessão
(prioridade 0,0556 vs. 0,0526). Nenhuma retratação necessária — a
alegação "36 de 38 têm só 1 sessão" já era precisa; a checagem só
confirma que nenhuma das 38 está de fato *validada* a 0,05, o que os
textos anteriores nunca alegaram.

**Integração ao `guaraci plan`**: `_menu_plan` (TUI, não há subcomando
`guaraci plan` de linha de comando — só o menu interativo, confirmado
antes de implementar) ganha um passo opcional no final ("Refinar com
amostragem ativa?") — carrega um `.joblib` já treinado (mesmo aviso de
segurança de pickle já usado no menu de Predição, não um mecanismo
novo) e mostra a lista priorizada numa tabela Rich.

**Achado corrigido durante a verificação**: minha primeira versão do
refinamento quebrava `test_menu_plan_cli_end_to_end_conformal` (a nova
pergunta consumia o input que o teste reservava para o `_pause()`
final) — pego pela suíte antes do commit, corrigido adicionando 1 input
("n") à sequência do teste existente.

12 testes novos (8 em `test_amostragem_ativa.py`, 4 em
`test_plano_coleta.py`). Suíte completa (1327 passed, 19 skipped — +12
vs. Passo 137; 1 falha em `test_regressao_pooled_com_benchmark_ligado_
roda_sem_erro` numa rodada intermediária, confirmada como flakiness
pré-existente não relacionada a este diff — passa isolado e a suíte
inteira repetida ficou 100% verde). Contrato de API pública regravado
(módulo novo). Ruff/mypy limpos, `amostragem_ativa.py` adicionado à
lista de módulos puros do CI.

---

# PROGRESSO — Passo 137 (2026-09-04)

## Passo 137 — Faixa de decisão (Bloco 24)

Todo resultado de quantificação passa a ser categorizado em 3 estados
usando os **mesmos** limiares de LOD/LOQ já validados no Bloco 12
(`chemometric_stats.regression_figures_of_merit`, nunca recalculados):
`nao_detectavel` (< LOD), `zona_cinzenta` (LOD ≤ x < LOQ — detecção
possível, quantificação não confiável), `quantificado_com_confianca`
(≥ LOQ). Nova função pura `chemometric_stats.faixa_decisao(valor, lod,
loq)`; `None` (nunca "não detectável" por omissão) quando LOD/LOQ não são
computáveis (sem réplicas físicas suficientes) — categorizar contra um
limiar inexistente fabricaria confiança sem lastro.

**Wiring**: LOD/LOQ (já calculados em `pls_regression_by_species`, só
não persistidos até agora) passam a viajar dentro do pacote `.joblib`
por espécie (`pipelines_especie[esp]["lod"/"loq"]`) — sem isso,
`quantify_sample` (que só recebe UMA amostra nova, sem acesso ao dado de
calibração/réplicas) não teria como categorizar. `QuantificationResult`
ganha `faixa_decisao`/`lod`/`loq`; pacotes salvos ANTES deste passo
simplesmente não têm essas chaves (`.get()`, sem exceção, faixa fica
`None`).

**Exibição proeminente** (CLI, não nota de rodapé): coluna
`faixa_decisao`/`lod`/`loq` no CSV de saída do fluxo cego (ao lado de
`teor_estimado`, `guaraci.py`) + linha "📏 Faixa de decisão" no painel
resumo do terminal, com contagem por estado (não detectável / zona
cinzenta / quantificado com confiança), só aparece quando há pelo menos
1 amostra quantificada.

9 testes novos (4 em `test_pipeline_core.py`, 5 em `test_predicao.py`).
Suíte completa (1314 passed, 19 skipped — +9 vs. Passo 136). Contrato de
API pública regravado (4 nomes novos em `chemometric_stats.__all__`, 3
campos novos em `QuantificationResult`, ambos aditivos/retrocompatíveis).

---

# PROGRESSO — Passo 136 (2026-09-04)

## Passo 136 — Aviso de escopo permanente do MCR-ALS (Bloco 23)

Aviso explícito adicionado em 3 lugares (nenhum novo mecanismo de
exibição — reaproveita superfícies já existentes):

1. **Docstring do módulo** (`mcr_als.py`, seção "AVISO DE ESCOPO"): texto
   completo com os números reais do Passo 131 (nenhuma correlação
   detectável em nenhuma das duas combinações testadas, mesmo onde
   PLS-R supervisionado funciona bem).
2. **`technique_registry.py`** (campo `limitacao` da entrada `mcr_als`):
   mesmo aviso, versão resumida — aparece em `[4] Técnicas disponíveis`
   do assistente `G`.
3. **Fluxo de decisão do assistente `G`** (`_FLUXO_DECISAO`, opção
   "resolver mistura"): novo campo `aviso` (dict PT/EN), exibido de
   forma proeminente (estilo `[warn]`, com ⚠) sempre que o usuário
   seleciona essa opção — é exatamente o ponto de entrada onde alguém
   com objetivo de QUANTIFICAR um traço poderia ser levado ao MCR-ALS
   por engano. Redireciona explicitamente para "[3] Quantificar teor
   (PLS-R)" quando o objetivo real é um número com garantia.

**Experimento opcional (MCR-ALS como feature extra do PLS-R)**: testado
via o portão (Bloco 20) se concatenar as proporções resolvidas pelo
MCR-ALS como variáveis de entrada EXTRA (não como preditor direto)
ajuda o PLS-R supervisionado — `scripts/medicoes/portao_mcr_als_como_
feature.py`, Babaçu+milho (a combinação com sinal supervisionado forte
do Passo 131). **Resultado: NEUTRO** (RMSEP sem=1,42 com=1,42, p=0,375)
— os scores por seed são quase idênticos com/sem as 2 colunas extras
(PLS aparentemente já ignora as variáveis não-informativas, mesmo
padrão de robustez a ruído não-correlacionado já visto no Bloco 20).
Tentativa negativa honesta, documentada — **não implementado** como
feature padrão, conforme a própria instrução pedia para esse caso.

4 testes novos (`tests/test_assistente_guaraci.py`). Suíte completa
executada junto com o Passo 135 acima (mesmo lote). Ruff/mypy limpos.

---

# PROGRESSO — Passo 135 (2026-09-04)

## Passo 135 — Portão aplicado ao PDS/DS (Bloco 22) — PDS confirmado, DS retratado

Reaplicação formal do portão de aceite (Bloco 20) à transferência de
calibração PDS/DS no Corn real, exigida pela própria regra de pausa da
instrução ("se reaplicar o portão ao PDS do Corn não confirmar o
resultado já conhecido, é achado grave"). Script:
`scripts/medicoes/portao_pds_ds_corn.py`; teste permanente:
`tests/test_validacao_publica.py::test_portao_correcao_sinal_reproduz_
pds_e_RETRATA_ds_no_corn`.

**PDS: confirmado.** 10 seeds independentes (não só o seed=0 fixo do
teste original) — aprovado, RMSEP médio 0,91→0,18, chega abaixo de 0,25
como esperado. O portão reproduz o resultado já conhecido — mecanismo
validado antes de confiar nele em EMSC/OSC (Passo 134) ou casos novos.

**DS: retratação, não confirmação.** A nota do Passo 86 ("DS não
reduziu o erro de forma relevante") foi medida contra 1 único split.
20 seeds independentes mostram o oposto: **DS ajuda de verdade**
(RMSEP médio 0,88→0,50, p<0,001, vence em 16/20 seeds) — só que muito
mais fraco e inconsistente que PDS (que chega a ~0,15-0,22 sempre; DS
fica em ~0,44-0,58). Nota original corrigida em linha no Passo 86,
acima. Regra que fica, explícita no teste: nunca generalizar "PDS
sempre funciona"/"DS nunca funciona" — o veredito é por par de
instrumentos/dataset.

Suíte completa não re-executada isoladamente neste passo (mesmo lote do
Passo 134, ver abaixo).

---

## Passo 134 — Portão aplicado a EMSC/OSC (Bloco 21)

EMSC e OSC (implementados no Bloco 16, testados só estruturalmente até
aqui) passam pelo portão de aceite (Bloco 20) contra dois cenários reais:
acervo privado de óleo (quantificação pooled de teor de adulterante,
todas as espécies/adulterantes juntos, split group-aware por `mae_id`,
1633 amostras, 549 grupos) e Corn público (proteína, m5).
Script: `scripts/medicoes/portao_emsc_osc.py`; teste permanente do
cenário Corn: `tests/test_validacao_publica.py::
test_portao_correcao_sinal_aprova_emsc_e_osc_no_corn`.

| Técnica | Cenário | Veredito |
|---|---|---|
| EMSC | óleo pooled | ✅ aprovado (4,70→4,39 RMSEP, p=0,002) |
| OSC | óleo pooled | ❌ rejeitado (4,70→4,99 RMSEP — PIOROU, p=0,002) |
| EMSC | Corn/m5 | ✅ aprovado (0,164→0,132, p=0,002) |
| OSC | Corn/m5 | ✅ aprovado (0,164→0,145, p=0,002) |

**Resultado misto, honesto**: EMSC ajuda nos dois cenários; OSC ajuda no
Corn mas **piora** no óleo. Nenhuma das duas é "sempre boa" — exatamente
o problema estrutural que o Bloco 20 existe para impedir (recomendar sem
prova por cenário). Veredito exibido **na docstring das duas classes**
(`preprocessamento.EMSC`/`OSC`) — EMSC/OSC nunca foram expostas em
menu/CLI (mesmo padrão de `apply_snv`/`apply_sg`/`apply_mc`, campos
internos do `Config`, ver Passo 127), então não há um "ao lado da
opção" de UI interativa para anexar; a docstring é a superfície real
que existe hoje. Se/quando EMSC/OSC ganharem exposição de menu (fora do
escopo deste bloco), o veredito deve migrar pra lá também.

`docs/VALIDACAO_PUBLICA.md` §9 registra a parte pública (Corn); o
resultado do óleo fica só aqui (política da própria página, ver seu
cabeçalho).

Ruff limpo nos scripts novos. Suíte completa (roda com Passo 135 abaixo,
mesmo lote): 1299 passed, 23 skipped — +2 vs. Passo 133 (os 2 testes
novos gated por Corn, que passam quando `GUARACI_DATASETS_DIR` aponta
pro corn.mat baixado nesta sessão; sem ele, pulam como sempre).

---

# PROGRESSO — Passo 133 (2026-09-04)

## Passo 133 — Portão de aceite automático para correção de sinal (Bloco 20)

Novo módulo `portao_correcao_sinal.py`: `avaliar_correcao_sinal(nome,
avaliar_sem_fn, avaliar_com_fn, metrica, ...)` roda um pipeline com/sem
uma técnica de correção sob o MESMO split group-aware bloqueado, repetido
em `n_seeds` (default 10) partições independentes, e decide via Wilcoxon
pareado (mesmo método já validado na comparação N-PLS vs. PLS-DA por
pixel, Passo 132): `aprovado` (p<0,05 e ganho), `rejeitado` (p<0,05 e
piora), `neutro` (sem diferença significativa) — sempre com
`poder_suficiente` (n≥8 pares) reportado junto, nunca escondido.
`avaliar_correcao_sinal_pls` é o atalho para o caso comum (alternar um
transformer sklearn dentro de um Pipeline PLS-R/PLS-DA fixo, exatamente o
caso de EMSC/OSC/PDS/DS dos próximos blocos).

**Contra-prova exigida pelo bloco — achado real, não assumido**: tentei
3 desenhos diferentes de "correção com ganho sintético claro" usando
ganho multiplicativo + MSC/correção-oráculo (removendo o ganho exato,
por construção) — **as três pioraram o RMSEP** em vez de melhorar,
mesmo com conhecimento perfeito do ganho verdadeiro. Motivo, confirmado
por medição direta: PLS é SUPERVISIONADO — sua própria definição
(maximizar covariância com y) já ignora uma direção de ruído
não-correlacionada com y, sem precisar de pré-processamento; qualquer
correção adicional só soma variância de estimação sem ganho real
disponível para capturar. O cenário "aprovado" que de fato funciona usa
ruído gaussiano de alta frequência dominando um sinal fraco + suavização
Savitzky-Golay (reduz SNR de qualquer entrada, ajuda mesmo modelo
supervisionado) — 1,44→1,25 RMSEP, aprovado com p=0,008. Achado
registrado no docstring do teste, não escondido: **a intuição de que
"remover um ganho conhecido sempre ajuda" é falsa para PLS
supervisionado** — prenuncia o que o Bloco 21 vai medir de verdade para
EMSC/OSC (pode muito bem sair neutro/rejeitado).

Ruído aleatório como correção: rejeitado corretamente (RMSEP piora,
p<0,05). Identidade (mesmos valores dos dois lados): neutro corretamente
(p=1,0 por definição matemática — não "rejeitado", que seria incorreto:
não há diferença nenhuma para rejeitar).

Veredito registrado no model card (`resultados_io.append_correcao_sinal_
model_card`, addendum "Bloco 20") — lista TODO veredito recebido, nunca
filtra só os aprovados, mesmo padrão append-only das demais funções de
model card já existentes (`append_linearity_robustness_model_card` etc.).

**Achado colateral, corrigido nesta sessão**: `gh run list` mostrou o job
`typecheck` do CI falhando (exit 126) em **todo push desde antes desta
sessão** — faltava um `\` de continuação de linha entre
`model_registry.py` e `technique_registry.py` na lista de módulos puros
do workflow, fazendo o shell tentar EXECUTAR `technique_registry.py`
como comando. Isolado (só esse job falhava; suíte completa/lint/
validação pública sempre verdes) mas real — o type-check não rodava de
verdade nesse período. Corrigido, e os 4 módulos puros criados nas duas
últimas sessões (`mcr_als.py`, `hsi_multiway.py`, `importadores_
proprietarios.py`, `portao_correcao_sinal.py`) adicionados à lista —
nenhum estava lá antes.

14 testes (`tests/test_portao_correcao_sinal.py`). Suíte completa (1297
passed, 23 skipped — +14 vs. Passo 132). Contrato de API pública
regravado (módulo novo + 1 nome novo em `resultados_io.__all__`).

---

# PROGRESSO — Passo 132 (2026-09-04)

## Passo 132 — N-PLS vs. PLS-DA por pixel em dado público real (fecha a pendência do Passo 129)

**Retratação do Passo 129**: o empate 1,0 vs. 1,0 em dado sintético
"bem separável de propósito" foi aceito como resultado ali — correto
apontar que não discriminava nada. Corrigido aqui com dado público
real: DeepHS Fruit, **Mango/VIS** (56 gravações, 36 objetos físicos, 3
classes de `ripeness_state`, 224 bandas).

**Achado ao tentar baixar** (`scripts/download_datasets/
baixar_deephs_fruit_todas.py`, a mesma ferramenta cogitada no Passo
129): está **quebrada para qualquer fruta além de Kaki** — depende de
`_deephs_fruit_todas_pins.json`, que a própria docstring do script
descreve como "versionado junto com este script", mas nunca foi de fato
commitado (confirmado por `git ls-files` + `git check-ignore`: nem
rastreado, nem ignorado — só ausente). Corrigir esse gap está fora do
escopo deste passo (achado registrado, não corrigido em silêncio);
contornado baixando Mango/VIS com pins gerados nesta sessão (TOFU —
sha256 do que foi baixado agora, **não** auditado externamente como os
pins de Kaki). Conectividade de rede funciona neste ambiente (não era o
bloqueio real do Passo 129) — Mango/VIS: 56 gravações, ~197MB via HTTP
Range (bem abaixo do "~44GB só pra Kiwi" que motivou desistir no Passo
129 — Mango é a menor das 4 frutas extras, escolha deliberada).

**Comparação** (`scripts/medicoes/comparar_npls_pixelwise_mango.py`,
mesmo split group-aware por objeto físico para os dois métodos, 10
seeds independentes):

| Método | balanced_accuracy (média±desvio, 10 seeds) |
|---|---|
| N-PLS | 0,389 ± 0,044 |
| PLS-DA por pixel | 0,375 ± 0,050 |

N-PLS venceu em 6/10 seeds. **Teste de Wilcoxon pareado: p=0,625 — sem
diferença estatisticamente significativa.** Resultado honesto conforme
a própria instrução previu como possível: os dois métodos são
estatisticamente equivalentes nesta tarefa, não um "vencedor". Ambos
ficam pertinho do nível de chance para 3 classes (0,333) — a tarefa
(ripeness de manga por HSI) é genuinamente difícil, consistente com o
desempenho modesto já registrado em `docs/VALIDACAO_PUBLICA.md` para
Kaki/VIS (5/8 objetos corretos), não um sinal de bug num dos dois
métodos.

**Nenhuma preferência prática recomendada** a partir deste resultado —
seria forçar uma conclusão que a evidência (p=0,625) não sustenta.

Suíte completa não re-executada neste passo (script de medição novo,
mesmo padrão de `scripts/medicoes/`, não reproduzível em CI por
depender de dataset público baixado sob demanda). Ruff limpo no script
novo.

---

# PROGRESSO — Passo 131 (2026-09-04)

## Passo 131 — MCR-ALS validado contra dado REAL do acervo privado (fecha a pendência do Passo 125)

**Diagnóstico do "bloqueio" reportado no Passo 125**: era caminho errado
meu, não ausência de dado. Eu tinha verificado só a pasta `dados/` do
repo (vazia de propósito — `.gitignore`, dado de terceiro nunca
versionado) e concluído "acervo indisponível" sem checar
`config.yaml` (`pasta_dados:`), que já apontava para
`C:\Users\erley\OneDrive\Documentos\ERLEY\dados oleos\Por óleos` — 1741
arquivos `.dx` reais, organizados por espécie, com adulterante e teor
declarado no nome (padrão `COD-DD-MM-AAAA-AD-X-teor%-T_N.dx`, ex.
`AND-10-02-2099-AD-S-1,1%-T_1.dx`), o mesmo acervo que
`scripts/run_benchmark_tcc.py` já referencia diretamente. Confirmado com
`load_dx` real: 1672 amostras carregadas, 13 espécies, 35 combinações
espécie×adulterante com 15 níveis de teor declarado cada (1%-15%) + puros.
Retratação: o Passo 125 registrou isso como "limitação de ambiente" —
não era; é achado corrigido aqui.

**Validação real** (`scripts/medicoes/validar_mcr_als_oleos_reais.py`,
janela 4550-9000 cm⁻¹ — mesma de `config.yaml`, fora dela há saturação de
detector; SEM SNV/MSC/SG, só clip de ruído residual <0 a zero — MCR-ALS
pressupõe mistura aditiva/bilinear na absorbância bruta, pré-processamento
padrão do pipeline quebraria essa premissa). Duas combinações
deliberadamente contrastantes (nunca só a "melhor caso" vista de trás pra
frente): Andiroba+algodão (pior sinal supervisionado num sanity-check PLS-R
prévio, Q2=-0,29) e Babaçu+milho (melhor, Q2=0,82):

| Combinação | lof% MCR-ALS | melhor \|r\| c/ teor declarado | R² calibrado | RMSE | desvio rotacional |
|---|---|---|---|---|---|
| Andiroba+algodão | 2,58% | 0,173 (p=0,24) | 0,030 | 4,54pp | 0,283 |
| Babaçu+milho | 1,80% | 0,094 (p=0,53) | 0,009 | 4,60pp | 0,218 |

**Resultado honesto**: MCR-ALS (2 componentes) **não recupera** um
componente que rastreie o teor declarado de adulterante em nenhuma das
duas combinações — mesmo na combinação onde PLS-R **supervisionado**
consegue (Q2=0,82), a correlação do MCR-ALS não-supervisionado com o
alvo é estatisticamente não-significativa (p>0,24 em ambas). Testado
também com 3-4 componentes (Babaçu+milho): melhor \|r\| sobe de 0,17 a
só 0,26, ainda fraco, com componente degenerado em k=4 — não é questão
de sub-parametrização. Interpretação (não um bug de implementação — lof%
baixo confirma que a reconstrução em si é boa): MCR-ALS é
NÃO-SUPERVISIONADO, otimiza fidelidade de reconstrução (variância total
explicada), não correlação com uma variável externa que nunca vê — numa
mistura real onde o adulterante é um componente MINORITÁRIO (1-15%) em
meio a ruído/variação instrumental de replicagem, a maior fonte de
variância espectral não é necessariamente o sinal do adulterante. A alta
sensibilidade à inicialização (desvio rotacional 0,22-0,28, bem acima do
limiar "baixo" <0,15 usado no teste sintético do Passo 125) reforça que a
solução não está bem restringida pelos dados reais nesse regime.
Limitação genuína do método nesse tipo de mistura real, documentada como
tal — não escondida atrás de "implementado e testado".

`docs/VALIDACAO_PUBLICA.md` **não foi tocado**: sua própria política
("validado exclusivamente em datasets públicos... nenhuma métrica desta
página vem de dado privado", linha 1-5) proíbe explicitamente registrar
resultado de dado privado ali — este resultado é só do acervo privado do
PIBIC, então fica registrado aqui no PROGRESSO.md (mesmo tratamento já
dado à rodada TCC 2026-07-10).

Suíte completa não re-executada neste passo (nenhum código de produção
mudou, só o script de medição novo em `scripts/medicoes/`, mesmo padrão
já estabelecido de `medir_selecao_variaveis.py` etc. — não versionado com
teste pytest próprio porque depende de caminho absoluto privado desta
máquina, não reproduzível em CI). Ruff/mypy limpos no script novo.

---

# PROGRESSO — Passo 130 (2026-09-04)

## Passo 130 — Fluxo de decisão do assistente `G` (Bloco 19, fecha a instrução de expansão técnica)

Novo item `[6] O que você precisa decidir?` no menu do assistente `G`
(`_abrir_assistente`, `guaraci.py`), aberto por `_guaraci_fluxo_decisao`:
pergunta o objetivo do usuário em vez de exigir que ele já saiba o nome
da técnica — 6 opções (`_FLUXO_DECISAO`): autenticidade, identificar
espécie, quantificar teor, adulterante desconhecido, transferência entre
instrumentos, resolver mistura. Cada opção mostra técnica(s) sugerida(s)
(puxadas de `technique_registry.REGISTRY`, nunca uma lista escrita a mão
— mesmo princípio anti-duplicação que motivou o catálogo no Agente 6),
pré-processamento default, tipo de validação e critério de aceitação; se
a opção tiver um nível associado, oferece (opt-in, pergunta antes) aplicar
`cfg.level`/`cfg.default_preprocessing` à sessão atual. **Modo avançado
preservado**: nenhum menu existente foi removido/alterado, o fluxo novo é
só mais uma entrada no assistente.

**MCR-ALS (Passo 125) registrado em `technique_registry.py`** (categoria
nova `resolucao_mistura`, id `mcr_als`) — sem isso o Bloco 19 não teria
como conectar "resolver mistura" a uma técnica real do catálogo.
CARS/UVE/EMSC/OSC/PARAFAC/N-PLS **deliberadamente não registrados**:
seguem o mesmo padrão já existente de VIP/SR/iPLS/SPA/AG/SNV/MSC (métodos
de seleção/pré-processamento internos, nunca estiveram no catálogo — só
métodos de decisão de nível superior como PLS-DA/DD-SIMCA/PLS-R estão).

**Achado durante a implementação**: `_guaraci_tecnicas` (`[4]` do
assistente) mapeia categoria→rótulo bilíngue via um dict local
(`rotulos_categoria`) que **descarta silenciosamente** qualquer categoria
do `REGISTRY` sem entrada nesse dict — uma categoria nova cadastrada sem
atualizar esse mapa nunca apareceria na tela, sem erro nenhum. Corrigido
para a categoria nova (`resolucao_mistura` adicionada ao dict); a
armadilha em si (categoria sem rótulo = invisível, não é uma propriedade
testada automaticamente) fica registrada aqui para a próxima categoria
nova não cair no mesmo buraco.

19 testes novos (`tests/test_assistente_guaraci.py`,
`tests/test_technique_registry.py`): dispatch `[6]`, cada opção roda sem
exceção, aplicar nível/pré-processamento muda `cfg` de verdade quando
confirmado (e não muda quando recusado), toda técnica referenciada existe
no `REGISTRY`, `resolver_mistura` está de fato conectada a `mcr_als`
(não só presente no catálogo sem uso).

Suíte completa (1283 passed, 23 skipped — +8 vs. Passo 129). Contrato de
API pública **inalterado** (mudança é só em `guaraci.py`, fora do escopo
mypy/API pública por design, e uma entrada de dado em
`technique_registry.REGISTRY`, não em `__all__`).

**Fecha a instrução de expansão técnica (Blocos 14-19)**: das 6
capacidades pedidas, MCR-ALS/CARS/UVE/EMSC-OSC/importador
OPUS/PARAFAC-N-PLS/fluxo de decisão — todas **implementadas e testadas**
com dado sintético. Ficou de fora por limitação real (não por escolha):
validação de MCR-ALS contra dataset real de óleo (dados/ vazio neste
checkout) e comparação N-PLS vs. PLS-DA por pixel em dataset público real
(DeepHS Fruit, Kiwi ~44GB na fonte, inviável baixar nesta sessão) — ambas
documentadas explicitamente nos Passos 125 e 129, não escondidas. O
GUARACI cobre agora a lacuna de composição de mistura (MCR-ALS) e
estrutura multiway do HSI (PARAFAC/N-PLS) que não existia antes deste
lote — com a ressalva honesta de que a validação contra dado público real
dessas duas capacidades específicas ainda está pendente.

---

# PROGRESSO — Passo 129 (2026-09-04)

## Passo 129 — PARAFAC e N-PLS multiway para HSI (Bloco 15)

Novo módulo `hsi_multiway.py`: decomposição multiway do cubo
hiperespectral. Referências verificadas no Crossref: Bro (1997) "PARAFAC.
Tutorial and applications", DOI 10.1016/S0169-7439(97)00032-4; Bro (1996)
"Multiway calibration. Multilinear PLS", DOI
10.1002/(SICI)1099-128X(199601)10:1<47::AID-CEM400>3.0.CO;2-C.

**Biblioteca avaliada antes de implementar PARAFAC do zero**: `tensorly`
(BSD, madura, dedicada a decomposição tensorial) — usada via
`tensorly.decomposition.parafac`, import lazy, novo extra `[multiway]`
em `pyproject.toml`. N-PLS não tem biblioteca madura equivalente em
Python (não está em `tensorly.regression`) — implementado seguindo o
NIPALS multiway do artigo original (truque de Kroonenberg: SVD do tensor
ponderado por `u` para achar o par de pesos rank-1 espacial/espectral a
cada componente).

**Problema de engenharia resolvido antes da decomposição em si**:
PARAFAC/N-PLS exigem um array N-way REGULAR, mas gravações HSI reais de
objetos físicos diferentes quase nunca têm a mesma resolução espacial
(achado do agente de exploração: Kaki 64×64, Avocado/VIS ~286×294).
`construir_tensor_amostras` resolve isso reduzindo a ROI de cada
gravação a uma grade espacial FIXA por média de bloco (não interpolação)
antes de empilhar — modo espacial fica com o mesmo significado relativo
entre objetos (célula [0,0] = canto superior-esquerdo da bounding box da
ROI em toda amostra).

**Group-aware confirmado por teste dedicado**
(`test_comparar_npls_vs_pixelwise_nunca_vaza_grupo_entre_treino_teste`):
`comparar_npls_vs_pixelwise` levanta `RuntimeError` internamente se
qualquer fold vazar `group_id` entre treino/teste — nunca silencioso.

**Comparação N-PLS vs. PLS-DA por pixel** (reaproveita
`hsi_classification.fit_predict_pixel_plsda`, mesmo split group-aware,
mesmo nível de agregação por objeto): em dado sintético bem separável
(estrutura de sinal forte de propósito, para confirmar que ambas as
implementações estão corretas), **os dois métodos empatam em
balanced_accuracy=1.0 em todos os folds** — resultado honesto, mas
inconclusivo por efeito-teto: o dataset sintético não discrimina qual
método generaliza melhor em situação real, ambíguo/difícil (só confirma
que ambos funcionam). A comparação que responderia "qual performa
melhor" de verdade — Kiwi/VIS + outra combinação do DeepHS Fruit, como
pedido — está **bloqueada**: o dataset público (Kiwi sozinho ~44GB na
fonte, ver `scripts/download_datasets/baixar_deephs_fruit_todas.py`) não
está baixado neste ambiente e não foi baixado nesta sessão (inviável no
tempo/rede disponível) — mesma classe de limitação já documentada no
Passo 125 (MCR-ALS x dataset de óleo).

12 testes (`tests/test_hsi_multiway.py`): grade regular com cubos de
tamanho diferente, reconstrução PARAFAC de tensor de baixo posto
sintético (erro<5%), N-PLS recupera estrutura supervisionada conhecida e
separa classes (acc>85%), `NPLSClassifier.predict` em dado NOVO
(transform sequencial com pesos do treino, não reajuste), propriedade
group-aware, comparação honesta.

`technique_registry.py` **não foi tocado** neste passo — registro de
PARAFAC/N-PLS (e de MCR-ALS/CARS/UVE dos passos anteriores) fica para o
Passo 130 (Bloco 19), quando o fluxo de decisão do assistente `G` de fato
consome esse catálogo — evita registrar entradas antes de terem
consumidor real.

Suíte completa (1275 passed, 23 skipped — +12 vs. Passo 128). Contrato
de API pública regravado (módulo novo, 6 nomes em `__all__`).

---

# PROGRESSO — Passo 128 (2026-09-04)

## Passo 128 — Importador OPUS (Bloco 18 da instrução de expansão técnica)

Novo módulo `importadores_proprietarios.py`: `parse_opus(filepath) ->
(X, Y)`, mesmo contrato de `dados_io.parse_dx`/`parse_spectrum` (dois
arrays 1D — eixo espectral, intensidade), lendo arquivos binários OPUS
(Bruker FT-NIR/FT-MIR).

**Biblioteca avaliada ANTES de escrever parser do zero** (pedido
explícito do bloco): comparadas `brukeropusreader` (mais antiga, GPLv3,
sem manutenção desde 2019) e `brukeropus` (Josh Duran) — escolhida a
segunda: release mais recente (2025-11-14), **MIT** (compatível com
GPL-3.0-or-later deste projeto — confirmado pelo classifier
`License :: OSI Approved :: MIT License` no pacote instalado, não só
pela página do PyPI), dependência única `numpy`. Import LAZY dentro de
`parse_opus` — pacote opcional (`pip install guaraci-chemometrics[opus]`,
novo extra em `pyproject.toml`), não força a dependência em quem nunca
abre arquivo OPUS.

**Escopo deliberadamente limitado**: converte UM arquivo por vez, não
generaliza a varredura de pasta de `dados_io.load_dx`
(`_detectar_subpastas_classe`/`_listar_arquivos_espectro`) — arquivos
OPUS usam extensão numérica por repetição de medida (`.0`, `.1`, `.2`…),
não um marcador de formato filtrável como `.dx`; generalizar a varredura
para esse padrão seria trabalho de escopo próprio e arriscaria código já
congelado (Bloco B) por uma extensão que não é pura adição. O bloco pediu
"converter para a estrutura já usada", não "generalizar a varredura".

**Pendência honesta** (documentada na própria docstring do módulo, não
escondida): nenhum arquivo OPUS binário real de teste estava disponível
neste ambiente. A lógica de extração/preferência de bloco (`a` >
`t` > `r` > `sm`, fallback pra primeira chave disponível) foi validada
contra um DOUBLE que reproduz exatamente a forma documentada e
confirmada por leitura do código-fonte da biblioteca instalada
(`brukeropus.file.data.Data.x`/`.y`, `OPUSFile.data_keys`/`.is_opus`) —
não contra um binário OPUS de verdade. Cobertura fim-a-fim com
instrumento real fica pendente até haver um arquivo de exemplo genuíno.

8 testes (`tests/test_importadores_proprietarios.py`): ausência do
pacote opcional dá `ImportError` claro; preferência de bloco (absorbância
> transmitância > fallback); arquivo não-OPUS e sem blocos de dados dão
`ValueError`; formas x/y inconsistentes detectadas.

`requirements.txt`/`requirements-lock.txt` ganham `brukeropus` (mesmo
motivo já documentado ali para `prcv`: sem isso o CI nunca exercita o
caminho real, só o de "pacote ausente") — **nota de honestidade**: o
lock foi atualizado por inserção manual de uma linha (`brukeropus==1.4.3`,
versão realmente instalada e usada nos testes desta sessão), não por
regeneração completa a partir de venv limpo (processo documentado no
cabeçalho do arquivo) — o resto do lock não foi reverificado do zero.

Suíte completa (1262 passed, 23 skipped — +8 vs. Passo 127). Contrato de
API pública regravado (1 nome novo em `importadores_proprietarios.__all__`,
módulo novo).

---

# PROGRESSO — Passo 127 (2026-09-04)

## Passo 127 — EMSC e OSC (Bloco 16 da instrução de expansão técnica)

`preprocessamento.py` ganha duas transformações sklearn-compatíveis
novas, `EMSC` e `OSC`, integradas ao mesmo leque configurável de
`apply_snv`/`apply_sg`/`apply_mc` (campos `Config` internos, só
efetivos com `default_preprocessing='custom'` — **não** expostos em
`config.yaml`/menu, mesmo precedente de `apply_snv`/`apply_sg`/
`apply_mc`, confirmado por `grep` antes de decidir: zero referência a
esses três em `cli_assistente.py`/`guaraci.py`/`app_tabs/`).

- **EMSC** (Martens & Stark, 1991, DOI 10.1016/0731-7085(91)80188-F):
  generaliza MSC — além do termo multiplicativo contra o espectro médio
  de referência, ajusta linha de base POLINOMIAL (ordem configurável) e,
  opcionalmente, espectros de interferentes conhecidos, numa única
  regressão por amostra. Sem `eixo` explícito, usa o índice do canal
  normalizado (suficiente para linha de base — forma polinomial não
  muda por reescala/deslocamento linear do eixo).
- **OSC** (Wold, Antti, Lindgren & Öhman, 1998, DOI
  10.1016/S0169-7439(98)00109-9): remove de X só a variação ORTOGONAL ao
  alvo `y` (NIPALS iterativo por componente). Ao contrário de
  SNV/MSC/SG, **exige** `y` em `fit` — dentro de um `Pipeline`,
  `fit(X, y)` já repassa `y` de treino a toda etapa que aceite, sem
  vazamento adicional ao que o resto do pipeline já evita.

Referências verificadas no Crossref em 2026-09-04.

**Validado** (`tests/test_emsc_osc.py`, 12 testes): EMSC produz espectro
numericamente diferente de MSC/SNV e estável; com `ordem_polinomial=0`
sem interferentes reduz ao MSC (mesma base de regressão, `[1, ref]`);
com linha de base linear sintética conhecida, EMSC(ordem 1) recupera o
espectro puro com erro menor que MSC (caso construído para favorecer
EMSC, não uma alegação geral de superioridade). OSC: produz resultado
diferente de centrar-só, reduz variância total sem destruir a separação
de classe (PLS pós-OSC ainda classifica bem), `fit` sem `y` lança
`TypeError` (assinatura exige), transform em dado novo usa os pesos do
treino. Integração fim-a-fim de `build_preprocessor` com `apply_osc=True`
dentro de um `Pipeline` + `PLSRegression` completo.

Suíte completa (1253 passed, 23 skipped — +11 vs. Passo 126). Contrato
de API pública regravado (2 nomes novos em `__all__` de
`preprocessamento`, 4 campos novos no `Config`).

---

# PROGRESSO — Passo 126 (2026-09-04)

## Passo 126 — CARS e UVE (Bloco 17 da instrução de expansão técnica)

`selecao_variaveis.py` ganha dois métodos novos de seleção de variáveis,
opt-in (`cfg.run_cars`/`cfg.run_uve`, default `False`, mesmo motivo do
SPA/AG — mais avaliações de CV que iPLS/VIP/SR/sPLS-DA):

- **CARS** (Li, Liang, Xu & Cao 2009, DOI 10.1016/j.aca.2009.06.046):
  amostragem Monte Carlo + função exponencialmente decrescente (EDF) +
  Adaptive Reweighted Sampling (roleta ponderada por |coeficiente|, não
  corte duro por ranking). Adaptado de RMSECV (regressão univariada no
  artigo original) para balanced_accuracy via CV, mesma adaptação já
  usada pelo AG/SPA deste módulo (PLS-DA multi-classe, não regressão).
- **UVE** (Centner et al. 1996, DOI 10.1021/ac960321m): concatena
  variáveis de ruído artificial às reais, mede estabilidade do
  coeficiente PLS (média/desvio entre repetições Monte Carlo) e elimina
  variáveis reais indistinguíveis do ruído.

Ambas as referências verificadas no Crossref em 2026-09-04.

**Nested-CV garantido por reuso**: em vez de mecanismo novo, CARS entra
em `_avaliar_busca_nested_cv` (mesmo arcabouço já usado e testado por
AG/SPA) e UVE em `_avaliar_subset_nested_cv` (mesmo de VIP/SR/iPLS) — a
seleção é sempre refeita usando só as amostras de treino de cada fold
externo, nunca vê o fold de validação. Confirmado por teste de
propriedade dedicado (`test_cars_nested_cv_nunca_ve_o_fold_de_validacao`,
`test_uve_nested_cv_nunca_ve_o_fold_de_validacao`): um espião registra o
tamanho de X recebido pela seleção em cada fold e confirma que bate
exatamente com `len(treino)`, nunca com o dataset inteiro.

**Estabilidade entre repetições** (`estabilidade_selecao_entre_repeticoes`,
Jaccard pareado entre execuções com seeds diferentes): medido e
confirmado no teste `test_estabilidade_cars_uve_menor_que_vip_deterministico`
— VIP é perfeitamente estável (Jaccard=1.0, determinístico, mesmos dados
sempre produzem a mesma máscara) enquanto CARS/UVE têm Jaccard<1.0 (usam
amostragem Monte Carlo, esperado). Achado honesto, não um defeito: é o
preço de usar amostragem estocástica para robustez a colinearidade, que
VIP/SR/iPLS (determinísticos, ou quase) não pagam.

Testado em `tests/test_cars_uve.py` (10 testes: EDF, recall de variáveis
informativas em dataset sintético group-aware, corte do UVE em dataset
100% ruído, propriedade de nested-CV, comparação de estabilidade) +
integração de ponta a ponta de `etapa4_selecao_variaveis` com
`run_cars=True, run_uve=True` verificada manualmente (CSV de iterações do
CARS, tabela final, figura comparativa — sem erro).

Wiring de interface (achado durante a suíte completa, mesma classe dos
achados de 2026-08-06 documentados em `guaraci.py`): `selecao_cars`/
`selecao_uve` precisaram ser adicionados em 4 lugares além do
`_CONFIG_SPEC` para ficarem de fato editáveis/visíveis — `RISK_CLASS`,
rótulos PT/EN e texto de ajuda (`cli_assistente.py`), `MENU_FIELDS`
(`cli_assistente.py`), o menu real do CLI interativo
(`_menu_modeling` em `guaraci.py`) e `_MODELO_KEYS_EXTRAS` (app web,
`app_tabs/modelo.py`) — pego pelos 4 testes de cobertura de interface já
existentes (`test_contrato_api_publica.py` cobre schema, os outros 3
cobrem alcançabilidade nos menus).

Suíte completa (1242 passed, 23 skipped — +11 vs. Passo 125). Contrato
de API pública regravado intencionalmente (`config`/`guaraci`/
`selecao_variaveis`/schema do config.yaml mudaram: 2 funções + 1 helper
novos em `__all__`, 6 campos novos no `Config`, 2 chaves novas no
`config.yaml`).

---

# PROGRESSO — Passo 125 (2026-09-04)

## Passo 125 — MCR-ALS (Bloco 14 da instrução de expansão técnica)

Novo módulo `mcr_als.py`: Resolução de Curvas Multivariada por Mínimos
Quadrados Alternados. Referência verificada no Crossref: Tauler, R.
(1995), *Chemometrics and Intelligent Laboratory Systems* 30(1):133-146,
DOI 10.1016/0169-7439(95)00047-X; restrições revisadas em Tauler & de Juan
(2006), DOI 10.1201/9781420018301.ch11.

**O que faz.** Decompõe uma matriz de espectros de mistura `D` em perfis
de concentração `C` e perfis espectrais puros `S`, com restrições de
não-negatividade (ambos, default ligado), normalização configurável
(soma unitária / norma unitária) e unimodalidade opcional em `C` (só faz
sentido quando a ordem das amostras é significativa — off por default).
API pública: `mcr_als`, `MCRALSResultado`, `avaliar_incerteza_rotacional`.

**Ambiguidade rotacional** (limitação conhecida do método, não bug):
`mcr_als` nunca reporta `(C, S)` como solução única — todo resultado
carrega `aviso_ambiguidade_rotacional`. `avaliar_incerteza_rotacional`
roda múltiplas inicializações aleatórias, alinha componentes entre
execuções por correlação máxima (assignment ótimo via
`scipy.optimize.linear_sum_assignment`, necessário porque o rótulo dos
componentes não é preservado entre execuções independentes) e reporta o
desvio-padrão das proporções recuperadas como proxy de sensibilidade —
explicitamente **não** é o cálculo formal de banda de ambiguidade
(MCR-BANDS, Jaumot & Tauler 2010), documentado como tal no próprio aviso
de retorno.

**Validação**: mistura sintética de 3 espectros puros (gaussianos bem
separados) combinados em proporções `Dirichlet` conhecidas —
`tests/test_mcr_als.py`, 10 testes. Achado durante a validação: o
critério inicial de lack-of-fit (`<5%` fixo) estava errado — o "piso de
ruído" da própria mistura sintética (LOF contra os parâmetros
VERDADEIROS, não os ajustados) já fica em ~7% porque o sinal é pequeno
na maior parte dos canais espectrais (só é grande perto dos picos); o
teste foi corrigido para comparar o LOF ajustado contra esse piso
calculado a partir do próprio dataset, não contra um número arbitrário.
Dois bugs reais de implementação pegos pelos testes antes do commit:
troca de ordem de argumentos em `_normalizar_S` (S/C invertidos na
chamada) e uma transposição a mais na atribuição inicial de `S` — ambos
None dos testes de reconstrução falhavam sem eles.

**Pendência honesta**: o checklist do Bloco 14 pede validação contra o
dataset real de óleo (misturas espécie+adulterante em teor declarado).
A pasta `dados/` deste checkout está vazia (dado de terceiro, nunca
versionado — ver `.gitignore`) — essa validação fica bloqueada até o
dado estar acessível neste ambiente, não foi pulada por escolha.

Suíte completa (1231 passed, 23 skipped — inalterado fora do novo
módulo), ruff/mypy limpos em `mcr_als.py`. Contrato de API pública
regravado intencionalmente (módulo novo, 3 nomes novos em `__all__`).

---

# PROGRESSO — Passo 124 (2026-09-03)

## Passo 124 — Lista de técnicas de imagem generalizada (fecha o ciclo de adaptabilidade)

Último achado pendente da auditoria do Passo 117:
`perfil_matriz.PERFIS_TECNICA` era um frozenset fixo de 3 nomes
(`bancada`/`celular`/`scanner`) usado só pra' filtrar
`perfis_disponiveis(apenas="tecnica"/"matriz")` — uma técnica nova
carregava e funcionava normalmente, mas nunca aparecia na listagem
filtrada por "técnica".

**Corrigido**: classificação agora é por CONTEÚDO
(`_e_perfil_tecnica`, novo — declara `resolucao_esperada`/
`formatos_aceitos`/`nivel_agrupamento_tipico`), não por nome de
arquivo contra uma lista fixa. `PERFIS_TECNICA` continua existindo,
mas só como registro dos 3 exemplos pré-cadastrados de conveniência —
não decide mais classificação nenhuma. Zero mudança de assinatura
pública (`perfis_disponiveis` continua igual por fora) — golden de
contrato de API confirmado sem alteração.

Teste que documentava o achado (Passo 117,
`test_achado_perfis_tecnica_e_lista_fixa_nao_generica`) INVERTIDO pra'
`test_perfis_tecnica_e_generico_por_conteudo_nao_por_nome`, confirmando
a correção: uma 4ª técnica (inventada na hora) aparece corretamente na
listagem filtrada, as 3 originais continuam lá (22/22 testes de
`test_perfil_matriz.py` sem mudança). O teste de aceite completo
(técnica fictícia rodando mode="imagem" ponta-a-ponta) já existia do
Passo 117 (`test_aceitacao_imagem_tecnica_ficticia_nova`) — reaproveitado,
não duplicado.

README.md/README.pt-br.md: "com 1/2 ressalvas honestas" vira **"sem
nenhuma ressalva conhecida"** — a alegação multimatriz/multitécnica
agora está inteiramente sustentada por teste. MANUAL.md §4b.2 documenta
a classificação por conteúdo.

Suite completa (1185 passed), ruff/mypy limpos. Commit, push. **Fecha
o ciclo de adaptabilidade aberto no Passo 117.**

---

# PROGRESSO — Passo 123 (2026-09-03)

## Passo 123 — Hipótese NIR registrada (especulativa, distinta da conclusão definitiva do Passo 121)

Achado do Passo 121: efeito por-banda de firmeza ~2,5× maior em Kiwi/NIR
que Kiwi/VIS -- registrado em `docs/VALIDACAO_PUBLICA.md` §7 como
hipótese explícita ("sinal de maturação mais concentrado fora do
visível"), com o texto exato pedido, claramente separada da conclusão
DEFINITIVA do Passo 121 (aquela é confirmada; esta é especulativa).

**Checagem em outra fruta** (Avocado, única outra opção com câmera NIR
no DeepHS Fruit): o padrão NÃO se repete — Avocado/NIR (mediana 0,301)
é MENOR que Avocado/VIS (mediana 0,548), razão 0,55× (oposto de Kiwi).
Conclusão honesta registrada: a hipótese é **específica do Kiwi neste
dataset, não um padrão geral** — ambas as amostras de NIR são pequenas
(n_unripe=6-7), nenhuma direção deve ser tratada como estabelecida.
Nenhuma alegação forte feita.

Sem mudança de código (só documentação). Commit, push.

---

# PROGRESSO — Passo 122 (2026-09-03)

## Passo 122 — Identificação generalizada: aceita qualquer convenção de nome

Achado do Passo 117: a Identificação (Bloco 9b) só produzia combinações
calibradas com o padrão de letra do dataset original de óleo (A/M/S).

**Diagnóstico**: problema de PARSING (mapa hardcoded), não de conceito
semântico. `identificacao.train_identification_ensemble` chamava
`dados_io.adulterant_from_mae_id(mae_id)`, que consulta um dicionário
GLOBAL `ADULTERANTE_NOME = {"A":"algodão","M":"milho","S":"soja"}` fixo
no módulo. A ESTRUTURA do token (`{cod}-{data}-{letra}{teor}`, 1 letra
+ dígitos no último segmento) já era genérica — usada por qualquer
matriz em mode `dx`/`sintetico` — só o mapa letra→nome não tinha como
ser trocado sem editar `dados_io.py`. `session_from_mae_id` (a outra
função envolvida) já era agnóstica à letra (só olha a estrutura), não
precisou mudar. `identify_sample` (predição em amostra nova) também já
era genérico (nunca re-deriva o adulterante, só casa contra o ensemble
já calibrado).

**Generalização implementada**: `MatrixProfile.codigos_adulterante`
(novo campo, mesmo padrão já usado por `codigos_classe`) — vazio
(default) preserva o mapa global `ADULTERANTE_NOME`. Repassado
explicitamente como `mapa_adulterante` por `train_identification_
ensemble`, `r2cv_species_by_adulterant` e `adulterant_from_mae_id`.
Sem mudança de schema do `.joblib` persistido — `identify_sample`
nunca chamava essas funções (só casa contra o ensemble já calibrado),
então nenhum modelo antigo muda de comportamento ao carregar.

**Testes** (`tests/test_identificacao_generica.py`): (1) dataset
sintético com `synthetic_adulterants=("X","Y")` — letras DIFERENTES de
A/M/S — e perfil fictício `codigos_adulterante={"X":"quitosana",
"Y":"amido"}`, `executar()` ponta-a-ponta, ensemble NÃO-VAZIO com os
nomes certos, Detectar→Identificar→Quantificar (`predict_blind`) roda
sem exceção; (2) contra-prova de retrocompatibilidade: dataset com
A/M/S e perfil SEM `codigos_adulterante` produz exatamente o mesmo
ensemble de antes (43 testes existentes de Identificação/modo cego/
heatmap confirmados sem mudança).

README.md/README.pt-br.md corrigidos: "duas ressalvas honestas" vira
"uma ressalva honesta restante" (só o `PERFIS_TECNICA` fixo do Passo
117 continua). MANUAL.md §4b.1 documenta `codigos_adulterante`.
`docs/COMPATIBILITY.md` não precisou de entrada nova (mudança aditiva,
golden de contrato regravado só com adições).

Suite completa (1185 passed), ruff/mypy limpos. Commit, push.

---

# PROGRESSO — Passo 121 (2026-09-03)

## Passo 121 — Hipótese D registrada como conclusão DEFINITIVA (fechada, não mais em aberto)

O achado só existia em resumo de conversa. Formalizado:

- Teste confirmado ja' commitado (`80483e5`,
  `test_hipotese_d_firmeza_objetiva_confirma_rotulo_nao_e_ruido`).
- `docs/VALIDACAO_PUBLICA.md` §7 ganhou a conclusão definitiva textual:
  a diferença física é real e robusta (Mann-Whitney p=8,87×10⁻⁸,
  Cohen's d=1,64); a câmera VIS especificamente não capta essa
  diferença -- limite de sensibilidade espectral da técnica, não ruído
  de rótulo. Não é mais "questão em aberto".
- README.md/README.pt-br.md ganham essa conclusão como 6ª evidência da
  seção de validação honesta (Passo 115).
- **Checagem adicional (não bloqueante, exploratória):** Kiwi também
  tem câmera NIR (58 gravações, n_unripe=7 -- amostra pequena). Efeito
  por-banda medido: mediana 0,968 (contra 0,376 em VIS) -- ~2,5× maior,
  fisicamente plausível (NIR sensível a umidade/estrutura celular). Mas
  a classificação por pixel em Kiwi/NIR TAMBÉM falhou pra' `unripe`
  (mesma tabela do Passo 104) -- com n=7, causa mais provável é tamanho
  de amostra, não falta de sinal. Evidência adicional (não confirmação
  completa) de que o limite é da câmera VIS especificamente. Novo teste
  `test_checagem_adicional_camera_nir_kiwi_efeito_por_banda`.

Suite completa (6/6 na investigação, incl. o teste novo), ruff limpo.
Commit, push.

---

# PROGRESSO — Passo 120 (2026-09-03)

## Passo 120 — README, MANUAL.md, CITATION.cff, paper.md atualizados por completo

Revisao de toda a escrita de alto nivel a luz do trabalho desta rodada
(Passos 111-119). Achado real: o bullet de HSI em README.md/README.
pt-br.md/MANUAL.md ainda descrevia "prototipo minimo viavel" exigindo
`manifest.json` -- STALE desde o Passo 111 (HSI ja aceita dado proprio
do usuario, offline).

- **README.md/README.pt-br.md**: bullet de HSI reescrito (aceita cubo
  proprio, offline provado por teste, dataset publico rebaixado a
  fixture); nova subsecao "Multimatrix and multitechnique by design —
  with two honest caveats" / "Multimatriz e multitécnica por design —
  com 2 ressalvas honestas" declarando explicitamente a alegacao
  multimatriz/multitecnica **com** os 2 achados do Passo 117
  (Identificacao amarrada a "adulterante"; PERFIS_TECNICA fixo) --
  nunca alegacao sem a ressalva ao lado. "Known limitations"/
  "Limitações conhecidas" ganham bullet de HSI atualizado (perfomance
  no fixture publico ainda modesta, matriz propria nao testada ate
  testar).
- **MANUAL.md**: secao HSI (§4) reescrita por completo (mesmo
  conteudo do README, mais detalhado); tabela de modos de entrada
  atualizada; **secao nova 4b.2** "Perfil de tecnica de aquisicao +
  perfil combinado" -- documentando pela PRIMEIRA VEZ o fluxo
  combinar/salvar (Agente 5B, ja implementado mas nunca documentado) +
  o achado do Passo 117 sobre eixo espectral herdado da matriz ao
  combinar com tecnica de imagem (renumerado 4b.2→4b.3→4b.4). Mapa de
  modulos (§7) ganha `agrupamento_pastas.py` e atualiza `hsi_io.py`/
  `hsi_validation.py`/`hsi_pipeline.py`.
- **CITATION.cff/paper.md**: abstract/summary/tags ganham "hyperspectral
  imaging"/"HSI"; paper.md (JOSS) ganha paragrafo no "State of the
  field" ligando o modo HSI diretamente ao gap ja apontado do
  `hyperSpec` (estrutura de dado sem camada de modelagem/validacao) --
  com a mesma ressalva de desempenho honesto do README. Versao
  (31.9.0) e data-released NAO alterados (diretriz permanente: nao
  mexer em versao antes de fechar as pendencias 3/6/7).
- Comparativo com concorrentes (`app_tabs/sobre.py`) ja revisado no
  Passo 115.

Suite completa (1183 passed), ruff limpo (unico arquivo de codigo
tocado, `sobre.py`, so' texto). Commit, push.

---

# PROGRESSO — Passo 119 (2026-09-03)

## Passo 119 — Inventario essencial/util/obsoleto/duplicado

Nao havia inventario anterior persistido em arquivo pra' "reexecutar"
(buscado em docs/*.md e *.md da raiz -- zero resultado) -- feito do
zero, por comando direto.

- **`pyproject.toml` packaging**: `[tool.setuptools.package-data]`
  cobre `perfis_matriz/*.yaml` -- nenhum modulo novo desta rodada
  (`agrupamento_pastas.py`, os `hsi_*` incrementais) precisa de entrada
  nova (sao `.py` dentro do pacote `guaraci` ja listado, incluidos
  automaticamente). Nenhuma pasta de dado nova foi criada. **Nada a
  corrigir.**
- **`scripts/medicoes/*.py` (10 arquivos)**: checado se cada um e'
  referenciado por NOME em docs/README -- 6 deram "0 referencias"
  (`medir_achados`, `medir_ad_vies_insample`, `medir_bug_progresso_cli`,
  `medir_permutacao_grupos`, `medir_sessoes_especie_adulterante`,
  `medir_sr_ranking`). Inspecionado cada um: **falso alarme** -- sao
  recibos de reprodutibilidade de achados/numeros ESPECIFICOS ja
  citados em docs (ex.: `medir_sessoes_especie_adulterante.py` produz
  o "36 de 38" citado em MANUAL.md/technique_registry.py) -- o NUMERO e'
  citado, nao o nome do arquivo. Nenhum e' obsoleto ou duplicado.
  **Nada a remover.**
- **2 scripts + 1 doc pessoais** (`scripts/gerar_relatorio_abnt.py`,
  `scripts/run_benchmark_tcc.py`, `docs/_AUDITORIA_ESTADO.md`):
  presentes no disco mas deliberadamente `.gitignore`d (padrao
  `docs/_*.md` e nome explicito por script) -- ja corretamente
  excluidos do pacote publicavel. **Nada a corrigir.**
- **`INSTRUCAO_*.md`** citados em varios docstrings (ex.
  `INSTRUCAO_HSI_MINIMO_VIAVEL.md`): confirmado que NUNCA foram
  arquivos reais -- e' a convencao do projeto de referenciar a
  instrucao de CHAT que motivou cada mudanca, consistente em toda a
  historia do codigo. Nao e' arquivo faltando.
- Working tree confirmado limpo (`git status --short`) ao final.

**Lista de remocao proposta: vazia.** Repositorio ja esta organizado
para publicacao nesta dimensao -- nenhum achado, nada apagado.

---

# PROGRESSO — Passo 115 (2026-09-03)

## Passo 115 — Secao de validacao externa honesta (README/app)

Reunidas 5 evidencias ja existentes (nunca antes apresentadas como
argumento coeso) numa secao nova, "Why the numbers here sometimes look
modest" / "Por que os números aqui às vezes parecem modestos", em
README.md e README.pt-br.md (entre "Validation"/"Validação" e
"Security"/"Segurança"):

1. RMSEP Mendeley nao reproduziu (R²val negativo) -- VALIDACAO_PUBLICA §2.
2. Q² negativo na Hipotese B do unripe -- VALIDACAO_PUBLICA §7, Passo 112.
3. DD-SIMCA platoa em ~0,94-0,945, nunca converge ao nominal -- MANUAL.md.
4. Identificacao nao-validavel em 36/38 combinacoes -- technique_registry.py.
5. Kiwi/VIS falha em unripe com n suficiente -- VALIDACAO_PUBLICA §7,
   Passos 112/114.

Framing: filosofia deliberada (holdout externo + grupo fisico protegido
+ cobertura formal), nao pedido de desculpas -- reporta desempenho mais
modesto quando e' o que a medicao honesta mostra, em vez de esconder
atras de uma particao mais favoravel.

Comparativo com concorrentes (`app_tabs/sobre.py`, aba "Sobre") revisado
a luz dessa secao: adicionada legenda ligando a linha "Validacao
anti-vazamento: Padrao" a essa filosofia (e corrigido o asterisco
"Pagos*" que ficava sem nota de rodape nenhuma).

Suite completa (1183 passed), ruff limpo. Commit, push.

---

# PROGRESSO — Passo 118 (2026-09-03)

## Passo 118 — Isolamento fisico de dataset de terceiro (P0)

Checagem P0 explicita, por comando direto -- resultado: **nenhum
achado grave**.

- `git ls-files` (arvore atual): zero arquivo com extensao de dado
  bruto de terceiro, zero arquivo > 512KB.
- `git rev-list --objects --all` + `cat-file --batch-check` (HISTORICO
  COMPLETO, nao so' a arvore atual): maior blob de todo o historico e'
  `guaraci_icon.png` (~2,7MB, icone legitimo) -- nenhum dataset publico
  jamais foi commitado, em nenhuma revisao. `.git` total = 24MB,
  consistente com isso (23GB do DeepHS Fruit deixaria rastro MUITO
  maior se tivesse passado por 1 commit sequer).
- Mecanismo unico confirmado: os 3 scripts de download usam o MESMO
  padrao (`GUARACI_DATASETS_DIR`, fallback `datasets_publicos/`, ja
  coberto pelo `.gitignore`) -- sem segundo mecanismo paralelo.
- Prova automatizada em `tests/test_isolamento_datasets.py` (roda
  sempre, nao gated -- checagem sobre o repositorio em si): 4 testes
  (arvore atual, historico completo, cobertura do .gitignore,
  consistencia entre scripts).
- `datasets/README.md` atualizado: tabela completa (faltava DeepHS
  Fruit/todas-as-frutas), linguagem explicita "e' e SERA" reforcada.
  `docs/VALIDACAO_PUBLICA.md` secao 8 documenta a auditoria.

Suite completa (1183 passed), ruff limpo. Commit, push.

---

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

  **RETRATAÇÃO (Passo 135, 2026-09-04)**: a frase acima vinha de checar
  um único split (seed=0). Reavaliado formalmente contra 20 seeds
  independentes pelo portão de aceite (Bloco 20): **DS ajuda de
  verdade** (RMSEP médio 0,88→0,50, p<0,001, vence em 16/20 seeds) — só
  que muito mais fraco e menos consistente que PDS. No split seed=0
  específico, DS por coincidência saiu ligeiramente pior (0,510→0,528),
  o que produziu a impressão errada de "não ajuda". Ver
  `docs/VALIDACAO_PUBLICA.md` §9 para a tabela completa.
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
