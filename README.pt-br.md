# GUARACI — Plataforma quimiométrica com validação anti-vazamento por padrão

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Licença: GPLv3" src="https://img.shields.io/badge/Licen%C3%A7a-GPLv3-3D8B57">
  <img alt="Licença comercial disponível" src="https://img.shields.io/badge/comercial-licen%C3%A7a%20dispon%C3%ADvel-B8963E">
  <img alt="Version" src="https://img.shields.io/badge/version-31.9.0-B8963E">
  <img alt="Interface" src="https://img.shields.io/badge/UI-Rich%20CLI%20%2B%20Streamlit-4A9E5C">
  <img alt="Idiomas" src="https://img.shields.io/badge/i18n-PT%20%2F%20EN-686868">
  <img alt="Status" src="https://img.shields.io/badge/status-ativo-55B06A">
</p>

> 🇧🇷 Versão em português (de trabalho). • 🇬🇧 [English version](README.md)

### 🚀 [Teste a demo ao vivo — sem instalar nada](https://guaraci.streamlit.app/)

[![Abrir no Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://guaraci.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErleySC/guaraci/blob/master/notebooks/guaraci_5_minutos.ipynb)

Uma alternativa **livre e aberta** aos softwares pagos de quimiometria
(MATLAB/PLS_Toolbox, The Unscrambler, SIMCA): plataforma reprodutível de
**quimiometria multitécnica** para **classificação, autenticação e exploração**
de matrizes complexas, com **validação robusta a vazamento de réplicas**
(*group-aware*) — o diferencial metodológico do projeto.

**Missão:** democratizar a quimiometria de alto nível — entregar a pesquisadores
o rigor de um software comercial, sem custo de licença e sem prender ninguém a
um formato fechado.

Suporta dados vibracionais (**FT-NIR, NIR, MIR, Raman, UV-Vis**), de luminescência
(**fluorescência**), cromatográficos (**HPLC, GC-MS**) e de ressonância (**NMR,
IMS**), por uma interface de terminal bilíngue (**GUARACI**) e um app Streamlit —
sem precisar programar.

**Agnóstico de matriz por construção.** O que é específico de uma matriz —
faixa espectral, pré-processamento padrão, unidade do eixo e o *vocabulário*
da saída — vive num **perfil de matriz** (um YAML), nunca no código-fonte.
Trocar de óleo vegetal para milho em grão é um campo de configuração, não
uma edição de código; uma matriz sem perfil cadastrado **falha com mensagem
acionável em vez de predizer**.

**Validado exclusivamente em datasets públicos.** Ver [Validação](#validação)
— é a única base de evidência sobre a qual este repositório faz afirmações.

---

## Por que este projeto existe

As ferramentas de quimiometria hoje se dividem em dois grupos:

| | scikit-learn puro | Unscrambler / SIMCA / PLS_Toolbox | **Este projeto** |
|---|---|---|---|
| Custo | Grátis | Pago, licença fechada | **Grátis, aberto** |
| Diagnósticos quimiométricos (VIP, SR, Hotelling T², Q-resíduos, DD-SIMCA, OPLS-DA) | ❌ (você implementa) | ✅ | ✅ |
| Validação *group-aware* (réplicas T1/T2/T3 nunca vazam entre treino/teste) | ❌ (manual) | ⚠️ limitado | ✅ **por padrão** |
| Reprodutível (sementes fixas, saída versionada) | ⚠️ | ❌ | ✅ |
| Usável **sem programar** (YAML + menu + web) | ❌ | ✅ (GUI paga) | ✅ |
| Planejamento de tamanho amostral + auditoria automática de confundimento classe×sessão + identificação de conjunto aberto calibrada por predição conforme, num único fluxo | ❌ | ❌ (não encontrado em documentação pública até 2026-08) | ✅ |

**A lacuna preenchida:** rigor de publicação (Q1) + reprodutibilidade +
acessibilidade, sem custo de licença. (Seleção de amostras por Kennard-Stone
e transferência de calibração, que o GUARACI também implementa, NÃO são
reivindicadas como diferenciais aqui — o Unscrambler já traz Kennard-Stone,
e transferência de calibração é técnica clássica de quimiometria,
razoavelmente presumida presente em suites comerciais maduras.)

---

## Diferencial metodológico: validação *group-aware*

Cada amostra é medida em **triplicata** (T1/T2/T3). Se essas réplicas forem
distribuídas livremente entre treino e teste, o modelo "decora" a amostra e a
acurácia fica **inflada** (vazamento de dados). Aqui, o identificador `mae_id`
mantém as três réplicas **sempre no mesmo lado** da partição
(`StratifiedGroupKFold` / `GroupShuffleSplit`), tanto na validação cruzada
quanto no *holdout* externo. É o que separa um número honesto de um artefato.

---

## O que o pipeline faz

- **Modos de análise**
  - **Classificação por classe** (multiclasse; código interno N1)
  - **Discriminação puro × adulterado** (autenticação; código interno N2)
  - **Quantificação de teor** (% de adulterante, regressão; código interno N3)
- **Pré-processamento** (ordem de Rinnan et al., 2009): MSC ou SNV → Savitzky-Golay → *mean-centering*. Presets: `MSC+SG+MC`, `SNV+SG+MC`, `Autoscaling`, `Mean-centering`.
- **Modelos**: PLS-DA, PLS regressão, PCA, HCA (Ward), **DD-SIMCA** (one-class), **OPLS-DA**.
- **Bateria de validação**: teste de permutação (Y-randomization), interceptos R²Y/Q²Y de Wold, CV-ANOVA (Eriksson), **IC por bootstrap BCa**, Hotelling T² (UCL Tracy-Young-Mason), Q-resíduos (Jackson-Mudholkar), linearidade formal (teste F de falta de ajuste) e protocolo de robustez (perturbação de pré-processamento/ruído/deriva de linha de base, reportado como intervalo — nunca aprovado/reprovado).
- **Interpretação química**: VIP (Chong & Jun, 2005), Selectivity Ratio (Rajalahti, 2009), anotação de bandas.
- **Etapa 4 — seleção de variáveis**: iPLS (intervalos), corte por VIP, top-fração por SR, sPLS-DA (NIPALS).
- **Mode cego (padrão)**: uma amostra desconhecida é classificada e SÓ ENTÃO quantificada a partir da classe PREDITA — nunca a verdadeira —, com uma etapa de identificação de conjunto aberto calibrada por predição conforme (espécie × adulterante) que reporta uma garantia de cobertura por combinação, não um rótulo nu.
- **Planejamento experimental**: orientação de tamanho amostral (alvo conformal ou de cobertura DD-SIMCA) e plano de sessões de coleta com ordem de leitura aleatorizada, além de auditoria automática de delineamento que sinaliza confundimento classe×sessão antes de virar uma métrica errada.
- **Transferência de calibração** (Direct/Piecewise Direct Standardization) e **seleção do conjunto de calibração** (Kennard-Stone, Duplex, SPXY) para mover um modelo entre instrumentos ou escolher um subconjunto representativo a partir de dados já medidos.
- **Sentinela de deriva do domínio de aplicabilidade**: acumula a taxa de rejeição AD (dentro/fora do domínio) ao longo de execuções sucessivas de predição em lote e testa formalmente (teste binomial exato, H0: taxa = alfa nominal) se ela está subindo — sinal de deriva de instrumento/processo desde a calibração. Estado persistido em JSON ao lado do modelo; atualizado automaticamente pela Predição em Lote da CLI.
- **Modo imagem (colorimetria digital, protótipo)**: converte fotos (RGB/HSV/Lab, opcionalmente textura GLCM) na mesma matriz numérica que os outros modos consomem — PCA, PLS-DA, DD-SIMCA etc. funcionam sem alteração. Detecta automaticamente o nível de garantia de agrupamento contra vazamento de réplicas disponível na pasta de dados (subpasta por amostra física / CSV de associação manual / nenhum) e declara o nível usado no log, no model card e no manifesto — nunca finge uma garantia que não tem.
- **Modo HSI (imageamento hiperespectral)**: DISTINTO do modo imagem — opera POR PIXEL de um cubo hiperespectral (formato ENVI, `.hdr`+`.bin`), não por foto inteira. Quality gate (saturação/SNR/pixels válidos), segmentação PCA+Otsu, PLS-DA por pixel com agregação por objeto físico (voto majoritário + heterogeneidade), mapa de classificação espacial. **Funciona com cubos do próprio usuário, offline, sem dataset público nenhum** — basta apontar pra' qualquer pasta com uma subpasta por classe; vale a mesma hierarquia de 3 níveis de garantia de agrupamento do modo imagem (subpasta por amostra / CSV de associação / nenhuma, sempre declarada explicitamente, nunca presumida). O dataset público (DeepHS Fruit) serve só de fixture de validação do próprio projeto, detectado automaticamente pelo `manifest.json` (ou forçado explicitamente pra' qualquer um dos dois lados) — passar por ele também libera validação externa por partição de dia/lote e explicabilidade cruzada VIP×banda química, nenhuma das duas aplicável a uma pasta genérica sem essa partição nativa. Funcionamento offline é PROVADO, não só alegado: um teste constrói um cubo sintético e roda o pipeline inteiro com o socket de rede desabilitado. Acessível pela tecla `[X]` do menu principal da CLI. Desempenho no fixture público ainda é modesto em alguns pontos, reportado sem inflar (ver `docs/VALIDACAO_PUBLICA.md` §7).

### Multimatriz e multitécnica por design — sem nenhuma ressalva conhecida

Qualquer matriz (óleo, milho, mel, um cubo hiperespectral de fruta ou
comprimido) e qualquer técnica de aquisição (espectrômetro, câmera de
celular, câmera hiperespectral, ou uma que você inventar) entra via
**perfil** (um arquivo YAML) ou, no HSI, uma pasta simples com dado
próprio — nunca uma alteração de código-fonte. Isso foi reverificado
por comando direto (não só alegado) para os 3 modos, com perfil/
domínio inventado NA HORA pro teste: uma matriz tabular fictícia
carregada de um YAML solto nunca distribuído com o pacote, uma técnica
de aquisição de imagem fictícia combinada com o perfil genérico, e um
domínio de HSI sem nenhuma relação com qualquer dataset já usado neste
projeto (autenticidade de comprimido farmacêutico) — ver
`tests/test_aceitacao_adaptabilidade.py`.

Uma auditoria de adaptabilidade achou 2 limites reais e estreitos —
ambos já **corrigidos**:

- **A Identificação espécie×adulterante** (Bloco 9b do modo cego,
  `identificacao.py`) era estruturalmente amarrada aos códigos de letra
  específicos do dataset original de óleo (A/M/S → algodão/milho/soja)
  — rodava sem erro em qualquer matriz, mas produzia silenciosamente
  zero combinações calibradas pra' qualquer convenção de nome
  diferente. Corrigido movendo o mapa letra→nome pro perfil de matriz
  (`MatrixProfile.codigos_adulterante`), de forma aditiva — todo
  `.joblib` já persistido continua carregando sem alteração, já que
  nunca re-deriva esse mapa no momento da predição. Verificado
  diretamente com um dataset sintético usando uma convenção de nome
  deliberadamente diferente da do óleo, rodando
  Detectar→Identificar→Quantificar de ponta a ponta e confirmando que
  os próprios testes do dataset de óleo original continuam passando
  sem mudança — ver `tests/test_identificacao_generica.py`.
- **Os perfis de técnica de aquisição de imagem eram reconhecidos por
  uma lista fixa de 3 nomes** (`bancada`/`celular`/`scanner`) pra' um
  filtro de listagem de menu — uma técnica nova sempre carregava e
  funcionava corretamente, mas não aparecia nessa listagem filtrada.
  Corrigido classificando um perfil como "técnica" pelo seu *conteúdo*
  (se declara os campos de resolução esperada/formatos aceitos/nível
  de agrupamento típico) em vez do nome do arquivo — esses 3 perfis
  continuam como exemplos pré-cadastrados de conveniência, não um
  limite rígido. Verificado com uma quarta técnica, inventada na hora,
  aparecendo corretamente na listagem filtrada, sem afetar as 3
  originais — ver `tests/test_perfil_matriz.py`/
  `tests/test_aceitacao_adaptabilidade.py`.

---

## Instalação

Requer **Python 3.10+**. O código fica no pacote `guaraci`, em `src/`.

```bash
pip install -e .          # pacote `guaraci` + núcleo científico (adiciona o comando `guaraci`)
# tudo (web + relatórios + benchmark + imagem):
pip install -e .[all]
# alternativa p/ deploy (ex.: Streamlit Cloud): pip install -r requirements.txt
```

**Checkout de 5 minutos, sem precisar de dado seu:**

```bash
guaraci doctor    # checa Python/RAM/CPU/dependências, grava guaraci_doctor.txt
guaraci demo      # roda o pipeline completo com espectros sintéticos, abre a pasta de resultados
guaraci --version
```

Ou rode no navegador sem instalar nada: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErleySC/guaraci/blob/master/notebooks/guaraci_5_minutos.ipynb)

---

## Como usar (3 formas — sem editar código)

A configuração vive em `config.yaml` (linguagem simples, com comentário acima de
cada campo). Comece copiando o modelo:

```bash
cp config.example.yaml config.yaml   # Windows: copy config.example.yaml config.yaml
```

Edite `pasta_dados` para apontar à pasta com seus arquivos `.dx`.

O código fica no pacote `guaraci`, em `src/`. Instale uma vez com
`pip install -e .` (ou `pip install -e .[all]` para web/relatórios/benchmark);
isso disponibiliza o comando `guaraci`. Sem instalar, use `PYTHONPATH=src`.

### 1. Interface GUARACI (terminal, recomendada)
```bash
guaraci                               # ou: PYTHONPATH=src python -m guaraci.guaraci
```
Interface bilíngue (PT/EN) em Rich: menus guiados para configurar tudo, escolher
a técnica analítica (FT-NIR, NIR, MIR, Raman, UV-Vis, fluorescência, HPLC, GC-MS,
NMR, IMS), aplicar perfis prontos e rodar — sem editar código. Tecla **G** abre o
assistente científico em qualquer menu.

### 2. Direto pelo `config.yaml`
```bash
python -m guaraci.pipeline --rodar
```

### 3. Interface web (navegador)
```bash
streamlit run app_quimiometria.py
```
Campos clicáveis, validação imediata da pasta, execução com log ao vivo, exibição
do resumo + figuras e download `.zip` de todos os resultados. O app coloca `src/`
no path sozinho, então funciona mesmo sem `pip install -e .`.

> Modo legado: `python -m guaraci.pipeline --codigo` usa a `Config`
> embutida no código (para quem prefere editar o `.py`).

---

## Perfis de matriz — como se troca de matriz

Tudo que é específico de matriz vive num perfil
(`src/guaraci/perfis_matriz/*.yaml`): faixa e unidade do eixo,
pré-processamento padrão, faixa de trabalho esperada e o **vocabulário** da
saída. O motor nunca lê esses termos para decidir nada — é exatamente isso
que impede o vocabulário de uma matriz de contaminar os resultados de outra.

```bash
guaraci perfis                    # generico · oleo_nir · milho_nir · mel_vis_nir
guaraci --perfil=milho_nir        # ou o caminho do seu próprio YAML
```

Escrever um perfil para uma matriz nova é copiar o `generico.yaml` e
preencher — sem mudar código, sem fork. Uma matriz não cadastrada levanta
`PerfilDesconhecidoError` **antes de carregar qualquer dado**, listando o que
existe e como adicionar.

> O `mel_vis_nir` está **declarado, mas não validado com dado real** — o
> dataset público de mel não foi obtido (ver
> [`docs/VALIDACAO_PUBLICA.md`](docs/VALIDACAO_PUBLICA.md)).
> O próprio YAML diz isso.

## Modo cego é o padrão

Quem manda uma amostra para um modelo de quantificação **não sabe** a classe
dela. Então a calibração também não pode saber: por padrão, a regressão por
classe usa a classe **predita**, não a verdadeira.

```bash
guaraci                       # cego (padrão)
guaraci --modo=controle       # usa a classe VERDADEIRA — só diagnóstico interno
```

O `--modo=controle` existe para exatamente uma coisa: separar erro de
quantificação de erro de classificação durante o desenvolvimento. Números
obtidos assim não representam desempenho de uso, e a saída marca isso. Um
teste prova a separação envenenando os rótulos verdadeiros: em modo cego, o
resultado não pode mudar.

## Estrutura dos dados de entrada

Pasta-raiz com **uma subpasta por classe**, cada uma com os `.dx` da classe
(auto-detectado). A classe e os metadados também são lidos do campo `##TITLE=`
do JCAMP-DX:

```
PURO:       {COD}-{DD-MM-YYYY}_T{N}
ADULTERADO: {COD}-{DD-MM-YYYY}-AD-{A|M|S}-{NN}%_T{N}
```
Adulterantes: **A** = algodão, **M** = milho, **S** = soja. **T{N}** = réplica
(triplicata) → vira o `mae_id` que protege contra vazamento.

A faixa espectral vem do **perfil de matriz ativo** — 4000–10000 cm⁻¹ no
`oleo_nir` (o truncamento remove ruído de borda da FFT que, com SG
derivativo, vira falso top-VIP), 1100–2498 nm no `milho_nir`, e o que você
declarar no seu próprio YAML.

**A proveniência nunca viaja junto com os resultados.** O parser lê apenas os
9 campos numéricos/de rótulo de que precisa — `##AUDIT TRAIL` (operador,
local) não é lido em momento nenhum — e os identificadores de amostra são
retirados do `metadados.csv` antes da gravação, substituídos por rótulos
anônimos de grupo de réplica que preservam a auditabilidade *group-aware*.
Uma suíte de testes varre todo artefato gerado, inclusive os bytes do
`.joblib`, e falha se algo escapar.

---

## Saída

Cada execução grava em `resultados_tcc/<amostra>/<Modo>/<execução>/`, onde
`<amostra>` é o rótulo do conjunto de dados (`tag`, ou derivado da pasta/
arquivo de entrada) e `<Modo>` é o objetivo científico resolvido para a
execução (`Exploratorio` / `Classificacao` / `Quantificacao` — ver
`docs/MANUAL.md`, seções 2.2 e 3):
```
resultados_tcc/<amostra>/<Modo>/PLSDA_OE_{nível-amigável}_{pré-proc}_{AAAAMMDD_HHMMSS}/
├── Graficos/    # scores, VIP, dendrograma, acceptance plots, etc.
├── Tabelas/     # metadados, identificadores, comparações (.csv)
├── Relatorios/  # resumo_modelo.txt, model_card.md
└── Modelos/     # modelo final (.joblib: pré-proc + PLS + LabelBinarizer + wavenumbers)
```

---

## Validação

Duas camadas independentes, ambas automatizadas.

**1. Contra as fórmulas que definem cada método.** O PLS-DA reproduz
`sklearn.PLSRegression` + argmax exatamente (max|Δcoef| = 0,0); SNV/VIP/MSC/
CV-ANOVA batem com suas fórmulas de definição dentro da tolerância
numérica; o UCL do DD-SIMCA bate com as fórmulas fechadas de
Tracy-Young-Mason/χ²; a componente ortogonal do OPLS-DA é ortogonal a
menos de 1e-6. Tabela completa e tolerâncias em
[`docs/VALIDATION.md`](docs/VALIDATION.md).

**2. Contra resultados publicados em dados públicos** — a única base de
evidência para alegação de desempenho:

| Dataset | Matriz | Alvo | GUARACI | Literatura |
|---|---|---|---|---|
| [Eigenvector Corn](https://eigenvector.com/data/Corn/) (m5) | milho moído | proteína | **RMSEP 0,144 %m/m** | 0,1–0,2 |
| Tecator | carne moída | gordura | RMSEP 2,001 | ver [`docs/BENCHMARK_TECATOR.md`](docs/BENCHMARK_TECATOR.md) |
| [Mendeley `ctgg7k4m5g`](https://data.mendeley.com/datasets/ctgg7k4m5g/2) (CC BY 4.0) | 19 óleos comestíveis (NIR, 8mm) | espécie (8, n≥5) | **acurácia balanceada 0,35 CV / 0,475 holdout** | sem número publicado nesse formato; prova de multimatriz, não reprodução de benchmark |

O número do milho e a classificação Mendeley são **jobs de CI**
(`validacao-publica`, `validacao-publica-mendeley`, matriz de 3 SOs para
o segundo): se o motor parar de reproduzir a literatura/piso, o build
falha. Reproduza localmente com `GUARACI_DATASETS_DIR=<pasta>
pytest tests/test_validacao_publica.py tests/test_validacao_publica_mendeley.py`
(baixe os arquivos Mendeley antes com
`python scripts/download_datasets/baixar_mendeley_oleos.py`). Nenhum dos
dois datasets é versionado neste repositório — ver
[`datasets/README.md`](datasets/README.md) para a política. O RMSEP de
valor de peróxido publicado para o Mendeley (4,9) **não** reproduziu com
um holdout independente usando os presets padrão do GUARACI — documentado
como limitação honesta em
[`docs/VALIDACAO_PUBLICA.md`](docs/VALIDACAO_PUBLICA.md), não escondido.

Quantificação nunca reporta um RMSEP nu: **SEP, RPD e RER** acompanham,
com o RPD carregando sua faixa de interpretação publicada (Williams 2014;
AACC 39-00.01). LOD/LOQ são calculados quando réplicas físicas permitem
estimar o ruído instrumental, e retornam `N/A` quando não permitem — um
LOD sem base de repetibilidade seria um número inventado.

## Por que os números aqui às vezes parecem modestos

Isso é deliberado, não uma desculpa. O GUARACI é construído pra' medir
com **holdout externo**, **proteção de grupo físico** (nunca separa
réplicas da mesma amostra entre treino/teste) e **cobertura estatística
formal** (predição conformal, validação cruzada LOGO) — e reportar o
resultado mesmo quando isso é menos favorável do que uma validação
interna ingênua sugeriria. Um modelo que parece ótimo sob K-fold
aleatório em réplicas correlacionadas e desmorona num lote de verdade
nunca visto é um falso positivo disfarçado de resultado; este projeto
escolhe expor esse desmoronamento em vez de escondê-lo atrás da
partição mais gentil. Cinco casos concretos, cada um medido e
documentado de forma independente:

1. **O RMSEP de valor de peróxido do Mendeley não reproduziu.** O
   número publicado (4,9) falhou num holdout independente com os
   presets padrão do GUARACI — R²val saiu **negativo** (pior que prever
   a média) — ver [`docs/VALIDACAO_PUBLICA.md`](docs/VALIDACAO_PUBLICA.md)
   §2.
2. **Uma reformulação contínua de um problema de classificação difícil
   deu Q² negativo.** Investigando por que a classificação hiperespectral
   de kiwi `unripe` colapsa (abaixo), reformular a maturação como um
   alvo contínuo de PLS-R (`storage_days`) generalizou *pior* entre dias
   do que prever a média — reportado como resultado negativo, não
   escondido como tentativa fracassada — ver `docs/VALIDACAO_PUBLICA.md`
   §7, Passo 112.
3. **A cobertura do DD-SIMCA platôa em ~0,94-0,945 e nunca chega ao
   nominal 0,95**, mesmo com n=1200 amostras de calibração sob um
   processo gerador de dados gaussiano perfeitamente especificado — o
   cenário mais favorável possível pro método. Nenhuma forma funcional
   ajusta a curva como "converge ao nominal com mais dados"; é "subida
   rápida, depois um piso persistente". Medido com
   `scripts/medicoes/medir_ddsimca_cobertura_vs_n.py`, documentado em
   [`docs/MANUAL.md`](docs/MANUAL.md).
4. **A identificação espécie×adulterante é estruturalmente não-validável
   em 36 das 38 combinações** do dataset privado do TCC — cada uma tem
   exatamente 1 sessão de coleta independente, abaixo do mínimo que o
   gate conformal precisa pra' certificar uma garantia de cobertura. O
   pipeline reporta `DESCONHECIDO`/não-validado em vez de um rótulo
   confiante — ver `src/guaraci/technique_registry.py` e "Limitações
   conhecidas" abaixo.
5. **O `unripe` hiperespectral de Kiwi/VIS falha mesmo com tamanho de
   amostra estatisticamente suficiente** (n≥19, a única combinação do
   dataset DeepHS Fruit que passa desse limiar) — 4 hipóteses testadas
   (seleção de banda, alvo contínuo, sobreposição espectral, checagem
   de ruído de rótulo contra uma medição independente de firmeza),
   nenhuma resgatou. Ver `docs/VALIDACAO_PUBLICA.md` §7, Passos 112 e
   114, e o próximo item pra' leitura definitiva da checagem de ruído
   de rótulo.
6. **A checagem de ruído de rótulo (hipótese 4 acima) fechou com uma
   resposta quantitativa, não um palpite.** A diferença física entre
   kiwi `unripe` e `perfect` é real e estatisticamente robusta — uma
   medição de firmeza objetiva e independente, no mesmo dataset,
   separa as duas classes de forma limpa (Mann-Whitney p=8,87×10⁻⁸,
   Cohen's d=1,64). A câmera VIS especificamente não captura essa
   diferença no espectro de refletância (efeito por-banda ≈0,38,
   fraco-a-moderado) — não é ruído de rótulo, é um limite genuíno de
   sensibilidade espectral dessa técnica de aquisição pra' esse estágio
   de maturação. Ver `docs/VALIDACAO_PUBLICA.md` §7, Passo 121.

Nenhum desses números foi suavizado, re-rodado até dar certo, ou
deixado fora de uma tabela. Se um número aqui parece pouco
impressionante, é a filosofia de validação externa funcionando como
projetada, não uma lacuna no texto.

## Segurança

Carregar um modelo `.joblib` executa código arbitrário (é um pickle) — ver
[`SECURITY.md`](SECURITY.md). Todo carregamento na CLI e no app web passa
por um único portão que recusa rodar sem confirmação explícita, mais um
manifesto SHA-256 que bloqueia o carregamento se o arquivo foi alterado
depois de exportado.

## Limitações conhecidas (honestidade científica)

- **Validado em três datasets públicos até agora** (milho NIR, Tecator NIT,
  óleos comestíveis Mendeley `ctgg7k4m5g` NIR). O motor é agnóstico de
  matriz, mas "agnóstico" é propriedade de arquitetura, não resultado de
  validação — um perfil que você escreve para uma matriz nova está por
  testar até que você o teste.
- **`mel_vis_nir` está declarado, não validado** — a origem do dataset de
  mel foi identificada (Downey, Fouratier & Kelly 2003), mas não existe
  repositório público para ele; busca ativa reconfirmada em 2026-08-27,
  ainda sem achar. O perfil carrega e funciona; nenhuma afirmação é feita
  sobre seus números.
- **DD-SIMCA one-class depende de amostras FÍSICAS suficientes por classe**,
  não de espectros. Com um único ponto de amostragem físico por classe, os
  limites são calibrados contra uma só observação independente — e o software
  diz isso, em vez de reportar um número confiante.
- **n pequeno**: todas as métricas vêm com **intervalo de confiança** (BCa), e
  a predição conformal recusa devolver intervalo quando o *n* é insuficiente,
  em vez de extrapolar.
- **Identificação de adulterante não pode ser agregada entre espécies** — a
  matriz-hospedeira domina o sinal de adulteração mais que o próprio
  adulterante (espécie explica 21-175× mais variância do delta espectral
  que o tipo de adulterante, medido com script reprodutível; ver
  `docs/MANUAL.md`). Calibrado por
  espécie×adulterante, com cobertura não validada no dataset atual (mesmo
  padrão do gate DD-SIMCA: 36 de 38 combinações têm 1 só sessão de coleta
  independente). Implementado como fluxo completo Detectar → Identificar →
  Quantificar em amostra nova (`identificacao.py`,
  `predicao.predict_blind`), que nunca força uma classe/número quando a
  cobertura não é validada — ver `docs/MANUAL.md`.
- **Modo imagem (colorimetria digital) é protótipo só sem garantia de
  agrupamento** — com subpasta por amostra física ou CSV de associação, tem
  a mesma proteção anti-vazamento dos demais mode; sem nenhum dos dois,
  cai em `StratifiedKFold` e o relatório carimba isso explicitamente. Não
  usar o nível sem garantia para resultado publicável.
- **Modo HSI aceita cubo próprio, mas a validação de desempenho ainda é
  só contra o dataset público** (DeepHS Fruit, várias frutas/câmeras) —
  sem calibração radiométrica própria (usa reflectância já calibrada
  pelo dataset), com desempenho de classificação ainda modesto em
  várias combinações fruta×câmera por desbalanceamento severo de
  classes, e com uma investigação de 4 hipóteses que não resgatou a
  separabilidade de `unripe` em Kiwi/VIS mesmo com amostra
  estatisticamente suficiente (ver `docs/VALIDACAO_PUBLICA.md` §7 para
  os números honestos). Rodar em cubo próprio funciona mecanicamente
  (provado offline, ver acima) — mas, como qualquer perfil novo, o
  desempenho na SUA matriz é não testado até você testar.

---

## Referências dos métodos

- Rinnan, Å.; van den Berg, F.; Engelsen, S. B. *Review of the most common
  pre-processing techniques for near-infrared spectra.* TrAC, 2009.
- Chong, I.-G.; Jun, C.-H. *Performance of some variable selection methods…
  (VIP).* Chemom. Intell. Lab. Syst., 2005.
- Rajalahti, T. et al. *Discriminating variable test… (Selectivity Ratio).*
  Anal. Chem., 2009.
- Eriksson, L. et al. *CV-ANOVA for significance testing of PLS and OPLS models.*
  J. Chemom., 2008.
- Tracy, N. D.; Young, J. C.; Mason, R. L. *Multivariate control charts
  (Hotelling T²).* J. Qual. Technol., 1992.

---

## Autor

**Erley S. da Costa** — Pesquisador / Desenvolvedor
[GitHub](https://github.com/ErleySC) ·
[Lattes](http://lattes.cnpq.br/5755582193284309) ·
erleysdacosta@gmail.com

## Licença

Licenciado sob a **Licença Pública Geral GNU v3.0** ([GPLv3](LICENSE)) —
© 2026 Erley S. da Costa. Livre para pesquisa, ensino e uso
acadêmico, desde que a autoria seja creditada e os derivados permaneçam abertos.

**Uso comercial:** embutir a Guaraci em produtos proprietários/fechados ou
comerciais exige uma **licença comercial** separada — veja
[`COMMERCIAL.md`](docs/COMMERCIAL.md) (contato: erleysdacosta@gmail.com). O autor
retém integralmente o copyright (licenciamento duplo).

Metadados legíveis por máquina em [`CITATION.cff`](CITATION.cff). **Não há
DOI de arquivamento no momento:** os depósitos anteriores no Zenodo foram
retirados pelo autor em 2026-08-04 e seus DOIs não resolvem mais para um
registro. Cite o repositório e uma versão com tag até que um novo depósito
seja feito.

## Como citar

**ABNT (NBR 6023:2018)**

> COSTA, E. S. da. **GUARACI: Plataforma quimiométrica com validação anti-vazamento por padrão**.
> Versão 31.9.0. 2026. Disponível em:
> <https://github.com/ErleySC/guaraci>.

**APA**

> Costa, E. S. da. (2026). *GUARACI: Chemometrics platform with leakage-safe validation by default*
> (v31.9.0) [Software].
> https://github.com/ErleySC/guaraci

**BibTeX**

```bibtex
@software{guaraci_2026,
  author      = {Costa, Erley S. da},
  title       = {{GUARACI: Plataforma quimiométrica com validação anti-vazamento por padrão}},
  version     = {31.9.0},
  year        = {2026},
  url         = {https://github.com/ErleySC/guaraci},
  license     = {GPL-3.0-or-later}
}
```
