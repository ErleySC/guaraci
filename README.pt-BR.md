# GUARACI — quimiometria aberta, com validação anti-vazamento por padrão

**Público:** todos · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Licença: GPLv3" src="https://img.shields.io/badge/Licen%C3%A7a-GPLv3-2F7A48">
  <img alt="Versão" src="https://img.shields.io/badge/version-1.0.0-C24A00">
  <img alt="Interface" src="https://img.shields.io/badge/UI-Rich%20CLI%20%2B%20Streamlit-2F7A48">
  <img alt="Idiomas" src="https://img.shields.io/badge/idioma-PT%20%C2%B7%20EN-686868">
</p>

> 🇧🇷 Versão completa em português. • 🇬🇧 [English](README.md)

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErleySC/guaraci/blob/master/notebooks/guaraci_5_minutos.ipynb)

**Quimiometria de qualidade não deveria depender de quem pode pagar uma licença.** O GUARACI é
uma plataforma livre e aberta que leva quem pesquisa, quem estuda e quem faz controle de qualidade
do espectro bruto até um resultado defensável — exploração, classificação, autenticação,
quantificação, comparação de grupos e monitoramento de processos — por uma interface de terminal
guiada e um aplicativo web, sem precisar programar.

Ferramentas livres já cobrem a maior parte dos métodos: o `mdatools` (R) traz PCA, PLS, PLS-DA e
DD-SIMCA com diagnósticos maduros, e o scikit-learn tem os divisores por grupo. O que elas deixam
com o usuário é o fluxo em volta dos métodos, e as armadilhas de vazamento que há nele; as suítes
comerciais consagradas guiam esse fluxo atrás de uma licença paga e fechada. O GUARACI torna o
fluxo guiado livre, sob a GPLv3.

**O que você ganha**

- **Resultado que se sustenta.** Réplicas da mesma amostra física nunca caem nos dois lados de uma
  partição, na validação cruzada *e* no holdout externo. Isso é o padrão, não uma opção que você
  precisa saber ligar.
- **O fluxo inteiro num lugar só.** Planejar a coleta, auditar o delineamento, pré-processar,
  modelar, validar, interpretar, monitorar e escrever o relatório.
- **Números honestos.** Intervalo de confiança nas métricas principais, predição conformal que se
  recusa a responder quando os dados não sustentam, e um holdout relatado mesmo quando é menos
  lisonjeiro que uma partição ingênua.
- **Prático de verdade.** Um menu que explica cada opção (o que faz, quando usar, quando não usar),
  um assistente offline, cenários sintéticos para aprender e relatórios em PDF, Word, Excel,
  PowerPoint e LaTeX.
- **Seus dados continuam seus.** Tudo roda na sua máquina; identificadores de amostra e a
  procedência do instrumento são retirados antes de qualquer arquivo ser gravado.
- **Aberto em todos os sentidos.** Código-fonte, formatos de entrada, o modelo portátil em JSON e as
  conferências numéricas estão todos no repositório.

A interface de terminal (`guaraci`) abre em inglês e passa para o português com a tecla `I`; o
aplicativo web (Streamlit) fala português e inglês.

### Por que ele existe

| | Pacotes abertos (`mdatools`, scikit-learn, `chemotools`) | Suítes comerciais de quimiometria | **GUARACI** |
|---|---|---|---|
| Custo e licença | Grátis, aberta | Paga, fechada | **Grátis, aberta (GPLv3)** |
| Diagnósticos quimiométricos (VIP, SR, T² de Hotelling, resíduos Q, DD-SIMCA, OPLS-DA) | Incluídos no `mdatools` (R); parciais nos pacotes Python | Incluídos | **Incluídos** |
| Réplicas mantidas de um só lado de cada partição | Disponível (segmentos do `mdatools`, `StratifiedGroupKFold`); você precisa montar | Disponível, configurada pelo usuário | **Padrão, inclusive no holdout** |
| Uso sem programar | Em geral não (o `mdatools` tem apps web para DD-SIMCA e PLS) | Sim (interface paga) | **Sim (menu, web, YAML)** |
| Execução reprodutível (sementes fixas, pasta de saída versionada) | Por sua conta | Possível por script | **Padrão** |

**Quando outra ferramenta é melhor escolha.** Se você usa R, o `mdatools` é a referência madura e
publicada para PCA, PLS e DD-SIMCA; o DD-SIMCA do GUARACI está só parcialmente verificado
(`guaraci confianca`). Se você usa Python e só precisa de pré-processamento ou de um estimador, o
`chemotools` e o scikit-learn são menores e mais consolidados. O GUARACI serve para conduzir o estudo
inteiro (auditoria do delineamento, validação agrupada, holdout, relatório) sem programar.

<!-- guaraci:tecnicas:inicio -->
**Técnicas de medida:** **FT-NIR**, **NIR Dispersivo**, **MIR/FTIR**, **Raman**, **UV-Vis**, **Fluorescencia Molecular**, **HPLC**, **GC-MS**, **RMN / NMR**, **IMS**, **Fluorescencia 3D**, **ICP-OES / ICP-MS**, **Imagem digital**, **Imagem hiperespectral**, outras (perfil `generico`).

**Tarefas:**
- Ver como os dados se organizam, sem treinar modelo
- Dizer a que classe (especie, origem, produtor) cada amostra pertence
- Saber se uma amostra e autentica (puro x adulterado)
- Estimar o teor de um analito ou adulterante
- Testar se grupos diferem numa resposta
- Acompanhar deriva de um processo ou instrumento
- Analisar fotos ou cubos hiperespectrais

**Cenários sintéticos prontos** (`guaraci exemplo <cenário>`): `explorar`, `classificar`, `quantificar`, `multielementar`, `comparar`, `imagem`, `hsi`, `transferir`, `monitorar`, `fluorescencia_eem`, `cromatograma_gcms`, `vip`.
<!-- guaraci:tecnicas:fim -->

O mapa técnica × tarefa × comando × exemplo sintético × teste é gerado do catálogo em
[`docs/qualidade/mapa_tecnica_tarefa.md`](docs/qualidade/mapa_tecnica_tarefa.md).
Todos os documentos estão indexados em [`docs/projeto/INDICE.md`](docs/projeto/INDICE.md).

**Agnóstica de técnica por construção.** O que depende da medição — faixa espectral,
pré-processamento padrão, unidade do eixo, o *vocabulário* da saída — mora em um **perfil**
(um arquivo YAML), nunca no código. Uma técnica sem perfil registrado **falha com mensagem
acionável em vez de prever**.

---

## Validação agrupada, ativa por padrão

Cada amostra é medida em **réplicas** (T1/T2/T3). Deixar essas réplicas dos dois lados de uma
partição treino/teste infla a acurácia (vazamento). A chave de grupo `mae_id` mantém as réplicas
do **mesmo lado** (`StratifiedGroupKFold` / `GroupShuffleSplit`), na validação cruzada e no
holdout externo. Outras ferramentas podem ser configuradas para isso; aqui é o padrão, não uma
opção que você precisa saber ligar.

## Funcionalidades

- **Modos de análise:** discriminação de classes · puro × adulterado · quantificação de teor (regressão).
- **Pré-processamento** (ordem de Rinnan et al. 2009): MSC/SNV → Savitzky-Golay → centragem na média. Presets: `MSC+SG+MC`, `SNV+SG+MC`, `Autoscaling`, `Mean-centering`.
- **Modelos:** PLS-DA, regressão PLS, PCA, HCA (Ward), DD-SIMCA (uma classe), OPLS-DA.
- **Bateria de validação:** teste de permutação, interceptos R²Y/Q²Y de Wold, CV-ANOVA, **ICs bootstrap BCa**, T² de Hotelling, resíduos Q, teste F de falta de ajuste e protocolo de robustez relatado como intervalo (nunca aprovado/reprovado).
- **Interpretação e seleção:** VIP com estabilidade por dobra (`guaraci vip`, que mostra a estabilidade e não reduz variáveis), Razão de Seletividade, iPLS, sPLS-DA; a seleção de variáveis do `guaraci run` fica aninhada na validação e não pode vazar.
- **Comparação de grupos sobre a unidade independente** (`guaraci anova`): Tukey–Kramer, Games–Howell, Kruskal–Wallis com Dunn, diagnóstico de premissas, FDR entre variáveis.
- **Modo cego (padrão):** uma amostra desconhecida é classificada e depois quantificada *a partir da classe prevista* — nunca da verdadeira — com identificação de conjunto aberto calibrada por conformal, que reporta uma garantia de cobertura em vez de um rótulo nu.
- **Conjuntos de predição conformal** para classificação e regressão; `NÃO VALIDADO` em vez de um conjunto inventado quando o holdout não tem grupos independentes.
- **Transferência de calibração** (DS, PDS) e **seleção do conjunto de calibração** (Kennard-Stone, Duplex, SPXY).
- **Monitoramento de processo** (`guaraci monitorar`): PCA com limites T²/Q e sentinela de deriva entre lotes.
- **Planejamento experimental:** tamanho de amostra, plano de sessões com ordem de leitura aleatorizada e auditoria do delineamento que sinaliza confundimento classe/sessão.
- **Multivias e cromatografia:** PARAFAC para matrizes de excitação-emissão, MCR-ALS, alinhamento de retenção (COW).
- **Fusão multibloco** (prova de conceito, com guarda contra blocos desalinhados) e **análise de perfil de elementos** (censura, padrão interno, QC).
- **Imagens e cubos hiperespectrais:** estatísticas de cor por foto e classificação HSI por pixel com agregação por objeto. Ambos declaram qual garantia de agrupamento os dados sustentam.
- **Leitores de instrumento** (OPUS, SPC, PerkinElmer `.sp`, espectros processados de RMN Bruker, cromatogramas Agilent/Waters, ANDI-MS, arquivos de excitação-emissão) convertem um arquivo de instrumento no mesmo par `(X, Y)` que o pipeline consome.
- **Exportação portátil de modelo** (JSON, sem pickle) para os presets suportados.

## Instalação

**Nunca instalou um programa Python? Siga estes passos** (Windows; em macOS/Linux troque `.venv\Scripts\activate` por `source .venv/bin/activate`):

1. Instale o **Python 3.12** em <https://www.python.org/downloads/> e marque *"Add Python to PATH"*.
2. Baixe o GUARACI (botão verde **Code → Download ZIP** no GitHub) e descompacte em uma pasta de nome curto.
3. Abra um terminal nessa pasta e rode, uma linha por vez:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[web,reports]"
   guaraci doctor --instalar
   guaraci demo
   ```
4. Interface no navegador: `guaraci web`. Menu no terminal: `guaraci`.

Ainda não há pacote no PyPI. **Não rode `pip install guaraci`**: esse nome pertence a outro projeto, sem relação com este (instala um programa diferente, que também tem um comando `guaraci`). Instale a partir de um clone, como abaixo, **de dentro da pasta clonada** (não instale por nome pelo PyPI). Python 3.11+; o código mora no pacote `guaraci` em `src/`. O macOS não é plataforma suportada nem verificada nesta versão: a suíte completa de testes passa no macOS da CI (Python 3.11 e 3.13), mas não houve uso real verificado lá.

Guia passo a passo para Windows (inclusive a mensagem de política de execução do PowerShell), Linux (`python3.x-venv`) e WSL, com requisitos mínimos: [`docs/usuario/instalacao.md`](docs/usuario/instalacao.md). Em terminal de fundo claro, defina `GUARACI_TERMINAL=claro` para manter as cores legíveis.

```bash
pip install -e .        # o pacote `guaraci` + dependências essenciais
# ou, para a pilha completa de web/relatórios/benchmark:  pip install -e .[all]
```

A instalação base **não** traz os leitores de Bruker OPUS (`opus`), Thermo/Galactic SPC (`spc`), pasta Bruker de RMN (`rmn`) e pasta Agilent/Waters de HPLC/GC (`hplc`), nem PARAFAC (`multiway`), textura de imagem (`imagem`) e o diagnóstico PCV do DD-SIMCA (`robusto`). A tela [2.4] marca esses formatos com `(extra: pip install ...)` quando o pacote falta, e `guaraci doctor` e `guaraci checar` listam o que falta e o que isso afeta. Para instalar tudo de uma vez:

```bash
pip install ".[all]"   # dentro da pasta clonada; para desenvolver: pip install -e ".[all]"
```

Não sabe o que falta? `guaraci doctor` lista, e `guaraci doctor --instalar` instala depois da sua confirmação (com `--sim` não pergunta). Ele instala o navegador, os escritores de relatório e todos os leitores que existem para o seu Python, com as faixas de versão do pacote instalado, então também funciona de dentro de um clone. O extra de benchmark (`xgboost`, `shap`, grande) fica de fora; peça com `pip install -e ".[benchmark]"`.

**Primeiros passos, sem precisar de dado seu:**

```bash
guaraci doctor    # confere Python/RAM/CPU/dependências
guaraci demo      # roda o pipeline completo em espectros sintéticos
guaraci perfis    # lista os perfis disponíveis
guaraci --help    # lista completa de opções e códigos de saída
guaraci exemplo   # lista os cenários sintéticos guiados; `guaraci exemplo <cenário>` monta um
guaraci run       # roda o pipeline uma vez a partir de ./config.yaml e sai (automação e CI)
guaraci ajuda     # explica qualquer opção, técnica, gráfico ou tela (`guaraci buscar TERMO` busca em tudo)
guaraci assistente "sua pergunta"   # resposta offline a partir do catálogo; `guaraci confianca <método>` mostra selo e evidência
guaraci --version
```

Também: `anova`, `vip`, `monitorar`, `transferir`, `relatorio` — cada um é descrito por `guaraci --help`.

## Uso (3 jeitos, sem editar código)

A configuração fica em `config.yaml` (linguagem simples). Parta do modelo:

```bash
cp config.example.yaml config.yaml   # depois aponte `pasta_dados` para a sua pasta de dados
```

```bash
guaraci                                       # 1. interface de terminal (recomendada)
python -m guaraci.pipeline --rodar            # 2. rodar direto do config.yaml
guaraci web                                   # 3. aplicativo web (navegador, modo local)
```

`guaraci web` abre o aplicativo em **modo local** (caminhos livres, modelos `.joblib`, ligado a 127.0.0.1). O `streamlit run app_quimiometria.py` puro sobe o **modo servidor** (só upload, pasta de sessão isolada, só modelo JSON portátil) — o padrão seguro para qualquer coisa alcançável por rede. Veja [`SECURITY.md`](SECURITY.md).

## Perfis — como você troca de técnica

Um perfil (`src/guaraci/perfis_matriz/*.yaml`) guarda faixa e unidade do eixo, pré-processamento padrão, faixa de trabalho esperada e o vocabulário da saída. O motor nunca lê esses termos para decidir nada.

```bash
guaraci perfis                              # lista todos os perfis registrados (8 distribuídos)
guaraci --perfil=nir_dispersivo_1100_2498nm  # ou o caminho do seu próprio YAML
```

Escrever um perfil para uma técnica nova é copiar `generico.yaml` e preencher — sem mexer no código, sem fork. Um perfil não registrado levanta `PerfilDesconhecidoError` **antes de carregar qualquer dado**, listando o que existe. `visnir_400_1100nm` está **declarado, mas não validado com dado real**; o próprio YAML diz isso.

## O modo cego é o padrão

Quem envia uma amostra a um modelo de quantificação **não conhece** a classe dela, então por padrão a regressão por classe usa a classe **prevista**. `guaraci --modo=controle` usa a classe verdadeira e existe só para separar erro de quantificação de erro de classificação durante o desenvolvimento; a saída rotula esses números como não representativos. Um teste envenena os rótulos verdadeiros e confere que o resultado do modo cego não muda.

## Entrada e saída

Pasta raiz com **uma subpasta por classe** de arquivos `.dx`, ou uma tabela (CSV/XLSX), pasta de imagens, pasta de cubos ou arquivos de instrumento. Classe e metadados saem do campo `##TITLE=` do JCAMP-DX; a marca de réplica `T{N}` vira a guarda de vazamento `mae_id`. A faixa espectral vem do perfil ativo.

**A proveniência nunca viaja com os seus resultados.** O leitor lê só os campos de que precisa (nunca `##AUDIT TRAIL`) e os identificadores de amostra são removidos do `metadados.csv` antes de gravar. Um teste varre todo artefato gerado, inclusive os bytes do `.joblib`.

Cada execução grava em `resultados_guaraci/{amostra}/{Modo}/PLSDA_OE_{nivel}_{preproc}_{timestamp}/` com `Graficos/`, `Tabelas/`, `Relatorios/` (`resumo_modelo.txt`, `model_card.md`) e `Modelos/`.

## Validação

Duas camadas independentes.

**1. Contra as fórmulas que definem os métodos.** O PLS-DA bate exatamente com `sklearn.PLSRegression` + argmax; SNV, VIP, MSC e CV-ANOVA batem com as definições até a tolerância numérica; o limite do DD-SIMCA bate com as formas fechadas; o componente ortogonal do OPLS-DA é ortogonal a < 1e-6. Tabela e tolerâncias: [`docs/cientifico/validacao_numerica.md`](docs/cientifico/validacao_numerica.md).

**2. Valores padrão.** Limiares e parâmetros padrão são ponto de partida, tirados da literatura citada e não universais, e ajustáveis pelas opções de cada técnica (`guaraci confianca` mostra a evidência por trás do selo de cada técnica). O GUARACI não distribui dados de terceiros; a suíte de testes usa dados sintéticos gerados por código do próprio projeto.

A quantificação nunca reporta um RMSEP nu: **SEP, RPD e RER** vão junto; LOD/LOQ voltam `N/A` quando as réplicas não permitem estimar o ruído instrumental.

### Como ler os números

O GUARACI mede com **holdout externo**, **proteção de grupo físico** e **cobertura estatística formal**, e reporta o resultado mesmo quando ele é menos lisonjeiro que uma partição ingênua daria. Um modelo que parece ótimo em K-fold aleatório sobre réplicas correlacionadas e desaba num lote realmente novo é um falso positivo vestido de resultado; este projeto escolhe mostrar o desabamento. Resultados fracos ou nulos também são relatados.

## Segurança

Carregar um modelo `.joblib` executa código arbitrário (é um pickle) — veja [`SECURITY.md`](SECURITY.md). Todo carregamento na CLI e no aplicativo web passa por uma única guarda que se recusa a rodar sem confirmação explícita, mais um manifesto SHA-256 que bloqueia o carregamento se o arquivo foi adulterado depois da exportação.

## Limitações conhecidas

A lista completa, com a evidência de cada ponto, está em [`docs/cientifico/limitacoes.md`](docs/cientifico/limitacoes.md). As principais:

- **Verificada técnica a técnica.** "Agnóstica de técnica" é propriedade de arquitetura, não resultado de validação: um perfil que você escreve para uma técnica nova não está testado até você testar.
- **O resultado depende da unidade independente.** Se a réplica for tratada como unidade, significância e acurácia saem infladas; os programas dizem qual unidade usaram.
- **DD-SIMCA de uma classe exige amostras *físicas* suficientes por classe**, não espectros; com um ponto de amostragem por classe os limites se apoiam em uma observação, e o programa avisa.
- ***n* pequeno:** as métricas principais vêm com intervalo de confiança, e a predição conformal se recusa a produzir um conjunto quando o n é insuficiente.
- **O modo imagem sem garantia de agrupamento é protótipo** e o relatório carimba isso.
- **HSI** funciona com cubos seus, offline, e ainda é protótipo: trate o desempenho na sua técnica como não testado até você testar.
- **macOS e Docker só são conferidos pela CI.** A suíte completa passa no macOS (Python 3.11 e 3.13), e a CI constrói a imagem Docker e confere que o app web sobe em modo servidor sem root. Nenhum dos dois foi verificado em uso real, e análises dentro do contêiner não são testadas.
- **O assistente** responde a partir do catálogo embutido (busca offline); ele não gera respostas livres.
- **Idioma da interface:** o terminal e o aplicativo web rodam em português ou inglês; alguns nomes de pasta e de coluna (`dados`, `classe`) ficam em português porque são o que você digita.

## Autor

**Erley Silva da Costa** — Pesquisador / Desenvolvedor · [GitHub](https://github.com/ErleySC) · [ORCID](https://orcid.org/0009-0005-9655-6349)

O GUARACI é um projeto individual: não há equipe, empresa ou instituição por trás dele ([`GOVERNANCE.md`](GOVERNANCE.md)). Foi desenvolvido com auxílio de assistentes de IA para programação; o autor revisou e responde por todo o código e documentação.

## Licença e citação

Licenciado sob a **GNU General Public License v3.0 ou posterior** (`GPL-3.0-or-later`, [LICENSE](LICENSE)) — Copyright (C) 2026 Erley Silva da Costa. Qualquer pessoa, inclusive empresas, pode usar, modificar e redistribuir o software, desde que preserve os avisos de direito autoral e libere qualquer derivado distribuído sob a GPLv3 com o código-fonte. O software é distribuído somente sob esses termos.

Metadados legíveis por máquina em [`CITATION.cff`](CITATION.cff). Não há DOI de arquivamento no momento; cite o repositório e uma versão marcada.

**APA**

> Costa, Erley Silva da. (2026). *GUARACI: Plataforma quimiométrica com validação anti-vazamento por padrão*
> (v1.0.0) [Software].
> https://github.com/ErleySC/guaraci

**ABNT (NBR 6023:2018)**

> COSTA, Erley Silva da. **GUARACI: Plataforma quimiométrica com validação anti-vazamento por padrão**.
> Versão 1.0.0. 2026. Disponível em:
> <https://github.com/ErleySC/guaraci>.

**BibTeX**

```bibtex
@software{guaraci_2026,
  author      = {Costa, Erley Silva da},
  title       = {{GUARACI: Plataforma quimiométrica com validação anti-vazamento por padrão}},
  version     = {1.0.0},
  year        = {2026},
  url         = {https://github.com/ErleySC/guaraci},
  license     = {GPL-3.0-or-later}
}
```

## Como contribuir e pedir ajuda

Relatos de erro, pedidos de funcionalidade e pull requests são bem-vindos — veja [`CONTRIBUTING.md`](CONTRIBUTING.md) e o [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Dúvidas vão para as [GitHub Discussions](https://github.com/ErleySC/guaraci/discussions) ([`SUPPORT.md`](SUPPORT.md)); vulnerabilidades vão pelo canal privado descrito no [`SECURITY.md`](SECURITY.md), nunca numa issue pública.
