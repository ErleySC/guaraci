# GUARACI — Inteligência Quimiométrica para Matrizes Amazônicas

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Licença: GPLv3" src="https://img.shields.io/badge/Licen%C3%A7a-GPLv3-3D8B57">
  <img alt="Licença comercial disponível" src="https://img.shields.io/badge/comercial-licen%C3%A7a%20dispon%C3%ADvel-B8963E">
  <img alt="Version" src="https://img.shields.io/badge/version-31.8.0-B8963E">
  <img alt="Interface" src="https://img.shields.io/badge/UI-Rich%20CLI%20%2B%20Streamlit-4A9E5C">
  <img alt="Idiomas" src="https://img.shields.io/badge/i18n-PT%20%2F%20EN-686868">
  <img alt="Status" src="https://img.shields.io/badge/status-ativo-55B06A">
  <a href="https://doi.org/10.5281/zenodo.21313436"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.21313436.svg"></a>
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

Originalmente desenvolvido como Trabalho de Conclusão de Curso (Química, UFPA)
sobre óleos amazônicos medidos por FT-NIR em **ABB MB3600** (arquivos **JCAMP-DX
`.dx`**), agora generalizado para outras técnicas e matrizes.

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

**A lacuna preenchida:** rigor de publicação (Q1) + reprodutibilidade +
acessibilidade, sem custo de licença.

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
  - **Classificação por espécie** (14 classes; código interno N1)
  - **Discriminação puro × adulterado** (autenticação; código interno N2)
  - **Quantificação de teor** (% de adulterante, regressão; código interno N3)
- **Pré-processamento** (ordem de Rinnan et al., 2009): MSC ou SNV → Savitzky-Golay → *mean-centering*. Presets: `MSC+SG+MC`, `SNV+SG+MC`, `Autoscaling`, `Mean-centering`.
- **Modelos**: PLS-DA, PLS regressão, PCA, HCA (Ward), **DD-SIMCA** (one-class), **OPLS-DA**.
- **Bateria de validação**: teste de permutação (Y-randomization), interceptos R²Y/Q²Y de Wold, CV-ANOVA (Eriksson), **IC por bootstrap BCa**, Hotelling T² (UCL Tracy-Young-Mason), Q-resíduos (Jackson-Mudholkar).
- **Interpretação química**: VIP (Chong & Jun, 2005), Selectivity Ratio (Rajalahti, 2009), anotação de bandas.
- **Etapa 4 — seleção de variáveis**: iPLS (intervalos), corte por VIP, top-fração por SR, sPLS-DA (NIPALS).

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

## Estrutura dos dados de entrada

Pasta-raiz com **uma subpasta por espécie**, cada uma com os `.dx` da espécie
(auto-detectado). A classe e os metadados também são lidos do campo `##TITLE=`
do JCAMP-DX:

```
PURO:       {COD}-{DD-MM-YYYY}_T{N}
ADULTERADO: {COD}-{DD-MM-YYYY}-AD-{A|M|S}-{NN}%_T{N}
```
Adulterantes: **A** = algodão, **M** = milho, **S** = soja. **T{N}** = réplica
(triplicata) → vira o `mae_id` que protege contra vazamento.

**14 espécies:** Açaí, Andiroba, Babaçu, Bacaba, Buriti, Castanha do Pará, Coco,
Copaíba, Goiaba, Graviola, Maracujá, Palmiste, Patauá, Pracaxi.

Faixa espectral útil: **4000–10000 cm⁻¹** (truncamento remove ruído de borda da
FFT que, com SG derivativo, vira falso top-VIP).

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

Cada número da tabela abaixo vem de um teste automatizado que roda a cada
commit — tabela completa, tolerâncias e como reproduzir em
[`docs/VALIDATION.md`](docs/VALIDATION.md): o PLS-DA reproduz
`sklearn.PLSRegression` + argmax exatamente (max|Δcoef| = 0,0), SNV/VIP/MSC/
CV-ANOVA batem com suas fórmulas de definição dentro da tolerância
numérica, o UCL do DD-SIMCA bate com as fórmulas fechadas de
Tracy-Young-Mason/χ², a componente ortogonal do OPLS-DA é ortogonal a
menos de 1e-6.

## Segurança

Carregar um modelo `.joblib` executa código arbitrário (é um pickle) — ver
[`SECURITY.md`](SECURITY.md). Todo carregamento na CLI e no app web passa
por um único portão que recusa rodar sem confirmação explícita, mais um
manifesto SHA-256 que bloqueia o carregamento se o arquivo foi alterado
depois de exportado.

## Limitações conhecidas (honestidade científica)

- **Babaçu × Palmiste**: classe mais fraca — ambas são palmáceas e têm assinatura
  NIR muito próxima. É uma limitação **botânica/química** do FT-NIR, não do código.
- **DD-SIMCA one-class**: o modo `puros` exige ≥15 amostras puras por classe; o
  dataset atual tem ~3/classe, então o modo padrão (`todos`) é exploratório — as
  métricas de sensibilidade/especificidade **não** equivalem a autenticação one-class.
- **n pequeno**: por isso todas as métricas vêm com **intervalo de confiança** (BCa).

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

**Erley S. da Costa** — Pesquisador / Desenvolvedor · GEAAp/UFPA
[GitHub](https://github.com/ErleySC) ·
[Lattes](http://lattes.cnpq.br/5755582193284309) ·
erleysdacosta@gmail.com

## Licença

Licenciado sob a **Licença Pública Geral GNU v3.0** ([GPLv3](LICENSE)) —
© 2026 Erley S. da Costa & GEAAp/UFPA. Livre para pesquisa, ensino e uso
acadêmico, desde que a autoria seja creditada e os derivados permaneçam abertos.

**Uso comercial:** embutir a Guaraci em produtos proprietários/fechados ou
comerciais exige uma **licença comercial** separada — veja
[`COMMERCIAL.md`](docs/COMMERCIAL.md) (contato: erleysdacosta@gmail.com). O autor
retém integralmente o copyright (licenciamento duplo).

Metadados legíveis por máquina em [`CITATION.cff`](CITATION.cff). DOI Zenodo
permanente: [10.5281/zenodo.21313436](https://doi.org/10.5281/zenodo.21313436).

## Como citar

**ABNT (NBR 6023:2018)**

> COSTA, E. S. da. **GUARACI: Inteligência Quimiométrica para Matrizes
> Amazônicas**. Versão 31.8.0. GEAAp/UFPA, 2026. Disponível em:
> <https://github.com/ErleySC/guaraci>.

**APA**

> Costa, E. S. da. (2026). *GUARACI: Chemometric Intelligence for Amazonian
> Matrices* (v31.8.0) [Software]. GEAAp/UFPA.
> https://github.com/ErleySC/guaraci

**BibTeX**

```bibtex
@software{guaraci_2026,
  author      = {Costa, Erley S. da},
  title       = {{GUARACI: Inteligência Quimiométrica para Matrizes Amazônicas}},
  version     = {31.8.0},
  year        = {2026},
  institution = {GEAAp/UFPA},
  url         = {https://github.com/ErleySC/guaraci},
  license     = {GPL-3.0-or-later},
  doi         = {10.5281/zenodo.21313436}
}
```
