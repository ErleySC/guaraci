# Pipeline Quimiométrico FT-NIR — Autenticação de Óleos Vegetais Amazônicos

> 🇧🇷 Versão em português (de trabalho). • 🇬🇧 [English version](README.md)

Pipeline reprodutível de **quimiometria** para **classificação e autenticação**
de óleos vegetais por **espectroscopia FT-NIR**, com **validação robusta a
vazamento de réplicas** (*group-aware*) — o diferencial metodológico do projeto.

Desenvolvido como Trabalho de Conclusão de Curso (Química, UFPA) sobre óleos
amazônicos medidos em **ABB MB3600** (arquivos **JCAMP-DX `.dx`**).

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

- **Níveis de análise**
  - **N1** — classificação por espécie (14 classes)
  - **N2** — puro × adulterado
  - **N3** — regressão de teor (% de adulterante)
- **Pré-processamento** (ordem de Rinnan et al., 2009): MSC ou SNV → Savitzky-Golay → *mean-centering*. Presets: `MSC+SG+MC`, `SNV+SG+MC`, `Autoscaling`, `Mean-centering`.
- **Modelos**: PLS-DA, PLS regressão, PCA, HCA (Ward), **DD-SIMCA** (one-class), **OPLS-DA**.
- **Bateria de validação**: teste de permutação (Y-randomization), interceptos R²Y/Q²Y de Wold, CV-ANOVA (Eriksson), **IC por bootstrap BCa**, Hotelling T² (UCL Tracy-Young-Mason), Q-resíduos (Jackson-Mudholkar).
- **Interpretação química**: VIP (Chong & Jun, 2005), Selectivity Ratio (Rajalahti, 2009), anotação de bandas.
- **Etapa 4 — seleção de variáveis**: iPLS (intervalos), corte por VIP, top-fração por SR, sPLS-DA (NIPALS).

---

## Instalação

Requer **Python 3.10+**.

```bash
pip install -r requirements.txt
```

---

## Como usar (3 formas — sem editar código)

A configuração vive em `config.yaml` (linguagem simples, com comentário acima de
cada campo). Comece copiando o modelo:

```bash
cp config.example.yaml config.yaml   # Windows: copy config.example.yaml config.yaml
```

Edite `pasta_dados` para apontar à pasta com seus arquivos `.dx`.

### 1. Assistente de terminal (estilo CMD)
```bash
python pineline_quimiometria_14.py
```
Menu numerado para editar campos, salvar/carregar e rodar.

### 2. Direto pelo `config.yaml`
```bash
python pineline_quimiometria_14.py --rodar
```

### 3. Interface web (navegador)
```bash
streamlit run app_quimiometria.py
```
Campos clicáveis, validação imediata da pasta, execução com log ao vivo, exibição
do resumo + figuras e download `.zip` de todos os resultados.

> Modo legado: `python pineline_quimiometria_14.py --codigo` usa a `Config`
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

Cada execução cria uma pasta versionada:
```
resultados_tcc/PLSDA_OE_{nível}_{pré-proc}_{AAAAMMDD_HHMMSS}/
├── dados/      # metadados, identificadores, comparações (.csv)
├── figuras/    # scores, VIP, dendrograma, acceptance plots, etc.
├── modelos/    # modelo final (.joblib: pré-proc + PLS + LabelBinarizer + wavenumbers)
└── logs/       # resumo_modelo.txt
```

---

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

## Licença

[MIT](LICENSE) — © 2026 Erley S. da Costa. Uso livre com atribuição.

## Como citar

> da Costa, E. S. *Pipeline quimiométrico FT-NIR para autenticação de óleos
> vegetais amazônicos com validação group-aware.* TCC, Química, UFPA, 2026.
