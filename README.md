# GUARACI — Chemometric Intelligence for Amazonian Matrices

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License: GPLv3" src="https://img.shields.io/badge/License-GPLv3-3D8B57">
  <img alt="Commercial license available" src="https://img.shields.io/badge/commercial-license%20available-B8963E">
  <img alt="Version" src="https://img.shields.io/badge/version-31.2.0-B8963E">
  <img alt="Interface" src="https://img.shields.io/badge/UI-Rich%20CLI%20%2B%20Streamlit-4A9E5C">
  <img alt="Languages" src="https://img.shields.io/badge/i18n-PT%20%2F%20EN-686868">
  <img alt="Status" src="https://img.shields.io/badge/status-active-55B06A">
  <a href="https://doi.org/10.5281/zenodo.21313436"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.21313436.svg"></a>
</p>

> 🇬🇧 English (lean). • 🇧🇷 [Versão completa em português](README.pt-br.md)

### 🚀 [Try the live demo — no install required](https://guaraci.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://guaraci.streamlit.app/)

A **free and open** alternative to paid chemometrics suites (MATLAB/PLS_Toolbox,
The Unscrambler, SIMCA): a reproducible **multi-technique chemometrics platform**
for **classification, authentication and exploration** of complex matrices, with
**replicate-leakage-safe (group-aware) validation** — the project's
methodological differentiator.

**Mission:** democratize high-end chemometrics — give researchers the rigor of
commercial software, at zero license cost and with no closed-format lock-in.

It supports vibrational (**FT-NIR, NIR, MIR, Raman, UV-Vis**), luminescence
(**fluorescence**), chromatographic (**HPLC, GC-MS**) and resonance (**NMR,
IMS**) data, through a guided bilingual terminal interface (**GUARACI**) and a
Streamlit web app — no coding required.

Originally built for an undergraduate thesis (Chemistry, UFPA) on Amazonian
vegetable oils measured by FT-NIR on an **ABB MB3600** (JCAMP-DX `.dx` files),
now generalized to other analytical techniques and matrices.

---

## Why it exists

| | scikit-learn | Unscrambler / SIMCA / PLS_Toolbox | **This project** |
|---|---|---|---|
| Cost | Free | Paid, closed | **Free, open** |
| Chemometric diagnostics (VIP, SR, Hotelling T², Q-residuals, DD-SIMCA, OPLS-DA) | ❌ DIY | ✅ | ✅ |
| Group-aware validation (T1/T2/T3 replicates never leak) | ❌ manual | ⚠️ limited | ✅ **by default** |
| Reproducible (fixed seeds, versioned output) | ⚠️ | ❌ | ✅ |
| Usable **without coding** (YAML + menu + web) | ❌ | ✅ (paid GUI) | ✅ |

It fills the gap between low-level libraries and paid closed GUIs: **Q1-grade
rigor + reproducibility + accessibility, at no license cost.**

---

## Methodological differentiator: group-aware validation

Each sample is measured in **triplicate** (T1/T2/T3). Letting those replicates
fall on both sides of a train/test split inflates accuracy (data leakage). A
`mae_id` group key keeps the three replicates on the **same side**
(`StratifiedGroupKFold` / `GroupShuffleSplit`), in both cross-validation and the
external hold-out. That is what separates an honest metric from an artifact.

---

## Features

- **Analysis modes:** species classification (14 classes) · pure vs. adulterated discrimination · adulterant-content quantification (regression).
- **Preprocessing** (Rinnan et al. 2009 order): MSC/SNV → Savitzky-Golay → mean-centering. Presets: `MSC+SG+MC`, `SNV+SG+MC`, `Autoscaling`, `Mean-centering`.
- **Models:** PLS-DA, PLS regression, PCA, HCA (Ward), DD-SIMCA (one-class), OPLS-DA.
- **Validation battery:** permutation test, Wold R²Y/Q²Y intercepts, CV-ANOVA, **BCa bootstrap CIs**, Hotelling T², Q-residuals.
- **Interpretation:** VIP (Chong & Jun 2005), Selectivity Ratio (Rajalahti 2009).
- **Variable selection:** iPLS, VIP cutoff, SR top-fraction, sPLS-DA.

---

## Install

Python 3.10+. The code lives in the `guaraci` package under `src/`.

```bash
pip install -e .        # installs the `guaraci` package + core deps (adds the `guaraci` command)
# or, for the full web/reports/benchmark stack:  pip install -e .[all]
```

## Use (3 ways, no code editing)

Config lives in `config.yaml` (plain language). Start from the template:

```bash
cp config.example.yaml config.yaml   # then set `pasta_dados` to your .dx folder
```

```bash
guaraci                                       # 1. GUARACI terminal interface (recommended)
python -m guaraci.pipeline --rodar            # 2. run straight from config.yaml
streamlit run app_quimiometria.py             # 3. web app (browser)
```

> Without installing, prepend `PYTHONPATH=src` (e.g. `PYTHONPATH=src python -m guaraci.guaraci`).
> The web app bootstraps `src/` itself, so `streamlit run app_quimiometria.py` works either way.

**GUARACI** is the bilingual (PT/EN) Rich terminal interface: configure every
parameter through guided menus, pick analytical techniques (FT-NIR, NIR, MIR,
Raman, UV-Vis, fluorescence, HPLC, GC-MS, NMR, IMS), apply ready-made profiles,
and launch the pipeline — all without editing code. Press **G** in any menu to
open the built-in scientific assistant.

## Input

Root folder with **one subfolder per species** of `.dx` files. Class and
metadata are also parsed from the JCAMP-DX `##TITLE=` field. Triplicate tag
`T{N}` becomes the `mae_id` leakage guard. Useful range: **4000–10000 cm⁻¹**.

## Output

Each run writes to
`resultados_tcc/{sample}/{Mode}/PLSDA_OE_{level}_{preproc}_{timestamp}/`,
where `{sample}` is the dataset label (`tag`, or derived from the input
folder/file) and `{Mode}` is the scientific objective resolved for the run
(`Exploratorio` / `Classificacao` / `Quantificacao` — see `docs/MANUAL.md`).
Inside: `Graficos/` (figures), `Tabelas/` (CSV data), `Relatorios/`
(`resumo_modelo.txt`, `model_card.md`), and `Modelos/` (final `.joblib`).

## Validation

Every number in the table below comes from an automated test that runs on
every commit — see [`docs/VALIDATION.md`](docs/VALIDATION.md) for the full
table, tolerances, and how to reproduce: PLS-DA matches
`sklearn.PLSRegression` + argmax exactly (max|Δcoef| = 0.0), SNV/VIP/MSC/
CV-ANOVA match their defining formulas to numerical tolerance, DD-SIMCA's
UCL matches the Tracy-Young-Mason/χ² closed forms, OPLS-DA's orthogonal
component is orthogonal to <1e-6.

## Security

Loading a `.joblib` model executes arbitrary code (it's a pickle) — see
[`SECURITY.md`](SECURITY.md). Every load in the CLI and web app goes
through a single guard that refuses to run without explicit confirmation,
plus a SHA-256 manifest that blocks loading if the file was tampered with
after export.

## Known limitations

- **Babaçu vs. Palmiste** overlap — both are palms with near-identical NIR
  signatures (a botanical/chemical limit of FT-NIR, not a code bug).
- **DD-SIMCA** one-class `puros` mode needs ≥15 pure samples/class; current data
  has ~3, so the default `todos` mode is exploratory.
- Small *n* → all metrics ship with **confidence intervals** (BCa).

## Author

**Erley S. da Costa** — Researcher / Developer · GEAAp/UFPA
[GitHub](https://github.com/ErleySC) ·
[Lattes](http://lattes.cnpq.br/5755582193284309) ·
erleysdacosta@gmail.com

## License & citation

Licensed under the **GNU General Public License v3.0** ([GPLv3](LICENSE)) —
© 2026 Erley S. da Costa & GEAAp/UFPA. Free for research, teaching and academic
use, provided authorship is credited and derivatives stay open-source.

**Commercial use:** embedding Guaraci in proprietary/closed-source or commercial
products requires a separate **commercial license** — see
[`COMMERCIAL.md`](docs/COMMERCIAL.md) (contact: erleysdacosta@gmail.com). The author
retains full copyright (dual licensing).

Machine-readable metadata in [`CITATION.cff`](CITATION.cff). Permanent Zenodo
DOI: [10.5281/zenodo.21313436](https://doi.org/10.5281/zenodo.21313436).

## Contributing

Bug reports, feature requests and pull requests are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

**APA**

> Costa, E. S. da. (2026). *GUARACI: Chemometric Intelligence for Amazonian
> Matrices* (v31.2.0) [Software]. GEAAp/UFPA.
> https://github.com/ErleySC/guaraci

**ABNT (NBR 6023:2018)**

> COSTA, E. S. da. **GUARACI: Inteligência Quimiométrica para Matrizes
> Amazônicas**. Versão 31.2.0. GEAAp/UFPA, 2026. Disponível em:
> <https://github.com/ErleySC/guaraci>.

**BibTeX**

```bibtex
@software{guaraci_2026,
  author      = {Costa, Erley S. da},
  title       = {{GUARACI: Inteligência Quimiométrica para Matrizes Amazônicas}},
  version     = {31.2.0},
  year        = {2026},
  institution = {GEAAp/UFPA},
  url         = {https://github.com/ErleySC/guaraci},
  license     = {GPL-3.0-or-later},
  doi         = {10.5281/zenodo.21313436}
}
```
