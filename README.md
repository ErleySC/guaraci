# GUARACI — Chemometrics platform with leakage-safe validation by default

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License: GPLv3" src="https://img.shields.io/badge/License-GPLv3-3D8B57">
  <img alt="Commercial license available" src="https://img.shields.io/badge/commercial-license%20available-B8963E">
  <img alt="Version" src="https://img.shields.io/badge/version-31.9.0-B8963E">
  <img alt="Interface" src="https://img.shields.io/badge/UI-Rich%20CLI%20%2B%20Streamlit-4A9E5C">
  <img alt="Languages" src="https://img.shields.io/badge/i18n-PT%20%2F%20EN-686868">
  <img alt="Status" src="https://img.shields.io/badge/status-active-55B06A">
</p>

> 🇬🇧 English (lean). • 🇧🇷 [Versão completa em português](README.pt-br.md)

### 🚀 [Try the live demo — no install required](https://guaraci.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://guaraci.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErleySC/guaraci/blob/master/notebooks/guaraci_5_minutos.ipynb)

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

**Matrix-agnostic by construction.** What is specific to a matrix — spectral
range, default preprocessing, axis unit, and the *vocabulary* of the output —
lives in a **matrix profile** (a YAML file), never in the source. Switching
from vegetable oil to ground corn is one config field, not a code edit; a
matrix with no registered profile **fails with an actionable message instead
of predicting**.

**Validated exclusively on public datasets.** See
[Validation](#validation) — that is the only evidence base this repository
makes claims from.

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

- **Analysis modes:** class discrimination · pure vs. adulterated discrimination · adulterant-content quantification (regression).
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

**5-minute checkout, no data of your own required:**

```bash
guaraci doctor    # checks Python/RAM/CPU/deps, writes guaraci_doctor.txt
guaraci demo      # runs the full pipeline on synthetic spectra, opens the results folder
guaraci perfis    # lists the available matrix profiles
guaraci --help    # full option list and exit codes
guaraci --version
```

Or run it in the browser with zero local install: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErleySC/guaraci/blob/master/notebooks/guaraci_5_minutos.ipynb)

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

## Matrix profiles — how you switch matrices

Everything matrix-specific lives in a profile
(`src/guaraci/perfis_matriz/*.yaml`): axis range and unit, default
preprocessing, expected working range, and the **vocabulary** used in the
output. The engine never reads those terms to decide anything — which is
exactly what stops one matrix's vocabulary from contaminating another's
results.

```bash
guaraci perfis                    # generico · oleo_nir · milho_nir · mel_vis_nir
guaraci --perfil=milho_nir        # or the path to your own YAML
```

Writing a profile for a new matrix is copying `generico.yaml` and filling it
in — no source change, no fork. An unregistered matrix raises
`PerfilDesconhecidoError` **before any data is loaded**, listing what exists
and how to add one.

> `mel_vis_nir` is **declared but not validated against real data** — the
> public honey dataset was not obtained (see
> [`docs/VALIDACAO_PUBLICA.md`](docs/VALIDACAO_PUBLICA.md)).
> The YAML says so itself.

## Blind mode is the default

Whoever sends a sample to a quantification model does **not** know its class.
So the calibration must not know it either: by default the per-class
regression uses the **predicted** class, not the true one.

```bash
guaraci                       # blind (default)
guaraci --modo=controle       # uses the TRUE class — internal diagnosis only
```

`--modo=controle` exists for exactly one purpose: separating quantification
error from classification error while developing. Numbers obtained that way
do not represent real-world performance, and the output labels them as such.
A test proves the separation by poisoning the true labels: in blind mode the
result must not change.

## Input

Root folder with **one subfolder per class** of `.dx` files. Class and
metadata are also parsed from the JCAMP-DX `##TITLE=` field; the replicate
tag `T{N}` becomes the `mae_id` leakage guard. The spectral range comes from
the active matrix profile (e.g. 4000–10000 cm⁻¹ for `oleo_nir`,
1100–2498 nm for `milho_nir`).

**Provenance never travels with your results.** The parser reads only the 9
numeric/label fields it needs — `##AUDIT TRAIL` (operator, site) is never
read at all — and sample identifiers are stripped from `metadados.csv`
before it is written, replaced by anonymous replicate-group labels that
preserve group-aware auditability. A test suite scans every generated
artifact, including the `.joblib` bytes, and fails if any of it leaks.

## Output

Each run writes to
`resultados_tcc/{sample}/{Mode}/PLSDA_OE_{level_slug}_{preproc}_{timestamp}/`,
where `{sample}` is the dataset label (`tag`, or derived from the input
folder/file), `{Mode}` is the scientific objective resolved for the run
(`Exploratorio` / `Classificacao` / `Quantificacao` — see `docs/MANUAL.md`),
and `{level_slug}` is a friendly name for the analysis level (`PorEspecie` /
`Autenticacao` / `Quantificacao`).
Inside: `Graficos/` (figures), `Tabelas/` (CSV data), `Relatorios/`
(`resumo_modelo.txt`, `model_card.md`), and `Modelos/` (final `.joblib`).

## Validation

Two independent layers, both automated.

**1. Against the defining formulas.** PLS-DA matches `sklearn.PLSRegression`
+ argmax exactly (max|Δcoef| = 0.0); SNV/VIP/MSC/CV-ANOVA match their
definitions to numerical tolerance; DD-SIMCA's UCL matches the
Tracy-Young-Mason/χ² closed forms; OPLS-DA's orthogonal component is
orthogonal to <1e-6. Full table and tolerances in
[`docs/VALIDATION.md`](docs/VALIDATION.md).

**2. Against published results on public data** — the only evidence base for
performance claims:

| Dataset | Matrix | Target | GUARACI | Literature |
|---|---|---|---|---|
| [Eigenvector Corn](https://eigenvector.com/data/Corn/) (m5) | ground corn | protein | **RMSEP 0.144 %w/w** | 0.1–0.2 |
| Tecator | minced meat | fat | RMSEP 2.001 | see [`docs/BENCHMARK_TECATOR.md`](docs/BENCHMARK_TECATOR.md) |
| [Mendeley `ctgg7k4m5g`](https://data.mendeley.com/datasets/ctgg7k4m5g/2) (CC BY 4.0) | 19 edible oils (NIR, 8mm) | species (8, n≥5) | **balanced accuracy 0.35 CV / 0.475 holdout** | no published figure in this form; multimatrix proof, not a benchmark reproduction |

The corn figure and the Mendeley classification are **CI jobs**
(`validacao-publica`, `validacao-publica-mendeley`, matrix of 3 OSes for
the latter): if the engine stops reproducing the literature/floor, the
build fails. Reproduce locally with `GUARACI_DATASETS_DIR=<folder>
pytest tests/test_validacao_publica.py tests/test_validacao_publica_mendeley.py`
(download the Mendeley files first with
`python scripts/download_datasets/baixar_mendeley_oleos.py`). Neither
dataset is versioned in this repository — see
[`datasets/README.md`](datasets/README.md) for the policy. The
Mendeley dataset's published peroxide-value RMSEP (4.9) did **not**
reproduce with an independent holdout using GUARACI's default presets —
documented as an honest limitation in
[`docs/VALIDACAO_PUBLICA.md`](docs/VALIDACAO_PUBLICA.md), not hidden.

Quantification never reports a bare RMSEP: **SEP, RPD and RER** ship
alongside it, with RPD carrying its published interpretation band
(Williams 2014; AACC 39-00.01). LOD/LOQ are computed when physical
replicates allow estimating instrumental noise, and return `N/A` when they
do not — a LOD without a repeatability basis would be a made-up number.

## Security

Loading a `.joblib` model executes arbitrary code (it's a pickle) — see
[`SECURITY.md`](SECURITY.md). Every load in the CLI and web app goes
through a single guard that refuses to run without explicit confirmation,
plus a SHA-256 manifest that blocks loading if the file was tampered with
after export.

## Known limitations

- **Validated on three public datasets so far** (corn NIR, Tecator NIT,
  Mendeley `ctgg7k4m5g` edible oils NIR). The engine is matrix-agnostic,
  but "agnostic" is an architectural property, not a validation result —
  a profile you write for a new matrix is untested until you test it.
- **`mel_vis_nir` is declared, not validated** — the honey dataset's
  origin was identified (Downey, Fouratier & Kelly 2003) but no public
  repository for it exists; actively re-searched 2026-08-27, still not
  found. The profile loads and works; no claim is made about its numbers.
- **One-class DD-SIMCA needs enough *physical* samples per class**, not
  spectra. With one physical sampling point per class the limits are
  calibrated against a single independent observation, and the software says
  so rather than reporting a confident number.
- **Small *n*** → every metric ships with **confidence intervals** (BCa), and
  conformal prediction refuses to produce an interval when *n* is
  insufficient instead of extrapolating.
- **Adulterant identification cannot be pooled across species** — the host
  matrix dominates the adulteration signal more than the adulterant itself
  (species explains 21-175x more delta-spectrum variance than adulterant
  type, measured with a reproducible script; see `docs/MANUAL.md`).
  Calibrated per species×adulterant, with
  coverage reported as unvalidated on the current dataset (same pattern as
  the DD-SIMCA gate: 36 of 38 combinations have exactly 1 independent
  collection session). Implemented as a full Detect → Identify → Quantify
  blind-prediction flow (`identificacao.py`, `predicao.predict_blind`) that
  never forces a class or a number when coverage isn't validated — see
  `docs/MANUAL.md`.
- **Image mode (digital colorimetry) is a prototype only without a grouping
  guarantee** — with a per-sample subfolder or an association CSV, it has
  the same leakage protection as the other input modes; without either, it
  falls back to `StratifiedKFold` and the report explicitly stamps that.
  Do not use the no-guarantee level for publishable results.

## Author

**Erley S. da Costa** — Researcher / Developer
[GitHub](https://github.com/ErleySC) ·
[Lattes](http://lattes.cnpq.br/5755582193284309) ·
erleysdacosta@gmail.com

## License & citation

Licensed under the **GNU General Public License v3.0** ([GPLv3](LICENSE)) —
© 2026 Erley S. da Costa. Free for research, teaching and academic
use, provided authorship is credited and derivatives stay open-source.

**Commercial use:** embedding Guaraci in proprietary/closed-source or commercial
products requires a separate **commercial license** — see
[`COMMERCIAL.md`](docs/COMMERCIAL.md) (contact: erleysdacosta@gmail.com). The author
retains full copyright (dual licensing).

Machine-readable metadata in [`CITATION.cff`](CITATION.cff). There is no
archival DOI at present: the earlier Zenodo deposits were withdrawn by the
author on 2026-08-04 and their DOIs no longer resolve to a record. Cite the
repository and a tagged version until a new deposit is made.

## Contributing

Bug reports, feature requests and pull requests are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

**APA**

> Costa, E. S. da. (2026). *GUARACI: Chemometrics platform with leakage-safe validation by default*
> (v31.9.0) [Software].
> https://github.com/ErleySC/guaraci

**ABNT (NBR 6023:2018)**

> COSTA, E. S. da. **GUARACI: Plataforma quimiométrica com validação anti-vazamento por padrão**.
> Versão 31.9.0. 2026. Disponível em:
> <https://github.com/ErleySC/guaraci>.

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
