# GUARACI — open chemometrics with leakage-safe validation by default

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="License: GPLv3" src="https://img.shields.io/badge/License-GPLv3-2F7A48">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-C24A00">
  <img alt="Interface" src="https://img.shields.io/badge/UI-Rich%20CLI%20%2B%20Streamlit-2F7A48">
  <img alt="Languages" src="https://img.shields.io/badge/language-EN%20%C2%B7%20PT-686868">
</p>

> 🇬🇧 English. • 🇧🇷 [Versão completa em português](README.pt-BR.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErleySC/guaraci/blob/master/notebooks/guaraci_5_minutos.ipynb)

**Chemometrics should not depend on who can afford a license.** GUARACI is a free and open
platform that takes a researcher, a student or a quality-control lab from raw spectra to a
defensible result — exploration, classification, authentication, quantification, group
comparison and process monitoring — through a guided terminal interface and a web app, with
no coding required.

It exists because two things are usually missing at the same time. Libraries such as
scikit-learn are free but leave the chemometric diagnostics and the leakage traps to the user;
the established commercial suites ship those diagnostics behind a paid, closed license.
GUARACI puts both on the same desk, at zero license cost, under the GPLv3.

**What you get**

- **Results you can defend.** Replicates of the same physical sample never fall on both sides of
  a split, in cross-validation *and* in the external holdout. That is the default, not an option you
  have to know to turn on.
- **The whole workflow in one place.** Planning the collection, auditing the design, preprocessing,
  modelling, validating, interpreting, monitoring and writing the report.
- **Honest numbers.** Confidence intervals on the main metrics, conformal prediction that refuses
  to answer when the data cannot support it, and a holdout that is reported even when it is less
  flattering than a naive split.
- **Practical by design.** A menu that explains every option (what it does, when to use it, when
  not to), an offline assistant, synthetic scenarios to learn with, and reports in PDF, Word,
  Excel, PowerPoint and LaTeX.
- **Your data stays yours.** Everything runs on your machine; sample identifiers and instrument
  provenance are stripped before anything is written.
- **Open in every sense.** Source code, input formats, the portable JSON model and the
  numerical checks are all in the repository.

The terminal interface (`guaraci`) opens in English and switches to Portuguese with the `I` key;
the Streamlit web app speaks English and Portuguese.

### Why it exists

| | Python libraries (e.g. scikit-learn) | Commercial chemometrics suites | **GUARACI** |
|---|---|---|---|
| Cost and license | Free, open | Paid, closed | **Free, open (GPLv3)** |
| Chemometric diagnostics (VIP, SR, Hotelling T², Q-residuals, DD-SIMCA, OPLS-DA) | Do it yourself | Included | **Included** |
| Replicates kept on one side of every split | Manual | Available, set up by the user | **Default, holdout included** |
| Usable without coding | No | Yes (paid GUI) | **Yes (menu, web app, YAML)** |
| Reproducible run (fixed seeds, versioned output folder) | Up to you | Possible through scripting | **Default** |

<!-- guaraci:tecnicas:inicio -->
**Measurement techniques:** **FT-NIR**, **Dispersive NIR**, **MIR/FTIR**, **Raman Spectroscopy**, **UV-Vis**, **Molecular Fluorescence**, **HPLC**, **GC-MS**, **NMR**, **IMS**, **3D fluorescence**, **ICP-OES / ICP-MS**, **Digital image**, **Hyperspectral imaging**, others (`generico` profile).

**Tasks:**
- See how the data is organised, without training a model
- Say which class (species, origin, producer) each sample belongs to
- Know whether a sample is authentic (pure x adulterated)
- Estimate the content of an analyte or adulterant
- Test whether groups differ on a response
- Track drift of a process or instrument
- Analyse photos or hyperspectral cubes

**Ready synthetic scenarios** (`guaraci exemplo <scenario>`): `explorar`, `classificar`, `quantificar`, `multielementar`, `comparar`, `imagem`, `hsi`, `transferir`, `monitorar`, `fluorescencia_eem`, `cromatograma_gcms`, `vip`.
<!-- guaraci:tecnicas:fim -->

The map technique × task × command × synthetic example × test is generated from the
catalogue in [`docs/qualidade/mapa_tecnica_tarefa.md`](docs/qualidade/mapa_tecnica_tarefa.md).
Every document is indexed in [`docs/projeto/INDICE.md`](docs/projeto/INDICE.md).

**Technique-agnostic by construction.** What depends on the measurement — spectral
range, default preprocessing, axis unit, the *vocabulary* of the output — lives in a
**profile** (a YAML file), never in the source. A technique with no registered profile
**fails with an actionable message instead of predicting**.

---

## Group-aware validation, enabled by default

Each sample is measured in **replicates** (T1/T2/T3). Letting those replicates fall on
both sides of a train/test split inflates accuracy (leakage). A `mae_id` group key keeps
them on the **same side** (`StratifiedGroupKFold` / `GroupShuffleSplit`), in
cross-validation and in the external holdout. Other tools can be configured to do this;
here it is the default, not an option you have to know to turn on.

## Features

- **Analysis modes:** class discrimination · pure vs. adulterated discrimination · content quantification (regression).
- **Preprocessing** (Rinnan et al. 2009 order): MSC/SNV → Savitzky-Golay → mean-centering. Presets: `MSC+SG+MC`, `SNV+SG+MC`, `Autoscaling`, `Mean-centering`.
- **Models:** PLS-DA, PLS regression, PCA, HCA (Ward), DD-SIMCA (one-class), OPLS-DA.
- **Validation battery:** permutation test, Wold R²Y/Q²Y intercepts, CV-ANOVA, **BCa bootstrap CIs**, Hotelling T², Q-residuals, lack-of-fit F-test, and a robustness protocol reported as an interval (never pass/fail).
- **Interpretation and selection:** VIP with fold-wise stability (`guaraci vip`, which shows stability and does not reduce variables), Selectivity Ratio, iPLS, sPLS-DA; variable selection in `guaraci run` is nested inside the validation so it cannot leak.
- **Group comparison on the independent unit** (`guaraci anova`): Tukey–Kramer, Games–Howell, Kruskal–Wallis with Dunn, assumption diagnostics, FDR across variables.
- **Blind mode (default):** an unknown sample is classified, then quantified *from the predicted class* — never the true one — with conformal-calibrated open-set identification that reports a coverage guarantee instead of a bare label.
- **Conformal prediction sets** for classification and regression; `NOT VALIDATED` instead of a fabricated set when the holdout lacks independent groups.
- **Calibration transfer** (DS, PDS) and **calibration-set selection** (Kennard-Stone, Duplex, SPXY).
- **Process monitoring** (`guaraci monitorar`): PCA with T²/Q control limits and a drift sentinel across batches.
- **Experimental planning:** sample-size guidance, collection-session plan with randomized reading order, and a design audit that flags class/session confounding.
- **Multi-way and chromatography tools:** PARAFAC for excitation-emission matrices, MCR-ALS, retention alignment (COW).
- **Multiblock fusion** (proof of concept, with a guard against misaligned blocks), **element-profile analysis** (censoring, internal standard, QC).
- **Images and hyperspectral cubes:** per-photo colour statistics, and per-pixel HSI classification with object aggregation. Both state which grouping guarantee the data supports.
- **Instrument readers** (OPUS, SPC, PerkinElmer `.sp`, Bruker NMR processed spectra, Agilent/Waters chromatograms, ANDI-MS, excitation-emission files) convert one instrument file to the same `(X, Y)` pair the pipeline consumes.
- **Portable model export** (JSON, no pickle) for the supported presets.

## Install

**Never installed a Python program? Follow these steps** (Windows; on macOS/Linux replace `.venv\Scripts\activate` with `source .venv/bin/activate`):

1. Install **Python 3.12** from <https://www.python.org/downloads/> and tick *"Add Python to PATH"*.
2. Download GUARACI (green **Code → Download ZIP** button on GitHub) and unzip it into a short folder name.
3. Open a terminal in that folder and run, one line at a time:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[web,reports]"
   guaraci doctor --instalar
   guaraci demo
   ```
4. Browser interface: `guaraci web`. Terminal menu: `guaraci`.

There is no PyPI package yet. **Do not run `pip install guaraci`**: that name belongs to an unrelated project (it installs a different program that also provides a `guaraci` command). Install from a clone, as below, **from inside the cloned folder** (do not install by name from PyPI). Python 3.11+; the code lives in the `guaraci` package under `src/`. macOS is not a supported or verified platform in this release.

Step-by-step guide for Windows (including the PowerShell execution-policy message), Linux (`python3.x-venv`) and WSL, with minimum requirements: [`docs/usuario/instalacao.md`](docs/usuario/instalacao.md) (in Portuguese). On a light-background terminal, set `GUARACI_TERMINAL=claro` so the colours stay readable.

```bash
pip install -e .        # the `guaraci` package + core dependencies
# or, for the full web/reports/benchmark stack:  pip install -e .[all]
```

The base install does **not** include the readers for Bruker OPUS (`opus`), Thermo/Galactic SPC (`spc`), Bruker NMR folders (`rmn`) and Agilent/Waters HPLC/GC folders (`hplc`), nor PARAFAC (`multiway`), image texture (`imagem`) or the DD-SIMCA PCV diagnostic (`robusto`). Screen [2.4] marks those formats with `(extra: pip install ...)` when the package is missing, and `guaraci doctor` / `guaraci checar` list what is missing and what it affects. To get everything at once:

```bash
pip install ".[all]"   # run inside the cloned folder; for development: pip install -e ".[all]"
```

Not sure what is missing? `guaraci doctor` lists it, and `guaraci doctor --instalar` installs it after you confirm (add `--sim` to skip the question). It installs the web app, the report writers and every reader that exists for your Python, with the version ranges of the installed package, so it also works from a clone. The benchmark extra (`xgboost`, `shap`, large) is left out; ask for it with `pip install -e ".[benchmark]"`.

**First steps, no data of your own required:**

```bash
guaraci doctor    # checks Python/RAM/CPU/dependencies
guaraci demo      # runs the full pipeline on synthetic spectra
guaraci perfis    # lists the available profiles
guaraci --help    # full option list and exit codes
guaraci exemplo   # lists the guided synthetic scenarios; `guaraci exemplo <scenario>` builds one
guaraci run       # runs the pipeline once from ./config.yaml and exits (automation and CI)
guaraci ajuda     # explains any option, technique, plot or screen (`guaraci buscar TERM` searches all)
guaraci assistente "your question"   # offline answer from the catalogue; `guaraci confianca <method>` shows seal and evidence
guaraci --version
```

Also: `anova`, `vip`, `monitorar`, `transferir`, `relatorio` — each is described by `guaraci --help`.

## Use (3 ways, no code editing)

Config lives in `config.yaml` (plain language). Start from the template:

```bash
cp config.example.yaml config.yaml   # then set `pasta_dados` to your data folder
```

```bash
guaraci                                       # 1. terminal interface (recommended)
python -m guaraci.pipeline --rodar            # 2. run straight from config.yaml
guaraci web                                   # 3. web app (browser, local mode)
```

`guaraci web` opens the web app in **local mode** (free paths, `.joblib` models, bound to 127.0.0.1). Plain `streamlit run app_quimiometria.py` starts the **server mode** (upload only, isolated session folder, portable JSON model only) — the safe default for anything reachable from a network. See [`SECURITY.md`](SECURITY.md).

## Profiles — how you switch techniques

A profile (`src/guaraci/perfis_matriz/*.yaml`) holds axis range and unit, default preprocessing, expected working range and the output vocabulary. The engine never reads those terms to decide anything.

```bash
guaraci perfis                              # lists every registered profile (8 shipped)
guaraci --perfil=nir_dispersivo_1100_2498nm  # or the path to your own YAML
```

Writing a profile for a new technique is copying `generico.yaml` and filling it in — no source change, no fork. An unregistered profile raises `PerfilDesconhecidoError` **before any data is loaded**, listing what exists. `visnir_400_1100nm` is **declared but not validated against real data**; the YAML says so itself.

## Blind mode is the default

Whoever sends a sample to a quantification model does **not** know its class, so by default the per-class regression uses the **predicted** class. `guaraci --modo=controle` uses the true class and exists only to separate quantification error from classification error while developing; the output labels such numbers as not representative. A test poisons the true labels and checks that blind-mode results do not change.

## Input and output

Root folder with **one subfolder per class** of `.dx` files, or a table (CSV/XLSX), image folder, cube folder or instrument files. Class and metadata are parsed from the JCAMP-DX `##TITLE=` field; the replicate tag `T{N}` becomes the `mae_id` leakage guard. The spectral range comes from the active profile.

**Provenance never travels with your results.** The parser reads only the fields it needs (never `##AUDIT TRAIL`), and sample identifiers are stripped from `metadados.csv` before it is written. A test scans every generated artifact, including the `.joblib` bytes.

Each run writes to `resultados_guaraci/{sample}/{Mode}/PLSDA_OE_{level_slug}_{preproc}_{timestamp}/` with `Graficos/`, `Tabelas/`, `Relatorios/` (`resumo_modelo.txt`, `model_card.md`) and `Modelos/`.

## Validation

Two independent layers.

**1. Against the defining formulas.** PLS-DA matches `sklearn.PLSRegression` + argmax exactly; SNV, VIP, MSC and CV-ANOVA match their definitions to numerical tolerance; the DD-SIMCA limit matches the closed forms; the OPLS-DA orthogonal component is orthogonal to < 1e-6. Table and tolerances: [`docs/cientifico/validacao_numerica.md`](docs/cientifico/validacao_numerica.md).

**2. Default values.** Thresholds and default parameters are starting points taken from the cited literature, not universal values, and can be changed through each technique's options (`guaraci confianca` shows the evidence behind each technique's seal). GUARACI does not distribute third-party data; the test suite uses synthetic data generated by the project's own code.

Quantification never reports a bare RMSEP: **SEP, RPD and RER** ship alongside it; LOD/LOQ return `N/A` when replicates cannot estimate instrumental noise.

### Reading the numbers

GUARACI measures with an **external holdout**, **physical-group protection** and **formal statistical coverage**, and reports the result even when it is less flattering than a naive split would give. A model that looks great under random K-fold on correlated replicates and collapses on a truly unseen batch is a false positive dressed as a result; this project chooses to show that collapse. Weak and null results are reported too.

## Security

Loading a `.joblib` model executes arbitrary code (it is a pickle) — see [`SECURITY.md`](SECURITY.md). Every load in the CLI and web app goes through a single guard that refuses to run without explicit confirmation, plus a SHA-256 manifest that blocks loading if the file was tampered with after export.

## Known limitations

The full, evidence-backed list is in [`docs/cientifico/limitacoes.md`](docs/cientifico/limitacoes.md). The main ones:

- **Verified technique by technique.** "Technique-agnostic" is an architectural property, not a validation result: a profile you write for a new technique is untested until you test it.
- **The result depends on the independent unit.** If the replicate is treated as the unit, significance and accuracy are inflated; the programs say which unit they used.
- **One-class DD-SIMCA needs enough *physical* samples per class**, not spectra; with one sampling point per class the limits rest on one observation, and the software says so.
- **Small *n*:** the main metrics ship with confidence intervals, and conformal prediction refuses to produce a set when n is insufficient.
- **Image mode without a grouping guarantee is a prototype** and the report stamps that.
- **HSI** works with your own cubes offline and is still a prototype: treat its performance on your technique as untested until you test it.
- **macOS** is not supported or verified in this release; Docker is not verified.
- **The assistant** answers from the built-in catalogue (offline search); it does not generate free-form answers.
- **Interface language:** the terminal and the web app run in English or Portuguese; a few folder and column names (`dados`, `classe`) stay in Portuguese because they are what you type.

## Author

**Erley Silva da Costa** — Researcher / Developer · [GitHub](https://github.com/ErleySC) · [ORCID](https://orcid.org/0009-0005-9655-6349)

GUARACI is an individual project: no team, company or institution is behind it ([`GOVERNANCE.md`](GOVERNANCE.md)). It was developed with the assistance of AI coding assistants; the author reviewed and is responsible for all code and documentation.

## License & citation

Licensed under the **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`, [LICENSE](LICENSE)) — Copyright (C) 2026 Erley Silva da Costa. Anyone, including companies, may use, modify and redistribute the software provided copyright notices are kept and any distributed derivative is released under the GPLv3 with its source. The software is distributed under these terms only.

Machine-readable metadata in [`CITATION.cff`](CITATION.cff). There is no archival DOI at present; cite the repository and a tagged version.

**APA**

> Costa, Erley Silva da. (2026). *GUARACI: Chemometrics platform with leakage-safe validation by default*
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

## Contributing and support

Bug reports, feature requests and pull requests are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Questions go to [GitHub Discussions](https://github.com/ErleySC/guaraci/discussions) ([`SUPPORT.md`](SUPPORT.md)); security reports go through the private channel in [`SECURITY.md`](SECURITY.md), never a public issue.
