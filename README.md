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

Free tools already cover most of the methods: `mdatools` (R) ships PCA, PLS, PLS-DA and
DD-SIMCA with mature diagnostics, and scikit-learn has the group-aware splitters. What they
leave to the user is the workflow around the methods, and the leakage traps in it; the
established commercial suites guide that workflow behind a paid, closed license. GUARACI makes
the guided workflow free, under the GPLv3.

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

| | Open packages (`mdatools`, scikit-learn, `chemotools`) | Commercial chemometrics suites | **GUARACI** |
|---|---|---|---|
| Cost and license | Free, open | Paid, closed | **Free, open (GPLv3)** |
| Chemometric diagnostics (VIP, SR, Hotelling T², Q-residuals, DD-SIMCA, OPLS-DA) | Included in `mdatools` (R); partial in Python packages | Included | **Included** |
| Replicates kept on one side of every split | Available (`mdatools` segment vectors, `StratifiedGroupKFold`); you have to set it up | Available, set up by the user | **Default, holdout included** |
| Usable without coding | Mostly no (`mdatools` has web apps for DD-SIMCA and PLS) | Yes (paid GUI) | **Yes (menu, web app, YAML)** |
| Reproducible run (fixed seeds, versioned output folder) | Up to you | Possible through scripting | **Default** |

**When another tool is the better choice.** If you write R, `mdatools` is the mature, published
reference for PCA, PLS and DD-SIMCA; GUARACI's DD-SIMCA is only partially verified
(`guaraci confianca`). If you write Python and only need preprocessing or a single estimator,
`chemotools` and scikit-learn are smaller and better established. GUARACI is for running the
whole study (design audit, grouped validation, holdout, report) without writing code.

<!-- guaraci:tecnicas:inicio -->
**Measurement techniques:** **FT-NIR**, **Dispersive NIR**, **MIR/FTIR**, **Raman Spectroscopy**, **UV-Vis**, **Molecular Fluorescence**, **HPLC**, **GC-FID**, **GC-MS**, **LC-MS**, **NMR**, **IMS**, **3D fluorescence**, **ICP-OES / ICP-MS**, **XRF**, **Digital image**, **Hyperspectral imaging**, others (`generico` profile).

**Tasks:**
- See how the data is organised, without training a model
- Say which class (species, origin, producer) each sample belongs to
- Know whether a sample is authentic (pure x adulterated)
- Estimate the content of an analyte or adulterant
- Test whether groups differ on a response
- Track drift of a process or instrument
- Analyse photos or hyperspectral cubes

**Ready synthetic scenarios** (`guaraci exemplo <scenario>`): `explorar`, `classificar`, `quantificar`, `multielementar`, `comparar`, `imagem`, `hsi`, `transferir`, `monitorar`, `pares`, `fluorescencia_eem`, `cromatograma_gcms`, `vip`.
<!-- guaraci:tecnicas:fim -->

A technique on this list has a profile, a synthetic scenario and tests; that is **not** the same as having been
checked on real data from that technique. Which ones were is in [What has been run on real data](#what-has-been-run-on-real-data).

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
- **Models:** PLS-DA and PLS-LDA (`classificador: auto` picks PLS-LDA below 100 independent units; `plsda`/`plslda` fixes it), PLS regression, PCA, HCA (Ward), DD-SIMCA (one-class), OPLS-DA.
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
- **Instrument readers** (OPUS, SPC, PerkinElmer `.sp`, Bruker NMR processed spectra, Agilent/Waters chromatograms, ANDI-MS, excitation-emission files) convert one instrument file to the same `(X, Y)` pair the pipeline consumes. The public test suite uses only files generated from the format specification; real instrument files were read for a few formats only. See [What has been run on real data](#what-has-been-run-on-real-data).
- **Portable model export** (JSON, no pickle) for the supported presets.

## Install

It takes about 10 minutes, needs internet only for this step and **requires no programming**. Pick your system and follow the steps in order. On any of them, paste **one line at a time** and wait for it to finish before the next.

### Windows

1. Install **Python 3.12**: download it from <https://www.python.org/downloads/windows/> and, on the installer's first screen, tick **"Add python.exe to PATH"** before clicking *Install Now*.
2. Download GUARACI: on this GitHub page, green **Code → Download ZIP** button. Right-click the file and choose **Extract All** into a short folder name outside cloud-synced folders (for example, `C:\guaraci`).
3. Open the extracted folder (the one containing `pyproject.toml`), click the address bar, type `powershell` and press Enter. In PowerShell, run:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install ".[web,reports]"
   python -m guaraci doctor --instalar
   python -m guaraci demo
   ```

If PowerShell complains that "running scripts is disabled", run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and repeat the `Activate.ps1` line.

### macOS

1. Install **Python 3.12**: download the installer from <https://www.python.org/downloads/macos/> and open the `.pkg` file.
2. Download GUARACI: on this GitHub page, green **Code → Download ZIP** button, then double-click the file to extract it.
3. Open **Terminal** (Cmd + Space, type "Terminal"). Type `cd ` (with a trailing space), **drag the extracted folder onto the window** and press Enter. Then run:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install ".[web,reports]"
   python -m guaraci doctor --instalar
   python -m guaraci demo
   ```

macOS is not a supported or verified platform in this release: it has been verified only by automated tests, and nobody has really used the program there. If something fails, open a [Discussion](https://github.com/ErleySC/guaraci/discussions) with the error message.

### Linux (Ubuntu and Debian)

```bash
sudo apt update
sudo apt install python3 python3-venv unzip
unzip guaraci-master.zip && cd guaraci-master      # the ZIP from Code → Download ZIP
python3 -m venv .venv
source .venv/bin/activate
pip install ".[web,reports]"
python -m guaraci doctor --instalar
python -m guaraci demo
```

On other distributions, install the equivalent `python3-venv` package.

### After installing

In each new terminal session, open the GUARACI folder, activate the environment again (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on Mac and Linux) and use:

```bash
python -m guaraci web    # browser interface
python -m guaraci        # terminal menu
```

### Warnings

- **Do not run `pip install guaraci`.** That name on PyPI belongs to an unrelated project. Always install from inside the downloaded folder, as above.
- **Python version:** use 3.12. GUARACI requires 3.11 or newer; 3.11 to 3.13 are tested in CI. On 3.14 the program runs, but the Agilent/Waters `.D` folder reader does not install.
- Hit an error or stuck? See the [full installation guide](docs/usuario/instalacao.md) (in Portuguese: requirements, WSL, common error messages). On a light-background terminal, set `GUARACI_TERMINAL=claro`.

### Optional extras and development

The `python -m guaraci doctor --instalar` command above already installs what is missing (web app, report writers and the instrument readers available for your Python) after you confirm. Without an extra the program runs, but the matching format does not open: screen [2.4] marks it with `(extra: pip install ...)`, and `guaraci doctor` and `guaraci checar` list what is missing. Single readers: `opus`, `spc`, `rmn`, `hplc`; analyses: `multiway` (PARAFAC), `imagem`, `robusto` (DD-SIMCA PCV). `benchmark` (`xgboost` and `shap`, large) is left out: `pip install ".[benchmark]"` (on Mac, first: `brew install libomp`). `pip install ".[all]"` installs everything and on 3.14 ends **without** the HPLC/GC reader, silently.

To develop (from a clone): `pip install -e ".[all]"`. The code lives in the `guaraci` package under `src/`.

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

Also: `anova`, `vip`, `monitorar`, `pares`, `transferir`, `relatorio` — each is described by `guaraci --help`.

## Use (3 ways, no code editing)

Config lives in `config.yaml` (plain language). Start from the template:

```bash
cp config.example.yaml config.yaml   # then set `pasta_dados` to your data folder
```

```bash
guaraci                                       # 1. terminal interface (recommended)
guaraci run config.yaml                       # 2. run straight from config.yaml
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

Each run writes to `resultados_guaraci/{sample}/{Mode}/{PREFIX}_{level_slug}_{preproc}_{timestamp}/` (`PREFIX` is `PLSDA_OE` for classification, `PLSR_OE` for quantification and `PCA_OE` for exploration) with `Figures/`, `Tables/`, `Reports/` (`resumo_modelo.txt`, `model_card.md`) and `Models/` (in Portuguese: `Graficos/`, `Tabelas/`, `Relatorios/`, `Modelos/`).

## Validation

Two independent layers.

**1. Against the defining formulas.** PLS-DA matches `sklearn.PLSRegression` + argmax exactly; SNV, VIP, MSC and CV-ANOVA match their definitions to numerical tolerance; the DD-SIMCA limit matches the closed forms; the OPLS-DA orthogonal component is orthogonal to < 1e-6. Table and tolerances: [`docs/cientifico/validacao_numerica.md`](docs/cientifico/validacao_numerica.md).

**2. Default values.** Thresholds and default parameters are starting points taken from the cited literature, not universal values, and can be changed through each technique's options (`guaraci confianca` shows the evidence behind each technique's seal). GUARACI does not distribute third-party data; the test suite uses synthetic data generated by the project's own code.

### What has been run on real data

The test suite uses synthetic data only, and the project publishes no result obtained on third-party data. The
maintainer's own runs on real data covered NIR (regression and classification), Raman (regression) and a GC peak-area
table; in one NIR regression the external holdout failed (R² below 0), and the run now says so instead of calling the
number publishable. Those runs are evidence for the maintainer, not a validation you can cite.

**No result on real data has been checked for:** MIR/FTIR, UV-Vis, fluorescence and EEM, HPLC, GC-MS, LC-MS, raw GC-FID
chromatograms, NMR, IMS, ICP, XRF, digital image and the full hyperspectral pipeline. They work on synthetic data and
their readers follow the format specification; treat them as unvalidated until you check a sample you know.

**Readers checked with real public files** (reading only, no comparison with the vendor software): JCAMP-DX (official
test suite and public reference spectra), MATLAB `.mat` (126 files, v4 to v7.3), OPUS (4 files), ANDI-MS (8), Agilent/Waters (12
folders), one Agilent GC-FID `.D` folder. SPC, PerkinElmer `.sp`, Bruker NMR, Horiba EEM and an HPLC folder were read
from real instrument files only on the maintainer's machine (files not public; outcome in
[`docs/qualidade/rodada_local_formatos_reais.json`](docs/qualidade/rodada_local_formatos_reais.json)). CSV, Excel, `.mat` and JCAMP were also fed malformed files (decimal comma,
`;`/TAB separators, latin-1, UTF-16, blank rows, empty columns, NaN/Inf, truncated and empty `.dx`): each one is either
read, or refused with a message that says where the problem is. A file skipped inside a data folder is listed in
`ARQUIVOS_IGNORADOS.txt` and in the model card. Details: [`docs/cientifico/limitacoes.md`](docs/cientifico/limitacoes.md), section 6.

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
- **macOS and Docker are only checked by CI.** The full test suite passes on macOS (Python 3.11 and 3.13), and CI builds the Docker image and checks that the web app starts in server mode as a non-root user. Neither has been verified in real use, and analysis runs inside the container are not tested.
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
