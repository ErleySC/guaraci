---
title: 'GUARACI: Chemometrics platform with leakage-safe validation by default'
tags:
  - Python
  - chemometrics
  - spectroscopy
  - FT-NIR
  - hyperspectral imaging
  - PLS-DA
  - SIMCA
  - food authentication
  - multivariate analysis
authors:
  - name: Erley Silva da Costa
    orcid: "0009-0005-9655-6349"
    affiliation: 1
affiliations:
  - name: Independent Researcher, Brazil
    index: 1
date: 15 October 2026
bibliography: paper.bib
---

# Summary

`GUARACI` is an open-source Python platform for chemometric classification,
authentication and quantification of spectral, chromatographic and
hyperspectral-imaging data (FT-NIR, NIR, MIR, Raman, UV-Vis, fluorescence,
HPLC, GC-MS, NMR, ENVI cubes). It implements the standard multivariate toolkit
of analytical chemistry --- PLS-DA, OPLS-DA [@TryggWold2002], PLS regression,
PCA and DD-SIMCA one-class modelling [@Pomerantsev2008; @Kucheryavskiy2024] --- with the
diagnostics that turn a fitted model into a defensible claim: VIP
[@ChongJun2005], Selectivity Ratio [@Rajalahti2009], Hotelling T² and
Q-residuals, permutation tests, CV-ANOVA, BCa bootstrap intervals and
Martens' uncertainty test [@MartensMartens2000]. A bilingual terminal
assistant and a Streamlit web app expose the same package (`src/guaraci`)
without requiring code. Matrix-specific knowledge (spectral range,
preprocessing, working range) lives in YAML **matrix profiles**, so changing
matrix is a configuration change; a matrix without a profile fails before any
data is read.

# Statement of need

Spectroscopy studies routinely measure the same physical sample several
times. If replicates of one sample fall on both sides of a split, accuracy is
inflated --- a common, under-reported error. Laboratories without coding
staff have, in practice, no guided way to avoid it.

`GUARACI` makes **group-aware validation** the default. Every split
(`StratifiedGroupKFold`, `GroupShuffleSplit`, leave-one-group-out for
Q-residuals and DD-SIMCA sensitivity) honours a group key, and the run states
how strong that guarantee is: *high* (identifiers parsed from instrument file
names, checkable), *medium* (a group column or file declared by the user, which
the software cannot verify) or *none* (no key, so metrics are marked
exploratory). The independent unit is configurable --- sample or collection
session --- because replicates of one sample are not independent of the
session in which they were measured. A design audit flags class/session
confounding and single-class data, and a hold-out reserved before any tuning
choice is reported separately, for classification and quantification alike.
Runs write a model card [@Mitchell2019], the effective parameters (hardware
never changes a scientific parameter silently) and a fixed seed.
Saved models are verified by an HMAC-signed manifest before deserialisation,
because loading a `.joblib` file executes code; the web app defaults to a
server mode that accepts only uploads, isolated per session, and portable
JSON models.

Two further defaults follow from the end-user's position. Per-class
quantification is calibrated on the *predicted* class, since a user
submitting an unknown sample does not know its class; the label-aware path
exists behind an explicit flag and is marked in every artifact. An open-set
identification step matches an unknown sample against conformal-calibrated
species-by-adulterant combinations, and blocks quantification when the
coverage guarantee could not be validated on held-out sessions. **This
open-set component has not been demonstrated on independent data**; where it
has been used, most combinations were not validable, and the software reports
that status instead of a number.

# State of the field

Mature open-source options exist. `mdatools` [@Kucheryavskiy2020; @MdatoolsDocs]
(R) covers PCA, PLS, PLS-DA and SIMCA with resampling validation (including
user-supplied segment vectors), and also DD-SIMCA (with a three-way variant),
MCR (ALS and purity-based), interval PLS (iPLS), a randomisation test and
Procrustes cross-validation, with companion web applications for DD-SIMCA and
PLS. `chemotools` [@Lopez2024] provides scikit-learn-compatible preprocessing
and chemometric estimators in Python, including direct and piecewise direct
standardisation (DS, PDS) for calibration transfer
[@ChemotoolsPyPI], and `hoggorm` [@Tomic2019] and `mbpls`
[@Baum2019] cover explorative and multiblock PLS. `pyChemometrics`
[@Correia] targets metabolomics; `hyperSpec` [@BeleitesSergo] (R),
`Spectral Python` and the MIA Toolbox handle hyperspectral data.
`scikit-learn` [@Pedregosa2011] supplies `GroupKFold` and related splitters.
Commercial suites (The Unscrambler, SIMCA, PLS_Toolbox) are closed-source;
some, such as PLS_Toolbox on MATLAB, are scriptable.

Every one of these lets a careful user keep physical replicates on the same
side of a split; to the author's knowledge none makes that the guided default
for a user who does not write code, and none verifies how trustworthy the
grouping is. `GUARACI` implements DD-SIMCA, calibration transfer and
preprocessing as well, but they are not what distinguishes it: it positions
itself on the workflow around them, namely an automatic report (model card,
effective parameters, seed), conformal prediction for open-set identification,
and grouped validation as a guided default. Its DD-SIMCA implementation is only
partially verified (`guaraci confianca`), whereas `mdatools` is the mature,
published reference for that method. `GUARACI` was built rather than contributed to an existing
package because the contribution is an end-to-end *workflow default* (data
loading, validation, reporting, prediction), not a single estimator that
would fit an existing scikit-learn-style API. Reuse where possible remains
the goal: comparing against, and where sensible depending on, `chemotools`
for preprocessing is future work.

# Software design

The scientific engine (`pipeline.py`) is separated from I/O (`dados_io.py`,
readers), statistics (`chemometric_stats.py`, `conformal.py`), reporting and
two thin front ends. Registries decouple techniques, readers and matrix
profiles from the engine, so adding one does not touch the pipeline. Design
choices trade convenience for defensibility: fail-closed model loading,
errors raised before a run rather than defaults applied silently, and no
global side effects on import. Cross-validation uses the package's own
splitter rather than scikit-learn's, because fold assignment changed between
scikit-learn versions and moved results. Compatibility is guarded by a
public-API golden test and property-based tests (Hypothesis) on the
invariants most exposed to regression: leakage, configuration round-trip and
blind-mode label handling. CI runs linting, type checking and tests on Linux
and Windows with Python 3.11--3.13 (a reduced matrix on pull requests, the full
one on pushes to the main branch). A macOS job also runs on those pushes, but
macOS is not a supported or verified platform in this release.

# Research impact statement

Default parameters and thresholds are starting points taken from the cited
literature, not universal values, and can be changed through each technique's options; the package and its test suite use
synthetic data generated by the project's own code, and numerical correctness
is checked against the defining formulas (`docs/cientifico/validacao_numerica.md`).
Quantification never reports a
bare RMSEP (SEP, RPD and RER accompany it), and LOD/LOQ return `N/A` when
replicates cannot estimate instrumental noise. A peer-review-style
audit of the code found and drove the correction of errors that inflated
validation (the reviewers were independent AI agents, not people). At the time of writing the software has
**no external users, releases or citations**; near-term impact is its use in the author's own analytical work.

# AI usage disclosure

Parts of the code, tests, documentation and this manuscript were produced with
the assistance of AI coding assistants (Anthropic's Claude models, via Claude
Code), under the author's direction. The author chose the scientific scope and methods and reviewed the results;
correctness rests on the automated tests, the checks against the defining
formulas and the audit above. Bibliographic references were once assistant-suggested and
some proved wrong; all DOIs are now checked automatically against
Crossref/DataCite (`scripts/verificar_citacoes.py`). The author is responsible
for the content.

# Acknowledgements

The author thanks the maintainers of the open-source scientific Python stack on
which `GUARACI` is built (see `ACKNOWLEDGMENTS.md`).

# References
