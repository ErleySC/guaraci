---
title: 'GUARACI: An open, reproducible multi-technique chemometrics platform with leakage-safe validation'
tags:
  - Python
  - chemometrics
  - spectroscopy
  - FT-NIR
  - PLS-DA
  - SIMCA
  - food authentication
  - multivariate analysis
authors:
  - name: Erley S. da Costa
    orcid: "0009-0005-9655-6349"
    affiliation: 1
affiliations:
  - name: Grupo de Espectroscopia Analítica Aplicada (GEAAp), Universidade Federal do Pará (UFPA), Brazil
    index: 1
date: 12 July 2026
bibliography: paper.bib
---

# Summary

`GUARACI` is an open-source Python platform for chemometric classification,
authentication, and quantification of complex sample matrices. It targets
vibrational (FT-NIR, NIR, MIR, Raman, UV-Vis), luminescence, chromatographic
(HPLC, GC-MS), and resonance (NMR, IMS) data, and implements the standard
multivariate toolkit used in analytical chemistry — PLS-DA, OPLS-DA
[@TryggWold2002], PLS regression, PCA, hierarchical clustering, and DD-SIMCA
one-class modelling [@PomerantsevRodionova2014] — together with the
diagnostics and validation battery that turn a fitted model into a defensible
scientific claim: VIP [@ChongJun2005] and Selectivity Ratio
[@Rajalahti2009] for variable importance, Hotelling T² and Q-residuals for
outlier and applicability-domain detection, permutation testing, Wold
R²Y/Q²Y intercepts, CV-ANOVA, BCa bootstrap confidence intervals, and
Martens' uncertainty test for PLS coefficients [@MartensMartens2000]. Two
user-facing interfaces — a bilingual (Portuguese/English) guided terminal
application and a Streamlit web app — expose the full pipeline without
requiring the user to write code, while a shared Python package
(`src/guaraci`) keeps the scientific logic identical across both.

`GUARACI` was originally developed for an undergraduate thesis on the FT-NIR
authentication of Amazonian vegetable oils, and has since been generalised
into a matrix- and technique-agnostic platform.

# State of the field

Several existing packages address parts of the same problem space.
`mdatools` [@Kucheryavskiy2020] is a mature R package offering PCA, PLS,
PLS-DA and SIMCA with a broad validation toolkit, but reproducibility and
cross-validation splitting strategy are left entirely to the user's script,
and there is no distinct interface layer for users who do not write code.
`hyperSpec` [@BeleitesSergo] targets a narrower problem: representing and
manipulating hyperspectral data structures in R, without a built-in
modelling or validation layer. `pyChemometrics` [@Correia] implements
PCA, PLS and PLS-DA for NMR and mass-spectrometry metabolomics in Python,
but is a research codebase without a packaged CLI or web interface, and,
like `mdatools`, does not surface a first-class, guided option to keep
physical replicates on the same side of a cross-validation split.
`scikit-learn` [@Pedregosa2011] supplies general-purpose primitives
(including `GroupKFold`) that any of these packages, or a user's own
script, could combine to prevent replicate leakage — but doing so is
opt-in and requires the user to already know the risk exists; to the
author's knowledge, none of the surveyed domain-specific chemometrics
packages document group-aware splitting as a default or a guided setting.
`GUARACI`'s contribution is not a new statistical method but making this
protection, together with the accompanying chemometric diagnostics (VIP,
Selectivity Ratio, Hotelling T²/Q-residuals, DD-SIMCA sensitivity), the
default path for a user who does not write code.

# Statement of need

Laboratories that need rigorous multivariate analysis of spectral or
chromatographic data today face a choice between two unsatisfactory options.
General-purpose libraries such as scikit-learn [@Pedregosa2011] provide the
underlying algorithms but not the chemometric diagnostics
(VIP, Selectivity Ratio, Hotelling T²/Q-residuals, DD-SIMCA, OPLS-DA) that
analytical chemists rely on to interpret and validate a model, leaving users
to reimplement well-established statistics from scratch. Commercial suites
(e.g. The Unscrambler, SIMCA-P, PLS_Toolbox) provide those diagnostics but are
closed-source, expensive, and not reproducible or scriptable in an open
research pipeline.

`GUARACI` fills this gap. Its main methodological differentiator is
**group-aware (leakage-safe) validation**: samples measured in physical
replicate are tagged with a group key (`mae_id`) that is honoured by every
cross-validation and hold-out split (`StratifiedGroupKFold` /
`GroupShuffleSplit`), preventing replicates of the same physical sample from
appearing on both sides of a split — a common and under-reported source of
inflated accuracy in spectroscopy studies. This same principle governs
one-class (DD-SIMCA) sensitivity: it is estimated by leave-one-group-out
cross-validation rather than by re-substitution on the training set, and the
number of groups used is always reported alongside the estimate so that
users can judge its reliability instead of trusting a single inflated
percentage. Every run also produces a versioned, human- and
machine-readable record: figures of merit (LOD, LOQ, sensitivity,
selectivity), an automatically generated model card [@Mitchell2019]
documenting intended use and limitations, and a fixed random seed, so that a
reported result can be independently reproduced from the same configuration
file. Saved models are accompanied by a SHA-256 integrity manifest, and
loading a model requires an explicit trust flag, since deserialising a
`.joblib` file executes arbitrary code — a risk that is otherwise easy to
overlook when sharing pretrained models between labs.

The software is aimed at two audiences: academic researchers who need
citable, reproducible chemometric analysis without a commercial license, and
quality-control laboratories that need the same rigor with an auditable
trail. Its input/output layer is deliberately generic (JCAMP-DX and tabular
formats), so it applies to matrices and analytical techniques beyond the one
that motivated it, without code changes. The codebase is covered by an
automated test suite (550+ tests) and continuous integration (linting,
coverage gate), and each implemented method is checked against a reference
implementation or a closed-form analytical property (documented in
`docs/VALIDATION.md`), so contributions and future chemometric methods can
be added without regressing existing behaviour. Beyond internal validation,
the preprocessing and PLS regression engine has been benchmarked against
Tecator, a public NIR dataset unrelated to the authors' own data
[@Thodberg1996], reproducing RMSEP/R² in the range reported in the
chemometrics literature for this dataset (`docs/BENCHMARK_TECATOR.md`).

# Acknowledgements

The author thanks the Grupo de Espectroscopia Analítica Aplicada (GEAAp) at
Universidade Federal do Pará (UFPA) for the FT-NIR data and infrastructure
that motivated this work.

# References
