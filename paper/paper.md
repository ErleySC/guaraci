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
  - name: Erley S. da Costa
    orcid: "0009-0005-9655-6349"
    affiliation: 1
affiliations:
  - name: Independent Researcher, Brazil
    index: 1
date: 12 July 2026
bibliography: paper.bib
---

# Summary

`GUARACI` is an open-source Python platform for chemometric classification,
authentication, and quantification of complex sample matrices. It targets
vibrational (FT-NIR, NIR, MIR, Raman, UV-Vis), luminescence, chromatographic
(HPLC, GC-MS), resonance (NMR, IMS), and hyperspectral-imaging (ENVI cubes,
per-pixel classification with physical-object aggregation) data, and
implements the standard
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

Matrix-specific knowledge is externalised into **matrix profiles** (YAML):
spectral range and unit, default preprocessing, expected working range, and
the vocabulary used in reports. Switching from one matrix to another is a
configuration change, not a code change, and a matrix with no registered
profile raises an error before any data is loaded rather than predicting
under another matrix's assumptions.

# State of the field

Several existing packages address parts of the same problem space.
`mdatools` [@Kucheryavskiy2020] is a mature R package offering PCA, PLS,
PLS-DA and SIMCA with a broad validation toolkit, but reproducibility and
cross-validation splitting strategy are left entirely to the user's script,
and there is no distinct interface layer for users who do not write code.
`hyperSpec` [@BeleitesSergo] targets a narrower problem: representing and
manipulating hyperspectral data structures in R, without a built-in
modelling or validation layer. `GUARACI`'s hyperspectral-imaging (HSI)
mode addresses that gap directly: a generic ENVI-cube reader (no public
dataset required — any folder of the user's own cubes, organised one
subfolder per class, works offline), the same physical-group leakage
protection as the platform's other modes, per-pixel PLS-DA with
object-level aggregation, and a spatial classification map. Reported
performance on it is modest in places, honestly, on the public fixture
used for the project's own validation — a matrix-agnostic engine is an
architectural property, not a validation result; a profile or dataset
a user brings is untested until they test it.
`pyChemometrics` [@Correia] implements
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

A second default follows the same logic. In the surveyed packages, per-class
quantification calibrates on the analyst's known class labels, because that
is the natural thing to write in a script. But an end user submitting an
unknown sample does not know its class — so a figure of merit obtained with
the true label describes a situation the user will never be in, and silently
folds classification error out of the reported quantification error.
`GUARACI` calibrates on the *predicted* class by default; the label-aware
path exists, is reached only through an explicit `--modo=controle` flag, and
is marked as such in every artifact it produces. The same blind-prediction
path also drives an open-set identification step: an unknown sample's
adulterant is matched against a conformal-calibrated ensemble of
species-by-adulterant combinations, and quantification is only produced —
never a bare number — for a combination whose coverage guarantee was
actually validated on the training groups; an unmatched or statistically
unvalidated combination blocks the quantification step and reports why,
instead of returning a number with no error-rate guarantee behind it.

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
automated test suite (1000+ tests, including Hypothesis property tests for
the invariants most exposed to regression — grouped-validation leakage,
configuration-file round-tripping, blind-mode label handling) and
continuous integration (linting, type-checking, coverage gate) across
Linux, Windows and macOS on Python 3.10–3.13. Each implemented method is
checked against a reference implementation or a closed-form analytical
property (documented in `docs/VALIDATION.md`), so contributions and future
chemometric methods can be added without regressing existing behaviour.
Beyond the core modelling loop, `GUARACI` also covers the surrounding
workflow: experimental-design guidance and an automated audit that flags
class/session confounding before it inflates a metric, formal linearity
(lack-of-fit F-test [@DraperSmith1998]) and a robustness protocol
reporting result variation under perturbation as an interval rather than
a pass/fail verdict, and calibration transfer (Direct/Piecewise Direct
Standardization [@WangVeltkampKowalski1991]) alongside calibration-set
selection (Kennard-Stone, Duplex [@Snee1977], SPXY [@Galvao2005]) for
moving a model between instruments or picking a representative measured
subset. A drift sentinel accumulates the applicability-domain rejection
rate across successive batch-prediction runs and formally tests, via an
exact binomial test, whether that rate is rising above the nominal alpha —
the question that matters for continuous production use, as opposed to
flagging any single out-of-domain sample, which the nominal alpha already
allows for. A prototype image mode extends the same modelling and
diagnostic machinery to digital colorimetry (RGB/HSV/Lab statistics,
optionally GLCM texture) by treating each photograph as one spectrum-like
row, auto-detecting which replicate-grouping guarantee the data folder
supports and declaring it explicitly rather than assuming one.

Performance claims rest exclusively on **public datasets**. On the
Eigenvector *Corn* set (80 samples, 700 channels, 1100–2498 nm), `GUARACI`
predicts protein content with RMSEP = 0.144 %w/w, within the 0.1–0.2 range
reported in the literature for PLS on this benchmark; this runs as a
continuous-integration job, so the build fails if the engine stops
reproducing it. The preprocessing engine has additionally been benchmarked
against Tecator [@Thodberg1996], reproducing RMSEP/R² in the published range
(`docs/BENCHMARK_TECATOR.md`). Quantification output never reports a bare
RMSEP: SEP, RPD and RER accompany it, with RPD carrying its published
interpretation band, and LOD/LOQ return `N/A` rather than a number when
physical replicates are insufficient to estimate instrumental noise.

A related design decision concerns data provenance. Spectroscopic file
formats routinely embed operator name, site and instrument serial numbers in
their headers — information that is irrelevant to the analysis but travels
with the file. `GUARACI`'s parser reads only the numeric and label fields it
needs, never the audit-trail block, and strips sample identifiers from
exported metadata, replacing them with anonymous replicate-group labels that
preserve group-aware auditability. A test suite scans every generated
artifact, including serialised model bytes, and fails if such a field
leaks.

# Acknowledgements

The author thanks those who provided samples, instrument access and
spectral acquisition during the development of this software; see
`ACKNOWLEDGMENTS.md` in the repository. The author also thanks the
maintainers of the public datasets used for validation, and the maintainers
of the scientific Python stack on which `GUARACI` is built.

# References
