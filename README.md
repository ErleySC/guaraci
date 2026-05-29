# FT-NIR Chemometrics Pipeline — Authentication of Amazonian Vegetable Oils

> 🇬🇧 English (lean). • 🇧🇷 [Versão completa em português](README.pt-br.md)

A free, reproducible **chemometrics pipeline** for **classification and
authentication** of vegetable oils by **FT-NIR spectroscopy**, with
**replicate-leakage-safe (group-aware) validation** — the project's
methodological differentiator.

Built for an undergraduate thesis (Chemistry, UFPA) on Amazonian oils measured
on an **ABB MB3600** (JCAMP-DX `.dx` files).

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

- **Levels:** N1 species (14 classes) · N2 pure vs. adulterated · N3 adulterant-content regression.
- **Preprocessing** (Rinnan et al. 2009 order): MSC/SNV → Savitzky-Golay → mean-centering. Presets: `MSC+SG+MC`, `SNV+SG+MC`, `Autoscaling`, `Mean-centering`.
- **Models:** PLS-DA, PLS regression, PCA, HCA (Ward), DD-SIMCA (one-class), OPLS-DA.
- **Validation battery:** permutation test, Wold R²Y/Q²Y intercepts, CV-ANOVA, **BCa bootstrap CIs**, Hotelling T², Q-residuals.
- **Interpretation:** VIP (Chong & Jun 2005), Selectivity Ratio (Rajalahti 2009).
- **Variable selection:** iPLS, VIP cutoff, SR top-fraction, sPLS-DA.

---

## Install

Python 3.10+.

```bash
pip install -r requirements.txt
```

## Use (3 ways, no code editing)

Config lives in `config.yaml` (plain language). Start from the template:

```bash
cp config.example.yaml config.yaml   # then set `pasta_dados` to your .dx folder
```

```bash
python pineline_quimiometria_14.py            # 1. terminal wizard (CMD-style)
python pineline_quimiometria_14.py --rodar    # 2. run straight from config.yaml
streamlit run app_quimiometria.py             # 3. web app (browser)
```

## Input

Root folder with **one subfolder per species** of `.dx` files. Class and
metadata are also parsed from the JCAMP-DX `##TITLE=` field. Triplicate tag
`T{N}` becomes the `mae_id` leakage guard. Useful range: **4000–10000 cm⁻¹**.

## Output

Each run writes a versioned folder
`resultados_tcc/PLSDA_OE_{level}_{preproc}_{timestamp}/` with `dados/`,
`figuras/`, `modelos/` (final `.joblib`), and `logs/resumo_modelo.txt`.

## Known limitations

- **Babaçu vs. Palmiste** overlap — both are palms with near-identical NIR
  signatures (a botanical/chemical limit of FT-NIR, not a code bug).
- **DD-SIMCA** one-class `puros` mode needs ≥15 pure samples/class; current data
  has ~3, so the default `todos` mode is exploratory.
- Small *n* → all metrics ship with **confidence intervals** (BCa).

## License & citation

[MIT](LICENSE) — © 2026 Erley S. da Costa.

> da Costa, E. S. *FT-NIR chemometrics pipeline for authentication of Amazonian
> vegetable oils with group-aware validation.* B.Sc. thesis, Chemistry, UFPA, 2026.
