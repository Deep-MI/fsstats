# GLM, univariate

*Part 1 of 4 in the [fsstats tutorial](../TUTORIAL.md). See also: [GLM, mass-univariate](glm_massunivariate.md).*

## What this analysis is for

We start with the simplest version of our [running example](../TUTORIAL.md#the-running-example): comparing baseline hippocampal volume between the MCI and CN groups, using one visit (baseline) per subject. Because each subject contributes a single, independent observation, an ordinary least squares (GLM) model is appropriate — `fsstats_lme`'s clustered/repeated-measures machinery isn't needed here. This is the same kind of comparison FreeSurfer's `mri_glmfit` is used for.

## Inputs you need

- `--table`: a table of baseline hippocampal volumes, one row per subject, as produced by `asegstats2table`. Only baseline-visit rows are included.
- `--X`: the design matrix (below).
- `--C`: one contrast file (below).
- `--glmdir`: an output directory for `fsstats_glm` to write results into.

See [DOCUMENTATION.md](../DOCUMENTATION.md#input-data-and-format-specifications) for the exact expected file format of each.

## Building the design matrix and contrast

For this cross-sectional comparison, the design matrix has one row per subject and three columns:

| Column | Meaning |
|---|---|
| 1 | Intercept (all 1s) |
| 2 | Group (0 = CN, 1 = MCI) |
| 3 | Age at baseline (covariate) |

The question "does hippocampal volume differ by group, after accounting for age?" is a test of column 2, so the contrast file is a single row selecting that column:

```text
0 1 0
```

## Running the command

```bash
fsstats_glm \
--table /my/input/directory/baseline_hippocampus_table.txt \
--X /my/input/directory/baseline_design_matrix.txt \
--C /my/input/directory/group_contrast.txt \
--glmdir /my/output/directory/glm_univariate
```

## Understanding the output

`fsstats_glm` writes results into `/my/output/directory/glm_univariate`:

- `beta.txt` — the fitted regression coefficients (intercept, group effect, age effect).
- `Xg.dat` — the design matrix actually used, for reference.
- A subdirectory named after the contrast file (e.g. `group_contrast/`), containing:
  - `gamma.txt` — the estimated group effect (contrast applied to the coefficients).
  - `F.txt` — the F-statistic for the group effect.
  - `sig.txt` — the signed sig-value (`sig = -log10(p)`, sign matching the direction of the effect since this is a single-row, t-test-like contrast).
  - `z.txt` — the corresponding signed z-value.

A `sig` of, say, 2.0 corresponds to p = 0.01. See [DOCUMENTATION.md](../DOCUMENTATION.md#output-files-and-their-interpretation) for the full output reference and the sig/p conversion table.

Next: [GLM, mass-univariate](glm_massunivariate.md), which asks the same kind of question but across the whole cortical surface instead of a single region.
