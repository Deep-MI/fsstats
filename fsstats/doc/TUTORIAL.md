# Tutorial

```{toctree}
:maxdepth: 1
:hidden:

tutorial/glm_univariate
tutorial/glm_massunivariate
tutorial/lme_univariate
tutorial/lme_massunivariate
```

## Overview

Here we illustrate an example analysis with the `fsstats` command-line tools, `fsstats_glm` and `fsstats_lme`. 

The tutorial deliberately keeps things shorter and only explains what is needed to follow the example. For the full, authoritative description of every argument, input format, and output file, see the [documentation](DOCUMENTATION.md).

The tutorial covers command-line usage only. For using `fsstats` as a Python package or calling its low-level functions directly, see the sections for running the fsstats tools [as a Python package](documentation_as-a-python-package) and [as Python functions](documentation_as-python-functions).

___

## Contents

This tutorial provides walkthroughs for four analysis scenarios:

1. Cross-sectional analysis for extracted, tabular data ([GLM, univariate](tutorial/glm_univariate.md))
2. Cross-sectional analysis for whole-brain, vertex-based data ([GLM, mass-univariate](tutorial/glm_massunivariate.md))
3. Longitudinal analysis for extracted, tabular data ([LME, univariate](tutorial/lme_univariate.md))
4. Longitudinal analysis for whole-brain, vertex-based data ([LME, mass-univariate](tutorial/lme_massunivariate.md))

___

(tutorial_key-concepts)=
## Key concepts

The four walkthroughs share the concepts below. Read them once here; each section then only explains what's specific to that analysis.

### General linear and linear mixed-effects models

A **general linear model (GLM)** relates an outcome (e.g. a brain measure) to one or more predictors (e.g. group, age) through a set of fixed coefficients. These coefficients, also called beta estimates, quantify the contribution of each predictor in the prediction of the outcome in a model of the form `Y = beta_1 * X_1 + ... + beta_n * X_n + error`, where `Y` represents the outcome, `beta_1` to `beta_n` are the model coefficients, and `X_1` to `X_n` represent `n` predictors in the model. This model assumes that observations are independent of one another.

A **linear mixed-effects (LME) model** extends the GLM with *random effects*: in addition to the fixed, population-average coefficients, each case is allowed its own deviation from that average. This is what makes it possible to correctly analyze *clustered* data, such as repeated visits from the same person, where observations within a cluster are not independent. Ignoring that non-independence (e.g. by using a plain GLM on longitudinal data) tends to produce overconfident, artificially small p-values. LME also lets you directly ask questions a GLM cannot, such as whether individuals' *rates of change* differ by group.

### Model inputs: design matrix and contrasts

The **design matrix** contains -- in numerical form -- all variables that you want to use as predictors (or control variables) in your model. It typically contains an intercept (i.e., a column of ones) in addition to the variables of interest. The design matrix has one row per observation and one column per variable (e.g. intercept, group, age), and for each column / variable one beta estimate will be available after the model has been estimated. Categorical variables need to be recoded into one or more group indicator variables that can take values of either `0` or `1`; in models with a global intercept, categorical variables with `n` levels will result in `n-1` indicator variables, since one of the `n` groups serves as a reference group that will be modeled through the global intercept and therefore does not need its own indicator variable.

A **contrast vector** encodes the specific hypothesis that you want to test; it is a single row of numerical values (typically 0, -1, or +1) and has as many columns as the design matrix. The columns of the design matrix, the columns of the contrast vector, and the beta estimates all correspond to each other. A contrast selects those model parameters (beta estimates) that you want to test -- e.g. "is the group coefficient different from zero?" It then implements a linear combination of these parameters, resulting in the contrast estimate `gamma` -- a single number that represents an effect estimate and can be evaluated for statistical significance. A single-row contrast vector corresponds to a t-test that evaluates a single effect. 

A **contrast matrix** is a generalization of the contrast vector and serves the same purpose. Unlike the the contrast vector, it consists of two or more rows of numerical values (typically 0, -1, or +1), but still has as many columns as the design matrix. A multi-row contrast matrix behaves like a joint F-test across several effects at once: this can be useful if you want to test the overall effect of a group with more than 2 levels, for example to evaluate if there is a difference between *any* pair of subgroups while not primarily being interested between which specific pairs. Note that for testing multiple independent research questions -- e.g. testing effects of group as well as age -- we use multiple independent (possibly single-row) contrasts rather than one multi-row contrast, because the latter still encodes just a single research question.

(tutorial_model-outputs)=
### Model outputs: parameter estimates and statistical significance

The software uses the design matrix to fit the model to the data, and uses the contrasts to determine statistical significance for the effects of interest. This results in the following outputs:

- `beta`/`gamma` are regression coefficients / effect estimates — `gamma` is obtained by the applying the contrast vector or matrix to the fitted `beta` coefficients, and represents the estimated effect that the contrast is testing.
- `F` is an F-statistic. For a single-row (t-test-like) contrast, F is simply the square of the corresponding t-statistic (F = t²).
- `z` is the t-value expressed as a standard-normal quantile (a z-score), and its value will approach the t-value as sample size increases; z-values and t-values will be identical for practical purposes when df ≳ 30.
- `sig` is -- for numerical readability -- a transformation of the p-value rather than the p-value itself (smaller p-value → larger absolute sig-value: p = 0.05 → abs(sig) = 1.3; p = 0.01 → abs(sig) = 2; p = 0.001 → abs(sig) = 3). 

> **Sign:** for single-row contrasts, `sig` and `z` carry the sign of the estimated effect (`gamma`) — a positive value means the effect went in the direction the contrast defines as positive, a negative value the opposite. Multi-row (F-test) contrasts are left unsigned, since "direction" isn't a single number once more than one parameter is being tested jointly. This means flipping a contrast's sign (e.g. using `0 -1 0` instead of `0 1 0`) doesn't change the F-statistic or p-value at all — it only flips which direction is reported as positive. For single-row contrasts, the transformation between p-values and sig-values therefore is `sig = sign(gamma) * -log10(p)`. For multi-row contrasts, the transform is just `sig = -log10(p)`. To derive (unsigned) p-values from sig-values, use `p = 10^-abs(sig)`.

(tutorial_multiple-comparisons)=
### Multiple comparisons

A mass-univariate analysis runs one test per vertex — typically tens of thousands of tests at once. `fsstats` does not currently perform any multiple-comparison correction itself. Uncorrected `sig`/`z` values at individual vertices should therefore not be interpreted as significant on their own — apply an appropriate correction such as false discovery rate control (e.g. using FreeSurfer's `mri_fdr`), or a cluster-based approach (e.g. using FreeSurfer's `mri_glmfit-sim`) before drawing conclusions from a mass-univariate result. 

___

(tutorial_example)=
## Example analysis

All four [analysis scenarios](#contents) analyze the same hypothetical study: a longitudinal aging study comparing participants with dementia (AD) against cognitively normal (CN) controls, each scanned at multiple visits, and processed with the FastSurfer or FreeSurfer software. Two outcomes are used throughout:

- **Hippocampal volume**, a single value per participant per visit -- this is a *univariate* analysis, i.e. analysis of one or a few tabular variables ---.
- **Cortical thickness**, estimated at every vertex of a cortical surface — this is a *mass-univariate* analysis, i.e. an analysis of a very large number of variables (here, thousands of vertices) at once.

The four walkthroughs form a deliberate progression:

- The **GLM** walkthroughs (1-2) treat one visit per participant as an independent observation — a simple cross-sectional comparison of AD vs. CN at baseline.
- The **LME** walkthroughs (3-4) use the full longitudinal data, with multiple visits per participant. Because repeated visits from the same person are not independent observations, this requires a linear mixed-effects model rather than ordinary least squares, and lets us additionally ask whether AD and CN differ in their *rate of change* over time.

Working through all four in order (GLM before LME, univariate before mass-univariate) is a good way to build up from the simplest case to the full analysis, but each page is also self-contained if you only need one of the four.

> **Note:** `fsstats` does not contain an example dataset. Therefore, every command below uses placeholder paths (e.g. `/my/input/directory/...`), and these have to be substituted with your own paths and files for an analysis of your own data.

___

## Requirements

All four walkthroughs assume your data has already been processed with FastSurfer or FreeSurfer, and that you have prepared the inputs `fsstats` expects:

- For univariate analyses: a table of per-participant values, such as the output of FreeSurfer's `asegstats2table` (for hippocampal volume, an `aseg` measure) or `aparcstats2table` (for a parcellation-based thickness summary).
- For mass-univariate analyses: a concatenated surface overlay file produced by `mris_preproc`, containing cortical thickness at every vertex for every participant/visit, resampled to a common template surface (e.g. `fsaverage`).
- In both cases: a design matrix and one or more contrast files describing your model (built by hand or with your own scripts — `fsstats` does not build these for you), and, for the LME walkthroughs, a list identifying which rows belong to which case.

Each walkthrough describes the specific files it needs as they come up.
