# LME, mass-univariate

*Part 4 of 4 in the [fsstats tutorial](../TUTORIAL.md). See also: [LME, univariate](lme_univariate.md).*

## What this analysis is for

This combines the previous two extensions: like [LME, univariate](lme_univariate.md), we use every longitudinal visit per participant and model each participant as a random-effects cluster; like [GLM, mass-univariate](glm_massunivariate.md), the outcome is cortical thickness at every surface vertex rather than a single tabular value. This is the full analysis our running example has been building towards: does the rate of cortical thinning over time differ between AD and CN, and where on the cortex?

Because a separate model is fit at every vertex and each fit iterates to convergence, this is the most computationally demanding of the four walkthroughs — parallelism (`--cores`) matters here.

## Inputs you need

- `--y`: a concatenated cortical thickness overlay covering all participants' visits, produced by `mris_preproc`, resampled to a common template surface.
- `--surface`: template surface and hemisphere, e.g. `fsaverage lh`.
- `--label` (optional): restrict analysis to a region of interest, e.g. `lh.cortex.label`.
- `--X`, `--C`, `--rfx`, `--clusters`: the same longitudinal design, contrasts, random-effects columns, and case grouping (`case_ids.txt`) as [LME, univariate](lme_univariate.md) — again, only the outcome data changes.
- `--glmdir`: an output directory.

See the [documentation](documentation_input-data-format-specs) for exact format details.

## Building the design matrix and contrasts

Unchanged from [LME, univariate](lme_univariate.md): intercept, Time, Group, Group×Time, with `--rfx 1 2` (random intercept and random slope of Time) and the same two contrasts. The `group_by_time_contrast` (testing column 4) is again the one that answers our motivating question — now per vertex.

## Running the command

```bash
fsstats_lme \
--y /my/input/directory/longitudinal_thickness_lh.mgh \
--label /my/input/directory/lh.cortex.label \
--X /my/input/directory/longitudinal_design_matrix.txt \
--C /my/input/directory/group_contrast.txt \
--C /my/input/directory/group_by_time_contrast.txt \
--clusters /my/input/directory/case_ids.txt \
--rfx 1 2 \
--surface fsaverage lh \
--glmdir /my/output/directory/lme_massunivariate \
--cores 4
```

> **A note on runtimes and parallel processing:** unlike GLM's closed-form fit, a separate LME model is iteratively fit at every one of the (typically tens of thousands of) vertices, and these per-vertex fits are independent of one another — `--cores` distributes them across CPU cores to run in parallel. The default is 1 (no parallelism), which will be considerably slower. As a starting point, set `--cores` to the number of physical CPU cores available on your machine (maybe minus 1 or 2 to leave some capacity for other tasks); going beyond that rarely helps further, since the fits then have to compete for the same cores. Typical runtime for an analysis of **this ...** size, at various `--cores` settings: **to be measured and added here [...]**.

## Understanding the output

Same structure as [GLM, mass-univariate](glm_massunivariate.md): `beta.mgh`, and per-contrast `gamma.mgh`, `F.mgh`, `sig.mgh`, `z.mgh` in `group_contrast/` and `group_by_time_contrast/`. See [key concepts](tutorial_model-outputs) for what these represent.

Unlike the univariate analysis, mass-univariate LME does not compute per-case random-effects estimates at all (they are skipped here to keep runtime and output size manageable across tens of thousands of vertices) — only the fixed-effects results above are available.

`fsstats` caches the fitted model internally, so once estimation has completed for this `--glmdir`, you can add `--skip-estimation` on a later run to evaluate new or additional contrasts without re-fitting every vertex.

View the group-by-time effect's signed significance map with Freeview, for example:

```bash
freeview -f fsaverage/surf/lh.inflated:overlay=/my/output/directory/lme_massunivariate/group_by_time_contrast/sig.mgh:overlay_threshold=1.3,5
```

As in [GLM, mass-univariate](glm_massunivariate.md), this colors the surface by `sig` (here thresholded at sig ≥ 1.3, i.e. p ≤ 0.05): warm/cool colors mark where the group difference in the rate of thinning is positive/negative, and intensity reflects the strength of the effect. `sig.mgh`/`F.mgh` can also be further processed with `mri_surfcluster` for cluster-based summaries.

**A note on multiple comparisons:** as with [GLM, mass-univariate](glm_massunivariate.md), this map is one test per vertex and is not corrected for the many comparisons involved — see [key concepts](tutorial_multiple-comparisons) before interpreting individual vertices as significant.

See the [documentation](documentation_output-files) for the full output reference.

## Optional: modeling spatial covariance

By default, every vertex is fit independently. The `--spatial-covariance` flag instead pools covariance estimation and models residual correlation across neighboring vertices within homogeneous surface regions, which can improve statistical power. It requires `--surface` to resolve to an anatomical surface file with a sibling `<hemi>.sphere` file (true for standard templates like `fsaverage`), and is only available for mass-univariate analyses (`--y`, not `--table`):

```bash
fsstats_lme \
--y /my/input/directory/longitudinal_thickness_lh.mgh \
--label /my/input/directory/lh.cortex.label \
--X /my/input/directory/longitudinal_design_matrix.txt \
--C /my/input/directory/group_by_time_contrast.txt \
--clusters /my/input/directory/case_ids.txt \
--rfx 1 2 \
--surface fsaverage lh \
--glmdir /my/output/directory/lme_massunivariate_spatial \
--cores 4 \
--spatial-covariance
```

This additionally writes `regions.mgh`, showing which vertices were pooled together into each region for covariance estimation.

This concludes the tutorial. For the complete argument and output reference, deviations from the original FreeSurfer/Matlab tools, and citations, see the [documentation](documentation).
