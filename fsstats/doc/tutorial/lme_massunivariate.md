# LME, mass-univariate

*Part 4 of 4 in the [fsstats tutorial](../TUTORIAL.md). See also: [LME, univariate](lme_univariate.md).*

## What this analysis is for

This combines the previous two extensions: like [LME, univariate](lme_univariate.md), we use every longitudinal visit per subject and model subject as a random-effects cluster; like [GLM, mass-univariate](glm_massunivariate.md), the outcome is cortical thickness at every surface vertex rather than a single tabular value. This is the full analysis our running example has been building towards: does the rate of cortical thinning over time differ between MCI and CN, and where on the cortex?

Because a separate model is fit at every vertex and each fit iterates to convergence, this is the most computationally demanding of the four walkthroughs — parallelism (`--cores`) matters here.

## Inputs you need

- `--y`: a concatenated cortical thickness overlay covering all subjects' visits, produced by `mris_preproc`, resampled to a common template surface.
- `--surface`: template surface and hemisphere, e.g. `fsaverage lh`.
- `--label` (optional): restrict analysis to a region of interest, e.g. `lh.cortex.label`.
- `--X`, `--C`, `--rfx`, `--clusters`: the same longitudinal design, contrasts, random-effects columns, and subject grouping as [LME, univariate](lme_univariate.md) — again, only the outcome data changes.
- `--glmdir`: an output directory.

See [DOCUMENTATION.md](../DOCUMENTATION.md#input-data-and-format-specifications) for exact format details.

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
--clusters /my/input/directory/subject_ids.txt \
--rfx 1 2 \
--surface fsaverage lh \
--glmdir /my/output/directory/lme_massunivariate \
--cores 4
```

`--cores` distributes the per-vertex model fits across CPU cores — increase it to as many cores as you have available to speed up processing. Two related expert options are available if the default settings don't suit your data: `--iterations` (maximum iterations per vertex before giving up on convergence; default 20) and `--epsilon` (how strict the convergence criterion is; default 0.1).

## Understanding the output

Same structure as [GLM, mass-univariate](glm_massunivariate.md), but with `estimates.pkl`/`statistics.pkl` alongside the surface overlays: `beta.mgh`, and per-contrast `gamma.mgh`, `F.mgh`, `sig.mgh`, `z.mgh` in `group_contrast/` and `group_by_time_contrast/`. View them with `Freeview` or summarize clusters of significant vertices with `surfcluster`. See [DOCUMENTATION.md](../DOCUMENTATION.md#output-files-and-their-interpretation) for the full output reference.

Since model estimation is the expensive part here, `--skip-estimation` lets you re-run inference with new or additional contrasts against an existing `--glmdir` (which must already contain `estimates.pkl`) without re-fitting every vertex.

## Optional: modeling spatial covariance

By default, every vertex is fit independently. The `--spatial-covariance` flag instead pools covariance estimation and models residual correlation across neighboring vertices within homogeneous surface regions, which can improve statistical power. It requires `--surface` to resolve to an anatomical surface file with a sibling `<hemi>.sphere` file (true for standard templates like `fsaverage`), and is only available for mass-univariate analyses (`--y`, not `--table`):

```bash
fsstats_lme \
--y /my/input/directory/longitudinal_thickness_lh.mgh \
--label /my/input/directory/lh.cortex.label \
--X /my/input/directory/longitudinal_design_matrix.txt \
--C /my/input/directory/group_by_time_contrast.txt \
--clusters /my/input/directory/subject_ids.txt \
--rfx 1 2 \
--surface fsaverage lh \
--glmdir /my/output/directory/lme_massunivariate_spatial \
--cores 4 \
--spatial-covariance
```

This concludes the tutorial. For the complete argument and output reference, deviations from the original FreeSurfer/Matlab tools, and citations, see [DOCUMENTATION.md](../DOCUMENTATION.md).
