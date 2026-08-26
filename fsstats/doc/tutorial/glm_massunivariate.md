# GLM, mass-univariate

*Part 2 of 4 in the [fsstats tutorial](../TUTORIAL.md). See also: [GLM, univariate](glm_univariate.md).*

## What this analysis is for

Instead of a single volume measure per subject, we now ask the same baseline MCI-vs-CN question about cortical thickness at every point (vertex) of the cortical surface. This is a *mass-univariate* analysis: the same ordinary least squares model as [GLM, univariate](glm_univariate.md) is fit independently at each of the (typically tens of thousands of) vertices, rather than once for a single tabular variable. Observations are still independent across subjects, so `fsstats_glm` remains the right tool.

## Inputs you need

- `--y`: a concatenated cortical thickness overlay for all subjects' baseline visits, resampled to a common template surface, as produced by `mris_preproc`.
- `--surface`: the template surface and hemisphere used for resampling, e.g. `fsaverage lh`.
- `--label` (optional): a FreeSurfer label restricting the analysis to a region of interest, e.g. `lh.cortex.label`, so that non-cortical/medial-wall vertices are skipped. Without it, all vertices are analyzed.
- `--X` and `--C`: the same design matrix and contrast as [GLM, univariate](glm_univariate.md) — the model doesn't change, only the outcome data does.
- `--glmdir`: an output directory.

See [DOCUMENTATION.md](../DOCUMENTATION.md#input-data-and-format-specifications) for exact format details.

## Building the design matrix and contrast

Unchanged from [GLM, univariate](glm_univariate.md): intercept, group, age, with a single contrast row `0 1 0` testing the group effect. The design matrix still has one row per subject — what changes is that `--y` now supplies thousands of outcome columns (one per vertex) instead of the single column read from `--table`.

## Running the command

```bash
fsstats_glm \
--y /my/input/directory/baseline_thickness_lh.mgh \
--label /my/input/directory/lh.cortex.label \
--X /my/input/directory/baseline_design_matrix.txt \
--C /my/input/directory/group_contrast.txt \
--surface fsaverage lh \
--glmdir /my/output/directory/glm_massunivariate \
--yhat-save \
--eres-save
```

`--yhat-save` and `--eres-save` are optional; they additionally save the model's fitted signal (`yhat`) and residuals (`eres`) per vertex, which can be useful for diagnostics.

## Understanding the output

The output has the same structure as the univariate case, but every file is a surface overlay (`.mgh`) instead of a text table: `beta.mgh`, `mask.mgh` (the label actually used), and, in the `group_contrast/` subdirectory, `gamma.mgh`, `F.mgh`, `sig.mgh`, and `z.mgh` — one value per vertex.

These are FreeSurfer-format overlays, meant to be viewed with `Freeview` (e.g. `freeview -f fsaverage/surf/lh.inflated:overlay=sig.mgh`) or further processed with `surfcluster` for cluster-based summaries. See [DOCUMENTATION.md](../DOCUMENTATION.md#output-files-and-their-interpretation) for the full output reference.

Next: [LME, univariate](lme_univariate.md), which brings back the repeated visits per subject that this cross-sectional analysis set aside.
