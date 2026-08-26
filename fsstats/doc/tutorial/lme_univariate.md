# LME, univariate

*Part 3 of 4 in the [fsstats tutorial](../TUTORIAL.md). See also: [LME, mass-univariate](lme_massunivariate.md).*

## What this analysis is for

Now we use the full longitudinal data: every subject contributes multiple visits, not just baseline. This lets us ask a richer question — does hippocampal volume *decline over time* differently in MCI compared to CN? — but it also breaks the independence assumption the GLM walkthroughs relied on: visits from the same subject are correlated with each other. A linear mixed-effects (LME) model accounts for this by treating subject as a random-effects grouping ("cluster"), which is what `fsstats_lme` is for.

## Inputs you need

- `--table`: a table of hippocampal volumes, now with one row per subject *per visit* (so most subjects contribute several rows), as produced by `asegstats2table`.
- `--X`: the longitudinal design matrix (below).
- `--C`: two contrast files (below).
- `--rfx`: which columns of `--X` are random effects.
- `--clusters`: a file naming, for each row of `--X`, which subject it belongs to. Rows for the same subject must be grouped together (consecutive), matching the order of rows in `--X` and `--table`.
- `--glmdir`: an output directory.

See [DOCUMENTATION.md](../DOCUMENTATION.md#input-data-and-format-specifications) for exact format details.

## Building the design matrix and contrasts

The design matrix now has one row per *visit* and four columns:

| Column | Meaning |
|---|---|
| 1 | Intercept |
| 2 | Time (years since baseline) |
| 3 | Group (0 = CN, 1 = MCI) |
| 4 | Group × Time interaction |

We treat both the intercept and the slope of Time as random effects — i.e., we let each subject have their own baseline level and their own rate of change, rather than assuming everyone starts at the same level and declines at the same rate:

```text
--rfx 1 2
```

Two questions, two contrasts:

- Does baseline volume differ by group? Test column 3:
  ```text
  0 0 1 0
  ```
- Does the *rate of decline* differ by group — i.e. the effect we're really after? Test column 4:
  ```text
  0 0 0 1
  ```

## Running the command

```bash
fsstats_lme \
--table /my/input/directory/longitudinal_hippocampus_table.txt \
--X /my/input/directory/longitudinal_design_matrix.txt \
--C /my/input/directory/group_contrast.txt \
--C /my/input/directory/group_by_time_contrast.txt \
--clusters /my/input/directory/subject_ids.txt \
--rfx 1 2 \
--glmdir /my/output/directory/lme_univariate
```

## Understanding the output

`fsstats_lme` writes results into `/my/output/directory/lme_univariate`:

- `beta.txt` — the fitted fixed-effects coefficients.
- `estimates.pkl` — the cached model fit (internal use; enables `--skip-estimation` for re-running inference with additional contrasts without re-fitting the model).
- One subdirectory per contrast (`group_contrast/`, `group_by_time_contrast/`), each with `gamma.txt`, `F.txt`, `sig.txt`, and `z.txt` — same meaning as in the GLM walkthroughs.

The `group_by_time_contrast/sig.txt` result is the one that answers our motivating question: whether MCI and CN differ in their rate of hippocampal decline. See [DOCUMENTATION.md](../DOCUMENTATION.md#output-files-and-their-interpretation) for the full output reference.

Next: [LME, mass-univariate](lme_massunivariate.md), which asks this same longitudinal question across the whole cortical surface.
