# Tutorial

## Overview

This tutorial walks through a complete example analysis with the `fsstats` command-line tools, `fsstats_glm` and `fsstats_lme`. It is intended for readers who have some background in statistics (e.g., you know what a design matrix or a contrast is) but are new to `fsstats`.

## Description

This page introduces the example analysis scenario used throughout the tutorial and links to the walkthrough pages. For the full, authoritative description of every argument, input format, and output file, see [DOCUMENTATION.md](../DOCUMENTATION.md) — the tutorial deliberately keeps things shorter and only explains what is needed to follow the example.

This tutorial covers command-line usage only. For using `fsstats` as a Python package or calling its low-level functions directly, see the ["As a Python package"](../DOCUMENTATION.md#as-a-python-package) and ["As Python functions"](../DOCUMENTATION.md#as-python-functions) sections of DOCUMENTATION.md.

___

## Contents

The tutorial covers four analysis scenarios:

1. Cross-sectional analysis for extracted, tabular data ([GLM, univariate](tutorial/glm_univariate.md))
2. Cross-sectional analysis for whole-brain, vertex-based data ([GLM, mass-univariate](tutorial/glm_massunivariate.md))
3. Longitudinal analysis for extracted, tabular data ([LME, univariate](tutorial/lme_univariate.md))
4. Longitudinal analysis for whole-brain, vertex-based data ([LME, mass-univariate](tutorial/lme_massunivariate.md))

___

## The running example

To keep the four walkthroughs connected, they all analyze the same hypothetical study: a longitudinal aging study comparing participants with mild cognitive impairment (MCI) against cognitively normal (CN) controls, each scanned at multiple visits, and processed with the FastSurfer or FreeSurfer softwares. Two outcomes are used throughout:

- **Hippocampal volume**, a single value per subject per visit — this is a *univariate* analysis, i.e. analysis of one or a few tabular variables.
- **Cortical thickness**, estimated at every vertex of a cortical surface — this is a *mass-univariate* analysis, i.e. an analysis of a very large number of variables (here, thousands of vertices) at once.

The four walkthroughs form a deliberate progression:

- The **GLM** walkthroughs (1-2) treat one visit per subject as an independent observation — a simple cross-sectional comparison of MCI vs. CN at baseline.
- The **LME** walkthroughs (3-4) use the full longitudinal data, with multiple visits per subject. Because repeated visits from the same person are not independent observations, this requires a linear mixed-effects model rather than ordinary least squares, and lets us additionally ask whether MCI and CN differ in their *rate of change* over time.

Working through all four in order (GLM before LME, univariate before mass-univariate) is a good way to build up from the simplest case to the full analysis, but each page is also self-contained if you only need one of the four.

> **Note:** The `fsstats` does not contain an example dataset. Therefore, every command below uses placeholder paths (e.g. `/my/input/directory/...`), and these have to be substituted with your own paths and files.

## What you'll need

All four walkthroughs assume your data has already been processed with FastSurfer or FreeSurfer, and that you have prepared the inputs `fsstats` expects:

- For univariate analyses: a table of per-subject values, such as the output of FreeSurfer's `asegstats2table` (for hippocampal volume, an `aseg` measure) or `aparcstats2table` (for a parcellation-based thickness summary).
- For mass-univariate analyses: a concatenated surface overlay file produced by `mris_preproc`, containing cortical thickness at every vertex for every subject/visit, resampled to a common template surface (e.g. `fsaverage`).
- In both cases: a design matrix and one or more contrast files describing your model (built by hand or with your own scripts — `fsstats` does not build these for you), and, for the LME walkthroughs, a list identifying which rows belong to which subject.

Each walkthrough describes the specific files it needs as they come up.
