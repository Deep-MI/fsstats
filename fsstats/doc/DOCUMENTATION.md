(documentation)=
# Documentation

## Overview

The `fsstats` package provides univariate and mass-univariate statistical modeling and inference tools for [FastSurfer](http://fastsurfer.org) and [FreeSurfer](https://surfer.nmr.mgh.harvard.edu) imaging data. It contains two modules: `glm`, a lean Python port of a subset of FreeSurfer's [mri_glmfit](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_glmfit) tool for ordinary least squares (GLM) analyses, and `lme`, a revision, extension, and translation to the Python language of Freesurfer's Matlab-based [LME](https://surfer.nmr.mgh.harvard.edu/fswiki/LinearMixedEffectsModels) tools.

This page provides general, technical usage, and installation information about the `fsstats` toolbox. See the tutorial for an illustrative explanation about using the software.

___

## Contents

- [Functionality](#functionality)
- [Usage](#usage)
  - [As a command line tool](#as-a-command-line-tool)
  - [As a Python package](#as-a-python-package)
  - [As Python functions](#as-python-functions) 
- [Inputs and outputs](#inputs-and-outputs)
    - [Input data and format specifications](#input-data-and-format-specifications)
    - [Output files and their interpretation](#output-files-and-their-interpretation)
- [Deviations from original packages](#deviations-from-original-packages)
    - [Deviations from mri_glmfit](#deviations-from-mri-glmfit)
    - [Deviations from the lme toolbox](#deviations-from-the-lme-toolbox)
- [Citations](#citations)
- [License](#license)

___

## Functionality

The core functionality of this toolbox is to conduct statistical analyses for FastSurfer- and FreeSurfer-based imaging data, via two modules:

- `glm` provides univariate and mass-univariate ordinary least squares (general linear model) analyses for independent observations, following the conventions of FreeSurfer's `mri_glmfit`. 
- `lme` provides univariate and mass-univariate linear mixed-effects analyses for clustered (e.g., longitudinal) data, where observations belonging to the same cluster (e.g., repeated visits of the same subject) are not independent.

Here, *univariate analysis* refers to the analysis of one or few variables, where the input data can be represented in tabular format; a typical example is the analysis of extracted volume or thickness estimates in a limited set of larger brain regions (regions of interest). In contrast, *mass-univariate* refers to analysis of a very large number of variables, and typically means analyses of single voxels or vertices, which require dedicated, image-based preprocessing and input formats.
___

## Usage

### As a command line tool

#### The GLM module

```text
fsstats_glm --X <filename> --C <filename> [--C <filename>] --glmdir <directory>
    [--table <filename>] [--y <filename> ] [--surface <surface name> <lh|rh>] 
    [--mask <filename>]  [--label <filename>] [--yhat-save] [--eres-save]
    [--help] [--detailed-help] [--version]

Required arguments:
  --X <filename>        Filename of a design matrix.
  --C filename          Filename of a contrast matrix or vector. Can be used multiple times.
  --glmdir <directory>  Directory where the results will be written.

Required arguments for univariate analysis:
  --table <filename>    Filename of a table output, e.g. produced by asegstats2table or aparcstats2table.

Required and optional arguments for mass-univariate analysis:
  --y <filename>        Filename of a concatenated overlay file, e.g. produced by mris_preproc.
  --surface <surface name> <lh|rh>
                        Template surface (e.g., fsaverage) and hemisphere (lh or rh).
  --mask <filename>     Filename of a binary mask volume that will be used to select vertices/voxels for analysis.
                        Mutually exclusive with --label. Default: All vertices/voxels.
  --label <filename>    Filename of a FreeSurfer label file that will be used to select vertices for analysis.
                        Mutually exclusive with --mask. Default: All vertices.
  --yhat-save           Save the signal estimate (yhat). Default: False.
  --eres-save           Save the residual error (eres). Default: False.

Getting help:
  --help                Display this help message and exit
  --detailed-help       Display extensive help message and exit
  --version             Display version number and exit

```

*Examples:*

- Univariate analysis:

The following command can be used to conduct an ordinary least squares analysis of tabular input data, evaluating two contrasts via the `--C` arguments. Further details on the expected inputs and their formats are given in the [Inputs and outputs](#inputs-and-outputs) section.

```bash
fsstats_glm \
--table /my/input/directory/my_tabular_input_file.txt \
--X /my/input/directory/my_design_matrix.txt \
--C /my/input/directory/my_contrast_file_1.txt \
--C /my/input/directory/my_contrast_file_2.txt \
--glmdir /my/output/directory
```

- Mass-univariate analysis:

The following command can be used to conduct an ordinary least squares analysis of mass-univariate input data, evaluating one contrast via the `--C` argument. As specified by the `--surface` argument, the analysis will use the `fsaverage` template, and will be done for the left hemisphere. The `--yhat-save` and `--eres-save` flags additionally save the model's signal estimate and residual error. Further details on the expected inputs and their formats are given in the [Inputs and outputs](#inputs-and-outputs) section.

```bash
fsstats_glm \
--y /my/input/directory/my_preprocessed_image_data.mgh \
--label /input/directory/my_cortical_label_file.label \
--X /my/input/directory/my_design_matrix.txt \
--C /my/input/directory/my_contrast_file.txt \
--surface fsaverage lh \
--glmdir /my/output/directory \
--yhat-save \
--eres-save
```

#### The LME module

```text
fsstats_lme --X <filename> --C <filename> [--C <filename>]--rfx <int> [<int> ...]
    --clusters <filename> --glmdir <directory> [--table <filename>] [--y <filename> ]
    [--surface <surface name> <lh|rh>] [--label <filename>] [--spatial-covariance]
    [--cores <int>] [--iterations <int>] [--epsilon <float>] [--skip-estimation]
    [--help] [--detailed-help] [--version]

Required arguments:
  --X <filename>        Filename of a design matrix.
  --C filename          Filename of a contrast matrix or vector. Can be used multiple times.
  --rfx <int> [<int> ...]
                        Column indices (1-based) of random effects in the design matrix.
  --clusters <filename>
                        Filename of a list of clusters (e.g., case IDs), where each line corresponds to the same row in the design matrix.
  --glmdir <directory>  Directory where the results will be written.

Required arguments for univariate analysis:
  --table <filename>    Filename of a table output, e.g. produced by asegstats2table or aparcstats2table.

Required and optional arguments for mass-univariate analysis:
  --y <filename>        Filename of a concatenated overlay file, e.g. produced by mris_preproc.
  --surface <surface name> <lh|rh>
                        Template surface (e.g., fsaverage) and hemisphere (lh or rh).
  --label <filename>    Filename of a FreeSurfer label file that will be used to select vertices for analysis. Default: All vertices.
  --spatial-covariance  Model spatial covariance of the residuals within homogeneous surface regions instead of fitting each vertex independently.
                        Requires --surface to resolve to an anatomical surface file with a sibling <hemi>.sphere file. Default: False.

Expert options:
  --cores <int>         Number of CPU cores used in parallel processing. Default: 1.
  --iterations <int>    Maximum number of iterations to achieve convergence. Default: 20.
  --epsilon <float>     Epsilon for gradient norm to determine convergence. Default: 1e-1.
  --skip-estimation     Skip estimation. Useful for conducting statistical inference for additional contrasts with needing to run model estimation again.
                        Requires existing glmdir with estimates.pkl file. Default: False.

Getting help:
  --help                Display this help message and exit
  --detailed-help       Display extensive help message and exit
  --version             Display version number and exit

```

*Examples:*

- Univariate analysis:

The following command can be used to conduct a linear mixed-effects analysis of tabular input data. This example includes two random effects via the `--rfx` argument (e.g., random intercept and random slope of time, assuming that columns 1 and 2 of the design matrix are the global intercept and the regressor for *Time*, respectively) and the evaluation of two contrasts via the `--C` arguments. Further details on the expected inputs and their formats are given in the [Inputs and outputs](#inputs-and-outputs) section.

```bash
fsstats_lme \
--table /my/input/directory/my_tabular_input_file.txt \
--X /my/input/directory/my_design_matrix.txt \
--C /my/input/directory/my_contrast_file_1.txt \
--C /my/input/directory/my_contrast_file_2.txt \
--clusters /my/input/directory/my_list_of_clusters.txt \
--rfx 1 2 \
--glmdir /my/output/directory
```

- Mass-univariate analysis:

The following command can be used to conduct a linear mixed-effects analysis of mass-univariate input data. This example includes specification of one random effect via the `--rfx` argument (e.g., random intercept, assuming that column 1 of the design matrix is the global intercept), and the evaluation of one contrast via the `--C` argument. As specified by the `--surface` argument, the analysis will use the `fsaverage` template, and will be done for the left hemisphere. The `--cores` argument tells the program to run 4 processes in parallel, which will speed up processing, and can be increased depending on how many CPU cores are available on the system. Further details on the expected inputs and their formats are given in the [Inputs and outputs](#inputs-and-outputs) section.

```bash
fsstats_lme \
--y /my/input/directory/my_preprocessed_image_data.mgh \
--label /input/directory/my_cortical_label_file.label \
--X /my/input/directory/my_design_matrix.txt \
--C /my/input/directory/my_contrast_file.txt \
--clusters /my/input/directory/my_list_of_clusters.txt \
--rfx 1 \
--surface fsaverage lh \
--glmdir /my/output/directory \
--cores 4
```

(documentation_as-a-python-package)=
### As a Python package

As an alternative to their command-line usage, the fsstats scripts can also be run within a pure Python environment, i.e. installed and imported as a Python package.

Use `import fsstats` (or sth. equivalent) to import the package within a Python environment, and use the `run_glm` or `run_lme` function to run a ordinary least squares (GLM) or linear mixed-effects analysis, respectively.

(documentation_package-glm)=
#### The GLM module

- Univariate analysis:

```python
import fsstats
fsstats.run_glm(
    table="/my/input/directory/my_tabular_input_file.txt",
    design_matrix="/my/input/directory/my_design_matrix.txt",
    contrasts=["/my/input/directory/my_contrast_file_1.txt", "/my/input/directory/my_contrast_file_2.txt"],
    glmdir="/my/output/directory"
    )
```
- Mass-univariate analysis:

```python
import fsstats
fsstats.run_glm(
    filename="/my/input/directory/my_preprocessed_image_data.mgh",
    label="/input/directory/my_cortical_label_file.label",
    design_matrix="/my/input/directory/my_design_matrix.txt",
    contrasts=["/my/input/directory/my_contrast_file.txt"],
    surface="fsaverage",
    hemi="lh",
    glmdir="/my/output/directory",
    yhat_save=True,
    eres_save=True
    )
```

(documentation_package-lme)=
#### The LME module

- Univariate analysis:

```python
import fsstats
fsstats.run_lme(
    table="/my/input/directory/my_tabular_input_file.txt",
    design_matrix="/my/input/directory/my_design_matrix.txt",
    contrasts=["/my/input/directory/my_contrast_file_1.txt", "/my/input/directory/my_contrast_file_2.txt"],
    clusters="/my/input/directory/my_list_of_clusters.txt",
    random_effects=[1, 2],
    glmdir="/my/output/directory"
    )
```
- Mass-univariate analysis:

```python
import fsstats
fsstats.run_lme(
    filename="/my/input/directory/my_preprocessed_image_data.mgh",
    label="/input/directory/my_cortical_label_file.label",
    design_matrix="/my/input/directory/my_design_matrix.txt",
    contrasts=["/my/input/directory/my_contrast_file.txt"],
    clusters="/my/input/directory/my_list_of_clusters.txt",
    random_effects=[1],
    surface="fsaverage",
    hemi="lh",
    num_cores=4,
    glmdir="/my/output/directory"
    )
```

(documentation_as-python-functions)=
### As Python functions

The above `fsstats.run_glm` and `fsstats.run_lme` functions are wrappers around the two central functions of each module for fitting a model and subsequently performing statistical inference on it, i.e. `fsstats.glm.fit`/`fsstats.glm.test` and `fsstats.lme.fit`/`fsstats.lme.test`. Use of the wrapper functions is recommended for most cases, but in scenarios like customized analyses or processing pipelines, it may be desirable to call the low-level `fit` and `test` functions directly. In contrast to the wrapper functions, which in most cases take filenames as inputs, the `fit` and `test` functions directly operate on Python variables such as numpy arrays or lists. We refer to the API documentation for details about the expected inputs and the resulting outputs, and give short examples below.

(documentation_functions-glm)=
#### The GLM module

The following code will fit an ordinary least squares model for all (or a masked subset of) voxels/vertices/variables in a single vectorized computation. We need to provide a design matrix `X` and the data `Y`. `maskvtx` is optional (`None` fits every column of `Y`); `return_yhat`/`return_eres` control whether the signal estimate and residual error are additionally returned.

```python
from fsstats.glm import fit
stats = fit(X=design_matrix,
            Y=thickness_values,
            maskvtx=None,
            return_yhat=False,
            return_eres=False
            )
```

After model estimation, the following code will perform statistical inference for a previously estimated model, again for all voxels/vertices/variables in a single vectorized computation. We need to specify the estimated regression coefficients (`Bhat`), residual variance (`rvar`), degrees of freedom (`dof`), and design matrix (`X`) from `stats`, and a contrast matrix `C`.

```python
from fsstats.glm import test
fstats = test(Bhat=stats['Bhat'],
              rvar=stats['rvar'],
              dof=stats['dof'],
              X=stats['X'],
              C=contrast_matrix
              )
```

(documentation_functions-lme)=
#### The LME module

The following code will fit a linear mixed-effects model. We need to provide a design matrix `X`, the indices `Zcols` of the random effects in the design matrix, the concatenated thickness (or volume) data `Y`, and a list `clusters` of the cluster structure. All other inputs are optional, and are set to their default values in this example. In practice, it will make sense to increase the number of CPU cores (`prs`) to allow for parallel processing.

```python
from fsstats.lme import fit
stats = fit(X=design_matrix,
            Zcols=indices_of_random_effects,
            Y=thickness_values,
            clusters=cluster_structure,
            maskvtx=None,
            prs=1,
            nit=20,
            e=0.1,
            return_random_effects_estimates=False,
            return_minimal_information=True,
            verbose=False
            )
```

After model estimation, the following code will perform statistical inference for a previously estimated model. This can be used repeatedly, for example for evaluating different contrasts. We need to specify estimated model parameters (`stats`), and a contrast vector or matrix `C`. All other inputs are optional, and are set to their default values.

```python
from fsstats.lme import test
fstats = test(stats,
              C=contrast_vector,
              bias_correct=False,
              prs=1,
              verbose=False
              )
```

___

## Inputs and outputs

(documentation_input-data-format-specs)=
### Input data and format specifications

Argument | Used by | Description | Format
------|------|------------|------------------------
`--table` | lme, glm | Univariate data | Table including a header that describes the variable names and a first column of IDs; can have one or more variables. Output from FreeSurfer's `asegstats2table` or `aparcstats2table` will be accepted. Only needed for univariate analysis.
`--y`  | lme, glm | Preprocessed mass-univarate data | FreeSurfer mgh/mgz image file, can be obtained from `mris_preproc`. Only needed for mass-univariate analysis.
`--label` | lme, glm | Analysis mask for mass-univariate data | FreeSurfer label file, e.g. `lh.cortex`. Optional for mass-univariate analysis; if omitted (and, for `glm`, `--mask` is also omitted), all data will be used. Mutually exclusive with `--mask`.
`--mask` | glm | Analysis mask for mass-univariate data | Binary mask volume in mgh/mgz format, with the same number of voxels/vertices as `--y`. Optional; mutually exclusive with `--label`.
`--X`  | lme, glm | Design matrix | Table without header and without column of IDs; each row corresponds to one observation and each column corresponds to one parameter/variable in the model.
`--C`  | lme, glm | Contrast vector/matrix | Table without header, with one or more rows depending on the type of contrast, and as many columns as the design matrix.  Each column refers to the same column of the design matrix. One contrast vector/matrix per file; for multiple contrasts, specify multiple `--C` arguments.
`--clusters` | lme | Grouping structure of the data | Table without header and a single column. Must have as many rows as the design matrix, and each row corresponds to the same row in the design matrix. Indicates for every observation to which which cluster (e.g., person) it belongs; can be a list of repeated participant IDs. Not used by `glm`, which assumes independent observations.

All tables can be separated by tab, comma, or space.

(documentation_output-files)=
### Output files and their interpretation

The program will compute the following statistics, and produce text- or image-based output files. Text files can be accessed with any text editor. Image files follow the FreeSurfer format and conventions, and are intended to be used with visualization and post-processing tools like `Freeview` and `mri_surfcluster`, which are available from any FreeSurfer distribution.

file           | used by | description
---------------|---------|----------------------------------------------------------------
logfile.txt    | lme, glm | logfile
Xg.dat         | lme, glm | design matrix (univariate analysis only)
beta.{mgh,txt} | lme, glm | model parameters
rvar.{mgh,txt} | glm | residual error variance
rstd.{mgh,txt} | glm | residual error standard deviation (square root of rvar)
mask.mgh       | glm | the final mask/label used, if `--mask` or `--label` was given
yhat.{mgh,txt} | glm | signal estimate, only with `--yhat-save`
eres.{mgh,txt} | glm | residual error, only with `--eres-save`
estimates.pkl  | lme | internal file with parameter estimates (not written by `glm`, which re-fits its closed-form solution on demand instead of caching it)
statistics.pkl | lme | internal file with statistics
\<contrast_directory>/C.dat     | lme, glm | contrast vector or matrix
\<contrast_directory>/gamma.{mgh,txt} | lme, glm | contrast estimate
\<contrast_directory>/F.{mgh,txt}     | lme, glm | F-value of the contrast
\<contrast_directory>/sig.{mgh,txt}   | lme, glm | sig-value of the contrast, signed for single-row (t-test-like) contrasts
\<contrast_directory>/z.{mgh,txt}     | lme, glm | z-value corresponding to the sig-value's p-value, signed for single-row (t-test-like) contrasts

The contrast estimate results from the application of the contrast vector or matrix to the model parameters. For single-row (t-test-like) contrasts, the sign of the contrast (`sign(gamma)`) is folded directly into `sig` and `z`; for multi-row (F-test) contrasts, sign is not meaningful and `sig`/`z` are left unsigned. This matches the convention used by FreeSurfer's `mri_glmfit`. The program produces `sig-values` instead of `p-values` for better representation and visualization of values that are very small or close to zero. `p-values` and (unsigned) `sig-values` can be converted into each other: `p = 10^(-sig)` and `sig = -log10(p)`. The following table gives a few examples.

p       | sig
--------|----
0.05    | 1.3
0.01    | 2.0
0.005   | 2.3
0.001   | 3.0
0.0001  | 4.0
0.00001 | 5.0

___

## Deviations from original packages

### Deviations from mri_glmfit

`glm` implements a lean subset of FreeSurfer's `mri_glmfit`. Relative to `mri_glmfit`, the following are intentionally **not** implemented:

- Weighted least squares (`--w`/`--wls`) and fixed-effects analysis (`--yffxvar`/`--ffxdof`/`--ffxdofdat`)
- `--osgm`, `--pca`, `--sim` (permutation/Monte Carlo simulation and multiple-comparison correction), `--fsgd`, DTI and kinetic-modeling options (`--dti`, `--mrtm1`, `--mrtm2`, `--logan`)
- Pruning (`--prune`) and smoothing (`--fwhm`/`--var-fwhm`)

### Deviations from the lme toolbox (Matlab)

Introducing the `glm` module also brought two changes to the `lme` module's command-line interface and outputs, made for consistency across both modules:

- `--mask` renamed to `--label`: the flag was always used to pass a FreeSurfer label file, so it is now named `--label` (matching `glm`, which additionally supports a distinct `--mask` for binary mask *volumes*, following `mri_glmfit`'s own convention). The Python API's `run_lme(mask=...)` parameter is now `run_lme(label=...)`.
- Signed `sig` + `z`, no separate `sgn`: contrast output no longer includes a separate `sgn.mgh`/`sgn.txt` file. `sig.mgh`/`sig.txt` now has the contrast sign folded in directly for single-row (t-test-like) contrasts, and every contrast additionally gets a new `z.mgh`/`z.txt` (z-value) file — mirroring FreeSurfer's `mri_glmfit` convention.

___

## Citations

If you use these tools in your analysis please cite:

- Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer -
  A fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219
  (2020), 117012. <https://doi.org/10.1016/j.neuroimage.2020.117012>

- Henschel L*, Kuegler D*, Reuter M. (*co-first). FastSurferVINN:
  Building Resolution-Independence into Deep Learning Segmentation Methods -
  A Solution for HighRes Brain MRI. NeuroImage 251 (2022), 118933.
  <https://dx.doi.org/10.1016/j.neuroimage.2022.118933>

If using the GLM module, please additionally cite:

- Fischl, B. (2012). FreeSurfer. NeuroImage, 62(2), 774-781. 
 <https://dx.doi.org/10.1016/j.neuroimage.2012.01.021>

If using the LME module, please additionally cite:

- Bernal-Rusiel JL, Greve DN, Reuter M, Fischl B, Sabuncu MR.
  Statistical Analysis of Longitudinal Neuroimage Data with
  Linear Mixed Effects Models, NeuroImage 66 (2012), 249-260.
  <https://dx.doi.org/10.1016%2Fj.neuroimage.2012.10.065>

- Bernal-Rusiel JL, Greve DN, Reuter M, Fischl B, Sabuncu MR.
  Spatiotemporal Linear Mixed Effects Modeling for the
  Mass-univariate Analysis of Longitudinal Neuroimage Data, NeuroImage
  81 (2013), 358–370. <https://doi.org/10.1016/j.neuroimage.2013.05.049>

___

## License

This software is licensed under the Apache 2.0 License, see associated LICENSE file
for details.

Copyright (c) 2026 Artificial Intelligence in Medical Imaging Group, DZNE
