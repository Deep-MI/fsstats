# Documentation

## Overview

The `fsstats` package provides univariate and mass-univariate statistical modeling and inference tools for [FastSurfer](http://fastsurfer.org) and [FreeSurfer](https://surfer.nmr.mgh.harvard.edu) imaging data. It is a revision, extension, and translation to the Python language of Freesurfer’s Matlab-based [LME](https://surfer.nmr.mgh.harvard.edu/fswiki/LinearMixedEffectsModels) tools.


## Description

This page provides general, usage, and installation information about the `fsstats` toolbox.

___

## Contents

- [Functionality](#functionality)
- [Usage](#usage)
  - [As a command line tool](#as-a-command-line-tool)
  - [As a Python package](#as-a-python-package)
- [Inputs and outputs](#inputs-and-outputs)
    - [Input data and format specifications](#input-data-and-format-specifications)
    - [Output files and their interpretation](#output-files-and-their-interpretation)
- [Citations](#citations)
- [License](#license)

___

## Functionality

The core functionality of this toolbox is to conduct statistical analyses for FastSurfer- and FreeSurfer-based imaging data. Its current focus is on providing univariate and mass-univariate linear mixed-effects analyses for clustered (e.g., longitudinal) data. Here, *univariate analysis* refers to the nalysis of one or few variables, where the input data can be represented in tabular format; a typical example is the analysis of extracted volume or thickness estimates in a limited set of larger brain regions (regions of interest). In contrast, *mass-univariate* refers to analysis of a very large number of variables, and typically means analyses of single voxels or vertices, which require dedicated, image-based preprocessing and input formats.

___

## Usage

### As a command line tool

```text
mri_lmefit --X <filename> --C <filename> [--C <filename>]--rfx <int> [<int> ...] --clusters <filename> --glmdir <directory>
    [--table <filename>]
    [--y <filename> ] [--surface <surface name> <lh|rh>] [--mask <filename>]
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
  --mask <filename>     Filename of a FreeSurfer label file that will be used to select vertices for analysis. Default: All vertices.

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
mri_lmefit \
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
mri_lmefit \
--y /my/input/directory/my_preprocessed_image_data.mgh \
--mask /input/directory/my_cortical_label_file.label \
--X /my/input/directory/my_design_matrix.txt \
--C /my/input/directory/my_contrast_file.txt \
--clusters /my/input/directory/my_list_of_clusters.txt \
--rfx 1 \
--surface fsaverage lh \
--glmdir /my/output/directory \
--cores 4
```

### As a Python package

As an alternative to their command-line usage, the fsstats scripts can also be run within a pure Python environment, i.e. installed and imported as a Python package.

Use `import fsstats` (or sth. equivalent) to import the package within a Python environment, and use the `run_lme` function to
run a linear mixed-effcts analysis.

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
    mask="/input/directory/my_cortical_label_file.label",
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

### As Python functions

The above `fsstats.run_lme` function is  a wrapper around the two central functions for fitting linear mixed-effects models and subsequently performing statistical inference on them, i.e. `fsstats.lme.fit` and `fsstats.lme.test`. Use of the wrapper function is recommended for most cases, but in scenarios like customized analyses or processing pipelines, it may be desirable to call the low-level `fit` and `test` functions directly. We refer to the API documentation for details about the expected inputs and the resulting outputs, and give two short examples below.

The following code will fit a linear mixed-effects model. We need to provide a design matrix `X`, the indices `Zcols` of the random effects in the design matrix, the concatenated thickness (or volume) data, and a vector `ni` of the cluster structure. All other inputs are optional, and are set to their default values in this example. In practice, it will make sense to increase the number of CPU cores (`prs`) to allow for parallel processing.

```python
from fsstats.lme import fit
stats = fit(X=design_matrix,
            Zcols=indices_of_random_effects,
            Y=thickness_values,
            ni=cluster_structure,
            maskvtx=None,
            prs=1,
            nit=20,
            e=0.1,
            return_random_effects_estimates=False,
            return_minimal_information=True,
            verbose=False
            )
```

The following code will perform statistical inference for a previously estimated model. This can be used repeatedly, for example for evaluating different contrasts.

```python
from fsstats.lme import test
fstats = test(stats,
              C=contrast_vector,
              bias_correct=False,
              prs=8,
              verbose=False
              )
```

___


## Inputs and outputs

### Input data and format specifications

Argument | Description | Format
------|------------|------------------------
-- table | Univariate data | Table including a header that describes the variable names and a first column of IDs; can have one or more variables. Output from FreeSurfer's `asegstats2table` or `aparcstats2table` will be accepted. Only needed for univariate analysis.
--Y  | Preprocessed mass-univarate data | FreeSurfer mgh/mgz image file, can be obtained from `mris_preproc`. Only needed for mass-univariate analysis.
--mask | Analysis mask for mass-univariate data | FreeSurfer label file, e.g. `lh.cortex`. Optional for mass-univariate analysis; if omitted, all data will be used.
--X  | Design matrix | Table without header and without column of IDs; each row corresponds to one observation and each column corresponds to one parameter/variable in the model.
--C  | Contrast vector/matrix | Table without header, with one or more rows depending on the type of contrast, and as many columns as the design matrix.  Each column refers to the same column of the design matrix. One contrast vector/matrix per file; for multiple contrasts, specify multiple `--C` arguments.
--clusters | Grouping structure of the data | Table without header and a single column. Must have as many rows as the design matrix, and each row corresponds to the same row in the design matrix. Indicates for every observation to which which cluster (e.g., person) it belongs; can be a list of repeated participant IDs.

All tables can be separated by tab, comma, or space.

### Output files and their interpretation

The program will compute the following statistics, and produce text- or image-based output files. Text files can be accessed with any text editor. Image files follow the FreeSurfer format and conventions, and are intended to be used with visualization and post-processing tools like `Freeview` and `surfcluster`, which are available from any FreeSurfer distribution.

file           |   description
---------------|----------------------------------------------------------------
logfile.txt    | logfile
Xg.dat         | design matrix
beta.mgh       | model parameters
estimates.pkl  | internal file with parameter estimates
statistics.pkl | internal file with statistics
\<contrast_directory>/C.dat     | contrast vector or matrix
\<contrast_directory>/gamma.mgh | contrast estimate
\<contrast_directory>/sgn.mgh   | sign of the contrast
\<contrast_directory>/F.mgh     | F-value of the contrast
\<contrast_directory>/sig.mgh   | Sig-value of the contrast

The contrast estimate results from the application of the contrast vector or matrix to the model parameters. The sign of the contrast is computed as `sign(gamma)` and can be used for the interpretation of the results of the undirected *F* - test. The program produces `sig-values` instead of `p-values` for better representation and visualization of values that are very small or close to zero. `p-values` and `sig-values` can be converted into each other: `p = 10^(-sig)` and `sig = -log10(p)`. The following table gives a few examples.

p      |  sig
-------|-----
0.05   | 1.3
0.01   | 2.0
0.001  | 3.0
0.0001 | 4.0

___

## Citations

If you use these tools in your analysis please cite:

- Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer -
  A fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219
  (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012

- Henschel L*, Kuegler D*, Reuter M. (*co-first). FastSurferVINN:
  Building Resolution-Independence into Deep Learning Segmentation Methods -
  A Solution for HighRes Brain MRI. NeuroImage 251 (2022), 118933.
  https://dx.doi.org/10.1016/j.neuroimage.2022.118933

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

This software is licensed under the MIT License, see associated LICENSE file
for details.

Copyright (c) 2019 Image Analysis Group, DZNE e.V.
