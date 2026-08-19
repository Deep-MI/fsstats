"""
This module provides configuration defaults for the FastSurfer statistical
modeling and inference package.

"""


def get_defaults(x):
    """
    Set and retrieve default values.
    """
    defaults = dict(
        # settings that can also be changed via the cmd line / function arguments
        # will be taken from here and set in _parse_arguments() for cmdline and
        # run_lme() for module. if changing these values here, remember to adjust
        # the argparse help messages as well as the run_lme docstring also
        num_cores=1,
        num_iterations=20,
        epsilon=1e-1,
        # cap on the auto-detected CPU core count used when num_cores is not
        # specified (see _get_num_cores in fsstats.py)
        max_cores=16,
        # settings that are currently hardcoded in _run_lme_estimation and _run_lme_inference:
        # if univariate
        # - return_random_effects_estimates=True
        # - return_minimal_information=False
        # - verbose=True
        # - bias_correct=True
        # if mass-univariate
        # - return_random_effects_estimates=False
        # - return_minimal_information=True
        # - verbose=False
        # - bias_correct=False
    )

    # the glm module (a lean port of FreeSurfer's mri_glmfit) has no numeric
    # defaults of its own: its ordinary-least-squares fit/test are fully
    # vectorized and closed-form, so there is no iteration count, convergence
    # epsilon, or core count to configure. relative to mri_glmfit, glm
    # deliberately does not implement: weighting (--w/--wls), fixed-effects
    # analysis (--yffxvar/--ffxdof), --osgm, --pca, --sim (permutation/monte
    # carlo simulation), --fsgd, DTI/kinetic-modeling options, pruning
    # (--prune), smoothing (--fwhm/--var-fwhm), or --skip-estimation (OLS is
    # cheap enough to always re-fit). contrast sign is only folded into sig/z
    # for single-row (t-test-like) contrasts, matching mri_glmfit's own
    # MRIsetSign behavior; multi-row (F-test) contrasts are left unsigned.

    if x in defaults.keys():
        return defaults[x]
    else:
        raise ValueError(f"{x} is not a valid key in the defaults dict")
