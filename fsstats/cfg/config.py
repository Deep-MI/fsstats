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
        # will be set in _parse_arguments() for cmdline and run_lme() for module
        # note: if changing these values, remember to adjust the argparse help
        # messages as well as the run_lme docstring also
        num_cores=1,
        num_iterations=20,
        epsilon=1e-1,
        # cap on the auto-detected CPU core count used when num_cores is not
        # specified (see _get_num_cores in fsstats.py)
        max_cores=16,
        # settings that are currently hardcoded in _run_estimation and _run_inferences:
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

    if x in defaults.keys():
        return defaults[x]
    else:
        raise ValueError(f"{x} is not a valid key in the defaults dict")
