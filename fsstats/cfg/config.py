"""
This module provides configuration defaults for the FastSurfer statistical
modeling and inference package.

"""


def get_defaults(x):
    defaults = dict(
        # note: if changing these values, remember to adjust the argparse help
        # messages as well as the run_fsstats docstring also
        num_cores=1,
        num_iterations=20,
        epsilon=1e-1, # convergence epsilon (gradient norm)
        compute_random_effects=True,
        return_minimal_information=False,
        verbose=False,
    )

    if x in defaults.keys():
        return defaults[x]
    else:
        raise ValueError(x + " is not a valid key in the defaults dict")
