"""
fsstats CLI
"""

import warnings

from ..fsstats import _help, _run_glm, _run_lme
from ..utils.parser import _parse_arguments

warnings.simplefilter('always', DeprecationWarning)

def lmefit_deprecated():
    # message

    print("")
    print("---------------------------------------------")
    print("FastSurfer Statistical Modeling and Inference")
    print("---------------------------------------------")
    print("")

    warnings.warn(
        "mri_lmefit is deprecated and will be removed in a future version. Use fsstats_lme instead.",
        category=DeprecationWarning,
        stacklevel=2
    )

    # parse arguments

    args = _parse_arguments("lme")

    # return help or run analysis

    if args.more_help is True:
        _help()
    else:
        _run_lme(args)


def lmefit():
    # message

    print("")
    print("---------------------------------------------")
    print("FastSurfer Statistical Modeling and Inference")
    print("---------------------------------------------")
    print("")

    # parse arguments

    args = _parse_arguments("lme")

    # return help or run analysis

    if args.more_help is True:
        _help()
    else:
        _run_lme(args)


def glmfit():
    # message

    print("")
    print("---------------------------------------------")
    print("FastSurfer Statistical Modeling and Inference")
    print("---------------------------------------------")
    print("")

    # parse arguments

    args = _parse_arguments("glm")

    # return help or run analysis

    if args.more_help is True:
        _help()
    else:
        _run_glm(args)
