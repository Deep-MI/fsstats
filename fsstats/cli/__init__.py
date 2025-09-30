"""
fsstats CLI
"""

from ..fsstats import _help, _run_lme
from ..utils.parser import _parse_arguments


def lmefit():
    # message

    print("")
    print("---------------------------------------------")
    print("FastSurfer Statistical Modeling and Inference")
    print("---------------------------------------------")
    print("")

    # parse arguments

    args = _parse_arguments()

    # return help or run analysis

    if args.more_help is True:
        _help()
    else:
        _run_lme(args)
