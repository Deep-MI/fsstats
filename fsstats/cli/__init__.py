"""
fsstats CLI
"""

from ..fsstats import _parse_arguments, _run_lme, get_help


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
        get_help()
    else:
        _run_lme(args)
