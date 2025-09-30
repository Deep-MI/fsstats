"""
fsstats CLI
"""

from ..fsstats import _help, _parse_arguments, _run_lme


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
