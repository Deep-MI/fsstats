import argparse
import sys
from ..cfg.version import get_version
from ..cfg.config import get_defaults

# ------------------------------------------------------------------------------
# parse_arguments

def _parse_arguments():
    """
    an internal function to parse input arguments

    """
    # setup parser
    parser = argparse.ArgumentParser(
        description="This program conducts statistical modeling and inference for FastSurfer output.",
        add_help=False,
    )

    # required arguments (set to required=False, because not required when using --more-help)
    required = parser.add_argument_group("Required arguments")

    required.add_argument(
        "--y",
        dest="filename",
        help="Filename of a concatenated overlay file, e.g. produced by mris_preproc.",
        default=None,
        metavar="<filename>",
        required=False,
    )
    required.add_argument(
        "--mask",
        dest="mask",
        help="Filename of a FreeSurfer label file that will be used to select vertices for analysis.",
        default=None,
        metavar="<filename>",
        required=False,
    )    
    required.add_argument(
        "--table",
        dest="table",
        help="Filename of a table output, e.g. produced by asegstats2table or aparcstats2table.",
        default=None,
        metavar="<filename>",
        required=False,
    )
    required.add_argument(
        "--X",
        dest="design_matrix",
        help="Design matrix.",
        default=None,
        metavar="<filename>",
        required=False
    )
    required.add_argument(
        "--C",
        dest="contrasts",
        help="Filename of a contrast matrix or vector.",
        action="append",
        default=None,
        metavar="filename",
        required=False,
    )
    required.add_argument(
        "--rfx",
        dest="random_effects",
        help="Column indices (1-based) of random effects in the design matrix.",
        nargs="+",
        type=int,
        default=None,
        metavar="<int>",
        required=False
    )
    required.add_argument(
        "--clusters",
        dest="clusters",
        help="Filename of a list of clusters (e.g., case IDs), where each line corresponds to the same row in the design matrix.",
        default=None,
        metavar="<filename>",
        required=False
    )
    required.add_argument(
        "--glmdir",
        dest="glmdir",
        help="Directory where the results will be written.",
        default=None,
        metavar="<directory>",
        required=False,
    )
    required.add_argument(
        "--surface",
        dest="surface",
        help="Template surface (e.g., fsaverage) and hemisphere (lh or rh).",
        nargs=2,
        default=None,
        metavar="<surface name> <lh|rh>",
        required=False,
    )
    required.add_argument(
        "--cores",
        dest="num_cores",
        help="Number of CPU cores used in parallel processing. Default: 1.",
        type=int,
        default=get_defaults("num_cores"),
        metavar="<int>",
        required=False,
    )
    required.add_argument(
        "--iterations",
        dest="num_iterations",
        help="Maximum number of iterations to achieve convergence. Default: 20.",
        type=int,
        default=get_defaults("num_iterations"),
        metavar="<int>",
        required=False,
    )
    required.add_argument(
        "--epsilon",
        dest="epsilon",
        help="Epsilon for gradient norm to determine convergence. Default: 1e-1.",
        type=float,
        default=get_defaults("epsilon"),
        metavar="<float>",
        required=False,
    )       
    required.add_argument(
        "--skip-estimation",
        dest="skip_estimation",
        help="Skip estimation. Useful to compute additional contrasts. Requires existing glmdir with estimates.pkl file. Default: False.",
        default=False,
        action="store_true",
        required=False,
    )    

    # define help
    help = parser.add_argument_group("Getting help")

    help.add_argument("--help", help="Display this help message and exit", action="help")
    help.add_argument(
        "--detailed-help",
        dest="more_help",
        help="Display extensive help message and exit",
        default=False,
        action="store_true",
        required=False,
    )
    help.add_argument(
        "--version", help="Display version number and exit", action="version", version="%(prog)s " + get_version()
    )

    # check if there are any inputs; if not, print help
    if len(sys.argv) == 1:
        args = parser.parse_args(["--help"])
    else:
        args = parser.parse_args()

    #
    if args.more_help is False:
        # check for required arguments, print message and exit
        if args.filename is None and args.table is None:
            print("ERROR: either the --y or the --table argument is required, exiting. Use --help to see details.")
            print()
            sys.exit(1)

        if args.filename is not None and args.table is not None:
            print("ERROR: --y and --table cannot be present at the same time, exiting. Use --help to see details.")
            print()
            sys.exit(1)

        if args.design_matrix is None:
            print("ERROR: the --X argument is required, exiting. Use --help to see details.")
            print()
            sys.exit(1)

        if args.contrasts is None:
            print("ERROR: the --C argument is required, exiting. Use --help to see details.")
            print()
            sys.exit(1)

        if args.random_effects is None:
            print("ERROR: the --rfx argument is required, exiting. Use --help to see details.")
            print()
            sys.exit(1)

        if args.clusters is None:
            print("ERROR: the --clusters argument is required, exiting. Use --help to see details.")
            print()
            sys.exit(1)

        if args.glmdir is None:
            print("ERROR: the --glmdir argument is required, exiting. Use --help to see details.")
            print()
            sys.exit(1)

        if args.filename is not None and args.surface is None:
            print("ERROR: the --surface argument is required when using --filename, exiting. Use --help to see details.")
            print()
            sys.exit(1)

    #
    return args