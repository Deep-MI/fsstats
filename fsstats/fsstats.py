"""
This module provides the main functionality of the fastsurfer statistical
modeling and inference package.

"""

import argparse
import logging
from math import gamma
import os
import pickle
import sys

import numpy as np
import pandas as pd

from .cfg.config import get_defaults
from .cfg.logging import setup_logging
from .cfg.version import get_version
from .doc.documentation import get_help_text
from .utils.importing import read_mgh, read_table, read_design_matrix, read_contrasts, read_clusters
from .lme_fit import lme_fit
from .lme_test import lme_test

# ==============================================================================
# LOGGING

LOGGER = logging.getLogger(__name__)

# ==============================================================================
# FUNCTIONS

# ------------------------------------------------------------------------------
# get_help()

def get_help(print_help=True, return_help=False):
    """
    a function to return a help message

    """

    HELPTEXT = get_help_text()

    if print_help:
        print(HELPTEXT)

    if return_help:
        return HELPTEXT

# ------------------------------------------------------------------------------
# create directories

def _create_directories(args):

    # note that the main output directory has been created during
    # _start_logging() already

    # define list of directories
    list_of_directories = list()
    for i in args.contrasts:
        list_of_directories.append(os.path.join(args.glmdir, os.path.splitext(os.path.basename(i))[0]))

    # loop over list of directories
    for directory in list_of_directories:
        if not os.path.isdir(directory):
            try:
                LOGGER.info("Creating output directory " + directory)
                os.mkdir(directory)
            except OSError as e:
                LOGGER.error("Cannot create output directory " + args.glmdir)
                LOGGER.error("Reason: " + str(e))
                raise

    # return
    return

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
    # TODO: make this argument optional when exting the software to regular glms
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
        "--skip-estimation",
        dest="skip_estimation",
        help="Number of CPU cores used in parallel processing. Default: 1.",
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

# ------------------------------------------------------------------------------
#  convert arguments to params

def _evaluate_args(args):

    """
    an internal function to set and return internal parameters

    """

    # message

    LOGGER.info("Evaluating arguments ...")

    # initialize settings

    class settings:
        pass

    # clean up

    settings.epsilon = get_defaults("epsilon")
    settings.num_iterations = get_defaults("num_iterations")
    settings.compute_random_effects = get_defaults("compute_random_effects")
    settings.minimal_information = get_defaults("return_minimal_information")
    settings.verbose = get_defaults("verbose")

    # assemble and return params

    class params:
        pass

    if args.filename is not None:
        params.FILENAME = os.path.abspath(args.filename)
        params.ISTABLE = False
    if args.table is not None:
        params.FILENAME = os.path.abspath(args.table)
        params.ISTABLE = True
    params.DESIGN_MATRIX = os.path.abspath(args.design_matrix)
    params.CONTRASTS = [ os.path.abspath(x) for x in args.contrasts ]
    params.RFX = args.random_effects
    params.CLUSTERS = os.path.abspath(args.clusters)
    params.GLMDIR = os.path.abspath(args.glmdir)
    if args.surface is not None:
        params.SURFACE = args.surface[0]
        params.HEMI = args.surface[1]
    else:
        params.SURFACE = None
        params.HEMI = None
    params.SKIP_ESTIMATION = args.skip_estimation
    params.num_cores = args.num_cores

    params.internal = settings

    LOGGER.info("Using " + params.GLMDIR + " as output directory")

    return params

# ------------------------------------------------------------------------------
#  check params

def _check_params(params):
    """
    checks that input data corresponds to pre-specified formats
    Y     np.array, nm x nv
    X     np.array, nm x p
    C     np.array, nf x p
    U     np.array, nr
    ni    np.array, nc
    with
    nm    number of observations
    nv    number of vertices / variables
    p     number of parameters
    nf    number of rows in each contrast
    nr    numer of random effects
    nc    number of clusters
    """

    # data
    # data must have two dimensions
    if params.ISTABLE:
        params.Y = np.array(read_table(params.FILENAME).iloc[:,1:])
        params.H = None
    else:
        params.H, params.Y = read_mgh(params.FILENAME)
    if len(params.Y.shape) != 2:
        LOGGER.error("Design matrix must have two dimensions.")
        raise argparse.ArgumentTypeError

    # design matrix
    # design matrix must have two dimensions
    # design matrix and the data matrix must have the same number of rows
    # design matrix must contain a columnn of ones
    # design matrix must not contain redundant columns
    params.X = read_design_matrix(params.DESIGN_MATRIX)
    if len(params.X.shape) != 2:
        LOGGER.error("Design matrix must have two dimensions.")
        raise argparse.ArgumentTypeError("Design matrix must have two dimensions.")
    if params.X.shape[0] != params.Y.shape[0]:
        LOGGER.error("The design matrix X and the data matrix Y must have the same number of rows.")
        raise argparse.ArgumentTypeError("The design matrix X and the data matrix Y must have the same number of rows.")
    if not(np.any(np.sum(params.X==1, axis=0) == params.X.shape[0])):
        LOGGER.error("Design matrix must contain a columns of ones.")
        raise argparse.ArgumentTypeError("Design matrix must contain a columns of ones.")
    if np.linalg.matrix_rank(params.X) != params.X.shape[1]:
        LOGGER.error("Design matrix cannot have redundant columnns.")
        raise argparse.ArgumentTypeError("Design matrix cannot have redundant columnns.")

    # contrasts
    # each contrast matrix must have two dimensions
    # each contrast must have as many columns as the design matrix
    params.C = read_contrasts(params.CONTRASTS)
    if len(params.X.shape) != 2:
        LOGGER.error("Each contrast vector / matrix must have two dimensions (which can be of size 1).")
        raise argparse.ArgumentTypeError("Each contrast vector / matrix must have two dimensions (which can be of size 1).")
    for i in params.C:
        if params.C[i].shape[1] != params.X.shape[1]:
            LOGGER.error("Each contrast vector / matrix must have as many columns as the design matrix.")
            raise argparse.ArgumentTypeError("Each contrast vector / matrix must have as many columns as the design matrix.")

    # random effects
    # indices cannot be larger than number of columns in the design matrix
    params.U = np.array(params.RFX) - 1 # convert from 1-based to 0-based
    if np.max(params.U + 1) > params.X.shape[1]:
        LOGGER.error("Indices of random effects cannot be larger than the number of columns of the design matrix.")
        raise argparse.ArgumentTypeError("Indices of random effects cannot be larger than the number of columns of the design matrix.")

    # clusters
    # entries must correspond to rows of the design matrix
    params.Z = read_clusters(params.CLUSTERS)
    if len(params.Z) != params.X.shape[0]:
        LOGGER.error("List of clusters must have as many items as rows of the design matrix.")
        raise argparse.ArgumentTypeError("List of clusters must have as many items as rows of the design matrix.")
        

    # create number of observations from clusters; dictionary keys will be sorted
    # by order of appearance (in contrast to np.unique)
    ni = {}
    for item in params.Z[0]:
        ni[item] = ni.get(item, 0) + 1
    params.ni = np.array(list(ni.values()))

    # surface
    # surface must be readable
    if params.SURFACE is not None:
        # TODO: check surface
        pass

    # hemi
    # hemi must be lh or rh
    if params.HEMI is not None:
        if params.HEMI != "lh" and params.HEMI != "rh":
            LOGGER.error('Hemisphere after <SURFACE> must be either lh or rh.')
            raise argparse.ArgumentTypeError('Hemisphere after <SURFACE> must be either lh or rh.')

    # skip estimation
    # estimates.pkl must exist in glmdir
    if params.SKIP_ESTIMATION:
        if not(os.path.exists(os.path.join(params.GLMDIR, 'estimates.pkl'))):
            LOGGER.error('Skipping estimation requires existing GLM directory with estimates.pkl file.')
            raise argparse.ArgumentTypeError('Skipping estimation requires existing GLM directory with estimates.pkl file.')

    return params

# ------------------------------------------------------------------------------
#  run estimation

def _run_estimation(params):

    # estimation
    if params.ISTABLE:
        LOGGER.info("Running univariate analysis")
        stats = lme_fit(
            X=params.X,
            Y=params.Y,
            Zcols=params.U,
            ni=params.ni,
            prs=params.num_cores,
            return_minimal_information=False,
            compute_random_effects=True,
            verbose=True
            )
        pd.DataFrame(params.X).to_csv(os.path.join(params.GLMDIR, 'Xg.dat')) # TODO: comma or whitespace? rownames? colnames?
        betas = {}
        for s in range(len(stats)):
            betas.update({ s : stats[s]['Bhat'].squeeze()}) # TODO: keep data labels?
        pd.DataFrame(betas).to_csv(os.path.join(params.GLMDIR, 'beta.txt')) # TODO: comma or whitespace? rownames? colnames?
    else:
        LOGGER.info("Running mass-univariate analysis")
        if params.num_cores == 1:
            LOGGER.warning("Number of cores to use for parallel processing is set to 1.")
        stats = lme_fit(
            X=params.X,
            Y=params.Y,
            Zcols=params.U,
            ni=params.ni,
            prs=params.num_cores,
            return_minimal_information=True,
            compute_random_effects=False,
            verbose=False
            ) 
        # TODO: output
        
    # output
    with open(os.path.join(params.GLMDIR, "estimates.pkl"), "wb") as f:
        pickle.dump(stats, f)

    # return
    return params, stats

# ------------------------------------------------------------------------------
#  run inference

def _run_inference(params, stats):

    # inference
    fstats = dict()    
    if params.ISTABLE:
        LOGGER.info("Running univariate inference")
        for key, value in params.C.items():
            LOGGER.info("Evaluating contrast " + key)
            fstats[key] = lme_test(stats=stats, C=value, prs=params.num_cores, bias_correct=True)
            gamma = dict()
            F = dict()
            sig = dict()
            for s in range(len(fstats[key])):
                gamma.update({s : float((params.C[key] @ stats[s]['Bhat']).item())})
                F.update({ s : fstats[key][s]['F']}) # TODO: keep data labels?
                sig.update({ s : float(-np.log10(fstats[key][s]['pval']))}) # TODO: keep data labels?
            pd.DataFrame(params.C[key]).to_csv(os.path.join(params.GLMDIR, key, 'C.dat')) # TODO: comma or whitespace? rownames? colnames?    
            pd.DataFrame(gamma, index=[0]).to_csv(os.path.join(params.GLMDIR, key, 'gamma.txt')) # TODO: comma or whitespace? rownames? colnames?    
            pd.DataFrame(F, index=[0]).to_csv(os.path.join(params.GLMDIR, key, 'F.txt')) # TODO: comma or whitespace? rownames? colnames?       
            pd.DataFrame(sig, index=[0]).to_csv(os.path.join(params.GLMDIR, key, 'sig.txt')) # TODO: comma or whitespace? rownames? colnames?    

            
    else:
        LOGGER.info("Running mass-univariate inference")
        if params.num_cores == 1:
            LOGGER.warning("Number of cores to use for parallel processing is set to 1.")
        for key, value in params.C.items():
            LOGGER.info("Evaluating contrast " + key)
            fstats[key] = lme_test(stats=stats, C=value, prs=params.num_cores, bias_correct=False)
        # TODO: output

    # output
    with open(os.path.join(params.GLMDIR, "statistics.pkl"), "wb") as f:
        pickle.dump(fstats, f)        
          
    # return
    return fstats

# ------------------------------------------------------------------------------
# _run_fsstats

def _run_fsstats(args):
    """
    a function to run the fsstats submodules

    """

    #
    try:
        #
        setup_logging(args)

        # create directories
        _create_directories(args)

        # convert arguments to params
        params = _evaluate_args(args)

        # check params
        params = _check_params(params)

        # run estimation
        if params.SKIP_ESTIMATION is False:
            params, stats = _run_estimation(params)
        else:
            # TODO: needs to be tested
            with open(os.path.join(params.GLMDIR, 'estimates.pkl'), 'rb') as f:
                stats = pickle.load(f)

        # run inference
        params = _run_inference(params, stats)

    except Exception as e:
        LOGGER.error("Error Information:")
        LOGGER.error("Type: %s", type(e))
        LOGGER.error("Value: %s", e.args)

        raise


# ------------------------------------------------------------------------------
# run_fsstats

def run_fsstats(filename, design_matrix, contrasts, random_effects, clusters, glmdir, surface, skip_estimation, **kwargs):
    """
    Run FastSurfer statistical modeling and inference

    Parameters
    ----------

    filename :
        Filename of a concatenated overlay file, e.g. produced by mris_preproc.
    design_matrix :
        Filename of a design matrix.
    contrasts :
        Filename of a contrast matrix or vector.
    random_effects :
        List of column indices (1-based) of random effects in the design matrix.
    clusters :
        Filename of a list of clusters (e.g., case IDs), where each line corresponds to the same row in the design matrix.
    glmdir :
        Directory where the results will be written.
    surface :
        Name of template surface (e.g., fsaverage) and hemisphere (lh or rh).
    skip_estimation : 
        Skip estimation. Useful to compute additional contrasts. Requires existing glmdir with estimates.pkl file. Default: False.

    Returns
    -------
    dict
        a dictionary of input arguments
    """

    # define class

    class Args:
        def __init__(self, dct=None):
            # get defaults
            self.num_cores = get_defaults("num_cores")

            # parse kwargs
            if dct is not None:
                for key, value in dct.items():
                    if key in vars(self).keys():
                        setattr(self, key, value)
                    else:
                        raise AssertionError("Key " + key + " not included in list of default keys")

    #

    my_args = Args(kwargs)

    my_args.filename = filename
    my_args.design_matrix = design_matrix
    my_args.contrasts = contrasts
    my_args.random_effects = random_effects
    my_args.clusters = clusters
    my_args.glmdir = glmdir
    my_args.surface = surface
    my_args.skip_estimation = skip_estimation

    #

    _run_fsstats(my_args)

    #

    return vars(my_args)


#

if __name__ == "__main__":

    # message

    print("")
    print("---------------------------------------------")
    print("FastSurfer Statistical Modeling and Inference")
    print("---------------------------------------------")
    print("")

    print("Running directly")

    # parse arguments

    args = _parse_arguments()

    # return help or run analysis

    if args.more_help is True:
        get_help()
    else:
        _run_fsstats(args)