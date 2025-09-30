"""
This module provides the main functionality of the fastsurfer statistical
modeling and inference package.

"""

import argparse
import logging
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd

from .cfg.config import get_defaults
from .cfg.logging import setup_logging
from .doc.documentation import _get_help_text
from .lme.fit import fit
from .lme.test import test
from .utils.exporting import write_mgh
from .utils.importing import read_clusters, read_contrasts, read_design_matrix, read_label, read_mgh, read_table

# ==============================================================================
# LOGGING

LOGGER = logging.getLogger(__name__)

# ==============================================================================
# FUNCTIONS

# ------------------------------------------------------------------------------
# _help()

def _help(print_help=True, return_help=False):
    """
    a function to return a help message

    """

    HELPTEXT = _get_help_text()

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
#  convert arguments to params

def _evaluate_args(args):

    """
    an internal function to set and return internal parameters

    """

    # message

    LOGGER.info("Evaluating arguments ...")

    # assemble and return params

    class params:
        pass

    if args.filename is not None:
        params.FILENAME = os.path.abspath(args.filename)
        if args.mask is not None:
            params.MASK = os.path.abspath(args.mask)
        else:
            params.MASK = None
        params.ISTABLE = False
    if args.table is not None:
        params.FILENAME = os.path.abspath(args.table)
        params.MASK = None
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
    params.num_iterations = args.num_iterations
    params.epsilon = args.epsilon

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
    nr    number of random effects
    nc    number of clusters
    """

    # data
    # data must have two dimensions
    if params.ISTABLE:
        tbl = read_table(params.FILENAME)
        params.Y = np.array(tbl.iloc[:,1:])
        params.L = tbl.columns[1:].to_list()
        params.M = None
        params.H = None
    else:
        params.H, params.Y = read_mgh(params.FILENAME)
        if params.MASK is not None:
            params.M = read_label(params.MASK)
        else:
            params.M = None
        params.L = None
    if len(params.Y.shape) != 2:
        LOGGER.error("Data matrix must have two dimensions.")
        raise argparse.ArgumentTypeError

    # design matrix
    # design matrix must have two dimensions
    # design matrix and the data matrix must have the same number of rows
    # design matrix must contain a column of ones
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
        LOGGER.error("Design matrix cannot have redundant columns.")
        raise argparse.ArgumentTypeError("Design matrix cannot have redundant columns.")

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
    # surface must be present in the current directory, the data directory, subjects dir, or freesurfer home (if it exists)
    if params.SURFACE is not None:
        if os.path.exists(params.SURFACE):
            LOGGER.info("Found template surface " + params.SURFACE)
            params.SURFACE_DIR = os.path.abspath(params.SURFACE)
        elif os.path.exists(os.path.join(os.path.dirname(params.FILENAME), params.SURFACE)):
            LOGGER.info("Found template surface " + os.path.join(os.path.dirname(params.FILENAME), params.SURFACE))
            params.SURFACE_DIR = os.path.abspath(os.path.join(os.path.dirname(params.FILENAME), params.SURFACE))
        elif os.environ.get('SUBJECTS_DIR') is not None and os.path.exists(os.path.join(os.environ.get('SUBJECTS_DIR'), params.SURFACE)):
            LOGGER.info("Found template surface " + os.path.join(os.environ.get('SUBJECTS_DIR'), params.SURFACE))
            params.SURFACE_DIR = os.path.join(os.environ.get('SUBJECTS_DIR'), params.SURFACE)
        elif os.environ.get('FREESURFER_HOME') is not None and os.path.exists(os.path.join(os.environ.get('FREESURFER_HOME'), 'subjects', params.SURFACE)):
            LOGGER.info("Found template surface " + os.path.join(os.environ.get('FREESURFER_HOME'), params.SURFACE))
            params.SURFACE_DIR = os.path.join(os.environ.get('FREESURFER_HOME'), params.SURFACE)
        else:
            LOGGER.error("Template surface " + params.SURFACE + " could not be found.")
            raise argparse.ArgumentTypeError("Template surface " + params.SURFACE + " could not be found.")

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

    # determine cores
    if params.num_cores is None:
        prs = np.min(multiprocessing.cpu_count(), 16)
    else:
        prs = params.num_cores

    # estimation
    if params.ISTABLE:
        LOGGER.info("Running univariate analysis")
        stats = fit(
            X=params.X,
            Y=params.Y,
            Zcols=params.U,
            maskvtx=None,
            ni=params.ni,
            prs=prs,
            nit=params.num_iterations,
            e=params.epsilon,
            compute_random_effects=True,
            return_minimal_information=False,
            verbose=True
            )
        pd.DataFrame(params.X).to_csv(os.path.join(params.GLMDIR, 'Xg.dat'), sep=" ", header=False, index=False)
        betas = {}
        for s in stats.keys():
            betas.update({ params.L[s] : stats[s]['Bhat'].squeeze()})
        pd.DataFrame.from_dict(betas, orient="index").to_csv(os.path.join(params.GLMDIR, 'beta.txt'), sep=" ", header=False)
    else:
        LOGGER.info("Running mass-univariate analysis")
        if prs == 1:
            LOGGER.warning("Number of cores to use for parallel processing is set to 1.")
        stats = fit(
            X=params.X,
            Y=params.Y,
            Zcols=params.U,
            ni=params.ni,
            maskvtx=params.M,
            prs=prs,
            nit=params.num_iterations,
            e=params.epsilon,
            compute_random_effects=False,
            return_minimal_information=True,
            verbose=False
            )
        # output
        betas = np.zeros((params.Y.shape[1], 1, 1, params.X.shape[1]))
        for s in stats.keys():
            betas[s,0,0,:] = stats[s]['Bhat'].squeeze()
        write_mgh(dat=betas, filename=os.path.join(params.GLMDIR, 'beta.mgh'))

    # output
    with open(os.path.join(params.GLMDIR, "estimates.pkl"), "wb") as f:
        pickle.dump(stats, f)

    # return
    return params, stats

# ------------------------------------------------------------------------------
#  run inference

def _run_inference(params, stats):

    # determine cores
    if params.num_cores is None:
        prs = np.min(multiprocessing.cpu_count(), 16)
    else:
        prs = params.num_cores

    # inference
    fstats = dict()
    if params.ISTABLE:
        LOGGER.info("Running univariate inference")
        for key, value in params.C.items():
            LOGGER.info("Evaluating contrast " + key)
            fstats[key] = test(stats=stats, C=value, prs=prs, bias_correct=True, verbose=False)
            gamma = dict()
            F = dict()
            sig = dict()
            for s in range(len(fstats[key])):
                gamma.update({params.L[s] : float((params.C[key] @ stats[s]['Bhat']).item())})
                F.update({params.L[s] : fstats[key][s]['F']})
                sig.update({params.L[s] : float(-np.log10(fstats[key][s]['pval']))})
            pd.DataFrame(params.C[key]).to_csv(os.path.join(params.GLMDIR, key, 'C.dat'), header=False, index=False)
            pd.DataFrame.from_dict(gamma, orient='index').to_csv(os.path.join(params.GLMDIR, key, 'gamma.txt'), sep=" ", header=False)
            pd.DataFrame.from_dict(F, orient='index').to_csv(os.path.join(params.GLMDIR, key, 'F.txt'), sep=" ", header=False)
            pd.DataFrame.from_dict(sig, orient='index').to_csv(os.path.join(params.GLMDIR, key, 'sig.txt'), sep=" ", header=False)
    else:
        LOGGER.info("Running mass-univariate inference")
        if prs == 1:
            LOGGER.warning("Number of cores to use for parallel processing is set to 1.")
        for key, value in params.C.items():
            LOGGER.info("Evaluating contrast " + key)
            fstats[key] = test(stats=stats, C=value, prs=prs, bias_correct=False, verbose=False)
            gamma = np.zeros((params.Y.shape[1], 1, 1))
            F = np.zeros((params.Y.shape[1], 1, 1))
            sig = np.zeros((params.Y.shape[1], 1, 1))
            sgn = np.zeros((params.Y.shape[1], 1, 1))
            for s in fstats[key].keys():
                gamma[s,0,0] = float((params.C[key] @ stats[s]['Bhat']).item())
                F[s,0,0] = fstats[key][s]['F']
                sig[s,0,0] = fstats[key][s]['sig']
                sgn[s,0,0] = fstats[key][s]['sgn']
            pd.DataFrame(params.C[key]).to_csv(os.path.join(params.GLMDIR, key, 'C.dat'), header=False, index=False)
            write_mgh(dat=gamma, filename=os.path.join(params.GLMDIR, key, 'gamma.mgh'))
            write_mgh(dat=F, filename=os.path.join(params.GLMDIR, key, 'F.mgh'))
            write_mgh(dat=sig, filename=os.path.join(params.GLMDIR, key, 'sig.mgh'))
            write_mgh(dat=sgn, filename=os.path.join(params.GLMDIR, key, 'sgn.mgh'))

    # output
    with open(os.path.join(params.GLMDIR, "statistics.pkl"), "wb") as f:
        pickle.dump(fstats, f)

    # return
    return fstats

# ------------------------------------------------------------------------------
# _run_lme

def _run_lme(args):
    """
    A function to run the lme submodules

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
            with open(os.path.join(params.GLMDIR, 'estimates.pkl'), 'rb') as f:
                LOGGER.info("Reading " + os.path.join(params.GLMDIR, 'estimates.pkl'))
                stats = pickle.load(f)

        # run inference
        params = _run_inference(params, stats)

    except Exception as e:
        LOGGER.error("Error Information:")
        LOGGER.error("Type: %s", type(e))
        LOGGER.error("Value: %s", e.args)

        raise


# ------------------------------------------------------------------------------
# run_lme

def run_lme(table=None, filename=None, design_matrix=None, contrasts=None, random_effects=None, clusters=None, glmdir=None, mask=None, surface=None, hemi=None, skip_estimation=False, **kwargs):
    """
    Run FastSurfer statistical modeling and inference

    Parameters
    ----------

    table :
        Filename of a table output, e.g. produced by asegstats2table or aparcstats2table.
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
    mask
        Filename of a FreeSurfer label file that will be used to select vertices for analysis.
    surface :
        Name of template surface (e.g., fsaverage). Default: None.
    hemi :
        Name of hemisphere lh or rh). Default: None.
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
            self.num_iterations = get_defaults("num_iterations")
            self.epsilon = get_defaults("epsilon")

            # parse kwargs
            if dct is not None:
                for key, value in dct.items():
                    if key in vars(self).keys():
                        setattr(self, key, value)
                    else:
                        raise AssertionError("Key " + key + " not included in list of default keys")

    #

    my_args = Args(kwargs)

    my_args.table = table
    my_args.filename = filename
    my_args.mask = mask
    my_args.design_matrix = design_matrix
    my_args.contrasts = contrasts
    my_args.random_effects = random_effects
    my_args.clusters = clusters
    my_args.glmdir = glmdir
    if surface is not None and hemi is not None:
        my_args.surface = surface + " " + hemi
    else:
        my_args.surface = None
    my_args.skip_estimation = skip_estimation

    #

    _run_lme(my_args)

    #

    return vars(my_args)
