"""
This module provides the main functionality of the fastsurfer statistical
modeling and inference package, for both linear mixed-effects (lme) and
ordinary least squares (glm) modeling and inference.

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
from .glm.fit import fit as fit_glm
from .glm.test import test as test_glm
from .lme.fit import fit as fit_lme
from .lme.spatial import fit_spatial
from .lme.test import test as test_lme
from .utils.exporting import write_mgh
from .utils.helper import _all_identical_elements_consecutive, _signed_sig_and_z
from .utils.importing import (
    read_clusters,
    read_contrasts,
    read_design_matrix,
    read_label,
    read_mask,
    read_mgh,
    read_surface,
    read_table,
)

# ==============================================================================
# LOGGING

LOGGER = logging.getLogger(__name__)

# ==============================================================================
# FUNCTIONS

# ------------------------------------------------------------------------------
# _help()

def _help(print_help=True, return_help=False):
    """
    Returns a help message.

    Parameters
    ----------

    print_help : bool
        Prints the help message.
    return_help : bool
        Returns the help message.

    Returns
    -------
    str or None
        Helptext if return_help was True, otherwise None.
    """

    HELPTEXT = _get_help_text()

    if print_help:
        print(HELPTEXT)

    if return_help:
        return HELPTEXT

# ------------------------------------------------------------------------------
# create directories

def _create_directories(args):
    """
    Create directories.

    Parameters
    ----------

    args : dict
        Input arguments.

    Returns
    -------
    None
        The function does not return any specific objects.
    """

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
                LOGGER.info(f"Creating output directory {directory}")
                os.mkdir(directory)
            except OSError as e:
                LOGGER.error(f"Cannot create output directory {args.glmdir}")
                LOGGER.error(f"Reason: {e}")
                raise

    # return
    return

# ------------------------------------------------------------------------------
#  convert arguments to params

def _evaluate_args(args):
    """
    Set and return internal parameters.

    Shared between the lme and glm modules, since most of the fields (input data,
    design matrix, contrasts, output directory, surface, mask/label) are evaluated
    identically for both; args.module ("lme" or "glm") selects the module-specific
    fields.

    Parameters
    ----------

    args : dict
        Input arguments.

    Returns
    -------
    dict
        Parameters to configure the analysis.
    """

    # message

    LOGGER.info("Evaluating arguments ...")

    # assemble and return params

    class params:
        pass

    params.MODULE = args.module

    if args.filename is not None:
        params.FILENAME = os.path.abspath(args.filename)
        params.LABEL = os.path.abspath(args.label) if args.label is not None else None
        if params.MODULE == "glm":
            params.MASK = os.path.abspath(args.mask) if args.mask is not None else None
        params.ISTABLE = False
    if args.table is not None:
        params.FILENAME = os.path.abspath(args.table)
        params.LABEL = None
        if params.MODULE == "glm":
            params.MASK = None
        params.ISTABLE = True
    params.DESIGN_MATRIX = os.path.abspath(args.design_matrix)
    params.CONTRASTS = [ os.path.abspath(x) for x in args.contrasts ]
    params.GLMDIR = os.path.abspath(args.glmdir)
    if args.surface is not None:
        params.SURFACE = args.surface[0]
        params.HEMI = args.surface[1]
    else:
        params.SURFACE = None
        params.HEMI = None

    if params.MODULE == "lme":
        params.RFX = args.random_effects
        params.CLUSTERS = os.path.abspath(args.clusters)
        params.SKIP_ESTIMATION = args.skip_estimation
        params.SPATIAL_COVARIANCE = args.spatial_covariance
        params.num_cores = args.num_cores
        params.num_iterations = args.num_iterations
        params.epsilon = args.epsilon
    elif params.MODULE == "glm":
        params.YHAT_SAVE = args.yhat_save
        params.ERES_SAVE = args.eres_save

    LOGGER.info(f"Using {params.GLMDIR} as output directory")

    return params

# ------------------------------------------------------------------------------
#  check params

def _check_params(params):
    """
    Checks that input data corresponds to pre-specified formats.

    Shared between the lme and glm modules for the parts that are evaluated
    identically (data, design matrix, contrasts, surface, mask/label); params.MODULE
    ("lme" or "glm") selects the module-specific checks (random effects and clusters
    for lme; the volume mask for glm).

    Expected formats are:

    Y     np.array, nm x nv
    X     np.array, nm x p
    C     np.array, nf x p
    U     np.array, nr           (lme only)
    Z     list, nm               (lme only)
    with
    nm    number of observations
    nv    number of vertices / variables
    p     number of parameters
    nf    number of rows in each contrast
    nr    number of random effects

    Parameters
    ----------

    params : dict
        Parameters to configure the analysis.

    Returns
    -------
    dict
        Parameters to configure the analysis.
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
        if params.LABEL is not None:
            params.M = read_label(params.LABEL)
        elif params.MODULE == "glm" and params.MASK is not None:
            params.M = read_mask(params.MASK)
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
        LOGGER.error("Design matrix must contain a column of ones.")
        raise argparse.ArgumentTypeError("Design matrix must contain a column of ones.")
    if np.linalg.matrix_rank(params.X) != params.X.shape[1]:
        LOGGER.error("Design matrix cannot have redundant columns.")
        raise argparse.ArgumentTypeError("Design matrix cannot have redundant columns.")
    if params.MODULE == "glm" and params.X.shape[0] - params.X.shape[1] < 1:
        LOGGER.error("Design matrix must have at least one more row than columns (degrees of freedom must be at least 1).")
        raise argparse.ArgumentTypeError("Design matrix must have at least one more row than columns (degrees of freedom must be at least 1).")

    # contrasts
    # each contrast matrix must have two dimensions
    # each contrast must have as many columns as the design matrix
    params.C = read_contrasts(params.CONTRASTS)
    for i in params.C:
        if len(params.C[i].shape) != 2:
            LOGGER.error("Each contrast vector / matrix must have two dimensions (which can be of size 1).")
            raise argparse.ArgumentTypeError("Each contrast vector / matrix must have two dimensions (which can be of size 1).")
        if params.C[i].shape[1] != params.X.shape[1]:
            LOGGER.error("Each contrast vector / matrix must have as many columns as the design matrix.")
            raise argparse.ArgumentTypeError("Each contrast vector / matrix must have as many columns as the design matrix.")

    if params.MODULE == "lme":
        # random effects
        # indices cannot be larger than number of columns in the design matrix
        params.U = np.array(params.RFX) - 1 # convert from 1-based to 0-based
        if np.max(params.U + 1) > params.X.shape[1]:
            LOGGER.error("Indices of random effects cannot be larger than the number of columns of the design matrix.")
            raise argparse.ArgumentTypeError("Indices of random effects cannot be larger than the number of columns of the design matrix.")

        # clusters
        # entries must correspond to rows of the design matrix
        # entries must be sorted (not necessarily alphabetically or numerically) according to clusters
        params.Z = read_clusters(params.CLUSTERS)
        if len(params.Z) != params.X.shape[0]:
            LOGGER.error("List of clusters must have as many items as rows of the design matrix.")
            raise argparse.ArgumentTypeError("List of clusters must have as many items as rows of the design matrix.")
        if _all_identical_elements_consecutive(params.Z) is False:
            LOGGER.error("List of clusters must be sorted (not necessarily alphabetically or numerically) according to clusters, i.e. cluster elements must be in consecutive blocks.")
            raise argparse.ArgumentTypeError("List of clusters must be sorted (not necessarily alphabetically or numerically) according to clusters, i.e. cluster elements must be in consecutive blocks.")

    # surface
    # surface must be present in the current directory, the data directory, subjects dir, or freesurfer home (if it exists)
    if params.SURFACE is not None:
        if os.path.exists(params.SURFACE):
            LOGGER.info(f"Found template surface {params.SURFACE}")
            params.SURFACE_DIR = os.path.abspath(params.SURFACE)
        elif os.path.exists(os.path.join(os.path.dirname(params.FILENAME), params.SURFACE)):
            LOGGER.info(f"Found template surface {os.path.join(os.path.dirname(params.FILENAME), params.SURFACE)}")
            params.SURFACE_DIR = os.path.abspath(os.path.join(os.path.dirname(params.FILENAME), params.SURFACE))
        elif os.environ.get('SUBJECTS_DIR') is not None and os.path.exists(os.path.join(os.environ.get('SUBJECTS_DIR'), params.SURFACE)):
            LOGGER.info(f"Found template surface {os.path.join(os.environ.get('SUBJECTS_DIR'), params.SURFACE)}")
            params.SURFACE_DIR = os.path.join(os.environ.get('SUBJECTS_DIR'), params.SURFACE)
        elif os.environ.get('FREESURFER_HOME') is not None and os.path.exists(os.path.join(os.environ.get('FREESURFER_HOME'), 'subjects', params.SURFACE)):
            LOGGER.info(f"Found template surface {os.path.join(os.environ.get('FREESURFER_HOME'), params.SURFACE)}")
            params.SURFACE_DIR = os.path.join(os.environ.get('FREESURFER_HOME'), params.SURFACE)
        else:
            LOGGER.error(f"Template surface {params.SURFACE} could not be found.")
            raise argparse.ArgumentTypeError(f"Template surface {params.SURFACE} could not be found.")

    # hemi
    # hemi must be lh or rh
    if params.HEMI is not None:
        if params.HEMI != "lh" and params.HEMI != "rh":
            LOGGER.error('Hemisphere after <SURFACE> must be either lh or rh.')
            raise argparse.ArgumentTypeError('Hemisphere after <SURFACE> must be either lh or rh.')

    if params.MODULE == "lme":
        # spatial covariance
        # only available for mass-univariate analyses
        # requires --surface to resolve to an anatomical surface file, with a sibling <hemi>.sphere file
        if params.SPATIAL_COVARIANCE:
            if params.ISTABLE:
                LOGGER.error("Spatial covariance is only available for mass-univariate (--y) analyses.")
                raise argparse.ArgumentTypeError("Spatial covariance is only available for mass-univariate (--y) analyses.")
            if not os.path.isfile(params.SURFACE_DIR):
                LOGGER.error("Spatial covariance requires --surface to resolve to an anatomical surface file (e.g. lh.white).")
                raise argparse.ArgumentTypeError(
                    "Spatial covariance requires --surface to resolve to an anatomical surface file (e.g. lh.white)."
                )
            sphere = os.path.join(os.path.dirname(params.SURFACE_DIR), f"{params.HEMI}.sphere")
            if not os.path.exists(sphere):
                LOGGER.error(f"Spherical surface {sphere} could not be found; required for spatial covariance.")
                raise argparse.ArgumentTypeError(
                    f"Spherical surface {sphere} could not be found; required for spatial covariance."
                )
            params.SPHERE_DIR = sphere

        # skip estimation
        # estimates.pkl must exist in glmdir
        if params.SKIP_ESTIMATION:
            if not(os.path.exists(os.path.join(params.GLMDIR, 'estimates.pkl'))):
                LOGGER.error('Skipping estimation requires existing GLM directory with estimates.pkl file.')
                raise argparse.ArgumentTypeError('Skipping estimation requires existing GLM directory with estimates.pkl file.')

    return params

# ------------------------------------------------------------------------------
#  determine number of cores

def _get_num_cores(params, warn_if_single):
    """
    Determine the number of CPU cores to use for parallel processing.

    Only used by the lme module; glm's OLS fit/test are fully vectorized and do
    not use parallel processing.

    Parameters
    ----------

    params : dict
        Parameters to configure the analysis.
    warn_if_single : bool
        Log a warning if only a single core will be used.

    Returns
    -------
    int
        Number of CPU cores to use.
    """

    if params.num_cores is None:
        prs = min(multiprocessing.cpu_count(), get_defaults("max_cores"))
    else:
        prs = params.num_cores

    if warn_if_single and prs == 1:
        LOGGER.warning("Number of cores to use for parallel processing is set to 1.")

    return prs

# ------------------------------------------------------------------------------
#  run lme estimation

def _run_lme_estimation(params):
    """
    Run the lme estimation submodule.

    Parameters
    ----------

    params : dict
        Parameters to configure the analysis.

    Returns
    -------
    dict
        Parameters to configure the analysis.
    dict
        Model estimation results.
    """


    # determine cores
    prs = _get_num_cores(params, warn_if_single=not params.ISTABLE)

    # estimation
    if params.ISTABLE:
        LOGGER.info("Running univariate analysis")
        stats = fit_lme(
            X=params.X,
            Y=params.Y,
            Zcols=params.U,
            maskvtx=None,
            clusters=params.Z,
            prs=prs,
            nit=params.num_iterations,
            e=params.epsilon,
            return_random_effects_estimates=True,
            return_minimal_information=False,
            verbose=True
            )
        pd.DataFrame(params.X).to_csv(os.path.join(params.GLMDIR, 'Xg.dat'), sep=" ", header=False, index=False)
        betas = {}
        for s in stats.keys():
            betas.update({ params.L[s] : stats[s]['Bhat'].squeeze()})
        pd.DataFrame.from_dict(betas, orient="index").to_csv(os.path.join(params.GLMDIR, 'beta.txt'), sep=" ", header=False)
    elif params.SPATIAL_COVARIANCE:
        LOGGER.info("Running mass-univariate analysis with spatial covariance")
        anat_coords, _ = read_surface(params.SURFACE_DIR)
        sphere_coords, sphere_faces = read_surface(params.SPHERE_DIR)
        stats, regions = fit_spatial(
            X=params.X,
            Y=params.Y,
            Zcols=params.U,
            clusters=params.Z,
            maskvtx=params.M,
            coords=anat_coords,
            sphere_coords=sphere_coords,
            faces=sphere_faces,
            prs=prs,
            nit=params.num_iterations,
            e=params.epsilon,
            verbose=False
            )
        # output
        betas = np.zeros((params.Y.shape[1], 1, 1, params.X.shape[1]))
        for s in stats.keys():
            betas[s,0,0,:] = stats[s]['Bhat'].squeeze()
        write_mgh(dat=betas, filename=os.path.join(params.GLMDIR, 'beta.mgh'))
        write_mgh(dat=regions.reshape(-1, 1, 1).astype(np.float32), filename=os.path.join(params.GLMDIR, 'regions.mgh'))
    else:
        LOGGER.info("Running mass-univariate analysis")
        stats = fit_lme(
            X=params.X,
            Y=params.Y,
            Zcols=params.U,
            clusters=params.Z,
            maskvtx=params.M,
            prs=prs,
            nit=params.num_iterations,
            e=params.epsilon,
            return_random_effects_estimates=False,
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
#  run lme inference

def _run_lme_inference(params, stats):
    """
    Run the lme inference submodule.

    Parameters
    ----------

    params : dict
        Parameters to configure the analysis.
    stats : dict
        Results obtained from the estimation submodule.

    Returns
    -------
    dict
        Statistical inference results.

    """

    # determine cores
    prs = _get_num_cores(params, warn_if_single=not params.ISTABLE)

    # inference
    fstats = dict()
    if params.ISTABLE:
        LOGGER.info("Running univariate inference")
        for key, value in params.C.items():
            LOGGER.info(f"Evaluating contrast {key}")
            fstats[key] = test_lme(stats=stats, C=value, prs=prs, bias_correct=True, verbose=False)
            ncontrastrows = value.shape[0]
            gamma = dict()
            F = dict()
            sig = dict()
            z = dict()
            for s in range(len(fstats[key])):
                gamma.update({params.L[s] : float((params.C[key] @ stats[s]['Bhat']).item())})
                F.update({params.L[s] : fstats[key][s]['F']})
                signed_sig, z_val = _signed_sig_and_z(fstats[key][s]['sig'], fstats[key][s]['sgn'], ncontrastrows)
                sig.update({params.L[s] : signed_sig})
                z.update({params.L[s] : z_val})
            pd.DataFrame(params.C[key]).to_csv(os.path.join(params.GLMDIR, key, 'C.dat'), header=False, index=False)
            pd.DataFrame.from_dict(gamma, orient='index').to_csv(os.path.join(params.GLMDIR, key, 'gamma.txt'), sep=" ", header=False)
            pd.DataFrame.from_dict(F, orient='index').to_csv(os.path.join(params.GLMDIR, key, 'F.txt'), sep=" ", header=False)
            pd.DataFrame.from_dict(sig, orient='index').to_csv(os.path.join(params.GLMDIR, key, 'sig.txt'), sep=" ", header=False)
            pd.DataFrame.from_dict(z, orient='index').to_csv(os.path.join(params.GLMDIR, key, 'z.txt'), sep=" ", header=False)
    else:
        LOGGER.info("Running mass-univariate inference")
        for key, value in params.C.items():
            LOGGER.info(f"Evaluating contrast {key}")
            fstats[key] = test_lme(stats=stats, C=value, prs=prs, bias_correct=False, verbose=False)
            ncontrastrows = value.shape[0]
            gamma = np.zeros((params.Y.shape[1], 1, 1))
            F = np.zeros((params.Y.shape[1], 1, 1))
            sig = np.zeros((params.Y.shape[1], 1, 1))
            z = np.zeros((params.Y.shape[1], 1, 1))
            for s in fstats[key].keys():
                gamma[s,0,0] = float((params.C[key] @ stats[s]['Bhat']).item())
                F[s,0,0] = fstats[key][s]['F']
                signed_sig, z_val = _signed_sig_and_z(fstats[key][s]['sig'], fstats[key][s]['sgn'], ncontrastrows)
                sig[s,0,0] = signed_sig
                z[s,0,0] = z_val
            pd.DataFrame(params.C[key]).to_csv(os.path.join(params.GLMDIR, key, 'C.dat'), header=False, index=False)
            write_mgh(dat=gamma, filename=os.path.join(params.GLMDIR, key, 'gamma.mgh'))
            write_mgh(dat=F, filename=os.path.join(params.GLMDIR, key, 'F.mgh'))
            write_mgh(dat=sig, filename=os.path.join(params.GLMDIR, key, 'sig.mgh'))
            write_mgh(dat=z, filename=os.path.join(params.GLMDIR, key, 'z.mgh'))

    # output
    with open(os.path.join(params.GLMDIR, "statistics.pkl"), "wb") as f:
        pickle.dump(fstats, f)

    # return
    return fstats

# ------------------------------------------------------------------------------
# _run_lme

def _run_lme(args):
    """
    Run the lme submodules.

    Parameters
    ----------

    args : dict
        Parsed arguments.

    Returns
    -------
    None
        The function does not return any specific objects.
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
            params, stats = _run_lme_estimation(params)
        else:
            with open(os.path.join(params.GLMDIR, 'estimates.pkl'), 'rb') as f:
                LOGGER.info(f"Reading {os.path.join(params.GLMDIR, 'estimates.pkl')}")
                stats = pickle.load(f)

        # run inference
        #fstats = _run_lme_inference(params, stats)
        _run_lme_inference(params, stats)

    except Exception as e:
        LOGGER.error("Error Information:")
        LOGGER.error("Type: %s", type(e))
        LOGGER.error("Value: %s", e.args)

        raise


# ------------------------------------------------------------------------------
# run_lme

def run_lme(table=None, filename=None, design_matrix=None, contrasts=None, random_effects=None, clusters=None, glmdir=None, label=None, surface=None, hemi=None, skip_estimation=False, spatial_covariance=False, **kwargs):
    """
    Run mixed-effects modeling and inference.

    Parameters
    ----------

    table : str
        Filename of a table output, e.g. produced by asegstats2table or aparcstats2table.
    filename : str
        Filename of a concatenated overlay file, e.g. produced by mris_preproc.
    design_matrix : str
        Filename of a design matrix.
    contrasts : list
        Filename(s) of a contrast matrix or vector.
    random_effects : list
        List of column indices (1-based) of random effects in the design matrix.
    clusters : str
        Filename of a list of clusters (e.g., case IDs), where each line corresponds to the same row in the design matrix.
    glmdir : str
        Directory where the results will be written.
    label : str
        Filename of a FreeSurfer label file that will be used to select vertices for analysis.
    surface : str
        Name of template surface (e.g., fsaverage). Default: None.
    hemi : str
        Name of hemisphere (lh or rh). Default: None.
    skip_estimation : bool
        Skip estimation. Useful to compute additional contrasts. Requires existing glmdir with estimates.pkl file. Default: False.
    spatial_covariance : bool
        Model spatial covariance of the residuals within homogeneous surface regions instead of fitting each
        vertex independently. Only supported for mass-univariate analyses (filename, not table). Requires
        surface to resolve to an anatomical surface file with a sibling <hemi>.sphere file. Default: False.
    **kwargs : dict
        Additional keyword arguments, including:
        - num_cores (int):  Number of CPU cores used in parallel processing. Default: 1.
        - num_iterations (int): Maximum number of iterations to achieve convergence. Default: 20.
        - epsilon (float): Epsilon for gradient norm to determine convergence. Default: 1e-1.

    Returns
    -------
    dict
        A dictionary of input arguments.
    """

    # validate required argument combinations (mirrors the CLI's checks in
    # utils/parser.py._parse_arguments, adapted for library use)
    errors = []
    if filename is None and table is None:
        errors.append("either 'filename' or 'table' must be provided")
    if filename is not None and table is not None:
        errors.append("'filename' and 'table' cannot both be provided")
    if design_matrix is None:
        errors.append("'design_matrix' is required")
    if contrasts is None:
        errors.append("'contrasts' is required")
    if random_effects is None:
        errors.append("'random_effects' is required")
    if clusters is None:
        errors.append("'clusters' is required")
    if glmdir is None:
        errors.append("'glmdir' is required")
    if filename is not None and (surface is None or hemi is None):
        errors.append("'surface' and 'hemi' are both required when 'filename' is provided")
    if spatial_covariance and table is not None:
        errors.append("'spatial_covariance' is only supported for mass-univariate analyses (with 'filename', not 'table')")
    if errors:
        raise ValueError(f"Invalid arguments to run_lme(): {'; '.join(errors)}")

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
                        raise AssertionError(f"Key {key} not included in list of default keys")

    #

    my_args = Args(kwargs)
    my_args.module = "lme"

    my_args.table = table
    my_args.filename = filename
    my_args.label = label
    my_args.design_matrix = design_matrix
    my_args.contrasts = contrasts
    my_args.random_effects = random_effects
    my_args.clusters = clusters
    my_args.glmdir = glmdir
    if surface is not None and hemi is not None:
        my_args.surface = [surface, hemi]
    else:
        my_args.surface = None
    my_args.skip_estimation = skip_estimation
    my_args.spatial_covariance = spatial_covariance

    #

    _run_lme(my_args)

    #

    return vars(my_args)

# ------------------------------------------------------------------------------
#  run glm estimation

def _run_glm_estimation(params):
    """
    Run the glm estimation submodule.

    Parameters
    ----------

    params : dict
        Parameters to configure the analysis.

    Returns
    -------
    dict
        Parameters to configure the analysis.
    dict
        Model estimation results.
    """

    LOGGER.info("Running univariate analysis" if params.ISTABLE else "Running mass-univariate analysis")

    stats = fit_glm(
        X=params.X,
        Y=params.Y,
        maskvtx=params.M,
        return_yhat=params.YHAT_SAVE,
        return_eres=params.ERES_SAVE,
        )

    nv = params.Y.shape[1]
    maskvtx = stats['maskvtx']

    if params.ISTABLE:
        pd.DataFrame(params.X).to_csv(os.path.join(params.GLMDIR, 'Xg.dat'), sep=" ", header=False, index=False)
        labels = [params.L[i] for i in maskvtx]
        pd.DataFrame(stats['Bhat'].T, index=labels).to_csv(os.path.join(params.GLMDIR, 'beta.txt'), sep=" ", header=False)
        pd.Series(stats['rvar'], index=labels).to_csv(os.path.join(params.GLMDIR, 'rvar.txt'), sep=" ", header=False)
        pd.Series(np.sqrt(stats['rvar']), index=labels).to_csv(os.path.join(params.GLMDIR, 'rstd.txt'), sep=" ", header=False)
        if params.YHAT_SAVE:
            pd.DataFrame(stats['yhat'].T, index=labels).to_csv(os.path.join(params.GLMDIR, 'yhat.txt'), sep=" ", header=False)
        if params.ERES_SAVE:
            pd.DataFrame(stats['eres'].T, index=labels).to_csv(os.path.join(params.GLMDIR, 'eres.txt'), sep=" ", header=False)
    else:
        beta = np.zeros((nv, 1, 1, params.X.shape[1]))
        beta[maskvtx,0,0,:] = stats['Bhat'].T
        write_mgh(dat=beta, filename=os.path.join(params.GLMDIR, 'beta.mgh'))
        rvar = np.zeros((nv, 1, 1))
        rvar[maskvtx,0,0] = stats['rvar']
        write_mgh(dat=rvar, filename=os.path.join(params.GLMDIR, 'rvar.mgh'))
        write_mgh(dat=np.sqrt(rvar), filename=os.path.join(params.GLMDIR, 'rstd.mgh'))
        if params.M is not None:
            mask = np.zeros((nv, 1, 1))
            mask[maskvtx,0,0] = 1
            write_mgh(dat=mask, filename=os.path.join(params.GLMDIR, 'mask.mgh'))
        if params.YHAT_SAVE:
            yhat = np.zeros((nv, 1, 1, params.Y.shape[0]))
            yhat[maskvtx,0,0,:] = stats['yhat'].T
            write_mgh(dat=yhat, filename=os.path.join(params.GLMDIR, 'yhat.mgh'))
        if params.ERES_SAVE:
            eres = np.zeros((nv, 1, 1, params.Y.shape[0]))
            eres[maskvtx,0,0,:] = stats['eres'].T
            write_mgh(dat=eres, filename=os.path.join(params.GLMDIR, 'eres.mgh'))

    # return
    return params, stats

# ------------------------------------------------------------------------------
#  run glm inference

def _run_glm_inference(params, stats):
    """
    Run the glm inference submodule.

    Parameters
    ----------

    params : dict
        Parameters to configure the analysis.
    stats : dict
        Results obtained from the estimation submodule.

    Returns
    -------
    dict
        Statistical inference results.
    """

    LOGGER.info("Running univariate inference" if params.ISTABLE else "Running mass-univariate inference")

    fstats = dict()
    nv = params.Y.shape[1]
    maskvtx = stats['maskvtx']
    labels = [params.L[i] for i in maskvtx] if params.ISTABLE else None

    for key, value in params.C.items():
        LOGGER.info(f"Evaluating contrast {key}")
        fstats[key] = test_glm(Bhat=stats['Bhat'], rvar=stats['rvar'], dof=stats['dof'], X=stats['X'], C=value)
        ncontrastrows = value.shape[0]
        sig, z = _signed_sig_and_z(fstats[key]['sig'], fstats[key]['sgn'], ncontrastrows)

        pd.DataFrame(value).to_csv(os.path.join(params.GLMDIR, key, 'C.dat'), header=False, index=False)
        if params.ISTABLE:
            pd.DataFrame(fstats[key]['gamma'].T, index=labels).to_csv(os.path.join(params.GLMDIR, key, 'gamma.txt'), sep=" ", header=False)
            pd.Series(fstats[key]['F'], index=labels).to_csv(os.path.join(params.GLMDIR, key, 'F.txt'), sep=" ", header=False)
            pd.Series(sig, index=labels).to_csv(os.path.join(params.GLMDIR, key, 'sig.txt'), sep=" ", header=False)
            pd.Series(z, index=labels).to_csv(os.path.join(params.GLMDIR, key, 'z.txt'), sep=" ", header=False)
        else:
            gamma = np.zeros((nv, 1, 1, ncontrastrows))
            gamma[maskvtx,0,0,:] = fstats[key]['gamma'].T
            F = np.zeros((nv, 1, 1))
            F[maskvtx,0,0] = fstats[key]['F']
            sig_out = np.zeros((nv, 1, 1))
            sig_out[maskvtx,0,0] = sig
            z_out = np.zeros((nv, 1, 1))
            z_out[maskvtx,0,0] = z
            write_mgh(dat=gamma, filename=os.path.join(params.GLMDIR, key, 'gamma.mgh'))
            write_mgh(dat=F, filename=os.path.join(params.GLMDIR, key, 'F.mgh'))
            write_mgh(dat=sig_out, filename=os.path.join(params.GLMDIR, key, 'sig.mgh'))
            write_mgh(dat=z_out, filename=os.path.join(params.GLMDIR, key, 'z.mgh'))

    # return
    return fstats

# ------------------------------------------------------------------------------
# _run_glm

def _run_glm(args):
    """
    Run the glm submodules.

    Parameters
    ----------

    args : dict
        Parsed arguments.

    Returns
    -------
    None
        The function does not return any specific objects.
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
        params, stats = _run_glm_estimation(params)

        # run inference
        #fstats = _run_glm_inference(params, stats)
        _run_glm_inference(params, stats)

    except Exception as e:
        LOGGER.error("Error Information:")
        LOGGER.error("Type: %s", type(e))
        LOGGER.error("Value: %s", e.args)

        raise


# ------------------------------------------------------------------------------
# run_glm

def run_glm(table=None, filename=None, design_matrix=None, contrasts=None, glmdir=None, mask=None, label=None, surface=None, hemi=None, yhat_save=False, eres_save=False):
    """
    Run GLM (ordinary least squares) modeling and inference.

    Parameters
    ----------

    table : str
        Filename of a table output, e.g. produced by asegstats2table or aparcstats2table.
    filename : str
        Filename of a concatenated overlay file, e.g. produced by mris_preproc.
    design_matrix : str
        Filename of a design matrix.
    contrasts : list
        Filename(s) of a contrast matrix or vector.
    glmdir : str
        Directory where the results will be written.
    mask : str
        Filename of a binary mask volume that will be used to select vertices/voxels for analysis. Mutually
        exclusive with label.
    label : str
        Filename of a FreeSurfer label file that will be used to select vertices for analysis. Mutually
        exclusive with mask.
    surface : str
        Name of template surface (e.g., fsaverage). Default: None.
    hemi : str
        Name of hemisphere (lh or rh). Default: None.
    yhat_save : bool
        Save the signal estimate (yhat). Default: False.
    eres_save : bool
        Save the residual error (eres). Default: False.

    Returns
    -------
    dict
        A dictionary of input arguments.
    """

    # validate required argument combinations (mirrors the CLI's checks in
    # utils/parser.py._parse_arguments, adapted for library use)
    errors = []
    if filename is None and table is None:
        errors.append("either 'filename' or 'table' must be provided")
    if filename is not None and table is not None:
        errors.append("'filename' and 'table' cannot both be provided")
    if design_matrix is None:
        errors.append("'design_matrix' is required")
    if contrasts is None:
        errors.append("'contrasts' is required")
    if glmdir is None:
        errors.append("'glmdir' is required")
    if filename is not None and (surface is None or hemi is None):
        errors.append("'surface' and 'hemi' are both required when 'filename' is provided")
    if mask is not None and label is not None:
        errors.append("'mask' and 'label' cannot both be provided")
    if errors:
        raise ValueError(f"Invalid arguments to run_glm(): {'; '.join(errors)}")

    # define class

    class Args:
        pass

    #

    my_args = Args()
    my_args.module = "glm"

    my_args.table = table
    my_args.filename = filename
    my_args.mask = mask
    my_args.label = label
    my_args.design_matrix = design_matrix
    my_args.contrasts = contrasts
    my_args.glmdir = glmdir
    if surface is not None and hemi is not None:
        my_args.surface = [surface, hemi]
    else:
        my_args.surface = None
    my_args.yhat_save = yhat_save
    my_args.eres_save = eres_save

    #

    _run_glm(my_args)

    #

    return vars(my_args)
