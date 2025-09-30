"""
This module provides vertex/voxel-wise linear mixed-effects estimation.
"""

import logging

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .lme_fit_fs import lme_fit_fs


def lme_fit(X, Zcols, Y, ni, maskvtx, prs, nit, e, compute_random_effects, return_minimal_information, verbose):
    """
    Vertex/voxel-wise linear mixed-effects estimation.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    Y : ndarray
        Data matrix (nm, nv) whose columns represent the data vector for each voxel/vertex/variable. Rows must correspond to rows in X.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject. Order must correspond to order of X.
    maskvtx : ndarray
        List of zero-based mask indices (nv,). If None, all voxels/vertices/variables are included.
    prs : int
        Number of CPU cores used for parallel computing.
    nit : int
        Number of iterations.
    e : float
        Convergence epsilon (gradient's norm).
    compute_random_effects : bool
        Compute random effects estimates.
    return_minimal_information : bool
        Return minimal information.
    verbose : bool
        Increase verbosity.

    Returns
    -------
    stats : dict
        A dict of dictionaries containing estimates for every voxel/vertex/variable where the model was fit. Keys will be numerical indices.
    """

    LOGGER = logging.getLogger(__name__)

    # Prepare inputs

    nv = Y.shape[1]

    if maskvtx is None:
        maskvtx = np.arange(nv)

    Y_masked = Y[:, maskvtx]

    nv_masked = Y_masked.shape[1]

    # Processing

    if prs > 1:
        stats_masked = list(tqdm(Parallel(return_as="generator", n_jobs=prs)(delayed(lme_fit_fs)(
                    X=X,
                    Zcols=Zcols,
                    Y=np.expand_dims(Y[:, i], axis=1),
                    ni=ni,
                    nit=nit,
                    e=e,
                    label=str(i),
                    compute_random_effects=compute_random_effects,
                    return_minimal_information=return_minimal_information,
                    verbose=verbose
                    )
                for i in range(nv_masked)
                ),
                total = nv_masked
            ))
    else:
        stats_masked = [{} for _ in range(nv_masked)]
        for i in range(nv_masked):
            stats_masked[i] = lme_fit_fs(
                X=X,
                Zcols=Zcols,
                Y=np.expand_dims(Y[:, i], axis=1),
                ni=ni,
                nit=nit,
                e=e,
                label=str(i),
                compute_random_effects=compute_random_effects,
                return_minimal_information=return_minimal_information,
                verbose=verbose
            )

    # Prepare outputs

    stats = dict()
    for i in range(nv_masked):
        stats[maskvtx[i]] = stats_masked[i]

    # Summary
    converged = np.sum([ stats[i]['st'] for i in stats ])
    LOGGER.info(f'Summary: Algorithm did not converge at {100 * (1 - converged / len(stats)):.2f} percent of the total number of locations.')

    # Return

    return stats


