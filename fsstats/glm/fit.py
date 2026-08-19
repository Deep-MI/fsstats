"""
This module provides vertex/voxel-wise ordinary least squares (GLM) estimation.
"""

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def fit(X, Y, maskvtx, return_yhat, return_eres):
    """
    Vertex/voxel-wise ordinary least squares estimation.

    Unlike linear mixed-effects estimation, ordinary least squares has a closed-form
    solution and does not require iterating per vertex/voxel/variable; all columns of
    Y are fit in a single vectorized computation.

    Parameters
    ----------
    X : ndarray
        Design matrix (nm, p).
    Y : ndarray
        Data matrix (nm, nv) whose columns represent the data vector for each voxel/vertex/variable.
        Rows must correspond to rows in X.
    maskvtx : ndarray
        List of zero-based mask indices (nv,). If None, all voxels/vertices/variables are included.
    return_yhat : bool
        Return the signal estimate (yhat).
    return_eres : bool
        Return the residual error (eres).

    Returns
    -------
    stats : dict
        Dictionary containing the results of the estimation, with the following keys:

            - 'Bhat': regression coefficients (p, nv_masked),
            - 'rvar': residual error variance (nv_masked,),
            - 'dof': residual degrees of freedom,
            - 'X': design matrix,
            - 'maskvtx': the (zero-based) mask indices that were fit,
            - 'yhat': signal estimate (nm, nv_masked), if return_yhat,
            - 'eres': residual error (nm, nv_masked), if return_eres.

    Notes
    -----

    The dimensions of the inputs are based on:

    nm :
        number of observations
    p :
        number of regressors in the model, including the intercept
    nv :
        number of voxels/vertices/variables
    """

    nm, p = X.shape
    dof = nm - p
    if dof < 1:
        raise ValueError(f"Degrees of freedom must be at least 1 (X has {nm} rows and {p} columns).")

    nv = Y.shape[1]
    if maskvtx is None:
        maskvtx = np.arange(nv)

    Y_masked = Y[:, maskvtx]

    Bhat = np.linalg.pinv(X) @ Y_masked
    eres = Y_masked - X @ Bhat
    rvar = np.sum(eres ** 2, axis=0) / dof

    LOGGER.info(f'Fit {Y_masked.shape[1]} voxels/vertices/variables with {p} regressors and {dof} degrees of freedom.')

    stats = {
        'Bhat': Bhat,
        'rvar': rvar,
        'dof': dof,
        'X': X,
        'maskvtx': maskvtx,
    }
    if return_yhat:
        stats['yhat'] = X @ Bhat
    if return_eres:
        stats['eres'] = eres

    return stats
