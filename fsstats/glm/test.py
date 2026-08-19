"""
This module provides statistical inference after vertex/voxel-wise ordinary least squares estimation.
"""

import logging

import numpy as np
from scipy.stats import f as f_dist

LOGGER = logging.getLogger(__name__)


def test(Bhat, rvar, dof, X, C):
    """
    Vertex/voxel-wise inference for the fixed effects in the ordinary least squares model.

    All voxels/vertices/variables are tested in a single vectorized computation.

    Parameters
    ----------
    Bhat : ndarray
        Regression coefficients (p, nv).
    rvar : ndarray
        Residual error variance (nv,).
    dof : int
        Residual degrees of freedom.
    X : ndarray
        Design matrix (nm, p).
    C : ndarray
        Contrast matrix. Must have 2 dimensions, even for contrast vectors (1, p), where p is the
        number of regressors in the model.

    Returns
    -------
    fstats : dict
        Dictionary containing the results of statistical inference, with the following keys:

            - 'gamma': contrast estimate (J, nv), where J is the number of rows of C,
            - 'F': F-statistic (nv,),
            - 'pval': p-value (nv,),
            - 'sig': sig-value, where sig = -log10(p-value) (nv,),
            - 'sgn': sign of the contrast (nv,) if J == 1, else None,
            - 'df': degrees of freedom, [dof, J].
    """

    J = C.shape[0]
    invXtX = np.linalg.pinv(X.T @ X)
    invM = np.linalg.inv(C @ invXtX @ C.T)
    gamma = C @ Bhat

    F = np.einsum('jn,jk,kn->n', gamma, invM, gamma) / (J * rvar)
    pval = 1 - f_dist.cdf(F, J, dof)
    sig = -np.log10(pval)
    sgn = np.sign(gamma[0]) if J == 1 else None

    LOGGER.info(f'Evaluated contrast with {J} row(s) for {Bhat.shape[1]} voxels/vertices/variables.')

    return {'gamma': gamma, 'F': F, 'pval': pval, 'sig': sig, 'sgn': sgn, 'df': [float(dof), float(J)]}
