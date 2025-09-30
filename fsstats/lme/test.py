"""
This module provides statistical inference after vertex/voxel-wise linear mixed-effects estimation.
"""

import logging

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import f as f_dist
from tqdm import tqdm


def _test(C, Bhat, CBhat, Zcols, invEI, Pth, Qthth, X, W, ni, label, bias_correct, verbose):
    """
    Internal function to provide inference for the fixed-effects in the linear mixed-effects model.

    Parameters
    ----------
    C : ndarray
        Contrast vector or matrix.
    Bhat : ndarray
        Beta estimates.
    CBhat : ndarray
        Covariance of beta estimates.
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    invEI : ndarray
        Inverse of expected information matrix.
    Pth : ndarray
        ...
    Qthth : ndarray
        ...
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
        Can be None if bias_correct==False.
    W : ndarray
        ...
        Can be None if bias_correct==False.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject. Order must correspond to order of X.
        Can be None if bias_correct==False.
    label : str
        Label used for reporting. Can be None.
    bias_correct : bool
        Perform small-sample bias adjustment of covariance of fixed effects.
    verbose : bool
        Increase verbosity.

    Returns
    -------
    fstats : dict
        Dictionary containing the F-statistic, p-value, sign of the contrast, and degrees of freedom.
    """

    #
    LOGGER = logging.getLogger(__name__)

    # Prepare inputs
    q = len(Zcols)
    nth = q * (q + 1) // 2 + 1

    # Computation of Rijs: Kenward & Roger (1997) “A method for small sample inference in linear mixed models” (Biometrics); bias-adjusted covariance of fixed effects.
    if bias_correct:
        Z = X[:, Zcols]
        p = X.shape[1]
        Rthth = np.zeros((nth, nth, p, p))
        jk = 0
        for k in range(q):
            for j in range(k + 1):
                jk += 1
                uv = 0
                for v in range(q):
                    for u in range(v + 1):
                        uv += 1
                        posi = 1
                        SumR = 0
                        for i in range(len(ni)):
                            posf = posi + ni[i] - 1
                            Xi = X[posi-1:posf, :]
                            Zi = Z[posi-1:posf, :]
                            Wi = W[posi-1:posf, :ni[i]]
                            Ekj = np.zeros((q, q))
                            Ekj[k, j] = 1
                            Euv = np.zeros((q, q))
                            Euv[u, v] = 1
                            Ai = Zi @ Ekj @ Euv @ Zi.T
                            Ri = Xi.T @ Wi @ (Ai + Ai.T) @ Wi @ Xi
                            SumR += Ri
                            posi = posf + 1
                        Rthth[jk-1, uv-1, :, :] = SumR

    # Estimation of the bias in the covariance matrix and computation of the F-statistic and the degrees of freedom of the test
    Bias = 0
    OM = C.T @ np.linalg.inv(C @ CBhat @ C.T) @ C
    A1 = 0
    A2 = 0
    Term1 = OM @ CBhat
    Term2 = CBhat @ OM @ CBhat
    for k in range(nth):
        Pk = Pth[k, :, :]
        for j in range(nth):
            Qkj = Qthth[k, j, :, :]
            Pj = Pth[j, :, :]
            if bias_correct:
                Rkj = Rthth[k, j, :, :]
                Bias += invEI[k, j] * (Qkj - Pk @ CBhat @ Pj - 0.25 * Rkj)
            else:
                Bias += invEI[k, j] * (Qkj - Pk @ CBhat @ Pj)
            A1 += invEI[k, j] * np.trace(Term1 @ Pk @ CBhat) * np.trace(Term1 @ Pj @ CBhat)
            A2 += invEI[k, j] * np.trace(Term1 @ Pk @ Term2 @ Pj @ CBhat)

    szC = C.shape[0]
    Bdf = (A1 + 6 * A2) / (2 * szC)
    g = ((szC + 1) * A1 - (szC + 4) * A2) / ((szC + 2) * A2)
    d = 3 * szC + 2 * (1 - g)
    c1 = g / d
    c2 = (szC - g) / d
    c3 = (szC + 2 - g) / d
    EF = 1 / (1 - A2 / szC)
    VF = (2 / szC) * (1 + c1 * Bdf) / ((1 - c2 * Bdf)**2 * (1 - c3 * Bdf))
    ro = VF / (2 * EF**2)
    m = 4 + (szC + 2) / (szC * ro - 1)
    l = m / (EF * (m - 2))
    Bias = CBhat @ Bias @ CBhat
    CovBhat = CBhat + 2 * Bias
    Fstat_i = l * Bhat.T @ C.T @ np.linalg.inv(C @ CovBhat @ C.T) @ C @ Bhat / szC
    if Fstat_i < 0:
        LOGGER.warning('Index ' + label + ': Negative F-values encountered, setting to zero.')
        Fstat_i = np.array(0)
    pval_i = 1 - f_dist.cdf(Fstat_i, szC, m)
    sig_i = -np.log10(pval_i)
    sgn_i = np.sign(C @ Bhat)
    df_i = [m, szC]

    Fstat_i = Fstat_i.item()
    pval_i = pval_i.item()
    sig_i = sig_i.item()
    sgn_i = sgn_i.item()
    df_i = [ float(i) for i in df_i ]

    # Output
    if verbose:
        LOGGER.info('Index ' + label + ': F(' + f"{df_i[0]:.2f}" + ',' + f"{df_i[1]:.2f}" + ') = ' + str(Fstat_i) + ', p = ' + str(pval_i) + ', sgn = ' + str(sgn_i))

    # Return
    return { 'F' : Fstat_i, 'pval' : pval_i, 'sig' : sig_i, 'sgn' : sgn_i, 'df' : df_i }


def test(stats, C, bias_correct, prs, verbose):
    """
    Inference for the fixed-effects in the linear mixed-effects model.

    Parameters
    ----------
    stats : list
        Structure array containing statistics for every voxel/vertex/variable.
    C : array-like
        Contrast matrix.
    bias_correct : bool
        Bias-adjustment of covariance of fixed effects.
    prs : int
        Number of workers for parallel computing.
    verbose : bool
        Increase verbosity.

    Returns
    -------
    fstats : dict
        A dict of dictionaries containing the F-statistic, p-value, sign of the contrast, and degrees of freedom for every voxel/vertex/variable where estimation was performed. Keys are numerical indices.
    """

    # Processing function
    def process(key, value, C, bias_correct, verbose):
        out = _test(
            C = C,
            Bhat = value['Bhat'],
            CBhat = value['CovBhat'],
            Zcols = value['Zcols'],
            invEI = value['invEI'],
            Pth = value['Pth'],
            Qthth = value['Qthth'],
            X = value['X'],   # can be None, required if bias_correct is True
            W = value['W'],   # can be None, required if bias_correct is True
            ni = value['ni'], # can be None, required if bias_correct is True
            label = str(key),
            bias_correct=bias_correct,
            verbose=verbose
        )

        return key, out

    #
    fstats = dict()
    if prs > 1:
        fstats = dict(list(tqdm(Parallel(return_as="generator", n_jobs=prs)(delayed(process)(i, stats[i], C, bias_correct, verbose)
                for i in stats
                ),
                total = len(stats)
        )))
    else:
        for i in stats:
            key, value = (process(i, stats[i], C, bias_correct, verbose))
            fstats[key] = value

    # Return
    return fstats
