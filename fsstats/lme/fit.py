"""
This module provides vertex/voxel-wise linear mixed-effects estimation.
"""

import logging

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import cholesky, det, solve
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def _ei(X, Zcols, W, CBhat, SIGMA, L, phi, ni):
    """
    Expected information matrix for the restricted log-likelihood.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    W : ndarray
         W: Inverses of the estimated marginal covariance matrices for each subject stacked in W.
    CBhat :
        Asymptotic covariance matrix of the fixed effects.
    SIGMA :
        Estimated marginal covariance matrices for each subject stacked in SIGMA.
    L :
        Cholesky factor of the covariance matrix of the random effects (D).
    phi :
        Within-subject standard deviation of the errors.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject. Order must correspond to order of X.

    Returns
    -------
    EI :
        Expected information matrix.
    Pth:
        Matrices useful for inferences on the fixed effects.
    Qthth:
        Matrices useful for inferences on the fixed effects.
    """

    # Prepare inputs
    m = len(ni)
    n = sum(ni)
    q = len(Zcols)
    nth = q * (q + 1) // 2 + 1
    p = X.shape[1]
    Pth = np.zeros((nth, p, p))
    Qthth = np.zeros((nth, nth, p, p))
    Der = np.zeros((nth, n, W.shape[1]))

    # Computation of the first order derivatives of W
    jk = 0
    for k in range(q):
        for j in range(k + 1):
            jk += 1
            posi = 0
            for i in range(m):
                posf = posi + ni[i]
                Zi = X[posi:posf, :][:, Zcols]
                Zki = Zi[:, k]
                Mjki = Zki[:, np.newaxis] @ (L[j, :] @ Zi.T)[np.newaxis, :]
                Mjki = Mjki + Mjki.T
                Wi = W[posi:posf, :ni[i]]
                Der[jk - 1, posi:posf, :ni[i]] = -Wi @ Mjki @ Wi
                posi = posf

    posi = 0
    for i in range(m):
        posf = posi + ni[i]
        Wi = W[posi:posf, :ni[i]]
        Der[nth - 1, posi:posf, :ni[i]] = -2 * phi * Wi @ Wi
        posi = posf

    # Computation of Pth, Qthth and the expected information matrix EI.
    for j in range(nth):
        posi = 0
        Pj = 0
        Bj = Der[j]
        for i in range(m):
            posf = posi + ni[i]
            Pj += X[posi:posf].T @ Bj[posi:posf, :ni[i]] @ X[posi:posf]
            posi = posf
        Pth[j] = Pj

    EI = np.zeros((nth, nth))
    for k in range(nth):
        Bk = Der[k]
        Pk = Pth[k]
        for j in range(k + 1):
            Bj = Der[j]
            Pj = Pth[j]
            posi = 0
            Qkj = 0
            traceBkj = 0
            for i in range(m):
                posf = posi + ni[i]
                Vi = SIGMA[posi:posf, :ni[i]]
                Bkji = Bk[posi:posf, :ni[i]] @ Vi @ Bj[posi:posf, :ni[i]]
                traceBkj += np.trace(Bkji @ Vi)
                Qkji = X[posi:posf].T @ Bkji @ X[posi:posf]
                Qkj += Qkji
                posi = posf
            Qthth[k, j] = Qkj
            Qthth[j, k] = Qkj
            EI[k, j] = 0.5 * (traceBkj - np.trace(CBhat @ (2 * Qkj - Pk @ CBhat @ Pj)))
            EI[j, k] = EI[k, j]

    # Return
    return EI, Pth, Qthth


def _gradient(X, Zcols, W, invH, L, phi, re, ni):
    """
    Gradient vector for the restricted log-likelihood.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    W : ndarray
         W: Inverses of the estimated marginal covariance matrices for each subject stacked in W.
    invH :
        Asymptotic covariance matrix of the fixed effects.
    L :
        Cholesky factor of the covariance matrix of the random effects (D).
    phi :
        Within-subject standard deviation of the errors.
    re :
        Residuals.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject. Order must correspond to order of X.

    Returns
    -------
    gr :
        Gradient vector.
    """

    # Initialization
    m = len(ni)
    q = len(Zcols)
    Z = X[:, Zcols]
    nth = q * (q + 1) // 2 + 1
    gr = np.zeros(nth)

    # Derivatives for L
    jk = 0
    for k in range(q):
        for j in range(k + 1):
            jk += 1
            posi = 0
            a1 = 0
            a2 = 0
            M1 = 0
            for i in range(m):
                posf = posi + ni[i]
                Zi = Z[posi:posf, :]
                Wi = W[posi:posf, :ni[i]]
                Xi = X[posi:posf, :]
                rei = re[posi:posf]
                Mjki = np.outer(Zi[:, k], L[j, :] @ Zi.T)
                Mjki = Mjki + Mjki.T
                Maux = Mjki @ Wi
                a1 += np.trace(-Maux)
                a2 += rei.T @ Wi @ Maux @ rei
                M1 += Xi.T @ Wi @ Maux @ Xi
                posi = posf
            a3 = np.trace(-invH @ M1)
            gr[jk - 1] = (a1 + a2 - a3) / 2

    # Derivative for phi
    posi = 0
    a1 = 0
    a2 = 0
    M1 = 0
    for i in range(m):
        posf = posi + ni[i]
        Wi = W[posi:posf, :ni[i]]
        Xi = X[posi:posf, :]
        rei = re[posi:posf]
        a1 += np.trace(-Wi)
        Maux = Wi @ Wi
        a2 += rei.T @ Maux @ rei
        M1 += Xi.T @ Maux @ Xi
        posi = posf
    a3 = np.trace(-invH @ M1)
    gr[nth - 1] = phi * (a1 + a2 - a3)

    # Return
    return gr


def _fit_init(X, Zcols, y, ni, Bhat=None):
    """
    Starting values for linear mixed-effects iterative estimation.
    These starting values are based on the ordinary least squares estimators.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    y : ndarray
        Data matrix (nm, nv) whose columns represent the data vector for each voxel/vertex. Rows must correspond to rows in X.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject. Order must correspond to order of X.
    Bhat : ndarray, optional
        Precomputed OLS estimate (p, nv) for y, i.e. pinv(X) @ y. Useful when this function is called
        repeatedly for many columns of a shared y with the same X, to avoid recomputing pinv(X) each time.
        If None (default), computed internally.

    Returns
    -------
    phisq0 : float
        Estimated within-subject variability.
    D0 : ndarray
        Estimated random effects covariance matrix.
    """

    m = len(ni)
    p = X.shape[1]
    q = len(Zcols)
    n = sum(ni)
    Z = X[:, Zcols]
    if Bhat is None:
        Bhat = np.linalg.pinv(X) @ y
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    posi = 0

    for i in range(m):
        posf = posi + ni[i]
        Zi = Z[posi:posf, :]
        Xi = X[posi:posf, :]
        yi = y[posi:posf]
        ri = yi - Xi @ Bhat
        t = np.linalg.pinv(Zi.T @ Zi)
        t1 += t
        bihat = t @ Zi.T @ ri
        t2 += yi.T @ yi - bihat.T @ Zi.T @ ri
        t3 += Xi.T @ yi
        t4 += bihat @ bihat.T
        posi = posf

    phisq0 = (t2 - Bhat.T @ t3) / (n - (m - 1) * q - p)
    if phisq0 <= 0:
        phisq0 = 1

    D0 = (t4 - phisq0 * t1) / m

    try:
        np.linalg.cholesky(D0)
    except np.linalg.LinAlgError:
        # Handling non-positive definite initial D
        ED, EV = np.linalg.eigh(D0)
        ED[ED < 0] = 1e-4
        D0 = EV @ np.diag(ED) @ np.linalg.inv(EV)

    return D0, phisq0


def _fit_step(X, Z, Y, D, phisq, ni):
    """
    One Fisher-scoring update step: given the current random-effects covariance D and residual
    variance phisq, compute the per-cluster inverse marginal covariance W, the stacked marginal
    covariance SIGMA, the fixed-effects information matrix H (and its inverse), and the fixed-effects
    estimate(s) Bhat. Shared by the per-vertex (_fit_fs, below) and per-region
    (spatial.py::_fit_region) Fisher-scoring iterations, which differ only in whether Y (and hence
    Bhat) has one column (a single vertex) or one column per vertex in a region.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Z : ndarray
        Random-effects design matrix, X[:, Zcols] (nm, nr).
    Y : ndarray
        Data matrix (nm, ncols); ncols=1 for a single vertex, or one column per vertex in a region.
    D : ndarray
        Current estimate of the random effects covariance matrix (nr, nr).
    phisq : float
        Current estimate of the within-subject residual variance.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject. Order must correspond to order of X.

    Returns
    -------
    W : ndarray
        Inverses of the estimated marginal covariance matrices for each subject, stacked (nm, max(ni)).
    SIGMA : ndarray
        Estimated marginal covariance matrices for each subject, stacked (nm, max(ni)).
    H : ndarray
        Fisher information matrix for the fixed effects (p, p).
    invH : ndarray
        Inverse of H, the asymptotic covariance matrix of the fixed effects (p, p).
    Bhat : ndarray
        Fixed-effects estimate(s) (p, ncols).
    """

    n = X.shape[0]
    p = X.shape[1]
    q = Z.shape[1]
    m = len(ni)
    W = np.zeros((n, max(ni)))
    SIGMA = np.zeros_like(W)
    H = np.zeros((p, p))
    Term = np.zeros((p, Y.shape[1]))
    scInvD = solve(D, np.eye(q)) * phisq
    posi = 0
    for i in range(m):
        posf = posi + ni[i]
        Zi = Z[posi:posf, :]
        Wi = (np.eye(ni[i]) - Zi @ solve(Zi.T @ Zi + scInvD, Zi.T)) / phisq
        W[posi:posf, :ni[i]] = Wi
        SIGMA[posi:posf, :ni[i]] = Zi @ D @ Zi.T + np.eye(ni[i]) * phisq
        Xi = X[posi:posf, :]
        Ti = Xi.T @ Wi
        H += Ti @ Xi
        Term += Ti @ Y[posi:posf, :]
        posi = posf
    invH = solve(H, np.eye(p))
    Bhat = invH @ Term

    return W, SIGMA, H, invH, Bhat


def _fit_fs(X, Zcols, Y, ni, nit, e, label, return_random_effects_estimates, return_minimal_information, verbose, Bhat0=None):
    """
    Linear mixed-effects estimation by the Fisher scoring algorithm.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    Y : ndarray
        Data matrix (nm, nv) whose columns represent the data vector for each voxel/vertex. Rows must correspond to rows in X.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject. Order must correspond to order of X.
    nit : int
        Number of iterations.
    e : float
        Convergence epsilon (gradient norm).
    label : str
        Label used for reporting. Can be None.
    return_random_effects_estimates : bool
        Compute random effects estimates.
    return_minimal_information : bool
        Return minimal information.
    verbose : bool
        Increase verbosity.
    Bhat0 : ndarray, optional
        Precomputed OLS estimate (p, 1) for Y, i.e. pinv(X) @ Y, forwarded to _fit_init's starting-value
        computation. Since X (and hence pinv(X)) is the same for every vertex, callers that fit many
        vertices against the same X can precompute it once and pass it in here to avoid recomputing
        pinv(X) on every call. If None (default), computed internally by _fit_init.

    Returns
    -------
    stats : dict
        Dictionary containing the results of the estimation.
    """

    # Prepare inputs
    Z = X[:, Zcols]
    m = len(ni)
    p = X.shape[1]
    q = len(Zcols)
    nth = int(q * (q + 1) / 2 + 1)
    ind = np.concatenate([np.zeros(q*q, dtype=bool), np.ones(1, dtype=bool)])
    for k in range(q):
        for j in range(k+1):
            ind[(k*q)+j] = True
    lreml = np.zeros(nit)

    # Starting values
    D, phisq = _fit_init(X, Zcols, Y, ni, Bhat=Bhat0)
    L = cholesky(D)
    phi = np.sqrt(phisq)
    theta = np.concatenate([L.flatten('F'), phi.flatten()]) # phi has dims (1,1), therefore flatten

    # Iterations
    tf = True
    it = 0
    while tf:
        it += 1
        # Computation of W = SIGMA^-1 and H
        W, SIGMA, H, invH, Bhat = _fit_step(X, Z, Y, D, phisq, ni)
        r = Y - X @ Bhat
        posi = 0
        lremlit = 0
        for i in range(m):
            posf = posi + ni[i]
            Wi = W[posi:posf, :ni[i]]
            ri = r[posi:posf]
            lremlit += np.log(det(Wi)) - ri.T @ Wi @ ri
            posi = posf

        gr = _gradient(X, Zcols, W, invH, L, phi, r, ni)
        EI, Pth, Qthth = _ei(X, Zcols, W, invH, SIGMA, L, phi, ni)
        invEI = solve(EI, np.eye(nth))
        theta[ind] += invEI @ gr
        lremlit = 0.5 * (lremlit - np.log(det(H)))
        eps = np.linalg.norm(gr)
        lreml[it-1] = lremlit
        if verbose:
            LOGGER.info(f'Likelihood at FS iteration {it}: {lremlit.item()}; gradient norm: {eps}')

        # Termination
        if it == nit or eps < e:
            tf = False
            if return_random_effects_estimates:
                posi = 0
                Cbihat = np.zeros((q * m, q))
                bihat = np.zeros((m, q))
                for i in range(m):
                    posf = posi + ni[i]
                    Zi = Z[posi:posf, :]
                    Wi = W[posi:posf, :ni[i]]
                    Xi = X[posi:posf, :]
                    Pi = Wi - Wi @ Xi @ invH @ Xi.T @ Wi
                    ri = r[posi:posf]
                    bihat[i, :] = (D @ Zi.T @ Wi @ ri).flatten()
                    Cbihat[i*q:(i+1)*q, :] = D - D @ Zi.T @ Pi @ Zi @ D
                    posi = posf
            else:
                bihat = None
                Cbihat = None
            if return_minimal_information:
                X_out, r_out, ni_out = None, None, None
            else:
                X_out, r_out, ni_out = X, r, ni
            stats = {
                'X': X_out,         # optional in lme_F, (nm, p)
                'Zcols': Zcols,     # required in lme_F, (nr,)
                'Bhat': Bhat,       # required in lme_F,  (p, 1)
                'CovBhat': invH,    # required in lme_F,  (p, p)
                #'SIGMA': SIGMA,     # no longer needed in lme_F, (nm, p)
                #'Dhat': D,          # no longer needed in lme_F, (nr, nr)
                #'phisqhat': phisq,  # no longer needed in lme_F, (1, 1)
                'W': W,             # optional in lme_F, (nm, p)
                'ni': ni_out,       # optional in lme_F, (nc,)
                'invEI': invEI,     # required in lme_F, (2*nr, 2*nr)
                'Pth': Pth,         # required in lme_F, (2*nr, p, p)
                'Qthth': Qthth,     # required in lme_F, (2*nr, 2*nr, p, p)
                'bihat': bihat,     # optional output only, (nc, 1)
                'Covbihat': Cbihat, # optional output only (nc * nr, 1)
                're': r_out,        # output only, (nm, 1)
                'lreml': lreml[:it],# output only, (nit,)
                'st' : it != nit ,  # output only, (nit,)
            }
        else:
            L = theta[:q*q].reshape((q, q), order='F')
            D = L.T @ L
            phi = theta[-1]
            phisq = phi * phi

    # output
    if stats['st'] and stats['lreml'][-1]>=stats['lreml'][0]:
        LOGGER.info(f"Index {label}: Convergence at iteration {len(stats['lreml'])}. Initial and final likelihoods: {stats['lreml'][0]}, {stats['lreml'][-1]}")
    else:
        LOGGER.info(f"Index {label}: Algorithm did not converge. Initial and final likelihoods: {stats['lreml'][0]}, {stats['lreml'][-1]}")

    # return
    return stats


def fit(X, Zcols, Y, clusters, maskvtx, prs, nit, e, return_random_effects_estimates, return_minimal_information, verbose):
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
    clusters : list
        List of clusters (nm), where each item corresponds to the same row in the design matrix and indicates which cluster the observation belongs to. List must be sorted (not necessarily alphabetically or numerically) according to clusters, i.e. cluster elements must be in consecutive blocks.
    maskvtx : ndarray
        List of zero-based mask indices (nv,). If None, all voxels/vertices/variables are included.
    prs : int
        Number of CPU cores used for parallel computing.
    nit : int
        Number of iterations.
    e : float
        Convergence epsilon (gradient's norm).
    return_random_effects_estimates : bool
        Compute random effects estimates.
    return_minimal_information : bool
        Return minimal information.
    verbose : bool
        Increase verbosity.

    Returns
    -------
    stats : dict
        A dict of dictionaries containing estimates for every voxel/vertex/variable where the model was fit.
        The indices of the voxels/vertices/variables are the keys of the outer dictionary. Each inner dictionary
        has the following keys:

            - 'X': design matrix,
            - 'Zcols': indices of random effects,
            - 'Bhat': beta estimates,
            - 'CovBhat': covariance of beta estimates,
            - 'W': stacked inverses of the individual marginal covariance matrices,
            - 'ni': number of repeated measures for each subject,
            - 'invEI': inverse of expected information matrix,
            - 'Pth': auxiliary, precomputed matrices for useful for inferences on the fixed effects,
            - 'Qthth': auxiliary, precomputed matrices for useful for inferences on the fixed effects,
            - 'bihat': random effects estimates,
            - 'Covbihat': covariance of random effects,
            - 're': residuals,
            - 'lreml': log restricted maximum likelihood,
            - 'st' : status.

    Notes
    -----

    The dimensions of the inputs are based on:

    nm :
        number of observations
    p :
        number of fixed effects in the model, including the intercepts
    nr :
        number of random effects in the model
    nc :
        number of clusters
    """

    # Create number of observations from clusters; dictionary keys will be sorted by order of appearance (in contrast
    # to np.unique)

    ni = {}
    for item in clusters:
        ni[item] = ni.get(item, 0) + 1
    ni = np.array(list(ni.values()))

    # Prepare inputs

    nv = Y.shape[1]

    if maskvtx is None:
        maskvtx = np.arange(nv)

    Y_masked = Y[:, maskvtx]

    nv_masked = Y_masked.shape[1]

    # pinv(X) does not depend on Y, so it is the same for every vertex; compute the OLS starting-value
    # estimate for all (masked) vertices at once here, instead of recomputing pinv(X) inside _fit_init
    # on every one of the nv_masked calls to _fit_fs below.
    Bhat_ols = np.linalg.pinv(X) @ Y_masked

    # Processing function
    def process(i, X, Zcols, Y_masked, Bhat_ols, ni, nit, e, return_random_effects_estimates, return_minimal_information, verbose):
        return _fit_fs(
            X=X,
            Zcols=Zcols,
            Y=np.expand_dims(Y_masked[:, i], axis=1),
            ni=ni,
            nit=nit,
            e=e,
            label=str(i),
            return_random_effects_estimates=return_random_effects_estimates,
            return_minimal_information=return_minimal_information,
            verbose=verbose,
            Bhat0=Bhat_ols[:, i:i+1]
        )

    #
    if prs > 1:
        stats_masked = list(tqdm(Parallel(return_as="generator", n_jobs=prs)(delayed(process)(
                    i, X, Zcols, Y_masked, Bhat_ols, ni, nit, e, return_random_effects_estimates, return_minimal_information, verbose
                    )
                for i in range(nv_masked)
                ),
                total = nv_masked
            ))
    else:
        stats_masked = [
            process(i, X, Zcols, Y_masked, Bhat_ols, ni, nit, e, return_random_effects_estimates, return_minimal_information, verbose)
            for i in range(nv_masked)
        ]

    # Prepare outputs

    stats = dict()
    for i in range(nv_masked):
        stats[maskvtx[i]] = stats_masked[i]

    # Summary
    converged = np.sum([ stats[i]['st'] for i in stats ])
    LOGGER.info(f'Summary: Algorithm did not converge at {100 * (1 - converged / len(stats)):.2f} percent of the total number of locations.')

    # Return

    return stats


