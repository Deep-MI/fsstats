import logging
import numpy as np
from scipy.linalg import cholesky, solve, det
from .lme_fit_init import lme_fit_init
from .lme_Gradient import lme_Gradient
from .lme_EI import lme_EI

def lme_fit_FS(X, Zcols, Y, ni, nit=20, e=1e-1, label=None, compute_random_effects=False, return_minimal_information=False, verbose=False):
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
    nit : int, optional
        Number of iterations. Default is 20.
    e : float, optional
        Convergence epsilon (gradient's norm). Default 10^-1.
    label : str, optional
        Label used for reporting. Default: None.
    compute_random_effects : bool, optional
        Compute random effects estimates. Default: False.
    return_minimal_information : bool, optional
        Return minimal information. Default: False.
    verbose : bool, optional
        Increase verbosity. Default: False.

    Returns
    -------
    stats : dict
        Dictionary containing the results of the estimation.
    """

    # check for / deal with missing data # TODO: missing data
    #if Xrows is not None:
    #    rows = np.where(Xrows[:, i] == 1)[0]
    #    sID = np.repeat(np.arange(len(ni)), ni)
    #    _, ni_rows = np.unique(sID[rows], return_counts=True)
    #else:
    #    rows = np.arange(nm)
    #    ni_rows = ni

    # optionally catch errors # TODO: catch errors
    # try:
    #     stats_i, st_i = lme_fit_FS(
    #         X[rows, :],
    #         Zcols,
    #         Y=np.expand_dims(Y[rows, i], axis=1),
    #         ni=ni_rows,
    #         nit=nit,
    #         e=e,
    #         compute_random_effects=compute_random_effects,
    #         verbose=verbose)
    # except Exception as em:
    #     LOGGER.error(f'Location {i}: {em}')
    #     stats_i, st_i = None, False

    #
    LOGGER = logging.getLogger(__name__)

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
    n = np.sum(ni)
    W = np.zeros((n, max(ni)))
    SIGMA = np.zeros_like(W)
    lreml = np.zeros(nit)

    # Starting values
    D, phisq = lme_fit_init(X, Zcols, Y, ni)
    L = cholesky(D)
    phi = np.sqrt(phisq)
    theta = np.concatenate([L.flatten('F'), phi.flatten()]) # phi has dims (1,1), therefore flatten

    # Iterations
    tf = True
    it = 0
    while tf:
        it += 1
        # Computation of W = SIGMA^-1 and H
        posi = 0
        H = np.zeros((p, p))
        Term = np.zeros((p, 1))
        scInvD = solve(D, np.eye(q)) * phisq
        for i in range(m):
            posf = posi + ni[i]
            Zi = Z[posi:posf, :]
            #Wi = (np.eye(ni[i]) - Zi @ inv(Zi.T @ Zi + scInvD) @ Zi.T) / phisq
            Wi = (np.eye(ni[i]) - Zi @ solve(Zi.T @ Zi + scInvD, Zi.T)) / phisq
            W[posi:posf, :ni[i]] = Wi
            SIGMA[posi:posf, :ni[i]] = Zi @ D @ Zi.T + np.eye(ni[i]) * phisq
            Xi = X[posi:posf, :]
            Ti = Xi.T @ Wi
            H += Ti @ Xi
            Term += Ti @ Y[posi:posf]
            posi = posf
        invH = solve(H, np.eye(p))

        Bhat = invH @ Term
        r = Y - X @ Bhat
        posi = 0
        lremlit = 0
        for i in range(m):
            posf = posi + ni[i]
            Wi = W[posi:posf, :ni[i]]
            ri = r[posi:posf]
            lremlit += np.log(det(Wi)) - ri.T @ Wi @ ri
            posi = posf

        gr = lme_Gradient(X, Zcols, W, invH, L, phi, r, ni)
        EI, Pth, Qthth = lme_EI(X, Zcols, W, invH, SIGMA, L, phi, ni)
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
            if compute_random_effects:
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
                X = None
                r = None
                ni = None
            # TODO: assess output size, in particular for mass-univariate. should not grow too large.
            stats = {
                'X': X,             # optional in lme_F, (nm, p)
                'Zcols': Zcols,     # required in lme_F, (nr,)
                'Bhat': Bhat,       # required in lme_F,  (p, 1)
                'CovBhat': invH,    # required in lme_F,  (p, p)
                #'SIGMA': SIGMA,     # no longer needed in lme_F, (nm, p)
                #'Dhat': D,          # no longer needed in lme_F, (nr, nr)
                #'phisqhat': phisq,  # no longer needed in lme_F, (1, 1)
                'W': W,             # optional in lme_F, (nm, p)
                'ni': ni,           # optional in lme_F, (nc,)
                'invEI': invEI,     # required in lme_F, (2*nr, 2*nr)
                'Pth': Pth,         # required in lme_F, (2*nr, p, p)
                'Qthth': Qthth,     # required in lme_F, (2*nr, 2*nr, p, p)
                'bihat': bihat,     # optional output only, (nc, 1)
                'Covbihat': Cbihat, # optional output only (nc * nr, 1)
                're': r,            # output only, (nm, 1)
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
        LOGGER.info('Index ' + label + ': Convergence at iteration ' + str(len(stats['lreml'])) + '. Initial and final likelihoods: ' + str(stats['lreml'][0]) + ', ' + str(stats['lreml'][-1]));
    else:
        LOGGER.info('Index ' + label + ': Algorithm did not converge. Initial and final likelihoods: ' + str(stats['lreml'][0]) + ', ' + str(stats['lreml'][-1]))

    # return
    return stats

