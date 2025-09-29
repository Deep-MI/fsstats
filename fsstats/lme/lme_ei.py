import numpy as np


def lme_ei(X, Zcols, W, CBhat, SIGMA, L, phi, ni):
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

