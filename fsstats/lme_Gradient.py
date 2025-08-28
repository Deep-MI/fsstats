import numpy as np

def lme_Gradient(X, Zcols, W, invH, L, phi, re, ni):
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
