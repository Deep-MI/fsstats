import numpy as np

def lme_fit_init(X, Zcols, y, ni):
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

    # TODO: check
    try:
        np.linalg.cholesky(D0)
    except np.linalg.LinAlgError:
        # Handling non-positive definite initial D;
        ED, EV = np.linalg.eigh(D0)
        ED[ED < 0] = 10**-4 # TODO: check
        D0 = EV @ np.diag(ED) @ np.linalg.inv(EV)

    return D0, phisq0