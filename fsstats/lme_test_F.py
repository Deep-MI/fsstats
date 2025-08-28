import logging
import numpy as np
from scipy.stats import f as f_dist
from .lme_EI import lme_EI

def lme_test_F(C, Bhat, CBhat, Zcols, invEI, Pth, Qthth, X=None, W=None, ni=None, label=None, bias_correct=False, verbose=False):
    """
    Inference for the fixed-effects in the linear mixed-effects model.

    Parameters
    ----------
    stats : list of dict
        Structure array containing statistics for every voxel/vertex.
    C : array-like
        Contrast matrix.
    bias_correct : bool, optional
        Bias-adjustment of covariance of fixed effects. Default: False.
    verbose : bool, optional
        Increase verbosity. Default: False.        

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

    # Computation of Pis, Qijs and the expected information matrix EI // should exist already # TODO
    #V = stats['SIGMA']
    #Dhat = stats[i]['Dhat']
    #phisqhat = stats[i]['phisqhat']
    #phi = np.sqrt(phisqhat)
    #L = cholesky(Dhat)
    #EI, Pth, Qthth = lme_EI(X, Zcols, W, CBhat, V, L, phi, ni)
    #invEI = np.linalg.inv(EI)

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
        LOGGER.info('Index ' + label + ': Negative F-values encountered, setting to zero.')
        Fstat_i = np.array(0)
    pval_i = 1 - f_dist.cdf(Fstat_i, szC, m)
    sgn_i = np.sign(C @ Bhat)
    df_i = [m, szC]

    Fstat_i = Fstat_i.item()
    pval_i = pval_i.item()
    sgn_i = sgn_i.item()
    df_i = [ float(i) for i in df_i ]

    # Output
    LOGGER.info('Index ' + label + ': F(' + f"{df_i[0]:.2f}" + ',' + f"{df_i[1]:.2f}" + ') = ' + str(Fstat_i) + ', p = ' + str(pval_i) + ', sgn = ' + str(sgn_i))
        
    # Return
    return { 'F' : Fstat_i, 'pval' : pval_i, 'sgn' : sgn_i, 'df' : df_i }