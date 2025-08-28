import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from .lme_test_F import lme_test_F

def lme_test(stats, C, bias_correct=False, prs=1, verbose=False):
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
    prs : int, optional
        Number of workers for parallel computing. Default is the number of available cores.
    verbose : bool, optional
        Increase verbosity. Default: False.        

    Returns
    -------
    fstats : dict
        List of dictionaries containing the F-statistic, p-value, sign of the contrast, and degrees of freedom.
    """

    # Check parallel processing
    if prs is None:
        prs = multiprocessing.cpu_count()

    # Prepare inputs

    nv = len(stats)

    # Prepare outputs

    fstats = [ {'F': None, 'pval': None, 'sgn': None, 'df': None } for _ in range(nv) ]

    # Processing

    if prs > 1:
        results = Parallel(n_jobs=prs)(delayed(lme_test_F)(i)(
            # TODO: handle none or missing values, handle prs==1
                C = C, 
                Bhat = stats[i]['Bhat'], 
                CBhat = stats[i]['CBhat'], 
                Zcols = stats[i]['Zcols'], 
                invEI = stats[i]['invEI'], 
                Pth = stats[i]['Pth'], 
                Qthth = stats[i]['Qthth'], 
                X = stats[i]['X'],   # can be None, required if bias_correct is True
                W = stats[i]['W'],   # can be None, required if bias_correct is True
                ni = stats[i]['ni'], # can be None, required if bias_correct is True
                label = str(i),
                bias_correct=bias_correct,
                verbose=verbose
                )
            for i in range(nv)
            )
        for i in range(nv):
            fstats[i] = results[i]

    else:
        for i in range(nv):
            fstats[i] = lme_test_F(
            # TODO: handle none or missing values, handle prs==1
                C = C, 
                Bhat = stats[i]['Bhat'], 
                CBhat = stats[i]['CovBhat'], 
                Zcols = stats[i]['Zcols'], 
                invEI = stats[i]['invEI'], 
                Pth = stats[i]['Pth'], 
                Qthth = stats[i]['Qthth'], 
                X = stats[i]['X'],   # can be None, required if bias_correct is True
                W = stats[i]['W'],   # can be None, required if bias_correct is True
                ni = stats[i]['ni'], # can be None, required if bias_correct is True
                label = str(i),
                bias_correct=bias_correct,
                verbose=verbose
                )

    # Return

    return fstats
