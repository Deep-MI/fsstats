import multiprocessing
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from .lme_test_F import lme_test_F

def lme_test(stats, C, bias_correct, prs, verbose):
    """
    Inference for the fixed-effects in the linear mixed-effects model.

    Parameters
    ----------
    stats : list of dict
        Structure array containing statistics for every voxel/vertex.
    C : array-like
        Contrast matrix.
    bias_correct : bool
        Bias-adjustment of covariance of fixed effects.
    prs : int
        Number of workers for parallel computing. If None, the number of available cores.
    verbose : bool
        Increase verbosity.

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
        results = list(tqdm(Parallel(return_as="generator", n_jobs=prs)(delayed(lme_test_F)(i)(
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
                for i in range(nv)     # TODO: handle none or missing values
                ),
                total = nv
        ))
        for i in range(nv):
            fstats[i] = results[i]

    else:
        for i in range(nv):             # TODO: handle none or missing values
            fstats[i] = lme_test_F(
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
