from tqdm import tqdm
from joblib import Parallel, delayed
from .lme.lme_test_f import lme_test_f

def lme_test(stats, C, bias_correct, prs, verbose):
    """
    Inference for the fixed-effects in the linear mixed-effects model.

    Parameters
    ----------
    stats : list of dict
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
        out = lme_test_f(
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
