import logging
import multiprocessing
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from .lme_fit_FS import lme_fit_FS

def lme_fit(X, Zcols, Y, ni, maskvtx=None, prs=1, nit=20, e=1e-1, compute_random_effects=False, return_minimal_information=False, verbose=False):
    """
    Vertex/voxel-wise linear mixed-effects estimation.

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
    maskvtx : ndarray, optional
        List of zero-based mask indices (nv,). Default: None (all vertices included).
    prs : int, optional
        Number of CPU cores used for parallel computing. Default: 1.
    nit : int, optional
        Number of iterations. Default is 20.
    e : float, optional
        Convergence epsilon (gradient's norm). Default 10^-1.
    compute_random_effects : bool, optional
        Compute random effects estimates. Default: False.
    return_minimal_information : bool, optional
        Return minimal information. Default: False.
    verbose : bool, optional
        Increase verbosity. Default: False.

    Returns
    -------
    stats : list
        List of dictionaries containing estimates for every voxel/vertex.
    """

    LOGGER = logging.getLogger(__name__)

    # Prepare inputs

    nv0 = Y.shape[1]

    if maskvtx is None:
        maskvtx = np.arange(nv0)

    Y_masked = Y[:, maskvtx]

    nm, nv = Y_masked.shape

    if prs is None:
        prs = multiprocessing.cpu_count()

    # Prepare masked outputs

    stats_masked = [{} for _ in range(nv)]

    # Processing

    if prs > 1:
        results = list(tqdm(Parallel(return_as="generator", n_jobs=prs)(delayed(lme_fit_FS)(
                    X=X,
                    Zcols=Zcols,
                    Y=np.expand_dims(Y[:, i], axis=1),
                    ni=ni,
                    nit=nit,
                    e=e,
                    label=str(i),
                    compute_random_effects=compute_random_effects,
                    return_minimal_information=return_minimal_information,
                    verbose=verbose
                    )
                for i in range(nv)
                ),
                total = nv
            ))
        for i, stats_i in enumerate(results):
            stats_masked[i] = stats_i
    else:
        for i in range(nv):
            stats_masked[i] = lme_fit_FS(
                X=X,
                Zcols=Zcols,
                Y=np.expand_dims(Y[:, i], axis=1),
                ni=ni,
                nit=nit,
                e=e,
                label=str(i),
                compute_random_effects=compute_random_effects,
                return_minimal_information=return_minimal_information,
                verbose=verbose
            )


    # Prepare outputs

    stats = [{'Bhat': None, 'CovBhat': None, 'phisqhat': None, 'Dhat': None,
              'Zcols': None, 'invEI': None, 'Pth': None, 'Qthth': None, 'lreml': None, 'st' : None} for _ in range(nv0)]

    for i in range(nv):
        stats[maskvtx[i]] = stats_masked[i]

    # Summary
    #converged = np.sum([ stats[i]['st'] for i in range(nv0) ])
    #TODO: summation does not work for None
    #LOGGER.info(f'Summary: Algorithm did not converge at {100 * (1 - converged / nv):.2f} percent of the total number of locations.')

    # Return

    return stats


