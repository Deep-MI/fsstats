"""
This module provides region-wise linear mixed-effects estimation with spatial
covariance, pooling covariance-component estimation over homogeneous surface
regions instead of fitting every vertex independently. Regions are found by
a region-growing segmentation over a spherical parameterization of the
surface, and within each region the residual covariance across vertices is
modeled with an exponential spatial correlation model based on Euclidean
distance. Singleton regions fall back to the plain per-vertex fit in
``fit.py``.
"""

import logging
import warnings

import numpy as np

from joblib import Parallel, delayed
from scipy.linalg import cholesky, det, solve
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .fit import _fit_fs, _fit_init, _fit_step

LOGGER = logging.getLogger(__name__)

MAX_REGION_SIZE = 90

# ------------------------------------------------------------------------------
# surface geometry helpers


def _surface_adjacency(faces, nv, maskvtx=None):
    """
    Build a vertex adjacency list from a triangle mesh.

    Parameters
    ----------
    faces : ndarray
        Zero-based triangle vertex indices (nf, 3).
    nv : int
        Total number of vertices.
    maskvtx : ndarray, optional
        Zero-based vertex indices to restrict the mesh to. Triangles with any vertex outside
        this set are excluded, matching the original mask-restricted adjacency computation.

    Returns
    -------
    list of ndarray
        For each vertex, the zero-based indices of its adjacent vertices.
    """
    if maskvtx is not None:
        in_mask = np.zeros(nv, dtype=bool)
        in_mask[maskvtx] = True
        faces = faces[in_mask[faces].all(axis=1)]

    neighbors = [set() for _ in range(nv)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    return [np.array(sorted(s), dtype=np.int64) for s in neighbors]


def _spherical_coordinates(coords):
    """
    Convert 3D spherical-surface coordinates to 2D (phi, theta) coordinates.

    Parameters
    ----------
    coords : ndarray
        XYZ coordinates of a spherical surface (nv, 3).

    Returns
    -------
    ndarray
        (phi, theta) coordinates (nv, 2).
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    phi = np.arctan2(y, x)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    return np.column_stack([phi, theta])


def _region_means(labels, data):
    """
    Compute the mean of data within each region and broadcast it back to every vertex in that region.

    Parameters
    ----------
    labels : ndarray
        Region label per vertex (nv,). Label 0 denotes vertices excluded from analysis.
    data : ndarray
        Data whose columns will be averaged per region (np_, nv).

    Returns
    -------
    ndarray
        Per-vertex region means, same shape as data.
    """
    means = np.zeros_like(data)
    for region_id in np.unique(labels):
        idx = labels == region_id
        means[:, idx] = data[:, idx].mean(axis=1, keepdims=True)
    return means


def _region_distances(coords, vertex_indices):
    """
    Compute pairwise Euclidean distances among a set of surface vertices.

    Parameters
    ----------
    coords : ndarray
        Vertex coordinates (nv, 3).
    vertex_indices : ndarray
        Zero-based indices of the vertices belonging to one region.

    Returns
    -------
    ndarray
        Square, symmetric distance matrix (n, n).
    """
    if len(vertex_indices) == 1:
        return np.zeros((1, 1))
    region_coords = coords[vertex_indices, :]
    return cdist(region_coords, region_coords)


# ------------------------------------------------------------------------------
# region homogeneity, splitting, and merging


def _homogeneous(residuals, theta, labels, region_id, nst, prc, max_region_size=MAX_REGION_SIZE):
    """
    Check whether a region is homogeneous in its covariance-component estimates and residuals.

    Parameters
    ----------
    residuals : ndarray
        OLS residuals (n, nv).
    theta : ndarray
        Covariance-component estimates (nth, nv).
    labels : ndarray
        Region label per vertex (nv,).
    region_id : int
        Region to test.
    nst : float
        Number of standard deviations of theta inside a region for it to be considered homogeneous.
    prc : float
        Percent of vertices inside a region that are allowed to not meet the homogeneity criterion.
    max_region_size : int
        Regions larger than this are never considered homogeneous.

    Returns
    -------
    bool
        True if the region is homogeneous.
    """
    idx = np.flatnonzero(labels == region_id)
    if idx.size > max_region_size:
        return False

    region_theta = theta[:, idx]
    mean_theta = region_theta.mean(axis=1, keepdims=True)
    std_theta = region_theta.std(axis=1, ddof=1, keepdims=True) if idx.size > 1 else np.zeros_like(mean_theta)
    threshold = nst * std_theta
    deviation = np.abs(region_theta - mean_theta)
    compliant = np.sum(np.all(deviation <= threshold, axis=0))
    if compliant < prc * idx.size / 100:
        return False

    if idx.size > 1:
        correlation = np.corrcoef(residuals[:, idx], rowvar=False)
        if correlation.min() < 0.5:
            return False

    return True


def _merge_regions(labels, residuals, theta, region_a, region_b, nst, prc):
    """
    Tentatively merge two regions and keep the merge only if the result is homogeneous.

    Parameters
    ----------
    labels : ndarray
        Region label per vertex (nv,).
    residuals : ndarray
        OLS residuals (n, nv).
    theta : ndarray
        Covariance-component estimates (nth, nv).
    region_a, region_b : int
        Region labels to merge.
    nst : float
        Number of standard deviations of theta inside a region for it to be considered homogeneous.
    prc : float
        Percent of vertices inside a region that are allowed to not meet the homogeneity criterion.

    Returns
    -------
    labels : ndarray
        Updated (or unchanged) labels.
    merged : bool
        Whether the merge was accepted.
    """
    keep, absorb = (region_a, region_b) if region_a <= region_b else (region_b, region_a)
    trial = labels.copy()
    trial[trial == absorb] = keep
    if _homogeneous(residuals, theta, trial, keep, nst, prc):
        return trial, True
    return labels, False


def _split_region(labels, residuals, theta, coord2d, region_id, nst, prc):
    """
    Recursively split a non-homogeneous region into up to four spherical-coordinate quadrants,
    then greedily re-merge homogeneous quadrant pairs.

    Parameters
    ----------
    labels : ndarray
        Region label per vertex (nv,).
    residuals : ndarray
        OLS residuals (n, nv).
    theta : ndarray
        Covariance-component estimates (nth, nv).
    coord2d : ndarray
        (phi, theta) spherical coordinates (nv, 2).
    region_id : int
        Region to test/split.
    nst : float
        Number of standard deviations of theta inside a region for it to be considered homogeneous.
    prc : float
        Percent of vertices inside a region that are allowed to not meet the homogeneity criterion.

    Returns
    -------
    labels : ndarray
        Updated labels.
    split : bool
        Whether the region was split.
    """
    if _homogeneous(residuals, theta, labels, region_id, nst, prc):
        return labels, False

    labels = labels.copy()
    idx = np.flatnonzero(labels == region_id)
    centroid = coord2d[idx, :].mean(axis=0)
    quadrant = np.where(coord2d[idx, 0] <= centroid[0], 0, 2) + np.where(coord2d[idx, 1] <= centroid[1], 0, 1)

    present = np.unique(quadrant)
    next_label = labels.max() + 1
    quadrant_to_label = {present[0]: region_id}
    for q in present[1:]:
        quadrant_to_label[q] = next_label
        next_label += 1
    for q, lbl in quadrant_to_label.items():
        labels[idx[quadrant == q]] = lbl

    child_labels = list(quadrant_to_label.values())
    i = 0
    while i < len(child_labels):
        merged_any = False
        k = i + 1
        while k < len(child_labels):
            labels, merged = _merge_regions(labels, residuals, theta, child_labels[i], child_labels[k], nst, prc)
            if merged:
                del child_labels[k]
                merged_any = True
                break
            k += 1
        if not merged_any:
            i += 1

    return labels, True


def _compute_seeds(residuals, theta, coord2d, nst, prc):
    """
    Recursively split the (already mask-restricted) surface into homogeneous seed regions
    and locate one seed vertex per region.

    Parameters
    ----------
    residuals : ndarray
        OLS residuals, restricted to masked vertices (n, nv).
    theta : ndarray
        Covariance-component estimates, restricted to masked vertices (nth, nv).
    coord2d : ndarray
        (phi, theta) spherical coordinates, restricted to masked vertices (nv, 2).
    nst : float
        Number of standard deviations of theta inside a region for it to be considered homogeneous.
    prc : float
        Percent of vertices inside a region that are allowed to not meet the homogeneity criterion.

    Returns
    -------
    labels : ndarray
        Region label per (masked) vertex.
    seed_local : ndarray
        Index (into the masked vertex array) of the seed vertex of each region.
    seed_std : ndarray
        Standard deviation of theta within each seed region (nth, n_regions).
    """
    nv = theta.shape[1]
    labels = np.ones(nv, dtype=np.int64)

    changed = True
    while changed:
        changed = False
        for region_id in np.unique(labels).tolist():
            labels, split = _split_region(labels, residuals, theta, coord2d, region_id, nst, prc)
            changed = changed or split

    region_ids = np.unique(labels)
    n_regions = len(region_ids)
    seed_local = np.zeros(n_regions, dtype=np.int64)
    seed_std = np.zeros((theta.shape[0], n_regions))
    for k, region_id in enumerate(region_ids):
        idx = np.flatnonzero(labels == region_id)
        region_theta = theta[:, idx]
        seed_std[:, k] = region_theta.std(axis=1, ddof=1) if idx.size > 1 else 0.0
        centroid = coord2d[idx, :].mean(axis=0)
        # need to introduce a caveat here: if two points are (almost) identical, i.e. close to precision,
        # we identify all of them, and choose the one with the lower index
        # original: loc = np.argmin(np.sum(np.abs(coord2d[idx, :] - centroid), axis=1))
        arr = np.sum(np.abs(coord2d[idx, :] - centroid), axis=1)
        loc = np.argmin(arr)
        eps = 1e-12
        if np.sum(np.abs(arr - arr[loc]) < eps) > 1:
            loc = np.min(np.where(np.abs(arr - arr[loc]) < eps)[0]).astype(int)
        # end caveat
        seed_local[k] = idx[loc]

    return labels, seed_local, seed_std


def _grow_regions(residuals, theta, coord2d, adjacency, maskvtx, nst, prc, max_region_size=MAX_REGION_SIZE):
    """
    Grow homogeneous seed regions outward along the surface adjacency, then assign any
    remaining ("hole") vertices to their most similar adjacent region.

    Parameters
    ----------
    residuals : ndarray
        OLS residuals for all vertices (n, nv). Columns outside maskvtx are ignored.
    theta : ndarray
        Covariance-component estimates for all vertices (nth, nv). Columns outside maskvtx are ignored.
    coord2d : ndarray
        (phi, theta) spherical coordinates for all vertices (nv, 2).
    adjacency : list of ndarray
        Per-vertex adjacent vertex indices, as returned by _surface_adjacency.
    maskvtx : ndarray
        Zero-based indices of vertices included in the analysis.
    nst : float
        Number of standard deviations of theta inside a region for it to be considered homogeneous.
    prc : float
        Percent of vertices inside a region that are allowed to not meet the homogeneity criterion.
    max_region_size : int
        Regions are not grown beyond this many vertices.

    Returns
    -------
    ndarray
        Region label per vertex (nv,). Label 0 denotes vertices excluded from analysis.
    """
    nv = theta.shape[1]

    masked_labels, seed_local, seed_std = _compute_seeds(
        residuals[:, maskvtx], theta[:, maskvtx], coord2d[maskvtx, :], nst, prc
    )
    seeds = maskvtx[seed_local]
    n_seeds = len(seeds)

    labels = np.zeros(nv, dtype=np.int64)
    labels[maskvtx] = masked_labels
    not_included = labels != 0
    not_included[seeds] = False

    region_members = [np.zeros(max_region_size, dtype=np.int64) for _ in range(n_seeds)]
    for r in range(n_seeds):
        region_members[r][0] = seeds[r]
    process_ptr = np.zeros(n_seeds, dtype=np.int64)
    region_size = np.ones(n_seeds, dtype=np.int64)
    threshold = nst * seed_std

    active = list(range(n_seeds))
    while active:
        i = 0
        while i < len(active):
            r = active[i]
            prvtx = region_members[r][process_ptr[r]]
            process_ptr[r] += 1

            members = region_members[r][: region_size[r]]
            region_params = theta[:, members]
            mean_params = region_params.mean(axis=1, keepdims=True)

            candidates = adjacency[prvtx]
            candidates = candidates[not_included[candidates]]

            grown = True
            while grown and candidates.size > 0 and region_size[r] < max_region_size:
                distances = np.sum(np.abs(theta[:, candidates] - mean_params), axis=0)
                loc = np.argmin(distances)
                candidate = candidates[loc]

                trial_theta = np.concatenate([region_params, theta[:, candidate : candidate + 1]], axis=1)
                trial_mean = trial_theta.mean(axis=1, keepdims=True)
                deviation = np.abs(trial_theta - trial_mean)
                compliant = np.sum(np.all(deviation <= threshold[:, r : r + 1], axis=0))
                n_trial = trial_theta.shape[1]

                if compliant >= prc * n_trial / 100:
                    correlation = np.corrcoef(residuals[:, np.append(members, candidate)], rowvar=False)
                    if correlation.min() >= 0.5:
                        region_size[r] += 1
                        region_members[r][region_size[r] - 1] = candidate
                        not_included[candidate] = False
                        members = region_members[r][: region_size[r]]
                        region_params = theta[:, members]
                        mean_params = region_params.mean(axis=1, keepdims=True)
                    candidates = np.delete(candidates, loc)
                else:
                    grown = False

            if process_ptr[r] == region_size[r] or region_size[r] == max_region_size:
                del active[i]
            else:
                i += 1

    for r in range(n_seeds):
        labels[region_members[r][: region_size[r]]] = r + 1

    holes = np.flatnonzero(not_included)
    labels[holes] = np.arange(n_seeds + 1, n_seeds + 1 + holes.size)

    for h in holes:
        neighbors = adjacency[h]
        if neighbors.size == 0:
            continue
        neighbor_labels = labels[neighbors]
        best_label = neighbor_labels[0]
        best_dist = np.sum(np.abs(theta[:, h] - theta[:, labels == best_label].mean(axis=1)))
        for lbl in neighbor_labels[1:]:
            dist = np.sum(np.abs(theta[:, h] - theta[:, labels == lbl].mean(axis=1)))
            if dist < best_dist:
                best_dist = dist
                best_label = lbl
        labels, _ = _merge_regions(labels, residuals, theta, labels[h], best_label, nst, prc)

    return labels


def _create_regions(sphere_coords, faces, theta, residuals, maskvtx, nst=2, prc=95):
    """
    Segment a surface into homogeneous regions comprising locations with similar covariance components.

    Parameters
    ----------
    sphere_coords : ndarray
        Spherical-surface vertex coordinates (nv, 3).
    faces : ndarray
        Zero-based triangle vertex indices of the same mesh (nf, 3).
    theta : ndarray
        Covariance-component estimates for all vertices (nth, nv), e.g. from _fit_init_all.
    residuals : ndarray
        OLS residuals for all vertices (n, nv), e.g. from _fit_init_all.
    maskvtx : ndarray
        Zero-based indices of vertices included in the analysis.
    nst : float
        Number of standard deviations of theta inside a region for it to be considered homogeneous. Default 2.
    prc : float
        Percent of vertices inside a region that are allowed to not meet the homogeneity criterion. Default 95.

    Returns
    -------
    regions : ndarray
        Region label per vertex (nv,). Label 0 denotes vertices excluded from analysis.
    region_ids : ndarray
        Unique region labels
    region_means : ndarray
        Per-vertex mean of theta within its region, same shape as theta.
    """
    nv = theta.shape[1]
    adjacency = _surface_adjacency(faces, nv, maskvtx)
    coord2d = _spherical_coordinates(sphere_coords)
    regions = _grow_regions(residuals, theta, coord2d, adjacency, maskvtx, nst, prc)
    region_means = _region_means(regions, theta)
    n_regions = len(np.unique(regions[regions != 0]))
    LOGGER.info(f"{n_regions} homogeneous regions were obtained.")
    region_ids = np.unique(regions[regions != 0])
    return regions, region_ids, region_means


# ------------------------------------------------------------------------------
# whole-surface starting values

def _fit_init_all(X, Zcols, Y, ni, prs):
    """
    Starting values at each (already masked) location for the region-wise linear mixed-effects
    iterative estimation, based on the ordinary least squares estimators.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    Y : ndarray
        Data matrix (nm, nv) whose columns represent the ordered data vector for each vertex.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject.
    prs : int
        Number of CPU cores used for parallel computing.

    Returns
    -------
    Theta0 : ndarray
        Covariance-component estimates at each location (nth, nv).
    Re0 : ndarray
        OLS residuals at each location (nm, nv).
    """
    q = len(Zcols)
    nth = q * (q + 1) // 2 + 1
    nv = Y.shape[1]

    ind = np.zeros(q * q, dtype=bool)
    for k in range(q):
        for j in range(k + 1):
            ind[k * q + j] = True

    Bhat_ols = np.linalg.pinv(X) @ Y

    def process_compute_theta0_init(i):
        Theta0 = np.zeros((nth))
        D0, phisq0 = _fit_init(X, Zcols, Y[:, i : i + 1], ni, Bhat=Bhat_ols[:, i : i + 1])
        Theta0[:-1] = D0.flatten("F")[ind]
        Theta0[-1] = phisq0.item()
        return Theta0

    if prs > 1:
        results = list(
            tqdm(
                Parallel(return_as="generator", n_jobs=prs)(delayed(process_compute_theta0_init)(nvi) for nvi in range(nv)),
                total=nv,
            )
        )
    else:
        results = [process_compute_theta0_init(nvi) for nvi in range(nv)]

    Theta0 = np.transpose(np.asarray(results))

    # original, non-parallelized code:
    # Theta0 = np.zeros((nth, nv))
    # for i in range(nv):
    #     D0, phisq0 = _fit_init(X, Zcols, Y[:, i : i + 1], ni, Bhat=Bhat_ols[:, i : i + 1])
    #     Theta0[:-1, i] = D0.flatten("F")[ind]
    #     Theta0[-1, i] = phisq0.item()

    Re0 = Y - X @ Bhat_ols

    return Theta0, Re0


def _fit_em(X, Zcols, y, ni, ths, nit):

    def _safe_inv(A, label="matrix"):
        """
        Inverse of a square matrix, falling back to the Moore-Penrose
        pseudo-inverse if `A` is singular or numerically ill-conditioned.

        Parameters
        ----------
        A : ndarray
            Square matrix to invert.
        label : str
            Name used in the warning message if the fallback is triggered.

        Returns
        -------
        ndarray
            inv(A), or pinv(A) if A is (near-)singular.
        """
        try:
            return np.linalg.solve(A, np.eye(A.shape[0]))
        except np.linalg.LinAlgError:
            warnings.warn(
                f"{label} is singular; falling back to pseudo-inverse.",
                RuntimeWarning,
            )
            return np.linalg.pinv(A)

    # Auxiliary variables
    m = len(ni)
    n = np.sum(ni)
    nimax = np.max(ni)
    p = X.shape[1]
    q = len(Zcols)
    nDth = q * q
    ind = [False] * nDth
    for k in range(1,q+1):
        for j in range(1,k+1):
            ind[(k-1) * q + j - 1] = True

    # Iterations
    W = np.zeros((n, nimax))

    D = np.zeros((nDth,1))
    D[ind] = ths[:-1].reshape(-1,1)
    D = D.reshape((q, q), order='F')
    D = D + np.triu(D, 1).T
    phisq = ths[-1]

    for k in range(nit):
        # Computation of W = SIGMA^-1 and H.
        posi = 0
        H = np.zeros((p, p))
        Term = np.zeros(p)

        #scInvD = np.linalg.solve(D, np.eye(q)) * phisq
        scInvD = _safe_inv(D, label="D") * phisq

        for i in range(m):
            posf = posi + ni[i]  # exclusive end (posf = posi+ni(i)-1 in 1-based -> posi+ni[i] exclusive in 0-based)
            Zi = X[posi:posf, Zcols]
            Xi = X[posi:posf, :]

            M = Zi.T @ Zi + scInvD
            Wi = (np.eye(ni[i]) - Zi @ np.linalg.solve(M, Zi.T)) / phisq
            W[posi:posf, 0:ni[i]] = Wi

            Ti = Xi.T @ Wi
            H = H + Ti @ Xi
            Term = Term + Ti @ y[posi:posf]

            posi = posf

        #invH = np.linalg.solve(H, np.eye(p))
        invH = _safe_inv(H, label="H")

        # Estimation
        Bhat = invH @ Term
        r = y - X @ Bhat

        Dhat = np.zeros((q, q))
        phisqhat = 0.0
        posi = 0

        for i in range(m):
            posf = posi + ni[i]
            Zi = X[posi:posf, Zcols]
            Wi = W[posi:posf, 0:ni[i]]
            Xi = X[posi:posf, :]
            ri = r[posi:posf]

            bihat = D @ Zi.T @ Wi @ ri
            Pi = Wi - Wi @ Xi @ invH @ Xi.T @ Wi

            resid = ri - Zi @ bihat
            phisqhat = phisqhat + resid.T @ resid + phisq * np.trace(
                np.eye(ni[i]) - phisq * Pi
            )
            Dhat = Dhat + np.outer(bihat, bihat) + D - D @ Zi.T @ Pi @ Zi @ D

            posi = posf

        D = Dhat / m
        phisq = phisqhat / n

    D_vec = D.flatten(order='F')
    th = np.concatenate([D_vec[ind], [phisq]])

    return th, r


def _fit_em_all(X, Zcols, Y_masked, ni, prs, Theta0_masked, nit):
    """
    Region-wise linear mixed-effects iterative estimation of the covariance
    components, based on Newton-Raphson updates from starting values, at
    each (already masked) location.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (n, p).
    Zcols : ndarray
        0-based indices of the columns of X that are considered as random
        effects (nr,).
    Y_masked : ndarray
        Data matrix (n, nv) whose columns represent the ordered data vector
        for each vertex/voxel actually being processed.
    ni : ndarray
        List (m,) whose entries are the number of repeated measures for
        each subject.
    prs : int
        Number of CPU cores used for parallel computing.
    Theta0_masked : ndarray
        Starting values of the covariance-component parameters at each
        processed location (nth, nv); iteratively updated.
    nit : int
        Number of inner (Newton-Raphson) iterations.

    Returns
    -------
    Theta0 : ndarray
        Covariance-component estimates at each location (nth, nv0).
    Re0 : ndarray
        Residuals at each location (n, nv0).
    """

    # parallel

    nvm = Y_masked.shape[1]

    def process_compute_theta0_and_re0_em(i):
        Theta0, Re0 = _fit_em(X, Zcols, Y_masked[:, i], ni, Theta0_masked[:, i], nit)
        return Theta0, Re0

    if prs > 1:
        results = list(
            tqdm(
                Parallel(return_as="generator", n_jobs=prs)(delayed(process_compute_theta0_and_re0_em)(nvmi) for nvmi in range(nvm)),
                total=nvm,
            )
        )
    else:
        results = [process_compute_theta0_and_re0_em(nvmi) for nvmi in range(nvm)]

    # results to array

    Theta0 = np.transpose(np.array([x[0] for x in results]))
    Re0 = np.transpose(np.array([x[1] for x in results]))

    return Theta0, Re0

# ------------------------------------------------------------------------------
# region-wise Fisher scoring


def _spatial_derivative(Dist, a):
    """
    Exponential spatial correlation matrix and its derivative with respect to its parameter.

    Parameters
    ----------
    Dist : ndarray
        Square, symmetric matrix of distances among locations inside a region (nv, nv).
    a : float
        Spatial correlation parameter.

    Returns
    -------
    G : ndarray
        Spatial correlation matrix (nv, nv).
    GDa : ndarray
        Derivative of G with respect to a.
    """
    G = np.exp(-a * a * Dist)
    try:
        cholesky(G)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(G)
        G = eigvecs @ np.diag(np.maximum(eigvals, 1e-5)) @ eigvecs.T
    GDa = -2 * a * Dist * G
    return G, GDa


def _gradient_region(X, Zcols, W, invH, L, phi, re, ni, invG, GDa):
    """
    Gradient vector of the restricted log-likelihood for a whole region.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    W : ndarray
        Inverses of the estimated temporal covariance matrices for each subject, stacked.
    invH : ndarray
        Asymptotic covariance matrix of the fixed effects.
    L : ndarray
        Cholesky factor of the covariance matrix of the random effects (D).
    phi : float
        Within-subject standard deviation of the errors.
    re : ndarray
        Residuals (nm, nv_region).
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject.
    invG : ndarray
        Inverse of the spatial correlation matrix (nv_region, nv_region).
    GDa : ndarray
        Derivative of the spatial correlation matrix with respect to a.

    Returns
    -------
    ndarray
        Gradient vector (nth + 1,), the last entry corresponding to the spatial parameter a.
    """
    m = len(ni)
    q = len(Zcols)
    Z = X[:, Zcols]
    nth = q * (q + 1) // 2 + 1
    nv = invG.shape[0]
    gr = np.zeros(nth + 1)

    jk = 0
    for k in range(q):
        for j in range(k + 1):
            posi = 0
            a1 = 0.0
            a2 = 0.0
            M1 = 0
            for i in range(m):
                posf = posi + ni[i]
                Zi = Z[posi:posf, :]
                Wi = W[posi:posf, : ni[i]]
                Xi = X[posi:posf, :]
                Ri = re[posi:posf, :]
                Mjki = np.outer(Zi[:, k], L[j, :] @ Zi.T)
                Mjki = Mjki + Mjki.T
                Maux = Mjki @ Wi
                a1 += -np.trace(Maux)
                a2 += np.trace(Ri.T @ Wi @ Maux @ Ri @ invG)
                M1 += Xi.T @ Wi @ Maux @ Xi
                posi = posf
            a3 = -np.trace(invH @ M1)
            gr[jk] = (nv * a1 + a2 - nv * a3) / 2
            jk += 1

    posi = 0
    a1 = 0.0
    a2 = 0.0
    M1 = 0
    for i in range(m):
        posf = posi + ni[i]
        Wi = W[posi:posf, : ni[i]]
        Xi = X[posi:posf, :]
        Ri = re[posi:posf, :]
        a1 += -np.trace(Wi)
        Maux = Wi @ Wi
        a2 += np.trace(Ri.T @ Maux @ Ri @ invG)
        M1 += Xi.T @ Maux @ Xi
        posi = posf
    a3 = -np.trace(invH @ M1)
    gr[nth - 1] = phi * (nv * a1 + a2 - nv * a3)

    posi = 0
    a2 = 0.0
    Maux1 = GDa @ invG
    Maux2 = invG @ Maux1
    for i in range(m):
        posf = posi + ni[i]
        Wi = W[posi:posf, : ni[i]]
        Ri = re[posi:posf, :]
        a2 += np.trace(Ri.T @ Wi @ Ri @ Maux2)
        posi = posf
    aux = -np.trace(Maux1)
    a1 = np.sum(ni) * aux
    a3 = X.shape[1] * aux
    gr[nth] = (a1 + a2 - a3) / 2

    return gr


def _ei_region(X, Zcols, W, CBhat, L, phi, ni, invG, GDa):
    """
    Expected information matrix of the restricted log-likelihood for a whole region.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    W : ndarray
        Inverses of the estimated temporal covariance matrices for each subject, stacked.
    CBhat : ndarray
        Asymptotic covariance matrix of the fixed effects.
    L : ndarray
        Cholesky factor of the covariance matrix of the random effects (D).
    phi : float
        Within-subject standard deviation of the errors.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject.
    invG : ndarray
        Inverse of the spatial correlation matrix (nv_region, nv_region).
    GDa : ndarray
        Derivative of the spatial correlation matrix with respect to a.

    Returns
    -------
    EI : ndarray
        Expected information matrix (nth + 1, nth + 1).
    Pth : ndarray
        Matrices useful for inferences on the fixed effects.
    Qthth : ndarray
        Matrices useful for inferences on the fixed effects.
    """
    m = len(ni)
    n = int(np.sum(ni))
    nv = invG.shape[0]
    q = len(Zcols)
    nth = q * (q + 1) // 2 + 1
    p = X.shape[1]
    Pth = np.zeros((nth, p, p))
    Qthth = np.zeros((nth, nth, p, p))
    Der = np.zeros((nth, n, W.shape[1]))

    jk = 0
    for k in range(q):
        for j in range(k + 1):
            posi = 0
            for i in range(m):
                posf = posi + ni[i]
                Zi = X[posi:posf, :][:, Zcols]
                Zki = Zi[:, k]
                Mjki = Zki[:, np.newaxis] @ (L[j, :] @ Zi.T)[np.newaxis, :]
                Mjki = Mjki + Mjki.T
                Der[jk, posi:posf, : ni[i]] = Mjki
                posi = posf
            jk += 1
    posi = 0
    for i in range(m):
        posf = posi + ni[i]
        Der[nth - 1, posi:posf, : ni[i]] = 2 * phi * np.eye(ni[i])
        posi = posf

    for j in range(nth):
        posi = 0
        Pj = 0
        Bj = Der[j]
        for i in range(m):
            posf = posi + ni[i]
            Wi = W[posi:posf, : ni[i]]
            Pj = Pj - X[posi:posf, :].T @ Wi @ Bj[posi:posf, : ni[i]] @ Wi @ X[posi:posf, :]
            posi = posf
        Pth[j] = Pj

    EI = np.zeros((nth + 1, nth + 1))
    for k in range(nth):
        Bk = Der[k]
        Pk = Pth[k]
        for j in range(k + 1):
            Bj = Der[j]
            Pj = Pth[j]
            posi = 0
            Qkj = 0
            traceBkj = 0.0
            for i in range(m):
                posf = posi + ni[i]
                Wi = W[posi:posf, : ni[i]]
                Bkji = Wi @ Bk[posi:posf, : ni[i]] @ Wi @ Bj[posi:posf, : ni[i]]
                traceBkj += np.trace(Bkji)
                Bkji = Bkji @ Wi
                Qkji = X[posi:posf, :].T @ Bkji @ X[posi:posf, :]
                Qkj += Qkji
                posi = posf
            Qthth[k, j] = Qkj
            Qthth[j, k] = Qkj
            EI[k, j] = nv * (traceBkj - np.trace(CBhat @ (2 * Qkj - Pk @ CBhat @ Pj)))
            EI[j, k] = EI[k, j]

    Mauxa = GDa @ invG
    trMauxa = np.trace(Mauxa)
    for k in range(nth):
        Bk = Der[k]
        Pk = Pth[k]
        posi = 0
        traceBk = 0.0
        for i in range(m):
            posf = posi + ni[i]
            Wi = W[posi:posf, : ni[i]]
            Bki = Wi @ Bk[posi:posf, : ni[i]]
            traceBk += np.trace(Bki)
            posi = posf
        EI[k, nth] = trMauxa * (traceBk + np.trace(CBhat @ Pk))
        EI[nth, k] = EI[k, nth]

    EI[nth, nth] = (n - p) * np.trace(Mauxa @ Mauxa)
    EI = 0.5 * EI

    return EI, Pth, Qthth


def _fit_region(X, Zcols, Y, ni, Theta0, Dist, nit, e, label, verbose):
    """
    Linear mixed-effects estimation by the Fisher scoring algorithm for a whole region, modeling
    the residual spatial correlation among the region's vertices with an exponential correlation
    model based on Euclidean distance. A region of a single vertex is delegated directly to the
    plain per-vertex fit (fit.py::_fit_fs), since no spatial parameter can be estimated from it.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    Y : ndarray
        Data matrix (nm, nv_region) for the vertices inside this region.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject.
    Theta0 : ndarray
        Initial covariance-component estimates for the vertices inside this region (nth, nv_region).
    Dist : ndarray
        Square, symmetric matrix of distances among the region's vertices (nv_region, nv_region).
    nit : int
        Number of iterations.
    e : float
        Convergence epsilon (gradient's norm).
    label : str
        Label used for reporting.
    verbose : bool
        Increase verbosity.

    Returns
    -------
    list of dict
        One results dictionary per vertex in the region, in the same order as Y's columns, with
        the same keys as fit.py::_fit_fs's output (Bhat differs per vertex; CovBhat/invEI/Pth/Qthth
        are shared across the region).
    """
    nv_region = Y.shape[1]

    if nv_region == 1:
        stats = _fit_fs(
            X=X,
            Zcols=Zcols,
            Y=Y,
            ni=ni,
            nit=nit,
            e=e,
            label=label,
            return_random_effects_estimates=False,
            return_minimal_information=True,
            verbose=verbose,
        )
        return [stats]

    Z = X[:, Zcols]
    m = len(ni)
    n = int(np.sum(ni))
    p = X.shape[1]
    q = len(Zcols)
    nth = q * (q + 1) // 2 + 1

    ind_D = np.zeros(q * q, dtype=bool)
    for k in range(q):
        for j in range(k + 1):
            ind_D[k * q + j] = True

    mTheta0 = Theta0.mean(axis=1)
    D = np.zeros(q * q)
    D[ind_D] = mTheta0[:-1]
    D = D.reshape((q, q), order="F")
    D = D + np.triu(D, 1).T
    phisq = mTheta0[-1]
    L = cholesky(D)
    phi = np.sqrt(phisq)
    a = 0.05
    theta = np.concatenate([L.flatten("F"), [phi, a]])
    ind = np.concatenate([ind_D, np.ones(2, dtype=bool)])

    lreml = np.zeros(nit)

    tf = True
    it = 0
    while tf:
        it += 1
        # Computation of W = SIGMA^-1 and H (SIGMA is unused in the region-wise likelihood)
        W, _, H, invH, Bhat = _fit_step(X, Z, Y, D, phisq, ni)

        G, GDa = _spatial_derivative(Dist, a)
        try:
            invG = solve(G, np.eye(nv_region))
        except np.linalg.LinAlgError:
            invG = np.linalg.pinv(G)

        r = Y - X @ Bhat

        posi = 0
        lreml_w = 0.0
        term = 0.0
        for i in range(m):
            posf = posi + ni[i]
            Wi = W[posi:posf, : ni[i]]
            Ri = r[posi:posf, :]
            lreml_w += np.log(det(Wi))
            term += np.trace(Ri.T @ Wi @ Ri @ invG)
            posi = posf

        lremlit = 0.5 * ((n - p) * np.log(1.0 / det(G)) + nv_region * (lreml_w - np.log(det(H))) - term)

        gr = _gradient_region(X, Zcols, W, invH, L, phi, r, ni, invG, GDa)
        EI, Pth, Qthth = _ei_region(X, Zcols, W, invH, L, phi, ni, invG, GDa)
        invEI = solve(EI, np.eye(nth + 1))
        theta[ind] += invEI @ gr
        eps = np.linalg.norm(gr)
        lreml[it - 1] = lremlit
        if verbose:
            LOGGER.info(f"Likelihood at FS iteration {it} for region {label}: {lremlit}; gradient norm: {eps}")

        if it == nit or eps < e:
            tf = False
            invEI_final = invEI[:nth, :nth]
            stats = []
            for j in range(nv_region):
                stats.append(
                    {
                        "X": None,
                        "Zcols": Zcols,
                        "Bhat": Bhat[:, j : j + 1],
                        "CovBhat": invH,
                        "W": None,
                        "ni": None,
                        "invEI": invEI_final,
                        "Pth": Pth,
                        "Qthth": Qthth,
                        "bihat": None,
                        "Covbihat": None,
                        "re": None,
                        "lreml": lreml[:it],
                        "st": it != nit,
                    }
                )
        else:
            L = theta[: q * q].reshape((q, q), order="F")
            D = L.T @ L
            phi = theta[q * q]
            phisq = phi * phi
            a = theta[q * q + 1]

    if stats[0]["st"] and stats[0]["lreml"][-1] >= stats[0]["lreml"][0]:
        LOGGER.info(
            f"Region {label}: Convergence at iteration {len(stats[0]['lreml'])}. "
            f"Initial and final likelihoods: {stats[0]['lreml'][0]}, {stats[0]['lreml'][-1]}"
        )
    else:
        LOGGER.info(
            f"Region {label}: Algorithm did not converge. "
            f"Initial and final likelihoods: {stats[0]['lreml'][0]}, {stats[0]['lreml'][-1]}"
        )

    return stats


def _fit_region_all(X, Zcols, Y, ni, Theta0, regions, region_ids, coords, nit, e, prs, verbose):
    """
    Linear mixed-effects estimation by the Fisher scoring algorithm for a whole region, modeling
    the residual spatial correlation among the region's vertices with an exponential correlation
    model based on Euclidean distance.

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    Y : ndarray
        Data matrix (nm, nv) whose columns represent the data vector for each vertex.
    ni : ndarray
        List (nc,) whose entries are the number of repeated measures for each subject.
    Theta0 : ndarray
        ...
    regions : ndarray
        ...
    region_ids : ndarray
        ...
    coords : ndarray
        Anatomical surface vertex coordinates (nv, 3), used to compute within-region Euclidean distances.
    nit : int
        Number of iterations.
    e : float
        Convergence epsilon (gradient's norm).
    prs : int
        Number of CPU cores used for parallel computing.
    verbose : bool
        Increase verbosity.

    Returns
    -------
    stats : dict
        A dict of dictionaries containing estimates for every vertex where the model was fit, in the
        same format as fit.py::fit's return value.
    """

    def process_fit_region(region_id):
        vertex_idx = np.flatnonzero(regions == region_id)
        Dist = _region_distances(coords, vertex_idx)
        region_stats = _fit_region(
            X=X,
            Zcols=Zcols,
            Y=Y[:, vertex_idx],
            ni=ni,
            Theta0=Theta0[:, vertex_idx],
            Dist=Dist,
            nit=nit,
            e=e,
            label=str(region_id),
            verbose=verbose,
        )
        return vertex_idx, region_stats

    if prs > 1:
        results = list(
            tqdm(
                Parallel(return_as="generator", n_jobs=prs)(delayed(process_fit_region)(region_id) for region_id in region_ids),
                total=len(region_ids),
            )
        )
    else:
        results = [process_fit_region(region_id) for region_id in region_ids]

    # Output

    stats = dict()
    for vertex_idx, region_stats in results:
        for local_i, global_i in enumerate(vertex_idx):
            stats[global_i] = region_stats[local_i]

    return stats


# ------------------------------------------------------------------------------
# top-level orchestration


def fit_spatial(X, Zcols, Y, clusters, maskvtx, coords, sphere_coords, faces, prs, nit, e, nst=2, prc=95, verbose=False):
    """
    Vertex-wise linear mixed-effects estimation with spatial covariance: covariance components are
    pooled and residual spatial correlation is modeled over homogeneous surface regions found by a
    region-growing segmentation, rather than fitting every vertex independently (see fit.py::fit).

    Parameters
    ----------
    X : ndarray
        Design matrix, ordered according to time for each subject (nm, p).
    Zcols : ndarray
        List of indices of the columns of X that will be considered as random effects (nr,).
    Y : ndarray
        Data matrix (nm, nv) whose columns represent the data vector for each vertex.
    clusters : list
        List of clusters (nm), as in fit.py::fit.
    maskvtx : ndarray
        Zero-based mask indices (nv,). If None, all vertices are included.
    coords : ndarray
        Anatomical surface vertex coordinates (nv, 3), used to compute within-region Euclidean distances.
    sphere_coords : ndarray
        Spherical surface vertex coordinates (nv, 3), used for the region-growing segmentation.
    faces : ndarray
        Zero-based triangle vertex indices of the surface mesh (nf, 3).
    prs : int
        Number of CPU cores used for parallel computing.
    nit : int
        Number of iterations.
    e : float
        Convergence epsilon (gradient's norm).
    nst : float
        Number of standard deviations of theta inside a region for it to be considered homogeneous. Default 2.
    prc : float
        Percent of vertices inside a region that are allowed to not meet the homogeneity criterion. Default 95.
    verbose : bool
        Increase verbosity.

    Returns
    -------
    stats : dict
        A dict of dictionaries containing estimates for every vertex where the model was fit, in the
        same format as fit.py::fit's return value.
    regions : ndarray
        Region label per vertex (nv,). Label 0 denotes vertices excluded from analysis.
    """

    # Create auxiliary variables

    ni = {}
    for item in clusters:
        ni[item] = ni.get(item, 0) + 1
    ni = np.array(list(ni.values()))

    nv = Y.shape[1]
    if maskvtx is None:
        maskvtx = np.arange(nv)

    # Initial estimates

    LOGGER.info("Computing initial values ...")
    Theta0_masked_init, Re0_masked_init = _fit_init_all(X, Zcols, Y[:, maskvtx], ni, prs)
    Theta0_init = np.zeros((Theta0_masked_init.shape[0], nv))
    Theta0_init[:, maskvtx] = Theta0_masked_init
    Re0_init = np.zeros((Re0_masked_init.shape[0], nv))
    Re0_init[:, maskvtx] = Re0_masked_init

    # EM estimates

    LOGGER.info("Starting EM algorithm ...")
    Theta0_masked_em, Re0_masked_em = _fit_em_all(X, Zcols, Y[:, maskvtx], ni, prs, Theta0_init[:, maskvtx], nit)
    Theta0_em = np.zeros((Theta0_masked_em.shape[0], nv))
    Theta0_em[:, maskvtx] = Theta0_masked_em
    Re0_em = np.zeros((Re0_masked_em.shape[0], nv))
    Re0_em[:, maskvtx] = Re0_masked_em

    # Create regions

    LOGGER.info("Starting region growing algorithm ...")
    regions, region_ids, _ = _create_regions(sphere_coords, faces, Theta0_em, Re0_em, maskvtx, nst=nst, prc=prc)

    # Model fitting

    LOGGER.info("Starting model fitting at each region ...")
    stats = _fit_region_all(X, Zcols, Y, ni, Theta0_em, regions, region_ids, coords, nit, e, prs, verbose)

    # Output

    converged = np.sum([stats[i]["st"] for i in stats])
    LOGGER.info(
        f"Summary: Algorithm did not converge at {100 * (1 - converged / len(stats)):.2f} "
        "percent of the total number of locations."
    )

    return stats, regions
