"""
This module provides helper functions.
"""

from scipy.stats import norm

# ------------------------------------------------------------------------------
# _signed_sig_and_z

def _signed_sig_and_z(sig, sgn, ncontrastrows):
    """
    Fold contrast sign into a sig-value and compute the corresponding z-value.

    Sign is only meaningful for single-row (t-test-like) contrasts; for multi-row
    (F-test) contrasts, sig and z are left unsigned. This matches mri_glmfit's
    convention, which only applies a sign to single-row contrasts.

    Parameters
    ----------
    sig : float or ndarray
        Unsigned sig-value(s), where sig = -log10(p).
    sgn : float or ndarray
        Sign of the contrast, sign(C @ Bhat). Ignored if ncontrastrows != 1.
    ncontrastrows : int
        Number of rows of the contrast matrix/vector.

    Returns
    -------
    signed_sig : float or ndarray
        Sig-value, with sign folded in for single-row contrasts.
    z : float or ndarray
        Z-value corresponding to the (two-sided) p-value, signed for single-row contrasts.
    """
    p = 10.0 ** (-sig)
    z = norm.isf(p / 2)
    if ncontrastrows == 1:
        return sig * sgn, z * sgn
    return sig, z

# ------------------------------------------------------------------------------
# _all_identical_elements_consecutive

def _all_identical_elements_consecutive(lst):
    """
    Check if identical elements in a list are in consecutive blocks.

    Parameters
    ----------
    lst : list
        List to check.

    Returns
    -------
    bool
        True if identical elements occur only in consecutive blocks, False otherwise.
    """
    seen = set()
    prev = None
    for item in lst:
        if item != prev:
            if item in seen:
                return False
            seen.add(item)
        prev = item
    return True
