"""
This module provides helper functions.
"""

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
