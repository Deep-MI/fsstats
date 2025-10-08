"""
This module provides helper functions.
"""

# ------------------------------------------------------------------------------
# _all_identical_elements_consecutive

def _all_identical_elements_consecutive(lst):
    """
    check if identical elements in a list are in consecutive blocks
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