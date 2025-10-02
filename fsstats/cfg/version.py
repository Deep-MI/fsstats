"""
This module provides version information for the FastSurfer statistical
modeling and inference package.
"""

# ------------------------------------------------------------------------------
# get_version()


def get_version():
    """
    Get version from package metadata.
    """
    from importlib import metadata

    try:
        # requires existing installation
        version = metadata.version("fsstats")
    except Exception:
        # fall-back if package is not installed, but run directly
        version = "unknown"

    return version
