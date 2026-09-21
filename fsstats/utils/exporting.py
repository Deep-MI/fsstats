"""
This module provides file import / export functions for the FastSurfer statistical
modeling and inference package.
"""

import logging

import nibabel as nb
import numpy as np

# ==============================================================================
# LOGGING

LOGGER = logging.getLogger(__name__)

def write_mgh(dat, filename):
    """
    Write image data to disk in mgh format.

    Parameters
    ----------
    dat : ndarray
        Image data to write.
    filename : str
        Output filename.

    Returns
    -------
    None
        The function does not return any specific objects.
    """
    LOGGER.info(f"Writing {filename}")
    if len(dat.shape)==4:
        if dat.shape[3]==1:
            dat = dat.squeeze(axis=3)
    nb.freesurfer.save(nb.MGHImage(dataobj=dat.astype(np.float32), affine=np.eye(4)), filename)
