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
    Writes image data to disk.

    Expects numpy array.
    """
    LOGGER.info("Writing " + filename)
    nb.freesurfer.save(nb.MGHImage(dataobj=dat.astype(np.float32), affine=np.eye(4)), filename)
