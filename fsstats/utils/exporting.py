import logging
import os
import nibabel as nb
import numpy as np
import pandas as pd

# ==============================================================================
# LOGGING

LOGGER = logging.getLogger(__name__)

def write_mgh(dat, filename):
    """
    expects numpy array
    """
    LOGGER.info("Writing " + filename)
    nb.freesurfer.save(nb.MGHImage(dataobj=dat.astype(np.float32), affine=np.eye(4)), filename)
