"""
This module provides file import / export functions for the FastSurfer statistical
modeling and inference package.
"""

import logging
import os

import nibabel as nb
import numpy as np
import pandas as pd

# ==============================================================================
# LOGGING

LOGGER = logging.getLogger(__name__)

def read_mgh(filename):
    """
    Reads mgh files.

    expects mgh/mgz image file
    returns numpy array
    """
    LOGGER.info("Reading " + filename)
    img = nb.freesurfer.load(filename)
    dat = img.get_fdata().squeeze().T
    LOGGER.info("Found data with shape " + str(dat.shape))
    return img, dat

def read_label(filename):
    """
    Reads FreeSufer label files.

    expects label file
    returns numpy array
    """
    LOGGER.info("Reading " + filename)
    lbl = nb.freesurfer.io.read_label(filename)
    LOGGER.info("Found label with shape " + str(lbl.shape))
    return lbl

def read_table(filename):
    """
    Reads tabular data.

    expects table with header and first column of IDs
    returns pandas df
    """
    LOGGER.info("Reading " + filename)
    tab = pd.read_table(filename, sep=r'\s+|,', engine='python')
    LOGGER.info("Found data with shape " + str(tab.shape))
    return tab

def read_design_matrix(filename):
    """
    Reads design matrix file.

    expects table without header and without column of IDs
    returns numpy array
    """
    LOGGER.info("Reading " + filename)
    mat = np.array(pd.read_table(filename, sep=r'\s+|,', header=None, engine='python'))
    LOGGER.info("Read design matrix with shape " + str(mat.shape))
    return mat

def read_contrasts(list_of_filenames):
    """
    Reads one or more contrast files.

    expects table without header
    returns numpy array
    """
    contrasts = dict()
    for i in list_of_filenames:
        LOGGER.info("Reading " + i)
        con = np.array(pd.read_table(i, sep=r'\s+|,', header=None, engine='python'))
        LOGGER.info("Found contrasts file " + i + " with shape " + str(con.shape))
        contrasts[os.path.splitext(os.path.basename(i))[0]] = con
    return contrasts

def read_clusters(filename):
    """
    Reads cluster file.

    expects table without header
    returns pandas df
    """
    LOGGER.info("Reading " + filename)
    clust = pd.read_table(filename, header=None)
    LOGGER.info("Found clusters file with shape " + str(clust.shape))
    return clust