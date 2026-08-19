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
    Read mgh/mgz image files.

    Parameters
    ----------
    filename : str
        Filename of an mgh/mgz image file.

    Returns
    -------
    img : nibabel.freesurfer.mghformat.MGHImage
        The loaded image object.
    dat : ndarray
        The image data.
    """
    LOGGER.info(f"Reading {filename}")
    img = nb.freesurfer.load(filename)
    dat = img.get_fdata().squeeze().T
    LOGGER.info(f"Found data with shape {dat.shape}")
    return img, dat

def read_surface(filename):
    """
    Read a FreeSurfer surface file.

    Parameters
    ----------
    filename : str
        Filename of a FreeSurfer surface file (e.g., lh.white, lh.sphere).

    Returns
    -------
    coords : ndarray
        Vertex coordinates (nv, 3).
    faces : ndarray
        Zero-based triangle vertex indices (nf, 3).
    """
    LOGGER.info(f"Reading {filename}")
    coords, faces = nb.freesurfer.io.read_geometry(filename)
    LOGGER.info(f"Found surface with {coords.shape[0]} vertices and {faces.shape[0]} faces")
    return coords, faces

def read_label(filename):
    """
    Read FreeSurfer label files.

    Parameters
    ----------
    filename : str
        Filename of a FreeSurfer label file.

    Returns
    -------
    lbl : ndarray
        Array of vertex indices contained in the label.
    """
    LOGGER.info(f"Reading {filename}")
    lbl = nb.freesurfer.io.read_label(filename)
    LOGGER.info(f"Found label with shape {lbl.shape}")
    return lbl

def read_mask(filename):
    """
    Read a binary mask volume and return the indices of its nonzero voxels/vertices.

    Parameters
    ----------
    filename : str
        Filename of a binary mask volume (e.g., mgh/mgz).

    Returns
    -------
    msk : ndarray
        Array of zero-based voxel/vertex indices where the mask is nonzero.
    """
    LOGGER.info(f"Reading {filename}")
    _, dat = read_mgh(filename)
    msk = np.flatnonzero(dat)
    LOGGER.info(f"Found mask with {msk.shape[0]} nonzero voxels/vertices")
    return msk

def read_table(filename):
    """
    Read tabular data with a header and a first column of IDs.

    Parameters
    ----------
    filename : str
        Filename of a table, e.g. produced by asegstats2table or aparcstats2table.

    Returns
    -------
    tab : pandas.DataFrame
        The table contents.
    """
    LOGGER.info(f"Reading {filename}")
    tab = pd.read_table(filename, sep=r'\s+|,', engine='python')
    LOGGER.info(f"Found data with shape {tab.shape}")
    return tab

def read_design_matrix(filename):
    """
    Read a design matrix file without header and without column of IDs.

    Parameters
    ----------
    filename : str
        Filename of a design matrix.

    Returns
    -------
    mat : ndarray
        The design matrix.
    """
    LOGGER.info(f"Reading {filename}")
    mat = np.array(pd.read_table(filename, sep=r'\s+|,', header=None, engine='python'))
    LOGGER.info(f"Read design matrix with shape {mat.shape}")
    return mat

def read_contrasts(list_of_filenames):
    """
    Read one or more contrast files without header.

    Parameters
    ----------
    list_of_filenames : list of str
        Filenames of one or more contrast matrix/vector files.

    Returns
    -------
    contrasts : dict
        Dictionary mapping each contrast's filename (without extension) to its
        contrast matrix/vector as a numpy array.
    """
    contrasts = dict()
    for i in list_of_filenames:
        LOGGER.info(f"Reading {i}")
        con = np.array(pd.read_table(i, sep=r'\s+|,', header=None, engine='python'))
        LOGGER.info(f"Found contrasts file {i} with shape {con.shape}")
        contrasts[os.path.splitext(os.path.basename(i))[0]] = con
    return contrasts

def read_clusters(filename):
    """
    Read a cluster file without header.

    Parameters
    ----------
    filename : str
        Filename of a list of clusters (e.g., case IDs).

    Returns
    -------
    clust : list
        The list of cluster identifiers.
    """
    LOGGER.info(f"Reading {filename}")
    clust = pd.read_table(filename, header=None)[0].tolist()
    LOGGER.info(f"Found clusters file with length {len(clust)}")
    return clust
