"""
This module provides documentation for the fsstats package.

"""

# this is a backport to python 3.8; for newer versions (>=3.10) use built-in
# importlib.resources
from importlib_resources import files


def _get_help_text():
    txt = files("fsstats.doc").joinpath("DOCUMENTATION.md").read_text()

    return txt
