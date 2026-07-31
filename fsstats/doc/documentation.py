"""
This module provides documentation for the fsstats package.

"""

from importlib.resources import files


def _get_help_text():
    txt = files("fsstats.doc").joinpath("DOCUMENTATION.md").read_text()

    return txt
