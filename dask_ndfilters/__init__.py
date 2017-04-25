# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from dask_ndfilters.core import *
