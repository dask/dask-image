# -*- coding: utf-8 -*-


__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


import itertools
import numbers
import warnings

import dask
import dask.array as da
import dask.delayed
import numpy as np
import pims

from ._utils import imread
