# -*- coding: utf-8 -*-

from __future__ import division


__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import operator

import numpy
import dask.array

from . import _compat
from . import _utils


def center_of_mass(input, labels=None, index=None):
    """
    Calculate the center of mass of the values of an array at labels.

    Parameters
    ----------
    input : ndarray
        Data from which to calculate center-of-mass. The masses can either
        be positive or negative.
    labels : ndarray, optional
        Labels for objects in `input`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `input`.
    index : int or sequence of ints, optional
        Labels for which to calculate centers-of-mass. If not specified,
        all labels greater than zero are used.  Only used with `labels`.

    Returns
    -------
    center_of_mass : tuple, or list of tuples
        Coordinates of centers-of-mass.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    lbl_mtch, input_i_mtch, input_mtch = _utils._get_label_matches(
        input, labels, index
    )

    input_i_mtch_wt = (
        input_mtch[index.ndim * (slice(None),) + (None,)] *
        input_i_mtch
    )

    input_i_mtch_wt = input_i_mtch_wt.astype(numpy.float64)
    input_mtch = input_mtch.astype(numpy.float64)

    com_lbl = input_i_mtch_wt.sum(
        axis=tuple(range(1 + index.ndim, input_i_mtch_wt.ndim))
    )
    input_mtch_sum = input_mtch.sum(
        axis=tuple(range(index.ndim, input_mtch.ndim))
    )
    com_lbl /= input_mtch_sum[index.ndim * (slice(None),) + (None,)]

    return com_lbl
