# -*- coding: utf-8 -*-


import numpy

import dask.array

from . import _compat


def _norm_input_labels_index(input, labels=None, index=None):
    """
    Normalize arguments to a standard form.
    """

    input = _compat._asarray(input)

    if labels is None:
        labels = (input != 0).astype(numpy.int64)
        index = None

    if index is None:
        labels = (labels > 0).astype(numpy.int64)
        index = dask.array.ones(tuple(), dtype=numpy.int64, chunks=tuple())

    labels = _compat._asarray(labels)
    index = _compat._asarray(index)

    # SciPy transposes these for some reason.
    # So we do the same thing here.
    # This only matters if index is some array.
    index = index.T

    if input.shape != labels.shape:
        raise ValueError("The input and labels arrays must be the same shape.")

    return (input, labels, index)
