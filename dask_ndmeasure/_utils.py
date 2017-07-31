# -*- coding: utf-8 -*-


import operator

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

    if input.shape != labels.shape:
        raise ValueError("The input and labels arrays must be the same shape.")

    return (input, labels, index)


def _get_label_matches(labels, index):
    lbl_mtch = operator.eq(
        index[(Ellipsis,) + labels.ndim * (None,)],
        labels[index.ndim * (None,)]
    )

    return lbl_mtch
