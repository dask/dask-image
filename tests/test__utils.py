#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import operator

import pytest

import numpy as np

import dask.array as da
import dask.array.utils as dau

import dask_ndmeasure._utils


def test__norm_input_labels_index_err():
    shape = (15, 16)
    chunks = (4, 5)
    ind = None

    a = np.random.random(shape)
    d = da.from_array(a, chunks=chunks)

    lbls = (a < 0.5).astype(np.int64)
    d_lbls = da.from_array(lbls, chunks=d.chunks)

    lbls = lbls[:-1]
    d_lbls = d_lbls[:-1]

    with pytest.raises(ValueError):
        dask_ndmeasure._utils._norm_input_labels_index(d, d_lbls, ind)


def test__norm_input_labels_index():
    shape = (15, 16)
    chunks = (4, 5)
    ind = None

    a = np.random.random(shape)
    d = da.from_array(a, chunks=chunks)

    lbls = (a < 0.5).astype(np.int64)
    d_lbls = da.from_array(lbls, chunks=d.chunks)

    d_n, d_lbls_n, ind_n = dask_ndmeasure._utils._norm_input_labels_index(
        d, d_lbls, ind
    )

    assert isinstance(d_n, da.Array)
    assert isinstance(d_lbls_n, da.Array)
    assert isinstance(ind_n, da.Array)

    dau.assert_eq(d_n, d)
    dau.assert_eq(d_lbls_n, d_lbls)
    dau.assert_eq(ind_n, np.array([1], dtype=np.int64))


@pytest.mark.parametrize(
    "shape, chunks, ind", [
        ((15, 16), (4, 5), 0),
        ((15, 16), (4, 5), 1),
        ((15, 16), (4, 5), [1]),
        ((15, 16), (4, 5), [1, 2]),
        ((15, 16), (4, 5), [1, 100]),
        ((15, 16), (4, 5), [[1, 2, 3, 4]]),
        ((15, 16), (4, 5), [[1, 2], [3, 4]]),
        ((15, 16), (4, 5), [[[1], [2], [3], [4]]]),
    ]
)
def test__get_label_matches(shape, chunks, ind):
    a = np.random.random(shape)
    d = da.from_array(a, chunks=chunks)

    lbls = np.zeros(a.shape, dtype=np.int64)
    lbls += (
        (a < 0.5).astype(lbls.dtype) +
        (a < 0.25).astype(lbls.dtype) +
        (a < 0.125).astype(lbls.dtype) +
        (a < 0.0625).astype(lbls.dtype)
    )
    d_lbls = da.from_array(lbls, chunks=d.chunks)

    ind = np.array(ind)
    d_ind = da.from_array(ind, chunks=1)

    lbl_mtch = operator.eq(
        ind[(Ellipsis,) + lbls.ndim * (None,)],
        lbls[ind.ndim * (None,)]
    )
    input_i_mtch = (
        lbl_mtch.astype(np.int64)[ind.ndim * (slice(None),) + (None,)] *
        np.indices(a.shape, dtype=np.int64)[ind.ndim * (None,)]
    )
    input_mtch = lbl_mtch.astype(a.dtype) * a[ind.ndim * (None,)]

    d_lbl_mtch, d_input_i_mtch, d_input_mtch = dask_ndmeasure._utils._get_label_matches(
        d, d_lbls, ind
    )

    assert issubclass(d_lbl_mtch.dtype.type, np.bool8)
    assert issubclass(d_input_i_mtch.dtype.type, np.int64)
    assert issubclass(d_input_mtch.dtype.type, a.dtype.type)

    dau.assert_eq(d_lbl_mtch, lbl_mtch)
    dau.assert_eq(d_input_i_mtch, input_i_mtch)
    dau.assert_eq(d_input_mtch, input_mtch)
