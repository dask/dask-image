#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

import numpy as np

import dask.array as da
import dask.array.utils as dau

import dask_image.ndmeasure._utils


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
        dask_image.ndmeasure._utils._norm_input_labels_index(d, d_lbls, ind)


def test__norm_input_labels_index():
    shape = (15, 16)
    chunks = (4, 5)
    ind = None

    a = np.random.random(shape)
    d = da.from_array(a, chunks=chunks)

    lbls = (a < 0.5).astype(int)
    d_lbls = da.from_array(lbls, chunks=d.chunks)

    d_n, d_lbls_n, ind_n = dask_image.ndmeasure._utils._norm_input_labels_index(
        d, d_lbls, ind
    )

    assert isinstance(d_n, da.Array)
    assert isinstance(d_lbls_n, da.Array)
    assert isinstance(ind_n, da.Array)

    assert d_n.shape == d.shape
    assert d_lbls_n.shape == d_lbls.shape
    assert ind_n.shape == ()

    dau.assert_eq(d_n, d)
    dau.assert_eq(d_lbls_n, d_lbls)
    dau.assert_eq(ind_n, np.array(1, dtype=int))


@pytest.mark.parametrize(
    "shape, chunks, ind", [
        ((15, 16), (4, 5), 1),
        ((15, 16), (4, 5), [1]),
        ((15, 16), (4, 5), [[1, 2, 3, 4]]),
        ((15, 16), (4, 5), [[1, 2], [3, 4]]),
        ((15, 16), (4, 5), [[[1], [2], [3], [4]]]),
    ]
)
def test__norm_input_labels_index_warn(shape, chunks, ind):
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

    with pytest.warns(None) as w:
        dask_image.ndmeasure._utils._norm_input_labels_index(
            d, d_lbls, d_ind
        )

    if ind.ndim > 1:
        assert len(w) == 1
        w.pop(FutureWarning)
    else:
        assert len(w) == 0


@pytest.mark.parametrize(
    "shape, chunks", [
        ((15,), (4,)),
        ((15, 16), (4, 5)),
        ((15, 1, 16), (4, 1, 5)),
        ((15, 12, 16), (4, 5, 6)),
    ]
)
def test___ravel_shape_indices(shape, chunks):
    a = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    d = dask_image.ndmeasure._utils._ravel_shape_indices(
        shape, dtype=np.int64, chunks=chunks
    )

    dau.assert_eq(d, a)
