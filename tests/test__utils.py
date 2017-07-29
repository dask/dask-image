#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

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
