#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

import numpy as np

import scipy.ndimage as spnd

import dask.array as da
import dask.array.utils as dau

import dask_ndmeasure
import dask_ndmeasure._test_utils


@pytest.mark.parametrize(
    "funcname", [
        "center_of_mass",
        "mean",
        "standard_deviation",
        "sum",
        "variance",
    ]
)
def test_measure_props_err(funcname):
    da_func = getattr(dask_ndmeasure, funcname)

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
        da_func(d, lbls, ind)


@pytest.mark.parametrize(
    "funcname", [
        "center_of_mass",
        "mean",
        "standard_deviation",
        "sum",
        "variance",
    ]
)
@pytest.mark.parametrize(
    "shape, chunks, has_lbls, ind", [
        ((15, 16), (4, 5), False, None),
        ((15, 16), (4, 5), True, None),
        ((15, 16), (4, 5), True, 0),
        ((15, 16), (4, 5), True, 1),
        ((15, 16), (4, 5), True, [1]),
        ((15, 16), (4, 5), True, [1, 2]),
        ((15, 16), (4, 5), True, [1, 100]),
        ((15, 16), (4, 5), True, [[1, 2, 3, 4]]),
        ((15, 16), (4, 5), True, [[1, 2], [3, 4]]),
        ((15, 16), (4, 5), True, [[[1], [2], [3], [4]]]),
    ]
)
def test_measure_props(funcname, shape, chunks, has_lbls, ind):
    sp_func = getattr(spnd, funcname)
    da_func = getattr(dask_ndmeasure, funcname)

    a = np.random.random(shape)
    d = da.from_array(a, chunks=chunks)

    lbls = None
    d_lbls = None

    if has_lbls:
        lbls = np.zeros(a.shape, dtype=np.int64)
        lbls += (
            (d < 0.5).astype(lbls.dtype) +
            (d < 0.25).astype(lbls.dtype) +
            (d < 0.125).astype(lbls.dtype) +
            (d < 0.0625).astype(lbls.dtype)
        )
        d_lbls = da.from_array(lbls, chunks=d.chunks)

    a_r = np.array(sp_func(a, lbls, ind))
    d_r = da_func(d, d_lbls, ind)

    dask_ndmeasure._test_utils._assert_eq_nan(a_r, d_r)
