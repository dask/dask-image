#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

import numpy as np
import scipy.ndimage.filters as sp_ndf

import dask
import dask.array as da
import dask.array.utils as dau

import dask_image.ndfilters as da_ndf


assert dask


@pytest.mark.parametrize(
    "err_type, axis",
    [
        (ValueError, 0.0),
        (ValueError, 2),
        (ValueError, -3),
    ]
)
@pytest.mark.parametrize(
    "da_func",
    [
        da_ndf.prewitt,
        da_ndf.sobel,
    ]
)
def test_edge_func_params(da_func, err_type, axis):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_func(d, axis)


@pytest.mark.parametrize(
    "da_func",
    [
        da_ndf.prewitt,
        da_ndf.sobel,
    ]
)
def test_edge_comprehensions(da_func):
    np.random.seed(0)

    a = np.random.random((3, 12, 14))
    d = da.from_array(a, chunks=(3, 6, 7))

    l2s = [da_func(d[i]) for i in range(len(d))]
    l2c = [da_func(d[i])[None] for i in range(len(d))]

    dau.assert_eq(np.stack(l2s), da.stack(l2s))
    dau.assert_eq(np.concatenate(l2c), da.concatenate(l2c))


@pytest.mark.parametrize(
    "axis",
    [
        0,
        1,
        2,
        -1,
        -2,
        -3,
    ]
)
@pytest.mark.parametrize(
    "da_func, sp_func",
    [
        (da_ndf.prewitt, sp_ndf.prewitt),
        (da_ndf.sobel, sp_ndf.sobel),
    ]
)
def test_edge_func_compare(da_func, sp_func, axis):
    s = (10, 11, 12)
    a = np.arange(float(np.prod(s))).reshape(s)
    d = da.from_array(a, chunks=(5, 5, 6))

    dau.assert_eq(
        sp_func(a, axis),
        da_func(d, axis)
    )
