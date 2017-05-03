#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

import numpy as np
import scipy.ndimage.filters as sp_ndf

import dask.array as da
import dask.array.utils as dau

import dask_ndfilters as da_ndf


@pytest.mark.parametrize(
    "err_type, axis",
    [
        (ValueError, 0.0),
        (ValueError, 2),
        (ValueError, -3),
    ]
)
def test_prewitt_params(err_type, axis):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_ndf.prewitt(d, axis)


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
def test_prewitt_compare(axis):
    s = (10, 11, 12)
    a = np.arange(float(np.prod(s))).reshape(s)
    d = da.from_array(a, chunks=(5, 5, 6))

    dau.assert_eq(
        sp_ndf.prewitt(a, axis),
        da_ndf.prewitt(d, axis)
    )
