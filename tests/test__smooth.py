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
    "err_type, size, origin",
    [
        (TypeError, 3.0, 0),
        (TypeError, 3, 0.0),
        (RuntimeError, [3], 0),
        (RuntimeError, 3, [0]),
        (RuntimeError, [[3]], 0),
        (RuntimeError, 3, [[0]]),
    ]
)
def test_uniform_filter_params(err_type, size, origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_ndf.uniform_filter(d, size, origin=origin)


@pytest.mark.parametrize(
    "size, origin",
    [
        (1, 0),
    ]
)
def test_uniform_identity(size, origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        d, da_ndf.uniform_filter(d, size, origin=origin)
    )

    dau.assert_eq(
        sp_ndf.uniform_filter(a, size, origin=origin),
        da_ndf.uniform_filter(d, size, origin=origin)
    )


@pytest.mark.parametrize(
    "size, origin",
    [
        (2, 0),
        (3, 0),
        (3, 1),
        (3, (1, 0)),
        ((1, 2), 0),
        ((3, 2), (1, 0)),
    ]
)
def test_uniform_compare(size, origin):
    s = (100, 110)
    a = np.arange(float(np.prod(s))).reshape(s)
    d = da.from_array(a, chunks=(50, 55))

    dau.assert_eq(
        sp_ndf.uniform_filter(a, size, origin=origin),
        da_ndf.uniform_filter(d, size, origin=origin)
    )
