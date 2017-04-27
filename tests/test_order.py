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
    "da_func",
    [
        da_ndf.minimum_filter,
        da_ndf.median_filter,
        da_ndf.maximum_filter,
    ]
)
@pytest.mark.parametrize(
    "err_type, size, footprint, origin",
    [
        (RuntimeError, None, None, 0),
        (RuntimeError, 1, np.ones((1,)), 0),
        (RuntimeError, None, np.ones((1,)), 0),
        (RuntimeError, None, np.ones((1, 0)), 0),
        (RuntimeError, 1, None, (0,)),
        (RuntimeError, 1, None, [(0,)]),
        (ValueError, 1, None, 1),
    ]
)
def test_order_filter_params(da_func, err_type, size, footprint, origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_func(d, size=size, footprint=footprint, origin=origin)


@pytest.mark.parametrize(
    "sp_func, da_func",
    [
        (sp_ndf.minimum_filter, da_ndf.minimum_filter),
        (sp_ndf.median_filter, da_ndf.median_filter),
        (sp_ndf.maximum_filter, da_ndf.maximum_filter),
    ]
)
@pytest.mark.parametrize(
    "size, footprint",
    [
        (1, None),
        ((1, 1), None),
        (None, np.ones((1, 1))),
    ]
)
def test_ordered_filter_identity(sp_func, da_func, size, footprint):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(d, da_func(d, size=size, footprint=footprint))

    dau.assert_eq(
        sp_func(a, size=size, footprint=footprint),
        da_func(d, size=size, footprint=footprint)
    )


@pytest.mark.parametrize(
    "sp_func, da_func",
    [
        (sp_ndf.minimum_filter, da_ndf.minimum_filter),
        (sp_ndf.median_filter, da_ndf.median_filter),
        (sp_ndf.maximum_filter, da_ndf.maximum_filter),
    ]
)
@pytest.mark.parametrize(
    "size, footprint, origin",
    [
        (2, None, 0),
        (None, np.ones((2, 3)), 0),
        (None, np.ones((2, 3)), (0, 1)),
        (None, np.ones((2, 3)), (0, -1)),
        (None, (np.mgrid[-2: 2+1, -2: 2+1]**2).sum(axis=0) < 2.5**2, 0),
        (None, (np.mgrid[-2: 2+1, -2: 2+1]**2).sum(axis=0) < 2.5**2, (1, 2)),
        (None, (np.mgrid[-2: 2+1, -2: 2+1]**2).sum(axis=0) < 2.5**2, (-1, -2)),
        (5, None, 0),
        (7, None, 0),
        (8, None, 0),
        (10, None, 0),
        (5, None, 2),
        (5, None, -2),
    ]
)
def test_ordered_filter_compare(sp_func, da_func, size, footprint, origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        sp_func(a, size=size, footprint=footprint, origin=origin),
        da_func(d, size=size, footprint=footprint, origin=origin)
    )


@pytest.mark.parametrize(
    "size, footprint, origin",
    [
        (5, None, 2.6),
        (5, None, 2.4),
        (5, None, -1.6),
        (5, None, -1.4),
    ]
)
def test_median_float_origin_compare(size, footprint, origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        sp_ndf.median_filter(a, size=size, footprint=footprint, origin=origin),
        da_ndf.median_filter(d, size=size, footprint=footprint, origin=origin)
    )
