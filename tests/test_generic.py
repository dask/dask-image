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
        da_ndf.generic_filter,
    ]
)
@pytest.mark.parametrize(
    "err_type, function, size, footprint, origin",
    [
        (RuntimeError, lambda x: x, None, None, 0),
        (TypeError, lambda x: x, 1.0, None, 0),
        (RuntimeError, lambda x: x, (1,), None, 0),
        (RuntimeError, lambda x: x, [(1,)], None, 0),
        (RuntimeError, lambda x: x, 1, np.ones((1,)), 0),
        (RuntimeError, lambda x: x, None, np.ones((1,)), 0),
        (RuntimeError, lambda x: x, None, np.ones((1, 0)), 0),
        (RuntimeError, lambda x: x, 1, None, (0,)),
        (RuntimeError, lambda x: x, 1, None, [(0,)]),
        (ValueError, lambda x: x, 1, None, 1),
        (TypeError, lambda x: x, 1, None, 0.0),
        (TypeError, lambda x: x, 1, None, (0.0, 0.0)),
        (TypeError, lambda x: x, 1, None, 1+0j),
        (TypeError, lambda x: x, 1, None, (0+0j, 1+0j)),
    ]
)
def test_generic_filters_params(da_func,
                                err_type,
                                function,
                                size,
                                footprint,
                                origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_func(d,
                function,
                size=size,
                footprint=footprint,
                origin=origin)


@pytest.mark.parametrize(
    "sp_func, da_func",
    [
        (sp_ndf.generic_filter, da_ndf.generic_filter),
    ]
)
@pytest.mark.parametrize(
    "function, size, footprint",
    [
        (lambda x: x, 1, None),
        (lambda x: x, (1, 1), None),
        (lambda x: x, None, np.ones((1, 1))),
    ]
)
def test_generic_filter_identity(sp_func,
                                 da_func,
                                 function,
                                 size,
                                 footprint):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        d, da_func(d, function, size=size, footprint=footprint)
    )

    dau.assert_eq(
        sp_func(a, function, size=size, footprint=footprint),
        da_func(d, function, size=size, footprint=footprint),
    )


@pytest.mark.parametrize(
    "sp_func, da_func",
    [
        (sp_ndf.generic_filter, da_ndf.generic_filter),
    ]
)
@pytest.mark.parametrize(
    "function, size, footprint, origin",
    [
        (
            lambda x: (np.array(x)**2).sum(),
            2,
            None,
            0
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            None,
            np.ones((2, 3)),
            0
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            None,
            np.ones((2, 3)),
            (0, 1)
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            None,
            np.ones((2, 3)),
            (0, -1)
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            None,
            (np.mgrid[-2: 2+1, -2: 2+1]**2).sum(axis=0) < 2.5**2,
            0
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            None,
            (np.mgrid[-2: 2+1, -2: 2+1]**2).sum(axis=0) < 2.5**2,
            (1, 2)
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            None,
            (np.mgrid[-2: 2+1, -2: 2+1]**2).sum(axis=0) < 2.5**2,
            (-1, -2)
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            5,
            None,
            0
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            7,
            None,
            0
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            8,
            None,
            0
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            10,
            None,
            0
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            5,
            None,
            2
        ),
        (
            lambda x: (np.array(x)**2).sum(),
            5,
            None,
            -2
        ),
    ]
)
def test_generic_filter_compare(sp_func,
                                da_func,
                                function,
                                size,
                                footprint,
                                origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        sp_func(
            a, function, size=size, footprint=footprint, origin=origin
        ),
        da_func(
            d, function, size=size, footprint=footprint, origin=origin
        )
    )
