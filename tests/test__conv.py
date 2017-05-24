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
        (da_ndf.convolve),
        (da_ndf.correlate),
    ]
)
@pytest.mark.parametrize(
    "err_type, weights, origin",
    [
        (ValueError, np.ones((1,)), 0),
        (ValueError, np.ones((1, 0)), 0),
        (RuntimeError, np.ones((1, 1)), (0,)),
        (RuntimeError, np.ones((1, 1)), [(0,)]),
        (ValueError, np.ones((1, 1)), 1),
        (TypeError, np.ones((1, 1)), 0.0),
        (TypeError, np.ones((1, 1)), (0.0, 0.0)),
        (TypeError, np.ones((1, 1)), 1+0j),
        (TypeError, np.ones((1, 1)), (0+0j, 1+0j)),
    ]
)
def test_convolutions_params(da_func,
                             err_type,
                             weights,
                             origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_func(d,
                weights,
                origin=origin)


@pytest.mark.parametrize(
    "da_func",
    [
        da_ndf.convolve,
        da_ndf.correlate,
    ]
)
def test_convolutions_shape_type(da_func):
    weights = np.ones((1, 1))

    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    assert all([(type(s) is int) for s in d.shape])

    d2 = da_func(d, weights)

    assert all([(type(s) is int) for s in d2.shape])


@pytest.mark.parametrize(
    "da_func",
    [
        da_ndf.convolve,
        da_ndf.correlate,
    ]
)
def test_convolutions_comprehensions(da_func):
    np.random.seed(0)

    a = np.random.random((3, 12, 14))
    d = da.from_array(a, chunks=(3, 6, 7))

    weights = np.ones((1, 1))

    l2s = [da_func(d[i], weights) for i in range(len(d))]
    l2c = [da_func(d[i], weights)[None] for i in range(len(d))]

    dau.assert_eq(np.stack(l2s), da.stack(l2s))
    dau.assert_eq(np.concatenate(l2c), da.concatenate(l2c))


@pytest.mark.parametrize(
    "sp_func, da_func",
    [
        (sp_ndf.convolve, da_ndf.convolve),
        (sp_ndf.correlate, da_ndf.correlate),
    ]
)
@pytest.mark.parametrize(
    "weights",
    [
        np.ones((1, 1)),
    ]
)
def test_convolutions_identity(sp_func,
                               da_func,
                               weights):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        d, da_func(d, weights)
    )

    dau.assert_eq(
        sp_func(a, weights),
        da_func(d, weights)
    )


@pytest.mark.parametrize(
    "sp_func, da_func",
    [
        (sp_ndf.convolve, da_ndf.convolve),
        (sp_ndf.correlate, da_ndf.correlate),
    ]
)
@pytest.mark.parametrize(
    "weights, origin",
    [
        (np.ones((2, 2)), 0),
        (np.ones((2, 3)), 0),
        (np.ones((2, 3)), (0, 1)),
        (np.ones((2, 3)), (0, -1)),
        ((np.mgrid[-2: 2+1, -2: 2+1]**2).sum(axis=0) < 2.5**2, 0),
        ((np.mgrid[-2: 2+1, -2: 2+1]**2).sum(axis=0) < 2.5**2, (1, 2)),
        ((np.mgrid[-2: 2+1, -2: 2+1]**2).sum(axis=0) < 2.5**2, (-1, -2)),
        (np.ones((5, 5)), 0),
        (np.ones((7, 7)), 0),
        (np.ones((8, 8)), 0),
        (np.ones((10, 10)), 0),
        (np.ones((5, 5)), 2),
        (np.ones((5, 5)), -2),
    ]
)
def test_convolutions_compare(sp_func,
                              da_func,
                              weights,
                              origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        sp_func(
            a, weights, origin=origin
        ),
        da_func(
            d, weights, origin=origin
        )
    )
