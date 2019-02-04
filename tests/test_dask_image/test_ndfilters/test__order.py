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
    "da_func, extra_kwargs",
    [
        (da_ndf.minimum_filter, {}),
        (da_ndf.median_filter, {}),
        (da_ndf.maximum_filter, {}),
        (da_ndf.rank_filter, {"rank": 0}),
        (da_ndf.percentile_filter, {"percentile": 0}),
    ]
)
@pytest.mark.parametrize(
    "err_type, size, footprint, origin",
    [
        (RuntimeError, None, None, 0),
        (TypeError, 1.0, None, 0),
        (RuntimeError, (1,), None, 0),
        (RuntimeError, [(1,)], None, 0),
        (RuntimeError, 1, np.ones((1,)), 0),
        (RuntimeError, None, np.ones((1,)), 0),
        (RuntimeError, None, np.ones((1, 0)), 0),
        (RuntimeError, 1, None, (0,)),
        (RuntimeError, 1, None, [(0,)]),
        (ValueError, 1, None, 1),
        (TypeError, 1, None, 0.0),
        (TypeError, 1, None, (0.0, 0.0)),
        (TypeError, 1, None, 1+0j),
        (TypeError, 1, None, (0+0j, 1+0j)),
    ]
)
def test_order_filter_params(da_func,
                             extra_kwargs,
                             err_type,
                             size,
                             footprint,
                             origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_func(d,
                size=size,
                footprint=footprint,
                origin=origin,
                **extra_kwargs)


@pytest.mark.parametrize(
    "da_func, extra_kwargs",
    [
        (da_ndf.minimum_filter, {}),
        (da_ndf.median_filter, {}),
        (da_ndf.maximum_filter, {}),
        (da_ndf.rank_filter, {"rank": 0}),
        (da_ndf.percentile_filter, {"percentile": 0}),
    ]
)
def test_ordered_filter_shape_type(da_func,
                                   extra_kwargs):
    size = 1

    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    assert all([(type(s) is int) for s in d.shape])

    d2 = da_func(d, size=size, **extra_kwargs)

    assert all([(type(s) is int) for s in d2.shape])


@pytest.mark.parametrize(
    "sp_func, da_func, extra_kwargs",
    [
        (sp_ndf.minimum_filter,
         da_ndf.minimum_filter, {}),
        (sp_ndf.median_filter,
         da_ndf.median_filter, {}),
        (sp_ndf.maximum_filter,
         da_ndf.maximum_filter, {}),
        (sp_ndf.rank_filter,
         da_ndf.rank_filter, {"rank": 0}),
        (sp_ndf.percentile_filter,
         da_ndf.percentile_filter, {"percentile": 0}),
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
def test_ordered_filter_identity(sp_func,
                                 da_func,
                                 extra_kwargs,
                                 size,
                                 footprint):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        d, da_func(d, size=size, footprint=footprint, **extra_kwargs)
    )

    dau.assert_eq(
        sp_func(a, size=size, footprint=footprint, **extra_kwargs),
        da_func(d, size=size, footprint=footprint, **extra_kwargs)
    )


@pytest.mark.parametrize(
    "da_func, kwargs",
    [
        (da_ndf.minimum_filter, {"size": 1}),
        (da_ndf.median_filter, {"size": 1}),
        (da_ndf.maximum_filter, {"size": 1}),
        (da_ndf.rank_filter, {"size": 1, "rank": 0}),
        (da_ndf.percentile_filter, {"size": 1, "percentile": 0}),
    ]
)
def test_order_comprehensions(da_func, kwargs):
    np.random.seed(0)

    a = np.random.random((3, 12, 14))
    d = da.from_array(a, chunks=(3, 6, 7))

    l2s = [da_func(d[i], **kwargs) for i in range(len(d))]
    l2c = [da_func(d[i], **kwargs)[None] for i in range(len(d))]

    dau.assert_eq(np.stack(l2s), da.stack(l2s))
    dau.assert_eq(np.concatenate(l2c), da.concatenate(l2c))


@pytest.mark.parametrize(
    "sp_func, da_func, extra_kwargs",
    [
        (sp_ndf.minimum_filter,
         da_ndf.minimum_filter, {}),
        (sp_ndf.median_filter,
         da_ndf.median_filter, {}),
        (sp_ndf.maximum_filter,
         da_ndf.maximum_filter, {}),
        (sp_ndf.rank_filter,
         da_ndf.rank_filter, {"rank": 1}),
        (sp_ndf.percentile_filter,
         da_ndf.percentile_filter, {"percentile": 10}),
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
def test_ordered_filter_compare(sp_func,
                                da_func,
                                extra_kwargs,
                                size,
                                footprint,
                                origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        sp_func(
            a, size=size, footprint=footprint, origin=origin, **extra_kwargs
        ),
        da_func(
            d, size=size, footprint=footprint, origin=origin, **extra_kwargs
        )
    )
