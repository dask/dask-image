#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import scipy.ndimage

import dask.array as da

import dask_image.ndfilters


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
        dask_image.ndfilters.uniform_filter(d, size, origin=origin)


def test_uniform_shape_type():
    size = 1
    origin = 0

    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    assert all([(type(s) is int) for s in d.shape])

    d2 = dask_image.ndfilters.uniform_filter(d, size, origin=origin)

    assert all([(type(s) is int) for s in d2.shape])


def test_uniform_comprehensions():
    da_func = lambda arr: dask_image.ndfilters.uniform_filter(arr, 1, origin=0)  # noqa: E731, E501

    np.random.seed(0)

    a = np.random.random((3, 12, 14))
    d = da.from_array(a, chunks=(3, 6, 7))

    l2s = [da_func(d[i]) for i in range(len(d))]
    l2c = [da_func(d[i])[None] for i in range(len(d))]

    da.utils.assert_eq(np.stack(l2s), da.stack(l2s))
    da.utils.assert_eq(np.concatenate(l2c), da.concatenate(l2c))


@pytest.mark.parametrize(
    "size, origin",
    [
        (1, 0),
    ]
)
def test_uniform_identity(size, origin):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    da.utils.assert_eq(
        d, dask_image.ndfilters.uniform_filter(d, size, origin=origin)
    )

    da.utils.assert_eq(
        scipy.ndimage.uniform_filter(a, size, origin=origin),
        dask_image.ndfilters.uniform_filter(d, size, origin=origin)
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

    da.utils.assert_eq(
        scipy.ndimage.uniform_filter(a, size, origin=origin),
        dask_image.ndfilters.uniform_filter(d, size, origin=origin)
    )
