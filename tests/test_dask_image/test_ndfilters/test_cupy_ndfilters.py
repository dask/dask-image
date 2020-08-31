#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

from dask_image import ndfilters

cupy = pytest.importorskip("cupy", minversion="7.7.0")


@pytest.fixture
def array():
    s = (10, 10)
    a = da.from_array(cupy.arange(int(np.prod(s)),
                      dtype=cupy.float32).reshape(s), chunks=5)
    return a


@pytest.mark.cupy
@pytest.mark.parametrize("func", [
    ndfilters.convolve,
    ndfilters.correlate,
])
def test_cupy_conv(array, func):
    """Test convolve & correlate filters with cupy input arrays."""
    weights = cupy.ones(array.ndim * (3,), dtype=cupy.float32)
    result = func(array, weights)
    result.compute()


@pytest.mark.cupy
@pytest.mark.parametrize("func", [
    ndfilters.laplace,
])
def test_cupy_diff(array, func):
    result = func(array)
    result.compute()


@pytest.mark.cupy
@pytest.mark.parametrize("func", [
    ndfilters.prewitt,
    ndfilters.sobel,
])
def test_cupy_edge(array, func):
    result = func(array)
    result.compute()


@pytest.mark.cupy
@pytest.mark.parametrize("func", [
    ndfilters.gaussian_filter,
    ndfilters.gaussian_gradient_magnitude,
    ndfilters.gaussian_laplace,
])
def test_cupy_gaussian(array, func):
    sigma = 1
    result = func(array, sigma)
    result.compute()


@pytest.mark.parametrize(
    "size, footprint",
    [
        (1, None),
        ((1, 1), None),
        (None, np.ones((1, 1))),
    ]
)
def test_cupy_generic(array, size, footprint):
    my_sum = cupy.ReductionKernel(
        'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')
    result = ndfilters.generic_filter(array, my_sum, size=size,
                                      footprint=footprint)
    result.compute()


@pytest.mark.cupy
@pytest.mark.parametrize("func, extra_arg, size", [
    (ndfilters.minimum_filter, None, 3),
    (ndfilters.median_filter, None, 3),
    (ndfilters.maximum_filter, None, 3),
    (ndfilters.rank_filter, 5, 3),
    (ndfilters.percentile_filter, 50, 3),
])
def test_cupy_order(array, func, extra_arg, size):
    if extra_arg is not None:
        result = func(array, extra_arg, size=size)
    else:
        result = func(array, size=size)
    result.compute()


@pytest.mark.cupy
@pytest.mark.parametrize("func", [
    ndfilters.uniform_filter,
])
def test_cupy_smooth(array, func):
    result = func(array)
    result.compute()
