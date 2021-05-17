#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import scipy.ndimage

import dask.array as da

import dask_image.ndfilters as da_ndf


@pytest.mark.parametrize(
    "err_type, sigma, truncate",
    [
        (RuntimeError, [[1.0]], 4.0),
        (RuntimeError, [1.0], 4.0),
        (TypeError, 1.0 + 0.0j, 4.0),
        (TypeError, 1.0, 4.0 + 0.0j),
    ]
)
@pytest.mark.parametrize(
    "da_func",
    [
        da_ndf.gaussian_filter,
        da_ndf.gaussian_gradient_magnitude,
        da_ndf.gaussian_laplace,
    ]
)
def test_gaussian_filters_params(da_func, err_type, sigma, truncate):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_func(d, sigma, truncate=truncate)


@pytest.mark.parametrize(
    "sigma, truncate",
    [
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, 4.0),
        (1.0, 0.0),
    ]
)
@pytest.mark.parametrize(
    "order", [0, 1, 2, 3]
)
@pytest.mark.parametrize(
    "sp_func, da_func",
    [
        (scipy.ndimage.filters.gaussian_filter, da_ndf.gaussian_filter),
    ]
)
def test_gaussian_filters_identity(sp_func, da_func, order, sigma, truncate):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    if order % 2 == 1 and sigma != 0 and truncate == 0:
        pytest.skip(
            "SciPy zeros the result of a Gaussian filter with odd derivatives"
            " when sigma is non-zero, truncate is zero, and derivative is odd."
            "\n\nxref: https://github.com/scipy/scipy/issues/7364"
        )

    da.utils.assert_eq(
        d, da_func(d, sigma, order, truncate=truncate)
    )

    da.utils.assert_eq(
        sp_func(a, sigma, order, truncate=truncate),
        da_func(d, sigma, order, truncate=truncate)
    )


@pytest.mark.parametrize(
    "da_func",
    [
        da_ndf.gaussian_filter,
        da_ndf.gaussian_gradient_magnitude,
        da_ndf.gaussian_laplace,
    ]
)
def test_gaussian_filter_shape_type(da_func):
    sigma = 1.0
    truncate = 4.0

    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    assert all([(type(s) is int) for s in d.shape])

    d2 = da_func(d, sigma=sigma, truncate=truncate)

    assert all([(type(s) is int) for s in d2.shape])


@pytest.mark.parametrize(
    "da_func",
    [
        da_ndf.gaussian_filter,
        da_ndf.gaussian_gradient_magnitude,
        da_ndf.gaussian_laplace,
    ]
)
def test_gaussian_filter_comprehensions(da_func):
    da_wfunc = lambda arr: da_func(arr, 1.0, truncate=4.0)  # noqa: E731

    np.random.seed(0)

    a = np.random.random((3, 12, 14))
    d = da.from_array(a, chunks=(3, 6, 7))

    l2s = [da_wfunc(d[i]) for i in range(len(d))]
    l2c = [da_wfunc(d[i])[None] for i in range(len(d))]

    da.utils.assert_eq(np.stack(l2s), da.stack(l2s))
    da.utils.assert_eq(np.concatenate(l2c), da.concatenate(l2c))


@pytest.mark.parametrize(
    "sigma, truncate",
    [
        (1.0, 2.0),
        (1.0, 4.0),
        (2.0, 2.0),
        (2.0, 4.0),
        ((1.0, 2.0), 4.0),
    ]
)
@pytest.mark.parametrize(
    "sp_func, da_func",
    [
        (scipy.ndimage.filters.gaussian_filter,
         da_ndf.gaussian_filter),
        (scipy.ndimage.filters.gaussian_gradient_magnitude,
         da_ndf.gaussian_gradient_magnitude),
        (scipy.ndimage.filters.gaussian_laplace,
         da_ndf.gaussian_laplace),
    ]
)
def test_gaussian_filters_compare(sp_func, da_func, sigma, truncate):
    s = (100, 110)
    a = np.arange(float(np.prod(s))).reshape(s)
    d = da.from_array(a, chunks=(50, 55))

    da.utils.assert_eq(
        sp_func(a, sigma, truncate=truncate),
        da_func(d, sigma, truncate=truncate)
    )


@pytest.mark.parametrize(
    "sigma, truncate",
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (1.0, 4.0),
        (2.0, 2.0),
        (2.0, 4.0),
        ((1.0, 2.0), 4.0),
    ]
)
@pytest.mark.parametrize(
    "order", [
        0,
        1,
        2,
        3,
        (0, 1),
        (2, 3),
    ]
)
@pytest.mark.parametrize(
    "sp_func, da_func",
    [
        (scipy.ndimage.filters.gaussian_filter, da_ndf.gaussian_filter),
    ]
)
def test_gaussian_derivative_filters_compare(sp_func, da_func,
                                             order, sigma, truncate):
    s = (100, 110)
    a = np.arange(float(np.prod(s))).reshape(s)
    d = da.from_array(a, chunks=(50, 55))

    da.utils.assert_eq(
        sp_func(a, sigma, order, truncate=truncate),
        da_func(d, sigma, order, truncate=truncate)
    )
