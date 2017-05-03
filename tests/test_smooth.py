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
    "err_type, sigma, truncate",
    [
        (RuntimeError, [[1.0]], 4.0),
        (RuntimeError, [1.0], 4.0),
        (TypeError, 1.0 + 0.0j, 4.0),
        (TypeError, 1.0, 4.0 + 0.0j),
    ]
)
def test_gaussian_filter_params(err_type, sigma, truncate):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_ndf.gaussian_filter(d, sigma, truncate=truncate)


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
def test_gaussian_identity(order, sigma, truncate):
    a = np.arange(140.0).reshape(10, 14)
    d = da.from_array(a, chunks=(5, 7))

    if order % 2 == 1 and sigma != 0 and truncate == 0:
        pytest.skip(
            "SciPy zeros the result of a Gaussian filter with odd derivatives"
            " when sigma is non-zero, truncate is zero, and derivative is odd."
            "\n\nxref: https://github.com/scipy/scipy/issues/7364"
        )

    dau.assert_eq(
        d, da_ndf.gaussian_filter(d, sigma, order, truncate=truncate)
    )

    dau.assert_eq(
        sp_ndf.gaussian_filter(a, sigma, order, truncate=truncate),
        da_ndf.gaussian_filter(d, sigma, order, truncate=truncate)
    )


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
    "order", [
        0,
        1,
        2,
        3,
        (0, 1),
        (2, 3),
    ]
)
def test_gaussian_compare(order, sigma, truncate):
    s = (100, 110)
    a = np.arange(float(np.prod(s))).reshape(s)
    d = da.from_array(a, chunks=(50, 55))

    dau.assert_eq(
        sp_ndf.gaussian_filter(a, sigma, order, truncate=truncate),
        da_ndf.gaussian_filter(d, sigma, order, truncate=truncate)
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
