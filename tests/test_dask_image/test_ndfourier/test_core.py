#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numbers
from distutils.version import LooseVersion

import pytest

import numpy as np
import scipy as sp
import scipy.ndimage.fourier as sp_ndf

import dask.array as da
import dask.array.utils as dau
import dask_image.ndfourier as da_ndf


@pytest.mark.parametrize(
    "err_type, s, n",
    [
        (NotImplementedError, 0.0, 0),
        (TypeError, 0.0 + 0.0j, 0),
        (TypeError, {}, 0),
        (RuntimeError, [0.0], 0),
        (RuntimeError, [[0.0], [0.0]], 0),
        (TypeError, [0.0, 0.0 + 0.0j], 0),
        (NotImplementedError, 0, 0),
    ]
)
@pytest.mark.parametrize(
    "funcname",
    [
        "fourier_shift",
        "fourier_gaussian",
        "fourier_uniform",
    ]
)
def test_fourier_filter_err(funcname, err_type, s, n):
    da_func = getattr(da_ndf, funcname)

    a = np.arange(140.0).reshape(10, 14).astype(complex)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_func(d, s, n)


@pytest.mark.parametrize(
    "s",
    [
        0,
        (0, 0),
    ]
)
@pytest.mark.parametrize(
    "funcname",
    [
        "fourier_shift",
        "fourier_gaussian",
    ]
)
def test_fourier_filter_identity(funcname, s):
    da_func = getattr(da_ndf, funcname)
    sp_func = getattr(sp_ndf, funcname)

    a = np.arange(140.0).reshape(10, 14).astype(complex)
    d = da.from_array(a, chunks=(5, 7))

    r_a = sp_func(a, s)
    r_d = da_func(d, s)

    assert d.chunks == r_d.chunks

    dau.assert_eq(d, r_d)
    dau.assert_eq(r_a, r_d)


@pytest.mark.parametrize(
    "dtype",
    [
        np.int64,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
)
@pytest.mark.parametrize(
    "funcname, upcast_type",
    [
        ("fourier_shift", numbers.Real),
        ("fourier_gaussian", numbers.Integral),
        ("fourier_uniform", numbers.Integral),
    ]
)
def test_fourier_filter_type(funcname, upcast_type, dtype):
    if (
            LooseVersion(sp.__version__) >= "1.0.0" and
            dtype in [np.int64, np.float64] and
            funcname in ["fourier_gaussian", "fourier_uniform"]
       ):
        pytest.skip(
            "SciPy 1.0.0+ doesn't handle double precision values correctly."
        )

    dtype = np.dtype(dtype).type

    s = 1

    da_func = getattr(da_ndf, funcname)
    sp_func = getattr(sp_ndf, funcname)

    a = np.arange(140.0).reshape(10, 14).astype(dtype)
    d = da.from_array(a, chunks=(5, 7))

    r_a = sp_func(a, s)
    r_d = da_func(d, s)

    assert d.chunks == r_d.chunks

    dau.assert_eq(r_a, r_d)

    if issubclass(dtype, upcast_type):
        assert r_d.real.dtype.type is np.float64
    else:
        assert r_d.dtype.type is dtype


@pytest.mark.parametrize(
    "shape, chunks",
    [
        ((10, 14), (10, 14)),
        ((10, 14), (5, 7)),
        ((10, 14), (6, 8)),
        ((10, 14), (4, 6)),
        ((16,), (3, 6, 2, 5)),
    ]
)
@pytest.mark.parametrize(
    "funcname",
    [
        "fourier_shift",
        "fourier_gaussian",
        "fourier_uniform",
    ]
)
def test_fourier_filter_chunks(funcname, shape, chunks):
    dtype = np.dtype(complex).type

    s = 1

    da_func = getattr(da_ndf, funcname)
    sp_func = getattr(sp_ndf, funcname)

    a = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    d = da.from_array(a, chunks=chunks)

    r_a = sp_func(a, s)
    r_d = da_func(d, s)

    assert d.chunks == r_d.chunks

    dau.assert_eq(r_a, r_d)


@pytest.mark.parametrize(
    "s",
    [
        -1,
        (-1, -1),
        (-1, 2),
        (10, -9),
        (1, 0),
        (0, 2),
    ]
)
@pytest.mark.parametrize(
    "funcname",
    [
        "fourier_shift",
        "fourier_gaussian",
    ]
)
def test_fourier_filter_non_positive(funcname, s):
    da_func = getattr(da_ndf, funcname)
    sp_func = getattr(sp_ndf, funcname)

    a = np.arange(140.0).reshape(10, 14).astype(complex)
    d = da.from_array(a, chunks=(5, 7))

    r_a = sp_func(a, s)
    r_d = da_func(d, s)

    assert d.chunks == r_d.chunks

    dau.assert_eq(r_a, r_d)


@pytest.mark.parametrize(
    "s",
    [
        1,
        0.5,
        (1, 1),
        (0.8, 1.5),
        np.ones((2,)),
        da.ones((2,), chunks=(2,)),
        da.ones((2,), chunks=(1,)),
    ]
)
@pytest.mark.parametrize(
    "funcname",
    [
        "fourier_shift",
        "fourier_gaussian",
        "fourier_uniform",
    ]
)
def test_fourier_filter(funcname, s):
    da_func = getattr(da_ndf, funcname)
    sp_func = getattr(sp_ndf, funcname)

    a = np.arange(140.0).reshape(10, 14).astype(complex)
    d = da.from_array(a, chunks=(5, 7))

    r_a = sp_func(a, s)
    r_d = da_func(d, s)

    assert d.chunks == r_d.chunks

    dau.assert_eq(r_a, r_d)
