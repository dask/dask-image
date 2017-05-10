#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

import numpy as np
import scipy.ndimage.fourier as sp_ndf

import dask.array as da
import dask.array.utils as dau
import dask_ndfourier as da_ndf


def test_import_core():
    try:
        from dask_ndfourier import core
    except ImportError:
        pytest.fail("Unable to import `core`.")


@pytest.mark.parametrize(
    "err_type, arg1_, n",
    [
        (NotImplementedError, 0.0, 0),
        (TypeError, 0.0 + 0.0j, 0),
        (TypeError, {}, 0),
        (RuntimeError, [0.0], 0),
        (TypeError, [0.0, 0.0 + 0.0j], 0),
        (NotImplementedError, 0, 0),
    ]
)
@pytest.mark.parametrize(
    "funcname",
    [
        "fourier_shift",
    ]
)
def test_fourier_filter_err(funcname, err_type, arg1_, n):
    da_func = getattr(da_ndf, funcname)

    a = np.arange(140.0).reshape(10, 14).astype(complex)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_func(d, arg1_, n)


@pytest.mark.parametrize(
    "arg1_",
    [
        0,
        (0, 0),
    ]
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        (complex, complex),
        (float, complex),
    ]
)
@pytest.mark.parametrize(
    "funcname",
    [
        "fourier_shift",
    ]
)
def test_fourier_filter_identity(funcname, arg1_, in_dtype, out_dtype):
    da_func = getattr(da_ndf, funcname)
    sp_func = getattr(sp_ndf, funcname)

    a = np.arange(140.0).reshape(10, 14).astype(in_dtype)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(d.astype(out_dtype), da_func(d, arg1_))

    dau.assert_eq(
        sp_func(a, arg1_), da_func(d, arg1_)
    )


@pytest.mark.parametrize(
    "arg1_",
    [
        1,
        0.5,
        -1,
        (1, 1),
        (0.8, 1.5),
        (-1, -1),
        (1, 0),
        (0, 2),
        (-1, 2),
        (10, -9),
    ]
)
@pytest.mark.parametrize(
    "funcname",
    [
        "fourier_shift",
    ]
)
def test_fourier_filter(funcname, arg1_):
    da_func = getattr(da_ndf, funcname)
    sp_func = getattr(sp_ndf, funcname)

    a = np.arange(140.0).reshape(10, 14).astype(complex)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        sp_func(a, arg1_), da_func(d, arg1_)
    )
