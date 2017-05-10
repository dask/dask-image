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
def test_fourier_shift_err(err_type, arg1_, n):
    a = np.arange(140.0).reshape(10, 14).astype(complex)
    d = da.from_array(a, chunks=(5, 7))

    with pytest.raises(err_type):
        da_ndf.fourier_shift(d, arg1_, n)


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
def test_fourier_shift_identity(arg1_, in_dtype, out_dtype):
    a = np.arange(140.0).reshape(10, 14).astype(in_dtype)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(d.astype(out_dtype), da_ndf.fourier_shift(d, arg1_))

    dau.assert_eq(
        sp_ndf.fourier_shift(a, arg1_), da_ndf.fourier_shift(d, arg1_)
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
def test_fourier_shift(arg1_):
    a = np.arange(140.0).reshape(10, 14).astype(complex)
    d = da.from_array(a, chunks=(5, 7))

    dau.assert_eq(
        sp_ndf.fourier_shift(a, arg1_), da_ndf.fourier_shift(d, arg1_)
    )
