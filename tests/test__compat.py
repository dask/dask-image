#!/usr/bin/env python
# -*- coding: utf-8 -*-


import distutils.version as ver

import pytest

import numpy as np

import dask
import dask.array as da
import dask.array.utils as dau

import dask_ndmeasure._compat


dask_0_14_0 = ver.LooseVersion(dask.__version__) >= ver.LooseVersion("0.14.0")
dask_0_14_1 = ver.LooseVersion(dask.__version__) >= ver.LooseVersion("0.14.1")


@pytest.mark.parametrize("x", [
    list(range(5)),
    np.random.randint(10, size=(15, 16)),
    da.random.randint(10, size=(15, 16), chunks=(5, 5)),
])
def test_asarray(x):
    d = dask_ndmeasure._compat._asarray(x)

    assert isinstance(d, da.Array)

    if not isinstance(x, (np.ndarray, da.Array)):
        x = np.asarray(x)

    dau.assert_eq(d, x)


def test_indices_no_chunks():
    with pytest.raises(ValueError):
        dask_ndmeasure._compat._indices((1,))


def test_indices_wrong_chunks():
    with pytest.raises(ValueError):
        dask_ndmeasure._compat._indices((1,), chunks=tuple())


@pytest.mark.parametrize(
    "dimensions, dtype, chunks",
    [
        (tuple(), int, tuple()),
        (tuple(), float, tuple()),
        ((0,), float, (1,)),
        ((0, 1, 2), float, (1, 1, 2)),
    ]
)
def test_empty_indicies(dimensions, dtype, chunks):
    darr = dask_ndmeasure._compat._indices(dimensions, dtype, chunks=chunks)
    nparr = np.indices(dimensions, dtype)

    assert darr.shape == nparr.shape
    assert darr.dtype == nparr.dtype

    try:
        dau.assert_eq(darr, nparr)
    except IndexError:
        if len(dimensions) and not dask_0_14_0:
            pytest.skip(
                "Dask pre-0.14.0 is unable to compute this empty array."
            )
        else:
            raise


def test_indicies():
    darr = dask_ndmeasure._compat._indices((1,), chunks=(1,))
    nparr = np.indices((1,))
    dau.assert_eq(darr, nparr)

    darr = dask_ndmeasure._compat._indices((1,), float, chunks=(1,))
    nparr = np.indices((1,), float)
    dau.assert_eq(darr, nparr)

    darr = dask_ndmeasure._compat._indices((2, 1), chunks=(2, 1))
    nparr = np.indices((2, 1))
    dau.assert_eq(darr, nparr)

    darr = dask_ndmeasure._compat._indices((2, 3), chunks=(1, 2))
    nparr = np.indices((2, 3))
    dau.assert_eq(darr, nparr)


@pytest.mark.parametrize("shape, chunks", [
    (0, ()), ((0, 0), (0, 0)), ((15, 16), (15, 16))
])
def test_argwhere(shape, chunks):
    if not np.prod(shape) and not dask_0_14_1:
        pytest.skip(
            "Dask pre-0.14.1 is unable to compute this empty array."
        )

    x = np.random.randint(10, size=shape)
    d = da.from_array(x, chunks=chunks)

    x_nz = np.argwhere(x)
    d_nz = dask_ndmeasure._compat._argwhere(d)

    dau.assert_eq(d_nz, x_nz)


def test_argwhere_arr():
    x = np.random.randint(10, size=(15, 16)).astype(object)

    x_nz = np.argwhere(x)
    d_nz = dask_ndmeasure._compat._argwhere(x)

    dau.assert_eq(d_nz, x_nz)


@pytest.mark.skipif(
    not dask_0_14_1,
    reason="Dask pre-0.14.1 is unable to compute this object array."
)
def test_argwhere_obj():
    x = np.random.randint(10, size=(15, 16)).astype(object)
    d = da.from_array(x, chunks=(4, 5))

    x_nz = np.argwhere(x)
    d_nz = dask_ndmeasure._compat._argwhere(d)

    dau.assert_eq(d_nz, x_nz)


def test_argwhere_str():
    x = np.array(list("Hello world"))
    d = da.from_array(x, chunks=(4,))

    x_nz = np.argwhere(x)
    d_nz = dask_ndmeasure._compat._argwhere(d)

    dau.assert_eq(d_nz, x_nz)


@pytest.mark.parametrize("axis", [
    None, 0, 1,
])
def test_compress_err_condition(axis):
    shape = (15, 16)
    chunks = (15, 16)

    x = np.random.randint(10, size=shape)
    d = da.from_array(x, chunks=chunks)

    m = (x < 5)
    d_m = da.from_array(m, chunks=chunks)

    with pytest.raises(ValueError):
        d_nz = dask_ndmeasure._compat._compress(d_m, d, axis)


@pytest.mark.parametrize("axis", [
    0.5, -3, 2,
])
def test_compress_err_axis(axis):
    shape = (15, 16)
    chunks = (15, 16)

    x = np.random.randint(10, size=shape)
    d = da.from_array(x, chunks=chunks)

    m = (x < 5)
    d_m = da.from_array(m, chunks=chunks)
    if axis is None:
        m = m.flatten()
        d_m = d_m.flatten()
    else:
        sl = tuple(slice(None) if i == axis else 0 for i in range(m.ndim))
        m = m[sl]
        d_m = d_m[sl]

    with pytest.raises(ValueError):
        d_nz = dask_ndmeasure._compat._compress(d_m, d, axis)


@pytest.mark.parametrize("axis", [
    0, 1,
])
def test_compress_err_len(axis):
    shape = (15, 16)
    chunks = (15, 16)

    x = np.random.randint(10, size=shape)
    d = da.from_array(x, chunks=chunks)

    m = (x < 5)
    d_m = da.from_array(m, chunks=chunks)
    m = m.flatten()
    d_m = d_m.flatten()

    with pytest.raises(IndexError):
        d_nz = dask_ndmeasure._compat._compress(d_m, d, axis)


@pytest.mark.parametrize("shape, chunks, axis", [
    (0, (), None),
    ((0, 0), (0, 0), None),
    ((0, 0), (0, 0), 0),
    ((0, 0), (0, 0), 1),
    ((15, 16), (15, 16), None),
    ((15, 16), (15, 16), 0),
    ((15, 16), (15, 16), 1),
])
def test_compress(shape, chunks, axis):
    if not np.prod(shape) and not dask_0_14_1:
        pytest.skip(
            "Dask pre-0.14.1 is unable to compute this empty array."
        )

    x = np.random.randint(10, size=shape)
    d = da.from_array(x, chunks=chunks)

    m = (x < 5)
    d_m = da.from_array(m, chunks=chunks)
    if axis is None:
        m = m.flatten()
        d_m = d_m.flatten()
    else:
        sl = tuple(slice(None) if i == axis else 0 for i in range(m.ndim))
        m = m[sl]
        d_m = d_m[sl]

    x_nz = np.compress(m, x, axis)
    d_nz = dask_ndmeasure._compat._compress(d_m, d, axis)

    dau.assert_eq(d_nz, x_nz)
