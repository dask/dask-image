# -*- coding: utf-8 -*-


import distutils.version as ver

import pytest

import numpy as np

import dask
import dask.array as da
import dask.array.utils as dau

import dask_ndmeasure._compat


old_dask = ver.LooseVersion(dask.__version__) <= ver.LooseVersion("0.13.0")


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
        if len(dimensions) and old_dask:
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


def test_argwhere():
    for shape, chunks in [(0, ()), ((0, 0), (0, 0)), ((15, 16), (4, 5))]:
        x = np.random.randint(10, size=shape)
        d = da.from_array(x, chunks=chunks)

        x_nz = np.argwhere(x)
        d_nz = dask_ndmeasure._compat._argwhere(d)

        dau.assert_eq(d_nz, x_nz)


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
