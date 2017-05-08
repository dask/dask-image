# -*- coding: utf-8 -*-


import distutils.version as ver

import pytest

import numpy as np

import dask
import dask.array.utils as dau

import dask_ndfourier._compat


old_dask = ver.LooseVersion(dask.__version__) <= ver.LooseVersion("0.13.0")


def test_indices_no_chunks():
    with pytest.raises(ValueError):
        dask_ndfourier._compat._indices((1,))


def test_indices_wrong_chunks():
    with pytest.raises(ValueError):
        dask_ndfourier._compat._indices((1,), chunks=tuple())


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
    darr = dask_ndfourier._compat._indices(dimensions, dtype, chunks=chunks)
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
    darr = dask_ndfourier._compat._indices((1,), chunks=(1,))
    nparr = np.indices((1,))
    dau.assert_eq(darr, nparr)

    darr = dask_ndfourier._compat._indices((1,), float, chunks=(1,))
    nparr = np.indices((1,), float)
    dau.assert_eq(darr, nparr)

    darr = dask_ndfourier._compat._indices((2, 1), chunks=(2, 1))
    nparr = np.indices((2, 1))
    dau.assert_eq(darr, nparr)

    darr = dask_ndfourier._compat._indices((2, 3), chunks=(1, 2))
    nparr = np.indices((2, 3))
    dau.assert_eq(darr, nparr)
