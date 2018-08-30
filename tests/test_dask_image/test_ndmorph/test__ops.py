#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import


import pytest

import numpy

import dask.array

from dask_image.ndmorph import _ops


@pytest.mark.parametrize(
    "condition, x, y",
    [
        (
            True,
            dask.array.arange(2, chunks=(2,)),
            dask.array.arange(2, 4, chunks=(2,))
        ),
        (
            False,
            dask.array.arange(2, chunks=(2,)),
            dask.array.arange(2, 4, chunks=(2,))
        ),
        (
            True,
            dask.array.arange(2, chunks=(2,)),
            dask.array.arange(2, 4, dtype=float, chunks=(2,))
        ),
        (
            False,
            dask.array.arange(2, dtype=float, chunks=(2,)),
            dask.array.arange(2, 4, chunks=(2,))
        ),
        (
            numpy.bool8(True),
            dask.array.arange(2, chunks=(2,)),
            dask.array.arange(2, 4, chunks=(2,))
        ),
        (
            numpy.bool8(False),
            dask.array.arange(2, chunks=(2,)),
            dask.array.arange(2, 4, chunks=(2,))
        ),
        (
            dask.array.arange(2, dtype=bool, chunks=(2,)),
            dask.array.arange(2, chunks=(2,)),
            dask.array.arange(2, 4, chunks=(2,))
        ),
    ]
)
def test__where(condition, x, y):
    dask_result = dask.array.where(condition, x, y)

    result = _ops._where(condition, x, y)

    assert result.dtype.type == dask_result.dtype.type
    assert numpy.array((result == dask_result).all())[()]

    if isinstance(condition, (bool, numpy.bool8)):
        dtype = numpy.promote_types(x.dtype, y.dtype)
        if condition:
            return x.astype(dtype)
        else:
            return y.astype(dtype)

        assert numpy.array((result == expected).all())[()]
        assert result is expected
