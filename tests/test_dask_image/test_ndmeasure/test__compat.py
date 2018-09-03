#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest

import numpy as np

import dask
import dask.array as da
import dask.array.utils as dau

import dask_image.ndmeasure._compat


@pytest.mark.parametrize("x", [
    list(range(5)),
    np.random.randint(10, size=(15, 16)),
    da.random.randint(10, size=(15, 16), chunks=(5, 5)),
])
def test_asarray(x):
    d = dask_image.ndmeasure._compat._asarray(x)

    assert isinstance(d, da.Array)

    if not isinstance(x, (np.ndarray, da.Array)):
        x = np.asarray(x)

    dau.assert_eq(d, x)
