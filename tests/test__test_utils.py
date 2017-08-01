#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest

import numpy as np

import dask.array as da

import dask_ndmeasure._test_utils


nan = np.nan


@pytest.mark.parametrize("match, a, b", [
    [True] + 2 * [np.array(2)[()]],
    [True] + 2 * [np.array(nan)[()]],
    [True] + 2 * [np.array(2)],
    [True] + 2 * [np.array(nan)],
    [True] + [np.array(1.0), da.ones(tuple(), chunks=tuple())],
    [True] + 2 * [np.random.randint(10, size=(15, 16))],
    [True] + 2 * [da.random.randint(10, size=(15, 16), chunks=(5, 5))],
    [True, np.array([2, nan]), np.array([2, nan])],
    [False, np.array([2, nan]), np.array([3, nan])],
    [False, np.array([2, nan]), np.array([2, 3])],
    [True, np.array([2, 3]), da.from_array(np.array([2, 3]), chunks=1)],
    [True, np.array([nan]), da.from_array(np.array([nan]), chunks=1)],
    [False, np.array([2]), da.from_array(np.array([nan]), chunks=1)],
    [False, np.array([nan]), da.from_array(np.array([2]), chunks=1)],
    [True, np.array([2, nan]), da.from_array(np.array([2, nan]), chunks=1)],
    [False, np.array([2, nan]), da.from_array(np.array([3, nan]), chunks=1)],
    [False, np.array([2, nan]), da.from_array(np.array([2, 3]), chunks=1)],
])
def test_assert_eq_nan(match, a, b):
    if match:
        dask_ndmeasure._test_utils._assert_eq_nan(a, b)
    else:
        with pytest.raises(AssertionError):
            dask_ndmeasure._test_utils._assert_eq_nan(a, b)
