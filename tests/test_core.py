#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

import numpy as np

import scipy.ndimage as spnd

import dask.array as da

import dask_ndmeasure
import dask_ndmeasure._test_utils


@pytest.mark.parametrize(
    "funcname", [
        "center_of_mass",
        "extrema",
        "maximum",
        "maximum_position",
        "mean",
        "minimum",
        "minimum_position",
        "standard_deviation",
        "sum",
        "variance",
    ]
)
def test_measure_props_err(funcname):
    da_func = getattr(dask_ndmeasure, funcname)

    shape = (15, 16)
    chunks = (4, 5)
    ind = None

    a = np.random.random(shape)
    d = da.from_array(a, chunks=chunks)

    lbls = (a < 0.5).astype(np.int64)
    d_lbls = da.from_array(lbls, chunks=d.chunks)

    lbls = lbls[:-1]
    d_lbls = d_lbls[:-1]

    with pytest.raises(ValueError):
        da_func(d, lbls, ind)


@pytest.mark.parametrize(
    "funcname", [
        "center_of_mass",
        "maximum",
        "maximum_position",
        "mean",
        "minimum",
        "minimum_position",
        "standard_deviation",
        "sum",
        "variance",
    ]
)
@pytest.mark.parametrize(
    "shape, chunks, has_lbls, ind", [
        ((5, 6, 4), (2, 3, 2), False, None),
        ((15, 16), (4, 5), False, None),
        ((15, 16), (4, 5), True, None),
        ((15, 16), (4, 5), True, 0),
        ((15, 16), (4, 5), True, 1),
        ((15, 16), (4, 5), True, [1]),
        ((15, 16), (4, 5), True, [1, 2]),
        ((5, 6, 4), (2, 3, 2), True, [1, 2]),
        ((15, 16), (4, 5), True, [1, 100]),
        ((5, 6, 4), (2, 3, 2), True, [1, 100]),
        ((15, 16), (4, 5), True, [[1, 2, 3, 4]]),
        ((15, 16), (4, 5), True, [[1, 2], [3, 4]]),
        ((15, 16), (4, 5), True, [[[1], [2], [3], [4]]]),
    ]
)
def test_measure_props(funcname, shape, chunks, has_lbls, ind):
    sp_func = getattr(spnd, funcname)
    da_func = getattr(dask_ndmeasure, funcname)

    a = np.random.random(shape)
    d = da.from_array(a, chunks=chunks)

    lbls = None
    d_lbls = None

    if has_lbls:
        lbls = np.zeros(a.shape, dtype=np.int64)
        lbls += (
            (a < 0.5).astype(lbls.dtype) +
            (a < 0.25).astype(lbls.dtype) +
            (a < 0.125).astype(lbls.dtype) +
            (a < 0.0625).astype(lbls.dtype)
        )
        d_lbls = da.from_array(lbls, chunks=d.chunks)

    a_r = np.array(sp_func(a, lbls, ind))
    d_r = da_func(d, d_lbls, ind)

    assert a_r.dtype == d_r.dtype
    assert a_r.shape == d_r.shape
    assert np.allclose(np.array(a_r), np.array(d_r), equal_nan=True)


@pytest.mark.parametrize(
    "shape, chunks, has_lbls, ind", [
        ((15, 16), (4, 5), False, None),
        ((5, 6, 4), (2, 3, 2), False, None),
        ((15, 16), (4, 5), True, None),
        ((15, 16), (4, 5), True, 0),
        ((15, 16), (4, 5), True, 1),
        ((15, 16), (4, 5), True, [1]),
        ((15, 16), (4, 5), True, [1, 2]),
        ((5, 6, 4), (2, 3, 2), True, [1, 2]),
        ((15, 16), (4, 5), True, [1, 100]),
        ((5, 6, 4), (2, 3, 2), True, [1, 100]),
        ((15, 16), (4, 5), True, [[1, 2, 3, 4]]),
        ((15, 16), (4, 5), True, [[1, 2], [3, 4]]),
        ((15, 16), (4, 5), True, [[[1], [2], [3], [4]]]),
    ]
)
def test_extrema(shape, chunks, has_lbls, ind):
    a = np.random.random(shape)
    d = da.from_array(a, chunks=chunks)

    lbls = None
    d_lbls = None

    if has_lbls:
        lbls = np.zeros(a.shape, dtype=np.int64)
        lbls += (
            (a < 0.5).astype(lbls.dtype) +
            (a < 0.25).astype(lbls.dtype) +
            (a < 0.125).astype(lbls.dtype) +
            (a < 0.0625).astype(lbls.dtype)
        )
        d_lbls = da.from_array(lbls, chunks=d.chunks)

    a_r = spnd.extrema(a, lbls, ind)
    d_r = dask_ndmeasure.extrema(d, d_lbls, ind)

    assert len(a_r) == len(d_r)

    for i in range(len(a_r)):
        a_r_i = np.array(a_r[i])
        assert a_r_i.dtype == d_r[i].dtype
        assert a_r_i.shape == d_r[i].shape
        assert np.allclose(a_r_i, np.array(d_r[i]), equal_nan=True)


@pytest.mark.parametrize(
    "shape, chunks, ind", [
        ((15, 16), (4, 5), None),
        ((5, 6, 4), (2, 3, 2), None),
        ((15, 16), (4, 5), 0),
        ((15, 16), (4, 5), 1),
        ((15, 16), (4, 5), [1]),
        ((15, 16), (4, 5), [1, 2]),
        ((5, 6, 4), (2, 3, 2), [1, 2]),
        ((15, 16), (4, 5), [1, 100]),
        ((5, 6, 4), (2, 3, 2), [1, 100]),
    ]
)
@pytest.mark.parametrize(
    "default", [
        None,
        0,
        1.5,
    ]
)
@pytest.mark.parametrize(
    "pass_positions", [
        False,
        True,
    ]
)
def test_labeled_comprehension(shape, chunks, ind, default, pass_positions):
    a = np.random.random(shape)
    d = da.from_array(a, chunks=chunks)

    lbls = np.zeros(a.shape, dtype=np.int64)
    lbls += (
        (a < 0.5).astype(lbls.dtype) +
        (a < 0.25).astype(lbls.dtype) +
        (a < 0.125).astype(lbls.dtype) +
        (a < 0.0625).astype(lbls.dtype)
    )
    d_lbls = da.from_array(lbls, chunks=d.chunks)

    def func(val, pos=None):
        if pos is None:
            pos = 0 * val + 1

        return (val * pos).sum() / (1 + val.max() * pos.max())

    a_cm = spnd.labeled_comprehension(
        a, lbls, ind, func, np.float64, default, pass_positions
    )
    d_cm = dask_ndmeasure.labeled_comprehension(
        d, d_lbls, ind, func, np.float64, default, pass_positions
    )

    assert a_cm.dtype == d_cm.dtype
    assert a_cm.shape == d_cm.shape
    assert np.allclose(np.array(a_cm), np.array(d_cm), equal_nan=True)
