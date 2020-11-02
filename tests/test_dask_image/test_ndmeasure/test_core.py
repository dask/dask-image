#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from distutils.version import LooseVersion
import scipy
import itertools as it
import warnings as wrn

import pytest

import numpy as np

import scipy
import scipy.ndimage as spnd

import dask.array as da

import dask_image.ndmeasure


@pytest.mark.parametrize(
    "funcname", [
        "center_of_mass",
        "extrema",
        "maximum",
        "maximum_position",
        "mean",
        "median",
        "minimum",
        "minimum_position",
        "standard_deviation",
        "sum_labels",
        "variance",
    ]
)
def test_measure_props_err(funcname):
    da_func = getattr(dask_image.ndmeasure, funcname)

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
    "datatype", [
        int,
        float,
        np.bool,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ]
)
def test_center_of_mass(datatype):
    a = np.array([[1, 1], [0, 0]]).astype(datatype)
    d = da.from_array(a, chunks=(1, 2))

    actual = dask_image.ndmeasure.center_of_mass(d).compute()
    expected = [0., 0.5]

    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "funcname", [
        "center_of_mass",
        "maximum",
        "maximum_position",
        "mean",
        "median",
        "minimum",
        "minimum_position",
        "standard_deviation",
        "sum_labels",
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
    # early scipy version uses a different name sum insted of sum_labels.
    if funcname == 'sum_labels' and scipy.__version__ < LooseVersion('1.6.0'):
        scipy_funcname = 'sum'
    else:
        scipy_funcname = funcname

    sp_func = getattr(spnd, scipy_funcname)
    da_func = getattr(dask_image.ndmeasure, funcname)

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

    if a_r.dtype != d_r.dtype:
        wrn.warn(
            "Encountered a type mismatch."
            " Expected type, %s, but got type, %s."
            "" % (str(a_r.dtype), str(d_r.dtype)),
            RuntimeWarning
        )
    assert a_r.shape == d_r.shape

    # See the linked issue for details.
    # ref: https://github.com/scipy/scipy/issues/7706
    if (
        funcname == "median" and
        ind is not None and
        not np.in1d(np.atleast_1d(ind), lbls).all()
    ):
        pytest.skip("SciPy's `median` mishandles missing labels.")

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
def test_area(shape, chunks, has_lbls, ind):
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

    a_r = None
    if has_lbls:
        if ind is None:
            a_r = lbls.astype(bool).astype(np.int64).sum()
        else:
            a_r = np.bincount(
                lbls.flatten(),
                minlength=(1 + max(np.array(ind).flatten()))
            )
            a_r = a_r[np.asarray(ind)]
    else:
        a_r = np.array(a.size)[()]

    d_r = dask_image.ndmeasure.area(d, d_lbls, ind)

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
    d_r = dask_image.ndmeasure.extrema(d, d_lbls, ind)

    assert len(a_r) == len(d_r)

    for i in range(len(a_r)):
        a_r_i = np.array(a_r[i])
        if a_r_i.dtype != d_r[i].dtype:
            wrn.warn(
                "Encountered a type mismatch."
                " Expected type, %s, but got type, %s."
                "" % (str(a_r_i.dtype), str(d_r[i].dtype)),
                RuntimeWarning
            )
        assert a_r_i.shape == d_r[i].shape
        assert np.allclose(a_r_i, np.array(d_r[i]), equal_nan=True)


@pytest.mark.parametrize(
    "shape, chunks, has_lbls, ind", [
        ((15, 16), (4, 5), False, None),
        ((5, 6, 4), (2, 3, 2), False, None),
        ((15, 16), (4, 5), True, None),
        ((15, 16), (4, 5), True, 0),
        ((15, 16), (4, 5), True, 1),
        ((15, 16), (4, 5), True, 100),
        ((15, 16), (4, 5), True, [1]),
        ((15, 16), (4, 5), True, [1, 2]),
        ((5, 6, 4), (2, 3, 2), True, [1, 2]),
        ((15, 16), (4, 5), True, [1, 100]),
        ((5, 6, 4), (2, 3, 2), True, [1, 100]),
    ]
)
@pytest.mark.parametrize(
    "min, max, bins", [
        (0, 1, 5),
    ]
)
def test_histogram(shape, chunks, has_lbls, ind, min, max, bins):
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

    a_r = spnd.histogram(a, min, max, bins, lbls, ind)
    d_r = dask_image.ndmeasure.histogram(d, min, max, bins, d_lbls, ind)

    if ind is None or np.isscalar(ind):
        if a_r is None:
            assert d_r.compute() is None
        else:
            np.allclose(a_r, d_r.compute(), equal_nan=True)
    else:
        assert a_r.dtype == d_r.dtype
        assert a_r.shape == d_r.shape
        for i in it.product(*[range(_) for _ in a_r.shape]):
            if a_r[i] is None:
                assert d_r[i].compute() is None
            else:
                assert np.allclose(a_r[i], d_r[i].compute(), equal_nan=True)


def _assert_equivalent_labeling(labels0, labels1):
    """Make sure the two label arrays are equivalent.

    In the sense that if two pixels have the same label in labels0, they will
    also have the same label in labels1, and vice-versa.

    We check this by verifying that there is exactly a one-to-one mapping
    between the two label volumes.
    """
    matching = np.stack((labels0.ravel(), labels1.ravel()), axis=1)
    unique_matching = dask_image.ndmeasure._label._unique_axis(matching)
    bincount0 = np.bincount(unique_matching[:, 0])
    bincount1 = np.bincount(unique_matching[:, 1])
    assert np.all(bincount0 == 1)
    assert np.all(bincount1 == 1)


@pytest.mark.parametrize(
    "seed, prob, shape, chunks, connectivity", [
        (42, 0.4, (15, 16), (15, 16), 1),
        (42, 0.4, (15, 16), (4, 5), 1),
        (42, 0.4, (15, 16), (4, 5), 2),
        (42, 0.4, (15, 16), (8, 5), 1),
        (42, 0.4, (15, 16), (8, 5), 2),
        (42, 0.3, (10, 8, 6), (5, 4, 3), 1),
        (42, 0.3, (10, 8, 6), (5, 4, 3), 2),
        (42, 0.3, (10, 8, 6), (5, 4, 3), 3),
    ]
)
def test_label(seed, prob, shape, chunks, connectivity):
    np.random.seed(seed)

    a = np.random.random(shape) < prob
    d = da.from_array(a, chunks=chunks)

    s = spnd.generate_binary_structure(a.ndim, connectivity)

    a_l, a_nl = spnd.label(a, s)
    d_l, d_nl = dask_image.ndmeasure.label(d, s)

    assert a_nl == d_nl.compute()

    assert a_l.dtype == d_l.dtype
    assert a_l.shape == d_l.shape
    _assert_equivalent_labeling(a_l, d_l.compute())


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
    d_cm = dask_image.ndmeasure.labeled_comprehension(
        d, d_lbls, ind, func, np.float64, default, pass_positions
    )

    assert a_cm.dtype == d_cm.dtype
    assert a_cm.shape == d_cm.shape
    assert np.allclose(np.array(a_cm), np.array(d_cm), equal_nan=True)


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
def test_labeled_comprehension_struct(shape, chunks, ind):
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

    dtype = np.dtype([("val", np.float64), ("pos", np.int)])
    default = np.array((np.nan, -1), dtype=dtype)

    def func_max(val):
        return np.max(val)

    def func_argmax(val, pos):
        return pos[np.argmax(val)]

    def func_max_argmax(val, pos):
        result = np.empty((), dtype=dtype)

        i = np.argmax(val)

        result["val"] = val[i]
        result["pos"] = pos[i]

        return result[()]

    a_max = spnd.labeled_comprehension(
        a, lbls, ind, func_max, dtype["val"], default["val"], False
    )
    a_argmax = spnd.labeled_comprehension(
        a, lbls, ind, func_argmax, dtype["pos"], default["pos"], True
    )

    d_max_argmax = dask_image.ndmeasure.labeled_comprehension(
        d, d_lbls, ind, func_max_argmax, dtype, default, True
    )
    d_max = d_max_argmax["val"]
    d_argmax = d_max_argmax["pos"]

    assert dtype == d_max_argmax.dtype

    for e_a_r, e_d_r in zip([a_max, a_argmax], [d_max, d_argmax]):
        assert e_a_r.dtype == e_d_r.dtype
        assert e_a_r.shape == e_d_r.shape
        assert np.allclose(np.array(e_a_r), np.array(e_d_r), equal_nan=True)


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
def test_labeled_comprehension_object(shape, chunks, ind):
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

    dtype = np.dtype(object)

    default = None

    def func_min_max(val):
        return np.array([np.min(val), np.max(val)])

    a_r = spnd.labeled_comprehension(
        a, lbls, ind, func_min_max, dtype, default, False
    )

    d_r = dask_image.ndmeasure.labeled_comprehension(
        d, d_lbls, ind, func_min_max, dtype, default, False
    )

    if ind is None or np.isscalar(ind):
        if a_r is None:
            assert d_r.compute() is None
        else:
            np.allclose(a_r, d_r.compute(), equal_nan=True)
    else:
        assert a_r.dtype == d_r.dtype
        assert a_r.shape == d_r.shape
        for i in it.product(*[range(_) for _ in a_r.shape]):
            if a_r[i] is None:
                assert d_r[i].compute() is None
            else:
                assert np.allclose(a_r[i], d_r[i].compute(), equal_nan=True)
