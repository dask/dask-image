#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
import warnings as wrn

import pytest
import numpy as np
import scipy
import scipy.ndimage

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
        np.bool_,
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
    sp_func = getattr(scipy.ndimage, funcname)
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

    a_r = scipy.ndimage.extrema(a, lbls, ind)
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

    a_r = scipy.ndimage.histogram(a, min, max, bins, lbls, ind)
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
    assert len(np.unique(unique_matching[:, 0])) == \
           len(np.unique(unique_matching[:, 1]))


def assert_sequential_labeling(labels):
    """Assert that the labels are sequential starting at 1.

    I.e. the labels are in {0, 1, 2, ..., N} where 0 is background.
    """
    u = np.unique(labels)
    assert len(u) == 1 + u.max()


@pytest.mark.parametrize(
    "seed, prob, shape, chunks, connectivity, sequential", [
        (42, 0.4, (15, 16), (15, 16), 1, False),
        (42, 0.4, (15, 16), (15, 16), 1, True),
        (42, 0.4, (15, 16), (4, 5), 1, False),
        (42, 0.4, (15, 16), (4, 5), 1, True),
        (42, 0.4, (15, 16), (4, 5), 2, False),
        (42, 0.4, (15, 16), (4, 5), None, False),
        (42, 0.4, (15, 16), (8, 5), 1, False),
        (42, 0.4, (15, 16), (8, 5), 2, False),
        (42, 0.3, (10, 8, 6), (5, 4, 3), 1, False),
        (42, 0.3, (10, 8, 6), (5, 4, 3), 1, True),
        (42, 0.3, (10, 8, 6), (5, 4, 3), 2, False),
        (42, 0.3, (10, 8, 6), (5, 4, 3), 3, False),
    ]
)
def test_label(seed, prob, shape, chunks, connectivity, sequential):
    np.random.seed(seed)

    a = np.random.random(shape) < prob
    d = da.from_array(a, chunks=chunks)

    if connectivity is None:
        s = None
    else:
        s = scipy.ndimage.generate_binary_structure(a.ndim, connectivity)

    a_l, a_nl = scipy.ndimage.label(a, s)
    d_l, d_nl = dask_image.ndmeasure.label(
        d, s, produce_sequential_labels=sequential)

    assert a_nl == d_nl.compute()

    assert a_l.dtype == d_l.dtype
    assert a_l.shape == d_l.shape
    _assert_equivalent_labeling(a_l, d_l.compute())

    if sequential:
        assert_sequential_labeling(d_l.compute())


a = np.array(
    [
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    ]
)


@pytest.mark.parametrize(
    "a, a_res, wrap_axes, connectivity, chunks",
    [
        pytest.param(
            a,
            np.array(
                [
                    [0, 0, 1, 0, 0, 3, 3, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 0, 0, 0, 2, 0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 2, 0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 4, 0, 0, 5, 5, 0, 0, 0],
                ]
            ),
            (1,),
            2,
            (5, 5),
            id="2d, wrapping 1st axis.",
        ),
        pytest.param(
            a,
            np.array(
                [
                    [0, 0, 1, 0, 0, 3, 3, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 4, 4, 4, 4],
                    [1, 1, 0, 0, 0, 0, 4, 4, 4, 4],
                    [1, 0, 0, 0, 2, 0, 4, 4, 4, 0],
                    [0, 1, 0, 0, 2, 0, 4, 4, 4, 0],
                    [0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 3, 3, 0, 0, 0],
                ]
            ),
            (0,),
            2,
            (5, 5),
            id="2d, wrapping 0th axes.",
        ),
        pytest.param(
            a,
            np.array(
                [
                    [0, 0, 1, 0, 0, 3, 3, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 0, 0, 0, 2, 0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 2, 0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 3, 3, 0, 0, 0],
                ]
            ),
            (0, 1),
            2,
            (5, 5),
            id="2d, wrapping both axes",
        ),
        pytest.param(
            np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]),
            (0, 1),
            2,
            "auto",
            id="2d, full wrap, high connectivity (corners).",
        ),
        pytest.param(
            np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]),
            # Corners should not be connected for lower connectivity.
            np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2]]),
            (0, 1),
            1,
            "auto",
            id="2d, full wrap, low connectivity (no corners).",
        ),
        # 3d
        pytest.param(
            np.array(
                [
                    [[0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]],
                ]
            ),
            np.array(
                [
                    [[0, 0, 0, 0, 0], [1, 0, 0, 0, 2], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [3, 0, 0, 0, 4], [3, 0, 0, 0, 4]],
                ]
            ),
            None,
            3,
            "auto",
            id="3d no wrap",
        ),
        pytest.param(
            np.array(
                [
                    [[0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]],
                ]
            ),
            np.array(
                [
                    [[0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [2, 0, 0, 0, 2], [2, 0, 0, 0, 2]],
                ]
            ),
            (2,),
            3,
            "auto",
            id="3d wrap 2nd axis",
        ),
        pytest.param(
            np.array(
                [
                    [
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 2],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [3, 0, 0, 0, 3],
                    ],
                ]
            ),
            (1, 2),
            3,
            "auto",
            id="3d, wrap 1st and 2nd axis, with corners",
        ),
        pytest.param(
            np.array(
                [
                    [
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 2],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1],
                    ],
                ]
            ),
            (1, 2),
            3,
            "auto",
            id="3d, with corners, connection through adjacent timesteps.",
        ),
    ],
)
def test_label_wrap(a, a_res, wrap_axes, connectivity, chunks):
    d = da.from_array(a, chunks=chunks)

    s = scipy.ndimage.generate_binary_structure(a.ndim, connectivity)

    d_l, _ = dask_image.ndmeasure.label(d, s, wrap_axes=wrap_axes)

    _assert_equivalent_labeling(a_res, d_l.compute())


@pytest.mark.parametrize(
    "ndim", (2, 3, 4, 5)
)
def test_label_full_struct_element(ndim):

    full_s = scipy.ndimage.generate_binary_structure(ndim, ndim)
    orth_s = scipy.ndimage.generate_binary_structure(ndim, ndim - 1)

    # create a mask that represents a single connected component
    # under the full (highest rank) structuring element
    # but several connected components under the orthogonal
    # structuring element
    mask = full_s ^ orth_s
    mask[tuple([1] * ndim)] = True

    # create dask array with chunk boundary
    # that passes through the mask
    mask_da = da.from_array(mask, chunks=[2] * ndim)

    labels_ndi, N_ndi = scipy.ndimage.label(mask, structure=full_s)
    labels_di_da, N_di_da = dask_image.ndmeasure.label(
        mask_da, structure=full_s)

    assert N_ndi == N_di_da.compute()

    _assert_equivalent_labeling(
        labels_ndi,
        labels_di_da.compute())


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

    a_cm = scipy.ndimage.labeled_comprehension(
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

    dtype = np.dtype([("val", np.float64), ("pos", int)])
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

    a_max = scipy.ndimage.labeled_comprehension(
        a, lbls, ind, func_max, dtype["val"], default["val"], False
    )
    a_argmax = scipy.ndimage.labeled_comprehension(
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

    a_r = scipy.ndimage.labeled_comprehension(
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


def test_make_labels_unique():

    np.random.seed(42)
    labels = da.random.randint(0, 1, size=(6, 6), chunks=(3, 3))
    unique_labels = dask_image.ndmeasure._label._make_labels_unique(labels)

    assert unique_labels.shape == labels.shape
    assert len(np.unique(unique_labels.compute())) >= \
           len(np.unique(labels.compute()))


@pytest.mark.parametrize(
        "ndim, overlap_depth, produce_sequential_labels",
        [
            (1, 0, False),
            (1, 1, True),
            (2, 0, False),
            (2, 1, True),
            (3, 0, False),
            (3, 1, True),
        ]
)
def test_merge_labels_across_chunk_boundaries(
    ndim, overlap_depth, produce_sequential_labels
):

    # create a segmentation ground truth
    # e.g. in 2d:
    # 0 0 0 0 0 0
    # 0 1 1 1 1 0
    # 0 1 1 1 1 0
    # 0 2 2 2 2 0
    # 0 2 2 2 2 0
    # 0 0 0 0 0 0

    im = np.zeros((6, ) * ndim, dtype=np.uint16)
    im[tuple([slice(1, -1)] * ndim)] = 1

    # along dimension 0 introduce two "instances"
    # with labels 1 and 2
    # along the remaining dimensions the labels don't change

    dim = da.from_array(im, chunks=(3, ) * ndim)

    def label_block_dim0(x, block_id=None):
        return x * (block_id[0] + 1)

    dim = dim.map_blocks(
        label_block_dim0,
        dtype=dim.dtype,
    )

    # simulate a segmentation method which works
    # perfectly within each chunk, but cannot
    # guarantee object identities across chunks.
    # the segmentation is applied with different
    # overlap depths

    # input as seen by the segmentation method
    dim_chunkwise_view = da.overlap.overlap(
        dim,
        depth={i: overlap_depth for i in range(ndim)},
        boundary='none',
    )

    # simulate instance label ids that vary over chunks
    random_factor_per_chunk = da.random.randint(
        0,
        1000,
        size=dim_chunkwise_view.numblocks,
        chunks=(1, ) * ndim,
    )

    dim_chunkwise_segmentation = da.map_blocks(
            lambda x, y: x * y,
            dim_chunkwise_view,
            random_factor_per_chunk,
            chunks=dim_chunkwise_view.chunks,
            dtype=dim.dtype,
        )

    # merge labels across chunk boundaries
    dim_segmentation_merged = dask_image.ndmeasure\
        .merge_labels_across_chunk_boundaries(
            dim_chunkwise_segmentation,
            overlap_depth=overlap_depth,
            iou_threshold=0,
        )['labels']

    dim_segmentation_merged_c = \
        dim_segmentation_merged.compute(scheduler='single-threaded')

    if not overlap_depth:
        # merging labels with zero overlap should merge all labels
        # into a single connected component
        _assert_equivalent_labeling(
            dim_segmentation_merged_c,
            scipy.ndimage.label(dim > 0)[0]
        )
    else:
        # merging labels with overlap recovers the ground truth
        _assert_equivalent_labeling(
            dim.compute(),
            dim_segmentation_merged_c
        )

    if produce_sequential_labels:
        assert_sequential_labeling(dim_segmentation_merged_c)
