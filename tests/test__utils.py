#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import inspect

import pytest

import numpy
import dask.array

from dask_ndmorph import _utils


def test__get_docstring():
    f = lambda : 0

    result = _utils._get_docstring(f)

    expected = """
    Wrapped copy of "{mod_name}.{func_name}"


    Excludes the output parameter as it would not work with Dask arrays.


    Original docstring:

    {doc}
    """.format(
        mod_name=inspect.getmodule(f).__name__,
        func_name=f.__name__,
        doc="",
    )

    assert result == expected


def test__update_wrapper():
    f = lambda : 0

    @_utils._update_wrapper(f)
    def g():
        return f()


    assert f.__name__ == g.__name__

    expected = """
    Wrapped copy of "{mod_name}.{func_name}"


    Excludes the output parameter as it would not work with Dask arrays.


    Original docstring:

    {doc}
    """.format(
        mod_name=inspect.getmodule(g).__name__,
        func_name=g.__name__,
        doc="",
    )

    assert g.__doc__ == expected


@pytest.mark.parametrize(
    "err_type, ndim, depth, boundary",
    [
        (TypeError, lambda : 0, 1, None),
        (TypeError, 1.0, 1, None),
        (ValueError, -1, 1, None),
        (TypeError, 1, lambda : 0, None),
        (TypeError, 1, 1.0, None),
        (ValueError, 1, -1, None),
        (ValueError, 1, (1, 1), None),
        (ValueError, 1, {0: 1, 1: 1}, None),
        (TypeError, 1, {1}, None),
        (TypeError, 1, 1, 1),
        (ValueError, 1, 1, (None, None)),
        (ValueError, 1, 1, {0: None, 1: None}),
        (TypeError, 1, 1, (1,)),
        (TypeError, 1, 1, {1}),
    ]
)
def test_errs__get_depth_boundary(err_type, ndim, depth, boundary):
    with pytest.raises(err_type):
        _utils._get_depth_boundary(ndim, depth, boundary)


@pytest.mark.parametrize(
    "err_type, ndim, size",
    [
        (TypeError, 1.0, 1),
        (RuntimeError, 1, [[1]]),
        (TypeError, 1, 1.0),
        (TypeError, 1, [1.0]),
        (RuntimeError, 1, [1, 1]),
    ]
)
def test_errs__get_size(err_type, ndim, size):
    with pytest.raises(err_type):
        _utils._get_size(ndim, size)


@pytest.mark.parametrize(
    "err_type, size, origin",
    [
        (TypeError, [1], 1.0),
        (TypeError, [1], [1.0]),
        (RuntimeError, [1], [[1]]),
        (RuntimeError, [1], [1, 1]),
        (ValueError, [1], [2]),
    ]
)
def test_errs__get_origin(err_type, size, origin):
    with pytest.raises(err_type):
        _utils._get_origin(size, origin)


@pytest.mark.parametrize(
    "err_type, ndim, size, footprint",
    [
        (RuntimeError, 1, None, None),
        (RuntimeError, 1, [2], numpy.ones((2,), dtype=bool)),
        (RuntimeError, 1, None, numpy.ones((1, 2), dtype=bool)),
        (RuntimeError, 1, None, numpy.ones([0], dtype=bool)),
    ]
)
def test_errs__get_footprint(err_type, ndim, size, footprint):
    with pytest.raises(err_type):
        _utils._get_footprint(ndim, size=size, footprint=footprint)


@pytest.mark.parametrize(
    "err_type, input, structure",
    [
        (
            RuntimeError,
            dask.array.ones([1, 2], dtype=bool, chunks=(1, 2,)),
            dask.array.arange(2, dtype=bool, chunks=(2,))
        ),
        (
            TypeError,
            dask.array.arange(2, dtype=bool, chunks=(2,)),
            2.0
        ),
    ]
)
def test_errs__get_structure(err_type, input, structure):
    with pytest.raises(err_type):
        _utils._get_structure(input, structure)


@pytest.mark.parametrize(
    "err_type, iterations",
    [
        (TypeError, 0.0),
        (NotImplementedError, 0),
    ]
)
def test_errs__get_iterations(err_type, iterations):
    with pytest.raises(err_type):
        _utils._get_iterations(iterations)


@pytest.mark.parametrize(
    "err_type, input, mask",
    [
        (
            RuntimeError,
            dask.array.arange(2, dtype=bool, chunks=(2,)),
            dask.array.arange(1, dtype=bool, chunks=(2,))
        ),
        (
            TypeError,
            dask.array.arange(2, dtype=bool, chunks=(2,)),
            2.0
        ),
    ]
)
def test_errs__get_mask(err_type, input, mask):
    with pytest.raises(err_type):
        _utils._get_mask(input, mask)


@pytest.mark.parametrize(
    "err_type, border_value",
    [
        (TypeError, 0.0),
        (TypeError, 1.0),
    ]
)
def test_errs__get_border_value(err_type, border_value):
    with pytest.raises(err_type):
        _utils._get_border_value(border_value)


@pytest.mark.parametrize(
    "err_type, brute_force",
    [
        (NotImplementedError, True),
        (TypeError, 1),
    ]
)
def test_errs__get_brute_force(err_type, brute_force):
    with pytest.raises(err_type):
        _utils._get_brute_force(brute_force)


@pytest.mark.parametrize(
    "expected, ndim, depth, boundary",
    [
        (({0: 0}, {0: "reflect"}), 1, 0, "none"),
        (({0: 0}, {0: "reflect"}), 1, 0, "reflect"),
        (({0: 0}, {0: "periodic"}), 1, 0, "periodic"),
        (({0: 1}, {0: "none"}), 1, 1, "none"),
    ]
)
def test__get_depth_boundary(expected, ndim, depth, boundary):
    assert expected == _utils._get_depth_boundary(ndim, depth, boundary)


@pytest.mark.parametrize(
    "expected, ndim, size",
    [
        ((1,), 1, 1),
        ((3, 3), 2, 3),
        ((2, 4), 2, (2, 4)),
    ]
)
def test__get_size(expected, ndim, size):
    assert expected == _utils._get_size(ndim, size)


@pytest.mark.parametrize(
    "expected, size, origin",
    [
        ((0,), (1,), 0),
        ((1,), (3,), 1),
        ((1, 2), (3, 5), (1, 2)),
    ]
)
def test__get_origin(expected, size, origin):
    assert expected == _utils._get_origin(size, origin)


@pytest.mark.parametrize(
    "expected, size, origin",
    [
        ((0,), (1,), 0),
        ((1,), (3,), 0),
        ((2,), (3,), 1),
        ((2, 4), (3, 5), (1, 2)),
    ]
)
def test__get_depth(expected, size, origin):
    assert expected == _utils._get_depth(size, origin)


@pytest.mark.parametrize(
    "expected, ndim, size, footprint",
    [
        (numpy.ones((2,), dtype=bool), 1, 2, None),
        (numpy.ones((2,), dtype=bool), 1, None, numpy.ones((2,), dtype=bool)),
    ]
)
def test__get_footprint(expected, ndim, size, footprint):
    assert (expected == _utils._get_footprint(ndim, size, footprint)).all()


@pytest.mark.parametrize(
    "expected, input, structure",
    [
        (
            numpy.array([0, 1, 0], dtype=bool),
            (dask.array.arange(10, chunks=(10,)) % 2).astype(bool),
            None
        ),
        (
            numpy.array([1, 1, 1], dtype=bool),
            (dask.array.arange(10, chunks=(10,)) % 2).astype(bool),
            numpy.array([1, 1, 1], dtype=int)
        ),
        (
            numpy.array([1, 1, 1], dtype=bool),
            (dask.array.arange(10, chunks=(10,)) % 2).astype(bool),
            numpy.array([1, 1, 1], dtype=bool)
        ),
    ]
)
def test__get_structure(expected, input, structure):
    result = _utils._get_structure(input, structure)

    assert expected.dtype.type == result.dtype.type
    assert numpy.array((expected == result).all())[()]


@pytest.mark.parametrize(
    "expected, iterations",
    [
        (1, 1),
        (4, 4),
    ]
)
def test__get_iterations(expected, iterations):
    assert expected == _utils._get_iterations(iterations)


@pytest.mark.parametrize(
    "expected, a",
    [
        (numpy.bool8, False),
        (numpy.int64, 2),
        (numpy.float64, 3.1),
        (numpy.complex128, 1 + 2j),
        (numpy.int16, numpy.int16(6)),
        (numpy.uint32, numpy.arange(3, dtype=numpy.uint32)),
    ]
)
def test__get_dtype(expected, a):
    assert expected == _utils._get_dtype(a)


@pytest.mark.parametrize(
    "expected, input, mask",
    [
        (True, dask.array.arange(2, dtype=bool, chunks=(2,)), None),
        (True, dask.array.arange(2, dtype=bool, chunks=(2,)), True),
        (False, dask.array.arange(2, dtype=bool, chunks=(2,)), False),
        (
            True,
            dask.array.arange(2, dtype=bool, chunks=(2,)),
            numpy.bool8(True)
        ),
        (
            False,
            dask.array.arange(2, dtype=bool, chunks=(2,)),
            numpy.bool8(False)
        ),
        (
            numpy.arange(2, dtype=bool),
            dask.array.arange(2, dtype=bool, chunks=(2,)),
            numpy.arange(2, dtype=bool)
        ),
        (
            dask.array.arange(2, dtype=bool, chunks=(2,)),
            dask.array.arange(2, dtype=bool, chunks=(2,)),
            dask.array.arange(2, dtype=int, chunks=(2,))
        ),
    ]
)
def test__get_mask(expected, input, mask):
    result = _utils._get_mask(input, mask)

    assert type(expected) == type(result)

    if isinstance(expected, (numpy.ndarray, dask.array.Array)):
        assert numpy.array((expected == result).all())[()]
    else:
        assert expected == result


@pytest.mark.parametrize(
    "expected, border_value",
    [
        (False, False),
        (True, True),
        (False, 0),
        (True, 1),
        (True, 5),
        (True, -2),
    ]
)
def test__get_border_value(expected, border_value):
    assert expected == _utils._get_border_value(border_value)


@pytest.mark.parametrize(
    "expected, brute_force",
    [
        (False, False),
    ]
)
def test__get_brute_force(expected, brute_force):
    assert expected == _utils._get_brute_force(brute_force)
