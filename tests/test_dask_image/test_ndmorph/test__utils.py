#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

import numpy as np
import dask.array as da

from dask_image.ndmorph import _utils


@pytest.mark.parametrize(
    "err_type, input, structure",
    [
        (
            RuntimeError,
            da.ones([1, 2], dtype=bool, chunks=(1, 2,)),
            da.arange(2, dtype=bool, chunks=(2,))
        ),
        (
            TypeError,
            da.arange(2, dtype=bool, chunks=(2,)),
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
            da.arange(2, dtype=bool, chunks=(2,)),
            da.arange(1, dtype=bool, chunks=(2,))
        ),
        (
            TypeError,
            da.arange(2, dtype=bool, chunks=(2,)),
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
    "expected, input, structure",
    [
        (
            np.array([1, 1, 1], dtype=bool),
            (da.arange(10, chunks=(10,)) % 2).astype(bool),
            None
        ),
        (
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
            (da.arange(100, chunks=10).reshape(10, 10) % 2).astype(bool),  # noqa: E501
            None
        ),
        (
            np.array([1, 1, 1], dtype=bool),
            (da.arange(10, chunks=(10,)) % 2).astype(bool),
            np.array([1, 1, 1], dtype=int)
        ),
        (
            np.array([1, 1, 1], dtype=bool),
            (da.arange(10, chunks=(10,)) % 2).astype(bool),
            np.array([1, 1, 1], dtype=bool)
        ),
    ]
)
def test__get_structure(expected, input, structure):
    result = _utils._get_structure(input, structure)

    assert expected.dtype.type == result.dtype.type
    assert np.array((expected == result).all())[()]


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
        (np.bool8, False),
        (np.int_, 2),
        (np.float64, 3.1),
        (np.complex128, 1 + 2j),
        (np.int16, np.int16(6)),
        (np.uint32, np.arange(3, dtype=np.uint32)),
    ]
)
def test__get_dtype(expected, a):
    assert expected == _utils._get_dtype(a)


@pytest.mark.parametrize(
    "expected, input, mask",
    [
        (True, da.arange(2, dtype=bool, chunks=(2,)), None),
        (True, da.arange(2, dtype=bool, chunks=(2,)), True),
        (False, da.arange(2, dtype=bool, chunks=(2,)), False),
        (
            True,
            da.arange(2, dtype=bool, chunks=(2,)),
            np.bool8(True)
        ),
        (
            False,
            da.arange(2, dtype=bool, chunks=(2,)),
            np.bool8(False)
        ),
        (
            np.arange(2, dtype=bool),
            da.arange(2, dtype=bool, chunks=(2,)),
            np.arange(2, dtype=bool)
        ),
        (
            da.arange(2, dtype=bool, chunks=(2,)),
            da.arange(2, dtype=bool, chunks=(2,)),
            da.arange(2, dtype=int, chunks=(2,))
        ),
    ]
)
def test__get_mask(expected, input, mask):
    result = _utils._get_mask(input, mask)

    assert type(expected) == type(result)

    if isinstance(expected, (np.ndarray, da.Array)):
        assert np.array((expected == result).all())[()]
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
