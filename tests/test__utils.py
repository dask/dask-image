#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import inspect

import pytest

from dask_ndfilters import _utils


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
