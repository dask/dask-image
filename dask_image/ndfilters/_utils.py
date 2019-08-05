# -*- coding: utf-8 -*-

from __future__ import division

import collections
import inspect
import numbers
import re

import numpy


def _get_docstring(func):
    # Drop the output parameter from the docstring.
    split_doc_params = lambda s: re.subn(                         # noqa: E731
        "(    [A-Za-z]+ : )", "\0\\1", s)[0].split("\0")
    drop_doc_param = lambda s: not s.startswith("    output : ")  # noqa: E731
    func_doc = "" if func.__doc__ is None else func.__doc__
    cleaned_docstring = "".join([
        l for l in split_doc_params(func_doc) if drop_doc_param(l)
    ])
    cleaned_docstring = cleaned_docstring.replace('input', 'image')
    cleaned_docstring = cleaned_docstring.replace('labels', 'label_image')
    cleaned_docstring = cleaned_docstring.split('Examples')[0].strip()

    docstring = """
    Wrapped copy of "{mod_name}.{func_name}"


    Excludes the output parameter as it would not work with Dask arrays.


    Original docstring:

    {doc}
    """.format(
        mod_name=inspect.getmodule(func).__name__,
        func_name=func.__name__,
        doc=cleaned_docstring,
    )

    return docstring


def _update_wrapper(func):
    def _updater(wrapper):
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = _get_docstring(func)
        return wrapper

    return _updater


def _get_depth_boundary(ndim, depth, boundary=None):
    strlike = (bytes, str)

    if not isinstance(ndim, numbers.Integral):
        raise TypeError("Expected integer value for `ndim`.")
    if ndim <= 0:
        raise ValueError("Expected positive value for `ndim`.")

    if isinstance(depth, numbers.Number):
        depth = ndim * (depth,)
    if not isinstance(depth, collections.Sized):
        raise TypeError("Unexpected type for `depth`.")
    if len(depth) != ndim:
        raise ValueError("Expected `depth` to have a length equal to `ndim`.")
    if isinstance(depth, collections.Sequence):
        depth = dict(zip(range(ndim), depth))
    if not isinstance(depth, collections.Mapping):
        raise TypeError("Unexpected type for `depth`.")

    if not all(map(lambda d: isinstance(d, numbers.Integral), depth.values())):
        raise TypeError("Expected integer values for `depth`.")
    if not all(map(lambda d: d >= 0, depth.values())):
        raise ValueError("Expected positive semidefinite values for `depth`.")

    depth = dict([(a, int(d)) for a, d in depth.items()])

    if (boundary is None) or isinstance(boundary, strlike):
        boundary = ndim * (boundary,)
    if not isinstance(boundary, collections.Sized):
        raise TypeError("Unexpected type for `boundary`.")
    if len(boundary) != ndim:
        raise ValueError(
            "Expected `boundary` to have a length equal to `ndim`."
        )
    if isinstance(boundary, collections.Sequence):
        boundary = dict(zip(range(ndim), boundary))
    if not isinstance(boundary, collections.Mapping):
        raise TypeError("Unexpected type for `boundary`.")

    type_check = lambda b: (b is None) or isinstance(b, strlike)  # noqa: E731
    if not all(map(type_check, boundary.values())):
        raise TypeError("Expected string-like values for `boundary`.")

    return depth, boundary


def _get_size(ndim, size):
    if not isinstance(ndim, numbers.Integral):
        raise TypeError("The ndim must be of integral type.")

    if isinstance(size, numbers.Number):
        size = ndim * (size,)
    size = numpy.array(size)

    if size.ndim != 1:
        raise RuntimeError("The size must have only one dimension.")
    if len(size) != ndim:
        raise RuntimeError(
            "The size must have a length equal to the number of dimensions."
        )
    if not issubclass(size.dtype.type, numbers.Integral):
        raise TypeError("The size must be of integral type.")

    size = tuple(size)

    return size


def _get_origin(size, origin=0):
    size = numpy.array(size)
    ndim = len(size)

    if isinstance(origin, numbers.Number):
        origin = ndim * (origin,)

    origin = numpy.array(origin)

    if not issubclass(origin.dtype.type, numbers.Integral):
        raise TypeError("The origin must be of integral type.")

    # Validate dimensions.
    if origin.ndim != 1:
        raise RuntimeError("The origin must have only one dimension.")
    if len(origin) != ndim:
        raise RuntimeError(
            "The origin must have the same length as the number of dimensions"
            " as the array being filtered."
        )

    # Validate origin is bounded.
    if not (origin < ((size + 1) // 2)).all():
        raise ValueError("The origin must be within the footprint.")

    origin = tuple(origin)

    return origin


def _get_depth(size, origin=0):
    origin = numpy.array(_get_origin(size, origin))
    size = numpy.array(size)

    half_size = size // 2
    depth = half_size + abs(origin)

    depth = tuple(depth)

    return depth


def _get_footprint(ndim, size=None, footprint=None):
    # Verify that we only got size or footprint.
    if size is None and footprint is None:
        raise RuntimeError("Must provide either size or footprint.")
    if size is not None and footprint is not None:
        raise RuntimeError("Provide either size or footprint, but not both.")

    # Get a footprint based on the size.
    if size is not None:
        size = _get_size(ndim, size)
        footprint = numpy.ones(size, dtype=bool)

    # Validate the footprint.
    if footprint.ndim != ndim:
        raise RuntimeError(
            "The footprint must have the same number of dimensions as"
            " the array being filtered."
        )
    if footprint.size == 0:
        raise RuntimeError("The footprint must have only non-zero dimensions.")

    # Convert to Boolean.
    footprint = (footprint != 0)

    return footprint
