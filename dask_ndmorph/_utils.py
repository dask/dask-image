# -*- coding: utf-8 -*-


import collections
import inspect
import numbers
import re

import numpy
import scipy.ndimage

import dask.array


try:
    from itertools import imap, izip
except ImportError:
    imap, izip = map, zip

try:
    irange = xrange
except NameError:
    irange = range

try:
    unicode
except NameError:
    unicode = str

strlike = (bytes, unicode)


def _get_docstring(func):
    # Drop the output parameter from the docstring.
    split_doc_params = lambda s: \
        re.subn("(    [A-Za-z]+ : )", "\0\\1", s)[0].split("\0")
    drop_doc_param = lambda s: not s.startswith("    output : ")
    func_doc = "" if func.__doc__ is None else func.__doc__
    cleaned_docstring = "".join([
        l for l in split_doc_params(func_doc) if drop_doc_param(l)
    ])

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
        depth = dict(izip(irange(ndim), depth))
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
        boundary = dict(izip(irange(ndim), boundary))
    if not isinstance(boundary, collections.Mapping):
        raise TypeError("Unexpected type for `boundary`.")

    type_check = lambda b: (b is None) or isinstance(b, strlike)
    if not all(map(type_check, boundary.values())):
        raise TypeError("Expected string-like values for `boundary`.")

    # Workaround for a bug in Dask with 0 depth.
    #
    # ref: https://github.com/dask/dask/issues/2258
    #
    for i in irange(ndim):
        if boundary[i] == "none" and depth[i] == 0:
            boundary[i] = "reflect"

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


def _get_structure(input, structure):
    # Create square connectivity as default
    if structure is None:
        structure = scipy.ndimage.generate_binary_structure(input.ndim, 1)
    elif isinstance(structure, (numpy.ndarray, dask.array.Array)):
        if structure.ndim != input.ndim:
            raise RuntimeError(
                "`structure` must have the same rank as `input`."
            )
        if not issubclass(structure.dtype.type, numpy.bool8):
            structure = (structure != 0)
    else:
        raise TypeError("`structure` must be an array.")

    return structure


def _get_iterations(iterations):
    if not isinstance(iterations, numbers.Integral):
        raise TypeError("`iterations` must be of integral type.")
    if iterations < 1:
        raise NotImplementedError(
            "`iterations` must be equal to 1 or greater not less."
        )

    return iterations


def _get_dtype(a):
    # Get the dtype of a value or an array.
    # Even handle non-NumPy types.
    return getattr(a, "dtype", numpy.dtype(type(a)))


def _get_mask(input, mask):
    if mask is None:
        mask = True

    mask_type = _get_dtype(mask).type
    if isinstance(mask, (numpy.ndarray, dask.array.Array)):
        if mask.shape != input.shape:
            raise RuntimeError("`mask` must have the same shape as `input`.")
        if not issubclass(mask_type, numpy.bool8):
            mask = (mask != 0)
    elif issubclass(mask_type, numpy.bool8):
        mask = bool(mask)
    else:
        raise TypeError("`mask` must be a Boolean or an array.")

    return mask


def _get_border_value(border_value):
    if not isinstance(border_value, numbers.Integral):
        raise TypeError("`border_value` must be of integral type.")

    border_value = (border_value != 0)

    return border_value


def _get_brute_force(brute_force):
    if brute_force is not False:
        if brute_force is True:
            raise NotImplementedError(
                "`brute_force` other than `False` is not yet supported."
            )
        else:
            raise TypeError(
                "`brute_force` must be `bool`."
            )

    return brute_force
