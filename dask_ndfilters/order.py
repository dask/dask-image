# -*- coding: utf-8 -*-


import inspect
import numbers
import re

import numpy
import scipy.ndimage.filters


def _get_footprint(ndim, size=None, footprint=None):
    # Verify that we only got size or footprint.
    if size is None and footprint is None:
        raise RuntimeError("Must provide either size or footprint.")
    if size is not None and footprint is not None:
        raise RuntimeError("Provide either size or footprint, but not both.")

    # Get a footprint based on the size.
    if size is not None:
        if isinstance(size, numbers.Integral):
            size = ndim * (size,)
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


def _get_origin(footprint, origin=0):
    size = numpy.array(footprint.shape)
    ndim = footprint.ndim

    if isinstance(origin, numbers.Real):
        origin = ndim * (origin,)

    origin = numpy.array(origin)
    origin = numpy.fix(origin).astype(int)

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

    return origin


def _get_depth_boundary(footprint, origin):
    origin = _get_origin(footprint, origin)

    size = numpy.array(footprint.shape)
    half_size = size // 2
    depth = half_size + abs(origin)
    depth = tuple(depth)

    # Workaround for a bug in Dask with 0 depth.
    #
    # ref: https://github.com/dask/dask/issues/2258
    #
    boundary = dict()
    for i in range(len(depth)):
        d = depth[i]
        if d == 0:
            boundary[i] = None
        else:
            boundary[i] = "none"

    return depth, boundary


def _get_normed_args(ndim, size=None, footprint=None, origin=0):
    footprint = _get_footprint(ndim, size, footprint)
    origin = _get_origin(footprint, origin)
    depth, boundary = _get_depth_boundary(footprint, origin)

    return footprint, origin, depth, boundary


def _get_docstring(func):
    # Drop the output parameter from the docstring.
    split_doc_params = lambda s: \
        re.subn("(    [A-Za-z]+ : )", "\0\\1", s)[0].split("\0")
    drop_doc_param = lambda s: not s.startswith("    output : ")
    cleaned_docstring = "".join([
        l for l in split_doc_params(func.__doc__) if drop_doc_param(l)
    ])

    docstring = """
    Wrapped copy of "{mod_name}.{func_name}"


    Excludes the output parameter as it would not work Dask arrays.


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


def _ordering_filter_wrapper(func):
    @_update_wrapper(func)
    def _wrapped_ordering_filter(input,
                                 size=None,
                                 footprint=None,
                                 mode='reflect',
                                 cval=0.0,
                                 origin=0):
        footprint, origin, depth, boundary = _get_normed_args(input.ndim,
                                                              size,
                                                              footprint,
                                                              origin)

        result = input.map_overlap(
            func,
            depth=depth,
            boundary=boundary,
            dtype=input.dtype,
            name=func.__name__,
            footprint=footprint,
            mode=mode,
            cval=cval,
            origin=origin
        )

        return result

    return _wrapped_ordering_filter


minimum_filter = _ordering_filter_wrapper(scipy.ndimage.filters.minimum_filter)
median_filter = _ordering_filter_wrapper(scipy.ndimage.filters.median_filter)
maximum_filter = _ordering_filter_wrapper(scipy.ndimage.filters.maximum_filter)


@_update_wrapper(scipy.ndimage.filters.rank_filter)
def rank_filter(input,
                rank,
                size=None,
                footprint=None,
                mode='reflect',
                cval=0.0,
                origin=0):
    footprint, origin, depth, boundary = _get_normed_args(input.ndim,
                                                          size,
                                                          footprint,
                                                          origin)

    result = input.map_overlap(
        scipy.ndimage.filters.rank_filter,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        name=scipy.ndimage.filters.rank_filter.__name__,
        rank=rank,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result


@_update_wrapper(scipy.ndimage.filters.percentile_filter)
def percentile_filter(input,
                      percentile,
                      size=None,
                      footprint=None,
                      mode='reflect',
                      cval=0.0,
                      origin=0):
    footprint, origin, depth, boundary = _get_normed_args(input.ndim,
                                                          size,
                                                          footprint,
                                                          origin)

    result = input.map_overlap(
        scipy.ndimage.filters.percentile_filter,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        name=scipy.ndimage.filters.percentile_filter.__name__,
        percentile=percentile,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
