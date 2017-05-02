# -*- coding: utf-8 -*-


import numbers

import numpy
import scipy.ndimage.filters

import dask_ndfilters._utils as _utils


def _get_footprint(ndim, size=None, footprint=None):
    # Verify that we only got size or footprint.
    if size is None and footprint is None:
        raise RuntimeError("Must provide either size or footprint.")
    if size is not None and footprint is not None:
        raise RuntimeError("Provide either size or footprint, but not both.")

    # Get a footprint based on the size.
    if size is not None:
        if isinstance(size, numbers.Number):
            size = ndim * (size,)
        size = numpy.array(size)

        if size.ndim != 1:
            raise RuntimeError("The size must have only one dimension.")
        if not issubclass(size.dtype.type, numbers.Integral):
            raise TypeError("The size must be of integral type.")

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

    return origin


def _get_depth_boundary(footprint, origin):
    origin = _get_origin(footprint, origin)

    size = numpy.array(footprint.shape)
    half_size = size // 2
    depth = half_size + abs(origin)
    depth = tuple(depth)

    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    return depth, boundary


def _get_normed_args(ndim, size=None, footprint=None, origin=0):
    footprint = _get_footprint(ndim, size, footprint)
    origin = _get_origin(footprint, origin)
    depth, boundary = _get_depth_boundary(footprint, origin)

    return footprint, origin, depth, boundary


def _ordering_filter_wrapper(func):
    @_utils._update_wrapper(func)
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


@_utils._update_wrapper(scipy.ndimage.filters.rank_filter)
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


@_utils._update_wrapper(scipy.ndimage.filters.percentile_filter)
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
