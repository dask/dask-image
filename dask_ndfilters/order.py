# -*- coding: utf-8 -*-


import numbers

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


def median_filter(input,
                  size=None,
                  footprint=None,
                  mode='reflect',
                  cval=0.0,
                  origin=0):
    footprint = _get_footprint(input.ndim, size, footprint)
    origin = _get_origin(footprint, origin)
    depth, boundary = _get_depth_boundary(footprint, origin)

    result = input.map_overlap(
        scipy.ndimage.filters.median_filter,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        name="median_filter",
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
