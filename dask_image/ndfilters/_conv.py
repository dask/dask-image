# -*- coding: utf-8 -*-


import scipy.ndimage.filters

from . import _utils


@_utils._update_wrapper(scipy.ndimage.filters.convolve)
def convolve(input,
             weights,
             mode='reflect',
             cval=0.0,
             origin=0):
    origin = _utils._get_origin(weights.shape, origin)
    depth = _utils._get_depth(weights.shape, origin)
    depth, boundary = _utils._get_depth_boundary(input.ndim, depth, "none")

    result = input.map_overlap(
        scipy.ndimage.filters.convolve,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        weights=weights,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result


@_utils._update_wrapper(scipy.ndimage.filters.correlate)
def correlate(input,
              weights,
              mode='reflect',
              cval=0.0,
              origin=0):
    origin = _utils._get_origin(weights.shape, origin)
    depth = _utils._get_depth(weights.shape, origin)
    depth, boundary = _utils._get_depth_boundary(input.ndim, depth, "none")

    result = input.map_overlap(
        scipy.ndimage.filters.correlate,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        weights=weights,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
