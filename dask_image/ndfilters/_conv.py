# -*- coding: utf-8 -*-
import dask.array as da
import scipy.ndimage.filters

from . import _utils
from ..utils._dispatcher import dispatch_convolve, check_arraytypes_compatible


@_utils._update_wrapper(scipy.ndimage.filters.convolve)
def convolve(image,
             weights,
             mode='reflect',
             cval=0.0,
             origin=0):
    check_arraytypes_compatible(image, weights)

    origin = _utils._get_origin(weights.shape, origin)
    depth = _utils._get_depth(weights.shape, origin)
    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")

    result = da.map_overlap(
        dispatch_convolve,
        image,
        depth=depth,
        boundary=boundary,
        meta=image._meta,
        weights=weights,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result


@_utils._update_wrapper(scipy.ndimage.filters.correlate)
def correlate(image,
              weights,
              mode='reflect',
              cval=0.0,
              origin=0):
    origin = _utils._get_origin(weights.shape, origin)
    depth = _utils._get_depth(weights.shape, origin)
    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")

    result = image.map_overlap(
        scipy.ndimage.filters.correlate,
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        weights=weights,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
