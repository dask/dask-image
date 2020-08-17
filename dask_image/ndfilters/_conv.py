# -*- coding: utf-8 -*-
import scipy.ndimage.filters

from . import _utils
from ..dispatch._utils import check_arraytypes_compatible
from ..dispatch._dispatch_ndfilters import (
    dispatch_convolve,
    dispatch_correlate)

__all__ = [
    "convolve",
    "correlate",
]


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

    result = image.map_overlap(
        dispatch_convolve(image),
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
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
    check_arraytypes_compatible(image, weights)

    origin = _utils._get_origin(weights.shape, origin)
    depth = _utils._get_depth(weights.shape, origin)
    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")

    result = image.map_overlap(
        dispatch_correlate(image),
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        meta=image._meta,
        weights=weights,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
