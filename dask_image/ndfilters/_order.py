# -*- coding: utf-8 -*-

import scipy.ndimage

from ..dispatch._dispatch_ndfilters import (dispatch_maximum_filter,
                                            dispatch_median_filter,
                                            dispatch_minimum_filter,
                                            dispatch_percentile_filter,
                                            dispatch_rank_filter)
from . import _utils

__all__ = [
    "minimum_filter",
    "median_filter",
    "maximum_filter",
    "rank_filter",
    "percentile_filter",
]


@_utils._update_wrapper(scipy.ndimage.minimum_filter)
def minimum_filter(image,
                   size=None,
                   footprint=None,
                   mode='reflect',
                   cval=0.0,
                   origin=0):
    footprint = _utils._get_footprint(image.ndim, size, footprint)
    origin = _utils._get_origin(footprint.shape, origin)
    depth = _utils._get_depth(footprint.shape, origin)
    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    result = image.map_overlap(
        dispatch_minimum_filter(image),
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        meta=image._meta,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result


@_utils._update_wrapper(scipy.ndimage.median_filter)
def median_filter(image,
                  size=None,
                  footprint=None,
                  mode='reflect',
                  cval=0.0,
                  origin=0):
    footprint = _utils._get_footprint(image.ndim, size, footprint)
    origin = _utils._get_origin(footprint.shape, origin)
    depth = _utils._get_depth(footprint.shape, origin)
    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    result = image.map_overlap(
        dispatch_median_filter(image),
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        meta=image._meta,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result


@_utils._update_wrapper(scipy.ndimage.maximum_filter)
def maximum_filter(image,
                   size=None,
                   footprint=None,
                   mode='reflect',
                   cval=0.0,
                   origin=0):
    footprint = _utils._get_footprint(image.ndim, size, footprint)
    origin = _utils._get_origin(footprint.shape, origin)
    depth = _utils._get_depth(footprint.shape, origin)
    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    result = image.map_overlap(
        dispatch_maximum_filter(image),
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        meta=image._meta,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result


@_utils._update_wrapper(scipy.ndimage.rank_filter)
def rank_filter(image,
                rank,
                size=None,
                footprint=None,
                mode='reflect',
                cval=0.0,
                origin=0):
    footprint = _utils._get_footprint(image.ndim, size, footprint)
    origin = _utils._get_origin(footprint.shape, origin)
    depth = _utils._get_depth(footprint.shape, origin)
    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    result = image.map_overlap(
        dispatch_rank_filter(image),
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        meta=image._meta,
        rank=rank,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result


@_utils._update_wrapper(scipy.ndimage.percentile_filter)
def percentile_filter(image,
                      percentile,
                      size=None,
                      footprint=None,
                      mode='reflect',
                      cval=0.0,
                      origin=0):
    footprint = _utils._get_footprint(image.ndim, size, footprint)
    origin = _utils._get_origin(footprint.shape, origin)
    depth = _utils._get_depth(footprint.shape, origin)
    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    result = image.map_overlap(
        dispatch_percentile_filter(image),
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        meta=image._meta,
        percentile=percentile,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
