# -*- coding: utf-8 -*-

import scipy.ndimage.filters

from . import _utils
from ..dispatch._dispatch_ndfilters import (
    dispatch_minimum_filter,
    dispatch_median_filter,
    dispatch_maximum_filter,
    dispatch_rank_filter,
    dispatch_percentile_filter)

__all__ = [
    "minimum_filter",
    "median_filter",
    "maximum_filter",
    "rank_filter",
    "percentile_filter",
]


@_utils._update_wrapper(scipy.ndimage.filters.minimum_filter)
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


@_utils._update_wrapper(scipy.ndimage.filters.median_filter)
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


@_utils._update_wrapper(scipy.ndimage.filters.maximum_filter)
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


@_utils._update_wrapper(scipy.ndimage.filters.rank_filter)
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


@_utils._update_wrapper(scipy.ndimage.filters.percentile_filter)
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
