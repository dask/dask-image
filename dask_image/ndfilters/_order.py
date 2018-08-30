# -*- coding: utf-8 -*-


import numbers

import numpy
import scipy.ndimage.filters

from . import _utils


def _ordering_filter_wrapper(func):
    @_utils._update_wrapper(func)
    def _wrapped_ordering_filter(input,
                                 size=None,
                                 footprint=None,
                                 mode='reflect',
                                 cval=0.0,
                                 origin=0):
        footprint = _utils._get_footprint(input.ndim, size, footprint)
        origin = _utils._get_origin(footprint.shape, origin)
        depth = _utils._get_depth(footprint.shape, origin)
        depth, boundary = _utils._get_depth_boundary(footprint.ndim,
                                                     depth,
                                                     "none")

        result = input.map_overlap(
            func,
            depth=depth,
            boundary=boundary,
            dtype=input.dtype,
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
    footprint = _utils._get_footprint(input.ndim, size, footprint)
    origin = _utils._get_origin(footprint.shape, origin)
    depth = _utils._get_depth(footprint.shape, origin)
    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    result = input.map_overlap(
        scipy.ndimage.filters.rank_filter,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
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
    footprint = _utils._get_footprint(input.ndim, size, footprint)
    origin = _utils._get_origin(footprint.shape, origin)
    depth = _utils._get_depth(footprint.shape, origin)
    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    result = input.map_overlap(
        scipy.ndimage.filters.percentile_filter,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        percentile=percentile,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
