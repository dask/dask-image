# -*- coding: utf-8 -*-


import numbers

import numpy
import scipy.ndimage.filters

import dask_ndfilters._utils as _utils


@_utils._update_wrapper(scipy.ndimage.filters.generic_filter)
def generic_filter(input,
                   function,
                   size=None,
                   footprint=None,
                   mode='reflect',
                   cval=0.0,
                   origin=0,
                   extra_arguments=tuple(),
                   extra_keywords=dict()):
    footprint = _utils._get_footprint(input.ndim, size, footprint)
    origin = _utils._get_origin(footprint.shape, origin)
    depth = _utils._get_depth(footprint.shape, origin)
    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    result = input.map_overlap(
        scipy.ndimage.filters.generic_filter,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        name=scipy.ndimage.filters.generic_filter.__name__,
        function=function,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin,
        extra_arguments=extra_arguments,
        extra_keywords=extra_keywords
    )

    return result
