# -*- coding: utf-8 -*-


import scipy.ndimage.filters

from . import _utils
from ._gaussian import gaussian_filter


@_utils._update_wrapper(scipy.ndimage.filters.uniform_filter)
def uniform_filter(input,
                   size=3,
                   mode='reflect',
                   cval=0.0,
                   origin=0):
    size = _utils._get_size(input.ndim, size)
    depth = _utils._get_depth(size, origin)

    depth, boundary = _utils._get_depth_boundary(input.ndim, depth, "none")

    result = input.map_overlap(
        scipy.ndimage.filters.uniform_filter,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        size=size,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
