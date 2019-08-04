# -*- coding: utf-8 -*-


import scipy.ndimage.filters

from . import _utils
from ._gaussian import gaussian_filter


gaussian_filter = gaussian_filter


@_utils._update_wrapper(scipy.ndimage.filters.uniform_filter)
def uniform_filter(image,
                   size=3,
                   mode='reflect',
                   cval=0.0,
                   origin=0):
    size = _utils._get_size(image.ndim, size)
    depth = _utils._get_depth(size, origin)

    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")

    result = image.map_overlap(
        scipy.ndimage.filters.uniform_filter,
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        size=size,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
