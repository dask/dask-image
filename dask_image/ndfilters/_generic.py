# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage

from ..dispatch._dispatch_ndfilters import dispatch_generic_filter
from . import _utils

__all__ = [
    "generic_filter",
]


@_utils._update_wrapper(scipy.ndimage.generic_filter)
def generic_filter(image,
                   function,
                   size=None,
                   footprint=None,
                   mode='reflect',
                   cval=0.0,
                   origin=0,
                   extra_arguments=tuple(),
                   extra_keywords=dict()):
    footprint = _utils._get_footprint(image.ndim, size, footprint)
    origin = _utils._get_origin(footprint.shape, origin)
    depth = _utils._get_depth(footprint.shape, origin)
    depth, boundary = _utils._get_depth_boundary(footprint.ndim, depth, "none")

    if type(image._meta) == np.ndarray:
        kwargs = {"extra_arguments": extra_arguments,
                  "extra_keywords": extra_keywords}
    else:  # pragma: no cover
        # cupy generic_filter doesn't support extra_arguments or extra_keywords
        kwargs = {}

    result = image.map_overlap(
        dispatch_generic_filter(image),
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        meta=image._meta,
        function=function,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin,
        **kwargs
    )

    return result
