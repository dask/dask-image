# -*- coding: utf-8 -*-


import scipy.ndimage.filters

from ..dispatch._dispatch_ndfilters import dispatch_laplace
from . import _utils

__all__ = [
    "laplace",
]


@_utils._update_wrapper(scipy.ndimage.filters.laplace)
def laplace(image, mode='reflect', cval=0.0):
    result = image.map_overlap(
        dispatch_laplace(image),
        depth=(image.ndim * (1,)),
        boundary="none",
        dtype=image.dtype,
        meta=image._meta,
        mode=mode,
        cval=cval
    )

    return result
