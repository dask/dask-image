# -*- coding: utf-8 -*-


import scipy.ndimage.filters

from . import _utils


@_utils._update_wrapper(scipy.ndimage.filters.laplace)
def laplace(image, mode='reflect', cval=0.0):
    result = image.map_overlap(
        scipy.ndimage.filters.laplace,
        depth=(image.ndim * (1,)),
        boundary="none",
        dtype=image.dtype,
        mode=mode,
        cval=cval
    )

    return result
