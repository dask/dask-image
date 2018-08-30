# -*- coding: utf-8 -*-


import numbers

import scipy.ndimage.filters

from . import _utils


@_utils._update_wrapper(scipy.ndimage.filters.laplace)
def laplace(input, mode='reflect', cval=0.0):
    result = input.map_overlap(
        scipy.ndimage.filters.laplace,
        depth=(input.ndim * (1,)),
        boundary="none",
        dtype=input.dtype,
        mode=mode,
        cval=cval
    )

    return result
