# -*- coding: utf-8 -*-


import numbers

import scipy.ndimage.filters

from . import _utils


def _validate_axis(ndim, axis):
    if not isinstance(axis, numbers.Integral):
        raise ValueError("The axis must be of integral type.")
    if axis < -ndim or axis >= ndim:
        raise ValueError("The axis is out of range.")


@_utils._update_wrapper(scipy.ndimage.filters.prewitt)
def prewitt(image, axis=-1, mode='reflect', cval=0.0):
    _validate_axis(image.ndim, axis)

    result = image.map_overlap(
        scipy.ndimage.filters.prewitt,
        depth=(image.ndim * (1,)),
        boundary="none",
        dtype=image.dtype,
        axis=axis,
        mode=mode,
        cval=cval
    )

    return result


@_utils._update_wrapper(scipy.ndimage.filters.sobel)
def sobel(input, axis=-1, mode='reflect', cval=0.0):
    _validate_axis(input.ndim, axis)

    result = input.map_overlap(
        scipy.ndimage.filters.sobel,
        depth=(input.ndim * (1,)),
        boundary="none",
        dtype=input.dtype,
        axis=axis,
        mode=mode,
        cval=cval
    )

    return result
