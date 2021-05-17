# -*- coding: utf-8 -*-


import numbers

import scipy.ndimage

from ..dispatch._dispatch_ndfilters import dispatch_prewitt, dispatch_sobel
from . import _utils

__all__ = [
    "prewitt",
    "sobel",
]


def _validate_axis(ndim, axis):
    if not isinstance(axis, numbers.Integral):
        raise ValueError("The axis must be of integral type.")
    if axis < -ndim or axis >= ndim:
        raise ValueError("The axis is out of range.")


@_utils._update_wrapper(scipy.ndimage.filters.prewitt)
def prewitt(image, axis=-1, mode='reflect', cval=0.0):
    _validate_axis(image.ndim, axis)

    result = image.map_overlap(
        dispatch_prewitt(image),
        depth=(image.ndim * (1,)),
        boundary="none",
        dtype=image.dtype,
        meta=image._meta,
        axis=axis,
        mode=mode,
        cval=cval
    )

    return result


@_utils._update_wrapper(scipy.ndimage.filters.sobel)
def sobel(image, axis=-1, mode='reflect', cval=0.0):
    _validate_axis(image.ndim, axis)

    result = image.map_overlap(
        dispatch_sobel(image),
        depth=(image.ndim * (1,)),
        boundary="none",
        dtype=image.dtype,
        meta=image._meta,
        axis=axis,
        mode=mode,
        cval=cval
    )

    return result
