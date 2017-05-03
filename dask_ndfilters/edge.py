# -*- coding: utf-8 -*-


import numbers

import scipy.ndimage.filters

import dask_ndfilters._utils as _utils


def _validate_axis(ndim, axis):
    if not isinstance(axis, numbers.Integral):
        raise ValueError("The axis must be of integral type.")
    if axis < -ndim or axis >= ndim:
        raise ValueError("The axis is out of range.")


@_utils._update_wrapper(scipy.ndimage.filters.prewitt)
def prewitt(input, axis=-1, mode='reflect', cval=0.0):
    _validate_axis(input.ndim, axis)

    result = input.map_overlap(
        scipy.ndimage.filters.prewitt,
        depth=(input.ndim * (1,)),
        boundary="none",
        dtype=input.dtype,
        name=scipy.ndimage.filters.prewitt.__name__,
        axis=axis,
        mode=mode,
        cval=cval
    )

    return result
