# -*- coding: utf-8 -*-


import numbers

import dask.array
import numpy as np
import scipy.ndimage

from ..dispatch._dispatch_ndmorph import dispatch_binary_structure
from ..ndfilters._utils import (_get_depth, _get_depth_boundary, _get_origin,
                                _update_wrapper)

_update_wrapper = _update_wrapper
_get_depth_boundary = _get_depth_boundary
_get_origin = _get_origin
_get_depth = _get_depth


def _get_structure(image, structure):
    # Create square connectivity as default
    if structure is None:
        generate_binary_structure = dispatch_binary_structure(image)
        structure = generate_binary_structure(image.ndim, 1)
    elif hasattr(structure, 'ndim'):
        if structure.ndim != image.ndim:
            raise RuntimeError(
                "`structure` must have the same rank as `image`."
            )
        if not issubclass(structure.dtype.type, np.bool8):
            structure = (structure != 0)
    else:
        raise TypeError("`structure` must be an array.")

    return structure


def _get_iterations(iterations):
    if not isinstance(iterations, numbers.Integral):
        raise TypeError("`iterations` must be of integral type.")
    if iterations < 1:
        raise NotImplementedError(
            "`iterations` must be equal to 1 or greater not less."
        )

    return iterations


def _get_dtype(a):
    # Get the dtype of a value or an array.
    # Even handle non-NumPy types.
    return getattr(a, "dtype", np.dtype(type(a)))


def _get_mask(image, mask):
    if mask is None:
        mask = True

    mask_type = _get_dtype(mask).type
    if isinstance(mask, (np.ndarray, dask.array.Array)):
        if mask.shape != image.shape:
            raise RuntimeError("`mask` must have the same shape as `image`.")
        if not issubclass(mask_type, np.bool8):
            mask = (mask != 0)
    elif issubclass(mask_type, np.bool8):
        mask = bool(mask)
    else:
        raise TypeError("`mask` must be a Boolean or an array.")

    return mask


def _get_border_value(border_value):
    if not isinstance(border_value, numbers.Integral):
        raise TypeError("`border_value` must be of integral type.")

    border_value = (border_value != 0)

    return border_value


def _get_brute_force(brute_force):
    if brute_force is not False:
        if brute_force is True:
            raise NotImplementedError(
                "`brute_force` other than `False` is not yet supported."
            )
        else:
            raise TypeError(
                "`brute_force` must be `bool`."
            )

    return brute_force
