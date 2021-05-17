# -*- coding: utf-8 -*-
import scipy.ndimage

from ..dispatch._dispatch_ndmorph import (dispatch_binary_dilation,
                                          dispatch_binary_erosion)
from . import _ops, _utils

__all__ = [
    "binary_closing",
    "binary_dilation",
    "binary_erosion",
    "binary_opening",
]


@_utils._update_wrapper(scipy.ndimage.binary_closing)
def binary_closing(image,
                   structure=None,
                   iterations=1,
                   origin=0,
                   mask=None,
                   border_value=0,
                   brute_force=False):
    image = (image != 0)

    structure = _utils._get_structure(image, structure)
    iterations = _utils._get_iterations(iterations)
    origin = _utils._get_origin(structure.shape, origin)

    kwargs =  dict(
        structure=structure,
        iterations=iterations,
        origin=origin,
        mask=mask,
        border_value=border_value,
        brute_force=brute_force
    )

    result = image
    result = binary_dilation(result, **kwargs)
    result = binary_erosion(result, **kwargs)

    return result


@_utils._update_wrapper(scipy.ndimage.binary_dilation)
def binary_dilation(image,
                    structure=None,
                    iterations=1,
                    mask=None,
                    border_value=0,
                    origin=0,
                    brute_force=False):
    border_value = _utils._get_border_value(border_value)

    result = _ops._binary_op(
        dispatch_binary_dilation(image),
        image,
        structure=structure,
        iterations=iterations,
        mask=mask,
        origin=origin,
        brute_force=brute_force,
        border_value=border_value
    )

    return result


@_utils._update_wrapper(scipy.ndimage.binary_erosion)
def binary_erosion(image,
                   structure=None,
                   iterations=1,
                   mask=None,
                   border_value=0,
                   origin=0,
                   brute_force=False):
    border_value = _utils._get_border_value(border_value)

    result = _ops._binary_op(
        dispatch_binary_erosion(image),
        image,
        structure=structure,
        iterations=iterations,
        mask=mask,
        origin=origin,
        brute_force=brute_force,
        border_value=border_value
    )

    return result


@_utils._update_wrapper(scipy.ndimage.binary_opening)
def binary_opening(image,
                   structure=None,
                   iterations=1,
                   origin=0,
                   mask=None,
                   border_value=0,
                   brute_force=False):
    image = (image != 0)

    structure = _utils._get_structure(image, structure)
    iterations = _utils._get_iterations(iterations)
    origin = _utils._get_origin(structure.shape, origin)

    kwargs =  dict(
        structure=structure,
        iterations=iterations,
        origin=origin,
        mask=mask,
        border_value=border_value,
        brute_force=brute_force
    )

    result = image
    result = binary_erosion(result, **kwargs)
    result = binary_dilation(result, **kwargs)

    return result
