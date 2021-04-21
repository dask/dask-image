# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


import scipy.ndimage

from . import _utils
from . import _ops
from ..dispatch._dispatch_ndmorph import (
    dispatch_binary_dilation,
    dispatch_binary_erosion)

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
                   origin=0):
    image = (image != 0)

    structure = _utils._get_structure(image, structure)
    iterations = _utils._get_iterations(iterations)
    origin = _utils._get_origin(structure.shape, origin)

    result = image
    result = binary_dilation(
        result, structure=structure, iterations=iterations, origin=origin
    )
    result = binary_erosion(
        result, structure=structure, iterations=iterations, origin=origin
    )
    result._meta = image._meta.astype(bool)

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
    result._meta = image._meta.astype(bool)

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
    result._meta = image._meta.astype(bool)

    return result


@_utils._update_wrapper(scipy.ndimage.binary_opening)
def binary_opening(image,
                   structure=None,
                   iterations=1,
                   origin=0):
    image = (image != 0)

    structure = _utils._get_structure(image, structure)
    iterations = _utils._get_iterations(iterations)
    origin = _utils._get_origin(structure.shape, origin)

    result = image
    result = binary_erosion(
        result, structure=structure, iterations=iterations, origin=origin
    )
    result = binary_dilation(
        result, structure=structure, iterations=iterations, origin=origin
    )
    result._meta = image._meta.astype(bool)

    return result
