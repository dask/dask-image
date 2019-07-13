# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


import scipy.ndimage

from . import _utils
from . import _ops


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
        scipy.ndimage.binary_dilation,
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
def binary_erosion(input,
                   structure=None,
                   iterations=1,
                   mask=None,
                   border_value=0,
                   origin=0,
                   brute_force=False):
    border_value = _utils._get_border_value(border_value)

    result = _ops._binary_op(
        scipy.ndimage.binary_erosion,
        input,
        structure=structure,
        iterations=iterations,
        mask=mask,
        origin=origin,
        brute_force=brute_force,
        border_value=border_value
    )

    return result


@_utils._update_wrapper(scipy.ndimage.binary_opening)
def binary_opening(input,
                   structure=None,
                   iterations=1,
                   origin=0):
    input = (input != 0)

    structure = _utils._get_structure(input, structure)
    iterations = _utils._get_iterations(iterations)
    origin = _utils._get_origin(structure.shape, origin)

    result = input
    result = binary_erosion(
        result, structure=structure, iterations=iterations, origin=origin
    )
    result = binary_dilation(
        result, structure=structure, iterations=iterations, origin=origin
    )

    return result
