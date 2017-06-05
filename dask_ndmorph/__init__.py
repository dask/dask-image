# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import scipy.ndimage.morphology

import dask_ndmorph._utils as _utils
import dask_ndmorph._ops as _ops


@_utils._update_wrapper(scipy.ndimage.morphology.binary_closing)
def binary_closing(input,
                   structure=None,
                   iterations=1,
                   origin=0):
    input = (input != 0)

    structure = _utils._get_structure(input, structure)
    iterations = _utils._get_iterations(iterations)
    origin = _utils._get_origin(structure.shape, origin)

    result = input
    result = binary_dilation(
        result, structure=structure, iterations=iterations, origin=origin
    )
    result = binary_erosion(
        result, structure=structure, iterations=iterations, origin=origin
    )

    return result


@_utils._update_wrapper(scipy.ndimage.morphology.binary_dilation)
def binary_dilation(input,
                    structure=None,
                    iterations=1,
                    mask=None,
                    border_value=0,
                    origin=0,
                    brute_force=False):
    border_value = _utils._get_border_value(border_value)

    result = _ops._binary_op(
        scipy.ndimage.morphology.binary_dilation,
        input,
        structure=structure,
        iterations=iterations,
        mask=mask,
        origin=origin,
        brute_force=brute_force,
        border_value=border_value
    )

    return result


@_utils._update_wrapper(scipy.ndimage.morphology.binary_erosion)
def binary_erosion(input,
                   structure=None,
                   iterations=1,
                   mask=None,
                   border_value=0,
                   origin=0,
                   brute_force=False):
    border_value = _utils._get_border_value(border_value)

    result = _ops._binary_op(
        scipy.ndimage.morphology.binary_erosion,
        input,
        structure=structure,
        iterations=iterations,
        mask=mask,
        origin=origin,
        brute_force=brute_force,
        border_value=border_value
    )

    return result


@_utils._update_wrapper(scipy.ndimage.morphology.binary_opening)
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
