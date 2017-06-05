# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import scipy.ndimage.morphology

import dask_ndmorph._utils as _utils
import dask_ndmorph._ops as _ops


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
