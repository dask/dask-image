# -*- coding: utf-8 -*-

import functools
import math

import dask.array as da
import numpy as np
import scipy

from ..dispatch._dispatch_ndinterp import (
    dispatch_spline_filter,
    dispatch_spline_filter1d,
)
from ..ndfilters._utils import _get_depth_boundary, _update_wrapper


# magnitude of the maximum filter pole for each order
# (obtained from scipy/ndimage/src/ni_splines.c)
_maximum_pole = {
    2: 0.171572875253809902396622551580603843,
    3: 0.267949192431122706472553658494127633,
    4: 0.361341225900220177092212841325675255,
    5: 0.430575347099973791851434783493520110,
}


def _get_default_depth(order, tol=1e-8):
    """Determine the approximate depth needed for a given tolerance.

    Here depth is chosen as the smallest integer such that ``|p| ** n < tol``
    where `|p|` is the magnitude of the largest pole in the IIR filter.
    """

    return math.ceil(np.log(tol) / np.log(_maximum_pole[order]))


@_update_wrapper(scipy.ndimage.spline_filter)
def spline_filter(
        image,
        order=3,
        output=np.float64,
        mode='mirror',
        output_chunks=None,
        *,
        depth=None,
        **kwargs
):

    if not isinstance(image, da.core.Array):
        image = da.from_array(image)

    # use dispatching mechanism to determine backend
    spline_filter_method = dispatch_spline_filter(image)

    try:
        dtype = np.dtype(output)
    except TypeError:     # pragma: no cover
        raise TypeError(  # pragma: no cover
            "Could not coerce the provided output to a dtype. "
            "Passing array to output is not currently supported."
        )

    if depth is None:
        depth = _get_default_depth(order)

    if mode == 'wrap':
        raise NotImplementedError(
            "mode='wrap' is unsupported. It is recommended to use 'grid-wrap' "
            "instead."
        )

    # Note: depths of 12 and 24 give results matching SciPy to approximately
    #       single and double precision accuracy, respectively.
    boundary = "periodic" if mode == 'grid-wrap' else "none"
    depth, boundary = _get_depth_boundary(image.ndim, depth, boundary)

    # cannot pass a func kwarg named "output" to map_overlap
    spline_filter_method = functools.partial(spline_filter_method,
                                             output=dtype)

    result = image.map_overlap(
        spline_filter_method,
        depth=depth,
        boundary=boundary,
        dtype=dtype,
        meta=image._meta,
        # spline_filter kwargs
        order=order,
        mode=mode,
    )

    return result


@_update_wrapper(scipy.ndimage.spline_filter1d)
def spline_filter1d(
        image,
        order=3,
        axis=-1,
        output=np.float64,
        mode='mirror',
        output_chunks=None,
        *,
        depth=None,
        **kwargs
):

    if not isinstance(image, da.core.Array):
        image = da.from_array(image)

    # use dispatching mechanism to determine backend
    spline_filter1d_method = dispatch_spline_filter1d(image)

    try:
        dtype = np.dtype(output)
    except TypeError:     # pragma: no cover
        raise TypeError(  # pragma: no cover
            "Could not coerce the provided output to a dtype. "
            "Passing array to output is not currently supported."
        )

    if depth is None:
        depth = _get_default_depth(order)

    # use depth 0 on all axes except the filtered axis
    if not np.isscalar(depth):
        raise ValueError("depth must be a scalar value")
    depths = [0] * image.ndim
    depths[axis] = depth

    if mode == 'wrap':
        raise NotImplementedError(
            "mode='wrap' is unsupported. It is recommended to use 'grid-wrap' "
            "instead."
        )

    # cannot pass a func kwarg named "output" to map_overlap
    spline_filter1d_method = functools.partial(spline_filter1d_method,
                                               output=dtype)

    result = image.map_overlap(
        spline_filter1d_method,
        depth=tuple(depths),
        boundary="periodic" if mode == 'grid-wrap' else "none",
        dtype=dtype,
        meta=image._meta,
        # spline_filter1d kwargs
        order=order,
        axis=axis,
        mode=mode,
    )

    return result
