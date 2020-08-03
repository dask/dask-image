# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage.filters

from . import _utils
from ..utils import Dispatcher

convolve = Dispatcher(name="convolve")


@_utils._update_wrapper(scipy.ndimage.filters.convolve)
def convolve_func(func,
                  image,
                  weights,
                  mode='reflect',
                  cval=0.0,
                  origin=0):
    origin = _utils._get_origin(weights.shape, origin)
    depth = _utils._get_depth(weights.shape, origin)
    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")

    result = image.map_overlap(
        func,
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        weights=weights,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result


@convolve.register(np.ndarray)
def numpy_convolve(*args, **kwargs):
    return convolve_func(scipy.ndimage.filters.convolve, *args, **kwargs)


@convolve.register_lazy("cupy")
def register_cupy():
    import cupy
    import cupyx

    @convolve.register(cupy.ndarray)
    def cupy_convolve(*args, **kwargs):
        return convolve_func(cupyx.scipy.ndimage.filters.convolve, *args, **kwargs)


@_utils._update_wrapper(scipy.ndimage.filters.correlate)
def correlate(image,
              weights,
              mode='reflect',
              cval=0.0,
              origin=0):
    origin = _utils._get_origin(weights.shape, origin)
    depth = _utils._get_depth(weights.shape, origin)
    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")

    result = image.map_overlap(
        scipy.ndimage.filters.correlate,
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        weights=weights,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
