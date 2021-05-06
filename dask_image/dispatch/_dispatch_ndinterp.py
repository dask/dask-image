# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage

from ._dispatcher import Dispatcher

__all__ = [
    "dispatch_affine_transform",
    "dispatch_asarray",
]


dispatch_affine_transform = Dispatcher(name="dispatch_affine_transform")

# ================== affine_transform ==================
@dispatch_affine_transform.register(np.ndarray)
def numpy_affine_transform(*args, **kwargs):
    return ndimage.affine_transform


@dispatch_affine_transform.register_lazy("cupy")
def register_cupy_affine_transform():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_affine_transform.register(cupy.ndarray)
    def cupy_affine_transform(*args, **kwargs):

        return cupyx.scipy.ndimage.affine_transform


dispatch_spline_filter = Dispatcher(name="dispatch_spline_filter")

# ================== spline_filter ==================
@dispatch_spline_filter.register(np.ndarray)
def numpy_spline_filter(*args, **kwargs):
    return ndimage.spline_filter


@dispatch_spline_filter.register_lazy("cupy")
def register_cupy_spline_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_spline_filter.register(cupy.ndarray)
    def cupy_spline_filter(*args, **kwargs):

        return cupyx.scipy.ndimage.spline_filter


dispatch_spline_filter1d = Dispatcher(name="dispatch_spline_filter1d")

# ================== spline_filter1d ==================
@dispatch_spline_filter1d.register(np.ndarray)
def numpy_spline_filter1d(*args, **kwargs):
    return ndimage.spline_filter1d


@dispatch_spline_filter1d.register_lazy("cupy")
def register_cupy_spline_filter1d():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_spline_filter1d.register(cupy.ndarray)
    def cupy_spline_filter1d(*args, **kwargs):

        return cupyx.scipy.ndimage.spline_filter1d


dispatch_asarray = Dispatcher(name="dispatch_asarray")

# ===================== asarray ========================
@dispatch_asarray.register(np.ndarray)
def numpy_asarray(*args, **kwargs):
    return np.asarray


@dispatch_asarray.register_lazy("cupy")
def register_cupy_asarray():
    import cupy

    @dispatch_asarray.register(cupy.ndarray)
    def cupy_asarray(*args, **kwargs):

        return cupy.asarray
