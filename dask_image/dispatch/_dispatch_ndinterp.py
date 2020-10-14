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


dispatch_asarray = Dispatcher(name="dispatch_asarray")

# ================== affine_transform ==================
@dispatch_asarray.register(np.ndarray)
def numpy_asarray(*args, **kwargs):
    return np.asarray


@dispatch_asarray.register_lazy("cupy")
def register_cupy_asarray():
    import cupy

    @dispatch_asarray.register(cupy.ndarray)
    def cupy_asarray(*args, **kwargs):

        return cupy.asarray
