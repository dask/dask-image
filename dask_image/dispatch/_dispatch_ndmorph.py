# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage

from ._dispatcher import Dispatcher

__all__ = [
    "dispatch_binary_dilation",
    "dispatch_binary_erosion",
]

dispatch_binary_dilation = Dispatcher(name="dispatch_binary_dilation")
dispatch_binary_erosion = Dispatcher(name="dispatch_binary_erosion")


# ================== binary_dilation ==================
@dispatch_binary_dilation.register(np.ndarray)
def numpy_binary_dilation(*args, **kwargs):
    return scipy.ndimage.binary_dilation


@dispatch_binary_dilation.register_lazy("cupy")
def register_cupy_binary_dilation():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_binary_dilation.register(cupy.ndarray)
    def cupy_binary_dilation(*args, **kwargs):
        return cupyx.scipy.ndimage.binary_dilation


# ================== binary_erosion ==================
@dispatch_binary_erosion.register(np.ndarray)
def numpy_binary_erosion(*args, **kwargs):
    return scipy.ndimage.binary_erosion


@dispatch_binary_erosion.register_lazy("cupy")
def register_cupy_binary_erosion():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_binary_erosion.register(cupy.ndarray)
    def cupy_binary_erosion(*args, **kwargs):
        return cupyx.scipy.ndimage.binary_erosion
