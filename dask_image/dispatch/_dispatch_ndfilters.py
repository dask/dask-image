# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage.filters

from ._dispatcher import Dispatcher

__all__ = [
    "dispatch_convolve",
    "dispatch_correlate",
    "dispatch_laplace",
    "dispatch_prewitt",
    "dispatch_sobel",
    "dispatch_gaussian_filter",
    "dispatch_gaussian_gradient_magnitude",
    "dispatch_gaussian_laplace",
    "dispatch_generic_filter",
    "dispatch_minimum_filter",
    "dispatch_median_filter",
    "dispatch_maximum_filter",
    "dispatch_rank_filter",
    "dispatch_percentile_filter",
    "dispatch_uniform_filter",
]


dispatch_convolve = Dispatcher(name="dispatch_convolve")
dispatch_correlate = Dispatcher(name="dispatch_correlate")
dispatch_laplace = Dispatcher(name="dispatch_laplace")
dispatch_prewitt = Dispatcher(name="dispatch_prewitt")
dispatch_sobel = Dispatcher(name="dispatch_sobel")
dispatch_gaussian_filter = Dispatcher(name="dispatch_gaussian_filter")
dispatch_gaussian_gradient_magnitude = Dispatcher(name="dispatch_gaussian_gradient_magnitude")  # noqa: E501
dispatch_gaussian_laplace = Dispatcher(name="dispatch_gaussian_laplace")
dispatch_generic_filter = Dispatcher(name="dispatch_generic_filter")
dispatch_minimum_filter = Dispatcher(name="dispatch_minimum_filter")
dispatch_median_filter = Dispatcher(name="dispatch_median_filter")
dispatch_maximum_filter = Dispatcher(name="dispatch_maximum_filter")
dispatch_rank_filter = Dispatcher(name="dispatch_rank_filter")
dispatch_percentile_filter = Dispatcher(name="dispatch_percentile_filter")
dispatch_uniform_filter = Dispatcher(name="dispatch_uniform_filter")


# ================== convolve ==================
@dispatch_convolve.register(np.ndarray)
def numpy_convolve(*args, **kwargs):
    return scipy.ndimage.filters.convolve


@dispatch_convolve.register_lazy("cupy")
def register_cupy_convolve():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_convolve.register(cupy.ndarray)
    def cupy_convolve(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.convolve


# ================== correlate ==================
@dispatch_correlate.register(np.ndarray)
def numpy_correlate(*args, **kwargs):
    return scipy.ndimage.filters.correlate


@dispatch_correlate.register_lazy("cupy")
def register_cupy_correlate():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_correlate.register(cupy.ndarray)
    def cupy_correlate(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.correlate


# ================== laplace ==================
@dispatch_laplace.register(np.ndarray)
def numpy_laplace(*args, **kwargs):
    return scipy.ndimage.filters.laplace


@dispatch_laplace.register_lazy("cupy")
def register_cupy_laplace():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_laplace.register(cupy.ndarray)
    def cupy_laplace(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.laplace


# ================== prewitt ==================
@dispatch_prewitt.register(np.ndarray)
def numpy_prewitt(*args, **kwargs):
    return scipy.ndimage.filters.prewitt


@dispatch_correlate.register_lazy("cupy")
def register_cupy_prewitt():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_prewitt.register(cupy.ndarray)
    def cupy_prewitt(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.prewitt


# ================== sobel ==================
@dispatch_sobel.register(np.ndarray)
def numpy_sobel(*args, **kwargs):
    return scipy.ndimage.filters.sobel


@dispatch_sobel.register_lazy("cupy")
def register_cupy_sobel():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_sobel.register(cupy.ndarray)
    def cupy_sobel(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.sobel


# ================== gaussian_filter ==================
@dispatch_gaussian_filter.register(np.ndarray)
def numpy_gaussian_filter(*args, **kwargs):
    return scipy.ndimage.filters.gaussian_filter


@dispatch_gaussian_filter.register_lazy("cupy")
def register_cupy_gaussian_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_gaussian_filter.register(cupy.ndarray)
    def cupy_gaussian_filter(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.gaussian_filter


# ================== gaussian_gradient_magnitude ==================
@dispatch_gaussian_gradient_magnitude.register(np.ndarray)
def numpy_gaussian_gradient_magnitude(*args, **kwargs):
    return scipy.ndimage.filters.gaussian_gradient_magnitude


@dispatch_gaussian_gradient_magnitude.register_lazy("cupy")
def register_cupy_gaussian_gradient_magnitude():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_gaussian_gradient_magnitude.register(cupy.ndarray)
    def cupy_gaussian_gradient_magnitude(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.gaussian_gradient_magnitude


# ================== gaussian_laplace ==================
@dispatch_gaussian_laplace.register(np.ndarray)
def numpy_gaussian_laplace(*args, **kwargs):
    return scipy.ndimage.filters.gaussian_laplace


@dispatch_gaussian_laplace.register_lazy("cupy")
def register_cupy_gaussian_laplace():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_gaussian_laplace.register(cupy.ndarray)
    def cupy_gaussian_laplace(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.gaussian_laplace


# ================== generic_filter ==================
@dispatch_generic_filter.register(np.ndarray)
def numpy_generic_filter(*args, **kwargs):
    return scipy.ndimage.filters.generic_filter


@dispatch_generic_filter.register_lazy("cupy")
def register_cupy_generic_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_generic_filter.register(cupy.ndarray)
    def cupy_generic_filter(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.generic_filter


# ================== minimum_filter ==================
@dispatch_minimum_filter.register(np.ndarray)
def numpy_minimum_filter(*args, **kwargs):
    return scipy.ndimage.filters.minimum_filter


@dispatch_minimum_filter.register_lazy("cupy")
def register_cupy_minimum_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_minimum_filter.register(cupy.ndarray)
    def cupy_minimum_filter(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.minimum_filter


# ================== median_filter ==================
@dispatch_median_filter.register(np.ndarray)
def numpy_median_filter(*args, **kwargs):
    return scipy.ndimage.filters.median_filter


@dispatch_median_filter.register_lazy("cupy")
def register_cupy_median_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_median_filter.register(cupy.ndarray)
    def cupy_median_filter(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.median_filter


# ================== maximum_filter ==================
@dispatch_maximum_filter.register(np.ndarray)
def numpy_maximum_filter(*args, **kwargs):
    return scipy.ndimage.filters.maximum_filter


@dispatch_maximum_filter.register_lazy("cupy")
def register_cupy_maximum_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_maximum_filter.register(cupy.ndarray)
    def cupy_maximum_filter(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.maximum_filter


# ================== rank_filter ==================
@dispatch_rank_filter.register(np.ndarray)
def numpy_rank_filter(*args, **kwargs):
    return scipy.ndimage.filters.rank_filter


@dispatch_rank_filter.register_lazy("cupy")
def register_cupy_rank_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_rank_filter.register(cupy.ndarray)
    def cupy_rank_filter(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.rank_filter


# ================== percentile_filter ==================
@dispatch_percentile_filter.register(np.ndarray)
def numpy_percentile_filter(*args, **kwargs):
    return scipy.ndimage.filters.percentile_filter


@dispatch_percentile_filter.register_lazy("cupy")
def register_cupy_percentile_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_percentile_filter.register(cupy.ndarray)
    def cupy_percentile_filter(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.percentile_filter


# ================== uniform_filter ==================
@dispatch_uniform_filter.register(np.ndarray)
def numpy_uniform_filter(*args, **kwargs):
    return scipy.ndimage.filters.uniform_filter


@dispatch_uniform_filter.register_lazy("cupy")
def register_cupy_uniform_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_uniform_filter.register(cupy.ndarray)
    def cupy_uniform_filter(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.uniform_filter
