# -*- coding: utf-8 -*-


import numbers

import numpy
import scipy.ndimage.filters

from . import _utils


def _get_sigmas(image, sigma):
    ndim = image.ndim

    nsigmas = numpy.array(sigma)
    if nsigmas.ndim == 0:
        nsigmas = numpy.array(ndim * [nsigmas[()]])

    if nsigmas.ndim != 1:
        raise RuntimeError(
            "Must have a single sigma or a single sequence."
        )

    if ndim != len(nsigmas):
        raise RuntimeError(
            "Must have an equal number of sigmas to image dimensions."
        )

    if not issubclass(nsigmas.dtype.type, numbers.Real):
        raise TypeError("Must have real sigmas.")

    nsigmas = tuple(nsigmas)

    return nsigmas


def _get_border(image, sigma, truncate):
    sigma = numpy.array(_get_sigmas(image, sigma))

    if not isinstance(truncate, numbers.Real):
        raise TypeError("Must have a real truncate value.")

    half_shape = tuple(numpy.ceil(sigma * truncate).astype(int))

    return half_shape


@_utils._update_wrapper(scipy.ndimage.filters.gaussian_filter)
def gaussian_filter(image,
                    sigma,
                    order=0,
                    mode='reflect',
                    cval=0.0,
                    truncate=4.0):
    sigma = _get_sigmas(image, sigma)
    depth = _get_border(image, sigma, truncate)

    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")

    result = image.map_overlap(
        scipy.ndimage.filters.gaussian_filter,
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        sigma=sigma,
        order=order,
        mode=mode,
        cval=cval,
        truncate=truncate
    )

    return result


@_utils._update_wrapper(scipy.ndimage.filters.gaussian_gradient_magnitude)
def gaussian_gradient_magnitude(image,
                                sigma,
                                mode='reflect',
                                cval=0.0,
                                truncate=4.0,
                                **kwargs):
    sigma = _get_sigmas(image, sigma)
    depth = _get_border(image, sigma, truncate)

    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")

    result = image.map_overlap(
        scipy.ndimage.filters.gaussian_gradient_magnitude,
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        sigma=sigma,
        mode=mode,
        cval=cval,
        truncate=truncate,
        **kwargs
    )

    return result


@_utils._update_wrapper(scipy.ndimage.filters.gaussian_laplace)
def gaussian_laplace(image,
                     sigma,
                     mode='reflect',
                     cval=0.0,
                     truncate=4.0,
                     **kwargs):
    sigma = _get_sigmas(image, sigma)
    depth = _get_border(image, sigma, truncate)

    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")

    result = image.map_overlap(
        scipy.ndimage.filters.gaussian_laplace,
        depth=depth,
        boundary=boundary,
        dtype=image.dtype,
        sigma=sigma,
        mode=mode,
        cval=cval,
        truncate=truncate,
        **kwargs
    )

    return result
