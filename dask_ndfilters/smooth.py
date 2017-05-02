# -*- coding: utf-8 -*-


import numbers

import numpy
import scipy.ndimage.filters

import dask_ndfilters._utils as _utils


def _get_sigmas(input, sigma):
    ndim = input.ndim

    nsigmas = numpy.array(sigma)
    if nsigmas.ndim == 0:
        nsigmas = numpy.array(ndim * [nsigmas[()]])

    if nsigmas.ndim != 1:
        raise RuntimeError(
            "Must have a single sigma or a single sequence."
        )

    if ndim != len(nsigmas):
        raise RuntimeError(
            "Must have an equal number of sigmas to input dimensions."
        )

    if not issubclass(nsigmas.dtype.type, numbers.Real):
        raise TypeError("Must have real sigmas.")

    nsigmas = tuple(nsigmas)

    return nsigmas


def _get_border(input, sigma, truncate):
    sigma = numpy.array(_get_sigmas(input, sigma))

    if not isinstance(truncate, numbers.Real):
        raise TypeError("Must have a real truncate value.")

    half_shape = tuple(numpy.ceil(sigma * truncate).astype(int))

    return half_shape


@_utils._update_wrapper(scipy.ndimage.filters.gaussian_filter)
def gaussian_filter(input,
                    sigma,
                    order=0,
                    mode='reflect',
                    cval=0.0,
                    truncate=4.0):
    sigma = _get_sigmas(input, sigma)
    depth = _get_border(input, sigma, truncate)

    depth, boundary = _utils._get_depth_boundary(input.ndim, depth, "none")

    result = input.map_overlap(
        scipy.ndimage.filters.gaussian_filter,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        name=scipy.ndimage.filters.gaussian_filter.__name__,
        sigma=sigma,
        order=order,
        mode=mode,
        cval=cval,
        truncate=truncate
    )

    return result


@_utils._update_wrapper(scipy.ndimage.filters.uniform_filter)
def uniform_filter(input,
                   size=3,
                   mode='reflect',
                   cval=0.0,
                   origin=0):
    size = _utils._get_size(input.ndim, size)
    depth = _utils._get_depth(size, origin)

    depth, boundary = _utils._get_depth_boundary(input.ndim, depth, "none")

    result = input.map_overlap(
        scipy.ndimage.filters.uniform_filter,
        depth=depth,
        boundary=boundary,
        dtype=input.dtype,
        name=scipy.ndimage.filters.uniform_filter.__name__,
        size=size,
        mode=mode,
        cval=cval,
        origin=origin
    )

    return result
