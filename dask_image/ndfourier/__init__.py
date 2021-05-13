# -*- coding: utf-8 -*-

from __future__ import division


__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


import numbers

import dask.array

from . import _utils

__all__ = [
    "fourier_gaussian",
    "fourier_shift",
    "fourier_uniform",
]


def fourier_gaussian(image, sigma, n=-1, axis=-1):
    """
    Multi-dimensional Gaussian fourier filter.

    The array is multiplied with the fourier transform of a Gaussian
    kernel.

    Parameters
    ----------
    image : array_like
        The input image.
    sigma : float or sequence
        The sigma of the Gaussian kernel. If a float, `sigma` is the same for
        all axes. If a sequence, `sigma` has to contain one value for each
        axis.
    n : int, optional
        If `n` is negative (default), then the image is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the image is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.

    Returns
    -------
    fourier_gaussian : Dask Array

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = misc.ascent()
    >>> image = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_gaussian(image, sigma=4)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    """

    # Validate and normalize arguments
    image, sigma, n, axis = _utils._norm_args(image, sigma, n=n, axis=axis)

    # Compute frequencies
    ang_freq_grid = _utils._get_ang_freq_grid(
        image.shape,
        chunks=image.chunks,
        n=n,
        axis=axis,
        dtype=sigma.dtype
    )

    # Compute Fourier transformed Gaussian
    result = image.copy()
    scale = (sigma ** 2) / -2
    for ax, f in enumerate(ang_freq_grid):
        f *= f
        gaussian = dask.array.exp(scale[ax] * f)
        gaussian = _utils._reshape_nd(gaussian, ndim=image.ndim, axis=ax)
        result *= gaussian

    return result


def fourier_shift(image, shift, n=-1, axis=-1):
    """
    Multi-dimensional fourier shift filter.

    The array is multiplied with the fourier transform of a shift operation.

    Parameters
    ----------
    image : array_like
        The input image.
    shift : float or sequence
        The size of the box used for filtering.
        If a float, `shift` is the same for all axes. If a sequence, `shift`
        has to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the image is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the image is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.

    Returns
    -------
    fourier_shift : Dask Array

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import matplotlib.pyplot as plt
    >>> import numpy.fft
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = misc.ascent()
    >>> image = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_shift(image, shift=200)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """

    if issubclass(image.dtype.type, numbers.Real):
        image = image.astype(complex)

    # Validate and normalize arguments
    image, shift, n, axis = _utils._norm_args(image, shift, n=n, axis=axis)

    # Constants with type converted
    J = image.dtype.type(1j)

    # Get the grid of frequencies
    ang_freq_grid = _utils._get_ang_freq_grid(
        image.shape,
        chunks=image.chunks,
        n=n,
        axis=axis,
        dtype=shift.dtype
    )

    # Apply shift
    result = image.copy()
    for ax, f in enumerate(ang_freq_grid):
        phase_shift = dask.array.exp((-J) * shift[ax] * f)
        phase_shift = _utils._reshape_nd(phase_shift, ndim=image.ndim, axis=ax)
        result *= phase_shift

    return result


def fourier_uniform(image, size, n=-1, axis=-1):
    """
    Multi-dimensional uniform fourier filter.

    The array is multiplied with the fourier transform of a box of given
    size.

    Parameters
    ----------
    image : array_like
        The input image.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the image is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the image is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.

    Returns
    -------
    fourier_uniform : Dask Array
        The filtered image. If `output` is given as a parameter, None is
        returned.

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = misc.ascent()
    >>> image = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_uniform(image, size=20)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """

    # Validate and normalize arguments
    image, size, n, axis = _utils._norm_args(image, size, n=n, axis=axis)

    # Get the grid of frequencies
    freq_grid = _utils._get_freq_grid(
        image.shape,
        chunks=image.chunks,
        n=n,
        axis=axis,
        dtype=size.dtype
    )

    # Compute uniform filter
    result = image.copy()
    for ax, f in enumerate(freq_grid):
        uniform = dask.array.sinc(size[ax] * f)
        uniform = _utils._reshape_nd(uniform, ndim=image.ndim, axis=ax)
        result *= uniform

    return result
