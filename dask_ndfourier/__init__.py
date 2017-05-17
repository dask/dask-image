# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import numbers

import dask.array

from dask_ndfourier import _compat
from dask_ndfourier import _utils


def fourier_gaussian(input, sigma, n=-1, axis=-1):
    """
    Multi-dimensional Gaussian fourier filter.

    The array is multiplied with the fourier transform of a Gaussian
    kernel.

    Parameters
    ----------
    input : array_like
        The input array.
    sigma : float or sequence
        The sigma of the Gaussian kernel. If a float, `sigma` is the same for
        all axes. If a sequence, `sigma` has to contain one value for each
        axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
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
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_gaussian(input_, sigma=4)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    """

    # Validate and normalize arguments
    input, sigma, n, axis = _utils._norm_args(input, sigma, n=n, axis=axis)

    # Compute frequencies
    ang_freq_grid = _utils._get_ang_freq_grid(
        input.shape,
        chunks=input.chunks,
        dtype=sigma.dtype
    )

    # Compute Fourier transformed Gaussian
    gaussian = dask.array.exp(
        - dask.array.tensordot(sigma ** 2, ang_freq_grid ** 2, axes=1) / 2
    )

    result = input * gaussian

    return result


def fourier_shift(input, shift, n=-1, axis=-1):
    """
    Multi-dimensional fourier shift filter.

    The array is multiplied with the fourier transform of a shift operation.

    Parameters
    ----------
    input : array_like
        The input array.
    shift : float or sequence
        The size of the box used for filtering.
        If a float, `shift` is the same for all axes. If a sequence, `shift`
        has to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
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
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_shift(input_, shift=200)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """

    if issubclass(input.dtype.type, numbers.Real):
        input = input.astype(complex)

    # Validate and normalize arguments
    input, shift, n, axis = _utils._norm_args(input, shift, n=n, axis=axis)

    # Constants with type converted
    J = input.dtype.type(1j)

    # Get the grid of frequencies
    ang_freq_grid = _utils._get_ang_freq_grid(
        input.shape,
        chunks=input.chunks,
        dtype=shift.dtype
    )

    # Apply shift
    phase_shift = dask.array.exp(
        - J * dask.array.tensordot(shift, ang_freq_grid, axes=1)
    )
    result = input * phase_shift

    return result


def fourier_uniform(input, size, n=-1, axis=-1):
    """
    Multi-dimensional uniform fourier filter.

    The array is multiplied with the fourier transform of a box of given
    size.

    Parameters
    ----------
    input : array_like
        The input array.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.

    Returns
    -------
    fourier_uniform : Dask Array
        The filtered input. If `output` is given as a parameter, None is
        returned.

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = misc.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_uniform(input_, size=20)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """

    # Validate and normalize arguments
    input, size, n, axis = _utils._norm_args(input, size, n=n, axis=axis)

    # Get the grid of frequencies
    freq_grid = _utils._get_freq_grid(
        input.shape,
        chunks=input.chunks,
        dtype=size.dtype
    )

    # Compute uniform filter
    uniform = _compat._sinc(
        size[(slice(None),) + input.ndim * (None,)] * freq_grid
    )
    uniform = dask.array.prod(uniform, axis=0)

    result = input * uniform

    return result
