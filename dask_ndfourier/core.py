# -*- coding: utf-8 -*-


import collections
import itertools
import numbers

import numpy

import dask.array

from dask_ndfourier import _compat

try:
    from itertools import imap
except ImportError:
    imap = map

try:
    irange = xrange
except NameError:
    irange = range


def _get_freq_grid(shape, chunks, dtype=float):
    assert len(shape) == len(chunks)

    shape = tuple(shape)
    ndim = len(shape)
    dtype = numpy.dtype(dtype)

    pi = dtype.type(numpy.pi).real

    freq_grid = []
    for i in irange(ndim):
        sl = ndim * [None]
        sl[i] = slice(None)
        sl = tuple(sl)

        freq_grid_i = _compat._fftfreq(shape[i], chunks=chunks[i])[sl]
        for j in itertools.chain(range(i), range(i + 1, ndim)):
            freq_grid_i = freq_grid_i.repeat(shape[j], axis=j)

        freq_grid.append(freq_grid_i)

    freq_grid = dask.array.stack(freq_grid)
    freq_grid *= 2 * pi

    return freq_grid


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

    # Validate and normalize shift
    if isinstance(shift, numbers.Number):
        shift = input.ndim * [shift]
    elif not isinstance(shift, collections.Sequence):
        raise TypeError("The `shift` must be a number or a sequence.")
    if len(shift) != input.ndim:
        raise RuntimeError(
            "The `shift` must have a length equal to the input's rank."
        )
    if all(imap(lambda i: isinstance(i, numbers.Integral), shift)):
        pass
    elif all(imap(lambda i: isinstance(i, numbers.Real), shift)):
        raise NotImplementedError("Real value(s) unsupported in the `shift`.")
    else:
        raise TypeError("The `shift` must contain integral value(s).")
    shift = numpy.array(shift)

    if n != -1:
        raise NotImplementedError(
            "Currently `n` other than -1 is unsupported."
        )

    # Constants with type converted
    J = input.dtype.type(1j)

    # Get the grid of frequencies
    freq_grid = _get_freq_grid(
        input.shape, dtype=input.dtype, chunks=input.chunks
    )

    # Apply shift
    phase_shift = dask.array.exp(
        - J * dask.array.tensordot(shift, freq_grid, axes=1)
    )
    result = input * phase_shift

    return result
