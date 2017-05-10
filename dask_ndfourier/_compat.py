# -*- coding: utf-8 -*-


import itertools

import numpy

import dask.array


def _indices(dimensions, dtype=int, chunks=None):
    """
    Implements NumPy's ``indices`` for Dask Arrays.
    Generates a grid of indices covering the dimensions provided.
    The final array has the shape ``(len(dimensions), *dimensions)``. The
    chunks are used to specify the chunking for axis 1 up to
    ``len(dimensions)``. The 0th axis always has chunks of length 1.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the index grid.
    dtype : dtype, optional
        Type to use for the array. Default is ``int``.
    chunks : sequence of ints
        The number of samples on each block. Note that the last block will
        have fewer samples if ``len(array) % chunks != 0``.

    Returns
    -------
    grid : dask array

    Notes
    -----
    Borrowed from my Dask Array contribution.
    """
    if chunks is None:
        raise ValueError("Must supply a chunks= keyword argument")

    dimensions = tuple(dimensions)
    dtype = numpy.dtype(dtype)
    chunks = tuple(chunks)

    if len(dimensions) != len(chunks):
        raise ValueError("Need one more chunk than dimensions.")

    grid = []
    if numpy.prod(dimensions):
        for i in range(len(dimensions)):
            s = len(dimensions) * [None]
            s[i] = slice(None)
            s = tuple(s)

            r = dask.array.arange(dimensions[i], dtype=dtype, chunks=chunks[i])
            r = r[s]

            for j in itertools.chain(range(i), range(i + 1, len(dimensions))):
                r = r.repeat(dimensions[j], axis=j)

            grid.append(r)

    if grid:
        grid = dask.array.stack(grid)
    else:
        grid = dask.array.empty(
            (len(dimensions),) + dimensions, dtype=dtype, chunks=(1,) + chunks
        )

    return grid


def _fftfreq(n, d=1.0, chunks=None):
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    grid : dask array

    Examples
    --------
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = np.fft.fft(signal)
    >>> n = signal.size
    >>> timestep = 0.1
    >>> freq = np.fft.fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 ,  3.75, -5.  , -3.75, -2.5 , -1.25])

    Notes
    -----
    Borrowed from my Dask Array contribution.
    """
    n = int(n)
    chunks = dask.array.core.normalize_chunks(chunks, (n,))[0] + (1,)

    n_1 = n + 1
    n_2 = n_1 // 2

    l = dask.array.linspace(0, 1, n_1, chunks=chunks)
    r = dask.array.concatenate([l[:n_2], l[n_2:-1] - 1])
    r /= d

    return r
