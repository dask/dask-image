# -*- coding: utf-8 -*-

"""
Content here is borrowed from our contributions to Dask.
"""


import numpy

import dask.array


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
    chunks = dask.array.core.normalize_chunks(chunks, (n,))

    n_1 = n + 1
    n_2 = n_1 // 2

    s = dask.array.linspace(0, 1, n_1, chunks=(chunks[0] + (1,),))

    l, r = s[:n_2], s[n_2:-1]

    a = l
    if len(r):
        a = dask.array.concatenate([l, r - 1])

    if a.chunks != chunks:
        a = a.rechunk(chunks)

    a /= d

    return a


_sinc = dask.array.ufunc.wrap_elemwise(numpy.sinc)
