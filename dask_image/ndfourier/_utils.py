# -*- coding: utf-8 -*-

import functools
import operator
import numbers

import dask.array as da
import numpy as np


def _get_freq_grid(shape, chunks, axis, n, dtype=float):
    assert len(shape) == len(chunks)

    shape = tuple(shape)
    dtype = np.dtype(dtype).type

    assert (issubclass(dtype, numbers.Real) and
            not issubclass(dtype, numbers.Integral))

    axis = axis % len(shape)

    freq_grid = []
    for ax, (s, c) in enumerate(zip(shape, chunks)):
        if axis == ax and n > 0:
            f = da.fft.rfftfreq(n, chunks=c).astype(dtype)
        else:
            f = da.fft.fftfreq(s, chunks=c).astype(dtype)
        freq_grid.append(f)

    freq_grid = da.meshgrid(*freq_grid, indexing="ij", sparse=True)

    return freq_grid


def _get_ang_freq_grid(shape, chunks, axis, n, dtype=float):
    dtype = np.dtype(dtype).type

    assert (issubclass(dtype, numbers.Real) and
            not issubclass(dtype, numbers.Integral))

    pi = dtype(np.pi)

    freq_grid = _get_freq_grid(shape, chunks, axis, n, dtype=dtype)
    ang_freq_grid = tuple((2 * pi) * f for f in freq_grid)

    return ang_freq_grid


def _norm_args(a, s, n=-1, axis=-1):
    if issubclass(a.dtype.type, numbers.Integral):
        a = a.astype(float)

    if isinstance(s, numbers.Number):
        s = np.array(a.ndim * [s])
    elif not isinstance(s, da.Array):
        s = np.array(s)

    if issubclass(s.dtype.type, numbers.Integral):
        s = s.astype(a.real.dtype)
    elif not issubclass(s.dtype.type, numbers.Real):
        raise TypeError("The `s` must contain real value(s).")
    if s.shape != (a.ndim,):
        raise RuntimeError(
            "Shape of `s` must be 1-D and equal to the input's rank."
        )

    if n != -1 and a.shape[axis] != (n // 2 + 1):
        raise NotImplementedError(
            "In the case of real-valued images, it is required that "
            "(n // 2 + 1) == image.shape[axis]."
        )

    return (a, s, n, axis)


def _reshape_nd(arr, ndim, axis):
    """Promote a 1d array to ndim with non-singleton size along axis."""
    nd_shape = (1,) * axis + (arr.size,) + (1,) * (ndim - axis - 1)
    return arr.reshape(nd_shape)
