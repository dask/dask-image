# -*- coding: utf-8 -*-


import numbers

import numpy as np

import dask.array as da


def _get_freq_grid(shape, chunks, dtype=float):
    assert len(shape) == len(chunks)

    shape = tuple(shape)
    dtype = np.dtype(dtype).type

    assert (issubclass(dtype, numbers.Real) and
            not issubclass(dtype, numbers.Integral))

    freq_grid = [
        da.fft.fftfreq(s, chunks=c).astype(dtype)
        for s, c in zip(shape, chunks)
    ]
    freq_grid = da.meshgrid(*freq_grid, indexing="ij")
    freq_grid = da.stack(freq_grid)

    return freq_grid


def _get_ang_freq_grid(shape, chunks, dtype=float):
    dtype = np.dtype(dtype).type

    assert (issubclass(dtype, numbers.Real) and
            not issubclass(dtype, numbers.Integral))

    pi = dtype(np.pi)

    freq_grid = _get_freq_grid(shape, chunks, dtype=dtype)
    ang_freq_grid = (2 * pi) * freq_grid

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

    if n != -1:
        raise NotImplementedError(
            "Currently `n` other than -1 is unsupported."
        )

    return (a, s, n, axis)
