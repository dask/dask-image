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


def _get_freq_grid(shape, chunks):
    assert len(shape) == len(chunks)

    shape = tuple(shape)
    ndim = len(shape)

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

    return freq_grid


def _get_ang_freq_grid(shape, chunks):
    freq_grid = _get_freq_grid(shape, chunks)
    ang_freq_grid = 2 * numpy.pi * freq_grid

    return ang_freq_grid


def _norm_args(a, s, n=-1, axis=-1):
    if isinstance(s, numbers.Number):
        s = numpy.array(a.ndim * [s])
    elif not isinstance(s, dask.array.Array):
        s = numpy.array(s)

    if not issubclass(s.dtype.type, numbers.Real):
        raise TypeError("The `s` must contain real value(s).")
    if s.shape != (a.ndim,):
        raise RuntimeError(
            "Shape of `s` must be 1-D and equal to the input's rank."
        )

    if n != -1:
        raise NotImplementedError(
            "Currently `n` other than -1 is unsupported."
        )

    return (s, n, axis)
