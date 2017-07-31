# -*- coding: utf-8 -*-


import functools
import itertools
import numbers

import numpy

import dask.array

from . import _pycompat


def _asarray(a):
    """
    Creates a Dask array based on ``a``.

    Parameters
    ----------
    a : array-like
        Object to convert to a Dask Array.

    Returns
    -------
    a : Dask Array
    """

    if not isinstance(a, dask.array.Array):
        a = numpy.asarray(a)
        a = dask.array.from_array(a, a.shape)

    return a


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
        for i in _pycompat.irange(len(dimensions)):
            s = len(dimensions) * [None]
            s[i] = slice(None)
            s = tuple(s)

            r = dask.array.arange(dimensions[i], dtype=dtype, chunks=chunks[i])
            r = r[s]

            for j in itertools.chain(_pycompat.irange(i),
                                     _pycompat.irange(i + 1, len(dimensions))):
                r = r.repeat(dimensions[j], axis=j)

            grid.append(r)

    if grid:
        grid = dask.array.stack(grid)
    else:
        grid = dask.array.empty(
            (len(dimensions),) + dimensions, dtype=dtype, chunks=(1,) + chunks
        )

    return grid


def _isnonzero_vec(v):
    return bool(numpy.count_nonzero(v))


_isnonzero_vec = numpy.vectorize(_isnonzero_vec, otypes=[bool])


def _isnonzero(a):
    try:
        numpy.zeros(tuple(), dtype=a.dtype).astype(bool)
    except ValueError:
        return a.map_blocks(_isnonzero_vec, dtype=bool)
    else:
        return a.astype(bool)


@functools.wraps(numpy.argwhere)
def _argwhere(a):
    a = _asarray(a)

    nz = _isnonzero(a).flatten()

    ind = _indices(a.shape, dtype=numpy.int64, chunks=a.chunks)
    if ind.ndim > 1:
        ind = dask.array.stack(
            [ind[i].ravel() for i in _pycompat.irange(len(ind))], axis=1
        )
    ind = _compress(nz, ind, 0)

    return ind


@functools.wraps(numpy.compress)
def _compress(condition, a, axis=None):
    condition = _asarray(condition)
    a = _asarray(a)

    if condition.ndim > 1:
        raise ValueError("condition must be 1-D.")

    if axis is None:
        axis = 0
        a = a.flatten()

    if not isinstance(axis, numbers.Integral):
        raise ValueError("axis must be an integer.")

    if (axis < -a.ndim) or (axis >= a.ndim):
        raise ValueError("axis is out of bounds.")

    if len(condition) > a.shape[axis]:
        raise IndexError("condition is too long.")

    axes = tuple(_pycompat.irange(a.ndim))

    # Shrink `axis` in `a` to the length of `condition`.
    sl = tuple(
        slice(0, len(condition), 1) if i == axis else slice(None) for i in axes
    )
    a = a[sl]

    r = dask.array.atop(
        numpy.compress, axes,
        condition, (axis,),
        a, axes,
        axis=axis,
        dtype=a.dtype,
    )
    r._chunks = (
        r.chunks[:axis] +
        (len(r.chunks[axis]) * (numpy.nan,),) +
        r.chunks[axis+1:]
    )

    return r
