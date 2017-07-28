# -*- coding: utf-8 -*-


import functools
import itertools

import numpy

import dask.array


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
            [ind[i].ravel() for i in range(len(ind))], axis=1
        )

    axis = 0
    axes = tuple(range(ind.ndim))

    ind = dask.array.atop(
        numpy.compress, axes,
        nz, (axis,),
        ind, axes,
        axis=axis,
        dtype=ind.dtype,
    )
    ind._chunks = (
        ind.chunks[:axis] +
        (len(ind.chunks[axis]) * (numpy.nan,),) +
        ind.chunks[axis+1:]
    )

    return ind
