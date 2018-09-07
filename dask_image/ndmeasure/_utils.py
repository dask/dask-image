# -*- coding: utf-8 -*-


import operator
import warnings

import numpy

import dask
import dask.array

from .. import _pycompat


def _norm_input_labels_index(input, labels=None, index=None):
    """
    Normalize arguments to a standard form.
    """

    input = dask.array.asarray(input)

    if labels is None:
        labels = dask.array.ones(input.shape, dtype=int, chunks=input.chunks)
        index = dask.array.ones(tuple(), dtype=int, chunks=tuple())
    elif index is None:
        labels = (labels > 0).astype(int)
        index = dask.array.ones(tuple(), dtype=int, chunks=tuple())

    labels = dask.array.asarray(labels)
    index = dask.array.asarray(index)

    if index.ndim > 1:
        warnings.warn(
            "Having index with dimensionality greater than 1 is undefined.",
            FutureWarning
        )

    if input.shape != labels.shape:
        raise ValueError("The input and labels arrays must be the same shape.")

    return (input, labels, index)


def _get_label_matches(labels, index):
    lbl_mtch = operator.eq(
        index[(Ellipsis,) + labels.ndim * (None,)],
        labels[index.ndim * (None,)]
    )

    return lbl_mtch


def _ravel_shape_indices(dimensions, dtype=int, chunks=None):
    """
    Gets the raveled indices shaped like input.
    """

    indices = sum([
        dask.array.arange(
            0,
            numpy.prod(dimensions[i:], dtype=dtype),
            numpy.prod(dimensions[i + 1:], dtype=dtype),
            dtype=dtype,
            chunks=c
        )[i * (None,) + (slice(None),) + (len(dimensions) - i - 1) * (None,)]
        for i, c in enumerate(chunks)
    ])

    return indices


def _unravel_index_kernel(indices, func_kwargs):
    return numpy.stack(numpy.unravel_index(indices, **func_kwargs), axis=1)


def _unravel_index(indices, dims, order='C'):
    """
    Unravels the indices like NumPy's ``unravel_index``.

    Uses NumPy's ``unravel_index`` on Dask Array blocks.
    """

    if dims and indices.size:
        unraveled_indices = indices.map_blocks(
            _unravel_index_kernel,
            dtype=numpy.intp,
            chunks=indices.chunks + ((len(dims),),),
            new_axis=1,
            func_kwargs={"dims": dims, "order": order}
        )
    else:
        unraveled_indices = dask.array.empty(
            (0, len(dims)), dtype=numpy.intp, chunks=1
        )

    return unraveled_indices


def _argmax(a, positions):
    """
    Find original array position corresponding to the maximum.
    """

    return positions[numpy.argmax(a)]


def _argmin(a, positions):
    """
    Find original array position corresponding to the minimum.
    """

    return positions[numpy.argmin(a)]


@dask.delayed
def _histogram(input,
               min,
               max,
               bins):
    """
    Delayed wrapping of NumPy's histogram

    Also reformats the arguments.
    """

    if input.size:
        return numpy.histogram(input, bins, (min, max))[0]
    else:
        return None


@dask.delayed
def _labeled_comprehension_delayed(func,
                                   out_dtype,
                                   default,
                                   a,
                                   positions=None):
    """
    Wrapped delayed labeled comprehension function

    Included in the module for pickling purposes. Also handle cases where
    computation should not occur.
    """

    if a.size:
        if positions is None:
            return out_dtype.type(func(a))
        else:
            return out_dtype.type(func(a, positions))
    else:
        return default


def _labeled_comprehension_func(func,
                                out_dtype,
                                default,
                                a,
                                positions=None):
    """
    Wrapped labeled comprehension function

    Ensures the result is a proper Dask Array and the computation delayed.
    """

    return dask.array.from_delayed(
        _labeled_comprehension_delayed(
            func, out_dtype, default, a, positions
        ),
        tuple(),
        out_dtype
    )
