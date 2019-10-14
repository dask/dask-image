# -*- coding: utf-8 -*-


from __future__ import division

import warnings

import numpy

import dask
import dask.array


try:
    from dask.array import blockwise as da_blockwise
except ImportError:
    from dask.array import atop as da_blockwise


def _norm_input_labels_index(image, labels=None, index=None):
    """
    Normalize arguments to a standard form.
    """

    image = dask.array.asarray(image)

    if labels is None:
        labels = dask.array.ones(image.shape, dtype=int, chunks=image.chunks)
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

    if image.shape != labels.shape:
        raise ValueError("The image and labels arrays must be the same shape.")

    return (image, labels, index)


def _ravel_shape_indices_kernel(*args):
    args2 = tuple(
        a[i * (None,) + (slice(None),) + (len(args) - i - 1) * (None,)]
        for i, a in enumerate(args)
    )
    return sum(args2)


def _ravel_shape_indices(dimensions, dtype=int, chunks=None):
    """
    Gets the raveled indices shaped like input.
    """

    indices = [
        dask.array.arange(
            0,
            numpy.prod(dimensions[i:], dtype=dtype),
            numpy.prod(dimensions[i + 1:], dtype=dtype),
            dtype=dtype,
            chunks=c
        )
        for i, c in enumerate(chunks)
    ]

    indices = da_blockwise(
        _ravel_shape_indices_kernel, tuple(range(len(indices))),
        *sum([(a, (i,)) for i, a in enumerate(indices)], tuple()),
        dtype=dtype
    )

    return indices


def _argmax(a, positions, shape, dtype):
    """
    Find original array position corresponding to the maximum.
    """

    result = numpy.empty((1,), dtype=dtype)

    pos_nd = numpy.unravel_index(positions[numpy.argmax(a)], shape)
    for i, pos_nd_i in enumerate(pos_nd):
        result["pos"][0, i] = pos_nd_i

    return result[0]


def _argmin(a, positions, shape, dtype):
    """
    Find original array position corresponding to the minimum.
    """

    result = numpy.empty((1,), dtype=dtype)

    pos_nd = numpy.unravel_index(positions[numpy.argmin(a)], shape)
    for i, pos_nd_i in enumerate(pos_nd):
        result["pos"][0, i] = pos_nd_i

    return result[0]


def _center_of_mass(a, positions, shape, dtype):
    """
    Find the center of mass for each ROI.
    """

    result = numpy.empty((1,), dtype=dtype)

    positions_nd = numpy.unravel_index(positions, shape)
    a_sum = numpy.sum(a)

    a_wt_i = numpy.empty(a.shape)
    for i, pos_nd_i in enumerate(positions_nd):
        a_wt_sum_i = numpy.multiply(a, pos_nd_i, out=a_wt_i).sum()
        result["com"][0, i] = a_wt_sum_i / a_sum

    return result[0]


def _extrema(a, positions, shape, dtype):
    """
    Find minimum and maximum as well as positions for both.
    """

    result = numpy.empty((1,), dtype=dtype)

    int_min_pos = numpy.argmin(a)
    int_max_pos = numpy.argmax(a)

    result["min_val"] = a[int_min_pos]
    result["max_val"] = a[int_max_pos]

    min_pos_nd = numpy.unravel_index(positions[int_min_pos], shape)
    max_pos_nd = numpy.unravel_index(positions[int_max_pos], shape)
    for i in range(len(shape)):
        result["min_pos"][0, i] = min_pos_nd[i]
        result["max_pos"][0, i] = max_pos_nd[i]

    return result[0]


def _histogram(image,
               min,
               max,
               bins):
    """
    Delayed wrapping of NumPy's histogram

    Also reformats the arguments.
    """

    return numpy.histogram(image, bins, (min, max))[0]


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

    result = numpy.empty((1,), dtype=out_dtype)

    if a.size:
        if positions is None:
            result[0] = func(a)
        else:
            result[0] = func(a, positions)
    else:
        result[0] = default[0]

    return result


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
        _labeled_comprehension_delayed(func, out_dtype, default, a, positions),
        (1,),
        out_dtype
    )
