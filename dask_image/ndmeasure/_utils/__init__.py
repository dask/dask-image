# -*- coding: utf-8 -*-
import warnings

import dask
import dask.array as da
import numpy as np


def _norm_input_labels_index(image, label_image=None, index=None):
    """
    Normalize arguments to a standard form.
    """

    image = da.asarray(image)

    if label_image is None:
        label_image = da.ones(
            image.shape, dtype=int, chunks=image.chunks,
        )
        index = da.from_array(np.array(1, dtype=int))
    elif index is None:
        label_image = (label_image > 0).astype(int)
        index = da.from_array(np.array(1, dtype=int))

    label_image = da.asarray(label_image)
    index = da.asarray(index)

    if index.ndim > 1:
        warnings.warn(
            "Having index with dimensionality greater than 1 is undefined.",
            FutureWarning
        )

    if image.shape != label_image.shape:
        raise ValueError(
            "The image and label_image arrays must be the same shape."
        )

    return (image, label_image, index)


def _grouped_sums(values, labels, dtype):
    """Return sorted labels and their partial sums."""
    if labels.size == 0:
        return labels, np.empty(0, dtype=dtype)

    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    return labels[starts], np.add.reduceat(values[order], starts, dtype=dtype)


def _sum_labels_block(image, labels, index, dtype):
    present = np.isin(labels, index)
    return _grouped_sums(image[present], labels[present], dtype)


def _merge_label_sums(blocks):
    labels, values = zip(*blocks)
    return _grouped_sums(np.concatenate(values), np.concatenate(labels),
                         values[0].dtype)


def _select_label_sums(grouped, index, sum_dtype):
    labels, sums = grouped
    positions = np.searchsorted(labels, index)
    candidates = np.flatnonzero(positions < labels.size)
    matches = candidates[labels[positions[candidates]] == index[candidates]]
    result = np.zeros(index.shape, dtype=np.float64)
    result[matches] = sums[positions[matches]].astype(sum_dtype, copy=False)
    return result


def _sum_labels_grouped(image, labels, index):
    """Build a tree of sparse per-block label sums."""
    sum_dtype = np.sum(np.empty(0, dtype=image.dtype)).dtype
    partial_dtype = sum_dtype
    if sum_dtype.kind == "f":
        # Cast to the input sum dtype only after combining partial sums.
        partial_dtype = np.result_type(sum_dtype, np.float64)

    axes = tuple(range(image.ndim))
    _, (image, labels) = da.core.unify_chunks(image, axes, labels, axes)
    index_block = index.rechunk(-1).to_delayed().item()
    sum_block = dask.delayed(_sum_labels_block, pure=True)
    partials = [
        sum_block(image_block, label_block, index_block, partial_dtype)
        for image_block, label_block in zip(
            image.to_delayed().flat, labels.to_delayed().flat
        )
    ]

    merge = dask.delayed(_merge_label_sums, pure=True)
    fan_in = 8
    while len(partials) > 1:
        partials = [
            merge(partials[start:start + fan_in])
            for start in range(0, len(partials), fan_in)
        ]

    select_sums = dask.delayed(_select_label_sums, pure=True)
    return da.from_delayed(
        select_sums(partials[0], index_block, sum_dtype),
        shape=index.shape,
        dtype=np.float64,
    )


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
        da.arange(
            0,
            np.prod(dimensions[i:], dtype=dtype),
            np.prod(dimensions[i + 1:], dtype=dtype),
            dtype=dtype,
            chunks=c
        )
        for i, c in enumerate(chunks)
    ]

    indices = da.blockwise(
        _ravel_shape_indices_kernel, tuple(range(len(indices))),
        *sum([(a, (i,)) for i, a in enumerate(indices)], tuple()),
        dtype=dtype
    )

    return indices


def _argmax(a, positions, shape, dtype):
    """
    Find original array position corresponding to the maximum.
    """

    result = np.empty((1,), dtype=dtype)

    pos_nd = np.unravel_index(positions[np.argmax(a)], shape)
    for i, pos_nd_i in enumerate(pos_nd):
        result["pos"][0, i] = pos_nd_i

    return result[0]


def _argmin(a, positions, shape, dtype):
    """
    Find original array position corresponding to the minimum.
    """

    result = np.empty((1,), dtype=dtype)

    pos_nd = np.unravel_index(positions[np.argmin(a)], shape)
    for i, pos_nd_i in enumerate(pos_nd):
        result["pos"][0, i] = pos_nd_i

    return result[0]


def _center_of_mass(a, positions, shape, dtype):
    """
    Find the center of mass for each ROI.
    """

    result = np.empty((1,), dtype=dtype)

    positions_nd = np.unravel_index(positions, shape)
    a_sum = np.sum(a)

    a_wt_i = np.empty(a.shape)
    for i, pos_nd_i in enumerate(positions_nd):
        a_wt_sum_i = np.multiply(a, pos_nd_i, out=a_wt_i).sum()
        result["com"][0, i] = a_wt_sum_i / a_sum

    return result[0]


def _extrema(a, positions, shape, dtype):
    """
    Find minimum and maximum as well as positions for both.
    """

    result = np.empty((1,), dtype=dtype)

    int_min_pos = np.argmin(a)
    int_max_pos = np.argmax(a)

    result["min_val"] = a[int_min_pos]
    result["max_val"] = a[int_max_pos]

    min_pos_nd = np.unravel_index(positions[int_min_pos], shape)
    max_pos_nd = np.unravel_index(positions[int_max_pos], shape)
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

    return np.histogram(image, bins, (min, max))[0]


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

    result = np.empty((1,), dtype=out_dtype)

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

    return da.from_delayed(
        _labeled_comprehension_delayed(func, out_dtype, default, a, positions),
        (1,),
        out_dtype
    )
