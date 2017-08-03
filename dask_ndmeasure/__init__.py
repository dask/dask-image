# -*- coding: utf-8 -*-

from __future__ import division


__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import itertools
import operator

import numpy

import dask.array

from . import _compat
from . import _pycompat
from . import _utils


def center_of_mass(input, labels=None, index=None):
    """
    Calculate the center of mass of the values of an array at labels.

    Parameters
    ----------
    input : ndarray
        Data from which to calculate center-of-mass. The masses can either
        be positive or negative.
    labels : ndarray, optional
        Labels for objects in `input`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `input`.
    index : int or sequence of ints, optional
        Labels for which to calculate centers-of-mass. If not specified,
        all labels greater than zero are used.  Only used with `labels`.

    Returns
    -------
    center_of_mass : tuple, or list of tuples
        Coordinates of centers-of-mass.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    # SciPy transposes these for some reason.
    # So we do the same thing here.
    # This only matters if index is some array.
    index = index.T

    input_mtch_sum = sum(input, labels, index)

    input_i = _compat._indices(
        input.shape, dtype=numpy.int64, chunks=input.chunks
    )

    input_i_wt = input[None] * input_i

    input_i_wt_mtch_sum = []
    for i in _pycompat.irange(len(input_i_wt)):
        input_i_wt_mtch_sum.append(sum(input_i_wt[i], labels, index))
    input_i_wt_mtch_sum = dask.array.stack(input_i_wt_mtch_sum, axis=-1)

    com_lbl = input_i_wt_mtch_sum / input_mtch_sum[..., None]

    return com_lbl


def extrema(input, labels=None, index=None):
    """
    Calculate the minimums and maximums of the values of an array
    at labels, along with their positions.

    Parameters
    ----------
    input : ndarray
        Nd-image data to process.
    labels : ndarray, optional
        Labels of features in input.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero `labels` are used.

    Returns
    -------
    minimums, maximums : int or ndarray
        Values of minimums and maximums in each feature.
    min_positions, max_positions : tuple or list of tuples
        Each tuple gives the n-D coordinates of the corresponding minimum
        or maximum.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    min_lbl = minimum(
        input, labels, index
    )
    max_lbl = maximum(
        input, labels, index
    )
    min_pos_lbl = minimum_position(
        input, labels, index
    )
    max_pos_lbl = maximum_position(
        input, labels, index
    )

    return min_lbl, max_lbl, min_pos_lbl, max_pos_lbl


def labeled_comprehension(input,
                          labels,
                          index,
                          func,
                          out_dtype,
                          default,
                          pass_positions=False):
    """
    Roughly equivalent to [func(input[labels == i]) for i in index].

    Sequentially applies an arbitrary function (that works on array_like input)
    to subsets of an n-D image array specified by `labels` and `index`.
    The option exists to provide the function with positional parameters as the
    second argument.

    Parameters
    ----------
    input : array_like
        Data from which to select `labels` to process.
    labels : array_like or None
        Labels to objects in `input`.
        If not None, array must be same shape as `input`.
        If None, `func` is applied to raveled `input`.
    index : int, sequence of ints or None
        Subset of `labels` to which to apply `func`.
        If a scalar, a single value is returned.
        If None, `func` is applied to all non-zero values of `labels`.
    func : callable
        Python function to apply to `labels` from `input`.
    out_dtype : dtype
        Dtype to use for `result`.
    default : int, float or None
        Default return value when a element of `index` does not exist
        in `labels`.
    pass_positions : bool, optional
        If True, pass linear indices to `func` as a second argument.
        Default is False.

    Returns
    -------
    result : ndarray
        Result of applying `func` to each of `labels` to `input` in `index`.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )
    out_dtype = numpy.dtype(out_dtype)
    default = out_dtype.type(default)
    pass_positions = bool(pass_positions)

    lbl_mtch = _utils._get_label_matches(labels, index)

    args = (input,)
    if pass_positions:
        positions = _utils._ravel_shape_indices(
            input.shape, dtype=numpy.int64, chunks=input.chunks
        )
        args = (input, positions)

    index_ranges = [_pycompat.irange(e) for e in index.shape]

    result = numpy.empty(index.shape, dtype=object)
    for i in itertools.product(*index_ranges):
        lbl_mtch_i = lbl_mtch[i]
        args_lbl_mtch_i = tuple(e[lbl_mtch_i] for e in args)
        result[i] = _utils._labeled_comprehension_func(
            func, out_dtype, default, *args_lbl_mtch_i
        )

    for i in _pycompat.irange(result.ndim - 1, -1, -1):
        index_ranges_i = itertools.product(*(index_ranges[:i]))
        result2 = result[..., 0]
        for j in index_ranges_i:
            result2[j] = dask.array.stack(result[j].tolist(), axis=0)
        result = result2
    result = result[()]

    return result


def maximum(input, labels=None, index=None):
    """
    Calculate the maximum of the values of an array over labeled regions.

    Parameters
    ----------
    input : ndarray
        Array_like of values. For each region specified by `labels`, the
        maximal values of `input` over the region is computed.
    labels : ndarray, optional
        An array of integers marking different regions over which the
        maximum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the maximum
        over the whole array is returned.
    index : array-like, optional
        A list of region labels that are taken into account for computing the
        maxima. If index is None, the maximum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    output : array of floats
        List of maxima of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the maximal value of `input` if `labels` is None,
        and the maximal value of elements where `labels` is greater than zero
        if `index` is None.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    return labeled_comprehension(
        input, labels, index, numpy.max, input.dtype, input.dtype.type(0)
    )


def maximum_position(input, labels=None, index=None):
    """
    Find the positions of the maximums of the values of an array at labels.

    For each region specified by `labels`, the position of the maximum
    value of `input` within the region is returned.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the maximum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first maximum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.
    index : array_like, optional
        A list of region labels that are taken into account for finding the
        location of the maxima.  If `index` is None, the first maximum
        over all elements where `labels` is non-zero is returned.

        The `index` argument only works when `labels` is specified.

    Returns
    -------
    output : array of ints
        Array of ints that specify the location of maxima of
        `input` over the regions determined by `labels` and whose index
        is in `index`.

        If `index` or `labels` are not specified, an array of ints is
        returned specifying the location of the ``first`` maximal value
        of `input`.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    if index.shape:
        index = index.flatten()

    indices = _utils._ravel_shape_indices(
        input.shape, dtype=numpy.int64, chunks=input.chunks
    )

    max_lbl = maximum(input, labels=labels, index=index)

    lbl_mtch = _utils._get_label_matches(labels, index)
    input_mtch = dask.array.where(
        lbl_mtch, input[index.ndim * (None,)], numpy.nan
    )

    max_lbl_mask = operator.eq(
        max_lbl[max_lbl.ndim * (slice(None),) + input.ndim * (None,)],
        input_mtch
    )
    max_lbl_mask_any = max_lbl_mask.any(
        axis=tuple(_pycompat.irange(index.ndim, max_lbl_mask.ndim))
    )
    max_lbl_indices = dask.array.where(
        max_lbl_mask, indices, numpy.nan
    )

    max_1dpos_lbl = dask.array.where(
        max_lbl_mask_any,
        dask.array.nanmin(
            max_lbl_indices,
            axis=tuple(_pycompat.irange(index.ndim, max_lbl_indices.ndim))
        ),
        0
    ).astype(numpy.int64)

    max_pos_lbl = []
    max_1dpos_lbl_rem = max_1dpos_lbl
    if not max_1dpos_lbl_rem.ndim:
        max_1dpos_lbl_rem = max_1dpos_lbl_rem[None]
    for i in _pycompat.irange(input.ndim):
        d = numpy.int64(numpy.prod(input.shape[i + 1:]))
        max_pos_lbl.append(max_1dpos_lbl_rem // d)
        max_1dpos_lbl_rem %= d
    max_pos_lbl = dask.array.stack(max_pos_lbl, axis=1)

    if index.shape == tuple():
        max_pos_lbl = dask.array.squeeze(max_pos_lbl)

    return max_pos_lbl


def mean(input, labels=None, index=None):
    """
    Calculate the mean of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array on which to compute the mean of elements over distinct
        regions.
    labels : array_like, optional
        Array of labels of same shape, or broadcastable to the same shape as
        `input`. All elements sharing the same label form one region over
        which the mean of the elements is computed.
    index : int or sequence of ints, optional
        Labels of the objects over which the mean is to be computed.
        Default is None, in which case the mean for all values where label is
        greater than 0 is calculated.

    Returns
    -------
    out : array_like
        Sequence of same length as `index`, with the mean of the different
        regions labeled by the labels in `index`.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    input_sum = sum(input, labels, index)
    input_norm = sum(
        dask.array.ones(input.shape, dtype=input.dtype, chunks=input.chunks),
        labels,
        index
    )

    com_lbl = input_sum / input_norm

    return com_lbl


def minimum(input, labels=None, index=None):
    """
    Calculate the minimum of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        minimal values of `input` over the region is computed.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        minimum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the minimum
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        minima. If index is None, the minimum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    minimum : array of floats
        List of minima of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the minimal value of `input` if `labels` is None,
        and the minimal value of elements where `labels` is greater than zero
        if `index` is None.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    return labeled_comprehension(
        input, labels, index, numpy.min, input.dtype, input.dtype.type(0)
    )


def minimum_position(input, labels=None, index=None):
    """
    Find the positions of the minimums of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the minimum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first minimum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.
    index : array_like, optional
        A list of region labels that are taken into account for finding the
        location of the minima. If `index` is None, the ``first`` minimum
        over all elements where `labels` is non-zero is returned.

        The `index` argument only works when `labels` is specified.

    Returns
    -------
    output : list of tuples of ints
        Tuple of ints or list of tuples of ints that specify the location
        of minima of `input` over the regions determined by `labels` and
        whose index is in `index`.

        If `index` or `labels` are not specified, a tuple of ints is
        returned specifying the location of the first minimal value of `input`.

    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    if index.shape:
        index = index.flatten()

    indices = _utils._ravel_shape_indices(
        input.shape, dtype=numpy.int64, chunks=input.chunks
    )

    min_lbl = minimum(input, labels=labels, index=index)

    lbl_mtch = _utils._get_label_matches(labels, index)
    input_mtch = dask.array.where(
        lbl_mtch, input[index.ndim * (None,)], numpy.nan
    )

    min_lbl_mask = operator.eq(
        min_lbl[min_lbl.ndim * (slice(None),) + input.ndim * (None,)],
        input_mtch
    )
    min_lbl_mask_any = min_lbl_mask.any(
        axis=tuple(_pycompat.irange(index.ndim, min_lbl_mask.ndim))
    )
    min_lbl_indices = dask.array.where(
        min_lbl_mask, indices, numpy.nan
    )

    min_1dpos_lbl = dask.array.where(
        min_lbl_mask_any,
        dask.array.nanmin(
            min_lbl_indices,
            axis=tuple(_pycompat.irange(index.ndim, min_lbl_indices.ndim))
        ),
        0
    ).astype(numpy.int64)

    min_pos_lbl = []
    min_1dpos_lbl_rem = min_1dpos_lbl
    if not min_1dpos_lbl_rem.ndim:
        min_1dpos_lbl_rem = min_1dpos_lbl_rem[None]
    for i in _pycompat.irange(input.ndim):
        d = numpy.int64(numpy.prod(input.shape[i + 1:]))
        min_pos_lbl.append(min_1dpos_lbl_rem // d)
        min_1dpos_lbl_rem %= d
    min_pos_lbl = dask.array.stack(min_pos_lbl, axis=1)

    if index.shape == tuple():
        min_pos_lbl = dask.array.squeeze(min_pos_lbl)

    return min_pos_lbl


def standard_deviation(input, labels=None, index=None):
    """
    Calculate the standard deviation of the values of an n-D image array,
    optionally at specified sub-regions.

    Parameters
    ----------
    input : array_like
        Nd-image data to process.
    labels : array_like, optional
        Labels to identify sub-regions in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        `labels` to include in output.  If None (default), all values where
        `labels` is non-zero are used.

    Returns
    -------
    standard_deviation : float or ndarray
        Values of standard deviation, for each sub-region if `labels` and
        `index` are specified.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    std_lbl = dask.array.sqrt(variance(input, labels, index))

    return std_lbl


def sum(input, labels=None, index=None):
    """
    Calculate the sum of the values of the array.

    Parameters
    ----------
    input : array_like
        Values of `input` inside the regions defined by `labels`
        are summed together.
    labels : array_like of ints, optional
        Assign labels to the values of the array. Has to have the same shape as
        `input`.
    index : array_like, optional
        A single label number or a sequence of label numbers of
        the objects to be measured.

    Returns
    -------
    sum : ndarray or scalar
        An array of the sums of values of `input` inside the regions defined
        by `labels` with the same shape as `index`. If 'index' is None or scalar,
        a scalar is returned.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    lbl_mtch = _utils._get_label_matches(labels, index)

    input_mtch = dask.array.where(
        lbl_mtch, input[index.ndim * (None,)], input.dtype.type(0)
    )

    input_mtch = input_mtch.astype(numpy.float64)

    sum_lbl = input_mtch.sum(
        axis=tuple(_pycompat.irange(index.ndim, input_mtch.ndim))
    )

    return sum_lbl


def variance(input, labels=None, index=None):
    """
    Calculate the variance of the values of an n-D image array, optionally at
    specified sub-regions.

    Parameters
    ----------
    input : array_like
        Nd-image data to process.
    labels : array_like, optional
        Labels defining sub-regions in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        `labels` to include in output.  If None (default), all values where
        `labels` is non-zero are used.

    Returns
    -------
    variance : array-like
        Values of variance, for each sub-region if `labels` and `index` are
        specified.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    input_2_mean = mean(dask.array.square(input), labels, index)
    input_mean_2 = dask.array.square(mean(input, labels, index))

    var_lbl = input_2_mean - input_mean_2

    return var_lbl
