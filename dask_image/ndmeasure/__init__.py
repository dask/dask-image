# -*- coding: utf-8 -*-

from __future__ import division


__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


import itertools
from warnings import warn

import numpy
import scipy.ndimage

import dask.array

from . import _compat
from . import _pycompat
from . import _utils


def center_of_mass(input, labels=None, index=None):
    """
    Find the center of mass over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    center_of_mass : ndarray
        Coordinates of centers-of-mass of ``input`` over the ``index`` selected
        regions from ``labels``.
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
        input.shape, chunks=input.chunks
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
    Find the min and max with positions over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    minimums, maximums, min_positions, max_positions : tuple of ndarrays
        Values and coordinates of minimums and maximums in each feature.
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


def histogram(input,
              min,
              max,
              bins,
              labels=None,
              index=None):
    """
    Find the histogram over an image at specified subregions.

    Histogram calculates the frequency of values in an array within bins
    determined by ``min``, ``max``, and ``bins``. The ``labels`` and ``index``
    keywords can limit the scope of the histogram to specified sub-regions
    within the array.

    Parameters
    ----------
    input : ndarray
        N-D image data
    min : int
        Minimum value of range of histogram bins.
    max : int
        Maximum value of range of histogram bins.
    bins : int
        Number of bins.
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    histogram : ndarray
        Histogram of ``input`` over the ``index`` selected regions from
        ``labels``.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )
    min = int(min)
    max = int(max)
    bins = int(bins)

    lbl_mtch = _utils._get_label_matches(labels, index)

    index_ranges = [_pycompat.irange(e) for e in index.shape]

    result = numpy.empty(index.shape, dtype=object)
    for i in itertools.product(*index_ranges):
        result[i] = _utils._histogram(
            input[lbl_mtch[i]], min, max, bins
        )
    result = result[()]

    return result


def label(input, structure=None):
    """
    Label features in an array.

    Parameters
    ----------
    input : ndarray
        An array-like object to be labeled.  Any non-zero values in ``input``
        are counted as features and zero values are considered the background.
    structure : ndarray, optional
        A structuring element that defines feature connections.
        ``structure`` must be symmetric.  If no structuring element is
        provided, one is automatically generated with a squared connectivity
        equal to one.  That is, for a 2-D ``input`` array, the default
        structuring element is::

            [[0,1,0],
             [1,1,1],
             [0,1,0]]

    Returns
    -------
    label : ndarray or int
        An integer ndarray where each unique feature in ``input`` has a unique
        label in the returned array.
    num_features : int
        How many objects were found.
    """

    input = _compat._asarray(input)

    if not all([len(c) == 1 for c in input.chunks]):
        warn("``input`` does not have 1 chunk in all dimensions; it will be consolidated first", RuntimeWarning)

    result = dask.delayed(scipy.ndimage.label)(input, structure)

    label = dask.array.from_delayed(
        result[0],
        input.shape,
        numpy.int32
    )

    num_features = dask.array.from_delayed(
        result[1],
        tuple(),
        int
    )

    result = (label, num_features)

    return result


def labeled_comprehension(input,
                          labels,
                          index,
                          func,
                          out_dtype,
                          default,
                          pass_positions=False):
    """
    Compute a function over an image at specified subregions.

    Roughly equivalent to [func(input[labels == i]) for i in index].

    Sequentially applies an arbitrary function (that works on array_like input)
    to subsets of an n-D image array specified by ``labels`` and ``index``.
    The option exists to provide the function with positional parameters as the
    second argument.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    func : callable
        Python function to apply to ``labels`` from ``input``.
    out_dtype : dtype
        Dtype to use for ``result``.
    default : int, float or None
        Default return value when a element of ``index`` does not exist
        in ``labels``.
    pass_positions : bool, optional
        If True, pass linear indices to ``func`` as a second argument.
        Default is False.

    Returns
    -------
    result : ndarray
        Result of applying ``func`` on ``input`` over the ``index`` selected
        regions from ``labels``.
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
            input.shape, chunks=input.chunks
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
    Find the maxima over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    maxima : ndarray
        Maxima of ``input`` over the ``index`` selected regions from
        ``labels``.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    return labeled_comprehension(
        input, labels, index, numpy.max, input.dtype, input.dtype.type(0)
    )


def maximum_position(input, labels=None, index=None):
    """
    Find the positions of maxima over an image at specified subregions.

    For each region specified by ``labels``, the position of the maximum
    value of ``input`` within the region is returned.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    maxima_positions : ndarray
        Maxima positions of ``input`` over the ``index`` selected regions from
        ``labels``.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    if index.shape:
        index = index.flatten()

    max_1dpos_lbl = labeled_comprehension(
        input, labels, index, _utils._argmax, int, 0, pass_positions=True
    )

    if not max_1dpos_lbl.ndim:
        max_1dpos_lbl = max_1dpos_lbl[None]

    max_pos_lbl = []
    max_1dpos_lbl_rem = max_1dpos_lbl
    for i in _pycompat.irange(input.ndim):
        d = int(numpy.prod(input.shape[i + 1:]))
        max_pos_lbl.append(max_1dpos_lbl_rem // d)
        max_1dpos_lbl_rem %= d
    max_pos_lbl = dask.array.stack(max_pos_lbl, axis=1)

    if index.shape == tuple():
        max_pos_lbl = dask.array.squeeze(max_pos_lbl)

    return max_pos_lbl


def mean(input, labels=None, index=None):
    """
    Find the mean over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    means : ndarray
        Mean of ``input`` over the ``index`` selected regions from ``labels``.
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


def median(input, labels=None, index=None):
    """
    Find the median over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    medians : ndarray
        Median of ``input`` over the ``index`` selected regions from
        ``labels``.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    nan = numpy.float64(numpy.nan)

    return labeled_comprehension(
        input, labels, index, numpy.median, numpy.float64, nan
    )


def minimum(input, labels=None, index=None):
    """
    Find the minima over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    minima : ndarray
        Minima of ``input`` over the ``index`` selected regions from
        ``labels``.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    return labeled_comprehension(
        input, labels, index, numpy.min, input.dtype, input.dtype.type(0)
    )


def minimum_position(input, labels=None, index=None):
    """
    Find the positions of minima over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    minima_positions : ndarray
        Maxima positions of ``input`` over the ``index`` selected regions from
        ``labels``.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    if index.shape:
        index = index.flatten()

    min_1dpos_lbl = labeled_comprehension(
        input, labels, index, _utils._argmin, int, 0, pass_positions=True
    )

    if not min_1dpos_lbl.ndim:
        min_1dpos_lbl = min_1dpos_lbl[None]

    min_pos_lbl = []
    min_1dpos_lbl_rem = min_1dpos_lbl
    for i in _pycompat.irange(input.ndim):
        d = int(numpy.prod(input.shape[i + 1:]))
        min_pos_lbl.append(min_1dpos_lbl_rem // d)
        min_1dpos_lbl_rem %= d
    min_pos_lbl = dask.array.stack(min_pos_lbl, axis=1)

    if index.shape == tuple():
        min_pos_lbl = dask.array.squeeze(min_pos_lbl)

    return min_pos_lbl


def standard_deviation(input, labels=None, index=None):
    """
    Find the standard deviation over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    standard_deviation : ndarray
        Standard deviation of ``input`` over the ``index`` selected regions
        from ``labels``.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    std_lbl = dask.array.sqrt(variance(input, labels, index))

    return std_lbl


def sum(input, labels=None, index=None):
    """
    Find the sum over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    sum : ndarray
        Sum of ``input`` over the ``index`` selected regions from ``labels``.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    sum_lbl = labeled_comprehension(
        input, labels, index, numpy.sum, numpy.float64, numpy.float64(0)
    )

    return sum_lbl


def variance(input, labels=None, index=None):
    """
    Find the variance over an image at specified subregions.

    Parameters
    ----------
    input : ndarray
        N-D image data
    labels : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``labels`` are used.

        The ``index`` argument only works when ``labels`` is specified.

    Returns
    -------
    variance : ndarray
        Variance of ``input`` over the ``index`` selected regions from
        ``labels``.
    """

    input, labels, index = _utils._norm_input_labels_index(
        input, labels, index
    )

    input_2_mean = mean(dask.array.square(input), labels, index)
    input_mean_2 = dask.array.square(mean(input, labels, index))

    var_lbl = input_2_mean - input_mean_2

    return var_lbl
