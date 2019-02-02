# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


import collections
import functools
from warnings import warn

import numpy
import scipy.ndimage

import dask.array

from .. import _pycompat
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

    out_dtype = numpy.dtype([("com", input.dtype, (input.ndim,))])
    default_1d = numpy.full((1,), numpy.nan, dtype=out_dtype)

    func = functools.partial(
        _utils._center_of_mass, shape=input.shape, dtype=out_dtype
    )
    com_lbl = labeled_comprehension(
        input, labels, index,
        func, out_dtype, default_1d[0], pass_positions=True
    )
    com_lbl = com_lbl["com"]

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

    out_dtype = numpy.dtype([
        ("min_val", input.dtype),
        ("max_val", input.dtype),
        ("min_pos", numpy.dtype(numpy.int), input.ndim),
        ("max_pos", numpy.dtype(numpy.int), input.ndim)
    ])
    default_1d = numpy.zeros((1,), dtype=out_dtype)

    func = functools.partial(
        _utils._extrema, shape=input.shape, dtype=out_dtype
    )
    extrema_lbl = labeled_comprehension(
        input, labels, index,
        func, out_dtype, default_1d[0], pass_positions=True
    )
    extrema_lbl = collections.OrderedDict([
        (k, extrema_lbl[k])
        for k in ["min_val", "max_val", "min_pos", "max_pos"]
    ])

    for pos_key in ["min_pos", "max_pos"]:
        pos_nd = extrema_lbl[pos_key]

        if index.ndim == 0:
            pos_nd = dask.array.squeeze(pos_nd)
        elif index.ndim > 1:
            pos_nd = pos_nd.reshape(
                (int(numpy.prod(pos_nd.shape[:-1])), pos_nd.shape[-1])
            )

        extrema_lbl[pos_key] = pos_nd

    result = tuple(extrema_lbl.values())

    return result


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

    func = functools.partial(_utils._histogram, min=min, max=max, bins=bins)
    result = labeled_comprehension(input, labels, index, func, object, None)

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

    input = dask.array.asarray(input)

    if not all([len(c) == 1 for c in input.chunks]):
        warn(
            "``input`` does not have 1 chunk in all dimensions; "
            "it will be consolidated first",
            RuntimeWarning
        )

    label_func = dask.delayed(scipy.ndimage.label, nout=2)
    label, num_features = label_func(input, structure)

    label = dask.array.from_delayed(
        label,
        input.shape,
        numpy.int32
    )

    num_features = dask.array.from_delayed(
        num_features,
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
    default_1d = numpy.full((1,), default, dtype=out_dtype)

    pass_positions = bool(pass_positions)

    args = (input,)
    if pass_positions:
        positions = _utils._ravel_shape_indices(
            input.shape, chunks=input.chunks
        )
        args = (input, positions)

    result = numpy.empty(index.shape, dtype=object)
    for i in numpy.ndindex(index.shape):
        lbl_mtch_i = (labels == index[i])
        args_lbl_mtch_i = tuple(e[lbl_mtch_i] for e in args)
        result[i] = _utils._labeled_comprehension_func(
            func, out_dtype, default_1d, *args_lbl_mtch_i
        )

    for i in _pycompat.irange(result.ndim - 1, -1, -1):
        result2 = result[..., 0]
        for j in numpy.ndindex(index.shape[:i]):
            result2[j] = dask.array.stack(result[j].tolist(), axis=0)
        result = result2
    result = result[()][..., 0]

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

    out_dtype = numpy.dtype([("pos", int, (input.ndim,))])
    default_1d = numpy.zeros((1,), dtype=out_dtype)

    func = functools.partial(
        _utils._argmax, shape=input.shape, dtype=out_dtype
    )
    max_pos_lbl = labeled_comprehension(
        input, labels, index,
        func, out_dtype, default_1d[0], pass_positions=True
    )
    max_pos_lbl = max_pos_lbl["pos"]

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

    nan = numpy.float64(numpy.nan)

    mean_lbl = labeled_comprehension(
        input, labels, index, numpy.mean, numpy.float64, nan
    )

    return mean_lbl


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

    out_dtype = numpy.dtype([("pos", int, (input.ndim,))])
    default_1d = numpy.zeros((1,), dtype=out_dtype)

    func = functools.partial(
        _utils._argmin, shape=input.shape, dtype=out_dtype
    )
    min_pos_lbl = labeled_comprehension(
        input, labels, index,
        func, out_dtype, default_1d[0], pass_positions=True
    )
    min_pos_lbl = min_pos_lbl["pos"]

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

    nan = numpy.float64(numpy.nan)

    std_lbl = labeled_comprehension(
        input, labels, index, numpy.std, numpy.float64, nan
    )

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

    nan = numpy.float64(numpy.nan)

    var_lbl = labeled_comprehension(
        input, labels, index, numpy.var, numpy.float64, nan
    )

    return var_lbl
