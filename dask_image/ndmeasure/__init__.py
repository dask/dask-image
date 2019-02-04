# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


import operator
import collections
import functools

import numpy
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
import scipy.ndimage
import scipy.ndimage as ndi
import skimage.util

import dask
import dask.array
import dask.array as da

from .. import _pycompat
from . import _utils


def _get_ndimage_label_dtype():
    return scipy.ndimage.label([1, 0, 1])[0].dtype


LABEL_DTYPE = _get_ndimage_label_dtype()


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


def _relabel_components(array, labeling):
    """Relabel the input array based on correspondences in comp_labels

    Parameters
    ----------
    array : array of int
        The input label array.
    labeling : 1D array of int
        A new labeling, such that ``labeling[i] = j`` implies that
        any element in ``array`` valued ``i`` should be relabeled to ``j``.

    Returns
    -------
    result : array of int, same shape as ``array``
        The relabeled input array.
    """
    result = da.map_blocks(operator.getitem, labeling, array,
                           dtype=labeling.dtype, chunks=array.chunks)
    return result


@dask.delayed
def get_valid_matches(face):
    common_labels = ndi.label(face)[0]
    matching = np.stack((common_labels.ravel(), face.ravel()), axis=1)
    unique_matching = skimage.util.unique_rows(matching)
    valid = np.all(unique_matching, axis=1)
    unique_valid_matching = unique_matching[valid]
    relabeled_unique, relabeled_counts = np.unique(unique_valid_matching[:, 0],
                                                   return_counts=True)
    real_mapped_labels = relabeled_unique[relabeled_counts > 1]
    rows_we_keep = np.in1d(unique_valid_matching[:, 0], real_mapped_labels)
    mapped = unique_valid_matching[rows_we_keep, 1].reshape((-1, 2)).T
    return mapped


@dask.delayed
def csr(i, j, n):
    v = np.ones_like(i)
    mat = sparse.coo_matrix((v, (i, j)), shape=(n, n))
    return mat.tocsr()


def _label_adj_graph(array, structure, nlabels):
    """Adjacency graph of labels between chunks of ``array``.
    """
    faces = chunk_faces(array.chunks, array.shape)
    all_mappings = []
    for face_slice in faces:
        chunky_face = array[face_slice]
        face = chunky_face.rechunk(-1)
        mapped = da.from_delayed(get_valid_matches(face), (2, np.nan),
                                 dtype=LABEL_DTYPE)
        all_mappings.append(mapped)
    all_mappings = da.concatenate(all_mappings, axis=1)
    i, j = all_mappings
    mat = csr(i, j, nlabels + 1)
    return mat


def chunk_faces(chunks, shape):
    """Return slices for two-pixel-wide boundaries between chunks.

    Parameters
    ----------
    chunks : tuple of tuple of int
        The chunk specification of the array.
    shape : tuple of int
        The shape of the array.

    Returns
    -------
    faces : list of tuple of slices
        Each element in this list indexes a face between two chunks.

    Examples
    --------
    >>> a = da.arange(110, chunks=110).reshape((10, 11)).rechunk(5)
    >>> chunk_faces(a.chunks, a.shape)
    [(slice(4, 6, None), slice(0, 5, None)),
     (slice(4, 6, None), slice(5, 10, None)),
     (slice(4, 6, None), slice(10, 11, None)),
     (slice(0, 5, None), slice(4, 6, None)),
     (slice(0, 5, None), slice(9, 11, None)),
     (slice(5, 10, None), slice(4, 6, None)),
     (slice(5, 10, None), slice(9, 11, None))]
    """
    slices = da.core.slices_from_chunks(chunks)
    ndim = len(shape)
    faces = []
    for ax in range(ndim):
        for sl in slices:
            if sl[ax].stop == shape[ax]:
                continue
            slice_to_append = list(sl)
            slice_to_append[ax] = slice(sl[ax].stop - 1, sl[ax].stop + 1)
            faces.append(tuple(slice_to_append))
    return faces


def label(input, structure=None):
    """Label features in an array.

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

    label = dask.delayed(functools.partial(scipy.ndimage.label,
                                           structure=structure), nout=2)
    labeled_blocks = np.empty(input.numblocks, dtype=object)
    total = 0

    for i, index in enumerate(np.ndindex(*input.numblocks)):
        input_block = input.blocks[index]
        labeled_block, n = label(input_block)
        labeled_block = da.from_delayed(labeled_block, shape=input_block.shape,
                                        dtype=LABEL_DTYPE)
        labeled_block += da.where(labeled_block > 0, total, 0)
        labeled_blocks[index] = labeled_block
        total += da.from_delayed(n, shape=(), dtype=LABEL_DTYPE)

    result_array = da.block(labeled_blocks.tolist())

    # _label_adj_graph needs to be defined still; returns a csr_matrix
    correspondences = _label_adj_graph(result_array, structure, total)
    conn_comp = dask.delayed(functools.partial(csgraph.connected_components,
                                               directed=False), nout=2)
    _, comp_labels = conn_comp(correspondences)
    comp_labels = da.from_delayed(comp_labels, shape=(np.nan,), dtype=int)
    relabeled_result_array = _relabel_components(result_array, comp_labels)
    result = (relabeled_result_array, da.max(relabeled_result_array))

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
