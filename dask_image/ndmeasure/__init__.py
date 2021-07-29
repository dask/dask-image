# -*- coding: utf-8 -*-

import collections
import functools
import operator
import warnings
from dask import compute, delayed

import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import numpy as np

from . import _utils
from ._utils import _label
from ._utils._find_objects import _array_chunk_location, _find_bounding_boxes, _find_objects

__all__ = [
    "area",
    "center_of_mass",
    "extrema",
    "histogram",
    "label",
    "labeled_comprehension",
    "maximum",
    "maximum_position",
    "mean",
    "median",
    "minimum",
    "minimum_position",
    "standard_deviation",
    "sum",
    "sum_labels",
    "variance",
]


def area(image, label_image=None, index=None):
    """Find the area of specified subregions in an image.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers.
        If None (default), returns area of total image dimensions.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.
        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    area : ndarray
        Area of ``index`` selected regions from ``label_image``.

    Example
    -------
    >>> import dask.array as da
    >>> image = da.random.random((3, 3))
    >>> label_image = da.from_array(
        [[1, 1, 0],
         [1, 0, 3],
         [0, 7, 0]], chunks=(1, 3))

    >>> # No labels given, returns area of total image dimensions
    >>> area(image)
    9

    >>> # Combined area of all non-zero labels
    >>> area(image, label_image).compute()
    5

    >>> # Areas of selected labels selected with the ``index`` keyword argument
    >>> area(image, label_image, index=[0, 1, 2, 3]).compute()
    array([4, 3, 0, 1], dtype=int64)
    """

    if label_image is None:
        return da.prod(np.array([i for i in image.shape]))

    else:
        image, label_image, index = _utils._norm_input_labels_index(
            image, label_image, index
        )

        ones = da.ones(
            label_image.shape, dtype=bool, chunks=label_image.chunks
        )

        area_lbl = labeled_comprehension(
            ones, label_image, index, len, int, int(0)
        )

        return area_lbl


def center_of_mass(image, label_image=None, index=None):
    """
    Find the center of mass over an image at specified subregions.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    center_of_mass : ndarray
        Coordinates of centers-of-mass of ``image`` over the ``index`` selected
        regions from ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    # SciPy transposes these for some reason.
    # So we do the same thing here.
    # This only matters if index is some array.
    index = index.T

    out_dtype = np.dtype([("com", float, (image.ndim,))])
    default_1d = np.full((1,), np.nan, dtype=out_dtype)

    func = functools.partial(
        _utils._center_of_mass, shape=image.shape, dtype=out_dtype
    )
    com_lbl = labeled_comprehension(
        image, label_image, index,
        func, out_dtype, default_1d[0], pass_positions=True
    )
    com_lbl = com_lbl["com"]

    return com_lbl


def extrema(image, label_image=None, index=None):
    """
    Find the min and max with positions over an image at specified subregions.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    minimums, maximums, min_positions, max_positions : tuple of ndarrays
        Values and coordinates of minimums and maximums in each feature.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    out_dtype = np.dtype([
        ("min_val", image.dtype),
        ("max_val", image.dtype),
        ("min_pos", np.dtype(int), image.ndim),
        ("max_pos", np.dtype(int), image.ndim)
    ])
    default_1d = np.zeros((1,), dtype=out_dtype)

    func = functools.partial(
        _utils._extrema, shape=image.shape, dtype=out_dtype
    )
    extrema_lbl = labeled_comprehension(
        image, label_image, index,
        func, out_dtype, default_1d[0], pass_positions=True
    )
    extrema_lbl = collections.OrderedDict([
        (k, extrema_lbl[k])
        for k in ["min_val", "max_val", "min_pos", "max_pos"]
    ])

    for pos_key in ["min_pos", "max_pos"]:
        pos_nd = extrema_lbl[pos_key]

        if index.ndim == 0:
            pos_nd = da.squeeze(pos_nd)
        elif index.ndim > 1:
            pos_nd = pos_nd.reshape(
                (int(np.prod(pos_nd.shape[:-1])), pos_nd.shape[-1])
            )

        extrema_lbl[pos_key] = pos_nd

    result = tuple(extrema_lbl.values())

    return result


def find_objects(label_image):
    """Return bounding box slices for each object labelled by integers.

    Parameters
    ----------
    label_image : ndarray
        Image features noted by integers.
    """
    if label_image.dtype.char not in np.typecodes['AllInteger']:
        raise ValueError("find_objects only accepts integer dtype arrays")

    block_iter = zip(
        np.ndindex(*label_image.numblocks),
        map(functools.partial(operator.getitem, label_image),
            da.core.slices_from_chunks(label_image.chunks))
    )

    arrays = []
    for block_id, block in block_iter:
        array_location = _array_chunk_location(block_id, label_image.chunks)
        arrays.append(delayed(_find_bounding_boxes)(block, array_location))

    bag = db.from_sequence(arrays)
    result = bag.fold(functools.partial(_find_objects, label_image.ndim), split_every=2).to_delayed()
    meta = dd.utils.make_meta([(i, object) for i in range(label_image.ndim)])
    result = delayed(compute)(result)[0]  # avoid the user having to call compute twice on result
    result = dd.from_delayed(result, meta=meta, prefix="find-objects-", verify_meta=False)

    return result


def histogram(image,
              min,
              max,
              bins,
              label_image=None,
              index=None):
    """
    Find the histogram over an image at specified subregions.

    Histogram calculates the frequency of values in an array within bins
    determined by ``min``, ``max``, and ``bins``. The ``label_image`` and
    ``index`` keywords can limit the scope of the histogram to specified
    sub-regions within the array.

    Parameters
    ----------
    image : ndarray
        N-D image data
    min : int
        Minimum value of range of histogram bins.
    max : int
        Maximum value of range of histogram bins.
    bins : int
        Number of bins.
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    histogram : ndarray
        Histogram of ``image`` over the ``index`` selected regions from
        ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )
    min = int(min)
    max = int(max)
    bins = int(bins)

    func = functools.partial(_utils._histogram, min=min, max=max, bins=bins)
    result = labeled_comprehension(
        image, label_image, index, func, object, None
    )

    return result


def label(image, structure=None):
    """
    Label features in an array.

    Parameters
    ----------
    image : ndarray
        An array-like object to be labeled.  Any non-zero values in ``image``
        are counted as features and zero values are considered the background.
    structure : ndarray, optional
        A structuring element that defines feature connections.
        ``structure`` must be symmetric.  If no structuring element is
        provided, one is automatically generated with a squared connectivity
        equal to one.  That is, for a 2-D ``image`` array, the default
        structuring element is::

            [[0,1,0],
             [1,1,1],
             [0,1,0]]

    Returns
    -------
    label : ndarray or int
        An integer ndarray where each unique feature in ``image`` has a unique
        label in the returned array.
    num_features : int
        How many objects were found.
    """

    image = da.asarray(image)

    labeled_blocks = np.empty(image.numblocks, dtype=object)

    # First, label each block independently, incrementing the labels in that
    # block by the total number of labels from previous blocks. This way, each
    # block's labels are globally unique.
    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks))
    )
    index, input_block = next(block_iter)
    labeled_blocks[index], total = _label.block_ndi_label_delayed(input_block,
                                                                  structure)
    for index, input_block in block_iter:
        labeled_block, n = _label.block_ndi_label_delayed(input_block,
                                                          structure)
        block_label_offset = da.where(labeled_block > 0,
                                      total,
                                      _label.LABEL_DTYPE.type(0))
        labeled_block += block_label_offset
        labeled_blocks[index] = labeled_block
        total += n

    # Put all the blocks together
    block_labeled = da.block(labeled_blocks.tolist())

    # Now, build a label connectivity graph that groups labels across blocks.
    # We use this graph to find connected components and then relabel each
    # block according to those.
    label_groups = _label.label_adjacency_graph(block_labeled, structure,
                                                total)
    new_labeling = _label.connected_components_delayed(label_groups)
    relabeled = _label.relabel_blocks(block_labeled, new_labeling)
    n = da.max(relabeled)

    return (relabeled, n)


def labeled_comprehension(image,
                          label_image,
                          index,
                          func,
                          out_dtype,
                          default,
                          pass_positions=False):
    """
    Compute a function over an image at specified subregions.

    Roughly equivalent to [func(image[labels == i]) for i in index].

    Sequentially applies an arbitrary function (that works on array_like image)
    to subsets of an n-D image array specified by ``label_image`` and
    ``index``. The option exists to provide the function with positional
    parameters as the second argument.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    func : callable
        Python function to apply to ``label_image`` from ``image``.
    out_dtype : dtype
        Dtype to use for ``result``.
    default : int, float or None
        Default return value when a element of ``index`` does not exist
        in ``label_image``.
    pass_positions : bool, optional
        If True, pass linear indices to ``func`` as a second argument.
        Default is False.

    Returns
    -------
    result : ndarray
        Result of applying ``func`` on ``image`` over the ``index`` selected
        regions from ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    out_dtype = np.dtype(out_dtype)
    default_1d = np.full((1,), default, dtype=out_dtype)

    pass_positions = bool(pass_positions)

    args = (image,)
    if pass_positions:
        positions = _utils._ravel_shape_indices(
            image.shape, chunks=image.chunks
        )
        args = (image, positions)

    result = np.empty(index.shape, dtype=object)
    for i in np.ndindex(index.shape):
        lbl_mtch_i = (label_image == index[i])
        args_lbl_mtch_i = tuple(e[lbl_mtch_i] for e in args)
        result[i] = _utils._labeled_comprehension_func(
            func, out_dtype, default_1d, *args_lbl_mtch_i
        )

    for i in range(result.ndim - 1, -1, -1):
        result2 = result[..., 0]
        for j in np.ndindex(index.shape[:i]):
            result2[j] = da.stack(result[j].tolist(), axis=0)
        result = result2
    result = result[()][..., 0]

    return result


def maximum(image, label_image=None, index=None):
    """
    Find the maxima over an image at specified subregions.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    maxima : ndarray
        Maxima of ``image`` over the ``index`` selected regions from
        ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    return labeled_comprehension(
        image, label_image, index, np.max, image.dtype, image.dtype.type(0)
    )


def maximum_position(image, label_image=None, index=None):
    """
    Find the positions of maxima over an image at specified subregions.

    For each region specified by ``label_image``, the position of the maximum
    value of ``image`` within the region is returned.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    maxima_positions : ndarray
        Maxima positions of ``image`` over the ``index`` selected regions from
        ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    if index.shape:
        index = index.flatten()

    out_dtype = np.dtype([("pos", int, (image.ndim,))])
    default_1d = np.zeros((1,), dtype=out_dtype)

    func = functools.partial(
        _utils._argmax, shape=image.shape, dtype=out_dtype
    )
    max_pos_lbl = labeled_comprehension(
        image, label_image, index,
        func, out_dtype, default_1d[0], pass_positions=True
    )
    max_pos_lbl = max_pos_lbl["pos"]

    if index.shape == tuple():
        max_pos_lbl = da.squeeze(max_pos_lbl)

    return max_pos_lbl


def mean(image, label_image=None, index=None):
    """
    Find the mean over an image at specified subregions.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    means : ndarray
        Mean of ``image`` over the ``index`` selected regions from
        ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    nan = np.float64(np.nan)

    mean_lbl = labeled_comprehension(
        image, label_image, index, np.mean, np.float64, nan
    )

    return mean_lbl


def median(image, label_image=None, index=None):
    """
    Find the median over an image at specified subregions.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    medians : ndarray
        Median of ``image`` over the ``index`` selected regions from
        ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    nan = np.float64(np.nan)

    return labeled_comprehension(
        image, label_image, index, np.median, np.float64, nan
    )


def minimum(image, label_image=None, index=None):
    """
    Find the minima over an image at specified subregions.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    minima : ndarray
        Minima of ``image`` over the ``index`` selected regions from
        ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    return labeled_comprehension(
        image, label_image, index, np.min, image.dtype, image.dtype.type(0)
    )


def minimum_position(image, label_image=None, index=None):
    """
    Find the positions of minima over an image at specified subregions.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    minima_positions : ndarray
        Maxima positions of ``image`` over the ``index`` selected regions from
        ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    if index.shape:
        index = index.flatten()

    out_dtype = np.dtype([("pos", int, (image.ndim,))])
    default_1d = np.zeros((1,), dtype=out_dtype)

    func = functools.partial(
        _utils._argmin, shape=image.shape, dtype=out_dtype
    )
    min_pos_lbl = labeled_comprehension(
        image, label_image, index,
        func, out_dtype, default_1d[0], pass_positions=True
    )
    min_pos_lbl = min_pos_lbl["pos"]

    if index.shape == tuple():
        min_pos_lbl = da.squeeze(min_pos_lbl)

    return min_pos_lbl


def standard_deviation(image, label_image=None, index=None):
    """
    Find the standard deviation over an image at specified subregions.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    standard_deviation : ndarray
        Standard deviation of ``image`` over the ``index`` selected regions
        from ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    nan = np.float64(np.nan)

    std_lbl = labeled_comprehension(
        image, label_image, index, np.std, np.float64, nan
    )

    return std_lbl


def sum_labels(image, label_image=None, index=None):
    """
    Find the sum of all pixels over specified subregions of an image.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    sum_lbl : ndarray
        Sum of ``image`` over the ``index`` selected regions from
        ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    sum_lbl = labeled_comprehension(
        image, label_image, index, np.sum, np.float64, np.float64(0)
    )

    return sum_lbl


def sum(image, label_image=None, index=None):
    """DEPRECATED FUNCTION. Use `sum_labels` instead."""
    warnings.warn("DEPRECATED FUNCTION. Use `sum_labels` instead.",
                  DeprecationWarning)
    return sum_labels(image, label_image=label_image, index=index)


def variance(image, label_image=None, index=None):
    """
    Find the variance over an image at specified subregions.

    Parameters
    ----------
    image : ndarray
        N-D image data
    label_image : ndarray, optional
        Image features noted by integers. If None (default), all values.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero ``label_image`` are used.

        The ``index`` argument only works when ``label_image`` is specified.

    Returns
    -------
    variance : ndarray
        Variance of ``image`` over the ``index`` selected regions from
        ``label_image``.
    """

    image, label_image, index = _utils._norm_input_labels_index(
        image, label_image, index
    )

    nan = np.float64(np.nan)

    var_lbl = labeled_comprehension(
        image, label_image, index, np.var, np.float64, nan
    )

    return var_lbl
