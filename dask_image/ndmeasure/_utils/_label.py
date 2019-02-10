# -*- coding: utf-8 -*-

import operator

import numpy
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph

import dask
import dask.array


def _get_ndimage_label_dtype():
    return scipy.ndimage.label([1, 0, 1])[0].dtype


LABEL_DTYPE = _get_ndimage_label_dtype()


def _get_connected_components_dtype():
    a = numpy.empty((0, 0), dtype=int)
    return scipy.sparse.csgraph.connected_components(a)[1].dtype


CONN_COMP_DTYPE = _get_connected_components_dtype()


def relabel_blocks(block_labeled, new_labeling):
    """
    Relabel a block-labeled array based on ``new_labeling``.

    Parameters
    ----------
    block_labeled : array of int
        The input label array.
    new_labeling : 1D array of int
        A new labeling, such that ``labeling[i] = j`` implies that
        any element in ``array`` valued ``i`` should be relabeled to ``j``.

    Returns
    -------
    relabeled : array of int, same shape as ``array``
        The relabeled input array.
    """
    new_labeling = new_labeling.astype(LABEL_DTYPE)
    relabeled = dask.array.map_blocks(operator.getitem,
                                      new_labeling,
                                      block_labeled,
                                      dtype=LABEL_DTYPE,
                                      chunks=block_labeled.chunks)
    return relabeled


def _unique_axis(a, axis=0):
    """Find unique subarrays in axis in N-D array."""
    at = numpy.ascontiguousarray(a.swapaxes(0, axis))
    dt = numpy.dtype([("values", at.dtype, at.shape[1:])])
    atv = at.view(dt)
    r = numpy.unique(atv)["values"].swapaxes(0, axis)
    return r


def _across_block_label_grouping(face, structure):
    """
    Find a grouping of labels across block faces.

    We assume that the labels on either side of the block face are unique to
    that block. This is enforced elsewhere.

    Parameters
    ----------
    face : array-like
        This is the boundary, of thickness (2,), between two blocks.
    structure : array-like
        Structuring element for the labeling of the face. This should have
        length 3 along each axis and have the same number of dimensions as
        ``face``.

    Returns
    -------
    grouped : array of int, shape (2, M)
        If a column of ``grouped`` contains the values ``i`` and ``j``, it
        implies that labels ``i`` and ``j`` belong in the same group. These
        are edges in a global label connectivity graph.

    Examples
    --------
    >>> face = numpy.array([[1, 1, 0, 2, 2, 0, 8],
    ...                     [0, 7, 7, 7, 7, 0, 9]])
    >>> structure = numpy.ones((3, 3), dtype=bool)
    >>> _across_block_label_grouping(face, structure)
    array([[1, 2, 8],
           [2, 7, 9]], dtype=numpy.int32)

    This shows that 1-2 are connected, 2-7 are connected, and 8-9 are
    connected. The resulting graph is (1-2-7), (8-9).
    """
    common_labels = scipy.ndimage.label(face, structure)[0]
    matching = numpy.stack((common_labels.ravel(), face.ravel()), axis=1)
    unique_matching = _unique_axis(matching)
    valid = numpy.all(unique_matching, axis=1)
    unique_valid_matching = unique_matching[valid]
    common_labels, labels = unique_valid_matching.T
    in_group = numpy.flatnonzero(numpy.diff(common_labels) == 0)
    i = numpy.take(labels, in_group)
    j = numpy.take(labels, in_group + 1)
    grouped = numpy.stack((i, j), axis=0)
    return grouped


def _across_block_label_grouping_delayed(face, structure):
    """Delayed version of :func:`_across_block_label_grouping`."""
    _across_block_label_grouping_ = dask.delayed(_across_block_label_grouping)
    grouped = _across_block_label_grouping_(face, structure)
    return dask.array.from_delayed(grouped,
                                   shape=(2, numpy.nan),
                                   dtype=LABEL_DTYPE)


@dask.delayed
def _to_csr_matrix(i, j, n):
    """Using i and j as coo-format coordinates, return csr matrix."""
    v = numpy.ones_like(i)
    mat = scipy.sparse.coo_matrix((v, (i, j)), shape=(n, n))
    return mat.tocsr()


def label_adjacency_graph(labels, structure, nlabels):
    """
    Adjacency graph of labels between chunks of ``labels``.

    Each chunk in ``labels`` has been labeled independently, and the labels
    in different chunks are guaranteed to be unique.

    Here we construct a graph connecting labels in different chunks that
    correspond to the same logical label in the global volume. This is true
    if the two labels "touch" across the block face as defined by the input
    ``structure``.

    Parameters
    ----------
    labels : dask array of int
        The input labeled array, where each chunk is independently labeled.
    structure : array of bool
        Structuring element, shape (3,) * labels.ndim.
    nlabels : delayed int
        The total number of labels in ``labels`` *before* correcting for
        global consistency.

    Returns
    -------
    mat : delayed scipy.sparse.csr_matrix
        This matrix has value 1 at (i, j) if label i is connected to
        label j in the global volume, 0 everywhere else.
    """
    faces = _chunk_faces(labels.chunks, labels.shape)
    all_mappings = [dask.array.empty((2, 0), dtype=LABEL_DTYPE, chunks=1)]
    for face_slice in faces:
        face = labels[face_slice]
        mapped = _across_block_label_grouping_delayed(face, structure)
        all_mappings.append(mapped)
    all_mappings = dask.array.concatenate(all_mappings, axis=1)
    i, j = all_mappings
    mat = _to_csr_matrix(i, j, nlabels + 1)
    return mat


def _chunk_faces(chunks, shape):
    """
    Return slices for two-pixel-wide boundaries between chunks.

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
    >>> a = dask.array.arange(110, chunks=110).reshape((10, 11)).rechunk(5)
    >>> chunk_faces(a.chunks, a.shape)
    [(slice(4, 6, None), slice(0, 5, None)),
     (slice(4, 6, None), slice(5, 10, None)),
     (slice(4, 6, None), slice(10, 11, None)),
     (slice(0, 5, None), slice(4, 6, None)),
     (slice(0, 5, None), slice(9, 11, None)),
     (slice(5, 10, None), slice(4, 6, None)),
     (slice(5, 10, None), slice(9, 11, None))]
    """
    slices = dask.array.core.slices_from_chunks(chunks)
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


def block_ndi_label_delayed(block, structure):
    """
    Delayed version of ``scipy.ndimage.label``.

    Parameters
    ----------
    block : dask array (single chunk)
        The input array to be labeled.
    structure : array of bool
        Structure defining the connectivity of the labeling.

    Returns
    -------
    labeled : dask array, same shape as ``block``.
        The labeled array.
    n : delayed int
        The number of labels in ``labeled``.
    """
    label = dask.delayed(scipy.ndimage.label, nout=2)
    labeled_block, n = label(block, structure=structure)
    n = dask.delayed(LABEL_DTYPE.type)(n)
    labeled = dask.array.from_delayed(labeled_block, shape=block.shape,
                                      dtype=LABEL_DTYPE)
    n = dask.array.from_delayed(n, shape=(), dtype=LABEL_DTYPE)
    return labeled, n


def connected_components_delayed(csr_matrix):
    """
    Delayed version of scipy.sparse.csgraph.connected_components.

    This version only returns the (delayed) connected component labelling, not
    the number of components.
    """
    conn_comp = dask.delayed(scipy.sparse.csgraph.connected_components, nout=2)
    return dask.array.from_delayed(conn_comp(csr_matrix, directed=False)[1],
                                   shape=(numpy.nan,), dtype=CONN_COMP_DTYPE)
