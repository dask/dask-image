# -*- coding: utf-8 -*-

import operator

import dask
import dask.array as da
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph


def _get_ndimage_label_dtype():
    return scipy.ndimage.label([1, 0, 1])[0].dtype


LABEL_DTYPE = _get_ndimage_label_dtype()


def _get_connected_components_dtype():
    a = np.empty((0, 0), dtype=int)
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
    relabeled = da.map_blocks(operator.getitem,
                              new_labeling,
                              block_labeled,
                              dtype=LABEL_DTYPE,
                              chunks=block_labeled.chunks)
    return relabeled


def _unique_axis(a, axis=0):
    """Find unique subarrays in axis in N-D array."""
    at = np.ascontiguousarray(a.swapaxes(0, axis))
    dt = np.dtype([("values", at.dtype, at.shape[1:])])
    atv = at.view(dt)
    r = np.unique(atv)["values"].swapaxes(0, axis)
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
    >>> face = np.array([[1, 1, 0, 2, 2, 0, 8],
    ...                     [0, 7, 7, 7, 7, 0, 9]])
    >>> structure = np.ones((3, 3), dtype=bool)
    >>> _across_block_label_grouping(face, structure)
    array([[1, 2, 8],
           [2, 7, 9]], dtype=np.int32)

    This shows that 1-2 are connected, 2-7 are connected, and 8-9 are
    connected. The resulting graph is (1-2-7), (8-9).
    """
    common_labels = scipy.ndimage.label(face, structure)[0]
    matching = np.stack((common_labels.ravel(), face.ravel()), axis=1)
    unique_matching = _unique_axis(matching)
    valid = np.all(unique_matching, axis=1)
    unique_valid_matching = unique_matching[valid]
    common_labels, labels = unique_valid_matching.T
    in_group = np.flatnonzero(np.diff(common_labels) == 0)
    i = np.take(labels, in_group)
    j = np.take(labels, in_group + 1)
    grouped = np.stack((i, j), axis=0)
    return grouped


def _across_block_label_grouping_delayed(face, structure):
    """Delayed version of :func:`_across_block_label_grouping`."""
    _across_block_label_grouping_ = dask.delayed(_across_block_label_grouping)
    grouped = _across_block_label_grouping_(face, structure)
    return da.from_delayed(grouped, shape=(2, np.nan), dtype=LABEL_DTYPE)


@dask.delayed
def _to_csr_matrix(i, j, n):
    """Using i and j as coo-format coordinates, return csr matrix."""
    v = np.ones_like(i)
    mat = scipy.sparse.coo_matrix((v, (i, j)), shape=(n, n))
    return mat.tocsr()


def set_tup_value(tup, idx, value):
    """Return a copy of `tup` with `value` at `idx`."""
    return tuple((elem if i == idx else value) for i, elem in enumerate(tup))


def label_adjacency_graph(labels, structure, nlabels, wrap_axes=None):
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
    wrap_axes : tuple of int, optional
        Should labels be wrapped across array boundaries, and if so which axes.
        - (0,) only wrap over the 0th axis.
        - (0, 1) wrap over the 0th and 1st axis.
        - (0, 1, 3)  wrap over 0th, 1st and 3rd axis.

    Returns
    -------
    mat : delayed scipy.sparse.csr_matrix
        This matrix has value 1 at (i, j) if label i is connected to
        label j in the global volume, 0 everywhere else.
    """

    if structure is None:
        structure = scipy.ndimage.generate_binary_structure(labels.ndim, 1)

    face_slices = _chunk_faces(
        labels.chunks, labels.shape, structure, wrap_axes=wrap_axes
    )
    all_mappings = [da.empty((2, 0), dtype=LABEL_DTYPE, chunks=1)]
    faces = []

    for face_slice in face_slices:
        faces.append(labels[face_slice])

    for face in faces:
        mapped = _across_block_label_grouping_delayed(face, structure)
        all_mappings.append(mapped)

    all_mappings = da.concatenate(all_mappings, axis=1)
    i, j = all_mappings
    mat = _to_csr_matrix(i, j, nlabels + 1)

    return mat


def _chunk_faces(chunks, shape, structure, wrap_axes=None):
    """
    Return slices for two-pixel-wide boundaries between chunks.

    Parameters
    ----------
    chunks : tuple of tuple of int
        The chunk specification of the array.
    shape : tuple of int
        The shape of the array.
    structure: array of bool
        Structuring element, shape (3,) * ndim.
    wrap_axes : tuple of int, optional
        Should labels be wrapped across array boundaries, and if so which axes.
        - (0,) only wrap over the 0th axis.
        - (0, 1) wrap over the 0th and 1st axis.
        - (0, 1, 3)  wrap over 0th, 1st and 3rd axis.

    Returns
    -------
    faces : list of tuple of slices
        Each element in this list indexes a face between two chunks.

    Examples
    --------
    >>> import dask.array as da
    >>> import scipy.ndimage as ndi
    >>> a = da.arange(110, chunks=110).reshape((10, 11)).rechunk(5)
    >>> structure = ndi.generate_binary_structure(2, 1)
    >>> chunk_faces(a.chunks, a.shape, structure)
    [(slice(4, 6, None), slice(0, 5, None)),
     (slice(4, 6, None), slice(5, 10, None)),
     (slice(4, 6, None), slice(10, 11, None)),
     (slice(0, 5, None), slice(4, 6, None)),
     (slice(0, 5, None), slice(9, 11, None)),
     (slice(5, 10, None), slice(4, 6, None)),
     (slice(5, 10, None), slice(9, 11, None))]
    """

    ndim = len(shape)
    numblocks = tuple(list(len(c) for c in chunks))

    slices = da.core.slices_from_chunks(chunks)

    # arrange block/chunk indices on grid
    block_summary = np.arange(len(slices)).reshape(numblocks)

    faces = []
    for ind_curr_block, curr_block in enumerate(np.ndindex(numblocks)):

        for pos_structure_coord in np.array(np.where(structure)).T:

            # only consider forward neighbors
            if min(pos_structure_coord) < 1 or \
               max(pos_structure_coord) < 2: continue

            neigh_block = [curr_block[dim] + pos_structure_coord[dim] - 1
                           for dim in range(ndim)]

            if max([neigh_block[dim] >= numblocks[dim] for dim in range(ndim)]): continue

            # get neighbor slice index
            ind_neigh_block = block_summary[tuple(neigh_block)]

            curr_slice = []
            for dim in range(ndim):
                # keep slice if not on boundary
                if slices[ind_curr_block][dim] == slices[ind_neigh_block][dim]:
                    curr_slice.append(slices[ind_curr_block][dim])
                # otherwise, add two-pixel-wide boundary
                else:
                    curr_slice.append(slice(
                        slices[ind_curr_block][dim].stop - 1,
                        slices[ind_curr_block][dim].stop + 1))

            faces.append(tuple(curr_slice))

    if wrap_axes is not None:
        for ax in wrap_axes:
            none_slice = (slice(None),) * ndim
            wrap_slice = set_tup_value(none_slice, ax, [0, -1])
            faces.append(wrap_slice)
        # Stupidly hard code corners
        faces.append(tuple(slice(None, None, i - 1) for i in shape))

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
    labeled = da.from_delayed(labeled_block, shape=block.shape,
                              dtype=LABEL_DTYPE)
    n = da.from_delayed(n, shape=(), dtype=LABEL_DTYPE)
    return labeled, n


def connected_components_delayed(csr_matrix):
    """
    Delayed version of scipy.sparse.csgraph.connected_components.

    This version only returns the (delayed) connected component labelling, not
    the number of components.
    """
    conn_comp = dask.delayed(scipy.sparse.csgraph.connected_components, nout=2)
    return da.from_delayed(conn_comp(csr_matrix, directed=False)[1],
                           shape=(np.nan,), dtype=CONN_COMP_DTYPE)
