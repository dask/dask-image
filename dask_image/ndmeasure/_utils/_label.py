# -*- coding: utf-8 -*-

import operator

import dask
import dask.array as da
import numpy as np
import pandas as pd
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


def relabel_blocks(block_labeled, mapping):
    """
    Relabel a block-labeled array based on ``mapping``.

    Parameters
    ----------
    block_labeled : array of int
        The input label array.
    mapping : dict
        A label mapping, such that ``mapping[i] = j`` implies that
        any element in ``array`` valued ``i`` should be relabeled to ``j``.

    Returns
    -------
    relabeled : array of int, same shape as ``array``
        The relabeled input array.
    """
    
    def relabel_block(block, mapping):
        # pandas replace works well for both dense and sparse mappings
        # including large integer labels
        s = pd.Series(block.flatten())
        result = s.replace(mapping).to_numpy()
        return result.reshape(block.shape)

    relabeled = block_labeled.map_blocks(
        relabel_block,
        mapping=mapping,
        dtype=block_labeled.dtype
    )

    return relabeled


def relabel_sequential(labels):
    """
    Relabel the labels in ``labels`` to be sequential starting at 1.
    """

    u = da.unique(
        da.concatenate(
            [da.unique(b) for b in labels.blocks]
            )
        )
    mapping = dask.delayed(lambda x: {k: v for k, v in zip(x, np.arange(len(x)))})(u)

    relabeled = relabel_blocks(labels, mapping)

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


def _to_csr_matrix(i, j, n):
    """Using i and j as coo-format coordinates, return csr matrix."""
    v = np.ones_like(i)
    mat = scipy.sparse.coo_matrix((v, (i, j)), shape=(n, n))
    return mat.tocsr()


def label_adjacency_mapping(labels, structure, wrap_axes=None):
    """
    Adjacency graph of labels between chunks of ``labels``.

    Each chunk in ``labels`` has been labeled independently, and the labels
    in different chunks are guaranteed to be unique.

    Here we construct a mapping connecting labels in different chunks that
    correspond to the same logical label in the global volume. This is true
    if the two labels "touch" across the block face as defined by the input
    ``structure``.

    Parameters
    ----------
    labels : dask array of int
        The input labeled array, where each chunk is independently labeled.
    structure : array of bool
        Structuring element, shape (3,) * labels.ndim.
    wrap_axes : tuple of int, optional
        Should labels be wrapped across array boundaries, and if so which axes.
        - (0,) only wrap over the 0th axis.
        - (0, 1) wrap over the 0th and 1st axis.
        - (0, 1, 3)  wrap over 0th, 1st and 3rd axis.

    Returns
    -------
    mat : delayed dict
        This is a mapping such that if ``mapping[i] = j``, then labels ``i``
        and ``j`` should be merged.
    """

    if structure is None:
        structure = scipy.ndimage.generate_binary_structure(labels.ndim, 1)

    face_slices = _chunk_faces(
        labels.chunks, labels.shape, structure, wrap_axes=wrap_axes
    )
    all_mappings = [da.empty((2, 0), dtype=LABEL_DTYPE, chunks=1)]

    for face_slice in face_slices:
        face = labels[face_slice]
        mapped = _across_block_label_grouping_delayed(face, structure)
        all_mappings.append(mapped)

    all_mappings = da.concatenate(all_mappings, axis=1)

    return all_mappings


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

    Yields
    -------
    tuple of slices
        Each element indexes a face between two chunks.

    Examples
    --------
    >>> import dask.array as da
    >>> import scipy.ndimage as ndi
    >>> a = da.arange(110, chunks=110).reshape((10, 11)).rechunk(5)
    >>> structure = ndi.generate_binary_structure(2, 1)
    >>> list(chunk_faces(a.chunks, a.shape, structure))
    [(slice(4, 6, None), slice(0, 5, None)),
     (slice(4, 6, None), slice(5, 10, None)),
     (slice(4, 6, None), slice(10, 11, None)),
     (slice(0, 5, None), slice(4, 6, None)),
     (slice(0, 5, None), slice(9, 11, None)),
     (slice(5, 10, None), slice(4, 6, None)),
     (slice(5, 10, None), slice(9, 11, None))]
    """

    ndim = len(shape)

    slices = da.core.slices_from_chunks(chunks)

    # arrange block/chunk indices on grid
    block_summary = np.arange(len(slices)).reshape(
        [len(c) for c in chunks])

    # Iterate over all blocks and use the structuring element
    # to determine which blocks should be connected.
    # For wrappped axes, we need to consider the block
    # before the current block with index -1 as well.
    numblocks = [len(c) if wrap_axes is None or ax not in wrap_axes
                 else len(c) + 1 for ax, c in enumerate(chunks)]
    for curr_block in np.ndindex(tuple(numblocks)):

        curr_block = list(curr_block)

        if wrap_axes is not None:
            # start at -1 indices for wrapped axes
            for wrap_axis in wrap_axes:
                curr_block[wrap_axis] = curr_block[wrap_axis] - 1

        # iterate over neighbors of the current block
        for pos_structure_coord in np.array(np.where(structure)).T:

            # only consider forward neighbors
            if min(pos_structure_coord) < 1 or max(pos_structure_coord) < 2:
                continue

            neigh_block = [
                curr_block[dim] + pos_structure_coord[dim] - 1
                for dim in range(ndim)
            ]

            if max([neigh_block[dim] >= block_summary.shape[dim]
                    for dim in range(ndim)]):
                continue

            # get current slice index
            ind_curr_block = block_summary[tuple(curr_block)]

            curr_slice = []
            for dim in range(ndim):
                # keep slice if not on boundary
                if neigh_block[dim] == curr_block[dim]:
                    curr_slice.append(slices[ind_curr_block][dim])
                # otherwise, add two-pixel-wide boundary
                else:
                    if slices[ind_curr_block][dim].stop == shape[dim]:
                        curr_slice.append(slice(None, None, shape[dim] - 1))
                    else:
                        curr_slice.append(slice(
                            slices[ind_curr_block][dim].stop - 1,
                            slices[ind_curr_block][dim].stop + 1))

            yield tuple(curr_slice)


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


@dask.delayed(nout=2)
def connected_components_delayed(all_mappings):
    """
    Use scipy.sparse.csgraph.connected_components to find the connected
    components of the graph defined by ``all_mappings``.

    Return the mapping from old labels to new labels, and the number of
    labels that were collapsed.
    """

    if not all_mappings.shape[1]:
        return {}, 0

    # relabel all_mappings to consecutive integers starting at 1
    unique_labels, unique_inverse = np.unique(
        all_mappings,
        return_inverse=True)

    unique_labels_new = np.arange(
        0, len(unique_labels), dtype=all_mappings.dtype)

    relabeled_mappings = unique_labels_new[unique_inverse]

    i, j = relabeled_mappings
    csr_matrix = _to_csr_matrix(i, j, len(unique_labels) + 1)

    conn_comp = scipy.sparse.csgraph.connected_components
    new_labeling = unique_labels[conn_comp(csr_matrix, directed=False)[1]]

    n_red = len(unique_labels) - len(np.unique(new_labeling)) + 1

    mapping = {k: v for k, v in zip(unique_labels, new_labeling)}

    return mapping, n_red


def _encode_label(label, block_id, encoding_dtype=np.uint32):
    bit_shift = np.iinfo(encoding_dtype).bits // 2
    nonzero = label != 0
    label = np.array(label, dtype=encoding_dtype)
    label[nonzero] = label[nonzero] + (block_id << bit_shift)
    return label


# def _decode_label(encoded_label, encoding_dtype=np.uint32, decoding_dtype=np.uint16):
#     bit_shift = np.iinfo(encoding_dtype).bits // 2
#     label = encoded_label & ((1 << bit_shift) - 1)
#     # block_id = encoded_label >> bit_shift
#     return label.astype(decoding_dtype)


def _make_labels_unique(labels, encoding_dtype=np.uint16):
    """
    Make labels unique across blocks by using
    - lower bits to encode the block ID
    - higher bits to encode the label ID within the block

    Parameters
    ----------
    labels: dask array
        Array containing labels
    encoding_dtype: numpy dtype
        Dtype used to encode the labels.
        Must be an unsigned integer dtype.
    """

    assert np.issubdtype(encoding_dtype, np.unsignedinteger), \
        "encoding_dtype must be an unsigned integer dtype"
        
    def _unique_block_labels(
            block, block_id=None,
            encoding_dtype=encoding_dtype,
            numblocks=labels.numblocks
            ):
        
        if block_id is None:
            block_id = (0,) * block.ndim

        block_id = np.ravel_multi_index(
            block_id,
            numblocks,
            )
        block = block.astype(encoding_dtype)
        block = _encode_label(block, block_id, encoding_dtype=encoding_dtype)

        return block

    unique_labels = da.map_blocks(_unique_block_labels,
                                  labels,
                                  dtype=encoding_dtype,
                                  chunks=labels.chunks,
                                  encoding_dtype=encoding_dtype,
                                  )
    
    return unique_labels
