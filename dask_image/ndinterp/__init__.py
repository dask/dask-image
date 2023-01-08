# -*- coding: utf-8 -*-

import functools
import math
from itertools import product
import warnings

from dask import compute, delayed
import dask.array as da
import numpy as np
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
import scipy
from scipy.ndimage import affine_transform as ndimage_affine_transform
from scipy.ndimage import map_coordinates as ndimage_map_coordinates

from ..dispatch._dispatch_ndinterp import (
    dispatch_affine_transform,
    dispatch_asarray,
    dispatch_spline_filter,
    dispatch_spline_filter1d,
)
from ..ndfilters._utils import _get_depth_boundary

from ..dispatch._dispatch_ndinterp import (dispatch_affine_transform,
                                           dispatch_asarray)

__all__ = [
    "affine_transform",
]


def affine_transform(
        image,
        matrix,
        offset=0.0,
        output_shape=None,
        order=1,
        output_chunks=None,
        **kwargs
):
    """Apply an affine transform using Dask. For every
    output chunk, only the slice containing the relevant part
    of the image is processed. Chunkwise processing is performed
    either using `ndimage.affine_transform` or
    `cupyx.scipy.ndimage.affine_transform`, depending on the input type.

    Notes
    -----
        Differences to `ndimage.affine_transformation`:
        - currently, prefiltering is not supported
          (affecting the output in case of interpolation `order > 1`)
        - default order is 1
        - modes 'reflect', 'mirror' and 'wrap' are not supported

        Arguments equal to `ndimage.affine_transformation`,
        except for `output_chunks`.

    Parameters
    ----------
    image : array_like (Numpy Array, Cupy Array, Dask Array...)
        The image array.
    matrix : array (ndim,), (ndim, ndim), (ndim, ndim+1) or (ndim+1, ndim+1)
        Transformation matrix.
    offset : float or sequence, optional
        The offset into the array where the transform is applied. If a float,
        `offset` is the same for each axis. If a sequence, `offset` should
        contain one value for each axis.
    output_shape : tuple of ints, optional
        The shape of the array to be returned.
    order : int, optional
        The order of the spline interpolation. Note that for order>1
        scipy's affine_transform applies prefiltering, which is not
        yet supported and skipped in this implementation.
    output_chunks : tuple of ints, optional
        The shape of the chunks of the output Dask Array.

    Returns
    -------
    affine_transform : Dask Array
        A dask array representing the transformed output

    """

    if not type(image) == da.core.Array:
        image = da.from_array(image)

    if output_shape is None:
        output_shape = image.shape

    if output_chunks is None:
        output_chunks = image.shape

    # Perform test run to ensure parameter validity.
    ndimage_affine_transform(np.zeros([0] * image.ndim),
                             matrix,
                             offset)

    # Make sure parameters contained in matrix and offset
    # are not overlapping, i.e. that the offset is valid as
    # it needs to be modified for each chunk.
    # Further parameter checks are performed directly by
    # `ndimage.affine_transform`.

    matrix = np.asarray(matrix)
    offset = np.asarray(offset).squeeze()

    # these lines were copied and adapted from `ndimage.affine_transform`
    if (matrix.ndim == 2 and matrix.shape[1] == image.ndim + 1 and
            (matrix.shape[0] in [image.ndim, image.ndim + 1])):

        # assume input is homogeneous coordinate transformation matrix
        offset = matrix[:image.ndim, image.ndim]
        matrix = matrix[:image.ndim, :image.ndim]

    cval = kwargs.pop('cval', 0)
    mode = kwargs.pop('mode', 'constant')
    prefilter = kwargs.pop('prefilter', False)

    supported_modes = ['constant', 'nearest']
    if scipy.__version__ > np.lib.NumpyVersion('1.6.0'):
        supported_modes += ['grid-constant']
    if mode in ['wrap', 'reflect', 'mirror', 'grid-mirror', 'grid-wrap']:
        raise NotImplementedError(
            f"Mode {mode} is not currently supported. It must be one of "
            f"{supported_modes}.")

    # process kwargs
    if prefilter and order > 1:
        # prefilter is not yet supported for all modes
        if mode in ['nearest', 'grid-constant']:
            raise NotImplementedError(
                f"order > 1 with mode='{mode}' is not supported. Currently "
                f"prefilter is only supported with mode='constant'."
            )
        image = spline_filter(image, order, output=np.float64,
                              mode=mode)

    n = image.ndim
    image_shape = image.shape

    # calculate output array properties
    normalized_chunks = da.core.normalize_chunks(output_chunks,
                                                 tuple(output_shape))
    block_indices = product(*(range(len(bds)) for bds in normalized_chunks))
    block_offsets = [np.cumsum((0,) + bds[:-1]) for bds in normalized_chunks]

    # use dispatching mechanism to determine backend
    affine_transform_method = dispatch_affine_transform(image)
    asarray_method = dispatch_asarray(image)

    # construct dask graph for output array
    # using unique and deterministic identifier
    output_name = 'affine_transform-' + tokenize(image, matrix, offset,
                                                 output_shape, output_chunks,
                                                 kwargs)
    output_layer = {}
    rel_images = []
    for ib, block_ind in enumerate(block_indices):

        out_chunk_shape = [normalized_chunks[dim][block_ind[dim]]
                           for dim in range(n)]
        out_chunk_offset = [block_offsets[dim][block_ind[dim]]
                            for dim in range(n)]

        out_chunk_edges = np.array([i for i in np.ndindex(tuple([2] * n))])\
            * np.array(out_chunk_shape) + np.array(out_chunk_offset)

        # map output chunk edges onto input image coordinates
        # to define the input region relevant for the current chunk
        if matrix.ndim == 1 and len(matrix) == image.ndim:
            rel_image_edges = matrix * out_chunk_edges + offset
        else:
            rel_image_edges = np.dot(matrix, out_chunk_edges.T).T + offset

        rel_image_i = np.min(rel_image_edges, 0)
        rel_image_f = np.max(rel_image_edges, 0)

        # Calculate edge coordinates required for the footprint of the
        # spline kernel according to
        # https://github.com/scipy/scipy/blob/9c0d08d7d11fc33311a96d2ac3ad73c8f6e3df00/scipy/ndimage/src/ni_interpolation.c#L412-L419 # noqa: E501
        # Also see this discussion:
        # https://github.com/dask/dask-image/issues/24#issuecomment-706165593 # noqa: E501
        for dim in range(n):

            if order % 2 == 0:
                rel_image_i[dim] += 0.5
                rel_image_f[dim] += 0.5

            rel_image_i[dim] = np.floor(rel_image_i[dim]) - order // 2
            rel_image_f[dim] = np.floor(rel_image_f[dim]) - order // 2 + order

            if order == 0:  # required for consistency with scipy.ndimage
                rel_image_i[dim] -= 1

        # clip image coordinates to image extent
        for dim, s in zip(range(n), image_shape):
            rel_image_i[dim] = np.clip(rel_image_i[dim], 0, s - 1)
            rel_image_f[dim] = np.clip(rel_image_f[dim], 0, s - 1)

        rel_image_slice = tuple([slice(int(rel_image_i[dim]),
                                       int(rel_image_f[dim]) + 2)
                                 for dim in range(n)])

        rel_image = image[rel_image_slice]

        """Block comment for future developers explaining how `offset` is
        transformed into `offset_prime` for each output chunk.
        Modify offset to point into cropped image.
        y = Mx + o
        Coordinate substitution:
        y' = y - y0(min_coord_px)
        x' = x - x0(chunk_offset)
        Then:
        y' = Mx' + o + Mx0 - y0
        M' = M
        o' = o + Mx0 - y0
        """

        offset_prime = offset + np.dot(matrix, out_chunk_offset) - rel_image_i

        output_layer[(output_name,) + block_ind] = (
                        affine_transform_method,
                        (da.core.concatenate3, rel_image.__dask_keys__()),
                        asarray_method(matrix),
                        offset_prime,
                        tuple(out_chunk_shape),  # output_shape
                        None,  # out
                        order,
                        mode,
                        cval,
                        False  # prefilter
        )

        rel_images.append(rel_image)

    graph = HighLevelGraph.from_collections(output_name, output_layer,
                                            dependencies=[image] + rel_images)

    meta = dispatch_asarray(image)([0]).astype(image.dtype)

    transformed = da.Array(graph,
                           output_name,
                           shape=tuple(output_shape),
                           # chunks=output_chunks,
                           chunks=normalized_chunks,
                           meta=meta)

    return transformed


# magnitude of the maximum filter pole for each order
# (obtained from scipy/ndimage/src/ni_splines.c)
_maximum_pole = {
    2: 0.171572875253809902396622551580603843,
    3: 0.267949192431122706472553658494127633,
    4: 0.361341225900220177092212841325675255,
    5: 0.430575347099973791851434783493520110,
}


def _get_default_depth(order, tol=1e-8):
    """Determine the approximate depth needed for a given tolerance.

    Here depth is chosen as the smallest integer such that ``|p| ** n < tol``
    where `|p|` is the magnitude of the largest pole in the IIR filter.
    """
    return math.ceil(np.log(tol) / np.log(_maximum_pole[order]))


def spline_filter(
        image,
        order=3,
        output=np.float64,
        mode='mirror',
        output_chunks=None,
        *,
        depth=None,
        **kwargs
):

    if not type(image) == da.core.Array:
        image = da.from_array(image)

    # use dispatching mechanism to determine backend
    spline_filter_method = dispatch_spline_filter(image)

    try:
        dtype = np.dtype(output)
    except TypeError:     # pragma: no cover
        raise TypeError(  # pragma: no cover
            "Could not coerce the provided output to a dtype. "
            "Passing array to output is not currently supported."
        )

    if depth is None:
        depth = _get_default_depth(order)

    if mode == 'wrap':
        raise NotImplementedError(
            "mode='wrap' is unsupported. It is recommended to use 'grid-wrap' "
            "instead."
        )

    # Note: depths of 12 and 24 give results matching SciPy to approximately
    #       single and double precision accuracy, respectively.
    boundary = "periodic" if mode == 'grid-wrap' else "none"
    depth, boundary = _get_depth_boundary(image.ndim, depth, boundary)

    # cannot pass a func kwarg named "output" to map_overlap
    spline_filter_method = functools.partial(spline_filter_method,
                                             output=dtype)

    result = image.map_overlap(
        spline_filter_method,
        depth=depth,
        boundary=boundary,
        dtype=dtype,
        meta=image._meta,
        # spline_filter kwargs
        order=order,
        mode=mode,
    )

    return result


def spline_filter1d(
        image,
        order=3,
        axis=-1,
        output=np.float64,
        mode='mirror',
        output_chunks=None,
        *,
        depth=None,
        **kwargs
):

    if not type(image) == da.core.Array:
        image = da.from_array(image)

    # use dispatching mechanism to determine backend
    spline_filter1d_method = dispatch_spline_filter1d(image)

    try:
        dtype = np.dtype(output)
    except TypeError:     # pragma: no cover
        raise TypeError(  # pragma: no cover
            "Could not coerce the provided output to a dtype. "
            "Passing array to output is not currently supported."
        )

    if depth is None:
        depth = _get_default_depth(order)

    # use depth 0 on all axes except the filtered axis
    if not np.isscalar(depth):
        raise ValueError("depth must be a scalar value")
    depths = [0] * image.ndim
    depths[axis] = depth

    if mode == 'wrap':
        raise NotImplementedError(
            "mode='wrap' is unsupported. It is recommended to use 'grid-wrap' "
            "instead."
        )

    # cannot pass a func kwarg named "output" to map_overlap
    spline_filter1d_method = functools.partial(spline_filter1d_method,
                                               output=dtype)

    result = image.map_overlap(
        spline_filter1d_method,
        depth=tuple(depths),
        boundary="periodic" if mode == 'grid-wrap' else "none",
        dtype=dtype,
        meta=image._meta,
        # spline_filter1d kwargs
        order=order,
        axis=axis,
        mode=mode,
    )

    return result


def _map_single_coordinates_array_chunk(image, coordinates, order=3,
                    mode='constant', cval=0.0, prefilter=False):
    """
    Central helper function for implementing map_coordinates.

    Receives 'image' as a dask array and computed coordinates.

    Implementation details and steps:
    1) associate each coordinate in coordinates with the chunk
       it maps to in the input image
    2) for each input image chunk that has been associated to at least one
       coordinate, calculate the minimal slice required to map
       all coordinates that are associated to it (note that resulting slice
       coordinates can lie outside of the coordinate's chunk)
    3) for each previously obtained slice and its associated coordinates,
       define a dask task and apply ndimage.map_coordinates
    4) outputs of ndimage.map_coordinates are rearranged to match input order
    """

    # STEP 1: Associate each coordinate in coordinates with
    # the chunk it maps to in the input image

    # determine which chunk each coordinate maps to
    coord_chunk_locations = np.array([
        np.searchsorted(np.cumsum(image.chunks[dim]), coordinates[dim])
        for dim in range(image.ndim)]).T
    # associate coordinates_chunk with the chunks they map into
    required_input_chunks, coord_rc_inds, coord_rc_count = np.unique(
        coord_chunk_locations, axis=0, return_inverse=True, return_counts=True)

    # STEP 2: for each input image chunk that has been associated to at least
    # one coordinate, calculate the minimal slice required to map all
    # coordinates that are associated to it (note that resulting slice
    # coordinates can lie outside of the coordinate's chunk)

    input_slices = np.ones((2, len(required_input_chunks), image.ndim)) * np.nan
    rc_coord_inds = [np.where(coord_rc_inds == irc)[0]
                     for irc in range(len(required_input_chunks))]
    for ic, c in enumerate(coordinates.T):

        input_slices[0][coord_rc_inds[ic]]\
            = np.nanmin([input_slices[0][coord_rc_inds[ic]],
                         np.floor(c - order // 2)], 0)
        input_slices[1][coord_rc_inds[ic]]\
            = np.nanmax([input_slices[1][coord_rc_inds[ic]],
                         np.ceil(c + order // 2)], 0)

    # STEP 3: For each previously obtained slice and its associated
    # coordinates, define a dask task and apply ndimage.map_coordinates

    # prepare building of dask graph
    name = "map_coordinates_chunk-%s" % tokenize(image,
                                           coordinates,
                                           order,
                                           mode,
                                           cval,
                                           prefilter)

    keys = [(name, i) for i in range(len(required_input_chunks))]

    values = []
    for irc, _rc in enumerate(required_input_chunks):
        offset = np.min([np.max([[0] * image.ndim,
                                 input_slices[0][irc].astype(np.int64)], 0),
                         np.array(image.shape) - 1], 0)
        sl = [slice(offset[dim],
                    min(image.shape[dim], int(input_slices[1][irc][dim]) + 1))
              for dim in range(image.ndim)]

        values.append((ndimage_map_coordinates,
                       image[tuple(sl)],
                       coordinates[:, rc_coord_inds[irc]] - offset[:, None],
                       None,
                       order,
                       mode,
                       cval,
                       prefilter))

    dsk = dict(zip(keys, values))
    ar = da.Array(dsk, name, tuple([list(coord_rc_count)]), image.dtype)

    # STEP 4: rearrange outputs of ndimage.map_coordinates
    # to match input order
    orig_order = np.argsort([ic for rc_ci in rc_coord_inds for ic in rc_ci])

    return ar[orig_order].compute()


def map_coordinates(image, coordinates, order=3,
                    mode='constant', cval=0.0, prefilter=False):
    """
    Wraps ndimage.map_coordinates.

    Both the image and coordinate arrays can be dask arrays.

    For each chunk in the coordinates array, the coordinates are computed
    and mapped onto the required slices of the input image. One task is
    is defined for each input image chunk that has been associated to at
    least one coordinate. The outputs of the tasks are then rearranged to
    match the input order. For more details see the docstring of
    '_map_single_coordinates_array_chunk'.

    image : array_like
        The input array.
    coordinates : array_like
        The coordinates at which to sample the input.
    order : int, optional
        The order of the spline interpolation, default is 3. The order has to
        be in the range 0-5.
    mode : boundary behavior mode, optional
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'. Default is 0.0
    prefilter : bool, optional

    Comments:
     - in case of a small coordinate array, it might make sense to rechunk
       it into a single chunk
    """

    # if coordinate array is not a dask array, convert it into one
    if type(coordinates) is not da.Array:
        coordinates = da.from_array(coordinates, chunks=coordinates.shape)
    else:
        # make sure indices are not split across chunks, i.e. that there's
        # no chunking along the first dimension 
        if len(coordinates.chunks[0]) > 1:
            coordinates = da.rechunk(coordinates,
                                    (-1,) + coordinates.chunks[1:])

    # if image array is not a dask array, convert it into one
    if type(image) is not da.Array:
        image = da.from_array(image, chunks=image.shape)

    # Map each chunk of the coordinates array onto the entire input image.
    # 'image' is passed to `_map_single_coordinates_array_chunk` using a
    # dirty trick: it is split into its components and passed as a delayed
    # object, which reconstructs the original array when the task is
    # executed. Therefore two `compute` calls are required to obtain the
    # final result, one of which is peformed by
    # `_map_single_coordinates_array_chunk`
    output = da.map_blocks(
        _map_single_coordinates_array_chunk,
        delayed(da.Array)(image.dask, image.name, image.chunks, image.dtype),
        coordinates,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
        dtype=image.dtype,
        chunks=coordinates.chunks[1:],
        drop_axis=0,
    )

    return output
