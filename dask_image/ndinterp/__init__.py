# -*- coding: utf-8 -*-

import functools
import math
from itertools import product

from dask import delayed
import dask.array as da
import numpy as np
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
import scipy
from scipy.ndimage import affine_transform as ndimage_affine_transform
from scipy.ndimage import map_coordinates as ndimage_map_coordinates
from scipy.ndimage import labeled_comprehension as\
    ndimage_labeled_comprehension
from scipy.special import sindg, cosdg

from ..dispatch._dispatch_ndinterp import (
    dispatch_affine_transform,
    dispatch_asarray,
    dispatch_spline_filter,
    dispatch_spline_filter1d,
)
from ..dispatch._utils import get_type
from ..ndfilters._utils import _get_depth_boundary

__all__ = [
    "affine_transform",
    "map_coordinates",
    "rotate",
    "spline_filter",
    "spline_filter1d",
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

    if not isinstance(image, da.core.Array):
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


def rotate(
        input_arr,
        angle,
        axes=(1, 0),
        reshape=True,
        output_chunks=None,
        **kwargs,
        ):
    """Rotate an array using Dask.

    The array is rotated in the plane defined by the two axes given by the
    `axes` parameter using spline interpolation of the requested order.

    Chunkwise processing is performed using
    `dask_image.ndinterp.affine_transform`, for which further parameters
    supported by the ndimage functions can be passed as keyword arguments.

    Notes
    -----
        Differences to `ndimage.rotate`:
        - currently, prefiltering is not supported
          (affecting the output in case of interpolation `order > 1`)
        - default order is 1
        - modes 'reflect', 'mirror' and 'wrap' are not supported

        Arguments are equal to `ndimage.rotate` except for
        - `output` (not present here)
        - `output_chunks` (relevant in the dask array context)

    Parameters
    ----------
    input_arr : array_like (Numpy Array, Cupy Array, Dask Array...)
        The image array.
    angle : float
        The rotation angle in degrees.
    axes : tuple of 2 ints, optional
        The two axes that define the plane of rotation. Default is the first
        two axes.
    reshape : bool, optional
        If `reshape` is true, the output shape is adapted so that the input
        array is contained completely in the output. Default is True.
    output_chunks : tuple of ints, optional
        The shape of the chunks of the output Dask Array.
    **kwargs : dict, optional
        Additional keyword arguments are passed to
        `dask_image.ndinterp.affine_transform`.

    Returns
    -------
    rotate : Dask Array
        A dask array representing the rotated input.

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import matplotlib.pyplot as plt
    >>> import dask.array as da
    >>> fig = plt.figure(figsize=(10, 3))
    >>> ax1, ax2, ax3 = fig.subplots(1, 3)
    >>> img = da.from_array(misc.ascent(),chunks=(64,64))
    >>> img_45 = dask_image.ndinterp.rotate(img, 45, reshape=False)
    >>> full_img_45 = dask_image.ndinterp.rotate(img, 45, reshape=True)
    >>> ax1.imshow(img, cmap='gray')
    >>> ax1.set_axis_off()
    >>> ax2.imshow(img_45, cmap='gray')
    >>> ax2.set_axis_off()
    >>> ax3.imshow(full_img_45, cmap='gray')
    >>> ax3.set_axis_off()
    >>> fig.set_tight_layout(True)
    >>> plt.show()
    >>> print(img.shape)
    (512, 512)
    >>> print(img_45.shape)
    (512, 512)
    >>> print(full_img_45.shape)
    (724, 724)

    """
    if not isinstance(input_arr, da.core.Array):
        input_arr = da.from_array(input_arr)

    if output_chunks is None:
        output_chunks = input_arr.chunksize

    ndim = input_arr.ndim

    if ndim < 2:
        raise ValueError('input array should be at least 2D')

    axes = list(axes)

    if len(axes) != 2:
        raise ValueError('axes should contain exactly two values')

    if not all([float(ax).is_integer() for ax in axes]):
        raise ValueError('axes should contain only integer values')

    if axes[0] < 0:
        axes[0] += ndim
    if axes[1] < 0:
        axes[1] += ndim
    if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
        raise ValueError('invalid rotation plane specified')

    axes.sort()

    c, s = cosdg(angle), sindg(angle)

    rot_matrix = np.array([[c, s],
                           [-s, c]])

    img_shape = np.asarray(input_arr.shape)
    in_plane_shape = img_shape[axes]

    if reshape:
        # Compute transformed input bounds
        iy, ix = in_plane_shape
        in_bounds = np.array([[0, 0, iy, iy],
                              [0, ix, 0, ix]])
        out_bounds = rot_matrix @ in_bounds
        # Compute the shape of the transformed input plane
        out_plane_shape = (np.ptp(out_bounds, axis=1) + 0.5).astype(int)
    else:
        out_plane_shape = img_shape[axes]

    output_shape = np.array(img_shape)
    output_shape[axes] = out_plane_shape
    output_shape = tuple(output_shape)

    out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
    in_center = (in_plane_shape - 1) / 2
    offset = in_center - out_center

    matrix_nd = np.eye(ndim)
    offset_nd = np.zeros(ndim)

    for o_x, idx in enumerate(axes):

        matrix_nd[idx, axes[0]] = rot_matrix[o_x, 0]
        matrix_nd[idx, axes[1]] = rot_matrix[o_x, 1]

        offset_nd[idx] = offset[o_x]

    output = affine_transform(
        input_arr,
        matrix=matrix_nd,
        offset=offset_nd,
        output_shape=output_shape,
        output_chunks=output_chunks,
        **kwargs,
        )

    return output


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

    if not isinstance(image, da.core.Array):
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

    if not isinstance(image, da.core.Array):
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


def _map_single_coordinates_array_chunk(
        input, coordinates, order=3, mode='constant',
        cval=0.0, prefilter=False):
    """
    Central helper function for implementing map_coordinates.

    Receives 'input' as a dask array and computed coordinates.

    Implementation details and steps:
    1) associate each coordinate in coordinates with the chunk
       it maps to in the input
    2) for each input chunk that has been associated to at least one
       coordinate, calculate the minimal slice required to map
       all coordinates that are associated to it (note that resulting slice
       coordinates can lie outside of the coordinate's chunk)
    3) for each previously obtained slice and its associated coordinates,
       define a dask task and apply ndimage.map_coordinates
    4) outputs of ndimage.map_coordinates are rearranged to match input order
    """

    # STEP 1: Associate each coordinate in coordinates with
    # the chunk it maps to in the input array

    # get the input chunks each coordinate maps onto
    coords_input_chunk_locations = coordinates.T // np.array(input.chunksize)

    # map out-of-bounds chunk locations to valid input chunks
    coords_input_chunk_locations = np.clip(
        coords_input_chunk_locations, 0, np.array(input.numblocks) - 1
    )

    # all input chunk locations
    input_chunk_locations = np.array([i for i in np.ndindex(input.numblocks)])

    # linearize input chunk locations
    coords_input_chunk_locations_linear = np.sum(
        coords_input_chunk_locations * np.array(
            [np.prod(input.numblocks[:dim])
                for dim in range(input.ndim)])[::-1],
        axis=1, dtype=np.int64)

    # determine the input chunks that have coords associated and
    # count how many coords map onto each input chunk
    chunk_indices_count = np.bincount(coords_input_chunk_locations_linear,
                                      minlength=np.prod(input.numblocks))
    required_input_chunk_indices = np.where(chunk_indices_count > 0)[0]
    required_input_chunks = input_chunk_locations[required_input_chunk_indices]
    coord_rc_count = chunk_indices_count[required_input_chunk_indices]

    # inverse mapping: input chunks to coordinates
    required_input_chunk_coords_indices = \
        [np.where(coords_input_chunk_locations_linear == irc)[0]
            for irc in required_input_chunk_indices]

    # STEP 2: for each input chunk that has been associated to at least
    # one coordinate, calculate the minimal slice required to map all
    # coordinates that are associated to it (note that resulting slice
    # coordinates can lie outside of the coordinate's chunk)

    # determine the slices of the input array that are required for
    # mapping all coordinates associated to a given input chunk.
    # Note that this slice can be larger than the given chunk when coords
    # lie at chunk borders.
    # (probably there's a more efficient way to do this)
    input_slices_lower = np.array([np.clip(
            ndimage_labeled_comprehension(
                np.floor(coordinates[dim] - order // 2),
                coords_input_chunk_locations_linear,
                required_input_chunk_indices,
                np.min, np.int64, 0),
            0, input.shape[dim] - 1)
        for dim in range(input.ndim)])

    input_slices_upper = np.array([np.clip(
            ndimage_labeled_comprehension(
                np.ceil(coordinates[dim] + order // 2) + 1,
                coords_input_chunk_locations_linear,
                required_input_chunk_indices,
                np.max, np.int64, 0),
            0, input.shape[dim])
        for dim in range(input.ndim)])

    input_slices = np.array([input_slices_lower, input_slices_upper])\
        .swapaxes(1, 2)

    # STEP 3: For each previously obtained slice and its associated
    # coordinates, define a dask task and apply ndimage.map_coordinates

    # prepare building dask graph
    # define one task per associated input chunk
    name = "map_coordinates_chunk-%s" % tokenize(
        input,
        coordinates,
        order,
        mode,
        cval,
        prefilter
        )

    keys = [(name, i) for i in range(len(required_input_chunks))]

    # pair map_coordinates calls with input slices and mapped coordinates
    values = []
    for irc in range(len(required_input_chunks)):

        ric_slice = [slice(
            input_slices[0][irc][dim],
            input_slices[1][irc][dim])
            for dim in range(input.ndim)]
        ric_offset = input_slices[0][irc]

        values.append((
            ndimage_map_coordinates,
            input[tuple(ric_slice)],
            coordinates[:, required_input_chunk_coords_indices[irc]]
            - ric_offset[:, None],
            None,
            order,
            mode,
            cval,
            prefilter
        ))

    # build dask graph
    dsk = dict(zip(keys, values))
    ar = da.Array(dsk, name, tuple([list(coord_rc_count)]), input.dtype)

    # STEP 4: rearrange outputs of ndimage.map_coordinates
    # to match input order
    orig_order = np.argsort(
        [ic for ric_ci in required_input_chunk_coords_indices
            for ic in ric_ci])

    # compute result and reorder
    # (ordering first would probably unnecessarily inflate the task graph)
    return ar.compute()[orig_order]


def map_coordinates(input, coordinates, order=3,
                    mode='constant', cval=0.0, prefilter=False):
    """
    Wraps ndimage.map_coordinates.

    Both the input and coordinate arrays can be dask arrays.
    GPU arrays are not supported.

    For each chunk in the coordinates array, the coordinates are computed
    and mapped onto the required slices of the input array. One task is
    is defined for each input array chunk that has been associated to at
    least one coordinate. The outputs of the tasks are then rearranged to
    match the input order. For more details see the docstring of
    '_map_single_coordinates_array_chunk'.

    Using this function together with schedulers that support
    parallelism (threads, processes, distributed) makes sense in the
    case of either a large input array or a large coordinates array.
    When both arrays are large, it is recommended to use the
    single-threaded scheduler. A scheduler can be specified using e.g.
    `with dask.config.set(scheduler='threads'): ...`.

    input : array_like
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
        If True, prefilter the input before interpolation. Default is False.
        Warning: prefilter is True by default in
        `scipy.ndimage.map_coordinates`. Prefiltering here is performed on a
        chunk-by-chunk basis, which may lead to different results than
        `scipy.ndimage.map_coordinates` in case of chunked input arrays
        and order > 1.
        Note: prefilter is not necessary when:
          - You are using nearest neighbour interpolation, by setting order=0
          - You are using linear interpolation, by setting order=1, or
          - You have already prefiltered the input array,
          using the spline_filter or spline_filter1d functions.

    Comments:
      - in case of a small coordinate array, it might make sense to rechunk
        it into a single chunk
      - note the different default for `prefilter` compared to
        `scipy.ndimage.map_coordinates`, which is True by default.
    """
    if "cupy" in str(get_type(input)) or "cupy" in str(get_type(coordinates)):
        raise NotImplementedError("GPU cupy arrays are not supported by dask_image.ndinterp.map_overlap")

    # if coordinate array is not a dask array, convert it into one
    if type(coordinates) is not da.Array:
        coordinates = da.from_array(coordinates, chunks=coordinates.shape)
    else:
        # make sure indices are not split across chunks, i.e. that there's
        # no chunking along the first dimension
        if len(coordinates.chunks[0]) > 1:
            coordinates = da.rechunk(
                coordinates,
                (-1,) + coordinates.chunks[1:])

    # if the input array is not a dask array, convert it into one
    if type(input) is not da.Array:
        input = da.from_array(input, chunks=input.shape)

    # Map each chunk of the coordinates array onto the entire input array.
    # 'input' is passed to `_map_single_coordinates_array_chunk` using a bit of
    # a dirty trick: it is split into its components and passed as a delayed
    # object, which reconstructs the original array when the task is
    # executed. Therefore two `compute` calls are required to obtain the
    # final result, one of which is peformed by
    # `_map_single_coordinates_array_chunk`
    # Discussion https://dask.discourse.group/t/passing-dask-objects-to-delayed-computations-without-triggering-compute/1441
    output = da.map_blocks(
        _map_single_coordinates_array_chunk,
        delayed(da.Array)(input.dask, input.name, input.chunks, input.dtype),
        coordinates,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
        dtype=input.dtype,
        chunks=coordinates.chunks[1:],
        drop_axis=0,
    )

    return output
