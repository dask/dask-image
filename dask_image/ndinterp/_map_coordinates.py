# -*- coding: utf-8 -*-

from dask import delayed
import dask.array as da
import numpy as np
from dask.base import tokenize
from scipy.ndimage import map_coordinates as ndimage_map_coordinates
from scipy.ndimage import labeled_comprehension as\
    ndimage_labeled_comprehension

from ._spline_filters import _get_default_depth, spline_filter
from ..dispatch._utils import get_type


def _map_single_coordinates_array_chunk(
        input, coordinates, order, mode, cval, output_dtype):
    """
    Central helper function for implementing map_coordinates.

    Receives 'input' as a dask array and computed coordinates.

    Implementation details and steps:
    1) associate each coordinate in coordinates with the chunk
       it maps to in the input
    2) for each input chunk that has been associated to at least one
       coordinate, calculate a slice covering the interpolation samples for
       those coordinates. The slice can extend beyond the chunk.
    3) for each previously obtained slice and its associated coordinates,
       define a dask task and apply ndimage.map_coordinates
    4) outputs of ndimage.map_coordinates are rearranged to match input order
    """

    output_shape = coordinates.shape[1:]
    coordinates = coordinates.astype(np.float64, copy=False)
    coordinates = coordinates.reshape(input.ndim, -1)
    if coordinates.shape[1] == 0:
        return np.empty(output_shape, dtype=output_dtype)

    # STEP 1: Associate coordinates with input chunks. Use the actual chunk
    # edges since prefilter padding can introduce uneven chunk sizes.
    coords_input_chunk_locations = tuple(
        np.searchsorted(np.cumsum(chunks)[:-1], axis_coords, side='right')
        for chunks, axis_coords in zip(input.chunks, coordinates)
    )
    coords_input_chunk_locations_linear = np.ravel_multi_index(
        coords_input_chunk_locations, input.numblocks
    )
    required_input_chunk_indices, coord_rc_count = np.unique(
        coords_input_chunk_locations_linear, return_counts=True
    )

    # inverse mapping: input chunks to coordinates
    required_input_chunk_coords_indices = \
        [np.where(coords_input_chunk_locations_linear == irc)[0]
            for irc in required_input_chunk_indices]

    # STEP 2: Find the source slice for the samples associated with each chunk.

    # determine the slices of the input array that are required for
    # mapping all coordinates associated to a given input chunk.
    # Note that this slice can be larger than the given chunk when coords
    # lie at chunk borders.
    # (probably there's a more efficient way to do this)
    input_slices_lower = np.array([
        ndimage_labeled_comprehension(
            np.floor(coordinates[dim] - order // 2),
            coords_input_chunk_locations_linear,
            required_input_chunk_indices,
            np.min, np.int64, 0,
        )
        for dim in range(input.ndim)
    ])

    input_slices_upper = np.array([
        ndimage_labeled_comprehension(
            np.ceil(coordinates[dim] + order // 2) + 1,
            coords_input_chunk_locations_linear,
            required_input_chunk_indices,
            np.max, np.int64, 0,
        )
        for dim in range(input.ndim)
    ])

    shape = np.asarray(input.shape)[:, None]
    if mode in ('reflect', 'grid-mirror', 'mirror', 'wrap', 'grid-wrap'):
        # These modes can map a coordinate or spline tap to a distant sample.
        # Give SciPy the full extent of an axis when the slice crosses an edge.
        crosses_edge = (input_slices_lower < 0) | (input_slices_upper > shape)
        input_slices_lower = np.where(crosses_edge, 0, input_slices_lower)
        input_slices_upper = np.where(crosses_edge, shape, input_slices_upper)
    input_slices_lower = np.clip(input_slices_lower, 0, shape - 1)
    input_slices_upper = np.clip(
        input_slices_upper, input_slices_lower + 1, shape
    )

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
        output_dtype
        )

    keys = [(name, i) for i in range(len(required_input_chunk_indices))]

    # pair map_coordinates calls with input slices and mapped coordinates
    values = []
    for irc in range(len(required_input_chunk_indices)):

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
            output_dtype,
            order,
            mode,
            cval,
            False  # the source has already been prefiltered if requested
        ))

    # build dask graph
    dsk = dict(zip(keys, values))
    ar = da.Array(dsk, name, (tuple(coord_rc_count),), output_dtype)

    # STEP 4: rearrange outputs of ndimage.map_coordinates
    # to match input order
    orig_order = np.argsort(
        [ic for ric_ci in required_input_chunk_coords_indices
            for ic in ric_ci])

    # compute result and reorder
    # (ordering first would probably unnecessarily inflate the task graph)
    return ar.compute()[orig_order].reshape(output_shape)


def map_coordinates(input, coordinates, order=3,
                    mode='constant', cval=0.0, prefilter=True):
    """
    Wraps ndimage.map_coordinates.

    Both the input and coordinate arrays can be dask arrays.
    GPU arrays are not supported.

    For each chunk in the coordinates array, the coordinates are computed
    and mapped onto the required slices of the input array. One task is
    defined for each input array chunk that has been associated to at
    least one coordinate. The outputs of the tasks are then rearranged to
    match the input order. For more details see the docstring of
    '_map_single_coordinates_array_chunk'.
    Reflecting or wrapping at the image boundary can require reading the full
    extent of the affected input axes.

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
        Fill value for 'constant' and 'grid-constant' modes. Default is 0.0.
    prefilter : bool, optional
        If True and order is greater than 1, prefilter the input with
        `dask_image.ndinterp.spline_filter` before extracting the slices for
        interpolation. The filter overlaps input chunks to approximate the
        full-array spline coefficients. The default is True, as in
        `scipy.ndimage.map_coordinates`.
        Note: prefilter is not necessary when:
          - You are using nearest neighbour interpolation, by setting order=0
          - You are using linear interpolation, by setting order=1, or
          - You have already prefiltered the input array,
          using the spline_filter or spline_filter1d functions.

    Comments:
      - in case of a small coordinate array, it might make sense to rechunk
        it into a single chunk
    """
    if "cupy" in str(get_type(input)) or "cupy" in str(get_type(coordinates)):
        raise NotImplementedError(
            "GPU cupy arrays are not supported by "
            "dask_image.ndinterp.map_overlap")

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

    if coordinates.ndim < 2 or coordinates.shape[0] != input.ndim:
        raise RuntimeError("invalid shape for coordinate array")

    output_dtype = input.dtype
    if prefilter and 2 <= order <= 5:
        if mode in ('nearest', 'grid-constant'):
            # Match the padding SciPy uses before spline filtering.
            pad_width = 12
            if mode == 'nearest':
                input = da.pad(input, pad_width, mode='edge')
            else:
                input = da.pad(input, pad_width, mode='constant',
                               constant_values=cval)
            coordinates = coordinates.astype(np.float64) + pad_width

        # SciPy uses mirror boundary conditions for the spline in wrap mode.
        filter_mode = 'mirror' if mode == 'wrap' else mode
        coefficient_dtype = (np.complex128
                             if np.issubdtype(output_dtype, np.complexfloating)
                             else np.float64)
        overlap = _get_default_depth(order, tol=np.finfo(np.float64).eps)
        depth = tuple(min(size, overlap) for size in input.shape)
        input = spline_filter(input, order=order, output=coefficient_dtype,
                              mode=filter_mode, depth=depth)

    # Map each chunk of the coordinates array onto the entire input array.
    # 'input' is passed to `_map_single_coordinates_array_chunk` using a bit of
    # a dirty trick: it is split into its components and passed as a delayed
    # object, which reconstructs the original array when the task is
    # executed. Therefore two `compute` calls are required to obtain the
    # final result, one of which is peformed by
    # `_map_single_coordinates_array_chunk`
    # Discussion https://dask.discourse.group/t/passing-dask-objects-to-delayed-computations-without-triggering-compute/1441 # noqa: E501
    output = da.map_blocks(
        _map_single_coordinates_array_chunk,
        delayed(da.Array)(input.dask, input.name, input.chunks, input.dtype),
        coordinates,
        order=order,
        mode=mode,
        cval=cval,
        output_dtype=output_dtype,
        dtype=output_dtype,
        chunks=coordinates.chunks[1:],
        drop_axis=0,
    )

    return output
