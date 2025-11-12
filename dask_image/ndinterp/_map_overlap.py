# -*- coding: utf-8 -*-

from dask import delayed
import dask.array as da
import numpy as np
from dask.base import tokenize
from scipy.ndimage import map_coordinates as ndimage_map_coordinates
from scipy.ndimage import labeled_comprehension as\
    ndimage_labeled_comprehension

from ..dispatch._utils import get_type


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
        prefilter=prefilter,
        dtype=input.dtype,
        chunks=coordinates.chunks[1:],
        drop_axis=0,
    )

    return output
