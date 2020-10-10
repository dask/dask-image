# -*- coding: utf-8 -*-


import numpy as np
import dask.array as da
from scipy import ndimage


__all__ = [
    "affine_transform",
]


def affine_transform(
        input,
        matrix,
        offset=0.0,
        output_shape=None,
        output_chunks=None,
        **kwargs
):
    """Apply an affine transform using Dask.
    `ndimage.affine_transformation` for Dask Arrays. For every
    output chunk, only the slice containing the relevant part
    of the input is passed on to `ndimage.affine_transformation`.

    Notes
    -----
        Differences to `ndimage.affine_transformation`:
        - currently, prefiltering is not supported
          (affecting the output in case of interpolation `order > 1`)
        - default order is 1
        - currently, matrix is expected to have shape (ndim,) or (ndim, ndim)
          (because offset needs to be set for each chunk)
        - modes 'reflect', 'mirror' and 'wrap' may yield different results

        To do:
        - optionally use cupyx.scipy.ndimage.affine_transform

        Arguments equal to `ndimage.affine_transformation`,
        except for `output_chunks`.

    Parameters
    ----------
    input : array_like (Numpy Array, Dask Array, ...)
        The input array.
    matrix : array (ndim,) or (ndim, ndim)
        Transformation matrix.
    offset : array (ndim,)
        Transformation offset.
    output_shape : array (ndim,), optional
        The size of the array to be returned.
    output_chunks : array (ndim,), optional
        The chunks of the output Dask Array.

    Returns
    -------
    affine_transform : Dask Array
        A dask array representing the transformed output

    """

    if output_shape is None:
        output_shape = input.shape

    # process kwargs

    # set default order to 1, warn if different
    if 'order' in kwargs:
        order = kwargs['order']
        del kwargs['order']
        if order > 1:
            UserWarning('Currently, for order > 1 the output of'
                        '`dask_image.ndinterp.affine_transform` can differ'
                        'from that of `ndimage.affine_transform`')
    else:
        order = 1

    # prefilter is not yet supported
    if 'prefilter' in kwargs:
        if kwargs['prefilter'] and order > 1:
            UserWarning('Currently, `dask_image.ndinterp.affine_transform`'
                        'does not support `prefilter=True`')
        del kwargs['prefilter']

    transformed = da.zeros(output_shape,
                           dtype=input.dtype,
                           chunks=output_chunks)

    transformed = transformed.map_blocks(resample_chunk,
                                         dtype=input.dtype,
                                         input=input,
                                         matrix=matrix,
                                         offset=offset,
                                         order=order,
                                         func_kwargs=kwargs,
                                         )

    return transformed


def resample_chunk(chunk, input, matrix, offset, order, func_kwargs, block_info=None):
    """Resample a given chunk, used by `affine_transform`."""

    n = chunk.ndim
    input_shape = input.shape
    chunk_shape = chunk.shape

    chunk_offset = [i[0] for i in block_info[0]['array-location']]

    chunk_edges = np.array([i for i in np.ndindex(tuple([2] * n))])\
        * np.array(chunk_shape) + np.array(chunk_offset)

    rel_input_edges = np.dot(matrix, chunk_edges.T).T + offset
    rel_input_i = np.min(rel_input_edges, 0)
    rel_input_f = np.max(rel_input_edges, 0)

    # expand relevant input to contain footprint of the spline kernel
    # according to
    #
    # https://github.com/scipy/scipy/blob/
    # 9c0d08d7d11fc33311a96d2ac3ad73c8f6e3df00/
    # scipy/ndimage/src/ni_interpolation.c#L412-L419
    #
    # Also see https://github.com/dask/dask-image/
    # issues/24#issuecomment-706165593
    for dim in range(n):
        if order % 1:
            start_lower = np.floor(rel_input_i[dim]) - order // 2
            start_upper = np.floor(rel_input_f[dim]) - order // 2
        else:
            start_lower = np.floor(rel_input_i[dim] + 0.5) - order // 2
            start_upper = np.floor(rel_input_f[dim] + 0.5) - order // 2

        stop_upper = start_upper + order
        # for some reason the behaviour is different to ndimage's
        # if leaving out the -1 in the next two lines
        rel_input_i[dim] = start_lower - 1
        rel_input_f[dim] = stop_upper + 1

    # clip input coordinates to input image extent
    for dim, upper in zip(range(n), input_shape):
        rel_input_i[dim] = np.clip(rel_input_i[dim], 0, upper)
        rel_input_f[dim] = np.clip(rel_input_f[dim], 0, upper)

    rel_input_slice = tuple([slice(int(rel_input_i[dim]),
                                   int(rel_input_f[dim]))
                             for dim in range(n)])

    rel_input = input[rel_input_slice]

    # modify offset to point into cropped input
    # y = Mx + o
    # coordinate substitution:
    # y' = y - y0(min_coord_px)
    # x' = x - x0(chunk_offset)
    # then:
    # y' = Mx' + o + Mx0 - y0
    # M' = M
    # o' = o + Mx0 - y0
    offset_prime = offset + np.dot(matrix, chunk_offset) - rel_input_i

    chunk = ndimage.affine_transform(rel_input,
                                     matrix,
                                     offset_prime,
                                     output_shape=chunk_shape,
                                     order=order,
                                     prefilter=False,
                                     **func_kwargs)

    return chunk
