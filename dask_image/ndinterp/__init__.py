# -*- coding: utf-8 -*-


import numpy as np
import dask.array as da
from scipy.ndimage import affine_transform as ndimage_affine_transform
import warnings

from ..dispatch._dispatch_ndinterp import (
    dispatch_affine_transform,
    dispatch_asarray,
)


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
    offset : array (ndim,)
        Transformation offset.
    output_shape : array (ndim,), optional
        The size of the array to be returned.
    order : int, optional
        The order of the spline interpolation. Note that for order>1
        scipy's affine_transform applies prefiltering, which is not
        yet supported and skipped in this implementation.
    output_chunks : array (ndim,), optional
        The chunks of the output Dask Array.

    Returns
    -------
    affine_transform : Dask Array
        A dask array representing the transformed output

    """

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
        # if matrix.shape[0] == image.ndim + 1:
        #     exptd = [0] * image.ndim + [1]
        #     if not np.all(matrix[image.ndim] == exptd):
        #         msg = ('Expected homogeneous transformation matrix with '
        #                'shape %s for image shape %s, but bottom row was '
        #                'not equal to %s' % (matrix.shape, image.shape, exptd))
        #         raise ValueError(msg)
        # assume input is homogeneous coordinate transformation matrix
        offset = matrix[:image.ndim, image.ndim]
        matrix = matrix[:image.ndim, :image.ndim]

    # process kwargs
    # prefilter is not yet supported
    if 'prefilter' in kwargs:
        if kwargs['prefilter'] and order > 1:
            warnings.warn('Currently, `dask_image.ndinterp.affine_transform` '
                          'doesn\'t support `prefilter=True`. Proceeding with'
                          ' `prefilter=False`, which if order > 1 can lead '
                          'to the output containing more blur than with '
                          'prefiltering.', UserWarning)
        del kwargs['prefilter']

    if 'mode' in kwargs:
        if kwargs['mode'] in ['wrap', 'reflect', 'mirror']:
            raise(NotImplementedError("Mode %s is not currently supported."
                                      % kwargs['mode']))

    transformed = da.zeros(output_shape,
                           dtype=image.dtype,
                           chunks=output_chunks)

    # define meta array to inform dask of the output chunk type
    meta = dispatch_asarray(image)([]).astype(image.dtype)

    # map affine_transform onto array
    transformed = transformed.map_blocks(resample_chunk,
                                         dtype=image.dtype,
                                         image=image,
                                         matrix=matrix,
                                         offset=offset,
                                         order=order,
                                         func_kwargs=kwargs,
                                         meta=meta
                                         )

    return transformed


def resample_chunk(chunk, image, matrix, offset, order, func_kwargs, block_info=None):
    """Resample a given chunk, used by `affine_transform`."""

    n = chunk.ndim
    image_shape = image.shape
    chunk_shape = chunk.shape

    # extract coordinates of chunk edges
    chunk_offset = [i[0] for i in block_info[0]['array-location']]
    chunk_edges = np.array([i for i in np.ndindex(tuple([2] * n))])\
        * np.array(chunk_shape) + np.array(chunk_offset)

    # map output chunk edges onto input image coordinates
    # to define the input region relevant for the current chunk
    if matrix.ndim == 1 and len(matrix) == image.ndim:
        rel_image_edges = matrix * chunk_edges + offset
    else:
        rel_image_edges = np.dot(matrix, chunk_edges.T).T + offset

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

        # for some reason the behaviour becomes inconsistent with ndimage
        # if leaving out the -1 in the next line
        rel_image_i[dim] = np.floor(rel_image_i[dim]) - order // 2
        rel_image_f[dim] = np.floor(rel_image_f[dim]) - order // 2 + order

        if order == 0:
            rel_image_i[dim] -= 1

    # clip image coordinates to image extent
    for dim, s in zip(range(n), image_shape):
        rel_image_i[dim] = np.clip(rel_image_i[dim], 0, s - 1)
        rel_image_f[dim] = np.clip(rel_image_f[dim], 0, s - 1)

    rel_image_slice = tuple([slice(int(rel_image_i[dim]),
                                   int(rel_image_f[dim]) + 1)
                             for dim in range(n)])

    rel_image = image[rel_image_slice]

    # Modify offset to point into cropped image.
    # y = Mx + o
    # Coordinate substitution:
    # y' = y - y0(min_coord_px)
    # x' = x - x0(chunk_offset)
    # Then:
    # y' = Mx' + o + Mx0 - y0
    # M' = M
    # o' = o + Mx0 - y0
    offset_prime = offset + np.dot(matrix, chunk_offset) - rel_image_i

    affine_transform_method = dispatch_affine_transform(image)
    asarray_method = dispatch_asarray(image)

    chunk = affine_transform_method(rel_image,
                                    asarray_method(matrix),
                                    offset_prime,
                                    output_shape=chunk_shape,
                                    order=order,
                                    prefilter=False,
                                    **func_kwargs)

    return chunk
