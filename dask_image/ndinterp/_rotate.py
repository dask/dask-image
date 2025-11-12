# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
from scipy.special import sindg, cosdg

from ._affine_transform import affine_transform


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
