#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import dask_image.ndinterp as da_ndinterp

import numpy as np
import dask.array as da


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("interp_order", [0, 1, 3])
def test_affine_transformation(n, interp_order):
    """
    Compare the outputs of `ndimage.affine_transformation`
    and `dask_image.ndinterp.affine_transformation`.

    Notes
    -----
        Currently, prefilter is disabled and therefore the output
        of `dask_image.ndinterp.affine_transformation` is compared
        to `prefilter=False`.
    """

    # define test image
    a = 25
    im = np.random.random([a] * n)

    # transform into dask array
    chunksize = [16] * n
    dim = da.from_array(im, chunks=chunksize)

    # define (random) transformation
    matrix = np.eye(n) + (np.random.random((n, n)) - 0.5) / 5.
    offset = (np.random.random(n) - 0.5) / 5. * np.array(im.shape)

    # define resampling options
    output_shape = [int(a / 2)] * n
    output_chunks = [16] * n

    from scipy import ndimage
    # transform with scipy
    im_t_scipy = ndimage.affine_transform(im, matrix, offset,
                                          output_shape=output_shape,
                                          order=interp_order,
                                          prefilter=False)

    # transform with dask-image
    im_t_dask = da_ndinterp.affine_transform(dim, matrix, offset,
                                             output_shape=output_shape,
                                             output_chunks=output_chunks,
                                             order=interp_order)
    im_t_dask_computed = im_t_dask.compute()

    print(im_t_scipy, im_t_dask_computed)
    assert np.allclose(im_t_scipy, im_t_dask_computed)

