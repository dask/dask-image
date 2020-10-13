#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import dask_image.ndinterp as da_ndinterp

import numpy as np
import dask.array as da
from scipy import ndimage


@pytest.mark.parametrize("n",
                         [1, 2, 3])
@pytest.mark.parametrize("input_output_shape_per_dim",
                         [(25, 25), (25, 10)])
@pytest.mark.parametrize("interp_order",
                         range(6))
@pytest.mark.parametrize("input_output_chunksize_per_dim",
                         [(16, 16), (16, 7), (7, 16)])
@pytest.mark.parametrize("random_seed",
                         [0, 1, 2])
def test_affine_transform(n,
                          input_output_shape_per_dim,
                          interp_order,
                          input_output_chunksize_per_dim,
                          random_seed):
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
    a = input_output_shape_per_dim[0]
    np.random.seed(random_seed)
    image = np.random.random([a] * n)

    # transform into dask array
    chunksize = [input_output_chunksize_per_dim[0]] * n
    image_da = da.from_array(image, chunks=chunksize)

    # define (random) transformation
    matrix = np.eye(n) + (np.random.random((n, n)) - 0.5) / 5.
    offset = (np.random.random(n) - 0.5) / 5. * np.array(image.shape)

    # define resampling options
    output_shape = [input_output_shape_per_dim[1]] * n
    output_chunks = [input_output_chunksize_per_dim[1]] * n

    # transform with scipy
    image_t_scipy = ndimage.affine_transform(image, matrix, offset,
                                             output_shape=output_shape,
                                             order=interp_order,
                                             prefilter=False)

    # transform with dask-image
    image_t_dask = da_ndinterp.affine_transform(image_da, matrix, offset,
                                                output_shape=output_shape,
                                                output_chunks=output_chunks,
                                                order=interp_order)
    image_t_dask_computed = image_t_dask.compute()

    assert np.allclose(image_t_scipy, image_t_dask_computed)


def test_affine_transform_no_output_shape_or_chunks_specified():

    image = np.ones((3, 3))
    image_t = da_ndinterp.affine_transform(image, np.eye(2), [0, 0])

    assert image_t.shape == image.shape
    assert image_t.chunks == tuple([(s,) for s in image.shape])


def test_affine_transform_prefilter_warning():

    with pytest.warns(UserWarning):
        da_ndinterp.affine_transform(np.ones(3), [1], [0],
                                     order=3, prefilter=True)


@pytest.mark.filterwarnings("ignore:The behavior of affine_transform "
                            "with a 1-D array supplied for the matrix "
                            "parameter has changed")
@pytest.mark.parametrize("n",
                         [1, 2, 3, 4])
def test_affine_transform_parameter_formats(n):

    # define reference parameters
    scale_factors = np.ones(n, dtype=np.float) * 2.
    matrix_n = np.diag(scale_factors)
    offset = -np.ones(n)

    # convert into different formats
    matrix_only_scaling = scale_factors
    matrix_pre_homogeneous = np.hstack((matrix_n, offset[:, None]))
    matrix_homogeneous = np.vstack((matrix_pre_homogeneous,
                                   [0] * n + [1]))

    np.random.seed(0)
    image = np.random.random([5] * n)

    # reference run
    image_t_0 = da_ndinterp.affine_transform(image,
                                             matrix_n,
                                             offset).compute()

    # assert that the different parameter formats
    # lead to the same output
    image_t_scale = da_ndinterp.affine_transform(image,
                                                 matrix_only_scaling,
                                                 offset).compute()
    assert (np.allclose(image_t_0, image_t_scale))

    for matrix in [matrix_pre_homogeneous, matrix_homogeneous]:

        image_t = da_ndinterp.affine_transform(image,
                                               matrix,
                                               offset + 10.,  # ignored
                                               ).compute()

        assert(np.allclose(image_t_0, image_t))

    # catch matrices that are not homogeneous transformation matrices
    with pytest.raises(ValueError):
        matrix_not_homogeneous = np.vstack((matrix_pre_homogeneous,
                                           [-1] * n + [1]))
        da_ndinterp.affine_transform(image,
                                     matrix_not_homogeneous,
                                     offset)
