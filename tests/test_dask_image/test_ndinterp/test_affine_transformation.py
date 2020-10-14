#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import dask_image.ndinterp as da_ndinterp

import numpy as np
import dask.array as da
from scipy import ndimage


class Helpers:
    @staticmethod
    def test_affine_transform(n=2,
                              matrix=None,
                              offset=None,
                              input_output_shape_per_dim=(16, 16),
                              interp_order=1,
                              interp_mode='constant',
                              input_output_chunksize_per_dim=(6, 6),
                              random_seed=0,
                              use_cupy=False,
                              ):
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
        if use_cupy:
            import cupy as cp
            image_da = image_da.map_blocks(cp.asarray)

        # define (random) transformation
        if matrix is None:
            matrix = np.eye(n) + (np.random.random((n, n)) - 0.5) / 5.
        if offset is None:
            offset = (np.random.random(n) - 0.5) / 5. * np.array(image.shape)

        # define resampling options
        output_shape = [input_output_shape_per_dim[1]] * n
        output_chunks = [input_output_chunksize_per_dim[1]] * n

        # transform with scipy
        image_t_scipy = ndimage.affine_transform(
            image, matrix, offset,
            output_shape=output_shape,
            order=interp_order,
            mode=interp_mode,
            prefilter=False)

        # transform with dask-image
        image_t_dask = da_ndinterp.affine_transform(
            image_da, matrix, offset,
            output_shape=output_shape,
            output_chunks=output_chunks,
            order=interp_order,
            mode=interp_mode)
        image_t_dask_computed = image_t_dask.compute()

        assert np.allclose(image_t_scipy, image_t_dask_computed)


@pytest.fixture
def helpers():
    return Helpers


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
def test_affine_transform_general(n,
                                  input_output_shape_per_dim,
                                  interp_order,
                                  input_output_chunksize_per_dim,
                                  random_seed, helpers):

    kwargs = dict()
    kwargs['n'] = n
    kwargs['input_output_shape_per_dim'] = input_output_shape_per_dim
    kwargs['interp_order'] = interp_order
    kwargs['input_output_chunksize_per_dim'] = input_output_chunksize_per_dim
    kwargs['random_seed'] = random_seed

    helpers.test_affine_transform(**kwargs)


@pytest.mark.cupy
@pytest.mark.parametrize("n",
                         [1, 2, 3])
@pytest.mark.parametrize("input_output_shape_per_dim",
                         [(25, 25), (25, 10)])
@pytest.mark.parametrize("interp_order",
                         [0, 1])
@pytest.mark.parametrize("input_output_chunksize_per_dim",
                         [(16, 16), (16, 7)])
@pytest.mark.parametrize("random_seed",
                         [0])
def test_affine_transform_cupy(n,
                               input_output_shape_per_dim,
                               interp_order,
                               input_output_chunksize_per_dim,
                               random_seed, helpers):

    pytest.importorskip("cupy", minversion="6.0.0")

    # somehow, these lines are required for the first parametrized
    # test to succeed
    import cupy as cp
    from dask_image.dispatch._dispatch_ndinterp import (
        dispatch_affine_transform)
    dispatch_affine_transform(cp.asarray([]))

    kwargs = dict()
    kwargs['n'] = n
    kwargs['input_output_shape_per_dim'] = input_output_shape_per_dim
    kwargs['interp_order'] = interp_order
    kwargs['input_output_chunksize_per_dim'] = input_output_chunksize_per_dim
    kwargs['random_seed'] = random_seed
    kwargs['use_cupy'] = True

    helpers.test_affine_transform(**kwargs)


@pytest.mark.parametrize("n",
                         [1, 2, 3])
@pytest.mark.parametrize("interp_mode",
                         ['constant', 'nearest'])
@pytest.mark.parametrize("input_output_shape_per_dim",
                         [(20, 30)])
@pytest.mark.parametrize("input_output_chunksize_per_dim",
                         [(15, 10)])
def test_affine_transform_modes(n,
                                interp_mode,
                                input_output_shape_per_dim,
                                input_output_chunksize_per_dim,
                                helpers):

    kwargs = dict()
    kwargs['n'] = n
    kwargs['interp_mode'] = interp_mode
    kwargs['input_output_shape_per_dim'] = input_output_shape_per_dim
    kwargs['input_output_chunksize_per_dim'] = input_output_chunksize_per_dim

    helpers.test_affine_transform(**kwargs)


@pytest.mark.parametrize("interp_mode",
                         ['wrap', 'reflect', 'mirror'])
def test_affine_transform_unsupported_modes(interp_mode, helpers):

    kwargs = dict()
    kwargs['interp_mode'] = interp_mode

    with pytest.raises(NotImplementedError):
        helpers.test_affine_transform(**kwargs)


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
