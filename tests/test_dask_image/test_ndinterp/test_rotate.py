#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import dask_image.ndinterp as da_ndinterp

import numpy as np
import dask.array as da
from scipy import ndimage


def validate_rotate(n=2,
                    angle=0,
                    axes=(0,1),
                    resize=False,
                    input_output_shape_per_dim=(16,16),
                    interp_order=1,
                    interp_mode='constant',
                    input_output_chunksize_per_dim=(6,6),
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
    
    angle = np.random.random() * 360 - 180

    # transform into dask array
    chunksize = [input_output_chunksize_per_dim[0]] * n
    image_da = da.from_array(image, chunks=chunksize)
    if use_cupy:
        import cupy as cp
        image_da = image_da.map_blocks(cp.asarray)


    # define resampling options
    output_chunks = [input_output_chunksize_per_dim[1]] * n

    # transform with scipy
    image_t_scipy = ndimage.rotate(
        image, angle,
        axes=axes,
        resize=resize,
        order=interp_order,
        mode=interp_mode,
        prefilter=False)

    # transform with dask-image
    image_t_dask = da_ndinterp.rotate(
        image, angle,
        axes=axes,
        resize=resize,
        order=interp_order,
        mode=interp_mode,)
    image_t_dask_computed = image_t_dask.compute()

    assert np.allclose(image_t_scipy, image_t_dask_computed)


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
                                  random_seed):

    kwargs = dict()
    kwargs['n'] = n
    kwargs['input_output_shape_per_dim'] = input_output_shape_per_dim
    kwargs['interp_order'] = interp_order
    kwargs['input_output_chunksize_per_dim'] = input_output_chunksize_per_dim
    kwargs['random_seed'] = random_seed

    validate_rotate(**kwargs)


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
                               random_seed):
    cupy = pytest.importorskip("cupy", minversion="6.0.0")

    kwargs = dict()
    kwargs['n'] = n
    kwargs['input_output_shape_per_dim'] = input_output_shape_per_dim
    kwargs['interp_order'] = interp_order
    kwargs['input_output_chunksize_per_dim'] = input_output_chunksize_per_dim
    kwargs['random_seed'] = random_seed
    kwargs['use_cupy'] = True

    validate_rotate(**kwargs)


@pytest.mark.parametrize("n",
                         [1, 2, 3])
@pytest.mark.parametrize("interp_mode",
                         ['constant', 'nearest'])
@pytest.mark.parametrize("input_output_shape_per_dim",
                         [(20, 30)])
@pytest.mark.parametrize("input_output_chunksize_per_dim",
                         [(15, 10)])
def test_rotate_modes(n,
                                interp_mode,
                                input_output_shape_per_dim,
                                input_output_chunksize_per_dim,
                                ):

    kwargs = dict()
    kwargs['n'] = n
    kwargs['interp_mode'] = interp_mode
    kwargs['input_output_shape_per_dim'] = input_output_shape_per_dim
    kwargs['input_output_chunksize_per_dim'] = input_output_chunksize_per_dim
    kwargs['interp_order'] = 0

    validate_rotate(**kwargs)


@pytest.mark.parametrize("interp_mode",
                         ['wrap', 'reflect', 'mirror'])
def test_rotate_unsupported_modes(interp_mode):

    kwargs = dict()
    kwargs['interp_mode'] = interp_mode

    with pytest.raises(NotImplementedError):
        validate_rotate(**kwargs)


def test_rotate_numpy_input():

    image = np.ones((3, 3))
    image_t = da_ndinterp.rotate(image, 15, [0, 0])

    assert image_t.shape == image.shape
    assert (image == image_t).min()


def test_rotate_minimal_input():

    image = np.ones((3, 3))
    image_t = da_ndinterp.affine_transform(np.ones((3, 3)), 0)

    assert image_t.shape == image.shape


def test_rotate_type_consistency():

    image = da.ones((3, 3))
    image_t = da_ndinterp.rotate(image, 0)

    assert isinstance(image, type(image_t))
    assert isinstance(image[0, 0].compute(), type(image_t[0, 0].compute()))


@pytest.mark.cupy
def test_rotate_type_consistency_gpu():

    cupy = pytest.importorskip("cupy", minversion="6.0.0")

    image = da.ones((3, 3))
    image_t = da_ndinterp.rotate(image, 0)

    image.map_blocks(cupy.asarray)

    assert isinstance(image, type(image_t))
    assert isinstance(image[0, 0].compute(), type(image_t[0, 0].compute()))


def test_rotate_no_output_shape_or_chunks_specified():

    image = da.ones((3, 3))
    image_t = da_ndinterp.affine_transform(image, 0)

    assert image_t.shape == image.shape
    assert image_t.chunks == tuple([(s,) for s in image.shape])


def test_rotate_prefilter_warning():

    with pytest.warns(UserWarning):
        da_ndinterp.rotate(da.ones(3), 0,
                           order=3, prefilter=True)


@pytest.mark.timeout(15)
def test_rotate_large_input_small_output_cpu():
    """
    Make sure input array does not need to be computed entirely
    """

    # fully computed, this array would occupy 8TB
    image = da.random.random([10000] * 3, chunks=(200, 200, 200))
    image_t = da_ndinterp.rotate(image, 0,
                                 output_chunks=[1, 1, 1])

    # if more than the needed chunks should be computed,
    # this would take long and eventually raise a MemoryError
    image_t[0, 0, 0].compute()


@pytest.mark.cupy
@pytest.mark.timeout(15)
def test_rotate_large_input_small_output_gpu():
    """
    Make sure input array does not need to be computed entirely
    """
    cupy = pytest.importorskip("cupy", minversion="6.0.0")

    # this array would occupy more than 24GB on a GPU
    image = da.random.random([2000] * 3, chunks=(50, 50, 50))
    image.map_blocks(cupy.asarray)

    image_t = da_ndinterp.rotate(image, 0,
                                 output_chunks=[1, 1, 1])
    
    # if more than the needed chunks should be computed,
    # this would take long and eventually raise a MemoryError
    image_t[0, 0, 0].compute()

