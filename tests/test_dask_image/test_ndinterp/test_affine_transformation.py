#!/usr/bin/env python
# -*- coding: utf-8 -*-

from packaging import version

import dask
import dask.array as da
import numpy as np
import pytest
import scipy
from scipy import ndimage

import dask_image.ndinterp as da_ndinterp

# mode lists for the case with prefilter = False
_supported_modes = ['constant', 'nearest']
_unsupported_modes = ['wrap', 'reflect', 'mirror']

# mode lists for the case with prefilter = True
_supported_prefilter_modes = ['constant']
_unsupported_prefilter_modes = _unsupported_modes + ['nearest']

have_scipy16 = version.parse(scipy.__version__) >= version.parse('1.6.0')

# additional modes are present in SciPy >= 1.6.0
if have_scipy16:
    _supported_modes += ['grid-constant']
    _unsupported_modes += ['grid-mirror', 'grid-wrap']
    _unsupported_prefilter_modes += ['grid-constant', 'grid-mirror',
                                     'grid-wrap']


def validate_affine_transform(n=2,
                              matrix=None,
                              offset=None,
                              input_output_shape_per_dim=(16, 16),
                              interp_order=1,
                              interp_mode='constant',
                              input_output_chunksize_per_dim=(6, 6),
                              random_seed=0,
                              use_cupy=False,
                              prefilter=False
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

    if (interp_order > 1 and interp_mode == 'nearest' and not have_scipy16):
        # not clear on the underlying cause, but this fails on older SciPy
        pytest.skip("requires SciPy >= 1.6.0")

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

    if (prefilter
        and interp_mode in _supported_prefilter_modes
        and interp_order > 1
        and version.parse(dask.__version__) < version.parse("2020.1.0")
    ):
        # older dask will fail if any chunks have size smaller than depth
        depth = da_ndinterp._get_default_depth(interp_order)
        in_size = input_output_shape_per_dim[0]
        in_chunksize = input_output_chunksize_per_dim[0]
        rem = in_size % in_chunksize
        if in_size < depth or (rem != 0 and rem < depth):
            pytest.skip("older dask doesn't automatically rechunk")

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
        prefilter=prefilter)

    # transform with dask-image
    image_t_dask = da_ndinterp.affine_transform(
        image_da, matrix, offset,
        output_shape=output_shape,
        output_chunks=output_chunks,
        order=interp_order,
        mode=interp_mode,
        prefilter=prefilter)
    image_t_dask_computed = image_t_dask.compute()

    assert np.allclose(image_t_scipy, image_t_dask_computed)


@pytest.mark.parametrize("n",
                         [1, 2, 3])
@pytest.mark.parametrize("input_output_shape_per_dim",
                         [(25, 25)])
@pytest.mark.parametrize("interp_order",
                         range(6))
@pytest.mark.parametrize("input_output_chunksize_per_dim",
                         [(16, 16), (16, 7), (7, 16)])
@pytest.mark.parametrize("random_seed",
                         [0, 2])
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

    validate_affine_transform(**kwargs)


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

    validate_affine_transform(**kwargs)


@pytest.mark.parametrize("n",
                         [1, 2, 3])
@pytest.mark.parametrize("interp_mode",
                         _supported_modes)
@pytest.mark.parametrize("interp_order",
                         [0, 3])
@pytest.mark.parametrize("input_output_shape_per_dim",
                         [(20, 30)])
@pytest.mark.parametrize("input_output_chunksize_per_dim",
                         [(15, 10)])
def test_affine_transform_modes(n,
                                interp_mode,
                                interp_order,
                                input_output_shape_per_dim,
                                input_output_chunksize_per_dim,
                                ):

    kwargs = dict()
    kwargs['n'] = n
    kwargs['interp_mode'] = interp_mode
    kwargs['input_output_shape_per_dim'] = input_output_shape_per_dim
    kwargs['input_output_chunksize_per_dim'] = input_output_chunksize_per_dim
    kwargs['interp_order'] = interp_order
    kwargs['prefilter'] = False

    validate_affine_transform(**kwargs)


@pytest.mark.parametrize("interp_mode",
                         _unsupported_modes)
def test_affine_transform_unsupported_modes(interp_mode):

    with pytest.raises(NotImplementedError):
        validate_affine_transform(interp_mode=interp_mode)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("interp_order", range(6))
@pytest.mark.parametrize("interp_mode", _supported_prefilter_modes)
def test_affine_transform_prefilter_modes(n, interp_order, interp_mode):

    validate_affine_transform(
        n=n,
        input_output_shape_per_dim=(32, 32),
        input_output_chunksize_per_dim=(24, 24),
        interp_order=interp_order,
        interp_mode=interp_mode,
        prefilter=True,
    )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("interp_order", range(2, 6))
@pytest.mark.parametrize("interp_mode", _unsupported_prefilter_modes)
def test_affine_transform_prefilter_not_implemented(
    n, interp_order, interp_mode
):

    with pytest.raises(NotImplementedError):
        validate_affine_transform(
            n=n,
            interp_order=interp_order,
            interp_mode=interp_mode,
            prefilter=True,
        )


def test_affine_transform_numpy_input():

    image = np.ones((3, 3))
    image_t = da_ndinterp.affine_transform(image, np.eye(2), [0, 0])

    assert image_t.shape == image.shape
    assert (image == image_t).min()


def test_affine_transform_minimal_input():

    image = np.ones((3, 3))
    image_t = da_ndinterp.affine_transform(np.ones((3, 3)), np.eye(2))

    assert image_t.shape == image.shape


def test_affine_transform_type_consistency():

    image = da.ones((3, 3))
    image_t = da_ndinterp.affine_transform(image, np.eye(2), [0, 0])

    assert isinstance(image, type(image_t))
    assert isinstance(image[0, 0].compute(), type(image_t[0, 0].compute()))


@pytest.mark.cupy
def test_affine_transform_type_consistency_gpu():

    cupy = pytest.importorskip("cupy", minversion="6.0.0")

    image = da.ones((3, 3))
    image_t = da_ndinterp.affine_transform(image, np.eye(2), [0, 0])

    image.map_blocks(cupy.asarray)

    assert isinstance(image, type(image_t))
    assert isinstance(image[0, 0].compute(), type(image_t[0, 0].compute()))


def test_affine_transform_no_output_shape_or_chunks_specified():

    image = da.ones((3, 3))
    image_t = da_ndinterp.affine_transform(image, np.eye(2), [0, 0])

    assert image_t.shape == image.shape
    assert image_t.chunks == tuple([(s,) for s in image.shape])


def test_affine_transform_prefilter_warning():

    with pytest.warns(UserWarning):
        da_ndinterp.affine_transform(da.ones(3), [1], [0],
                                     order=3, prefilter=False)


@pytest.mark.timeout(15)
def test_affine_transform_large_input_small_output_cpu():
    """
    Make sure input array does not need to be computed entirely
    """

    # fully computed, this array would occupy 8TB
    image = da.random.random([10000] * 3, chunks=(200, 200, 200))
    image_t = da_ndinterp.affine_transform(image, np.eye(3), [0, 0, 0],
                                           output_chunks=[1, 1, 1],
                                           output_shape=[1, 1, 1])

    # if more than the needed chunks should be computed,
    # this would take long and eventually raise a MemoryError
    image_t[0, 0, 0].compute()


@pytest.mark.cupy
@pytest.mark.timeout(15)
def test_affine_transform_large_input_small_output_gpu():
    """
    Make sure input array does not need to be computed entirely
    """
    cupy = pytest.importorskip("cupy", minversion="6.0.0")

    # this array would occupy more than 24GB on a GPU
    image = da.random.random([2000] * 3, chunks=(50, 50, 50))
    image.map_blocks(cupy.asarray)

    image_t = da_ndinterp.affine_transform(image, np.eye(3), [0, 0, 0],
                                           output_chunks=[1, 1, 1],
                                           output_shape=[1, 1, 1])
    # if more than the needed chunks should be computed,
    # this would take long and eventually raise a MemoryError
    image_t[0, 0, 0].compute()


@pytest.mark.filterwarnings("ignore:The behavior of affine_transform "
                            "with a 1-D array supplied for the matrix "
                            "parameter has changed")
@pytest.mark.parametrize("n",
                         [1, 2, 3, 4])
def test_affine_transform_parameter_formats(n):

    # define reference parameters
    scale_factors = np.ones(n, dtype=float) * 2.
    matrix_n = np.diag(scale_factors)
    offset = -np.ones(n)

    # convert into different formats
    matrix_only_scaling = scale_factors
    matrix_pre_homogeneous = np.hstack((matrix_n, offset[:, None]))
    matrix_homogeneous = np.vstack((matrix_pre_homogeneous,
                                   [0] * n + [1]))

    np.random.seed(0)
    image = da.random.random([5] * n)

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
