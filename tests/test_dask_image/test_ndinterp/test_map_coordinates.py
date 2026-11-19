#!/usr/bin/env python
# -*- coding: utf-8 -*-
from packaging import version

import dask.array as da
import numpy as np
import pytest
import scipy
import scipy.ndimage

import dask_image.ndinterp

have_scipy16 = version.parse(scipy.__version__) >= version.parse('1.6.0')
_boundary_modes = [
    'constant', 'nearest', 'grid-constant', 'reflect', 'grid-mirror',
    'mirror', 'wrap', 'grid-wrap',
]


def validate_map_coordinates_general(n=2,
                                     interp_order=1,
                                     interp_mode='constant',
                                     coord_len=12,
                                     coord_chunksize=6,
                                     coord_offset=0.,
                                     im_shape_per_dim=12,
                                     im_chunksize_per_dim=6,
                                     random_seed=0,
                                     prefilter=False,
                                     ):

    if interp_order > 1 and interp_mode == 'nearest' and not have_scipy16:
        # not clear on the underlying cause, but this fails on older SciPy
        pytest.skip("requires SciPy >= 1.6.0")

    # define test input
    np.random.seed(random_seed)
    input = np.random.random([im_shape_per_dim] * n)
    input_da = da.from_array(input, chunks=im_chunksize_per_dim)

    # define test coordinates
    coords = np.random.random((n, coord_len)) * im_shape_per_dim + coord_offset
    coords_da = da.from_array(coords, chunks=(n, coord_chunksize))

    # ndimage result
    mapped_scipy = scipy.ndimage.map_coordinates(
        input,
        coords,
        order=interp_order,
        mode=interp_mode,
        cval=0.0,
        prefilter=prefilter)

    # dask-image results
    for input_array in [input, input_da]:
        for coords_array in [coords, coords_da]:
            mapped_dask = dask_image.ndinterp.map_coordinates(
                input_array,
                coords_array,
                order=interp_order,
                mode=interp_mode,
                cval=0.0,
                prefilter=prefilter)

            mapped_dask_computed = mapped_dask.compute()

            assert np.allclose(mapped_scipy, mapped_dask_computed)


@pytest.mark.parametrize("n",
                         [1, 2, 3, 4])
@pytest.mark.parametrize("random_seed",
                         range(2))
def test_map_coordinates_basic(n,
                               random_seed,
                               ):

    kwargs = dict()
    kwargs['n'] = n
    kwargs['random_seed'] = random_seed

    validate_map_coordinates_general(**kwargs)


@pytest.mark.timeout(3)
def test_map_coordinates_large_input():

    """
    This test assesses whether relatively large
    inputs are processed before timeout.
    """

    # define large test image
    image_da = da.random.random([1000] * 3, chunks=200)

    # define sparse test coordinates
    coords = np.random.random((3, 2)) * 1000

    # dask-image result
    dask_image.ndinterp.map_coordinates(
        image_da,
        coords,
        prefilter=False).compute()


@pytest.mark.parametrize("interp_mode",
                         _boundary_modes)
def test_map_coordinates_out_of_bounds(interp_mode):
    """
    Out-of-bounds coordinates follow the boundary mode of the full image.
    """

    kwargs = dict()
    kwargs['random_seed'] = 0
    kwargs['interp_mode'] = interp_mode
    kwargs['im_shape_per_dim'] = 10
    kwargs['coord_offset'] = 10  # coordinates will be out of bounds

    validate_map_coordinates_general(**kwargs)


@pytest.mark.parametrize("mode", _boundary_modes)
@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_map_coordinates_prefilters_before_cropping(mode, order):
    rng = np.random.default_rng(82)
    image = rng.random((61, 72))
    coords = np.array([
        [-70.2, -0.6, 0.15, 16.7, 17.1, 50.2, 59.6, 61.5, 132.6],
        [131.4, 21.6, 0.5, 19.4, 20.2, 69.9, 71.1, -2.1, -82.7],
    ])
    expected = scipy.ndimage.map_coordinates(
        image, coords, order=order, mode=mode, cval=-0.3, prefilter=True,
    )
    result = dask_image.ndinterp.map_coordinates(
        da.from_array(image, chunks=(17, 20)),
        da.from_array(coords, chunks=(1, 3)),
        order=order, mode=mode, cval=-0.3, prefilter=True,
    )
    np.testing.assert_allclose(result.compute(), expected, atol=1e-11)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.complex64])
def test_map_coordinates_default_preserves_dtype_and_coordinate_shape(dtype):
    image = np.arange(99).reshape(9, 11).astype(dtype)
    if np.iscomplexobj(image):
        image += 1j * image[::-1]
    coords = np.array([
        [[0.2, 4.3, 8.0], [3.0, 4.8, 7.2]],
        [[0.7, 5.1, 10.0], [2.0, 5.7, 9.1]],
    ])
    expected = scipy.ndimage.map_coordinates(image, coords)
    result = dask_image.ndinterp.map_coordinates(
        da.from_array(image, chunks=(3, 6)),
        da.from_array(coords, chunks=(1, 1, 2)),
    )
    assert result.dtype == image.dtype
    assert result.chunks == ((1, 1), (2, 1))
    np.testing.assert_allclose(result.compute(), expected)


def test_map_coordinates_prefilter_does_not_overflow_integer_coordinates():
    image = np.arange(270, dtype=float)
    coords = np.array([[245, 250, 255]], dtype=np.uint8)
    expected = scipy.ndimage.map_coordinates(image, coords, mode='nearest')
    result = dask_image.ndinterp.map_coordinates(
        da.from_array(image, chunks=70), coords, mode='nearest',
    )
    np.testing.assert_allclose(result.compute(), expected)
