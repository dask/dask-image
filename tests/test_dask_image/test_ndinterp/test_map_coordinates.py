#!/usr/bin/env python
# -*- coding: utf-8 -*-
from packaging import version

import dask.array as da
import numpy as np
import pytest
import scipy
import scipy.ndimage

import dask_image.ndinterp

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
        coords).compute()
