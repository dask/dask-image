#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest
import scipy
from scipy import ndimage

import dask_image.ndinterp as da_ndinterp

# mode lists for the case with prefilter = False
_supported_modes = ['constant', 'nearest', 'reflect', 'mirror']
_unsupported_modes = ['wrap']

# additional modes are present in SciPy >= 1.6.0
if np.lib.NumpyVersion(scipy.__version__) >= '1.6.0':
    _supported_modes += ['grid-constant', 'grid-mirror']
    _unsupported_modes += ['grid-wrap']


def validate_spline_filter(n=2,
                           axis_size=64,
                           interp_order=1,
                           interp_mode='constant',
                           chunksize=32,
                           output=np.float64,
                           random_seed=0,
                           use_cupy=False,
                           axis=None):
    """
    Compare the outputs of `ndimage.spline_transform`
    and `dask_image.ndinterp.spline_transform`. If axis is not None, then
    `spline_transform1d` is tested instead.

    """

    # define test image
    np.random.seed(random_seed)
    image = np.random.random([axis_size] * n)

    # transform into dask array
    image_da = da.from_array(image, chunks=[chunksize] * n)
    if use_cupy:
        import cupy as cp
        image_da = image_da.map_blocks(cp.asarray)

    if axis is not None:
        scipy_func = ndimage.spline_filter1d
        dask_image_func = da_ndinterp.spline_filter1d
        kwargs = {'axis': axis}
    else:
        scipy_func = ndimage.spline_filter
        dask_image_func = da_ndinterp.spline_filter
        kwargs = {}

    # transform with scipy
    image_t_scipy = scipy_func(
        image,
        output=output,
        order=interp_order,
        mode=interp_mode,
        **kwargs)

    # transform with dask-image
    image_t_dask = dask_image_func(
        image_da,
        output=output,
        order=interp_order,
        mode=interp_mode,
        **kwargs)
    image_t_dask_computed = image_t_dask.compute()

    rtol = atol = 1e-6
    out_dtype = np.dtype(output)
    assert image_t_scipy.dtype == image_t_dask_computed.dtype == out_dtype
    assert np.allclose(image_t_scipy, image_t_dask_computed,
                       rtol=rtol, atol=atol)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("axis_size", [64])
@pytest.mark.parametrize("interp_order", range(2, 6))
@pytest.mark.parametrize("interp_mode", _supported_modes)
@pytest.mark.parametrize("chunksize", [32, 15])
def test_spline_filter_general(
    n,
    axis_size,
    interp_order,
    interp_mode,
    chunksize,
):

    validate_spline_filter(
        n=n,
        axis_size=axis_size,
        interp_order=interp_order,
        interp_mode=interp_mode,
        chunksize=chunksize,
        axis=None,
    )


@pytest.mark.cupy
@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("axis_size", [32])
@pytest.mark.parametrize("interp_order", range(2, 6))
@pytest.mark.parametrize("interp_mode", _supported_modes[::2])
@pytest.mark.parametrize("chunksize", [16])
@pytest.mark.parametrize("axis", [None, -1])
def test_spline_filter_cupy(
    n,
    axis_size,
    interp_order,
    interp_mode,
    chunksize,
    axis,
):

    cupy = pytest.importorskip("cupy", minversion="6.0.0")

    validate_spline_filter(
        n=n,
        axis_size=axis_size,
        interp_order=interp_order,
        interp_mode=interp_mode,
        chunksize=chunksize,
        axis=axis,
        use_cupy=True,
    )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("axis_size", [48, 27])
@pytest.mark.parametrize("interp_order", range(2, 6))
@pytest.mark.parametrize("interp_mode", _supported_modes)
@pytest.mark.parametrize("chunksize", [33])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_spline_filter1d_general(
    n,
    axis_size,
    interp_order,
    interp_mode,
    chunksize,
    axis,
):
    if axis == 1 and n < 2:
        pytest.skip(msg="skip axis=1 for 1d signals")

    validate_spline_filter(
        n=n,
        axis_size=axis_size,
        interp_order=interp_order,
        interp_mode=interp_mode,
        chunksize=chunksize,
        axis=axis,
    )

@pytest.mark.parametrize("axis_size", [32])
@pytest.mark.parametrize("interp_order", range(2, 6))
@pytest.mark.parametrize("interp_mode", _unsupported_modes)
@pytest.mark.parametrize("axis", [None, -1])
def test_spline_filter_unsupported_modes(
    axis_size,
    interp_order,
    interp_mode,
    axis,
):

    with pytest.raises(NotImplementedError):
        validate_spline_filter(
            axis_size=axis_size,
            interp_order=interp_order,
            interp_mode=interp_mode,
            axis=axis,
        )


@pytest.mark.parametrize(
    "output", [np.float64, np.float32, "float32", np.dtype(np.float32)]
)
@pytest.mark.parametrize("axis", [None, -1])
def test_spline_filter_output_dtype(
    output,
    axis,
):

    validate_spline_filter(
        axis_size=32,
        interp_order=3,
        output=output,
        axis=axis,
    )
