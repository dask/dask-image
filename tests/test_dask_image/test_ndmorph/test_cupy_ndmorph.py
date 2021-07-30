#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

import dask_image.ndmorph

cupy = pytest.importorskip("cupy", minversion="7.7.0")


@pytest.fixture
def array():
    s = (10, 10)
    a = da.from_array(cupy.arange(int(np.prod(s)),
                      dtype=cupy.float32).reshape(s), chunks=5)
    return a


@pytest.mark.cupy
@pytest.mark.parametrize("func", [
    dask_image.ndmorph.binary_closing,
    dask_image.ndmorph.binary_dilation,
    dask_image.ndmorph.binary_erosion,
    dask_image.ndmorph.binary_opening,
])
def test_cupy_ndmorph(array, func):
    """Test convolve & correlate filters with cupy input arrays."""
    result = func(array)
    assert result.dtype == bool
    assert result._meta.dtype == bool
    assert isinstance(result._meta, cupy.ndarray)
    computed = result.compute()
    assert computed.dtype == bool
    assert isinstance(computed, cupy.ndarray)
