#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

from dask_image import ndmorph

cupy = pytest.importorskip("cupy", minversion="7.7.0")


@pytest.fixture
def array():
    s = (10, 10)
    a = da.from_array(cupy.arange(int(np.prod(s)),
                      dtype=cupy.float32).reshape(s), chunks=5)
    return a


@pytest.mark.cupy
@pytest.mark.parametrize("func", [
    ndmorph.binary_closing,
    ndmorph.binary_dilation,
    ndmorph.binary_erosion,
    ndmorph.binary_opening,
])
def test_cupy_ndmorph(array, func):
    """Test convolve & correlate filters with cupy input arrays."""
    result = func(array)
    result.compute()
