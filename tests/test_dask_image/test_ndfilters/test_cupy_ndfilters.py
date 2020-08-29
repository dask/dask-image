#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

import dask_image.ndfilters as da_ndf

cupy = pytest.importorskip("cupy", minversion="7.0.0")


@pytest.mark.cupy
def test_cupy_convolve():
    """Test convolve filter with cupy input arrays."""
    s = (10, 10)
    a = da.from_array(cupy.arange(int(np.prod(s)), dtype=cupy.float32).reshape(s), chunks=5)
    w = cupy.ones(a.ndim * (3,), dtype=cupy.float32)
    result = da_ndf.convolve(a, w)
    result.compute()
