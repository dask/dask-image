# -*- coding: utf-8 -*-

"""
Content here is borrowed from our contributions to Dask.
"""


import pytest

import numpy as np

import dask.array.core as dac
import dask.array.utils as dau

import dask_image.ndfourier._compat


@pytest.mark.parametrize("n", [1, 2, 3, 6, 7])
@pytest.mark.parametrize("d", [1.0, 0.5, 2 * np.pi])
@pytest.mark.parametrize("c", [lambda m: m, lambda m: (1, m - 1)])
def test_fftfreq(n, d, c):
    c = c(n)

    r1 = np.fft.fftfreq(n, d)
    r2 = dask_image.ndfourier._compat._fftfreq(n, d, chunks=c)

    assert dac.normalize_chunks(c, r2.shape) == r2.chunks

    dau.assert_eq(r1, r2)
