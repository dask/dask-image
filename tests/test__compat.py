# -*- coding: utf-8 -*-


import pytest

import numpy as np

import dask.array.utils as dau

import dask_ndfourier._compat


@pytest.mark.parametrize("n", [1, 2, 3, 6, 7])
@pytest.mark.parametrize("d", [1.0, 0.5, 2 * np.pi])
def test_fftfreq(n, d):
    dau.assert_eq(
        dask_ndfourier._compat._fftfreq(n, d, chunks=((n,),)),
        np.fft.fftfreq(n, d)
    )
