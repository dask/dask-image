#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage.filters as sp_ndf

import dask
import dask.array as da
import dask.array.utils as dau

import dask_image.ndfilters as da_ndf


def test_laplace_comprehensions():
    np.random.seed(0)

    a = np.random.random((3, 12, 14))
    d = da.from_array(a, chunks=(3, 6, 7))

    l2s = [da_ndf.laplace(d[i]) for i in range(len(d))]
    l2c = [da_ndf.laplace(d[i])[None] for i in range(len(d))]

    dau.assert_eq(np.stack(l2s), da.stack(l2s))
    dau.assert_eq(np.concatenate(l2c), da.concatenate(l2c))


def test_laplace_compare():
    s = (10, 11, 12)
    a = np.arange(float(np.prod(s))).reshape(s)
    d = da.from_array(a, chunks=(5, 5, 6))

    dau.assert_eq(
        sp_ndf.laplace(a),
        da_ndf.laplace(d)
    )
