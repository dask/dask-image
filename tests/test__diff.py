#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import scipy.ndimage.filters as sp_ndf

import dask.array as da
import dask.array.utils as dau

import dask_ndfilters as da_ndf


def test_laplace_compare():
    s = (10, 11, 12)
    a = np.arange(float(np.prod(s))).reshape(s)
    d = da.from_array(a, chunks=(5, 5, 6))

    dau.assert_eq(
        sp_ndf.laplace(a),
        da_ndf.laplace(d)
    )
