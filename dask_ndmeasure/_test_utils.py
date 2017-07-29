# -*- coding: utf-8 -*-

from __future__ import absolute_import

import dask.array.utils


def _assert_eq_nan(a, b, **kwargs):
    a = a.copy()
    b = b.copy()

    a_nan = (a != a)
    b_nan = (b != b)

    a[a_nan] = 0
    b[b_nan] = 0

    dask.array.utils.assert_eq(a_nan, b_nan, **kwargs)
    dask.array.utils.assert_eq(a, b, **kwargs)
