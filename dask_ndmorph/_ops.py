# -*- coding: utf-8 -*-


import numpy

import dask.array


def _where(condition, x, y):
    if isinstance(condition, (bool, numpy.bool8)):
        dtype = numpy.promote_types(x.dtype, y.dtype)
        if condition:
            return x.astype(dtype)
        else:
            return y.astype(dtype)
    else:
        return dask.array.where(condition, x, y)
