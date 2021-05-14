# -*- coding: utf-8 -*-
import numpy
import pims


def _read_frame(fn, i, *, arrayfunc=numpy.asanyarray):
    with pims.open(fn) as imgs:
        return arrayfunc(imgs[i])
