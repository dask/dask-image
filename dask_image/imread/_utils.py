# -*- coding: utf-8 -*-
import numpy as np
import pims


def _read_frame(fn, i, *, arrayfunc=np.asanyarray):
    with pims.open(fn) as imgs:
        return arrayfunc(imgs[i])
