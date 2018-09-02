# -*- coding: utf-8 -*-


__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


import numpy
import pims


def _read_frame(fn, i):
    with pims.open(fn) as imgs:
        return numpy.asanyarray(imgs[i])
