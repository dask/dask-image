# -*- coding: utf-8 -*-


__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import dask
import dask.array
import dask.delayed
import numpy
import pims


try:
    irange = xrange
except NameError:
    irange = range


def imread(fn):
    with pims.open(fn) as imgs:
        shape = (len(imgs),) + imgs.frame_shape
        dtype = numpy.dtype(imgs.pixel_type)

    def _read_frame(fn, i):
        with pims.open(fn) as imgs:
            return imgs[i]

    a = []
    for i in irange(shape[0]):
        a.append(dask.array.from_delayed(
            dask.delayed(_read_frame)(fn, i),
            shape[1:],
            dtype
        ))
    a = dask.array.stack(a)

    return a
