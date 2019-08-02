# -*- coding: utf-8 -*-


__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


import itertools
import numbers
import warnings

import dask
import dask.array
import dask.delayed
import numpy
import pims

from . import _utils


def imread(fname, nframes=1):
    """
    Read image data into a Dask Array.

    Provides a simple, fast mechanism to ingest image data into a
    Dask Array.

    Parameters
    ----------
    fname : str
        A glob like string that may match one or multiple filenames.
    nframes : int, optional
        Number of the frames to include in each chunk (default: 1).

    Returns
    -------
    array : dask.array.Array
        A Dask Array representing the contents of all image files.
    """

    if not isinstance(nframes, numbers.Integral):
        raise ValueError("`nframes` must be an integer.")
    if (nframes != -1) and not (nframes > 0):
        raise ValueError("`nframes` must be greater than zero.")

    with pims.open(fname) as imgs:
        shape = (len(imgs),) + imgs.frame_shape
        dtype = numpy.dtype(imgs.pixel_type)

    if nframes == -1:
        nframes = shape[0]

    if nframes > shape[0]:
        warnings.warn(
            "`nframes` larger than number of frames in file."
            " Will truncate to number of frames in file.",
            RuntimeWarning
        )
    elif shape[0] % nframes != 0:
        warnings.warn(
            "`nframes` does not nicely divide number of frames in file."
            " Last chunk will contain the remainder.",
            RuntimeWarning
        )

    lower_iter, upper_iter = itertools.tee(itertools.chain(
        range(0, shape[0], nframes),
        [shape[0]]
    ))
    next(upper_iter)

    a = []
    for i, j in zip(lower_iter, upper_iter):
        a.append(dask.array.from_delayed(
            dask.delayed(_utils._read_frame)(fname, slice(i, j)),
            (j - i,) + shape[1:],
            dtype
        ))
    a = dask.array.concatenate(a)

    return a
