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


def imread(fname, nframes=1, *, arraytype="numpy"):
    """
    Read image data into a Dask Array.

    Provides a simple, fast mechanism to ingest image data into a
    Dask Array.

    Parameters
    ----------
    fname : str or pathlib.Path
        A glob like string that may match one or multiple filenames.
    nframes : int, optional
        Number of the frames to include in each chunk (default: 1).
    arraytype : str, optional
        Array type for dask chunks. Available options: "numpy", "cupy".

    Returns
    -------
    array : dask.array.Array
        A Dask Array representing the contents of all image files.
    """

    sfname = str(fname)
    if not isinstance(nframes, numbers.Integral):
        raise ValueError("`nframes` must be an integer.")
    if (nframes != -1) and not (nframes > 0):
        raise ValueError("`nframes` must be greater than zero.")

    if arraytype == "numpy":
        arrayfunc = numpy.asanyarray
    elif arraytype == "cupy":   # pragma: no cover
        import cupy
        arrayfunc = cupy.asanyarray

    with pims.open(sfname) as imgs:
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

    a = dask.array.map_blocks(
        _map_read_frame,
        chunks=dask.array.core.normalize_chunks(
            (nframes,) + shape[1:], shape),
        fn=sfname,
        arrayfunc=arrayfunc,
        meta=arrayfunc([]).astype(dtype),  # meta overwrites `dtype` argument
    )

    return a


def _map_read_frame(block_info=None, **kwargs):

    i, j = block_info[None]['array-location'][0]

    return _utils._read_frame(i=slice(i, j), **kwargs)
