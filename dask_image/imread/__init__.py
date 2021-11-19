# -*- coding: utf-8 -*-
import glob
import numbers
import warnings

import dask.array as da
import dask.array.image
import numpy as np
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

    if arraytype == "cupy":   # pragma: no cover
        import cupy
        preprocess = cupy.asanyarray
    else:
        preprocess = None

    result = dask.array.image.imread(fname, preprocess=preprocess)

    if nframes != 1:
        chunks = [nframes] + ['auto' for _ in range(result.ndim - 1)]
        result = result.rechunk(chunks=chunks)

    return result
