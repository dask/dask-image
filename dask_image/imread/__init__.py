# -*- coding: utf-8 -*-
import glob
import numbers
import warnings

import dask.array as da
import numpy as np
import pims

from . import _utils


def imread(fname, nframes=1, *, arraytype="numpy", sortfunc=sorted):
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
    sortfunc: Callable, optional
        A function for sorting the glob results, default is "sorted".

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
        arrayfunc = np.asanyarray
    elif arraytype == "cupy":   # pragma: no cover
        import cupy
        arrayfunc = cupy.asanyarray

    with pims.open(sfname) as imgs:
        shape = (len(imgs),) + imgs.frame_shape
        dtype = np.dtype(imgs.pixel_type)

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

    # place source filenames into dask array
    filenames = sortfunc(glob.glob(sfname))  # pims also does this
    if len(filenames) > 1:
        ar = da.from_array(filenames, chunks=(nframes,))
        multiple_files = True
    else:
        ar = da.from_array(filenames * shape[0], chunks=(nframes,))
        multiple_files = False

    # read in data using encoded filenames
    a = ar.map_blocks(
        _map_read_frame,
        chunks=da.core.normalize_chunks(
            (nframes,) + shape[1:], shape),
        multiple_files=multiple_files,
        new_axis=list(range(1, len(shape))),
        arrayfunc=arrayfunc,
        meta=arrayfunc([]).astype(dtype),  # meta overwrites `dtype` argument
    )
    return a


def _map_read_frame(x, multiple_files, block_info=None, **kwargs):

    fn = x[0]  # get filename from input chunk

    if multiple_files:
        i, j = 0, 1
    else:
        i, j = block_info[None]['array-location'][0]

    return _utils._read_frame(fn=fn, i=slice(i, j), **kwargs)
