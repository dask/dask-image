#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import

import numbers

import pytest

import numpy as np
import tifffile

import dask.array.utils as dau
import dask_image.imread


@pytest.mark.parametrize(
    "err_type, nframes",
    [
        (ValueError, 1.0),
        (ValueError, 0),
        (ValueError, -2),
    ]
)
def test_errs_imread(err_type, nframes):
    with pytest.raises(err_type):
        dask_image.imread.imread("test.tiff", nframes=nframes)


@pytest.mark.parametrize(
    "seed",
    [
        0,
        1,
    ]
)
@pytest.mark.parametrize(
    "nframes, shape",
    [
        (1, (1, 4, 3)),
        (-1, (1, 4, 3)),
        (3, (1, 4, 3)),
        (1, (5, 4, 3)),
        (2, (5, 4, 3)),
        (1, (10, 5, 4, 3)),
        (5, (10, 5, 4, 3)),
        (10, (10, 5, 4, 3)),
        (-1, (10, 5, 4, 3)),
    ]
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.int16,
        np.int32,
        np.float32,
    ]
)
def test_tiff_imread(tmpdir, seed, nframes, shape, dtype):
    np.random.seed(seed)

    dirpth = tmpdir.mkdir("test_imread")
    dtype = np.dtype(dtype).type

    low, high = 0.0, 1.0
    if isinstance(dtype, numbers.Integral):
        low, high = np.iinfo(dtype).min, np.iinfo(dtype).max

    a = np.random.uniform(low=low, high=high, size=shape).astype(dtype)

    fn = str(dirpth.join("test.tiff"))
    with tifffile.TiffWriter(fn) as fh:
        for i in range(len(a)):
            fh.save(a[i])

    d = dask_image.imread.imread(fn, nframes=nframes)

    if nframes == -1:
        nframes = shape[0]

    assert min(nframes, shape[0]) == max(d.chunks[0])

    if shape[0] % nframes == 0:
        assert nframes == d.chunks[0][-1]
    else:
        assert (shape[0] % nframes) == d.chunks[0][-1]

    dau.assert_eq(a, d)
