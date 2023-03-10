#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numbers
import pathlib

import pytest

import numpy as np
import tifffile

import dask.array as da
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
    "nframes, shape, runtime_warning",
    [
        (1, (1, 4, 3), None),
        (-1, (1, 4, 3), None),
        (3, (1, 4, 3), "`nframes` larger than"),
        (1, (5, 4, 3), None),
        (2, (5, 4, 3), "`nframes` does not nicely divide"),
        (1, (10, 5, 4, 3), None),
        (5, (10, 5, 4, 3), None),
        (10, (10, 5, 4, 3), None),
        (-1, (10, 5, 4, 3), None),
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
@pytest.mark.parametrize(
    "is_pathlib_Path",
    [
        True,
        False,
    ]
)
def test_tiff_imread(tmpdir, seed, nframes, shape, runtime_warning, dtype, is_pathlib_Path):  # noqa: E501
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
            fh.save(a[i], contiguous=True)

    with pytest.warns(None if runtime_warning is None
                      else RuntimeWarning, match=runtime_warning):
        if is_pathlib_Path:
            fn = pathlib.Path(fn)
        d = dask_image.imread.imread(fn, nframes=nframes)

    if nframes == -1:
        nframes = shape[0]

    assert min(nframes, shape[0]) == max(d.chunks[0])

    if shape[0] % nframes == 0:
        assert nframes == d.chunks[0][-1]
    else:
        assert (shape[0] % nframes) == d.chunks[0][-1]

    da.utils.assert_eq(a, d)


def test_tiff_imread_glob_natural_sort(tmpdir):
    dirpth = tmpdir.mkdir("test_imread")
    tifffile.imwrite(dirpth.join("10.tif"), np.array([10]))
    tifffile.imwrite(dirpth.join("9.tif"), np.array([9]))
    actual = np.array(dask_image.imread.imread(dirpth.join("*.tif")))
    assert np.all(actual == np.array([[9], [10]]))
