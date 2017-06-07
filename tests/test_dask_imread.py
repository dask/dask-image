#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import

import pytest

import dask_imread


@pytest.mark.parametrize(
    "err_type, nframes",
    [
        (ValueError, 1.0),
        (ValueError, 0),
        (ValueError, -1),
    ]
)
def test_errs_imread(err_type, nframes):
    with pytest.raises(err_type):
        dask_imread.imread("test.tiff", nframes=nframes)
