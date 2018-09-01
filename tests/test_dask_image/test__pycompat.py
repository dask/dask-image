#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import

import dask_image.imread._pycompat


def test_irange():
    r = dask_image.imread._pycompat.irange(5)

    assert not isinstance(r, list)

    assert list(r) == [0, 1, 2, 3, 4]


def test_izip():
    r = dask_image.imread._pycompat.izip([1, 2], [3, 4, 5])

    assert not isinstance(r, list)

    assert list(r) == [(1, 3), (2, 4)]
