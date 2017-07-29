#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import

import dask_ndmeasure._pycompat


def test_irange():
    r = dask_ndmeasure._pycompat.irange(5)

    assert not isinstance(r, list)

    assert list(r) == [0, 1, 2, 3, 4]
