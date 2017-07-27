#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest


def test_import():
    try:
        import dask_ndmeasure
    except ImportError:
        pytest.fail("Unable to import `dask_ndmeasure`.")
