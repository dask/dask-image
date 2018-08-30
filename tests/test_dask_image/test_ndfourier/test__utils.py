# -*- coding: utf-8 -*-


import pytest

import dask.array as da

import dask_image.ndfourier._utils


@pytest.mark.parametrize(
    "a, s, n, axis", [
        (da.ones((3, 4), chunks=(3, 4)), da.ones((2,), chunks=(2,)), -1, -1),
    ]
)
def test_norm_args(a, s, n, axis):
    a2, s2, n2, axis2 = dask_image.ndfourier._utils._norm_args(
        a, s, n=n, axis=axis
    )

    assert isinstance(a2, da.Array)
    assert isinstance(s2, da.Array)
