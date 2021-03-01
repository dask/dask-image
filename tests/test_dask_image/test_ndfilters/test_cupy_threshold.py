#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

from dask_image.ndfilters import threshold_local

cupy = pytest.importorskip("cupy", minversion="7.7.0")


@pytest.mark.parametrize('block_size', [
    3,
    [3, 3],
    np.array([3, 3]),
    da.from_array(np.array([3, 3]), chunks=1),
    da.from_array(np.array([3, 3]), chunks=2),
])
class TestSimpleImage:
    def setup(self):
        self.image = da.from_array(cupy.array(
            [[0, 0, 1, 3, 5],
             [0, 1, 4, 3, 4],
             [1, 2, 5, 4, 1],
             [2, 4, 5, 2, 1],
             [4, 5, 1, 0, 0]], dtype=int), chunks=(5, 5))

    def test_threshold_local_gaussian(self, block_size):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [True,  True, False, False, False]]
        )
        out = threshold_local(self.image, block_size, method='gaussian')
        cupy.testing.assert_array_equal(ref, (self.image > out).compute())

        out = threshold_local(self.image, block_size, method='gaussian',
                              param=1./3.)
        cupy.testing.assert_array_equal(ref, (self.image > out).compute())

    def test_threshold_local_mean(self, block_size):
        ref = cupy.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [True,  True, False, False, False]]
        )
        out = threshold_local(self.image, block_size, method='mean')
        cupy.testing.assert_array_equal(ref, (self.image > out).compute())

    def test_threshold_local_median(self, block_size):
        ref = cupy.array(
            [[False, False, False, False,  True],
             [False, False,  True, False, False],
             [False, False,  True, False, False],
             [False, False,  True,  True, False],
             [False,  True, False, False, False]]
        )
        out = threshold_local(self.image, block_size, method='median')
        cupy.testing.assert_array_equal(ref, (self.image > out).compute())


class TestGenericFilter:
    def setup(self):
        self.image = da.from_array(cupy.array(
            [[0, 0, 1, 3, 5],
             [0, 1, 4, 3, 4],
             [1, 2, 5, 4, 1],
             [2, 4, 5, 2, 1],
             [4, 5, 1, 0, 0]], dtype=int), chunks=(5, 5))

    def test_threshold_local_generic(self):
        ref = cupy.array(
            [[1.,  7., 16., 29., 37.],
             [5., 14., 23., 30., 30.],
             [13., 24., 30., 29., 21.],
             [25., 29., 28., 19., 10.],
             [34., 31., 23., 10.,  4.]]
        )
        my_sum = cupy.ReductionKernel(
            'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')
        unchanged = threshold_local(self.image, 1, method='generic', param=my_sum)  # noqa: E501
        out = threshold_local(self.image, 3, method='generic', param=my_sum)
        assert cupy.allclose(unchanged.compute(), self.image.compute())
        assert cupy.allclose(out.compute(), ref)

    def test_threshold_local_generic_invalid(self):
        expected_error_message = "Must include a valid function to use as "
        "the 'param' keyword argument."
        with pytest.raises(ValueError) as e:
            threshold_local(self.image, 3, method='generic', param='sum')
            assert e == expected_error_message


class TestInvalidArguments:
    def setup(self):
        self.image = da.from_array(cupy.array(
            [[0, 0, 1, 3, 5],
             [0, 1, 4, 3, 4],
             [1, 2, 5, 4, 1],
             [2, 4, 5, 2, 1],
             [4, 5, 1, 0, 0]], dtype=int), chunks=(5, 5))

    @pytest.mark.parametrize("method, block_size, error_type", [
        ('gaussian', cupy.nan, ValueError),
        ('median', cupy.nan, TypeError),
    ])
    def test_nan_blocksize(self, method, block_size, error_type):
        with pytest.raises(error_type):
            threshold_local(self.image, block_size, method=method)

    def test_invalid_threshold_method(self):
        with pytest.raises(ValueError):
            threshold_local(self.image, 3, method='invalid')
