import dask.array as da
import numpy as np
from numpy.testing import assert_equal
import pytest

from dask_image.ndfilters import threshold_local

@pytest.fixture
def simple_test_image():
    image = da.from_array(np.array(
        [[0, 0, 1, 3, 5],
         [0, 1, 4, 3, 4],
         [1, 2, 5, 4, 1],
         [2, 4, 5, 2, 1],
         [4, 5, 1, 0, 0]], dtype=int), chunks=(5, 5))
    return image


# ==================================================
# Test Threshold Filters
# ==================================================

@pytest.mark.parametrize('block_size', [
    3,
    [3, 3],
    np.array([3, 3]),
    da.from_array(np.array([3, 3]), chunks=1),
    da.from_array(np.array([3, 3]), chunks=2),
])
def test_threshold_local_gaussian(simple_test_image, block_size):
    ref = np.array(
        [[False, False, False, False,  True],
            [False, False,  True, False,  True],
            [False, False,  True,  True, False],
            [False,  True,  True, False, False],
            [True,  True, False, False, False]]
    )
    out = threshold_local(simple_test_image, block_size, method='gaussian')
    assert_equal(ref, (simple_test_image > out).compute())

    out = threshold_local(simple_test_image, block_size, method='gaussian',
                            param=1./3.)
    assert_equal(ref, (simple_test_image > out).compute())


@pytest.mark.parametrize('block_size', [
    3,
    [3, 3],
    np.array([3, 3]),
    da.from_array(np.array([3, 3]), chunks=1),
    da.from_array(np.array([3, 3]), chunks=2),
])
def test_threshold_local_mean(simple_test_image, block_size):
    ref = np.array(
        [[False, False, False, False,  True],
            [False, False,  True, False,  True],
            [False, False,  True,  True, False],
            [False,  True,  True, False, False],
            [True,  True, False, False, False]]
    )
    out = threshold_local(simple_test_image, block_size, method='mean')
    assert_equal(ref, (simple_test_image > out).compute())


@pytest.mark.parametrize('block_size', [
    3,
    [3, 3],
    np.array([3, 3]),
    da.from_array(np.array([3, 3]), chunks=1),
    da.from_array(np.array([3, 3]), chunks=2),
])
def test_threshold_local_median(simple_test_image, block_size):
    ref = np.array(
        [[False, False, False, False,  True],
            [False, False,  True, False, False],
            [False, False,  True, False, False],
            [False, False,  True,  True, False],
            [False,  True, False, False, False]]
    )
    out = threshold_local(simple_test_image, block_size, method='median')
    assert_equal(ref, (simple_test_image > out).compute())


# ==================================================
# Test Generic Filters
# ==================================================

def test_threshold_local_generic(simple_test_image):
    ref = np.array(
        [[1.,  7., 16., 29., 37.],
         [5., 14., 23., 30., 30.],
         [13., 24., 30., 29., 21.],
         [25., 29., 28., 19., 10.],
         [34., 31., 23., 10.,  4.]]
    )
    unchanged = threshold_local(simple_test_image, 1, method='generic', param=sum)
    out = threshold_local(simple_test_image, 3, method='generic', param=sum)
    assert np.allclose(unchanged.compute(), simple_test_image.compute())
    assert np.allclose(out.compute(), ref)

def test_threshold_local_generic_invalid(simple_test_image):
    expected_error_message = "Must include a valid function to use as "
    "the 'param' keyword argument."
    with pytest.raises(ValueError) as e:
        threshold_local(simple_test_image, 3, method='generic', param='sum')
        assert e == expected_error_message


# ==================================================
# Test Invalid Arguments
# ==================================================

@pytest.mark.parametrize("method, block_size, error_type", [
    ('median', np.nan, TypeError),
])
def test_nan_blocksize(simple_test_image, method, block_size, error_type):
    with pytest.raises(error_type):
        threshold_local(simple_test_image, block_size, method=method)

def test_invalid_threshold_method(simple_test_image):
    with pytest.raises(ValueError):
        threshold_local(simple_test_image, 3, method='invalid')
