import dask.array as da
import numpy as np
from numpy.testing import assert_equal

from dask_image.ndfilters._thresholding import threshold_local


class TestSimpleImage():
    def setup(self):
        self.image = da.from_array(np.array(
            [[0, 0, 1, 3, 5],
             [0, 1, 4, 3, 4],
             [1, 2, 5, 4, 1],
             [2, 4, 5, 2, 1],
             [4, 5, 1, 0, 0]], dtype=int), chunks=(5, 5))

    def test_threshold_local_gaussian(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [True,  True, False, False, False]]
        )
        out = threshold_local(self.image, 3, method='gaussian')
        assert_equal(ref, (self.image > out).compute())

        out = threshold_local(self.image, 3, method='gaussian',
                              param=1./3.)
        assert_equal(ref, (self.image > out).compute())

    def test_threshold_local_mean(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [True,  True, False, False, False]]
        )
        out = threshold_local(self.image, 3, method='mean')
        assert_equal(ref, (self.image > out).compute())

    def test_threshold_local_median(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False, False],
             [False, False,  True, False, False],
             [False, False,  True,  True, False],
             [False,  True, False, False, False]]
        )
        out = threshold_local(self.image, 3, method='median')
        assert_equal(ref, (self.image > out).compute())
