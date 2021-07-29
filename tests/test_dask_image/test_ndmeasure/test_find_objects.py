import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import dask_image.ndmeasure


@pytest.fixture
def label_image():
    """Return small label image for tests.

    dask.array<array, shape=(5, 10), dtype=int64, chunksize=(5, 5), chunktype=numpy.ndarray>

    array([[  0,   0,   0,   0,   0,   0,   0, 333, 333, 333],
            [111, 111,   0,   0,   0,   0,   0, 333, 333, 333],
            [111, 111,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0, 222, 222, 222, 222, 222, 222,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]])

    """
    label_image = np.zeros((5, 10)).astype(int)
    label_image[1:3,0:2] = 111
    label_image[3,3:-2] = 222
    label_image[0:2,-3:] = 333
    label_image = da.from_array(label_image, chunks=(5, 5))
    return label_image


def test_bounding_boxes(label_image):
    result = dask_image.ndmeasure.find_objects(label_image)
    assert isinstance(result, dd.DataFrame)
    computed_result = result.compute()
    assert isinstance(computed_result, pd.DataFrame)
    expected = pd.DataFrame.from_dict(
        {0: {111: slice(1, 3), 222: slice(3, 4), 333: slice(0, 2)},
        1: {111: slice(0, 2), 222: slice(3, 8), 333: slice(7, 10)}}
    )
    assert computed_result.equals(expected)
