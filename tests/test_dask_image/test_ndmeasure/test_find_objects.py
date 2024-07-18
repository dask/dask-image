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

    array([[   0,   0,   0,   0,   0,   0,   0, 333, 333, 333],
            [111, 111,   0,   0,   0,   0,   0, 333, 333, 333],
            [111, 111,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0, 222, 222, 222, 222, 222, 222,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]])

    """
    label_image = np.zeros((5, 10)).astype(int)
    label_image[1:3, 0:2] = 111
    label_image[3, 3:-2] = 222
    label_image[0:2, -3:] = 333
    label_image = da.from_array(label_image, chunks=(5, 5))
    return label_image


@pytest.fixture
def label_image_with_empty_chunk():
    """Return small label image with an empty chunk for tests.

    dask.array<array, shape=(6, 6), dtype=int64, chunksize=(3, 3), chunktype=numpy.ndarray>

    array([[   0,   0,   0,   0,   0,   0],
            [111, 111,   0,   0,   0,   0],
            [111, 111,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0],
            [  0,   0,   0, 222, 222, 222],
            [  0,   0,   0,   0,   0,   0]])
    """
    label_image = np.zeros((6, 6)).astype(int)
    label_image[1:3, 0:2] = 111
    label_image[4, 3:] = 222
    label_image = da.from_array(label_image, chunks=(3, 3))
    return label_image


def test_find_objects_err(label_image):
    label_image = label_image.astype(float)
    with pytest.raises(ValueError):
        dask_image.ndmeasure.find_objects(label_image)


def test_empty_chunk():
    test_labels = da.zeros((10, 10), dtype='int', chunks=(3, 3))
    test_labels[0, 0] = 1
    computed_result = dask_image.ndmeasure.find_objects(test_labels).compute()
    expected = pd.DataFrame.from_dict({0: {1: slice(0, 1)},
                                       1: {1: slice(0, 1)}, })
    assert computed_result.equals(expected)


def test_find_objects(label_image):
    result = dask_image.ndmeasure.find_objects(label_image)
    assert isinstance(result, dd.DataFrame)
    computed_result = result.compute()
    assert isinstance(computed_result, pd.DataFrame)
    expected = pd.DataFrame.from_dict(
        {0: {111: slice(1, 3), 222: slice(3, 4), 333: slice(0, 2)},
         1: {111: slice(0, 2), 222: slice(3, 8), 333: slice(7, 10)}}
    )
    assert computed_result.equals(expected)


def test_3d_find_objects(label_image):
    label_image = da.stack([label_image, label_image], axis=2)
    result = dask_image.ndmeasure.find_objects(label_image)
    assert isinstance(result, dd.DataFrame)
    computed_result = result.compute()
    assert isinstance(computed_result, pd.DataFrame)
    expected = pd.DataFrame.from_dict(
        {0: {111: slice(1, 3), 222: slice(3, 4), 333: slice(0, 2)},
         1: {111: slice(0, 2), 222: slice(3, 8), 333: slice(7, 10)},
         2: {111: slice(0, 2), 222: slice(0, 2), 333: slice(0, 2)}}
    )
    assert computed_result.equals(expected)


def test_find_objects_with_empty_chunks(label_image_with_empty_chunk):
    result = dask_image.ndmeasure.find_objects(label_image_with_empty_chunk)
    assert isinstance(result, dd.DataFrame)
    computed_result = result.compute()
    assert isinstance(computed_result, pd.DataFrame)
    expected = pd.DataFrame.from_dict(
        {0: {111: slice(1, 3, None), 222: slice(4, 5, None)},
         1: {111: slice(0, 2, None), 222: slice(3, 6, None)}}
    )
    assert computed_result.equals(expected)
