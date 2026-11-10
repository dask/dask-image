"""
Test that ``find_objects`` and ``label`` raise a helpful ``ImportError`` when the
optional ``dask[dataframe]`` / ``pandas`` dependencies are not installed.

This is skipped if both dependencies are installed.

"""

import dask.array as da
import pytest

import dask_image.ndmeasure

try:
    import pandas  # noqa: F401

    pandas_available = True
except ImportError:
    pandas_available = False

if pandas_available is True:
    try:
        import dask.dataframe  # noqa: F401

        dataframe_available = True
    except ImportError:
        dataframe_available = False
elif pandas_available is False:
    dataframe_available = False


@pytest.mark.skipif(
    dataframe_available,
    reason="dataframe dependencies are installed; "
    "ImportError path only triggers without them",
)
def test_find_objects_raises_import_error_without_pandas():
    label_image = da.zeros((3, 3), dtype=int, chunks=(3, 3))
    with pytest.raises(
        ImportError,
        match=(
            r"dask_image\.ndmeasure\.find_objects requires the optional "
            r"dependencies `dask\[dataframe\]` and `pandas`\. "
            r"Install them with `pip install dask-image\[dataframe\]`\."
        ),
    ):
        dask_image.ndmeasure.find_objects(label_image)


@pytest.mark.skipif(
    pandas_available,
    reason="dataframe dependencies are installed; "
    "ImportError path only triggers without them",
)
def test_label_objects_raises_import_error_without_pandas():
    label_image = da.zeros((3, 3), dtype=int, chunks=(3, 3))
    with pytest.raises(
        ImportError,
        match=(
            r"dask_image\.ndmeasure\.label requires the optional "
            r"dependency `pandas`\."
            r"Install with `pip install dask-image\[dataframe\]`\."
        ),
    ):
        dask_image.ndmeasure.label(label_image)
