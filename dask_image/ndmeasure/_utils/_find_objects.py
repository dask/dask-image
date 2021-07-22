import numpy as np
import pandas as pd
from dask.delayed import delayed
import dask.dataframe as dd


def _array_chunk_location(block_id, chunks):
    """Pixel coordinate of top left corner of the array chunk."""
    array_location = []
    for idx, chunk in zip(block_id, chunks):
        array_location.append(sum(chunk[:idx]))
    return tuple(array_location)


@delayed
def _find_bounding_boxes(x, array_location):
    """An alternative to scipy.ndi.find_objects"""
    unique_vals = np.unique(x)
    unique_vals = unique_vals[unique_vals != 0]
    result = {}
    for val in unique_vals:
        positions = np.where(x == val)
        slices = tuple(slice(np.min(pos) + array_location[i], np.max(pos) + 1 + array_location[i], 1) for i, pos in enumerate(positions))
        result[val] = slices
    return pd.DataFrame.from_dict(result, orient='index')


def isnan(value):
    try:
        if np.isnan(value):
            return True
    except Exception:
        if value is np.nan:
            return True
    else:
        return False


def _combine_series(a, b):
    if isnan(a):
        return b
    elif isnan(b):
        return a
    else:
        start = min(a.start, b.start)
        stop = max(a.stop, b.stop)
        return slice(start, stop, 1)


def _combine_dataframes(s1, s2):
    combined = s1.combine(s2, _combine_series)
    return combined


def _merge_bounding_boxes(iterable):
    iterable = list(iterable)
    if len(iterable) == 1:
        df1 = iterable[0]
        return df1
    else:
        df1, df2 = iterable
        result = df1.combine(df2, _combine_dataframes)
        return result
