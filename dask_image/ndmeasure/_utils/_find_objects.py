import numpy as np
import pandas as pd
from dask.delayed import Delayed
import dask.dataframe as dd


def _array_chunk_location(block_id, chunks):
    """Pixel coordinate of top left corner of the array chunk."""
    array_location = []
    for idx, chunk in zip(block_id, chunks):
        array_location.append(sum(chunk[:idx]))
    return tuple(array_location)


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


def _combine_slices(slices):
    "Return the union of all slices."
    if len(slices) == 1:
        return slices[0]
    else:
        start = min([sl.start for sl in slices])
        stop = max([sl.stop for sl in slices])
        return slice(start, stop, 1)


def _merge_bounding_boxes(x, ndim):
    x = x.dropna()
    data = {}
    for i in range(ndim):
        slices = [x[ii] for ii in x.index if str(ii).startswith(str(i))]
        combined_slices = _combine_slices(slices)
        data[i] = combined_slices
    result = pd.Series(data=data, index=[i for i in range(ndim)], name=x.name)
    return result


def _find_objects(df1, df2, ndim=2):
    meta = dd.utils.make_meta([(i, object) for i in range(ndim)])
    if isinstance(df1, Delayed):
        df1 = dd.from_delayed(df1, meta=meta)
    if isinstance(df2, Delayed):
        df2 = dd.from_delayed(df2, meta=meta)
    ddf = dd.merge(df1, df2, how="outer", left_index=True, right_index=True)
    result = ddf.apply(_merge_bounding_boxes, ndim=ndim, axis=1, meta=meta)
    return result
