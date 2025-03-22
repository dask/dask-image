import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.delayed import Delayed
import dask.config as dask_config


def _array_chunk_location(block_id, chunks):
    """Pixel coordinate of top left corner of the array chunk."""
    array_location = []
    for idx, chunk in zip(block_id, chunks):
        array_location.append(sum(chunk[:idx]))
    return tuple(array_location)


def _find_bounding_boxes(x, array_location):
    """An alternative to scipy.ndimage.find_objects.

    We use this alternative because scipy.ndimage.find_objects
    returns a tuple of length N, where N is the largest integer label.
    This is not ideal for distributed labels, where there might be only
    one or two objects in an image chunk labelled with very large integers.

    This alternative function returns a pandas dataframe,
    with one row per object found in the image chunk.
    """
    unique_vals = np.unique(x)
    unique_vals = unique_vals[unique_vals != 0]
    result = {}
    for val in unique_vals:
        positions = np.where(x == val)
        slices = tuple(slice(np.min(pos) + array_location[i], np.max(pos) + 1 + array_location[i]) for i, pos in
                       enumerate(positions))
        result[val] = slices
    column_names = [i for i in range(x.ndim)]  # column names are: 0, 1, ... nD
    return pd.DataFrame.from_dict(result, orient='index', columns=column_names)


def _combine_slices(slices):
    "Return the union of all slices."
    if len(slices) == 1:
        return slices[0]
    else:
        start = min([sl.start for sl in slices])
        stop = max([sl.stop for sl in slices])
        return slice(start, stop)


def _merge_bounding_boxes(x, ndim):
    """Merge the bounding boxes describing objects over multiple image chunks."""
    x = x.dropna()
    data = {}
    # For each dimension in the array,
    # pick out the slice values belonging to that dimension
    # and combine the slices
    # (i.e. find the union; the slice expanded to all input slices).
    for i in range(ndim):
        # Array dimensions are labelled by a number followed by an underscroe
        # i.e. column labels are: 0_x, 1_x, 2_x, ... 0_y, 1_y, 2_y, ...
        # (x and y represent the pair of chunks label slices are merged from)
        slices = [x[ii] for ii in x.index if str(ii).startswith(str(i))]
        combined_slices = _combine_slices(slices)
        data[i] = combined_slices
    result = pd.Series(data=data, index=[i for i in range(ndim)], name=x.name)
    return result


def _find_objects(ndim, df1, df2):
    """Main utility function for find_objects."""
    meta = dd.utils.make_meta([(i, object) for i in range(ndim)])
    if isinstance(df1, Delayed):
        with dask_config.set({'dataframe.convert-string': False}):
            df1 = dd.from_delayed(df1, meta=meta)
    if isinstance(df2, Delayed):
        with dask_config.set({'dataframe.convert-string': False}):
            df2 = dd.from_delayed(df2, meta=meta)

    if len(df1) > 0 and len(df2) > 0:
        ddf = dd.merge(df1, df2, how="outer", left_index=True, right_index=True)
    elif len(df1) > 0:
        ddf = df1
    elif len(df2) > 0:
        ddf = df2
    else:
        ddf = pd.DataFrame()

    result = ddf.apply(_merge_bounding_boxes, ndim=ndim, axis=1, meta=meta)
    return result
