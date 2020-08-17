# -*- coding: utf-8 -*-

from dask.utils import Dispatch


def get_type(array):
    """Return type of arrays contained within the dask array chunks."""
    try:
        datatype = type(array._meta)  # Check chunk type backing dask array
    except AttributeError:
        datatype = type(array)  # For all non-dask arrays
    return datatype


class Dispatcher(Dispatch):
    """Simple single dispatch for different dask array types."""

    def __call__(self, arg, *args, **kwargs):
        """
        Call the corresponding method based on type of dask array.
        """
        datatype = get_type(arg)
        meth = self.dispatch(datatype)
        return meth(arg, *args, **kwargs)
