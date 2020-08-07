import numpy as np
import scipy.ndimage.filters
from dask.utils import Dispatch


def check_arraytypes_compatible(*args):
    """"""
    arraytypes = [get_type(arg) for arg in args]
    if len(set(arraytypes)) != 1:
        raise ValueError("Array types must be compatible.")



def get_type(arg):
    """"""
    try:
        datatype = type(arg._meta)  # Check chunk type backing dask array
    except AttributeError:
        datatype = type(arg)  # For all non-dask arrays
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


dispatch_convolve = Dispatcher(name="dispatch_convolve")


@dispatch_convolve.register(np.ndarray)
def numpy_convolve(*args, **kwargs):
    return scipy.ndimage.filters.convolve(*args, **kwargs)


@dispatch_convolve.register_lazy("cupy")
def register_cupy():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_convolve.register(cupy.ndarray)
    def cupy_convolve(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.convolve(*args, **kwargs)
