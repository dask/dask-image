import numpy as np
import scipy.ndimage.filters
from dask.utils import Dispatch


class Dispatcher(Dispatch):
    """Simple single dispatch for different dask array types."""

    def __call__(self, arg, *args, **kwargs):
        """
        Call the corresponding method based on type of dask array.
        """
        meth = self.dispatch(type(arg._meta))
        return meth(arg, *args, **kwargs)


convolve_dispatch = Dispatcher(name="convolve_dispatch")


@convolve_dispatch.register(np.ndarray)
def numpy_convolve(*args, **kwargs):
    return scipy.ndimage.filters.convolve


@convolve_dispatch.register_lazy("cupy")
def register_cupy():
    import cupy
    import cupyx.scipy.ndimage

    @convolve_dispatch.register(cupy.ndarray)
    def cupy_convolve(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.convolve
