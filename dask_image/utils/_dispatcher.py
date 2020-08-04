from dask.utils import Dispatch


class Dispatcher(Dispatch):
    """Simple single dispatch for different dask array types."""

    def __call__(self, arg, *args, **kwargs):
        """
        Call the corresponding method based on type of dask array.
        """
        meth = self.dispatch(type(arg._meta))
        return meth(arg, *args, **kwargs)
