# -*- coding: utf-8 -*-

import numpy as np

from ._dispatcher import Dispatcher, get_type

__all__ = [
    "dispatch_array",
    "check_arraytypes_compatible",
]

dispatch_array = Dispatcher(name="dispatch_array")


@dispatch_array.register(np.ndarray)
def numpy_array(*args, **kwargs):
    return np.array


@dispatch_array.register_lazy("cupy")
def register_cupy_array():
    import cupy

    @dispatch_array.register(cupy.ndarray)
    def cupy_array(*args, **kwargs):
        return cupy.array


def check_arraytypes_compatible(*args):
    """Check array types are compatible.

    For arrays to be compatible they must either have the same type,
    or a dask array where the chunks match the same array type.

    Examples of compatible arrays:
    * Two (or more) numpy arrays
    * A dask array with numpy chunks, and a numpy array

    Examples of incompatible arrays:
    * A numpy array and a cupy array
    """
    arraytypes = [get_type(arg) for arg in args]
    if len(set(arraytypes)) != 1:
        raise ValueError("Array types must be compatible.")
