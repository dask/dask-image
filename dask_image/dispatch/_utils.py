# -*- coding: utf-8 -*-

from ._dispatcher import get_type


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
