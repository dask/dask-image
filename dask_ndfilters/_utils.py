# -*- coding: utf-8 -*-


import collections
import inspect
import numbers
import re


try:
    from itertools import imap, izip
except ImportError:
    imap, izip = map, zip

try:
    irange = xrange
except NameError:
    irange = range

try:
    unicode
except NameError:
    unicode = str

strlike = (bytes, unicode)


def _get_docstring(func):
    # Drop the output parameter from the docstring.
    split_doc_params = lambda s: \
        re.subn("(    [A-Za-z]+ : )", "\0\\1", s)[0].split("\0")
    drop_doc_param = lambda s: not s.startswith("    output : ")
    func_doc = "" if func.__doc__ is None else func.__doc__
    cleaned_docstring = "".join([
        l for l in split_doc_params(func_doc) if drop_doc_param(l)
    ])

    docstring = """
    Wrapped copy of "{mod_name}.{func_name}"


    Excludes the output parameter as it would not work Dask arrays.


    Original docstring:

    {doc}
    """.format(
        mod_name=inspect.getmodule(func).__name__,
        func_name=func.__name__,
        doc=cleaned_docstring,
    )

    return docstring


def _update_wrapper(func):
    def _updater(wrapper):
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = _get_docstring(func)
        return wrapper

    return _updater


def _get_depth_boundary(ndim, depth, boundary=None):
    if not isinstance(ndim, numbers.Integral):
        raise TypeError("Expected integer value for `ndim`.")
    if ndim <= 0:
        raise ValueError("Expected positive value for `ndim`.")

    if isinstance(depth, numbers.Number):
        depth = ndim * (depth,)
    if not isinstance(depth, collections.Sized):
        raise TypeError("Unexpected type for `depth`.")
    if len(depth) != ndim:
        raise ValueError("Expected `depth` to have a length equal to `ndim`.")
    if isinstance(depth, collections.Sequence):
        depth = dict(izip(irange(ndim), depth))
    if not isinstance(depth, collections.Mapping):
        raise TypeError("Unexpected type for `depth`.")

    if not all(map(lambda d: isinstance(d, numbers.Integral), depth.values())):
        raise TypeError("Expected integer values for `depth`.")
    if not all(map(lambda d: d >= 0, depth.values())):
        raise ValueError("Expected positive semidefinite values for `depth`.")

    if (boundary is None) or isinstance(boundary, strlike):
        boundary = ndim * (boundary,)
    if not isinstance(boundary, collections.Sized):
        raise TypeError("Unexpected type for `boundary`.")
    if len(boundary) != ndim:
        raise ValueError(
            "Expected `boundary` to have a length equal to `ndim`."
        )
    if isinstance(boundary, collections.Sequence):
        boundary = dict(izip(irange(ndim), boundary))
    if not isinstance(boundary, collections.Mapping):
        raise TypeError("Unexpected type for `boundary`.")

    type_check = lambda b: (b is None) or isinstance(b, strlike)
    if not all(map(type_check, boundary.values())):
        raise TypeError("Expected string-like values for `boundary`.")

    # Workaround for a bug in Dask with 0 depth.
    #
    # ref: https://github.com/dask/dask/issues/2258
    #
    for i in irange(ndim):
        if boundary[i] == "none" and depth[i] == 0:
            boundary[i] = "reflect"

    return depth, boundary
