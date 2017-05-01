# -*- coding: utf-8 -*-


import inspect
import re


def _get_docstring(func):
    # Drop the output parameter from the docstring.
    split_doc_params = lambda s: \
        re.subn("(    [A-Za-z]+ : )", "\0\\1", s)[0].split("\0")
    drop_doc_param = lambda s: not s.startswith("    output : ")
    cleaned_docstring = "".join([
        l for l in split_doc_params(func.__doc__) if drop_doc_param(l)
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
