# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


from dask_ndfilters._conv import (
    convolve,
    correlate,
)

from dask_ndfilters._diff import (
    laplace,
)

from dask_ndfilters._edge import (
    prewitt,
    sobel,
)

from dask_ndfilters._gaussian import (
    gaussian_filter,
    gaussian_gradient_magnitude,
    gaussian_laplace,
)

from dask_ndfilters._generic import (
    generic_filter,
)

from dask_ndfilters._order import (
    minimum_filter,
    median_filter,
    maximum_filter,
    rank_filter,
    percentile_filter,
)

from dask_ndfilters._smooth import (
    uniform_filter,
)
