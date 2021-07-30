# -*- coding: utf-8 -*-

__all__ = [
    "convolve",
    "correlate",
    "laplace",
    "prewitt",
    "sobel",
    "gaussian_filter",
    "gaussian_gradient_magnitude",
    "gaussian_laplace",
    "generic_filter",
    "minimum_filter",
    "median_filter",
    "maximum_filter",
    "rank_filter",
    "percentile_filter",
    "uniform_filter",
    "threshold_local",
]

from ._conv import convolve, correlate
from ._diff import laplace
from ._edge import prewitt, sobel
from ._gaussian import (gaussian_filter, gaussian_gradient_magnitude,
                        gaussian_laplace)
from ._generic import generic_filter
from ._order import (maximum_filter, median_filter, minimum_filter,
                     percentile_filter, rank_filter)
from ._smooth import uniform_filter
from ._threshold import threshold_local

convolve.__module__ = __name__
correlate.__module__ = __name__

laplace.__module__ = __name__

prewitt.__module__ = __name__
sobel.__module__ = __name__

gaussian_filter.__module__ = __name__
gaussian_gradient_magnitude.__module__ = __name__
gaussian_laplace.__module__ = __name__

generic_filter.__module__ = __name__

minimum_filter.__module__ = __name__
median_filter.__module__ = __name__
maximum_filter.__module__ = __name__
rank_filter.__module__ = __name__
percentile_filter.__module__ = __name__

uniform_filter.__module__ = __name__

threshold_local.__module__ = __name__
