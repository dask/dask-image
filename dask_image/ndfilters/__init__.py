# -*- coding: utf-8 -*-

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


from ._conv import (
    convolve,
    correlate,
)
convolve.__module__ = __name__
correlate.__module__ = __name__

from ._diff import (
    laplace,
)
laplace.__module__ = __name__


from ._edge import (
    prewitt,
    sobel,
)
prewitt.__module__ = __name__
sobel.__module__ = __name__


from ._gaussian import (
    gaussian_filter,
    gaussian_gradient_magnitude,
    gaussian_laplace,
)
gaussian_filter.__module__ = __name__
gaussian_gradient_magnitude.__module__ = __name__
gaussian_laplace.__module__ = __name__


from ._generic import (
    generic_filter,
)
generic_filter.__module__ = __name__


from ._order import (
    minimum_filter,
    median_filter,
    maximum_filter,
    rank_filter,
    percentile_filter,
)
minimum_filter.__module__ = __name__
median_filter.__module__ = __name__
maximum_filter.__module__ = __name__
rank_filter.__module__ = __name__
percentile_filter.__module__ = __name__


from ._smooth import (
    uniform_filter,
)
uniform_filter.__module__ = __name__
