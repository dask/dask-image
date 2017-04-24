# -*- coding: utf-8 -*-


import numbers

import numpy


def _get_footprint(ndim, size=None, footprint=None):
    # Verify that we only got size or footprint.
    if size is None and footprint is None:
        raise RuntimeError("Must provide either size or footprint.")
    if size is not None and footprint is not None:
        raise RuntimeError("Provide either size or footprint, but not both.")

    # Get a footprint based on the size.
    if size is not None:
        if isinstance(size, numbers.Integral):
            size = ndim * (size,)
        footprint = numpy.ones(size, dtype=bool)

    # Validate the footprint.
    if footprint.ndim != ndim:
        raise RuntimeError(
            "The footprint must have the same number of dimensions as"
            " the array being filtered."
        )
    if footprint.size == 0:
        raise ValueError("The footprint must have only non-zero dimensions.")

    # Convert to Boolean.
    footprint = (footprint != 0)

    return footprint
