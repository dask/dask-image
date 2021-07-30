# -*- coding: utf-8 -*-


import dask.array as da

from . import _utils


def _binary_op(func,
               image,
               structure=None,
               iterations=1,
               mask=None,
               origin=0,
               brute_force=False,
               **kwargs):
    image = (image != 0)

    structure = _utils._get_structure(image, structure)
    iterations = _utils._get_iterations(iterations)
    mask = _utils._get_mask(image, mask)
    origin = _utils._get_origin(structure.shape, origin)
    brute_force = _utils._get_brute_force(brute_force)
    depth = _utils._get_depth(structure.shape, origin)
    depth, boundary = _utils._get_depth_boundary(structure.ndim, depth, "none")

    result = image
    for i in range(iterations):
        iter_result = result.map_overlap(
            func,
            depth=depth,
            boundary=boundary,
            dtype=bool,
            meta=image._meta,
            structure=structure,
            origin=origin,
            **kwargs
        )
        result = da.where(mask, iter_result, result)
        result._meta = image._meta.astype(bool)

    return result
