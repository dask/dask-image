#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import scipy.ndimage as spnd

import dask.array as da
import dask_image.ndmorph as da_ndm


@pytest.mark.parametrize(
    "funcname",
    [
        "binary_closing",
        "binary_dilation",
        "binary_erosion",
        "binary_opening",
    ]
)
@pytest.mark.parametrize(
    "err_type, input, structure, origin",
    [
        (
            RuntimeError,
            da.ones([1, 2], dtype=bool, chunks=(1, 2,)),
            da.arange(2, dtype=bool, chunks=(2,)),
            0
        ),
        (
            TypeError,
            da.arange(2, dtype=bool, chunks=(2,)),
            2.0,
            0
        ),
        (
            TypeError,
            da.ones([2], dtype=bool, chunks=(2,)),
            da.arange(2, dtype=bool, chunks=(2,)),
            0.0
        ),
    ]
)
def test_errs_binary_ops(funcname,
                         err_type,
                         input,
                         structure,
                         origin):
    da_func = getattr(da_ndm, funcname)

    with pytest.raises(err_type):
        da_func(
            input,
            structure=structure,
            origin=origin
        )


@pytest.mark.parametrize(
    "funcname",
    [
        "binary_closing",
        "binary_dilation",
        "binary_erosion",
        "binary_opening",
    ]
)
@pytest.mark.parametrize(
    "err_type, input, structure, iterations, origin",
    [
        (
            TypeError,
            da.ones([2], dtype=bool, chunks=(2,)),
            da.arange(2, dtype=bool, chunks=(2,)),
            1.0,
            0
        ),
        (
            NotImplementedError,
            da.ones([2], dtype=bool, chunks=(2,)),
            da.arange(2, dtype=bool, chunks=(2,)),
            0,
            0
        )
    ]
)
def test_errs_binary_ops_iter(funcname,
                              err_type,
                              input,
                              structure,
                              iterations,
                              origin):
    da_func = getattr(da_ndm, funcname)

    with pytest.raises(err_type):
        da_func(
            input,
            structure=structure,
            iterations=iterations,
            origin=origin
        )


@pytest.mark.parametrize(
    "funcname",
    [
        "binary_closing",
        "binary_dilation",
        "binary_erosion",
        "binary_opening",
    ]
)
@pytest.mark.parametrize(
    "err_type, input, structure, iterations, mask, border_value, origin"
    ", brute_force",
    [
        (
            RuntimeError,
            da.ones([2], dtype=bool, chunks=(2,)),
            da.arange(2, dtype=bool, chunks=(2,)),
            1,
            da.arange(2, dtype=bool, chunks=(2,))[None],
            0,
            0,
            False
        ),
        (
            TypeError,
            da.ones([2], dtype=bool, chunks=(2,)),
            da.arange(2, dtype=bool, chunks=(2,)),
            1,
            da.arange(2, dtype=bool, chunks=(2,)),
            2.0,
            0,
            False
        ),
        (
            NotImplementedError,
            da.ones([2], dtype=bool, chunks=(2,)),
            da.arange(2, dtype=bool, chunks=(2,)),
            1,
            da.arange(2, dtype=bool, chunks=(2,)),
            0,
            0,
            True
        ),
    ]
)
def test_errs_binary_ops_expanded(funcname,
                                  err_type,
                                  input,
                                  structure,
                                  iterations,
                                  mask,
                                  border_value,
                                  origin,
                                  brute_force):
    da_func = getattr(da_ndm, funcname)

    with pytest.raises(err_type):
        da_func(
            input,
            structure=structure,
            iterations=iterations,
            mask=mask,
            border_value=border_value,
            origin=origin,
            brute_force=brute_force
        )


@pytest.mark.parametrize(
    "funcname",
    [
        "binary_closing",
        "binary_dilation",
        "binary_erosion",
        "binary_opening",
    ]
)
@pytest.mark.parametrize(
    "input, structure, origin",
    [
        (
            da.from_array(
                np.array(
                    [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            None,
            0
        ),
        (
            da.from_array(
                np.array(
                    [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            np.ones([3, 3], dtype=bool),
            0
        ),
        (
            da.from_array(
                np.array(
                    [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            np.ones([3, 3], dtype=bool),
            1
        ),
        (
            da.from_array(
                np.array(
                    [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            np.ones([3, 3], dtype=bool),
            -1
        ),
    ]
)
def test_binary_ops(funcname,
                    input,
                    structure,
                    origin):
    da_func = getattr(da_ndm, funcname)
    sp_func = getattr(spnd, funcname)

    da_result = da_func(
        input,
        structure=structure,
        origin=origin
    )

    sp_result = sp_func(
        input,
        structure=structure,
        origin=origin
    )

    da.utils.assert_eq(sp_result, da_result)


@pytest.mark.parametrize(
    "funcname",
    [
        "binary_closing",
        "binary_dilation",
        "binary_erosion",
        "binary_opening",
    ]
)
@pytest.mark.parametrize(
    "input, structure, iterations, origin",
    [
        (
            da.from_array(
                np.array(
                    [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            np.ones([3, 3], dtype=bool),
            3,
            0
        ),
        (
            da.from_array(
                np.array(
                    [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            np.ones([3, 3], dtype=bool),
            3,
            1
        ),
    ]
)
def test_binary_ops_iter(funcname,
                         input,
                         structure,
                         iterations,
                         origin):
    da_func = getattr(da_ndm, funcname)
    sp_func = getattr(spnd, funcname)

    da_result = da_func(
        input,
        structure=structure,
        iterations=iterations,
        origin=origin
    )

    sp_result = sp_func(
        input,
        structure=structure,
        iterations=iterations,
        origin=origin
    )

    da.utils.assert_eq(sp_result, da_result)


@pytest.mark.parametrize(
    "funcname",
    [
        "binary_closing",
        "binary_dilation",
        "binary_erosion",
        "binary_opening",
    ]
)
@pytest.mark.parametrize(
    "input, structure, iterations, mask, border_value, origin, brute_force",
    [
        (
            da.from_array(
                np.array(
                    [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            np.ones([3, 3], dtype=bool),
            1,
            None,
            1,
            0,
            False
        ),
        (
            da.from_array(
                np.array(
                    [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            np.ones([3, 3], dtype=bool),
            1,
            da.from_array(
                np.array(
                    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            0,
            0,
            False
        ),
        (
            da.from_array(
                np.array(
                    [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            np.ones([3, 3], dtype=bool),
            3,
            da.from_array(
                np.array(
                    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    dtype=bool
                ),
                chunks=(5, 6)
            ),
            0,
            0,
            False
        ),
    ]
)
def test_binary_ops_expanded(funcname,
                             input,
                             structure,
                             iterations,
                             mask,
                             border_value,
                             origin,
                             brute_force):
    da_func = getattr(da_ndm, funcname)
    sp_func = getattr(spnd, funcname)

    da_result = da_func(
        input,
        structure=structure,
        iterations=iterations,
        mask=mask,
        border_value=border_value,
        origin=origin,
        brute_force=brute_force
    )

    sp_result = sp_func(
        input,
        structure=structure,
        iterations=iterations,
        mask=mask,
        border_value=border_value,
        origin=origin,
        brute_force=brute_force
    )

    da.utils.assert_eq(sp_result, da_result)
