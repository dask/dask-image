import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from itertools import product

import numpy as np
import scipy.signal



import dask.array as da
from dask.array.utils import assert_eq
from dask_image.signal import convolve 



def test_convolve_invalid_shapes():
    a = np.arange(1, 7).reshape((2, 3))
    b = np.arange(-6, 0).reshape((3, 2))
    with pytest.raises(
        ValueError,
        match="For 'valid' mode, one must be at least "
        "as large as the other in every dimension",
    ):
        convolve(a, b, mode="valid")


@pytest.mark.parametrize("method", ["fft", "oa"])
def test_convolve_invalid_shapes_axes(method):
    a = np.zeros([5, 6, 2, 1])
    b = np.zeros([5, 6, 3, 1])
    with pytest.raises(
        ValueError,
        match=r"incompatible shapes for in1 and in2:"
        r" \(5L?, 6L?, 2L?, 1L?\) and"
        r" \(5L?, 6L?, 3L?, 1L?\)",
    ):
        convolve(a, b, method=method, axes=[0, 1])


@pytest.mark.parametrize("a,b", [([1], 2), (1, [2]), ([3], [[2]])])
def test_convolve_mismatched_dims(a, b):
    with pytest.raises(
        ValueError, match="in1 and in2 should have the same" " dimensionality"
    ):
        convolve(a, b)


def test_convolve_invalid_flags():
    with pytest.raises(
        ValueError,
        match="acceptable mode flags are 'valid'," " 'same', 'full' or 'periodic'",
    ):
        convolve([1], [2], mode="chips")

    with pytest.raises(ValueError, match="acceptable method flags are 'oa', or 'fft'"):
        convolve([1], [2], method="chips")

    with pytest.raises(ValueError, match="when provided, axes cannot be empty"):
        convolve([1], [2], axes=[])

    with pytest.raises(
        ValueError, match="axes must be a scalar or " "iterable of integers"
    ):
        convolve([1], [2], axes=[[1, 2], [3, 4]])

    with pytest.raises(
        ValueError, match="axes must be a scalar or " "iterable of integers"
    ):
        convolve([1], [2], axes=[1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="axes exceeds dimensionality of input"):
        convolve([1], [2], axes=[1])

    with pytest.raises(ValueError, match="axes exceeds dimensionality of input"):
        convolve([1], [2], axes=[-2])

    with pytest.raises(ValueError, match="all axes must be unique"):
        convolve([1], [2], axes=[0, 0])


@pytest.mark.parametrize("method", ["fft", "oa"])
def test_convolve_basic(method):
    a = [3, 4, 5, 6, 5, 4]
    b = [1, 2, 3]
    c = convolve(a, b, method=method)
    assert_eq(c, np.array([3, 10, 22, 28, 32, 32, 23, 12], dtype="float"))


@pytest.mark.parametrize("method", ["fft", "oa"])
def test_convolve_same(method):
    a = [3, 4, 5]
    b = [1, 2, 3, 4]
    c = convolve(a, b, mode="same", method=method)
    assert_eq(c, np.array([10, 22, 34], dtype="float"))


def test_convolve_broadcastable():
    a = np.arange(27).reshape(3, 3, 3)
    b = np.arange(3)
    for i in range(3):
        b_shape = [1] * 3
        b_shape[i] = 3
        x = convolve(a, b.reshape(b_shape), method="oa")
        y = convolve(a, b.reshape(b_shape), method="fft")
        assert_eq(x, y)


def test_zero_rank():
    a = 1289
    b = 4567
    c = convolve(a, b)
    assert_eq(c, a * b)


def test_single_element():
    a = np.array([4967])
    b = np.array([3920])
    c = convolve(a, b)
    assert_eq(c, a * b)


@pytest.mark.parametrize("method", ["fft", "oa"])
def test_2d_arrays(method):
    a = [[1, 2, 3], [3, 4, 5]]
    b = [[2, 3, 4], [4, 5, 6]]
    c = convolve(a, b, method=method)
    d = np.array(
        [[2, 7, 16, 17, 12], [10, 30, 62, 58, 38], [12, 31, 58, 49, 30]], dtype="float"
    )
    assert_eq(c, d)


@pytest.mark.parametrize("method", ["fft", "oa"])
def test_input_swapping(method):
    small = np.arange(8).reshape(2, 2, 2)
    big = 1j * np.arange(27).reshape(3, 3, 3)
    big += np.arange(27)[::-1].reshape(3, 3, 3)

    out_array = np.array(
        [
            [
                [0 + 0j, 26 + 0j, 25 + 1j, 24 + 2j],
                [52 + 0j, 151 + 5j, 145 + 11j, 93 + 11j],
                [46 + 6j, 133 + 23j, 127 + 29j, 81 + 23j],
                [40 + 12j, 98 + 32j, 93 + 37j, 54 + 24j],
            ],
            [
                [104 + 0j, 247 + 13j, 237 + 23j, 135 + 21j],
                [282 + 30j, 632 + 96j, 604 + 124j, 330 + 86j],
                [246 + 66j, 548 + 180j, 520 + 208j, 282 + 134j],
                [142 + 66j, 307 + 161j, 289 + 179j, 153 + 107j],
            ],
            [
                [68 + 36j, 157 + 103j, 147 + 113j, 81 + 75j],
                [174 + 138j, 380 + 348j, 352 + 376j, 186 + 230j],
                [138 + 174j, 296 + 432j, 268 + 460j, 138 + 278j],
                [70 + 138j, 145 + 323j, 127 + 341j, 63 + 197j],
            ],
            [
                [32 + 72j, 68 + 166j, 59 + 175j, 30 + 100j],
                [68 + 192j, 139 + 433j, 117 + 455j, 57 + 255j],
                [38 + 222j, 73 + 499j, 51 + 521j, 21 + 291j],
                [12 + 144j, 20 + 318j, 7 + 331j, 0 + 182j],
            ],
        ]
    )

    assert_eq(convolve(small, big, "full", method=method), out_array)
    assert_eq(convolve(big, small, "full", method=method), out_array)
    assert_eq(convolve(small, big, "same", method=method), out_array[1:3, 1:3, 1:3])
    assert_eq(convolve(big, small, "same", method=method), out_array[0:3, 0:3, 0:3])
    assert_eq(convolve(big, small, "valid", method=method), out_array[1:3, 1:3, 1:3])

    with pytest.raises(
        ValueError,
        match="For 'valid' mode in1 has to be at least as large as in2 in every dimension",
    ):
        convolve(small, big, "valid")


@pytest.mark.parametrize("axes", ["", None, 0, [0], -1, [-1]])
@pytest.mark.parametrize("method", ["fft", "oa"])
def test_convolve_real(axes, method):
    a = np.array([1, 2, 3])
    expected = np.array([1.0, 4.0, 10.0, 12.0, 9.0])

    if axes == "":
        out = convolve(a, a, method=method)
    else:
        out = convolve(a, a, method=method, axes=axes)
    assert_eq(out, expected)


@pytest.mark.parametrize("axes", [1, [1], -1, [-1]])
@pytest.mark.parametrize("method", ["fft", "oa"])
def test_convolve_real_axes(axes, method):
    a = np.array([1, 2, 3])
    expected = np.array([1.0, 4.0, 10.0, 12.0, 9.0])
    a = np.tile(a, [2, 1])
    expected = np.tile(expected, [2, 1])

    out = convolve(a, a, method=method, axes=axes)
    assert_eq(out, expected)


@pytest.mark.parametrize("method", ["fft", "oa"])
@pytest.mark.parametrize("axes", ["", None, 0, [0], -1, [-1]])
def test_convolve_complex(method, axes):
    a = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype="complex")
    expected = np.array([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j], dtype="complex")

    if axes == "":
        out = convolve(a, a, method=method)
    else:
        out = convolve(a, a, axes=axes, method=method)
    assert_eq(out, expected)


@pytest.mark.skip(reason="Utils function, not a test function")
def gen_convolve_shapes(sizes):
    return [(a, b) for a, b in product(sizes, repeat=2) if abs(a - b) > 3]


@pytest.mark.skip(reason="Utils function, not a test function")
def gen_convolve_shapes_eq(sizes):
    return [(a, b) for a, b in product(sizes, repeat=2) if a >= b]


@pytest.mark.skip(reason="Utils function, not a test function")
def gen_convolve_shapes_2d(sizes):
    shapes0 = gen_convolve_shapes_eq(sizes)
    shapes1 = gen_convolve_shapes_eq(sizes)
    shapes = [ishapes0 + ishapes1 for ishapes0, ishapes1 in zip(shapes0, shapes1)]

    modes = ["full", "valid", "same"]
    return [
        ishapes + (imode,)
        for ishapes, imode in product(shapes, modes)
        if imode != "valid" or (ishapes[0] > ishapes[1] and ishapes[2] > ishapes[3])
    ]


@pytest.mark.slow
@pytest.mark.parametrize("method", ["fft", "oa"])
@pytest.mark.parametrize(
    "shape_a_0, shape_b_0",
    gen_convolve_shapes_eq(list(range(100)) + list(range(100, 1000, 23))),
)
def test_convolve_real_manylens(method, shape_a_0, shape_b_0):
    a = np.random.rand(shape_a_0)
    b = np.random.rand(shape_b_0)

    expected = scipy.signal.fftconvolve(a, b)

    out = convolve(a, b, method=method)
    assert_eq(out, expected)


@pytest.mark.parametrize("method", ["fft", "oa"])
@pytest.mark.parametrize("shape_a_0, shape_b_0", gen_convolve_shapes([50, 47, 6, 4]))
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("chunks", [5, 10])
def test_convolve_1d_noaxes(shape_a_0, shape_b_0, is_complex, mode, method, chunks):

    a = np.random.rand(shape_a_0)
    b = np.random.rand(shape_b_0)
    if is_complex:
        a = a + 1j * np.random.rand(shape_a_0)
        b = b + 1j * np.random.rand(shape_b_0)
    if mode != "valid" or (shape_a_0 > shape_b_0):
        a = da.from_array(a, chunks=chunks)

        expected = scipy.signal.fftconvolve(a, b, mode=mode)

        out = convolve(a, b, mode=mode, method=method)

        assert_eq(out, expected)


@pytest.mark.parametrize("method", ["fft", "oa"])
@pytest.mark.parametrize("axes", [0, 1])
@pytest.mark.parametrize("shape_a_0, shape_b_0", gen_convolve_shapes([50, 47, 6, 4]))
@pytest.mark.parametrize("shape_a_extra", [1, 3])
@pytest.mark.parametrize("shape_b_extra", [1, 3])
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("chunks", [5, 10])
def test_convolve_1d_axes(
    axes,
    shape_a_0,
    shape_b_0,
    shape_a_extra,
    shape_b_extra,
    is_complex,
    mode,
    method,
    chunks,
):
    ax_a = [shape_a_extra] * 2
    ax_b = [shape_b_extra] * 2
    ax_a[axes] = shape_a_0
    ax_b[axes] = shape_b_0

    a = np.random.rand(*ax_a)
    b = np.random.rand(*ax_b)
    if is_complex:
        a = a + 1j * np.random.rand(*ax_a)
        b = b + 1j * np.random.rand(*ax_b)

    if mode != "valid" or a.shape[axes] > b.shape[axes]:
        a = da.from_array(a, chunks=chunks)

        expected = scipy.signal.fftconvolve(a, b, mode=mode, axes=axes)

        out = convolve(a, b, mode=mode, method=method, axes=axes)

        assert_eq(out, expected)


@pytest.mark.parametrize("method", ["fft", "oa"])
@pytest.mark.parametrize(
    "shape_a_0, shape_b_0, " "shape_a_1, shape_b_1, mode",
    gen_convolve_shapes_2d([50, 47, 6, 4]),
)
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("chunks", [15, 25])
def test_convolve_2d_noaxes(
    shape_a_0, shape_b_0, shape_a_1, shape_b_1, mode, is_complex, method, chunks
):
    a = np.random.rand(shape_a_0, shape_a_1)
    b = np.random.rand(shape_b_0, shape_b_1)
    if is_complex:
        a = a + 1j * np.random.rand(shape_a_0, shape_a_1)
        b = b + 1j * np.random.rand(shape_b_0, shape_b_1)

    a = da.from_array(a, chunks=chunks)

    expected = scipy.signal.fftconvolve(a, b, mode=mode)

    out = convolve(a, b, mode=mode, method=method)

    assert_eq(out, expected)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["fft", "oa"])
@pytest.mark.parametrize("axes", [[0, 1], [0, 2], [1, 2]])
@pytest.mark.parametrize(
    "shape_a_0, shape_b_0, " "shape_a_1, shape_b_1, mode",
    gen_convolve_shapes_2d([50, 47, 6, 4]),
)
@pytest.mark.parametrize("shape_a_extra", [1, 3])
@pytest.mark.parametrize("shape_b_extra", [1, 3])
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("chunks", [15, 25])
def test_convolve_2d_axes(
    axes,
    shape_a_0,
    shape_b_0,
    shape_a_1,
    shape_b_1,
    mode,
    shape_a_extra,
    shape_b_extra,
    is_complex,
    method,
    chunks,
):
    ax_a = [shape_a_extra] * 3
    ax_b = [shape_b_extra] * 3
    ax_a[axes[0]] = shape_a_0
    ax_b[axes[0]] = shape_b_0
    ax_a[axes[1]] = shape_a_1
    ax_b[axes[1]] = shape_b_1

    a = np.random.rand(*ax_a)
    b = np.random.rand(*ax_b)
    if is_complex:
        a = a + 1j * np.random.rand(*ax_a)
        b = b + 1j * np.random.rand(*ax_b)

    a = da.from_array(a, chunks=chunks)
    expected = scipy.signal.fftconvolve(a, b, mode=mode, axes=axes)

    out = convolve(a, b, mode=mode, method=method, axes=axes)

    assert_eq(out, expected)
