import numpy as np
import dask.array as da

def convolve(in1, in2, mode="full", method="fft", axes=None):
    """Convolve a N-dimensional dask array with an N-dimensional array
    using either the fft or the overlap-add method.

    Some parts of this docstring are copied from scipy.signal.
    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : parallel array
        First input. Will be cast to a dask array

    in2 : sequential array_like
        Second input. Should have the same number of dimensions as `in1`.
        Will be cast to a numpy array.

    mode : str {'full', 'valid', 'same', 'periodic'}, optional
        A string indicating the size of the output.

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
        ``periodic``
           `in1` is assumed to be periodic for padding purposes, The output
           is the same size as `in1`.

    method : str {'oa', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``oa``
           Overlap-add method. This is generally much faster than `fft` when
           `in1` is much larger than `in2` but can be slower when only a few
           output values are needed or when the arrays are very similar in
           shape and can only output float arrays (int or object array inputs
           will be cast to float).
        ``fft``
           Convolve `in1` and `in2` using the fast Fourier transform method.
           Can only output float arrays (int or object array inputs will be
           cast to float). (Default)

    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.


    Returns
    -------
    out : parallel array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.


    See Also
    --------
    scipy.signal.oaconvolve : Equivalent Scipy operation for the overlap-add
        method
    scipy.signal.fftconvolve : Equivalent Scipy operation for the FFT
        method


    Notes
    -----
    This function is a work in progress and there are some possible improvements
    that could be added.
    These include but are not limited to:
        * Working out the case where both input are dask arrays.
        * Giving users the possibility to have the `mode` argument differ between axes.
        * Giving users the possibility to use the direct convolution with sums as in ``scipy.signal.convolve``



    Examples
    --------
    Convolve a 100,000 sample signal chunked in 100 1,000 elements chunks
    with a 512-sample filter.

    >>> from scipy import signal
    >>> import dask.array as da
    >>> rng = np.random.default_rng()
    >>> sig = da.from_array(rng.standard_normal(100000), chunks = (1000,))
    >>> filt = signal.firwin(512, 0.01)
    >>> fsig = da.linalg.convolve(sig, filt).compute()

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> _ = ax_orig.plot(sig)
    >>> _ = ax_orig.set_title('White noise')
    >>> _ = ax_mag.plot(fsig)
    >>> _ = ax_mag.set_title('Filtered noise')
    >>> fig.tight_layout()
    >>> fig.show()

    References
    ----------
    .. [1] Wikipedia, "Overlap-add_method".
           https://en.wikipedia.org/wiki/Overlap-add_method

    .. [2] Richard G. Lyons. Understanding Digital Signal Processing,
           Third Edition, 2011. Chapter 13.10.
           ISBN 13: 978-0137-02741-5




    """

    from scipy.signal import fftconvolve, oaconvolve
    from scipy.signal.signaltools import _init_freq_conv_axes


    in1 = da.asarray(in1)
    in2 = np.asarray(in2)

    # Checking for trivial cases and incorrect flags
    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    if mode != "full" and mode != "same" and mode != "valid" and mode != "periodic":
        raise ValueError(
            "acceptable mode flags are 'valid'," " 'same', 'full' or 'periodic'"
        )
    if method not in ["fft", "oa"]:
        raise ValueError("acceptable method flags are 'oa', or 'fft'")

    # Pre-formatting or the the inputs, mainly for the `axes` argument
    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

    # _init_freq_conv_axes calls a function that will swap out inputs if required
    # when mode == "valid".We want to avoid having in2 be a dask array thus check
    # to see if the inputs were swapped and raise an error.
    if type(in1) == np.ndarray:
        raise ValueError(
            "For 'valid' mode in1 has to be at least as large as in2 in every dimension"
        )

    s1 = in1.shape
    s2 = in2.shape

    # If all axe were removed by the preformatting we only have to rely
    # on multiplication broadcasting rules.
    if not len(axes):
        in_cv = in1 * in2
        # This is the "full" output that is also valid.
        # To get the "same" output we need to center in some dimensions.
        if mode == "same" or mode == "periodic":
            not_axes_but_s1_1 = [
                a
                for a in range(in1.ndim)
                if a not in axes and s1[a] == 1 and s2[a] != 1
            ]
            in_cv = in_cv[
                tuple(
                    slice((s2[a] - 1) // 2, (s2[a] - 1) // 2 + 1)
                    if a in not_axes_but_s1_1
                    else slice(None, None)
                    for a in range(in1.ndim)
                )
            ]
            return in_cv

    else:
        # This iskind of a hack but it works. 
        not_axes_but_s1_1 = [
            a for a in range(in1.ndim) if a not in axes and s1[a] == 1 and s2[a] != 1
        ]
        if len(not_axes_but_s1_1) and (mode == "full" or mode == "valid"):
            new_shape = tuple(
                s1[i] for i in range(in1.ndim) if i not in not_axes_but_s1_1
            )
            in1 = in1.reshape(new_shape)
            for a in not_axes_but_s1_1:
                in1 = da.stack([in1] * s2[a], axis=a)
            return convolve(in1, in2, mode=mode, method=method, axes=axes)

        # Deals with the case where there is at least one axis a in which we do not
        # do the convolution that has s2[a] == s1[a] != 1
        not_axes_but_same_shape = [
            a for a in range(in1.ndim) if a not in axes and s1[a] == s2[a] != 1
        ]
        if len(not_axes_but_same_shape):
            to_rechunk = [a for a in not_axes_but_same_shape if len(in1.chunks[a]) != 1]
            new_chunk_size = tuple(
                -1 if a in to_rechunk else "auto" for a in range(in1.ndim)
            )
            in1 = in1.rechunk(new_chunk_size)

        depth = {i: s2[i] // 2 for i in axes}

        # Flags even dimensions and removes them by adding zeros
        # This is done to avoid from having some results show up twice
        # at the edge of blocks
        even_flag = np.r_[[1 - s2[a] % 2 if a in axes else 0 for a in range(in1.ndim)]]
        target_shape = np.asarray(s2)
        target_shape += even_flag

        if any(target_shape != np.asarray(s2)):
            # padding axes where in2 is even
            pad_width = tuple(
                (even_flag[a], 0) if a in axes else (0, 0) for a in range(in1.ndim)
            )
            in2 = da.pad(in2, pad_width)

        if mode != "valid":
            pad_width = tuple(
                (depth[i] - even_flag[i], depth[i]) if i in axes else (0, 0)
                for i in range(in1.ndim)
            )
            in1 = da.pad(in1, pad_width)

        if mode == "periodic":
            boundary = "periodic"
        else:
            boundary = 0

        cv_dict = {"oa": oaconvolve, "fft": fftconvolve}

        cv_func = lambda x: cv_dict[method](x, in2, mode="same", axes=axes)

        complex_result = in1.dtype.kind == "c" or in2.dtype.kind == "c"

        if complex_result:
            dtype = "complex"
        else:
            dtype = "float"

        # Actualy does the convolution with all the parameters preformatted
        in_cv = in1.map_overlap(
            cv_func, depth=depth, boundary=boundary, trim=True, dtype=dtype
        )

        # The output as to be reduced depending on the `mode` argument
        if mode == "valid":
            output_slicing = tuple(
                slice(depth[i], s1[i] - (depth[i] - even_flag[i]), 1)
                if i in depth.keys()
                else slice(0, None)
                for i in range(in1.ndim)
            )
            in_cv = in_cv[output_slicing]

        elif mode != "full":
            # Only have to undo the padding
            output_slicing = tuple(
                slice(p[0], -p[1]) if p != (0, 0) else slice(0, None) for p in pad_width
            )
            in_cv = in_cv[output_slicing]

    return in_cv