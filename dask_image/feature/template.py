import numpy as np
import dask.array as da
from dask_image import ndfilters


def _window_sum_2d(image, window_shape):

    window_sum = da.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0] - 1])

    window_sum = da.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                  - window_sum[:, :-window_shape[1] - 1])

    return window_sum


def _window_sum_3d(image, window_shape):

    window_sum = _window_sum_2d(image, window_shape)

    window_sum = da.cumsum(window_sum, axis=2)
    window_sum = (window_sum[:, :, window_shape[2]:-1]
                  - window_sum[:, :, :-window_shape[2] - 1])

    return window_sum


def match_template(image, template, pad_input=False, mode='constant',
                   constant_values=0):
    '''
    Ported from skimage.feature.match_template
    Replaced scipy.signal.fftconvolve by dask_image.ndfilters.convolve and
    address boundary issue accordingly
    Worked around fancy indexing like array[boolean_mask]

    Match a template to a 2-D or 3-D image using normalized correlation.
    The output is an array with values between -1.0 and 1.0. The value at a
    given position corresponds to the correlation coefficient between the image
    and the template.
    For `pad_input=True` matches correspond to the center and otherwise to the
    top-left corner of the template. To find the best match you must search for
    peaks in the response (output) image.
    Parameters
    ----------
    image : (M, N[, D]) array
        2-D or 3-D input image.
    template : (m, n[, d]) array
        Template to locate. It must be `(m <= M, n <= N[, d <= D])`.
    pad_input : bool
        If True, pad `image` so that output is the same size as the image, and
        output values correspond to the template center. Otherwise, the output
        is an array with shape `(M - m + 1, N - n + 1)` for an `(M, N)` image
        and an `(m, n)` template, and matches correspond to origin
        (top-left corner) of the template.
    mode : see `numpy.pad`, optional
        Padding mode.
    constant_values : see `numpy.pad`, optional
        Constant values used in conjunction with ``mode='constant'``.
    Returns
    -------
    output : array
        Response image with correlation coefficients.
    '''
    if image.ndim < template.ndim:
        raise ValueError("Dimensionality of template must be less than or "
                         "equal to the dimensionality of image.")
    if np.any(np.less(image.shape, template.shape)):
        raise ValueError("Image must be larger than template.")

    image_shape = image.shape
    image = image.astype(np.float64)

    pad_width = tuple((width, width) for width in template.shape)
    if mode == 'constant':
        image = da.pad(image, pad_width=pad_width, mode=mode,
                       constant_values=constant_values)
    else:
        image = da.pad(image, pad_width=pad_width, mode=mode)

    # Use special case for 2-D images for much better performance in
    # computation of integral images
    if image.ndim == 2:
        image_window_sum = _window_sum_2d(image, template.shape)
        image_window_sum2 = _window_sum_2d(image ** 2, template.shape)
    elif image.ndim == 3:
        image_window_sum = _window_sum_3d(image, template.shape)
        image_window_sum2 = _window_sum_3d(image ** 2, template.shape)

    template_mean = template.mean()
    template_volume = np.prod(template.shape)
    template_ssd = da.sum((template - template_mean) ** 2)

    indent = [d//2+1 for d in template.shape]
    if image.ndim == 2:
        xcorr = (ndfilters.convolve(image, template[::-1, ::-1], mode=mode)
                 [indent[0]:-indent[0], indent[1]:-indent[1]])
    elif image.ndim == 3:
        xcorr = (ndfilters.convolve(image, template[::-1, ::-1, ::-1],
                 mode=mode)[indent[0]:-indent[0], indent[1]:-indent[1],
                 indent[2]:-indent[2]])

    numerator = xcorr - image_window_sum * template_mean

    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, template_volume, out=image_window_sum)
    denominator -= image_window_sum
    denominator *= template_ssd

    # sqrt of negative number not allowed
    da.maximum(denominator, 0, out=denominator)
    da.sqrt(denominator, out=denominator)

    # avoid zero-division
    # dask hasn't supported fancy indexing, so here's a workaround
    mask = denominator <= np.finfo(np.float64).eps
    numerator[mask] = 0.0
    denominator[mask] = 1.0
    response = numerator / denominator

    slices = []
    for i in range(template.ndim):
        if pad_input:
            d0 = (template.shape[i] - 1) // 2
            d1 = d0 + image_shape[i]
        else:
            d0 = template.shape[i] - 1
            d1 = d0 + image_shape[i] - template.shape[i] + 1
        slices.append(slice(d0, d1))

    return response[tuple(slices)]
