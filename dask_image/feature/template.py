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
    if image.ndim < template.ndim:
        raise ValueError("Dimensionality of template must be less than or "
                         "equal to the dimensionality of image.")
    if da.any(np.less(image.shape, template.shape)):
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
        xcorr = ndfilters.convolve(image, template[::-1, ::-1],
                mode=mode)[indent[0]:-indent[0], indent[1]:-indent[1]]
    elif image.ndim == 3:
        xcorr = ndfilters.convolve(image, template[::-1, ::-1, ::-1],
                mode=mode)[indent[0]:-indent[0], indent[1]:-indent[1],
                        indent[2]:-indent[2]]

    numerator = xcorr - image_window_sum * template_mean

    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, template_volume, out=image_window_sum)
    denominator -= image_window_sum
    denominator *= template_ssd
    da.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
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
