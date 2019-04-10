import dask.array as da
import numpy as np

from . import _conv, _gaussian, _generic, _order


def threshold_local(image, block_size, method='gaussian', offset=0,
                    mode='reflect', param=None, cval=0):
    """Compute a threshold mask image based on local pixel neighborhood.

    Also known as adaptive or dynamic thresholding. The threshold value is
    the weighted mean for the local neighborhood of a pixel subtracted by a
    constant. Alternatively the threshold can be determined dynamically by a
    given function, using the 'generic' method.

    Parameters
    ----------
    image : (N, M) dask ndarray
        Input image.
    block_size : int
        Size of pixel neighborhood which is used to calculate the
        threshold value.
    method : {'generic', 'gaussian', 'mean', 'median'}, optional
        Method used to determine adaptive threshold for local neighbourhood in
        weighted mean image.

        * 'generic': use custom function (see `param` parameter)
        * 'gaussian': apply gaussian filter (see `param` parameter for custom\
                      sigma value)
        * 'mean': apply arithmetic mean filter
        * 'median': apply median rank filter

        By default the 'gaussian' method is used.
    offset : float, optional
        Constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value. Default offset is 0.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
        Default is 'reflect'.
    param : {int, function}, optional
        Either specify sigma for 'gaussian' method or function object for
        'generic' method. This functions takes the flat array of local
        neighbourhood as a single argument and returns the calculated
        threshold for the centre pixel.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    threshold : (N, M) dask ndarray
        Threshold image. All pixels in the input image higher than the
        corresponding pixel in the threshold image are considered foreground.

    References
    ----------
    # adaptivethreshold
    .. [1] https://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold

    Examples
    --------
    >>> import dask.array as da
    >>> image = da.random.random((1000, 1000), chunks=(100, 100))
    >>> result = threshold_local(image, 15, 'gaussian')
    """
    image = image.astype(np.float64)
    if method == 'generic':
        thresh_image = _generic.generic_filter(image, param, block_size,
                                               mode=mode, cval=cval)
    if method == 'gaussian':
        if param is None:
            sigma = (da.from_array(block_size, chunks=1) - 1) / 6.0
        else:
            sigma = param
        thresh_image = _gaussian.gaussian_filter(image, sigma, mode=mode,
                                                 cval=cval)
    elif method == 'mean':
        if isinstance(block_size, int or float):
            block_size = [block_size] * image.ndim
        elif isinstance(block_size, list or tuple):
            if len(block_size) == image.ndim:
                pass
            elif block_size.ndim == 0:
                block_size = [block_size] * image.ndim
            else:
                raise ValueError("Invalid block size. Please enter either:\n"
                                 "(1) A single value for use in all dimensions"
                                 " or\n(2) A tuple or list with length equal "
                                 "to image.ndim")
        else:
            raise ValueError("{} type ".format(type(block_size)) +
                             "of 'block_size' input argument not "
                             "recognised! Must be numeric or listlike")
        mask = da.ones(block_size, chunks=block_size)
        mask /= mask.sum()
        thresh_image = _conv.convolve(image, mask, mode=mode, cval=cval)
    elif method == 'median':
        thresh_image = _order.median_filter(image, block_size, mode=mode,
                                            cval=cval)
    else:
        raise ValueError("Invalid method specified. Please use `generic`, "
                         "`gaussian`, `mean`, or `median`.")
    return thresh_image - offset
