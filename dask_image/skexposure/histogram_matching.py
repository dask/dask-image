import dask
import dask.array as da
import numpy as np

# Only these types are allowed, since some intermediate memory storage scales
# as the number of unique values in the range of the dtype.
DTYPES = (np.uint8, np.uint16, np.int8, np.int16)


def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    if ((source.dtype not in DTYPES) or (template.dtype not in DTYPES)):
        raise ValueError('Parallelization of match_histograms implemented '
                         'only for small integer types, not: %s %s' %
                         (source.dtype, template.dtype))

    _, src_unique_indices, src_counts = da.unique(source.ravel(),
                                                  return_inverse=True,
                                                  return_counts=True)
    tmpl_values, tmpl_counts = da.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = da.cumsum(src_counts) / source.size
    tmpl_quantiles = da.cumsum(tmpl_counts) / template.size

    # The interpolation is a bottleneck and must be done on local node.
    # This requires an in-memory array which could have a length equal in size
    # to the unique values of the underlying dtype.  This is why we limit to
    # small integers.
    interp_a_values = dask.delayed(_interpolate)(src_quantiles, tmpl_quantiles, tmpl_values)
    interp_a_values = da.from_delayed(interp_a_values, dtype=np.float, shape=src_quantiles.shape)

    result = src_unique_indices.map_blocks(lambda chunk, values: values[chunk],
                                           interp_a_values,
                                           dtype=interp_a_values.dtype)
    return result.reshape(source.shape)


def _interpolate(src_quantiles, tmpl_quantiles, tmpl_values):
    src_quantiles, tmpl_quantiles, tmpl_values = dask.compute(src_quantiles,
                                                              tmpl_quantiles,
                                                              tmpl_values)
    return np.interp(src_quantiles, tmpl_quantiles, tmpl_values)


def match_histograms(image, reference, *, multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.

    The adjustment is applied separately for each channel.

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.

    Returns
    -------
    matched : ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and '
                             'reference image must match!')

        matched = []
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel],
                                                    reference[..., channel])
            matched.append(matched_channel)
        matched = da.stack(matched, axis=-1)
    else:
        matched = _match_cumulative_cdf(image, reference)

    return matched
