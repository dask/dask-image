from __future__ import absolute_import

import dask.array as da
import numpy as np

from dask_image.skexposure import histogram_matching
from dask_image import skexposure
from skimage import data

from skimage._shared.testing import assert_array_almost_equal, \
    assert_almost_equal

import pytest


@pytest.mark.parametrize('array, template, expected_array', [
    (da.arange(10, dtype=np.int16, chunks=5),
        da.arange(100, dtype=np.int16, chunks=10),
        da.arange(9, 100, 10, dtype=np.float)),
    # (da.arange(-5, 5, dtype=np.uint16, chunks=5),
    #     da.arange(-50, 50, dtype=np.uint16, chunks=10),
    #     da.arange(-41, 50, 10, dtype=np.float)),
    (da.random.randint(0, 10, 4, dtype=np.int16, chunks=2),
        da.ones(3, dtype=np.int16),
        da.ones(4))
])
def test_match_array_values(array, template, expected_array):
    # when
    matched = histogram_matching._match_cumulative_cdf(array, template)

    # then
    assert_array_almost_equal(matched, expected_array)


@pytest.mark.parametrize('dtype1, dtype2', [
    (np.int64, np.int16),
    (np.int16, np.int64),
    (np.float, np.int16),
    (np.int16, np.float),
])
def test_raises_value_error_on_unimplemented_dtypes(dtype1, dtype2):
    array = da.arange(10, dtype=dtype1)
    template = da.arange(100, dtype=dtype2)
    with pytest.raises(ValueError):
        histogram_matching._match_cumulative_cdf(array, template)


class TestMatchHistogram:

    image_rgb = da.from_array(data.chelsea(), chunks=128)
    template_rgb = da.from_array(data.astronaut(), chunks=128)

    @pytest.mark.parametrize('image, reference, multichannel', [
        (image_rgb, template_rgb, True),
        (image_rgb[:, :, 0], template_rgb[:, :, 0], False)
    ])
    def test_match_histograms(self, image, reference, multichannel):
        """Assert that pdf of matched image is close to the reference's pdf for
        all channels and all values of matched"""

        # when
        matched = skexposure.match_histograms(image, reference,
                                              multichannel=multichannel)

        matched_pdf = self._calculate_image_empirical_pdf(matched)
        reference_pdf = self._calculate_image_empirical_pdf(reference)

        # then
        for channel in range(len(matched_pdf)):
            reference_values, reference_quantiles = reference_pdf[channel]
            matched_values, matched_quantiles = matched_pdf[channel]

            for i, matched_value in enumerate(matched_values):
                closest_id = (
                    np.abs(reference_values - matched_value)
                ).argmin()
                assert_almost_equal(matched_quantiles[i],
                                    reference_quantiles[closest_id],
                                    decimal=1)

    @pytest.mark.parametrize('image, reference', [
        (image_rgb, template_rgb[:, :, 0]),
        (image_rgb[:, :, 0], template_rgb)
    ])
    def test_raises_value_error_on_channels_mismatch(self, image, reference):
        with pytest.raises(ValueError):
            skexposure.match_histograms(image, reference)

    def test_raises_value_error_on_multichannels_mismatch(self):
        with pytest.raises(ValueError):
            skexposure.match_histograms(self.image_rgb,
                                        self.template_rgb[..., :1],
                                        multichannel=True)

    @classmethod
    def _calculate_image_empirical_pdf(cls, image):
        """Helper function for calculating empirical probability density
        function of a given image for all channels"""

        if image.ndim > 2:
            image = image.transpose(2, 0, 1)
        channels = np.array(image, copy=False, ndmin=3)

        channels_pdf = []
        for channel in channels:
            channel_values, counts = np.unique(channel, return_counts=True)
            channel_quantiles = np.cumsum(counts).astype(np.float64)
            channel_quantiles /= channel_quantiles[-1]

            channels_pdf.append((channel_values, channel_quantiles))

        return np.asarray(channels_pdf)
