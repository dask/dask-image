#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import dask.array as da

from skimage import data, color, morphology, feature

import dask_image.feature as da_feat


def test_match_template():
    img = data.astronaut()
    img = color.rgb2gray(img)
    img = da.from_array(img, chunks=img.shape)
    p = dict(image=img, template=morphology.square(3), mode='constant',
             pad_input=True)

    response_sk = feature.match_template(**p)
    response_da = da_feat.match_template(**p).compute()

    assert np.array_equal(response_sk, response_da)
