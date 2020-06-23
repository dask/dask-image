import numpy as np
import dask.array as da

from skimage import data, color, morphology, feature

import template

img = data.astronaut()
img = color.rgb2gray(img)
img = da.from_array(img)
p = dict(image=img, template=morphology.square(3), mode='constant',
        pad_input=True)

response_sk = feature.match_template(**p)
response_di = template.match_template(**p).compute()

print(np.array_equal(response_sk, response_di))
