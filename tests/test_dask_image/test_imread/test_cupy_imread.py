import numpy as np
import tifffile
import pytest

import dask_image.imread

cupy = pytest.importorskip("cupy", minversion="6.0.0")


@pytest.mark.cupy
def test_cupy_imread(tmp_path):
    a = np.random.uniform(low=0.0, high=1.0, size=(1, 4, 3)).astype(np.float32)

    fn = str(tmp_path/"test.tiff")
    with tifffile.TiffWriter(fn) as fh:
        for i in range(len(a)):
            fh.save(a[i])

    result = dask_image.imread.imread(fn, arraytype="cupy")
    assert type(result._meta) == cupy.ndarray
    assert type(result.compute()) == cupy.ndarray
