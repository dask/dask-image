==============
Array backends
==============

Dask arrays describe a computation split into chunks. Each chunk is itself
an array, usually a NumPy array on the CPU. For functions with GPU support,
dask-image can also process chunks stored as CuPy arrays on a GPU.

The chunk type selects the implementation. For example,
``dask_image.ndfilters.gaussian_filter`` uses ``scipy.ndimage.gaussian_filter``
for NumPy chunks and ``cupyx.scipy.ndimage.gaussian_filter`` for CuPy chunks.
You call the same dask-image function in both cases. Passing a NumPy-backed
Dask array does not automatically move data to a GPU.

See :doc:`coverage` for the functions that support GPU arrays. Support for
one function does not imply support for every function or every argument
combination in a module.

CPU example
-----------

Create an image with NumPy chunks and apply a Gaussian filter:

.. code-block:: python

    import dask.array as da
    import numpy as np
    from dask_image.ndfilters import gaussian_filter

    image = da.from_array(
        np.arange(256, dtype=np.float32).reshape(16, 16), chunks=(8, 8)
    )
    filtered = gaussian_filter(image, sigma=1)
    result = filtered.compute()
    assert isinstance(result, np.ndarray)
    assert result.shape == image.shape

``filtered`` is a lazy Dask array. ``compute()`` evaluates it and assembles
the complete result, so that result must fit in memory.

GPU example
-----------

This example requires a working CuPy installation and a compatible GPU.
See the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_
for installation instructions matching your hardware.

Starting with ``image`` from the CPU example, convert each chunk to CuPy:

.. code-block:: python

    import cupy as cp

    gpu_image = image.map_blocks(
        cp.asarray, meta=cp.empty((0, 0), dtype=image.dtype)
    )
    gpu_filtered = gaussian_filter(gpu_image, sigma=1)
    gpu_result = gpu_filtered.compute()
    assert isinstance(gpu_result, cp.ndarray)

The conversion is lazy. During computation, chunks are transferred to the
GPU before filtering. ``gpu_result`` occupies GPU memory; use
``cp.asnumpy(gpu_result)`` if you need a NumPy result on the CPU. As with the
CPU example, computing the complete array requires enough memory to hold it.

How dispatch works
------------------

Internally, dask-image inspects a Dask array's metadata to identify its chunk
type without computing its values. A dispatcher selects a registered
implementation for that type. CuPy implementations are registered lazily,
so CPU-only users do not need to install CuPy.

Dispatch selects the function that processes each chunk. Dask still manages
the task graph, scheduling, and chunk overlaps. Changing the backend does not
remove the need to choose suitable chunk sizes or configure the scheduler.

When a function takes several arrays, keep their backing array types
compatible. Convert NumPy chunks to CuPy explicitly before combining them
with GPU-backed arrays. An unsupported backend does not imply an automatic
fallback to NumPy; check the coverage table and the function's documentation.
