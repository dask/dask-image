.. highlight:: shell

==========
Quickstart
==========


Importing dask-image
--------------------
Import dask image is with an underscore, like this example:

.. code-block:: python

    import dask_image.imread
    import dask_image.ndfilters


Dask Examples
-------------
We highly recommend checking out the image-processing.ipynb notebook 
in the ``applications/`` folder of the dask-examples repository:

https://github.com/dask/dask-examples

All the example notebooks are available to launch with
mybinder and test out interactively.


An Even Quicker Start
---------------------

You can read files stored on disk into a dask array
by passing the filename, or regex matching multiple filenames
into ``imread()``.

.. code-block:: python

    filename_pattern = 'path/to/image-*.png'
    images = dask_image.imread.imread(filename_pattern)

If your images are parts of a much larger image, 
dask can stack, concatenate or block chunks together:
http://docs.dask.org/en/latest/array-stack.html


Calling dask-image functions is also easy.

.. code-block:: python

    import dask_image.ndfilters
    blurred_image = dask_image.ndfilters.gaussian_filter(images, sigma=10)


Many other functions can be applied to dask arrays.
See the dask_array_documentation_ for more detail on general array operations.

.. _dask_array_documentation: http://docs.dask.org/en/latest/array.html

.. code-block:: python

    result = function_name(images)


Further Reading
---------------

Good places to start include:

* The dask-image API documentation: http://image.dask.org/en/latest/api.html
* The documentation on working with dask arrays: http://docs.dask.org/en/latest/array.html
