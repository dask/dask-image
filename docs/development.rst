.. highlights:: shell

===============
Developer guide
===============


Recommended development environment setup
-----------------------------------------

To setup development environment, run this command in your terminal:

.. code-block:: console

    $ conda env create -f=environment.yml

where the ``environment.yml`` for each operating system can be accessed from the
root of repository. For example, the files for Python 3.8 would be:

.. list-table:: Conda environment.yml file for each operating system
    :widths: 20 50
    :header-rows: 1

    * - OS
      - path
    * - Linux
      - .circleci/environments/tst_py38.yml
    * - OSX
      - .travis_support/environment/tst_py38.yml
    * - Windows
      - .appveyor_support/environments/tst_py38.yml


This command will create a new conda environment called ``dask_image_py38_env``
with all the dependency requirements.

Now you can activate your testing environment with:

.. code-block:: console

    $ conda activate dask_image_py38_env

Finally, installing the development version of the dask-image to start the
development.

.. code-block:: console

    $ pip install -e .

Testing the code
----------------

For local testing, please just run ``pytest`` in the development environment.

.. code-block:: console

    $ pytest
