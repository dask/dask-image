.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/dask/dask-image/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

dask-image could always use more documentation, whether as part of the
official dask-image docs, in docstrings, or even on the web in blog posts,
articles, and such.

To build the documentation locally and preview your changes, first set up the
conda environment for building the dask-image documentation:

.. code-block:: console

    $ conda env create -f environment_doc.yml
    $ conda activate dask_image_doc_env

This conda environment contains dask-image and its dependencies, sphinx,
and the dask-sphinx-theme.

Next, build the documentation with sphinx:

.. code-block:: console

    $ cd dask-image/docs
    $ make html

Now you can preview the html documentation in your browser by opening the file:
dask-image/docs/_build/html/index.html

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/dask/dask-image/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `dask-image` for local development.

1. Fork the `dask-image` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/dask-image.git

3. Install your local copy into an environment. Assuming you have conda installed, this is how you set up your fork for local development (on Windows drop `source`). Replace `"<some version>"` with the Python version used for testing.::

    $ conda create -n dask-image-env python="<some version>"
    $ source activate dask-image-env
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions::

    $ flake8 dask_image tests
    $ python setup.py test or py.test

   To get flake8, just conda install it into your environment.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for all supported Python versions. Check CIs
   and make sure that the tests pass for all supported Python versions
   and platforms.

Running tests locally
---------------------

To setup development environment, run the following command in your terminal.

.. list-table:: Conda command for each operating system
    :widths: 20 50
    :header-rows: 1

    * - OS
      - path
    * - Linux
      - conda env create -f .circleci/environments/tst_py38.yml
    * - OSX
      - conda env create -f .travis_support/environment/tst_py38.yml
    * - Windows
      - conda env create -f .appveyor_support/environments/tst_py38.yml

.. note::
    If you do not have Anaconda/miniconda installed, please follow `the instruction <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.


This command will create a new conda environment called ``dask_image_py38_env``
with all the dependency requirements.

Now you can activate your testing environment with:

.. code-block:: console

    $ conda activate dask_image_py38_env

Finally, installing the development version of the dask-image to start the
development.

.. code-block:: console

    $ pip install -e .

For local testing, please run ``pytest`` in the development environment.

.. code-block:: console

    $ pytest



Tips
----

To run a subset of tests::

    $ py.test tests/test_dask_image.py
