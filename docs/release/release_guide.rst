=============
Release Guide
=============

This guide documents the ``dask-image`` release process.
It is based on the ``napari`` release guide created by Kira Evans.

This guide is primarily intended for core developers of `dask-image`.
They will need to have a `PyPI <https://pypi.org>`_ account
with upload permissions to the ``dask-image`` package.
They will also need permissions to merge pull requests
in the ``dask-image`` conda-forge feedstock repository:
https://github.com/conda-forge/dask-image-feedstock.

You will also need these additional release dependencies
to complete the release process:


.. code-block:: bash

   pip install PyGithub>=1.44.1 twine>=3.1.1 tqdm



Set PyPI password as GitHub secret
----------------------------------

The `dask/dask-image` repository must have a PyPI API token as a GitHub secret.

This likely has been done already, but if it has not, follow
`this guide <https://pypi.org/help/#apitoken>`_ to gain a token and
`this other guide <https://help.github.com/en/actions/automating-your-workflow-with-github-actions/creating-and-using-encrypted-secrets>`_
to add it as a secret.


Determining the new version number
----------------------------------

We use `semantic versioning <https://medium.com/the-non-traditional-developer/semantic-versioning-for-dummies-45c7fe04a1f8>`_
for `dask-image`. This means version numbers have the format
`Major.Minor.Patch`.

`Versioneer <https://github.com/warner/python-versioneer>`_
then determines the exact version from the latest
`git tag <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`_
beginning with `v`.


Generate the release notes
--------------------------

The release notes contain a list of merges, contributors, and reviewers.

1. Crate a GH_TOKEN environment variable on your computer.

    On Linux/Mac:

    .. code-block:: bash

       export GH_TOKEN=<your-gh-api-token>

    On Windows:

    .. code-block::

       set GH_TOKEN <your-gh-api-token>


    If you don't already have a
    `personal GitHub API token <https://github.blog/2013-05-16-personal-api-tokens/>`_,
    you can create one from the developer settings of your GitHub account:
    `<https://github.com/settings/tokens>`_


2. Run the python script to generate the release notes,
including all changes since the last tagged release.

    Call the script like this:

    .. code-block:: bash

       python docs/release/generate_release_notes.py  <last-version-tag> master --version <new-version-number>


    An example:

    .. code-block:: bash

       python docs/release/generate_release_notes.py  v0.14.0 master --version 0.15.0


    See help for this script with:

    .. code-block:: bash

       python docs/release/generate_release_notes.py -h


3. Scan the PR titles for highlights, deprecations, API changes,
   and bugfixes, and mention these in the relevant sections of the notes.
   Try to present the information in an expressive way by mentioning
   the affected functions, elaborating on the changes and their
   consequences. If possible, organize semantically close PRs in groups.

4. Copy your edited release notes into the file ``HISTORY.rst``.

5. Make and merge a PR with the release notes before moving onto the next steps.


Create the release candidate
-----------------------------

Go to the dask-image releases page: https://github.com/dask/dask-image/releases

Click the "Draft Release" button to create a new release candidate.

- Both the tag version and release title should have the format ``vX.Y.Zrc1``.
- Copy-paste the release notes from ``HISTORY.rst`` for this release into the
  description text box.

Note here how we are using ``rc`` for release candidate to create a version
of our release we can test before making the real release.

Creating the release will trigger a GitHub actions script,
which automatically uploads the release to PyPI.


Testing the release candidate
-----------------------------

The release candidate can then be tested with

.. code-block:: bash

   pip install --pre dask-image


It is recommended that the release candidate is tested in a virtual environment
in order to isolate dependencies.

If the release candidate is not what you want, make your changes and
repeat the process from the beginning but
incrementing the number after ``rc`` (e.g. ``vX.Y.Zrc2``).

Once you are satisfied with the release candidate it is time to generate
the actual release.

Generating the actual release
-----------------------------

To generate the actual release you will now repeat the processes above
but now dropping the ``rc`` suffix from the version number.

This will automatically upload the release to PyPI, and will also
automatically begin the process to release the new version on conda-forge.

Releasing on conda-forge
------------------------

It usually takes about an hour or so for the conda-forge bot 
``regro-cf-autotick-bot`` to see that there is a new release
available on PyPI, and open a pull request in the ``dask-image``
conda-forge feedstock here: https://github.com/conda-forge/dask-image-feedstock

Note: the conda-forge bot will not open a PR for any of the release candidates,
only for the final release. Only one PR is opened for 

Before merging the pull request, first you should check:
* That all the tests have passed on CI for this pull request
* If any dependencies were changed, and should be updated in the pull request

Once that all looks good you can merge the pull request,
and the newest version of ``dask-image`` will automatically be made
available on conda-forge.

