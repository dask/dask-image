=======
History
=======

0.6.0 (2021-04-20)
------------------

We're pleased to announce the release of dask-image 0.6.0!

Highlights

The highlights of this release include GPU support for binary morphological
functions, and improvements to the performance of ``imread``.

Cupy version 9.0.0 or higher is required for GPU support of the ``ndmorph`` subpackage.
Cupy version 7.7.0 or higher is required for GPU support of the ``ndfilters`` and ``imread`` subpackages.

New Features

* GPU support for ndmorph subpackage: binary morphological functions (#157)

Improvements

* Improve imread performance: reduced overhead of pim.open calls when reading from image sequence (#182)

Bug Fixes

* dask-image imread v0.5.0 not working with dask distributed Client & napari (#194)
* Not able to map actual image name with dask_image.imread (#200, fixed by #182)

API Changes

* Add alias ``gaussian`` pointing to ``gaussian_filter`` (#193)

Other Pull Requests

* Change default branch from master to main (#185)
* Fix rst formatting in release_guide.rst (#186)

4 authors added to this release (alphabetical)

* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `Julia Signell <https://github.com/dask/dask-image/commits?author=jsignell>`_ - @jsignell
* `KM Goh <https://github.com/dask/dask-image/commits?author=K-Monty>`_ - @K-Monty
* `Marvin Albert <https://github.com/dask/dask-image/commits?author=m-albert>`_ - @m-albert

2 reviewers added to this release (alphabetical)

* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `KM Goh <https://github.com/dask/dask-image/commits?author=K-Monty>`_ - @K-Monty

0.5.0 (2021-02-01)
------------------

We're pleased to announce the release of dask-image 0.5.0!

Highlights

The biggest highlight of this release is our new affine transformation feature, contributed by Marvin Albert.
The SciPy Japan sprint in November 2020 led to many improvements, and I'd like to recognise the hard work by Tetsuo Koyama and Kuya Takami.
Special thanks go to everyone who joined us at the conference!

New Features

* Affine transformation feature added: from dask_image.ndinterp import affine_transform (#159)
* GPU support added for local_threshold with method='mean' (#158)
* Pathlib input now accepted for imread functions (#174)

Improvements

* Performance improvement for 'imread', we now use `da.map_blocks` instead of `da.concatenate` (#165)

Bug Fixes

* Fixed imread tests (add `contiguous=True` when saving test data with tifffile) (#164)
* FIXed scipy LooseVersion for sum_labels check (#176)

API Changes

* 'sum' is renamed to 'sum_labels' and a add deprecation warning added (#172)

Documentation improvements

* Add section Talks and Slides #163 (#169)
* Add link to SciPy Japan 2020 talk (#171)
* Add development guide to setup environment and run tests (#170)
* Update information in AUTHORS.rst (#167)

Maintenance

* Update dependencies in Read The Docs environment (#168)

6 authors added to this release (alphabetical)

* `Fabian Chong <https://github.com/dask/dask-image/commits?author=feiming>`_ - @feiming
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Kuya Takami <https://github.com/dask/dask-image/commits?author=ku-ya>`_ - @ku-ya
* `Marvin Albert <https://github.com/dask/dask-image/commits?author=m-albert>`_ - @m-albert
* `Tetsuo Koyama <https://github.com/dask/dask-image/commits?author=tkoyama010>`_ - @tkoyama010


7 reviewers added to this release (alphabetical)

* `Fabian Chong <https://github.com/dask/dask-image/commits?author=feiming>`_ - @feiming
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `Gregory R. Lee <https://github.com/dask/dask-image/commits?author=grlee77>`_ - @grlee77
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Juan Nunez-Iglesias <https://github.com/dask/dask-image/commits?author=jni>`_ - @jni
* `Marvin Albert <https://github.com/dask/dask-image/commits?author=m-albert>`_ - @m-albert
* `Tetsuo Koyama <https://github.com/dask/dask-image/commits?author=tkoyama010>`_ - @tkoyama010

0.4.0 (2020-09-02)
------------------

We're pleased to announce the release of dask-image 0.4.0!

Highlights

The major highlight of this release is support for cupy GPU arrays for dask-image subpackages imread and ndfilters.
Cupy version 7.7.0 or higher is required to use this functionality.
GPU support for the remaining dask-image subpackages (ndmorph, ndfourier, and ndmeasure) will be rolled out at a later date, beginning with ndmorph.

We also have a new function, threshold_local, similar to the scikit-image local threshold function.

Lastly, we've made more improvements to the user documentation, which includes work by new contributor @abhisht51.

New Features

* GPU support for ndfilters & imread modules (#151)
* threshold_local function for dask-image ndfilters (#112)

Improvements

* Add function coverage table to the dask-image docs (#155)
* Developer documentation: release guide (#142)
* Use tifffile for testing instead of scikit-image (#145)


3 authors added to this release (alphabetical)

* `Abhisht Singh <https://github.com/dask/dask-image/commits?author=abhisht51>`_ - @abhisht51
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham


2 reviewers added to this release (alphabetical)

* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `Juan Nunez-Iglesias <https://github.com/dask/dask-image/commits?author=jni>`_ - @jni

0.3.0 (2020-06-06)
------------------

We're pleased to announce the release of dask-image 0.3.0!

Highlights

* Python 3.8 is now supported (#131)
* Support for Python 2.7 and 3.5 has been dropped (#119) (#131)
* We have a dask-image quickstart guide (#108), available from the dask examples page: https://examples.dask.org/applications/image-processing.html

New Features

* Distributed labeling has been implemented (#94)
* Area measurement function added to dask_image.ndmeasure (#115)

Improvements

* Optimize out first `where` in `label` (#102)

Bug Fixes

* Bugfix in `center_of_mass` to correctly handle integer input arrays (#122)
* Test float cast in `_norm_args` (#105)
* Handle Dask's renaming of `atop` to `blockwise` (#98)

API Changes

* Rename the input argument to image in the ndimage functions (#117)
* Rename labels in ndmeasure function arguments (#126)

Support

* Update installation instructions so conda is the preferred method (#88)
* Add Python 3.7 to Travis CI (#89)
* Add instructions for building docs with sphinx to CONTRIBUTING.rst (#90)
* Sort Python 3.7 requirements (#91)
* Use double equals for exact package versions (#92)
* Use flake8 (#93)
* Note Python 3.7 support (#95)
* Fix the Travis MacOS builds (update XCode to version 9.4 and use matplotlib 'Agg' backend) (#113)

7 authors added to this release (alphabetical)

* `Amir Khalighi <https://github.com/dask/dask-image/commits?author=akhalighi>`_ - @akhalighi
* `Elliana May <https://github.com/dask/dask-image/commits?author=Mause>`_ - @Mause
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Jaromir Latal <https://github.com/dask/dask-image/commits?author=jermenkoo>`_ - @jermenkoo
* `Juan Nunez-Iglesias <https://github.com/dask/dask-image/commits?author=jni>`_ - @jni
* `timbo8 <https://github.com/dask/dask-image/commits?author=timbo8>`_ - @timbo8

2 reviewers added to this release (alphabetical)

- `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
- `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham

0.2.0 (2018-10-10)
------------------

* Construct separate label masks in `labeled_comprehension` (#82)
* Use `full` to construct 1-D NumPy array (#83)
* Use NumPy's `ndindex` in `labeled_comprehension` (#81)
* Cleanup `test_labeled_comprehension_struct` (#80)
* Use 1-D structured array fields for position-based kernels in `ndmeasure` (#79)
* Rewrite `center_of_mass` using `labeled_comprehension` (#78)
* Adjust `extrema`'s internal structured type handling (#77)
* Test labeled_comprehension with object type (#76)
* Rewrite `histogram` to use `labeled_comprehension` (#75)
* Use labeled_comprehension directly in more function in ndmeasure (#74)
* Update mean's variables to match other functions (#73)
* Consolidate summation in `_ravel_shape_indices` (#72)
* Update HISTORY for 0.1.2 release (#71)
* Bump dask-sphinx-theme to 1.1.0 (#70)

0.1.2 (2018-09-17)
------------------

* Ensure `labeled_comprehension`'s `default` is 1D. (#69)
* Bump dask-sphinx-theme to 1.0.5. (#68)
* Use nout=2 in ndmeasure's label. (#67)
* Use custom kernel for extrema. (#61)
* Handle structured dtype in labeled_comprehension. (#66)
* Fixes for `_unravel_index`. (#65)
* Bump dask-sphinx-theme to 1.0.4. (#64)
* Unwrap some lines. (#63)
* Use dask-sphinx-theme. (#62)
* Refactor out `_unravel_index` function. (#60)
* Divide `sigma` by `-2`. (#59)
* Use Python 3's definition of division in Python 2. (#58)
* Force dtype of `prod` in `_ravel_shape_indices`. (#57)
* Drop vendored compatibility code. (#54)
* Drop vendored copy of indices and uses thereof. (#56)
* Drop duplicate utility tests from `ndmorph`. (#55)
* Refactor utility module for imread. (#53)
* Reuse `ndfilter` utility function in `ndmorph`. (#52)
* Cleanup freq_grid_i construction in _get_freq_grid. (#51)
* Use shared Python 2/3 compatibility module. (#50)
* Consolidate Python 2/3 compatibility code. (#49)
* Refactor Python 2/3 compatibility from imread. (#48)
* Perform `2 * pi` first in `_get_ang_freq_grid`. (#47)
* Ensure `J` is negated first in `fourier_shift`. (#46)
* Breakout common changes in fourier_gaussian. (#45)
* Use conda-forge badge. (#44)

0.1.1 (2018-08-31)
------------------

* Fix a bug in an ndmeasure test of an internal function.

0.1.0 (2018-08-31)
------------------

* First release on PyPI.
* Pulls in content from dask-image org.
* Supports reading of image files into Dask.
* Provides basic N-D filters with options to extend.
* Provides a few N-D Fourier filters.
* Provides a few N-D morphological filters.
* Provides a few N-D measurement functions for label images.
* Has 100% line coverage in test suite.
