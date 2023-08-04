=======
History
=======

2023.08.1 (2023-08-04)
----------------------

We're pleased to announce the release of dask-image 2023.08.1!

This is a patch release to complete the dropping of python 3.8
in the previous release.

* Use `>=3.9` in `python_requires` in `setup.py` (#336)

2 authors added to this release (alphabetical)

* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Marvin Albert <https://github.com/dask/dask-image/commits?author=m-albert>`_ - @m-albert


0 reviewers added to this release (alphabetical)


2023.08.0 (2023-08-03)
----------------------

We're pleased to announce the release of dask-image 2023.08.0!

Highlights

This version fixes bugs related to processing CuPy backed dask arrays
and improves testing on GPU CI. It drops support for python 3.8 and
adds pandas as a dependency. As a feature improvement, the dask-image
equivalent of ``scipy.ndimage.label`` now supports arbitrary
structuring elements.

For full support of all GPU functionality in dask-image we recommend
using CuPy version 9.0.0 or higher.

Improvements

* Generalised ndmeasure.label to arbitrary structuring elements (#321)

Bug Fixes

* Added missing cupy test mark and fixed cupy threshold (#329)
* Moved functions from ndimage submodules to ndimage namespace (#325)

Updated requirements

* Drop Python 3.8, in accordance with NEP29 recommendation (#315)
* Require NumPy 1.18+ (#304)
* Add pandas requirement for find_objs function (#309)

Build Tools

* Continuous integration
   * Update GPU conda environment before running tests (#318)
   * Fix GitHub actions README badge (#323)
* Dependabot updates
   * Bump coverallsapp/github-action from 2.0.0 to 2.1.2 (#313)
   * Bump coverallsapp/github-action from 2.1.2 to 2.2.0 (#322)
   * Bump coverallsapp/github-action from 2.2.0 to 2.2.1 (#326)


6 authors added to this release (alphabetical)

* `Charles Blackmon-Luca <https://github.com/dask/dask-image/commits?author=charlesbluca>`_ - @charlesbluca
* `David Stansby <https://github.com/dask/dask-image/commits?author=dstansby>`_ - @dstansby
* `dependabot[bot] <https://github.com/dask/dask-image/commits?author=dependabot[bot]>`_ - @dependabot[bot]
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Marvin Albert <https://github.com/dask/dask-image/commits?author=m-albert>`_ - @m-albert


4 reviewers added to this release (alphabetical)

* `Charles Blackmon-Luca <https://github.com/dask/dask-image/commits?author=charlesbluca>`_ - @charlesbluca
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Juan Nunez-Iglesias <https://github.com/dask/dask-image/commits?author=jni>`_ - @jni


v2023.03.0 (2023-03-27)
-----------------------

We're pleased to announce the release of dask-image v2023.03.0!

Highlights

This version of dask-image drops support for python 3.7,
now requires a minimum Dask version of 2021.10.0 or higher 
(due to a security patch), and makes tifffile a regular requirement.
We also now build and publish wheel files to PyPI.

Improvements

* Documentation
   * Add GPU CI info to contributing docs (#300)
   * Docs: add GPU support info to coverage table (#301)

* Testing
   * Test `gaussian` alias (#287)
   * Update NaN block size tests for threshold_local function (#289)
   * Test `find_objects` w/incorrect array type (#292)

Deprecations and updated requirements

* Update supported python versions to 3.8, 3.9, 3.10, & 3.11 (drop python 3.7) (#284)
* Security update: Dask v2021.10.0 as minimum allowable version (#288)
* Make tifffile regular requirement (#295)

Build Tools

* Continuous integration
   * Refresh doc environment (#273)
   * Setup Coveralls with GitHub Actions (#274)
   * Pin to jinja2<3.1 to avoid Readthedocs build error (#278)
   * Updates `setup.py`'s Python versions (#285)
   * Combine CI workflows for testing and release upload to PyPI (#291)
   * Enable option to restart GHA (#293)
   * Readd `environment-latest.yml` symlink (#294)
   * Add python 3.10 to gpuCI matrix (#298)
* Releases
   * ENH: Build and publish wheels in GitHub CI (#272)
   * Update release notes script (#299)
   * Release notes for v2022.09.0 (#270)
* Dependabot updates
   * Create dependabot.yml (#279)
   * Bump actions/setup-python from 2 to 4 (#280)
   * Bump actions/checkout from 2 to 3 (#281)
   * Bump coverallsapp/github-action from 1.1.3 to 1.2.2 (#282)
   * Bump coverallsapp/github-action from 1.2.2 to 1.2.4 (#283)
   * Bump coverallsapp/github-action from 1.2.4 to 2.0.0 (#296)

Other Pull Requests

* Group all imread functions together in the same file (#290)

7 authors added to this release (alphabetical)

* `Charles Blackmon-Luca <https://github.com/dask/dask-image/commits?author=charlesbluca>`_ - @charlesbluca
* `dependabot[bot] <https://github.com/dask/dask-image/commits?author=dependabot[bot]>`_ - @dependabot[bot]
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Marvin Albert <https://github.com/dask/dask-image/commits?author=m-albert>`_ - @m-albert
* `Matt McCormick <https://github.com/dask/dask-image/commits?author=thewtex>`_ - @thewtex
* `Volker Hilsenstein <https://github.com/dask/dask-image/commits?author=VolkerH>`_ - @VolkerH


3 reviewers added to this release (alphabetical)

* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Matt McCormick <https://github.com/dask/dask-image/commits?author=thewtex>`_ - @thewtex


v2022.09.0 (2022-09-19)
-----------------------

We're pleased to announce the release of dask-image v2022.09.0!

Not much has changed since the last release.
Volker Hilsenstein has improved imread, which now uses natural sorting for strings.
Fred Blunt has fixed deprecation warnings from scipy.ndimage,
and we've also done some miscellaneous maintenance work.

Improvements

* Use natural sorting in  `imread(...)` when globbing multiple files  (#265)
* Avoid DeprecationWarnings when importing scipy.ndimage filter functions (#261)


Maintenance

* Remove/add testing for python 3.6/3.9, update CI pinnings (#257)
* Update docs theme for rebranding (#263)
* Run CI on `main` (#264)


6 authors added to this release (alphabetical)

* `Charles Blackmon-Luca <https://github.com/dask/dask-image/commits?author=charlesbluca>`_ - @charlesbluca
* `Fred Bunt <https://github.com/dask/dask-image/commits?author=fbunt>`_ - @fbunt
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Sarah Charlotte Johnson <https://github.com/dask/dask-image/commits?author=scharlottej13>`_ - @scharlottej13
* `Volker Hilsenstein <https://github.com/dask/dask-image/commits?author=VolkerH>`_ - @VolkerH


3 reviewers added to this release (alphabetical)

* `Charles Blackmon-Luca <https://github.com/dask/dask-image/commits?author=charlesbluca>`_ - @charlesbluca
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham


2021.12.0
----------

We're pleased to announce the release of dask-image 2021.12.0!

Highlights

The major highlights of this release include the introduction of new featurees for ``find_objects`` and spline filters.
We have also moved to using CalVer (calendar version numbers) to match the main Dask project.

New Features

* Find objects bounding boxes (#240)
* Add spline_filter and spline_filter1d (#215)


Improvements

* ENH: add remaining kwargs to binary_closing and binary_opening (#221)
* ndfourier: support n > 0 (for rfft) and improve performance (#222)
* affine_transform: increased shape of required input array slices (#216)


Bug Fixes

* BUG: add missing import of warnings in dask_image.ndmeasure (#224)
* Fix wrap bug in ndfilters convolve and correlate (#243)
* Upgrade for compatibility with latest dask release (#241)


Test infrastructure

* GitHub actions testing (#188)
* Set up gpuCI testing on PRs (#248)
* Remove `RAPIDS_VER` axis, bump `CUDA_VER` in gpuCI matrix (#249)


Documentation updates

* Code style cleanup (#227)
* Remove out of date email address, strip __author__ & __email__ (#225)
* Update release guide, Dask CalVer uses YYYY.MM.DD (#236)
* Update min python version in setup.py (#250)
* Use new Dask docs theme (#245)
* Docs: Add `find_objects` to the coverage table (#254)


Other Pull Requests

* Switch to CalVer (calendar versioning) (#233)


6 authors added to this release (alphabetical)

* `anlavandier <https://github.com/dask/dask-image/commits?author=anlavandier>`_ - @anlavandier
* `Charles Blackmon-Luca <https://github.com/dask/dask-image/commits?author=charlesbluca>`_ - @charlesbluca
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `Gregory R. Lee <https://github.com/dask/dask-image/commits?author=grlee77>`_ - @grlee77
* `Jacob Tomlinson <https://github.com/dask/dask-image/commits?author=jacobtomlinson>`_ - @jacobtomlinson
* `Marvin Albert <https://github.com/dask/dask-image/commits?author=m-albert>`_ - @m-albert


6 reviewers added to this release (alphabetical)

* `anlavandier <https://github.com/dask/dask-image/commits?author=anlavandier>`_ - @anlavandier
* `Genevieve Buckley <https://github.com/dask/dask-image/commits?author=GenevieveBuckley>`_ - @GenevieveBuckley
* `Gregory R. Lee <https://github.com/dask/dask-image/commits?author=grlee77>`_ - @grlee77
* `Jacob Tomlinson <https://github.com/dask/dask-image/commits?author=jacobtomlinson>`_ - @jacobtomlinson
* `jakirkham <https://github.com/dask/dask-image/commits?author=jakirkham>`_ - @jakirkham
* `Marvin Albert <https://github.com/dask/dask-image/commits?author=m-albert>`_ - @m-albert


0.6.0 (2021-05-06)
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
* affine_transform: Remove inconsistencies with ndimage implementation #205

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
