=======
History
=======

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
