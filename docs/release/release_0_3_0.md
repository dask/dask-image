# dask-image 0.3.0

We're pleased to announce the release of dask-image 0.3.0!

Released 2020-05-27

## Highlights
- Python 3.8 is now supported (#131)
- Support for Python 2.7 and 3.5 has been dropped (#119) (#131)
- We have a dask-image quickstart guide (#108), available from the dask examples page: https://examples.dask.org/applications/image-processing.html


## New Features
- Distributed labeling has been impolemented (#94)
- Area function added to dask_image.ndmeasure (#115)


## Improvements
- Optimize out first `where` in `label` (#102)


## Bug Fixes
- Bugfix in `center_of_mass` to correctly handle integer input arrays (#122)
- Test float cast in `_norm_args` (#105)
- Handle Dask's renaming of `atop` to `blockwise` (#98)


## API Changes
- Rename the input argument to image in the ndimage functions (#117)
- Rename labels in ndmeasure function arguments (#126)


## Support
- Update HISTORY for 0.2.0 release  [ci skip] (#85)
- Update installation instructions so conda is the preferred method (#88)
- Add Python 3.7 to Travis CI (#89)
- Add instructions for building docs with sphinx to CONTRIBUTING.rst (#90)
- Sort Python 3.7 requirements (#91)
- Use double equals for exact package versions (#92)
- Use flake8 (#93)
- Note Python 3.7 support (#95)
- Fix the Travis MacOS builds (update XCode to version 9.4 and use matplotlib 'Agg' backend) (#113)


## 7 authors added to this release (alphabetical)

- [Amir Khalighi](https://github.com/dask/dask-image/commits?author=akhalighi) - @akhalighi
- [Elliana May](https://github.com/dask/dask-image/commits?author=Mause) - @Mause
- [Genevieve Buckley](https://github.com/dask/dask-image/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [jakirkham](https://github.com/dask/dask-image/commits?author=jakirkham) - @jakirkham
- [Jaromir Latal](https://github.com/dask/dask-image/commits?author=jermenkoo) - @jermenkoo
- [Juan Nunez-Iglesias](https://github.com/dask/dask-image/commits?author=jni) - @jni
- [timbo8](https://github.com/dask/dask-image/commits?author=timbo8) - @timbo8


## 2 reviewers added to this release (alphabetical)

- [Genevieve Buckley](https://github.com/dask/dask-image/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [jakirkham](https://github.com/dask/dask-image/commits?author=jakirkham) - @jakirkham
