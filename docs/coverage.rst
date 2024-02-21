*****************
Function Coverage
*****************

Coverage of dask-image vs scipy ndimage functions
*************************************************

This table shows which SciPy ndimage functions are supported by dask-image.

.. list-table::
   :widths: 25 25 25 30
   :header-rows: 0

   * - Function name
     - SciPy ndimage
     - dask-image
     - dask-image GPU support
   * - ``affine_transform``
     - ✓
     - ✓
     - ✓
   * - ``binary_closing``
     - ✓
     - ✓
     - ✓
   * - ``binary_dilation``
     - ✓
     - ✓
     - ✓
   * - ``binary_erosion``
     - ✓
     - ✓
     - ✓
   * - ``binary_fill_holes``
     - ✓
     -
     -
   * - ``binary_hit_or_miss``
     - ✓
     -
     -
   * - ``binary_opening``
     - ✓
     - ✓
     - ✓
   * - ``binary_propagation``
     - ✓
     -
     -
   * - ``black_tophat``
     - ✓
     -
     -
   * - ``center_of_mass``
     - ✓
     - ✓
     -
   * - ``convolve``
     - ✓
     - ✓
     - ✓
   * - ``convolve1d``
     - ✓
     -
     -
   * - ``correlate``
     - ✓
     - ✓
     - ✓
   * - ``correlate1d``
     - ✓
     -
     -
   * - ``distance_transform_bf``
     - ✓
     -
     -
   * - ``distance_transform_cdt``
     - ✓
     -
     -
   * - ``distance_transform_edt``
     - ✓
     -
     -
   * - ``extrema``
     - ✓
     - ✓
     -
   * - ``find_objects``
     - ✓
     - ✓
     -
   * - ``fourier_ellipsoid``
     - ✓
     -
     -
   * - ``fourier_gaussian``
     - ✓
     - ✓
     -
   * - ``fourier_shift``
     - ✓
     - ✓
     -
   * - ``fourier_uniform``
     - ✓
     - ✓
     -
   * - ``gaussian_filter``
     - ✓
     - ✓
     - ✓
   * - ``gaussian_filter1d``
     - ✓
     -
     -
   * - ``gaussian_gradient_magnitude``
     - ✓
     - ✓
     - ✓
   * - ``gaussian_laplace``
     - ✓
     - ✓
     - ✓
   * - ``generate_binary_structure``
     - ✓
     -
     -
   * - ``generic_filter``
     - ✓
     - ✓
     - ✓
   * - ``generic_filter1d``
     - ✓
     -
     -
   * - ``generic_gradient_magnitude``
     - ✓
     -
     -
   * - ``generic_laplace``
     - ✓
     -
     -
   * - ``geometric_transform``
     - ✓
     -
     -
   * - ``grey_closing``
     - ✓
     -
     -
   * - ``grey_dilation``
     - ✓
     -
     -
   * - ``grey_erosion``
     - ✓
     -
     -
   * - ``grey_opening``
     - ✓
     -
     -
   * - ``histogram``
     - ✓
     - ✓
     -
   * - ``imread``
     - ✓
     - ✓
     - ✓
   * - ``iterate_structure``
     - ✓
     -
     -
   * - ``label``
     - ✓
     - ✓
     -
   * - ``labeled_comprehension``
     - ✓
     - ✓
     -
   * - ``laplace``
     - ✓
     - ✓
     - ✓
   * - ``map_coordinates``
     - ✓
     -
     -
   * - ``maximum``
     - ✓
     - ✓
     -
   * - ``maximum_filter``
     - ✓
     - ✓
     - ✓
   * - ``maximum_filter1d``
     - ✓
     -
     -
   * - ``maximum_position``
     - ✓
     - ✓
     -
   * - ``mean``
     - ✓
     - ✓
     -
   * - ``median``
     - ✓
     - ✓
     -
   * - ``median_filter``
     - ✓
     - ✓
     - ✓
   * - ``minimum``
     - ✓
     - ✓
     -
   * - ``minimum_filter``
     - ✓
     - ✓
     - ✓
   * - ``minimum_filter1d``
     - ✓
     -
     -
   * - ``minimum_position``
     - ✓
     - ✓
     -
   * - ``morphological_gradient``
     - ✓
     -
     -
   * - ``morphological_laplace``
     - ✓
     -
     -
   * - ``percentile_filter``
     - ✓
     - ✓
     - ✓
   * - ``prewitt``
     - ✓
     - ✓
     - ✓
   * - ``rank_filter``
     - ✓
     - ✓
     - ✓
   * - ``rotate``
     - ✓
     - ✓
     -
   * - ``shift``
     - ✓
     -
     -
   * - ``sobel``
     - ✓
     - ✓
     - ✓
   * - ``spline_filter``
     - ✓
     - ✓
     - ✓
   * - ``spline_filter1d``
     - ✓
     - ✓
     - ✓
   * - ``standard_deviation``
     - ✓
     - ✓
     -
   * - ``sum_labels``
     - ✓
     - ✓
     -
   * - ``threshold_local``
     - scikit-image function
     - ✓
     - ✓
   * - ``uniform_filter``
     - ✓
     - ✓
     - ✓
   * - ``uniform_filter1d``
     - ✓
     -
     -
   * - ``variance``
     - ✓
     - ✓
     -
   * - ``watershed_ift``
     - ✓
     -
     -
   * - ``white_tophat``
     - ✓
     -
     -
   * - ``zoom``
     - ✓
     -
     -
