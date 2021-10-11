import pytest
from typing import Callable
import tempfile
import numpy as np
import numpy.testing as npt
from patchedgeoarray.patchedgeoarray import NdArrayPatchedGeoArray, BoundingBox, OutOfBoundsError, PatchedGeoArray, \
    FileBasedPatchedGeoArray


PgaFactory = Callable[[BoundingBox, float, int], PatchedGeoArray]


def perform_single_small_array_tests(factory: PgaFactory):
    box = BoundingBox(0, 10000, 0, 10000)
    ga = factory(box, 10, 100)
    sub_solution = np.arange(20).reshape(4, 5)
    ga.insert(20, 20, sub_solution)

    # Test extracting defined area
    x0, y0, data = ga.extract(BoundingBox(19, 61, 19, 71))
    assert data.shape == sub_solution.shape
    npt.assert_allclose(sub_solution, data)

    # Test extracting defined area, including boundary coordinates
    x0, y0, data = ga.extract(BoundingBox(20, 60, 20, 70))
    assert data.shape == sub_solution.shape
    npt.assert_allclose(sub_solution, data)

    # Test extracting defined area, including lower boundary
    x0, y0, data = ga.extract(BoundingBox(20, 61, 20, 71))
    assert data.shape == sub_solution.shape
    npt.assert_allclose(sub_solution, data)

    # Test extracting defined area, including upper boundary
    x0, y0, data = ga.extract(BoundingBox(19, 60, 19, 70))
    assert data.shape == sub_solution.shape
    npt.assert_allclose(sub_solution, data)

    # Test going beyond defined box
    solution = np.full((6, 8), fill_value=np.nan)
    solution[1:5, 1:6] = sub_solution
    x0, y0, data = ga.extract(BoundingBox(9, 71, 9, 91))
    npt.assert_allclose(np.isnan(data), np.isnan(solution))
    npt.assert_allclose(data[~np.isnan(data)], solution[~np.isnan(solution)])

    # Test outside defined area
    with pytest.raises(OutOfBoundsError):
        ga.extract(BoundingBox(-10000, -1000, -100000, -9000))
    with pytest.raises(OutOfBoundsError):
        ga.extract(BoundingBox(10, 90000000000, 10, 100))


def perform_multi_patch_test(factory: PgaFactory):
    inner = np.arange(200).reshape(10, 20)
    box = BoundingBox(0, 1000, 0, 1000)
    ga = factory(box, 1.0, 10)
    ga.insert(5.0, 25.0, inner)

    # Extract exact
    x0, y0, data = ga.extract(BoundingBox(4.9, 15.1, 24.9, 45.1))
    npt.assert_allclose(data, inner)
    assert x0 == 5.0
    assert y0 == 25.0


def test_ndarray_pga_single_small_array():
    perform_single_small_array_tests(NdArrayPatchedGeoArray)


def test_ndarray_pga_multi_patch():
    perform_multi_patch_test(NdArrayPatchedGeoArray)


def test_file_based_pga_single_small_array():
    with tempfile.TemporaryDirectory() as t:
        perform_single_small_array_tests(lambda box, res, patch: FileBasedPatchedGeoArray(box, res, patch, t))


def test_file_based_pga_multi_patch():
    with tempfile.TemporaryDirectory() as t:
        perform_multi_patch_test(lambda box, res, patch: FileBasedPatchedGeoArray(box, res, patch, t))
