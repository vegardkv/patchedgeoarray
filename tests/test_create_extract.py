import tempfile

import numpy as np
import numpy.testing as npt
import pytest
from patchedgeoarray.common import BoundingBox, OutOfBoundsError, GeoarrayConfig
from patchedgeoarray import creation, extraction


def test_single_small_arrays():
    with tempfile.TemporaryDirectory() as td:
        config = GeoarrayConfig(
            outer_box=BoundingBox(0, 10000, 0, 10000),
            resolution=10,
            patch_size=100,
            mode='file',
            file_directory=td,
        )
        sub_solution = np.arange(20).reshape(4, 5)
        creation.insert(20, 20, sub_solution, config)

        # Test extracting defined area
        x0, y0, data = extraction.extract(BoundingBox(19, 61, 19, 71), config)
        assert data.shape == sub_solution.shape
        npt.assert_allclose(sub_solution, data)

        # Test extracting defined area, including boundary coordinates
        x0, y0, data = extraction.extract(BoundingBox(20, 60, 20, 70), config)
        assert data.shape == sub_solution.shape
        npt.assert_allclose(sub_solution, data)

        # Test extracting defined area, including lower boundary
        x0, y0, data = extraction.extract(BoundingBox(20, 61, 20, 71), config)
        assert data.shape == sub_solution.shape
        npt.assert_allclose(sub_solution, data)

        # Test extracting defined area, including upper boundary
        x0, y0, data = extraction.extract(BoundingBox(19, 60, 19, 70), config)
        assert data.shape == sub_solution.shape
        npt.assert_allclose(sub_solution, data)

        # Test going beyond defined box
        solution = np.full((6, 8), fill_value=np.nan)
        solution[1:5, 1:6] = sub_solution
        x0, y0, data = extraction.extract(BoundingBox(9, 71, 9, 91), config)
        npt.assert_allclose(np.isnan(data), np.isnan(solution))
        npt.assert_allclose(data[~np.isnan(data)], solution[~np.isnan(solution)])

        # Test outside defined area
        with pytest.raises(OutOfBoundsError):
            extraction.extract(BoundingBox(-10000, -1000, -100000, -9000), config)
        with pytest.raises(OutOfBoundsError):
            extraction.extract(BoundingBox(10, 90000000000, 10, 100), config)


def test_multi_patches():
    with tempfile.TemporaryDirectory() as td:
        inner = np.arange(200).reshape(10, 20)
        config = GeoarrayConfig(
            outer_box=BoundingBox(0, 1000, 0, 1000),
            resolution=1.0,
            patch_size=10,
            mode='file',
            file_directory=td,
        )
        creation.insert(5.0, 25.0, inner, config)

        # Extract exact
        x0, y0, data = extraction.extract(BoundingBox(4.9, 15.1, 24.9, 45.1), config)
        npt.assert_allclose(data, inner)
        assert x0 == 5.0
        assert y0 == 25.0
