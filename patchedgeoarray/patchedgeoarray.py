import pathlib
from typing import List, Optional
import numpy as np
from dataclasses import dataclass


class OutOfBoundsError(Exception):
    pass


@dataclass
class BoundingBox:
    left: float
    right: float
    bottom: float
    top: float

    def contained_by(self, other: 'BoundingBox') -> bool:
        return other.left <= self.left \
           and other.bottom <= self.bottom \
           and other.right >= self.right \
           and other.top >= self.top


class PatchedGeoArray:
    def __init__(self, box: BoundingBox, resolution: float, patch_size: int) -> None:
        self._box = box
        self._resolution = resolution
        self._patch_size = patch_size

    @property
    def resolution(self):
        return self._resolution

    def _read_patch(self, i: int, j: int) -> np.ndarray:
        raise NotImplementedError

    def _store_patch(self, i: int, j: int, data: np.ndarray) -> None:
        raise NotImplementedError

    def _patch_index(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        patch_length = self._patch_size * self._resolution
        return ((x - self._box.left) / patch_length).astype(int),\
               ((y - self._box.bottom) / patch_length).astype(int)

    def _local_index(self, x, y):
        s = self._patch_start(*self._patch_index(x, y))
        return ((x - s[0]) / self._resolution).astype(int),\
               ((y - s[1]) / self._resolution).astype(int)

    def _patch_start(self, i, j):
        return self._box.left + i * self._patch_size * self._resolution,\
               self._box.bottom + j * self._patch_size * self._resolution

    def extract(self, box: BoundingBox):
        if not box.contained_by(self._box):
            raise OutOfBoundsError
        i_lo, j_lo = self._patch_index(box.left, box.bottom)
        i_hi, j_hi = self._patch_index(box.right, box.top)
        blocked = np.block([
            [
                self._read_patch(i, j)
                for j in range(j_lo, j_hi + 1)
            ]
            for i in range(i_lo, i_hi + 1)
        ])
        # Next: extract sub-array and
        x_start, y_start = self._patch_start(i_lo, j_lo)
        i0 = np.ceil((box.left - x_start) / self._resolution).astype(int)
        j0 = np.ceil((box.bottom - y_start) / self._resolution).astype(int)
        x0 = x_start + i0 * self._resolution
        y0 = y_start + j0 * self._resolution
        ni = ((box.right - x0) / self._resolution).astype(int)
        nj = ((box.top - y0) / self._resolution).astype(int)
        return x0, y0, blocked[i0:i0+ni, j0:j0+nj]

    def insert(self, x0: float, y0: float, data: np.ndarray):
        assert np.isclose((x0 - self._box.left) % self._resolution, 0)
        assert np.isclose((y0 - self._box.bottom) % self._resolution, 0)
        x1 = x0 + self._resolution * (data.shape[0] - 1)
        y1 = y0 + self._resolution * (data.shape[1] - 1)
        p0x, p0y = self._patch_index(x0, y0)
        p1x, p1y = self._patch_index(x1, y1)
        if p0x < p1x:
            sx, sy = self._patch_start(p1x, p1y)
            split = ((sx - x0) / self._resolution).astype(int)
            assert split not in (0, data.shape[0])
            self.insert(x0, y0, data[:split, :])
            self.insert(sx, y0, data[split:, :])
        elif p0y < p1y:
            sx, sy = self._patch_start(p1x, p1y)
            split = ((sy - y0) / self._resolution).astype(int)
            assert split not in (0, data.shape[1])
            self.insert(x0, y0, data[:, :split])
            self.insert(x0, sy, data[:, split:])
        else:
            s0x, s0y = self._local_index(x0, y0)
            s1x, s1y = self._local_index(x1, y1)
            d = np.full((self._patch_size, self._patch_size), fill_value=np.nan)
            d[s0x:s1x + 1, s0y:s1y + 1] = data
            self._store_patch(p0x, p0y, d)

    def insert_geotiff(self, filename: str):
        import rasterio
        with rasterio.open(filename) as tif:
            bounds = tif.bounds.astype(int)
            self.insert(bounds.left, bounds.bottom, tif.read[0][::-1, :].T)


class NdArrayPatchedGeoArray(PatchedGeoArray):
    def __init__(self, box: BoundingBox, resolution: float, patch_size: int) -> None:
        super().__init__(box, resolution, patch_size)
        self._arrays: List[List[Optional[np.ndarray]]] = []

    def _store_patch(self, i: int, j: int, data: np.ndarray) -> None:
        assert data.shape == (self._patch_size, self._patch_size)
        while len(self._arrays) < i + 1:
            self._arrays.append([])
        while len(self._arrays[i]) < j + 1:
            self._arrays[i].append(None)
        if self._arrays[i][j] is None:
            self._arrays[i][j] = data
        else:
            undefined = np.isnan(self._arrays[i][j])
            self._arrays[i][j][undefined] = data[undefined]

    def _read_patch(self, i: int, j: int) -> np.ndarray:
        try:
            out = self._arrays[i][j]
        except IndexError:
            raise OutOfBoundsError
        if out is None:
            raise OutOfBoundsError
        return out


class FileBasedPatchedGeoArray(PatchedGeoArray):
    def __init__(self, box: BoundingBox, resolution: float, patch_size: int, directory: str) -> None:
        super().__init__(box, resolution, patch_size)
        self._directory = pathlib.Path(directory)

    def _data_file(self, i, j):
        return self._directory / f'data_{i}_{j}.npy'

    def _read_patch(self, i: int, j: int) -> np.ndarray:
        return np.load(self._data_file(i, j))

    def _store_patch(self, i: int, j: int, data: np.ndarray) -> None:
        assert data.shape == (self._patch_size, self._patch_size)
        fn = self._data_file(i, j)
        if fn.is_file():
            existing_data = np.load(fn)
            already_set = ~np.isnan(existing_data)
            data[already_set] = existing_data[already_set]
        np.save(fn, data)
