from typing import Optional
import numpy as np
from . import common
from patchedgeoarray.common import GeoarrayConfig


def insert_geotiff(tiff_file: str, config: GeoarrayConfig, dtype: Optional = None):
    import rasterio
    with rasterio.open(tiff_file) as tif:
        data = tif.read()[0][::-1, :].T
        if dtype is not None:
            data = data.astype(dtype)
        insert(tif.bounds.left, tif.bounds.bottom, data, config)


def insert(x0: float, y0: float, data: np.ndarray, config: GeoarrayConfig):
    assert np.isclose((x0 - config.outer_box.left) % config.resolution, 0)
    assert np.isclose((y0 - config.outer_box.bottom) % config.resolution, 0)
    x1 = x0 + config.resolution * (data.shape[0] - 1)
    y1 = y0 + config.resolution * (data.shape[1] - 1)
    p0x, p0y = common.patch_index(x0, y0, config)
    p1x, p1y = common.patch_index(x1, y1, config)
    if p0x < p1x:
        sx, sy = common.patch_start(p1x, p1y, config)
        split = ((sx - x0) / config.resolution).astype(int)
        assert split not in (0, data.shape[0])
        insert(x0, y0, data[:split, :], config)
        insert(sx, y0, data[split:, :], config)
    elif p0y < p1y:
        sx, sy = common.patch_start(p1x, p1y, config)
        split = ((sy - y0) / config.resolution).astype(int)
        assert split not in (0, data.shape[1])
        insert(x0, y0, data[:, :split], config)
        insert(x0, sy, data[:, split:], config)
    else:
        s0x, s0y = common.local_index(x0, y0, config)
        s1x, s1y = common.local_index(x1, y1, config)
        d = np.full((config.patch_size, config.patch_size), fill_value=np.nan)
        d[s0x:s1x + 1, s0y:s1y + 1] = data
        _store_patch(p0x, p0y, d, config)


def _store_patch(i: int, j: int, data: np.ndarray, config: GeoarrayConfig) -> None:
    assert data.shape == (config.patch_size, config.patch_size)
    fn = common.patch_name(i, j, config)
    if fn.is_file():
        existing_data = np.load(fn)
        already_set = ~np.isnan(existing_data)
        data[already_set] = existing_data[already_set]
    np.save(fn, data)
