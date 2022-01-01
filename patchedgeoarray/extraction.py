import numpy as np
from . import readers
from . import common
from .common import OutOfBoundsError, BoundingBox, GeoarrayConfig


def extract(box: BoundingBox, config: GeoarrayConfig):
    if not box.contained_by(config.outer_box):
        raise OutOfBoundsError
    i_lo, j_lo = common.patch_index(box.left, box.bottom, config)
    i_hi, j_hi = common.patch_index(box.right, box.top, config)
    blocked = np.block([
        [
            readers.read(i, j, config)
            for j in range(j_lo, j_hi + 1)
        ]
        for i in range(i_lo, i_hi + 1)
    ])
    # Next: extract sub-array and
    res = config.resolution
    x_start, y_start = common.patch_start(i_lo, j_lo, config)
    i0 = np.ceil((box.left - x_start) / res).astype(int)
    j0 = np.ceil((box.bottom - y_start) / res).astype(int)
    x0 = x_start + i0 * res
    y0 = y_start + j0 * res
    ni = ((box.right - x0) / res).astype(int)
    nj = ((box.top - y0) / res).astype(int)
    return x0, y0, blocked[i0:i0 + ni, j0:j0 + nj]
