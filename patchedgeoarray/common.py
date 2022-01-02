import pathlib

import numpy as np
from dataclasses import dataclass
from typing import Literal


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


@dataclass
class GeoarrayConfig:
    outer_box: BoundingBox
    resolution: float
    patch_size: int
    mode: Literal['numpy', 'file', 's3']
    name_scheme: Literal['default'] = 'default'
    # reader specific settings
    file_directory: str = ''
    s3_profile_name: str = ''
    s3_bucket_name: str = ''

    def __post_init__(self):
        if isinstance(self.outer_box, dict):
            self.outer_box = BoundingBox(**self.outer_box)

    @staticmethod
    def from_json(config_path):
        import json
        config_path = pathlib.Path(config_path)
        meta = json.load(open(config_path))
        config = GeoarrayConfig(**meta)
        config.file_directory = config_path.parent
        return config


def patch_index(x, y, config: GeoarrayConfig):
    x = np.asarray(x)
    y = np.asarray(y)
    patch_length = config.patch_size * config.resolution
    return ((x - config.outer_box.left) / patch_length).astype(int),\
           ((y - config.outer_box.bottom) / patch_length).astype(int)


def local_index(x, y, config: GeoarrayConfig):
    pi, pj = patch_index(x, y, config)
    s = patch_start(pi, pj, config)
    return ((x - s[0]) / config.resolution).astype(int),\
           ((y - s[1]) / config.resolution).astype(int)


def patch_start(i, j, config: GeoarrayConfig):
    return config.outer_box.left + i * config.patch_size * config.resolution,\
           config.outer_box.bottom + j * config.patch_size * config.resolution


def patch_name(i: int, j: int, config: GeoarrayConfig):
    if config.name_scheme == 'default':
        base = f'data_{i}_{j}.npy'
        if config.mode == 'file':
            return pathlib.Path(config.file_directory) / base
        elif config.mode == 's3':
            return base
        else:
            raise AssertionError
    else:
        raise NotImplementedError

