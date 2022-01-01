import pathlib
import numpy as np
from . import common
from patchedgeoarray.common import GeoarrayConfig


def read(i: int, j: int, config: GeoarrayConfig) -> np.ndarray:
    return {
        'numpy': None,
        'file': _file_read,
        's3': _s3_read
    }[config.mode](i, j, config)


""" Numpy read """


def _numpy_read(i: int, j: int) -> np.ndarray:
    raise NotImplementedError


""" File read """


def _file_read(i: int, j: int, config: GeoarrayConfig) -> np.ndarray:
    file_path = common.patch_name(i, j, config)
    return np.load(file_path)


""" S3 read """


def _s3_read(i: int, j: int, config: GeoarrayConfig) -> np.ndarray:
    import io
    import boto3
    # TODO: no existing boto3 session. Might impact quota? Performance?
    session = boto3.session.Session(profile_name=config.s3_profile_name)
    s3_resource = session.resource('s3')
    key = common.patch_name(i, j, config)
    s3_obj = s3_resource.Object(bucket_name=config.s3_bucket_name, key=key)
    the_bytes = io.BytesIO()
    s3_obj.download_fileobj(the_bytes)  # TODO: Compare to download_file to tempfile
    the_bytes.seek(0)
    return np.load(the_bytes, allow_pickle=True)  # TODO: allow_pickle required?
