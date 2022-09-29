#  Copyright (c) 2022. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""IO utilities"""

import multiprocessing as mp
import os
import pathlib
import shutil
from io import BytesIO
from pathlib import Path
from typing import IO, Optional, Tuple, Union
from urllib.parse import urlparse

import requests
from requests.auth import AuthBase

import lance
from lance.logging import logger
from pyarrow import fs

USER_AGENT = f"User-Agent: Lance/{lance.__version__} (contact@eto.ai)"


def open_uri(
    uri: Union[str, Path],
    mode: str = "rb",
    http_auth: Optional[Union[AuthBase, Tuple[str, str]]] = None,
    http_headers: Optional[dict] = None,
) -> IO:
    """Open URI for reading. Supports the following URI formats:

    - File System: ``/path/to/file`` or ``file:///path/to/file``
    - Http(s): ``http://`` or ``https://``
    - AWS S3: ``s3://``
    - Google Cloud Storage: ``gs://``

    Parameters
    ----------
    uri : str or Path
        URI to open
    mode : str, default 'rb'
        the file mode
    http_auth : AuthBase or tuple of (str, str), optional
        Authentication details when using http(s) uri's
    http_headers : dict, optional
        Extra headers when using http(s) uri's

    Return
    ------
    IO
    """
    if isinstance(uri, Path):
        return uri.open(mode=mode)
    parsed_uri = urlparse(uri)
    scheme = parsed_uri.scheme
    if not scheme or scheme == "file":
        # This is a local file
        return open(uri, mode=mode)
    elif scheme in ("http", "https"):
        headers = {}
        headers.update(http_headers or {})
        if "User-Agent" not in headers:
            headers["User-Agent"] = "lance"
        resp = requests.get(uri, auth=http_auth, headers=headers)
        return BytesIO(resp.content)
    else:
        filesystem, path = fs.FileSystem.from_uri(uri)
        return filesystem.open_input_file(path)


def copy(source: Union[str, Path], dest: Union[str, Path]) -> str:
    """Copy a file from source to destination, and return the URI of
    the copied file.

    Parameters
    ----------
    source : str or Path
        The source URI to copy from
    dest : str or Path
        The destination uri or the destination directory. If ``dest`` is
        an URI ends with "/", it represents a directory.

    Return
    ------
    str
        The URI of destination.
    """
    parsed_source = urlparse(source)
    if dest and dest.endswith("/"):
        dest = os.path.join(dest, os.path.basename(parsed_source.path))
    parsed_dest = urlparse(dest)
    logger.debug("Copying %s to %s", source, dest)

    if parsed_dest.scheme == parsed_source.scheme:
        # Direct copy with the same file system
        filesystem, source_path = fs.FileSystem.from_uri(str(source))
        _, dest_path = fs.FileSystem.from_uri(str(dest))
        filesystem.copy(source_path, dest_path)
        return dest

    source_fs, source_path = fs.FileSystem.from_uri(str(source))
    dest_fs, dest_path = fs.FileSystem.from_uri(str(dest))
    with source_fs.open_input_file(source_path) as in_stream:
        with dest_fs.open_output_stream(dest_path) as out_stream:
            shutil.copyfileobj(in_stream, out_stream)
    return dest


def read_file(uri: str) -> bytes:
    """
    Download a single file from given uri

    Parameters
    ----------
    uri: str
        The uri from where the file should be downloaded
    """
    if not urlparse(uri).scheme:
        uri = pathlib.Path(uri).expanduser().absolute()
    _fs, key = fs.FileSystem.from_uri(uri)
    return _fs.open_input_file(key).read()


def download_uris(uris, num_workers=mp.cpu_count() - 1, func=read_file):
    """
    Download an iterable of uris in parallel

    Parameters
    ----------
    uris: list-like of str
        Uris to be downloaded
    num_workers: int, default multiprocessing.cpu_count() - 1
        The number of workers to spin up in the Pool
    func: str -> bytes, default read_file
        Can be overridden to provide custom behavior or pre/post-processing
    """
    pool = mp.Pool(num_workers)
    data = pool.map(func, uris)
    return data
