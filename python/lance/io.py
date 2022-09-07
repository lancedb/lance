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
from io import BytesIO
from pathlib import Path
from typing import Optional, IO
from urllib.parse import urlparse

from pyarrow import fs
import requests
from requests.auth import AuthBase

import lance

USER_AGENT = f"User-Agent: Lance/{lance.__version__} (contact@eto.ai)"


def open_uri(
        uri: str | Path,
        mode: str = "rb",
        http_auth: Optional[AuthBase | tuple[str, str]] = None,
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
