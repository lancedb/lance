#  Copyright 2022 Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import pathlib
import platform
import urllib.request
import zipfile
from io import BytesIO

import duckdb


def install_duckdb_extension(version="latest"):
    """
    Install the lance duckdb extension

    Parameters
    ----------
    version: str, default 'latest'
        The version of the extension to install
    """
    _check_duckdb_version()
    if version == "latest":
        version = _get_latest_version("lance", "lance_duckdb")
    uri = _get_uri(version)
    local_path = _download_and_unzip("lance", "lance_duckdb", uri)
    con = duckdb.connect(config={"allow_unsigned_extensions": True})
    con.install_extension(local_path, force_install=True)


def _check_duckdb_version():
    """
    Currently the extension is pre-built for v0.6.0 of duckdb and duckdb
    does not support binary compatibility between patch versions
    """
    CURR_VER = "0.6.0"
    if duckdb.__version__ != CURR_VER:
        msg = (
            f"The lance extension is built against DuckDB version f{CURR_VER} but "
            f"{duckdb.__version__} was found. Please `pip install --force-reinstall duckdb=={CURR_VER}` "
            f"OR build the extension from source against your version of duckdb"
        )
        raise ImportError(msg)


def _get_uri(version):
    uname = platform.uname()
    arch = uname.machine  # arm64, x86_64
    system = uname.system
    system = "osx" if system.lower() == "darwin" else system
    device = _get_device()
    zip_name = f"lance.duckdb_extension.{system}.{arch}.{device}.zip"
    uri_root = "https://eto-public.s3.us-west-2.amazonaws.com/"
    uri = os.path.join(
        uri_root, "artifacts", "lance", "lance_duckdb", version, zip_name
    )
    return uri


def _get_device():
    import torch

    if torch.cuda.is_available():
        return f"cu{torch.version.cuda.replace('.', '')}"
    else:
        return "cpu"


def _get_latest_version(org="lance", ext="lance_duckdb"):
    uri_root = "https://eto-public.s3.us-west-2.amazonaws.com/"
    uri = os.path.join(uri_root, "artifacts", org, ext, "latest")
    with urllib.request.urlopen(uri) as fh:
        return fh.read().decode().strip()


def _download_and_unzip(org, ext, uri):
    with urllib.request.urlopen(uri) as fh:
        zip_file_object = zipfile.ZipFile(BytesIO(fh.read()), "r")
        first_file = zip_file_object.namelist()[0]
        ext_path = f"/tmp/{org}/{ext}"
        os.makedirs(ext_path, exist_ok=True)
        output_path = f"{ext_path}/{pathlib.Path(first_file).name}"
        with zip_file_object.open(first_file) as in_:
            content = in_.read()
            with open(output_path, "wb") as out_:
                out_.write(content)
        return output_path
