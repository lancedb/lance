import duckdb
import os
import pathlib
import platform
import torch
import urllib.request
import zipfile


def install_duckdb_extension(version='latest', unsigned=True):
    """
    Install the lance duckdb extension

    Parameters
    ----------
    version: str, default 'latest'
        The version of the extension to install
    unsigned: bool, default True
        Whether to turn on allow_unsigned_extensions in duckdb
    """
    if version == 'latest':
        version = _get_latest_version('lance', 'lance_duckdb')
    uri = _get_uri(version)
    local_path = _download_and_unzip('lance', 'lance_duckdb', uri)
    con = duckdb.connect(config={"allow_unsigned_extensions": unsigned})
    con.install_extension(local_path, force_install=True)


def _get_uri(version):
    uname = platform.uname()
    arch = uname.machine  # arm64, x86_64
    system = uname.system
    device = _get_device()
    zip_name = f'lance.duckdb_extension.{system}.{arch}.{device}.zip'
    uri_root = 'https://eto-public.s3.us-west-2.amazonaws.com/'
    uri = os.path.join(uri_root, 'artifacts', 'lance', 'lance_duckdb',
                       version, zip_name)
    return uri


def _get_device():
    import torch
    if torch.cuda.is_available():
        return f"cu{torch.version.cuda.replace('.', '')}"
    else:
        return 'cpu'


def _get_latest_version(org='lance', ext='lance_duckdb'):
    uri_root = 'https://eto-public.s3.us-west-2.amazonaws.com/'
    uri = os.path.join(uri_root, 'artifacts', org, ext, 'latest')
    filehandle, _ = urllib.request.urlretrieve(uri)
    with open(filehandle, 'r') as fh:
        return fh.read().strip()


def _download_and_unzip(org, ext, uri):
    filehandle, _ = urllib.request.urlretrieve(uri)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    first_file = zip_file_object.namelist()[0]
    ext_path = f'/tmp/{org}/{ext}'
    os.makedirs(ext_path, exist_ok=True)
    output_path = f'{ext_path}/{pathlib.Path(first_file).name}'
    with zip_file_object.open(first_file) as in_:
        content = in_.read()
        with open(output_path, 'wb') as out_:
            out_.write(content)
    return output_path
