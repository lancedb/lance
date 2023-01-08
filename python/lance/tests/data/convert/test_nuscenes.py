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
import requests
import tarfile

BASE_URI_FIXTURES = "https://eto-public.s3.us-west-2.amazonaws.com"
URI_FIXTURES = "/tests/datasets/nuimages-v1.0-mini-test-fixtures.tgz"
LOCAL_ZIP_PATH = "nuimages-v1.0-mini-test-fixtures.tgz"

from lance.data.convert.nuscenes import NuscenesConverter

def download_fixtures(tmp_path):
    r = requests.get(f"{BASE_URI_FIXTURES}{URI_FIXTURES}", stream=True)
    if r.status_code == 200:
        with open(tmp_path / LOCAL_ZIP_PATH, "wb") as f:
            f.write(r.raw.read())

def unpack_fixtures(tmp_path):
    if tarfile.is_tarfile(tmp_path / LOCAL_ZIP_PATH):
        with tarfile.open(tmp_path / LOCAL_ZIP_PATH) as f:
            f.extractall(path=tmp_path)

def test_nuscenes_dataset_converter(tmp_path):
    download_fixtures(tmp_path)
    unpack_fixtures(tmp_path)
    c = NuscenesConverter(
        uri_root=tmp_path / "nuimages-v1.0-mini-test-fixtures",
        images_root=tmp_path / "nuimages-v1.0-mini-test-fixtures",
        dataset_verson="v1.0-mini"
    )
    nuscenes_df = c.read_metadata()
    assert nuscenes_df.shape[0] == 2
    table = c.to_table(nuscenes_df, to_image=True)
    c.write_dataset(table, fmt="lance", output_path=str(tmp_path / "nuscenes.lance"))
    c.write_dataset(table, fmt="parquet", output_path=str(tmp_path / "nuscenes.parquet"))