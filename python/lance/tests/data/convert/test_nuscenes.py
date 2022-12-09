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
import numpy as np
import pyarrow as pa
import requests
import tarfile

BASE_URI_FIXTURES = "https://eto-public.s3.us-west-2.amazonaws.com"
URI_FIXTURES = "/tests/datasets/nuimages-v1.0-mini-test-fixtures.tgz"
LOCAL_ZIP_PATH = "/tmp/nuimages-v1.0-mini-test-fixtures.tgz"
LOCAL_FIXTURES_PATH = "/tmp"

from lance.data.convert.nuscenes import NuscenesConverter

def prep_test_fixtures():
    r = requests.get(f"{BASE_URI_FIXTURES}{URI_FIXTURES}", stream=True)
    if r.status_code == 200:
        with open(LOCAL_ZIP_PATH, 'wb') as f:
            f.write(r.raw.read())

def unpack_fixtures():
    if tarfile.is_tarfile(LOCAL_ZIP_PATH):
        with tarfile.open(LOCAL_ZIP_PATH) as f:
            f.extractall(path=LOCAL_FIXTURES_PATH)

def test_nuscenes_dataset_converter():
    prep_test_fixtures()
    unpack_fixtures()
    c = NuscenesConverter(
        uri_root=f"{LOCAL_FIXTURES_PATH}/nuimages-v1.0-mini-test-fixtures",
        dataset_verson="v1.0-mini"
    )
    nuscenes_df = c.instances_to_df()
    assert nuscenes_df.shape[0] == 2
    c.write_dataset(nuscenes_df, LOCAL_FIXTURES_PATH, "lance")