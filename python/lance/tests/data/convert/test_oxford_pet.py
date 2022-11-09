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

from lance.data.convert.oxford_pet import OxfordPetConverter


def test_basic(tmp_path):
    num_rows = 2
    c = OxfordPetConverter(
        uri_root="s3://eto-public/datasets/oxford_pet",
        images_root="https://eto-public.s3.us-west-2.amazonaws.com/datasets/oxford_pet/"
    )
    df = c.read_metadata(num_rows)
    c.make_embedded_dataset(df, fmt="lance", output_path=str(tmp_path / "oxford_pet.lance"))
    c.make_embedded_dataset(df, fmt="parquet", output_path=str(tmp_path / "oxford_pet.parquet"))
