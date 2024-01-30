#  Copyright 2023 Lance Developers
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

from pathlib import Path

import lance
import pytest

datasets = pytest.importorskip("datasets")


def test_write_hf_dataset(tmp_path: Path):
    hf_ds = datasets.load_dataset(
        "poloclub/diffusiondb", name="2m_first_1k", split="train[:50]"
    )

    ds = lance.write_dataset(hf_ds, tmp_path)
    assert ds.count_rows() == 50

    assert ds.schema == hf_ds.features.arrow_schema
