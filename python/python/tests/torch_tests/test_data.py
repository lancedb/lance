#  Copyright (c) 2023. Lance Developers
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


import torch
import torchdata.datapipes.iter.util
import numpy as np
import pyarrow as pa
import lance
from lance.torch.data import LanceDataLoader


def test_iter_over_dataset(tmp_path):
    # 10240 of 32-d vectors.
    data = np.random.random(10240 * 32).astype("f")

    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    ids = pa.array(range(0, 10240))
    tbl = pa.Table.from_arrays([ids, fsl], ["ids", "vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance")

    loader = LanceDataLoader(ds, batch_size=256, samples=2048, columns=["ids", "vec"])

    cache_file = tmp_path / "cache.data"
    def _filepath_fn(data):
        return str(cache_file)

    dp = loader.on_disk_cache(filepath_fn=_filepath_fn).end_caching()
    for a in dp:
        print(a)
