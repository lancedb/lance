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
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

N_DIMS = 768


@pytest.fixture(scope="module")
def test_dataset(tmpdir_factory):
    # We are writing to this, so it's not beneficial to cache it in the data_dir.
    tmp_path = Path(tmpdir_factory.mktemp("index_dataset"))
    num_rows = 1_000

    values = pc.random(num_rows * N_DIMS).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, N_DIMS)
    table = pa.table({"vector": vectors})

    dataset = lance.write_dataset(table, tmp_path)

    return dataset


@pytest.mark.benchmark(group="create_index")
def test_create_ivf_pq(test_dataset, benchmark):
    benchmark(
        test_dataset.create_index,
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=8,
        num_sub_vectors=2,
        num_bits=8,
        replace=True,
    )


@pytest.mark.benchmark(group="create_index")
@pytest.mark.cuda
def test_create_ivf_pq_cuda(test_dataset, benchmark):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")

    benchmark(
        test_dataset.create_index,
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=8,
        num_sub_vectors=2,
        num_bits=8,
        accelerator="cuda",
        replace=True,
    )
