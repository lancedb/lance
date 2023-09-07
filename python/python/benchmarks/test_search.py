import shutil
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

N_DIMS = 128


@pytest.fixture(scope="module")
def test_dataset(data_dir: Path) -> lance.LanceDataset:
    tmp_path = data_dir / "search_dataset"
    num_rows = 100_000

    if tmp_path.exists():
        try:
            dataset = lance.LanceDataset(tmp_path)
        except Exception:
            pass
        else:
            return dataset

    # clear any old data there
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    values = pc.random(num_rows * N_DIMS).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, N_DIMS)
    table = pa.table({"vector": vectors})

    dataset = lance.write_dataset(table, tmp_path)

    dataset.create_index(
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=32,
        num_sub_vectors=16,
        num_bits=8,
    )

    return dataset


@pytest.mark.benchmark(group="query_ann")
def test_knn_search(test_dataset, benchmark):
    q = pc.random(N_DIMS).cast(pa.float32())
    result = benchmark(
        test_dataset.to_table,
        nearest=dict(
            column="vector",
            q=q,
            k=100,
            nprobes=10,
            use_index=False,
        ),
    )
    assert result.num_rows > 0


@pytest.mark.benchmark(group="query_ann")
def test_flat_index_search(test_dataset, benchmark):
    q = pc.random(N_DIMS).cast(pa.float32())
    result = benchmark(
        test_dataset.to_table,
        nearest=dict(
            column="vector",
            q=q,
            k=100,
            nprobes=10,
        ),
    )
    assert result.num_rows > 0


@pytest.mark.benchmark(group="query_ann")
def test_ivf_pq_index_search(test_dataset, benchmark):
    q = pc.random(N_DIMS).cast(pa.float32())
    result = benchmark(
        test_dataset.to_table,
        nearest=dict(
            column="vector",
            q=q,
            k=100,
            nprobes=10,
            refine_factor=2,
        ),
    )
    assert result.num_rows > 0
