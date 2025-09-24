# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import threading

import lance
import numpy as np
import pyarrow as pa
import pytest


def test_gil_release(tmp_path):
    dataset_path = tmp_path / "test_data"
    data = [
        {"id": i, "vector": np.random.rand(128).astype(np.float32)} for i in range(1000)
    ]
    table = pa.Table.from_pylist(data)
    lance.write_dataset(table, dataset_path)

    ds = lance.dataset(dataset_path)
    fragment_ids = [frag.fragment_id for frag in ds.get_fragments()]

    def process_fragment(fragment_id: int):
        try:
            frag_ds = lance.dataset(dataset_path)
            fragment = frag_ds.get_fragment(fragment_id)
            scanner = frag_ds.scanner(fragments=[fragment])
            for _ in scanner.to_batches():
                pass
        except Exception as e:
            pytest.fail(f"Fragment {fragment_id} failed: {str(e)}")

    threads = [
        threading.Thread(target=process_fragment, args=(fid,), daemon=True)
        for fid in fragment_ids
    ]

    for t in threads:
        t.start()

    assert all(t.is_alive() for t in threads), "Threads should still be running"
