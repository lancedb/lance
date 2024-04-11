# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import pyarrow as pa


def test_train_hnsw():
    arrays = pa.array([1, 2, 3, 4], type=pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(arrays, 2)
    hnsw = lance.util.HNSW.build([vectors])
    hnsw.to_lance_file("/tmp/hnsw.idx")


def test_build_sq_storage():
    arrays = pa.array([1, 2, 3, 4], type=pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(arrays, 2)
    hnsw = lance.util.HNSW.build([vectors])
    hnsw.to_lance_file("/tmp/hnsw.idx")

    lance.util.build_sq_storage()
