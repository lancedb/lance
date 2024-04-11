# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import pyarrow as pa


def train_hnsw():
    data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    arrays = [pa.array(sublist, type=pa.float32()) for sublist in data]
    vectors = pa.FixedSizeListArray.from_arrays(arrays, size=2)
    hnsw = lance.util.HNSW.build(vectors)
    hnsw.to_lance_file("hnsw.idx")
