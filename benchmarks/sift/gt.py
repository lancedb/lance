#!/usr/bin/env python3
#
#  Copyright (c) 2024. Lance Developers
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

"""Generate ground truth from a dataset"""

import argparse
from pathlib import Path
import shutil

import lance
import numpy as np
import pyarrow as pa

from lance.torch.bench_utils import infer_vector_column, ground_truth


def generate_gt(args):
    ds = lance.dataset(args.uri)
    print(ds.schema)

    col = args.col or infer_vector_column(ds)
    if col is None:
        raise ValueError(
            "Can not infer vector column, please specifiy the column explicitly"
        )

    samples = ds.sample(args.samples, columns=[col])[col]
    queries = np.stack(samples.to_numpy())

    gt_rows = (
        ground_truth(
            ds, col, queries, metric_type=args.metric, k=args.k, batch_size=args.batch
        )
        .cpu()
        .numpy()
    )

    rows_col = pa.FixedShapeTensorArray.from_numpy_ndarray(gt_rows)

    query_col = pa.FixedShapeTensorArray.from_numpy_ndarray(queries)

    gt_table = pa.Table.from_arrays(
        [query_col, rows_col], names=["query", "ground_truth"]
    )
    if Path(args.out).exists():
        shutil.rmtree(args.out)
    lance.write_dataset(gt_table, args.out)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("uri", metavar="URI", help="dataset uri")
    parser.add_argument(
        "-o", "--out", help="output directory", default="ground_truth.lance"
    )
    parser.add_argument("-k", help="top K results", default=1000, type=int)
    parser.add_argument("--col", help="vector column", default=None, metavar="NAME")
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        metavar="NUM",
        default=50,
        help="Number of sample queries",
    )
    parser.add_argument(
        "-m", "--metric", choices=["l2", "cosine"], default="l2", help="metric type"
    )
    parser.add_argument(
        "--batch", help="batch size", metavar="NUM", default=1024 * 80, type=int
    )
    args = parser.parse_args()

    generate_gt(args)


if __name__ == "__main__":
    main()
