#!/usr/bin/env python3
#
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

import argparse
import time
from typing import Optional

import duckdb
import lance
import numpy as np
import pandas as pd


def recall(actual_sorted: np.ndarray, results: np.ndarray):
    """
    Recall-at-k
    """
    len = results.shape[1]
    recall_at_k = np.array([np.sum([1 if id in results[i, :] else 0 for id in row]) * 1.0 / len
                            for i, row in enumerate(actual_sorted)])
    return (recall_at_k.mean(), recall_at_k.std(), recall_at_k)


def l2_argsort(mat, q):
    """
    Parameters
    ----------
    mat: ndarray
        shape is (n, d) where n is number of vectors and d is number of dims
    q: ndarray
        shape is d, this is the query vector
    """
    return np.argsort(((mat - q) ** 2).sum(axis=1))


def test_dataset(
    uri, nsamples=100, k=10, nprobes=1, refine_factor: Optional[int] = None
):
    dataset = lance.dataset(uri)
    tbl = dataset.to_table()
    v = tbl["vector"].combine_chunks()
    all_vectors = v.values.to_numpy().reshape(len(tbl), v.type.list_size)
    # all_vectors = all_vectors / np.linalg.norm(all_vectors, axis=1)[:, None]

    query_vectors = duckdb.query(
        f"SELECT vector FROM tbl USING SAMPLE {nsamples}"
    ).to_df()
    query_vectors = np.array([np.array(x) for x in query_vectors.vector.values])
    # query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1)[:, None]

    actual_sorted = []
    results = []

    tot = 0
    for i in range(nsamples):
        q = query_vectors[i, :]
        actual_sorted.append(l2_argsort(all_vectors, q)[:k])
        start = time.time()
        rs = dataset.to_table(
            nearest={
                "column": "vector",
                "q": q,
                "k": k,
                "nprobes": nprobes,
                "refine_factor": refine_factor,
            }
        )
        end = time.time()
        tot += end - start
        results.append(rs["id"].combine_chunks().to_numpy())
    avg_latency = tot / nsamples
    return recall(np.array(actual_sorted), np.array(results)), avg_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("uri", help="Dataset URI", metavar="URI")
    parser.add_argument("out", help="Output file", metavar="FILE")
    parser.add_argument("-i", "--ivf-partitions", type=int, metavar="N")
    parser.add_argument("-s", "--samples", default=10, type=int, metavar="N")
    parser.add_argument("-k", "--top_k", default=10, type=int, metavar="N")
    args = parser.parse_args()

    columns = [
        "ivf",
        "pq",
        "nprobes",
        "nsamples",
        "topk",
        "refine_factor",
        "recall@k",
        "mean_time_sec",
    ]
    ivf = []
    pq = []
    nprobes = []
    nsamples = []
    topk = []
    refine_factor = []
    recall_at_k = []
    mean_time = []
    for n in [1, 10, 50, 100]:
        for rf in [None, 1, 5, 10]:
            recalls, times = test_dataset(
                args.uri,
                nsamples=args.samples,
                k=args.top_k,
                nprobes=n,
                refine_factor=rf,
            )
            ivf.append(args.ivf_partitions)
            pq.append(16)
            nprobes.append(n)
            nsamples.append(args.samples)
            topk.append(args.top_k)
            refine_factor.append(rf)
            recall_at_k.append(recalls[0])
            mean_time.append(times)
            print(
                f"nprobes: {n}, refine={rf}, recall@{args.top_k}={recalls[0]:0.3f}, mean(s)={times}"
            )

    df = pd.DataFrame(
        {
            k: v
            for k, v in zip(
                columns,
                [
                    ivf,
                    pq,
                    nprobes,
                    nsamples,
                    topk,
                    refine_factor,
                    recall_at_k,
                    mean_time,
                ],
            )
        }
    )
    df.to_csv(args.out, index=False)
