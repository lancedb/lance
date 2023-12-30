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

import lance
import numpy as np
import pandas as pd
from lance.torch.bench_utils import ground_truth as gt_func, recall


def get_query_vectors(uri, nsamples=1000, normalize=False):
    """Get the query vectors as a 2d numpy array

    Parameters
    ----------
    uri: str
        Sample the vector column from this lance datasets as query vectors.
    nsamples: int
        Number of samples to read from the dataset
    """
    tbl = lance.dataset(uri)
    query_vectors = np.stack(
        tbl.sample(nsamples, columns=["vector"])["vector"].to_numpy()
    )
    if normalize:
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1)[:, None]
    return query_vectors.astype(np.float32)


def test_dataset(
    uri,
    query_vectors,
    ground_truth,
    k=10,
    nprobes=1,
    refine_factor: Optional[int] = None,
):
    """
    Compute the recall for a given query configuration

    Parameters
    ----------
    uri: str
        Dataset URI for the database vectors
    query_vectors: ndarray
        Query vectors
    ground_truth: ndarray
        Ground truth computed by brute force KNN
    k: int
        Number of nearest neighbors to search for
    nprobes: int
        Number of probes during search
    refine_factor: int
        Refine factor during search
    """
    dataset = lance.dataset(uri)
    actual_sorted = []
    results = []

    tot = 0
    # call ANN for each ground truth set
    for i in range(ground_truth.shape[0]):
        q = query_vectors[i, :]
        actual_sorted.append(ground_truth[i, :k])
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
    avg_latency = tot / ground_truth.shape[0]
    recalls = recall(np.array(actual_sorted), np.array(results))
    return recalls.mean(), avg_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("uri", help="Dataset URI", metavar="URI")
    parser.add_argument("out", help="Output file", metavar="FILE")
    parser.add_argument("-i", "--ivf-partitions", type=int, metavar="N")
    parser.add_argument("-p", "--pq", type=int, metavar="N")
    parser.add_argument("-s", "--samples", default=1000, type=int, metavar="N")
    parser.add_argument(
        "-q",
        "--queries",
        type=str,
        default=None,
        help="lance dataset uri containing query vectors",
        metavar="URI",
    )
    parser.add_argument("-k", "--top_k", default=10, type=int, metavar="N")
    parser.add_argument("-n", "--normalize", action="store_true")
    args = parser.parse_args()

    columns = [
        "ivf",
        "pq",
        "nprobes",
        "nsamples",
        "queries",
        "topk",
        "refine_factor",
        "recall@k",
        "mean_time_sec",
    ]
    ivf = []
    pq = []
    nprobes = []
    nsamples = []
    queries = []
    topk = []
    refine_factor = []
    recall_at_k = []
    mean_time = []
    query_vectors = get_query_vectors(
        args.queries, nsamples=args.samples, normalize=args.normalize
    )
    ds = lance.dataset(args.uri)
    tbl = ds.to_table()
    v = tbl["vector"].combine_chunks()
    all_vectors = v.values.to_numpy().reshape(len(tbl), v.type.list_size)
    print("Computing ground truth")
    start = time.time()
    gt = (
        gt_func(ds, "vector", query_vectors.astype(np.float32), k=args.top_k)
        .cpu()
        .numpy()
    )
    print(f"Get ground truth in: {time.time() - start:0.3f}s")
    print("Starting benchmarks")
    for n in [1, 10, 25, 50, 75, 100]:
        for rf in [None, 1, 10, 20, 30, 40, 50]:
            recalls, times = test_dataset(
                args.uri,
                query_vectors,
                gt,
                k=args.top_k,
                nprobes=n,
                refine_factor=rf,
            )
            ivf.append(args.ivf_partitions)
            pq.append(args.pq)
            nprobes.append(n)
            nsamples.append(args.samples)
            queries.append(args.queries)
            topk.append(args.top_k)
            refine_factor.append(rf)
            recall_at_k.append(recalls)
            mean_time.append(times)
            print(
                f"nprobes: {n}, refine={rf}, recall@{args.top_k}={recalls:0.3f}, mean(s)={times}"
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
                    queries,
                    topk,
                    refine_factor,
                    recall_at_k,
                    mean_time,
                ],
            )
        }
    )
    df.to_csv(args.out, index=False)
