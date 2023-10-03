#!/usr/bin/env python3
#

import argparse

import lance
import numpy as np


def run_query(
    ds: lance.LanceDataset,
    queries: np.ndarray,
    k: int,
    metric: str,
    *,
    refine_factor: int | None = None,
    use_index: bool = True,
) -> list[list[str]]:
    results = []
    for query in queries:
        tbl = ds.scanner(
            columns=["_id"],
            nearest={
                "column": "openai",
                "q": query,
                "k": k,
                "metric": metric,
                "refine_factor": refine_factor,
                "use_index": use_index,
            },
        ).to_table()
        results.append(tbl["_id"].to_numpy())
    return results


def ground_truth(
    dataset: lance.LanceDataset,
    queries: np.ndarray,
    top_k: int,
    metric: str,
) -> np.ndarray:
    return run_query(dataset, queries, top_k, metric, refine_factor=None, use_index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("uri", help="dataset uri")
    parser.add_argument(
        "-k",
        "--top-k",
        metavar="K",
        type=int,
        default=10,
        help="top k nearest neighbors",
    )
    args = parser.parse_args()

    ds = lance.dataset(args.uri)

    queries = np.random.rand(20, 1536)  # out of distribution
    gt = ground_truth(ds, queries, args.top_k, "cosine")

    for ivf in [32, 128, 256, 1024]:
        for pq in [32, 96, 192]:
            ds.create_index(
                "openai", "IVF_PQ", num_partitions=ivf, num_sub_vectors=pq, replace=True
            )
            for refine in [0, 2, 5, 10]:
                results = run_query(ds, queries, args.top_k, "cosine", refine_factor=refine)
                print(results)
                print(f"IVF{ivf},PQ{pq}: refine={refine}, recall={0.0}")


if __name__ == "__main__":
    main()
