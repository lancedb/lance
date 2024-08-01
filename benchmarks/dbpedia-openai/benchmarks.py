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
    nprobes: int = 10,
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
                "nprobes": nprobes,
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
    return run_query(
        dataset, queries, top_k, metric, refine_factor=None, use_index=False
    )


def compute_recall(gt: np.ndarray, result: np.ndarray) -> float:
    recalls = [
        np.isin(rst, gt_vector).sum() / rst.shape[0]
        for (rst, gt_vector) in zip(result, gt)
    ]
    return np.mean(recalls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("uri", help="dataset uri")
    parser.add_argument(
        "-k",
        "--top-k",
        metavar="K",
        type=int,
        default=100,
        help="top k nearest neighbors",
    )
    parser.add_argument(
        "--ood", action="store_true", help="out of distribution query", default=False
    )
    parser.add_argument(
        "-m",
        "--metric",
        choices=["l2", "cosine"],
        default="cosine",
        help="distance metric type",
    )
    args = parser.parse_args()

    ds = lance.dataset(args.uri)

    # queries = np.random.rand(20, 1536)  # out of distribution
    if args.ood:
        queries = np.random.rand(20, 1536) * 2 - 1  # make [-1, 1] distribubtion.
    else:
        queries = ds.take(
            np.random.randint(0, ds.count_rows(), size=20), columns=["openai"]
        )["openai"].to_numpy()
    gt = ground_truth(ds, queries, args.top_k, args.metric)

    for ivf in [256, 512, 1024]:
        for pq in [32, 96, 192]:
            ds.create_index(
                "openai",
                "IVF_PQ",
                num_partitions=ivf,
                num_sub_vectors=pq,
                replace=True,
                metric=args.metric,
            )
            for refine in [None, 2, 5, 10, 50, 100]:
                results = run_query(
                    ds,
                    queries,
                    args.top_k,
                    args.metric,
                    refine_factor=refine,
                    nprobes=ivf // 10,
                )
                recall = compute_recall(gt, results)
                print(
                    f"IVF{ivf},PQ{pq}: refine={refine}, recall@{args.top_k}={recall:0.2f}"
                )


if __name__ == "__main__":
    main()
