#!/usr/bin/env python3

import argparse

import lance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        default="cosine",
        help="The metric type (L2, cosine, dot) to use for indexing",
    )
    parser.add_argument(
        "--num-partitions",
        default=2048,
        type=int,
        help="The number of partitions to use for indexing",
    )
    parser.add_argument(
        "--num-sub-vectors",
        default=96,
        type=int,
        help="The number of sub-vectors to use for indexing",
    )

    args = parser.parse_args()
    ds = lance.dataset("wiki.lance")
    ds.create_index(
        "emb",
        "IVF_PQ",
        # only meant for benchmarking indexing speed
        # this dataset is not meant for recall/latency benchmarking
        metric=args.metric,
        num_partitions=args.num_partitions,
        num_sub_vectors=args.num_sub_vectors,
        replace=True,
    )


if __name__ == "__main__":
    main()
