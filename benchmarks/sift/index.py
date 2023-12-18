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
import platform
import time
from subprocess import check_output

import lance
import numpy as np
import pyarrow as pa


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("uri", help="lance path", metavar="FILE")
    parser.add_argument("-t", "--index-type", choices=["ivf_pq", "diskann"], default="ivf_pq")
    parser.add_argument("-m", "--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument(
        "-c",
        "--column-name",
        type=str,
        metavar="NAME",
        default="vector",
        help="Name of the vector column",
    )
    parser.add_argument(
        "-i",
        "--ivf-partitions",
        type=int,
        default=256,
        help="Number of IVF partitions",
        metavar="NUM",
    )
    parser.add_argument(
        "-p",
        "--pq-subvectors",
        type=int,
        default=16,
        help="Number of subvectors for product quantization",
        metavar="NUM",
    )
    args = parser.parse_args()

    dataset = lance.dataset(args.uri)
    start = time.time()
    dataset = dataset.create_index(
        args.column_name,
        index_type=args.index_type,
        metric=args.metric,
        num_partitions=args.ivf_partitions,  # IVF
        num_sub_vectors=args.pq_subvectors,
        replace=True
    )  # PQ
    end = time.time()

    GIT_COMMIT = (
        check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    )
    bench_results = {
        "commit": GIT_COMMIT,
        "name": "index_creation",
        "platform": platform.platform(),
        "arch": platform.machine(),
        "params": {
            "type": args.index_type,
        },
        "duration": end - start,
    }
    print(bench_results)

    return dataset


if __name__ == "__main__":
    main()
