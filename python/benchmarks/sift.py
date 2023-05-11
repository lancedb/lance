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
import shutil
import struct
from pathlib import Path
from subprocess import check_output
import time

import lance
import numpy as np
from lance.vector import vec_to_table

GIT_COMMIT = (
    check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
)

def create_dataset(args):
    nvecs = 1000000
    ndims = 128
    with open(Path(args.path) / "sift_base.fvecs", mode="rb") as fobj:
        buf = fobj.read()
        data = np.array(
            struct.unpack("<128000000f", buf[4 : 4 + 4 * nvecs * ndims])
        ).reshape((nvecs, ndims))
        dd = dict(zip(range(nvecs), data))

    table = vec_to_table(dd)
    lance.write_dataset(
        table, args.output, max_rows_per_group=8192, max_rows_per_file=1024 * 1024
    )


def create_index_benchmark(args) -> float:
    ds = lance.dataset(args.output)
    start = time.time()
    ds.create_index("vector", index_type="IVF_PQ", num_partitions=256, num_sub_vectors=16)
    end = time.time()
    return end - start


def run_benchmark():
    pass


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on the sift dataset")
    parser.add_argument("path", help="path of the sift dataset")
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="output directory",
        default="./sift.lance",
    )

    args = parser.parse_args()
    print("Git commit:", GIT_COMMIT)

    shutil.rmtree(args.output, ignore_errors=True)
    create_dataset(args)

    index_time = create_index_benchmark(args)
    index_record = {
        "git_commit": GIT_COMMIT,
        "name": "sift.create_ivfpq",
        "time": index_time,
    }
    print(index_record)


if __name__ == "__main__":
    main()
