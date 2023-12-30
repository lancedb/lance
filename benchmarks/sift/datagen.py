#!/usr/bin/env python3
#
import argparse
import struct

import lance
import numpy as np
import pyarrow as pa
from ml_dtypes import bfloat16


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data", help="input dataset (sift_base.fvecs)", metavar="FILE")
    parser.add_argument("out", help="output directory")
    parser.add_argument(
        "-g",
        "--max-rows-per-group",
        type=int,
        default=8192,
        help="Max number of rows per group",
        metavar="NUM",
    )
    parser.add_argument(
        "-m",
        "--max-rows-per-file",
        type=int,
        default=1024 * 1024,
        help="Max number of records per data file",
        metavar="NUM",
    )
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=128,
        help="Dimension of the vector",
        metavar="NUM",
    )
    parser.add_argument(
        "-t",
        "--datatype",
        help="data type of the vector column",
        choices=["f32", "f16", "bf16"],
        default="f32",
    )
    parser.add_argument(
        "-n",
        "--num-vectors",
        type=int,
        default=1000000,
        help="Number of vectors to generate",
        metavar="NUM",
    )
    args = parser.parse_args()

    dt = {"f32": pa.float32(), "f16": pa.float16(), "bf16": lance.arrow.BFloat16Type()}[
        args.datatype
    ]
    vector_dt = pa.list_(dt, args.dimension)

    with open(args.data, mode="rb") as fobj:
        buf = fobj.read()
        data = np.array(
            struct.unpack(
                f"<{args.dimension * args.num_vectors}f",
                buf[4 : 4 + 4 * args.num_vectors * args.dimension],
            )
        )

        if args.datatype == "f16":
            data = data.astype(np.float16)
        elif args.datatype == "bf16":
            data = data.astype(bfloat16)
        elif args.datatype == "f32":
            data = data.astype(np.float32)

        schema = pa.schema(
            [
                pa.field("id", pa.uint32(), False),
                pa.field("vector", vector_dt, False),
            ]
        )
        table = pa.Table.from_arrays(
            [
                pa.array(range(args.num_vectors), type=pa.uint32()),
                pa.FixedSizeListArray.from_arrays(
                    pa.array(data, type=dt), list_size=args.dimension
                ),
            ],
            schema=schema,
        )

        lance.write_dataset(
            table,
            args.out,
            max_rows_per_group=args.max_rows_per_group,
            max_rows_per_file=args.max_rows_per_file,
        )


if __name__ == "__main__":
    main()
