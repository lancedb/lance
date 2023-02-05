#!/usr/bin/env python3
#

import lance
import pyarrow as pa
import argparse
import numpy as np
import struct
import pyarrow as pa


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
    args = parser.parse_args()

    with open(args.data, mode="rb") as fobj:
        buf = fobj.read()
        data = np.array(struct.unpack("<128000000f", buf[4 : 4 + 4 * 1000000 * 128]))
        data = data.reshape((1000000, 128))
        print(f"Dataset shape: {data.shape}")

        schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), 128))])
        table = pa.Table.from_arrays([data.tolist()], schema=schema)

        lance.write_dataset(
            table,
            args.out,
            max_rows_per_group=args.max_rows_per_group,
            max_rows_per_file=args.max_rows_per_file,
        )


if __name__ == "__main__":
    main()
