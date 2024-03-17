#!/usr/bin/env python3
#

import argparse
import shutil
from pathlib import Path

import numpy as np
import lance
import pyarrow as pa
import torch
from lance.torch.bench_utils import ground_truth


def create_yfcc_10m(args):
    input_dir = args.input
    print(input_dir)


def read_ndarray(filename: str, dtype: np.dtype = np.float32) -> np.ndarray:
    (total, dim) = np.fromfile(filename, dtype=np.uint32, count=2)
    return np.fromfile(filename, dtype=dtype, count=total * dim).reshape(total, dim)


def create_text2image_10m(args):
    input_dir = args.input

    base_dir = Path(input_dir)
    print(base_dir)

    vectors = read_ndarray(base_dir / "base.1B.fbin.crop_nb_10000000")
    print(vectors.shape)

    shutil.rmtree("text2image-10m.lance", ignore_errors=True)
    arr = pa.array(vectors.reshape(-1))
    fsl = pa.FixedSizeListArray.from_arrays(arr, 200)
    ids = pa.array(range(vectors.shape[0]))
    tbl = pa.Table.from_arrays([ids, fsl], ["id", "vector"])

    ds: lance.LanceDataset = lance.write_dataset(
        tbl, "text2image-10m.lance", max_rows_per_group=10240, max_rows_per_file=1000000
    )

    query_path = "text2image-10m-queries.lance"
    queries = read_ndarray(base_dir / "query.public.100K.fbin")
    queries = queries[:1000]

    queries_tensor = torch.from_numpy(queries).cuda()
    gt_row_ids = ground_truth(
        ds, "vector", queries_tensor, metric_type="L2", k=args.k, batch_size=81960
    )

    ids = []
    for row_id in gt_row_ids.cpu().numpy():
        ids.append(ds._take_rows(row_id, columns=["id"]))

    shutil.rmtree(query_path, ignore_errors=True)
    arr = pa.array(queries.reshape(-1))
    fsl = pa.FixedSizeListArray.from_arrays(arr, 200)
    gt = pa.array(np.stack(ids).reshape(-1))
    gt_fsl = pa.FixedSizeListArray.from_arrays(gt, args.k)
    tbl = pa.Table.from_arrays([fsl, gt_fsl], ["vector", "groud_truth"])
    lance.write_dataset(
        tbl, query_path, max_rows_per_group=10240, max_rows_per_file=1000000
    )


def main():
    parser = argparse.ArgumentParser(description="BigANN dataset")
    parser.add_argument("input", type=str, metavar="DIR")
    parser.add_argument(
        "-t",
        "--type",
        choices=["yfcc-10m", "text2image-10m"],
        required=True,
        help="Benchmark type",
    )
    parser.add_argument(
        "-k", "--k", type=int, default=100, help="Top-k of nearest vectors"
    )

    args = parser.parse_args()

    match args.type.lower():
        case "yfcc-10m":
            create_yfcc_10m(args)
        case "text2image-10m":
            create_text2image_10m(args)
        case _:
            raise ValueError(f"Invalid benchmark type: {args.type}")


if __name__ == "__main__":
    main()
