#!/usr/bin/env python3
#
#
# Data generation for KShivendu/dbpedia-entities-openai-1M dataset on huggingface.

import shutil
from argparse import ArgumentParser
from pathlib import Path

import lance
import pyarrow as pa
from datasets import DownloadConfig, load_dataset

schema = pa.schema(
    [
        pa.field("_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("text", pa.string()),
        pa.field("openai", pa.list_(pa.float32(), 1536)),
    ]
)


def to_fixed_size_array(array, dim):
    return pa.FixedSizeListArray.from_arrays(array.values, dim)


def convert_dataset():
    for batch in load_dataset(
        "KShivendu/dbpedia-entities-openai-1M",
        download_config=DownloadConfig(num_proc=8, resume_download=True),
        split="train",
    ).data.to_batches():
        yield pa.RecordBatch.from_arrays(
            [
                batch["_id"],
                batch["title"],
                batch["text"],
                to_fixed_size_array(batch["openai"], 1536),
            ],
            schema=schema,
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="dbpedia.lance")
    parser.add_argument(
        "-g",
        "--max_rows_per_group",
        type=int,
        default=10240,
        metavar="ROWS",
        help="set max rows per group",
    )
    parser.add_argument(
        "-f",
        "--max_rows_per_file",
        type=int,
        default=2048 * 100,
        metavar="ROWS",
        help="set max rows per file",
    )
    args = parser.parse_args()

    if Path(args.output).exists():
        shutil.rmtree(args.output)

    lance.write_dataset(
        convert_dataset(),
        args.output,
        schema=schema,
        max_rows_per_group=args.max_rows_per_group,
        max_rows_per_file=args.max_rows_per_file,
    )


if __name__ == "__main__":
    main()
