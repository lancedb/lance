#!/usr/bin/env python3
#
#
# Data generation for KShivendu/dbpedia-entities-openai-1M dataset on huggingface.

from argparse import ArgumentParser

from datasets import load_dataset, DownloadConfig
import pyarrow as pa
import lance


schema = pa.schema(
    [
        pa.field("_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("text", pa.string()),
        pa.field("openai", pa.list_(pa.float32(), 1536)),
    ]
)


def convert_dataset():
    for batch in load_dataset(
        "KShivendu/dbpedia-entities-openai-1M",
        download_config=DownloadConfig(num_proc=8, resume_download=True),
    ).data.to_batches():
        print(batch)
        break


def datagen(args):
    lance.write_dataset(
        convert_dataset(),
        args.output,
        max_rows_per_group=10240,
        max_rows_per_file=204800,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="dbpedia.lance")
    args = parser.parse_args()

    datagen(args)


if __name__ == "__main__":
    main()
