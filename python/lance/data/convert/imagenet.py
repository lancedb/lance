#  Copyright (c) 2022. Lance Developers
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


import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset

import lance
from lance.io import download_uris
from lance.types.image import Image, ImageArray, ImageBinaryType

__all__ = ["convert_imagenet_1k"]

_SPLITS = ["train", "validation", "test"]


def _read_labels(base_uri: str) -> tuple[dict, pd.CategoricalDtype]:
    LABEL_FILE = "LOC_synset_mapping.txt"
    labels = {}
    label_names = OrderedDict()
    label_names["unknown"] = True
    with open(os.path.join(base_uri, LABEL_FILE)) as fobj:
        for idx, line in enumerate(fobj.readlines()):
            tag, label = line.strip().split(maxsplit=1)
            labels[tag] = label
            if label not in label_names:
                label_names[label] = True
    dtype = pd.CategoricalDtype(label_names.keys(), ordered=True)
    return labels, dtype


def _generate_image_uri(df: pd.DataFrame) -> pd.DataFrame:
    def generate_uri(image_id, split):
        if split == "train":
            image_id = f"{image_id.split('_')[0]}/{image_id}"
        return (
            f"s3://eto-public/datasets/imagenet_1k/Data/CLS-LOC/{split}/{image_id}.JPEG"
        )

    df["image_uri"] = df.apply(
        lambda row: generate_uri(row["ImageId"], row["split"]), axis=1
    )


def _embedded_images(base_dir: str, df: pd.DataFrame) -> pd.DataFrame:
    """Include embedded images."""

    def gen_image_uris(row):
        if row.split == "train":
            image_id = row.ImageId
            image_id = f"{image_id.split('_')[0]}/{image_id}.JPEG"
        else:
            image_id = f"{row.ImageId}.JPEG"
        image_uri = os.path.join(base_dir, "Data", "CLS-LOC", row.split, image_id)
        return image_uri

    image_uris = df.apply(lambda row: gen_image_uris(row), axis=1)
    images = download_uris(image_uris)
    return lance.types.ImageArray.from_pandas(images)


def convert_imagenet_1k(
    uri: str | Path,
    out: str | Path,
    group_size: int,
    limit: Optional[int] = None,
    split: Optional[str | list[str]] = None,
) -> None:
    """Convert ImageNet 1K dataset to lance format

    Parameters
    ----------
    uri : str or Path
        Input Dataset URI
    out : str or Path
        Output URI
    group_size : int
        The size of each row group.
    limit : int, optional
        Limit number of records to generate, useful for testing.

    It expects the input directory has the following directories:

      - Data
      - Annotations
      - ImageSet
      - LOC_val_solution.csv
      - LOC_train_solution.csv

    You can obtain the dataset from Kaggle:

        kaggle competitions download -c imagenet-object-localization-challenge

    """
    labels_map, label_dtype = _read_labels(uri)
    IMAGESET_FILES = {
        "train": "LOC_train_solution.csv",
        "val": "LOC_val_solution.csv",
        "test": "ImageSets/CLS-LOC/test.txt",
    }
    if split is None:
        split = ["train", "val", "test"]
    if isinstance(split, str):
        split = [split]

    dfs = []
    for s in split:
        if s == "test":
            split_df = pd.read_csv(
                os.path.join(uri, IMAGESET_FILES[s]),
                sep=" ",
                header=None,
            )
            split_df = split_df.rename(columns={0: "ImageId"}).drop(columns=[1])
        else:
            split_df = pd.read_csv(os.path.join(uri, IMAGESET_FILES[s]))
            split_df["class"] = split_df["PredictionString"].apply(
                lambda s: labels_map[s.split()[0]]
            )
            split_df = split_df.drop(columns=["PredictionString"])
        split_df["split"] = s
        dfs.append(split_df)
    df = pd.concat(dfs)
    df["split"] = df["split"].astype("category")
    df["class"] = df["class"].astype(label_dtype)
    _generate_image_uri(df)

    if limit:
        frac = limit * 1.0 / len(df)
        print("Limit fraction: ", frac)
        df = df.groupby(["split", "class"]).apply(lambda f: f.sample(frac=frac))

    table = pa.Table.from_pandas(df, preserve_index=False)
    image_arr = _embedded_images(uri, df)
    table = table.append_column(
        pa.field("image", ImageBinaryType()), image_arr
    )


    # batch_reader = pa.RecordBatchReader.from_batches(schema, _record_batch_gen())
    # TODO: Pending the response / fix from arrow to support directly write RecordBatchReader, so that
    # it allows to write larger-than-memory data.
    dataset = pa.dataset.dataset(table)

    lance.write_dataset(dataset, out, max_rows_per_group=group_size)


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("uri")
    @click.argument("out", default="imagenet_1k.lance")
    @click.option(
        "-g",
        "--group-size",
        type=int,
        default=2048,
        help="group size",
        show_default=True,
    )
    @click.option(
        "--limit",
        type=int,
        help="limit the number of examples in total",
        default=None,
        metavar="N",
        show_default=True,
    )
    def main(uri, out, group_size, limit):
        convert_imagenet_1k(uri, out, group_size, limit=limit)

    main()
