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
from torchvision.models._meta import _IMAGENET_CATEGORIES

import lance
from lance.io import download_uris
from lance.types.image import ImageArray, ImageBinaryType

__all__ = ["convert_imagenet_1k"]

_SPLITS = ["train", "val", "test"]


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


def _read_labels(base_uri: str) -> tuple[dict, pd.CategoricalDtype]:
    LABEL_FILE = "LOC_synset_mapping.txt"
    labels = {}
    with open(os.path.join(base_uri, LABEL_FILE)) as fobj:
        for idx, line in enumerate(fobj.readlines()):
            tag, label = line.strip().split(maxsplit=1)
            label = label.strip()
            if label == "maillot, tank suit":  # this is Kaggle label
                label = "maillot tank suit"  # this is Torchvision label
            else:
                label = label.split(",")[0].strip()
            if label == "crane" and label not in set(
                [x["name"] for x in labels.values()]
            ):
                # Fix Kaggle and Torchvision label discrepancy.
                label = "crane bird"
            assert (
                label in _IMAGENET_CATEGORIES
            ), f"{label} not found in torchvision imagenet categories"
            labels[tag] = {"name": label, "label": idx}
    dtype = pd.CategoricalDtype(_IMAGENET_CATEGORIES, ordered=True)
    return labels, dtype


def _generate_image_uri(df: pd.DataFrame) -> pd.DataFrame:
    def generate_uri(image_id, split):
        if split == "train":
            image_id = f"{image_id.split('_')[0]}/{image_id}"
        return f"https://eto-public.s3.amazonaws.com/datasets/imagenet_1k/Data/CLS-LOC/{split}/{image_id}.JPEG"

    df["image_uri"] = df.apply(
        lambda row: generate_uri(row.image_id, row.split), axis=1
    ).astype("image[uri]")


def _embedded_images(base_dir: str, df: pd.DataFrame) -> ImageArray:
    """Include embedded images."""

    def gen_image_uris(row):
        if row.split == "train":
            image_id = row.image_id
            image_id = f"{image_id.split('_')[0]}/{image_id}.JPEG"
        else:
            image_id = f"{row.image_id}.JPEG"
        image_uri = os.path.join(base_dir, "Data", "CLS-LOC", row.split, image_id)
        return image_uri

    image_uris = df.apply(lambda row: gen_image_uris(row), axis=1)
    images = download_uris(image_uris)
    return ImageArray.from_pandas(images)


def read_metadata(uri, split=None):
    labels_map, label_dtype = _read_labels(uri)
    IMAGESET_FILES = {
        "train": "LOC_train_solution.csv",
        "val": "LOC_val_solution.csv",
        "test": "ImageSets/CLS-LOC/test.txt",
    }
    if split is None:
        split = _SPLITS
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
            split_df["label"] = split_df["PredictionString"].apply(
                lambda s: labels_map[s.split()[0]]["label"]
            )
            split_df["name"] = split_df["PredictionString"].apply(
                lambda s: labels_map[s.split()[0]]["name"]
            )
            split_df = split_df.drop(columns=["PredictionString"])
        split_df["split"] = s
        dfs.append(split_df)
    df = pd.concat(dfs)
    df["split"] = df["split"].astype("category")
    df = df.rename(columns={"ImageId": "image_id"})
    return df


def convert_imagenet_1k(
    uri: str | Path,
    out: str | Path,
    group_size: int,
    max_rows_per_file: int = 0,
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
    df = read_metadata(uri, split=split)
    _generate_image_uri(df)

    if limit:
        df = _sample_data(df, limit)
    _write_data(df, uri, out, group_size, max_rows_per_file)


def _sample_data(df, limit):
    frac = limit * 1.0 / len(df)
    print("Limit fraction: ", frac)
    label_grouper = df["label"].fillna(-1)
    df = (
        df.groupby(["split", label_grouper], dropna=False)
        .apply(lambda f: f.sample(frac=frac))
        .reset_index(drop=True)
    )
    return df


def _write_data(frame, uri, out_path, group_size, max_rows_per_file):
    table = pa.Table.from_pandas(
        frame, schema=_arrow_schema(frame), preserve_index=False
    )
    image_arr = _embedded_images(uri, frame)
    table = table.append_column(pa.field("image", ImageBinaryType()), image_arr)
    dataset = pa.dataset.dataset(table)
    lance.write_dataset(
        dataset,
        out_path,
        max_rows_per_group=group_size,
        max_rows_per_file=max_rows_per_file,
    )


def _arrow_schema(df) -> pa.Schema:
    base_schema = pa.Table.from_pandas(df, preserve_index=False).schema
    fields = []
    for f in base_schema:
        if f.name != "label":
            fields.append(f)
        else:
            fields.append(pa.field("label", pa.int16()))
    return pa.schema(fields)


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
        "--max-rows-per-file",
        type=int,
        default=0,
        help="max rows per file",
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
    def main(uri, out, group_size, max_rows_per_file, limit):
        convert_imagenet_1k(
            uri, out, group_size, max_rows_per_file=max_rows_per_file, limit=limit
        )

    main()
