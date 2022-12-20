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


import itertools
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset

import lance
from lance.types.image import Image, ImageArray

__all__ = ["convert_imagenet_1k"]

_SPLITS = ["train", "validation", "test"]


def _to_record_batch(batch: Dict, label_names: List[str]) -> pa.RecordBatch:
    """Convert a batch to RecordBatch."""
    image_arr = ImageArray.from_images(batch["image"])
    label_arr = pa.array(batch["label"], type=pa.int16())
    names_arr = pa.DictionaryArray.from_arrays(
        label_arr,
        label_names,
    )
    split_arr = pa.DictionaryArray.from_arrays(
        pa.array(batch["split"], type=pa.int8()), _SPLITS
    )
    id_arr = pa.array(batch["id"], pa.int32())
    return pa.RecordBatch.from_arrays(
        [id_arr, image_arr, label_arr, names_arr, split_arr],
        ["id", "image", "label", "name", "split"],
    )


def _record_batch_gen(
    batch_size: int = 1024, limit: Optional[int] = None
) -> Generator[pa.RecordBatch, None, None]:
    """Generator of RecordBatch."""
    try:
        from datasets import load_dataset
    except ImportError as ie:
        raise ImportError(
            "Please install huggingface dataset via 'pip install datasets'"
        ) from ie

    sample_id = 1
    splits = ["train", "validation", "test"]
    for split in splits:
        hg_dataset = load_dataset("imagenet-1k", split=split, streaming=True)
        hg_features = hg_dataset.features
        batch = {"image": [], "label": [], "split": [], "id": []}
        if limit:
            hg_dataset = itertools.islice(hg_dataset, limit)
        for sample in hg_dataset:
            batch["image"].append(Image.create(sample["image"]))
            batch["label"].append(sample["label"] if split != "test" else None)
            batch["split"].append(splits.index(split))
            batch["id"].append(sample_id)
            sample_id += 1

            if len(batch["image"]) >= batch_size:
                yield _to_record_batch(batch, hg_features["label"].names)
                batch = {"image": [], "label": [], "split": [], "id": []}

        if batch["image"]:
            yield _to_record_batch(batch, hg_features["label"].names)


def _read_labels(base_uri: str) -> tuple[dict, pd.CategoricalDtype]:
    LABEL_FILE = "LOC_synset_mapping.txt"
    labels = {}
    label_names = ["unknown"]
    with open(os.path.join(base_uri, LABEL_FILE)) as fobj:
        for idx, line in enumerate(fobj.readlines()):
            tag, label = line.strip().split(maxsplit=1)
            labels[tag] = (idx + 1, label)
            label_names.append(label)
    print(sorted(label_names))
    for i in range(len(label_names) - 1):
        if label_names[i] == label_names[i + 1]:
            print(label_names[i])
            import sys
            sys.exit(1)
    dtype = pd.CategoricalDtype(label_names, ordered=True)
    return labels, dtype


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
    labels_map, dtype = _read_labels(uri)
    print(labels_map, dtype)
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
        else:
            split_df = pd.read_csv(os.path.join(uri, IMAGESET_FILES[s]))
        split_df["split"] = s
        split_df = split_df.rename(columns={0: "filename"}).drop(columns=[1])
        split_df.drop(1)
        dfs.append(split_df)
    df = pd.concat(dfs)
    df["split"] = df["split"].astype("category")

    print(df)

    if limit:
        frac = limit * 1.0 / len(df)
        print(frac)
        df = df.groupby(["split", "class"]).apply(lambda f: f.sample(frac=frac))
    print(df)
    return

    # batch_reader = pa.RecordBatchReader.from_batches(schema, _record_batch_gen())
    # TODO: Pending the response / fix from arrow to support directly write RecordBatchReader, so that
    # it allows to write larger-than-memory data.
    dataset = pa.dataset.dataset(list(_record_batch_gen(limit=limit)))
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
