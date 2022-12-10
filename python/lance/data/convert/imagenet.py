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


from pathlib import Path

import pyarrow

import lance
import itertools
from typing import Generator
import pyarrow as pa
import pyarrow.dataset
from lance.types.image import Image, ImageArray, ImageBinaryType
import lance.data.huggingface as hf


def _record_batch_gen(batch_size: int = 1024) -> Generator[pa.RecordBatch, None, None]:
    """Generator of RecordBatch."""
    from datasets import load_dataset

    splits = ["train", "validation", "test"]
    for split in splits:
        hg_dataset = load_dataset("imagenet-1k", split=split, streaming=True)
        hg_features = hg_dataset.features
        batch = {"image": [], "label": [], "split": []}
        for example in itertools.islice(hg_dataset, 2000):
            batch["image"].append(Image.create(example["image"]))
            if split == "test":
                batch["label"].append(None)
            else:
                batch["label"].append(example["label"])
            batch["split"].append(splits.index(split))

            if len(batch["image"]) >= batch_size:
                image_arr = ImageArray.from_images(batch["image"])
                label_arr = pyarrow.DictionaryArray.from_arrays(
                    pa.array(batch["label"], type=pa.int16()), hg_features["label"].names
                )
                split_arr = pyarrow.DictionaryArray.from_arrays(pa.array(batch["split"]), splits)
                yield pa.RecordBatch.from_arrays(
                    [image_arr, label_arr, split_arr], ["image", "label", "split"]
                )
                batch = {"image": [], "label": [], "split": []}

        if batch["image"]:
            image_arr = ImageArray.from_images(batch["image"])
            label_arr = pyarrow.DictionaryArray.from_arrays(
                pa.array(batch["label"], type=pa.int16()), hg_features["label"].names)
            split_arr = pyarrow.DictionaryArray.from_arrays(pa.array(batch["split"]), splits)
            yield pa.RecordBatch.from_arrays(
                [image_arr, label_arr, split_arr], ["image", "label", "split"]
            )


def convert_imagenet_1k(out: str | Path) -> lance.FileSystemDataset:
    """Converting ImageNet 1K dataset to lance format

    Parameters
    ----------
    out : str or Path
        Output URI

    Returns
    -------
    An opened lance dataset.
    """
    schema = pa.schema(
        [
            pa.field("image", ImageBinaryType()),
            pa.field("label", pa.dictionary(pa.int16(), pa.utf8())),
            pa.field("split", pa.dictionary(pa.int8(), pa.utf8())),
        ]
    )
    # batch_reader = pa.RecordBatchReader.from_batches(schema, _record_batch_gen())
    dataset = pa.dataset.dataset(list(_record_batch_gen()))
    lance.write_dataset(dataset, out)


if __name__ == "__main__":
    import click


    @click.command()
    @click.argument("out")
    @click.option(
        "--limit",
        type=int,
        help="limit the number of examples in total",
        default=None,
        metavar="N",
        show_default=True,
    )
    def main(out, limit):
        convert_imagenet_1k(out)


    main()
