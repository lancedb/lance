#!/usr/bin/env python
"""Parse Berkley 100K dataset."""

import os
from pathlib import Path
from typing import Union

import click
import lance
import pandas as pd
import pyarrow as pa
import pyarrow.fs

from bench_utils import DatasetConverter


class BDD100kConverter(DatasetConverter):
    def __init__(self, uri_root: Union[str, Path]):
        super().__init__("bdd100k", uri_root)

    def read_metadata(self) -> pd.DataFrame:
        frames = []
        for split in ["train", "val"]:
            annotation = pd.read_json(
                os.path.join(
                    self.uri_root, "bdd100k", "labels", f"bdd100k_labels_images_{split}.json"
                )
            )
            annotation["split"] = split
            annotation["image_uri"] = annotation["name"].map(
                lambda name: os.path.join(self.uri_root, "bdd100k", "images", "100k", split, name))
            frames.append(annotation)

        return pd.concat(frames)

    def image_uris(self, table):
        return table["image_uri"].to_numpy()

    def get_schema(self):
        attributes = pa.struct(
            [
                ("weather", pa.dictionary(pa.uint8(), pa.utf8())),
                ("scene", pa.dictionary(pa.uint8(), pa.utf8())),
                ("timeofday", pa.dictionary(pa.uint8(), pa.utf8())),
            ]
        )
        labels = pa.list_(
            pa.struct(
                [
                    ("category", pa.dictionary(pa.int8(), pa.utf8())),
                    (
                        "attributes",
                        pa.struct(
                            [
                                ("occluded", pa.bool_()),
                                ("truncated", pa.bool_()),
                                (
                                    "trafficLightColor",
                                    pa.dictionary(pa.uint8(), pa.utf8()),
                                ),
                            ]
                        ),
                    ),
                    ("manualShape", pa.bool_()),
                    ("manualAttributes", pa.bool_()),
                    ("box", pa.list_(pa.float32())),
                ]
            )
        )
        schema = pa.schema(
            [
                ("name", pa.utf8()),
                ("image_uri", pa.utf8()),
                ("timestamp", pa.int32()),
                ("split", pa.dictionary(pa.uint8(), pa.utf8())),
                ("attributes", attributes),
                ("labels", labels),
            ]
        )
        return schema


@click.command
@click.option("-u", "--base-uri", type=str, required=True, help="Coco dataset root")
@click.option("-f", "--fmt", type=click.Choice(["lance", "parquet"]), help="Output format (parquet or lance)")
@click.option("-e", "--embedded", type=bool, default=True, help="Embed images")
@click.option(
    "-o",
    "--output-path",
    type=str,
    required=True,
    help="Output path. Default is {base_uri}/coco_links.{fmt}",
)
def main(base_uri, fmt, embedded, output_path):
    converter = BDD100kConverter(base_uri)
    df = converter.read_metadata()
    known_formats = ["lance", "parquet"]
    if fmt is not None:
        assert fmt in known_formats
        fmt = [fmt]
    else:
        fmt = known_formats
    partitioning = ["split"]
    for f in fmt:
        if embedded:
            converter.make_embedded_dataset(df, f, output_path, partitioning=partitioning)
        else:
            return converter.save_df(df, f, output_path, partitioning=partitioning)


if __name__ == "__main__":
    main()
