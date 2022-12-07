#!/usr/bin/env python

"""Parse coco dataset"""

import json
import os
import sys
from collections import defaultdict
from typing import Iterable, List

import numpy as np
import pandas as pd
import pyarrow as pa

from lance.types import ImageType
import click

sys.path.append("..")

from converter import DatasetConverter, PUBLIC_URI_ROOT

FORMATS = ["lance", "parquet"]


class CocoConverter(DatasetConverter):
    def __init__(
        self,
        uri_root: str,
        splits: List[str] = ["train", "val", "test"],
        version: str = "2017",
    ):
        super(CocoConverter, self).__init__("coco", uri_root)

        self.splits = splits
        self.version = version

    def _get_instances_json(self, split):
        """
        Read the annotations json data
        """
        uri = os.path.join(
            self.uri_root, "annotations", f"instances_{split}{self.version}.json"
        )
        fs, path = pa.fs.FileSystem.from_uri(uri)
        with fs.open_input_file(path) as fobj:
            return json.load(fobj)

    def _instances_to_df(self, split, instances_json):
        """
        Read instances and join to images
        """
        annotations_df = self._ann_to_df(instances_json)
        anno_df = (
            pd.DataFrame(
                {
                    "image_id": annotations_df.image_id,
                    "annotations": annotations_df.drop(
                        columns=["image_id"], axis=1
                    ).to_dict(orient="records"),
                }
            )
            .groupby("image_id")
            .agg(_aggregate_annotations)
        ).reset_index()
        images_df = pd.DataFrame(instances_json["images"]).rename(
            {"id": "image_id"}, axis=1
        )
        images_df["split"] = split
        images_df["image_uri"] = images_df["file_name"].apply(
            lambda fname: os.path.join(
                PUBLIC_URI_ROOT, "coco", f"{split}{self.version}", fname
            )
        )
        # TODO join images_df.license to instances_json['license']
        return images_df.merge(anno_df, on="image_id")

    def _ann_to_df(self, instances_json):
        """
        Read annotations and map to string category names

        Returns
        -------
        ann_df: pd.DataFrame
        """
        df = pd.DataFrame(instances_json["annotations"])
        cat_df = pd.DataFrame(instances_json["categories"])
        df["segmentation"] = df.segmentation.apply(_convert_segmentation)
        df["iscrowd"] = df.iscrowd.astype("bool")
        # Convert coco dataset bounding box [x,y,width,height] to [x0,y0,x1,y1] format.
        df["bbox"] = df.bbox.apply(
            lambda arr: [arr[0], arr[1], arr[0] + arr[2], arr[1] + arr[3]]
        )
        category_df = cat_df.rename({"id": "category_id"}, axis=1)
        return df.merge(category_df, on="category_id")

    def read_metadata(self, num_rows: int = 0) -> pd.DataFrame:
        def read_split(split):
            json_data = self._get_instances_json(split)
            return self._instances_to_df(split, json_data)

        split_dfs = []
        for split in self.splits:
            if split == "test":
                test_images = self._get_test_images("test")
                split_dfs.append(
                    pd.DataFrame({"image_uri": test_images, "split": split})
                )
            else:
                split_dfs.append(read_split(split))

        df = self._concat_frames(split_dfs, num_rows)
        df["date_captured"] = pd.to_datetime(df.date_captured)
        return df

    def _concat_frames(
        self, frames: Iterable[pd.DataFrame], num_rows: int
    ) -> pd.DataFrame:
        if num_rows > 0:
            sizes = np.array([len(df) for df in frames])
            rows = np.round(sizes / sizes.sum() * num_rows).astype(int)
            return pd.concat([df.sample(n) for df, n in zip(frames, rows)])
        return pd.concat(frames)

    def _get_test_images(self, dirname: str = "test"):
        uri = os.path.join(self.uri_root, f"{dirname}{self.version}")
        fs, path = pa.fs.FileSystem.from_uri(uri)
        public_uri = os.path.join(PUBLIC_URI_ROOT, "coco", f"{dirname}{self.version}")
        return [
            os.path.join(public_uri, file.base_name)
            for file in fs.get_file_info(pa.fs.FileSelector(path, recursive=True))
        ]

    def image_uris(self, table):
        prefix = os.path.join(PUBLIC_URI_ROOT, "coco/")
        uris = np.array(
            [
                os.path.join(self.uri_root, image_uri[len(prefix) :])
                for image_uri in table["image_uri"].to_numpy()
            ]
        )
        return uris

    @property
    def schema(self):
        names = [
            "license",
            "file_name",
            "coco_url",
            "height",
            "width",
            "date_captured",
            "flickr_url",
            "image_id",
            "split",
            "image_uri",
            "annotations",
        ]
        types = [
            pa.int64(),
            pa.utf8(),
            ImageType.from_storage(pa.utf8()),
            pa.int16(),
            pa.int16(),
            pa.timestamp("ns"),
            ImageType.from_storage(pa.utf8()),
            pa.int64(),
            pa.dictionary(pa.int8(), pa.utf8()),
            ImageType.from_storage(pa.utf8()),
            self._ann_schema(),
        ]
        return pa.schema([pa.field(name, dtype) for name, dtype in zip(names, types)])

    @staticmethod
    def _ann_schema():
        segmentation_type = pa.struct(
            [
                pa.field("counts", pa.list_(pa.int32())),
                pa.field("polygon", pa.list_(pa.list_(pa.float32()))),
                pa.field("size", pa.list_(pa.int32())),
            ]
        )
        names = [
            "segmentation",
            "area",
            "iscrowd",
            "bbox",
            "category_id",
            "id",
            "supercategory",
            "name",
        ]
        types = [
            segmentation_type,
            pa.float64(),
            pa.bool_(),
            pa.list_(pa.float32(), 4),
            pa.int16(),
            pa.int64(),
            pa.string(),  # TODO https://github.com/duckdb/duckdb/issues/4812
            pa.string(),  # TODO https://github.com/duckdb/duckdb/issues/4812
        ]
        schema = pa.struct(
            [pa.field(name, pa.list_(dtype)) for name, dtype in zip(names, types)]
        )
        return schema


def _convert_segmentation(s):
    if isinstance(s, list):
        return {"polygon": s}
    return s


def _aggregate_annotations(annotations):
    ret = defaultdict(list)
    for ann in annotations:
        for k, v in ann.items():
            ret[k].append(v)
    return ret


@click.command()
@click.argument("uri")
@click.option(
    "-f",
    "--fmt",
    type=click.Choice(FORMATS),
    default="lance",
    help="Output format (parquet or lance)",
)
@click.option("-e", "--embedded", type=bool, default=True, help="Embed images")
@click.option(
    "-g",
    "--group-size",
    type=int,
    default=1024,
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
    "-o",
    "--output-path",
    type=str,
    help="Output path. Default is under the base_uri",
)
@click.option(
    "--num-rows",
    type=int,
    default=0,
    help="Max rows in the dataset (0 means this is ignored)",
)
@click.option(
    "--split",
    type=str,
    help="comma separated split strings",
    default="train,val,test",
    show_default=True,
)
def main(
    uri,
    fmt,
    embedded,
    output_path,
    group_size: int,
    max_rows_per_file: int,
    num_rows: int,
    split: str,
):
    converter = CocoConverter(uri, splits=split.split(","))
    df = converter.read_metadata(num_rows=num_rows)
    if fmt is not None:
        assert fmt in FORMATS
        fmt = [fmt]
    else:
        fmt = FORMATS

    for f in fmt:
        if f == "lance":
            kwargs = {
                "existing_data_behavior": "overwrite_or_ignore",
                "max_rows_per_group": group_size,
                "max_rows_per_file": max_rows_per_file,
            }
        elif f == "parquet":
            kwargs = {
                "row_group_size": group_size,
            }
        if embedded:
            converter.make_embedded_dataset(df, f, output_path, **kwargs)
        else:
            return converter.save_df(df, f, output_path, **kwargs)


if __name__ == "__main__":
    main()
