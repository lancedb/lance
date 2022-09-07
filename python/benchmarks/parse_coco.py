#!/usr/bin/env python
"""Parse coco dataset"""
import json
import os

import click
import pandas as pd
import pyarrow as pa
from bench_utils import DatasetConverter


class CocoConverter(DatasetConverter):
    def __init__(self, uri_root, version="2017"):
        super(CocoConverter, self).__init__("coco", uri_root)
        self.version = version

    def _get_instances_json(self, split):
        uri = os.path.join(
            self.uri_root, "annotations", f"instances_{split}{self.version}.json"
        )
        fs, path = pa.fs.FileSystem.from_uri(uri)
        with fs.open_input_file(path) as fobj:
            return json.load(fobj)

    def _instances_to_df(self, split, instances_json):
        df = pd.DataFrame(instances_json["annotations"])
        df["segmentation"] = df.segmentation.apply(_convert_segmentation)
        df["iscrowd"] = df.iscrowd.astype("bool")
        category_df = pd.DataFrame(instances_json["categories"]).rename(
            {"id": "category_id"}, axis=1
        )
        annotations_df = df.merge(category_df, on="category_id")
        # annotations_df['iscrowd'] = annotations_df.iscrowd.astype(bool)
        anno_df = (
            pd.DataFrame(
                {
                    "image_id": df.image_id,
                    "annotations": annotations_df.drop(
                        columns=["image_id"], axis=1
                    ).to_dict(orient="records"),
                }
            )
            .groupby("image_id")
            .agg(list)
        ).reset_index()
        images_df = pd.DataFrame(instances_json["images"]).rename(
            {"id": "image_id"}, axis=1
        )
        images_df["split"] = split
        images_df["image_uri"] = images_df["file_name"].apply(
            lambda fname: os.path.join(self.uri_root, f"{split}{self.version}", fname)
        )
        # TODO join images_df.license to instances_json['license']
        return images_df.merge(anno_df, on="image_id")

    def read_metadata(self) -> pd.DataFrame:
        def read_split(split):
            json_data = self._get_instances_json(split)
            return self._instances_to_df(split, json_data)

        df = pd.concat([read_split(split) for split in ["train", "val"]])
        # df['date_captured'] = pd.to_datetime(df.date_captured)  # lance GH#98
        return df

    def image_uris(self, table):
        return table["image_uri"].to_numpy()

    def get_schema(self):
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
            pa.utf8(),
            pa.int64(),
            pa.int64(),
            pa.utf8(),
            pa.utf8(),
            pa.int64(),
            pa.dictionary(pa.uint8(), pa.utf8()),
            pa.utf8(),
            self._ann_schema(),
        ]
        return pa.schema([pa.field(name, dtype) for name, dtype in zip(names, types)])

    @staticmethod
    def _ann_schema():
        segmentation_type = pa.struct(
            [
                pa.field("polygon", pa.list_(pa.list_(pa.int32()))),
                pa.field(
                    "coco_rle",
                    pa.struct(
                        [
                            pa.field("counts", pa.list_(pa.int32())),
                            pa.field("size", pa.list_(pa.int32())),
                        ]
                    ),
                ),
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
            pa.list_(pa.float32()),
            pa.int16(),
            pa.int64(),
            pa.dictionary(pa.uint8(), pa.utf8()),
            pa.dictionary(pa.uint8(), pa.utf8()),
        ]
        schema = pa.list_(
            pa.struct([pa.field(name, dtype) for name, dtype in zip(names, types)])
        )
        return schema


def _convert_segmentation(s):
    if isinstance(s, list):
        return {"polygon": s}
    return s


@click.command
@click.option("-u", "--base-uri", type=str, help="Coco dataset root")
@click.option(
    "-v", "--version", type=str, default="2017", help="Dataset version. Default 2017"
)
@click.option("-f", "--fmt", type=str, help="Output format (parquet or lance)")
@click.option("-e", "--embedded", type=bool, default=True, help="Embed images")
@click.option(
    "-o",
    "--output-path",
    type=str,
    help="Output path. Default is {base_uri}/coco_links.{fmt}",
)
def main(base_uri, version, fmt, embedded, output_path):
    converter = CocoConverter(base_uri, version=version)
    df = converter.read_metadata()
    known_formats = ["lance", "parquet"]
    if fmt is not None:
        assert fmt in known_formats
        fmt = [fmt]
    else:
        fmt = known_formats
    for f in fmt:
        if embedded:
            converter.make_embedded_dataset(df, f, output_path)
        else:
            return converter.save_df(df, f, output_path)


if __name__ == "__main__":
    main()
