#!/usr/bin/env python

"""Parse coco dataset"""

import json
import os
from collections import defaultdict
import sys

sys.path.append("..")


import click
import pandas as pd
import pyarrow as pa
from bench_utils import DatasetConverter
from lance.types import ImageType

import lance
import lance.types


class CocoConverter(DatasetConverter):
    def __init__(self, uri_root: str, version: str = "2017"):
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
        # Convert coco dataset bounding box [x,y,width,height] to [x0,y0,x1,y1] format.
        df["bbox"] = df.bbox.apply(
            lambda arr: [arr[0], arr[1], arr[0] + arr[2], arr[1] + arr[3]]
        )
        category_df = pd.DataFrame(instances_json["categories"]).rename(
            {"id": "category_id"}, axis=1
        )
        annotations_df = df.merge(category_df, on="category_id")
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
            .agg(_aggregate_annotations)
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
        df["date_captured"] = pd.to_datetime(df.date_captured)  # lance GH#98
        return df

    def _convert_metadata_df(self, df: pd.DataFrame) -> pa.Table:
        """Convert each metdata column to pyarrow with lance types"""
        schema = self.get_schema()
        arrays = []
        for name, col in df.items():
            field = schema.field(name)
            arr = self._convert_field(field.name, field.type, col)
            arrays.append(arr)
        table = pa.Table.from_arrays(arrays, schema=schema)
        return table

    def _convert_field(self, name, typ, col):
        if isinstance(typ, pa.ExtensionType):
            storage = pa.array(col, type=typ.storage_type)
            arr = pa.ExtensionArray.from_storage(typ, storage)
        elif pa.types.is_list(typ):
            native_arr = pa.array(col)
            offsets = native_arr.offsets
            values = native_arr.values.to_numpy(zero_copy_only=False)
            return pa.ListArray.from_arrays(
                offsets, self._convert_field(f"{name}.elements", typ.value_type, values)
            )
        elif pa.types.is_struct(typ):
            native_arr = pa.array(col)
            arrays = []
            for subfield in typ:
                sub_arr = native_arr.field(subfield.name)
                if name == "annotations" and subfield.name == "name":
                    converted = self._convert_name_column(
                        sub_arr, native_arr.field("category_id")
                    )
                else:
                    converted = self._convert_field(
                        f"{name}.{subfield.name}",
                        subfield.type,
                        sub_arr.to_numpy(zero_copy_only=False),
                    )
                arrays.append(converted)
            return pa.StructArray.from_arrays(arrays, fields=typ)
        else:
            arr = pa.array(col, type=typ)
        return arr

    def _convert_name_column(self, name_arr, category_id_arr):
        coco_classes = pd.read_csv("coco_classes.csv", header=0, index_col=None)
        # let's make sure the actual data matches
        check = pd.Series(
            dict(
                zip(
                    name_arr.values.to_numpy(False),
                    category_id_arr.values.to_numpy(False),
                )
            )
        ).to_frame(name="check_id")
        joined = coco_classes.set_index("name").join(check, how="right")
        mask = pd.notnull(joined.check_id)
        filtered = joined[mask]
        if not (filtered.check_id == filtered.category_id).all():
            raise ValueError(f"Category id check failed")
        dict_arr = pa.DictionaryArray.from_pandas(
            pd.Categorical(name_arr.values.to_numpy(False), coco_classes.name.values)
        )
        assert not pd.isna(dict_arr.indices.to_numpy()).all()
        return pa.ListArray.from_arrays(name_arr.offsets, dict_arr)

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
            ImageType.from_storage(pa.utf8()),
            pa.int64(),
            pa.int64(),
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
            pa.dictionary(pa.int8(), pa.utf8()),
            pa.dictionary(pa.int8(), pa.utf8()),
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
@click.argument("base_uri")
@click.option(
    "-v", "--version", type=str, default="2017", help="Dataset version. Default 2017"
)
@click.option(
    "-f",
    "--fmt",
    type=click.Choice(["lance", "parquet"]),
    default="lance",
    help="Output format (parquet or lance)",
)
@click.option("-e", "--embedded", type=bool, default=True, help="Embed images")
@click.option(
    "-g",
    "--group-size",
    type=int,
    default=1024,
    help="set the group size",
    show_default=True,
)
@click.option(
    "--max-rows-per-file",
    type=int,
    default=0,
    help="set the max rows per file",
    show_default=True,
)
@click.option(
    "-o",
    "--output-path",
    type=str,
    help="Output path. Default is {base_uri}/coco_links.{fmt}",
)
def main(
    base_uri,
    version,
    fmt,
    embedded,
    output_path,
    group_size: int,
    max_rows_per_file: int,
):
    converter = CocoConverter(base_uri, version=version)
    df = converter.read_metadata()
    known_formats = ["lance", "parquet"]
    if fmt is not None:
        assert fmt in known_formats
        fmt = [fmt]
    else:
        fmt = known_formats

    kwargs = {
        "existing_data_behavior": "overwrite_or_ignore",
        "partitioning": ["split"],
        "partitioning_flavor": "hive",
        "max_rows_per_group": group_size,
        "max_rows_per_file": max_rows_per_file,
    }
    for f in fmt:
        if embedded:
            converter.make_embedded_dataset(df, f, output_path, **kwargs)
        else:
            return converter.save_df(df, f, output_path, **kwargs)


if __name__ == "__main__":
    main()
