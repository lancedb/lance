#!/usr/bin/env python

"""Parse coco dataset"""

import json
import os
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

from lance.types import ImageType

sys.path.append("..")

from converter import DatasetConverter, PUBLIC_URI_ROOT


class CocoConverter(DatasetConverter):
    def __init__(self, uri_root: str, version: str = "2017"):
        super(CocoConverter, self).__init__("coco", uri_root)
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
            lambda fname: os.path.join(PUBLIC_URI_ROOT,
                                       "coco",
                                       f"{split}{self.version}",
                                       fname)
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
        df = pd.DataFrame(instances_json['annotations'])
        cat_df = pd.DataFrame(instances_json['categories'])
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

        splits = [read_split(split) for split in ["train", "val"]]
        test_images = self._get_test_images("test")
        splits.append(pd.DataFrame({"image_uri": test_images, "split": "test"}))
        df = self._concat_frames(splits, num_rows)
        df["date_captured"] = pd.to_datetime(df.date_captured)
        return df

    def _concat_frames(self,
                       frames: Iterable[pd.DataFrame],
                       num_rows: int) -> pd.DataFrame:
        if num_rows > 0:
            sizes = np.array([len(df) for df in frames])
            rows = np.round(sizes / sizes.sum() * num_rows).astype(int)
            return pd.concat([df.sample(n) for df, n in zip(frames, rows)])
        return pd.concat(frames)

    def _get_test_images(self, dirname: str = "test"):
        uri = os.path.join(self.uri_root,
                           f"{dirname}{self.version}")
        fs, path = pa.fs.FileSystem.from_uri(uri)
        public_uri = os.path.join(PUBLIC_URI_ROOT,
                                  "coco",
                                  f"{dirname}{self.version}")
        return [
            os.path.join(public_uri, file.base_name)
            for file in fs.get_file_info(pa.fs.FileSelector(path, recursive=True))
        ]

    def image_uris(self, table):
        prefix = os.path.join(PUBLIC_URI_ROOT, "coco/")
        uris = np.array([
            os.path.join(self.uri_root, image_uri[len(prefix):])
            for image_uri in table["image_uri"].to_numpy()
        ])
        return uris

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
            pa.string(),  # TODO https://github.com/duckdb/duckdb/issues/4812
            pa.string()  # TODO https://github.com/duckdb/duckdb/issues/4812
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


if __name__ == "__main__":
    main = CocoConverter.create_main()
    main()
