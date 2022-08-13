#!/usr/bin/env python
"""Parse coco dataset"""
import click
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet as pq

import lance
from bench_utils import download_uris


class CocoConverter:

    def __init__(self, dataset_root, version='2017'):
        self.dataset_root = dataset_root
        self.version = version

    def _get_instances_json(self, split):
        uri = os.path.join(self.dataset_root, 'annotations',
                           f'instances_{split}{self.version}.json')
        fs, path = pa.fs.FileSystem.from_uri(uri)
        with fs.open_input_file(path) as fobj:
            return json.load(fobj)

    def _instances_to_df(self, split, instances_json):
        df = pd.DataFrame(instances_json["annotations"])
        df['segmentation'] = df.segmentation.apply(_convert_segmentation)
        category_df = (pd.DataFrame(instances_json["categories"])
                       .rename({'id': 'category_id'}, axis=1))
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
            .agg(list)
        ).reset_index()
        images_df = (pd.DataFrame(instances_json["images"])
                     .rename({'id': 'image_id'}, axis=1))
        images_df["split"] = split
        images_df["image_uri"] = images_df["file_name"].apply(
            lambda fname: os.path.join(self.dataset_root,
                                       f"{split}{self.version}", fname)
        )
        # TODO join images_df.license to instances_json['license']
        return images_df.merge(anno_df, on="image_id")

    def read_instances(self) -> pd.DataFrame:
        def read_split(split):
            json_data = self._get_instances_json(split)
            return self._instances_to_df(split, json_data)
        df = pd.concat([read_split(split) for split in ['train', 'val']])
        # df['date_captured'] = pd.to_datetime(df.date_captured)  # lance GH#98
        return df

    def save_df(self, df, fmt='lance', output_path=None):
        if output_path is None:
            output_path = os.path.join(self.dataset_root, f'coco_links.{fmt}')
        table = pa.Table.from_pandas(df, get_schema())
        if fmt == 'parquet':
            pq.write_table(table, output_path)
        elif fmt == 'lance':
            lance.write_table(table, output_path)
        return table

    def make_embedded_dataset(self, table: pa.Table, fmt='lance', output_path=None):
        if output_path is None:
            output_path = os.path.join(self.dataset_root, f'coco.{fmt}')
        uris = table["image_uri"].to_numpy()
        images = download_uris(pd.Series(uris))
        arr = pa.BinaryArray.from_pandas(images)
        embedded = table.append_column(pa.field("image", pa.binary()), arr)
        if fmt == 'parquet':
            pq.write_table(embedded, output_path)
        elif fmt == 'lance':
            lance.write_table(embedded, output_path)
        return embedded


def _ann_schema():
    segmentation_type = pa.struct([
        pa.field("polygon", pa.list_(pa.list_(pa.int32()))),
        pa.field("coco_rle", pa.struct([
            pa.field('counts', pa.list_(pa.int32())),
            pa.field('size', pa.list_(pa.int32()))])),
    ])
    names = ['segmentation', 'area', 'iscrowd', 'bbox', 'category_id', 'id',
             'supercategory', 'name']
    types = [segmentation_type, pa.float64(), pa.int8(),
             pa.list_(pa.float32()), pa.int16(), pa.int64(),
             pa.utf8(), pa.utf8()]
    schema = pa.list_(pa.struct([pa.field(name, dtype)
                                 for name, dtype in zip(names, types)]))
    return schema


def get_schema():
    names = ['license', 'file_name', 'coco_url', 'height', 'width',
             'date_captured', 'flickr_url', 'image_id', 'split', 'image_uri',
             'annotations']
    types = [pa.int64(), pa.utf8(), pa.utf8(), pa.int64(), pa.int64(),
             pa.utf8(), pa.utf8(), pa.int64(), pa.utf8(), pa.utf8(),
             _ann_schema()]
    return pa.schema([pa.field(name, dtype)
                      for name, dtype in zip(names, types)])


def _convert_segmentation(s):
    if isinstance(s, list):
        return {'polygon': s}
    return s


@click.command
@click.option('-u', '--base-uri', type=str, help='Coco dataset root')
@click.option('-v', '--version', type=str, default='2017',
              help='Dataset version. Default 2017')
@click.option('-f', '--fmt', type=str,
              help='Output format (parquet or lance)')
@click.option('-o', '--output-path', type=str,
              help='Output path. Default is {base_uri}/coco_links.{fmt}')
def main(base_uri, version, fmt, output_path):
    converter = CocoConverter(base_uri, version=version)
    df = converter.read_instances()
    known_formats = ['lance', 'parquet']
    if fmt is not None:
        assert fmt in known_formats
        fmt = [fmt]
    else:
        fmt = known_formats
    for f in fmt:
        return converter.save_df(df, f, output_path)


if __name__ == '__main__':
    main()
