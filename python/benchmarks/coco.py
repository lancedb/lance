#!/usr/bin/env python3

import argparse
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.fs

from bench_utils import download_uris, timeit


def get_metadata(base_uri: str, split: str = "val"):
    annotation_uri = os.path.join(base_uri, f"annotations/instances_{split}2017.json")
    fs, path = pa.fs.FileSystem.from_uri(annotation_uri)
    with fs.open_input_file(path) as fobj:
        annotation_json = json.load(fobj)
    df = pd.DataFrame(annotation_json["annotations"])
    category_df = pd.DataFrame(annotation_json["categories"])
    annotations_df = df.merge(category_df, left_on="category_id", right_on="id").rename(
        {"id": "category_id"}
    )
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
    )
    # print(anno_df, anno_df.columns)
    images_df = pd.DataFrame(annotation_json["images"])
    images_df["split"] = split
    images_df["image_uri"] = images_df["file_name"].apply(
        lambda fname: os.path.join(base_uri, f"{split}2017", fname)
    )
    return images_df.merge(anno_df, left_on="id", right_on="image_id")


@timeit
def get_label_distribution(base_uri: str):
    """Minic
    SELECT label, count(1) FROM coco_dataset GROUP BY 1
    """
    metadata = get_metadata(base_uri)
    exploded_series = (
        metadata["annotations"].explode("annotations").apply(lambda r: r["name"])
    )
    return exploded_series.value_counts()


@timeit
def get_filtered_data(url: str, klass="cat", offset=20, limit=50):
    """SELECT image, annotations FROM coco WHERE annotations.label = 'cat' LIMIT 50 OFFSET 20"""
    # %time rs = bench.get_pets_filtered_data(url, "pug", 20, 50)
    df = get_metadata(url)
    print(df["annotations"])
    filtered = df[["image_uri", "annotations"]].loc[df["annotations"].apply(
        lambda annos: any([a["name"] == "cat" for a in annos])
    )]
    limited = filtered[offset:offset + limit]
    limited["image"] = download_uris(limited.image_uri)
    return limited


def main():
    parser = argparse.ArgumentParser(description="Benchmarks on COCO dataset")
    parser.add_argument("uri", help="base uri for coco dataset")
    args = parser.parse_args()

    get_label_distribution(args.uri)
    get_filtered_data(args.uri)


if __name__ == "__main__":
    main()
