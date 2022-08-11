#!/usr/bin/env python3
from typing import Union

import click
import json
import os

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs
import pyarrow.compute as pc

import lance
from bench_utils import download_uris, get_uri, get_dataset, BenchmarkSuite, timeit


coco_benchmarks = BenchmarkSuite()


@coco_benchmarks.benchmark("label_distribution")
def label_distribution(base_uri: str, fmt: str, flavor: str = None):
    if fmt == 'raw':
        return _label_distribution_raw(base_uri)
    elif fmt == 'lance':
        uri = get_uri(base_uri, "coco", fmt, flavor)
        dataset = get_dataset(uri)
        return _label_distribution_lance(dataset)
    elif fmt == 'parquet':
        uri = get_uri(base_uri, "coco", fmt, flavor)
        dataset = get_dataset(uri)
        return _label_distribution_duckdb(dataset)
    raise NotImplementedError()

@coco_benchmarks.benchmark("filter_data")
def filter_data(base_uri: str, fmt: str, flavor: str = None):
    if fmt == 'raw':
        return _filter_data_raw(base_uri)
    elif fmt == 'lance':
        return _filter_data_lance(base_uri)
    elif fmt == 'parquet':
        return _filter_data_parquet(base_uri)
    raise NotImplementedError()


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
    images_df = pd.DataFrame(annotation_json["images"])
    images_df["split"] = split
    images_df["image_uri"] = images_df["file_name"].apply(
        lambda fname: os.path.join(base_uri, f"{split}2017", fname)
    )
    return images_df.merge(anno_df, left_on="id", right_on="image_id")


@timeit
def _label_distribution_raw(base_uri: str):
    """Minic
    SELECT label, count(1) FROM coco_dataset GROUP BY 1
    """
    metadata = get_metadata(base_uri)
    exploded_series = (
        metadata["annotations"].explode("annotations").apply(lambda r: r["name"])
    )
    return exploded_series.value_counts()


@timeit
def _filter_data_raw(url: str, klass="cat", offset=20, limit=50):
    """SELECT image, annotations FROM coco WHERE annotations.label = 'cat' LIMIT 50 OFFSET 20"""
    df = get_metadata(url)
    filtered = df[["image_uri", "annotations"]].loc[df["annotations"].apply(
        lambda annos: any([a["name"] == "cat" for a in annos])
    )]
    limited = filtered[offset:offset + limit]
    limited.assign(image=download_uris(limited.image_uri))
    return limited


@timeit
def _filter_data_lance(base_uri: str, klass="cat", offset=20, limit=50):
    uri = get_uri(base_uri, "coco", "lance", None)
    index_scanner = lance.scanner(uri, columns=['id', 'annotations.label'])
    query = (f"SELECT distinct id FROM ("
             f"  SELECT id, UNNEST(annotations) as ann FROM index_scanner"
             f") WHERE ann.label == '{klass}'")
    filtered_ids = duckdb.query(query).arrow().column("id").to_numpy().tolist()
    scanner = lance.scanner(uri, ['image', 'annotations.label'],
                            limit=50, offset=20)
    return scanner.to_table().to_pandas()


@timeit
def _filter_data_parquet(base_uri: str, klass="cat", offset=20, limit=50):
    uri = get_uri(base_uri, "coco", "parquet", None)
    dataset = ds.dataset(uri)
    query = (f"SELECT distinct id FROM ("
             f"  SELECT id, UNNEST(annotations) as ann FROM dataset"
             f") WHERE ann.label == '{klass}'")
    filtered_ids = duckdb.query(query).arrow().column("id").to_numpy().tolist()
    df: pd.DataFrame = duckdb.query("SELECT image, annotations FROM dataset LIMIT 50 OFFSET 20").to_df()
    df["label"] = df.annotations.apply(lambda x: [elm["label"] for elm in x])
    return df.drop(columns='annotations')


def _label_distribution_lance(dataset: ds.Dataset):
    scanner = lance.scanner(dataset, columns=['annotations.label'])
    return _label_distribution_duckdb(scanner)


@timeit
def _label_distribution_duckdb(arrow_obj: Union[ds.Dataset | ds.Scanner]):
    query = """\
      SELECT ann.label, COUNT(1) FROM (
        SELECT UNNEST(annotations) as ann FROM arrow_obj
      ) GROUP BY 1
    """
    return duckdb.query(query).to_df()


KNOWN_FORMATS = ["lance", "parquet", "raw"]


@click.command
@click.option('-u', '--base-uri', required=True, type=str,
              help="Base uri to the benchmark dataset catalog")
@click.option('-f', '--format', 'fmt',
              help="'lance', 'parquet', or 'raw'. Omit for all")
@click.option('--flavor', type=str,
              help="external if parquet/lance had external images version")
@click.option('-b', '--benchmark', type=str,
              help="which benchmark to run. Omit for all")
def main(base_uri, fmt, flavor, benchmark):
    if fmt:
        fmt = fmt.strip().lower()
        assert fmt in KNOWN_FORMATS
    base_uri = f'{base_uri}/datasets/coco'

    if benchmark is not None:
        coco_benchmarks.get_benchmark(benchmark).run(base_uri, fmt, flavor)
    else:
        for b in coco_benchmarks.list_benchmarks():
            if fmt:
                b.run(base_uri, fmt, flavor)
            else:
                for f in KNOWN_FORMATS:
                    b.run(base_uri, f, flavor)


if __name__ == "__main__":
    main()
