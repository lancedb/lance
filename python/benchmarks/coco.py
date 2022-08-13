#!/usr/bin/env python3
import pathlib
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
from parse_coco import CocoConverter
from bench_utils import download_uris, get_uri, get_dataset, BenchmarkSuite


coco_benchmarks = BenchmarkSuite()


@coco_benchmarks.benchmark("label_distribution", key=['fmt', 'flavor'])
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


@coco_benchmarks.benchmark("filter_data", key=['fmt', 'flavor'])
def filter_data(base_uri: str, fmt: str, flavor: str = None):
    if fmt == 'raw':
        return _filter_data_raw(base_uri)
    elif fmt == 'lance':
        return _filter_data_lance(base_uri, flavor=flavor)
    elif fmt == 'parquet':
        return _filter_data_parquet(base_uri, flavor=flavor)
    raise NotImplementedError()

def _label_distribution_raw(base_uri: str):
    """Minic
    SELECT label, count(1) FROM coco_dataset GROUP BY 1
    """
    c = CocoConverter(base_uri)
    df = c.read_instances()
    return pd.json_normalize(df.annotations.explode()).name.value_counts()


def _filter_data_raw(base_uri: str, klass="cat", offset=20, limit=50):
    """SELECT image, annotations FROM coco WHERE annotations.label = 'cat' LIMIT 50 OFFSET 20"""
    c = CocoConverter(base_uri)
    df = c.read_instances()
    mask = df.annotations.apply(lambda ann: any([a["name"] == klass for a in ann]))
    filtered = df.loc[mask, ["image_uri", "annotations"]]
    limited = filtered[offset:offset + limit]
    limited.assign(image=download_uris(limited.image_uri))
    return limited


def _filter_data_lance(base_uri: str, klass="cat", offset=20, limit=50, flavor=None):
    uri = get_uri(base_uri, "coco", "lance", flavor)
    index_scanner = lance.scanner(uri, columns=['image_id', 'annotations.name'])
    query = (f"SELECT distinct image_id FROM ("
             f"  SELECT image_id, UNNEST(annotations) as ann FROM index_scanner"
             f") WHERE ann.name == '{klass}'")
    filtered_ids = duckdb.query(query).arrow().column("image_id").combine_chunks()
    scanner = lance.scanner(uri, ['image_id', 'image', 'annotations.name'],
                            #filter=pc.field("image_id").isin(filtered_ids),
                            limit=50, offset=20)
    return scanner.to_table().to_pandas()


def _filter_data_parquet(base_uri: str, klass="cat", offset=20, limit=50, flavor=None):
    uri = get_uri(base_uri, "coco", "parquet", flavor)
    dataset = ds.dataset(uri)
    query = (f"SELECT distinct image_id FROM ("
             f"  SELECT image_id, UNNEST(annotations) as ann FROM dataset"
             f") WHERE ann.name == '{klass}'")
    filtered_ids = duckdb.query(query).arrow().column("image_id").to_numpy().tolist()
    return duckdb.query("SELECT image, annotations FROM dataset LIMIT 50 OFFSET 20").to_arrow_table()


def _label_distribution_lance(dataset: ds.Dataset):
    scanner = lance.scanner(dataset, columns=['annotations.name'])
    return _label_distribution_duckdb(scanner)


def _label_distribution_duckdb(arrow_obj: Union[ds.Dataset | ds.Scanner]):
    query = """\
      SELECT ann.name, COUNT(1) FROM (
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
@click.option('-r', '--repeats', type=int,
              help="number of times to run each benchmark")
@click.option('-o', '--output', type=str,
              help="save timing results to directory")
def main(base_uri, fmt, flavor, benchmark, repeats, output):
    if fmt:
        fmt = fmt.strip().lower()
        assert fmt in KNOWN_FORMATS
        fmt = [fmt]
    else:
        fmt = KNOWN_FORMATS
    base_uri = f'{base_uri}/datasets/coco'

    def run_benchmark(bmark):
        b = bmark.repeat(repeats or 1)
        for f in fmt:
            b.run(base_uri=base_uri, fmt=f, flavor=flavor)
        if output:
            path = pathlib.Path(output) / f"{bmark.name}.csv"
            b.to_df().to_csv(path, index=False)

    if benchmark is not None:
        b = coco_benchmarks.get_benchmark(benchmark)
        run_benchmark(b)
    else:
        [run_benchmark(b) for b in coco_benchmarks.list_benchmarks()]


if __name__ == "__main__":
    main()
