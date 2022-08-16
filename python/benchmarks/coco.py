#!/usr/bin/env python3

from typing import Union

import duckdb
import pandas as pd

import lance
import pyarrow.compute as pc
import pyarrow.dataset as ds
from bench_utils import download_uris, get_uri, get_dataset, BenchmarkSuite
from parse_coco import CocoConverter

coco_benchmarks = BenchmarkSuite("coco")


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
    df = c.read_metadata()
    return pd.json_normalize(df.annotations.explode()).name.value_counts()


def _filter_data_raw(base_uri: str, klass="cat", offset=20, limit=50):
    """SELECT image, annotations FROM coco WHERE annotations.label = 'cat' LIMIT 50 OFFSET 20"""
    c = CocoConverter(base_uri)
    df = c.read_metadata()
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
                            # filter=pc.field("image_id").isin(filtered_ids),
                            limit=50, offset=20)
    return scanner.to_table().to_pandas()


def _filter_data_parquet(base_uri: str, klass="cat", offset=20, limit=50, flavor=None):
    uri = get_uri(base_uri, "coco", "parquet", flavor)
    dataset = ds.dataset(uri)
    query = (f"SELECT distinct image_id FROM ("
             f"  SELECT image_id, UNNEST(annotations) as ann FROM dataset"
             f") WHERE ann.name == '{klass}'")
    filtered_ids = duckdb.query(query).arrow().column("image_id").to_numpy().tolist()
    id_string = ','.join([f"'{x}'" for x in filtered_ids])
    return duckdb.query(f"SELECT image, annotations "
                        f"FROM dataset "
                        f"WHERE image_id in ({id_string}) "
                        f"LIMIT 50 OFFSET 20").to_arrow_table()


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


if __name__ == "__main__":
    main = coco_benchmarks.create_main()
    main()
