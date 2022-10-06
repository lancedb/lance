#!/usr/bin/env python3
import sys
from typing import Union

import duckdb
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from lance.io import download_uris

sys.path.append("..")

from suite import BenchmarkSuite, get_dataset, get_uri
from datagen import CocoConverter

import lance

coco_benchmarks = BenchmarkSuite("coco")


@coco_benchmarks.benchmark("label_distribution", key=["fmt", "flavor"])
def label_distribution(base_uri: str, fmt: str, flavor: str = None):
    if fmt == "raw":
        return _label_distribution_raw(base_uri)
    elif fmt == "lance":
        uri = get_uri(base_uri, "coco", fmt, flavor)
        dataset = get_dataset(uri)
        return _label_distribution_lance(dataset)
    elif fmt == "parquet":
        uri = get_uri(base_uri, "coco", fmt, flavor)
        dataset = get_dataset(uri)
        return _label_distribution_duckdb(dataset)
    raise NotImplementedError()


@coco_benchmarks.benchmark("filter_data", key=["fmt", "flavor"])
def filter_data(base_uri: str, fmt: str, flavor: str = None):
    if fmt == "raw":
        return _filter_data_raw(base_uri)
    elif fmt == "lance":
        return _filter_data_lance(base_uri, flavor=flavor)
    elif fmt == "parquet":
        return _filter_data_parquet(base_uri, flavor=flavor)
    raise NotImplementedError()


def _label_distribution_raw(base_uri: str):
    """Minic
    SELECT label, count(1) FROM coco_dataset GROUP BY 1
    """
    c = CocoConverter(base_uri)
    df = c.read_metadata()
    return pd.json_normalize(df.annotations).name.explode().value_counts()


def _filter_data_raw(base_uri: str, klass="cat", offset=20, limit=50):
    """SELECT image, annotations FROM coco WHERE annotations.label = 'cat' LIMIT 50 OFFSET 20"""
    c = CocoConverter(base_uri)
    df = c.read_metadata()
    ser = pd.json_normalize(df.annotations)["name"]

    def has_klass(names):
        if isinstance(names, list):
            return (np.array(names) == klass).any()
        return False

    mask = ser.apply(has_klass)
    filtered = df.loc[mask, ["image_uri", "annotations"]]
    limited = filtered[offset: offset + limit]
    limited.assign(image=download_uris(limited.image_uri))
    return limited


def _filter_data_lance(base_uri: str, klass="cat", offset=20, limit=50, flavor=None):
    uri = get_uri(base_uri, "coco", "lance", flavor)
    # TODO restore after projection bug
    index_scanner = lance.dataset(uri)
    # index_scanner = index_scanner.scanner(columns=["image_id", "annotations.name"])
    query = (
        f"SELECT distinct image_id FROM ("
        f"  SELECT image_id, UNNEST(annotations.name) as name FROM index_scanner"
        f") WHERE name = '{klass}'"
    )
    filtered_ids = duckdb.query(query).arrow().column("image_id").combine_chunks()
    scanner = lance.dataset(uri).scanner(
        # TODO restore after projection bug
        # columns=["image_id", "image", "annotations.name"],
        columns=["image_id", "image", "annotations"],
        filter=pc.field("image_id").isin(filtered_ids),
        limit=50,
        offset=20,
    )
    return scanner.to_table()


def _filter_data_parquet(base_uri: str, klass="cat", offset=20, limit=50, flavor=None):
    uri = get_uri(base_uri, "coco", "parquet", flavor)
    dataset = ds.dataset(uri)
    query = (
        f"SELECT distinct image_id FROM ("
        f"  SELECT image_id, UNNEST(annotations.name) as name FROM dataset"
        f") WHERE name == '{klass}'"
    )
    filtered_ids = duckdb.query(query).arrow().column("image_id").to_numpy().tolist()
    id_string = ",".join([f"'{x}'" for x in filtered_ids])
    return duckdb.query(
        f"SELECT image, annotations "
        f"FROM dataset "
        f"WHERE image_id in ({id_string}) "
        f"LIMIT 50 OFFSET 20"
    ).to_arrow_table()


def _label_distribution_lance(dataset: ds.Dataset):
    # TODO restore after projection bug
    # scanner = dataset.scanner(columns=["annotations.name"])
    # return _label_distribution_duckdb(scanner)
    return _label_distribution_duckdb(dataset)


def _label_distribution_duckdb(arrow_obj: Union[ds.Dataset | ds.Scanner]):
    query = """\
      SELECT name, COUNT(1) FROM (
        SELECT UNNEST(annotations.name) as name FROM arrow_obj
      ) GROUP BY 1
    """
    return duckdb.query(query).to_df()


if __name__ == "__main__":
    main = coco_benchmarks.create_main()
    main()
