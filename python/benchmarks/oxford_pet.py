#!/usr/bin/env python3

import os
from typing import Optional

import duckdb
import pandas as pd

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset
from bench_utils import BenchmarkSuite, download_uris
from parse_pet import OxfordPetConverter

oxford_pet_benchmarks = BenchmarkSuite("oxford_pet")


@oxford_pet_benchmarks.benchmark("label_distribution", key=['fmt', 'flavor'])
def label_distribution(base_uri: str, fmt: str, flavor: Optional[str]):
    if fmt == "raw":
        return get_pets_class_distribution(base_uri)
    suffix = '' if not flavor else f'_{flavor}'
    ds = _get_dataset(os.path.join(base_uri, f'oxford_pet{suffix}.{fmt}'), fmt)
    query = "SELECT class, count(1) FROM ds GROUP BY 1"
    return duckdb.query(query).to_df()


@oxford_pet_benchmarks.benchmark("filter_data", key=['fmt', 'flavor'])
def filter_data(base_uri: str, fmt: str, flavor: Optional[str]):
    if fmt == "raw":
        return get_pets_filtered_data(base_uri)
    suffix = '' if not flavor else f'_{flavor}'
    uri = os.path.join(base_uri, f'oxford_pet{suffix}.{fmt}')
    if fmt == "parquet":
        ds = _get_dataset(uri, fmt)
        query = ("SELECT image, class FROM ds WHERE class='pug' "
                 "LIMIT 50 OFFSET 20")
        return duckdb.query(query).to_df()
    elif fmt == "lance":
        scanner = lance.scanner(uri, columns=["image", "class"],
                                filter=pc.field("class") == "pug",
                                limit=50, offset=20)
        return scanner.to_table().to_pandas()


def _get_dataset(uri, fmt):
    if fmt == "parquet":
        return pa.dataset.dataset(uri)
    elif fmt == "lance":
        return lance.dataset(uri)
    raise NotImplementedError()


def get_pets_class_distribution(base_uri):
    c = OxfordPetConverter(base_uri)
    df = c.read_metadata()
    return df.groupby("class")["class"].count()


def get_pets_filtered_data(base_uri, klass="pug", offset=20, limit=50):
    c = OxfordPetConverter(base_uri)
    df = c.read_metadata()
    filtered = df.loc[df["class"] == klass, ["class", "filename"]]
    limited = filtered[offset: offset + limit]
    uris = [os.path.join(base_uri, f"images/{x}.jpg")
            for x in limited.filename.values]
    limited.assign(images=download_uris(pd.Series(uris)))
    return limited


if __name__ == "__main__":
    main = oxford_pet_benchmarks.create_main()
    main()
