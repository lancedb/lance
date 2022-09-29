#!/usr/bin/env python3

import os
import sys
from typing import Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset

from lance.io import download_uris

sys.path.append("..")

from suite import BenchmarkSuite, get_dataset, get_uri
from datagen import OxfordPetConverter

import lance

oxford_pet_benchmarks = BenchmarkSuite("oxford_pet")


@oxford_pet_benchmarks.benchmark("label_distribution", key=["fmt", "flavor"])
def label_distribution(base_uri: str, fmt: str, flavor: Optional[str]):
    if fmt == "raw":
        return get_pets_class_distribution(base_uri)
    uri = get_uri(base_uri, "oxford_pet", fmt, flavor)
    ds = get_dataset(uri)
    query = "SELECT class, count(1) FROM ds GROUP BY 1"
    return duckdb.query(query).to_df()


@oxford_pet_benchmarks.benchmark("filter_data", key=["fmt", "flavor"])
def filter_data(base_uri: str, fmt: str, flavor: Optional[str]):
    if fmt == "raw":
        return get_pets_filtered_data(base_uri)
    uri = get_uri(base_uri, "oxford_pet", fmt, flavor)
    if fmt == "parquet":
        ds = get_dataset(uri)
        query = "SELECT image, class FROM ds WHERE class='pug' " "LIMIT 50 OFFSET 20"
        return duckdb.query(query).to_arrow_table()
    elif fmt == "lance":
        scanner = lance.dataset(uri).scanner(
            columns=["image", "class"],
            filter=pc.field("class") == "pug",
            limit=50,
            offset=20,
        )
        return scanner.to_table()


@oxford_pet_benchmarks.benchmark("area_histogram", key=["fmt", "flavor"])
def compute_histogram(base_uri: str, fmt: str, flavor: Optional[str]):
    if fmt == "raw":
        return area_histogram_raw(base_uri)
    uri = get_uri(base_uri, "oxford_pet", fmt, flavor)
    ds = get_dataset(uri)
    query = "SELECT histogram(size.width * size.height) FROM ds"
    return duckdb.query(query).to_df()


def get_pets_class_distribution(base_uri):
    c = OxfordPetConverter(base_uri)
    df = c.read_metadata()
    return df.groupby("class")["class"].count()


def get_pets_filtered_data(base_uri, klass="pug", offset=20, limit=50):
    c = OxfordPetConverter(base_uri)
    df = c.read_metadata()
    filtered = df.loc[df["class"] == klass, ["class", "filename"]]
    limited: pd.DataFrame = filtered[offset: offset + limit]
    uris = [os.path.join(base_uri, f"images/{x}.jpg") for x in limited.filename.values]
    return limited.assign(images=download_uris(pd.Series(uris)))


def area_histogram_raw(base_uri):
    c = OxfordPetConverter(base_uri)
    df = c.read_metadata()
    sz = pd.json_normalize(df["size"])
    query = "SELECT histogram(width * height) FROM sz"
    return duckdb.query(query).to_df()


if __name__ == "__main__":
    main = oxford_pet_benchmarks.create_main()
    main()
