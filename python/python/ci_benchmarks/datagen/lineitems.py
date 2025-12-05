# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Creates a dataset containing the TPC-H lineitems table using a prebuilt Parquet file

import duckdb
import lance
from lance.log import LOGGER

from ci_benchmarks.datasets import get_dataset_uri

NUM_ROWS = 59986052


def _gen_data(scale_factor: int):
    LOGGER.info("Using DuckDB to generate TPC-H dataset")
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={scale_factor})")
    res = con.query("SELECT * FROM lineitem")
    return res.to_arrow_table()


def _create(dataset_uri: str, data_storage_version: str, scale_factor: int = 10):
    try:
        ds = lance.dataset(dataset_uri)
        print(ds.count_rows())
        if ds.count_rows() == NUM_ROWS:
            return
        elif ds.count_rows() == 0:
            ds = lance.write_dataset(
                _gen_data(scale_factor),
                dataset_uri,
                mode="append",
                data_storage_version=data_storage_version,
            )
        else:
            raise Exception(
                "Cannot generate TPC-H dataset because a dataset with the URI "
                f"{dataset_uri} already exists and doesn't appear to be the "
                "same dataset"
            )
    except ValueError:
        ds = lance.write_dataset(
            _gen_data(scale_factor),
            dataset_uri,
            mode="create",
            data_storage_version=data_storage_version,
        )
    return ds


def gen_tcph():
    dataset_uri = get_dataset_uri("tpch")
    _create(dataset_uri, data_storage_version="2.0")
    dataset_uri = get_dataset_uri("tpch-2.1")
    _create(dataset_uri, data_storage_version="2.1")


def gen_mem_tcph(data_storage_version: str):
    dataset_uri = "memory://tpch"
    return _create(
        dataset_uri, data_storage_version=data_storage_version, scale_factor=1
    )
