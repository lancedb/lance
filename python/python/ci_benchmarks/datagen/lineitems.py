# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Creates a dataset containing the TPC-H lineitems table using a prebuilt Parquet file

import duckdb
import lance
from lance.log import LOGGER

from ci_benchmarks.datasets import get_dataset_uri

NUM_ROWS = 59986052


def _gen_data():
    LOGGER.info("Using DuckDB to generate TPC-H dataset")
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute("CALL dbgen(sf=10)")
    res = con.query("SELECT * FROM lineitem")
    return res.to_arrow_table()


def _create(dataset_uri: str):
    try:
        ds = lance.dataset(dataset_uri)
        print(ds.count_rows())
        if ds.count_rows() == NUM_ROWS:
            return
        elif ds.count_rows() == 0:
            lance.write_dataset(
                _gen_data(), dataset_uri, mode="append", use_legacy_format=False
            )
        else:
            raise Exception(
                "Cannot generate TPC-H dataset because a dataset with the URI "
                f"{dataset_uri} already exists and doesn't appear to be the "
                "same dataset"
            )
    except ValueError:
        lance.write_dataset(
            _gen_data(), dataset_uri, mode="create", use_legacy_format=False
        )


def gen_tcph():
    dataset_uri = get_dataset_uri("tpch")
    _create(dataset_uri)
