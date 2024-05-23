# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import csv
import os
import shutil
from datetime import datetime

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from lance.file import LanceFileReader, LanceFileWriter
from lance.tracing import trace_to_chrome

trace_to_chrome(level="trace")

"""
Data generation (in-memory)
import duckdb
import pyarrow.parquet as pq

con = duckdb.connect(database=':memory:')
con.execute("INSTALL tpch; LOAD tpch")
con.execute("CALL dbgen(sf=10)")

tables = ["lineitem"]
for t in tables:
    res = con.query("SELECT * FROM " + t)
    pq.write_table(res.to_arrow_table(), t + ".parquet")
"""


def measure_pyarrow_read_time(path, num_trials, verbose=False):
    """
    Measures the time required to read a parquet file using pyarrow,
    averaged over multiple trials
    """
    parquet_read_time = 0
    for trial in range(1, num_trials + 1):
        start = datetime.now()
        pa.parquet.read_table(path)
        end = datetime.now()
        parquet_read_time += (end - start).total_seconds()

        if verbose:
            print(
                f"Parquet Read Time for trial {trial}: {(end - start).total_seconds()}s"
            )

    avg_read_time = parquet_read_time / num_trials
    print(f"Parquet Read Time: {avg_read_time}s")
    return avg_read_time


def measure_parquet_write_time(table, path, num_trials, verbose=False):
    """
    Measures the time required to write a parquet file using pyarrow
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    parquet_write_time = 0
    for trial in range(1, num_trials + 1):
        start = datetime.now()
        pq.write_table(table, path, compression="snappy")
        end = datetime.now()
        parquet_write_time += (end - start).total_seconds()

        if verbose:
            print(
                f"Parquet Write Time for trial {trial}: \
                    {(end - start).total_seconds()}s"
            )

    avg_write_time = parquet_write_time / num_trials
    print(f"Parquet Write Time: {avg_write_time}s")
    return avg_write_time


def measure_lance_read_time(path, num_trials, batch_size, verbose=False):
    """
    Measures the time required to read a lance file using lance,
    averaged over multiple trials
    """
    lance_read_time = 0
    for trial in range(1, num_trials + 1):
        start = datetime.now()
        reader = LanceFileReader(path)
        for batch in reader.read_all(batch_size=batch_size).to_batches():
            pass
        end = datetime.now()

        lance_read_time += (end - start).total_seconds()

        if verbose:
            print(
                f"Lance Read Time for trial {trial}: {(end - start).total_seconds()}s"
            )

    avg_read_time = lance_read_time / num_trials
    print(f"Lance Read Time: {avg_read_time}s")
    return avg_read_time


def measure_lance_write_time(dataset, path, num_trials=10, verbose=False):
    """
    Takes a lance dataset object as input, and writes a lance
    file to disk in a directory
    """

    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    lance_write_time = 0
    for trial in range(1, num_trials + 1):
        start = datetime.now()
        with LanceFileWriter(path, dataset.schema) as writer:
            for batch in dataset.to_batches():
                writer.write_batch(batch)
        end = datetime.now()
        lance_write_time += (end - start).total_seconds()

        if verbose:
            print(
                f"Lance Write Time for trial {trial}: {(end - start).total_seconds()}s"
            )

    avg_write_time = lance_write_time / num_trials
    print(f"Lance Write Time: {avg_write_time}s")
    return avg_write_time


def benchmark_tpch_lineitem(dataset_dir, num_trials=10, verbose=False):
    name = benchmark_tpch_lineitem.__name__
    dataset_path = dataset_dir + "lineitem.parquet"
    output = csv.writer(open(name + ".csv", "w"))
    output.writerow(
        [
            "name",
            "dataset",
            "num_trials",
            "num_rows",
            "parquet_size",
            "pyarrow_read_time",
            "lance_size",
            "lance_read_time",
        ]
    )

    dataset = ds.dataset(dataset_path)
    num_rows = dataset.count_rows()
    parquet_size = os.path.getsize(dataset_path) / (1024**2)

    pyarrow_read_time = measure_pyarrow_read_time(dataset_path, num_trials=10)

    dataset = ds.dataset(dataset_path)
    lance_path = "/tmp/tpch.lancev2"
    measure_lance_write_time(dataset, path=lance_path, num_trials=1)
    lance_size = os.path.getsize(lance_path) / (1024**2)

    lance_read_time = measure_lance_read_time(
        lance_path, num_trials=10, batch_size=1024 * 8
    )

    output.writerow(
        [
            name,
            "tpch_lineitem",
            num_trials,
            num_rows,
            parquet_size,
            pyarrow_read_time,
            lance_size,
            lance_read_time,
        ]
    )


def benchmark_tpch_encodings(
    dataset_dir, dataset_name, encoding_type, num_trials=10, verbose=False
):
    """
    Loads numeric columns from TPCH tables and benchmarks the read times for
    parquet and lance files
    """

    benchmark_name = benchmark_tpch_encodings.__name__
    output = csv.writer(open(benchmark_name + "_" + encoding_type + ".csv", "w"))
    output.writerow(
        [
            "name",
            "dataset",
            "lance_encoding_type",
            "num_trials",
            "num_rows",
            "parquet_size",
            "parquet_write_time",
            "parquet_read_time",
            "lance_size",
            "lance_write_time",
            "lance_read_time",
        ]
    )

    tables = [
        "nation",
        "region",
        "part",
        "supplier",
        "customer",
        "partsupp",
        "orders",
        "lineitem",
    ]

    for table_name in tables:
        print("Table: {}".format(table_name))
        PATH = dataset_dir + table_name + ".parquet"

        orig_dataset = ds.dataset(PATH)
        orig_schema = orig_dataset.schema

        # for field in orig_schema:
        #     print(field.name, field.type)

        if encoding_type == "plain_numeric":
            target_types = [pa.int32(), pa.int64(), pa.float32(), pa.float64()]
        elif encoding_type == "plain_non_numeric":
            target_types = [pa.string(), pa.date32()]

        target_columns = [
            field.name for field in orig_schema if field.type in target_types
        ]

        table = orig_dataset.to_table(columns=target_columns)
        parquet_path = "/tmp/parquet_ds.parquet"

        parquet_write_time = measure_parquet_write_time(
            table, parquet_path, num_trials=num_trials
        )

        dataset = ds.dataset(parquet_path)
        num_rows = dataset.count_rows()
        parquet_size = os.path.getsize(parquet_path) / (1024**2)

        parquet_read_time = measure_pyarrow_read_time(
            parquet_path, num_trials=num_trials
        )

        lance_path = "/tmp/tpch.lancev2"
        lance_write_time = measure_lance_write_time(
            table, lance_path, num_trials=num_trials
        )
        lance_size = os.path.getsize(lance_path) / (1024**2)

        lance_read_time = measure_lance_read_time(
            lance_path, num_trials=num_trials, batch_size=1024 * 8
        )

        output.writerow(
            [
                benchmark_name,
                dataset_name + "_" + table_name,
                encoding_type,
                num_trials,
                num_rows,
                parquet_size,
                parquet_write_time,
                parquet_read_time,
                lance_size,
                lance_write_time,
                lance_read_time,
            ]
        )

        os.remove(parquet_path)

        if os.path.isdir(lance_path):
            shutil.rmtree(lance_path)
        else:
            os.remove(lance_path)


"""
Data generation (sift)
    nvecs = 10000
    ndims = 128
    with open("../sift/sift_base.fvecs", mode="rb") as fobj:
        buf = fobj.read()
        data = np.array(struct.unpack("<1280000f",
                        buf[4 : 4 + 4 * nvecs * ndims])).reshape((nvecs, ndims))
        dd = dict(zip(range(nvecs), data))

    table = vec_to_table(dd)
    parquet_path = "/home/ubuntu/test/sift1m.parquet"

"""


def benchmark_sift_vector_encodings(
    dataset_dir, dataset_name, encoding_type, num_trials=10, verbose=False
):
    benchmark_name = benchmark_sift_vector_encodings.__name__
    output = csv.writer(open(benchmark_name + "_" + encoding_type + ".csv", "w"))
    output.writerow(
        [
            "name",
            "dataset",
            "lance_encoding_type",
            "num_trials",
            "num_rows",
            "parquet_size",
            "parquet_write_time",
            "parquet_read_time",
            "lance_size",
            "lance_write_time",
            "lance_read_time",
        ]
    )

    PATH = dataset_dir + "sift1m.parquet"
    orig_dataset = ds.dataset(PATH)

    table = orig_dataset.to_table()
    parquet_path = "/tmp/parquet_ds.parquet"

    parquet_write_time = measure_parquet_write_time(
        table, parquet_path, num_trials=num_trials
    )

    dataset = ds.dataset(parquet_path)
    num_rows = dataset.count_rows()
    parquet_size = os.path.getsize(parquet_path) / (1024**2)

    parquet_read_time = measure_pyarrow_read_time(parquet_path, num_trials=num_trials)

    lance_path = "/tmp/tpch.lancev2"
    lance_write_time = measure_lance_write_time(
        table, lance_path, num_trials=num_trials
    )
    lance_size = os.path.getsize(lance_path) / (1024**2)

    lance_read_time = measure_lance_read_time(
        lance_path, num_trials=num_trials, batch_size=1024 * 8
    )

    output.writerow(
        [
            benchmark_name,
            dataset_name,
            encoding_type,
            num_trials,
            num_rows,
            parquet_size,
            parquet_write_time,
            parquet_read_time,
            lance_size,
            lance_write_time,
            lance_read_time,
        ]
    )


if __name__ == "__main__":
    if os.path.exists("benchmark_tpch_encodings.csv"):
        os.remove("benchmark_tpch_encodings.csv")

    # benchmark_tpch_lineitem("/home/ubuntu/test/TPCH_SF1/")
    benchmark_tpch_encodings(
        "/home/ubuntu/test/TPCH_SF1/",
        dataset_name="tpch_sf1",
        encoding_type="plain_numeric",
    )
    benchmark_tpch_encodings(
        "/home/ubuntu/test/TPCH_SF1/",
        dataset_name="tpch_sf1",
        encoding_type="plain_non_numeric",
    )
    benchmark_sift_vector_encodings(
        "/home/ubuntu/test/",
        dataset_name="sift1m",
        encoding_type="plain_fixed_size_list",
        num_trials=5,
    )
