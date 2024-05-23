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


def write_lance_file(dataset, dir_path="/tmp/tpch.lancev2"):
    """
    Takes a pyarrow dataset object as input, and writes a lance
    file to disk in a directory
    """

    with LanceFileWriter(dir_path, dataset.schema) as writer:
        for batch in dataset.to_batches():
            writer.write_batch(batch)


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
            print(f"Parquet Read Time: {parquet_read_time/trial}s")

    avg_read_time = parquet_read_time / num_trials
    print(f"Parquet Read Time: {avg_read_time}s")
    return avg_read_time


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
            print(f"V2 Read Time: {lance_read_time/trial}s")

    avg_read_time = lance_read_time / num_trials
    print(f"V2 Read Time: {avg_read_time}s")
    return avg_read_time


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
    write_lance_file(dataset, dir_path=lance_path)
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


def benchmark_tpch_numeric_encodings(dataset_dir, num_trials=10, verbose=False):
    """
    Loads numeric columns from TPCH tables and benchmarks the read times for
    parquet and lance files
    """

    name = benchmark_tpch_numeric_encodings.__name__
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

        # Only specified the numeric types relevant to TPCH. We can expand it
        # if we generalize this later
        int_columns = [
            field.name
            for field in orig_schema
            if field.type
            in (pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.decimal128(15, 2))
        ]

        # Write table with only numeric columns to disk
        table = orig_dataset.to_table(columns=int_columns)
        parquet_path = "/tmp/numeric_ds.parquet"
        pq.write_table(table, parquet_path)
        dataset = ds.dataset(parquet_path)

        num_rows = dataset.count_rows()
        parquet_size = os.path.getsize(PATH) / (1024**2)

        pyarrow_read_time = measure_pyarrow_read_time(parquet_path, num_trials=10)

        lance_path = "/tmp/tpch.lancev2"
        write_lance_file(dataset, dir_path=lance_path)
        lance_size = os.path.getsize(lance_path) / (1024**2)

        lance_read_time = measure_lance_read_time(
            lance_path, num_trials=10, batch_size=1024 * 8
        )

        output.writerow(
            [
                name,
                "tpch_" + table_name,
                num_trials,
                num_rows,
                parquet_size,
                pyarrow_read_time,
                lance_size,
                lance_read_time,
            ]
        )

        os.remove(parquet_path)

        if os.path.isdir(lance_path):
            shutil.rmtree(lance_path)
        else:
            os.remove(lance_path)


# benchmark_tpch_lineitem("/home/ubuntu/test/TPCH_SF1/")
benchmark_tpch_numeric_encodings("/home/ubuntu/test/TPCH_SF1/")
