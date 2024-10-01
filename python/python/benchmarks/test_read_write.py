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

# from lance.tracing import trace_to_chrome

# trace_to_chrome(level="trace")

"""
Data generation (in-memory)
import duckdb
import pyarrow.parquet as pq

con = duckdb.connect(database=":memory:")
con.execute("INSTALL tpch; LOAD tpch")
con.execute("CALL dbgen(sf=1)")

tables = ["lineitem", "nation", "region", "part", "supplier", "customer", "partsupp",
    "orders"]
for t in tables:
    res = con.query("SELECT * FROM " + t)
    pq.write_table(res.to_arrow_table(), tpch_dir_path + t + ".parquet")
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

    file_size_bytes = os.path.getsize(path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(
        f"Parquet Read Time: {avg_read_time:.2f}s for file {path} of size: "
        f"{file_size_mb:.2f} MiB"
    )
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

    file_size_bytes = os.path.getsize(path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(
        f"Parquet Write Time: {avg_write_time:.2f}s for file {path} of size: "
        f"{file_size_mb:.2f} MiB"
    )
    return avg_write_time


def measure_lance_read_time(
    path, num_trials, batch_size, verbose=False, lance_version="2.0"
):
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

    file_size_bytes = os.path.getsize(path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(
        f"Lance {lance_version} Read Time: {avg_read_time:.2f}s for file {path} "
        f"of size: {file_size_mb:.2f} MiB"
    )
    return avg_read_time


def measure_lance_write_time(
    dataset, path, num_trials=10, verbose=False, lance_version="2.0"
):
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
        with LanceFileWriter(path, dataset.schema, version=lance_version) as writer:
            for batch in dataset.to_batches():
                writer.write_batch(batch)
        end = datetime.now()
        lance_write_time += (end - start).total_seconds()

        if verbose:
            print(
                f"Lance Write Time for trial {trial}: {(end - start).total_seconds()}s"
            )

    avg_write_time = lance_write_time / num_trials
    if os.path.exists(path):
        file_size_bytes = os.path.getsize(path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        print(
            f"Lance {lance_version} Write Time: {avg_write_time:.2f}s for file {path} "
            f"of size: {file_size_mb:.2f} MiB"
        )
    else:
        raise (f"Lance {lance_version} write file {path} failed")

    return avg_write_time


def benchmark_tpch_lineitem(
    dataset_dir, num_trials=10, verbose=False, lance_version="2.0"
):
    name = benchmark_tpch_lineitem.__name__
    dataset_path = dataset_dir + "lineitem.parquet"
    output = csv.writer(open(name + ".csv", "w"))

    dataset = ds.dataset(dataset_path)
    num_rows = dataset.count_rows()
    parquet_size = os.path.getsize(dataset_path)
    parquet_size_mb = parquet_size / (1024 * 1024)

    pyarrow_read_time = measure_pyarrow_read_time(dataset_path, num_trials=10)

    dataset = ds.dataset(dataset_path)
    lance_path = "/tmp/tpch.lanceV" + lance_version
    measure_lance_write_time(
        dataset, path=lance_path, num_trials=1, lance_version=lance_version
    )
    lance_size = os.path.getsize(lance_path)
    lance_size_mb = lance_size / (1024 * 1024)

    lance_read_time = measure_lance_read_time(
        lance_path, num_trials=10, batch_size=1024 * 8, lance_version=lance_version
    )

    output.writerows(
        [
            [
                "name",
                "dataset",
                "num_trials",
                "num_rows",
                "parquet_size",
                "lance_size",
                "pyarrow_read_time",
                "lance_read_time",
            ],
            [
                name,
                "tpch_lineitem",
                num_trials,
                num_rows,
                f"{parquet_size_mb:.2f} MiB",
                f"{lance_size_mb:.2f} MiB",
                f"{pyarrow_read_time:.2f} seconds",
                f"{lance_read_time:.2f} seconds",
            ],
        ]
    )


def benchmark_tpch_encodings(
    dataset_dir,
    dataset_name,
    encoding_type,
    num_trials=10,
    verbose=False,
    lance_version="2.0",
):
    """
    Loads numeric columns from TPCH tables and benchmarks the read times for
    parquet and lance files
    """

    benchmark_name = benchmark_tpch_encodings.__name__
    output = csv.writer(
        open(benchmark_name + "_" + dataset_name + "_" + encoding_type + ".csv", "w")
    )
    output.writerow(
        [
            "name",
            "dataset",
            "lance_encoding_type",
            "num_trials",
            "num_rows",
            "parquet_size",
            "lance_size",
            "parquet_write_time",
            "lance_write_time",
            "parquet_read_time",
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

        if encoding_type == "plain_numeric":
            target_types = [pa.int32(), pa.int64(), pa.float32(), pa.float64()]
        elif encoding_type == "plain_non_numeric":
            target_types = [pa.string()]
        elif encoding_type == "plain_timestamp":
            target_types = [pa.date32()]
        elif encoding_type == "integer":
            target_types = [
                pa.int8(),
                pa.int16(),
                pa.int32(),
                pa.int64(),
                pa.uint8(),
                pa.uint16(),
                pa.uint32(),
                pa.uint64(),
            ]

        target_columns = [
            field.name for field in orig_schema if field.type in target_types
        ]

        if len(target_columns) == 0:
            continue
        table = orig_dataset.to_table(columns=target_columns)
        parquet_path = "/tmp/parquet_ds.parquet"

        parquet_write_time = measure_parquet_write_time(
            table, parquet_path, num_trials=num_trials
        )

        dataset = ds.dataset(parquet_path)
        num_rows = dataset.count_rows()
        parquet_size = os.path.getsize(parquet_path)
        parquet_size_mb = parquet_size / (1024 * 1024)

        parquet_read_time = measure_pyarrow_read_time(
            parquet_path, num_trials=num_trials
        )

        lance_path = "/tmp/tpch.lanceV" + lance_version
        lance_write_time = measure_lance_write_time(
            table, lance_path, num_trials=num_trials, lance_version=lance_version
        )
        lance_size = os.path.getsize(lance_path)
        lance_size_mb = lance_size / (1024 * 1024)

        lance_read_time = measure_lance_read_time(
            lance_path,
            num_trials=num_trials,
            batch_size=1024 * 8,
            lance_version=lance_version,
        )

        output.writerow(
            [
                benchmark_name,
                dataset_name + "_" + table_name,
                encoding_type,
                num_trials,
                num_rows,
                f"{parquet_size_mb:.2f} MiB",
                f"{lance_size_mb:.2f} MiB",
                f"{parquet_write_time:.2f} seconds",
                f"{lance_write_time:.2f} seconds",
                f"{parquet_read_time:.2f} seconds",
                f"{lance_read_time:.2f} seconds",
            ]
        )

        os.remove(parquet_path)

        if os.path.isdir(lance_path):
            shutil.rmtree(lance_path)
        else:
            os.remove(lance_path)


"""
Data generation (sift)
    nvecs = 1000000
    ndims = 128
    with open("../sift/sift_base.fvecs", mode="rb") as fobj:
        buf = fobj.read()
        data = np.array(struct.unpack("<128000000f",
                        buf[4 : 4 + 4 * nvecs * ndims])).reshape((nvecs, ndims))
        dd = dict(zip(range(nvecs), data))

    table = vec_to_table(dd)
    parquet_path = "/tmp/sift1m.parquet"

"""


def benchmark_sift_vector_encodings(
    dataset_dir,
    dataset_name,
    encoding_type,
    num_trials=10,
    verbose=False,
    lance_version="2.0",
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
            "lance_size",
            "parquet_write_time",
            "lance_write_time",
            "parquet_read_time",
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
    parquet_size = os.path.getsize(parquet_path) / (1024**3)

    parquet_read_time = measure_pyarrow_read_time(parquet_path, num_trials=num_trials)

    lance_path = "/tmp/tpch.lanceV" + lance_version
    lance_write_time = measure_lance_write_time(
        table, lance_path, num_trials=num_trials, lance_version=lance_version
    )
    lance_size = os.path.getsize(lance_path) / (1024**3)

    lance_read_time = measure_lance_read_time(
        lance_path,
        num_trials=num_trials,
        batch_size=1024 * 8,
        lance_version=lance_version,
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


# Should contain all table files
tpch_dir_path = "/tmp/TPCH_SF1/"
dataset_name = "tpch_sf1"

if __name__ == "__main__":
    # Version "2.1" test is temporarily disabled due to currently issues when
    # activate it
    for lance_version in ["2.0"]:
        benchmark_tpch_lineitem(tpch_dir_path, lance_version)
        benchmark_tpch_encodings(
            tpch_dir_path,
            dataset_name=dataset_name,
            encoding_type="plain_numeric",
            lance_version=lance_version,
        )
        benchmark_tpch_encodings(
            tpch_dir_path,
            dataset_name=dataset_name,
            encoding_type="plain_non_numeric",
            lance_version=lance_version,
        )
        benchmark_tpch_encodings(
            tpch_dir_path,
            dataset_name=dataset_name,
            encoding_type="plain_timestamp",
            lance_version=lance_version,
        )
        benchmark_tpch_encodings(
            tpch_dir_path,
            dataset_name=dataset_name,
            encoding_type="integer",
            lance_version=lance_version,
        )

        """
        benchmark_sift_vector_encodings(
            "/home/ubuntu/test/",
            dataset_name="sift1m",
            encoding_type="plain_fixed_size_list",
            num_trials=5,
            lance_version=lance_version,
        )
        """
