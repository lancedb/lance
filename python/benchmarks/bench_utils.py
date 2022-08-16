#!/usr/bin/env python3
# Copyright 2022 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pathlib
from abc import ABC, abstractmethod
from functools import wraps
import multiprocessing as mp
from typing import Iterable

import click
import pandas as pd
import time

import pyarrow as pa
import pyarrow.fs
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import lance

__all__ = ["download_uris", "timeit", "get_dataset", "get_uri", "BenchmarkSuite"]

KNOWN_FORMATS = ["lance", "parquet", "raw"]


def read_file(uri) -> bytes:
    fs, key = pyarrow.fs.FileSystem.from_uri(uri)
    return fs.open_input_file(key).read()


def download_uris(uris: Iterable[str], func=read_file) -> Iterable[bytes]:
    if isinstance(uris, pd.Series):
        uris = uris.values
    pool = mp.Pool(mp.cpu_count() - 1)
    data = pool.map(func, uris)
    return data


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def get_dataset(uri: str) -> ds.Dataset:
    """
    Return a pyarrow Dataset stored at the given uri
    """
    if uri.endswith(".lance"):
        return lance.dataset(uri)
    return ds.dataset(uri)


def get_uri(base_uri: str, dataset_name: str, fmt: str, flavor: str = None) -> str:
    """
    Return the uri to the dataset with the given specifications

    Parameters
    ----------
    base_uri: str
        Base uri to the root of the benchmark dataset catalog
    dataset_name: str
        Catalog name of the dataset (e.g., coco, oxford_pet)
    fmt: str
        'lance', 'parquet', or 'raw'
    flavor: str, optional
        We may store different flavors for parquet and lance,
        e.g., with image links but not bytes
    """
    return f"{base_uri}/{dataset_name}{('_' + flavor) if flavor else ''}.{fmt}"


class BenchmarkSuite:
    def __init__(self, name: str):
        self.name = name
        self._benchmarks = {}
        self._results = {}

    def benchmark(self, name, key=None):
        def decorator(func):
            b = Benchmark(name, func, key=key)
            self._benchmarks[name] = b
            return func

        return decorator

    def get_benchmark(self, name):
        return self._benchmarks[name]

    def list_benchmarks(self):
        return self._benchmarks.values()

    def create_main(self):
        @click.command
        @click.option(
            "-u",
            "--base-uri",
            required=True,
            type=str,
            help="Base uri to the benchmark dataset catalog",
        )
        @click.option(
            "-f", "--format", "fmt", help="'lance', 'parquet', or 'raw'. Omit for all"
        )
        @click.option(
            "--flavor",
            type=str,
            help="external if parquet/lance had external images version",
        )
        @click.option(
            "-b", "--benchmark", type=str, help="which benchmark to run. Omit for all"
        )
        @click.option(
            "-r", "--repeats", type=int, help="number of times to run each benchmark"
        )
        @click.option(
            "-o", "--output", type=str, help="save timing results to directory"
        )
        def main(base_uri, fmt, flavor, benchmark, repeats, output):
            if fmt:
                fmt = fmt.strip().lower()
                assert fmt in KNOWN_FORMATS
                fmt = [fmt]
            else:
                fmt = KNOWN_FORMATS
            base_uri = f"{base_uri}/datasets/{self.name}"

            def run_benchmark(bmark):
                b = bmark.repeat(repeats or 1)
                for f in fmt:
                    b.run(base_uri=base_uri, fmt=f, flavor=flavor)
                if output:
                    path = pathlib.Path(output) / f"{bmark.name}.csv"
                    b.to_df().to_csv(path, index=False)

            if benchmark is not None:
                b = self.get_benchmark(benchmark)
                run_benchmark(b)
            else:
                [run_benchmark(b) for b in self.list_benchmarks()]

        return main


class Benchmark:
    def __init__(self, name, func, key=None, num_runs=1):
        self.name = name
        self.func = func
        self.key = key
        self.num_runs = num_runs
        self._timings = {}

    def repeat(self, num_runs: int):
        return Benchmark(self.name, self.func, key=self.key, num_runs=num_runs)

    def run(self, *args, **kwargs):
        output = None
        func = self.timeit("total")(self.func)
        for i in range(self.num_runs):
            output = func(*args, **kwargs)
        return output

    def to_df(self):
        return pd.DataFrame(self._timings)

    def timeit(self, name):
        def benchmark_decorator(func):
            @wraps(func)
            def timeit_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                # first item in the args, ie `args[0]` is `self`
                print(
                    f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds"
                )
                key = tuple([name] + [kwargs.get(k) for k in self.key])
                self._timings.setdefault(key, []).append(total_time)
                return result

            return timeit_wrapper

        return benchmark_decorator


class DatasetConverter(ABC):
    def __init__(self, name, uri_root):
        self.name = name
        self.uri_root = uri_root

    @abstractmethod
    def read_metadata(self) -> pd.DataFrame:
        pass

    def default_dataset_path(self, fmt, flavor=None):
        suffix = f"_{flavor}" if flavor else ""
        return os.path.join(self.uri_root, f"{self.name}{suffix}.{fmt}")

    def save_df(self, df, fmt="lance", output_path=None):
        output_path = output_path or self.default_dataset_path(fmt, "links")
        table = pa.Table.from_pandas(df, self.get_schema())
        if fmt == "parquet":
            pq.write_table(table, output_path)
        elif fmt == "lance":
            lance.write_table(table, output_path)
        return table

    @abstractmethod
    def image_uris(self, table):
        pass

    def make_embedded_dataset(
        self, table: pa.Table, fmt="lance", output_path=None, **kwargs
    ):
        output_path = output_path or self.default_dataset_path(fmt)
        uris = self.image_uris(table)
        images = download_uris(pd.Series(uris))
        arr = pa.BinaryArray.from_pandas(images)
        embedded = table.append_column(pa.field("image", pa.binary()), arr)
        if fmt == "parquet":
            pq.write_table(embedded, output_path, **kwargs)
        elif fmt == "lance":
            lance.write_table(embedded, output_path, **kwargs)
        return embedded

    @abstractmethod
    def get_schema(self):
        pass
