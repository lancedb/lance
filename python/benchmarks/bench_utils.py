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

from functools import wraps
import multiprocessing as mp
import pandas as pd
import time

import pyarrow.fs
import pyarrow.dataset as ds

import lance

__all__ = ["download_uris", "timeit", "get_dataset", "get_uri", "BenchmarkSuite"]


def get_bytes(uri):
    fs, key = pyarrow.fs.FileSystem.from_uri(uri)
    return fs.open_input_file(key).read()


def download_uris(uris: pd.Series) -> pd.Series:
    pool = mp.Pool(mp.cpu_count() - 1)
    data = pool.map(get_bytes, uris.values)
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
    if uri.endswith('.lance'):
        return lance.dataset(uri)
    return ds.dataset(uri)


def get_uri(base_uri: str, dataset_name: str, fmt: str,
            flavor: str = None) -> str:
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

    def __init__(self):
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


class Benchmark:

    def __init__(self, name, func, key=None, num_runs=1):
        self.name = name
        self.func = func
        self.key = key
        self.num_runs = num_runs
        self._components = {}

    def repeat(self, num_runs: int):
        return Benchmark(self.name, self.func, key=self.key, num_runs=num_runs)

    def run(self, *args, **kwargs):
        self._components = {}
        output = None
        func = self.timeit("total")(self.func)
        for i in range(self.num_runs):
            output = func(*args, **kwargs)
        return output

    def to_df(self):
        return pd.DataFrame(self._components)

    def timeit(self, name):
        def benchmark_decorator(func):
            @wraps(func)
            def timeit_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                # first item in the args, ie `args[0]` is `self`
                print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
                key = tuple([name] + [kwargs.get(k) for k in self.key])
                self._components.setdefault(key, []).append(total_time)
                return result
            return timeit_wrapper
        return benchmark_decorator

