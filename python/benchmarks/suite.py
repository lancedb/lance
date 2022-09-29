#  Copyright (c) 2022. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Benchmark suite"""
import pathlib
import time
from functools import wraps

import click
import pandas as pd
import pyarrow.dataset as ds

import lance

KNOWN_FORMATS = ["lance", "parquet", "raw"]


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
    """A group of related performance benchmarks"""

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
        @click.command()
        @click.argument("base_uri")
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
    """A single performance benchmark with convenience to repeat and time"""

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
