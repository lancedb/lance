#  Copyright 2023 Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Microbenchmark for performance"""
import shutil
import time

import duckdb
import lance
from lance.vector import vec_to_table
import numpy as np
import pandas as pd
import pyarrow as pa


def gen_vectors(num_vectors, num_dimensions) -> pa.Table:
    mat = np.random.randn(num_vectors, num_dimensions)
    assert mat.shape == (num_vectors, num_dimensions)
    return vec_to_table(mat)


def gen_index(ds, ivf, pq):
    ds.create_index(
        "vector", index_type="IVF_PQ", num_partitions=ivf, num_sub_vectors=pq
    )


def sample(ds, nsamples):
    vec = (
        duckdb.query(f"SELECT vector FROM ds USING SAMPLE {nsamples}")
        .to_arrow_table()["vector"]
        .to_numpy()
    )
    return [np.array(v) for v in vec]


class Suite:
    def __init__(self, benchmark_names):
        self._benchmarks = {k: Benchmark(k) for k in benchmark_names}

    def __getitem__(self, item):
        return self._benchmarks[item]

    def summary(self):
        return {k: v.summary() for k, v in self._benchmarks.items()}


class Benchmark:
    def __init__(self, name):
        self.name = name
        self._configs = {}
        self._timers = {}

    def timer(self, conf):
        conf_tups = list(conf.items())
        sorted(conf_tups)
        conf_str = f"{conf_tups}"
        if conf_str in self._timers:
            return self._timers[conf_str]
        self._configs[conf_str] = conf
        timer = Timer()
        self._timers[conf_str] = timer
        return timer

    def summary(self):
        series = []
        for k, v in self._configs.items():
            timer = self._timers[k]
            config_ser = pd.Series(v)
            time_ser = timer.summary()
            series.append(pd.concat([config_ser, time_ser]))
        return pd.DataFrame(series)


class Timer:
    def __init__(self):
        self.runs = []

    def time(self, func, *args, **kwargs):
        start = time.time()
        rs = func(*args, **kwargs)
        end = time.time()
        self.runs.append(end - start)
        return rs

    def summary(self):
        runs = pd.Series(self.runs)
        return runs.describe()


def run_test(
    nvecs, ndims, ivf_range, pq_range, nprobe_range, refine_factor_range, nsamples=100
):
    print(f"Generating {nvecs} vectors of {ndims}d")
    vectors = gen_vectors(nvecs, ndims)
    uri = f"test_base_{nvecs}_by_{ndims}.lance"
    print(f"Writing to {uri}")
    shutil.rmtree(uri, ignore_errors=True)
    lance.write_dataset(
        vectors, uri, max_rows_per_group=8192, max_rows_per_file=1024 * 1024 * 10
    )
    suite = Suite(["create_index", "query"])

    for ivf in ivf_range:
        for pq in pq_range:
            print(f"Creating index for ivf{ivf}_pq{pq}")
            dest = f"test_{nvecs}_by_{ndims}_ivf{ivf}_pq{pq}.lance"
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(uri, dest)
            ds = lance.dataset(dest)
            bench = suite["create_index"]
            config = {
                "num_vectors": nvecs,
                "num_dimensions": ndims,
                "ivf": ivf,
                "pq": pq,
            }
            timer = bench.timer(config)
            ds = timer.time(
                ds.create_index,
                "vector",
                index_type="IVF_PQ",
                num_partitions=ivf,
                num_sub_vectors=pq,
            )

            for nprobes in nprobe_range:
                for rf in refine_factor_range:
                    print(f"Querying nprobes={nprobes} refine_factor={rf}")
                    bench = suite["query"]
                    config = config.copy()
                    config["nprobes"] = nprobes
                    config["refine_factor"] = rf
                    config["num_samples"] = nsamples
                    timer = bench.timer(config)

                    queries = sample(ds, nsamples)
                    nearest = {
                        "column": "vector",
                        "k": 10,
                        "nprobes": nprobes,
                        "refine_factor": rf,
                    }
                    for q in queries:
                        nearest["q"] = q
                        timer.time(ds.to_table, nearest=nearest)

    summary = suite.summary()
    for name, df in summary.items():
        df.to_csv(f"{name}.csv", index=False)
    return summary
