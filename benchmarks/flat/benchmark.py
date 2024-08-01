#!/usr/bin/env python3
#
#  Copyright (c) 2023. Lance Developers
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

import time

import lance
import numpy as np
import pandas as pd
import pyarrow as pa

# use plotly for interactive plots
pd.options.plotting.backend = "plotly"


def benchmark(
    ds: lance.LanceDataset,
    dim: int,
    metric: str,
):
    querys = [np.random.random((dim,)).reshape(-1) for _ in range(32)]
    # warmup
    for query in querys:
        ds.to_table(
            nearest={"column": "vector", "k": 10, "q": query, "use_index": False}
        )

    latency = []

    for _ in range(10):
        for query in querys:
            start = time.perf_counter()
            ds.to_table(
                nearest={
                    "column": "vector",
                    "k": 10,
                    "q": query,
                    "use_index": False,
                    "metric": metric,
                }
            )
            latency.append(time.perf_counter() - start)

    latency = np.array(latency)
    mean = latency.mean() * 1000
    std = latency.std() * 1000
    print(f"Latency: {mean} ms, std: {std} ms")
    return mean, std


def main():
    # make sure we cover sift, BERT, ada2
    dims = [128, 512, 768, 1536, 2048]
    lengths = [10, 100, 1000, 5000, 10000, 20000]
    metrics = ["L2", "cosine", "dot"]

    latency_data = []

    for dim in dims:
        for length in lengths:
            print("Generating {} vectors of dimension {}".format(length, dim))
            data = np.random.random((length, dim)).reshape(-1).astype("f")
            arr = pa.FixedSizeListArray.from_arrays(data, list_size=dim)
            t = pa.Table.from_arrays([arr], names=["vector"])
            ds = lance.write_dataset(
                t,
                "test.lance",
                mode="overwrite",
            )
            for metric in metrics:
                latency, std = benchmark(ds, dim, metric)
                latency_data.append(
                    {
                        # this is an unfortunate hack to make the plot work
                        # plotly can't handle multiindex
                        "dim_metric": f"{dim}_{metric}",
                        "length": length,
                        "latency": latency,
                        "std": std,
                    }
                )

    df = pd.DataFrame(latency_data)
    df.to_csv("benchmark.csv", index=False)

    fig = df.pivot(index="length", columns="dim_metric", values="latency").plot(
        title="Flat Vector Search Latency vs. length of dataset",
        labels={"value": "latency (ms)"},
    )
    fig.write_html("benchmark.html")


if __name__ == "__main__":
    main()
