#  Copyright (c) 2024. Lance Developers
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


import numpy as np
from os import path
from vbench.experiment import Benchmark
from vbench.runner import BenchmarkRunner


def test_config(tempdir, random_dataset):
    # insample query of top1 should all hit itself
    query = np.stack(random_dataset.to_table()["vec"].to_numpy())
    gt = np.arange(len(query)).reshape(-1, 1)

    query_path = path.join(tempdir, "query.npy")
    np.save(query_path, query)
    gt_path = path.join(tempdir, "gt.npy")
    np.save(gt_path, gt)

    config = {
        "name": "test",
        "data": random_dataset.uri,
        "desc": "test",
        "test_cases": {
            "test": {
                "query": query_path,
                "ground_truth": gt_path,
            }
        },
    }

    benchmark = Benchmark(**config)

    res = BenchmarkRunner(
        benchmark=benchmark,
        test_name="test",
        query_params={"column": "vec"},
        num_threads=1,
        index_params={},
    ).run()

    assert res.recall == 1.0
    assert res.throughput > 0
    assert res.index_params == {}
    assert res.query_params == {"num_threads_in_runner": 1, "column": "vec"}
